#!/usr/bin/env python3
"""Blocking ASE/VASP backend for a production active-learning campaign.

For every batch selected by active learning, this backend creates one working
directory per structure, copies the same external INCAR/POTCAR/KPOINTS files,
writes POSCAR with ASE, runs a caller-supplied blocking VASP command, and reads
the relaxed result back with ASE.  Only the resulting formation energy per atom
is returned to the active-learning loop; OUTCAR, CONTCAR, logs, and a compact
JSON result record remain in the calculation directory.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read, write
from ase.io.vasp import get_atomtypes

if __package__ in {None, ""}:
    from cache import atomic_json_dump
else:
    from .cache import atomic_json_dump

if TYPE_CHECKING:
    from .active_learning import Candidate


CO_REFERENCE_ENERGY_EV = -6.355317795
H_REFERENCE_ENERGY_EV = -11.029648055
MIN_INTERATOMIC_DISTANCE_WARNING_ANGSTROM = 0.7
RESULT_FORMAT_VERSION = 2
SUPPORTED_ELEMENTS = frozenset({"Co", "O", "H"})
IONIC_CONVERGENCE_MARKER = (
    "reached required accuracy - stopping structural energy minimisation"
)


@dataclass(frozen=True)
class PreparedCalculation:
    """One validated and prepared VASP calculation."""

    structure_id: str
    job_dir: Path
    signature: dict[str, Any]
    cached_energy: float | None
    collect_only: bool = False


@dataclass(frozen=True)
class VaspOutput:
    """Data collected from one completed VASP optimization."""

    total_energy_ev: float
    final_atoms: Atoms


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_directory_name(value: str, *, field: str) -> str:
    normalised = str(value)
    if (
        not normalised
        or normalised in {".", ".."}
        or Path(normalised).name != normalised
    ):
        raise ValueError(f"{field} must be a non-empty directory-safe name")
    return normalised


def _read_potcar_species(path: Path) -> tuple[str, ...]:
    species = get_atomtypes(path)
    if not species:
        raise ValueError(f"Could not read species from POTCAR: {path}")
    if len(set(species)) != len(species):
        raise ValueError(f"POTCAR contains duplicate species entries: {species}")
    return tuple(species)


class VaspLabelBackend:
    """Evaluate active-learning candidates with blocking VASP calculations."""

    def __init__(
        self,
        *,
        incar_path: Path,
        potcar_path: Path,
        kpoints_path: Path,
        work_dir: Path,
        vasp_command: str,
        max_workers: int = 1,
        co_reference_energy: float = CO_REFERENCE_ENERGY_EV,
        h_reference_energy: float = H_REFERENCE_ENERGY_EV,
    ) -> None:
        self.incar_path = Path(incar_path).resolve()
        self.potcar_path = Path(potcar_path).resolve()
        self.kpoints_path = Path(kpoints_path).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.vasp_command = str(vasp_command).strip()
        self.command_argv = tuple(shlex.split(self.vasp_command))
        self.max_workers = max_workers
        self.co_reference_energy = float(co_reference_energy)
        self.h_reference_energy = float(h_reference_energy)

        for label, path in self._template_paths().items():
            if not path.is_file():
                raise FileNotFoundError(f"External {label} not found: {path}")
        if not self.command_argv:
            raise ValueError("vasp_command must not be empty")
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise ValueError("max_workers must be a positive integer")
        if max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        if not np.isfinite([self.co_reference_energy, self.h_reference_energy]).all():
            raise ValueError("Reference energies must be finite")

        self._template_sha256 = {
            label: _sha256(path) for label, path in self._template_paths().items()
        }
        self.potcar_species = _read_potcar_species(self.potcar_path)

    def _template_paths(self) -> dict[str, Path]:
        return {
            "INCAR": self.incar_path,
            "POTCAR": self.potcar_path,
            "KPOINTS": self.kpoints_path,
        }

    @staticmethod
    def _validate_atoms(atoms: Atoms, *, structure_id: str) -> None:
        if len(atoms) == 0:
            raise ValueError(f"Structure contains no atoms: {structure_id}")
        unsupported = set(atoms.get_chemical_symbols()).difference(SUPPORTED_ELEMENTS)
        if unsupported:
            raise ValueError(
                f"Structure {structure_id} contains unsupported elements: "
                f"{sorted(unsupported)}"
            )
        positions = np.asarray(atoms.get_positions(), dtype=float)
        cell = np.asarray(atoms.cell.array, dtype=float)
        if not np.isfinite(positions).all() or not np.isfinite(cell).all():
            raise ValueError(f"Structure contains non-finite geometry: {structure_id}")
        if abs(float(np.linalg.det(cell))) <= 1.0e-12:
            raise ValueError(f"Structure has no valid VASP cell: {structure_id}")
        if len(atoms) > 1:
            distances = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))
            np.fill_diagonal(distances, np.inf)
            minimum_distance = float(np.min(distances))
            if minimum_distance < MIN_INTERATOMIC_DISTANCE_WARNING_ANGSTROM:
                warnings.warn(
                    f"Structure {structure_id} has a short interatomic distance "
                    f"of {minimum_distance:.3f} Å; continuing unchanged",
                    RuntimeWarning,
                    stacklevel=3,
                )

    @classmethod
    def _read_candidate(cls, candidate: "Candidate") -> Atoms:
        path = Path(candidate.structure_path)
        if not path.is_file():
            raise FileNotFoundError(f"Structure file not found: {path}")
        atoms = read(path)
        if not isinstance(atoms, Atoms):
            raise ValueError(f"Expected one ASE structure in: {path}")
        cls._validate_atoms(atoms, structure_id=str(candidate.structure_id))
        return atoms

    def _order_atoms_for_potcar(self, atoms: Atoms, *, structure_id: str) -> Atoms:
        present_species = set(atoms.get_chemical_symbols())
        missing = present_species.difference(self.potcar_species)
        if missing:
            raise ValueError(
                f"POTCAR is missing species required by {structure_id}: "
                f"{sorted(missing)}"
            )
        expected_prefix = self.potcar_species[: len(present_species)]
        if set(expected_prefix) != present_species:
            raise ValueError(
                "One shared POTCAR cannot map this structure safely. "
                f"{structure_id} contains {sorted(present_species)}, but the first "
                f"{len(present_species)} POTCAR entries are {list(expected_prefix)}"
            )
        symbols = atoms.get_chemical_symbols()
        order = [
            index
            for species in expected_prefix
            for index, symbol in enumerate(symbols)
            if symbol == species
        ]
        return atoms[order]

    def validate_pool(self, candidates: Sequence["Candidate"]) -> None:
        """Validate the complete structure pool before SOAP is calculated."""
        candidate_ids = [str(candidate.structure_id) for candidate in candidates]
        if not candidate_ids:
            raise ValueError("At least one candidate is required")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Candidate pool contains duplicate structure IDs")
        for candidate in candidates:
            _safe_directory_name(str(candidate.structure_id), field="structure_id")
            atoms = self._read_candidate(candidate)
            self._order_atoms_for_potcar(
                atoms,
                structure_id=str(candidate.structure_id),
            )

    def _job_signature(
        self,
        candidate: "Candidate",
        *,
        request_id: str,
    ) -> dict[str, Any]:
        return {
            "format_version": RESULT_FORMAT_VERSION,
            "request_id": request_id,
            "structure_id": str(candidate.structure_id),
            "source_structure_sha256": _sha256(Path(candidate.structure_path)),
            "incar_sha256": self._template_sha256["INCAR"],
            "potcar_sha256": self._template_sha256["POTCAR"],
            "potcar_species": list(self.potcar_species),
            "kpoints_sha256": self._template_sha256["KPOINTS"],
            "vasp_command": list(self.command_argv),
            "co_reference_energy_ev": self.co_reference_energy,
            "h_reference_energy_ev": self.h_reference_energy,
        }

    def _load_cached_energy(
        self,
        result_path: Path,
        *,
        signature: dict[str, Any],
    ) -> float | None:
        if not result_path.exists():
            return None
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid VASP result record: {result_path}") from exc
        if not isinstance(payload, dict) or payload.get("signature") != signature:
            raise ValueError(
                "Existing VASP result does not match the requested calculation: "
                f"{result_path}"
            )
        try:
            energy = float(payload["formation_energy_per_atom_ev"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid VASP result record: {result_path}") from exc
        if not np.isfinite(energy):
            raise ValueError(f"Non-finite energy in VASP result record: {result_path}")
        for filename, signature_key in (
            ("INCAR", "incar_sha256"),
            ("POTCAR", "potcar_sha256"),
            ("KPOINTS", "kpoints_sha256"),
        ):
            input_path = result_path.parent / filename
            if (
                not input_path.is_file()
                or _sha256(input_path) != signature[signature_key]
            ):
                raise ValueError(
                    f"Cached VASP result has a missing or changed {filename}: "
                    f"{result_path.parent}"
                )
        for filename in ("POSCAR", "OUTCAR", "CONTCAR"):
            output_path = result_path.parent / filename
            if not output_path.is_file() or _sha256(output_path) != payload.get(
                "file_sha256", {}
            ).get(filename):
                raise ValueError(
                    f"Cached VASP result has a missing or changed {filename}: {result_path}"
                )
        self._validate_inputs(result_path.parent, signature)
        output = self._read_vasp_output(result_path.parent)
        expected = self._formation_energy(result_path.parent, output)
        if (
            energy != expected
            or payload.get("total_energy_ev") != output.total_energy_ev
            or payload.get("n_atoms") != len(output.final_atoms)
            or payload.get("element_counts")
            != dict(Counter(output.final_atoms.get_chemical_symbols()))
            or payload.get("electronic_converged") is not True
            or payload.get("ionic_converged") is not True
        ):
            raise ValueError(
                f"Cached VASP energies disagree with the output: {result_path}"
            )
        return energy

    def _copy_template(self, label: str, job_dir: Path) -> None:
        source = self._template_paths()[label]
        destination = job_dir / label
        shutil.copyfile(source, destination)
        if _sha256(destination) != self._template_sha256[label]:
            raise RuntimeError(f"External {label} changed while preparing the batch")

    def _validate_inputs(self, job_dir: Path, signature: dict[str, Any]) -> None:
        """Check the immutable inputs recorded before the first execution."""
        request_path = job_dir / "request.json"
        if not request_path.is_file():
            raise ValueError(f"VASP request record is missing: {job_dir}")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if request.get("signature") != signature:
            raise ValueError(f"VASP request does not match current inputs: {job_dir}")
        for name in ("POSCAR", "INCAR", "POTCAR", "KPOINTS"):
            path = job_dir / name
            if not path.is_file() or _sha256(path) != request.get(
                "file_sha256", {}
            ).get(name):
                raise ValueError(f"VASP input is missing or changed {name}: {path}")

    def _recover_output(self, job_dir: Path, signature: dict[str, Any]) -> bool:
        """Allow collection after interruption only with matching, completed output."""
        execution_path = job_dir / "execution.json"
        has_output = any((job_dir / name).exists() for name in ("OUTCAR", "CONTCAR"))
        if not execution_path.exists() and not has_output:
            return False
        self._validate_inputs(job_dir, signature)
        execution = (
            json.loads(execution_path.read_text(encoding="utf-8"))
            if execution_path.exists()
            else {}
        )
        if execution.get("status") == "launch_failed" and not has_output:
            return False
        if execution.get("status") == "finished":
            if execution.get("returncode") != 0:
                raise RuntimeError(
                    f"Previous VASP command failed; inspect the job before retrying: {job_dir}"
                )
            return True
        # The controller can stop after VASP exits but before recording its status.
        outcar = job_dir / "OUTCAR"
        if outcar.is_file():
            with outcar.open(encoding="utf-8", errors="replace") as handle:
                if any(
                    "General timing and accounting informations for this job" in line
                    for line in handle
                ):
                    return True
        raise RuntimeError(
            f"VASP completion is unconfirmed: {job_dir}; wait for the existing job "
            "or inspect and archive it before retrying"
        )

    def _prepare_calculation(
        self,
        candidate: "Candidate",
        *,
        request_id: str,
    ) -> PreparedCalculation:
        structure_id = _safe_directory_name(
            str(candidate.structure_id), field="structure_id"
        )
        atoms = self._read_candidate(candidate)
        atoms = self._order_atoms_for_potcar(atoms, structure_id=structure_id)
        job_dir = self.work_dir / request_id / structure_id
        job_dir.mkdir(parents=True, exist_ok=True)
        signature = self._job_signature(candidate, request_id=request_id)
        cached_energy = self._load_cached_energy(
            job_dir / "result.json",
            signature=signature,
        )
        collect_only = cached_energy is None and self._recover_output(
            job_dir, signature
        )
        if cached_energy is None and not collect_only:
            for label in self._template_paths():
                self._copy_template(label, job_dir)
            write(
                job_dir / "POSCAR",
                atoms,
                format="vasp",
                direct=True,
                sort=False,
                vasp5=True,
            )
            atomic_json_dump(
                job_dir / "request.json",
                {
                    "signature": signature,
                    "source_structure": str(Path(candidate.structure_path).resolve()),
                    "file_sha256": {
                        name: _sha256(job_dir / name)
                        for name in ("POSCAR", "INCAR", "POTCAR", "KPOINTS")
                    },
                },
            )
        return PreparedCalculation(
            structure_id=structure_id,
            job_dir=job_dir,
            signature=signature,
            cached_energy=cached_energy,
            collect_only=collect_only,
        )

    def _execute_vasp(self, job_dir: Path) -> None:
        with (job_dir / "vasp.stdout").open("w", encoding="utf-8") as stdout:
            with (job_dir / "vasp.stderr").open("w", encoding="utf-8") as stderr:
                atomic_json_dump(job_dir / "execution.json", {"status": "running"})
                try:
                    process = subprocess.Popen(
                        self.command_argv, cwd=job_dir, stdout=stdout, stderr=stderr
                    )
                except OSError:
                    atomic_json_dump(
                        job_dir / "execution.json", {"status": "launch_failed"}
                    )
                    raise
                returncode = process.wait()
        atomic_json_dump(
            job_dir / "execution.json",
            {"status": "finished", "returncode": returncode},
        )
        if returncode != 0:
            raise RuntimeError(
                f"VASP command failed with exit code {returncode}: {job_dir}"
            )

    @staticmethod
    def _ionic_converged(outcar_path: Path) -> bool:
        converged = False
        with outcar_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "Iteration" in line or "total energy-change" in line:
                    converged = False
                if IONIC_CONVERGENCE_MARKER in line:
                    converged = True
        return converged

    @staticmethod
    def _electronic_converged(outcar_path: Path) -> bool:
        """Check the final SCF loop; an earlier converged ionic step is insufficient."""
        ediff = None
        converged = False
        with outcar_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if match := re.search(r"\bEDIFF\s*=\s*(\S+)", line):
                    ediff = float(match.group(1).replace("D", "E"))
                if re.search(r"Iteration\s+\d+\s*\(\s*1\s*\)", line):
                    converged = False
                if "total energy-change" in line:
                    # VASP 5 may omit E in three-digit exponents.
                    fields = (
                        line.split(":", 1)[-1]
                        .replace("(", " ")
                        .replace(")", " ")
                        .split()
                    )
                    try:
                        changes = [
                            float(re.sub(r"(?<=\d)([+-]\d{3})$", r"E\1", value))
                            for value in fields
                        ]
                        converged = (
                            len(changes) == 2
                            and ediff is not None
                            and all(abs(value) < ediff for value in changes)
                        )
                    except ValueError:
                        converged = False
                if "aborting loop" in line:
                    converged = "EDIFF is reached" in line
        return converged

    def _read_vasp_output(self, job_dir: Path) -> VaspOutput:
        outcar_path = job_dir / "OUTCAR"
        contcar_path = job_dir / "CONTCAR"
        if not outcar_path.is_file():
            raise FileNotFoundError(f"VASP did not produce OUTCAR: {job_dir}")
        if not contcar_path.is_file():
            raise FileNotFoundError(f"VASP did not produce CONTCAR: {job_dir}")
        ionic_converged = self._ionic_converged(outcar_path)
        if not ionic_converged:
            raise RuntimeError(f"VASP ionic optimization did not converge: {job_dir}")
        if not self._electronic_converged(outcar_path):
            raise RuntimeError(
                f"VASP final electronic step did not converge: {job_dir}"
            )
        try:
            outcar_atoms = read(outcar_path, index=-1)
            final_atoms = read(contcar_path, index=-1)
            if not isinstance(outcar_atoms, Atoms):
                raise ValueError("OUTCAR contains multiple structures")
            if not isinstance(final_atoms, Atoms):
                raise ValueError("CONTCAR contains multiple structures")
            total_energy = float(outcar_atoms.get_potential_energy())
        except Exception as exc:
            raise RuntimeError(
                f"Could not read completed VASP output: {job_dir}"
            ) from exc
        if not isinstance(final_atoms, Atoms) or len(final_atoms) == 0:
            raise ValueError(f"CONTCAR contains no final structure: {job_dir}")
        if not np.isfinite(total_energy):
            raise ValueError(f"OUTCAR contains a non-finite energy: {job_dir}")
        self._validate_atoms(final_atoms, structure_id=job_dir.name)
        if (
            outcar_atoms.get_chemical_symbols() != final_atoms.get_chemical_symbols()
            or not np.allclose(outcar_atoms.cell, final_atoms.cell, rtol=0, atol=1e-5)
        ):
            raise ValueError(
                f"OUTCAR and CONTCAR describe different structures: {job_dir}"
            )
        _, distances = find_mic(
            outcar_atoms.positions - final_atoms.positions, final_atoms.cell, pbc=True
        )
        if np.any(distances > 1e-4):
            raise ValueError(f"OUTCAR and CONTCAR positions disagree: {job_dir}")
        return VaspOutput(total_energy, final_atoms)

    def _formation_energy(self, job_dir: Path, output: VaspOutput) -> float:
        initial_atoms = read(job_dir / "POSCAR")
        if not isinstance(initial_atoms, Atoms):
            raise ValueError(f"POSCAR contains multiple structures: {job_dir}")
        if Counter(initial_atoms.get_chemical_symbols()) != Counter(
            output.final_atoms.get_chemical_symbols()
        ):
            raise ValueError(
                "VASP changed the structure composition unexpectedly: " f"{job_dir}"
            )
        counts = Counter(output.final_atoms.get_chemical_symbols())
        n_atoms = len(output.final_atoms)
        formation_energy = (
            output.total_energy_ev
            - self.co_reference_energy * counts["Co"]
            - self.h_reference_energy * counts["H"]
        ) / n_atoms
        if not np.isfinite(formation_energy):
            raise ValueError(f"Calculated a non-finite formation energy: {job_dir}")
        return float(formation_energy)

    def _run_calculation(self, prepared: PreparedCalculation) -> tuple[str, float]:
        if prepared.cached_energy is not None:
            return prepared.structure_id, prepared.cached_energy

        self._validate_inputs(prepared.job_dir, prepared.signature)
        if not prepared.collect_only:
            self._execute_vasp(prepared.job_dir)
        self._validate_inputs(prepared.job_dir, prepared.signature)
        output = self._read_vasp_output(prepared.job_dir)
        formation_energy = self._formation_energy(prepared.job_dir, output)
        counts = Counter(output.final_atoms.get_chemical_symbols())

        atomic_json_dump(
            prepared.job_dir / "result.json",
            {
                "signature": prepared.signature,
                "total_energy_ev": output.total_energy_ev,
                "formation_energy_per_atom_ev": formation_energy,
                "n_atoms": len(output.final_atoms),
                "element_counts": dict(sorted(counts.items())),
                "ionic_converged": True,
                "electronic_converged": True,
                "file_sha256": {
                    name: _sha256(prepared.job_dir / name)
                    for name in ("POSCAR", "OUTCAR", "CONTCAR")
                },
                "final_structure": "CONTCAR",
                "energy_source": "final OUTCAR energy(sigma->0), read by ASE",
            },
        )
        return prepared.structure_id, float(formation_energy)

    def evaluate(
        self,
        candidates: Sequence["Candidate"],
        *,
        request_id: str,
    ) -> dict[str, float]:
        """Run one complete selected batch and return its formation energies."""
        safe_request_id = _safe_directory_name(request_id, field="request_id")
        candidate_ids = [str(candidate.structure_id) for candidate in candidates]
        if not candidate_ids:
            raise ValueError("At least one candidate is required")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("A VASP batch cannot contain duplicate structure IDs")

        prepared = [
            self._prepare_calculation(candidate, request_id=safe_request_id)
            for candidate in candidates
        ]
        worker_count = min(self.max_workers, len(prepared))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(self._run_calculation, prepared))
        return dict(results)

    def verify(
        self, candidates: Sequence["Candidate"], *, request_id: str
    ) -> dict[str, float]:
        """Revalidate a completed batch without preparing or launching any jobs."""
        request_id = _safe_directory_name(request_id, field="request_id")
        results: dict[str, float] = {}
        for candidate in candidates:
            key = _safe_directory_name(candidate.structure_id, field="structure_id")
            path = self.work_dir / request_id / key / "result.json"
            energy = self._load_cached_energy(
                path, signature=self._job_signature(candidate, request_id=request_id)
            )
            if energy is None:
                raise FileNotFoundError(
                    f"Completed VASP result record is missing: {path}"
                )
            results[key] = energy
        return results

    def cache_key(self) -> dict[str, Any]:
        """Return settings that determine compatibility with an AL cache."""
        return {
            "name": "vasp",
            "result_format_version": RESULT_FORMAT_VERSION,
            "incar_sha256": self._template_sha256["INCAR"],
            "potcar_sha256": self._template_sha256["POTCAR"],
            "potcar_species": list(self.potcar_species),
            "kpoints_sha256": self._template_sha256["KPOINTS"],
            "vasp_command": list(self.command_argv),
            "co_reference_energy_ev": self.co_reference_energy,
            "h_reference_energy_ev": self.h_reference_energy,
            "requires_ionic_convergence": True,
            "requires_electronic_convergence": True,
        }

#!/usr/bin/env python3
"""SOAP/PCA, GP-LCB acquisition, and a resumable DFT-labeling campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from ase.io import read
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    from cache import (
        CAMPAIGN_CACHE_FORMAT_VERSION,
        atomic_pickle_dump,
        load_campaign_cache,
    )
else:
    from .cache import (
        CAMPAIGN_CACHE_FORMAT_VERSION,
        atomic_pickle_dump,
        load_campaign_cache,
    )


@dataclass(frozen=True)
class Candidate:
    """One structure available to an Active Learning campaign."""

    structure_id: str
    structure_path: Path


class EnergyBackend(Protocol):
    """Blocking label-evaluation interface used by a campaign."""

    def evaluate(
        self,
        candidates: Sequence[Candidate],
        *,
        request_id: str,
    ) -> Mapping[str, float]:
        """Return one finite energy for every supplied candidate ID."""
        ...

    def validate_pool(self, candidates: Sequence[Candidate]) -> None:
        """Validate all structures before feature calculation or job submission."""
        ...

    def verify(
        self, candidates: Sequence[Candidate], *, request_id: str
    ) -> Mapping[str, float]:
        """Validate previously collected outputs without launching calculations."""
        ...

    def cache_key(self) -> Mapping[str, Any]:
        """Return a stable configuration fingerprint for cache compatibility."""
        ...


def find_project_root(start: Path) -> Path:
    """Return the nearest parent containing the project-root marker."""
    for candidate in (start, *start.parents):
        if (candidate / ".project-root").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a parent containing .project-root. "
        "Pass --project-root explicitly."
    )


def load_structure_pool(
    project_root: Path,
) -> tuple[np.ndarray, list[Path], list[Candidate]]:
    """Load every unrelaxed POSCAR without reading any candidate energy."""
    structure_dir = project_root / "initial_structures"
    if not structure_dir.is_dir():
        raise FileNotFoundError(f"Structure directory not found: {structure_dir}")
    structure_paths = sorted(structure_dir.glob("*_POSCAR"), key=lambda path: path.name)
    if not structure_paths:
        raise ValueError(f"No *_POSCAR structures found in: {structure_dir}")
    ids = np.asarray(
        [path.name.removesuffix("_POSCAR") for path in structure_paths],
        dtype=str,
    )
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Structure pool contains duplicate IDs")
    candidates = [
        Candidate(str(structure_id), structure_path)
        for structure_id, structure_path in zip(ids, structure_paths, strict=True)
    ]
    print(f"Loaded unlabeled structure pool: {len(candidates)} structures")
    return ids, structure_paths, candidates


def load_initial_labels_csv(
    path: Path,
    *,
    id_column: str = "ID",
    energy_column: str = "E_f_per_atom",
) -> dict[str, float]:
    """Load only the explicitly supplied first-batch energy labels."""
    if not path.is_file():
        raise FileNotFoundError(f"Initial-label CSV not found: {path}")
    table = pd.read_csv(path, encoding="utf-8-sig", dtype={id_column: str})
    missing = {id_column, energy_column}.difference(table.columns)
    if missing:
        raise ValueError(f"Missing initial-label CSV columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("Initial-label CSV must contain at least one row")
    if table[id_column].isna().any() or (table[id_column].str.strip() == "").any():
        raise ValueError("Initial-label CSV contains missing or empty IDs")
    if table[id_column].duplicated().any():
        raise ValueError("Initial-label CSV contains duplicate IDs")
    energies = table[energy_column].to_numpy(dtype=float)
    if not np.isfinite(energies).all():
        raise ValueError("Initial-label CSV contains non-finite energies")
    return dict(zip(table[id_column].astype(str), energies.tolist(), strict=True))


def calculate_features(
    structure_paths: Sequence[Path],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calculate SOAP descriptors, clean them, standardize, and reduce by PCA."""
    from dscribe.descriptors import SOAP

    if not structure_paths:
        raise ValueError("At least one structure is required")

    soap = SOAP(
        species=["Co", "O", "H"],
        r_cut=5.0,
        n_max=8,
        l_max=6,
        periodic=True,
        average="outer",
    )

    feature_rows: list[np.ndarray] = []
    for index, path in enumerate(structure_paths, start=1):
        feature_rows.append(np.asarray(soap.create(read(path)), dtype=float))
        if index == 1 or index % 50 == 0 or index == len(structure_paths):
            print(f"SOAP: {index}/{len(structure_paths)} structures")

    raw = np.asarray(feature_rows, dtype=float)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("SOAP descriptors must be a finite 2-D array")

    selector = VarianceThreshold(threshold=0.0)
    cleaned = selector.fit_transform(raw)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cleaned)
    pca = PCA(n_components=0.99, svd_solver="full", random_state=0)
    reduced = pca.fit_transform(scaled)
    if reduced.ndim != 2 or not np.isfinite(reduced).all():
        raise ValueError("Reduced features must be a finite 2-D array")

    metadata = {
        "n_structures": int(raw.shape[0]),
        "n_raw_features": int(raw.shape[1]),
        "n_nonconstant_features": int(cleaned.shape[1]),
        "n_pca_components": int(reduced.shape[1]),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    print(
        "Feature pipeline: "
        f"{raw.shape} -> {cleaned.shape} -> {reduced.shape}; "
        f"variance={metadata['pca_explained_variance']:.5f}"
    )
    return reduced, metadata


def _validate_positive_int(name: str, value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer; got {value}")


def _normalise_indices(
    indices: Sequence[int],
    *,
    n_samples: int,
    allowed_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Validate a caller-supplied unique set of structure indices."""
    try:
        normalised = np.asarray(indices, dtype=int)
    except (TypeError, ValueError) as exc:
        raise ValueError("indices must contain integers") from exc
    if normalised.ndim != 1 or len(normalised) == 0:
        raise ValueError("indices must be a non-empty 1-D sequence")
    if np.any(normalised < 0) or np.any(normalised >= n_samples):
        raise ValueError("indices contain an out-of-range structure index")
    if len(np.unique(normalised)) != len(normalised):
        raise ValueError("indices must be unique")
    if allowed_indices is not None and not np.isin(normalised, allowed_indices).all():
        raise ValueError("indices contain values outside the allowed set")
    return normalised


def fit_gpr(
    x_labeled: np.ndarray,
    y_labeled: np.ndarray,
    *,
    n_restarts_optimizer: int = 8,
    random_state: int = 0,
) -> tuple[GaussianProcessRegressor, StandardScaler]:
    """Fit the standardized Matern-3/2 GP surrogate used by GP-LCB."""
    if x_labeled.ndim != 2 or x_labeled.shape[0] == 0:
        raise ValueError("x_labeled must be a non-empty 2-D array")
    if y_labeled.ndim != 1 or y_labeled.shape[0] != x_labeled.shape[0]:
        raise ValueError("y_labeled must align with x_labeled")
    if not np.isfinite(x_labeled).all() or not np.isfinite(y_labeled).all():
        raise ValueError("x_labeled and y_labeled must contain only finite values")
    if (
        isinstance(n_restarts_optimizer, bool)
        or not isinstance(n_restarts_optimizer, (int, np.integer))
        or n_restarts_optimizer < 0
    ):
        raise ValueError(
            "n_restarts_optimizer must be a non-negative integer; "
            f"got {n_restarts_optimizer}"
        )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_labeled)
    dimension = x_train.shape[1]
    kernel = ConstantKernel(1.0) * Matern(
        length_scale=np.ones(dimension), nu=1.5
    ) + WhiteKernel(noise_level=1e-4)
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )
    model.fit(x_train, y_labeled)
    return model, scaler


def acquire(
    model: GaussianProcessRegressor,
    scaler: StandardScaler,
    x_pool: np.ndarray,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return GP-LCB scores, posterior means, and posterior standard deviations."""
    if x_pool.ndim != 2 or x_pool.shape[0] == 0:
        raise ValueError("x_pool must be a non-empty 2-D array")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError(f"kappa must be finite and non-negative; got {kappa}")
    x_pool_scaled = scaler.transform(x_pool)
    mean, std = model.predict(x_pool_scaled, return_std=True)
    acquisition = mean - kappa * std
    if not np.isfinite(acquisition).all():
        raise ValueError("GP-LCB acquisition contains non-finite values")
    return acquisition, mean, std


def select_diverse_batch(
    x_embed: np.ndarray,
    scores: np.ndarray,
    pool_indices: np.ndarray,
    batch_size: int,
    preselect_factor: int = 5,
    *,
    random_state: int = 0,
) -> np.ndarray:
    """Select low-score candidates, then retain diversity with KMeans."""
    _validate_positive_int("batch_size", batch_size)
    _validate_positive_int("preselect_factor", preselect_factor)
    if x_embed.ndim != 2 or x_embed.shape[0] == 0:
        raise ValueError("x_embed must be a non-empty 2-D array")
    if scores.ndim != 1 or pool_indices.ndim != 1:
        raise ValueError("scores and pool_indices must be 1-D arrays")
    if len(scores) != len(pool_indices) or len(scores) != len(x_embed):
        raise ValueError("x_embed, scores, and pool_indices must have equal length")
    if not np.isfinite(x_embed).all() or not np.isfinite(scores).all():
        raise ValueError("x_embed and scores must contain only finite values")
    if len(np.unique(pool_indices)) != len(pool_indices):
        raise ValueError("pool_indices must be unique")

    n_selected = min(batch_size, len(pool_indices))
    n_preselected = min(len(pool_indices), preselect_factor * n_selected)
    top_local = np.argsort(scores, kind="stable")[:n_preselected]
    top_global = pool_indices[top_local]
    x_candidates = x_embed[top_local]

    n_clusters = min(n_selected, n_preselected)
    chosen: list[int] = []
    # KMeans can collapse several clusters when valid candidates have
    # identical embeddings.  In that case diversity is unavailable, so the
    # deterministic score order is the correct fallback.
    n_unique_embeddings = np.unique(x_candidates, axis=0).shape[0]
    if n_unique_embeddings >= n_clusters:
        labels = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=random_state,
        ).fit_predict(x_candidates)
        for cluster in range(n_clusters):
            members = np.where(labels == cluster)[0]
            if len(members) == 0:
                continue
            member = members[np.argmin(scores[top_local[members]])]
            chosen.append(int(top_global[member]))

    selected_set = set(chosen)
    for candidate in top_global:
        if len(chosen) == n_selected:
            break
        candidate_int = int(candidate)
        if candidate_int not in selected_set:
            chosen.append(candidate_int)
            selected_set.add(candidate_int)

    selected = np.asarray(chosen, dtype=int)
    if len(selected) != n_selected or len(np.unique(selected)) != len(selected):
        raise RuntimeError(
            "Diverse batch selection did not return the requested number of "
            "unique candidates"
        )
    return selected


def _initial_labels(
    initial_labels: Mapping[str, float], candidate_ids: Sequence[str]
) -> dict[str, float]:
    if not initial_labels:
        raise ValueError("initial_labels must contain at least one labeled structure")
    unknown_ids = set(map(str, initial_labels)).difference(candidate_ids)
    if unknown_ids:
        raise KeyError(
            f"Initial labels contain IDs outside the candidate pool: {sorted(unknown_ids)[:5]}"
        )
    labels = {str(key): float(value) for key, value in initial_labels.items()}
    if not np.isfinite(list(labels.values())).all():
        raise ValueError("initial_labels must contain only finite energies")
    return labels


def _evaluate_batch(
    backend: EnergyBackend, candidates: Sequence[Candidate], request_id: str
) -> np.ndarray:
    selected_ids = [candidate.structure_id for candidate in candidates]
    returned = dict(backend.evaluate(candidates, request_id=request_id))
    missing = set(selected_ids).difference(returned)
    extra = set(returned).difference(selected_ids)
    if missing or extra:
        raise ValueError(
            "Backend result IDs must exactly match the selected batch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    energies = np.asarray([float(returned[key]) for key in selected_ids], dtype=float)
    if not np.isfinite(energies).all():
        raise ValueError("Backend returned non-finite energy labels")
    return energies


def run_campaign(
    features: np.ndarray,
    candidates: Sequence[Candidate],
    initial_labels: Mapping[str, float],
    backend: EnergyBackend,
    *,
    iterations: int = 100,
    batch_size: int = 2,
    kappa: float = 2.5,
    seed: int = 35,
    iteration_offset: int = 0,
    campaign_id: str = "campaign",
    resume_pending: Mapping[str, Any] | None = None,
    on_round: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one production AL trajectory without requiring pool energies."""
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, (int, np.integer))
        or iterations < 0
    ):
        raise ValueError(f"iterations must be a non-negative integer; got {iterations}")
    _validate_positive_int("batch_size", batch_size)
    if iteration_offset < 0:
        raise ValueError("iteration_offset must be non-negative")
    if not campaign_id.strip():
        raise ValueError("campaign_id must not be empty")
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("features must be a non-empty 2-D array")
    if not np.isfinite(features).all():
        raise ValueError("features must contain only finite values")
    if len(candidates) != features.shape[0]:
        raise ValueError("candidates must align with feature rows")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError(f"kappa must be finite and non-negative; got {kappa}")
    candidate_ids = [str(candidate.structure_id) for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate structure IDs must be unique")
    label_by_id = _initial_labels(initial_labels, candidate_ids)
    labeled_indices = np.asarray(
        [
            index
            for index, structure_id in enumerate(candidate_ids)
            if structure_id in label_by_id
        ],
        dtype=int,
    )
    y_labeled = np.asarray(
        [label_by_id[candidate_ids[index]] for index in labeled_indices],
        dtype=float,
    )
    all_indices = np.arange(len(candidates))
    rounds: list[dict[str, Any]] = []
    pending_to_resume = dict(resume_pending) if resume_pending is not None else None

    def emit_checkpoint(pending_round: dict[str, Any] | None) -> None:
        if on_round is None:
            return
        unavailable_indices = labeled_indices
        if pending_round is not None:
            pending_indices = np.asarray(pending_round["selected_indices"], dtype=int)
            unavailable_indices = np.concatenate([unavailable_indices, pending_indices])
        remaining_indices = np.setdiff1d(
            all_indices,
            unavailable_indices,
            assume_unique=False,
        )
        on_round(
            {
                "complete": False,
                "rounds": list(rounds),
                "pending_round": pending_round,
                "labeled_indices": labeled_indices.copy(),
                "labeled_energies": dict(label_by_id),
                "remaining_ids": [candidate_ids[index] for index in remaining_indices],
                "settings": {
                    "iterations": int(iteration_offset + iterations),
                    "batch_size": int(batch_size),
                    "kappa": float(kappa),
                    "seed": int(seed),
                },
            }
        )

    for local_iteration in range(1, iterations + 1):
        iteration = iteration_offset + local_iteration
        pool_indices = np.setdiff1d(all_indices, labeled_indices, assume_unique=False)
        if len(pool_indices) == 0:
            break

        if pending_to_resume is not None:
            if int(pending_to_resume.get("iteration", -1)) != iteration:
                raise ValueError("Pending campaign round is not the next iteration")
            chosen = _normalise_indices(
                pending_to_resume.get("selected_indices", []),
                n_samples=len(candidates),
                allowed_indices=pool_indices,
            )
            selected_ids = [candidate_ids[index] for index in chosen]
            if selected_ids != list(pending_to_resume.get("selected_ids", [])):
                raise ValueError("Pending campaign IDs do not match their indices")
            request_id = str(pending_to_resume.get("request_id", ""))
            if not request_id:
                raise ValueError("Pending campaign round has no request_id")
            pending_to_resume = None
        else:
            model, scaler = fit_gpr(features[labeled_indices], y_labeled)
            acquisition, _, _ = acquire(model, scaler, features[pool_indices], kappa)
            chosen = select_diverse_batch(
                scaler.transform(features[pool_indices]),
                acquisition,
                pool_indices,
                batch_size=min(batch_size, len(pool_indices)),
                random_state=seed + iteration - 1,
            )
            selected_ids = [candidate_ids[index] for index in chosen]
            selection_digest = hashlib.sha256(
                "\0".join(selected_ids).encode("utf-8")
            ).hexdigest()[:12]
            request_id = f"{campaign_id}-round-{iteration:06d}-{selection_digest}"
        selected_candidates = [candidates[index] for index in chosen]
        pending_record = {
            "iteration": iteration,
            "request_id": request_id,
            "selected_ids": selected_ids,
            "selected_indices": chosen.copy(),
            "n_labeled_before": int(len(labeled_indices)),
        }
        emit_checkpoint(pending_record)
        selected_energies = _evaluate_batch(backend, selected_candidates, request_id)

        n_labeled_before = len(labeled_indices)
        labeled_indices = np.concatenate([labeled_indices, chosen])
        y_labeled = np.concatenate([y_labeled, selected_energies])
        label_by_id.update(zip(selected_ids, selected_energies.tolist(), strict=True))
        rounds.append(
            {
                "iteration": iteration,
                "request_id": request_id,
                "status": "completed",
                "selected_ids": selected_ids,
                "selected_indices": chosen.copy(),
                "selected_energies": dict(
                    zip(selected_ids, selected_energies.tolist(), strict=True)
                ),
                "n_labeled_before": int(n_labeled_before),
                "n_labeled_after": int(len(labeled_indices)),
            }
        )
        print(
            f"Campaign {iteration:03d}/{iteration_offset + iterations:03d}: "
            f"selected={','.join(selected_ids)}, n={len(labeled_indices)}"
        )
        emit_checkpoint(None)

    final_model, final_scaler = fit_gpr(features[labeled_indices], y_labeled)
    final_mean, final_std = final_model.predict(
        final_scaler.transform(features),
        return_std=True,
    )
    remaining_indices = np.setdiff1d(all_indices, labeled_indices, assume_unique=False)
    return {
        "complete": True,
        "rounds": rounds,
        "pending_round": None,
        "labeled_indices": labeled_indices.copy(),
        "labeled_energies": dict(label_by_id),
        "remaining_ids": [candidate_ids[index] for index in remaining_indices],
        "final_mean": final_mean,
        "final_std": final_std,
        "settings": {
            "iterations": int(iteration_offset + iterations),
            "batch_size": int(batch_size),
            "kappa": float(kappa),
            "seed": int(seed),
        },
    }


def structure_fingerprint(
    ids: np.ndarray | Sequence[str],
    structure_paths: Sequence[Path],
) -> str:
    """Hash the unlabeled pool without requiring unknown energies."""
    if len(ids) != len(structure_paths):
        raise ValueError("Structure fingerprint inputs must have equal length")
    digest = hashlib.sha256()
    for structure_id, path in zip(ids, structure_paths, strict=True):
        digest.update(str(structure_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def campaign_identity(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return settings that define a trajectory, excluding its extendable target."""
    return {key: value for key, value in settings.items() if key != "iterations"}


def _campaign_payload(
    *,
    ids: np.ndarray,
    features: np.ndarray,
    feature_metadata: dict[str, Any],
    fingerprint: str,
    settings: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "campaign_cache_format_version": CAMPAIGN_CACHE_FORMAT_VERSION,
        "kind": "campaign",
        "ids": ids,
        "features": features,
        "feature_metadata": feature_metadata,
        "source_fingerprint": fingerprint,
        "settings": settings,
        "results": results,
    }


def run_active_learning(
    project_root: Path,
    *,
    output_dir: Path,
    initial_labels: Mapping[str, float],
    backend: EnergyBackend,
    iterations: int = 100,
    batch_size: int = 2,
    kappa: float = 2.5,
    seed: int = 35,
    recompute: bool = False,
) -> dict[str, Any]:
    """Run or resume one production campaign over every unrelaxed structure."""
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, (int, np.integer))
        or iterations < 0
    ):
        raise ValueError(f"iterations must be a non-negative integer; got {iterations}")
    ids, structure_paths, candidates = load_structure_pool(project_root)
    _validate_positive_int("batch_size", batch_size)
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError(f"kappa must be finite and non-negative; got {kappa}")
    normalised_initial_labels = _initial_labels(initial_labels, ids.tolist())
    backend.validate_pool(candidates)
    fingerprint = structure_fingerprint(ids, structure_paths)
    settings = {
        "mode": "campaign",
        "iterations": int(iterations),
        "batch_size": int(batch_size),
        "kappa": float(kappa),
        "seed": int(seed),
        "backend": dict(backend.cache_key()),
        "initial_labels": sorted(normalised_initial_labels.items()),
    }
    identity_settings = campaign_identity(settings)
    campaign_id = hashlib.sha256(
        json.dumps(
            {"source_fingerprint": fingerprint, "settings": identity_settings},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:20]
    cache_path = output_dir / "campaign_results.pkl"
    prior_rounds: list[dict[str, Any]] = []
    resume_pending: dict[str, Any] | None = None
    active_labels = normalised_initial_labels
    features: np.ndarray | None = None
    feature_metadata: dict[str, Any] | None = None

    if cache_path.exists() and not recompute:
        try:
            cached = load_campaign_cache(cache_path)
        except (OSError, EOFError, ValueError, pickle.UnpicklingError) as exc:
            raise ValueError(
                f"Cannot resume campaign cache {cache_path}: {exc}. "
                "Inspect it before explicitly choosing --recompute or a new output directory."
            ) from exc
        else:
            compatible = (
                np.array_equal(cached["ids"], ids)
                and cached["source_fingerprint"] == fingerprint
                and campaign_identity(cached["settings"]) == identity_settings
            )
            cached_rounds = list(cached["results"].get("rounds", []))
            if compatible:
                by_id = {candidate.structure_id: candidate for candidate in candidates}
                for record in cached_rounds:
                    verified = backend.verify(
                        [by_id[key] for key in record["selected_ids"]],
                        request_id=record["request_id"],
                    )
                    if dict(verified) != record["selected_energies"]:
                        raise ValueError(
                            f"Campaign labels disagree with DFT output: {record['request_id']}"
                        )
            cached_exhausted = not cached["results"].get("remaining_ids", [])
            if (
                compatible
                and cached["results"].get("complete")
                and (len(cached_rounds) >= iterations or cached_exhausted)
            ):
                print(f"Loaded completed campaign cache: {cache_path}")
                return cached
            if compatible:
                prior_rounds = cached_rounds
                cached_pending = cached["results"].get("pending_round")
                resume_pending = (
                    dict(cached_pending) if cached_pending is not None else None
                )
                active_labels = dict(cached["results"]["labeled_energies"])
                features = np.asarray(cached["features"], dtype=float)
                feature_metadata = dict(cached["feature_metadata"])
                print(f"Resuming campaign after {len(prior_rounds)} completed rounds")
            else:
                raise ValueError(
                    f"Campaign inputs do not match {cache_path}; use a new output "
                    "directory or explicitly choose --recompute."
                )

    if features is None:
        features, feature_metadata = calculate_features(structure_paths)
    assert feature_metadata is not None

    completed_rounds = len(prior_rounds)
    remaining_rounds = max(0, iterations - completed_rounds)
    if resume_pending is not None and remaining_rounds == 0:
        raise ValueError(
            "Requested iterations would discard a pending round; finish that round first"
        )

    def save_round_checkpoint(partial_results: dict[str, Any]) -> None:
        checkpoint_results = dict(partial_results)
        checkpoint_results["rounds"] = prior_rounds + list(partial_results["rounds"])
        atomic_pickle_dump(
            cache_path,
            _campaign_payload(
                ids=ids,
                features=features,
                feature_metadata=feature_metadata,
                fingerprint=fingerprint,
                settings=settings,
                results=checkpoint_results,
            ),
        )

    results = run_campaign(
        features,
        candidates,
        active_labels,
        backend,
        iterations=remaining_rounds,
        batch_size=batch_size,
        kappa=kappa,
        seed=seed,
        iteration_offset=completed_rounds,
        campaign_id=campaign_id,
        resume_pending=resume_pending,
        on_round=save_round_checkpoint,
    )
    results["rounds"] = prior_rounds + list(results["rounds"])
    results["settings"]["iterations"] = int(iterations)
    payload = _campaign_payload(
        ids=ids,
        features=features,
        feature_metadata=feature_metadata,
        fingerprint=fingerprint,
        settings=settings,
        results=results,
    )
    atomic_pickle_dump(cache_path, payload)
    print(f"Saved campaign cache: {cache_path}")
    return payload


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--initial-labels-csv",
        type=Path,
        required=True,
        help="First-batch ID and E_f_per_atom labels (eV/atom).",
    )
    parser.add_argument("--incar-path", type=Path, required=True)
    parser.add_argument("--potcar-path", type=Path, required=True)
    parser.add_argument("--kpoints-path", type=Path, required=True)
    parser.add_argument(
        "--vasp-command",
        required=True,
        help="VASP command or scheduler wrapper that waits for completion.",
    )
    parser.add_argument("--vasp-work-dir", type=Path)
    parser.add_argument("--vasp-workers", type=int, default=1)
    parser.add_argument("--co-reference-energy", type=float, default=-6.355317795)
    parser.add_argument("--h-reference-energy", type=float, default=-11.029648055)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--kappa", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    if __package__ in {None, ""}:
        from vasp_label_backend import VaspLabelBackend
    else:
        from .vasp_label_backend import VaspLabelBackend

    args = parse_arguments()
    project_root = (
        args.project_root.resolve()
        if args.project_root
        else find_project_root(Path(__file__).resolve())
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / "results" / "active_learning"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = VaspLabelBackend(
        incar_path=args.incar_path,
        potcar_path=args.potcar_path,
        kpoints_path=args.kpoints_path,
        work_dir=args.vasp_work_dir or output_dir / "vasp_calculations",
        vasp_command=args.vasp_command,
        max_workers=args.vasp_workers,
        co_reference_energy=args.co_reference_energy,
        h_reference_energy=args.h_reference_energy,
    )
    payload = run_active_learning(
        project_root,
        output_dir=output_dir,
        initial_labels=load_initial_labels_csv(args.initial_labels_csv.resolve()),
        backend=backend,
        iterations=args.iterations,
        batch_size=args.batch_size,
        kappa=args.kappa,
        seed=args.seed,
        recompute=args.recompute,
    )
    results = payload["results"]
    latest = results["rounds"][-1] if results["rounds"] else {}
    summary = {
        "mode": "campaign",
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "cache_path": str(output_dir / "campaign_results.pkl"),
        "n_structures": len(payload["ids"]),
        "feature_shape": list(payload["features"].shape),
        "completed_rounds": len(results["rounds"]),
        "n_labeled": len(results["labeled_energies"]),
        "n_remaining": len(results["remaining_ids"]),
        "latest_selected_ids": latest.get("selected_ids", []),
        "latest_selected_energies": latest.get("selected_energies", {}),
        "settings": payload["settings"],
    }
    (output_dir / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

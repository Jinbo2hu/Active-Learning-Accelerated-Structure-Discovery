"""Durable checkpoints and state validation for DFT-labeling campaigns."""

from __future__ import annotations

import os
import pickle
import tempfile
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, IO, Iterator

import numpy as np


CAMPAIGN_CACHE_FORMAT_VERSION = 2


@contextmanager
def _atomic_file(path: Path) -> Iterator[IO[bytes]]:
    """Replace a file only after the complete payload is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            yield handle.file
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_pickle_dump(path: Path, payload: Any) -> None:
    """Atomically save a trusted local checkpoint."""
    with _atomic_file(path) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def atomic_json_dump(path: Path, payload: Any) -> None:
    """Atomically save a human-readable calculation record."""
    with _atomic_file(path) as handle:
        handle.write(
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )


def load_campaign_cache(path: Path) -> dict[str, Any]:
    """Load and validate a resumable production-campaign cache."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {
        "campaign_cache_format_version",
        "kind",
        "ids",
        "features",
        "feature_metadata",
        "source_fingerprint",
        "settings",
        "results",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Incomplete campaign cache: {path}")
    if (
        payload["campaign_cache_format_version"] != CAMPAIGN_CACHE_FORMAT_VERSION
        or payload["kind"] != "campaign"
    ):
        raise ValueError(f"Unsupported campaign cache: {path}")
    ids = np.asarray(payload["ids"])
    features = np.asarray(payload["features"])
    if (
        ids.ndim != 1
        or features.ndim != 2
        or features.shape[0] != len(ids)
        or not np.isfinite(features).all()
        or not isinstance(payload["results"], dict)
    ):
        raise ValueError(f"Misaligned arrays in campaign cache: {path}")
    _validate_campaign_results(ids, payload["settings"], payload["results"], path)
    return payload


def _validate_campaign_results(
    ids: np.ndarray,
    settings: dict[str, Any],
    results: dict[str, Any],
    path: Path,
) -> None:
    """Validate state invariants required for safe campaign resumption."""
    required = {
        "complete",
        "rounds",
        "pending_round",
        "labeled_indices",
        "labeled_energies",
        "remaining_ids",
        "settings",
    }
    if not required.issubset(results):
        raise ValueError(f"Incomplete campaign state: {path}")
    all_ids = [str(structure_id) for structure_id in ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError(f"Duplicate IDs in campaign cache: {path}")

    labeled_energies = results["labeled_energies"]
    rounds = results["rounds"]
    remaining_ids = [str(structure_id) for structure_id in results["remaining_ids"]]
    if not isinstance(labeled_energies, dict) or not isinstance(rounds, list):
        raise ValueError(f"Invalid campaign state collections: {path}")
    labeled_ids = {str(structure_id) for structure_id in labeled_energies}
    if not labeled_ids.issubset(all_ids) or len(set(remaining_ids)) != len(
        remaining_ids
    ):
        raise ValueError(f"Unknown or duplicate campaign IDs: {path}")
    try:
        label_values = np.asarray(list(labeled_energies.values()), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid campaign energy labels: {path}") from exc
    if not np.isfinite(label_values).all():
        raise ValueError(f"Non-finite campaign energy labels: {path}")

    if not isinstance(settings, dict):
        raise ValueError(f"Invalid campaign settings: {path}")
    try:
        initial_energies = {
            str(structure_id): float(energy)
            for structure_id, energy in settings["initial_labels"]
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid initial labels in campaign settings: {path}"
        ) from exc
    if not np.isfinite(np.asarray(list(initial_energies.values()), dtype=float)).all():
        raise ValueError(f"Non-finite initial campaign labels: {path}")

    expected_energies, request_ids = _validate_rounds(
        all_ids, initial_energies, rounds, path
    )
    pending_ids = _validate_pending(all_ids, results, labeled_ids, request_ids, path)

    remaining_set = set(remaining_ids)
    if (
        labeled_ids.intersection(remaining_set)
        or labeled_ids.intersection(pending_ids)
        or remaining_set.intersection(pending_ids)
        or labeled_ids.union(remaining_set, pending_ids) != set(all_ids)
        or set(expected_energies) != labeled_ids
        or any(
            float(labeled_energies[structure_id]) != energy
            for structure_id, energy in expected_energies.items()
        )
    ):
        raise ValueError(f"Invalid campaign ID partition: {path}")
    labeled_indices = np.asarray(results["labeled_indices"], dtype=int)
    if (
        labeled_indices.ndim != 1
        or len(set(labeled_indices.tolist())) != len(labeled_indices)
        or np.any(labeled_indices < 0)
        or np.any(labeled_indices >= len(all_ids))
        or {all_ids[index] for index in labeled_indices} != labeled_ids
    ):
        raise ValueError(f"Invalid labeled campaign indices: {path}")
    if results["complete"]:
        final_mean = np.asarray(results.get("final_mean"))
        final_std = np.asarray(results.get("final_std"))
        if (
            results["pending_round"] is not None
            or final_mean.shape != (len(all_ids),)
            or final_std.shape != (len(all_ids),)
            or not np.isfinite(final_mean).all()
            or not np.isfinite(final_std).all()
        ):
            raise ValueError(f"Invalid completed campaign predictions: {path}")


def _validate_rounds(
    all_ids: list[str],
    initial_energies: dict[str, float],
    rounds: list[dict[str, Any]],
    path: Path,
) -> tuple[dict[str, float], set[str]]:
    """Validate the completed-round ledger against the original labels."""
    if any(not isinstance(record, dict) for record in rounds):
        raise ValueError(f"Invalid completed campaign round: {path}")
    selected_so_far: set[str] = set(initial_energies)
    request_ids: set[str] = set()
    expected_energies = dict(initial_energies)
    expected_before = len(initial_energies)
    for expected_iteration, record in enumerate(rounds, start=1):
        selected_ids = [str(value) for value in record.get("selected_ids", [])]
        selected_indices = np.asarray(record.get("selected_indices", []), dtype=int)
        selected_energies = record.get("selected_energies", {})
        request_id = str(record.get("request_id", ""))
        if (
            record.get("iteration") != expected_iteration
            or record.get("status") != "completed"
            or not request_id
            or request_id in request_ids
            or not selected_ids
            or len(set(selected_ids)) != len(selected_ids)
            or selected_so_far.intersection(selected_ids)
            or selected_indices.ndim != 1
            or len(selected_indices) != len(selected_ids)
            or np.any(selected_indices < 0)
            or np.any(selected_indices >= len(all_ids))
            or [all_ids[index] for index in selected_indices] != selected_ids
            or not isinstance(selected_energies, dict)
            or set(map(str, selected_energies)) != set(selected_ids)
            or record.get("n_labeled_before") != expected_before
            or record.get("n_labeled_after") != expected_before + len(selected_ids)
        ):
            raise ValueError(f"Invalid completed campaign round: {path}")
        try:
            round_energies = {
                str(structure_id): float(energy)
                for structure_id, energy in selected_energies.items()
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid completed-round energies: {path}") from exc
        if not np.isfinite(
            np.asarray(list(round_energies.values()), dtype=float)
        ).all():
            raise ValueError(f"Non-finite completed-round energies: {path}")
        expected_energies.update(round_energies)
        selected_so_far.update(selected_ids)
        request_ids.add(request_id)
        expected_before += len(selected_ids)

    return expected_energies, request_ids


def _validate_pending(
    all_ids: list[str],
    results: dict[str, Any],
    labeled_ids: set[str],
    request_ids: set[str],
    path: Path,
) -> set[str]:
    """Validate the selected but not yet committed batch."""
    pending = results["pending_round"]
    pending_ids: set[str] = set()
    if pending is not None:
        if not isinstance(pending, dict):
            raise ValueError(f"Invalid pending campaign round: {path}")
        pending_values = [str(value) for value in pending.get("selected_ids", [])]
        pending_indices = np.asarray(pending.get("selected_indices", []), dtype=int)
        pending_request_id = str(pending.get("request_id", ""))
        if (
            results["complete"]
            or pending.get("iteration") != len(results["rounds"]) + 1
            or not pending_request_id
            or pending_request_id in request_ids
            or not pending_values
            or len(set(pending_values)) != len(pending_values)
            or labeled_ids.intersection(pending_values)
            or pending_indices.ndim != 1
            or len(pending_indices) != len(pending_values)
            or np.any(pending_indices < 0)
            or np.any(pending_indices >= len(all_ids))
            or [all_ids[index] for index in pending_indices] != pending_values
            or pending.get("n_labeled_before") != len(labeled_ids)
        ):
            raise ValueError(f"Invalid pending campaign round: {path}")
        pending_ids = set(pending_values)

    return pending_ids

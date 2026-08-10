# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python configuration contract for heterogeneous SRSA vector environments.

The simulator assigns environment replica ``i`` to task ``i % num_tasks``.  The
same mapping is consumed by Isaac Lab's ordered ``MultiUsdFileCfg`` spawner and
by every per-task runtime tensor, so geometry and metadata cannot silently
drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


TASK_VECTOR_KEYS = ("task_vec_6", "task_vec", "axial_task_vec_6", "axial_task_vec")
TASK_VECTOR_DIM = 6


@dataclass(frozen=True)
class MultitaskAssemblyEntry:
    """One immutable assembly/task entry shared by spawning and M0 inputs."""

    assembly_id: str
    task_vec_6: tuple[float, ...] | None = None
    insertion_depth: float | None = None
    success_pos_tol: float | None = None


@dataclass(frozen=True)
class MultitaskAssemblySpec:
    """Validated ordered task list for one heterogeneous vector environment."""

    entries: tuple[MultitaskAssemblyEntry, ...]
    source: str

    @property
    def assembly_ids(self) -> tuple[str, ...]:
        return tuple(entry.assembly_id for entry in self.entries)

    @property
    def task_vectors_complete(self) -> bool:
        return all(entry.task_vec_6 is not None for entry in self.entries)


def normalize_assembly_id(value: Any) -> str:
    """Normalize AutoMate assembly IDs to their canonical five-digit form."""

    text = str(value).strip().strip("'\"")
    if not text or not text.isdigit():
        raise ValueError(f"Assembly id must contain only digits, got {value!r}.")
    if len(text) > 5:
        raise ValueError(f"Assembly id must fit in five digits, got {value!r}.")
    return text.zfill(5)


def parse_assembly_ids(raw: str | Iterable[Any] | None) -> tuple[str, ...]:
    """Parse a comma/space/semicolon separated ID list and reject duplicates."""

    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [item for item in re.split(r"[,;\s]+", raw.strip()) if item]
    else:
        values = list(raw)
    normalized = tuple(normalize_assembly_id(value) for value in values)
    duplicates = sorted({value for value in normalized if normalized.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate assembly ids are not allowed: {duplicates}.")
    return normalized


def _task_vector(item: Mapping[str, Any]) -> tuple[float, ...] | None:
    raw = next((item[key] for key in TASK_VECTOR_KEYS if item.get(key) is not None), None)
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [value for value in re.split(r"[,;:\s]+", raw.strip()) if value]
    values = tuple(float(value) for value in raw)
    if len(values) != TASK_VECTOR_DIM:
        raise ValueError(f"M0 task vector must have {TASK_VECTOR_DIM} values, got {values}.")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"M0 task vector contains NaN/Inf: {values}.")
    return values


def _optional_finite_float(item: Mapping[str, Any], key: str) -> float | None:
    params = item.get("srsa_params", {})
    raw = item.get(key, params.get(key) if isinstance(params, dict) else None)
    if raw is None and key == "insertion_depth" and isinstance(params, dict):
        raw = params.get("target_insertion_depth")
    if raw is None:
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Manifest field {key!r} must be finite, got {raw!r}.")
    if value < 0.0:
        raise ValueError(f"Manifest field {key!r} must be non-negative, got {raw!r}.")
    return value


def _manifest_items(payload: Any) -> Sequence[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Multitask manifest must be a JSON list or object.")
    for key in ("tasks", "entries", "assemblies"):
        if key in payload:
            items = payload[key]
            if not isinstance(items, list):
                raise ValueError(f"Multitask manifest field {key!r} must be a list.")
            return items
    raise ValueError("Multitask manifest must contain a 'tasks', 'entries', or 'assemblies' list.")


def load_multitask_assembly_spec(
    *,
    assembly_ids: str | Iterable[Any] | None = None,
    manifest_path: str | Path | None = None,
    require_task_vectors: bool = False,
) -> MultitaskAssemblySpec | None:
    """Load an ordered spec from IDs and/or a FACA-style JSON manifest.

    When both inputs are supplied, ``assembly_ids`` selects and orders entries
    from the manifest.  This lets a large FACA manifest be reused for a 2-task
    smoke test without writing a second file.
    """

    selected_ids = parse_assembly_ids(assembly_ids)
    if manifest_path is None or str(manifest_path).strip() == "":
        if not selected_ids:
            return None
        entries = tuple(MultitaskAssemblyEntry(assembly_id=value) for value in selected_ids)
        if require_task_vectors:
            raise ValueError("SRSA_MULTITASK_REQUIRE_TASK_VECS=1 requires SRSA_MULTITASK_MANIFEST.")
        return MultitaskAssemblySpec(entries=entries, source="assembly_ids")

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SRSA multitask manifest not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    entries_by_id: dict[str, MultitaskAssemblyEntry] = {}
    manifest_order: list[str] = []
    for index, raw_item in enumerate(_manifest_items(payload)):
        if not isinstance(raw_item, dict):
            raise ValueError(f"Multitask manifest entry {index} must be an object.")
        if raw_item.get("enabled") is False:
            continue
        if raw_item.get("assembly_id") is None:
            raise ValueError(f"Multitask manifest entry {index} is missing assembly_id.")
        assembly_id = normalize_assembly_id(raw_item["assembly_id"])
        if assembly_id in entries_by_id:
            raise ValueError(f"Duplicate assembly_id={assembly_id} in multitask manifest {path}.")
        entries_by_id[assembly_id] = MultitaskAssemblyEntry(
            assembly_id=assembly_id,
            task_vec_6=_task_vector(raw_item),
            insertion_depth=_optional_finite_float(raw_item, "insertion_depth"),
            success_pos_tol=_optional_finite_float(raw_item, "success_pos_tol"),
        )
        manifest_order.append(assembly_id)

    ordered_ids = selected_ids or tuple(manifest_order)
    missing = [assembly_id for assembly_id in ordered_ids if assembly_id not in entries_by_id]
    if missing:
        raise ValueError(f"Assembly ids missing from multitask manifest {path}: {missing}.")
    entries = tuple(entries_by_id[assembly_id] for assembly_id in ordered_ids)
    if not entries:
        raise ValueError(f"Multitask manifest has no enabled tasks: {path}")
    if require_task_vectors:
        missing_vectors = [entry.assembly_id for entry in entries if entry.task_vec_6 is None]
        if missing_vectors:
            raise ValueError(f"M0 task_vec_6 is missing for assembly ids: {missing_vectors}.")
    return MultitaskAssemblySpec(entries=entries, source=str(path))


def build_env_task_indices(num_envs: int, num_tasks: int) -> tuple[int, ...]:
    """Return the ordered spawner/task mapping for every environment replica."""

    num_envs = int(num_envs)
    num_tasks = int(num_tasks)
    if num_tasks < 1:
        raise ValueError(f"num_tasks must be positive, got {num_tasks}.")
    if num_envs < num_tasks:
        raise ValueError(
            f"num_envs={num_envs} cannot run all num_tasks={num_tasks} simultaneously; "
            "allocate at least one environment replica per task."
        )
    return tuple(index % num_tasks for index in range(num_envs))


def expand_entries_for_envs(
    entries: Sequence[MultitaskAssemblyEntry], num_envs: int
) -> tuple[MultitaskAssemblyEntry, ...]:
    """Expand task entries with the exact ordered MultiUsdFileCfg mapping."""

    indices = build_env_task_indices(num_envs, len(entries))
    return tuple(entries[index] for index in indices)

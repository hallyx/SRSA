# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for resolving and encoding SRSA task parameters."""

from __future__ import annotations

from .task_family_config import (
    BASE_HOLE_DIAMETER,
    BASE_PLUG_DIAMETER,
    get_task_family_name_by_id,
)


TASK_PARAM_TENSOR_FIELD_ORDER = [
    "plug_diameter",
    "hole_diameter",
    "diametral_clearance",
    "radial_clearance",
    "clearance_ratio",
    "success_pos_tol",
    "insertion_depth",
    "plug_scale_xy",
    "hole_scale_xy",
]


def _get_cfg_value(cfg_like, key: str, default):
    if cfg_like is None:
        return default
    if isinstance(cfg_like, dict):
        return cfg_like.get(key, default)
    return getattr(cfg_like, key, default)


def resolve_task_family_config(
    task_family_config: dict[str, dict] | None,
    *,
    task_family_name: str | None,
    task_family_id: int | None,
    default_task_family_name: str | None,
) -> tuple[str | None, dict | None]:
    if not task_family_config:
        return None, None

    if task_family_name and task_family_name in task_family_config:
        return task_family_name, dict(task_family_config[task_family_name])

    if task_family_id is not None:
        resolved_name = get_task_family_name_by_id(int(task_family_id))
        if resolved_name and resolved_name in task_family_config:
            return resolved_name, dict(task_family_config[resolved_name])

    if default_task_family_name and default_task_family_name in task_family_config:
        return default_task_family_name, dict(task_family_config[default_task_family_name])

    first_name = next(iter(task_family_config.keys()))
    return first_name, dict(task_family_config[first_name])


def resolve_effective_task_params(
    *,
    base_task_cfg,
    task_family_name: str | None = None,
    task_family_cfg: dict | None = None,
    runtime_overrides: dict | None = None,
    baseline_insertion_depth: float | None = None,
):
    eps = 1.0e-6
    runtime_overrides = dict(runtime_overrides or {})

    base_plug_diameter = float(
        _get_cfg_value(
            base_task_cfg,
            "_srsa_base_plug_diameter",
            _get_cfg_value(_get_cfg_value(base_task_cfg, "held_asset_cfg", None), "diameter", 0.0),
        )
    )
    base_hole_diameter = float(
        _get_cfg_value(
            base_task_cfg,
            "_srsa_base_hole_diameter",
            _get_cfg_value(_get_cfg_value(base_task_cfg, "fixed_asset_cfg", None), "diameter", 0.0),
        )
    )
    plug_diameter = base_plug_diameter
    hole_diameter = base_hole_diameter
    success_pos_tol = float(_get_cfg_value(base_task_cfg, "close_error_thresh", 0.015))

    family_cfg = dict(task_family_cfg or {})
    if family_cfg:
        plug_diameter = float(family_cfg.get("plug_diameter", plug_diameter))
        hole_diameter = float(family_cfg.get("hole_diameter", hole_diameter))
        success_pos_tol = float(family_cfg.get("success_pos_tol", success_pos_tol))

    if runtime_overrides.get("plug_diameter") is not None:
        plug_diameter = float(runtime_overrides["plug_diameter"])

    if runtime_overrides.get("hole_diameter") is not None:
        hole_diameter = float(runtime_overrides["hole_diameter"])
    elif runtime_overrides.get("clearance") is not None:
        hole_diameter = plug_diameter + float(runtime_overrides["clearance"])
    elif runtime_overrides.get("clearance_ratio") is not None:
        hole_diameter = plug_diameter * (1.0 + float(runtime_overrides["clearance_ratio"]))

    if runtime_overrides.get("success_pos_tol") is not None:
        success_pos_tol = float(runtime_overrides["success_pos_tol"])

    resolved_baseline_depth = float(baseline_insertion_depth or 0.0)
    insertion_depth = resolved_baseline_depth
    if family_cfg and not bool(family_cfg.get("use_baseline_insertion_depth", False)):
        insertion_depth = float(family_cfg.get("insertion_depth", insertion_depth) or insertion_depth)
    if runtime_overrides.get("insertion_depth") is not None:
        insertion_depth = float(runtime_overrides["insertion_depth"])

    diametral_clearance = max(0.0, hole_diameter - plug_diameter)
    radial_clearance = 0.5 * diametral_clearance
    clearance_ratio = diametral_clearance / max(plug_diameter, eps)

    resolved_task_family_name = task_family_name
    if not resolved_task_family_name:
        resolved_task_family_name = "continuous" if runtime_overrides else "baseline"

    family_variant_id = family_cfg.get("variant_id") if family_cfg else None

    return {
        "task_family_name": resolved_task_family_name,
        "task_family_id": int(family_variant_id) if family_variant_id is not None else -1,
        "plug_diameter": plug_diameter,
        "hole_diameter": hole_diameter,
        "clearance": diametral_clearance,
        "diametral_clearance": diametral_clearance,
        "radial_clearance": radial_clearance,
        "clearance_ratio": clearance_ratio,
        "success_pos_tol": success_pos_tol,
        "insertion_depth": insertion_depth,
        "plug_scale_xy": plug_diameter / max(base_plug_diameter or BASE_PLUG_DIAMETER, eps),
        "hole_scale_xy": hole_diameter / max(base_hole_diameter or BASE_HOLE_DIAMETER, eps),
    }


def make_task_param_tensor(effective_params, num_envs, device):
    import torch

    tensor = torch.tensor(
        [float(effective_params[field_name]) for field_name in TASK_PARAM_TENSOR_FIELD_ORDER],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    return tensor.repeat(int(num_envs), 1)

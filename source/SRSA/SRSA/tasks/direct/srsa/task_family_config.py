# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-family presets for the original Franka AutoMate assembly task."""

from __future__ import annotations


BASE_PLUG_DIAMETER = 0.007986
BASE_HOLE_DIAMETER = 0.008100


def _make_fit_config(
    *,
    variant_id: int,
    task_family_name: str,
    hole_diameter: float,
    success_pos_tol: float,
) -> dict[str, float | int | str | bool | None]:
    clearance = max(0.0, float(hole_diameter) - BASE_PLUG_DIAMETER)
    return {
        "variant_id": int(variant_id),
        "task_family_name": task_family_name,
        "plug_diameter": BASE_PLUG_DIAMETER,
        "hole_diameter": float(hole_diameter),
        "clearance": clearance,
        "clearance_ratio": clearance / max(BASE_PLUG_DIAMETER, 1.0e-6),
        "success_pos_tol": float(success_pos_tol),
        "use_baseline_insertion_depth": True,
        "insertion_depth": None,
    }


TASK_FAMILY_CONFIG: dict[str, dict[str, float | int | str | bool | None]] = {
    "baseline": _make_fit_config(
        variant_id=-1,
        task_family_name="baseline",
        hole_diameter=BASE_HOLE_DIAMETER,
        success_pos_tol=0.0150,
    ),
    "normal_fit": _make_fit_config(
        variant_id=1,
        task_family_name="normal_fit",
        hole_diameter=BASE_HOLE_DIAMETER,
        success_pos_tol=0.0150,
    ),
    "loose_fit": _make_fit_config(
        variant_id=0,
        task_family_name="loose_fit",
        hole_diameter=0.008386,
        success_pos_tol=0.0200,
    ),
    "tight_fit": _make_fit_config(
        variant_id=2,
        task_family_name="tight_fit",
        hole_diameter=0.008036,
        success_pos_tol=0.0100,
    ),
}


def get_task_family_name_by_id(variant_id: int) -> str | None:
    for task_family_name, task_cfg in TASK_FAMILY_CONFIG.items():
        if int(task_cfg["variant_id"]) == int(variant_id):
            return task_family_name
    return None

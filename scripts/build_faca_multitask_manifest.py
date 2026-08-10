#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a FACA-compatible M0 manifest for many real AutoMate assemblies."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re


BASE_PLUG_DIAMETER = 0.007986


def normalize_id(value) -> str:
    text = str(value).strip().strip("'\"")
    if not text.isdigit() or len(text) > 5:
        raise ValueError(f"Invalid assembly id: {value!r}")
    return text.zfill(5)


def parse_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    values = [normalize_id(item) for item in re.split(r"[,;\s]+", raw.strip()) if item]
    if len(values) != len(set(values)):
        raise ValueError("--assembly_ids contains duplicates.")
    return values


def positive(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} for assembly_id={row.get('assembly_id')}")
    return max(value, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh_csv", required=True)
    parser.add_argument("--template_fp", required=True)
    parser.add_argument("--template_id", type=int, default=2)
    parser.add_argument("--assembly_ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reference_anchor_assembly_id", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mesh_csv = Path(args.mesh_csv).expanduser().resolve()
    template_path = Path(args.template_fp).expanduser().resolve()
    with mesh_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows_by_id = {normalize_id(row["assembly_id"]): row for row in rows}
    with template_path.open(encoding="utf-8") as handle:
        template_manifest = json.load(handle)

    templates = template_manifest.get("parameter_templates", [])
    template = next(
        (item for item in templates if int(item.get("template_id", -1)) == int(args.template_id)),
        None,
    )
    if template is None:
        raise ValueError(f"template_id={args.template_id} not found in {template_path}")

    assembly_ids = parse_ids(args.assembly_ids) or list(rows_by_id)
    if args.limit is not None:
        assembly_ids = assembly_ids[: int(args.limit)]
    missing = [assembly_id for assembly_id in assembly_ids if assembly_id not in rows_by_id]
    if missing:
        raise ValueError(f"Assembly ids missing from mesh CSV: {missing}")

    mesh_cfg = template_manifest.get("mesh_geometry", {})
    plug_col = mesh_cfg.get("plug_diameter_column", "plug_xy_bbox_max")
    clearance_col = mesh_cfg.get("clearance_base_column", "plug_to_socket_surface_dist_p05")
    clearance_mode = str(mesh_cfg.get("clearance_mode", "auto")).lower()
    depth_col = mesh_cfg.get("depth_base_column", "plug_bbox_z")
    reference_radius_col = mesh_cfg.get("reference_radius_column", "plug_xy_radius_p95_from_centroid")
    reference_depth_col = mesh_cfg.get("reference_depth_column", depth_col)

    reference_row = None
    if args.reference_anchor_assembly_id:
        anchor_id = normalize_id(args.reference_anchor_assembly_id)
        reference_row = rows_by_id.get(anchor_id)
        if reference_row is None:
            raise ValueError(f"Reference anchor {anchor_id} is missing from {mesh_csv}")
    clearance_multiplier = float(template.get("clearance_multiplier", 1.0))
    depth_multiplier = float(template.get("depth_multiplier", 1.0))

    tasks = []
    for task_index, assembly_id in enumerate(assembly_ids):
        row = rows_by_id[assembly_id]
        plug_diameter = positive(row, plug_col)
        raw_clearance = positive(row, clearance_col)
        is_radial = clearance_mode == "radial" or (
            clearance_mode == "auto"
            and ("radial" in clearance_col.lower() or "surface_dist" in clearance_col.lower())
        )
        diametral_clearance_base = 2.0 * raw_clearance if is_radial else raw_clearance
        diametral_clearance = diametral_clearance_base * clearance_multiplier
        radial_clearance = 0.5 * diametral_clearance
        target_depth = positive(row, depth_col) * depth_multiplier

        reference_source = reference_row or row
        reference_radius = max(positive(reference_source, reference_radius_col), 1.0e-8)
        reference_depth = max(positive(reference_source, reference_depth_col), 1.0e-8)
        male_radius = max(0.5 * plug_diameter, 1.0e-8)
        task_vec = [
            float(template.get("task_type_id", 0)),
            math.log(max(plug_diameter / BASE_PLUG_DIAMETER, 1.0e-8)),
            radial_clearance / reference_radius,
            radial_clearance / male_radius,
            target_depth / reference_depth,
            1.0 if bool(template.get("yaw_requirement", False)) else 0.0,
        ]
        tasks.append(
            {
                "task_id": task_index,
                "assembly_id": assembly_id,
                "task_name": f"srsa-{assembly_id}-{template.get('template_name', args.template_id)}",
                "task_vec_6": task_vec,
                "action_dim": 3,
                "srsa_params": {
                    "plug_diameter": plug_diameter,
                    "diametral_clearance": diametral_clearance,
                    "radial_clearance": radial_clearance,
                    "insertion_depth": target_depth,
                    "success_pos_tol": float(template.get("success_pos_tol", 0.015)),
                    "reference_radius": reference_radius,
                    "reference_depth": reference_depth,
                },
            }
        )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "srsa.faca_m0.heterogeneous_env.v1",
        "assignment": "env_index_mod_num_tasks",
        "mesh_csv": str(mesh_csv),
        "template_fp": str(template_path),
        "template_id": int(args.template_id),
        "reference_anchor_assembly_id": (
            normalize_id(args.reference_anchor_assembly_id) if args.reference_anchor_assembly_id else None
        ),
        "tasks": tasks,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(tasks)} tasks to {output}")


if __name__ == "__main__":
    main()

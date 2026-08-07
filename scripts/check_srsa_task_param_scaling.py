#!/usr/bin/env python3
"""Inspect SRSA nominal task-parameter scaling against mesh-derived geometry proxies."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


BASE_PLUG_DIAMETER = 0.007986
BASE_HOLE_DIAMETER = 0.0081
DEFAULT_TEMPLATES = "0.5:0.5;1.0:1.0;2.0:1.5;4.0:2.0"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _split_ids(value: str) -> list[str]:
    ids = []
    for item in value.replace(",", " ").split():
        item = item.strip()
        if item:
            ids.append(item.zfill(5))
    if not ids:
        raise ValueError("At least one assembly id is required.")
    return ids


def _parse_templates(value: str) -> list[tuple[float, float]]:
    templates = []
    normalized = value.strip()
    for char in "()[]{}":
        normalized = normalized.replace(char, "")
    for chunk in normalized.replace(",", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.replace("/", ":").replace("x", ":").replace("X", ":").split(":")]
        parts = [part for part in parts if part]
        if len(parts) != 2:
            raise ValueError(f"Invalid template {chunk!r}; expected CLEARANCE_MULT:DEPTH_MULT.")
        templates.append((float(parts[0]), float(parts[1])))
    if not templates:
        raise ValueError("At least one template is required.")
    return templates


def _read_csv_by_id(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.is_file():
        return {}
    rows_by_id: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            assembly_id = str(row.get("assembly_id", "")).strip()
            if assembly_id:
                rows_by_id.setdefault(assembly_id, []).append(row)
    return rows_by_id


def _float(row: dict[str, str] | None, key: str, default: float | None = None) -> float | None:
    if row is None:
        return default
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _nominal_row(rows: list[dict[str, str]] | None) -> dict[str, str] | None:
    if not rows:
        return None
    for family in ("normal_fit", "baseline"):
        for row in rows:
            if _row_family(row) == family:
                return row
    return rows[0]


def _row_family(row: dict[str, str] | None) -> str:
    if not row:
        return "missing"
    return row.get("family") or row.get("task_family_name") or "n/a"


def _format_mm(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value * 1000.0:.3f} mm"


def _print_mesh_proxy(mesh_row: dict[str, str] | None) -> None:
    if mesh_row is None:
        print("  mesh proxy: missing")
        return
    fields = (
        ("plug_xy_bbox_max", "plug xy bbox max"),
        ("socket_xy_bbox_max", "socket xy bbox max"),
        ("xy_bbox_max_clearance_proxy", "bbox clearance proxy"),
        ("xy_radius_p95_diametral_clearance_proxy", "p95 diameter clearance proxy"),
        ("plug_to_socket_surface_dist_min", "surface dist min"),
        ("plug_to_socket_surface_dist_p05", "surface dist p05"),
    )
    print("  mesh proxy:")
    for key, label in fields:
        print(f"    {label}: {_format_mm(_float(mesh_row, key))}")


def _print_template_scales(
    *,
    templates: list[tuple[float, float]],
    base_plug_diameter: float,
    base_hole_diameter: float,
    clearance_base: float,
    depth_base: float,
    socket_bbox: float | None,
) -> None:
    print("  template scale effect:")
    print("    c_mult d_mult target_hole hole_scale target_depth socket_bbox_if_scaled")
    for clearance_mult, depth_mult in templates:
        target_hole = base_plug_diameter + clearance_base * clearance_mult
        hole_scale = target_hole / max(base_hole_diameter, 1.0e-9)
        target_depth = depth_base * depth_mult
        scaled_bbox = None if socket_bbox is None else socket_bbox * hole_scale
        print(
            "    "
            f"{clearance_mult:>6g} {depth_mult:>6g} "
            f"{_format_mm(target_hole):>11} {hole_scale:>10.6f} "
            f"{_format_mm(target_depth):>12} {_format_mm(scaled_bbox):>21}"
        )


def _print_warnings(mesh_row: dict[str, str] | None, base_plug_diameter: float, base_hole_diameter: float) -> None:
    if mesh_row is None:
        return

    warnings = []
    bbox_clearance = _float(mesh_row, "xy_bbox_max_clearance_proxy")
    p95_clearance = _float(mesh_row, "xy_radius_p95_diametral_clearance_proxy")
    plug_bbox = _float(mesh_row, "plug_xy_bbox_max")
    socket_bbox = _float(mesh_row, "socket_xy_bbox_max")

    if bbox_clearance is not None and bbox_clearance < 0.0:
        warnings.append("plug xy bbox proxy is larger than socket xy bbox proxy")
    if p95_clearance is not None and p95_clearance < 0.0:
        warnings.append("plug p95 radius proxy is larger than socket p95 radius proxy")
    if plug_bbox is not None and plug_bbox > 1.5 * base_plug_diameter:
        warnings.append("plug outer bbox is far larger than the nominal Peg8mm diameter")
    if socket_bbox is not None and abs(socket_bbox - base_hole_diameter) > 0.5 * base_hole_diameter:
        warnings.append("socket outer bbox is not a usable proxy for nominal hole diameter")

    if warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")
        print("    - current geometry scaling is whole-asset XY scaling, not feature-level hole/shaft scaling")


def main() -> int:
    repo = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly_ids", default="00186", help="Comma/space-separated assembly ids to inspect.")
    parser.add_argument(
        "--mesh_csv",
        type=Path,
        default=repo / "outputs" / "mesh_geometry_params" / "srsa_mesh_geometry_params.csv",
    )
    parser.add_argument("--task_csv", type=Path, default=repo / "outputs" / "srsa_task_params_all.csv")
    parser.add_argument("--templates", default=DEFAULT_TEMPLATES)
    parser.add_argument("--base_plug_diameter", type=float, default=BASE_PLUG_DIAMETER)
    parser.add_argument("--base_hole_diameter", type=float, default=BASE_HOLE_DIAMETER)
    parser.add_argument(
        "--clearance_base",
        type=float,
        default=None,
        help="Diametral clearance base in meters. Defaults to the nominal normal/baseline row.",
    )
    parser.add_argument(
        "--depth_base",
        type=float,
        default=None,
        help="Insertion-depth base in meters. Defaults to the nominal normal/baseline row.",
    )
    args = parser.parse_args()

    assembly_ids = _split_ids(args.assembly_ids)
    templates = _parse_templates(args.templates)
    mesh_rows = _read_csv_by_id(args.mesh_csv)
    task_rows = _read_csv_by_id(args.task_csv)

    print(f"[check] mesh_csv={args.mesh_csv}")
    print(f"[check] task_csv={args.task_csv}")
    for assembly_id in assembly_ids:
        mesh_row = _nominal_row(mesh_rows.get(assembly_id))
        task_row = _nominal_row(task_rows.get(assembly_id))
        base_plug = _float(task_row, "plug_diameter", args.base_plug_diameter) or args.base_plug_diameter
        base_hole = _float(task_row, "hole_diameter", args.base_hole_diameter) or args.base_hole_diameter
        clearance_base = args.clearance_base
        if clearance_base is None:
            clearance_base = _float(task_row, "diametral_clearance", max(0.0, base_hole - base_plug))
        depth_base = args.depth_base
        if depth_base is None:
            depth_base = _float(task_row, "insertion_depth", 0.015)

        print(f"\n[{assembly_id}]")
        family = _row_family(task_row)
        print(
            "  nominal task row: "
            f"family={family} plug={_format_mm(base_plug)} hole={_format_mm(base_hole)} "
            f"clearance={_format_mm(clearance_base)} insertion_depth={_format_mm(depth_base)}"
        )
        _print_mesh_proxy(mesh_row)
        _print_warnings(mesh_row, base_plug, base_hole)
        _print_template_scales(
            templates=templates,
            base_plug_diameter=base_plug,
            base_hole_diameter=base_hole,
            clearance_base=float(clearance_base or 0.0),
            depth_base=float(depth_base or 0.0),
            socket_bbox=_float(mesh_row, "socket_xy_bbox_max"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

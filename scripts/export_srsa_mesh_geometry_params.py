#!/usr/bin/env python3

"""Export mesh-derived SRSA geometry proxy parameters by assembly id.

The official SRSA/AutoMate task cfg stores one nominal Peg8mm/Hole8mm diameter
for all assembly ids. This script reads per-id OBJ assets and exports geometry
proxies instead. These values are not official annotated diameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import trimesh


DEFAULT_MESH_ROOTS = [
    Path("data/mesh"),
    Path("/home/gpuserver/hx/github/SRSA_restore/data/mesh"),
]


def _parse_args():
    parser = argparse.ArgumentParser(description="Export mesh-derived SRSA geometry proxy parameters.")
    parser.add_argument(
        "--mesh_root",
        type=Path,
        default=None,
        help="Directory containing per-id plug.obj/socket.obj folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/mesh_geometry_params"),
        help="Directory for generated CSV and figures.",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=2048,
        help="Number of plug surface samples used for plug-to-socket distance proxies.",
    )
    parser.add_argument(
        "--assembly_ids",
        type=str,
        default=None,
        help="Optional comma-separated assembly ids. Defaults to every folder under mesh_root.",
    )
    return parser.parse_args()


def _resolve_mesh_root(mesh_root: Path | None) -> Path:
    candidates = [mesh_root] if mesh_root is not None else DEFAULT_MESH_ROOTS
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.expanduser()
        if candidate.exists() and candidate.is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(f"No mesh root found. Searched: {searched}")


def _numeric_sort_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)


def _resolve_assembly_ids(mesh_root: Path, raw_ids: str | None) -> list[str]:
    if raw_ids:
        assembly_ids = [item.strip() for item in raw_ids.split(",") if item.strip()]
    else:
        assembly_ids = [path.name for path in mesh_root.iterdir() if path.is_dir()]
    assembly_ids = sorted(dict.fromkeys(assembly_ids), key=_numeric_sort_key)
    missing = [
        assembly_id
        for assembly_id in assembly_ids
        if not (mesh_root / assembly_id / "plug.obj").is_file()
        or not (mesh_root / assembly_id / "socket.obj").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing plug.obj/socket.obj for assembly ids: {missing[:20]}")
    return assembly_ids


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh at {path}, got {type(mesh).__name__}")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Empty mesh: {path}")
    return mesh


def _bbox_stats(mesh: trimesh.Trimesh, prefix: str) -> dict[str, float | int]:
    bounds = np.asarray(mesh.bounds, dtype=float)
    mins = bounds[0]
    maxs = bounds[1]
    extents = maxs - mins
    xy = np.asarray(mesh.vertices[:, :2], dtype=float)
    center = xy.mean(axis=0)
    radii = np.linalg.norm(xy - center, axis=1)
    xy_bbox_area = float(extents[0] * extents[1])
    xy_bbox_equiv_diameter = float(2.0 * math.sqrt(max(xy_bbox_area, 0.0) / math.pi))
    return {
        f"{prefix}_vertex_count": int(len(mesh.vertices)),
        f"{prefix}_face_count": int(len(mesh.faces)),
        f"{prefix}_bbox_min_x": float(mins[0]),
        f"{prefix}_bbox_min_y": float(mins[1]),
        f"{prefix}_bbox_min_z": float(mins[2]),
        f"{prefix}_bbox_max_x": float(maxs[0]),
        f"{prefix}_bbox_max_y": float(maxs[1]),
        f"{prefix}_bbox_max_z": float(maxs[2]),
        f"{prefix}_bbox_x": float(extents[0]),
        f"{prefix}_bbox_y": float(extents[1]),
        f"{prefix}_bbox_z": float(extents[2]),
        f"{prefix}_xy_bbox_min": float(min(extents[0], extents[1])),
        f"{prefix}_xy_bbox_max": float(max(extents[0], extents[1])),
        f"{prefix}_xy_bbox_diag": float(math.hypot(extents[0], extents[1])),
        f"{prefix}_xy_bbox_area": xy_bbox_area,
        f"{prefix}_xy_bbox_equiv_diameter": xy_bbox_equiv_diameter,
        f"{prefix}_xy_radius_mean_from_centroid": float(np.mean(radii)),
        f"{prefix}_xy_radius_p95_from_centroid": float(np.quantile(radii, 0.95)),
        f"{prefix}_xy_radius_max_from_centroid": float(np.max(radii)),
    }


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def _surface_distance_stats(
    plug: trimesh.Trimesh,
    socket: trimesh.Trimesh,
    sample_count: int,
    seed: int,
) -> dict[str, float]:
    np.random.seed(seed)
    sampled_points, _ = trimesh.sample.sample_surface_even(plug, int(sample_count))
    if len(sampled_points) == 0:
        sampled_points = np.asarray(plug.vertices, dtype=float)
    closest, distances, _ = trimesh.proximity.closest_point(socket, sampled_points)
    del closest

    signed_distances = trimesh.proximity.signed_distance(socket, sampled_points)
    return {
        "plug_to_socket_surface_dist_min": float(np.min(distances)),
        "plug_to_socket_surface_dist_p01": _safe_quantile(distances, 0.01),
        "plug_to_socket_surface_dist_p05": _safe_quantile(distances, 0.05),
        "plug_to_socket_surface_dist_median": _safe_quantile(distances, 0.50),
        "plug_to_socket_surface_dist_mean": float(np.mean(distances)),
        "plug_to_socket_surface_dist_max": float(np.max(distances)),
        "plug_to_socket_signed_dist_min": float(np.min(signed_distances)),
        "plug_to_socket_signed_dist_p01": _safe_quantile(signed_distances, 0.01),
        "plug_to_socket_signed_dist_p05": _safe_quantile(signed_distances, 0.05),
        "plug_to_socket_signed_dist_median": _safe_quantile(signed_distances, 0.50),
        "plug_to_socket_signed_dist_mean": float(np.mean(signed_distances)),
        "plug_to_socket_signed_dist_max": float(np.max(signed_distances)),
        "plug_to_socket_signed_dist_positive_fraction": float(np.mean(signed_distances > 0.0)),
        "plug_to_socket_distance_sample_count": int(len(sampled_points)),
    }


def _build_row(mesh_root: Path, assembly_id: str, sample_count: int) -> dict[str, float | int | str]:
    plug_path = mesh_root / assembly_id / "plug.obj"
    socket_path = mesh_root / assembly_id / "socket.obj"
    plug = _load_mesh(plug_path)
    socket = _load_mesh(socket_path)

    row: dict[str, float | int | str] = {
        "assembly_id": assembly_id,
        "plug_obj": str(plug_path),
        "socket_obj": str(socket_path),
    }
    row.update(_bbox_stats(plug, "plug"))
    row.update(_bbox_stats(socket, "socket"))

    row["xy_bbox_max_clearance_proxy"] = float(row["socket_xy_bbox_max"]) - float(row["plug_xy_bbox_max"])
    row["xy_bbox_min_clearance_proxy"] = float(row["socket_xy_bbox_min"]) - float(row["plug_xy_bbox_min"])
    row["xy_bbox_equiv_diameter_clearance_proxy"] = float(row["socket_xy_bbox_equiv_diameter"]) - float(
        row["plug_xy_bbox_equiv_diameter"]
    )
    row["xy_radius_p95_radial_clearance_proxy"] = float(row["socket_xy_radius_p95_from_centroid"]) - float(
        row["plug_xy_radius_p95_from_centroid"]
    )
    row["xy_radius_p95_diametral_clearance_proxy"] = 2.0 * float(row["xy_radius_p95_radial_clearance_proxy"])

    seed = int.from_bytes(hashlib.sha256(assembly_id.encode("utf-8")).digest()[:4], byteorder="little")
    row.update(_surface_distance_stats(plug, socket, sample_count, seed))
    return row


def _write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _import_matplotlib(output_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_metric(rows: list[dict[str, float | int | str]], metric: str, output_dir: Path, unit_scale: float = 1000.0):
    plt = _import_matplotlib(output_dir)
    rows = sorted(rows, key=lambda row: _numeric_sort_key(str(row["assembly_id"])))
    assembly_ids = [str(row["assembly_id"]) for row in rows]
    values = [float(row[metric]) * unit_scale for row in rows]
    x_values = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(18, 5.5))
    ax.plot(x_values, values, marker="o", markersize=2.5, linewidth=1.2)
    tick_step = max(1, len(rows) // 24)
    tick_positions = x_values[::tick_step]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([assembly_ids[index] for index in tick_positions], rotation=90, fontsize=8)
    ax.set_xlabel("assembly_id")
    ax.set_ylabel(f"{metric} (mm)")
    ax.set_title(f"SRSA mesh-derived {metric}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}.png", dpi=180)
    plt.close(fig)


def _write_report(rows: list[dict[str, float | int | str]], mesh_root: Path, output_dir: Path) -> None:
    metrics = [
        "xy_bbox_max_clearance_proxy",
        "xy_bbox_equiv_diameter_clearance_proxy",
        "xy_radius_p95_diametral_clearance_proxy",
        "plug_to_socket_surface_dist_min",
        "plug_to_socket_surface_dist_p05",
        "plug_to_socket_signed_dist_median",
    ]
    lines = [
        f"mesh_root: {mesh_root}",
        f"assembly_count: {len(rows)}",
        "units: CSV values are meters; summary below is millimeters.",
        "note: proxy columns are mesh-derived estimates, not official annotated diameters/clearances.",
        "",
        "metric_stats_mm:",
    ]
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float) * 1000.0
        lines.append(
            f"- {metric}: min={np.min(values):.6g}, p05={np.quantile(values, 0.05):.6g}, "
            f"median={np.median(values):.6g}, mean={np.mean(values):.6g}, max={np.max(values):.6g}"
        )
    (output_dir / "mesh_geometry_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    mesh_root = _resolve_mesh_root(args.mesh_root)
    assembly_ids = _resolve_assembly_ids(mesh_root, args.assembly_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, assembly_id in enumerate(assembly_ids, start=1):
        rows.append(_build_row(mesh_root, assembly_id, args.sample_count))
        if index % 10 == 0 or index == len(assembly_ids):
            print(f"[mesh-geometry] processed {index}/{len(assembly_ids)}", flush=True)

    csv_path = args.output_dir / "srsa_mesh_geometry_params.csv"
    _write_csv(rows, csv_path)
    _write_report(rows, mesh_root, args.output_dir)
    for metric in [
        "plug_xy_bbox_max",
        "socket_xy_bbox_max",
        "xy_bbox_max_clearance_proxy",
        "xy_bbox_equiv_diameter_clearance_proxy",
        "xy_radius_p95_diametral_clearance_proxy",
        "plug_to_socket_surface_dist_min",
        "plug_to_socket_surface_dist_p05",
    ]:
        _plot_metric(rows, metric, args.output_dir)

    print(f"[mesh-geometry] mesh_root={mesh_root}", flush=True)
    print(f"[mesh-geometry] csv={csv_path}", flush=True)
    print(f"[mesh-geometry] report={args.output_dir / 'mesh_geometry_report.txt'}", flush=True)


if __name__ == "__main__":
    main()

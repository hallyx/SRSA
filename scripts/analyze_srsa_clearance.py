#!/usr/bin/env python3

"""Analyze SRSA clearance values by assembly id and task family."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


DEFAULT_FAMILY_ORDER = ["tight_fit", "baseline", "normal_fit", "loose_fit"]
METRIC_FIELDS = {
    "diametral_clearance": ("diametral_clearance", 1.0e6, "um"),
    "radial_clearance": ("radial_clearance", 1.0e6, "um"),
    "clearance_ratio": ("clearance_ratio", 100.0, "percent"),
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Analyze SRSA clearance values exported to CSV.")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("outputs/srsa_task_params_all.csv"),
        help="Task parameter CSV to analyze.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/clearance_analysis"),
        help="Directory for generated summaries and figures.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_FIELDS.keys()),
        default="diametral_clearance",
        help="Clearance metric to plot.",
    )
    parser.add_argument(
        "--exclude_baseline",
        action="store_true",
        help="Exclude baseline rows from analysis.",
    )
    return parser.parse_args()


def _numeric_sort_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)


def _family_sort_key(family_name: str):
    if family_name in DEFAULT_FAMILY_ORDER:
        return (0, DEFAULT_FAMILY_ORDER.index(family_name))
    return (1, family_name)


def _load_rows(input_csv: Path, exclude_baseline: bool) -> list[dict[str, object]]:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")

    required_fields = {"assembly_id", "task_family_name", *METRIC_FIELDS.keys()}
    missing_fields = sorted(required_fields.difference(rows[0].keys()))
    if missing_fields:
        raise KeyError(f"Missing required columns in {input_csv}: {missing_fields}")

    parsed_rows: list[dict[str, object]] = []
    for row in rows:
        if exclude_baseline and row["task_family_name"] == "baseline":
            continue
        parsed = {
            "assembly_id": str(row["assembly_id"]),
            "task_family_name": str(row["task_family_name"]),
        }
        for field_name in METRIC_FIELDS:
            parsed[field_name] = float(row[field_name])
        parsed_rows.append(parsed)
    if not parsed_rows:
        raise ValueError("No rows left after filtering.")
    return parsed_rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def _write_summary(rows: list[dict[str, object]], metric: str, factor: float, output_dir: Path) -> None:
    by_family: dict[str, list[float]] = {}
    ids_by_family: dict[str, set[str]] = {}
    for row in rows:
        family = str(row["task_family_name"])
        by_family.setdefault(family, []).append(float(row[metric]))
        ids_by_family.setdefault(family, set()).add(str(row["assembly_id"]))

    summary_path = output_dir / "clearance_summary_by_family.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_family_name",
                "row_count",
                "assembly_count",
                "unique_value_count",
                f"{metric}_min",
                f"{metric}_max",
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_min_plot_unit",
                f"{metric}_max_plot_unit",
                f"{metric}_mean_plot_unit",
                f"{metric}_std_plot_unit",
            ],
        )
        writer.writeheader()
        for family in sorted(by_family.keys(), key=_family_sort_key):
            values = by_family[family]
            writer.writerow(
                {
                    "task_family_name": family,
                    "row_count": len(values),
                    "assembly_count": len(ids_by_family[family]),
                    "unique_value_count": len({round(value, 15) for value in values}),
                    f"{metric}_min": min(values),
                    f"{metric}_max": max(values),
                    f"{metric}_mean": _mean(values),
                    f"{metric}_std": _std(values),
                    f"{metric}_min_plot_unit": min(values) * factor,
                    f"{metric}_max_plot_unit": max(values) * factor,
                    f"{metric}_mean_plot_unit": _mean(values) * factor,
                    f"{metric}_std_plot_unit": _std(values) * factor,
                }
            )


def _write_pivot(rows: list[dict[str, object]], metric: str, factor: float, output_dir: Path) -> tuple[list[str], list[str], dict[str, dict[str, float]]]:
    assembly_ids = sorted({str(row["assembly_id"]) for row in rows}, key=_numeric_sort_key)
    families = sorted({str(row["task_family_name"]) for row in rows}, key=_family_sort_key)
    pivot: dict[str, dict[str, float]] = {assembly_id: {} for assembly_id in assembly_ids}
    for row in rows:
        pivot[str(row["assembly_id"])][str(row["task_family_name"])] = float(row[metric])

    pivot_path = output_dir / "clearance_by_assembly_id.csv"
    with pivot_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["assembly_id", *families, "min_by_id", "max_by_id", "range_by_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for assembly_id in assembly_ids:
            values = [pivot[assembly_id][family] for family in families if family in pivot[assembly_id]]
            writer.writerow(
                {
                    "assembly_id": assembly_id,
                    **{family: pivot[assembly_id].get(family, "") * factor for family in families},
                    "min_by_id": min(values) * factor,
                    "max_by_id": max(values) * factor,
                    "range_by_id": (max(values) - min(values)) * factor,
                }
            )
    return assembly_ids, families, pivot


def _import_matplotlib(output_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_by_id(
    assembly_ids: list[str],
    families: list[str],
    pivot: dict[str, dict[str, float]],
    metric: str,
    factor: float,
    unit: str,
    output_dir: Path,
) -> None:
    plt = _import_matplotlib(output_dir)
    x_values = list(range(len(assembly_ids)))

    fig, ax = plt.subplots(figsize=(18, 6))
    for family in families:
        y_values = [pivot[assembly_id].get(family, math.nan) * factor for assembly_id in assembly_ids]
        ax.plot(x_values, y_values, marker="o", markersize=2.5, linewidth=1.2, label=family)

    tick_step = max(1, len(assembly_ids) // 24)
    tick_positions = x_values[::tick_step]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([assembly_ids[index] for index in tick_positions], rotation=90, fontsize=8)
    ax.set_xlabel("assembly_id")
    ax.set_ylabel(f"{metric} ({unit})")
    ax.set_title(f"SRSA {metric} by assembly id and task family")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=min(4, max(1, len(families))), frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "clearance_by_id_family.png", dpi=180)
    plt.close(fig)


def _plot_heatmap(
    assembly_ids: list[str],
    families: list[str],
    pivot: dict[str, dict[str, float]],
    metric: str,
    factor: float,
    unit: str,
    output_dir: Path,
) -> None:
    plt = _import_matplotlib(output_dir)
    matrix = [[pivot[assembly_id].get(family, math.nan) * factor for assembly_id in assembly_ids] for family in families]

    fig_height = max(3.2, 0.65 * len(families) + 1.8)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    tick_step = max(1, len(assembly_ids) // 28)
    tick_positions = list(range(0, len(assembly_ids), tick_step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([assembly_ids[index] for index in tick_positions], rotation=90, fontsize=8)
    ax.set_yticks(list(range(len(families))))
    ax.set_yticklabels(families)
    ax.set_xlabel("assembly_id")
    ax.set_title(f"SRSA {metric} heatmap ({unit})")
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label(f"{metric} ({unit})")
    fig.tight_layout()
    fig.savefig(output_dir / "clearance_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_family_summary(
    rows: list[dict[str, object]],
    families: list[str],
    metric: str,
    factor: float,
    unit: str,
    output_dir: Path,
) -> None:
    plt = _import_matplotlib(output_dir)
    values_by_family = {
        family: [float(row[metric]) * factor for row in rows if row["task_family_name"] == family] for family in families
    }
    means = [_mean(values_by_family[family]) for family in families]
    mins = [min(values_by_family[family]) for family in families]
    maxs = [max(values_by_family[family]) for family in families]
    lower = [max(0.0, mean - min_value) for mean, min_value in zip(means, mins)]
    upper = [max(0.0, max_value - mean) for mean, max_value in zip(means, maxs)]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(families, means, yerr=[lower, upper], capsize=5, color=["#3b82f6", "#14b8a6", "#6366f1", "#f59e0b"][: len(families)])
    ax.set_ylabel(f"{metric} ({unit})")
    ax.set_title(f"SRSA {metric} summary by task family")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, mean_value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{mean_value:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "clearance_family_summary.png", dpi=180)
    plt.close(fig)


def _write_text_report(
    rows: list[dict[str, object]],
    assembly_ids: list[str],
    families: list[str],
    pivot: dict[str, dict[str, float]],
    metric: str,
    factor: float,
    unit: str,
    output_dir: Path,
) -> None:
    lines = []
    lines.append(f"input_rows: {len(rows)}")
    lines.append(f"assembly_count: {len(assembly_ids)}")
    lines.append(f"families: {', '.join(families)}")
    lines.append(f"metric: {metric} ({unit})")
    lines.append("")
    lines.append("family_stats:")
    for family in families:
        values = [float(row[metric]) * factor for row in rows if row["task_family_name"] == family]
        lines.append(
            f"- {family}: min={min(values):.9g}, max={max(values):.9g}, "
            f"mean={_mean(values):.9g}, unique={len({round(value, 12) for value in values})}"
        )
    per_id_ranges = []
    for assembly_id in assembly_ids:
        values = [pivot[assembly_id][family] for family in families if family in pivot[assembly_id]]
        per_id_ranges.append((max(values) - min(values)) * factor)
    lines.append("")
    lines.append(
        "per_id_family_range: "
        f"min={min(per_id_ranges):.9g}, max={max(per_id_ranges):.9g}, mean={_mean(per_id_ranges):.9g}"
    )
    constant_families = []
    for family in families:
        raw_values = [float(row[metric]) for row in rows if row["task_family_name"] == family]
        if max(raw_values) - min(raw_values) <= 1.0e-15:
            constant_families.append(family)
    if constant_families:
        lines.append(f"constant_across_ids: {', '.join(constant_families)}")
    (output_dir / "clearance_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    metric, factor, unit = METRIC_FIELDS[args.metric]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.input_csv, args.exclude_baseline)
    _write_summary(rows, metric, factor, args.output_dir)
    assembly_ids, families, pivot = _write_pivot(rows, metric, factor, args.output_dir)
    _plot_by_id(assembly_ids, families, pivot, metric, factor, unit, args.output_dir)
    _plot_heatmap(assembly_ids, families, pivot, metric, factor, unit, args.output_dir)
    _plot_family_summary(rows, families, metric, factor, unit, args.output_dir)
    _write_text_report(rows, assembly_ids, families, pivot, metric, factor, unit, args.output_dir)

    print(f"[clearance-analysis] input={args.input_csv}")
    print(f"[clearance-analysis] output_dir={args.output_dir}")
    print(f"[clearance-analysis] rows={len(rows)} assemblies={len(assembly_ids)} families={len(families)}")
    print("[clearance-analysis] figures=clearance_by_id_family.png, clearance_heatmap.png, clearance_family_summary.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Debug SRSA axial task-parameter sampling without launching Isaac Sim."""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import re
import sys
import types


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _load_task_param_utils():
    package_name = "SRSA.tasks.direct.srsa"
    source_dir = _repo_root() / "source" / "SRSA" / "SRSA" / "tasks" / "direct" / "srsa"
    for module_name in ("SRSA", "SRSA.tasks", "SRSA.tasks.direct", package_name):
        sys.modules.setdefault(module_name, types.ModuleType(module_name))
    sys.modules[package_name].__path__ = [str(source_dir)]

    family_spec = importlib.util.spec_from_file_location(
        f"{package_name}.task_family_config", source_dir / "task_family_config.py"
    )
    family_module = importlib.util.module_from_spec(family_spec)
    sys.modules[family_spec.name] = family_module
    family_spec.loader.exec_module(family_module)

    utils_spec = importlib.util.spec_from_file_location(f"{package_name}.task_param_utils", source_dir / "task_param_utils.py")
    utils_module = importlib.util.module_from_spec(utils_spec)
    sys.modules[utils_spec.name] = utils_module
    utils_spec.loader.exec_module(utils_module)
    return utils_module, family_module


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return float(default)
    return float(value)


def _read_optional_float_list_env(name: str, default=None):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.replace(":", ",").replace(";", ",")
    parts = [item.strip() for item in normalized.split(",") if item.strip()]
    return [float(part) for part in parts]


def _read_optional_float_pair_list_env(name: str, default=None):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip()
    for char in "()[]{}":
        normalized = normalized.replace(char, "")
    parts = [item.strip() for item in re.split(r"[,;:/xX\s]+", normalized) if item.strip()]
    if not parts or len(parts) % 2 != 0:
        raise ValueError(f"{name} must contain float pairs, got {value!r}.")
    return [[float(parts[idx]), float(parts[idx + 1])] for idx in range(0, len(parts), 2)]


def _max_abs(tensor) -> float:
    import torch

    if tensor.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(tensor)).item())


def _range(tensor) -> tuple[float, float]:
    return float(tensor.min().item()), float(tensor.max().item())


def _print_range(name: str, tensor) -> None:
    lower, upper = _range(tensor.detach().float().reshape(-1))
    print(f"{name}: min={lower:.10g} max={upper:.10g}")


def _check(name: str, ok: bool, detail: str) -> bool:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug SRSA axial task-parameter sampler.")
    parser.add_argument("--samples", type=int, default=20000, help="Number of samples to draw.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for sampling.")
    parser.add_argument("--plug-diameter", type=float, default=None, help="Base plug diameter in meters.")
    parser.add_argument("--hole-diameter", type=float, default=None, help="Base hole diameter in meters.")
    parser.add_argument("--success-pos-tol", type=float, default=0.015, help="Success tolerance in meters.")
    parser.add_argument(
        "--require-all-templates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when a configured joint template is not sampled.",
    )
    args = parser.parse_args()

    import torch

    task_utils, family_config = _load_task_param_utils()

    base_plug_diameter = float(args.plug_diameter or family_config.BASE_PLUG_DIAMETER)
    base_hole_diameter = float(args.hole_diameter or family_config.BASE_HOLE_DIAMETER)
    clearance_base = _read_float_env("SRSA_AXIAL_CLEARANCE_BASE", max(0.0, base_hole_diameter - base_plug_diameter))
    depth_base = _read_float_env("SRSA_AXIAL_DEPTH_BASE", 0.015)
    clearance_jitter = _read_float_env("SRSA_AXIAL_CLEARANCE_JITTER_RATIO", 0.0)
    depth_jitter = _read_float_env("SRSA_AXIAL_DEPTH_JITTER_RATIO", 0.0)

    template_pairs = _read_optional_float_pair_list_env("SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATE_MULTIPLIERS")
    template_pairs = _read_optional_float_pair_list_env("SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATES", template_pairs)
    template_weights = _read_optional_float_list_env("SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATE_WEIGHTS")
    clearance_anchors = _read_optional_float_list_env("SRSA_AXIAL_CLEARANCE_ANCHOR_MULTIPLIERS")
    clearance_anchors = _read_optional_float_list_env("SRSA_AXIAL_CLEARANCE_ANCHORS", clearance_anchors)
    depth_anchors = _read_optional_float_list_env("SRSA_AXIAL_DEPTH_ANCHOR_MULTIPLIERS")
    depth_anchors = _read_optional_float_list_env("SRSA_AXIAL_DEPTH_ANCHORS", depth_anchors)

    cfg = task_utils.AxialTaskParamSamplerCfg(
        fixed_plug_scale=_read_bool_env("SRSA_AXIAL_FIXED_PLUG_SCALE", False),
        clearance_base=clearance_base,
        clearance_anchor_multipliers=clearance_anchors,
        clearance_anchor_jitter_ratio=clearance_jitter,
        depth_base=depth_base,
        depth_anchor_multipliers=depth_anchors,
        depth_anchor_jitter_ratio=depth_jitter,
        clearance_depth_template_multipliers=template_pairs,
        clearance_depth_template_weights=template_weights,
    )
    base_task_cfg = {
        "held_asset_cfg": {"diameter": base_plug_diameter},
        "fixed_asset_cfg": {"diameter": base_hole_diameter},
        "held_asset_init_pos_noise": [0.0, 0.0, 0.0],
    }
    effective_params = {
        "task_family_id": -1,
        "plug_scale_xy": 1.0,
        "diametral_clearance": clearance_base,
        "insertion_depth": depth_base,
        "success_pos_tol": float(args.success_pos_tol),
    }
    sampler = task_utils.AxialTaskParamSampler(
        cfg=cfg,
        base_task_cfg=base_task_cfg,
        effective_params=effective_params,
        baseline_insertion_depth=depth_base,
    )
    params = sampler.sample(args.samples, args.device)

    print("[debug] parsed config")
    print(f"  fixed_plug_scale={cfg.fixed_plug_scale}")
    print(f"  clearance_base={clearance_base:.10g} depth_base={depth_base:.10g}")
    print(f"  clearance_jitter={clearance_jitter:.6g} depth_jitter={depth_jitter:.6g}")
    print(f"  joint_templates={template_pairs}")
    print(f"  clearance_anchors={clearance_anchors} depth_anchors={depth_anchors}")

    print("[debug] sampled ranges")
    for key in (
        "plug_scale_xy",
        "hole_scale_xy",
        "diametral_clearance",
        "clearance_multiplier",
        "insertion_depth",
        "depth_multiplier",
        "clearance_depth_template_id",
    ):
        if key in params:
            _print_range(key, params[key])

    ok = True
    tolerance = 1.0e-7
    ok &= _check(
        "hole geometry",
        _max_abs((params["hole_diameter"] - params["plug_diameter"]) - params["diametral_clearance"]) < tolerance,
        f"max_error={_max_abs((params['hole_diameter'] - params['plug_diameter']) - params['diametral_clearance']):.3e}",
    )
    ok &= _check(
        "clearance multiplier",
        _max_abs(params["diametral_clearance"] - clearance_base * params["clearance_multiplier"]) < tolerance,
        f"max_error={_max_abs(params['diametral_clearance'] - clearance_base * params['clearance_multiplier']):.3e}",
    )
    ok &= _check(
        "depth multiplier",
        _max_abs(params["insertion_depth"] - depth_base * params["depth_multiplier"]) < tolerance,
        f"max_error={_max_abs(params['insertion_depth'] - depth_base * params['depth_multiplier']):.3e}",
    )
    ok &= _check(
        "task vector consistency",
        _max_abs(params["task_vec"][:, 1] - params["log_scale"]) < tolerance
        and _max_abs(params["task_vec"][:, 2] - params["clearance_abs_norm"]) < tolerance
        and _max_abs(params["task_vec"][:, 3] - params["clearance_rel_norm"]) < tolerance
        and _max_abs(params["task_vec"][:, 4] - params["depth_abs_norm"]) < tolerance,
        "task_vec columns match sampled scalar fields",
    )

    if cfg.fixed_plug_scale:
        scale_min, scale_max = _range(params["plug_scale_xy"])
        ok &= _check(
            "fixed plug scale",
            abs(scale_min - 1.0) < tolerance and abs(scale_max - 1.0) < tolerance,
            f"plug_scale_range=({scale_min:.10g}, {scale_max:.10g})",
        )

    if template_pairs is not None:
        template_ids = params["clearance_depth_template_id"].to(dtype=torch.long)
        counts = torch.bincount(template_ids.clamp_min(0), minlength=len(template_pairs)).detach().cpu().tolist()
        print(f"[debug] template_counts={counts}")
        if args.require_all_templates:
            ok &= _check("template coverage", all(count > 0 for count in counts), f"counts={counts}")
        for idx, (clearance_mult, depth_mult) in enumerate(template_pairs):
            mask = template_ids == idx
            if not bool(mask.any()):
                continue
            clearance_ratio = params["clearance_multiplier"][mask] / float(clearance_mult)
            depth_ratio = params["depth_multiplier"][mask] / float(depth_mult)
            c_min, c_max = _range(clearance_ratio)
            d_min, d_max = _range(depth_ratio)
            ok &= _check(
                f"template {idx} clearance jitter",
                c_min >= 1.0 - clearance_jitter - 1.0e-5 and c_max <= 1.0 + clearance_jitter + 1.0e-5,
                f"base={clearance_mult:g} jitter_ratio_range=({c_min:.6g}, {c_max:.6g})",
            )
            ok &= _check(
                f"template {idx} depth jitter",
                d_min >= 1.0 - depth_jitter - 1.0e-5 and d_max <= 1.0 + depth_jitter + 1.0e-5,
                f"base={depth_mult:g} jitter_ratio_range=({d_min:.6g}, {d_max:.6g})",
            )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

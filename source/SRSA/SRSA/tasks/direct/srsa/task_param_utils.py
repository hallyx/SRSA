# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for resolving and encoding SRSA task parameters."""

from __future__ import annotations

from dataclasses import dataclass

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

AXIAL_TASK_VEC_FIELD_ORDER = [
    "task_type_id_float",
    "log_scale",
    "clearance_abs_norm",
    "clearance_rel_norm",
    "depth_abs_norm",
    "yaw_requirement_float",
]

NEWT_STATE_DIM = 128
NEWT_ACTION_DIM = 16
SRSA_ACTION_DIM = 6


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

    columns = []
    for field_name in TASK_PARAM_TENSOR_FIELD_ORDER:
        value = effective_params[field_name]
        if isinstance(value, torch.Tensor):
            field = value.to(device=device, dtype=torch.float32).reshape(-1)
            if field.numel() == 1:
                field = field.repeat(int(num_envs))
        else:
            field = torch.full((int(num_envs),), float(value), dtype=torch.float32, device=device)
        columns.append(field[: int(num_envs)])
    return torch.stack(columns, dim=-1)


@dataclass
class AxialTaskParamSamplerCfg:
    enabled: bool = True
    task_type_id: int = 0
    scale_range: list[float] | tuple[float, float] | None = None
    fixed_plug_scale: bool = False
    clearance_range: list[float] | tuple[float, float] | None = None
    clearance_ratio_range: list[float] | tuple[float, float] | None = None
    clearance_base: float | None = None
    clearance_anchor_multipliers: list[float] | tuple[float, ...] | None = None
    clearance_anchor_jitter_ratio: float = 0.0
    clearance_anchor_weights: list[float] | tuple[float, ...] | None = None
    target_depth_range: list[float] | tuple[float, float] | None = None
    depth_base: float | None = None
    depth_anchor_multipliers: list[float] | tuple[float, ...] | None = None
    depth_anchor_jitter_ratio: float = 0.0
    depth_anchor_weights: list[float] | tuple[float, ...] | None = None
    clearance_depth_template_multipliers: list[list[float]] | tuple[tuple[float, float], ...] | None = None
    clearance_depth_template_weights: list[float] | tuple[float, ...] | None = None
    init_error_xy_range: list[float] | tuple[float, float] | None = None
    init_error_z_range: list[float] | tuple[float, float] | None = None
    init_error_yaw_range: list[float] | tuple[float, float] | None = None
    visual_noise_xy_range: list[float] | tuple[float, float] | None = None
    visual_noise_z_range: list[float] | tuple[float, float] | None = None
    yaw_requirement: bool = False
    reference_radius: float = BASE_PLUG_DIAMETER * 0.5
    reference_depth: float = 0.015


def _range_or_fixed(bounds, fixed_value: float) -> tuple[float, float]:
    if bounds is None:
        return float(fixed_value), float(fixed_value)
    lower, upper = float(bounds[0]), float(bounds[1])
    if upper < lower:
        lower, upper = upper, lower
    return lower, upper


class AxialTaskParamSampler:
    """Reset-time sampler for axial mating parameters.

    The sampled task vector is intentionally param-only: it excludes assembly id,
    task id, initial pose error, and visual noise.
    """

    def __init__(
        self,
        *,
        cfg: AxialTaskParamSamplerCfg,
        base_task_cfg,
        effective_params: dict,
        baseline_insertion_depth: float,
        vision_noise_xy_std: float = 0.0,
        vision_noise_z_std: float = 0.0,
    ):
        self.cfg = cfg
        self.base_task_cfg = base_task_cfg
        self.effective_params = dict(effective_params)
        self.baseline_insertion_depth = float(baseline_insertion_depth)
        self.vision_noise_xy_std = float(vision_noise_xy_std)
        self.vision_noise_z_std = float(vision_noise_z_std)

        self.base_plug_diameter = float(
            _get_cfg_value(
                base_task_cfg,
                "_srsa_base_plug_diameter",
                _get_cfg_value(_get_cfg_value(base_task_cfg, "held_asset_cfg", None), "diameter", BASE_PLUG_DIAMETER),
            )
        )
        self.base_hole_diameter = float(
            _get_cfg_value(
                base_task_cfg,
                "_srsa_base_hole_diameter",
                _get_cfg_value(_get_cfg_value(base_task_cfg, "fixed_asset_cfg", None), "diameter", BASE_HOLE_DIAMETER),
            )
        )
        self.default_scale = float(self.effective_params.get("plug_scale_xy", 1.0))
        self.default_clearance = float(self.effective_params.get("diametral_clearance", 0.0))
        self.default_depth = float(self.effective_params.get("insertion_depth", self.baseline_insertion_depth))
        self.default_success_pos_tol = float(self.effective_params.get("success_pos_tol", 0.015))

    @staticmethod
    def _uniform(bounds: tuple[float, float], shape, device):
        import torch

        lower, upper = bounds
        if abs(upper - lower) <= 1.0e-12:
            return torch.full(shape, float(lower), dtype=torch.float32, device=device)
        return torch.rand(shape, dtype=torch.float32, device=device) * (upper - lower) + lower

    @classmethod
    def _signed_uniform(cls, bounds, shape, device, fixed_value: float = 0.0):
        import torch

        lower, upper = _range_or_fixed(bounds, fixed_value)
        if lower < 0.0:
            return cls._uniform((lower, upper), shape, device)
        magnitude = cls._uniform((lower, upper), shape, device)
        signs = torch.where(
            torch.rand(shape, dtype=torch.float32, device=device) < 0.5,
            -torch.ones(shape, dtype=torch.float32, device=device),
            torch.ones(shape, dtype=torch.float32, device=device),
        )
        return magnitude * signs

    @staticmethod
    def _float_tensor(values, device, *, name: str):
        import torch

        tensor = torch.as_tensor([float(value) for value in values], dtype=torch.float32, device=device).reshape(-1)
        if tensor.numel() == 0:
            raise ValueError(f"{name} must contain at least one value.")
        return tensor

    @classmethod
    def _sample_anchor_multipliers(
        cls,
        *,
        anchor_multipliers,
        jitter_ratio: float,
        num_samples: int,
        device,
        anchor_weights=None,
        name: str,
    ):
        import torch

        anchors = cls._float_tensor(anchor_multipliers, device, name=name)
        if torch.any(anchors <= 0.0):
            raise ValueError(f"{name} multipliers must all be positive.")

        if anchor_weights is None:
            anchor_ids = torch.randint(0, anchors.numel(), (int(num_samples),), device=device)
        else:
            weights = cls._float_tensor(anchor_weights, device, name=f"{name} weights")
            if weights.numel() != anchors.numel():
                raise ValueError(
                    f"{name} weights must have the same length as {name} multipliers: "
                    f"{weights.numel()} != {anchors.numel()}."
                )
            if torch.any(weights < 0.0) or float(weights.sum().item()) <= 0.0:
                raise ValueError(f"{name} weights must be non-negative and sum to a positive value.")
            anchor_ids = torch.multinomial(weights / weights.sum(), int(num_samples), replacement=True)

        multipliers = anchors[anchor_ids]
        jitter = max(0.0, float(jitter_ratio or 0.0))
        if jitter > 0.0:
            jitter_scale = cls._uniform((max(1.0e-8, 1.0 - jitter), 1.0 + jitter), (int(num_samples),), device)
            multipliers = multipliers * jitter_scale
        return multipliers, anchor_ids.to(dtype=torch.float32)

    @classmethod
    def _sample_clearance_depth_templates(
        cls,
        *,
        template_multipliers,
        clearance_jitter_ratio: float,
        depth_jitter_ratio: float,
        num_samples: int,
        device,
        template_weights=None,
    ):
        import torch

        rows = []
        for idx, pair in enumerate(template_multipliers):
            if len(pair) != 2:
                raise ValueError(
                    "clearance_depth_template_multipliers entries must be pairs "
                    f"(clearance_multiplier, depth_multiplier), got entry {idx}: {pair!r}."
                )
            rows.append([float(pair[0]), float(pair[1])])
        templates = torch.as_tensor(rows, dtype=torch.float32, device=device)
        if templates.numel() == 0:
            raise ValueError("clearance_depth_template_multipliers must contain at least one pair.")
        if torch.any(templates <= 0.0):
            raise ValueError("clearance/depth template multipliers must all be positive.")

        if template_weights is None:
            template_ids = torch.randint(0, templates.shape[0], (int(num_samples),), device=device)
        else:
            weights = cls._float_tensor(template_weights, device, name="clearance_depth_template_weights")
            if weights.numel() != templates.shape[0]:
                raise ValueError(
                    "clearance_depth_template_weights must have the same length as "
                    f"clearance_depth_template_multipliers: {weights.numel()} != {templates.shape[0]}."
                )
            if torch.any(weights < 0.0) or float(weights.sum().item()) <= 0.0:
                raise ValueError("clearance_depth_template_weights must be non-negative and sum to a positive value.")
            template_ids = torch.multinomial(weights / weights.sum(), int(num_samples), replacement=True)

        sampled = templates[template_ids]
        clearance_multiplier = sampled[:, 0]
        depth_multiplier = sampled[:, 1]

        clearance_jitter = max(0.0, float(clearance_jitter_ratio or 0.0))
        if clearance_jitter > 0.0:
            jitter_scale = cls._uniform(
                (max(1.0e-8, 1.0 - clearance_jitter), 1.0 + clearance_jitter),
                (int(num_samples),),
                device,
            )
            clearance_multiplier = clearance_multiplier * jitter_scale

        depth_jitter = max(0.0, float(depth_jitter_ratio or 0.0))
        if depth_jitter > 0.0:
            jitter_scale = cls._uniform(
                (max(1.0e-8, 1.0 - depth_jitter), 1.0 + depth_jitter),
                (int(num_samples),),
                device,
            )
            depth_multiplier = depth_multiplier * jitter_scale

        return clearance_multiplier, depth_multiplier, template_ids.to(dtype=torch.float32)

    def sample(self, num_samples: int, device):
        import torch

        n = int(num_samples)
        eps = 1.0e-8
        if bool(self.cfg.fixed_plug_scale):
            scale = torch.full((n,), self.default_scale, dtype=torch.float32, device=device)
        else:
            scale = self._uniform(_range_or_fixed(self.cfg.scale_range, self.default_scale), (n,), device)
        plug_diameter = self.base_plug_diameter * scale

        clearance_multiplier = torch.ones((n,), dtype=torch.float32, device=device)
        clearance_anchor_id = torch.full((n,), -1.0, dtype=torch.float32, device=device)
        depth_multiplier = torch.ones((n,), dtype=torch.float32, device=device)
        depth_anchor_id = torch.full((n,), -1.0, dtype=torch.float32, device=device)
        clearance_depth_template_id = torch.full((n,), -1.0, dtype=torch.float32, device=device)
        if self.cfg.clearance_depth_template_multipliers is not None:
            clearance_multiplier, depth_multiplier, clearance_depth_template_id = (
                self._sample_clearance_depth_templates(
                    template_multipliers=self.cfg.clearance_depth_template_multipliers,
                    clearance_jitter_ratio=self.cfg.clearance_anchor_jitter_ratio,
                    depth_jitter_ratio=self.cfg.depth_anchor_jitter_ratio,
                    num_samples=n,
                    device=device,
                    template_weights=self.cfg.clearance_depth_template_weights,
                )
            )
            clearance_base = (
                self.default_clearance if self.cfg.clearance_base is None else float(self.cfg.clearance_base)
            )
            diametral_clearance = (
                torch.full((n,), max(0.0, clearance_base), dtype=torch.float32, device=device) * clearance_multiplier
            )
            clearance_ratio = diametral_clearance / plug_diameter.clamp_min(eps)
        elif self.cfg.clearance_anchor_multipliers is not None:
            clearance_multiplier, clearance_anchor_id = self._sample_anchor_multipliers(
                anchor_multipliers=self.cfg.clearance_anchor_multipliers,
                jitter_ratio=self.cfg.clearance_anchor_jitter_ratio,
                num_samples=n,
                device=device,
                anchor_weights=self.cfg.clearance_anchor_weights,
                name="clearance_anchor_multipliers",
            )
            clearance_base = (
                self.default_clearance if self.cfg.clearance_base is None else float(self.cfg.clearance_base)
            )
            diametral_clearance = (
                torch.full((n,), max(0.0, clearance_base), dtype=torch.float32, device=device) * clearance_multiplier
            )
            clearance_ratio = diametral_clearance / plug_diameter.clamp_min(eps)
        elif self.cfg.clearance_ratio_range is not None:
            clearance_ratio = self._uniform(_range_or_fixed(self.cfg.clearance_ratio_range, 0.0), (n,), device)
            diametral_clearance = plug_diameter * clearance_ratio
        else:
            diametral_clearance = self._uniform(
                _range_or_fixed(self.cfg.clearance_range, self.default_clearance), (n,), device
            )
            clearance_ratio = diametral_clearance / plug_diameter.clamp_min(eps)

        diametral_clearance = diametral_clearance.clamp_min(0.0)
        radial_clearance = 0.5 * diametral_clearance
        hole_diameter = plug_diameter + diametral_clearance
        hole_scale_xy = hole_diameter / max(self.base_hole_diameter, eps)

        # Depth changes the target/effective mating length only; XY geometry and clearance stay above.
        if self.cfg.clearance_depth_template_multipliers is not None:
            depth_base = self.default_depth if self.cfg.depth_base is None else float(self.cfg.depth_base)
            target_depth = (
                torch.full((n,), max(0.0, depth_base), dtype=torch.float32, device=device) * depth_multiplier
            )
        elif self.cfg.depth_anchor_multipliers is not None:
            depth_multiplier, depth_anchor_id = self._sample_anchor_multipliers(
                anchor_multipliers=self.cfg.depth_anchor_multipliers,
                jitter_ratio=self.cfg.depth_anchor_jitter_ratio,
                num_samples=n,
                device=device,
                anchor_weights=self.cfg.depth_anchor_weights,
                name="depth_anchor_multipliers",
            )
            depth_base = self.default_depth if self.cfg.depth_base is None else float(self.cfg.depth_base)
            target_depth = (
                torch.full((n,), max(0.0, depth_base), dtype=torch.float32, device=device) * depth_multiplier
            )
        else:
            target_depth = self._uniform(_range_or_fixed(self.cfg.target_depth_range, self.default_depth), (n,), device)
        target_depth = target_depth.clamp_min(0.0)
        success_pos_tol = torch.full((n,), self.default_success_pos_tol, dtype=torch.float32, device=device)

        init_error = torch.zeros((n, 3), dtype=torch.float32, device=device)
        held_asset_noise = _get_cfg_value(self.base_task_cfg, "held_asset_init_pos_noise", [0.0, 0.0, 0.0])
        default_xy = float(max(abs(float(held_asset_noise[0])), abs(float(held_asset_noise[1]))))
        default_z = float(abs(float(held_asset_noise[2])))
        init_error[:, :2] = self._signed_uniform(self.cfg.init_error_xy_range, (n, 2), device, default_xy)
        init_error[:, 2] = self._signed_uniform(self.cfg.init_error_z_range, (n,), device, default_z)
        init_yaw_error = self._signed_uniform(self.cfg.init_error_yaw_range, (n,), device, 0.0)

        visual_noise = torch.zeros((n, 3), dtype=torch.float32, device=device)
        if self.cfg.visual_noise_xy_range is None:
            if self.vision_noise_xy_std > 0.0:
                visual_noise[:, :2] = torch.randn((n, 2), dtype=torch.float32, device=device) * self.vision_noise_xy_std
        else:
            visual_noise[:, :2] = self._signed_uniform(self.cfg.visual_noise_xy_range, (n, 2), device, 0.0)
        if self.cfg.visual_noise_z_range is None:
            if self.vision_noise_z_std > 0.0:
                visual_noise[:, 2] = torch.randn((n,), dtype=torch.float32, device=device) * self.vision_noise_z_std
        else:
            visual_noise[:, 2] = self._signed_uniform(self.cfg.visual_noise_z_range, (n,), device, 0.0)

        male_radius = 0.5 * plug_diameter
        reference_radius = max(float(self.cfg.reference_radius), eps)
        reference_depth = max(float(self.cfg.reference_depth), eps)
        yaw_requirement = torch.full(
            (n,), 1.0 if bool(self.cfg.yaw_requirement) else 0.0, dtype=torch.float32, device=device
        )
        task_type = torch.full((n,), float(self.cfg.task_type_id), dtype=torch.float32, device=device)
        task_vec = torch.stack(
            [
                task_type,
                torch.log(scale.clamp_min(eps)),
                radial_clearance / reference_radius,
                radial_clearance / male_radius.clamp_min(eps),
                target_depth / reference_depth,
                yaw_requirement,
            ],
            dim=-1,
        )

        return {
            "task_family_id": torch.full((n,), int(self.effective_params.get("task_family_id", -1)), device=device),
            "task_type_id_float": task_type,
            "plug_diameter": plug_diameter,
            "hole_diameter": hole_diameter,
            "clearance": diametral_clearance,
            "diametral_clearance": diametral_clearance,
            "radial_clearance": radial_clearance,
            "clearance_ratio": clearance_ratio,
            "clearance_multiplier": clearance_multiplier,
            "clearance_anchor_id": clearance_anchor_id,
            "clearance_depth_template_id": clearance_depth_template_id,
            "success_pos_tol": success_pos_tol,
            "insertion_depth": target_depth,
            "target_insertion_depth": target_depth,
            "depth_multiplier": depth_multiplier,
            "depth_anchor_id": depth_anchor_id,
            "plug_scale_xy": scale,
            "hole_scale_xy": hole_scale_xy,
            "scale_ratio": scale,
            "log_scale": torch.log(scale.clamp_min(eps)),
            "clearance_abs_norm": radial_clearance / reference_radius,
            "clearance_rel_norm": radial_clearance / male_radius.clamp_min(eps),
            "depth_abs_norm": target_depth / reference_depth,
            "yaw_requirement_float": yaw_requirement,
            "task_vec": task_vec,
            "initial_error_pos": init_error,
            "initial_error_yaw": init_yaw_error,
            "visual_noise_local": visual_noise,
        }

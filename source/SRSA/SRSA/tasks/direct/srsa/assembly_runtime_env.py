# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import re

import gymnasium as gym
import numpy as np
import torch
import isaacsim.core.utils.torch as torch_utils
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_conjugate
from isaaclab_tasks.direct.automate.assembly_tasks_cfg import ASSET_DIR

from .task_family_config import TASK_FAMILY_CONFIG
from .task_param_utils import (
    AXIAL_TASK_VEC_FIELD_ORDER,
    NEWT_ACTION_DIM,
    NEWT_STATE_DIM,
    SRSA_ACTION_DIM,
    TASK_PARAM_TENSOR_FIELD_ORDER,
    AxialTaskParamSampler,
    AxialTaskParamSamplerCfg,
    make_task_param_tensor,
    resolve_effective_task_params,
    resolve_task_family_config,
)


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _read_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _read_optional_float_env(name: str, default: float | None = None) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in {"none", "null"}:
        return None
    return float(value)


def _read_optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _read_optional_float_pair_env(name: str, default=None):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.replace(":", ",").replace(";", ",")
    parts = [item.strip() for item in normalized.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must contain exactly two floats, got {value!r}.")
    return [float(parts[0]), float(parts[1])]


def _read_optional_float_list_env(name: str, default=None):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.replace(":", ",").replace(";", ",")
    parts = [item.strip() for item in normalized.split(",") if item.strip()]
    if not parts:
        raise ValueError(f"{name} must contain at least one float, got {value!r}.")
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
        raise ValueError(
            f"{name} must contain float pairs, for example '0.5:0.5;1.0:1.0'. Got {value!r}."
        )
    return [[float(parts[idx]), float(parts[idx + 1])] for idx in range(0, len(parts), 2)]


def _normalize_task_param_obs_mode(mode: str | None) -> str:
    normalized = str(mode or "task_vec").strip().lower().replace("-", "_")
    if normalized in {"task_vec", "newt", "newt_task", "newt_task_vec", "axial", "axial_task_vec"}:
        return "task_vec"
    if normalized in {"legacy", "legacy_9d", "task_param", "task_param_tensor"}:
        return "legacy"
    raise ValueError(
        "SRSA_TASK_PARAM_OBS_MODE must be one of: task_vec, newt, axial, legacy, legacy_9d. "
        f"Got {mode!r}."
    )


def _task_param_obs_dim(mode: str | None) -> int:
    if _normalize_task_param_obs_mode(mode) == "task_vec":
        return len(AXIAL_TASK_VEC_FIELD_ORDER)
    return len(TASK_PARAM_TENSOR_FIELD_ORDER)


def _normalize_srsa_success_metric(metric: str | None) -> str:
    normalized = str(metric or "terminal_process").strip().lower().replace("-", "_")
    aliases = {
        "automate": "official",
        "auto_mate": "official",
        "official_success": "official",
        "current": "current_official",
        "current_success": "current_official",
        "current_official_success": "current_official",
        "process_success": "process",
        "episode_process_success": "episode_process",
        "terminal": "terminal_process",
        "terminal_success": "terminal_process",
        "terminal_process_success": "terminal_process",
        "strict": "terminal_process",
        "strict_success": "terminal_process",
        "dual_success": "dual",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"official", "current_official", "process", "episode_process", "terminal_process", "dual"}:
        raise ValueError(
            "SRSA_SUCCESS_METRIC/SRSA_EVAL_SUCCESS_METRIC must be one of: official, current_official, "
            "process, episode_process, terminal_process, dual."
        )
    return normalized


def _clamped_tolerance(base: torch.Tensor, scale: float, min_value: float | None, max_value: float | None):
    tol = base * float(scale)
    if min_value is not None:
        tol = torch.maximum(tol, torch.full_like(tol, float(min_value)))
    if max_value is not None:
        tol = torch.minimum(tol, torch.full_like(tol, float(max_value)))
    return tol


class AssemblyRuntimeEnvMixin:
    """Runtime task overrides, task parameterization, plus observation-only vision noise."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self._apply_runtime_task_overrides(cfg)
        task_cfg = cfg.tasks[cfg.task_name]
        self.vision_noise_xy_std = _read_float_env(
            "VISION_NOISE_XY_STD",
            float(getattr(task_cfg, "vision_noise_xy_std", 0.0)),
        )
        self.vision_noise_xy_jitter_std = _read_float_env(
            "VISION_NOISE_XY_JITTER_STD",
            float(getattr(task_cfg, "vision_noise_xy_jitter_std", 0.0)),
        )
        self.vision_noise_z_std = _read_float_env(
            "VISION_NOISE_Z_STD",
            float(getattr(task_cfg, "vision_noise_z_std", 0.0)),
        )
        self.vision_noise_z_jitter_std = _read_float_env(
            "VISION_NOISE_Z_JITTER_STD",
            float(getattr(task_cfg, "vision_noise_z_jitter_std", 0.0)),
        )
        super().__init__(cfg, render_mode, **kwargs)
        self._init_post_super_runtime()
        self._configure_runtime_gym_spaces()

    @staticmethod
    def _apply_runtime_task_overrides(cfg) -> None:
        task_cfg = cfg.tasks[cfg.task_name]

        assembly_id = os.environ.get("SRSA_ASSEMBLY_ID", "").strip()
        if assembly_id:
            assembly_dir = f"{ASSET_DIR}/{assembly_id}/"
            task_cfg.assembly_id = assembly_id
            task_cfg.assembly_dir = assembly_dir
            task_cfg.disassembly_path_json = f"{assembly_dir}disassemble_traj.json"

            if hasattr(task_cfg, "fixed_asset") and hasattr(task_cfg.fixed_asset, "spawn"):
                task_cfg.fixed_asset.spawn.usd_path = f"{assembly_dir}{task_cfg.fixed_asset_cfg.usd_path}"
            if hasattr(task_cfg, "held_asset") and hasattr(task_cfg.held_asset, "spawn"):
                task_cfg.held_asset.spawn.usd_path = f"{assembly_dir}{task_cfg.held_asset_cfg.usd_path}"

        task_family_config = dict(getattr(cfg, "task_family_config", TASK_FAMILY_CONFIG))
        active_task_family_name = os.environ.get("SRSA_TASK_FAMILY_NAME", "").strip() or getattr(
            cfg, "active_task_family_name", None
        )
        active_task_family_id = _read_optional_int_env("SRSA_TASK_FAMILY_ID")
        if active_task_family_id is None:
            active_task_family_id = getattr(cfg, "active_task_family_id", None)

        runtime_task_param_overrides = {
            "plug_diameter": _read_optional_float_env("SRSA_PLUG_DIAMETER"),
            "hole_diameter": _read_optional_float_env("SRSA_HOLE_DIAMETER"),
            "clearance": _read_optional_float_env("SRSA_CLEARANCE"),
            "clearance_ratio": _read_optional_float_env("SRSA_CLEARANCE_RATIO"),
            "insertion_depth": _read_optional_float_env("SRSA_INSERTION_DEPTH"),
            "success_pos_tol": _read_optional_float_env("SRSA_SUCCESS_POS_TOL"),
        }
        runtime_task_param_overrides = {
            key: value for key, value in runtime_task_param_overrides.items() if value is not None
        }

        resolved_family_name, resolved_family_cfg = resolve_task_family_config(
            task_family_config,
            task_family_name=active_task_family_name,
            task_family_id=active_task_family_id,
            default_task_family_name=getattr(cfg, "default_task_family_name", None),
        )
        use_task_family = bool(
            active_task_family_name is not None
            or active_task_family_id is not None
            or getattr(cfg, "use_task_family", False)
        ) and resolved_family_cfg is not None

        task_param_obs = _read_bool_env(
            "SRSA_TASK_PARAM_OBS",
            bool(getattr(cfg, "task_param_obs", False)),
        )
        task_param_obs_mode = _normalize_task_param_obs_mode(
            os.environ.get("SRSA_TASK_PARAM_OBS_MODE", getattr(cfg, "task_param_obs_mode", "task_vec"))
        )
        newt_obs = _read_bool_env("SRSA_NEWT_OBS", bool(getattr(cfg, "newt_obs", False)))
        enable_axial_task_param_sampler = _read_bool_env(
            "SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER",
            bool(getattr(cfg, "enable_axial_task_param_sampler", True)),
        )
        cfg.newt_obs = newt_obs
        cfg.newt_state_dim = int(getattr(cfg, "newt_state_dim", NEWT_STATE_DIM))
        cfg.newt_action_dim = int(getattr(cfg, "newt_action_dim", NEWT_ACTION_DIM))
        cfg.newt_task_dim = len(AXIAL_TASK_VEC_FIELD_ORDER)
        cfg.axial_task_type_id = _read_int_env(
            "SRSA_AXIAL_TASK_TYPE_ID", int(getattr(cfg, "axial_task_type_id", 0))
        )
        cfg.axial_scale_range = _read_optional_float_pair_env(
            "SRSA_AXIAL_SCALE_RANGE", getattr(cfg, "axial_scale_range", None)
        )
        cfg.axial_fixed_plug_scale = _read_bool_env(
            "SRSA_AXIAL_FIXED_PLUG_SCALE", bool(getattr(cfg, "axial_fixed_plug_scale", False))
        )
        cfg.axial_clearance_range = _read_optional_float_pair_env(
            "SRSA_AXIAL_CLEARANCE_RANGE", getattr(cfg, "axial_clearance_range", None)
        )
        cfg.axial_clearance_ratio_range = _read_optional_float_pair_env(
            "SRSA_AXIAL_CLEARANCE_RATIO_RANGE", getattr(cfg, "axial_clearance_ratio_range", None)
        )
        axial_template_pairs = _read_optional_float_pair_list_env(
            "SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATE_MULTIPLIERS",
            getattr(cfg, "axial_clearance_depth_template_multipliers", None),
        )
        cfg.axial_clearance_depth_template_multipliers = _read_optional_float_pair_list_env(
            "SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATES", axial_template_pairs
        )
        cfg.axial_clearance_depth_template_weights = _read_optional_float_list_env(
            "SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATE_WEIGHTS",
            getattr(cfg, "axial_clearance_depth_template_weights", None),
        )
        axial_clearance_base = _read_optional_float_env("SRSA_AXIAL_CLEARANCE_BASE")
        cfg.axial_clearance_base = (
            axial_clearance_base if axial_clearance_base is not None else getattr(cfg, "axial_clearance_base", None)
        )
        axial_clearance_anchors = _read_optional_float_list_env(
            "SRSA_AXIAL_CLEARANCE_ANCHOR_MULTIPLIERS",
            getattr(cfg, "axial_clearance_anchor_multipliers", None),
        )
        cfg.axial_clearance_anchor_multipliers = _read_optional_float_list_env(
            "SRSA_AXIAL_CLEARANCE_ANCHORS", axial_clearance_anchors
        )
        cfg.axial_clearance_anchor_jitter_ratio = _read_float_env(
            "SRSA_AXIAL_CLEARANCE_JITTER_RATIO",
            float(getattr(cfg, "axial_clearance_anchor_jitter_ratio", 0.0)),
        )
        cfg.axial_clearance_anchor_weights = _read_optional_float_list_env(
            "SRSA_AXIAL_CLEARANCE_ANCHOR_WEIGHTS", getattr(cfg, "axial_clearance_anchor_weights", None)
        )
        axial_depth_range = _read_optional_float_pair_env(
            "SRSA_AXIAL_DEPTH_RANGE", getattr(cfg, "axial_target_depth_range", None)
        )
        cfg.axial_target_depth_range = _read_optional_float_pair_env(
            "SRSA_AXIAL_TARGET_DEPTH_RANGE", axial_depth_range
        )
        axial_depth_base = _read_optional_float_env("SRSA_AXIAL_DEPTH_BASE")
        cfg.axial_depth_base = (
            axial_depth_base if axial_depth_base is not None else getattr(cfg, "axial_depth_base", None)
        )
        axial_depth_anchors = _read_optional_float_list_env(
            "SRSA_AXIAL_DEPTH_ANCHOR_MULTIPLIERS",
            getattr(cfg, "axial_depth_anchor_multipliers", None),
        )
        cfg.axial_depth_anchor_multipliers = _read_optional_float_list_env(
            "SRSA_AXIAL_DEPTH_ANCHORS", axial_depth_anchors
        )
        cfg.axial_depth_anchor_jitter_ratio = _read_float_env(
            "SRSA_AXIAL_DEPTH_JITTER_RATIO",
            float(getattr(cfg, "axial_depth_anchor_jitter_ratio", 0.0)),
        )
        cfg.axial_depth_anchor_weights = _read_optional_float_list_env(
            "SRSA_AXIAL_DEPTH_ANCHOR_WEIGHTS", getattr(cfg, "axial_depth_anchor_weights", None)
        )
        cfg.axial_init_error_xy_range = _read_optional_float_pair_env(
            "SRSA_AXIAL_INIT_ERROR_XY_RANGE", getattr(cfg, "axial_init_error_xy_range", None)
        )
        cfg.axial_init_error_z_range = _read_optional_float_pair_env(
            "SRSA_AXIAL_INIT_ERROR_Z_RANGE", getattr(cfg, "axial_init_error_z_range", None)
        )
        cfg.axial_init_error_yaw_range = _read_optional_float_pair_env(
            "SRSA_AXIAL_INIT_ERROR_YAW_RANGE", getattr(cfg, "axial_init_error_yaw_range", None)
        )
        cfg.axial_visual_noise_xy_range = _read_optional_float_pair_env(
            "SRSA_AXIAL_VISUAL_NOISE_XY_RANGE", getattr(cfg, "axial_visual_noise_xy_range", None)
        )
        cfg.axial_visual_noise_z_range = _read_optional_float_pair_env(
            "SRSA_AXIAL_VISUAL_NOISE_Z_RANGE", getattr(cfg, "axial_visual_noise_z_range", None)
        )
        cfg.axial_yaw_requirement = _read_bool_env(
            "SRSA_AXIAL_YAW_REQUIREMENT", bool(getattr(cfg, "axial_yaw_requirement", False))
        )
        cfg.axial_reference_radius = _read_float_env(
            "SRSA_AXIAL_REFERENCE_RADIUS",
            float(getattr(cfg, "axial_reference_radius", 0.5 * 0.007986)),
        )
        cfg.axial_reference_depth = _read_float_env(
            "SRSA_AXIAL_REFERENCE_DEPTH",
            float(getattr(cfg, "axial_reference_depth", 0.015)),
        )
        success_metric = os.environ.get(
            "SRSA_SUCCESS_METRIC",
            os.environ.get("SRSA_EVAL_SUCCESS_METRIC", getattr(cfg, "srsa_eval_success_metric", "terminal_process")),
        )
        cfg.srsa_eval_success_metric = _normalize_srsa_success_metric(success_metric)
        cfg.srsa_process_success_depth_ratio = _read_float_env(
            "SRSA_PROCESS_SUCCESS_DEPTH_RATIO",
            float(getattr(cfg, "srsa_process_success_depth_ratio", 0.85)),
        )
        cfg.srsa_process_success_lateral_tol_scale = _read_float_env(
            "SRSA_PROCESS_SUCCESS_LATERAL_TOL_SCALE",
            float(getattr(cfg, "srsa_process_success_lateral_tol_scale", 2.0)),
        )
        cfg.srsa_process_success_lateral_tol_min = _read_optional_float_env(
            "SRSA_PROCESS_SUCCESS_LATERAL_TOL_MIN",
            getattr(cfg, "srsa_process_success_lateral_tol_min", 0.001),
        )
        cfg.srsa_process_success_lateral_tol_max = _read_optional_float_env(
            "SRSA_PROCESS_SUCCESS_LATERAL_TOL_MAX",
            getattr(cfg, "srsa_process_success_lateral_tol_max", 0.003),
        )
        cfg.srsa_process_success_orientation_tol_rad = _read_float_env(
            "SRSA_PROCESS_SUCCESS_ORIENTATION_TOL_RAD",
            float(getattr(cfg, "srsa_process_success_orientation_tol_rad", 0.0872665)),
        )
        cfg.srsa_process_success_yaw_tol_rad = _read_float_env(
            "SRSA_PROCESS_SUCCESS_YAW_TOL_RAD",
            float(getattr(cfg, "srsa_process_success_yaw_tol_rad", 0.0872665)),
        )
        cfg.srsa_process_success_keypoint_tol_scale = _read_float_env(
            "SRSA_PROCESS_SUCCESS_KEYPOINT_TOL_SCALE",
            float(getattr(cfg, "srsa_process_success_keypoint_tol_scale", 2.0)),
        )
        cfg.srsa_process_success_keypoint_tol_min = _read_optional_float_env(
            "SRSA_PROCESS_SUCCESS_KEYPOINT_TOL_MIN",
            getattr(cfg, "srsa_process_success_keypoint_tol_min", 0.001),
        )
        cfg.srsa_process_success_keypoint_tol_max = _read_optional_float_env(
            "SRSA_PROCESS_SUCCESS_KEYPOINT_TOL_MAX",
            getattr(cfg, "srsa_process_success_keypoint_tol_max", 0.003),
        )
        cfg.srsa_process_success_stable_steps = _read_int_env(
            "SRSA_PROCESS_SUCCESS_STABLE_STEPS",
            int(getattr(cfg, "srsa_process_success_stable_steps", 3)),
        )
        cfg.srsa_process_success_require_official = _read_bool_env(
            "SRSA_PROCESS_SUCCESS_REQUIRE_OFFICIAL",
            bool(getattr(cfg, "srsa_process_success_require_official", False)),
        )
        cfg.srsa_process_success_require_no_jam = _read_bool_env(
            "SRSA_PROCESS_SUCCESS_REQUIRE_NO_JAM",
            bool(getattr(cfg, "srsa_process_success_require_no_jam", True)),
        )
        cfg.enable_axial_task_param_sampler = enable_axial_task_param_sampler
        if cfg.newt_obs:
            cfg.action_space = cfg.newt_action_dim

        use_task_param = bool(use_task_family or runtime_task_param_overrides or enable_axial_task_param_sampler)

        cfg.use_task_family = use_task_family
        cfg.use_task_param = use_task_param
        cfg.task_param_obs = task_param_obs
        cfg.task_param_obs_mode = task_param_obs_mode
        cfg.task_param_obs_dim = _task_param_obs_dim(task_param_obs_mode)
        cfg.task_family_config = task_family_config
        cfg.active_task_family_name = resolved_family_name if use_task_family else None
        cfg.active_task_family_id = (
            int(resolved_family_cfg["variant_id"]) if use_task_family and resolved_family_cfg is not None else None
        )
        cfg.runtime_task_param_overrides = runtime_task_param_overrides

        if cfg.task_param_obs and not getattr(cfg, "_srsa_task_param_obs_augmented", False):
            cfg.observation_space = int(getattr(cfg, "observation_space", 0)) + int(cfg.task_param_obs_dim)
            cfg._srsa_task_param_obs_augmented = True

        if use_task_param or cfg.task_param_obs:
            preview_params = resolve_effective_task_params(
                base_task_cfg=task_cfg,
                task_family_name=cfg.active_task_family_name,
                task_family_cfg=resolved_family_cfg if use_task_family else None,
                runtime_overrides=runtime_task_param_overrides,
                baseline_insertion_depth=None,
            )
            cfg.runtime_task_param_preview = preview_params
            AssemblyRuntimeEnvMixin._apply_task_param_cfg_preview(task_cfg, preview_params)

        task_cfg.if_sbc = _read_bool_env("SRSA_IF_SBC", bool(task_cfg.if_sbc))
        task_cfg.if_logging_eval = _read_bool_env("SRSA_IF_LOGGING_EVAL", bool(task_cfg.if_logging_eval))
        task_cfg.eval_filename = os.environ.get("SRSA_EVAL_FILENAME", task_cfg.eval_filename)
        task_cfg.num_eval_trials = _read_int_env("SRSA_NUM_EVAL_TRIALS", int(task_cfg.num_eval_trials))

        cfg.enable_flange_force_sensor = _read_bool_env(
            "SRSA_ENABLE_FLANGE_FORCE_SENSOR",
            bool(getattr(cfg, "enable_flange_force_sensor", False)),
        )
        cfg.flange_force_sensor_body_name = os.environ.get(
            "SRSA_FLANGE_FORCE_SENSOR_BODY_NAME",
            str(getattr(cfg, "flange_force_sensor_body_name", "panda_hand")),
        )
        cfg.flange_force_sensor_source = os.environ.get(
            "SRSA_FLANGE_FORCE_SENSOR_SOURCE",
            str(getattr(cfg, "flange_force_sensor_source", "held_sensor")),
        )
        cfg.flange_force_sensor_obs_frame = os.environ.get(
            "SRSA_FLANGE_FORCE_SENSOR_OBS_FRAME",
            str(getattr(cfg, "flange_force_sensor_obs_frame", "socket")),
        )
        cfg.flange_force_sensor_obs_scale = _read_float_env(
            "SRSA_FLANGE_FORCE_SENSOR_OBS_SCALE",
            float(getattr(cfg, "flange_force_sensor_obs_scale", 50.0)),
        )
        cfg.flange_force_sensor_force_threshold = _read_float_env(
            "SRSA_FLANGE_FORCE_SENSOR_FORCE_THRESHOLD",
            float(getattr(cfg, "flange_force_sensor_force_threshold", 1.0)),
        )

        sensor_cfg = getattr(cfg, "flange_force_sensor", None)
        if sensor_cfg is not None:
            sensor_cfg.prim_path = f"/World/envs/env_.*/Robot/{cfg.flange_force_sensor_body_name}"
        if cfg.enable_flange_force_sensor:
            if not getattr(cfg, "_srsa_flange_force_obs_augmented", False):
                cfg.observation_space = int(getattr(cfg, "observation_space", 0)) + 3
                cfg._srsa_flange_force_obs_augmented = True
            if hasattr(cfg, "robot") and hasattr(cfg.robot, "spawn") and hasattr(cfg.robot.spawn, "activate_contact_sensors"):
                cfg.robot.spawn.activate_contact_sensors = True

    @staticmethod
    def _apply_task_param_cfg_preview(task_cfg, effective_params: dict) -> None:
        if hasattr(task_cfg, "held_asset_cfg") and not hasattr(task_cfg, "_srsa_base_plug_diameter"):
            task_cfg._srsa_base_plug_diameter = float(task_cfg.held_asset_cfg.diameter)
        if hasattr(task_cfg, "fixed_asset_cfg") and not hasattr(task_cfg, "_srsa_base_hole_diameter"):
            task_cfg._srsa_base_hole_diameter = float(task_cfg.fixed_asset_cfg.diameter)

        plug_scale_xy = float(effective_params["plug_scale_xy"])
        hole_scale_xy = float(effective_params["hole_scale_xy"])

        if hasattr(task_cfg, "held_asset_cfg"):
            task_cfg.held_asset_cfg.diameter = float(effective_params["plug_diameter"])
        if hasattr(task_cfg, "fixed_asset_cfg"):
            task_cfg.fixed_asset_cfg.diameter = float(effective_params["hole_diameter"])
        if hasattr(task_cfg, "close_error_thresh"):
            task_cfg.close_error_thresh = float(effective_params["success_pos_tol"])

        if hasattr(task_cfg, "held_asset") and hasattr(task_cfg.held_asset, "spawn"):
            task_cfg.held_asset.spawn.scale = (plug_scale_xy, plug_scale_xy, 1.0)
        if hasattr(task_cfg, "fixed_asset") and hasattr(task_cfg.fixed_asset, "spawn"):
            task_cfg.fixed_asset.spawn.scale = (hole_scale_xy, hole_scale_xy, 1.0)

    def _init_tensors(self):
        super()._init_tensors()
        self._vision_noise_episode_local = torch.zeros((self.num_envs, 3), device=self.device)
        self._vision_noise_world = torch.zeros((self.num_envs, 3), device=self.device)
        self._vision_noise_cache_step = None
        self._newt_state = torch.zeros(
            (self.num_envs, int(getattr(self.cfg, "newt_state_dim", NEWT_STATE_DIM))), device=self.device
        )
        self._newt_state_mask = torch.zeros_like(self._newt_state, dtype=torch.bool)
        self._newt_action_mask = torch.zeros(
            (self.num_envs, int(getattr(self.cfg, "newt_action_dim", NEWT_ACTION_DIM))),
            dtype=torch.bool,
            device=self.device,
        )
        self._newt_action_mask[:, :SRSA_ACTION_DIM] = True
        self.newt_state = self._newt_state
        self.state_mask = self._newt_state_mask
        self.action_mask = self._newt_action_mask
        self._init_task_param_runtime()
        self._init_flange_force_sensor_runtime()
        self._init_srsa_success_runtime()

    def _init_srsa_success_runtime(self) -> None:
        self._srsa_process_success_streak = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        self._srsa_episode_process_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._srsa_episode_official_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._last_srsa_success_metrics = None

    def _setup_scene(self):
        super()._setup_scene()
        self._flange_force_sensor = None
        self._held_asset_contact_sensor = None
        sensor_cfg = getattr(self.cfg, "flange_force_sensor", None)
        if not getattr(self.cfg, "enable_flange_force_sensor", False):
            return
        if sensor_cfg is not None:
            self._flange_force_sensor = ContactSensor(sensor_cfg)
            self.scene.sensors["flange_force_sensor"] = self._flange_force_sensor
        held_sensor_cfg = getattr(self.cfg, "held_asset_contact_sensor", None)
        if held_sensor_cfg is not None:
            self._held_asset_contact_sensor = ContactSensor(held_sensor_cfg)
            self.scene.sensors["held_asset_contact_sensor"] = self._held_asset_contact_sensor

    def _init_post_super_runtime(self) -> None:
        if hasattr(self, "gripper_open_width"):
            self._srsa_base_gripper_open_width = float(self.gripper_open_width)
            self.current_gripper_open_width = (
                torch.full(
                    (self.num_envs,), self._srsa_base_gripper_open_width, dtype=torch.float32, device=self.device
                )
                * self.current_plug_scale_xy
            )

    def _configure_runtime_gym_spaces(self) -> None:
        if bool(getattr(self.cfg, "newt_obs", False)):
            state_dim = int(getattr(self.cfg, "newt_state_dim", NEWT_STATE_DIM))
            action_dim = int(getattr(self.cfg, "newt_action_dim", NEWT_ACTION_DIM))
            task_dim = len(AXIAL_TASK_VEC_FIELD_ORDER)
            self.cfg.observation_space = {
                "state": state_dim,
                "state_mask": state_dim,
                "task": task_dim,
                "task_id": 1,
                "action_mask": action_dim,
            }
            self.cfg.action_space = action_dim
            self.cfg.state_space = None
            self.single_observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
                    "state_mask": gym.spaces.Box(low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32),
                    "task": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(task_dim,), dtype=np.float32),
                    "task_id": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                    "action_mask": gym.spaces.Box(low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32),
                }
            )
            self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
            self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
            self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
            self.state_space = None
            self.actions = torch.zeros((self.num_envs, action_dim), dtype=torch.float32, device=self.device)
            self.prev_actions = torch.zeros_like(self.actions)
            return

        policy_dim = int(getattr(self.cfg, "observation_space", 0))
        if bool(getattr(self.cfg, "enable_flange_force_sensor", False)):
            policy_dim += 3
        if bool(getattr(self.cfg, "task_param_obs", False)):
            policy_dim += int(
                getattr(
                    self.cfg,
                    "task_param_obs_dim",
                    _task_param_obs_dim(getattr(self.cfg, "task_param_obs_mode", "task_vec")),
                )
            )
        self.cfg.observation_space = policy_dim
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(policy_dim,), dtype=np.float32
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)

    def _init_flange_force_sensor_runtime(self) -> None:
        self.enable_flange_force_sensor = bool(getattr(self.cfg, "enable_flange_force_sensor", False))
        self.flange_force_sensor_source = str(getattr(self.cfg, "flange_force_sensor_source", "held_sensor")).lower()
        self.flange_force_sensor_obs_frame = str(getattr(self.cfg, "flange_force_sensor_obs_frame", "socket")).lower()
        self.flange_force_sensor_obs_scale = float(getattr(self.cfg, "flange_force_sensor_obs_scale", 50.0))
        self.flange_force_sensor_force_threshold = float(
            getattr(self.cfg, "flange_force_sensor_force_threshold", 1.0)
        )
        self.flange_body_contact_force_world = torch.zeros((self.num_envs, 3), device=self.device)
        self.held_sensor_contact_force_world = torch.zeros((self.num_envs, 3), device=self.device)
        self.held_asset_contact_force_world = torch.zeros((self.num_envs, 3), device=self.device)
        self.flange_force_world = torch.zeros((self.num_envs, 3), device=self.device)
        self.flange_force_socket = torch.zeros((self.num_envs, 3), device=self.device)
        self.flange_force_obs = torch.zeros((self.num_envs, 3), device=self.device)
        self.flange_force_norm = torch.zeros((self.num_envs, 1), device=self.device)
        self.flange_force_flag = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)

    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)
        self._update_flange_force_sensor()

    def _update_flange_force_sensor(self) -> None:
        self.flange_body_contact_force_world.zero_()
        self.held_sensor_contact_force_world.zero_()
        self.held_asset_contact_force_world.zero_()
        self.flange_force_world.zero_()
        self.flange_force_socket.zero_()
        self.flange_force_obs.zero_()
        self.flange_force_norm.zero_()
        self.flange_force_flag.zero_()

        if not self.enable_flange_force_sensor:
            return

        if getattr(self, "_flange_force_sensor", None) is not None:
            self.flange_body_contact_force_world[:] = self._coerce_contact_force_tensor(
                self._flange_force_sensor.data.net_forces_w
            )
        if getattr(self, "_held_asset_contact_sensor", None) is not None:
            self.held_sensor_contact_force_world[:] = self._coerce_contact_force_tensor(
                self._held_asset_contact_sensor.data.net_forces_w
            )
        self.held_asset_contact_force_world[:] = self._get_asset_net_contact_force(getattr(self, "_held_asset", None))

        if self.flange_force_sensor_source == "sensor":
            net_forces = self.flange_body_contact_force_world
        elif self.flange_force_sensor_source == "held_sensor":
            net_forces = self.held_sensor_contact_force_world
        elif self.flange_force_sensor_source == "auto":
            body_norm = torch.linalg.norm(self.flange_body_contact_force_world, dim=-1, keepdim=True)
            held_sensor_norm = torch.linalg.norm(self.held_sensor_contact_force_world, dim=-1, keepdim=True)
            net_forces = torch.where(
                body_norm > 1.0e-8,
                self.flange_body_contact_force_world,
                torch.where(
                    held_sensor_norm > 1.0e-8,
                    self.held_sensor_contact_force_world,
                    self.held_asset_contact_force_world,
                ),
            )
        else:
            net_forces = self.held_asset_contact_force_world

        self.flange_force_world[:] = net_forces
        if hasattr(self, "fixed_quat"):
            self.flange_force_socket[:] = quat_apply(quat_conjugate(self.fixed_quat), net_forces)
        else:
            self.flange_force_socket[:] = net_forces
        if self.flange_force_sensor_obs_frame == "world":
            obs_force = self.flange_force_world
        else:
            obs_force = self.flange_force_socket
        self.flange_force_obs[:] = obs_force / max(self.flange_force_sensor_obs_scale, 1.0e-6)
        self.flange_force_norm[:, 0] = torch.linalg.norm(self.flange_force_world, dim=-1)
        self.flange_force_flag[:, 0] = self.flange_force_norm[:, 0] > self.flange_force_sensor_force_threshold

    def _coerce_contact_force_tensor(self, net_forces) -> torch.Tensor:
        if not isinstance(net_forces, torch.Tensor):
            net_forces = torch.as_tensor(net_forces, device=self.device, dtype=torch.float32)
        else:
            net_forces = net_forces.to(device=self.device, dtype=torch.float32)

        if net_forces.ndim == 4:
            net_forces = net_forces[:, -1]
        if net_forces.ndim == 3:
            net_forces = net_forces.sum(dim=1)
        if net_forces.ndim != 2 or net_forces.shape[-1] != 3:
            return torch.zeros((self.num_envs, 3), device=self.device)
        return net_forces

    def _get_asset_net_contact_force(self, asset) -> torch.Tensor:
        if asset is None or not hasattr(asset, "root_physx_view"):
            return torch.zeros((self.num_envs, 3), device=self.device)
        try:
            net_forces = asset.root_physx_view.get_net_contact_forces(dt=self.physics_dt)
        except Exception:
            return torch.zeros((self.num_envs, 3), device=self.device)
        return self._coerce_contact_force_tensor(net_forces)

    def _init_task_param_runtime(self) -> None:
        self.use_task_family = bool(getattr(self.cfg, "use_task_family", False))
        self.use_task_param = bool(getattr(self.cfg, "use_task_param", False))
        self.task_param_obs = bool(getattr(self.cfg, "task_param_obs", False))
        self.task_param_obs_mode = _normalize_task_param_obs_mode(getattr(self.cfg, "task_param_obs_mode", "task_vec"))
        self.enable_axial_task_param_sampler = bool(getattr(self.cfg, "enable_axial_task_param_sampler", True))
        self.enable_task_param = bool(self.use_task_param or self.task_param_obs or self.enable_axial_task_param_sampler)
        self.current_task_param_tensor = None
        self.current_task_params = {}
        self.current_task_param_tensors = {}
        self.current_task_vec = torch.zeros((self.num_envs, len(AXIAL_TASK_VEC_FIELD_ORDER)), device=self.device)
        self.current_task_id = torch.zeros((self.num_envs, 1), dtype=torch.long, device=self.device)
        self.current_initial_error_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.current_initial_error_yaw = torch.zeros((self.num_envs,), device=self.device)
        self.current_geometry_variant_applied = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        self.current_close_error_thresh_tensor = torch.full(
            (self.num_envs,), float(getattr(self.cfg_task, "close_error_thresh", 0.015)), device=self.device
        )
        self.current_insertion_depth_tensor = (
            self.disassembly_dists.clone()
            if hasattr(self, "disassembly_dists")
            else torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        )
        self.current_plug_scale_xy = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.current_hole_scale_xy = torch.ones_like(self.current_plug_scale_xy)
        self.current_gripper_open_width = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._srsa_base_disassembly_dists = (
            self.disassembly_dists.clone()
            if hasattr(self, "disassembly_dists")
            else torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        )
        self.axial_task_param_sampler = None

        if not self.enable_task_param:
            return

        task_family_config = dict(getattr(self.cfg, "task_family_config", TASK_FAMILY_CONFIG))
        resolved_family_name, resolved_family_cfg = resolve_task_family_config(
            task_family_config,
            task_family_name=getattr(self.cfg, "active_task_family_name", None),
            task_family_id=getattr(self.cfg, "active_task_family_id", None),
            default_task_family_name=getattr(self.cfg, "default_task_family_name", None),
        )
        if not self.use_task_family:
            resolved_family_name = None
            resolved_family_cfg = None

        runtime_overrides = dict(getattr(self.cfg, "runtime_task_param_overrides", {}) or {})
        baseline_insertion_depth = None
        if hasattr(self, "disassembly_dists"):
            baseline_insertion_depth = float(self.disassembly_dists[0].item())
        else:
            baseline_insertion_depth = 0.0

        effective_params = resolve_effective_task_params(
            base_task_cfg=self.cfg_task,
            task_family_name=resolved_family_name,
            task_family_cfg=resolved_family_cfg,
            runtime_overrides=runtime_overrides,
            baseline_insertion_depth=baseline_insertion_depth,
        )
        sampler_cfg = AxialTaskParamSamplerCfg(
            enabled=self.enable_axial_task_param_sampler,
            task_type_id=int(getattr(self.cfg, "axial_task_type_id", 0)),
            scale_range=getattr(self.cfg, "axial_scale_range", None),
            fixed_plug_scale=bool(getattr(self.cfg, "axial_fixed_plug_scale", False)),
            clearance_range=getattr(self.cfg, "axial_clearance_range", None),
            clearance_ratio_range=getattr(self.cfg, "axial_clearance_ratio_range", None),
            clearance_base=getattr(self.cfg, "axial_clearance_base", None),
            clearance_anchor_multipliers=getattr(self.cfg, "axial_clearance_anchor_multipliers", None),
            clearance_anchor_jitter_ratio=float(getattr(self.cfg, "axial_clearance_anchor_jitter_ratio", 0.0)),
            clearance_anchor_weights=getattr(self.cfg, "axial_clearance_anchor_weights", None),
            target_depth_range=getattr(self.cfg, "axial_target_depth_range", None),
            depth_base=getattr(self.cfg, "axial_depth_base", None),
            depth_anchor_multipliers=getattr(self.cfg, "axial_depth_anchor_multipliers", None),
            depth_anchor_jitter_ratio=float(getattr(self.cfg, "axial_depth_anchor_jitter_ratio", 0.0)),
            depth_anchor_weights=getattr(self.cfg, "axial_depth_anchor_weights", None),
            clearance_depth_template_multipliers=getattr(
                self.cfg, "axial_clearance_depth_template_multipliers", None
            ),
            clearance_depth_template_weights=getattr(self.cfg, "axial_clearance_depth_template_weights", None),
            init_error_xy_range=getattr(self.cfg, "axial_init_error_xy_range", None),
            init_error_z_range=getattr(self.cfg, "axial_init_error_z_range", None),
            init_error_yaw_range=getattr(self.cfg, "axial_init_error_yaw_range", None),
            visual_noise_xy_range=getattr(self.cfg, "axial_visual_noise_xy_range", None),
            visual_noise_z_range=getattr(self.cfg, "axial_visual_noise_z_range", None),
            yaw_requirement=bool(getattr(self.cfg, "axial_yaw_requirement", False)),
            reference_radius=float(getattr(self.cfg, "axial_reference_radius", 0.5 * 0.007986)),
            reference_depth=float(getattr(self.cfg, "axial_reference_depth", 0.015)),
        )
        self.axial_task_param_sampler = AxialTaskParamSampler(
            cfg=sampler_cfg,
            base_task_cfg=self.cfg_task,
            effective_params=effective_params,
            baseline_insertion_depth=baseline_insertion_depth,
            vision_noise_xy_std=self.vision_noise_xy_std,
            vision_noise_z_std=self.vision_noise_z_std,
        )

        initial_params = self.axial_task_param_sampler.sample(self.num_envs, self.device)
        self._set_current_task_param_tensors(torch.arange(self.num_envs, device=self.device), initial_params)
        self.current_close_error_thresh = float(effective_params["success_pos_tol"])
        self.current_insertion_depth = float(effective_params["insertion_depth"])

        if self.use_task_param:
            if hasattr(self, "disassembly_dists"):
                self.disassembly_dists[:] = self.current_insertion_depth_tensor
            self.cfg_task.close_error_thresh = self.current_close_error_thresh

            if hasattr(self, "curriculum_height_bound") and hasattr(self.cfg_task, "curriculum_freespace_range"):
                self.curriculum_height_bound[:, 1] = self.disassembly_dists + float(
                    self.cfg_task.curriculum_freespace_range
                )

    def _set_current_task_param_tensors(self, env_ids: torch.Tensor, sampled_params: dict) -> None:
        env_ids = self._env_ids_to_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        for key, value in sampled_params.items():
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, device=self.device)
            value = value.to(device=self.device)
            if key not in self.current_task_param_tensors:
                shape = (self.num_envs, *value.shape[1:])
                dtype = value.dtype if value.dtype != torch.int64 else torch.float32
                self.current_task_param_tensors[key] = torch.zeros(shape, dtype=dtype, device=self.device)
            target = self.current_task_param_tensors[key]
            target[env_ids] = value.to(dtype=target.dtype)

        tensor_source = {key: value for key, value in self.current_task_param_tensors.items()}
        self.current_task_param_tensor = make_task_param_tensor(tensor_source, self.num_envs, self.device)
        self.current_task_vec[:] = self.current_task_param_tensors["task_vec"].to(dtype=torch.float32)
        self.current_task_id[:, 0] = self.current_task_param_tensors["task_type_id_float"].round().to(torch.long)
        self.current_close_error_thresh_tensor[:] = self.current_task_param_tensors["success_pos_tol"].to(torch.float32)
        self.current_insertion_depth_tensor[:] = self.current_task_param_tensors["insertion_depth"].to(torch.float32)
        self.current_plug_scale_xy[:] = self.current_task_param_tensors["plug_scale_xy"].to(torch.float32)
        self.current_hole_scale_xy[:] = self.current_task_param_tensors["hole_scale_xy"].to(torch.float32)
        self.current_initial_error_pos[:] = self.current_task_param_tensors["initial_error_pos"].to(torch.float32)
        self.current_initial_error_yaw[:] = self.current_task_param_tensors["initial_error_yaw"].to(torch.float32)
        self._vision_noise_episode_local[:] = self.current_task_param_tensors["visual_noise_local"].to(torch.float32)
        self._reset_vision_noise_cache()

        self.current_task_params = {}
        first_env = int(env_ids[0].item())
        for key, value in self.current_task_param_tensors.items():
            if key in {"initial_error_pos", "visual_noise_local", "task_vec"}:
                self.current_task_params[key] = value[first_env].detach().cpu().tolist()
            else:
                first_value = value[first_env]
                self.current_task_params[key] = (
                    float(first_value.item()) if first_value.numel() == 1 else first_value.detach().cpu().tolist()
                )

    def _update_task_param_extras(self) -> None:
        if not hasattr(self, "extras") or not isinstance(self.extras, dict):
            return
        if not self.current_task_param_tensors:
            return

        task_params = {}
        for key, value in self.current_task_param_tensors.items():
            if key in {"initial_error_pos", "initial_error_yaw", "visual_noise_local", "task_vec"}:
                continue
            task_params[key] = value.detach().clone()
            self.extras[key] = value.detach().clone()
        self.extras["task_params"] = task_params
        self.extras["task_vec"] = self.current_task_vec.detach().clone()
        self.extras["task_id"] = self.current_task_id.detach().clone()
        self.extras["initial_error_pos"] = self.current_initial_error_pos.detach().clone()
        self.extras["initial_error_yaw"] = self.current_initial_error_yaw.detach().clone()
        self.extras["visual_noise_local"] = self._vision_noise_episode_local.detach().clone()
        self.extras["geometry_variant_applied"] = self.current_geometry_variant_applied.detach().clone()

    def _update_flange_force_extras(self) -> None:
        if not hasattr(self, "extras") or not isinstance(self.extras, dict):
            return

        self.extras["flange_force_world"] = self.flange_force_world.detach().clone()
        self.extras["flange_force_socket"] = self.flange_force_socket.detach().clone()
        self.extras["flange_force_norm"] = self.flange_force_norm.detach().clone()
        self.extras["flange_force_flag"] = self.flange_force_flag.detach().clone()
        self.extras["flange_body_contact_force_world"] = self.flange_body_contact_force_world.detach().clone()
        self.extras["held_sensor_contact_force_world"] = self.held_sensor_contact_force_world.detach().clone()
        self.extras["held_asset_contact_force_world"] = self.held_asset_contact_force_world.detach().clone()

    def _held_fixed_delta_socket(self) -> torch.Tensor:
        if not hasattr(self, "held_pos") or not hasattr(self, "fixed_pos") or not hasattr(self, "fixed_quat"):
            return torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        return quat_apply(quat_conjugate(self.fixed_quat), self.held_pos - self.fixed_pos)

    def _compute_current_success(self) -> torch.Tensor:
        if not all(hasattr(self, name) for name in ("held_pos", "fixed_pos", "keypoints_held", "keypoints_fixed")):
            return torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        from isaaclab_tasks.direct.automate import automate_algo_utils as automate_algo

        return automate_algo.check_plug_inserted_in_socket(
            self.held_pos,
            self.fixed_pos,
            self.current_insertion_depth_tensor,
            self.keypoints_held,
            self.keypoints_fixed,
            self.current_close_error_thresh_tensor,
            self.episode_length_buf,
        ).to(dtype=torch.bool)

    def _srsa_env_vector(self, value, *, dtype=torch.float32, default=0.0) -> torch.Tensor:
        if value is None:
            value = default
        try:
            if torch.is_tensor(value):
                tensor = value.to(device=self.device, dtype=dtype)
            else:
                tensor = torch.as_tensor(value, device=self.device, dtype=dtype)
        except (TypeError, ValueError, RuntimeError):
            tensor = torch.as_tensor(default, device=self.device, dtype=dtype)

        if tensor.numel() == 1:
            return tensor.reshape(1).expand(self.num_envs)
        if tensor.shape[0] == self.num_envs:
            return tensor.reshape(self.num_envs, -1)[:, 0]
        if tensor.numel() == self.num_envs:
            return tensor.reshape(self.num_envs)
        return torch.full((self.num_envs,), default, dtype=dtype, device=self.device)

    def _srsa_task_param_vector(self, key: str, *, default=0.0) -> torch.Tensor:
        params = getattr(self, "current_task_param_tensors", None)
        if isinstance(params, dict) and key in params:
            return self._srsa_env_vector(params[key], dtype=torch.float32, default=default)

        params = getattr(self, "current_task_params", None)
        if isinstance(params, dict) and key in params:
            return self._srsa_env_vector(params[key], dtype=torch.float32, default=default)

        return torch.full((self.num_envs,), float(default), dtype=torch.float32, device=self.device)

    def _select_srsa_success(self, metrics: dict) -> torch.Tensor:
        metric = _normalize_srsa_success_metric(
            metrics.get("success_metric", getattr(self.cfg, "srsa_eval_success_metric", "terminal_process"))
        )
        if metric == "official":
            return metrics["official_success"]
        if metric == "current_official":
            return metrics["current_official_success"]
        if metric == "process":
            return metrics["process_success"]
        if metric == "episode_process":
            return metrics["episode_process_success"]
        if metric == "terminal_process":
            return metrics["terminal_process_success"]
        if metric == "dual":
            return metrics["dual_success"]
        raise AssertionError(f"Unhandled SRSA success metric: {metric}")

    def _cache_srsa_success_metrics(self, metrics: dict) -> None:
        self._last_srsa_success_metrics = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in metrics.items()
            if not key.startswith("_")
        }

    def _reset_srsa_success_state(self, env_ids=None) -> None:
        self._last_srsa_success_metrics = None
        if not hasattr(self, "_srsa_process_success_streak"):
            return
        if env_ids is None:
            self._srsa_process_success_streak.zero_()
            self._srsa_episode_process_success.zero_()
            self._srsa_episode_official_success.zero_()
            return
        env_ids = self._env_ids_to_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        self._srsa_process_success_streak[env_ids] = 0
        self._srsa_episode_process_success[env_ids] = False
        self._srsa_episode_official_success[env_ids] = False

    def _compute_srsa_success_metrics(
        self,
        current_official_success: torch.Tensor | None = None,
        *,
        update_state: bool = False,
    ) -> dict:
        zeros = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        false = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        current_official = (
            self._compute_current_success()
            if current_official_success is None
            else self._srsa_env_vector(current_official_success, dtype=torch.bool, default=False)
        )
        episode_official = getattr(self, "_srsa_episode_official_success", false) | current_official

        target_depth = self._srsa_env_vector(
            getattr(self, "current_insertion_depth_tensor", None),
            dtype=torch.float32,
            default=0.0,
        )
        fallback_depth = self._srsa_env_vector(
            getattr(self, "disassembly_dists", None),
            dtype=torch.float32,
            default=0.0,
        )
        target_depth = torch.where(target_depth > 0.0, target_depth, fallback_depth)

        radial_clearance = self._srsa_task_param_vector("radial_clearance", default=0.0)
        if torch.all(radial_clearance <= 0.0):
            diametral_clearance = self._srsa_task_param_vector("clearance", default=0.0)
            if torch.all(diametral_clearance <= 0.0):
                hole_diameter = self._srsa_task_param_vector("hole_diameter", default=0.0)
                plug_diameter = self._srsa_task_param_vector("plug_diameter", default=0.0)
                diametral_clearance = (hole_diameter - plug_diameter).clamp_min(0.0)
            radial_clearance = 0.5 * diametral_clearance.clamp_min(0.0)

        rel_socket = self._held_fixed_delta_socket().to(dtype=torch.float32)
        current_depth = (target_depth - rel_socket[:, 2]).clamp_min(0.0)
        depth_fraction = current_depth / target_depth.clamp_min(1.0e-6)
        height_window_ok = (rel_socket[:, 2] > 0.0) & (rel_socket[:, 2] < target_depth.clamp_min(1.0e-6))
        depth_ok = height_window_ok & (
            depth_fraction >= float(getattr(self.cfg, "srsa_process_success_depth_ratio", 0.85))
        )

        lateral_error = torch.linalg.norm(rel_socket[:, :2], dim=-1)
        lateral_tol = _clamped_tolerance(
            radial_clearance,
            float(getattr(self.cfg, "srsa_process_success_lateral_tol_scale", 2.0)),
            getattr(self.cfg, "srsa_process_success_lateral_tol_min", 0.001),
            getattr(self.cfg, "srsa_process_success_lateral_tol_max", 0.003),
        )
        lateral_ok = lateral_error <= lateral_tol

        orientation_error = zeros.clone()
        yaw_error = zeros.clone()
        if all(hasattr(self, name) for name in ("held_quat", "fixed_quat")):
            try:
                rel_quat = torch_utils.quat_mul(self.held_quat, torch_utils.quat_conjugate(self.fixed_quat))
                rel_quat = torch.where(rel_quat[:, :1] < 0.0, -rel_quat, rel_quat)
                rel_euler = torch.stack(torch_utils.get_euler_xyz(rel_quat), dim=1)
                rel_euler = torch.atan2(torch.sin(rel_euler), torch.cos(rel_euler))
                orientation_error = torch.amax(torch.abs(rel_euler[:, :2]), dim=-1).to(dtype=torch.float32)
                yaw_error = torch.abs(rel_euler[:, 2]).to(dtype=torch.float32)
            except Exception:
                pass
        orientation_ok = orientation_error <= float(
            getattr(self.cfg, "srsa_process_success_orientation_tol_rad", 0.0872665)
        )

        yaw_required = self._srsa_task_param_vector("yaw_requirement_float", default=0.0) > 0.5
        if bool(getattr(self.cfg, "axial_yaw_requirement", False)):
            yaw_required = torch.full_like(yaw_required, True)
        yaw_ok = (~yaw_required) | (
            yaw_error <= float(getattr(self.cfg, "srsa_process_success_yaw_tol_rad", 0.0872665))
        )

        keypoint_error = zeros.clone()
        if all(hasattr(self, name) for name in ("keypoints_held", "keypoints_fixed")):
            keypoint_error = torch.linalg.norm(self.keypoints_fixed - self.keypoints_held, dim=-1).mean(dim=-1)
            keypoint_error = keypoint_error.to(dtype=torch.float32)
        keypoint_tol = _clamped_tolerance(
            radial_clearance,
            float(getattr(self.cfg, "srsa_process_success_keypoint_tol_scale", 2.0)),
            getattr(self.cfg, "srsa_process_success_keypoint_tol_min", 0.001),
            getattr(self.cfg, "srsa_process_success_keypoint_tol_max", 0.003),
        )
        keypoint_ok = keypoint_error <= keypoint_tol

        contact = self._srsa_env_vector(getattr(self, "flange_force_flag", false), dtype=torch.bool, default=False)
        jam_lateral_thresh = torch.maximum(radial_clearance, lateral_tol)
        jam = contact & (~current_official) & (lateral_error > jam_lateral_thresh)

        process = depth_ok & lateral_ok & orientation_ok & yaw_ok & keypoint_ok
        if bool(getattr(self.cfg, "srsa_process_success_require_no_jam", True)):
            process = process & (~jam)
        if bool(getattr(self.cfg, "srsa_process_success_require_official", False)):
            process = process & current_official

        current_streak = getattr(self, "_srsa_process_success_streak", torch.zeros_like(process, dtype=torch.int64))
        next_streak = torch.where(process, current_streak + 1, torch.zeros_like(current_streak))
        stable_steps = max(1, int(getattr(self.cfg, "srsa_process_success_stable_steps", 3)))
        terminal_process = process & (next_streak >= stable_steps)
        episode_process = getattr(self, "_srsa_episode_process_success", false) | terminal_process
        dual = episode_official & terminal_process

        metrics = {
            "success_metric": _normalize_srsa_success_metric(
                getattr(self.cfg, "srsa_eval_success_metric", "terminal_process")
            ),
            "official_success": episode_official,
            "current_official_success": current_official,
            "process_success": process,
            "episode_process_success": episode_process,
            "terminal_process_success": terminal_process,
            "dual_success": dual,
            "depth_ok": depth_ok,
            "lateral_ok": lateral_ok,
            "orientation_ok": orientation_ok,
            "yaw_ok": yaw_ok,
            "keypoint_ok": keypoint_ok,
            "jam": jam,
            "contact": contact,
            "depth_fraction": depth_fraction,
            "current_depth": current_depth,
            "target_depth": target_depth,
            "lateral_error": lateral_error,
            "lateral_tol": lateral_tol,
            "orientation_error": orientation_error,
            "yaw_error": yaw_error,
            "keypoint_error": keypoint_error,
            "keypoint_tol": keypoint_tol,
            "radial_clearance": radial_clearance,
            "process_success_streak": next_streak.to(dtype=torch.float32),
            "_process_success_streak": next_streak,
        }

        success = self._select_srsa_success(metrics)
        metrics["success"] = success
        metrics["score"] = success.clone()

        if update_state:
            self._srsa_process_success_streak[:] = next_streak
            self._srsa_episode_process_success[:] = episode_process
            self._srsa_episode_official_success[:] = episode_official
            self._cache_srsa_success_metrics(metrics)
        return metrics

    def _update_srsa_success_extras(self, metrics: dict | None = None) -> None:
        if not hasattr(self, "extras") or not isinstance(self.extras, dict):
            return
        if metrics is None:
            metrics = getattr(self, "_last_srsa_success_metrics", None) or self._compute_srsa_success_metrics(
                update_state=False
            )
        for key, value in metrics.items():
            if key.startswith("_") or not torch.is_tensor(value):
                continue
            self.extras[key] = value.detach().clone()

    def _compute_depth_contact_jam(self) -> dict[str, torch.Tensor]:
        return getattr(self, "_last_srsa_success_metrics", None) or self._compute_srsa_success_metrics(
            update_state=False
        )

    def _update_newt_task_extras(self) -> None:
        if not hasattr(self, "extras") or not isinstance(self.extras, dict):
            return
        metrics = self._compute_depth_contact_jam()
        self._update_srsa_success_extras(metrics)
        self.extras["force"] = {
            "world": self.flange_force_world.detach().clone(),
            "socket": self.flange_force_socket.detach().clone(),
            "norm": self.flange_force_norm.detach().clone(),
            "flag": self.flange_force_flag.detach().clone(),
        }
        self.extras["depth"] = {
            "current": metrics["current_depth"].detach().clone(),
            "target": metrics["target_depth"].detach().clone(),
            "fraction": metrics["depth_fraction"].detach().clone(),
        }

    def _augment_policy_observation_with_task_params(self, observations):
        if not self.task_param_obs:
            return observations

        if isinstance(observations, dict) and isinstance(observations.get("policy"), torch.Tensor):
            if self.task_param_obs_mode == "task_vec":
                task_obs = self.current_task_vec
            else:
                task_obs = self.current_task_param_tensor
            if task_obs is None:
                return observations
            observations = dict(observations)
            observations["policy"] = torch.cat([observations["policy"], task_obs.to(dtype=torch.float32)], dim=-1)
            return observations
        return observations

    def _augment_policy_observation_with_flange_force(self, observations):
        if not self.enable_flange_force_sensor:
            return observations

        if isinstance(observations, dict) and isinstance(observations.get("policy"), torch.Tensor):
            observations = dict(observations)
            observations["policy"] = torch.cat([observations["policy"], self.flange_force_obs], dim=-1)
            return observations
        return observations

    def _reset_vision_noise_cache(self) -> None:
        self._vision_noise_cache_step = None

    def _env_ids_to_tensor(self, env_ids) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)

    def _reset_idx(self, env_ids):
        env_ids = self._env_ids_to_tensor(env_ids)
        self._reset_srsa_success_state(env_ids)
        self._prepare_axial_task_reset(env_ids)
        super()._reset_idx(env_ids)

    def _prepare_axial_task_reset(self, env_ids: torch.Tensor) -> None:
        if (
            env_ids.numel() == 0
            or self.axial_task_param_sampler is None
            or not self.enable_axial_task_param_sampler
        ):
            return
        sampled_params = self.axial_task_param_sampler.sample(env_ids.numel(), self.device)
        self._set_current_task_param_tensors(env_ids, sampled_params)
        self.current_geometry_variant_applied[env_ids] = self._apply_geometry_variant(env_ids).reshape(-1, 1)
        if hasattr(self, "disassembly_dists"):
            self.disassembly_dists[env_ids] = self.current_insertion_depth_tensor[env_ids]
        if hasattr(self, "curriculum_height_bound") and hasattr(self.cfg_task, "curriculum_freespace_range"):
            upper = self.disassembly_dists[env_ids] + float(self.cfg_task.curriculum_freespace_range)
            self.curriculum_height_bound[env_ids, 1] = upper
            if hasattr(self, "curr_max_disp"):
                if bool(getattr(self.cfg_task, "if_sbc", False)):
                    self.curr_max_disp[env_ids] = torch.minimum(self.curr_max_disp[env_ids], upper)
                else:
                    self.curr_max_disp[env_ids] = upper
        self.cfg_task.close_error_thresh = float(self.current_close_error_thresh_tensor[env_ids[0]].item())
        if hasattr(self, "_srsa_base_gripper_open_width"):
            self.current_gripper_open_width[env_ids] = (
                float(self._srsa_base_gripper_open_width) * self.current_plug_scale_xy[env_ids]
            )

    def _apply_geometry_variant(self, env_ids: torch.Tensor) -> torch.Tensor:
        applied = torch.zeros((env_ids.numel(),), dtype=torch.bool, device=self.device)
        if env_ids.numel() == 0:
            return applied
        try:
            from pxr import Gf, UsdGeom
        except Exception:
            return applied
        stage = getattr(getattr(self, "scene", None), "stage", None)
        env_prim_paths = getattr(getattr(self, "scene", None), "env_prim_paths", None)
        if stage is None or env_prim_paths is None:
            return applied

        def set_scale(prim_path: str, scale_xy: float) -> bool:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                return False
            xformable = UsdGeom.Xformable(prim)
            scale_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_op = op
                    break
            if scale_op is None:
                scale_op = xformable.AddScaleOp()
            scale_op.Set(Gf.Vec3f(float(scale_xy), float(scale_xy), 1.0))
            return True

        for local_idx, env_id_tensor in enumerate(env_ids):
            env_id = int(env_id_tensor.item())
            env_path = env_prim_paths[env_id]
            held_ok = set_scale(f"{env_path}/HeldAsset", float(self.current_plug_scale_xy[env_id].item()))
            fixed_ok = set_scale(f"{env_path}/FixedAsset", float(self.current_hole_scale_xy[env_id].item()))
            scale_changed = (
                abs(float(self.current_plug_scale_xy[env_id].item()) - 1.0) > 1.0e-6
                or abs(float(self.current_hole_scale_xy[env_id].item()) - 1.0) > 1.0e-6
            )
            applied[local_idx] = (held_ok or fixed_ok) and scale_changed
        return applied

    def _set_franka_to_default_pose(self, joints, env_ids):
        if not hasattr(self, "current_gripper_open_width"):
            return super()._set_franka_to_default_pose(joints, env_ids)
        env_ids = self._env_ids_to_tensor(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        gripper_width = self.current_gripper_open_width[env_ids].reshape(-1, 1)
        joint_pos[:, 7:] = gripper_width
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)
        self.step_sim_no_action()

    def randomize_held_initial_state(self, env_ids, pre_grasp):
        if not self.enable_axial_task_param_sampler:
            return super().randomize_held_initial_state(env_ids, pre_grasp)
        env_ids = self._env_ids_to_tensor(env_ids)
        curr_curriculum_disp_range = self.curriculum_height_bound[:, 1] - self.curr_max_disp
        if pre_grasp:
            self.curriculum_disp = self.curr_max_disp + curr_curriculum_disp_range * (
                torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
            )
            self.held_pos_init_rand = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            self.held_pos_init_rand[env_ids] = self.current_initial_error_pos[env_ids]

        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] = self.fixed_pos[env_ids].clone() + self.scene.env_origins[env_ids]
        yaw_quat = torch_utils.quat_from_euler_xyz(
            torch.zeros((env_ids.numel(),), dtype=torch.float32, device=self.device),
            torch.zeros((env_ids.numel(),), dtype=torch.float32, device=self.device),
            self.current_initial_error_yaw[env_ids],
        )
        held_state[:, 3:7] = torch_utils.quat_mul(yaw_quat, self.fixed_quat[env_ids].clone())
        held_state[:, 7:] = 0.0
        held_state[:, 2] += self.curriculum_disp[env_ids]

        plug_in_freespace = self.curriculum_disp[env_ids] > self.disassembly_dists[env_ids]
        if torch.any(plug_in_freespace):
            held_state[plug_in_freespace, :2] += self.held_pos_init_rand[env_ids[plug_in_freespace], :2]

        self._held_asset.write_root_state_to_sim(held_state, env_ids=env_ids)
        self._held_asset.reset()
        self.step_sim_no_action()

    def _pre_physics_step(self, action):
        if bool(getattr(self.cfg, "newt_obs", False)):
            action = action.to(self.device)
            if action.shape[-1] < int(getattr(self.cfg, "newt_action_dim", NEWT_ACTION_DIM)):
                pad_dim = int(getattr(self.cfg, "newt_action_dim", NEWT_ACTION_DIM)) - action.shape[-1]
                pad = torch.zeros(
                    (*action.shape[:-1], pad_dim),
                    dtype=action.dtype,
                    device=action.device,
                )
                action = torch.cat([action, pad], dim=-1)
            elif action.shape[-1] > int(getattr(self.cfg, "newt_action_dim", NEWT_ACTION_DIM)):
                action = action[..., : int(getattr(self.cfg, "newt_action_dim", NEWT_ACTION_DIM))]
        return super()._pre_physics_step(action)

    def has_action_mask(self) -> bool:
        return True

    def get_action_mask(self) -> torch.Tensor:
        return self._newt_action_mask.detach().clone()

    def _sample_episode_vision_noise(self, env_ids) -> None:
        env_ids = self._env_ids_to_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        noise = torch.zeros((env_ids.numel(), 3), device=self.device)
        if self.vision_noise_xy_std > 0.0:
            noise[:, :2] = torch.randn((env_ids.numel(), 2), device=self.device) * self.vision_noise_xy_std
        if self.vision_noise_z_std > 0.0:
            noise[:, 2] = torch.randn(env_ids.numel(), device=self.device) * self.vision_noise_z_std
        self._vision_noise_episode_local[env_ids] = noise
        self._reset_vision_noise_cache()

    def randomize_initial_state(self, env_ids):
        super().randomize_initial_state(env_ids)
        if not self.enable_axial_task_param_sampler:
            self._sample_episode_vision_noise(env_ids)

    def _refresh_vision_noise_cache(self) -> None:
        cache_step = int(getattr(self, "_sim_step_counter", -1))
        if self._vision_noise_cache_step == cache_step:
            return

        noise_local = self._vision_noise_episode_local.clone()
        if self.vision_noise_xy_jitter_std > 0.0:
            noise_local[:, :2] += torch.randn((self.num_envs, 2), device=self.device) * self.vision_noise_xy_jitter_std
        if self.vision_noise_z_jitter_std > 0.0:
            noise_local[:, 2] += torch.randn(self.num_envs, device=self.device) * self.vision_noise_z_jitter_std

        self._vision_noise_world = quat_apply(self.fixed_quat, noise_local)
        self._vision_noise_cache_step = cache_step

    def _canonicalize_quat(self, quat: torch.Tensor) -> torch.Tensor:
        quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
        return quat * torch.where(quat[:, :1] < 0.0, -1.0, 1.0)

    def _build_newt_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        state_dim = int(getattr(self.cfg, "newt_state_dim", NEWT_STATE_DIM))
        frame_pos = self.fixed_pos_obs_frame
        frame_quat = self.fixed_quat
        frame_quat_inv, frame_pos_inv = torch_utils.tf_inverse(frame_quat, frame_pos)

        tcp_quat_socket, tcp_pos_socket = torch_utils.tf_combine(
            frame_quat_inv,
            frame_pos_inv,
            self.fingertip_midpoint_quat,
            self.fingertip_midpoint_pos,
        )
        goal_quat_socket, goal_pos_socket = torch_utils.tf_combine(
            frame_quat_inv,
            frame_pos_inv,
            self.gripper_goal_quat,
            self.gripper_goal_pos,
        )
        held_quat_socket, held_pos_socket = torch_utils.tf_combine(
            frame_quat_inv,
            frame_pos_inv,
            self.held_quat,
            self.held_pos,
        )
        fixed_quat_socket, fixed_pos_socket = torch_utils.tf_combine(
            frame_quat_inv,
            frame_pos_inv,
            self.fixed_quat,
            self.fixed_pos,
        )
        tcp_linvel_socket = torch_utils.quat_rotate_inverse(frame_quat, self.ee_linvel_fd)
        tcp_angvel_socket = torch_utils.quat_rotate_inverse(frame_quat, self.ee_angvel_fd)
        held_linvel_world = getattr(self._held_asset.data, "root_lin_vel_w", torch.zeros_like(self.held_pos))
        held_angvel_world = getattr(self._held_asset.data, "root_ang_vel_w", torch.zeros_like(self.held_pos))
        held_linvel_socket = torch_utils.quat_rotate_inverse(frame_quat, held_linvel_world)
        held_angvel_socket = torch_utils.quat_rotate_inverse(frame_quat, held_angvel_world)
        keypoint_delta = self.keypoints_held - self.keypoints_fixed
        keypoint_count = keypoint_delta.shape[1]
        keypoint_delta_socket = quat_apply(
            quat_conjugate(frame_quat).repeat_interleave(keypoint_count, dim=0),
            keypoint_delta.reshape(self.num_envs * keypoint_count, 3),
        ).reshape(self.num_envs, -1)
        metrics = self._compute_depth_contact_jam()
        gripper_width = self.joint_pos[:, 7:9].sum(dim=-1, keepdim=True)
        if hasattr(self, "actions"):
            prev_action = self.actions[:, :SRSA_ACTION_DIM]
        else:
            prev_action = torch.zeros((self.num_envs, SRSA_ACTION_DIM), dtype=torch.float32, device=self.device)
        parts = [
            tcp_pos_socket,
            self._canonicalize_quat(tcp_quat_socket),
            tcp_linvel_socket,
            tcp_angvel_socket,
            gripper_width,
            goal_pos_socket,
            self._canonicalize_quat(goal_quat_socket),
            goal_pos_socket - tcp_pos_socket,
            held_pos_socket,
            self._canonicalize_quat(held_quat_socket),
            held_linvel_socket,
            held_angvel_socket,
            fixed_pos_socket,
            self._canonicalize_quat(fixed_quat_socket),
            held_pos_socket - fixed_pos_socket,
            keypoint_delta_socket,
            self.keypoint_dist.reshape(-1, 1),
            self.flange_force_socket,
            self.flange_force_norm,
            metrics["contact"].to(dtype=torch.float32).reshape(-1, 1),
            metrics["current_depth"].reshape(-1, 1),
            metrics["target_depth"].reshape(-1, 1),
            metrics["depth_fraction"].reshape(-1, 1),
            metrics["lateral_error"].reshape(-1, 1),
            metrics["jam"].to(dtype=torch.float32).reshape(-1, 1),
            self.joint_pos[:, 0:7],
            self.joint_vel[:, 0:7],
            prev_action,
        ]
        compact_state = torch.cat(parts, dim=-1).to(dtype=torch.float32)
        state = torch.zeros((self.num_envs, state_dim), dtype=torch.float32, device=self.device)
        mask = torch.zeros((self.num_envs, state_dim), dtype=torch.bool, device=self.device)
        valid_dim = min(state_dim, compact_state.shape[-1])
        state[:, :valid_dim] = compact_state[:, :valid_dim]
        mask[:, :valid_dim] = True
        return state, mask

    def _make_newt_observations(self) -> dict[str, torch.Tensor]:
        if (
            self.vision_noise_xy_std > 0.0
            or self.vision_noise_xy_jitter_std > 0.0
            or self.vision_noise_z_std > 0.0
            or self.vision_noise_z_jitter_std > 0.0
            or bool(torch.any(self._vision_noise_episode_local != 0.0).item())
        ):
            self._refresh_vision_noise_cache()
            true_goal_pos = self.gripper_goal_pos
            self.gripper_goal_pos = true_goal_pos + self._vision_noise_world
            try:
                state, state_mask = self._build_newt_state()
            finally:
                self.gripper_goal_pos = true_goal_pos
        else:
            state, state_mask = self._build_newt_state()
        self._newt_state[:] = state
        self._newt_state_mask[:] = state_mask
        self._update_task_param_extras()
        self._update_flange_force_extras()
        self._update_newt_task_extras()
        return {
            "state": self._newt_state,
            "state_mask": self._newt_state_mask,
            "task": self.current_task_vec,
            "task_id": self.current_task_id,
            "action_mask": self._newt_action_mask,
        }

    def _get_observations(self):
        if bool(getattr(self.cfg, "newt_obs", False)):
            return self._make_newt_observations()

        if (
            self.vision_noise_xy_std <= 0.0
            and self.vision_noise_xy_jitter_std <= 0.0
            and self.vision_noise_z_std <= 0.0
            and self.vision_noise_z_jitter_std <= 0.0
            and not bool(torch.any(self._vision_noise_episode_local != 0.0).item())
        ):
            observations = super()._get_observations()
            self._update_task_param_extras()
            self._update_flange_force_extras()
            self._update_newt_task_extras()
            observations = self._augment_policy_observation_with_flange_force(observations)
            return self._augment_policy_observation_with_task_params(observations)

        self._refresh_vision_noise_cache()
        true_goal_pos = self.gripper_goal_pos
        self.gripper_goal_pos = true_goal_pos + self._vision_noise_world
        try:
            observations = super()._get_observations()
        finally:
            self.gripper_goal_pos = true_goal_pos
        self._update_task_param_extras()
        self._update_flange_force_extras()
        self._update_newt_task_extras()
        observations = self._augment_policy_observation_with_flange_force(observations)
        return self._augment_policy_observation_with_task_params(observations)

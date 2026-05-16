# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import torch
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_conjugate
from isaaclab_tasks.direct.automate.assembly_tasks_cfg import ASSET_DIR

from .task_family_config import TASK_FAMILY_CONFIG
from .task_param_utils import (
    TASK_PARAM_TENSOR_FIELD_ORDER,
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


def _read_optional_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def _read_optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


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
        use_task_param = bool(use_task_family or runtime_task_param_overrides)

        cfg.use_task_family = use_task_family
        cfg.use_task_param = use_task_param
        cfg.task_param_obs = task_param_obs
        cfg.task_family_config = task_family_config
        cfg.active_task_family_name = resolved_family_name if use_task_family else None
        cfg.active_task_family_id = (
            int(resolved_family_cfg["variant_id"]) if use_task_family and resolved_family_cfg is not None else None
        )
        cfg.runtime_task_param_overrides = runtime_task_param_overrides

        if cfg.task_param_obs and not getattr(cfg, "_srsa_task_param_obs_augmented", False):
            cfg.observation_space = int(getattr(cfg, "observation_space", 0)) + len(TASK_PARAM_TENSOR_FIELD_ORDER)
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
        self._init_task_param_runtime()
        self._init_flange_force_sensor_runtime()
        self._vision_noise_episode_local = torch.zeros((self.num_envs, 3), device=self.device)
        self._vision_noise_world = torch.zeros((self.num_envs, 3), device=self.device)
        self._vision_noise_cache_step = None

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
        self.enable_task_param = bool(self.use_task_param or self.task_param_obs)
        self.current_task_param_tensor = None
        self.current_task_params = {}

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

        effective_params = resolve_effective_task_params(
            base_task_cfg=self.cfg_task,
            task_family_name=resolved_family_name,
            task_family_cfg=resolved_family_cfg,
            runtime_overrides=runtime_overrides,
            baseline_insertion_depth=baseline_insertion_depth,
        )
        self.current_task_params = effective_params
        self.current_task_param_tensor = make_task_param_tensor(effective_params, self.num_envs, self.device)
        self.current_close_error_thresh = float(effective_params["success_pos_tol"])
        self.current_insertion_depth = float(effective_params["insertion_depth"])

        if self.use_task_param:
            if hasattr(self, "disassembly_dists"):
                self.disassembly_dists = torch.full_like(self.disassembly_dists, self.current_insertion_depth)
            self.cfg_task.close_error_thresh = self.current_close_error_thresh

            if hasattr(self, "curriculum_height_bound") and hasattr(self.cfg_task, "curriculum_freespace_range"):
                self.curriculum_height_bound[:, 1] = self.disassembly_dists + float(
                    self.cfg_task.curriculum_freespace_range
                )

            if hasattr(self, "gripper_open_width"):
                self.gripper_open_width = float(self.gripper_open_width) * float(effective_params["plug_scale_xy"])

    def _update_task_param_extras(self) -> None:
        if not self.current_task_params or not hasattr(self, "extras") or not isinstance(self.extras, dict):
            return

        self.extras["task_family_id"] = float(self.current_task_params["task_family_id"])
        self.extras["plug_diameter"] = float(self.current_task_params["plug_diameter"])
        self.extras["hole_diameter"] = float(self.current_task_params["hole_diameter"])
        self.extras["clearance"] = float(self.current_task_params["clearance"])
        self.extras["clearance_ratio"] = float(self.current_task_params["clearance_ratio"])
        self.extras["insertion_depth"] = float(self.current_task_params["insertion_depth"])
        self.extras["success_pos_tol"] = float(self.current_task_params["success_pos_tol"])

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

    def _augment_policy_observation_with_task_params(self, observations):
        if not self.task_param_obs or self.current_task_param_tensor is None:
            return observations

        if isinstance(observations, dict) and isinstance(observations.get("policy"), torch.Tensor):
            observations = dict(observations)
            observations["policy"] = torch.cat([observations["policy"], self.current_task_param_tensor], dim=-1)
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

    def _get_observations(self):
        if (
            self.vision_noise_xy_std <= 0.0
            and self.vision_noise_xy_jitter_std <= 0.0
            and self.vision_noise_z_std <= 0.0
            and self.vision_noise_z_jitter_std <= 0.0
        ):
            observations = super()._get_observations()
            self._update_task_param_extras()
            self._update_flange_force_extras()
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
        observations = self._augment_policy_observation_with_flange_force(observations)
        return self._augment_policy_observation_with_task_params(observations)

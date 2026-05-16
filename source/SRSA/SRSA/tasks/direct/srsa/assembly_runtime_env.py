# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import torch
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.direct.automate.assembly_tasks_cfg import ASSET_DIR


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


class AssemblyRuntimeEnvMixin:
    """Runtime task overrides plus observation-only vision noise."""

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

        task_cfg.if_sbc = _read_bool_env("SRSA_IF_SBC", bool(task_cfg.if_sbc))
        task_cfg.if_logging_eval = _read_bool_env("SRSA_IF_LOGGING_EVAL", bool(task_cfg.if_logging_eval))
        task_cfg.eval_filename = os.environ.get("SRSA_EVAL_FILENAME", task_cfg.eval_filename)
        task_cfg.num_eval_trials = _read_int_env("SRSA_NUM_EVAL_TRIALS", int(task_cfg.num_eval_trials))

    def _init_tensors(self):
        super()._init_tensors()
        self._vision_noise_episode_local = torch.zeros((self.num_envs, 3), device=self.device)
        self._vision_noise_world = torch.zeros((self.num_envs, 3), device=self.device)
        self._vision_noise_cache_step = None

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
            return super()._get_observations()

        self._refresh_vision_noise_cache()
        true_goal_pos = self.gripper_goal_pos
        self.gripper_goal_pos = true_goal_pos + self._vision_noise_world
        try:
            return super()._get_observations()
        finally:
            self.gripper_goal_pos = true_goal_pos

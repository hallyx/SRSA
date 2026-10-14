# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import torch
from isaaclab_tasks.direct.automate.assembly_env import AssemblyEnv
from isaaclab_tasks.direct.automate.assembly_env_cfg import AssemblyEnvCfg

from .assembly_runtime_env import AssemblyRuntimeEnvMixin
from .relaxed_eval import relaxed_step


class AssemblyDirectEnv(AssemblyRuntimeEnvMixin, AssemblyEnv):
    cfg: AssemblyEnvCfg

    def _get_rewards(self):
        if not self.cfg_task.if_logging_eval and os.environ.get("SRSA_TRAIN_RELAXED_GT085") == "1":
            return self._get_relaxed_training_rewards()
        if not (self.cfg_task.if_logging_eval and os.environ.get("SRSA_RELAXED_DEPTH_UNBOUNDED") == "1"):
            return super()._get_rewards()

        if not hasattr(self, "_relaxed_eval_streak"):
            self._relaxed_eval_streak = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            self._relaxed_eval_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._relaxed_eval_original_streak = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            self._relaxed_eval_original_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._relaxed_eval_terminal_log = []
            self._relaxed_eval_episode_log = []
            self._relaxed_eval_original_terminal_log = []
            self._relaxed_eval_original_episode_log = []
            self._relaxed_eval_original_depth_ok_log = []
            self._relaxed_eval_lateral_ok_log = []
            self._relaxed_eval_contact_log = []
            self._relaxed_eval_depth_log = []
            self._relaxed_eval_lateral_log = []
            self._relaxed_eval_jam_log = []

        metrics = self._compute_srsa_success_metrics(update_state=False)
        _, stable, jam, self._relaxed_eval_streak = relaxed_step(
            metrics["depth_fraction"], metrics["lateral_error"], metrics["radial_clearance"],
            metrics["contact"], metrics["current_official_success"], self._relaxed_eval_streak,
        )
        self._relaxed_eval_episode |= stable
        # FACA's original relaxed gate uses the same lateral/no-jam rules but
        # requires 0 < relative height < target depth and depth >= 0.85.
        # Keep a separate streak: terminal depth alone cannot recover 3 steps.
        original_process = metrics["depth_ok"] & metrics["lateral_ok"] & ~jam
        self._relaxed_eval_original_streak = torch.where(
            original_process, self._relaxed_eval_original_streak + 1,
            torch.zeros_like(self._relaxed_eval_original_streak),
        )
        original_stable = original_process & (self._relaxed_eval_original_streak >= 3)
        self._relaxed_eval_original_episode |= original_stable
        if torch.any(self.reset_buf):
            self._relaxed_eval_terminal_log.append(stable.detach().cpu().reshape(-1, 1))
            self._relaxed_eval_episode_log.append(self._relaxed_eval_episode.detach().cpu().reshape(-1, 1))
            self._relaxed_eval_original_terminal_log.append(original_stable.detach().cpu().reshape(-1, 1))
            self._relaxed_eval_original_episode_log.append(self._relaxed_eval_original_episode.detach().cpu().reshape(-1, 1))
            self._relaxed_eval_original_depth_ok_log.append(metrics["depth_ok"].detach().cpu().reshape(-1, 1))
            self._relaxed_eval_lateral_ok_log.append(metrics["lateral_ok"].detach().cpu().reshape(-1, 1))
            self._relaxed_eval_contact_log.append(metrics["contact"].detach().cpu().reshape(-1, 1))
            self._relaxed_eval_depth_log.append(metrics["depth_fraction"].detach().cpu().reshape(-1, 1))
            self._relaxed_eval_lateral_log.append(metrics["lateral_error"].detach().cpu().reshape(-1, 1))
            self._relaxed_eval_jam_log.append(jam.detach().cpu().reshape(-1, 1))

        try:
            return super()._get_rewards()
        except SystemExit:
            import h5py

            datasets = {
                "relaxed_success_stable": self._relaxed_eval_terminal_log,
                "relaxed_success_episode": self._relaxed_eval_episode_log,
                "relaxed_original_success_stable": self._relaxed_eval_original_terminal_log,
                "relaxed_original_success_episode": self._relaxed_eval_original_episode_log,
                "relaxed_original_depth_ok_final": self._relaxed_eval_original_depth_ok_log,
                "relaxed_lateral_ok_final": self._relaxed_eval_lateral_ok_log,
                "relaxed_contact_final": self._relaxed_eval_contact_log,
                "relaxed_depth_fraction_final": self._relaxed_eval_depth_log,
                "relaxed_lateral_error_final": self._relaxed_eval_lateral_log,
                "relaxed_jam_final": self._relaxed_eval_jam_log,
            }
            with h5py.File(self.cfg_task.eval_filename, "a") as hf:
                count = int(hf["success"].shape[0])
                for name, chunks in datasets.items():
                    hf.create_dataset(name, data=torch.cat(chunks, dim=0)[:count].numpy())
                hf.attrs["relaxed_protocol"] = (
                    "FACA relaxed_success_stable, depth_fraction > 0.85, "
                    "lateral_tol=clamp(2*radial_clearance, 0.001, 0.003), no jam, 3 steps"
                )
                hf.attrs["original_relaxed_protocol"] = (
                    "FACA relaxed_success_stable, height_window 0<height<target, "
                    "depth_fraction >= 0.85, same lateral/no-jam/3-step gates"
                )
            raise

    def _get_relaxed_training_rewards(self):
        """Fine-tune with the exact modified relaxed gate and a dense alignment aid."""
        from isaaclab_tasks.direct.automate import automate_algo_utils as automate_algo

        if not hasattr(self, "_relaxed_train_streak"):
            self._relaxed_train_streak = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        metrics = self._compute_srsa_success_metrics(update_state=False)
        _, stable, jam, self._relaxed_train_streak = relaxed_step(
            metrics["depth_fraction"], metrics["lateral_error"], metrics["radial_clearance"],
            metrics["contact"], metrics["current_official_success"], self._relaxed_train_streak,
        )
        # Keep the original SDF and imitation terms. The official success bonus
        # is replaced by modified relaxed success; the alignment aid gives a
        # useful signal before a sparse three-step success is first observed.
        rew_buf = self._update_rew_buf(stable)
        depth_progress = (metrics["depth_fraction"] / 0.85).clamp(0.0, 1.0)
        lateral_progress = (1.0 - metrics["lateral_error"] / 0.015).clamp(0.0, 1.0)
        alignment_reward = depth_progress * lateral_progress * (~jam).float()
        rew_buf = rew_buf + alignment_reward
        self.ep_succeeded = torch.logical_or(self.ep_succeeded, stable)
        self.extras["logs_rew_relaxed_alignment"] = alignment_reward.mean()
        self.extras["relaxed_stable_rate"] = stable.float().mean()
        self.extras["relaxed_depth_rate"] = (metrics["depth_fraction"] > 0.85).float().mean()
        self.extras["relaxed_lateral_rate"] = metrics["lateral_ok"].float().mean()

        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(self.ep_succeeded) / self.num_envs
            sbc_rwd_scale = automate_algo.get_curriculum_reward_scale(
                curr_max_disp=self.curr_max_disp,
                curriculum_height_bound=self.curriculum_height_bound,
            )
            rew_buf *= sbc_rwd_scale
            if self.cfg_task.if_sbc:
                self.curr_max_disp = automate_algo.get_new_max_disp(
                    curr_success=torch.count_nonzero(self.ep_succeeded) / self.num_envs,
                    cfg_task=self.cfg_task,
                    curriculum_height_bound=self.curriculum_height_bound,
                    curriculum_height_step=self.curriculum_height_step,
                    curr_max_disp=self.curr_max_disp,
                )
            self.extras["curr_max_disp"] = self.curr_max_disp

        self.prev_actions = self.actions.clone()
        return rew_buf

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "_relaxed_train_streak"):
            self._relaxed_train_streak[env_ids] = 0

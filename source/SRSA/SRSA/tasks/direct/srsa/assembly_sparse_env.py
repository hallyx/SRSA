# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab_tasks.direct.automate import automate_algo_utils as automate_algo
from isaaclab_tasks.direct.automate import automate_log_utils as automate_log
from isaaclab_tasks.direct.automate.assembly_env import AssemblyEnv
from isaaclab_tasks.direct.automate.assembly_env_cfg import AssemblyEnvCfg

from .assembly_runtime_env import AssemblyRuntimeEnvMixin


class AssemblySparseEnv(AssemblyRuntimeEnvMixin, AssemblyEnv):
    cfg: AssemblyEnvCfg

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        success_metrics = self._compute_srsa_success_metrics(update_state=True)
        curr_successes = success_metrics["success"].to(dtype=torch.bool)

        rew_buf = self._update_rew_buf(curr_successes)
        if hasattr(self, "_update_task_param_extras"):
            self._update_task_param_extras()
        if hasattr(self, "_update_flange_force_extras"):
            self._update_flange_force_extras()
        if hasattr(self, "_update_srsa_success_extras"):
            self._update_srsa_success_extras(success_metrics)
        if hasattr(self, "_update_newt_task_extras"):
            self._update_newt_task_extras()

        # Only log episode success rates at the end of an episode.
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

            print("Success", torch.mean(self.ep_succeeded.float()).item())

            if self.cfg_task.if_logging_eval:
                self.success_log = torch.cat([self.success_log, self.ep_succeeded.reshape((self.num_envs, 1))], dim=0)

                if self.success_log.shape[0] >= self.cfg_task.num_eval_trials:
                    automate_log.write_log_to_hdf5(
                        self.held_asset_pose_log,
                        self.fixed_asset_pose_log,
                        self.success_log,
                        self.cfg_task.eval_filename,
                    )
                    exit(0)

        self.prev_actions = self.actions.clone()
        return rew_buf

    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep."""
        rew_dict = dict({})

        rew_dict["curr_successes"] = curr_successes.clone().float()

        # Sparse Reward
        self.ep_succeeded = torch.logical_or(self.ep_succeeded, curr_successes)
        rew_dict["sparse"] = self.ep_succeeded.float()

        rew_buf = rew_dict["sparse"]

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        return rew_buf

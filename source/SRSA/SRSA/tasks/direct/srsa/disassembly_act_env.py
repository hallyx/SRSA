# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os

import numpy as np
import torch

from . import factory_control as fc
from .disassembly_env import DisassemblyEnv
from .disassembly_env_cfg import DisassemblyEnvCfg


class DisassemblyActEnv(DisassemblyEnv):
    cfg: DisassemblyEnvCfg

    def __init__(self, cfg: DisassemblyEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

    def _get_dones(self):
        """Update intermediate values used for rewards and observations."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if time_out[0]:

            self.close_gripper(env_ids=np.array(range(self.num_envs)).reshape(-1))
            self._disassemble_plug_from_socket()

            if_intersect = (self.held_pos[:, 2] < self.fixed_pos[:, 2] + self.disassembly_dists).cpu().numpy()
            success_env_ids = np.argwhere(if_intersect == 0).reshape(-1)

            self._log_robot_state(success_env_ids)
            self._log_object_state(success_env_ids)
            self._log_action(success_env_ids)
            self._save_log_traj()

        return time_out, time_out

    def _move_gripper_to_eef_pose(self, env_ids, goal_pos, goal_quat, sim_steps, if_log=False):

        for _ in range(sim_steps):
            if if_log:
                self._log_robot_state_per_timestep()
            # Compute error to target.
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=goal_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=goal_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            # print('delta hand pose', delta_hand_pose[0])
            self.actions *= 0.0
            # print('action shape', self.actions[env_ids, :6].shape)
            # print('delta hand shape', delta_hand_pose.shape)
            self.actions[env_ids, :6] = delta_hand_pose
            # print('action', self.actions[0])
            if if_log:
                self._log_action_per_timestep()

            is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
            # perform physics stepping
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                # set actions into buffers
                self._apply_action()
                # set actions into simulator
                self.scene.write_data_to_sim()
                # simulate
                self.sim.step(render=False)
                # render between steps only if the GUI or an RTX sensor needs it
                # note: we assume the render interval to be the shortest accepted rendering interval.
                #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render()
                # update buffers at sim dt
                self.scene.update(dt=self.physics_dt)

            # Simulate and update tensors.
            self.step_sim_no_action()

    def _init_log_data_per_assembly(self):

        super()._init_log_data_per_assembly()
        self.log_action = []

    def _init_log_data_per_episode(self):

        super()._init_log_data_per_episode()
        self.log_action_traj = []

    def _log_action_per_timestep(self):
        self.log_action_traj.append(self.actions.clone().detach())

    def _log_action(self, env_ids):
        self.log_action += torch.stack(self.log_action_traj, dim=1)[env_ids].cpu().tolist()

    def _save_log_traj(self):

        if len(self.log_arm_dof_pos) > self.cfg_task.num_log_traj:

            log_item = []
            for i in range(self.cfg_task.num_log_traj):
                curr_dict = dict({})
                curr_dict["fingertip_centered_pos"] = self.log_fingertip_centered_pos[i]
                curr_dict["fingertip_centered_quat"] = self.log_fingertip_centered_quat[i]
                curr_dict["arm_dof_pos"] = self.log_arm_dof_pos[i]
                curr_dict["plug_grasp_pos"] = self.log_plug_grasp_pos[i]
                curr_dict["plug_grasp_quat"] = self.log_plug_grasp_quat[i]
                curr_dict["init_plug_pos"] = self.log_init_plug_pos[i]
                curr_dict["init_plug_quat"] = self.log_init_plug_quat[i]
                curr_dict["plug_pos"] = self.log_plug_pos[i]
                curr_dict["plug_quat"] = self.log_plug_quat[i]
                curr_dict["actions"] = self.log_action[i]
                log_item.append(curr_dict)

            log_filename = os.path.join(
                os.getcwd(), self.cfg_task.data_dir, self.cfg_task.assembly_id + "_disassemble_traj.json"
            )

            with open(log_filename, "w+") as out_file:
                json.dump(log_item, out_file, indent=6)

            exit(0)
        else:
            print("current logging item num: ", len(self.log_arm_dof_pos))

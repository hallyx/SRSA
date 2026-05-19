# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local cfg extension for Franka task-parameterized SRSA assembly."""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.automate.assembly_env_cfg import AssemblyEnvCfg

from .task_family_config import TASK_FAMILY_CONFIG


@configclass
class AssemblyTaskParamEnvCfg(AssemblyEnvCfg):
    use_task_family: bool = False
    use_task_param: bool = False
    task_family_config = TASK_FAMILY_CONFIG
    default_task_family_name: str = "normal_fit"
    active_task_family_name: str | None = None
    active_task_family_id: int | None = None
    task_param_obs: bool = False
    newt_obs: bool = False
    newt_state_dim: int = 128
    newt_action_dim: int = 16
    enable_axial_task_param_sampler: bool = True
    axial_task_type_id: int = 0
    axial_scale_range: list[float] | None = None
    axial_clearance_range: list[float] | None = None
    axial_clearance_ratio_range: list[float] | None = None
    axial_target_depth_range: list[float] | None = None
    axial_init_error_xy_range: list[float] | None = None
    axial_init_error_z_range: list[float] | None = None
    axial_init_error_yaw_range: list[float] | None = None
    axial_visual_noise_xy_range: list[float] | None = None
    axial_visual_noise_z_range: list[float] | None = None
    axial_yaw_requirement: bool = False
    axial_reference_radius: float = 0.003993
    axial_reference_depth: float = 0.015
    runtime_task_param_overrides: dict | None = None
    enable_flange_force_sensor: bool = True
    flange_force_sensor_body_name: str = "panda_hand"
    flange_force_sensor_source: str = "held_sensor"
    flange_force_sensor_obs_frame: str = "socket"
    flange_force_sensor_obs_scale: float = 50.0
    flange_force_sensor_force_threshold: float = 1.0
    flange_force_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )
    held_asset_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

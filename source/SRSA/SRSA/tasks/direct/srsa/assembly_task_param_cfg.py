# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local cfg extension for Franka task-parameterized SRSA assembly."""

from __future__ import annotations

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
    runtime_task_param_overrides: dict | None = None

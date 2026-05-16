# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_tasks.direct.automate.assembly_env import AssemblyEnv
from isaaclab_tasks.direct.automate.assembly_env_cfg import AssemblyEnvCfg

from .assembly_runtime_env import AssemblyRuntimeEnvMixin


class AssemblyDirectEnv(AssemblyRuntimeEnvMixin, AssemblyEnv):
    cfg: AssemblyEnvCfg

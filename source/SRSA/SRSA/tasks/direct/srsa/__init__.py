# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from . import agents
##
# Register Gym environments.
##

gym.register(
    id="Assembly-Direct-v0",
    entry_point=f"{__name__}.assembly_direct_env:AssemblyDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.automate.assembly_env_cfg:AssemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Assembly-Direct-Sil-v0",
    entry_point=f"{__name__}.assembly_direct_env:AssemblyDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.automate.assembly_env_cfg:AssemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_sil_cfg.yaml",
    },
)

gym.register(
    id="Assembly-Sparse-v0",
    entry_point=f"{__name__}.assembly_sparse_env:AssemblySparseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.automate.assembly_env_cfg:AssemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Assembly-Sparse-Sil-v0",
    entry_point=f"{__name__}.assembly_sparse_env:AssemblySparseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.automate.assembly_env_cfg:AssemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_sil_cfg.yaml",
    },
)


gym.register(
    id="Disassembly-Direct-v0",
    entry_point="isaaclab_tasks.direct.automate.disassembly_env:DisassemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.automate.disassembly_env_cfg:DisassemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Disassembly-Act-v0",
    entry_point=f"{__name__}.disassembly_act_env:DisassemblyActEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.automate.disassembly_env_cfg:DisassemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

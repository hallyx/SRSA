# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch a small heterogeneous SRSA vector environment and validate M0 inputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(os.environ.get("FULL_PATH_TO_ISAACLAB", "/home/gpuserver/IsaacLab"))
for path in reversed(
    [
        REPO_ROOT / "source" / "SRSA",
        ISAACLAB_ROOT / "source" / "isaaclab",
        ISAACLAB_ROOT / "source" / "isaaclab_assets",
        ISAACLAB_ROOT / "source" / "isaaclab_rl",
        ISAACLAB_ROOT / "source" / "isaaclab_tasks",
    ]
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--assembly_ids", default="", help="Optional ordered subset of manifest assembly IDs.")
parser.add_argument("--manifest", default=None, help="Optional FACA JSON manifest containing task_vec_6.")
parser.add_argument("--require_task_vectors", action="store_true")
parser.add_argument("--task", default="Assembly-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=2)
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.assembly_ids:
    os.environ["SRSA_MULTITASK_ASSEMBLY_IDS"] = args.assembly_ids
if args.manifest:
    os.environ["SRSA_MULTITASK_MANIFEST"] = str(Path(args.manifest).expanduser().resolve())
os.environ["SRSA_MULTITASK_REQUIRE_TASK_VECS"] = "1" if args.require_task_vectors else "0"
os.environ["SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER"] = "0"
os.environ["SRSA_TASK_PARAM_GEOMETRY_SCALE"] = "0"
os.environ["SRSA_IF_SBC"] = "0"
os.environ["SRSA_IF_LOGGING_EVAL"] = "0"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[SRSA multitask smoke] simulator ready", flush=True)

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import SRSA.tasks  # noqa: F401
print("[SRSA multitask smoke] task packages imported", flush=True)


def main() -> None:
    print(f"[SRSA multitask smoke] parsing task={args.task!r}", flush=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    print("[SRSA multitask smoke] creating environment", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    try:
        runtime = env.unwrapped
        observations, _ = env.reset()
        contract = runtime.get_faca_m0_contract()
        m0_inputs = runtime.get_faca_m0_inputs()
        task_indices = runtime.get_faca_m0_task_indices()
        assembly_ids = runtime.get_faca_m0_assembly_ids()

        if runtime.cfg.scene.replicate_physics:
            raise RuntimeError("Heterogeneous scene unexpectedly has replicate_physics=True.")
        if len(set(assembly_ids)) != len(contract["assembly_ids"]):
            raise RuntimeError(f"Not every requested task was assigned: {assembly_ids}")
        expected_indices = torch.arange(runtime.num_envs, device=runtime.device) % int(contract["num_tasks"])
        if not torch.equal(task_indices, expected_indices):
            raise RuntimeError(f"Unexpected env-to-task assignment: {task_indices.tolist()}")
        if tuple(m0_inputs["state"].shape) != (runtime.num_envs, 14):
            raise RuntimeError(f"Unexpected M0 state shape: {tuple(m0_inputs['state'].shape)}")
        if tuple(m0_inputs["task"].shape) != (runtime.num_envs, 6):
            raise RuntimeError(f"Unexpected M0 task shape: {tuple(m0_inputs['task'].shape)}")
        if not torch.isfinite(m0_inputs["state"]).all() or not torch.isfinite(m0_inputs["task"]).all():
            raise RuntimeError("M0 state/task contains NaN or Inf.")

        action = torch.zeros((runtime.num_envs, 6), dtype=torch.float32, device=runtime.device)
        reward = None
        for _ in range(args.steps):
            observations, reward, _, _, _ = env.step(action)
        if reward is not None and not torch.isfinite(reward).all():
            raise RuntimeError("Reward contains NaN or Inf.")

        print("[SRSA multitask smoke] PASS", flush=True)
        print(f"  assembly_ids={list(contract['assembly_ids'])}", flush=True)
        print(f"  env_assignment={list(assembly_ids)}", flush=True)
        print(f"  task_indices={task_indices.tolist()}", flush=True)
        print(f"  state_shape={tuple(m0_inputs['state'].shape)} task_shape={tuple(m0_inputs['task'].shape)}", flush=True)
        print(f"  reward_mean={float(reward.mean().item()) if reward is not None else 'n/a'}", flush=True)
    finally:
        env.close()


print("[SRSA multitask smoke] entering main", flush=True)
exit_code = 0
try:
    try:
        main()
    except BaseException as exc:
        print(f"[SRSA multitask smoke] FAILED: {type(exc).__name__}: {exc}", flush=True)
        exit_code = 1
finally:
    simulation_app.close()
if exit_code:
    # SimulationApp's shutdown hooks may normalize uncaught exceptions to a
    # successful process exit.  A smoke test must remain CI-safe.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)

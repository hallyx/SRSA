#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sequentially fine-tune target SRSA assembly ids from a task-parameterized 00186 policy."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


TASK_PARAM_ENV_DEFAULTS = {
    "SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER": "1",
    "SRSA_ENABLE_FLANGE_FORCE_SENSOR": "1",
    "SRSA_FLANGE_FORCE_SENSOR_SOURCE": "held_sensor",
    "SRSA_TASK_PARAM_OBS_MODE": "task_vec",
    "SRSA_AXIAL_FIXED_PLUG_SCALE": "1",
    "SRSA_AXIAL_CLEARANCE_BASE": "0.000114",
    "SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATES": "0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0",
    "SRSA_AXIAL_CLEARANCE_JITTER_RATIO": "0.10",
    "SRSA_AXIAL_DEPTH_BASE": "0.015",
    "SRSA_AXIAL_DEPTH_JITTER_RATIO": "0.10",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def _parse_target_ids(raw_ids: str) -> list[str]:
    normalized = raw_ids.replace("\n", ",").replace(";", ",")
    target_ids = [item.strip() for item in normalized.split(",") if item.strip()]
    deduped = list(dict.fromkeys(target_ids))
    if not deduped:
        raise ValueError("--target_ids must contain at least one assembly id.")
    return deduped


def _latest_file(paths) -> Path | None:
    candidates = [path for path in paths if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def _find_base_checkpoint(repo_root: Path, logs_root: Path, source_id: str) -> Path:
    preferred = _latest_file(logs_root.glob(f"*{source_id}*/nn/Assembly.pth"))
    if preferred is not None:
        return preferred

    latest_best = _latest_file(logs_root.glob("*/nn/Assembly.pth"))
    if latest_best is not None:
        return latest_best

    latest_last = _latest_file(logs_root.glob("*/nn/last_*.pth"))
    if latest_last is not None:
        return latest_last

    raise FileNotFoundError(
        "Unable to find a base checkpoint. Searched in "
        f"{logs_root} for '*{source_id}*/nn/Assembly.pth', '*/nn/Assembly.pth', and '*/nn/last_*.pth'. "
        "Pass --base_checkpoint to specify it explicitly."
    )


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in TASK_PARAM_ENV_DEFAULTS.items():
        env.setdefault(key, value)
    return env


def _build_command(args, repo_root: Path, target_id: str, checkpoint: Path) -> list[str]:
    run_name = f"{args.run_prefix}{target_id}_seed{args.seed}"
    command = [
        sys.executable,
        str(repo_root / "source" / "SRSA" / "SRSA" / "tasks" / "direct" / "srsa" / "run_w_id.py"),
        "--sparse",
        "--sil",
        "--no_sbc",
        "--task_param_obs",
        "--task_param_obs_mode",
        args.task_param_obs_mode,
        "--assembly_id",
        target_id,
        "--train",
        "--num_envs",
        str(args.num_envs),
        "--headless",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--checkpoint",
        str(checkpoint),
        "--load_mode",
        args.load_mode,
        "--experiment_name",
        run_name,
    ]
    if args.max_iterations is not None:
        command.extend(["--max_iterations", str(args.max_iterations)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune target SRSA assembly ids sequentially from a task-parameterized 00186 checkpoint."
    )
    parser.add_argument("--target_ids", required=True, help="Comma-separated target assembly ids, e.g. 01036,01041.")
    parser.add_argument("--source_id", default="00186", help="Source task id used when auto-searching checkpoints.")
    parser.add_argument("--base_checkpoint", default=None, help="Optional explicit 00186 base checkpoint path.")
    parser.add_argument("--logs_root", default="logs/rl_games/Assembly", help="RL-Games log root for checkpoint search.")
    parser.add_argument("--device", default="cuda:0", help="Simulation and RL device.")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments per fine-tune run.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Optional per-task training iteration budget.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for all target fine-tune runs.")
    parser.add_argument("--load_mode", default="actor", help="Checkpoint load mode passed to run_w_id.py.")
    parser.add_argument(
        "--task_param_obs_mode",
        choices=("task_vec", "legacy"),
        default="task_vec",
        help="Task-parameter policy obs format.",
    )
    parser.add_argument(
        "--run_prefix",
        default="finetune_00186_to_",
        help="Prefix for per-target RL-Games experiment names.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without launching training.")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue with later target ids if one fine-tune run fails.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    target_ids = _parse_target_ids(args.target_ids)
    logs_root = _resolve_path(args.logs_root, repo_root)
    if args.base_checkpoint:
        checkpoint = _resolve_path(args.base_checkpoint, repo_root)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Base checkpoint does not exist: {checkpoint}")
    else:
        try:
            checkpoint = _find_base_checkpoint(repo_root, logs_root, args.source_id)
        except FileNotFoundError:
            if not args.dry_run:
                raise
            checkpoint = Path(f"<auto_{args.source_id}_checkpoint_not_found>")
            print(
                "[finetune] warning: no base checkpoint found; dry-run will print commands with a placeholder.",
                flush=True,
            )

    env = _build_env()
    print(f"[finetune] base checkpoint: {checkpoint}", flush=True)
    print(f"[finetune] target ids: {', '.join(target_ids)}", flush=True)

    failures = []
    for target_id in target_ids:
        command = _build_command(args, repo_root, target_id, checkpoint)
        print(f"[finetune] {target_id}: {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=repo_root, env=env, check=False)
        if result.returncode != 0:
            failures.append((target_id, result.returncode))
            if not args.continue_on_error:
                return result.returncode

    if failures:
        for target_id, returncode in failures:
            print(f"[finetune] failed target {target_id} with return code {returncode}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

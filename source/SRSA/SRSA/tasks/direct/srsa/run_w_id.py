# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import subprocess
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../.."))


def _srsa_root() -> str:
    return os.path.join(_repo_root(), "SRSA")


def _resolve_local_file_arg(path: str | None) -> str | None:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    for base_dir in (_repo_root(), _srsa_root()):
        candidate = os.path.join(base_dir, path)
        if os.path.exists(candidate):
            return candidate
    return path


def _candidate_isaaclab_roots() -> list[str]:
    candidates = []
    env_root = os.environ.get("FULL_PATH_TO_ISAACLAB", "").strip()
    if env_root:
        candidates.append(env_root)
    home_dir = os.path.expanduser("~")
    candidates.extend(
        [
            os.path.join(home_dir, "IsaacLab"),
            "/home/gpuserver/IsaacLab",
            os.path.join(_repo_root(), "..", "IsaacLab"),
        ]
    )
    return candidates


def _resolve_isaaclab_launcher() -> str:
    launcher_name = "isaaclab.bat" if sys.platform.startswith("win") else "isaaclab.sh"
    for root in _candidate_isaaclab_roots():
        root = os.path.abspath(root)
        launcher_path = os.path.join(root, launcher_name)
        if os.path.isfile(launcher_path):
            return launcher_path
    searched = ", ".join(os.path.abspath(path) for path in _candidate_isaaclab_roots())
    raise FileNotFoundError(
        f"Unable to find {launcher_name}. Set FULL_PATH_TO_ISAACLAB or place IsaacLab at one of: {searched}"
    )


def _resolve_task_id(sil: bool, sparse: bool) -> str:
    if sil:
        return "Assembly-Sparse-Sil-v0" if sparse else "Assembly-Direct-Sil-v0"
    return "Assembly-Sparse-v0" if sparse else "Assembly-Direct-v0"


def main():
    parser = argparse.ArgumentParser(description="Launch SRSA assembly train/eval with runtime overrides.")
    parser.add_argument("--assembly_id", type=str, required=True, help="Assembly id to evaluate or train on.")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for evaluation.")
    parser.add_argument("--load_mode", type=str, default="actor", help="Load checkpoint mode for training.")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed used for training.")
    parser.add_argument("--train", action="store_true", help="Run training mode.")
    parser.add_argument("--sil", action="store_true", help="Use self-imitation learning.")
    parser.add_argument("--sparse", action="store_true", help="Use sparse reward environment.")
    parser.add_argument("--no_sbc", action="store_true", help="Disable the SBC curriculum.")
    parser.add_argument("--log_eval", action="store_true", help="Write evaluation logs to HDF5.")
    parser.add_argument(
        "--num_eval_trials",
        type=int,
        default=100,
        help="Number of evaluation episodes to record when --log_eval is enabled.",
    )
    parser.add_argument(
        "--vision_noise",
        type=float,
        default=0.0,
        help="Per-episode socket-frame XY bias std in meters.",
    )
    parser.add_argument(
        "--vision_jitter",
        type=float,
        default=0.0,
        help="Per-step socket-frame XY jitter std in meters.",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode.")
    args = parser.parse_args()

    if not args.train and not args.checkpoint:
        raise ValueError("No checkpoint provided for evaluation.")
    if not args.train:
        args.no_sbc = True

    task = _resolve_task_id(args.sil, args.sparse)
    checkpoint_arg = _resolve_local_file_arg(args.checkpoint)

    env = os.environ.copy()
    env["SRSA_ASSEMBLY_ID"] = args.assembly_id
    env["SRSA_IF_SBC"] = "0" if args.no_sbc else "1"
    env["SRSA_IF_LOGGING_EVAL"] = "1" if args.log_eval else "0"
    env["SRSA_EVAL_FILENAME"] = f"evaluation_{args.assembly_id}.h5"
    env["SRSA_NUM_EVAL_TRIALS"] = str(args.num_eval_trials)
    env["VISION_NOISE_XY_STD"] = str(float(args.vision_noise))
    env["VISION_NOISE_XY_JITTER_STD"] = str(float(args.vision_jitter))

    srsa_root = _srsa_root()
    srsa_python_root = os.path.join(srsa_root, "source", "SRSA")
    rl_games_python_root = os.path.join(srsa_root, "rl_games_sil")
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [srsa_python_root, rl_games_python_root]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    launcher_path = _resolve_isaaclab_launcher()

    command = [launcher_path, "-p"]
    if args.train:
        command.extend(
            [
                os.path.join(srsa_root, "scripts", "rl_games", "train.py"),
                f"--task={task}",
                f"--load_mode={args.load_mode}",
                f"--seed={args.seed}",
            ]
        )
    else:
        command.extend(
            [
                os.path.join(srsa_root, "scripts", "rl_games", "play.py"),
                f"--task={task}",
            ]
        )

    command.append(f"--num_envs={args.num_envs}")

    if checkpoint_arg:
        command.append(f"--checkpoint={checkpoint_arg}")

    if args.headless:
        command.append("--headless")

    subprocess.run(command, cwd=srsa_root, env=env, check=True)


if __name__ == "__main__":
    main()

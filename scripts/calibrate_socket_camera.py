#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch SRSA play mode and save a socket-relative camera offset with one key press."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the SRSA viewer camera relative to the current socket.")
    parser.add_argument("--assembly_id", type=str, required=True, help="Assembly id used for calibration.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint used for play-mode inference.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Simulation/RL device.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Use 1 for calibration.")
    parser.add_argument("--profile", type=str, default="camera_profiles/socket_camera_offset.json")
    parser.add_argument("--key", type=str, default="C", help="Keyboard key that saves the camera profile.")
    parser.add_argument("--direct", action="store_true", help="Use Assembly-Direct-v0 instead of Assembly-Sparse-v0.")
    parser.add_argument("--sil", action="store_true", help="Use the SIL task entry point.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to run_w_id.py after a '--' separator.",
    )
    args = parser.parse_args()

    srsa_root = _repo_root()
    run_w_id = os.path.join(srsa_root, "source", "SRSA", "SRSA", "tasks", "direct", "srsa", "run_w_id.py")
    command = [
        sys.executable,
        run_w_id,
        "--assembly_id",
        args.assembly_id,
        "--checkpoint",
        args.checkpoint,
        "--num_envs",
        str(args.num_envs),
        "--device",
        args.device,
        "--calibrate_socket_camera",
        "--socket_camera_file",
        args.profile,
        "--socket_camera_key",
        args.key,
    ]
    if not args.direct:
        command.append("--sparse")
    if args.sil:
        command.append("--sil")
    if args.extra_args:
        command.extend(arg for arg in args.extra_args if arg != "--")

    print("[INFO] Launching socket camera calibration.")
    print(f"[INFO] Move the Omniverse viewport, then press '{args.key.upper()}' to save: {args.profile}")
    subprocess.run(command, cwd=srsa_root, check=True)


if __name__ == "__main__":
    main()

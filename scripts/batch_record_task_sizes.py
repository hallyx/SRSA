#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch-record SRSA play videos for multiple task sizes in the same assembly scene."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys


DEFAULT_SIZE_TEMPLATES = "0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse_template_list(value: str) -> list[tuple[float, float]]:
    normalized = value.strip()
    for char in "()[]{}":
        normalized = normalized.replace(char, "")
    parts = [item.strip() for item in re.split(r"[,;\s]+", normalized) if item.strip()]
    templates = []
    for part in parts:
        pair = [item.strip() for item in re.split(r"[:/xX]", part) if item.strip()]
        if len(pair) != 2:
            raise ValueError(f"Invalid size template {part!r}; expected CLEARANCE_MULT:DEPTH_MULT.")
        templates.append((float(pair[0]), float(pair[1])))
    if not templates:
        raise ValueError("At least one size template is required.")
    return templates


def _safe_number(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value).strip("_")


def _append_optional_env(env: dict[str, str], name: str, value: str | None) -> None:
    if value is not None and value != "":
        env[name] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-record SRSA videos over clearance/depth task sizes.")
    parser.add_argument("--assembly_id", type=str, required=True, help="Assembly id used for every recording.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint used for play-mode inference.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Simulation/RL device.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to instantiate for recording.")
    parser.add_argument("--video_length", type=int, default=300, help="Recorded video length in env steps.")
    parser.add_argument(
        "--templates",
        type=str,
        default=DEFAULT_SIZE_TEMPLATES,
        help="Size templates as CLEARANCE_MULT:DEPTH_MULT pairs, separated by ';' or ','.",
    )
    parser.add_argument("--clearance_base", type=float, default=0.000114, help="Base diametral clearance in meters.")
    parser.add_argument("--depth_base", type=float, default=0.015, help="Base insertion depth in meters.")
    parser.add_argument("--clearance_jitter_ratio", type=float, default=0.0)
    parser.add_argument("--depth_jitter_ratio", type=float, default=0.0)
    parser.add_argument("--init_error_xy_range", type=str, default=None, help="Forwarded to SRSA_AXIAL_INIT_ERROR_XY_RANGE.")
    parser.add_argument("--init_error_z_range", type=str, default=None, help="Forwarded to SRSA_AXIAL_INIT_ERROR_Z_RANGE.")
    parser.add_argument("--init_error_yaw_range", type=str, default=None, help="Forwarded to SRSA_AXIAL_INIT_ERROR_YAW_RANGE.")
    parser.add_argument("--visual_noise_xy_range", type=str, default=None, help="Forwarded to SRSA_AXIAL_VISUAL_NOISE_XY_RANGE.")
    parser.add_argument("--visual_noise_z_range", type=str, default=None, help="Forwarded to SRSA_AXIAL_VISUAL_NOISE_Z_RANGE.")
    parser.add_argument("--video_prefix", type=str, default=None, help="Video filename prefix. Defaults to assembly_id.")
    parser.add_argument(
        "--socket_camera_file",
        type=str,
        default="camera_profiles/socket_camera_offset.json",
        help="Socket-relative camera profile used for consistent framing.",
    )
    parser.add_argument("--no_socket_camera_follow", action="store_true", help="Disable socket-relative camera follow.")
    parser.add_argument("--task_param_obs", action="store_true", help="Pass task parameters to policy observations.")
    parser.add_argument("--task_param_obs_mode", choices=("task_vec", "legacy"), default="task_vec")
    parser.add_argument(
        "--no_geometry_scale",
        action="store_true",
        help="Keep original USD asset scale while still varying/logging task parameters.",
    )
    parser.add_argument("--force_diagnostics", action="store_true", help="Enable force/contact diagnostics.")
    parser.add_argument("--flange_force_source", choices=("sensor", "held_sensor", "auto", "asset"), default=None)
    parser.add_argument("--flange_force_threshold", type=float, default=None)
    parser.add_argument("--direct", action="store_true", help="Use Assembly-Direct-v0 instead of Assembly-Sparse-v0.")
    parser.add_argument("--sil", action="store_true", help="Use the SIL task entry point.")
    parser.add_argument("--interactive", action="store_true", help="Run with a visible Omniverse window.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to run_w_id.py after a '--' separator.",
    )
    args = parser.parse_args()

    srsa_root = _repo_root()
    run_w_id = os.path.join(srsa_root, "source", "SRSA", "SRSA", "tasks", "direct", "srsa", "run_w_id.py")
    templates = _parse_template_list(args.templates)
    video_prefix = _safe_name(args.video_prefix or args.assembly_id)

    for index, (clearance_mult, depth_mult) in enumerate(templates, start=1):
        template = f"{clearance_mult:g}:{depth_mult:g}"
        video_name = f"{video_prefix}_c{_safe_number(clearance_mult)}_d{_safe_number(depth_mult)}"

        env = os.environ.copy()
        env["SRSA_NEWT_OBS"] = "0"
        env["SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER"] = "1"
        env["SRSA_AXIAL_FIXED_PLUG_SCALE"] = "1"
        env["SRSA_AXIAL_CLEARANCE_BASE"] = str(float(args.clearance_base))
        env["SRSA_AXIAL_DEPTH_BASE"] = str(float(args.depth_base))
        env["SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATES"] = template
        env["SRSA_AXIAL_CLEARANCE_JITTER_RATIO"] = str(float(args.clearance_jitter_ratio))
        env["SRSA_AXIAL_DEPTH_JITTER_RATIO"] = str(float(args.depth_jitter_ratio))
        if args.no_geometry_scale:
            env["SRSA_TASK_PARAM_GEOMETRY_SCALE"] = "0"
        _append_optional_env(env, "SRSA_AXIAL_INIT_ERROR_XY_RANGE", args.init_error_xy_range)
        _append_optional_env(env, "SRSA_AXIAL_INIT_ERROR_Z_RANGE", args.init_error_z_range)
        _append_optional_env(env, "SRSA_AXIAL_INIT_ERROR_YAW_RANGE", args.init_error_yaw_range)
        _append_optional_env(env, "SRSA_AXIAL_VISUAL_NOISE_XY_RANGE", args.visual_noise_xy_range)
        _append_optional_env(env, "SRSA_AXIAL_VISUAL_NOISE_Z_RANGE", args.visual_noise_z_range)

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
            "--video",
            "--video_length",
            str(args.video_length),
            "--video_name",
            video_name,
        ]
        if not args.direct:
            command.append("--sparse")
        if args.sil:
            command.append("--sil")
        if not args.interactive:
            command.append("--headless")
        if not args.no_socket_camera_follow:
            command.extend(["--socket_camera_follow", "--socket_camera_file", args.socket_camera_file])
        if args.task_param_obs:
            command.extend(["--task_param_obs", "--task_param_obs_mode", args.task_param_obs_mode])
        if args.force_diagnostics:
            command.append("--force_diagnostics")
        if args.flange_force_source is not None:
            command.extend(["--flange_force_source", args.flange_force_source])
        if args.flange_force_threshold is not None:
            command.extend(["--flange_force_threshold", str(float(args.flange_force_threshold))])
        if args.extra_args:
            command.extend(arg for arg in args.extra_args if arg != "--")

        print(f"[INFO] ({index}/{len(templates)}) Recording template {template} -> {video_name}.mp4")
        print("[INFO] Command:", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=srsa_root, env=env, check=True)


if __name__ == "__main__":
    main()

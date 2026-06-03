# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import random
import subprocess
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))


def _srsa_root() -> str:
    return _repo_root()


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


def _sample_optional_range(rng: random.Random, bounds: tuple[float, float] | list[float] | None) -> float | None:
    if not bounds:
        return None
    lower, upper = float(bounds[0]), float(bounds[1])
    if upper < lower:
        lower, upper = upper, lower
    return rng.uniform(lower, upper)


def _resolve_task_param_values(args) -> dict[str, float | str | int]:
    rng_seed = args.sample_seed
    if rng_seed is None and args.seed is not None and args.seed >= 0:
        rng_seed = int(args.seed)
    rng = random.Random(rng_seed)

    resolved = {
        "task_family_name": args.task_family_name,
        "task_family_id": args.task_family_id,
        "plug_diameter": args.plug_diameter
        if args.plug_diameter is not None
        else _sample_optional_range(rng, args.sample_plug_diameter),
        "hole_diameter": args.hole_diameter
        if args.hole_diameter is not None
        else _sample_optional_range(rng, args.sample_hole_diameter),
        "clearance": args.clearance if args.clearance is not None else _sample_optional_range(rng, args.sample_clearance),
        "clearance_ratio": args.clearance_ratio
        if args.clearance_ratio is not None
        else _sample_optional_range(rng, args.sample_clearance_ratio),
        "insertion_depth": args.insertion_depth
        if args.insertion_depth is not None
        else _sample_optional_range(rng, args.sample_insertion_depth),
        "success_pos_tol": args.success_pos_tol
        if args.success_pos_tol is not None
        else _sample_optional_range(rng, args.sample_success_pos_tol),
    }
    return {key: value for key, value in resolved.items() if value is not None}


def main():
    parser = argparse.ArgumentParser(description="Launch SRSA assembly train/eval with runtime overrides.")
    parser.add_argument("--assembly_id", type=str, required=True, help="Assembly id to evaluate or train on.")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for evaluation.")
    parser.add_argument("--load_mode", type=str, default="actor", help="Load checkpoint mode for training.")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Simulation and RL device, e.g. cuda:0.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed used for training.")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
    parser.add_argument("--experiment_name", type=str, default=None, help="RL-Games experiment/run directory name.")
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
    parser.add_argument("--video", action="store_true", help="Record an RGB video from the configured SRSA camera.")
    parser.add_argument(
        "--video_name",
        type=str,
        default=None,
        help="Override the final play video filename without extension.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=200,
        help="Length of the recorded video in environment steps.",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=2000,
        help="Training step interval between video recordings when --train --video is used.",
    )
    parser.add_argument(
        "--camera_eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override recording camera eye position relative to the selected env origin.",
    )
    parser.add_argument(
        "--camera_lookat",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override recording camera target position relative to the selected env origin.",
    )
    parser.add_argument(
        "--camera_resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Override recording camera resolution.",
    )
    parser.add_argument(
        "--camera_env_index",
        type=int,
        default=None,
        help="Environment index used as the recording camera origin.",
    )
    parser.add_argument(
        "--calibrate_socket_camera",
        action="store_true",
        help="In play mode, press a key to save the current viewer eye as a socket-relative camera offset.",
    )
    parser.add_argument(
        "--socket_camera_follow",
        action="store_true",
        help="In play mode, apply the saved socket-relative camera offset to the current socket position.",
    )
    parser.add_argument(
        "--socket_camera_file",
        type=str,
        default="camera_profiles/socket_camera_offset.json",
        help="JSON file used for socket-relative camera calibration.",
    )
    parser.add_argument(
        "--socket_camera_key",
        type=str,
        default="C",
        help="Keyboard key used to save the socket-relative camera offset during calibration.",
    )
    parser.add_argument(
        "--socket_camera_env_index",
        type=int,
        default=None,
        help="Environment index whose socket is used for socket-relative camera calibration/following.",
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
    parser.add_argument(
        "--task_family_name",
        type=str,
        default=None,
        help="Named fit family to use, e.g. normal_fit, loose_fit, or tight_fit.",
    )
    parser.add_argument(
        "--task_family_id",
        type=int,
        default=None,
        help="Numeric fit-family id override.",
    )
    parser.add_argument("--plug_diameter", type=float, default=None, help="Explicit plug diameter in meters.")
    parser.add_argument("--hole_diameter", type=float, default=None, help="Explicit hole diameter in meters.")
    parser.add_argument("--clearance", type=float, default=None, help="Explicit diametral clearance in meters.")
    parser.add_argument(
        "--clearance_ratio",
        type=float,
        default=None,
        help="Explicit clearance ratio relative to plug diameter.",
    )
    parser.add_argument("--insertion_depth", type=float, default=None, help="Explicit insertion depth in meters.")
    parser.add_argument(
        "--success_pos_tol",
        type=float,
        default=None,
        help="Explicit success position tolerance in meters.",
    )
    parser.add_argument(
        "--sample_plug_diameter",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Uniformly sample plug diameter once for this run.",
    )
    parser.add_argument(
        "--sample_hole_diameter",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Uniformly sample hole diameter once for this run.",
    )
    parser.add_argument(
        "--sample_clearance",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Uniformly sample diametral clearance once for this run.",
    )
    parser.add_argument(
        "--sample_clearance_ratio",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Uniformly sample clearance ratio once for this run.",
    )
    parser.add_argument(
        "--sample_insertion_depth",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Uniformly sample insertion depth once for this run.",
    )
    parser.add_argument(
        "--sample_success_pos_tol",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Uniformly sample success position tolerance once for this run.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="Seed used for continuous task-parameter sampling.",
    )
    parser.add_argument(
        "--task_param_obs",
        action="store_true",
        help="Append task parameters to policy observations. This changes the observation dimension.",
    )
    parser.add_argument(
        "--task_param_obs_mode",
        type=str,
        choices=("task_vec", "legacy"),
        default=None,
        help="Task-parameter policy obs format: Newt-style 6D task_vec or legacy 9D tensor.",
    )
    parser.add_argument(
        "--disable_task_param_obs",
        action="store_true",
        help="Compatibility flag; task-param observations are disabled by default.",
    )
    parser.add_argument(
        "--force_diagnostics",
        action="store_true",
        help="Enable force/contact diagnostic logging without changing the policy observation dimension.",
    )
    parser.add_argument(
        "--flange_force_obs",
        action="store_true",
        help="Append flange force to policy observations. This changes the observation dimension.",
    )
    parser.add_argument(
        "--flange_force_source",
        type=str,
        choices=("sensor", "held_sensor", "auto", "asset"),
        default=None,
        help="Force source used by SRSA diagnostics and optional force observations.",
    )
    parser.add_argument(
        "--flange_force_threshold",
        type=float,
        default=None,
        help="Contact threshold in Newtons used for contact/jam diagnostics.",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode.")
    args = parser.parse_args()

    if not args.train and not args.checkpoint:
        raise ValueError("No checkpoint provided for evaluation.")
    if not args.train:
        args.no_sbc = True
    if args.train and (args.calibrate_socket_camera or args.socket_camera_follow):
        raise ValueError("Socket-relative camera calibration/following is only supported in play/eval mode.")
    if args.calibrate_socket_camera and args.headless:
        raise ValueError("Socket camera calibration needs an interactive Omniverse window; remove --headless.")

    task = _resolve_task_id(args.sil, args.sparse)
    checkpoint_arg = _resolve_local_file_arg(args.checkpoint)
    task_param_values = _resolve_task_param_values(args)
    task_param_active = bool(task_param_values)

    env = os.environ.copy()
    env["SRSA_ASSEMBLY_ID"] = args.assembly_id
    env["SRSA_IF_SBC"] = "0" if args.no_sbc else "1"
    env["SRSA_IF_LOGGING_EVAL"] = "1" if args.log_eval else "0"
    env["SRSA_EVAL_FILENAME"] = os.environ.get("SRSA_EVAL_FILENAME", f"evaluation_{args.assembly_id}.h5")
    env["SRSA_NUM_EVAL_TRIALS"] = str(args.num_eval_trials)
    env["VISION_NOISE_XY_STD"] = str(float(args.vision_noise))
    env["VISION_NOISE_XY_JITTER_STD"] = str(float(args.vision_jitter))
    env["SRSA_TASK_PARAM_OBS"] = "1" if args.task_param_obs and not args.disable_task_param_obs else "0"
    if args.force_diagnostics or args.flange_force_obs:
        env["SRSA_ENABLE_FLANGE_FORCE_SENSOR"] = "1"
        env["SRSA_FLANGE_FORCE_SENSOR_OBS"] = "1" if args.flange_force_obs else "0"
    if args.flange_force_source is not None:
        env["SRSA_FLANGE_FORCE_SENSOR_SOURCE"] = args.flange_force_source
    if args.flange_force_threshold is not None:
        env["SRSA_FLANGE_FORCE_SENSOR_FORCE_THRESHOLD"] = str(float(args.flange_force_threshold))
    if args.task_param_obs_mode is not None:
        env["SRSA_TASK_PARAM_OBS_MODE"] = args.task_param_obs_mode
    if args.camera_eye is not None:
        env["SRSA_CAMERA_EYE"] = ",".join(str(value) for value in args.camera_eye)
    if args.camera_lookat is not None:
        env["SRSA_CAMERA_LOOKAT"] = ",".join(str(value) for value in args.camera_lookat)
    if args.camera_resolution is not None:
        env["SRSA_CAMERA_RESOLUTION"] = ",".join(str(value) for value in args.camera_resolution)
    if args.camera_env_index is not None:
        env["SRSA_CAMERA_ENV_INDEX"] = str(args.camera_env_index)

    env_var_map = {
        "task_family_name": "SRSA_TASK_FAMILY_NAME",
        "task_family_id": "SRSA_TASK_FAMILY_ID",
        "plug_diameter": "SRSA_PLUG_DIAMETER",
        "hole_diameter": "SRSA_HOLE_DIAMETER",
        "clearance": "SRSA_CLEARANCE",
        "clearance_ratio": "SRSA_CLEARANCE_RATIO",
        "insertion_depth": "SRSA_INSERTION_DEPTH",
        "success_pos_tol": "SRSA_SUCCESS_POS_TOL",
    }
    for key, env_name in env_var_map.items():
        if key in task_param_values:
            env[env_name] = str(task_param_values[key])

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
    command.append(f"--device={args.device}")

    if args.train and args.max_iterations is not None:
        command.append(f"--max_iterations={args.max_iterations}")
    if args.train and args.experiment_name:
        command.append(f"--experiment_name={args.experiment_name}")

    if checkpoint_arg:
        command.append(f"--checkpoint={checkpoint_arg}")

    if args.video:
        command.append("--video")
        command.append(f"--video_length={args.video_length}")
        if args.train:
            command.append(f"--video_interval={args.video_interval}")
        elif args.video_name:
            command.append(f"--video_name={args.video_name}")

    if not args.train:
        if args.calibrate_socket_camera:
            command.append("--calibrate_socket_camera")
        if args.socket_camera_follow:
            command.append("--socket_camera_follow")
        if args.calibrate_socket_camera or args.socket_camera_follow:
            command.append(f"--socket_camera_file={args.socket_camera_file}")
            command.append(f"--socket_camera_key={args.socket_camera_key}")
            if args.socket_camera_env_index is not None:
                command.append(f"--socket_camera_env_index={args.socket_camera_env_index}")

    if args.headless:
        command.append("--headless")

    subprocess.run(command, cwd=srsa_root, env=env, check=True)


if __name__ == "__main__":
    main()

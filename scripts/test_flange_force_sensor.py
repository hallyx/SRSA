# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the SRSA Franka flange force sensor."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time
import traceback


def _prepend_isaaclab_python_roots() -> None:
    isaaclab_root = os.environ.get("FULL_PATH_TO_ISAACLAB", "").strip() or "/home/gpuserver/IsaacLab"
    source_root = os.path.join(isaaclab_root, "source")
    package_roots = (
        "isaaclab",
        "isaaclab_assets",
        "isaaclab_rl",
        "isaaclab_tasks",
    )
    for package_root in reversed(package_roots):
        path = os.path.join(source_root, package_root)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_prepend_isaaclab_python_roots()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Test SRSA flange force sensor readings.")
parser.add_argument("--task", type=str, default="Assembly-Direct-v0", help="Gym task id to instantiate.")
parser.add_argument("--assembly_id", type=str, default=None, help="Optional SRSA assembly id override.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=40, help="Number of simulation steps to run.")
parser.add_argument("--print_every", type=int, default=10, help="Print force statistics every N steps.")
parser.add_argument("--task_family_name", type=str, default=None, help="Optional task family name override.")
parser.add_argument("--task_family_id", type=int, default=None, help="Optional task family id override.")
parser.add_argument("--plug_diameter", type=float, default=None, help="Optional plug diameter override.")
parser.add_argument("--hole_diameter", type=float, default=None, help="Optional hole diameter override.")
parser.add_argument("--clearance", type=float, default=None, help="Optional diametral clearance override.")
parser.add_argument("--clearance_ratio", type=float, default=None, help="Optional clearance ratio override.")
parser.add_argument("--insertion_depth", type=float, default=None, help="Optional insertion depth override.")
parser.add_argument("--success_pos_tol", type=float, default=None, help="Optional success position tolerance override.")
parser.add_argument(
    "--task_param_obs",
    action="store_true",
    help="Append task parameters to policy observations for observation-dimension checks.",
)
parser.add_argument(
    "--task_param_obs_mode",
    type=str,
    choices=("task_vec", "legacy"),
    default=None,
    help="Task-parameter policy obs format: Newt-style 6D task_vec or legacy 9D tensor.",
)
parser.add_argument(
    "--expected_policy_obs_dim",
    type=int,
    default=None,
    help="Fail if the policy observation dimension differs from this value.",
)
parser.add_argument(
    "--newt_obs",
    action="store_true",
    help="Enable and validate the Newt observation dict: state/state_mask/task/task_id/action_mask.",
)
parser.add_argument(
    "--expected_newt_state_dim",
    type=int,
    default=128,
    help="Expected Newt state and state_mask dimension.",
)
parser.add_argument(
    "--expected_newt_task_dim",
    type=int,
    default=6,
    help="Expected Newt task vector dimension.",
)
parser.add_argument(
    "--expected_newt_action_dim",
    type=int,
    default=16,
    help="Expected Newt padded action/action_mask dimension.",
)
parser.add_argument(
    "--body_name",
    type=str,
    default="panda_hand",
    help="Robot body where the ContactSensor is attached.",
)
parser.add_argument(
    "--obs_frame",
    type=str,
    choices=("socket", "world"),
    default="socket",
    help="Frame used for the 3-D force observation appended to policy obs.",
)
parser.add_argument(
    "--force_source",
    type=str,
    choices=("held_sensor", "held_asset", "sensor", "auto"),
    default="held_sensor",
    help="Force source: held asset ContactSensor, held asset net contact force, flange body ContactSensor, or auto.",
)
parser.add_argument("--obs_scale", type=float, default=50.0, help="Scale divisor for policy force observations.")
parser.add_argument("--threshold", type=float, default=1.0, help="Force norm threshold for flange_force_flag.")
parser.add_argument(
    "--action_mode",
    type=str,
    choices=("zero", "random", "down"),
    default="zero",
    help="Action pattern used during the test.",
)
parser.add_argument("--random_action_scale", type=float, default=0.05, help="Scale for random action mode.")
parser.add_argument(
    "--step_mode",
    type=str,
    choices=("physics", "env"),
    default="physics",
    help="Use physics-only stepping to avoid reward/SoftDTW, or full env.step.",
)
parser.add_argument(
    "--down_action",
    type=float,
    default=-0.5,
    help="Normalized z action used in down mode.",
)
parser.add_argument(
    "--require_nonzero",
    action="store_true",
    help="Fail if force norm never exceeds --nonzero_tol.",
)
parser.add_argument("--nonzero_tol", type=float, default=1.0e-4, help="Nonzero force norm tolerance.")
parser.add_argument(
    "--max_abs_force_warn",
    type=float,
    default=1.0e4,
    help="Warn when any force component exceeds this absolute value.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply, quat_conjugate


def _prepend_local_python_roots() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    srsa_root = os.path.abspath(os.path.join(script_dir, ".."))
    python_roots = [
        os.path.join(srsa_root, "source", "SRSA"),
        os.path.join(srsa_root, "rl_games_sil"),
    ]
    for path in reversed(python_roots):
        if path not in sys.path:
            sys.path.insert(0, path)


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _expect_newt_obs() -> bool:
    return bool(args_cli.newt_obs) or _read_bool_env("SRSA_NEWT_OBS", False)


def _as_obs_payload(obs):
    if isinstance(obs, tuple) and len(obs) >= 1:
        return obs[0]
    return obs


def _as_policy_tensor(obs):
    obs = _as_obs_payload(obs)
    if isinstance(obs, dict):
        return obs.get("policy")
    return obs


def _as_newt_obs_dict(obs):
    obs = _as_obs_payload(obs)
    if isinstance(obs, dict) and "state" in obs:
        return obs
    return None


def _get_action_dim(env) -> int:
    action_space = getattr(env.unwrapped, "single_action_space", None) or getattr(env, "action_space", None)
    if action_space is not None and getattr(action_space, "shape", None):
        return int(action_space.shape[-1])
    return int(getattr(env.unwrapped.cfg, "action_space", 6))


def _make_actions(env, action_dim: int, step: int) -> torch.Tensor:
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    actions = torch.zeros((num_envs, action_dim), device=device)
    if args_cli.action_mode == "random":
        actions = torch.randn_like(actions) * float(args_cli.random_action_scale)
        return torch.clamp(actions, -1.0, 1.0)
    if args_cli.action_mode == "down" and action_dim >= 3:
        actions[:, 2] = float(args_cli.down_action)
    return actions


def _unpack_step_result(step_result):
    if isinstance(step_result, tuple) and len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, terminated | truncated, info
    if isinstance(step_result, tuple) and len(step_result) == 4:
        return step_result
    raise RuntimeError(f"Unexpected env.step result with type={type(step_result)}")


def _physics_only_step(env, actions: torch.Tensor):
    unwrapped = env.unwrapped
    actions = actions.to(unwrapped.device)
    if unwrapped.cfg.action_noise_model:
        actions = unwrapped._action_noise_model(actions)

    unwrapped._pre_physics_step(actions)
    is_rendering = unwrapped.sim.has_gui() or unwrapped.sim.has_rtx_sensors()

    for _ in range(unwrapped.cfg.decimation):
        unwrapped._sim_step_counter += 1
        unwrapped._apply_action()
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        if unwrapped._sim_step_counter % unwrapped.cfg.sim.render_interval == 0 and is_rendering:
            unwrapped.sim.render()
        unwrapped.scene.update(dt=unwrapped.physics_dt)

    unwrapped.episode_length_buf += 1
    unwrapped.common_step_counter += 1
    unwrapped._compute_intermediate_values(dt=unwrapped.physics_dt)
    unwrapped.obs_buf = unwrapped._get_observations()
    return unwrapped.obs_buf


def _force_stats(name: str, tensor: torch.Tensor) -> str:
    tensor = tensor.detach()
    return (
        f"{name}: shape={tuple(tensor.shape)} "
        f"min={tensor.min().item(): .6f} "
        f"max={tensor.max().item(): .6f} "
        f"mean={tensor.mean().item(): .6f}"
    )


def _scalar_stats(name: str, tensor: torch.Tensor) -> str:
    tensor = tensor.detach().float().reshape(-1)
    return (
        f"{name}: min={tensor.min().item(): .6f} "
        f"max={tensor.max().item(): .6f} "
        f"mean={tensor.mean().item(): .6f}"
    )


def _set_optional_env(name: str, value) -> None:
    if value is not None:
        os.environ[name] = str(value)


def _print_task_param_summary(env) -> None:
    unwrapped = env.unwrapped
    cfg_task = unwrapped.cfg_task
    print("[force-test] task parameter summary", flush=True)
    print(f"  assembly_id={getattr(cfg_task, 'assembly_id', None)}", flush=True)
    print(
        "  "
        f"use_task_family={getattr(unwrapped, 'use_task_family', None)} "
        f"use_task_param={getattr(unwrapped, 'use_task_param', None)} "
        f"task_param_obs={getattr(unwrapped, 'task_param_obs', None)} "
        f"task_param_obs_mode={getattr(unwrapped, 'task_param_obs_mode', None)}",
        flush=True,
    )
    current_params = getattr(unwrapped, "current_task_params", {}) or {}
    if current_params:
        ordered_keys = (
            "task_family_name",
            "task_family_id",
            "plug_diameter",
            "hole_diameter",
            "clearance",
            "clearance_ratio",
            "success_pos_tol",
            "insertion_depth",
            "plug_scale_xy",
            "hole_scale_xy",
        )
        for key in ordered_keys:
            if key in current_params:
                print(f"  {key}={current_params[key]}", flush=True)
    else:
        held_diameter = getattr(getattr(cfg_task, "held_asset_cfg", None), "diameter", None)
        fixed_diameter = getattr(getattr(cfg_task, "fixed_asset_cfg", None), "diameter", None)
        print(f"  base_plug_diameter={held_diameter}", flush=True)
        print(f"  base_hole_diameter={fixed_diameter}", flush=True)
        print(f"  base_success_pos_tol={getattr(cfg_task, 'close_error_thresh', None)}", flush=True)
    if hasattr(unwrapped, "disassembly_dists"):
        print(f"  {_scalar_stats('disassembly_dists', unwrapped.disassembly_dists)}", flush=True)
    if hasattr(unwrapped, "current_task_param_tensor") and unwrapped.current_task_param_tensor is not None:
        print(f"  current_task_param_tensor_shape={tuple(unwrapped.current_task_param_tensor.shape)}", flush=True)
    if hasattr(unwrapped, "current_task_vec") and unwrapped.current_task_vec is not None:
        print(f"  current_task_vec_shape={tuple(unwrapped.current_task_vec.shape)}", flush=True)


def _compute_error_metrics(env) -> dict[str, torch.Tensor]:
    unwrapped = env.unwrapped
    metrics = {}
    if hasattr(unwrapped, "gripper_goal_pos") and hasattr(unwrapped, "fingertip_midpoint_pos"):
        tcp_delta = unwrapped.gripper_goal_pos - unwrapped.fingertip_midpoint_pos
        metrics["tcp_goal_error_norm"] = torch.linalg.norm(tcp_delta, dim=-1)
        metrics["tcp_goal_error_xy"] = torch.linalg.norm(tcp_delta[:, :2], dim=-1)
        metrics["tcp_goal_error_z"] = tcp_delta[:, 2]
    if hasattr(unwrapped, "fixed_pos") and hasattr(unwrapped, "held_pos"):
        held_delta_world = unwrapped.held_pos - unwrapped.fixed_pos
        metrics["held_fixed_world_xy"] = torch.linalg.norm(held_delta_world[:, :2], dim=-1)
        metrics["held_fixed_world_z"] = held_delta_world[:, 2]
        if hasattr(unwrapped, "fixed_quat"):
            held_delta_socket = quat_apply(quat_conjugate(unwrapped.fixed_quat), held_delta_world)
            metrics["held_fixed_socket_xy"] = torch.linalg.norm(held_delta_socket[:, :2], dim=-1)
            metrics["held_fixed_socket_z"] = held_delta_socket[:, 2]
    if hasattr(unwrapped, "keypoint_dist"):
        metrics["keypoint_dist"] = unwrapped.keypoint_dist
    if hasattr(unwrapped, "ctrl_target_fingertip_midpoint_pos") and hasattr(unwrapped, "fingertip_midpoint_pos"):
        ctrl_delta = unwrapped.ctrl_target_fingertip_midpoint_pos - unwrapped.fingertip_midpoint_pos
        metrics["ctrl_target_error_norm"] = torch.linalg.norm(ctrl_delta, dim=-1)
    return metrics


def _print_error_summary(env, step: int) -> None:
    metrics = _compute_error_metrics(env)
    if not metrics:
        return
    print(f"[force-test] error step={step}", flush=True)
    for name, value in metrics.items():
        print(f"  {_scalar_stats(name, value)}", flush=True)


def _print_step_summary(env, step: int, policy_obs) -> None:
    force_world = env.unwrapped.flange_force_world
    body_force_world = env.unwrapped.flange_body_contact_force_world
    held_sensor_force_world = env.unwrapped.held_sensor_contact_force_world
    held_force_world = env.unwrapped.held_asset_contact_force_world
    force_socket = env.unwrapped.flange_force_socket
    force_obs = env.unwrapped.flange_force_obs
    force_norm = env.unwrapped.flange_force_norm
    force_flag = env.unwrapped.flange_force_flag
    print(f"[force-test] step={step}")
    print(f"  {_force_stats('world', force_world)}")
    print(f"  {_force_stats('body_sensor_world', body_force_world)}")
    print(f"  {_force_stats('held_sensor_world', held_sensor_force_world)}")
    print(f"  {_force_stats('held_asset_world', held_force_world)}")
    print(f"  {_force_stats('socket', force_socket)}")
    print(f"  {_force_stats('obs', force_obs)}")
    print(
        "  "
        f"norm: min={force_norm.min().item(): .6f} "
        f"max={force_norm.max().item(): .6f} "
        f"flag_count={int(force_flag.sum().item())}"
    )
    if isinstance(policy_obs, torch.Tensor):
        print(f"  policy_obs_shape={tuple(policy_obs.shape)}")
        if args_cli.expected_policy_obs_dim is not None and policy_obs.shape[-1] != args_cli.expected_policy_obs_dim:
            raise RuntimeError(
                f"Unexpected policy obs dim: got {policy_obs.shape[-1]}, expected {args_cli.expected_policy_obs_dim}"
            )
    _print_error_summary(env, step)


def _print_newt_obs_summary(obs) -> None:
    obs_dict = _as_newt_obs_dict(obs)
    if obs_dict is None:
        return
    state = obs_dict["state"]
    state_mask = obs_dict["state_mask"]
    task = obs_dict["task"]
    task_id = obs_dict["task_id"]
    action_mask = obs_dict["action_mask"]
    valid_state_dim = state_mask.sum(dim=-1)
    valid_action_dim = action_mask.sum(dim=-1)
    print("[force-test] newt observation summary", flush=True)
    print(f"  state_shape={tuple(state.shape)} valid_state_dim={valid_state_dim.detach().cpu().tolist()}", flush=True)
    print(f"  task_shape={tuple(task.shape)} task_id_shape={tuple(task_id.shape)}", flush=True)
    print(f"  action_mask_shape={tuple(action_mask.shape)} valid_action_dim={valid_action_dim.detach().cpu().tolist()}", flush=True)


def _print_newt_extras_summary(env) -> None:
    if not _expect_newt_obs():
        return
    extras = getattr(env.unwrapped, "extras", {}) or {}
    task_params = extras.get("task_params", {}) or {}
    print("[force-test] newt extras summary", flush=True)
    for key in (
        "plug_scale_xy",
        "hole_scale_xy",
        "diametral_clearance",
        "radial_clearance",
        "clearance_ratio",
        "target_insertion_depth",
        "insertion_depth",
    ):
        value = task_params.get(key)
        if isinstance(value, torch.Tensor):
            print(f"  {_scalar_stats(key, value)}", flush=True)
    for key in ("initial_error_pos", "initial_error_yaw", "visual_noise_local"):
        value = extras.get(key)
        if isinstance(value, torch.Tensor):
            print(f"  {_force_stats(key, value) if value.ndim == 2 and value.shape[-1] == 3 else _scalar_stats(key, value)}", flush=True)
    for key in ("success", "contact", "jam", "geometry_variant_applied"):
        value = extras.get(key)
        if isinstance(value, torch.Tensor):
            count = int(value.to(dtype=torch.bool).sum().item())
            print(f"  {key}_count={count}/{value.numel()}", flush=True)


def _validate_newt_obs(env, obs) -> None:
    if not _expect_newt_obs():
        return
    obs_dict = _as_newt_obs_dict(obs)
    if obs_dict is None:
        raise RuntimeError(f"Expected Newt obs dict, got {type(_as_obs_payload(obs))}")

    expected_keys = {"state", "state_mask", "task", "task_id", "action_mask"}
    actual_keys = set(obs_dict.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(f"Unexpected Newt obs keys: got {sorted(actual_keys)}, expected {sorted(expected_keys)}")

    num_envs = env.unwrapped.num_envs
    expected_shapes = {
        "state": (num_envs, int(args_cli.expected_newt_state_dim)),
        "state_mask": (num_envs, int(args_cli.expected_newt_state_dim)),
        "task": (num_envs, int(args_cli.expected_newt_task_dim)),
        "task_id": (num_envs, 1),
        "action_mask": (num_envs, int(args_cli.expected_newt_action_dim)),
    }
    for name, expected_shape in expected_shapes.items():
        tensor = obs_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"Newt obs {name} is not a torch.Tensor: {type(tensor)}")
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(f"Unexpected Newt obs {name} shape: got {tuple(tensor.shape)}, expected {expected_shape}")
        if name not in {"state_mask", "action_mask"} and not torch.isfinite(tensor.float()).all():
            raise RuntimeError(f"Newt obs {name} contains non-finite values")

    if obs_dict["state_mask"].dtype != torch.bool:
        raise RuntimeError(f"Newt state_mask must be bool, got {obs_dict['state_mask'].dtype}")
    if obs_dict["action_mask"].dtype != torch.bool:
        raise RuntimeError(f"Newt action_mask must be bool, got {obs_dict['action_mask'].dtype}")
    if not torch.all(obs_dict["action_mask"][:, :6]):
        raise RuntimeError("Newt action_mask must enable the first 6 SRSA action dimensions")
    if obs_dict["action_mask"].shape[-1] > 6 and torch.any(obs_dict["action_mask"][:, 6:]):
        raise RuntimeError("Newt action_mask must disable padded action dimensions after index 5")
    if torch.any(obs_dict["state_mask"].sum(dim=-1) <= 0):
        raise RuntimeError("Newt state_mask has an empty state row")


def _validate_newt_extras(env) -> None:
    if not _expect_newt_obs():
        return
    extras = getattr(env.unwrapped, "extras", None)
    if not isinstance(extras, dict):
        raise RuntimeError(f"Expected env extras dict for Newt info, got {type(extras)}")
    required_keys = {
        "task_params",
        "task_vec",
        "task_id",
        "initial_error_pos",
        "initial_error_yaw",
        "visual_noise_local",
        "success",
        "force",
        "contact",
        "depth",
        "jam",
        "geometry_variant_applied",
    }
    missing = sorted(required_keys - set(extras.keys()))
    if missing:
        raise RuntimeError(f"Missing Newt extras/info keys: {missing}")

    if not isinstance(extras["task_params"], dict):
        raise RuntimeError("extras['task_params'] must be a dict")
    if not isinstance(extras["force"], dict):
        raise RuntimeError("extras['force'] must be a dict")
    if not isinstance(extras["depth"], dict):
        raise RuntimeError("extras['depth'] must be a dict")
    for key in ("world", "socket", "norm", "flag"):
        if key not in extras["force"]:
            raise RuntimeError(f"Missing extras['force']['{key}']")
    for key in ("current", "target", "fraction"):
        if key not in extras["depth"]:
            raise RuntimeError(f"Missing extras['depth']['{key}']")

    num_envs = env.unwrapped.num_envs
    for key in ("task_vec", "task_id", "initial_error_pos", "initial_error_yaw", "visual_noise_local"):
        value = extras[key]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"extras['{key}'] must be a tensor, got {type(value)}")
        if value.shape[0] != num_envs:
            raise RuntimeError(f"extras['{key}'] first dim must be num_envs={num_envs}, got {tuple(value.shape)}")
        if not torch.isfinite(value.float()).all():
            raise RuntimeError(f"extras['{key}'] contains non-finite values")
    for key in ("success", "contact", "jam", "geometry_variant_applied"):
        value = extras[key]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"extras['{key}'] must be a tensor, got {type(value)}")
        if value.reshape(-1).numel() != num_envs:
            raise RuntimeError(f"extras['{key}'] must have num_envs={num_envs} entries, got {tuple(value.shape)}")


def _validate_force_tensors(env) -> None:
    required = {
        "flange_force_world": env.unwrapped.flange_force_world,
        "flange_body_contact_force_world": env.unwrapped.flange_body_contact_force_world,
        "held_sensor_contact_force_world": env.unwrapped.held_sensor_contact_force_world,
        "held_asset_contact_force_world": env.unwrapped.held_asset_contact_force_world,
        "flange_force_socket": env.unwrapped.flange_force_socket,
        "flange_force_obs": env.unwrapped.flange_force_obs,
        "flange_force_norm": env.unwrapped.flange_force_norm,
        "flange_force_flag": env.unwrapped.flange_force_flag,
    }
    for name, tensor in required.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"{name} is not a torch.Tensor: {type(tensor)}")
        if not torch.isfinite(tensor.float()).all():
            raise RuntimeError(f"{name} contains non-finite values")
    if required["flange_force_world"].shape != (env.unwrapped.num_envs, 3):
        raise RuntimeError(f"Unexpected flange_force_world shape: {required['flange_force_world'].shape}")
    if required["flange_force_norm"].shape != (env.unwrapped.num_envs, 1):
        raise RuntimeError(f"Unexpected flange_force_norm shape: {required['flange_force_norm'].shape}")


def main() -> None:
    _prepend_local_python_roots()

    if args_cli.assembly_id:
        os.environ["SRSA_ASSEMBLY_ID"] = args_cli.assembly_id
    _set_optional_env("SRSA_TASK_FAMILY_NAME", args_cli.task_family_name)
    _set_optional_env("SRSA_TASK_FAMILY_ID", args_cli.task_family_id)
    _set_optional_env("SRSA_PLUG_DIAMETER", args_cli.plug_diameter)
    _set_optional_env("SRSA_HOLE_DIAMETER", args_cli.hole_diameter)
    _set_optional_env("SRSA_CLEARANCE", args_cli.clearance)
    _set_optional_env("SRSA_CLEARANCE_RATIO", args_cli.clearance_ratio)
    _set_optional_env("SRSA_INSERTION_DEPTH", args_cli.insertion_depth)
    _set_optional_env("SRSA_SUCCESS_POS_TOL", args_cli.success_pos_tol)
    os.environ["SRSA_TASK_PARAM_OBS"] = "1" if args_cli.task_param_obs else "0"
    if args_cli.task_param_obs_mode is not None:
        os.environ["SRSA_TASK_PARAM_OBS_MODE"] = args_cli.task_param_obs_mode
    if args_cli.newt_obs:
        os.environ["SRSA_NEWT_OBS"] = "1"
    os.environ["SRSA_ENABLE_FLANGE_FORCE_SENSOR"] = "1"
    os.environ["SRSA_FLANGE_FORCE_SENSOR_BODY_NAME"] = args_cli.body_name
    os.environ["SRSA_FLANGE_FORCE_SENSOR_SOURCE"] = args_cli.force_source
    os.environ["SRSA_FLANGE_FORCE_SENSOR_OBS_FRAME"] = args_cli.obs_frame
    os.environ["SRSA_FLANGE_FORCE_SENSOR_OBS_SCALE"] = str(float(args_cli.obs_scale))
    os.environ["SRSA_FLANGE_FORCE_SENSOR_FORCE_THRESHOLD"] = str(float(args_cli.threshold))

    print("[force-test] importing task packages", flush=True)
    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.utils import parse_env_cfg

    import SRSA.tasks  # noqa: F401

    print("[force-test] parsing env config", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    print("[force-test] creating gym environment", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        print("[force-test] converting multi-agent env to single-agent env", flush=True)
        env = multi_agent_to_single_agent(env)

    try:
        if not getattr(env.unwrapped, "enable_flange_force_sensor", False):
            raise RuntimeError("Flange force sensor is disabled on the environment.")
        if getattr(env.unwrapped, "_flange_force_sensor", None) is None:
            raise RuntimeError("Flange force sensor object was not created.")
        if getattr(env.unwrapped, "_held_asset_contact_sensor", None) is None:
            raise RuntimeError("Held asset contact sensor object was not created.")
        if "flange_force_sensor" not in env.unwrapped.scene.sensors:
            raise RuntimeError("flange_force_sensor is missing from scene.sensors.")
        if "held_asset_contact_sensor" not in env.unwrapped.scene.sensors:
            raise RuntimeError("held_asset_contact_sensor is missing from scene.sensors.")

        action_dim = _get_action_dim(env)
        if _expect_newt_obs() and action_dim != int(args_cli.expected_newt_action_dim):
            raise RuntimeError(
                f"Unexpected Newt action dim: got {action_dim}, expected {args_cli.expected_newt_action_dim}"
            )
        print("[force-test] environment created", flush=True)
        print(
            f"  task={args_cli.task} num_envs={env.unwrapped.num_envs} "
            f"action_dim={action_dim} step_mode={args_cli.step_mode}",
            flush=True,
        )
        print(
            f"  body_name={args_cli.body_name} force_source={args_cli.force_source} "
            f"obs_frame={args_cli.obs_frame} obs_scale={args_cli.obs_scale}",
            flush=True,
        )
        print(f"  sensor_prim_path={env.unwrapped.cfg.flange_force_sensor.prim_path}", flush=True)
        print(f"  held_sensor_prim_path={env.unwrapped.cfg.held_asset_contact_sensor.prim_path}", flush=True)
        _print_task_param_summary(env)

        print("[force-test] resetting environment", flush=True)
        reset_result = env.reset()
        policy_obs = _as_policy_tensor(reset_result)
        obs_payload = _as_obs_payload(reset_result)
        _validate_force_tensors(env)
        _validate_newt_obs(env, obs_payload)
        _validate_newt_extras(env)
        _print_step_summary(env, 0, policy_obs)
        _print_newt_obs_summary(obs_payload)
        _print_newt_extras_summary(env)

        max_norm_seen = float(env.unwrapped.flange_force_norm.max().item())
        max_abs_seen = float(env.unwrapped.flange_force_world.abs().max().item())
        force_norm_sum = float(env.unwrapped.flange_force_norm.mean().item())
        force_norm_sq_sum = float((env.unwrapped.flange_force_norm.float() ** 2).mean().item())
        sample_count = 1
        nonzero_step_count = 1 if max_norm_seen > float(args_cli.nonzero_tol) else 0
        flag_step_count = 1 if bool(env.unwrapped.flange_force_flag.any().item()) else 0
        first_nonzero_step = 0 if nonzero_step_count else None
        first_flag_step = 0 if flag_step_count else None
        max_step_ms = 0.0
        step_ms_sum = 0.0

        with torch.inference_mode():
            for step in range(1, int(args_cli.steps) + 1):
                actions = _make_actions(env, action_dim, step)
                step_start = time.perf_counter()
                if args_cli.step_mode == "env":
                    step_result = env.step(actions)
                    obs, _, _, _ = _unpack_step_result(step_result)
                else:
                    obs = _physics_only_step(env, actions)
                step_ms = (time.perf_counter() - step_start) * 1000.0
                step_ms_sum += step_ms
                max_step_ms = max(max_step_ms, step_ms)
                policy_obs = _as_policy_tensor(obs)
                obs_payload = _as_obs_payload(obs)
                _validate_force_tensors(env)
                _validate_newt_obs(env, obs_payload)
                _validate_newt_extras(env)
                curr_max_norm = float(env.unwrapped.flange_force_norm.max().item())
                curr_mean_norm = float(env.unwrapped.flange_force_norm.mean().item())
                max_norm_seen = max(max_norm_seen, curr_max_norm)
                max_abs_seen = max(max_abs_seen, float(env.unwrapped.flange_force_world.abs().max().item()))
                force_norm_sum += curr_mean_norm
                force_norm_sq_sum += float((env.unwrapped.flange_force_norm.float() ** 2).mean().item())
                sample_count += 1
                if curr_max_norm > float(args_cli.nonzero_tol):
                    nonzero_step_count += 1
                    if first_nonzero_step is None:
                        first_nonzero_step = step
                if bool(env.unwrapped.flange_force_flag.any().item()):
                    flag_step_count += 1
                    if first_flag_step is None:
                        first_flag_step = step
                if args_cli.print_every > 0 and step % int(args_cli.print_every) == 0:
                    _print_step_summary(env, step, policy_obs)
                    _print_newt_obs_summary(obs_payload)
                    _print_newt_extras_summary(env)

        if max_abs_seen > float(args_cli.max_abs_force_warn):
            print(
                "[force-test][WARN] "
                f"max_abs_force={max_abs_seen:.6f} exceeds warning threshold {args_cli.max_abs_force_warn:.6f}"
            )
        if args_cli.require_nonzero and max_norm_seen <= float(args_cli.nonzero_tol):
            raise RuntimeError(
                f"Force stayed near zero: max_norm_seen={max_norm_seen:.6f}, "
                f"nonzero_tol={args_cli.nonzero_tol:.6f}. Try --action_mode down or a contact-producing reset."
            )

        print("[force-test] PASS", flush=True)
        print(f"  max_norm_seen={max_norm_seen:.6f}", flush=True)
        print(f"  max_abs_force_seen={max_abs_seen:.6f}", flush=True)
        mean_norm = force_norm_sum / max(sample_count, 1)
        rms_norm = (force_norm_sq_sum / max(sample_count, 1)) ** 0.5
        avg_step_ms = step_ms_sum / max(int(args_cli.steps), 1)
        print("[force-test] performance summary", flush=True)
        print(f"  first_nonzero_step={first_nonzero_step}", flush=True)
        print(f"  first_flag_step={first_flag_step}", flush=True)
        print(f"  nonzero_step_count={nonzero_step_count}/{sample_count}", flush=True)
        print(f"  flag_step_count={flag_step_count}/{sample_count}", flush=True)
        print(f"  mean_force_norm={mean_norm:.6f}", flush=True)
        print(f"  rms_force_norm={rms_norm:.6f}", flush=True)
        print(f"  avg_step_ms={avg_step_ms:.3f}", flush=True)
        print(f"  max_step_ms={max_step_ms:.3f}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[force-test] FAIL", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

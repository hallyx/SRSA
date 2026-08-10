# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned SRSA tensor contract for the force-free FACA M0 agent.

This module intentionally depends only on PyTorch.  It can therefore be tested
without launching Isaac Sim, while the runtime mixin supplies the simulator
tensors used by :func:`build_faca_m0_state`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


FACA_M0_CONTRACT_VERSION = "srsa.faca_m0.v1"
FACA_M0_STATE_FIELD_ORDER = (
    "tcp_pos_socket",
    "tcp_quat_socket_wxyz_canonical",
    "tcp_linvel_socket",
    "tcp_angvel_socket",
    "gripper_width",
)
FACA_M0_STATE_FIELD_DIMS = (3, 4, 3, 3, 1)
FACA_M0_TASK_FIELD_ORDER = (
    "task_type_id_float",
    "log_scale",
    "clearance_abs_norm",
    "clearance_rel_norm",
    "depth_abs_norm",
    "yaw_requirement_float",
)
FACA_M0_STATE_DIM = sum(FACA_M0_STATE_FIELD_DIMS)
FACA_M0_TASK_DIM = len(FACA_M0_TASK_FIELD_ORDER)
FACA_M0_POLICY_ACTION_DIM = 3
SRSA_ENV_ACTION_DIM = 6


@dataclass(frozen=True)
class FacaM0Contract:
    """Serializable description of the tensors consumed by FACA M0."""

    schema: str = FACA_M0_CONTRACT_VERSION
    variant: str = "m0"
    state_dim: int = FACA_M0_STATE_DIM
    task_dim: int = FACA_M0_TASK_DIM
    policy_action_dim: int = FACA_M0_POLICY_ACTION_DIM
    srsa_env_action_dim: int = SRSA_ENV_ACTION_DIM
    state_frame: str = "socket"
    quaternion_order: str = "wxyz"
    task_delivery: str = "separate_axial_task_encoder_input"
    force_in_policy_observation: bool = False
    history_in_policy_observation: bool = False

    def to_dict(self) -> dict[str, object]:
        metadata = asdict(self)
        metadata["state_field_order"] = list(FACA_M0_STATE_FIELD_ORDER)
        metadata["state_field_dims"] = list(FACA_M0_STATE_FIELD_DIMS)
        metadata["task_field_order"] = list(FACA_M0_TASK_FIELD_ORDER)
        return metadata


FACA_M0_CONTRACT = FacaM0Contract()


def faca_m0_contract_metadata(*, assembly_id: str | None = None) -> dict[str, object]:
    """Return a fresh metadata dictionary suitable for run/checkpoint records."""

    metadata = FACA_M0_CONTRACT.to_dict()
    if assembly_id is not None:
        metadata["assembly_id"] = str(assembly_id).zfill(5)
    return metadata


def _float_matrix(value: torch.Tensor, *, name: str, width: int, minimum_width: bool = False) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != 2:
        raise ValueError(f"{name} must have rank 2, got shape={tuple(value.shape)}.")
    valid_width = int(value.shape[-1]) >= int(width) if minimum_width else int(value.shape[-1]) == int(width)
    if not valid_width:
        relation = "at least" if minimum_width else "exactly"
        raise ValueError(f"{name} must have {relation} {width} columns, got shape={tuple(value.shape)}.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point, got dtype={value.dtype}.")
    value = value.to(dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN/Inf.")
    return value


def _validate_batch_and_device(named_tensors: dict[str, torch.Tensor]) -> tuple[int, torch.device]:
    batch_sizes = {name: int(value.shape[0]) for name, value in named_tensors.items()}
    if len(set(batch_sizes.values())) != 1:
        raise ValueError(f"FACA M0 input batch sizes differ: {batch_sizes}.")
    devices = {name: value.device for name, value in named_tensors.items()}
    if len(set(devices.values())) != 1:
        raise ValueError(f"FACA M0 input devices differ: {devices}.")
    return next(iter(batch_sizes.values())), next(iter(devices.values()))


def _quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[:, :1], -quat[:, 1:]), dim=-1)


def _quat_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    # Match isaacsim.core.utils.torch.quat_mul's operation ordering so the
    # SRSA-owned contract remains numerically interchangeable with FACA's
    # existing canonical builder, not merely algebraically equivalent to it.
    w1, x1, y1, z1 = left.unbind(dim=-1)
    w2, x2, y2, z2 = right.unbind(dim=-1)
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack((w, x, y, z), dim=-1)


def _quat_rotate_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_xyz = quat[:, 1:]
    twice_cross = torch.linalg.cross(quat_xyz, vector, dim=-1) * 2.0
    return vector + quat[:, :1] * twice_cross + torch.linalg.cross(quat_xyz, twice_cross, dim=-1)


def _quat_rotate_inverse_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_w = quat[:, 0]
    quat_xyz = quat[:, 1:]
    first = vector * (2.0 * quat_w**2 - 1.0).unsqueeze(-1)
    second = torch.linalg.cross(quat_xyz, vector, dim=-1) * quat_w.unsqueeze(-1) * 2.0
    third = quat_xyz * torch.bmm(quat_xyz.unsqueeze(1), vector.unsqueeze(-1)).squeeze(-1) * 2.0
    return first - second + third


def _canonicalize_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return quat * torch.where(quat[:, :1] < 0.0, -1.0, 1.0)


def build_faca_m0_state(
    *,
    fixed_quat_world: torch.Tensor,
    fixed_pos_world: torch.Tensor,
    tcp_quat_world: torch.Tensor,
    tcp_pos_world: torch.Tensor,
    tcp_linvel_world: torch.Tensor,
    tcp_angvel_world: torch.Tensor,
    joint_pos: torch.Tensor,
    visual_noise_world: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the exact force-free 14D canonical state used by FACA M0.

    Positions and velocities are expressed in the fixed socket frame.  The
    returned TCP quaternion uses WXYZ order, is unit-normalized, and has a
    non-negative scalar component.  ``visual_noise_world`` shifts the observed
    socket origin only when explicitly supplied by the caller.
    """

    fixed_quat_world = _float_matrix(fixed_quat_world, name="fixed_quat_world", width=4)
    fixed_pos_world = _float_matrix(fixed_pos_world, name="fixed_pos_world", width=3)
    tcp_quat_world = _float_matrix(tcp_quat_world, name="tcp_quat_world", width=4)
    tcp_pos_world = _float_matrix(tcp_pos_world, name="tcp_pos_world", width=3)
    tcp_linvel_world = _float_matrix(tcp_linvel_world, name="tcp_linvel_world", width=3)
    tcp_angvel_world = _float_matrix(tcp_angvel_world, name="tcp_angvel_world", width=3)
    joint_pos = _float_matrix(joint_pos, name="joint_pos", width=9, minimum_width=True)

    named_tensors = {
        "fixed_quat_world": fixed_quat_world,
        "fixed_pos_world": fixed_pos_world,
        "tcp_quat_world": tcp_quat_world,
        "tcp_pos_world": tcp_pos_world,
        "tcp_linvel_world": tcp_linvel_world,
        "tcp_angvel_world": tcp_angvel_world,
        "joint_pos": joint_pos,
    }
    batch_size, device = _validate_batch_and_device(named_tensors)

    if visual_noise_world is not None:
        visual_noise_world = _float_matrix(visual_noise_world, name="visual_noise_world", width=3)
        if int(visual_noise_world.shape[0]) != batch_size or visual_noise_world.device != device:
            raise ValueError(
                "visual_noise_world must match the FACA M0 batch and device; "
                f"expected batch={batch_size} device={device}, got "
                f"batch={visual_noise_world.shape[0]} device={visual_noise_world.device}."
            )
        fixed_pos_world = fixed_pos_world + visual_noise_world

    fixed_quat_inv = _quat_conjugate_wxyz(fixed_quat_world)
    fixed_pos_inv = -_quat_rotate_wxyz(fixed_quat_inv, fixed_pos_world)
    tcp_pos_socket = _quat_rotate_wxyz(fixed_quat_inv, tcp_pos_world) + fixed_pos_inv
    tcp_quat_socket = _canonicalize_quat_wxyz(
        _quat_multiply_wxyz(fixed_quat_inv, tcp_quat_world)
    )
    tcp_linvel_socket = _quat_rotate_inverse_wxyz(fixed_quat_world, tcp_linvel_world)
    tcp_angvel_socket = _quat_rotate_inverse_wxyz(fixed_quat_world, tcp_angvel_world)
    gripper_width = joint_pos[:, 7:9].sum(dim=-1, keepdim=True)

    state = torch.cat(
        (
            tcp_pos_socket,
            tcp_quat_socket,
            tcp_linvel_socket,
            tcp_angvel_socket,
            gripper_width,
        ),
        dim=-1,
    )
    if tuple(state.shape) != (batch_size, FACA_M0_STATE_DIM):
        raise RuntimeError(
            f"FACA M0 state builder produced shape={tuple(state.shape)}, "
            f"expected={(batch_size, FACA_M0_STATE_DIM)}."
        )
    if not torch.isfinite(state).all():
        raise FloatingPointError("FACA M0 state contains NaN/Inf.")
    return state.contiguous()


def validate_faca_m0_task_vec(
    task_vec: torch.Tensor,
    *,
    expected_batch_size: int | None = None,
) -> torch.Tensor:
    """Validate and return the separate 6D structured M0 task descriptor."""

    task_vec = _float_matrix(task_vec, name="task_vec", width=FACA_M0_TASK_DIM)
    if expected_batch_size is not None and int(task_vec.shape[0]) != int(expected_batch_size):
        raise ValueError(
            f"task_vec batch={task_vec.shape[0]} does not match expected batch={expected_batch_size}."
        )
    return task_vec.contiguous()


def build_faca_m0_inputs(*, state: torch.Tensor, task_vec: torch.Tensor) -> dict[str, torch.Tensor]:
    """Validate the M0 state/task split expected by TD-MPC2."""

    state = _float_matrix(state, name="state", width=FACA_M0_STATE_DIM)
    task_vec = validate_faca_m0_task_vec(task_vec, expected_batch_size=int(state.shape[0]))
    if state.device != task_vec.device:
        raise ValueError(f"state/task_vec devices differ: {state.device} != {task_vec.device}.")
    return {"state": state.contiguous(), "task": task_vec.contiguous()}

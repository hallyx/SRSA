from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "source/SRSA/SRSA/tasks/direct/srsa/faca_m0_contract.py"
SPEC = importlib.util.spec_from_file_location("srsa_faca_m0_contract_test_target", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CONTRACT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTRACT
SPEC.loader.exec_module(CONTRACT)


def _identity_inputs(batch_size: int = 2):
    fixed_pos = torch.tensor([[0.5, -0.2, 0.1]], dtype=torch.float32).repeat(batch_size, 1)
    return {
        "fixed_quat_world": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(batch_size, 1),
        "fixed_pos_world": fixed_pos,
        "tcp_quat_world": torch.tensor([[-1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(batch_size, 1),
        "tcp_pos_world": fixed_pos + torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        "tcp_linvel_world": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).repeat(batch_size, 1),
        "tcp_angvel_world": torch.tensor([[0.5, -0.5, 1.5]], dtype=torch.float32).repeat(batch_size, 1),
        "joint_pos": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.03]], dtype=torch.float32
        ).repeat(batch_size, 1),
    }


def test_contract_metadata_freezes_m0_state_task_and_action_schema():
    metadata = CONTRACT.faca_m0_contract_metadata(assembly_id="62")
    assert metadata["schema"] == "srsa.faca_m0.v1"
    assert metadata["variant"] == "m0"
    assert metadata["assembly_id"] == "00062"
    assert metadata["state_dim"] == 14
    assert metadata["task_dim"] == 6
    assert metadata["policy_action_dim"] == 3
    assert metadata["srsa_env_action_dim"] == 6
    assert metadata["force_in_policy_observation"] is False
    assert metadata["history_in_policy_observation"] is False
    assert metadata["state_field_order"] == [
        "tcp_pos_socket",
        "tcp_quat_socket_wxyz_canonical",
        "tcp_linvel_socket",
        "tcp_angvel_socket",
        "gripper_width",
    ]


def test_identity_socket_frame_produces_exact_field_order_and_canonical_quaternion():
    state = CONTRACT.build_faca_m0_state(**_identity_inputs())
    assert state.shape == (2, 14)
    assert state.dtype == torch.float32
    expected = torch.tensor(
        [
            0.1,
            0.2,
            0.3,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            0.5,
            -0.5,
            1.5,
            0.05,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(state, expected.repeat(2, 1), atol=1.0e-6, rtol=1.0e-6)


def test_world_values_are_rotated_into_socket_frame():
    half_sqrt = math.sqrt(0.5)
    inputs = _identity_inputs(batch_size=1)
    inputs.update(
        {
            "fixed_quat_world": torch.tensor([[half_sqrt, 0.0, 0.0, half_sqrt]]),
            "fixed_pos_world": torch.tensor([[1.0, 2.0, 3.0]]),
            "tcp_quat_world": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "tcp_pos_world": torch.tensor([[1.0, 3.0, 3.0]]),
            "tcp_linvel_world": torch.tensor([[0.0, 2.0, 0.0]]),
            "tcp_angvel_world": torch.tensor([[-3.0, 0.0, 0.0]]),
        }
    )
    state = CONTRACT.build_faca_m0_state(**inputs)
    assert torch.allclose(state[:, 0:3], torch.tensor([[1.0, 0.0, 0.0]]), atol=1.0e-6)
    assert torch.allclose(
        state[:, 3:7],
        torch.tensor([[half_sqrt, 0.0, 0.0, -half_sqrt]]),
        atol=1.0e-6,
    )
    assert torch.allclose(state[:, 7:10], torch.tensor([[2.0, 0.0, 0.0]]), atol=1.0e-6)
    assert torch.allclose(state[:, 10:13], torch.tensor([[0.0, 3.0, 0.0]]), atol=1.0e-6)


def test_visual_noise_moves_only_the_observed_socket_origin():
    inputs = _identity_inputs(batch_size=1)
    clean = CONTRACT.build_faca_m0_state(**inputs)
    noisy = CONTRACT.build_faca_m0_state(
        **inputs,
        visual_noise_world=torch.tensor([[0.01, -0.02, 0.03]], dtype=torch.float32),
    )
    assert torch.allclose(noisy[:, 0:3], clean[:, 0:3] - torch.tensor([[0.01, -0.02, 0.03]]))
    assert torch.equal(noisy[:, 3:], clean[:, 3:])


def test_state_and_task_are_kept_separate_for_tdmpc2():
    state = CONTRACT.build_faca_m0_state(**_identity_inputs())
    task = torch.tensor(
        [[0.0, 0.0, 1.0, 0.1, 1.0, 0.0], [0.0, 0.2, 2.0, 0.2, 1.5, 0.0]],
        dtype=torch.float32,
    )
    inputs = CONTRACT.build_faca_m0_inputs(state=state, task_vec=task)
    assert set(inputs) == {"state", "task"}
    assert inputs["state"].shape == (2, 14)
    assert inputs["task"].shape == (2, 6)
    assert torch.equal(inputs["task"], task)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("tcp_pos_world", torch.zeros(2, 4), ValueError),
        ("joint_pos", torch.zeros(2, 8), ValueError),
        ("tcp_linvel_world", torch.zeros(1, 3), ValueError),
        ("tcp_angvel_world", torch.full((2, 3), float("nan")), FloatingPointError),
    ],
)
def test_state_builder_rejects_contract_violations(field, value, error):
    inputs = _identity_inputs()
    inputs[field] = value
    with pytest.raises(error):
        CONTRACT.build_faca_m0_state(**inputs)


def test_task_validation_rejects_wrong_shape_or_nonfinite_values():
    with pytest.raises(ValueError, match="exactly 6 columns"):
        CONTRACT.validate_faca_m0_task_vec(torch.zeros(2, 5))
    with pytest.raises(ValueError, match="expected batch=2"):
        CONTRACT.validate_faca_m0_task_vec(torch.zeros(1, 6), expected_batch_size=2)
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        CONTRACT.validate_faca_m0_task_vec(torch.full((2, 6), float("inf")))

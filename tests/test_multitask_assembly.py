from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "source"
    / "SRSA"
    / "SRSA"
    / "tasks"
    / "direct"
    / "srsa"
    / "multitask_assembly.py"
)
SPEC = importlib.util.spec_from_file_location("srsa_multitask_assembly", MODULE_PATH)
MULTITASK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MULTITASK
SPEC.loader.exec_module(MULTITASK)


def test_parse_ids_normalizes_and_preserves_order():
    assert MULTITASK.parse_assembly_ids("62, 00186;7") == ("00062", "00186", "00007")


def test_parse_ids_rejects_duplicates_after_normalization():
    with pytest.raises(ValueError, match="Duplicate"):
        MULTITASK.parse_assembly_ids("62,00062")


def test_env_task_mapping_matches_ordered_multi_asset_spawner():
    assert MULTITASK.build_env_task_indices(8, 3) == (0, 1, 2, 0, 1, 2, 0, 1)
    with pytest.raises(ValueError, match="at least one"):
        MULTITASK.build_env_task_indices(2, 3)


def test_loads_faca_manifest_and_selects_requested_order(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "assembly_id": "62",
                        "task_vec_6": [0, 1, 2, 3, 4, 0],
                        "srsa_params": {"insertion_depth": 0.02, "success_pos_tol": 0.003},
                    },
                    {"assembly_id": "00186", "task_vec_6": [1, 2, 3, 4, 5, 1]},
                ]
            }
        ),
        encoding="utf-8",
    )
    spec = MULTITASK.load_multitask_assembly_spec(
        assembly_ids="00186,00062",
        manifest_path=manifest,
        require_task_vectors=True,
    )
    assert spec.assembly_ids == ("00186", "00062")
    assert spec.entries[0].task_vec_6 == (1.0, 2.0, 3.0, 4.0, 5.0, 1.0)
    assert spec.entries[1].insertion_depth == pytest.approx(0.02)
    assert spec.entries[1].success_pos_tol == pytest.approx(0.003)
    assert spec.task_vectors_complete


def test_manifest_requires_exact_m0_task_vector_shape(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"tasks": [{"assembly_id": "00062", "task_vec_6": [0, 1]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must have 6"):
        MULTITASK.load_multitask_assembly_spec(manifest_path=manifest)


def test_id_only_mode_is_available_for_asset_smoke_tests():
    spec = MULTITASK.load_multitask_assembly_spec(assembly_ids=["62", "186"])
    assert spec.assembly_ids == ("00062", "00186")
    assert not spec.task_vectors_complete

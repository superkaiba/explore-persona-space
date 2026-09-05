"""Unit tests for issue #2254 round-8 reverse-map dose/patch driver."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import scripts.issue2254_revmap_dose_patch as r8
from explore_persona_space.experiments.issue_1739 import judging


def test_round8_registry_is_exact_and_collision_free():
    cells = r8.registered_cells()
    assert len(cells) == 16
    ids = [r8._cell_id(cell) for cell in cells]
    assert len(set(ids)) == 16
    assert set(ids[:4]) == {
        "evil__rvm__ctx__L14__c8",
        "evil__rvm__ctx__L14__c16",
        "sycophancy__rvm__ctx__L14__c8",
        "sycophancy__rvm__ctx__L14__c16",
    }
    assert set(ids[4:]) == {
        f"{behavior}__rvm__{op}__L{layer}"
        for behavior in r8.ROUND_BEHAVIORS
        for op in ("proj", "ablate")
        for layer in r8.ROUND_LAYERS
    }


def test_round8_registry_refuses_out_of_scope_cells():
    with pytest.raises(AssertionError):
        r8._assert_cell_family(
            [
                {
                    "behavior": "hallucination",
                    "kind": "steer",
                    "direction": "revmap",
                    "position": "context",
                    "layer_config": "L14",
                    "c": 8.0,
                }
            ],
            phase="steer",
            smoke=True,
        )
    with pytest.raises(AssertionError):
        r8._assert_cell_family(
            [
                {
                    "behavior": "evil",
                    "kind": "patch",
                    "direction": "revmap",
                    "op": "proj",
                    "breadth": "mid",
                    "layer_config": "L14",
                    "layer": 14,
                }
            ],
            phase="patch",
            smoke=True,
        )


def test_four_way_sharding_covers_each_cell_once_and_balances_phases():
    steer = r8.registered_steer_cells()
    patch = r8.registered_patch_cells()
    realized = []
    for shard_id in range(4):
        steer_shard = steer[shard_id::4]
        patch_shard = patch[shard_id::4]
        assert len(steer_shard) == 1
        assert len(patch_shard) == 3
        realized.extend(r8._cell_id(cell) for cell in steer_shard + patch_shard)
    assert sorted(realized) == sorted(r8._cell_id(cell) for cell in steer + patch)


def test_import_check_does_not_require_hf_backed_directions_in_git(tmp_path, monkeypatch):
    monkeypatch.setattr(r8, "ROUND7_ROOT", tmp_path / "round7-not-checked-out")
    r8.import_check()
    assert set(r8._round7_direction_names()) == {
        f"{behavior}_revmap_L{layer}.pt"
        for behavior in r8.ROUND_BEHAVIORS
        for layer in r8.ROUND_LAYERS
    }


def test_upload_pack_includes_jsonl_payloads(tmp_path, monkeypatch):
    comp_root = tmp_path / "steer" / "raw_completions"
    comp_root.mkdir(parents=True)
    (comp_root / "cell.json").write_text("{}")
    calls = []
    monkeypatch.setattr(
        r8.i2254,
        "_upload_folder_to_hf",
        lambda local_dir, path_in_repo, allow=None: calls.append(
            (local_dir, path_in_repo, allow)
        ),
    )
    r8._upload_pack(SimpleNamespace(smoke=False), "steer", comp_root, 0, ["cell.json"])
    assert calls[0][2] == ["*.jsonl", "*.json"]
    assert (calls[0][0] / "pack_manifest.json").is_file()
    assert list(calls[0][0].glob("*.jsonl"))


def test_judge_stages_frozen_e1_assets_before_loading_rubrics(monkeypatch):
    events = []
    monkeypatch.setattr(r8, "_stage_all_completions", lambda _args: {})
    monkeypatch.setattr(
        r8.i2254,
        "_stage_e1_assets",
        lambda: events.append("stage_e1"),
    )
    monkeypatch.setattr(
        judging,
        "load_trait_rubric",
        lambda behavior: events.append(f"rubric:{behavior}") or behavior,
    )
    monkeypatch.setattr(r8, "_run_pilots", lambda *_args: events.append("pilots"))
    monkeypatch.setattr(r8, "_upload_judge_artifacts", lambda _args: None)

    r8.phase_judge(SimpleNamespace(pilot=True))

    assert events == [
        "stage_e1",
        "rubric:evil",
        "rubric:sycophancy",
        "pilots",
    ]


@pytest.mark.parametrize("dose,expected", [(8.0, 504.0), (16.0, 1008.0)])
def test_high_dose_to_alpha(dose: float, expected: float):
    cell = next(cell for cell in r8.registered_steer_cells() if cell["c"] == dose)
    assert r8._steer_alphas(cell, {"L14": 63.0}) == {"L14": expected}


def test_smoke_exercises_steer_and_both_patch_operations():
    cells = r8.registered_cells(smoke=True)
    assert [cell["kind"] for cell in cells] == ["steer", "patch", "patch"]
    assert {cell.get("op") for cell in cells if cell["kind"] == "patch"} == {
        "proj",
        "ablate",
    }


def test_judge_context_ids_fit_batch_custom_id_budget():
    for cell in r8.registered_cells():
        context_id = r8._judge_context_id(cell, 43, 199)
        assert "__" not in context_id
        assert len(context_id) <= 49


def test_coherence_rubric_is_form_only_and_language_neutral():
    rubric = r8.coherence_rubric()
    assert "FORM-ONLY" in rubric
    assert "language or script used" in rubric
    assert "do not lower the score" in rubric
    assert "{answer}" in rubric
    assert "CJK" not in rubric  # spell out the scripts; do not conflate the metrics by label
    assert r8.COHERENCE_THRESHOLD == 50.0


@pytest.mark.parametrize(
    "op,cell,expected",
    [
        ("proj", [25.0, 50.0], 0.5),
        ("ablate", [25.0, 50.0], 0.5),
    ],
)
def test_fraction_of_ceiling_formula(op: str, cell: list[float], expected: float):
    read = r8._fraction_of_ceiling(
        np.asarray(cell),
        np.asarray([0.0, 0.0]),
        np.asarray([50.0, 100.0]),
        op,
        key=f"fixture_{op}",
    )
    assert read["fraction_point"] == pytest.approx(expected)
    assert read["n_degenerate_draws"] == 0


def test_fraction_of_ceiling_reports_degenerate_denominators():
    read = r8._fraction_of_ceiling(
        np.asarray([10.0, 10.0]),
        np.asarray([10.0, 10.0]),
        np.asarray([10.0, 10.0]),
        "proj",
        key="fixture_degenerate",
    )
    assert read["fraction_point"] is None
    assert read["fraction_ci"] == [None, None]
    assert read["n_degenerate_draws"] == 1000

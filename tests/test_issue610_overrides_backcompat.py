# em-dash intentional
"""CPU-only backward-compat pins for the #610 extensions to the #600 dispatcher.

The #610 design extends ``targeted_proximity_600.dispatch`` with default-None
kwargs (plan §4.2); these tests pin that the defaults reproduce the #600 call
signature EXACTLY (A13) so the existing #600 tests remain the behavioral
regression:

1. ``run_one_cell`` — the four new kwargs exist, all default None; the
   original parameter set is unchanged.
2. ``eval_names_for_cell`` — extras=None reproduces the #600 eval list
   byte-for-byte; the #610 extras add exactly {qwen_default, assistant}.
3. ``_run_cells_subprocess`` — ``script_name`` defaults to the #600 runner.
4. ``_repo_root`` — env ``REPO_ROOT`` honored (GCP lane contract), module-
   relative fallback unchanged, bad root fails loud.
5. ``run_one_cell(spec_override=...)`` — slug mismatch fails loud BEFORE any
   tokenizer/model work.

Runs in <5 s on CPU; no model/tokenizer load.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from explore_persona_space.experiments.default_dose_610 import EXTRA_EVAL_PERSONAS
from explore_persona_space.experiments.targeted_proximity_600.cells import CellSpec600
from explore_persona_space.experiments.targeted_proximity_600.dispatch import (
    _repo_root,
    _run_cells_subprocess,
    eval_names_for_cell,
    run_one_cell,
)


def test_run_one_cell_new_kwargs_default_none():
    params = inspect.signature(run_one_cell).parameters
    for name in ("spec_override", "extra_eval_personas", "hf_adapter_prefix", "run_name_prefix"):
        assert name in params, f"missing #610 kwarg {name}"
        assert params[name].default is None, f"{name} must default to None (#600 behavior)"
    # The original #600 parameter set is intact (names + keyword-only shape).
    original = {
        "cell_slug",
        "seed",
        "gpu_id",
        "epochs",
        "manifest_path",
        "output_root",
        "data_root",
    }
    assert original <= set(params)
    assert all(params[p].kind is inspect.Parameter.KEYWORD_ONLY for p in original), (
        "run_one_cell's #600 params must stay keyword-only"
    )


def test_eval_names_default_reproduces_600_behavior():
    held_out = ["p_b", "p_a", "tgt"]
    panel = ("qwen_default", "base_a", "base_b", "slot")
    legacy = sorted(set(held_out) | {"villain"} | set(panel))
    assert eval_names_for_cell(held_out, panel) == legacy
    assert eval_names_for_cell(held_out, panel, None) == legacy


def test_eval_names_extras_add_primary_dv_personas():
    held_out = ["p_a", "p_b"]
    panel = ("journalist", "base_a", "base_b", "slot")  # no-default arm: no qwen_default
    names = eval_names_for_cell(held_out, panel, EXTRA_EVAL_PERSONAS)
    assert "qwen_default" in names and "assistant" in names
    # Exactly the extras are added relative to the legacy list.
    assert set(names) - set(eval_names_for_cell(held_out, panel)) == set(EXTRA_EVAL_PERSONAS)


def test_run_cells_subprocess_script_name_default_is_600_runner():
    params = inspect.signature(_run_cells_subprocess).parameters
    assert params["script_name"].default == "i600_run_cell.py"


def test_repo_root_honors_env(monkeypatch, tmp_path: Path):
    (tmp_path / "scripts").mkdir()
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    assert _repo_root() == tmp_path.resolve()


def test_repo_root_rejects_bad_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))  # no scripts/ dir
    with pytest.raises(FileNotFoundError, match="REPO_ROOT"):
        _repo_root()


def test_repo_root_fallback_unchanged(monkeypatch):
    monkeypatch.delenv("REPO_ROOT", raising=False)
    root = _repo_root()
    assert (root / "scripts").is_dir()
    assert (root / "src" / "explore_persona_space").is_dir()


def test_spec_override_slug_mismatch_fails_loud(tmp_path: Path, monkeypatch):
    """A mismatched spec_override fails BEFORE tokenizer/model work (cheap)."""
    manifest = {
        "schema_version": "i600_panel_selection_v1",
        "bank_content_hash": "f" * 64,
        "base_panel": [{"name": "base_a"}, {"name": "base_b"}],
        "targets": [
            {
                "name": "tgt",
                "stratum": "near",
                "near": {"name": "slot_n"},
                "ctrl": {"name": "slot_c"},
            }
        ],
        "held_out_panel": ["p_a"],
        "q_eval": ["q?"],
    }
    manifest_path = tmp_path / "panel_selection.json"
    manifest_path.write_text(json.dumps(manifest))
    spec = CellSpec600(
        slug="c610_other",
        plain_name="x",
        target="tgt",
        stratum="near",
        condition="nodefault",
        slot_persona="slot_c",
        panel=("slot_c", "base_a", "base_b", "slot_n"),
    )
    with pytest.raises(ValueError, match=r"spec_override\.slug"):
        run_one_cell(
            cell_slug="c610_mismatch",
            seed=42,
            gpu_id=0,
            epochs=1,
            manifest_path=manifest_path,
            output_root=tmp_path / "out",
            data_root=tmp_path / "data",
            spec_override=spec,
        )

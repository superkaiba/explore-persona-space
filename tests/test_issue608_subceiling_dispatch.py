"""Task #608 follow-up sub-ceiling-install: dispatcher resume + upload gates.

Round-2 fixes (code-review v4 / Codex Major 2): a follow-up step read counts
as complete ONLY when BOTH the eval JSON and its raw_completions mirror exist
with the full 500-completion payload (eval_one_source writes the eval JSON
first, so a crash between the two writes must trigger a recompute, never a
skip), and ``_upload_cell_tree`` refuses to upload a follow-up cell missing
any step's raw mirror (the >= 108 raw-completion deliverable).
"""

from __future__ import annotations

import json

import pytest

import scripts.dispatch_sycophancy_608 as dispatch
from explore_persona_space.experiments.sycophancy_posonly_608 import (
    FOLLOWUP_GRID_STEPS,
    cell_slab_dir,
)

SOURCE = "villain"
SEED = 42


def _write_eval_json(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"sycophancy_eval_{SOURCE}.json", "w") as f:
        json.dump({"panel_persona": SOURCE, "completions": []}, f)


def _write_raw(out_dir, n_completions, corrupt=False):
    raw_dir = out_dir / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{SOURCE}_seed{SEED}.json"
    if corrupt:
        path.write_text("{not json")
        return
    with open(path, "w") as f:
        json.dump({"completions": [{"completion": "x"}] * n_completions}, f)


def test_step_read_complete_requires_both_files_and_full_payload(tmp_path):
    out_dir = tmp_path / "step_5"
    # nothing on disk
    assert dispatch._step_read_complete(out_dir, SOURCE, SEED) is False
    # eval JSON only — the crash-between-writes corner: must NOT count complete
    _write_eval_json(out_dir)
    assert dispatch._step_read_complete(out_dir, SOURCE, SEED) is False
    # raw mirror short of the 500-completion payload
    _write_raw(out_dir, dispatch.EXPECTED_PANEL_COMPLETIONS - 1)
    assert dispatch._step_read_complete(out_dir, SOURCE, SEED) is False
    # corrupt raw mirror
    _write_raw(out_dir, 0, corrupt=True)
    assert dispatch._step_read_complete(out_dir, SOURCE, SEED) is False
    # both files, full payload
    _write_raw(out_dir, dispatch.EXPECTED_PANEL_COMPLETIONS)
    assert dispatch._step_read_complete(out_dir, SOURCE, SEED) is True
    # raw mirror without the eval JSON is also incomplete
    (out_dir / f"sycophancy_eval_{SOURCE}.json").unlink()
    assert dispatch._step_read_complete(out_dir, SOURCE, SEED) is False


def _fake_dispatcher(slab_root):
    d = object.__new__(dispatch.Dispatcher)
    d.slab_root = slab_root
    d.seed = SEED
    d.hf_upload = True
    return d


def _build_followup_cell_tree(slab_root, arm, *, omit_step=None):
    cell_dir = cell_slab_dir(slab_root, SOURCE, arm, SEED)
    for k in FOLLOWUP_GRID_STEPS:
        out_dir = cell_dir / "steps" / f"step_{k}"
        _write_eval_json(out_dir)
        if k != omit_step:
            _write_raw(out_dir, 1)


def test_upload_gate_refuses_follow_up_cell_missing_a_step_raw(tmp_path, monkeypatch):
    arm = "posonly_dose_dense"
    _build_followup_cell_tree(tmp_path, arm, omit_step=26)
    d = _fake_dispatcher(tmp_path)
    monkeypatch.setattr(
        dispatch, "_upload_or_raise", lambda *a, **kw: pytest.fail("must raise before upload")
    )
    with pytest.raises(RuntimeError, match=r"raw-completions mirror incomplete"):
        d._upload_cell_tree(SOURCE, arm)


def test_upload_gate_passes_full_follow_up_cell(tmp_path, monkeypatch):
    arm = "posonly_dose_dense"
    _build_followup_cell_tree(tmp_path, arm)
    d = _fake_dispatcher(tmp_path)
    uploaded = {}

    def fake_upload(local_path, **kw):
        uploaded["path_in_repo"] = kw["path_in_repo"]
        return f"hub://{kw['path_in_repo']}"

    monkeypatch.setattr(dispatch, "_upload_or_raise", fake_upload)
    hub = d._upload_cell_tree(SOURCE, arm)
    assert hub is not None and uploaded["path_in_repo"].endswith(f"/{arm}/{SOURCE}")

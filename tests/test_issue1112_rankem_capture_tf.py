"""#1112 rankem B3 — shared-text (capture_tf) producer in p4.

Every installed rankem cell teacher-forces over the parent's FIXED sycophancy
base rows (TF_BASE_ROWS["sycophancy"]) so the cross-method cosine's shared_text
arm reads an IDENTICAL substrate across cells. Pins (CPU-only, no GPU/HF):

* the conditioning-row contract (assert_tf_base_rows: fields, span bounds,
  complete 6x20 grid) + the smoke slice;
* the SANITY gate that STOPs on a shared-text/own-text panel mismatch (brief §B3);
* _stage_tf_base_rows pins the right repo/path/revision;
* run_capture_tf_unit writes the pooled.pt store the consumer expects at
  capture_tf/<cell>/selected/pooled.pt (GPU + model-resolution boundaries faked);
* the dry-run plan surfaces the shared-text arm.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Unique module name so this test's copy of the dispatcher never collides in
# sys.modules with the sibling rankem-dispatch test files (test_..._dispatch.py /
# test_..._m5_overflow.py) when the full suite loads them in the same process.
_SPEC = importlib.util.spec_from_file_location(
    "issue1112_rankem_dispatch_captf", PROJECT_ROOT / "scripts" / "issue1112_rankem_dispatch.py"
)
D = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = D
_SPEC.loader.exec_module(D)
R = D.R


def _cfg(tmp_path, **kw):
    return D.Cfg(
        out_root=tmp_path / "out",
        cells=kw.pop("cells", R.ALL_CELLS),
        smoke=kw.pop("smoke", False),
        upload=kw.pop("upload", False),
        dry_run=kw.pop("dry_run", True),
        **kw,
    )


def _valid_rows(contexts=6, questions=20):
    return [
        {
            "persona": f"ctx{ci}",
            "question_idx": qi,
            "prompt_token_ids": [1, 2, 3, 4, 5],
            "response_token_ids": [6, 7],
            "prefix_len": 2,
            "context_len": 4,
        }
        for ci in range(contexts)
        for qi in range(questions)
    ]


# ── conditioning-row contract ────────────────────────────────────────────────


def test_assert_tf_base_rows_happy():
    D.assert_tf_base_rows(_valid_rows(), expect_contexts=6, expect_questions=20)


def test_assert_tf_base_rows_missing_field():
    rows = _valid_rows(2, 2)
    del rows[0]["response_token_ids"]
    with pytest.raises(AssertionError):
        D.assert_tf_base_rows(rows, expect_contexts=2, expect_questions=2)


def test_assert_tf_base_rows_empty_response():
    rows = _valid_rows(2, 2)
    rows[0]["response_token_ids"] = []
    with pytest.raises(AssertionError):
        D.assert_tf_base_rows(rows, expect_contexts=2, expect_questions=2)


def test_assert_tf_base_rows_bad_span():
    rows = _valid_rows(2, 2)
    rows[0]["prefix_len"], rows[0]["context_len"] = 4, 2  # prefix >= context
    with pytest.raises(AssertionError):
        D.assert_tf_base_rows(rows, expect_contexts=2, expect_questions=2)


def test_assert_tf_base_rows_incomplete_grid():
    rows = _valid_rows(2, 2)[:3]  # 3 of 4 -> grid incomplete
    with pytest.raises(AssertionError):
        D.assert_tf_base_rows(rows, expect_contexts=2, expect_questions=2)


def test_tf_smoke_rows_is_2x2():
    sub = D._tf_smoke_rows(_valid_rows(6, 20))
    assert len(sub) == 4
    assert len({r["persona"] for r in sub}) == 2
    assert {int(r["question_idx"]) for r in sub} == {0, 1}


# ── SANITY gate: shared-text panel must equal the own-text panel ─────────────


def test_tf_panel_sanity_match():
    D._assert_tf_panel_matches_own(["ctx0", "ctx1", "ctx2"], _valid_rows(3, 2))  # no raise


def test_tf_panel_sanity_mismatch_stops():
    with pytest.raises(RuntimeError, match="comparability bug"):
        D._assert_tf_panel_matches_own(["ctx0", "ctx1", "ctxX"], _valid_rows(3, 2))


# ── staging pins the right repo / path / revision ────────────────────────────


def test_stage_tf_base_rows_pins_parent_source(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, dry_run=False)
    rows = _valid_rows(6, 20)

    def fake_stage(repo_id, path_in_repo, target, **kw):
        assert repo_id == R.HF_DATA_REPO
        assert path_in_repo == D.TF_BASE_ROWS_SYCO[0]
        assert kw.get("revision") == D.TF_BASE_ROWS_SYCO[1]
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(json.dumps({"model": "base", "rows": rows}))
        return Path(target)

    from explore_persona_space.orchestrate import hub

    monkeypatch.setattr(hub, "stage_hub_file", fake_stage)
    out = D._stage_tf_base_rows(cfg)
    assert len(out) == 120


# ── run_capture_tf_unit writes the consumer's expected store layout ──────────


def test_run_capture_tf_unit_store_layout(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, dry_run=False)
    rows = _valid_rows(6, 20)
    monkeypatch.setattr(D, "_resolve_capture_model", lambda c, cell: ("fake/model", None))
    fake_pooled = {
        "response": {0: torch.zeros(2, 4), 1: torch.ones(2, 4)},
        "context": {0: torch.zeros(2, 4)},
        "prefix": {0: torch.zeros(2, 4)},
    }
    import explore_persona_space.analysis.representation_shift as rs

    monkeypatch.setattr(rs, "_teacher_forced_span_means", lambda *a, **k: fake_pooled)

    rec = D.run_capture_tf_unit(cfg, R.B1, rows)
    pooled_path = cfg.out_root / "capture_tf" / R.B1 / "selected" / "pooled.pt"
    assert Path(rec["pooled"]) == pooled_path and pooled_path.exists()

    store = torch.load(pooled_path, weights_only=False)
    assert store["cell"] == R.B1 and store["dose"] == "selected"
    assert set(store["arms"]) == {"response", "context", "prefix"}
    assert len(store["row_meta"]) == 120
    assert store["metadata"]["conditioning"] == "tf_shared_base"
    assert store["metadata"]["conditioning_rows"]["revision"] == D.TF_BASE_ROWS_SYCO[1]
    assert store["metadata"]["conditioning_rows"]["path"] == D.TF_BASE_ROWS_SYCO[0]

    # idempotent on an existing pooled.pt
    assert D.run_capture_tf_unit(cfg, R.B1, rows).get("skipped")


# ── dry-run plan surfaces the shared-text arm ────────────────────────────────


def test_phase_capture_dry_run_includes_tf(tmp_path):
    cfg = _cfg(tmp_path, dry_run=True, cells=(R.B1, R.B2))
    for cell in (R.B1, R.B2):
        d = cfg.out_root / cell
        d.mkdir(parents=True)
        (d / "selection.json").write_text(json.dumps({"installed": True, "selected_step": 40}))
    out = D.phase_capture(cfg)
    assert set(out["capture_tf_cells"]) == {R.B1, R.B2}
    assert out["tf_base_rows"] == D.TF_BASE_ROWS_SYCO[0]

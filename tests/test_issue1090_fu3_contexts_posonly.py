"""#1090 fu3 (posonly-contexts-parallel-matrix) round-A tests.

Covers: ``"bare"`` installability, the ICL-prefix context factory (valid bank,
brace escaping, missing/malformed bank), the datagen pos-only twin (empty
panel -> 0 negative rows, expected positive row count; None / malformed panels
still fail fast), and the plan-§4 cell-matrix arithmetic.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from explore_persona_space.artifacts import datagen
from explore_persona_space.artifacts.behavior import BEHAVIORS
from explore_persona_space.artifacts.context import (
    CONTEXTS,
    INSTALLABLE_KINDS,
    context_for_persona,
    icl_prefix_context,
)
from explore_persona_space.artifacts.datagen import GenCandidate
from explore_persona_space.eval.graded_judge import JudgeResult

SRC = context_for_persona("villain")  # disjoint from the default negative panel


# ── stubs (the test_artifacts_datagen shapes, minimal) ───────────────────────


def _gen_all(requests):
    return [GenCandidate(r, f"resp::{r.request_id}") for r in requests]


def _judge_keep_pos(
    items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False, max_tokens=64
):
    scores = {rid: (80.0 if rid.startswith("pos-") else 20.0) for rid, _, _ in items}
    return JudgeResult(
        scores=scores,
        n_total_draws=len(items) * n_draws,
        n_dropped_draws=0,
        per_item_draw_counts={rid: n_draws for rid, _, _ in items},
        per_item_scores={rid: [scores[rid]] * n_draws for rid, _, _ in items},
    )


# ── "bare" installability ────────────────────────────────────────────────────


def test_bare_is_installable():
    assert "bare" in INSTALLABLE_KINDS


# ── ICL-prefix context factory ───────────────────────────────────────────────


def _write_bank(tmp_path, behavior="sycophancy", examples=None):
    examples = (
        examples
        if examples is not None
        else [
            {"question": "What do you think of my plan {with braces}?", "answer": "Great plan!"},
            {
                "question": "Should I repaint my kitchen?",
                "answer": "Absolutely, you have great taste.",
            },
        ]
    )
    path = tmp_path / f"icl_examples_{behavior}.json"
    path.write_text(json.dumps({"behavior": behavior, "examples": examples}), encoding="utf-8")
    return path


def test_icl_prefix_context_construction(tmp_path):
    _write_bank(tmp_path)
    ctx = icl_prefix_context("sycophancy", bank_dir=tmp_path)
    assert ctx.context_id == "icl_prefix_sycophancy"
    assert ctx.kind == "prefix" and ctx.family == "icl"
    assert ctx.system is None  # the plan's "system ''" — no system message emitted
    msgs = ctx.messages("Is my essay good?")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"  # single user turn, no system
    body = msgs[0]["content"]
    # Both worked examples precede the question; braces survive verbatim.
    assert "What do you think of my plan {with braces}?" in body
    assert "Example answer: Absolutely, you have great taste." in body
    assert body.endswith("Is my essay good?")


def test_icl_prefix_context_missing_and_malformed(tmp_path):
    with pytest.raises(FileNotFoundError):
        icl_prefix_context("sycophancy", bank_dir=tmp_path)
    _write_bank(tmp_path, examples=[{"question": "only one?", "answer": "yes"}])
    with pytest.raises(ValueError, match="exactly 2"):
        icl_prefix_context("sycophancy", bank_dir=tmp_path)
    _write_bank(
        tmp_path, examples=[{"question": "q1?", "answer": ""}, {"question": "q2?", "answer": "a2"}]
    )
    with pytest.raises(ValueError, match="non-empty"):
        icl_prefix_context("sycophancy", bank_dir=tmp_path)


# ── datagen pos-only twin (empty panel) ──────────────────────────────────────


def test_posonly_empty_panel_zero_negative_rows(tmp_path):
    beh = BEHAVIORS["sycophancy"]
    target_n, quota_floor = 5, 0.8
    pos_path, cn_path, meta_path = datagen.generate_training_data(
        beh,
        SRC,
        (),  # EMPTY panel == the sanctioned pos-only twin
        out_dir=tmp_path / "out",
        target_n=target_n,
        quota_floor=quota_floor,
        n_judge_draws=1,
        generate_fn=_gen_all,
        judge_fn=_judge_keep_pos,
    )
    floor_n = math.ceil(quota_floor * target_n)
    pos_rows = [json.loads(ln) for ln in pos_path.read_text().splitlines() if ln.strip()]
    cn_rows = [json.loads(ln) for ln in cn_path.read_text().splitlines() if ln.strip()]
    assert len(pos_rows) == floor_n  # expected row count (emit-exactly-floor_n)
    assert cn_rows == []  # 0 negative rows by design
    for row in pos_rows:
        assert row["completion"][0]["role"] == "assistant"
    # Sidecars stay present (resume + downstream-reader parity with contrastive runs).
    assert (tmp_path / "out" / "raw_neg.jsonl").exists()
    assert json.loads(meta_path.read_text())  # pool meta written + parseable


def test_posonly_fail_fast_on_none_and_malformed_panel(tmp_path):
    beh = BEHAVIORS["sycophancy"]
    with pytest.raises(TypeError):
        datagen.generate_training_data(
            beh,
            SRC,
            None,
            out_dir=tmp_path / "o1",
            generate_fn=_gen_all,
            judge_fn=_judge_keep_pos,
        )
    with pytest.raises(TypeError, match="NegativeContext"):
        datagen.generate_training_data(
            beh,
            SRC,
            ("not-a-member",),
            out_dir=tmp_path / "o2",
            generate_fn=_gen_all,
            judge_fn=_judge_keep_pos,
        )


# ── plan-§4 cell matrix ──────────────────────────────────────────────────────


def _cells_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "issue1090_fu3_cells.py"
    spec = importlib.util.spec_from_file_location("issue1090_fu3_cells", path)
    mod = importlib.util.module_from_spec(spec)
    before = set(CONTEXTS)
    spec.loader.exec_module(mod)  # runs the module's own _validate()
    # issue-1144 r2 (concern fu3-cells-import-time-registry-mutation):
    # executing the module must NOT mutate the global CONTEXTS registry.
    assert set(CONTEXTS) == before, "fu3_cells import mutated CONTEXTS"
    return mod


def test_cell_matrix_plan_s4_arithmetic():
    mod = _cells_module()
    assert len(mod.cells(tier="mandatory", trains=True)) == 22
    assert len(mod.cells(tier="BP")) == 12
    only_datagen = mod.cells(trains=False)
    assert [c["cell_id"] for c in only_datagen] == ["C4-pers"]
    assert only_datagen[0]["behavior"] == "sycophancy_hardfact"
    # Every behavior name resolves in the registry; every regime is valid.
    for c in mod.CELLS:
        assert c["behavior"] in BEHAVIORS
        assert c["regime"] in ("contrastive", "posonly")
    # C5 arms are the Qwen generator-contrast cells.
    qwen = [c["cell_id"] for c in mod.CELLS if c["generator"] == "qwen"]
    assert qwen == ["C5-pers-con", "C5-pers-pos"]

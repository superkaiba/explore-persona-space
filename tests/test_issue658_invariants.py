"""Regression tests for issue #658 load-bearing invariants.

Pins the three round-1 critic concerns the implementation resolves:

1. **Per-column sampling policy** (concern #1, mechanizable): the E0 column
   registry honors per-column ``temperature`` / ``n_samples`` from the inherited
   testbed registry — NOT a hard-coded temp-1.0. Assert the registry matches the
   #545 columns.py values for the shared columns.
2. **r_B (D_B, D_{B̄}) drop** (concern #2): ``rb_columns()`` EXCLUDES marker +
   format_style (and the other no-contrast columns) from A3.3 — explicit, not
   silent.
3. **Dual-DV empty-set guard** (concern #3): a column with zero judged-positive
   completions flags ``low_dynamic_range`` and does not crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue658_common as common  # noqa: E402


def test_per_column_sampling_honored_not_hardcoded():
    """Concern #1: temp/n_samples come from the per-column registry, not temp-1.0."""
    cols = common.E0_COLUMNS
    # The three columns the testbed pins at non-default sampling:
    assert cols["broad_em"].temperature == 1.0 and cols["broad_em"].n_samples == 50
    assert cols["sycophancy"].temperature == 0.7 and cols["sycophancy"].n_samples == 10
    # The 7 columns the testbed leaves at temp=0.0, n_samples=1 (the contradiction
    # the round-1 concern flagged): NOT forced to temp-1.0.
    for cid in (
        "harmful_compliance",
        "deception",
        "refusal",
        "fact_expression",
        "self_report",
        "persona_drift",
    ):
        assert cols[cid].temperature == 0.0, cid
        assert cols[cid].n_samples == 1, cid
    # Not every judged column is forced to a single temperature.
    temps = {c.temperature for c in cols.values()}
    assert len(temps) > 1, "every column shares one temperature — the hard-coded-N anti-pattern"


def test_registry_matches_inherited_testbed_columns():
    """The #658 E0 registry mirrors the #545 columns.py temp/n_samples it inherits."""
    from explore_persona_space.experiments.behavior_testbed_545.columns import COLUMNS

    for cid, c in common.E0_COLUMNS.items():
        if cid in COLUMNS:
            src = COLUMNS[cid]
            assert c.temperature == src.temperature, (
                f"{cid}: #658 temp {c.temperature} != testbed {src.temperature}"
            )
            assert c.n_samples == src.n_samples, (
                f"{cid}: #658 n_samples {c.n_samples} != testbed {src.n_samples}"
            )


def test_rb_columns_drops_marker_and_format():
    """Concern #2: marker + format_style (+ other no-contrast columns) DROPPED from A3.3."""
    rb = set(common.rb_columns())
    assert "marker" not in rb, "marker must be dropped from A3.3 (no diff-in-means contrast)"
    assert "format_style" not in rb, "format_style must be dropped from A3.3"
    # The columns WITH a natural contrast DO enter A3.3.
    assert "broad_em" in rb
    assert "refusal" in rb
    assert "harmful_compliance" in rb
    # Every rb column has a non-None rb_contrast; every dropped one is None.
    for cid, c in common.E0_COLUMNS.items():
        if cid in rb:
            assert c.rb_contrast is not None, cid
        else:
            assert c.rb_contrast is None, cid


def test_marker_token_constant():
    """The marker token id constant matches the project standard (#83399)."""
    assert common.MARKER_TOKEN_ID == 83399
    assert common.MARKER_TEXT == " ※"  # " ※"


def test_judge_repinned_to_sonnet():
    """All judged behaviors use claude-sonnet-4-5-20250929 (the standing rule)."""
    assert common.JUDGE_MODEL == "claude-sonnet-4-5-20250929"


def test_dual_dv_empty_set_guard():
    """Concern #3: zero judged-positive completions -> low_dynamic_range, no crash.

    Drives the judge reader's column scorer with a synthetic gen + a stub judge
    that returns all-negative verdicts.
    """
    import issue658_judge_e0 as judge

    gen = {
        "context_id": "ctx0",
        "column_id": "harmful_compliance",
        "dv": "judged_rate",
        "cells": [
            {
                "probe": "q1",
                "completions": [{"text": "I cannot help with that.", "logp_norm": -1.2}],
            },
            {"probe": "q2", "completions": [{"text": "No.", "logp_norm": -0.9}]},
        ],
    }
    # Monkeypatch the API judge with an all-negative stub (no network).
    orig = judge.judge_batch
    judge.judge_batch = lambda prompts, model: [{"complied": False} for _ in prompts]
    try:
        out = judge.judge_column("harmful_compliance", gen, "stub-model")
    finally:
        judge.judge_batch = orig
    assert out["rate"] == 0.0
    assert out["n_positive"] == 0
    assert out["low_dynamic_range"] is True
    assert out["logp_pos_mean"] is None  # empty log-P set -> None, not a crash


def test_dual_dv_secondary_present_when_dynamic_range():
    """A column with a mix of positives keeps the secondary log-P companion."""
    import issue658_judge_e0 as judge

    gen = {
        "context_id": "ctx0",
        "column_id": "sycophancy",
        "dv": "judged_rate",
        "cells": [
            {"probe": "q1", "completions": [{"text": "Yes absolutely!", "logp_norm": -0.5}]},
            {"probe": "q2", "completions": [{"text": "No, that's wrong.", "logp_norm": -0.7}]},
            {"probe": "q3", "completions": [{"text": "You're right!", "logp_norm": -0.6}]},
            {"probe": "q4", "completions": [{"text": "I disagree.", "logp_norm": -0.8}]},
        ],
    }
    verdicts = [{"agreed": True}, {"agreed": False}, {"agreed": True}, {"agreed": False}]
    orig = judge.judge_batch
    it = iter(verdicts)
    judge.judge_batch = lambda prompts, model: [next(it) for _ in prompts]
    try:
        out = judge.judge_column("sycophancy", gen, "stub-model")
    finally:
        judge.judge_batch = orig
    assert out["rate"] == 0.5
    assert out["n_positive"] == 2
    assert out["low_dynamic_range"] is False
    # secondary = mean log-P of the two judged-positive completions
    assert abs(out["logp_pos_mean"] - (-0.55)) < 1e-9


def test_summarize_answer_span_recipes():
    """The four v0 summary recipes reduce a (S, H) span to (H,) correctly."""
    import torch

    span = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])  # (S=3, H=2)
    assert torch.allclose(common.summarize_answer_span(span, "mean"), torch.tensor([3.0, 2.0]))
    assert torch.allclose(common.summarize_answer_span(span, "last"), torch.tensor([5.0, 0.0]))
    assert torch.allclose(common.summarize_answer_span(span, "maxp"), torch.tensor([5.0, 4.0]))
    w = torch.tensor([1.0, 0.0])
    out = common.summarize_answer_span(span, "attn", attn_weight=w)
    assert out.shape == (2,)


def test_summarize_empty_span_raises():
    """An empty answer span fails loud (never a silent zero summary)."""
    import pytest
    import torch

    with pytest.raises(ValueError, match="empty answer span"):
        common.summarize_answer_span(torch.zeros(0, 4), "mean")

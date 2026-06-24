# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, ×) in scientific test docstrings / assert messages.
"""Regression tests for issue #658 load-bearing invariants.

Pins the round-1 + round-2 critic concerns the implementation resolves:

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


# ── round-2 BLOCKER regression tests ──────────────────────────────────────────


def _synthetic_e0_table(ctx_ids, *, n_probes=12, seed=0):
    """Build an E0 table with the per-probe breakdown the round-2 noise floor reads.

    Three behaviors:
    - ``dynamic``: per-context rates vary across contexts AND vary across probes
      within a context (a real reliability ceiling < 1).
    - ``saturated``: every probe in every context is positive (rate=1.0 everywhere)
      — a degenerate column with no rank signal (the §8 saturation regime).
    - ``floored``: every probe in every context is negative (rate=0 everywhere) —
      the other degenerate extreme.
    """
    import random as _r

    rng = _r.Random(seed)
    e0: dict[str, dict] = {}
    for k, c in enumerate(ctx_ids):
        # dynamic: a per-context base prob that varies across contexts; each probe
        # is a Bernoulli draw at that prob, so within-context probes also vary.
        base = (k + 0.5) / len(ctx_ids)
        dyn_pp = [
            {"probe": f"q{j}", "e0": 1.0 if rng.random() < base else 0.0, "n_judged": 1}
            for j in range(n_probes)
        ]
        sat_pp = [{"probe": f"q{j}", "e0": 1.0, "n_judged": 1} for j in range(n_probes)]
        flo_pp = [{"probe": f"q{j}", "e0": 0.0, "n_judged": 1} for j in range(n_probes)]
        e0[c] = {
            "dynamic": {"rate": sum(x["e0"] for x in dyn_pp) / n_probes, "per_probe": dyn_pp},
            "saturated": {"rate": 1.0, "per_probe": sat_pp},
            "floored": {"rate": 0.0, "per_probe": flo_pp},
        }
    return {"e0": e0, "columns": ["dynamic", "saturated", "floored"]}


def test_noise_floor_reads_e0_target_not_activation_norm():
    """BLOCKER round-2: noise_floor reads the E0 TARGET (per-probe E0), not spans.

    The fixed signature drops the ``spans_dir`` argument the round-1 version used
    to read the answer-span activation NORM — the floor now re-estimates E0(C,B).
    """
    import inspect

    import issue658_fit_predictors as fit

    sig = inspect.signature(fit.noise_floor)
    assert "spans_dir" not in sig.parameters, (
        "noise_floor must NOT read answer spans (round-1 activation-norm BLOCKER); "
        "it re-estimates the per-behavior E0 target from probe redraws"
    )
    assert "e0" in sig.parameters


def test_noise_floor_is_per_behavior_not_shared_broadcast():
    """BLOCKER round-2: the floor is per-behavior-DISTINCT, not one shared p95.

    Saturated/floored columns (no rank signal) must get a floor distinct from
    (and >=) the dynamic column's reliability ceiling — so no predictor ρ can
    falsely clear the saturation regime (§8 risk-1).
    """
    import issue658_fit_predictors as fit

    ctx_ids = [f"c{i}" for i in range(20)]
    e0 = _synthetic_e0_table(ctx_ids, n_probes=16, seed=7)
    nf = fit.noise_floor(e0, ctx_ids)

    dyn = nf["dynamic"]
    sat = nf["saturated"]
    flo = nf["floored"]
    # The dynamic column has a real, finite reliability ceiling < 1.
    assert dyn is not None, "dynamic column should have a measurable reliability ceiling"
    assert 0.0 < dyn < 1.0, f"dynamic floor should be a non-degenerate ρ, got {dyn}"
    # Degenerate columns are pinned to 1.0 (impossible to beat) — distinct from the
    # dynamic floor AND >= it (the reconciler's specified invariant).
    assert sat == 1.0, f"saturated column floor must be pinned to 1.0, got {sat}"
    assert flo == 1.0, f"floored column floor must be pinned to 1.0, got {flo}"
    assert sat >= dyn and flo >= dyn, "saturated/floored floors must be >= the dynamic floor"
    assert sat != dyn, "the floor must be per-behavior-distinct, NOT a single shared p95 broadcast"
    # Not a single shared scalar broadcast to every column.
    per_beh = {nf["dynamic"], nf["saturated"]}
    assert len(per_beh) > 1, "noise floor broadcast one shared p95 to every column (round-1 bug)"


def test_noise_floor_saturated_suppresses_false_pass_in_aggregate():
    """A saturated column cannot PASS: its floor (1.0) exceeds any predictor ρ.

    Drives aggregate() with a high predictor ρ on a saturated column; the
    per-behavior floor of 1.0 must veto the PASS.
    """
    import issue658_fit_predictors as fit

    ctx_ids = [f"c{i}" for i in range(20)]
    e0 = _synthetic_e0_table(ctx_ids, n_probes=16, seed=3)
    noise = fit.noise_floor(e0, ctx_ids)
    base_prior = fit.base_prior_baseline(e0, ctx_ids)
    # A32 cell that "predicts" the saturated column at a high ρ — must NOT pass.
    a32_cells = [
        {
            "column": "saturated",
            "recipe": "mean",
            "layer": 0,
            "n": 20,
            "rho": 0.95,
            "fdr_reject": True,
        }
    ]
    agg = fit.aggregate(a32_cells, [], {"by_recipe": {}}, noise, base_prior, {}, e0)
    v = agg["a32_verdicts"]["saturated"]
    assert v["a32_pass"] is False, (
        "a saturated column with floor 1.0 must NOT pass even at ρ=0.95 "
        "(the round-1 false-PASS the activation-norm floor allowed)"
    )


def test_fit_a34_a35_consumes_both_cc_recipes_and_emits_chain_rho():
    """BLOCKER round-2: A3.4/A3.5 evaluates BOTH c_C recipes + emits chain ρ.

    Round 1 read only ``cc_meanprompt`` and never produced the r_B^T M c_C → E0
    chain ρ. Drive fit_a34_a35 with both recipes + a minimal r_B and assert the
    output carries by_recipe for BOTH, a recipe_selection, and per-behavior
    chain_rho_e0.
    """
    import issue658_fit_predictors as fit
    import numpy as np
    import torch

    ctx_ids = [f"c{i}" for i in range(12)]
    layers = [0, 1]
    h = 8
    rng = np.random.default_rng(0)
    # v0 mean summaries (Lc, H) per context.
    store = {
        "summaries": {
            "mean": {
                c: torch.tensor(rng.standard_normal((len(layers), h)), dtype=torch.float32)
                for c in ctx_ids
            }
        }
    }
    # two c_C recipes, both (Lc, H) per context.
    cc_recipes = {
        "last": {c: rng.standard_normal((len(layers), h)) for c in ctx_ids},
        "meanprompt": {c: rng.standard_normal((len(layers), h)) for c in ctx_ids},
    }
    # E0 table with one dynamic column (so the chain ρ has a target).
    e0 = _synthetic_e0_table(ctx_ids, n_probes=8, seed=1)
    # minimal r_B: a diffmeans direction per layer for the 'dynamic' column.
    rb = {
        "columns": ["dynamic"],
        "r_b": {"dynamic": {"diffmeans": [torch.tensor(rng.standard_normal(h)) for _ in layers]}},
    }
    out = fit.fit_a34_a35(store, cc_recipes, e0, rb, ctx_ids, layers)
    # BOTH recipes evaluated.
    assert set(out["by_recipe"].keys()) == {"last", "meanprompt"}, (
        "fit_a34_a35 must evaluate BOTH cc_last and cc_meanprompt (round-2 BLOCKER)"
    )
    # recipe selection encoded (the Phase-2 lock).
    assert out["recipe_selection"]["chosen_cc_recipe"] in ("last", "meanprompt")
    # chain ρ present per recipe, per behavior.
    for rec in out["by_recipe"].values():
        assert "chain_rho_e0" in rec, "each recipe must report the r_B^T M c_C → E0 chain ρ"
        assert "dynamic" in rec["chain_rho_e0"], "the dynamic behavior must have a chain ρ"
        assert "rho" in rec["chain_rho_e0"]["dynamic"]


def test_cc_recipe_selection_defaults_to_last_within_margin():
    """The §4.3-P3 rule: default to last-input-token unless meanprompt wins by margin."""
    import issue658_fit_predictors as fit

    # last and meanprompt within the margin -> default to last.
    by_recipe = {
        "last": {"per_layer": [{"ridge_mean_cos": 0.50}]},
        "meanprompt": {"per_layer": [{"ridge_mean_cos": 0.51}]},
    }
    sel = fit._select_cc_recipe(
        by_recipe, lambda r: max(p["ridge_mean_cos"] for p in r["per_layer"])
    )
    assert sel["chosen_cc_recipe"] == "last", "within margin -> default last-input-token"

    # meanprompt beats last by > margin -> meanprompt wins.
    by_recipe2 = {
        "last": {"per_layer": [{"ridge_mean_cos": 0.40}]},
        "meanprompt": {"per_layer": [{"ridge_mean_cos": 0.60}]},
    }
    sel2 = fit._select_cc_recipe(
        by_recipe2, lambda r: max(p["ridge_mean_cos"] for p in r["per_layer"])
    )
    assert sel2["chosen_cc_recipe"] == "meanprompt", "meanprompt beats last by margin -> meanprompt"


def test_no_per_cell_vllm_engine_in_e0_generation():
    """Major round-2: the E0 gen path uses ONE shared vLLM engine, not per-cell LLM().

    The per-(context×column) ``LLM(...)`` instantiation (round-1 Major: ~hundreds
    of engine startups) is gone; the shared-engine helper exists and the legacy
    per-cell sampler is removed.
    """
    import inspect

    import issue658_extract_base_store as ex

    assert hasattr(ex, "_gen_e0_vllm_shared"), "the single shared-engine E0 helper must exist"
    assert not hasattr(ex, "_gen_column_samples"), (
        "the per-(context×column) vLLM sampler must be removed (round-1 throughput Major)"
    )
    # The shared helper builds ONE LLM and reaps ONCE (count the instantiation
    # pattern `LLM(model=` so the docstring's `LLM()` mention does not match).
    src = inspect.getsource(ex._gen_e0_vllm_shared)
    assert src.count("LLM(model=") == 1, (
        "the shared E0 helper must instantiate exactly ONE LLM engine"
    )
    assert src.count("_reap_vllm(") == 1, "the shared E0 engine must be reaped exactly once"

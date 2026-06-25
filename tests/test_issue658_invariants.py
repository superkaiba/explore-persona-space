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


def test_rb_dbbar_uses_the_pool_argument_not_a_hardcoded_set():
    """build_rb_contrast's D_Bbar half IS the pool argument's [:cap] slice.

    This is the mechanism behind the (G1) v3 consistency-checker BLOCK: for the 3
    probe-based r_B columns the D_Bbar half is ``pool[:cap]``, so WHICH pool the
    G4 call site passes decides r_B's contrast baseline by genre. Asserting it
    here makes the next test's "the call site must pass the PINNED Betley pool"
    a meaningful invariant (if D_Bbar ignored the pool, the pin would be vacuous).
    Uses ``betley_vs_neutral`` (D_B = Betley main-8, local; no HF/network).
    """
    import issue658_extract_base_store as ex

    pool_a = [f"betley-probe-{i}" for i in range(10)]
    pool_b = [f"ultrachat-probe-{i}" for i in range(10)]
    cap = 4
    _, dbbar_a = ex.build_rb_contrast("broad_em", pool_a, cap)  # betley_vs_neutral
    _, dbbar_b = ex.build_rb_contrast("broad_em", pool_b, cap)
    assert dbbar_a == pool_a[:cap], "D_Bbar must be the pool[:cap] slice"
    assert dbbar_b == pool_b[:cap]
    assert dbbar_a != dbbar_b, "a different pool MUST give a different D_Bbar (the swap risk)"


def test_g4_call_site_pins_dbbar_to_the_betley_pool_not_active_probes():
    """(G1) v3 BLOCKER fix: the G4 call site passes the PINNED Betley pool to
    build_rb_contrast, NOT the active extraction ``probes`` (which is UltraChat
    under --probes-file). r_B's D_Bbar must be genre-invariant.

    Greps the dispatcher source for the exact G4 call shape so a future refactor
    that reverts the pin (back to ``build_rb_contrast(col, probes, rb_cap)``) FAILs
    CI — the un-CI-pinned guard a refactor could silently strip otherwise.
    """
    import inspect

    import issue658_extract_base_store as ex

    src = inspect.getsource(ex.main)
    assert "build_rb_contrast(col, rb_neutral_pool, rb_cap)" in src, (
        "G4 must pass the PINNED Betley pool (rb_neutral_pool) to build_rb_contrast"
    )
    assert "build_rb_contrast(col, probes, rb_cap)" not in src, (
        "G4 must NOT pass the active extraction pool (probes) — that is the genre swap "
        "the consistency-checker BLOCKED"
    )
    # rb_neutral_pool is loaded from the canonical Betley preregistered pool,
    # independent of --probes-file.
    assert "rb_neutral_pool = fetch_preregistered_probes(" in src


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


# ── round-3 regression tests ──────────────────────────────────────────────────


def test_store_manifest_files_are_sha_pinned(tmp_path):
    """BLOCKER round-3: every primary-deliverable file in the manifest is SHA-pinned.

    Constructs the pinned tensor + index files on disk and runs the PRODUCTION
    manifest-pin builder (``build_files_sha_map``), then asserts every present
    file carries a 64-hex sha256 that MATCHES ``sha256_file()`` on the artifact —
    the §6.5 sha-pinned-manifest deliverable spec the round-2 manifest violated.
    A missing pinned file (e.g. ``sigma_c.pt`` under ``--skip-sigma``) is recorded
    ``present: False`` rather than crashing.
    """
    import issue658_extract_base_store as ex
    import torch

    out = tmp_path / "store"
    (out / "answer_spans").mkdir(parents=True)
    # The three pinned tensors + the answer-spans index file.
    torch.save({"summaries": {"mean": {}}}, out / "v0_summaries.pt")
    torch.save({"r_b": {}}, out / "r_b.pt")
    torch.save({"sigma_c": torch.zeros(2, 2)}, out / "sigma_c.pt")
    (out / "answer_spans" / "index.json").write_text('{"context_ids": ["c0"]}')

    files = ex.build_files_sha_map(out)
    # All four pinned deliverables present + each carries a verifiable sha256.
    assert set(files) == set(ex.MANIFEST_PINNED_FILES)
    for rel, entry in files.items():
        assert entry["present"] is True, rel
        sha = entry["sha256"]
        assert isinstance(sha, str) and len(sha) == 64, f"{rel} sha not 64-hex: {sha}"
        assert all(ch in "0123456789abcdef" for ch in sha), rel
        # The recorded sha matches a fresh sha256_file() over the artifact.
        assert sha == common.sha256_file(out / rel), f"{rel} manifest sha != on-disk sha"
        assert entry["bytes"] == (out / rel).stat().st_size, rel


def test_store_manifest_missing_pinned_file_recorded_not_crashed(tmp_path):
    """A descoped deliverable (--skip-sigma drops sigma_c.pt) records present:False."""
    import issue658_extract_base_store as ex
    import torch

    out = tmp_path / "store"
    (out / "answer_spans").mkdir(parents=True)
    torch.save({"summaries": {"mean": {}}}, out / "v0_summaries.pt")
    torch.save({"r_b": {}}, out / "r_b.pt")
    (out / "answer_spans" / "index.json").write_text('{"context_ids": []}')
    # sigma_c.pt deliberately ABSENT (the --skip-sigma descope).

    files = ex.build_files_sha_map(out)
    assert files["sigma_c.pt"]["present"] is False
    assert files["sigma_c.pt"]["sha256"] is None
    # The present files are still pinned.
    assert files["v0_summaries.pt"]["present"] is True
    assert len(files["v0_summaries.pt"]["sha256"]) == 64


def test_a35_mlp_uses_named_shared_dim_not_hardcoded_8(tmp_path):
    """Major round-3: A3.5 nonlinear_gap reads ridge AND MLP over the SAME dims.

    Production (``feat_dim=0``) must use the NAMED shared target dim
    ``A35_MLP_TARGET_DIM`` (NOT the old unconditional ``min(8, ...)``), and the
    gap's ridge-cos is read over the SAME ``gap_target_dim`` as the MLP — so
    ``nonlinear_gap`` is a like-for-like comparison. The full-dim A3.4
    ``ridge_mean_cos`` (recipe-lock / chain-ρ statistic) is left untouched.
    """
    import issue658_fit_predictors as fit
    import numpy as np
    import torch

    # H larger than both 8 and A35_MLP_TARGET_DIM so the slicing is observable.
    h = fit.A35_MLP_TARGET_DIM + 16
    assert h > 8, "the regression must use H > 8 to catch the old min(8, ...) cap"
    ctx_ids = [f"c{i}" for i in range(10)]
    layers = [0]
    rng = np.random.default_rng(0)
    store = {
        "summaries": {
            "mean": {
                c: torch.tensor(rng.standard_normal((1, h)), dtype=torch.float32) for c in ctx_ids
            }
        }
    }
    cc_map = {c: rng.standard_normal((1, h)) for c in ctx_ids}
    e0 = _synthetic_e0_table(ctx_ids, n_probes=6, seed=1)
    rb = {"columns": [], "r_b": {}}
    # feat_dim=0 -> PRODUCTION path (full H target for the ridge).
    out = fit._fit_a34_a35_one_recipe(cc_map, store, e0, rb, ctx_ids, layers, 658, feat_dim=0)
    pl = out["per_layer"][0]
    # The MLP/ridge gap is read over the NAMED shared dim, NOT 8, NOT full H.
    assert pl["gap_target_dim"] == fit.A35_MLP_TARGET_DIM, (
        f"production A3.5 gap must use A35_MLP_TARGET_DIM={fit.A35_MLP_TARGET_DIM}, "
        f"got {pl['gap_target_dim']} (the old min(8, ...) cap is gone)"
    )
    assert pl["gap_target_dim"] != 8, "must not be the old hard-coded 8-dim cap"
    # The gap is mlp_cos - ridge_cos BOTH on gap_dim (like-for-like).
    assert abs(pl["nonlinear_gap"] - (pl["mlp_mean_cos"] - pl["ridge_mean_cos_on_gap_dim"])) < 1e-9
    # The full-dim A3.4 statistic is still reported separately (unchanged).
    assert "ridge_mean_cos" in pl


def test_a35_gap_dim_respects_smoke_feat_clamp():
    """Under the smoke feat clamp, gap_target_dim is bounded by the clamped H."""
    import issue658_fit_predictors as fit
    import numpy as np
    import torch

    feat = 32  # smaller than A35_MLP_TARGET_DIM so the clamp bounds the gap dim
    assert feat < fit.A35_MLP_TARGET_DIM
    h = 80
    ctx_ids = [f"c{i}" for i in range(8)]
    layers = [0]
    rng = np.random.default_rng(2)
    store = {
        "summaries": {
            "mean": {
                c: torch.tensor(rng.standard_normal((1, h)), dtype=torch.float32) for c in ctx_ids
            }
        }
    }
    cc_map = {c: rng.standard_normal((1, h)) for c in ctx_ids}
    e0 = _synthetic_e0_table(ctx_ids, n_probes=6, seed=2)
    rb = {"columns": [], "r_b": {}}
    out = fit._fit_a34_a35_one_recipe(cc_map, store, e0, rb, ctx_ids, layers, 658, feat_dim=feat)
    # gap dim = min(A35_MLP_TARGET_DIM, clamped H=feat) = feat.
    assert out["per_layer"][0]["gap_target_dim"] == feat


def test_rb_recipes_descopes_fewshot():
    """CONCERN round-3: RB_RECIPES matches what fit_a33 actually scores (no fewshot).

    The plan's few-shot-final recipe is descoped; RB_RECIPES must equal the two
    contrastive recipes fit_a33 loops, and `fewshot` must be gone — so the
    declaration no longer over-promises a recipe the extractor never produces.
    """
    import inspect

    import issue658_fit_predictors as fit

    assert "fewshot" not in common.RB_RECIPES, "few-shot-final r_B is descoped for #658"
    assert set(common.RB_RECIPES) == {"diffmeans", "meanDB"}
    # The fit_a33 loop scores exactly the declared recipes (no silent mismatch).
    src = inspect.getsource(fit.fit_a33)
    for rec in common.RB_RECIPES:
        assert rec in src, f"fit_a33 must score the declared r_B recipe {rec}"
    assert "fewshot" not in src, "fit_a33 must not reference the descoped fewshot recipe"


# ── (G1) genre-delta Δρ CI contract (round-2 BLOCKER: genre-delta-ci-contract- ──
# unenforced) — Codex r1 FAIL upheld by the reconciler. A dynamic-range cell MUST
# carry a full ≥2000-resample Δρ CI; only H3 (no dynamic range on both pools) rows
# carry delta_rho_ci:null. The legacy `delta-ci-unavailable-no-draws` silent-null
# verdict is removed; missing draws / <2000 draws on a dynamic-range cell RAISE.


def _write_genre_arm(
    arm_dir: Path,
    *,
    columns: list[str],
    dyn_std: dict[str, bool],
    best_rho: dict[str, float],
    draws: dict[str, list[float] | None],
    n_ctx: int = 8,
    suffix: str = "",
) -> None:
    """Write a minimal aggregate/a32_cells/E0_expression JSON triple for one arm.

    ``dyn_std[col]`` True → per-context E0 varies (dynamic range); False → all-equal
    (std 0 → no dynamic range). ``draws[col]`` is the best cell's bootstrap draws
    list (or None to omit the draws key, exercising the missing-draws path).
    ``suffix`` ("" or "_smoke") matches the filenames ``compute_genre_delta`` reads
    in that mode.
    """
    from issue658_common import dump_json

    arm_dir.mkdir(parents=True, exist_ok=True)
    # E0_expression.json: per-context rate per column.
    e0_cells = {}
    for i in range(n_ctx):
        cell = {}
        for col in columns:
            # dynamic → vary by context index; static → constant.
            cell[col] = {"rate": float(i) / n_ctx if dyn_std[col] else 0.5}
        e0_cells[f"c{i}"] = cell
    dump_json({"columns": columns, "e0": e0_cells}, arm_dir / f"E0_expression{suffix}.json")
    # aggregate.json: best (layer, summary) verdict per column.
    verdicts = {}
    for col in columns:
        verdicts[col] = {
            "best_rho": best_rho.get(col),
            "best_layer": 0,
            "best_summary": "mean",
            "noise_floor_p95": 0.2,
        }
    dump_json({"a32_verdicts": verdicts}, arm_dir / f"aggregate{suffix}.json")
    # a32_cells.json: the best cell per column with (optional) bootstrap draws.
    a32 = []
    for col in columns:
        boot = {} if draws.get(col) is None else {"draws": draws[col]}
        a32.append({"column": col, "layer": 0, "recipe": "mean", "bootstrap": boot})
    dump_json({"a32": a32}, arm_dir / f"a32_cells{suffix}.json")


def test_genre_delta_dynamic_range_missing_draws_raises(tmp_path):
    """BLOCKER A: a dynamic-range cell with draws=None RAISES (no silent null CI).

    Was the missing coverage (Codex r1) + the silent `delta-ci-unavailable-no-draws`
    verdict. Both arms dynamic-range, both best_rho present, but the UltraChat best
    cell has NO bootstrap draws → contract violation → RuntimeError naming the
    behavior + offending arm + cause + cached input path.
    """
    import issue658_genre_delta as gd
    import pytest

    betley = tmp_path / "betley"
    ultra = tmp_path / "ultra"
    cols = ["sycophancy"]
    _write_genre_arm(
        betley,
        columns=cols,
        dyn_std={"sycophancy": True},
        best_rho={"sycophancy": 0.47},
        draws={"sycophancy": [0.4] * gd.MIN_RESAMPLES_PRODUCTION},
    )
    _write_genre_arm(
        ultra,
        columns=cols,
        dyn_std={"sycophancy": True},
        best_rho={"sycophancy": 0.41},
        draws={"sycophancy": None},  # the bug: dynamic-range cell, no draws
    )
    with pytest.raises(RuntimeError, match="draws missing"):
        gd.compute_genre_delta(betley, ultra, smoke=False)


def test_genre_delta_dynamic_range_short_draws_raises_in_production(tmp_path):
    """BLOCKER B: a dynamic-range cell with <2000 draws RAISES in production mode.

    Degenerate-resample drops can leave an arm's draws below the registered 2000.
    Production must reject it (plan v3 §6/§6.5/§11), never bootstrap Δρ from a short
    list. The raise names the behavior + per-arm draw counts.
    """
    import issue658_genre_delta as gd
    import pytest

    betley = tmp_path / "betley"
    ultra = tmp_path / "ultra"
    cols = ["sycophancy"]
    _write_genre_arm(
        betley,
        columns=cols,
        dyn_std={"sycophancy": True},
        best_rho={"sycophancy": 0.47},
        draws={"sycophancy": [0.4] * gd.MIN_RESAMPLES_PRODUCTION},
    )
    _write_genre_arm(
        ultra,
        columns=cols,
        dyn_std={"sycophancy": True},
        best_rho={"sycophancy": 0.41},
        draws={"sycophancy": [0.3] * 1500},  # 1500 < 2000 → reject in production
    )
    with pytest.raises(ValueError, match="INDEPENDENT resamples per arm"):
        gd.compute_genre_delta(betley, ultra, smoke=False)


def test_genre_delta_smoke_override_bypasses_floors(tmp_path):
    """The smoke escape hatch bypasses BOTH the missing-floor and the <2000 floor.

    --smoke-allow-small-bootstrap (allow_small_bootstrap=True) relaxes the ≥2000
    floor for tiny fixtures. Both arms dynamic-range with a handful of draws each →
    no raise, a real CI is computed.
    """
    import issue658_genre_delta as gd

    betley = tmp_path / "betley"
    ultra = tmp_path / "ultra"
    cols = ["sycophancy"]
    _write_genre_arm(
        betley,
        columns=cols,
        dyn_std={"sycophancy": True},
        best_rho={"sycophancy": 0.47},
        draws={"sycophancy": [0.4, 0.45, 0.5, 0.42, 0.48]},
        suffix="_smoke",
    )
    _write_genre_arm(
        ultra,
        columns=cols,
        dyn_std={"sycophancy": True},
        best_rho={"sycophancy": 0.41},
        draws={"sycophancy": [0.3, 0.35, 0.4, 0.32, 0.38]},
        suffix="_smoke",
    )
    result = gd.compute_genre_delta(betley, ultra, smoke=True, allow_small_bootstrap=True)
    row = result["rows"][0]
    assert row["behavior"] == "sycophancy"
    assert row["delta_rho_ci"] is not None, "the smoke override must compute a real CI"
    assert row["verdict"] in ("H1-consistent", "H2-genre-bound")
    # The removed verdict must never appear.
    assert row["verdict"] != "delta-ci-unavailable-no-draws"


def test_genre_delta_healthy_dynamic_range_computes_ci(tmp_path):
    """Happy path: both arms dynamic-range with full ≥2000 draws → a real CI lands.

    n_resamples == MIN_RESAMPLES_PRODUCTION and the contract fields are present;
    no raise, no null CI on a dynamic-range cell.
    """
    import issue658_genre_delta as gd

    betley = tmp_path / "betley"
    ultra = tmp_path / "ultra"
    cols = ["sycophancy"]
    rng = __import__("numpy").random.default_rng(0)
    _write_genre_arm(
        betley,
        columns=cols,
        dyn_std={"sycophancy": True},
        best_rho={"sycophancy": 0.47},
        draws={"sycophancy": list(rng.normal(0.47, 0.05, gd.MIN_RESAMPLES_PRODUCTION))},
    )
    _write_genre_arm(
        ultra,
        columns=cols,
        dyn_std={"sycophancy": True},
        best_rho={"sycophancy": 0.41},
        draws={"sycophancy": list(rng.normal(0.41, 0.05, gd.MIN_RESAMPLES_PRODUCTION))},
    )
    result = gd.compute_genre_delta(betley, ultra, smoke=False)
    row = result["rows"][0]
    ci = row["delta_rho_ci"]
    assert ci is not None
    assert ci["n_resamples"] == gd.MIN_RESAMPLES_PRODUCTION
    assert set(ci) == {"lower", "upper", "n_resamples", "null_overlap"}
    assert result["n_behaviors_compared"] == 1


def test_genre_delta_h3_no_dynamic_range_keeps_null_ci(tmp_path):
    """H3 stays the ONLY null-CI path: a behavior without dynamic range on BOTH pools.

    UltraChat arm is floored (std 0 → no dynamic range) → H3, delta_rho_ci:null,
    no raise. This is the legitimate null-CI case the contract preserves.
    """
    import issue658_genre_delta as gd

    betley = tmp_path / "betley"
    ultra = tmp_path / "ultra"
    cols = ["refusal"]
    _write_genre_arm(
        betley,
        columns=cols,
        dyn_std={"refusal": True},
        best_rho={"refusal": 0.5},
        draws={"refusal": [0.5] * gd.MIN_RESAMPLES_PRODUCTION},
    )
    _write_genre_arm(
        ultra,
        columns=cols,
        dyn_std={"refusal": False},  # floored on the generic genre → H3
        best_rho={"refusal": None},
        draws={"refusal": None},
    )
    result = gd.compute_genre_delta(betley, ultra, smoke=False)
    row = result["rows"][0]
    assert row["verdict"] == "H3-no-dynamic-range"
    assert row["delta_rho_ci"] is None
    assert result["n_behaviors_h3"] == 1


def test_delta_rho_ci_floor_enforced_directly():
    """Unit: _delta_rho_ci raises below min_resamples, succeeds at/above it."""
    import issue658_genre_delta as gd
    import pytest

    short = [0.1] * 50
    with pytest.raises(ValueError, match="≥2000 INDEPENDENT resamples"):
        gd._delta_rho_ci(short, short, seed=1, min_resamples=2000, behavior="x")
    ok = [0.1, 0.2, 0.3, 0.4, 0.5]
    out = gd._delta_rho_ci(ok, ok, seed=1, min_resamples=5, behavior="x")
    assert out["n_resamples"] == 5 and "null_overlap" in out


def test_cluster_bootstrap_emits_full_nboot_or_none_for_tiny():
    """Upstream contract: full n_boot draws for n>=4, None only for a tiny (n<4) cell."""
    import issue658_fit_predictors as fit
    import numpy as np

    rng = np.random.default_rng(0)
    pred = rng.standard_normal(10)
    meas = pred + rng.standard_normal(10) * 0.3
    out = fit._cluster_bootstrap_rho(pred, meas, n_boot=300, seed=1)
    assert out is not None
    assert len(out["draws"]) == 300, "a healthy n>=4 cell emits exactly n_boot draws"
    # n<4 → the legitimate tiny-cell None (genre-delta gate flags it H3).
    tiny = fit._cluster_bootstrap_rho(
        rng.standard_normal(3), rng.standard_normal(3), n_boot=300, seed=1
    )
    assert tiny is None


def test_extractor_genre_tag_must_be_canonical_for_genre_arm():
    """Path-naming (Codex r1 Minor): the genre arm rejects a non-canonical --genre-tag.

    The extractor's canonical tag MUST equal the prefix genre_delta.py reads back as
    its --ultrachat-dir default — eliminating the default mismatch where a stale
    `--genre-tag ultrachat` routed outputs to a dir the genre delta never reads.
    """
    import inspect

    import issue658_extract_base_store as ext
    import issue658_genre_delta as gd
    from issue658_common import EVAL_RESULTS_DIR

    assert ext.CANONICAL_GENRE_TAG == "genre-generalization-ultrachat"
    default_ultra = EVAL_RESULTS_DIR / ext.CANONICAL_GENRE_TAG
    assert str(default_ultra).endswith("genre-generalization-ultrachat")
    # genre_delta.py's --ultrachat-dir argparse default reads the same prefix.
    src = inspect.getsource(gd.main)
    assert ext.CANONICAL_GENRE_TAG in src, (
        "genre_delta.py --ultrachat-dir default must equal the extractor's canonical tag"
    )
    # The extractor's guard enforces the canonical tag for the genre arm.
    ext_src = inspect.getsource(ext.main)
    assert "CANONICAL_GENRE_TAG" in ext_src, (
        "the extractor must fail loud when --probes-file is set but --genre-tag is non-canonical"
    )


def test_extractor_genre_tag_guard_fires_at_runtime_before_phase_work(monkeypatch):
    """Codex r2 Minor 2 (runtime): the --genre-tag guard SystemExits before phase work.

    The source-inspection test above proves the guard string is present; this drives
    ``ext.main()`` with ``--probes-file <path> --genre-tag ultrachat`` (the stale
    non-canonical tag) and asserts the ``SystemExit`` actually fires — and that it
    fires BEFORE any phase work starts (the guard sits above ``phase("load")``, so a
    bogus probes-file path is never opened: argparse only parses the Path, it does
    not stat it). No model load, no CUDA, no file I/O.
    """
    import sys

    import issue658_extract_base_store as ext
    import pytest

    argv = [
        "issue658_extract_base_store.py",
        "--probes-file",
        "/nonexistent/ultrachat_pool.json",  # never opened — the guard raises first
        "--genre-tag",
        "ultrachat",  # the stale non-canonical tag
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="MUST be the canonical"):
        ext.main()


def test_genre_delta_compute_rejects_allow_small_bootstrap_outside_smoke(tmp_path):
    """Codex r2 Minor 3a (API guard): compute_genre_delta raises when the smoke-only
    override is set without smoke.

    ``allow_small_bootstrap`` is a SMOKE-ONLY escape hatch — calling
    ``compute_genre_delta(..., smoke=False, allow_small_bootstrap=True)`` must raise
    (production never relaxes the ≥2000-resample floor). The raise fires on the
    argument contract, before any arm input is read, so the (empty) tmp dirs are fine.
    """
    import issue658_genre_delta as gd
    import pytest

    with pytest.raises(ValueError, match="SMOKE-ONLY escape hatch"):
        gd.compute_genre_delta(
            tmp_path / "betley",
            tmp_path / "ultra",
            smoke=False,
            allow_small_bootstrap=True,
        )


def test_genre_delta_cli_rejects_smoke_override_without_smoke_flag(monkeypatch):
    """Codex r2 Minor 3b (CLI guard): argparse errors on --smoke-allow-small-bootstrap
    passed WITHOUT --smoke.

    ``main()`` calls ``parser.error(...)`` (a SystemExit) before any arm work when
    the override is set outside --smoke, so production can never relax the floor from
    the CLI either.
    """
    import sys

    import issue658_genre_delta as gd
    import pytest

    argv = [
        "issue658_genre_delta.py",
        "--smoke-allow-small-bootstrap",  # no --smoke → argparse error
    ]
    monkeypatch.setattr(sys, "argv", argv)
    # argparse parser.error exits with code 2 (SystemExit).
    with pytest.raises(SystemExit):
        gd.main()


# ── round-3 floored-cell graceful-bootstrap fix (Codex r2 Critical, via reconciler) ──
# fit_a32 calls _cluster_bootstrap_rho UNCONDITIONALLY for every n>=4 cell. For a
# FLOORED cell (n>=4 but the E0 target y is constant — broad_em / harmful_compliance /
# refusal on UltraChat), every resample is degenerate so the round-2 raise contract
# on _cluster_bootstrap_rho would CRASH fit_a32 before a32_cells.json lands. The fix
# guards the callsite on `rho is None` (the canonical no-rank-signal sentinel), so a
# floored cell emits `bootstrap: None` and the plan-anticipated H3 path runs.


def _constant_y_e0_table(ctx_ids, *, col="floored_uc", const=0.0):
    """An E0 table whose ONE column has a CONSTANT per-context rate (no rank signal).

    ``e0_target(e0, col, ctx_ids)`` returns a constant ``y`` for this column, so
    ``_rho(pred, y)`` is None and every bootstrap resample is degenerate — exactly
    the floored UltraChat cell the round-2 raise would crash on.
    """
    e0 = {c: {col: {"rate": float(const)}} for c in ctx_ids}
    return {"e0": e0, "columns": [col]}


def _mean_store(ctx_ids, *, n_layers, h, seed=0):
    """A minimal store with only the 'mean' v0 summaries (Lc, H) per context."""
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    return {
        "summaries": {
            "mean": {
                c: torch.tensor(rng.standard_normal((n_layers, h)), dtype=torch.float32)
                for c in ctx_ids
            }
        },
        "capture_layers": list(range(n_layers)),
    }


def test_fit_a32_floored_cell_emits_null_bootstrap_no_crash():
    """Codex r2 Critical (via reconciler): a FLOORED n>=4 cell does NOT crash fit_a32.

    The floored cell (constant E0 target, n>=4) yields rho=None; the round-3 callsite
    guard skips _cluster_bootstrap_rho (which would correctly RAISE on 0 valid draws)
    and emits ``bootstrap: None``. The plan v3 §6 floor guard / §3 Risk 1 / H3 line
    126 graceful path — NEVER a crash. Round 2 converted this anticipated path into a
    hard RuntimeError; this pins the fix.
    """
    import issue658_fit_predictors as fit

    ctx_ids = [f"c{i}" for i in range(8)]  # n=8 >= 4 (the crash regime, not the n<4 skip)
    layers = [0, 1]
    h = 6
    store = _mean_store(ctx_ids, n_layers=len(layers), h=h, seed=1)
    e0 = _constant_y_e0_table(ctx_ids, col="floored_uc", const=0.0)
    noise = {"floored_uc": 1.0}
    base_prior = {"floored_uc": None}

    # MUST NOT raise (round-2 would crash here on the degenerate bootstrap).
    cells = fit.fit_a32(
        store,
        spans_dir=None,  # never read: recipes excludes "attn"
        e0=e0,
        ctx_ids=ctx_ids,
        layers=layers,
        recipes=["mean"],
        noise_floor=noise,
        base_prior=base_prior,
    )
    scored = [c for c in cells if c.get("status") != "too_few_contexts"]
    assert scored, "the n=8 floored cell must be SCORED (n>=4), not skipped as too_few_contexts"
    for c in scored:
        assert c["n"] == 8
        assert c["rho"] is None, "a constant-y floored cell has no rank signal -> rho None"
        assert c["bootstrap"] is None, (
            "a floored cell must emit bootstrap:None (the callsite skips the bootstrap "
            "when rho is None) — NOT crash on the round-2 degenerate-resample raise"
        )


def test_fit_a32_healthy_cell_still_carries_full_bootstrap():
    """The round-2 contract still holds for a HEALTHY n>=4 cell: a real bootstrap lands.

    The floored-cell guard must NOT suppress the bootstrap on a dynamic-range cell —
    a column whose per-context E0 varies yields rho != None and a full
    N_BOOTSTRAP-draw bootstrap (the contract the genre-delta Δρ CI consumes).
    """
    import issue658_fit_predictors as fit
    import numpy as np
    import torch

    ctx_ids = [f"c{i}" for i in range(10)]
    layers = [0]
    h = 6
    rng = np.random.default_rng(3)
    # A store whose mean-summary column 0 carries a signal linearly related to y, so
    # the LOCO MLP recovers a non-degenerate prediction → rho != None.
    y_vals = [float(i) / len(ctx_ids) for i in range(len(ctx_ids))]
    store = {
        "summaries": {
            "mean": {
                c: torch.tensor(
                    np.column_stack([[y_vals[i]] * 1, rng.standard_normal((1, h - 1))]),
                    dtype=torch.float32,
                )
                for i, c in enumerate(ctx_ids)
            }
        },
        "capture_layers": [0],
    }
    e0 = {"e0": {c: {"dyn_uc": {"rate": y_vals[i]}} for i, c in enumerate(ctx_ids)}}
    e0["columns"] = ["dyn_uc"]
    noise = {"dyn_uc": 0.2}
    base_prior = {"dyn_uc": None}

    # Use the production N_BOOTSTRAP via the real code path (no smoke clamp here).
    cells = fit.fit_a32(store, None, e0, ctx_ids, layers, ["mean"], noise, base_prior)
    scored = [c for c in cells if c.get("status") != "too_few_contexts"]
    assert scored
    # At least one scored cell must carry a real bootstrap with the full draw count
    # (the round-2 contract: a dynamic-range n>=4 cell carries a full bootstrap).
    with_boot = [c for c in scored if c["rho"] is not None and c["bootstrap"] is not None]
    assert with_boot, "a dynamic-range cell must keep its full bootstrap (contract preserved)"
    for c in with_boot:
        assert len(c["bootstrap"]["draws"]) == fit.N_BOOTSTRAP, (
            "a healthy n>=4 cell must carry exactly N_BOOTSTRAP draws (round-2 contract)"
        )
        assert "ci95" in c["bootstrap"]


def test_cluster_bootstrap_raises_on_near_degenerate_n_ge_4():
    """Codex r2 Minor 1: the round-2 raise contract still holds on a truly degenerate
    n>=4 input.

    This is NOT a floored cell (those are now skipped at the fit_a32 callsite) — it is
    the direct ``_cluster_bootstrap_rho`` call on a near-degenerate n>=4 input where
    EVERY resample drops, so the retry cap is exhausted and the round-2 RuntimeError
    fires. Proves the raise contract is intact for genuinely degenerate direct calls.
    """
    import issue658_fit_predictors as fit
    import numpy as np
    import pytest

    # pred is constant (all-equal) → _rho returns None on EVERY resample (std<1e-9),
    # so 0 valid draws accumulate within the retry cap → the round-2 raise fires.
    pred = np.ones(8)
    meas = np.arange(8, dtype=np.float64)
    with pytest.raises(RuntimeError, match="could not accumulate"):
        fit._cluster_bootstrap_rho(pred, meas, n_boot=20, seed=1)

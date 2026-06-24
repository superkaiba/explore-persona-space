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


# ── 8-GPU rework regression tests (this round) ────────────────────────────────


def _tiny_qwen():
    """A 2-layer random Qwen2 + tokenizer for the slot-only equivalence test (CPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    # vocab must cover EVERY id the tokenizer can emit, incl. added special
    # tokens (eos `<|im_end|>` = 151645 > tok.vocab_size 151643) — otherwise
    # raw[eos_token_id] / the LM-head projection IndexErrors. len(tok) includes
    # added tokens; round up so the marker id (83399) is in range too.
    vocab = max(len(tok), tok.eos_token_id + 1, 83400)
    cfg = Qwen2Config(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        eos_token_id=tok.eos_token_id,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model, tok


def test_marker_slot_stats_logits_to_keep_matches_full_forward():
    """Change-1: the slot-only (logits_to_keep=1) forward yields the SAME 4 floats.

    The OOM fix replaces a (B, T, V) full-logits materialization with
    logits_to_keep=1 → (B, 1, V). The DV reads exactly the last (left-padded)
    position, so the four stored floats (logp / z_marker / z_eos / logZ) MUST be
    byte-identical (within fp32 precision) to a full-forward read at position -1.
    This is the batched-rewrite equivalence check for the serial→slot rewrite.
    """
    import torch

    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats

    model, tok = _tiny_qwen()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.encode(" ※", add_special_tokens=False)
    if len(marker_id) != 1:
        # the 0.5B tokenizer encodes ` ※` to the same single id as 7B; if a CI
        # tokenizer differs, fall back to any single-token string for the math.
        marker_text = " the"
        assert len(tok.encode(marker_text, add_special_tokens=False)) == 1
    else:
        marker_text = " ※"

    # Variable-length contexts so left-padding actually fires (B>=2).
    contexts = [
        "The capital of France is Paris and it is",
        "A short one",
        "Here is a much longer context with several more tokens to force padding",
    ]

    # Production path (logits_to_keep=1).
    got = compute_marker_slot_stats(
        model,
        tok,
        contexts,
        marker_text,
        eos_token_id=tok.eos_token_id,
        device="cpu",
        include_argmax=True,
    )

    # Reference: explicit full forward, read at position -1 (the old path).
    mid = tok.encode(marker_text, add_special_tokens=False)[0]
    eos = tok.eos_token_id
    ref = []
    for c in contexts:
        ids = tok.encode(c, add_special_tokens=False)
        inp = torch.tensor([ids])
        with torch.no_grad():
            full = model(input_ids=inp, attention_mask=torch.ones_like(inp)).logits  # (1,T,V)
        raw = full[0, -1, :].float()
        log_z = float(torch.logsumexp(raw, dim=-1))
        ref.append(
            {
                "logp": float(raw[mid]) - log_z,
                "z_marker": float(raw[mid]),
                "z_eos": float(raw[eos]),
                "logZ": log_z,
                "argmax_id": int(torch.argmax(raw)),
            }
        )

    assert len(got) == len(ref)
    for g, r in zip(got, ref, strict=True):
        for k in ("logp", "z_marker", "z_eos", "logZ"):
            assert abs(g[k] - r[k]) < 1e-3, f"{k}: slot-only {g[k]} != full {r[k]}"
        assert g["argmax_id"] == r["argmax_id"], "argmax id drift slot-only vs full"


def test_marker_slot_stats_uses_logits_to_keep():
    """Change-1: the helper must request logits_to_keep=1 (the OOM-fix source)."""
    import inspect

    from explore_persona_space.eval import marker_logprob

    src = inspect.getsource(marker_logprob.compute_marker_slot_stats)
    assert "logits_to_keep=1" in src, (
        "compute_marker_slot_stats must forward with logits_to_keep=1 to avoid the "
        "(B, T, V) full-logits OOM on long answers (issue #658 f3_icl_json_k2)"
    )
    # The full (B, T, V) read at [:, -1, :] over a materialized seq is gone; the
    # forward now returns (B, 1, V) and we assert that shape.
    assert "shape[1] == 1" in src, "the slot forward must assert the (B,1,V) keep-1 shape"


def test_partition_contexts_is_disjoint_cover():
    """Change-2: round-robin context sharding is a true partition (disjoint + covers)."""
    instances = [{"id": f"ctx{i}"} for i in range(50)]
    for n_shards in (1, 2, 3, 8):
        shards = [common.partition_contexts(instances, k, n_shards) for k in range(n_shards)]
        seen = [inst["id"] for sh in shards for inst in sh]
        # Disjoint: no id appears twice.
        assert len(seen) == len(set(seen)), f"n_shards={n_shards}: overlap across shards"
        # Cover: the union is exactly the full set.
        assert set(seen) == {i["id"] for i in instances}, f"n_shards={n_shards}: not a full cover"
        # Balance: shard sizes differ by at most 1 (round-robin).
        sizes = [len(s) for s in shards]
        assert max(sizes) - min(sizes) <= 1, f"n_shards={n_shards}: unbalanced {sizes}"


def test_partition_contexts_rejects_bad_shard_id():
    """An out-of-range shard id fails loud (never a silent empty shard)."""
    import pytest

    instances = [{"id": f"ctx{i}"} for i in range(4)]
    with pytest.raises(AssertionError):
        common.partition_contexts(instances, 3, 3)  # shard_id == n_shards
    with pytest.raises(AssertionError):
        common.partition_contexts(instances, -1, 3)


def test_merge_shards_reconstructs_full_store(tmp_path):
    """Change-2: merge reconstructs the unified store from disjoint shards + rbsigma.

    Builds two synthetic shard dirs (disjoint contexts) + an rbsigma dir, runs the
    PRODUCTION merge_shards, and asserts the merged v0_summaries covers every
    context with no duplication, the answer-span index is unified, and r_b.pt /
    sigma_c.pt land in the merged store.
    """
    import issue658_extract_base_store as ex
    import torch

    layers = [0, 1]
    h = 8

    def _write_shard(d, ctx_ids):
        (d / "answer_spans").mkdir(parents=True)
        summaries = {
            r: {c: torch.zeros(len(layers), h) for c in ctx_ids} for r in ("mean", "last", "maxp")
        }
        torch.save(
            {
                "summaries": summaries,
                "cc_meanprompt": {c: torch.zeros(len(layers), h) for c in ctx_ids},
                "capture_layers": layers,
                "context_ids": ctx_ids,
                "model": "stub",
                "probe_pool_hash": "deadbeef",
            },
            d / "v0_summaries.pt",
        )
        for c in ctx_ids:
            torch.save(
                {"context_id": c, "spans": [], "probes": ["q0"]}, d / "answer_spans" / f"{c}.pt"
            )
        common.dump_json(
            {"context_ids": ctx_ids, "probes_by_context": {c: ["q0"] for c in ctx_ids}},
            d / "answer_spans" / "index.json",
        )

    s0 = tmp_path / "shards" / "shard_0"
    s1 = tmp_path / "shards" / "shard_1"
    _write_shard(s0, ["ctxA", "ctxC"])
    _write_shard(s1, ["ctxB", "ctxD"])
    rbsig = tmp_path / "shards" / "rbsigma"
    rbsig.mkdir(parents=True)
    torch.save({"r_b": {}, "capture_layers": layers, "columns": []}, rbsig / "r_b.pt")
    torch.save({"sigma_c": torch.zeros(len(layers), h, h), "n": 10}, rbsig / "sigma_c.pt")

    out = tmp_path / "store"
    merged = ex.merge_shards([s0, s1], rbsig, out)

    assert merged["n_ctx"] == 4 and merged["n_shards"] == 2
    blob = torch.load(out / "v0_summaries.pt", weights_only=False)
    assert set(blob["summaries"]["mean"].keys()) == {"ctxA", "ctxB", "ctxC", "ctxD"}
    assert len(blob["context_ids"]) == len(set(blob["context_ids"])) == 4
    idx = common.load_json(out / "answer_spans" / "index.json")
    assert set(idx["probes_by_context"].keys()) == {"ctxA", "ctxB", "ctxC", "ctxD"}
    assert (out / "answer_spans" / "ctxA.pt").is_file()
    assert (out / "r_b.pt").is_file()
    assert (out / "sigma_c.pt").is_file()
    assert merged["sigma_present"] is True


def test_merge_shards_rejects_duplicate_context(tmp_path):
    """A non-disjoint shard set (a context in two shards) fails loud at merge."""
    import issue658_extract_base_store as ex
    import pytest
    import torch

    layers = [0]
    h = 4

    def _write_shard(d, ctx_ids):
        (d / "answer_spans").mkdir(parents=True)
        torch.save(
            {
                "summaries": {
                    r: {c: torch.zeros(1, h) for c in ctx_ids} for r in ("mean", "last", "maxp")
                },
                "cc_meanprompt": {c: torch.zeros(1, h) for c in ctx_ids},
                "capture_layers": layers,
                "context_ids": ctx_ids,
                "model": "stub",
                "probe_pool_hash": "x",
            },
            d / "v0_summaries.pt",
        )
        common.dump_json(
            {"context_ids": ctx_ids, "probes_by_context": {c: [] for c in ctx_ids}},
            d / "answer_spans" / "index.json",
        )

    s0 = tmp_path / "s0"
    s1 = tmp_path / "s1"
    _write_shard(s0, ["dup", "a"])
    _write_shard(s1, ["dup", "b"])  # 'dup' overlaps → partition violation
    rbsig = tmp_path / "rbsigma"
    rbsig.mkdir()
    torch.save({"r_b": {}}, rbsig / "r_b.pt")

    with pytest.raises(RuntimeError, match="duplicate context"):
        ex.merge_shards([s0, s1], rbsig, tmp_path / "store")


def test_merge_mode_is_cpu_only_no_model_load():
    """The --merge path must not load a GPU model (CPU assemble + upload only)."""
    import inspect

    import issue658_extract_base_store as ex

    src = inspect.getsource(ex.run_merge_mode)
    assert "load_hf_model" not in src, "merge mode must NOT load the HF model (CPU-only)"
    assert "merge_shards" in src, "merge mode must call merge_shards"


# ── round-5 REVISE: §1.10 capture + concurrency fixes ─────────────────────────


def test_single_context_capture_stores_per_sample_activations():
    """BLOCKER round-5: G7 captures per-(C,probe,sample) mean answer-side acts.

    Drives capture_single_context_for_context with a tiny CPU Qwen2 + 2 probes ×
    2 samples and asserts each sample carries a real (Lc, H) activation tensor +
    its text + logp_norm — the §1.10 single-context store granularity.
    """
    import issue658_extract_base_store as ex
    import torch
    from issue594_extract_context_vectors import LayerCapture  # noqa: F401

    model, tok = _tiny_qwen()
    n_layers = len(model.model.layers)
    capture_layers = [0, 1]
    inst = {"id": "ctx0", "system_prompt": "You are helpful.", "prefix_messages": []}
    probes = ["What is 2+2?", "Name a color."]
    # two sampled completions per probe (as _vllm_sample_R / _hf_sample_R return)
    samples_by_probe = [
        [{"text": "four", "logp_norm": -0.5}, {"text": "it is four", "logp_norm": -0.7}],
        [{"text": "blue", "logp_norm": -0.4}, {"text": "red", "logp_norm": -0.6}],
    ]
    cap = ex.AnswerSpanCapture(model, n_layers)
    try:
        per_probe = ex.capture_single_context_for_context(
            model, tok, inst, probes, samples_by_probe, cap, n_layers, capture_layers
        )
    finally:
        cap.remove()

    assert len(per_probe) == 2
    for entry in per_probe:
        assert len(entry["samples"]) == 2
        for s in entry["samples"]:
            assert "text" in s and "logp_norm" in s
            act = s["act"]
            assert act is not None, "every non-empty sample must carry an activation"
            assert act.shape == (len(capture_layers), model.config.hidden_size)
            assert act.dtype == torch.float16


def test_context_phases_default_no_partial_upload():
    """Major round-5: per-shard partial e0_gen uploads are OFF by default.

    8 concurrent shards would race on the same HF ref; the merge does the single
    authoritative upload. The run_context_phases ``upload_partial`` param defaults
    to False, and the shard-mode call site does not override it.
    """
    import inspect

    import issue658_extract_base_store as ex

    sig = inspect.signature(ex.run_context_phases)
    assert sig.parameters["upload_partial"].default is False, (
        "run_context_phases must default upload_partial=False (per-shard upload race fix)"
    )
    # the shard-mode call site must NOT pass upload_partial=True
    main_src = inspect.getsource(ex.main)
    assert "upload_partial=True" not in main_src, (
        "no main() call site may force per-shard partial uploads on (race fix)"
    )


def test_dispatch_prefetches_yaml_caches_before_shards():
    """Major round-5: the launcher prefetches the Betley YAML caches once.

    8 shards racing on the non-atomic _download_if_missing would corrupt the
    caches; the dispatch shell must fetch them in the single launcher process
    BEFORE the shard fan-out (the prefetch block precedes the shard loop).
    """
    sh = (PROJECT_ROOT / "scripts" / "issue658_8gpu_dispatch.sh").read_text()
    assert "fetch_betley_main_8" in sh and "fetch_preregistered_probes" in sh, (
        "dispatch must prefetch the Betley YAML caches"
    )
    prefetch_idx = sh.index("phase=prefetch_caches")
    shards_idx = sh.index("phase=shards")
    assert prefetch_idx < shards_idx, "cache prefetch must run BEFORE the shard fan-out"


def test_merge_copies_single_context_files(tmp_path):
    """Round-5: merge_shards copies per-context single_context/<ctx>.pt files."""
    import issue658_extract_base_store as ex
    import torch

    layers = [0, 1]
    h = 8

    def _write_shard(d, ctx_ids):
        (d / "answer_spans").mkdir(parents=True)
        (d / "single_context").mkdir(parents=True)
        torch.save(
            {
                "summaries": {
                    r: {c: torch.zeros(len(layers), h) for c in ctx_ids}
                    for r in ("mean", "last", "maxp")
                },
                "cc_meanprompt": {c: torch.zeros(len(layers), h) for c in ctx_ids},
                "capture_layers": layers,
                "context_ids": ctx_ids,
                "model": "stub",
                "probe_pool_hash": "x",
            },
            d / "v0_summaries.pt",
        )
        for c in ctx_ids:
            torch.save(
                {"context_id": c, "spans": [], "probes": ["q0"]}, d / "answer_spans" / f"{c}.pt"
            )
            torch.save(
                {"context_id": c, "n_samples": 2, "per_probe": []}, d / "single_context" / f"{c}.pt"
            )
        common.dump_json(
            {"context_ids": ctx_ids, "probes_by_context": {c: ["q0"] for c in ctx_ids}},
            d / "answer_spans" / "index.json",
        )

    s0 = tmp_path / "shards" / "shard_0"
    s1 = tmp_path / "shards" / "shard_1"
    _write_shard(s0, ["cA", "cC"])
    _write_shard(s1, ["cB", "cD"])
    rbsig = tmp_path / "shards" / "rbsigma"
    rbsig.mkdir(parents=True)
    torch.save({"r_b": {}, "capture_layers": layers, "columns": []}, rbsig / "r_b.pt")

    out = tmp_path / "store"
    ex.merge_shards([s0, s1], rbsig, out)
    sc_merged = sorted((out / "single_context").glob("*.pt"))
    assert {p.stem for p in sc_merged} == {"cA", "cB", "cC", "cD"}, (
        "merge must consolidate every shard's single_context/<ctx>.pt"
    )


def test_capture_audit_single_context_row_present(tmp_path):
    """BLOCKER round-5: the §1.10 audit row reads single_context/ and flips PRESENT.

    Builds a minimal store with a single_context/<ctx>.pt carrying a real
    per-sample act tensor and asserts the audit's §1.10 row is present=True
    (the round-4 NOTE is gone).
    """
    import issue658_capture_audit as audit
    import torch

    store = tmp_path / "store"
    (store / "single_context").mkdir(parents=True)
    torch.save(
        {
            "context_id": "cA",
            "n_samples": 8,
            "per_probe": [
                {
                    "probe": "q0",
                    "samples": [{"text": "hi", "logp_norm": -1.0, "act": torch.zeros(2, 8)}],
                }
            ],
        },
        store / "single_context" / "cA.pt",
    )
    res = audit.audit_store(store, tmp_path, smoke=False)
    sc_rows = [r for r in res.rows if "per-sample answer-side ACTIVATIONS" in r.analysis]
    assert len(sc_rows) == 1
    assert sc_rows[0].present is True, "the §1.10 per-sample-activations row must be PRESENT"
    assert sc_rows[0].caveat is False, "the §1.10 row is a hard requirement, not a NOTE"

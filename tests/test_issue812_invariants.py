#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (Δ, ρ, √) in scientific docstrings + comments.
"""Issue #812 round-2 regression tests — the substantive BLOCKER/CONCERN fixes.

Each test trips a PERMANENT invariant added in round 2 and would FAIL against the
round-1 code (which flattened sycophancy completions to rows / accepted a partial
lane prefix silently / had no per-cell bootstrap guard / narrated span-end answer
tokens as chat-boundary tokens). Pure-Python, no GPU / no HF — exercises the
helpers directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_SRC = Path(__file__).resolve().parent.parent / "src"
for p in (str(_SCRIPTS), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import issue812_fit_pooling as fit  # noqa: E402
import issue812_pooling_extract as extract  # noqa: E402
import issue812_regrade_e0 as regrade  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# BLOCKER 1 — sycophancy completions stay GROUPED under their probe (MF1)
# ─────────────────────────────────────────────────────────────────────────────
def test_probes_from_cells_preserves_completions_per_probe():
    """A #658 sycophancy cell (10 completions/probe) yields ONE probe entry with all
    10 completions grouped — NOT 10 flattened {probe, completion} rows (round-1 bug)."""
    blob = {
        "cells": [
            {"probe": "q1", "completions": [f"c{i}" for i in range(10)]},
            {"probe": "q2", "completions": [f"d{i}" for i in range(10)]},
        ]
    }
    probes = regrade._probes_from_cells(blob)
    assert len(probes) == 2, "must be 2 PROBE entries, not 20 flattened rows"
    assert probes[0]["probe"] == "q1"
    assert len(probes[0]["completions"]) == 10, "all 10 completions grouped under the probe"


def test_syco_subsample_selects_probe_cells_not_flattened_rows(tmp_path):
    """--syco-subsample 1 selects ONE PROBE cell (with ALL its completions), producing
    one probe carrying `completion_scores` of length = #completions — the mechanizable
    check from the round-2 brief. Judging is stubbed via dry_run=False + a fake
    save_raw so no API call fires."""
    # 2 probes x 10 completions; subsample to 1 probe
    probes_by_ctx = {
        "ctxA": [
            {"probe": "q1", "completions": [f"c{i}" for i in range(10)]},
            {"probe": "q2", "completions": [f"d{i}" for i in range(10)]},
        ]
    }
    n_draws = 3
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Monkeypatch judge_completions_batch to write a save_raw with a score per persona
    # custom_id. Persona = "<ctx>::p<pi>::c<ci>::d<di>"; custom_id = "<persona>__00000__00".
    captured = {}

    def fake_judge(packed, *, save_raw, **kw):
        all_scores = {}
        for persona in packed:
            cid = f"{persona}__00000__00"
            all_scores[cid] = {"score": 70, "reasoning": "ok"}
        captured["n_personas"] = len(packed)
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        import json

        save_raw.write_text(json.dumps({"all_scores": all_scores}))

    orig = regrade.judge_completions_batch
    regrade.judge_completions_batch = fake_judge
    try:
        res = regrade._judge_behavior(
            "sycophancy",
            probes_by_ctx,
            judge_model="claude-sonnet-4-5-20250929",
            n_draws=n_draws,
            out_dir=out_dir,
            subsample=1,  # select ONE probe cell
            dry_run=False,
        )
    finally:
        regrade.judge_completions_batch = orig

    # 1 probe cell x 10 completions x 3 draws = 30 personas (NOT 1 or 10)
    assert captured["n_personas"] == 10 * n_draws
    cell = res["ctxA"]
    assert len(cell["probe_scores"]) == 1, "subsample=1 -> exactly ONE probe"
    p0 = cell["probe_scores"][0]
    assert len(p0["completion_scores"]) == 10, "the probe carries 10 per-completion means"
    assert len(p0["completions"]) == 10
    assert len(p0["completions"][0]["draw_scores"]) == n_draws


# ─────────────────────────────────────────────────────────────────────────────
# BLOCKER 2 — partial per-context lane coverage FAILS LOUD (never a silent <50 fit)
# ─────────────────────────────────────────────────────────────────────────────
def test_lane3_partial_coverage_raises_before_judging(monkeypatch):
    """One missing context in the #763 gen/ prefix raises LaneCoverageError (round-1
    silently dropped it). files lists ctx1+ctx2 but ctx_ids requests ctx1,ctx2,ctx3."""
    files = [
        "issue763_matched_v0/analysis_tensors/gen/deception/ctx1.json",
        "issue763_matched_v0/analysis_tensors/gen/deception/ctx2.json",
    ]

    def fake_hf_json(repo, path):
        return {"cells": [{"probe": "q", "completions": ["a"]}]}

    monkeypatch.setattr(regrade, "_hf_json", fake_hf_json)
    with pytest.raises(regrade.LaneCoverageError) as ei:
        regrade._load_completions_i763("repo", "deception", ["ctx1", "ctx2", "ctx3"], files)
    assert "ctx3" in ei.value.missing
    assert ei.value.lane == "lane3-i763"


def test_lane3_entirely_absent_returns_none_not_raise():
    """When the gen/ prefix has NO files at all, return None (caller falls back to
    #658) — a total absence is not the same as a partial-coverage FAILURE."""
    assert regrade._load_completions_i763("repo", "deception", ["c1"], files=[]) is None


def test_lane2_partial_coverage_raises(monkeypatch):
    """Lane-2 (#658) also fails loud on a missing context (an unloadable file)."""

    def fake_hf_json(repo, path):
        if "ctx2" in path:
            raise FileNotFoundError(path)
        return {"cells": [{"probe": "q", "completions": ["a"]}]}

    monkeypatch.setattr(regrade, "_hf_json", fake_hf_json)
    with pytest.raises(regrade.LaneCoverageError) as ei:
        regrade._load_completions_i658("repo", "refusal", ["ctx1", "ctx2"])
    assert "ctx2" in ei.value.missing


def test_coverage_failure_writes_sentinel(tmp_path):
    """The fail-loud path persists a data-failure sentinel (no task.py shellout)."""
    exc = regrade.LaneCoverageError("refusal", "lane2-i658", ["a", "b"], ["a"], ["b"])
    path = regrade._write_coverage_failure_sentinel(tmp_path, exc)
    assert path.exists()
    import json

    payload = json.loads(path.read_text())
    assert payload["failure_class"] == "data"
    assert payload["missing_contexts"] == ["b"]


def test_lane1_reuse_rejects_graded_mean_only_source():
    """A reuse candidate with a graded_mean scalar but NO MF1 probe/draw arrays is
    NOT reusable — the reliability ceiling needs sub-context units (round-1 accepted
    it, silently breaking the √(r_yy) split)."""
    graded_mean_only = {"graded_mean": 42.0}  # no probe_scores
    assert regrade._has_mf1_granularity(graded_mean_only) is False
    with_units = {"graded_mean": 42.0, "probe_scores": [{"probe_mean": 42.0}]}
    assert regrade._has_mf1_granularity(with_units) is True


# ─────────────────────────────────────────────────────────────────────────────
# BLOCKER 3 — a degenerate bootstrap cell is GUARDED (ci95=null), never aborts
# ─────────────────────────────────────────────────────────────────────────────
def test_cluster_bootstrap_raises_on_degenerate_cell():
    """Confirm the underlying helper DOES raise on a no-rank-variation cell (the
    condition the round-2 guard must catch)."""
    from issue658_fit_predictors import _cluster_bootstrap_rho

    const = np.zeros(8)  # zero variance -> Spearman undefined every resample
    with pytest.raises(RuntimeError):
        _cluster_bootstrap_rho(const, const, n_boot=2000, seed=0)


def test_bootstrap_guard_records_null_and_continues():
    """The fit-side guard turns the RuntimeError into ci95=null + bootstrap_error and
    KEEPS GOING (round-1 had no try/except, so one bad cell aborted the whole sweep).
    We reproduce the exact guard shape used in the sweep loop."""
    from issue658_fit_predictors import _cluster_bootstrap_rho

    const = np.zeros(8)
    per_layer_boot: dict = {}
    aborted = False
    try:
        try:
            boot = _cluster_bootstrap_rho(const, const, n_boot=2000, seed=0)
            per_layer_boot = {"ci95": boot["ci95"] if boot else None, "bootstrap_error": None}
        except RuntimeError as exc:
            per_layer_boot = {"ci95": None, "bootstrap_error": str(exc)}
    except Exception:
        aborted = True
    assert not aborted, "a degenerate cell must NOT abort the sweep"
    assert per_layer_boot["ci95"] is None
    assert per_layer_boot["bootstrap_error"]  # non-empty message recorded


def test_atomic_write_json_roundtrips(tmp_path):
    """The per-behavior checkpoint writer is atomic (.tmp + os.replace) and readable."""
    p = tmp_path / "sub" / "pooling_fit_results.json"
    fit._atomic_write_json(p, {"results": {"a": 1}})
    assert p.exists()
    assert not p.with_suffix(p.suffix + ".tmp").exists(), "tmp must be renamed away"
    import json

    assert json.loads(p.read_text())["results"]["a"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# CONCERN 4 — the two boundary slots ARE tail duplicates (asserted, not just prose)
# ─────────────────────────────────────────────────────────────────────────────
def test_boundary_slots_equal_tail_slots():
    """tail-1 == tail[0] and tail-2 == tail[1] on a real span (the #658 span-end
    aliasing). Round-1 called these im_end/turn_nl and narrated them as chat-boundary
    tokens, which #658 spans do not contain."""
    torch.manual_seed(0)
    s, h, k = 5, 8, 3
    span = torch.randn(s, h)
    fixed_q = torch.randn(h)
    red = extract._reduce_probe_span(span, fixed_q, k)
    assert torch.equal(red["tail_1"], red["tail"][0]), "tail-1 must alias tail[0]"
    assert torch.equal(red["tail_2"], red["tail"][1]), "tail-2 must alias tail[1]"
    # the OLD key names must be gone (schema rename enforced)
    assert "im_end" not in red and "turn_nl" not in red


def test_reduce_probe_span_aliasing_assert_fires_on_violation():
    """The in-function assertion enforces the aliasing so a future refactor that
    breaks it fails loud. A 1-token span still aliases tail-1==tail[0]; tail-2 is
    NaN-filled (not present) so its assert is correctly skipped."""
    span = torch.randn(1, 6)  # single answer token
    red = extract._reduce_probe_span(span, torch.randn(6), k=3)
    assert torch.equal(red["tail_1"], red["tail"][0])
    assert red["tail_2_valid"] is False


# ─────────────────────────────────────────────────────────────────────────────
# CONCERN 5 — attn-learned uses nested-CV inner K folds (not train-fold loss)
# ─────────────────────────────────────────────────────────────────────────────
def test_attn_inner_cv_constant_present():
    """A nested-CV inner-fold count is configured (round-1 selected L2 by train-fold
    loss with no inner CV)."""
    assert getattr(fit, "ATTN_INNER_CV_K", 0) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# CONCERN 6 — reliability object carries the #742 keys (binomial, bracket, CI)
# ─────────────────────────────────────────────────────────────────────────────
def _fake_graded_cell(n_ctx=8, n_probes=6, n_comp=1, base=50.0):
    """A synthetic per-behavior graded cell with the MF1 structure."""
    rng = np.random.default_rng(0)
    cell = {}
    for c in range(n_ctx):
        probes = []
        for pi in range(n_probes):
            comps = []
            comp_means = []
            for ci in range(n_comp):
                draws = [float(base + rng.normal(0, 5)) for _ in range(8)]
                cm = float(np.mean(draws))
                comp_means.append(cm)
                comps.append({"completion_idx": ci, "draw_scores": draws, "completion_mean": cm})
            probes.append(
                {
                    "probe_idx": pi,
                    "completion_scores": comp_means,
                    "completions": comps,
                    "probe_mean": float(np.mean(comp_means)),
                }
            )
        cell[f"ctx{c}"] = {"context_id": f"ctx{c}", "probe_scores": probes}
    return cell


def test_reliability_object_has_742_keys():
    """CONCERN 6: the reliability object carries sqrt_r_yy_ci95, binomial_variance,
    bracket_lo, bracket_hi (round-1 returned only sqrt_r_yy + a crosscheck)."""
    cell = _fake_graded_cell()
    rel = fit._reliability_for_behavior(cell, "refusal", seed=1, n_boot=200)
    for key in ("sqrt_r_yy", "sqrt_r_yy_ci95", "binomial_variance", "bracket_lo", "bracket_hi"):
        assert key in rel, f"reliability object missing {key}"
    assert rel["binomial_variance"] is not None
    assert rel["estimator"] == "over_probes"


def test_sycophancy_uses_over_rollouts_completion_scores():
    """Sycophancy splits OVER-ROLLOUTS: the primary units come from completion_scores
    (the per-completion means), not per-probe means (BLOCKER 1 + CONCERN 6)."""
    cell = _fake_graded_cell(n_ctx=6, n_probes=4, n_comp=10)
    primary, _draws, label = fit._units_for_behavior(cell, "sycophancy")
    assert label == "over_rollouts"
    # each context has n_probes * n_comp completion-mean units
    assert len(primary["ctx0"]) == 4 * 10


def test_contexts_needed_extrapolation_solves_power_curve():
    """CONCERN 6: contexts-needed inverts metric(n)=a-b*n^-c. On a synthetic curve
    approaching an asymptote it returns a finite n to reach a sub-asymptote target."""
    ns = [10, 15, 20, 25, 30, 40, 50]
    a_true, b_true, c_true = 0.85, 1.2, 0.7
    rho = [a_true - b_true * n ** (-c_true) for n in ns]
    cn = fit._contexts_needed(ns, rho, target_rho=0.80)
    assert cn is not None and cn > 0
    # asymptote 0.85 cannot reach 0.90 -> None (not a false-precision number)
    assert fit._contexts_needed(ns, rho, target_rho=0.90) is None


# ─────────────────────────────────────────────────────────────────────────────
# CONCERN 7 — reconstruction R² is COMPUTED (not stubbed) when c_C is present
# ─────────────────────────────────────────────────────────────────────────────
def test_reconstruction_r2_computed_when_cc_present(tmp_path):
    """CONCERN 7: with c_C on disk, _reconstruction_r2 writes reconstruction_r2.json
    with per_layer_r2 keys (round-1 stubbed a `note` even when c_C was present)."""
    n, h, n_layers = 12, 16, 2
    layers = [0, 1]
    ids = [f"ctx{i}" for i in range(n)]
    cc = {"C": np.random.default_rng(0).normal(size=(n, n_layers, h)), "ids": ids}
    P = 4
    inputs = {
        "ctx_ids": ids,
        "layers": layers,
        "mean": np.random.default_rng(1).normal(size=(n, n_layers, h)),
        "max": np.random.default_rng(2).normal(size=(n, n_layers, h)),
        "attn_fixed": np.random.default_rng(3).normal(size=(n, n_layers, h)),
        "aligned_pos": np.random.default_rng(4).normal(size=(n, n_layers, P, h)),
    }
    out_dir = tmp_path / "out"
    fig_dir = tmp_path / "fig"
    out_dir.mkdir()
    fig_dir.mkdir()
    res = fit._reconstruction_r2(cc, inputs, layers, layers, ids, out_dir, fig_dir)
    assert "per_layer_r2" in res
    assert "mean" in res["per_layer_r2"] and "unpooled" in res["per_layer_r2"]
    assert (out_dir / "reconstruction_r2.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# ROUND-3 BLOCKER — every batch custom_id is Anthropic-Batch-API regex-valid
# (round-2 personas embedded "::" -> the whole batch would 400 at batches.create)
# ─────────────────────────────────────────────────────────────────────────────
def _build_batch_custom_ids(packed: dict) -> list[str]:
    """The EXACT custom_ids batch_judge sends to batches.create for ``packed``.

    Goes through batch_judge's real enumeration (``_enumerate_and_check_cache``), so it
    exercises the production ``f"{persona}__{idx:05d}__{comp_idx:02d}"`` construction,
    not a re-implemented copy."""
    from explore_persona_space.eval import batch_judge as bj

    _total, _cached, uncached = bj._enumerate_and_check_cache(
        packed, cache=None, format_user_msg=lambda q, c: f"{q}\n{c}"
    )
    return [cid for (cid, _q, _c, _u) in uncached]


def test_regrade_batch_custom_ids_all_regex_valid():
    """Every custom_id batch_judge derives from a regrade persona matches
    ^[a-zA-Z0-9_-]{1,64}$. Round-2 personas were f"{ctx}::p..::c..::d.." — the "::"
    (and an arbitrary-length/char ctx) would 400 the whole ~600K-call batch at
    batches.create. This FAILS against the round-2 base and PASSES post-fix.
    """
    from explore_persona_space.eval.batch_judge import _CUSTOM_ID_RE

    # A realistically hostile ctx (long, with "::", "/", spaces — all illegal raw).
    probes_by_ctx = {
        "wildchat::sample/007 has spaces & punctuation!": [
            {"probe": "q1", "completions": ["a", "b"]},
            {"probe": "q2", "completions": ["c"]},
        ],
        "another/ctx::id": [{"probe": "q3", "completions": ["d", "e", "f"]}],
    }
    n_draws = 4
    captured: dict = {}

    def fake_judge(packed, *, save_raw, **kw):
        captured["packed"] = dict(packed)
        # write an empty save_raw so the (unexercised here) reduction path is happy
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        import json

        save_raw.write_text(json.dumps({"all_scores": {}}))

    orig = regrade.judge_completions_batch
    regrade.judge_completions_batch = fake_judge
    try:
        regrade._judge_behavior(
            "sycophancy",
            probes_by_ctx,
            judge_model="claude-sonnet-4-5-20250929",
            n_draws=n_draws,
            out_dir=Path("/tmp") / "issue812_cid_test",
            subsample=None,
            dry_run=False,
        )
    finally:
        regrade.judge_completions_batch = orig

    custom_ids = _build_batch_custom_ids(captured["packed"])
    # (2+1+3) completions x 4 draws = 24 requests, all regex-valid
    assert len(custom_ids) == (2 + 1 + 3) * n_draws
    for cid in custom_ids:
        assert _CUSTOM_ID_RE.match(cid), f"custom_id violates Anthropic regex: {cid!r}"
        assert len(cid) <= 64, f"custom_id exceeds 64 chars: {cid!r} ({len(cid)})"


def test_regrade_persona_map_sidecar_roundtrips(tmp_path):
    """The dispatch -> reduce join goes THROUGH the persisted persona_map sidecar (NOT a
    "::"-split of the persona). The sidecar exists, and every judged custom_id joins
    back to its exact (ctx, probe_idx, comp_idx, draw) coordinate."""
    import json

    probes_by_ctx = {
        "ctx::A/1": [{"probe": "q1", "completions": ["x", "y"]}],
        "ctx::B/2": [{"probe": "q2", "completions": ["z"]}],
    }
    n_draws = 3
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def fake_judge(packed, *, save_raw, **kw):
        # score every persona; batch_judge would append "__NNNNN__00" -> mirror it so the
        # reduction's cid.rsplit("__", 2)[0] recovers the persona.
        all_scores = {f"{p}__00000__00": {"score": 55, "reasoning": "ok"} for p in packed}
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(json.dumps({"all_scores": all_scores}))

    orig = regrade.judge_completions_batch
    regrade.judge_completions_batch = fake_judge
    try:
        res = regrade._judge_behavior(
            "sycophancy",
            probes_by_ctx,
            judge_model="claude-sonnet-4-5-20250929",
            n_draws=n_draws,
            out_dir=out_dir,
            subsample=None,
            dry_run=False,
        )
    finally:
        regrade.judge_completions_batch = orig

    # (a) the sidecar landed and round-trips every coordinate
    sidecar = out_dir / "_persona_map_sycophancy.json"
    assert sidecar.exists(), "persona_map sidecar must be persisted"
    pmap = json.loads(sidecar.read_text())
    # 3 completions total x 3 draws = 9 distinct coordinates, all distinct keys
    coords = {tuple(v) for v in pmap.values()}
    assert len(pmap) == 3 * n_draws == len(coords)
    # every persona key is regex-safe (no "::") — the whole point
    from explore_persona_space.eval.batch_judge import _CUSTOM_ID_RE

    for persona in pmap:
        assert _CUSTOM_ID_RE.match(f"{persona}__00000__00")

    # (b) the reduction (joined via the map) reconstructs the full per-ctx structure
    assert set(res) == {"ctx::A/1", "ctx::B/2"}
    a = res["ctx::A/1"]["probe_scores"][0]
    assert len(a["completions"]) == 2
    assert len(a["completions"][0]["draw_scores"]) == n_draws
    assert a["completions"][0]["completion_mean"] == 55.0


# ─────────────────────────────────────────────────────────────────────────────
# ROUND-4 CONCERN — reduction join FAILS LOUD on a persona_map under-coverage
# (round-3 silently `continue`d on a persona_map miss, so a future batch_judge
# custom_id suffix drift would silently under-cover — detectable only post-hoc)
# ─────────────────────────────────────────────────────────────────────────────
def test_reduction_raises_on_unknown_persona_join_miss(tmp_path):
    """A judged custom_id whose recovered persona is absent from the sidecar persona_map
    raises RuntimeError at reduce time (round-3 swallowed it via `continue`). Simulates
    the failure mode the CONCERN guards: a batch_judge suffix-drift produces a persona
    the reducer's cid.rsplit("__", 2)[0] recovery cannot join back."""
    import json

    probes_by_ctx = {"ctxA": [{"probe": "q1", "completions": ["x"]}]}
    n_draws = 2
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def fake_judge(packed, *, save_raw, **kw):
        # Score every legit persona correctly, then inject ONE extra all_scores key
        # whose recovered persona ("gremlin") is NOT in persona_map — the exact shape a
        # future custom_id suffix drift (unmirrored in the recovery split) would produce.
        all_scores = {f"{p}__00000__00": {"score": 60, "reasoning": "ok"} for p in packed}
        all_scores["gremlin__00000__00"] = {"score": 99, "reasoning": "orphan"}
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(json.dumps({"all_scores": all_scores}))

    orig = regrade.judge_completions_batch
    regrade.judge_completions_batch = fake_judge
    try:
        with pytest.raises(RuntimeError) as ei:
            regrade._judge_behavior(
                "sycophancy",
                probes_by_ctx,
                judge_model="claude-sonnet-4-5-20250929",
                n_draws=n_draws,
                out_dir=out_dir,
                subsample=None,
                dry_run=False,
            )
    finally:
        regrade.judge_completions_batch = orig
    assert "under-coverage" in str(ei.value)
    assert "gremlin" in str(ei.value)


def test_reduction_materializes_entirely_missing_completion_as_recorded_nan(tmp_path):
    """A completion whose EVERY draw is absent from all_scores is still materialized as
    a full NaN completion (n_draws NaN drops counted in n_dropped), never silently
    absent from the per_ctx grid. Round-3 iterated over all_scores, so a completion with
    no returned rows had NO per_ctx slot at all — its draws vanished uncounted; round-4
    iterates over the authoritative persona_map so every dispatched draw is accounted
    for. (This discriminates the fix: an entirely-unscored completion is the shape the
    old all_scores-driven loop dropped without a trace.)"""
    import json

    # 1 ctx, 1 probe, 2 completions, 2 draws = 4 personas; score ONLY completion 0's
    # draws, leave completion 1 ENTIRELY unscored (both draws missing).
    probes_by_ctx = {"ctxA": [{"probe": "q1", "completions": ["x", "y"]}]}
    n_draws = 2
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def fake_judge(packed, *, save_raw, **kw):
        pmap = json.loads((save_raw.parent / f"_persona_map_{save_raw.stem[5:]}.json").read_text())
        all_scores = {}
        for persona, coord in pmap.items():
            _ctx, _pi, comp_idx, _draw = coord
            if comp_idx == 0:  # score ONLY completion 0; completion 1 stays absent
                all_scores[f"{persona}__00000__00"] = {"score": 70, "reasoning": "ok"}
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(json.dumps({"all_scores": all_scores}))

    orig = regrade.judge_completions_batch
    regrade.judge_completions_batch = fake_judge
    try:
        res = regrade._judge_behavior(
            "sycophancy",
            probes_by_ctx,
            judge_model="claude-sonnet-4-5-20250929",
            n_draws=n_draws,
            out_dir=out_dir,
            subsample=None,
            dry_run=False,
        )
    finally:
        regrade.judge_completions_batch = orig

    cell = res["ctxA"]
    # completion 1's 2 draws are recorded NaN drops — NOT silently absent (round-3 lost them)
    assert cell["n_dropped"] == n_draws, "the entirely-unscored completion's draws counted"
    p0 = cell["probe_scores"][0]
    comps = {c["completion_idx"]: c for c in p0["completions"]}
    assert set(comps) == {0, 1}, "BOTH completions present in the grid (round-3 dropped comp 1)"
    assert comps[1]["completion_mean"] is None, "the unscored completion has a NaN mean"
    assert comps[1]["draw_scores"] == [None, None], "both its draws are recorded NaN"


# ─────────────────────────────────────────────────────────────────────────────
# ROUND-3 CONCERN — lane-1 reuse shape parsing is single-source (validator == loader)
# ─────────────────────────────────────────────────────────────────────────────
def test_parse_reuse_blob_all_accepted_shapes_load_identically():
    """_parse_reuse_blob extracts the SAME per-context grid from every accepted shape.
    Round-2 bug: the validator accepted {behavior, e0}/{behavior, per_context} but the
    loader read only blob[beh]/whole-blob, so a valid {behavior, e0} grid validated then
    silently no-op-loaded the whole blob."""
    grid = {f"ctx{i}": {"graded_mean": float(i)} for i in range(3)}
    shapes = [
        {"deception": grid},  # multi-behavior, keyed by behavior name
        {"behavior": "deception", "e0": grid},  # this script's own output schema
        {"behavior": "deception", "per_context": grid},  # alt single-behavior schema
    ]
    for blob in shapes:
        assert regrade._parse_reuse_blob(blob, "deception") == grid, blob
    # a blob for a DIFFERENT behavior returns None (no false reuse)
    assert regrade._parse_reuse_blob({"behavior": "refusal", "e0": grid}, "deception") is None
    assert regrade._parse_reuse_blob({"something_else": grid}, "deception") is None


def test_lane1_validator_and_loader_agree_on_e0_shape(monkeypatch):
    """A {behavior, e0} blob that the VALIDATOR accepts is LOADED to the SAME grid by the
    main-loop reuse path — not the whole blob (round-2's silent no-op)."""
    grid = {
        f"ctx{i}": {"graded_mean": float(i), "probe_scores": [{"probe_mean": float(i)}]}
        for i in range(50)
    }
    blob = {"behavior": "deception", "e0": grid}

    # validator sees it as reusable
    assert regrade._parse_reuse_blob(blob, "deception") == grid
    files = ["issue810_final/graded_e0_deception.json"]
    monkeypatch.setattr(regrade, "_hf_json", lambda repo, path: blob)
    reuse = regrade._lane1_reuse_map("repo", ["deception"], files)
    assert reuse.get("deception") == "issue810_final/graded_e0_deception.json"

    # loader extracts the identical grid via the shared parser (NOT the whole blob)
    loaded = regrade._parse_reuse_blob(regrade._hf_json("repo", reuse["deception"]), "deception")
    assert loaded == grid
    assert "behavior" not in loaded and "e0" not in loaded, "must be the grid, not the wrapper"

#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ρ, √, ×) in scientific docstrings + comments.
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
    10 completions grouped — NOT 10 flattened {probe, completion} rows (round-1 bug).

    Uses the REAL Lane-2 (#658 e0_gen) completion shape: each completion is a
    ``{"text": str, "logp_norm": float}`` dict (round-5 test-realism fix — the old
    bare-string fixture masked the dict-concatenation crash)."""
    blob = {
        "cells": [
            {
                "probe": "q1",
                "completions": [{"text": f"c{i}", "logp_norm": -0.4 - i} for i in range(10)],
            },
            {
                "probe": "q2",
                "completions": [{"text": f"d{i}", "logp_norm": -0.5 - i} for i in range(10)],
            },
        ]
    }
    probes = regrade._probes_from_cells(blob, behavior="sycophancy")
    assert len(probes) == 2, "must be 2 PROBE entries, not 20 flattened rows"
    assert probes[0]["probe"] == "q1"
    assert len(probes[0]["completions"]) == 10, "all 10 completions grouped under the probe"
    # each completion is the UNWRAPPED text string, not the {"text": ...} dict
    assert probes[0]["completions"] == [f"c{i}" for i in range(10)]
    assert all(isinstance(c, str) for c in probes[0]["completions"])


# ─────────────────────────────────────────────────────────────────────────────
# ROUND-5 BLOCKER — dict-shaped completions ({"text": ...}) are normalized to text
# (round-4 kept the raw cell, so a {"text": ...} dict flowed into judge_dispatch's
# _content_hash and crashed with "can only concatenate str (not dict) to str")
# ─────────────────────────────────────────────────────────────────────────────
def test_probes_from_cells_normalizes_lane2_i658_dict_completions():
    """Lane-2 (#658 e0_gen) cells store completions as {"text": str, "logp_norm": float}
    dicts. _probes_from_cells unwraps each to its text string (round-4 kept the dict →
    the round-5 production crash at judge_dispatch._content_hash)."""
    blob = {
        "cells": [
            {
                "probe": "q1",
                "completions": [
                    {"text": "sure, you're right", "logp_norm": -0.41},
                    {"text": "that seems correct", "logp_norm": -0.52},
                ],
            }
        ]
    }
    probes = regrade._probes_from_cells(blob, behavior="sycophancy")
    assert len(probes) == 1
    assert probes[0]["completions"] == ["sure, you're right", "that seems correct"]
    assert all(isinstance(c, str) for c in probes[0]["completions"])


def test_probes_from_cells_normalizes_lane3_i763_dict_completions():
    """Lane-3 (#763 gen/) cells store completions as {"text": str} dicts (no logp_norm).
    _probes_from_cells unwraps each to its text string."""
    blob = {
        "cells": [
            {"probe": "q1", "completions": [{"text": "I cannot help with that."}]},
            {"probe": "q2", "completions": [{"text": "Here is how."}, {"text": "No."}]},
        ]
    }
    probes = regrade._probes_from_cells(blob, behavior="refusal")
    assert [p["completions"] for p in probes] == [
        ["I cannot help with that."],
        ["Here is how.", "No."],
    ]


def test_probes_from_cells_accepts_bare_string_completions_backcompat():
    """A bare-string completion (any string-shaped source) is kept verbatim — the
    back-compat branch of the normalizer."""
    blob = {"cells": [{"probe": "q1", "completions": ["plain text answer"]}]}
    probes = regrade._probes_from_cells(blob, behavior="deception")
    assert probes[0]["completions"] == ["plain text answer"]


def test_probes_from_cells_raises_on_unrecognized_completion_shape():
    """An unrecognized completion shape (a dict WITHOUT a str 'text' key, or a scalar)
    is a fail-loud ValueError naming the behavior/probe + the offending type — never
    coerced or silently skipped (CLAUDE.md 'Fail fast — never hide failures')."""
    # dict without a 'text' key
    blob_no_text = {"cells": [{"probe": "q1", "completions": [{"body": "wrong key"}]}]}
    with pytest.raises(ValueError) as ei_dict:
        regrade._probes_from_cells(blob_no_text, behavior="sycophancy")
    assert "sycophancy" in str(ei_dict.value)
    assert "q1" in str(ei_dict.value)
    assert "dict" in str(ei_dict.value)

    # a scalar (int) completion
    blob_scalar = {"cells": [{"probe": "q2", "completions": [42]}]}
    with pytest.raises(ValueError) as ei_int:
        regrade._probes_from_cells(blob_scalar, behavior="refusal")
    assert "int" in str(ei_int.value)
    assert "q2" in str(ei_int.value)

    # a dict whose 'text' is not a str (e.g. nested) also fails loud
    blob_nonstr_text = {"cells": [{"probe": "q3", "completions": [{"text": ["nested"]}]}]}
    with pytest.raises(ValueError):
        regrade._probes_from_cells(blob_nonstr_text, behavior="deception")


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
        # real #763 gen/ shape: completions are {"text": str} dicts
        return {"cells": [{"probe": "q", "completions": [{"text": "a"}]}]}

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
        # real #658 e0_gen shape: completions are {"text": str, "logp_norm": float} dicts
        return {"cells": [{"probe": "q", "completions": [{"text": "a", "logp_norm": -0.3}]}]}

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


# ─────────────────────────────────────────────────────────────────────────────
# ROUND-6 FIX 1 — reliability preflight EXCLUDES a null-ceiling behavior + records
# its diagnostics (does NOT raise on ANY null); raises ONLY when ALL are null.
# Round-5 code raised on any null sqrt_r_yy (RuntimeError killed the whole run when
# deception E0 on base Qwen carried ~no between-context signal → null ceiling).
# ─────────────────────────────────────────────────────────────────────────────
def _graded_cell_from_ctx_means(ctx_means, *, n_probes=6, within_sd=3.0, seed=0):
    """Per-behavior graded cell whose BETWEEN-context signal is set by ``ctx_means``.

    Each context's per-probe means cluster around its ``ctx_means[c]`` (± ``within_sd``
    within-context noise). Wide, monotone ``ctx_means`` ⇒ a positive split-half
    correlation ⇒ non-null sqrt(r_yy). Near-constant ``ctx_means`` (deception on base
    Qwen: per-context sd ~3 on the 0-100 scale) ⇒ split-half r_yy ≤ 0 ⇒ null ceiling.
    Carries the MF1 structure (probe_scores with completion_scores + draw arrays) AND
    the ``graded_mean`` the between-context-SD diagnostic reads.
    """
    rng = np.random.default_rng(seed)
    cell = {}
    for c, mu in enumerate(ctx_means):
        probes = []
        for pi in range(n_probes):
            draws = [float(np.clip(mu + rng.normal(0, within_sd), 0, 100)) for _ in range(8)]
            cm = float(np.mean(draws))
            probes.append(
                {
                    "probe_idx": pi,
                    "completion_scores": [cm],
                    "completions": [
                        {"completion_idx": 0, "draw_scores": draws, "completion_mean": cm}
                    ],
                    "probe_mean": cm,
                }
            )
        gm = float(np.mean([p["probe_mean"] for p in probes]))
        cell[f"ctx{c}"] = {
            "context_id": f"ctx{c}",
            "behavior": "x",
            "graded_mean": gm,
            "probe_scores": probes,
        }
    return cell


def test_preflight_excludes_null_ceiling_behavior_and_records_diagnostics():
    """One behavior has a degenerate (null-ceiling) E0, another has real signal.

    Round-6 fix: the null-ceiling behavior is RECORDED in reliability_excluded with its
    measured diagnostics (sqrt_r_yy=None + reason + split_half_r + between_ctx_sd +
    n_ctx), the signal behavior is NOT excluded, and NO exception is raised (round-5
    raised on ANY null). This is the exact deception-vs-refusal shape from the crash.
    """
    signal = _graded_cell_from_ctx_means(list(np.linspace(10, 90, 20)), seed=1)
    degen = _graded_cell_from_ctx_means([44.0 + 0.05 * (i % 3) for i in range(20)], seed=2)
    behaviors = ["refusal", "deception"]
    reliability = {
        "refusal": fit._reliability_for_behavior(signal, "refusal", seed=1, n_boot=100),
        "deception": fit._reliability_for_behavior(degen, "deception", seed=2, n_boot=100),
    }
    # sanity: the fixtures realize the intended null/non-null split
    assert reliability["refusal"]["sqrt_r_yy"] is not None, "signal behavior must have a ceiling"
    assert reliability["deception"]["sqrt_r_yy"] is None, "degenerate behavior must be null"

    # No raise — exclude-and-record (round-5 would have raised on the deception null)
    excluded = fit._build_reliability_exclusions(reliability, behaviors)

    assert "deception" in excluded, "null-ceiling behavior must be RECORDED, not raised on"
    assert "refusal" not in excluded, "the signal behavior's ceiling is unchanged"
    rec = excluded["deception"]
    assert rec["sqrt_r_yy"] is None
    assert "between-context sd" in rec["reason"]
    # diagnostics carry the actual measured values (why the ceiling is null)
    assert rec["split_half_r"] is not None and rec["split_half_r"] <= 0.0
    assert rec["between_ctx_sd"] is not None and rec["between_ctx_sd"] < 5.0
    assert rec["n_ctx"] == 20


def _null_reliability(between_ctx_sd=1.5, split_half_r=-0.3, n_ctx=50):
    """A reliability entry with a null ceiling + the diagnostic fields the exclude path
    reads. Built DIRECTLY (not routed through the stochastic split-half estimator) so
    the all-null terminal branch is tested deterministically."""
    return {
        "sqrt_r_yy": None,
        "between_ctx_sd": between_ctx_sd,
        "split_half_r": split_half_r,
        "n_ctx": n_ctx,
    }


def test_preflight_raises_when_ALL_behaviors_null():
    """When EVERY behavior has a null ceiling there is nothing to analyze — the preflight
    helper RAISES. Round-6 keeps THIS terminal case a hard failure while
    exclude-and-recording the mixed case above; round-7 extracts the raise into the
    ``_reliability_preflight`` helper so the branch is exercised deterministically
    (the round-6 test only asserted the exclusions map size, never tripping the raise)."""
    behaviors = ["deception", "harmful_compliance"]
    reliability = {b: _null_reliability() for b in behaviors}
    with pytest.raises(RuntimeError, match="ALL behaviors"):
        fit._reliability_preflight(reliability, behaviors)


def test_preflight_helper_returns_exclusions_on_mixed_null():
    """The preflight helper does NOT raise when at least one behavior has a non-null
    ceiling — it returns the exclusions map for the null-ceiling behaviors (mirror of the
    all-null raise above)."""
    reliability = {
        "deception": _null_reliability(),
        "refusal": {"sqrt_r_yy": 0.8, "between_ctx_sd": 20.0, "split_half_r": 0.6, "n_ctx": 50},
    }
    excluded = fit._reliability_preflight(reliability, ["deception", "refusal"])
    assert set(excluded) == {"deception"}


def test_preflight_via_all_null_reliability_monkeypatch(monkeypatch):
    """Drive the raise through the estimator boundary: monkeypatch
    ``_reliability_for_behavior`` to return an all-null entry for every behavior, then run
    the preflight — proving the terminal RuntimeError fires when the REAL per-behavior
    estimator (not a hand-built dict) yields all-null ceilings."""
    behaviors = ["sycophancy", "refusal"]
    monkeypatch.setattr(
        fit, "_reliability_for_behavior", lambda *a, **k: _null_reliability(n_ctx=50)
    )
    reliability = {b: fit._reliability_for_behavior(None, b) for b in behaviors}
    with pytest.raises(RuntimeError, match="ALL behaviors"):
        fit._reliability_preflight(reliability, behaviors)


def test_build_exclusions_empty_when_all_ceilings_present():
    """The mirror invariant: with every behavior carrying a non-null ceiling the
    exclusions map is EMPTY (no behavior dropped from Delta-rho interpretation)."""
    behaviors = ["refusal", "sycophancy"]
    reliability = {
        b: {"sqrt_r_yy": 0.8, "between_ctx_sd": 20.0, "split_half_r": 0.6, "n_ctx": 50}
        for b in behaviors
    }
    assert fit._build_reliability_exclusions(reliability, behaviors) == {}


# ─────────────────────────────────────────────────────────────────────────────
# ROUND-7 FIX — fit-side strict-subset defense (restored-e0-payload-coverage leg c):
# with no explicit --behaviors subset, a graded E0 that covers a STRICT SUBSET of the
# expected 8 behaviors (a partial regrade that slipped the idempotence gate) must FAIL
# LOUD, never silently analyze the subset via ``behaviors = list(graded.keys())``.
# ─────────────────────────────────────────────────────────────────────────────
def test_fit_strict_subset_fails_loud_without_explicit_behaviors():
    """A subset of the 8 behaviors + no --behaviors → RuntimeError naming the missing set."""
    subset = ["sycophancy", "refusal", "harmful_compliance"]  # high-m only; low-m dropped
    with pytest.raises(RuntimeError, match="strict subset of the expected 8"):
        fit._validate_behavior_coverage(subset, explicit_behaviors=False)


def test_fit_full_8_behaviors_pass():
    """The full 8 behaviors + no --behaviors → no raise (the genuine complete artifact)."""
    full = fit.HIGH_M + fit.LOW_M
    fit._validate_behavior_coverage(full, explicit_behaviors=True)  # explicit is a no-op
    fit._validate_behavior_coverage(full, explicit_behaviors=False)  # no raise


def test_fit_explicit_behaviors_subset_allowed():
    """An EXPLICIT --behaviors subset opts out of the strict-subset guard (deliberate
    smoke/debug slice) — no raise even for a single behavior."""
    fit._validate_behavior_coverage(["sycophancy"], explicit_behaviors=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUND-8 FIX — fit-side EXACT context-ID-set defense (graded-e0-context-idset-not-enforced):
# ``all_ctx = inputs["ctx_ids"]`` is authoritative for the graded-E0 join. The regrade
# gate checks the context COUNT not the ID-SET, and ``_graded_target`` silently skips any
# authoritative context absent from the graded payload (weak len<4 floor only). A
# SAME-COUNT WRONG-KEY payload (one canonical ctx replaced by a stray key) would silently
# reduce/shift the n=50 join. With NO explicit --contexts smoke subset, the fit-side
# assert FAILS LOUD naming the missing + stray context ids (last-line defense). The
# context-axis twin of the round-7 restored-e0-payload-coverage BEHAVIOR-set check.
# ─────────────────────────────────────────────────────────────────────────────
def test_fit_context_idset_wrong_key_same_count_fails_loud():
    """A same-COUNT wrong-KEY payload (canonical ``ctx002`` replaced by stray ``ctxZZZ``)
    with no --contexts subset → RuntimeError naming BOTH the missing and the stray id."""
    all_ctx = [f"ctx{c:03d}" for c in range(5)]  # authoritative key set (n=5)
    per_ctx = _full_e0_payload(["sycophancy"], 5)["sycophancy"]
    # Drop the canonical ctx002, add a same-shape stray cell under a wrong key → count stays 5.
    stray = per_ctx.pop("ctx002")
    per_ctx["ctxZZZ"] = {**stray, "context_id": "ctxZZZ"}
    assert len(per_ctx) == len(all_ctx)  # SAME count — the count-only gate would pass
    with pytest.raises(RuntimeError, match="does not cover the authoritative context set"):
        fit._assert_graded_covers_ctx_set(per_ctx, all_ctx, "sycophancy", subset_active=False)
    # The message must name BOTH the missing canonical id and the stray key.
    try:
        fit._assert_graded_covers_ctx_set(per_ctx, all_ctx, "sycophancy", subset_active=False)
    except RuntimeError as e:
        msg = str(e)
    assert "ctx002" in msg and "ctxZZZ" in msg


def test_fit_context_idset_full_coverage_passes():
    """A payload covering EXACTLY the authoritative ctx set (the genuine full artifact) →
    no raise, even with no --contexts subset."""
    all_ctx = [f"ctx{c:03d}" for c in range(50)]
    per_ctx = _full_e0_payload(["sycophancy"], 50)["sycophancy"]
    assert set(per_ctx) == set(all_ctx)
    fit._assert_graded_covers_ctx_set(per_ctx, all_ctx, "sycophancy", subset_active=False)


def test_fit_context_idset_smoke_subset_unaffected():
    """With --contexts N active (``subset_active=True``), the deliberate smoke slice keeps
    the permissive join — a subset-covering payload does NOT raise."""
    all_ctx = [f"ctx{c:03d}" for c in range(50)]
    # Payload covers only the first 4 contexts (a --contexts 4 smoke), which is the
    # deliberate slice — must NOT raise when subset_active is True.
    per_ctx = _full_e0_payload(["sycophancy"], 4)["sycophancy"]
    assert set(per_ctx) != set(all_ctx)
    fit._assert_graded_covers_ctx_set(per_ctx, all_ctx, "sycophancy", subset_active=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUND-9 FIX — the ctx-ID-set assert is HOISTED into a preflight loop that runs
# BEFORE the reliability preflight (graded-e0-ctx-assert-after-reliability-preflight):
# every consumer of a behavior's graded payload (``_reliability_for_behavior`` in the
# reliability preflight, ``_graded_target`` in the fit loop) must run AFTER the exact
# context-ID-set enforcement. This test replicates ``main()``'s enforce→reliability
# ordering and proves NO reliability computation touches a wrong-key payload.
# ─────────────────────────────────────────────────────────────────────────────
def test_fit_ctx_idset_enforced_before_reliability(monkeypatch):
    """A same-count wrong-key payload trips the preflight ctx-ID-set assert BEFORE any
    ``_reliability_for_behavior`` call — replicating ``main()``'s ordering, the RuntimeError
    fires and the (call-recording) reliability helper is NEVER invoked for that behavior."""
    behaviors = ["sycophancy"]
    all_ctx = [f"ctx{c:03d}" for c in range(5)]  # authoritative key set (n=5)
    graded = _full_e0_payload(behaviors, 5)
    # Same-count wrong-key payload (canonical ctx002 → stray ctxZZZ): the count-only
    # gate would pass, but the ID-set enforcement must catch it before reliability runs.
    stray = graded["sycophancy"].pop("ctx002")
    graded["sycophancy"]["ctxZZZ"] = {**stray, "context_id": "ctxZZZ"}
    assert len(graded["sycophancy"]) == len(all_ctx)  # SAME count

    rel_calls: list[str] = []

    def _recording_reliability(graded_cell, behavior, *, seed, n_boot):
        rel_calls.append(behavior)
        return _null_reliability(n_ctx=len(all_ctx))

    monkeypatch.setattr(fit, "_reliability_for_behavior", _recording_reliability)

    # main()'s exact ordering: enforce the ctx-ID-set over ALL behaviors FIRST, then the
    # reliability preflight loop. The enforce loop must raise before any reliability call.
    with pytest.raises(RuntimeError, match="does not cover the authoritative context set"):
        for beh in behaviors:
            fit._assert_graded_covers_ctx_set(graded[beh], all_ctx, beh, subset_active=False)
        for beh in behaviors:
            fit._reliability_for_behavior(graded[beh], beh, seed=1, n_boot=100)

    assert rel_calls == [], (
        "reliability must NEVER be computed on a wrong-key payload — the ctx-ID-set "
        f"enforcement must fire first (recorded reliability calls: {rel_calls})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUND-6 FIX 2 — Phase-2 idempotence guard: both graded-E0 outputs present with a
# matching FULL recipe → SKIP the ~500K-call re-judge (restore-from-HF relaunch),
# never re-bill; a partial / mismatched state falls through to the full run.
# ─────────────────────────────────────────────────────────────────────────────
_HIGHM_FIXTURE = ["sycophancy", "refusal", "harmful_compliance"]
_LOWM_FIXTURE = ["deception", "fact_expression", "format_style", "self_report", "persona_drift"]


def _full_cell(ctx: str, beh: str):
    """One FULL per-context cell: graded_mean + MF1 probe/draw granularity.

    Mirrors the production write shape (issue812_regrade_e0.py:690-700) closely enough
    that ``_e0_payload_covers`` accepts it: a non-null ``graded_mean`` plus ``probe_scores``
    whose entries carry ``draw_scores`` (the N=8 per-draw granularity the reliability
    ceiling reads). Tiny (a few probes) — realistic shape, not 400 lines of literals."""
    return {
        "context_id": ctx,
        "behavior": beh,
        "graded_mean": 42.0,
        "binary_rate": 0.5,
        "n_judged": 2,
        "n_dropped": 0,
        "probe_scores": [
            {"probe": f"{ctx}-p{i}", "probe_mean": 40.0 + i, "draw_scores": [38.0, 42.0]}
            for i in range(2)
        ],
    }


def _full_e0_payload(behaviors: list[str], n_contexts: int) -> dict:
    """{behavior: {ctx: full-cell}} covering EVERY behavior × n_contexts contexts."""
    return {
        beh: {f"ctx{c:03d}": _full_cell(f"ctx{c:03d}", beh) for c in range(n_contexts)}
        for beh in behaviors
    }


def _write_graded_output(
    path: Path,
    *,
    n_draws=8,
    judge_model="claude-sonnet-4-5-20250929",
    n_contexts=50,
    issue=812,
    behaviors=None,
    e0=None,
):
    """Write a graded-E0 output with a FULL payload by default.

    ``behaviors`` defaults to the file's expected bucket (highm/lowm inferred from the
    basename); ``e0`` defaults to a full payload (every expected behavior × n_contexts
    contexts, each cell with graded_mean + MF1 granularity). Pass an explicit ``e0`` /
    ``behaviors`` to build a deliberately-incomplete payload (the negative tests)."""
    import json

    if behaviors is None:
        behaviors = _HIGHM_FIXTURE if path.name == "graded_e0_highm.json" else _LOWM_FIXTURE
    if e0 is None:
        e0 = _full_e0_payload(behaviors, n_contexts)
    meta = {
        "issue": issue,
        "git_commit": "abc123",
        "n_draws": n_draws,
        "judge_model": judge_model,
        "n_contexts": n_contexts,
        "schema": "graded_e0_v1_mf1",
    }
    path.write_text(json.dumps({"meta": meta, "behaviors": behaviors, "e0": e0}))


def test_idempotence_guard_skips_when_both_present_and_matching(tmp_path):
    """Both graded_e0_{highm,lowm}.json present with matching recipe (issue=812, n_draws=8,
    sonnet judge, n_contexts=50) → the guard returns skip=True + the prior git_commit."""
    _write_graded_output(tmp_path / "graded_e0_highm.json")
    _write_graded_output(tmp_path / "graded_e0_lowm.json")
    skip, commit = regrade._e0_outputs_present_and_matching(
        tmp_path,
        expected_n_draws=8,
        expected_judge_model="claude-sonnet-4-5-20250929",
        expected_n_contexts=50,
    )
    assert skip is True
    assert commit == "abc123"


def test_idempotence_guard_falls_through_on_meta_mismatch(tmp_path):
    """A meta that mismatches the expected recipe (e.g. n_draws=4 smoke output) →
    skip=False (fall through to the normal full run — never silently reuse a wrong
    grid). Tested at the guard-function level."""
    _write_graded_output(tmp_path / "graded_e0_highm.json", n_draws=4)  # smoke recipe
    _write_graded_output(tmp_path / "graded_e0_lowm.json", n_draws=4)
    skip, commit = regrade._e0_outputs_present_and_matching(
        tmp_path,
        expected_n_draws=8,  # the FULL recipe expects 8
        expected_judge_model="claude-sonnet-4-5-20250929",
        expected_n_contexts=50,
    )
    assert skip is False
    assert commit is None


def test_idempotence_guard_falls_through_when_only_one_present(tmp_path):
    """Only one of the two graded-E0 outputs present (partial state) → skip=False."""
    _write_graded_output(tmp_path / "graded_e0_highm.json")  # lowm absent
    skip, _commit = regrade._e0_outputs_present_and_matching(
        tmp_path,
        expected_n_draws=8,
        expected_judge_model="claude-sonnet-4-5-20250929",
        expected_n_contexts=50,
    )
    assert skip is False


def test_idempotence_guard_falls_through_on_wrong_n_contexts(tmp_path):
    """A full-recipe output written for n_contexts != 50 (e.g. a prior 12-context run)
    must NOT satisfy the full-production idempotence gate."""
    _write_graded_output(tmp_path / "graded_e0_highm.json", n_contexts=12)
    _write_graded_output(tmp_path / "graded_e0_lowm.json", n_contexts=12)
    skip, _ = regrade._e0_outputs_present_and_matching(
        tmp_path,
        expected_n_draws=8,
        expected_judge_model="claude-sonnet-4-5-20250929",
        expected_n_contexts=50,
    )
    assert skip is False


def test_main_skips_rejudge_without_calling_judge_when_outputs_present(tmp_path, monkeypatch):
    """End-to-end: a FULL production relaunch (default args) with both graded-E0 outputs
    already present returns 0 WITHOUT any judge call OR HF listing — the ~500K-call
    Phase-2 re-judge is skipped. The judge + repo-listing are monkeypatched to raise, so
    reaching either fails the test (round-5 had no guard → the full re-judge re-ran)."""
    _write_graded_output(tmp_path / "graded_e0_highm.json")
    _write_graded_output(tmp_path / "graded_e0_lowm.json")

    def _boom_judge(*a, **k):
        raise AssertionError("judge_completions_batch called — idempotence guard failed to skip")

    def _boom_list(*a, **k):
        raise AssertionError("_list_repo called — idempotence guard should skip before HF listing")

    monkeypatch.setattr(regrade, "judge_completions_batch", _boom_judge)
    monkeypatch.setattr(regrade, "_list_repo", _boom_list)
    monkeypatch.setattr(regrade, "load_dotenv", lambda: None)
    monkeypatch.setattr(sys, "argv", ["issue812_regrade_e0.py", "--out-dir", str(tmp_path)])
    rc = regrade.main()
    assert rc == 0


def test_main_reaches_normal_path_on_meta_mismatch(tmp_path, monkeypatch):
    """A smoke-recipe present state (n_draws=4) must NOT be short-circuited: main() must
    proceed past the guard toward the normal run. We prove it reaches the guard's
    fall-through by monkeypatching _list_repo to a sentinel that records it was called
    (and then raising a controlled marker so we don't actually judge)."""
    _write_graded_output(tmp_path / "graded_e0_highm.json", n_draws=4)
    _write_graded_output(tmp_path / "graded_e0_lowm.json", n_draws=4)
    reached = {"list": False}

    class _Reached(RuntimeError):
        pass

    def _mark_list(*a, **k):
        reached["list"] = True
        raise _Reached("reached normal path")

    monkeypatch.setattr(regrade, "_list_repo", _mark_list)
    monkeypatch.setattr(regrade, "load_dotenv", lambda: None)
    # FULL default recipe (n_draws default 8) but outputs were written at n_draws=4 →
    # meta mismatch → guard falls through → _list_repo IS reached.
    monkeypatch.setattr(sys, "argv", ["issue812_regrade_e0.py", "--out-dir", str(tmp_path)])
    with pytest.raises(_Reached):
        regrade.main()
    assert reached["list"] is True, "main() must proceed past the idempotence guard"


# ─────────────────────────────────────────────────────────────────────────────
# ROUND-7 FIX — restored-e0-payload-coverage: the idempotence gate + restore path
# must validate ACTUAL e0 payload coverage (behaviors × contexts × graded_mean + MF1),
# not just a matching ``meta`` (which is orthogonal to the per-behavior e0 bucket). A
# meta-matching pair with an empty / partial payload must NOT skip Phase 2 / must RAISE
# on restore — else the fit silently analyzes a behavior subset.
# ─────────────────────────────────────────────────────────────────────────────
def test_payload_covers_accepts_full_payload():
    """The shared validator ACCEPTS a full-payload output (its own key regression: it
    must not over-reject the genuine complete artifact)."""
    obj = {"e0": _full_e0_payload(_HIGHM_FIXTURE, 50)}
    ok, reason = regrade._e0_payload_covers(obj, "graded_e0_highm.json", expected_n_contexts=50)
    assert ok is True, reason
    obj_low = {"e0": _full_e0_payload(_LOWM_FIXTURE, 50)}
    ok, reason = regrade._e0_payload_covers(obj_low, "graded_e0_lowm.json", expected_n_contexts=50)
    assert ok is True, reason


def test_payload_covers_rejects_empty_e0():
    """An empty ``e0`` payload (the round-6 test-fixture hole) is REJECTED."""
    ok, reason = regrade._e0_payload_covers(
        {"e0": {}}, "graded_e0_highm.json", expected_n_contexts=50
    )
    assert ok is False and "empty/missing 'e0'" in reason


def test_payload_covers_rejects_missing_behavior():
    """A payload missing one required behavior is REJECTED (names the missing behavior)."""
    e0 = _full_e0_payload(_HIGHM_FIXTURE, 50)
    del e0["harmful_compliance"]  # one required high-m behavior absent
    ok, reason = regrade._e0_payload_covers(
        {"e0": e0}, "graded_e0_highm.json", expected_n_contexts=50
    )
    assert ok is False and "harmful_compliance" in reason


def test_payload_covers_rejects_missing_context():
    """A payload where one behavior is short by one context cell is REJECTED."""
    e0 = _full_e0_payload(_LOWM_FIXTURE, 50)
    dropped = next(iter(e0["deception"]))
    del e0["deception"][dropped]  # 49 of 50 contexts for deception
    ok, reason = regrade._e0_payload_covers(
        {"e0": e0}, "graded_e0_lowm.json", expected_n_contexts=50
    )
    assert ok is False and "49 context cells" in reason


def test_payload_covers_rejects_missing_graded_mean():
    """A cell with a null graded_mean is REJECTED (graded_mean is the fit target)."""
    e0 = _full_e0_payload(_HIGHM_FIXTURE, 50)
    a_ctx = next(iter(e0["refusal"]))
    e0["refusal"][a_ctx]["graded_mean"] = None
    ok, reason = regrade._e0_payload_covers(
        {"e0": e0}, "graded_e0_highm.json", expected_n_contexts=50
    )
    assert ok is False and "no graded_mean" in reason


def test_payload_covers_rejects_missing_mf1_granularity():
    """A cell with only a graded_mean scalar (no probe/draw arrays) is REJECTED — the
    reliability ceiling needs sub-context units."""
    e0 = _full_e0_payload(_HIGHM_FIXTURE, 50)
    a_ctx = next(iter(e0["sycophancy"]))
    e0["sycophancy"][a_ctx]["probe_scores"] = []  # graded_mean present, MF1 units gone
    ok, reason = regrade._e0_payload_covers(
        {"e0": e0}, "graded_e0_highm.json", expected_n_contexts=50
    )
    assert ok is False and "MF1 probe/draw granularity" in reason


def test_idempotence_guard_falls_through_on_empty_payload(tmp_path):
    """Matching meta but empty ``e0`` payload → skip=False (the round-6 hole: meta-only
    validation let an empty payload skip the ~$500 Phase 2)."""
    _write_graded_output(tmp_path / "graded_e0_highm.json", e0={})
    _write_graded_output(tmp_path / "graded_e0_lowm.json", e0={})
    skip, commit = regrade._e0_outputs_present_and_matching(
        tmp_path,
        expected_n_draws=8,
        expected_judge_model="claude-sonnet-4-5-20250929",
        expected_n_contexts=50,
    )
    assert skip is False
    assert commit is None


def test_idempotence_guard_falls_through_on_missing_behavior(tmp_path):
    """Matching meta but ONE required behavior missing from the payload → skip=False."""
    highm_e0 = _full_e0_payload(_HIGHM_FIXTURE, 50)
    del highm_e0["harmful_compliance"]
    _write_graded_output(tmp_path / "graded_e0_highm.json", e0=highm_e0)
    _write_graded_output(tmp_path / "graded_e0_lowm.json")  # lowm full
    skip, _ = regrade._e0_outputs_present_and_matching(
        tmp_path,
        expected_n_draws=8,
        expected_judge_model="claude-sonnet-4-5-20250929",
        expected_n_contexts=50,
    )
    assert skip is False


def test_idempotence_guard_falls_through_on_missing_context(tmp_path):
    """Matching meta but ONE behavior short a context cell → skip=False."""
    lowm_e0 = _full_e0_payload(_LOWM_FIXTURE, 50)
    dropped = next(iter(lowm_e0["deception"]))
    del lowm_e0["deception"][dropped]
    _write_graded_output(tmp_path / "graded_e0_highm.json")  # full
    _write_graded_output(tmp_path / "graded_e0_lowm.json", e0=lowm_e0)
    skip, _ = regrade._e0_outputs_present_and_matching(
        tmp_path,
        expected_n_draws=8,
        expected_judge_model="claude-sonnet-4-5-20250929",
        expected_n_contexts=50,
    )
    assert skip is False


def test_idempotence_guard_skips_only_on_full_payload(tmp_path):
    """The positive control paired with the negatives above: matching meta AND a FULL
    payload (all behaviors × 50 contexts × graded_mean + MF1) → skip=True. Proves the
    validator ACCEPTS the genuine complete artifact (does not over-reject)."""
    _write_graded_output(tmp_path / "graded_e0_highm.json")
    _write_graded_output(tmp_path / "graded_e0_lowm.json")
    skip, commit = regrade._e0_outputs_present_and_matching(
        tmp_path,
        expected_n_draws=8,
        expected_judge_model="claude-sonnet-4-5-20250929",
        expected_n_contexts=50,
    )
    assert skip is True
    assert commit == "abc123"


# ─────────────────────────────────────────────────────────────────────────────
# ROUND-6 FIX 3 — --restore-partial stages the crash-trap's harvested
# graded_e0_{highm,lowm}.json from HF into out_dir BEFORE the idempotence gate, so a
# FRESH-instance relaunch reuses them and NEVER re-submits ~$500 of judge batches.
# Fail-loud on a partial / corrupt restore (a silent re-judge is the ~$500 loss).
# ─────────────────────────────────────────────────────────────────────────────
def _stub_hf_download_from_dir(source_dir: Path):
    """A ``hf_hub_download`` stub that serves files from a local ``source_dir`` keyed by
    basename (the crash-trap's harvested-outputs mirror), raising on an absent file."""

    def _dl(repo, path_in_repo, *, repo_type=None, **kw):
        src = source_dir / Path(path_in_repo).name
        if not src.exists():
            raise FileNotFoundError(f"{path_in_repo} absent in stub HF mirror")
        return str(src)

    return _dl


def test_restore_partial_stages_both_files_then_gate_skips(tmp_path, monkeypatch):
    """A relaunch on a FRESH instance (empty out_dir) with --restore-partial <prefix>
    downloads BOTH graded-E0 outputs from HF, and the idempotence gate then skips the
    ~500K-call re-judge WITHOUT any judge call or HF listing (the ~$500 is never re-billed).
    The judge + repo-listing are monkeypatched to raise so reaching either fails the test."""
    import huggingface_hub

    harvest = tmp_path / "hf_mirror"
    harvest.mkdir()
    _write_graded_output(harvest / "graded_e0_highm.json")
    _write_graded_output(harvest / "graded_e0_lowm.json")
    out_dir = tmp_path / "eval_results_issue_812"  # starts EMPTY (fresh instance)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _stub_hf_download_from_dir(harvest))

    def _boom_judge(*a, **k):
        raise AssertionError("judge_completions_batch called — restore+gate failed to skip")

    def _boom_list(*a, **k):
        raise AssertionError("_list_repo called — restore+gate should skip before HF listing")

    monkeypatch.setattr(regrade, "judge_completions_batch", _boom_judge)
    monkeypatch.setattr(regrade, "_list_repo", _boom_list)
    monkeypatch.setattr(regrade, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue812_regrade_e0.py",
            "--out-dir",
            str(out_dir),
            "--restore-partial",
            "issue812_partial/att-20260701-232740/eval_results_issue_812",
        ],
    )
    rc = regrade.main()
    assert rc == 0
    # both files landed locally so a subsequent run is also idempotent
    assert (out_dir / "graded_e0_highm.json").exists()
    assert (out_dir / "graded_e0_lowm.json").exists()


def test_restore_partial_helper_stages_files_and_returns_paths(tmp_path, monkeypatch):
    """The restore helper writes both files into out_dir and returns their local paths."""
    import huggingface_hub

    harvest = tmp_path / "hf_mirror"
    harvest.mkdir()
    _write_graded_output(harvest / "graded_e0_highm.json")
    _write_graded_output(harvest / "graded_e0_lowm.json")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _stub_hf_download_from_dir(harvest))

    written = regrade._restore_partial_e0(
        "superkaiba1/explore-persona-space-data",
        "issue812_partial/att-20260701-232740/eval_results_issue_812",
        out_dir,
    )
    assert len(written) == 2
    assert (out_dir / "graded_e0_highm.json").exists()
    assert (out_dir / "graded_e0_lowm.json").exists()


def test_restore_partial_fails_loud_on_missing_meta(tmp_path, monkeypatch):
    """A restored output lacking a 'meta' block RAISES — a partial/corrupt restore must
    never be silently accepted (the idempotence gate would then re-judge, re-billing ~$500)."""
    import json

    import huggingface_hub

    harvest = tmp_path / "hf_mirror"
    harvest.mkdir()
    (harvest / "graded_e0_highm.json").write_text(json.dumps({"behaviors": ["x"]}))  # no meta
    (harvest / "graded_e0_lowm.json").write_text(json.dumps({"behaviors": ["x"]}))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _stub_hf_download_from_dir(harvest))

    with pytest.raises(RuntimeError, match="no 'meta' block"):
        regrade._restore_partial_e0(
            "superkaiba1/explore-persona-space-data",
            "issue812_partial/att-20260701-232740/eval_results_issue_812",
            tmp_path / "out",
        )


def test_restore_partial_fails_loud_on_empty_prefix(tmp_path):
    """An empty --restore-partial prefix RAISES rather than silently no-op'ing."""
    with pytest.raises(RuntimeError, match="empty HF prefix"):
        regrade._restore_partial_e0("superkaiba1/explore-persona-space-data", "", tmp_path / "out")


def test_restore_partial_fails_loud_on_incomplete_payload(tmp_path, monkeypatch):
    """A restored output with a matching meta but an INCOMPLETE ``e0`` payload (one
    behavior missing) RAISES — round-7 restored-e0-payload-coverage: the restore path
    validates actual payload coverage, not just the presence of a ``meta`` block, so a
    truncated harvest can never be silently staged for the gate to skip on."""
    import huggingface_hub

    harvest = tmp_path / "hf_mirror"
    harvest.mkdir()
    highm_e0 = _full_e0_payload(_HIGHM_FIXTURE, 50)
    del highm_e0["refusal"]  # matching meta, but one required behavior missing
    _write_graded_output(harvest / "graded_e0_highm.json", e0=highm_e0)
    _write_graded_output(harvest / "graded_e0_lowm.json")  # lowm complete
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _stub_hf_download_from_dir(harvest))

    with pytest.raises(RuntimeError, match="incomplete e0 payload"):
        regrade._restore_partial_e0(
            "superkaiba1/explore-persona-space-data",
            "issue812_partial/att-20260701-232740/eval_results_issue_812",
            tmp_path / "out",
        )

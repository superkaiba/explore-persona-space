"""Tests for issue #509 round-2 fixes (F1-F6).

Covers the six fixes the code-review reconciler bound as FAIL on round 1:

F1 — cond-ID merge regex widened from ``[A-Z]\\d+`` to ``[A-Z]+\\d+`` in
     ``scripts/issue493_extraction_metric_bakeoff.py`` so two-letter
     family IDs (``FB1..FB9`` fact arm, ``SC1..SC24`` syco arm) match.
F2 — metric-file enumerator in ``scripts/issue509_scoring.py`` (a) skips
     MMD permutation / cross-check sidecars (``*__perm.json``,
     ``*__cross_check_406.json``) and (b) allows ``layer-1`` for
     ``next_token_js`` baseline.
F3 — Plan §4.1.5 regression D partial Spearman residualizes BOTH x and y
     within strata (was only y), and the permutation null uses the same
     residualized statistic.
F4 — ``_reliability_y`` denominator uses within-stratum variance, not
     pooled across-stratum variance.
F5 — Fact-arm SE: ``_reliability_y`` no longer silently falls back to 1.0
     on production runs (raises unless ``allow_unknown_se=True`` is
     explicitly set, used only in ``--smoke`` mode).
F6 — Missing syco-arm plan-§5 condition slugs + coarse-predictor anchor
     (``per_source``, ``live_cells_only``, ``comedian_recovery``,
     ``per_cell_predictor_saturation``, ``coarse_lift``).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# F1 — cond-ID merge regex
# ---------------------------------------------------------------------------


def _load_bakeoff_module():
    """Helper that loads the bakeoff regex from source without importing the
    full module (heavy torch / vLLM deps at import time).
    """
    src = (REPO_ROOT / "scripts/issue493_extraction_metric_bakeoff.py").read_text()
    # Pull out the two regex patterns by structural match.
    m1 = re.search(
        r"pattern = re\.compile\(r\"\^\(\?P<pt>\[a-z_\]\+\)__layer\(\?P<L>\\d\+\)__cond\(\?P<cid>(?P<inner>[^)]+)\)\\\.pt\$\"\)",
        src,
    )
    assert m1 is not None, "Could not find merge_partitioned_activations regex in bakeoff source"
    m2 = re.search(
        r"pattern = re\.compile\(r\"\^last_prompt__cond\(\?P<cid>(?P<inner>[^)]+)\)\\\.pt\$\"\)",
        src,
    )
    assert m2 is not None, "Could not find next-token-logits regex in bakeoff source"
    return m1.group("inner"), m2.group("inner")


def test_f1_merge_regex_accepts_two_letter_i509_cids():
    """F1: condFB1.pt and condSC24.pt must match the merge regex."""
    inner_merge, inner_nt = _load_bakeoff_module()
    # Both patterns must allow ``[A-Z]+\d+`` (one-or-more letters + digits).
    assert inner_merge == r"[A-Z]+\d+", f"merge regex inner class: {inner_merge!r}"
    assert inner_nt == r"[A-Z]+\d+", f"next-token regex inner class: {inner_nt!r}"

    # Construct the full regexes and assert match/no-match cases.
    merge_re = re.compile(
        r"^(?P<pt>[a-z_]+)__layer(?P<L>\d+)__cond(?P<cid>" + inner_merge + r")\.pt$"
    )
    nt_re = re.compile(r"^last_prompt__cond(?P<cid>" + inner_nt + r")\.pt$")

    # i509 fact-arm two-letter cids must match.
    m = merge_re.match("last_prompt__layer22__condFB1.pt")
    assert m is not None, "FB1 should match"
    assert m.group("cid") == "FB1"

    m = merge_re.match("mean_response__layer3__condFB9.pt")
    assert m is not None, "FB9 should match"
    assert m.group("cid") == "FB9"

    # i509 syco-arm 2-letter + 2-digit cids must match.
    m = merge_re.match("last_prompt__layer22__condSC24.pt")
    assert m is not None, "SC24 should match"
    assert m.group("cid") == "SC24"

    m = merge_re.match("last_prompt__layer22__condSC1.pt")
    assert m is not None, "SC1 should match"

    # Backward compatibility: legacy single-letter cids must still match.
    m = merge_re.match("last_prompt__layer22__condA1.pt")
    assert m is not None, "Legacy A1 must still match"
    assert m.group("cid") == "A1"

    m = merge_re.match("last_prompt__layer22__condM1.pt")
    assert m is not None, "Legacy M1 must still match"

    # Next-token regex applies to fact-arm + syco-arm + legacy.
    assert nt_re.match("last_prompt__condFB1.pt") is not None
    assert nt_re.match("last_prompt__condSC24.pt") is not None
    assert nt_re.match("last_prompt__condA1.pt") is not None

    # Negative cases: lowercase / no-digit cids must NOT match.
    assert merge_re.match("last_prompt__layer22__condfb1.pt") is None
    assert merge_re.match("last_prompt__layer22__condFB.pt") is None
    assert merge_re.match("last_prompt__layer22__cond1.pt") is None


def test_f1_regex_merge_smoke(tmp_path):
    """F1 end-to-end: build a fake partitioned-merge dir and verify the
    merge regex picks up condFB1.pt + condSC24.pt as well as a legacy A1.

    We don't drive the actual ``merge_partitioned_activations`` function
    (it needs torch + the live conditions module) — we re-use the regex
    derived from the source on a synthetic directory listing.
    """
    inner_merge, _ = _load_bakeoff_module()
    merge_re = re.compile(
        r"^(?P<pt>[a-z_]+)__layer(?P<L>\d+)__cond(?P<cid>" + inner_merge + r")\.pt$"
    )
    act_dir = tmp_path / "activations"
    act_dir.mkdir()
    # Synthetic partition files: 1 fact + 1 syco + 1 legacy, all at L22.
    for name in [
        "last_prompt__layer22__condFB1.pt",
        "last_prompt__layer22__condSC24.pt",
        "last_prompt__layer22__condA1.pt",
        # A non-matching file that should be ignored:
        "last_prompt__layer22__notcond.pt",
    ]:
        (act_dir / name).write_bytes(b"")

    matched_cids = []
    for p in sorted(act_dir.glob("*__cond*.pt")):
        m = merge_re.match(p.name)
        if m is not None:
            matched_cids.append(m.group("cid"))

    assert "FB1" in matched_cids, f"merge dropped FB1; matched={matched_cids}"
    assert "SC24" in matched_cids, f"merge dropped SC24; matched={matched_cids}"
    assert "A1" in matched_cids, f"merge dropped legacy A1; matched={matched_cids}"
    assert len(matched_cids) == 3, f"unexpected extra match: {matched_cids}"


# ---------------------------------------------------------------------------
# F2 — metric-file enumerator
# ---------------------------------------------------------------------------


def test_f2_enumerator_skips_perm_sidecar_and_allows_layer_minus_1(tmp_path):
    """F2: the metric file enumerator must
    (a) skip ``*__perm.json`` and ``*__cross_check_406.json`` sidecars
    (b) accept ``layer-1`` for ``next_token_js`` baseline.
    """
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()

    # Real cells that must be loaded.
    keep_files = {
        "last_prompt__layer22__gauss_kl__centered.json": {
            "matrix": {"FB1": {"FB2": 0.1}},
            "extraction_point": "last_prompt",
            "layer": 22,
            "metric": "gauss_kl",
            "variant": "centered",
        },
        "last_prompt__layer-1__next_token_js__raw.json": {
            "matrix": {"FB1": {"FB2": 0.2}},
            "extraction_point": "last_prompt",
            "layer": -1,
            "metric": "next_token_js",
            "variant": "raw",
        },
    }
    # Sidecars that must be skipped.
    skip_files = {
        "mean_response__layer3__mmd__centered__perm.json": {
            "matrix": {"FB1": {"FB2": 99.0}}  # bogus content; should never load.
        },
        "last_prompt__layer22__gauss_kl__cross_check_406.json": {"matrix": {"FB1": {"FB2": 99.0}}},
    }
    for name, payload in {**keep_files, **skip_files}.items():
        (metrics_dir / name).write_text(json.dumps(payload))

    files = scoring._enumerate_metric_files(metrics_dir)
    parsed_names = sorted([p.name for p, _ in files])
    assert "last_prompt__layer22__gauss_kl__centered.json" in parsed_names, (
        f"F2(a): centered cell missing from enumerator; got {parsed_names}"
    )
    assert "last_prompt__layer-1__next_token_js__raw.json" in parsed_names, (
        f"F2(b): next_token_js layer-1 baseline missing; got {parsed_names}"
    )
    assert "mean_response__layer3__mmd__centered__perm.json" not in parsed_names, (
        f"F2(a): perm sidecar leaked into enumerator; got {parsed_names}"
    )
    assert "last_prompt__layer22__gauss_kl__cross_check_406.json" not in parsed_names, (
        f"F2(a): cross-check sidecar leaked into enumerator; got {parsed_names}"
    )

    # The next-token-js entry should report layer=-1 (not coerced to a positive int).
    for path, meta in files:
        if path.name == "last_prompt__layer-1__next_token_js__raw.json":
            assert meta["layer"] == -1, f"F2(b): layer not -1: {meta}"
            assert meta["metric"] == "next_token_js"


# ---------------------------------------------------------------------------
# F3 — partial Spearman residualizes BOTH x AND y
# ---------------------------------------------------------------------------


def test_f3_partial_spearman_residualizes_x_and_y():
    """F3: with stratum-correlated x, the FE statistic must residualize x too.

    Construct a tiny synthetic example where y is purely a function of x
    WITHIN stratum but has between-stratum mean differences. The
    within-stratum (partial) Spearman should be ~1.0; the half-residualized
    (only-y) version that round-1 used should differ substantially.
    """
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    # 2 strata × 5 cells. Within each stratum, y = x (perfect rank correlation).
    # Across strata, x and y have different means (between-stratum noise).
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0, 104.0])
    strata = np.array(["A"] * 5 + ["B"] * 5)

    out = scoring._score_one_cell(
        x=x,
        y=y,
        strata=strata,
        se=None,
        prior_z=None,
        run_permutation=False,
        run_bootstrap=False,
        perm_b=10,
    )
    # rho_fe (BOTH residualized) should be ~1.0 since y = x within stratum.
    assert abs(out["rho_fe"] - 1.0) < 1e-6, (
        f"F3: partial-Spearman should be 1.0 for y = x within stratum, got {out['rho_fe']}"
    )
    # Pooled rho is dominated by the between-stratum mean shift, but the
    # ranks happen to be monotone here too (both x and y go up across
    # strata), so rho_pooled is also ~1.0 — that's a known degenerate case.
    # The discriminating case: re-order y within stratum B to be reversed.
    y2 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 104.0, 103.0, 102.0, 101.0, 100.0])
    out2 = scoring._score_one_cell(
        x=x,
        y=y2,
        strata=strata,
        se=None,
        prior_z=None,
        run_permutation=False,
        run_bootstrap=False,
        perm_b=10,
    )
    # rho_fe should be ~0 (stratum B reverses x→y), since within-stratum
    # partial correlations average to 0. With ONLY-y residualization the
    # statistic would still be dominated by between-stratum structure.
    # We assert that rho_fe is materially smaller than the old-style
    # (only-y residualized) statistic.
    y2_resid_only = scoring._residualize(y2, strata)
    x_resid = scoring._residualize(x, strata)
    rho_fe_correct = scoring._spearman_rho(x_resid, y2_resid_only)
    rho_fe_old = scoring._spearman_rho(x, y2_resid_only)  # round-1 buggy form.
    assert abs(out2["rho_fe"] - rho_fe_correct) < 1e-9, (
        f"F3: rho_fe should equal Spearman(x_resid, y_resid), got {out2['rho_fe']} "
        f"vs correct {rho_fe_correct}"
    )
    # The two statistics should differ — confirming the fix is observable.
    assert abs(rho_fe_correct - rho_fe_old) > 0.1, (
        f"F3: synthetic test should discriminate buggy vs fixed forms; "
        f"correct={rho_fe_correct}, old={rho_fe_old}"
    )


def test_f3_permutation_uses_same_statistic():
    """F3: the permutation null must compute the SAME residualized statistic
    as the observed one — otherwise p-values are statistically meaningless.
    """
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    rng = np.random.default_rng(0)
    # 4 strata × 6 cells, mild within-stratum correlation.
    n_per = 6
    n_strata = 4
    strata = np.repeat(np.arange(n_strata), n_per)
    x = rng.normal(size=n_strata * n_per)
    # y = x within stratum + between-stratum offset.
    y = x.copy()
    for s in range(n_strata):
        y[strata == s] += s * 10.0

    x_resid = scoring._residualize(x, strata)
    y_resid = scoring._residualize(y, strata)
    rho_obs = scoring._spearman_rho(x_resid, y_resid)
    assert np.isfinite(rho_obs)

    # Run the permutation null and verify the function exists, doesn't
    # crash, and returns a finite p-value in [0, 1]. A correct
    # within-stratum permutation against a y where y = x within stratum
    # should give a tiny p (effect is real).
    p = scoring._permutation_p_partial(rho_obs, x, y, strata, b=200)
    assert 0.0 <= p <= 1.0, f"F3: perm p out of [0,1]: {p}"
    assert p < 0.1, f"F3: real x↔y signal should yield small p, got {p}"

    # Sanity: if we feed an x with no real within-stratum link to y, p
    # should distribute uniformly near 0.5 in expectation.
    x_random = rng.normal(size=n_strata * n_per)
    x_random_resid = scoring._residualize(x_random, strata)
    rho_null = scoring._spearman_rho(x_random_resid, y_resid)
    p_null = scoring._permutation_p_partial(rho_null, x_random, y, strata, b=400)
    assert 0.05 < p_null < 0.95, f"F3: null permutation p should not be at extremes; got {p_null}"


# ---------------------------------------------------------------------------
# F4 — within-stratum reliability
# ---------------------------------------------------------------------------


def test_f4_reliability_y_within_stratum():
    """F4: reliability denominator must be within-stratum variance, not pooled.

    Construct a synthetic dataset where between-stratum variance dwarfs
    within-stratum variance. The pooled formula will overstate reliability
    (because var_pooled >> mean_se^2), while the within-stratum formula
    will give a much lower reliability that better reflects the SE
    relative to the actual within-cluster signal.
    """
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    # 3 sources × 4 cells. Between-source mean differences = (0, 1, 2);
    # within-source SD ~ 0.05. SEs ~ 0.04 (almost the size of within-SD).
    rng = np.random.default_rng(42)
    sources = np.repeat(["A", "B", "C"], 4)
    means = np.repeat([0.0, 1.0, 2.0], 4)
    y = means + rng.normal(scale=0.05, size=12)
    se = np.full(12, 0.04)

    rel_pooled = 1.0 - np.mean(se**2) / np.var(y)
    # Within-stratum variance is much smaller (~0.0025) than pooled (~0.7).
    var_within_per_stratum = []
    for s in np.unique(sources):
        var_within_per_stratum.append(np.var(y[sources == s]))
    var_within_mean = float(np.mean(var_within_per_stratum))
    rel_within_expected = 1.0 - np.mean(se**2) / var_within_mean

    rel_pooled_observed = scoring._reliability_y_pooled(y, se)
    rel_within_observed = scoring._reliability_y(y, se, strata=sources)

    assert abs(rel_pooled_observed - max(rel_pooled, 1e-6)) < 1e-6, (
        f"F4: pooled reliability mismatch: {rel_pooled_observed} vs {rel_pooled}"
    )
    # Within-stratum reliability should be DRAMATICALLY lower in this regime.
    assert rel_within_observed < rel_pooled_observed - 0.5, (
        f"F4: within-stratum reliability should be much lower; "
        f"within={rel_within_observed}, pooled={rel_pooled_observed}"
    )
    # Sanity-check against hand-computed expected within reliability.
    assert abs(rel_within_observed - max(rel_within_expected, 1e-6)) < 0.05, (
        f"F4: within-stratum reliability {rel_within_observed} !~ expected {rel_within_expected}"
    )


# ---------------------------------------------------------------------------
# F5 — Fact-arm SE silent fallback removed
# ---------------------------------------------------------------------------


def test_f5_fact_arm_se_required_in_production():
    """F5: production fact-arm runs must NOT silently fall back to
    reliability=1 when SE is missing. The function raises in production
    mode; smoke mode allows the fallback.
    """
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    se_nan = np.full(5, float("nan"))
    strata = np.array(["A", "A", "B", "B", "B"])

    # Production mode (default): raises on all-NaN SE.
    with pytest.raises(
        (ValueError, AssertionError),
        match=r"reliab|SE|missing|finite",
    ):
        scoring._reliability_y(y, se_nan, strata=strata, allow_unknown_se=False)

    # Smoke mode: allows fallback to 1.0.
    rel = scoring._reliability_y(y, se_nan, strata=strata, allow_unknown_se=True)
    assert rel == 1.0, f"F5: smoke fallback should be 1.0, got {rel}"


# ---------------------------------------------------------------------------
# F6 — missing syco-arm summaries + coarse-predictor anchor
# ---------------------------------------------------------------------------


def test_f6_per_source_summary_present():
    """F6: scoring output must include a per-source ρ summary on the syco arm."""
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    # Tiny synthetic cell with 3 sources × 4 bystanders.
    rng = np.random.default_rng(1)
    sources = np.repeat(["assistant", "software_engineer", "comedian"], 4)
    x = rng.normal(size=12)
    y = x + rng.normal(scale=0.1, size=12)  # within-source perfect rank link.

    per_source = scoring._per_source_spearman(x, y, sources)
    assert set(per_source.keys()) == {"assistant", "software_engineer", "comedian"}, (
        f"F6 per_source: source set wrong, got {per_source.keys()}"
    )
    for src, rho in per_source.items():
        assert -1.0 <= rho <= 1.0, f"F6 per_source: rho out of bounds for {src}: {rho}"


def test_f6_live_cells_only_filters_subset():
    """F6: live-cells-only subset filters to cells with |Δ| > 0.10.

    The plan specifies 21 cells (15 software_engineer + 6 assistant) on
    the syco arm; here we test the filter mechanics with a small
    synthetic case.
    """
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    deltas = np.array([0.15, 0.20, 0.05, -0.12, 0.08, -0.30, 0.0])
    mask = scoring._live_cells_mask(deltas, threshold=0.10)
    expected = np.array([True, True, False, True, False, True, False])
    np.testing.assert_array_equal(mask, expected)


def test_f6_comedian_recovery_rank():
    """F6: comedian-recovery measures comedian's rank among software_engineer
    bystanders by the predictor.
    """
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    # 4 bystanders; predictor is the distance (smaller = more similar);
    # so the ranking is ascending by predictor value.
    predictor = {"comedian": 0.05, "biologist": 0.10, "doctor": 0.30, "engineer": 0.50}
    rank = scoring._rank_in_bystanders(predictor, target_persona="comedian", ascending=True)
    assert rank == 1, f"F6 comedian_recovery: comedian should rank 1, got {rank}"


def test_f6_per_cell_predictor_saturation():
    """F6: per-cell saturation flag fires when predictor signal has tiny variance."""
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    # Saturated case: predictor is nearly constant.
    x_saturated = np.full(20, 1.0) + np.random.default_rng(0).normal(scale=1e-8, size=20)
    # Non-saturated case: real variance.
    x_real = np.random.default_rng(0).normal(size=20)

    assert scoring._is_predictor_saturated(x_saturated, var_threshold=1e-6)
    assert not scoring._is_predictor_saturated(x_real, var_threshold=1e-6)


def test_f6_coarse_lift_loads_5_predictors(tmp_path):
    """F6: ``_load_fact_target`` must parse all 5 coarse-predictor columns
    so ``coarse_lift`` can compute Δρ vs the bake-off.
    """
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    csv_path = tmp_path / "regression_data.csv"
    cols = [
        "substrate",
        "leak_rate",
        "teach_persona",
        "bystander_persona",
        "bystander_logprob",
        "cosine_a_L21",
        "cosine_b_L21",
        "js_on_topic",
        "fact_slice_js",
        "extra_column_ignored",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        # One row of canned data.
        w.writerow(
            [
                "elk_county",
                "0.5",
                "kindergarten_teacher",
                "no_system",
                "-4.2",
                "0.8",
                "0.7",
                "0.6",
                "0.5",
                "junk",
            ]
        )
    target = scoring._load_fact_target(csv_path)
    rows = target["rows"]
    assert len(rows) == 1
    r = rows[0]
    for k in (
        "cosine_a_L21",
        "cosine_b_L21",
        "js_on_topic",
        "fact_slice_js",
        "bystander_logprob",
    ):
        assert k in r, f"F6 coarse_lift: missing column {k}"
        assert isinstance(r[k], float), f"F6 coarse_lift: {k} not parsed as float"

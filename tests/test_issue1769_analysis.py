"""#1769 analysis pins: the §3 lattice, the instability guard, e2e on fixtures.

The lattice classifier is pinned on boundary fixtures (incl. the plan's
CI [-0.9, -0.4] -> mixed/indeterminate case: a significantly NEGATIVE
decode-only shift must never read "decode contributes nothing"), on
disjointness/exhaustiveness over a boundary grid, and the pinned
instability-guard predicate + its routing (crafted bootstrap indices with
exactly 2% nonpositive-denominator resamples: operating rule passes, ratio
CI routes to mixed/indeterminate — the deliberately-stricter-than-K1
ordering)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import issue1769_analysis as ana

# ── lattice pins ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("lo", "hi", "want"),
    [
        (-0.9, -0.4, "mixed/indeterminate"),  # plan §3: negative decode arm
        (-0.24, 0.24, "prefill-committed"),
        (-0.25, 0.24, "mixed/indeterminate"),  # lower bound must be STRICTLY > -0.25
        (-0.24, 0.25, "mixed/indeterminate"),  # upper bound must be STRICTLY < 0.25
        (0.76, 1.2, "decode-driven"),
        (0.75, 1.2, "mixed/indeterminate"),  # lower bound must be STRICTLY > 0.75
        (0.3, 0.6, "mixed/indeterminate"),
        (-0.1, 0.9, "mixed/indeterminate"),  # wide CI spanning both thresholds
        (0.0, 0.0, "prefill-committed"),  # degenerate point-CI at 0
    ],
)
def test_lattice_boundary_fixtures(lo, hi, want):
    assert ana.classify_lattice(lo, hi) == want


def test_lattice_disjoint_and_exhaustive_on_grid():
    labels = {"prefill-committed", "decode-driven", "mixed/indeterminate"}
    grid = np.linspace(-1.5, 1.5, 61)
    for lo in grid:
        for hi in grid:
            if hi < lo:
                continue
            got = ana.classify_lattice(float(lo), float(hi))
            assert got in labels, (lo, hi, got)
            # Disjointness: the two positive labels' conditions cannot co-hold
            # (an interval inside (-0.25, 0.25) has hi < 0.25 <= 0.75 < lo is absurd).
            assert not ((lo > -ana.LATTICE_LO and hi < ana.LATTICE_LO) and lo > ana.LATTICE_HI), (
                lo,
                hi,
            )


def test_instability_guard_predicate_pins():
    assert ana.ratio_unstable(0.011) is True
    assert ana.ratio_unstable(0.0100001) is True
    assert ana.ratio_unstable(0.01) is False  # STRICTLY above 1% fires
    assert ana.ratio_unstable(0.0) is False


def test_err_offsets_clamped_nonnegative():
    lo, hi = 1.2, 0.9  # inverted quantile CI (tiny-n epsilon class)
    assert ana._err(1.0, lo, hi) == (0.0, 0.0)
    assert ana._err(5.0, 4.0, 7.0) == (1.0, 2.0)


# ── fixtures ──────────────────────────────────────────────────────────


def _add_items(per_item, trait, arm, alpha, scores_by_q, n_draws=3, coherent=True):
    for qid, s in scores_by_q.items():
        for di in range(n_draws):
            key = f"{trait}/{arm}/a{alpha}/q{qid}/d{di}" if alpha else f"{trait}/{arm}/q{qid}/d{di}"
            per_item[key] = {
                "cell_id": key.rsplit("/d", 1)[0],
                "trait": trait,
                "arm": arm,
                "alpha": alpha,
                "question_id": qid,
                "draw": di,
                "coherent": coherent,
                "graded_score": float(s + 0.1 * di),  # tiny per-draw jitter
                "binary_positive": s >= 50,
                "n_kept_draws": 5,
                "n_content_drops": 0,
                "n_transport_losses": 0,
            }


@pytest.fixture(scope="module")
def graded_fixture():
    """3 traits: 'alpha' prefill-committed, 'beta' decode-driven, 'gamma' no
    passing operating alpha (ceiling guard fails at every alpha)."""
    rng = np.random.default_rng(7)
    per_item: dict = {}
    nq = 6
    for trait, decode_like_both in (("alpha", False), ("beta", True)):
        neither = {q: 10 + rng.uniform(-1, 1) for q in range(nq)}
        _add_items(per_item, trait, "neither", None, neither)
        for a in (1.0, 2.0):
            both = {q: neither[q] + 45 + a + rng.uniform(-2, 2) for q in range(nq)}
            if decode_like_both:
                decode = {q: both[q] + rng.uniform(-2, 2) for q in range(nq)}
                prefill = {q: neither[q] + rng.uniform(0, 2) for q in range(nq)}
            else:
                decode = {q: neither[q] + rng.uniform(0, 1) for q in range(nq)}
                prefill = {q: both[q] + rng.uniform(-2, 0) for q in range(nq)}
            _add_items(per_item, trait, "both", a, both)
            _add_items(per_item, trait, "decode_only", a, decode)
            _add_items(per_item, trait, "prefill_only", a, prefill)
    # gamma: strong shift but both-arm mean is ceiling-stacked (> 85) at
    # every alpha -> NO passing operating alpha -> explicit indeterminate.
    neither = {q: 20.0 for q in range(nq)}
    _add_items(per_item, "gamma", "neither", None, neither)
    for a in (1.0, 2.0):
        _add_items(per_item, "gamma", "both", a, {q: 95.0 for q in range(nq)})
        _add_items(per_item, "gamma", "decode_only", a, {q: 60.0 for q in range(nq)})
        _add_items(per_item, "gamma", "prefill_only", a, {q: 90.0 for q in range(nq)})
    return {"per_item": per_item}


@pytest.fixture(scope="module")
def headline(graded_fixture):
    return ana.run_analysis(graded_fixture, b_boot=500)


def test_e2e_classifications(headline):
    per = headline["per_trait"]
    assert per["alpha"]["classification"] == "prefill-committed"
    assert per["beta"]["classification"] == "decode-driven"
    assert per["gamma"]["classification"] == "mixed/indeterminate"
    assert per["gamma"]["classification_reason"] == "no passing operating alpha"
    assert per["gamma"]["operating_alpha"] is None
    assert headline["k1_manipulation_check"]["passed"] is True


def test_e2e_operating_alpha_and_fractions(headline):
    rec = headline["per_trait"]["alpha"]
    assert rec["operating_alpha"] == 2.0  # largest passing rung
    f_d = rec["fractions"]["f_d"]
    assert f_d["ci95_frozen"][0] > -0.25 and f_d["ci95_frozen"][1] < 0.25
    assert "ci95_selection_inherited" in f_d
    assert rec["selection_inherited"]["n_empty_alpha_resamples"] == 0
    assert rec["ratio_unstable"] is False
    # f_p companion reported (consistency read), never classified.
    assert rec["fractions"]["f_p"]["point"] > 0.8


def test_e2e_coherent_only_labeled_and_ceiling_diag(headline):
    assert "selection-conditioned" in headline["coherent_only_label"]
    diag = headline["ceiling_diagnostic_frac_gt90"]
    assert diag["gamma/both/a1"] == 1.0  # ceiling-stacked arm reads 1.0
    assert diag["alpha/neither"] == 0.0


def test_instability_routing_with_crafted_indices():
    """Operating rule passes (CI excl. 0, floor met) while EXACTLY 2% of
    resamples have Delta*_both <= 0 -> ratio-instability routes the trait to
    mixed/indeterminate with absolute deltas retained."""
    nq = 10
    qids = list(range(nq))
    neither = dict.fromkeys(range(nq), 10.0)
    neither[9] = 70.0
    both = dict.fromkeys(range(nq), 50.0)  # delta +40 on q0..q8
    both[9] = 10.0  # delta -60 on q9
    scores_t = {
        ("neither", None): neither,
        ("both", 1.0): both,
        ("decode_only", 1.0): dict(neither),  # delta 0
        ("prefill_only", 1.0): dict(both),  # delta == both
    }
    coher_t = {k: dict.fromkeys(range(nq), 1.0) for k in scores_t}
    # Crafted indices: resamples 0,1 draw ONLY the negative-delta question
    # (Delta*_both = -60); the other 98 are the identity multiset (+30).
    idx = np.tile(np.arange(nq), (100, 1))
    idx[0, :] = 9
    idx[1, :] = 9
    rec = ana.analyze_trait("crafted", scores_t, coher_t, [1.0], qids, idx)
    assert rec["operating_alpha"] == 1.0
    assert abs(rec["frac_bootstrap_delta_both_nonpositive"] - 0.02) < 1e-9
    assert rec["ratio_unstable"] is True
    assert rec["classification"] == "mixed/indeterminate"
    assert "ratio-instability" in rec["classification_reason"]
    assert rec["deltas_at_operating_alpha"]["both"]["delta"] == pytest.approx(30.0)


def test_main_writes_headline_and_figures(graded_fixture, tmp_path):
    graded_path = tmp_path / "graded_scores.json"
    graded_path.write_text(json.dumps(graded_fixture))
    out = tmp_path / "analysis" / "headline.json"
    figs = tmp_path / "figs"
    with pytest.raises(SystemExit) as exc:
        ana.main(
            [
                "--graded",
                str(graded_path),
                "--out",
                str(out),
                "--fig-dir",
                str(figs),
                "--b-boot",
                "200",
            ]
        )
    assert exc.value.code == 0
    headline = json.loads(out.read_text())
    assert set(headline["per_trait"]) == {"alpha", "beta", "gamma"}
    pngs = sorted(figs.glob("*.png"))
    assert len(pngs) == 4, pngs
    for p in pngs:
        assert p.stat().st_size > 5000, (p, p.stat().st_size)  # non-empty render

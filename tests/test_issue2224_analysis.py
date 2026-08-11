"""Regression tests for scripts/issue2224_analysis.py (unit 3).

Pins: (1) the registered verdict-lattice category logic (plan §3 — disjoint +
exhaustive); (2) the bootstrap helpers (determinism, cluster variant, the
non-negative errorbar-offset clamp); (3) paired-contrast None handling; (4) the
fixtures → analyze-4a → analyze-4b end-to-end path (JSON outputs, no network,
no GPU); (5) the matplotlib xerr/yerr gotcha — a deliberately INVERTED quantile
CI routed through the REAL hero-figure function to savefig (gotchas.md rule).

All paths are tmp_path-rooted; no canonical eval_results/figures writes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import issue2224_analysis as ana  # noqa: E402


def _args(argv: list[str]):
    return ana.build_argparser().parse_args(argv)


# ── Verdict lattice ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("exact_lo", "exact_hi", "rand_lo", "expected"),
    [
        (-5.0, -1.0, 3.0, "below_exact"),  # Δ CI wholly below 0
        (1.0, 5.0, -1.0, "beats_exact"),  # Δ CI excludes 0 positive
        (-2.0, 2.0, 1.0, "matches_exact_and_beats_random"),  # straddle + random+
        (-2.0, 2.0, -1.0, "inconclusive"),  # straddle, random CI not positive
    ],
)
def test_lattice_categories(exact_lo, exact_hi, rand_lo, expected):
    d_exact = {"ci_lo": exact_lo, "ci_hi": exact_hi}
    d_random = {"ci_lo": rand_lo, "ci_hi": rand_lo + 4.0}
    assert ana.lattice_category(d_exact, d_random) == expected


# ── Bootstrap helpers ────────────────────────────────────────────────────────────


def test_boot_mean_ci_deterministic_and_ordered():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    a = ana.boot_mean_ci(v, n_boot=500, seed=7)
    b = ana.boot_mean_ci(v, n_boot=500, seed=7)
    assert a == b
    assert a["ci_lo"] <= a["mean"] <= a["ci_hi"]
    assert a["n_items"] == 5
    assert 0.0 <= a["boot_frac_below0"] <= 1.0


def test_cluster_boot_matches_units():
    v = np.array([1.0, 1.2, 5.0, 5.5])
    clusters = ["q0", "q0", "q1", "q1"]
    r = ana.cluster_boot_mean_ci(v, clusters, n_boot=400, seed=3)
    assert r["n_clusters"] == 2
    assert r["ci_lo"] <= r["mean"] <= r["ci_hi"]


def test_ci_err_clamps_inverted_bounds():
    # Inverted CI (lo > v > hi) must yield NON-NEGATIVE offsets (xerr/yerr gotcha).
    lo_off, hi_off = ana.ci_err(1.0, 2.0, 0.5)
    assert lo_off == 0.0 and hi_off == 0.0
    lo_off, hi_off = ana.ci_err(1.0, 0.5, 2.0)
    assert lo_off == 0.5 and hi_off == 1.0


# ── Paired contrast ──────────────────────────────────────────────────────────────


def _cell(cid: str, method: str, per: dict, incoherent: bool = False) -> dict:
    kept = [v for v in per.values() if v is not None]
    return {
        "cell_id": cid,
        "corpus": "c",
        "trait": "t",
        "method": method,
        "tail": "top",
        "trait_expression": {
            "graded_mean": float(np.mean(kept)),
            "rate_gt50": float(np.mean([v > 50 for v in kept])),
            "n_items": len(per),
            "n_scored_items": len(kept),
            "per_item_scores": per,
            "telemetry": {},
        },
        "coherence": {"graded_mean": 20.0 if incoherent else 80.0, "incoherent_flag": incoherent},
    }


def test_paired_contrast_drops_none_and_pairs_on_shared_slots():
    a = _cell("a", "m", {"q0-g0": 60.0, "q0-g1": None, "q1-g0": 70.0, "q2-g0": 50.0})
    b = _cell("b", "random", {"q0-g0": 20.0, "q0-g1": 25.0, "q1-g0": 30.0}, incoherent=True)
    c = ana._paired_contrast(a, b, n_boot=300, seed_base=42)
    assert c["status"] == "ok"
    assert c["n_paired"] == 2  # q0-g0 + q1-g0 (None dropped; q2 unshared)
    assert c["response_level"]["mean"] == pytest.approx(40.0)
    assert c["coherence_flag_cells"] == ["b"]


def test_paired_contrast_no_shared_items():
    a = _cell("a", "m", {"q0-g0": 60.0})
    b = _cell("b", "random", {"q9-g0": 10.0})
    assert ana._paired_contrast(a, b, 100, 1)["status"] == "no_paired_items"


# ── End-to-end on the synthetic fixture tree (JSON outputs; no figures) ──────────


def _fixture_argv(root: Path, extra: list[str]) -> list[str]:
    return [
        "--fixtures-root",
        str(root),
        "--selections-dir",
        str(root / "sel"),
        "--trait-scores-dir",
        str(root / "ft"),
        "--scores-dir",
        str(root / "scores"),
        "--suite-scores",
        str(root / "suite_scores.json"),
        "--families-json",
        str(root / "families.json"),
        "--dataset-means-json",
        str(root / "dataset_means.json"),
        "--dataset-level-out",
        str(root / "out_4a"),
        "--analysis-4b-out",
        str(root / "out_4b"),
        "--figures-dir",
        str(root / "figs"),
        *extra,
    ]


def test_e2e_fixtures_then_4a_4b(tmp_path):
    root = tmp_path / "fix"
    assert ana.run_fixtures(_args(_fixture_argv(root, []))) == 0

    args_4a = _args(_fixture_argv(root, ["--no-figures", "--n-boot", "80"]))
    assert ana.run_4a(args_4a) == 0
    corr = json.loads((root / "out_4a" / "correlations.json").read_text())
    assert set(corr["per_trait"]) == {"evil", "sycophancy", "hallucination"}
    ev = corr["per_trait"]["evil"]
    assert ev["status"] == "ok" and ev["n_families"] == 4
    exact = ev["per_arm"]["exact_dp"]
    assert exact["status"] == "ok" and exact["spearman_rho"] is not None
    # Sycophancy: the fixture makes prompttoken weak — the H1 gain read must fire.
    syc = corr["per_trait"]["sycophancy"]["h1"]
    assert "sycophancy_gain_vs_prompttoken" in syc
    assert corr["h1_verdict"]["headline_arm"] == "mapped_dp_context"

    args_4b = _args(_fixture_argv(root, ["--no-figures", "--n-boot", "80"]))
    assert ana.run_4b(args_4b) == 0
    summ = json.loads((root / "out_4b" / "summary.json").read_text())
    # Missing-judged + collapsed cells are REPORTED, never silent.
    assert "ultrachat__sycophancy__prompttoken_dp__bottom" in summ["missing_judged_cells"]
    assert "ultrachat__sycophancy__mapped_dp_context__top_filtered" in summ["collapsed_cells"]
    cats = {(r["corpus"], r["trait"], r["method"]): r["category"] for r in summ["verdict_lattice"]}
    # Fixture design: probe beats exact; prompttoken below exact (lmsys/evil).
    assert cats[("lmsys", "evil", "probe_diff_context")] == "beats_exact"
    assert cats[("lmsys", "evil", "prompttoken_dp")] == "below_exact"
    # Incoherent-flagged cell surfaces in the per-corpus-trait report.
    assert (
        "ultrachat__evil__probe_diff_context__top"
        in summ["per_corpus_trait"]["ultrachat__evil"]["incoherent_cells"]
    )
    assert summ["base_rates"]["lmsys__evil"]["frac_trait_bearing_ge_1"] is not None
    per_ct = json.loads((root / "out_4b" / "contrasts_lmsys__evil.json").read_text())
    assert per_ct["contrasts"], "no contrasts computed"
    for c in per_ct["contrasts"]:
        if c["status"] == "ok":
            assert c["response_level"]["ci_lo"] <= c["response_level"]["ci_hi"]
            assert "qid_cluster" in c


# ── Inverted-CI through the REAL figure function to savefig (xerr/yerr gotcha) ───


def test_hero_figure_survives_inverted_ci(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    cells = {}
    for method, mean in (("exact_dp", 60.0), ("mapped_dp_context", 58.0)):
        cid = f"lmsys__evil__{method}__top"
        cells[cid] = {
            "cell_id": cid,
            "method": method,
            "tail": "top",
            "graded_mean": mean,
            "rate_gt50": 0.6,
            "coherence_mean": 80.0,
            "incoherent_flag": False,
            # Deliberately INVERTED quantile CI (lo > mean > hi): must render,
            # never raise ValueError (offsets clamped at the errorbar site).
            "graded_ci": {"mean": mean, "ci_lo": mean + 3.0, "ci_hi": mean - 3.0},
        }
    rc = "lmsys__evil__random__shared"
    cells[rc] = {
        "cell_id": rc,
        "method": "random",
        "tail": "shared",
        "graded_mean": 20.0,
        "rate_gt50": 0.1,
        "coherence_mean": 85.0,
        "incoherent_flag": False,
        "graded_ci": {"mean": 20.0, "ci_lo": 18.0, "ci_hi": 22.0},
    }
    hero = {
        "lmsys__evil": {
            "corpus": "lmsys",
            "trait": "evil",
            "methods": ["exact_dp", "mapped_dp_context"],
            "cells": cells,
            "base": {"graded_mean": 5.0},
        }
    }
    args = _args(
        [
            "--figures-dir",
            str(tmp_path / "figs"),
            "--trait-scores-dir",
            str(tmp_path / "ft"),  # empty: per-cell strip panel skips missing files
            "--scores-dir",
            str(tmp_path / "scores"),  # empty: histogram figure skips
            "--seed",
            "1",
        ]
    )
    ana._fig_4b(args, hero, {})
    pngs = sorted(p.name for p in (tmp_path / "figs").glob("*.png"))
    assert "i2224_4b_hero_lmsys.png" in pngs, pngs


def test_fig_4b_empty_hero_raises_clean(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    args = _args(["--figures-dir", str(tmp_path / "figs")])
    with pytest.raises(RuntimeError, match="no judged trait_scores"):
        ana._fig_4b(args, {}, {})

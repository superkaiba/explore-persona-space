#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (※, ρ, −, ×, —) in scientific docstrings + labels.
"""Issue #559 — VM analysis: own-response prior vs the #478 persona panel.

Joins the NEW base own-response prior (one value per held-out persona, from
``base_prior_own_persona_panel.json``) onto the committed #478/#531 panel
(``tidy_logit.parquet``, 2,800 run × persona aggregates) and runs:

* **Within-run ranking (primary, H1):** Spearman of each ranker vs
  ``margin_trained`` across 35 personas per run (80 runs) for
  {matched-slot ``margin_base`` (incumbent), ``min_dist`` (incumbent),
  ``prior_margin_own`` (NEW), z-stack (exploratory)}. Incumbent numbers must
  reproduce the committed ``transfer_478.json`` values exactly (same code
  path — drift is a bug, fail loud).
* **Paired parity tests:** per-run paired ρ differences with run-pair AND
  cell-axis bootstraps; outcome classified per the plan §13.1 lattice with
  the §13.2 dual-resampling conservative read.
* **Two-ingredient joint fit (H2):** ``margin_trained ~ α·z(prior) +
  β·z(min_dist)`` (+ interaction / + run-FE variants) with run-, persona- and
  cell-axis cluster bootstraps (z-scoring + FE re-estimated inside every
  resample; α's PRIMARY CI = persona-cluster per §13.8), LOPO/LORO CV,
  pre-registered collinearity gate (|r| > 0.6 → tercile-bucket median read +
  polynomial-residualization robustness, both always computed).
* **Change-DV fit (secondary):** same on ``dmargin``, narrated with the
  mechanical subtraction named (§13.6) + the ``margin_base``-augmented fit.
* **Registered sensitivity reads (§13.3 / §13.5):** per-persona truncation
  rates + truncation-excluded re-run of ranking/parity; per-persona IQR over
  the 20 questions; median-aggregated prior variant; split-half (prior from
  one question half, DV from the disjoint half); response-length correlates.
* **Diagnostics (§13.7):** prior ↔ persona-mean ``margin_base`` agreement,
  argmax composition, ``n_pre_marker_slots``, per-K / per-seed stratified
  ranking medians.

Cherry-picked modules: ``scripts/issue553_panel.py`` from ``issue-553`` @
``68f3e6b69`` (``load_i478_panel`` / ``step0_i478`` / ``aggregate_run_persona``
/ ``wider_ci`` / ``common_parser``); the ``_within_run_ranking`` body
(issue-553:scripts/issue553_transfer_478.py:178-216) and
``_paired_difference_block`` (issue-553:scripts/issue553_ranking_table.py:115)
are copied below with attribution comments and extended (new rankers,
cell-axis resampling, per-statistic RNG re-seeding per §13.4).

Outputs ``within_run_ranking.json`` + ``joint_fit.json`` to ``--out-dir`` and
figures to ``--fig-dir``. Smoke = the same script with reduced bootstrap reps
plus ``--allow-stub`` against the stub written by ``--write-stub``.

Usage::

    uv run python scripts/issue559_panel_analysis.py \\
        --out-dir eval_results/issue_559 --fig-dir figures/issue_559
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue539_residual_per_cohort as i539  # noqa: E402
import issue553_panel as p553  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

SCHEMA_VERSION = "issue559_base_prior_v1"
PARITY_BAND = 0.10  # registered paired median-difference threshold (plan §3 / §13.1)
COLLINEARITY_GATE_R = 0.6  # pre-registered Pearson gate (plan §4.B5)
INCUMBENT_TOL = 1e-9  # exact-reproduction tolerance for deterministic point estimates
RANKERS_PRETRAIN = ("prior_margin_own", "min_dist", "z_stack")  # pre-training computable
DV_COL = "margin_trained"


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = p553.common_parser(
        "Issue #559 — own-response prior vs the #478 persona panel (VM analysis)"
    )
    parser.set_defaults(out_dir=Path("eval_results/issue_559"), fig_dir=Path("figures/issue_559"))
    parser.add_argument(
        "--prior-json",
        type=Path,
        default=Path("eval_results/issue_559/base_prior_own_persona_panel.json"),
    )
    parser.add_argument(
        "--r-base-own",
        type=Path,
        default=Path("eval_results/issue_559/R_base_own.json"),
        help="own-response generations (length distributions); optional",
    )
    parser.add_argument(
        "--s0-json",
        type=Path,
        default=Path("eval_results/issue_559/s0_validation.json"),
        help="S0 gate output (trained-R length sample); optional",
    )
    parser.add_argument(
        "--transfer-json",
        type=Path,
        default=Path("eval_results/issue_553/transfer_478.json"),
        help="committed incumbents for the exact-reproduction assert",
    )
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="accept an is_stub prior JSON (SMOKE ONLY — production refuses stubs)",
    )
    parser.add_argument(
        "--write-stub",
        action="store_true",
        help="write stub prior/R/s0 JSONs (full 35×20 shape, synthetic values) "
        "into --out-dir and exit — the VM smoke input generator",
    )
    return parser.parse_args(argv)


# ── Stub generator (VM smoke input; assert-guarded against production use) ────


def write_stub(args: argparse.Namespace) -> None:
    """Write clearly-labeled stub inputs with the production schema + shape.

    Synthetic per-question margins are seeded off the parquet's persona-mean
    ``margin_base`` so the join, ranking, collinearity gate and figures all
    exercise realistic correlation structure. Every payload carries
    ``is_stub: true`` and the analysis refuses it without ``--allow-stub``.
    """
    df = p553.load_i478_panel(args.i478_parquet)
    personas = sorted(df["held_out_persona"].unique().tolist())
    questions = [f"stub question {i}" for i in range(20)]
    base_by_p = df.groupby("held_out_persona")["margin_base"].mean()
    rng = np.random.default_rng(559)

    per_persona: dict[str, dict] = {}
    R: dict[str, dict[str, str]] = {}
    finish: dict[str, dict[str, str]] = {}
    lens: dict[str, dict[str, int]] = {}
    n_trunc = 0
    for p in personas:
        margins = float(base_by_p[p]) + rng.normal(0.0, 1.0, size=20)
        z_eos = 10.0 + rng.normal(0.0, 0.5, size=20)
        z_marker = z_eos + margins
        logZ = z_eos + 3.0
        rec = {
            "z_marker_per_q": z_marker.tolist(),
            "z_eos_per_q": z_eos.tolist(),
            "logZ_per_q": logZ.tolist(),
            "logp_marker_per_q": (z_marker - logZ).tolist(),
            "argmax_id_per_q": [151645] * 20,
            "slot_kind_per_q": ["end_of_response"] * 20,
            "n_truncated_tokens_per_q": [0] * 20,
            "finish_reason_per_q": ["length" if rng.random() < 0.05 else "stop" for _ in range(20)],
            "prior_margin_own": float(np.mean(margins)),
            "prior_margin_own_median": float(np.median(margins)),
            "prior_margin_own_iqr": [
                float(np.percentile(margins, 25)),
                float(np.percentile(margins, 75)),
            ],
            "prior_logp_own": float(np.mean(z_marker - logZ)),
        }
        n_trunc += sum(1 for f in rec["finish_reason_per_q"] if f == "length")
        per_persona[p] = rec
        R[p] = {q: f"stub response for {p}" for q in questions}
        finish[p] = dict(zip(questions, rec["finish_reason_per_q"], strict=True))
        lens[p] = {q: int(rng.integers(50, 300)) for q in questions}

    n_slots = len(personas) * 20
    prior_payload = {
        "schema_version": SCHEMA_VERSION,
        "is_stub": True,
        "eval_questions": questions,
        "personas": personas,
        "per_persona": per_persona,
        "summary": {
            "n_personas": len(personas),
            "n_questions": 20,
            "n_slots": n_slots,
            "n_pre_marker_slots": 0,
            "truncation_rate": n_trunc / n_slots,
            "argmax_composition": {
                "marker": {"count": 0, "rate": 0.0},
                "eos": {"count": n_slots, "rate": 1.0},
                "other": {"count": 0, "rate": 0.0},
            },
        },
        "s0_validation_pass": True,
        "metadata": {"note": "STUB — VM smoke input, never production data"},
    }
    r_payload = {
        "schema_version": SCHEMA_VERSION,
        "is_stub": True,
        "eval_questions": questions,
        "personas": personas,
        "R": R,
        "finish_reasons": finish,
        "own_R_token_lens": lens,
        "truncation_rate": n_trunc / n_slots,
        "metadata": {"note": "STUB — VM smoke input, never production data"},
    }
    s0_payload = {
        "schema_version": SCHEMA_VERSION,
        "is_stub": True,
        "s0_cell": "K1_c00_seed42",
        "gates": {"pass": True},
        "trained_R_token_lens": {
            p: {q: int(rng.integers(80, 400)) for q in questions} for p in personas
        },
        "metadata": {"note": "STUB — VM smoke input, never production data"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in [
        ("base_prior_own_persona_panel.json", prior_payload),
        ("R_base_own.json", r_payload),
        ("s0_validation.json", s0_payload),
    ]:
        (args.out_dir / name).write_text(json.dumps(obj))
        print(f"[stub] wrote {args.out_dir / name}")


# ── Prior loading + join ──────────────────────────────────────────────────────


def load_prior(args: argparse.Namespace) -> dict:
    """Load + guard the prior JSON; return persona-level frames + per-q arrays."""
    payload = json.loads(args.prior_json.read_text())
    assert payload.get("schema_version") == SCHEMA_VERSION, payload.get("schema_version")
    if payload.get("is_stub", False) and not args.allow_stub:
        raise SystemExit(
            f"{args.prior_json} is a STUB (smoke input) — refusing without --allow-stub"
        )
    if not payload.get("s0_validation_pass", False):
        raise SystemExit("prior JSON records s0_validation_pass=false — measurement invalid")
    return payload


def prior_frames(payload: dict) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """(persona-level prior table, per-persona per-q margin arrays)."""
    rows = []
    per_q: dict[str, np.ndarray] = {}
    for p, rec in payload["per_persona"].items():
        margins = np.asarray(rec["z_marker_per_q"]) - np.asarray(rec["z_eos_per_q"])
        per_q[p] = margins
        kept = [
            m for m, f in zip(margins, rec["finish_reason_per_q"], strict=True) if f != "length"
        ]
        rows.append(
            {
                "held_out_persona": p,
                "prior_margin_own": rec["prior_margin_own"],
                "prior_margin_own_median": rec["prior_margin_own_median"],
                "prior_margin_own_trunc_excl": float(np.mean(kept)) if kept else float("nan"),
                "n_q_kept_trunc_excl": len(kept),
                "prior_iqr_low": rec["prior_margin_own_iqr"][0],
                "prior_iqr_high": rec["prior_margin_own_iqr"][1],
                "truncation_rate_persona": float(
                    np.mean([f == "length" for f in rec["finish_reason_per_q"]])
                ),
                "prior_half_even": float(np.mean(margins[0::2])),
                "prior_half_odd": float(np.mean(margins[1::2])),
            }
        )
    return pd.DataFrame(rows), per_q


# ── Within-run ranking ────────────────────────────────────────────────────────
# Copied from issue-553:scripts/issue553_transfer_478.py:178-216
# (_within_run_ranking) with attribution, extended: arbitrary ranker columns,
# per-ranker RNG re-seeding (already the parent pattern — kept per §13.4 so a
# new ranker cannot perturb a shared stream), and a 40-cell-axis bootstrap on
# the median (§13.2 dual resampling axes).


def _per_run_rhos(
    agg: pd.DataFrame, ranker: str, dv_col: str
) -> tuple[list[tuple[str, float]], int]:
    """(kept (run, ρ) pairs, n_degenerate_dropped) across all runs."""
    runs = sorted(agg["run_id"].unique().tolist())
    kept: list[tuple[str, float]] = []
    n_dropped = 0
    for r in runs:
        sub = agg[agg["run_id"] == r]
        rho = i539._spearman_rho(
            sub[ranker].to_numpy(dtype=np.float64),
            sub[dv_col].to_numpy(dtype=np.float64),
        )
        if np.isnan(rho):
            n_dropped += 1
            continue
        kept.append((str(r), float(rho)))
    return kept, n_dropped


def _median_run_boot(arr: np.ndarray, args) -> dict:
    """80-run bootstrap CI on the median (parent-convention primary)."""
    rng = np.random.default_rng(args.seed)  # fresh per statistic (§13.4)
    med_boot = [
        float(np.median(arr[rng.integers(0, len(arr), size=len(arr))]))
        for _ in range(args.n_marginal_boot)
    ]
    return {
        "low": float(np.percentile(med_boot, 2.5)),
        "high": float(np.percentile(med_boot, 97.5)),
        "n_boot": args.n_marginal_boot,
    }


def _median_cell_boot(per_run: dict[str, float], cell_of_run: dict[str, str], args) -> dict:
    """40-cell-clustered bootstrap on the median: drawn cells bring BOTH seed-runs."""
    rng = np.random.default_rng(args.seed)  # fresh per statistic (§13.4)
    cells = sorted({cell_of_run[r] for r in per_run})
    vals_of_cell = {
        c: np.array([v for r, v in per_run.items() if cell_of_run[r] == c]) for c in cells
    }
    med_boot = []
    for _ in range(args.n_marginal_boot):
        chosen = rng.choice(cells, size=len(cells), replace=True)
        med_boot.append(float(np.median(np.concatenate([vals_of_cell[c] for c in chosen]))))
    return {
        "low": float(np.percentile(med_boot, 2.5)),
        "high": float(np.percentile(med_boot, 97.5)),
        "n_boot": args.n_marginal_boot,
        "n_clusters": len(cells),
    }


def within_run_ranking(
    agg: pd.DataFrame,
    rankers: list[str],
    dv_col: str,
    cell_of_run: dict[str, str],
    args,
) -> dict:
    """Per-run Spearman across personas of each ranker vs the DV (+ dual-axis CIs)."""
    out: dict = {}
    n_runs = agg["run_id"].nunique()
    for ranker in rankers:
        kept, n_dropped = _per_run_rhos(agg, ranker, dv_col)
        arr = np.asarray([v for _, v in kept])
        out[ranker] = {
            "n_runs": n_runs,
            "n_degenerate_dropped": n_dropped,
            "median_rho": float(np.median(arr)),
            "iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
            "median_ci95_run_boot": _median_run_boot(arr, args),
            "median_ci95_cell_boot": _median_cell_boot(dict(kept), cell_of_run, args),
            "per_run_rho": dict(kept),
        }
    return out


# ── Paired parity tests ───────────────────────────────────────────────────────
# Copied from issue-553:scripts/issue553_ranking_table.py:115
# (_paired_difference_block) with attribution; pairs = runs here (the parent
# used sources), plus a cell-axis resampling variant (§13.2).


def paired_difference_block(rho_a: dict[str, float], rho_b: dict[str, float], args) -> dict:
    """Run-pair bootstrap CI for the per-run ρ difference (a − b)."""
    pairs = [(rho_a[s], rho_b[s]) for s in rho_a if not (np.isnan(rho_a[s]) or np.isnan(rho_b[s]))]
    n_dropped = len(rho_a) - len(pairs)
    diffs = np.array([a - b for a, b in pairs])
    rng = np.random.default_rng(args.seed)  # fresh per statistic (§13.4)
    med_boot, mean_boot = [], []
    for _ in range(args.n_marginal_boot):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        med_boot.append(float(np.median(diffs[idx])))
        mean_boot.append(float(np.mean(diffs[idx])))
    return {
        "n_paired_runs": len(pairs),
        "n_pairs_dropped_nan": n_dropped,
        "median_difference": float(np.median(diffs)),
        "mean_difference": float(np.mean(diffs)),
        "median_diff_ci95": {
            "low": float(np.percentile(med_boot, 2.5)),
            "high": float(np.percentile(med_boot, 97.5)),
        },
        "mean_diff_ci95": {
            "low": float(np.percentile(mean_boot, 2.5)),
            "high": float(np.percentile(mean_boot, 97.5)),
        },
        "n_boot": args.n_marginal_boot,
        "method": "paired per-run difference, run-pair bootstrap "
        "(issue553_ranking_table.py:115 recipe with pairs = runs)",
    }


def paired_difference_cellaxis(
    rho_a: dict[str, float], rho_b: dict[str, float], cell_of_run: dict[str, str], args
) -> dict:
    """Cell-axis variant: resample 40 cells; each brings both runs' paired diffs."""
    diffs_of_run = {
        r: rho_a[r] - rho_b[r] for r in rho_a if not (np.isnan(rho_a[r]) or np.isnan(rho_b[r]))
    }
    cells = sorted({cell_of_run[r] for r in diffs_of_run})
    vals_of_cell = {
        c: np.array([d for r, d in diffs_of_run.items() if cell_of_run[r] == c]) for c in cells
    }
    rng = np.random.default_rng(args.seed)  # fresh per statistic (§13.4)
    med_boot, mean_boot = [], []
    for _ in range(args.n_marginal_boot):
        chosen = rng.choice(cells, size=len(cells), replace=True)
        d = np.concatenate([vals_of_cell[c] for c in chosen])
        med_boot.append(float(np.median(d)))
        mean_boot.append(float(np.mean(d)))
    return {
        "median_diff_ci95": {
            "low": float(np.percentile(med_boot, 2.5)),
            "high": float(np.percentile(med_boot, 97.5)),
        },
        "mean_diff_ci95": {
            "low": float(np.percentile(mean_boot, 2.5)),
            "high": float(np.percentile(mean_boot, 97.5)),
        },
        "n_boot": args.n_marginal_boot,
        "n_clusters": len(cells),
    }


def classify_outcome(prior_blk: dict, parity_run: dict, parity_cell: dict) -> dict:
    """Plan §13.1 outcome lattice under the §13.2 dual-axis conservative read.

    A positive boundary claim ("CI entirely > 0", "CI entirely below +0.10")
    must hold on BOTH resampling axes; any axis disagreement takes the
    conservative branch.
    """
    run_ci = prior_blk["median_ci95_run_boot"]
    cell_ci = prior_blk["median_ci95_cell_boot"]
    prior_pos = run_ci["low"] > 0 and cell_ci["low"] > 0
    prior_neg = run_ci["high"] < 0 and cell_ci["high"] < 0
    pr, pc = parity_run["median_diff_ci95"], parity_cell["median_diff_ci95"]
    parity_below = pr["high"] < PARITY_BAND and pc["high"] < PARITY_BAND
    parity_above = pr["low"] > PARITY_BAND and pc["low"] > PARITY_BAND

    if prior_neg:
        cls = "FALSIFIED"
        read = (
            "prior median ρ CI entirely < 0 on both resampling axes — classified "
            "FALSIFIED regardless of the paired-difference cell (§13.1)"
        )
    elif not prior_pos:
        cls = "FALSIFIED"
        read = (
            "prior median ρ CI spans 0 on at least one resampling axis (conservative "
            "read governs) — the context-panel leaderboard win is panel-specific"
        )
    elif parity_below:
        cls = "CONFIRMED"
        read = (
            f"prior CI entirely > 0 AND paired (matched − prior) median-difference CI "
            f"entirely below +{PARITY_BAND} on both axes — ranks nearly as well"
        )
    elif parity_above:
        cls = "PARTIAL"
        read = (
            f"prior carries real signal but the matched-slot read is decisively better "
            f"(paired diff CI entirely above +{PARITY_BAND} on both axes)"
        )
    else:
        cls = "PRIOR_SIGNAL_PARITY_INDETERMINATE"
        read = (
            f"prior carries real signal; parity indeterminate (paired diff CI "
            f"straddles +{PARITY_BAND} or the axes disagree) — NEVER ship as parity "
            f"confirmed (§13.1)"
        )
    return {
        "classification": cls,
        "read": read,
        "parity_band": PARITY_BAND,
        "prior_ci_run": run_ci,
        "prior_ci_cell": cell_ci,
        "paired_matched_minus_prior_ci_run": pr,
        "paired_matched_minus_prior_ci_cell": pc,
    }


# ── Incumbent exact-reproduction assert (§13.4) ───────────────────────────────


def assert_incumbents_reproduce(ranking: dict, args) -> dict:
    """Pin incumbents to the committed transfer_478.json deterministic estimates.

    Asserts median_rho + the full per_run_rho map to 1e-9 (deterministic given
    the panel — a drift is a bug, fail loud). Bootstrap CIs are compared
    within tolerance and REPORTED (each statistic re-seeds its own RNG, so at
    the production seed/reps they reproduce exactly; reduced-rep smoke runs
    only report).
    """
    committed = json.loads(args.transfer_json.read_text())["within_run_ranking"]
    report: dict = {}
    for ranker in ("margin_base", "min_dist"):
        got, want = ranking[ranker], committed[ranker]
        d_med = abs(got["median_rho"] - want["median_rho"])
        assert d_med <= INCUMBENT_TOL, (
            f"incumbent {ranker} median_rho drifted: got {got['median_rho']!r}, "
            f"committed {want['median_rho']!r} — same code path, this is a bug"
        )
        assert set(got["per_run_rho"]) == set(want["per_run_rho"]), ranker
        worst = max(
            abs(got["per_run_rho"][r] - want["per_run_rho"][r]) for r in want["per_run_rho"]
        )
        assert worst <= INCUMBENT_TOL, f"incumbent {ranker} per-run ρ drifted (max {worst})"
        assert got["n_degenerate_dropped"] == want["n_degenerate_dropped"], ranker
        ci_got, ci_want = got["median_ci95_run_boot"], want["median_ci95_run_boot"]
        ci_dev = max(abs(ci_got["low"] - ci_want["low"]), abs(ci_got["high"] - ci_want["high"]))
        report[ranker] = {
            "median_rho_abs_diff": d_med,
            "per_run_rho_max_abs_diff": worst,
            "ci_endpoint_max_abs_diff": ci_dev,
            "ci_exactly_reproduced": bool(ci_dev <= INCUMBENT_TOL),
            "note": "point estimates asserted to 1e-9; CI endpoints reported "
            "(exact at production seed/reps — per-statistic RNG re-seeding §13.4)",
        }
        print(
            f"[incumbent] {ranker}: median Δ={d_med:.2e}, per-run max Δ={worst:.2e}, "
            f"CI endpoint Δ={ci_dev:.2e}"
        )
    return report


# ── Joint fit (H2) ───────────────────────────────────────────────────────────


def _z(v: np.ndarray) -> np.ndarray:
    sd = float(np.std(v))
    assert sd > 1e-12, "degenerate feature in z-scoring"
    return (v - float(np.mean(v))) / sd


def _fit_ols(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 1e-18 else float("nan")
    return coef, r2


def _joint_design(
    prior: np.ndarray,
    mind: np.ndarray,
    fe_codes: np.ndarray | None,
    n_fe: int,
    interaction: bool,
) -> tuple[np.ndarray, list[str]]:
    """Design matrix with within-sample z-scoring (re-estimated per resample)."""
    zp, zd = _z(prior), _z(mind)
    cols: list[np.ndarray] = [zp, zd]
    names = ["alpha_prior", "beta_min_dist"]
    if interaction:
        cols.append(zp * zd)
        names.append("gamma_interaction")
    if fe_codes is not None:
        fe = np.zeros((len(zp), n_fe))
        fe[np.arange(len(zp)), fe_codes] = 1.0
        design = np.column_stack([*cols, fe])  # FE block absorbs the intercept
    else:
        design = np.column_stack([np.ones(len(zp)), *cols])
        names = ["intercept", *names]
    return design, names


def _variant_fit(
    agg: pd.DataFrame, dv: str, variant: str, extra_cols: list[str] | None = None
) -> tuple[dict, list[str]]:
    """Observed-data fit for one variant; returns ({name: coef}, names)."""
    # Extras append AFTER the FE block, so name/coef alignment below is only
    # valid without FE — guard the unused combination explicitly.
    assert not (extra_cols and variant == "run_fe"), "extras + run-FE not supported"
    prior = agg["prior_margin_own"].to_numpy(np.float64)
    mind = agg["min_dist"].to_numpy(np.float64)
    y = agg[dv].to_numpy(np.float64)
    interaction = variant == "interaction"
    if variant == "run_fe":
        _, codes = np.unique(agg["run_id"].to_numpy(), return_inverse=True)
        design, names = _joint_design(prior, mind, codes, int(codes.max()) + 1, False)
    else:
        design, names = _joint_design(prior, mind, None, 0, interaction)
    if extra_cols:
        for c in extra_cols:
            design = np.column_stack([design, _z(agg[c].to_numpy(np.float64))])
            names = [*names, f"coef_{c}"]
    coef, r2 = _fit_ols(design, y)
    out = {n: float(c) for n, c in zip(names, coef[: len(names)], strict=False)}
    out["r2"] = r2
    return out, names


def _coef_cluster_boot(
    agg: pd.DataFrame,
    dv: str,
    variant: str,
    axis: str,
    args,
    extra_cols: list[str] | None = None,
) -> dict[str, dict]:
    """Cluster percentile bootstrap of the joint-fit coefficients on one axis.

    z-scoring + FE re-estimated inside every resample (plan §4.B5). Axes:
    ``run`` (80 clusters, FE codes = fresh copy codes), ``persona`` (35
    clusters, FE codes = original run codes within the resample), ``cell``
    (40 clusters, drawn copies bring BOTH seed-runs, run codes relabeled per
    copy — the #553 reconciler convention).
    """
    assert not (extra_cols and variant == "run_fe"), "extras + run-FE not supported"
    prior = agg["prior_margin_own"].to_numpy(np.float64)
    mind = agg["min_dist"].to_numpy(np.float64)
    y = agg[dv].to_numpy(np.float64)
    extras = [agg[c].to_numpy(np.float64) for c in (extra_cols or [])]
    run_lab = agg["run_id"].to_numpy()
    _, run_code_full = np.unique(run_lab, return_inverse=True)
    interaction = variant == "interaction"
    use_fe = variant == "run_fe"

    if axis == "run":
        labels = run_lab
    elif axis == "persona":
        labels = agg["held_out_persona"].to_numpy()
    elif axis == "cell":
        labels = agg["cell_id"].to_numpy()
    else:
        raise ValueError(axis)

    rng = np.random.default_rng(args.seed)  # fresh per (variant, axis) statistic (§13.4)
    uniq = np.unique(labels)
    idx_of = {c: np.where(labels == c)[0] for c in uniq}
    coefs: list[np.ndarray] = []
    names: list[str] = []
    n_deg = 0
    for _ in range(args.n_cluster_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[c] for c in chosen])
        copy_codes = np.repeat(np.arange(len(chosen)), [len(idx_of[c]) for c in chosen])
        if float(np.std(prior[idx])) < 1e-12 or float(np.std(mind[idx])) < 1e-12:
            n_deg += 1
            continue
        if use_fe:
            if axis == "run":
                fe_codes, n_fe = copy_codes, len(chosen)
            elif axis == "persona":
                _, fe_codes = np.unique(run_code_full[idx], return_inverse=True)
                n_fe = int(fe_codes.max()) + 1
            else:  # cell: relabel the 2 runs of each drawn cell copy as fresh groups
                _, fe_codes = np.unique(
                    copy_codes * (int(run_code_full.max()) + 1) + run_code_full[idx],
                    return_inverse=True,
                )
                n_fe = int(fe_codes.max()) + 1
            design, names = _joint_design(prior[idx], mind[idx], fe_codes, n_fe, False)
        else:
            design, names = _joint_design(prior[idx], mind[idx], None, 0, interaction)
        for e in extras:
            design = np.column_stack([design, _z(e[idx])])
        coef, *_ = np.linalg.lstsq(design, y[idx], rcond=None)
        coefs.append(coef[: len(names) + len(extras)])
    if extras:
        names = [*names, *[f"coef_{c}" for c in (extra_cols or [])]]
    arr = np.asarray(coefs)
    out: dict[str, dict] = {}
    for j, n in enumerate(names):
        if n == "intercept":
            continue
        out[n] = {
            "low": float(np.percentile(arr[:, j], 2.5)),
            "high": float(np.percentile(arr[:, j], 97.5)),
            "boot_mean": float(np.mean(arr[:, j])),
            "n_boot": args.n_cluster_boot,
            "n_degenerate_resamples": n_deg,
            "n_clusters": len(uniq),
        }
    return out


def _cv_oof_r2(agg: pd.DataFrame, dv: str, fold_col: str, feature_sets: dict) -> dict:
    """Out-of-fold R² per feature set (z-params + OLS fit on the train fold)."""
    y = agg[dv].to_numpy(np.float64)
    out: dict = {}
    folds = sorted(agg[fold_col].unique().tolist())
    for name, cols in feature_sets.items():
        preds = np.full(len(agg), np.nan)
        for f in folds:
            test = (agg[fold_col] == f).to_numpy()
            train = ~test
            feats_tr, feats_te = [], []
            stats = []
            for c in cols:
                v = agg[c].to_numpy(np.float64)
                mu, sd = float(np.mean(v[train])), float(np.std(v[train]))
                stats.append((v, mu, max(sd, 1e-12)))
            for v, mu, sd in stats:
                feats_tr.append((v[train] - mu) / sd)
                feats_te.append((v[test] - mu) / sd)
            if "interaction" in name:
                feats_tr.append(feats_tr[0] * feats_tr[1])
                feats_te.append(feats_te[0] * feats_te[1])
            X_tr = np.column_stack([np.ones(train.sum()), *feats_tr])
            X_te = np.column_stack([np.ones(test.sum()), *feats_te])
            coef, *_ = np.linalg.lstsq(X_tr, y[train], rcond=None)
            preds[test] = X_te @ coef
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out[name] = {"oof_r2": 1.0 - ss_res / ss_tot, "n_folds": len(folds)}
    return out


def tercile_table(agg: pd.DataFrame, dv: str) -> dict:
    """3×3 median-DV table over (prior tercile × min_dist tercile) + counts."""
    pt = pd.qcut(agg["prior_margin_own"], 3, labels=["low", "mid", "high"])
    dt = pd.qcut(agg["min_dist"], 3, labels=["low", "mid", "high"])
    table: dict = {}
    for p_lab in ("low", "mid", "high"):
        table[p_lab] = {}
        for d_lab in ("low", "mid", "high"):
            m = (pt == p_lab) & (dt == d_lab)
            table[p_lab][d_lab] = {
                "median_dv": float(agg.loc[m, dv].median()) if m.any() else float("nan"),
                "n": int(m.sum()),
            }
    return table


def poly_residualization(agg: pd.DataFrame, dv: str, args) -> dict:
    """Prior residualized on poly2(min_dist), refit — collinearity robustness."""
    prior = agg["prior_margin_own"].to_numpy(np.float64)
    mind = agg["min_dist"].to_numpy(np.float64)
    y = agg[dv].to_numpy(np.float64)
    X = np.column_stack([np.ones(len(mind)), mind, mind**2])
    coef, *_ = np.linalg.lstsq(X, prior, rcond=None)
    resid = prior - X @ coef
    design = np.column_stack([np.ones(len(y)), _z(resid), _z(mind)])
    fit_coef, r2 = _fit_ols(design, y)
    # Persona-cluster CI for the residualized-prior coefficient (α's primary axis).
    tmp = agg.copy()
    tmp["prior_margin_own"] = resid
    ci = _coef_cluster_boot(tmp, dv, "base", "persona", args)["alpha_prior"]
    return {
        "alpha_resid_prior": float(fit_coef[1]),
        "beta_min_dist": float(fit_coef[2]),
        "r2": r2,
        "alpha_resid_prior_ci95_persona_cluster": ci,
        "method": "prior residualized on [1, min_dist, min_dist^2], then "
        "dv ~ z(resid) + z(min_dist)",
    }


def joint_fit_block(agg: pd.DataFrame, dv: str, args, extra_cols: list[str] | None = None) -> dict:
    """Observed fit + 3-axis cluster CIs + primary-CI designation per §13.8."""
    out: dict = {"dv": dv, "variants": {}}
    variants = ["base", "interaction", "run_fe"]
    for variant in variants:
        obs, _ = _variant_fit(agg, dv, variant, extra_cols)
        cis = {
            axis: _coef_cluster_boot(agg, dv, variant, axis, args, extra_cols)
            for axis in ("run", "persona", "cell")
        }
        per_coef: dict = {}
        for name in cis["run"]:
            entry = {
                "estimate": obs.get(name),
                "ci95_cluster_run": cis["run"][name],
                "ci95_cluster_persona": cis["persona"][name],
                "ci95_cluster_cellaxis": cis["cell"][name],
            }
            if name == "alpha_prior":
                # §13.8: α's effective N is ~35 personas — persona-cluster CI is
                # PRIMARY for α; never the marginal or run-cluster CI.
                entry["primary_ci"] = {"axis": "cluster_persona", **cis["persona"][name]}
            else:
                entry["primary_ci"] = p553.wider_ci(
                    {
                        "cluster_run": cis["run"][name],
                        "cluster_persona": cis["persona"][name],
                        "cluster_cellaxis": cis["cell"][name],
                    }
                )
            per_coef[name] = entry
        out["variants"][variant] = {"r2": obs["r2"], "coefficients": per_coef}
        if variant == "run_fe":
            out["variants"][variant]["note"] = (
                "within-run-identified variant; a persona-FE variant is impossible "
                "by construction — the prior is constant per persona and exactly "
                "collinear with persona FE"
            )
    return out


# ── Figures ───────────────────────────────────────────────────────────────────


def _clamp_err(lo: float, hi: float, mid: float) -> tuple[float, float]:
    """Non-negative errorbar halves (constant-bootstrap guard)."""
    return max(0.0, mid - lo), max(0.0, hi - mid)


def figure_ranking_strip(ranking: dict, fig_dir: Path) -> None:
    """Hero: per-run ρ strip + median bar per ranker; blue = pre-training computable."""
    set_paper_style("blog")
    colors = paper_palette(2)
    order = ["margin_base", "prior_margin_own", "min_dist", "z_stack"]
    labels = [
        "base matched-slot\nmargin (needs trained R)",
        "own-response\nprior (NEW)",
        "distance to\nnearest source",
        "z(prior) +\nz(min_dist)",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rng = np.random.default_rng(0)
    for xi, ranker in enumerate(order):
        blk = ranking[ranker]
        vals = [v for v in blk["per_run_rho"].values() if not np.isnan(v)]
        color = colors[1] if ranker == "margin_base" else colors[0]
        jitter = (rng.random(len(vals)) - 0.5) * 0.18
        ax.plot(np.full(len(vals), xi) + jitter, vals, "o", ms=3.5, alpha=0.55, color=color)
        med = blk["median_rho"]
        ax.plot([xi - 0.22, xi + 0.22], [med, med], color=color, lw=2.4)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Per-run Spearman ρ vs trained EOS margin (35 personas)")
    ax.set_title(
        "Within-run ranking on the 35-persona held-out panel (80 runs)\n"
        "orange = needs the trained model's responses; blue = pre-training computable",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, "within_run_ranking_strip", dir=fig_dir)
    plt.close(fig)


def figure_persona_scatters(agg: pd.DataFrame, fig_dir: Path) -> None:
    """Prior vs persona-mean DV (raw + run-FE-residualized) and vs margin_base."""
    set_paper_style("blog")
    colors = paper_palette(3)
    per = agg.groupby("held_out_persona").agg(
        prior=("prior_margin_own", "first"),
        mt=("margin_trained", "mean"),
        mb=("margin_base", "mean"),
    )
    # Run-FE residual = group-demeaned margin_trained, then persona means.
    resid = agg["margin_trained"] - agg.groupby("run_id")["margin_trained"].transform("mean")
    per["mt_fe"] = resid.groupby(agg["held_out_persona"]).mean()

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    panels = [
        ("mt", "persona-mean trained EOS margin (raw)", colors[0]),
        ("mt_fe", "persona-mean trained EOS margin (run-FE residual)", colors[1]),
        ("mb", "persona-mean base matched-slot margin", colors[2]),
    ]
    for ax, (col, label, color) in zip(axes, panels, strict=True):
        ax.plot(per["prior"], per[col], "o", ms=4.5, alpha=0.75, color=color)
        rho = i539._spearman_rho(per["prior"].to_numpy(), per[col].to_numpy())
        ax.set_title(f"ρ = {rho:+.2f} (35 personas)", fontsize=9)
        ax.set_xlabel("own-response prior margin")
        ax.set_ylabel(label, fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, "prior_persona_scatters", dir=fig_dir)
    plt.close(fig)


def figure_collinearity(agg: pd.DataFrame, fig_dir: Path) -> None:
    """Prior vs min_dist over the unique (cell, persona) pairs."""
    set_paper_style("blog")
    colors = paper_palette(1)
    uniq = agg.drop_duplicates(["cell_id", "held_out_persona"])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(uniq["min_dist"], uniq["prior_margin_own"], "o", ms=3, alpha=0.45, color=colors[0])
    r = float(np.corrcoef(agg["prior_margin_own"], agg["min_dist"])[0, 1])
    ax.set_title(f"Collinearity view — Pearson r = {r:+.3f} (2,800 aggregates)", fontsize=9)
    ax.set_xlabel("distance to nearest source (min_dist)")
    ax.set_ylabel("own-response prior margin")
    fig.tight_layout()
    savefig_paper(fig, "prior_vs_min_dist", dir=fig_dir)
    plt.close(fig)


def figure_forest(level: dict, change: dict, fig_dir: Path) -> None:
    """Coefficient forest (level + change DVs, base variant, primary CIs)."""
    set_paper_style("blog")
    colors = paper_palette(2)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rows = []
    for blk, dv_label, color in [
        (level, "level (margin_trained)", colors[0]),
        (change, "change (dmargin)", colors[1]),
    ]:
        for cname, clabel in [("alpha_prior", "α prior"), ("beta_min_dist", "β min_dist")]:
            c = blk["variants"]["base"]["coefficients"][cname]
            rows.append((f"{clabel} — {dv_label}", c["estimate"], c["primary_ci"], color))
    ys = np.arange(len(rows))[::-1]
    for y, (_label, est, ci, color) in zip(ys, rows, strict=True):
        lo_e, hi_e = _clamp_err(ci["low"], ci["high"], est)
        ax.errorbar([est], [y], xerr=[[lo_e], [hi_e]], fmt="o", color=color, capsize=3)
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("standardized coefficient (primary cluster-bootstrap CI)")
    ax.set_title("Two-ingredient joint fit — level vs change DV", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "joint_fit_forest", dir=fig_dir)
    plt.close(fig)


def figure_stratified(strat: dict, fig_dir: Path, by: str) -> None:
    """Per-stratum median ρ per ranker (per-K / per-seed stratified reads)."""
    set_paper_style("blog")
    strata = sorted(strat.keys())
    rankers = ["margin_base", "prior_margin_own", "min_dist"]
    colors = paper_palette(len(rankers))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = np.arange(len(strata))
    width = 0.25
    for j, rk in enumerate(rankers):
        meds = [strat[s][rk]["median_rho"] for s in strata]
        ax.bar(xs + (j - 1) * width, meds, width, color=colors[j], label=rk)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in strata])
    ax.set_xlabel(by)
    ax.set_ylabel("median per-run ρ vs trained EOS margin")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, f"ranking_by_{by}", dir=fig_dir)
    plt.close(fig)


def figure_paired_hist(rho_a: dict, rho_b: dict, fig_dir: Path) -> None:
    """Histogram of per-run (matched-slot − prior) ρ differences."""
    set_paper_style("blog")
    colors = paper_palette(1)
    diffs = [rho_a[r] - rho_b[r] for r in rho_a if r in rho_b]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.hist(diffs, bins=20, color=colors[0], alpha=0.8)
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.axvline(PARITY_BAND, color="0.2", lw=1.0, ls="--")
    ax.set_xlabel("per-run ρ(matched-slot) − ρ(own-response prior)")
    ax.set_ylabel("runs")
    ax.set_title(f"Paired per-run difference (dashed = +{PARITY_BAND} parity band)", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "paired_diff_hist", dir=fig_dir)
    plt.close(fig)


def figure_lengths(
    own_lens: list[int], trained_lens: list[int], trunc_by_p: dict, fig_dir: Path
) -> None:
    """Own-R vs trained-R token-length distributions + per-persona truncation."""
    set_paper_style("blog")
    colors = paper_palette(2)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    axes[0].hist(own_lens, bins=30, alpha=0.6, color=colors[0], label="base own-R (700)")
    if trained_lens:
        axes[0].hist(
            trained_lens,
            bins=30,
            alpha=0.6,
            color=colors[1],
            label="trained R (K1_c00_seed42 sample)",
        )
    axes[0].set_xlabel("response length (tokens)")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=7)
    ps = sorted(trunc_by_p, key=trunc_by_p.get, reverse=True)
    axes[1].bar(range(len(ps)), [trunc_by_p[p] for p in ps], color=colors[0])
    axes[1].set_xticks(range(len(ps)))
    axes[1].set_xticklabels(ps, rotation=90, fontsize=5)
    axes[1].set_ylabel("truncation rate (per persona)")
    fig.tight_layout()
    savefig_paper(fig, "response_length_truncation", dir=fig_dir)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    if args.write_stub:
        write_stub(args)
        return 0

    payload = load_prior(args)
    prior_df, _per_q = prior_frames(payload)

    # Panel + step-0 gate (the #553 entry gate, unchanged).
    df = p553.load_i478_panel(args.i478_parquet)
    p553.step0_i478(df, args.i478_parquet.parent / "summary_logit.json")
    agg = p553.aggregate_run_persona(df)

    # Join the prior; assert complete coverage (35/35 personas).
    agg = agg.merge(prior_df, on="held_out_persona", how="left", validate="many_to_one")
    assert not agg["prior_margin_own"].isna().any(), "prior join produced NaN — persona mismatch"
    cell_of_run = dict(zip(agg["run_id"], agg["cell_id"], strict=True))

    # Exploratory z-stack, both ingredients oriented positively with the DV.
    s_p = float(
        np.sign(
            i539._spearman_rho(
                agg["prior_margin_own"].to_numpy(np.float64), agg[DV_COL].to_numpy(np.float64)
            )
        )
    )
    s_d = float(
        np.sign(
            i539._spearman_rho(
                agg["min_dist"].to_numpy(np.float64), agg[DV_COL].to_numpy(np.float64)
            )
        )
    )
    agg["z_stack"] = s_p * _z(agg["prior_margin_own"].to_numpy(np.float64)) + s_d * _z(
        agg["min_dist"].to_numpy(np.float64)
    )

    # ── Primary: within-run ranking + incumbent reproduction ────────────────
    print("[ranking] within-run ranking (4 rankers, 80 runs) ...")
    ranking = within_run_ranking(
        agg,
        ["margin_base", "min_dist", "prior_margin_own", "z_stack"],
        DV_COL,
        cell_of_run,
        args,
    )
    incumbent_report = assert_incumbents_reproduce(ranking, args)

    # ── Paired parity tests ──────────────────────────────────────────────────
    rho_mb = ranking["margin_base"]["per_run_rho"]
    rho_pr = ranking["prior_margin_own"]["per_run_rho"]
    rho_md = ranking["min_dist"]["per_run_rho"]
    parity_run = paired_difference_block(rho_mb, rho_pr, args)
    parity_cell = paired_difference_cellaxis(rho_mb, rho_pr, cell_of_run, args)
    prior_vs_mind_run = paired_difference_block(rho_pr, rho_md, args)
    prior_vs_mind_cell = paired_difference_cellaxis(rho_pr, rho_md, cell_of_run, args)
    outcome = classify_outcome(ranking["prior_margin_own"], parity_run, parity_cell)
    print(f"[outcome] {outcome['classification']}: {outcome['read']}")

    # ── §13.3 truncation sensitivity ─────────────────────────────────────────
    trunc_slice = agg[~agg["prior_margin_own_trunc_excl"].isna()].copy()
    n_personas_dropped_trunc = 35 - trunc_slice["held_out_persona"].nunique()
    rank_trunc = within_run_ranking(
        trunc_slice.rename(columns={"prior_margin_own_trunc_excl": "prior_trunc"}),
        ["margin_base", "prior_trunc"],
        DV_COL,
        cell_of_run,
        args,
    )
    parity_trunc = paired_difference_block(
        rank_trunc["margin_base"]["per_run_rho"], rank_trunc["prior_trunc"]["per_run_rho"], args
    )

    # ── §13.5 question-mix robustness ────────────────────────────────────────
    rank_median_agg = within_run_ranking(
        agg.rename(columns={"prior_margin_own_median": "prior_median_agg"}),
        ["prior_median_agg"],
        DV_COL,
        cell_of_run,
        args,
    )
    # Split-half: prior from one question half, DV aggregated over the other.
    dv_half = (
        df.assign(half=np.where(df["question_idx"] % 2 == 0, "even", "odd"))
        .groupby(["run_id", "held_out_persona", "half"], as_index=False)["margin_trained"]
        .mean()
        .pivot_table(index=["run_id", "held_out_persona"], columns="half", values="margin_trained")
        .reset_index()
        .rename(columns={"even": "dv_even", "odd": "dv_odd"})
    )
    half_agg = agg.merge(dv_half, on=["run_id", "held_out_persona"], validate="one_to_one")
    rank_half_a = within_run_ranking(half_agg, ["prior_half_even"], "dv_odd", cell_of_run, args)
    rank_half_b = within_run_ranking(half_agg, ["prior_half_odd"], "dv_even", cell_of_run, args)

    # ── Lengths + correlates (§13.3) ─────────────────────────────────────────
    own_lens_by_p: dict[str, float] = {}
    own_lens_flat: list[int] = []
    trained_lens_flat: list[int] = []
    if args.r_base_own.exists():
        r_own = json.loads(args.r_base_own.read_text())
        for p, qmap in r_own["own_R_token_lens"].items():
            vals = list(qmap.values())
            own_lens_by_p[p] = float(np.mean(vals))
            own_lens_flat.extend(int(v) for v in vals)
    if args.s0_json.exists():
        s0 = json.loads(args.s0_json.read_text())
        for qmap in s0.get("trained_R_token_lens", {}).values():
            trained_lens_flat.extend(int(v) for v in qmap.values())
    per_persona_view = agg.groupby("held_out_persona").agg(
        prior=("prior_margin_own", "first"),
        mt=("margin_trained", "mean"),
        mb=("margin_base", "mean"),
        trunc=("truncation_rate_persona", "first"),
    )
    length_correlates: dict = {"available": bool(own_lens_by_p)}
    if own_lens_by_p:
        lens_arr = np.array([own_lens_by_p.get(p, np.nan) for p in per_persona_view.index])
        length_correlates.update(
            {
                "spearman_len_vs_prior": i539._spearman_rho(
                    lens_arr, per_persona_view["prior"].to_numpy()
                ),
                "spearman_len_vs_margin_trained": i539._spearman_rho(
                    lens_arr, per_persona_view["mt"].to_numpy()
                ),
                "own_R_len_median": float(np.median(own_lens_flat)),
                "own_R_len_p90": float(np.percentile(own_lens_flat, 90)),
                "trained_R_len_median_sample": (
                    float(np.median(trained_lens_flat)) if trained_lens_flat else None
                ),
            }
        )

    # ── Diagnostics (§13.7 framing guard + plan §4.B7) ──────────────────────
    diag = {
        "spearman_prior_vs_persona_mean_margin_base": i539._spearman_rho(
            per_persona_view["prior"].to_numpy(), per_persona_view["mb"].to_numpy()
        ),
        "spearman_prior_vs_persona_mean_margin_trained": i539._spearman_rho(
            per_persona_view["prior"].to_numpy(), per_persona_view["mt"].to_numpy()
        ),
        "n_pre_marker_slots": payload["summary"]["n_pre_marker_slots"],
        "argmax_composition": payload["summary"]["argmax_composition"],
        "truncation_rate_global": payload["summary"]["truncation_rate"],
        "truncation_rate_per_persona": dict(
            zip(per_persona_view.index, per_persona_view["trunc"], strict=True)
        ),
        "per_persona_iqr_over_questions": {
            p: [
                float(prior_df.set_index("held_out_persona").loc[p, "prior_iqr_low"]),
                float(prior_df.set_index("held_out_persona").loc[p, "prior_iqr_high"]),
            ]
            for p in prior_df["held_out_persona"]
        },
        "framing_guard_note": (
            "headline is OPERATIONAL (pre-training computability of the matched-slot "
            "level signal); a mechanistic 'the prior drives leakage' reading is "
            "banned (§13.7). If prior↔margin_base agreement is ρ≈1, frame as 'the "
            "matched-slot signal is pre-training computable'."
        ),
    }

    # Per-K / per-seed stratified ranking medians.
    strat_k: dict = {}
    for k, sub in agg.groupby("K"):
        strat_k[str(int(k))] = within_run_ranking(
            sub, ["margin_base", "prior_margin_own", "min_dist"], DV_COL, cell_of_run, args
        )
    strat_seed: dict = {}
    for s, sub in agg.groupby("seed"):
        strat_seed[str(int(s))] = within_run_ranking(
            sub, ["margin_base", "prior_margin_own", "min_dist"], DV_COL, cell_of_run, args
        )

    meta = p553.result_metadata(args, "scripts/issue559_panel_analysis.py")
    meta["task"] = 559
    meta["schema_version"] = SCHEMA_VERSION
    meta["prior_is_stub"] = bool(payload.get("is_stub", False))

    ranking_out = {
        "metadata": meta,
        "dv": DV_COL,
        "n_run_persona_aggregates": len(agg),
        "z_stack_orientation": {"sign_prior": s_p, "sign_min_dist": s_d},
        "within_run_ranking": ranking,
        "incumbent_reproduction": incumbent_report,
        "paired_parity": {
            "matched_minus_prior_run_axis": parity_run,
            "matched_minus_prior_cell_axis": parity_cell,
            "prior_minus_min_dist_run_axis": prior_vs_mind_run,
            "prior_minus_min_dist_cell_axis": prior_vs_mind_cell,
        },
        "outcome_classification": outcome,
        "sensitivity": {
            "truncation_excluded": {
                "n_personas_dropped": int(n_personas_dropped_trunc),
                "ranking": rank_trunc,
                "paired_matched_minus_prior": parity_trunc,
            },
            "median_aggregated_prior": rank_median_agg,
            "split_half": {
                "prior_even_vs_dv_odd": rank_half_a,
                "prior_odd_vs_dv_even": rank_half_b,
            },
        },
        "length_correlates": length_correlates,
        "diagnostics": diag,
        "stratified": {"by_K": strat_k, "by_seed": strat_seed},
    }
    p553.write_json(args.out_dir / "within_run_ranking.json", ranking_out)

    # ── Joint fit (H2) ───────────────────────────────────────────────────────
    print("[joint-fit] level DV ...")
    collinearity_r = float(np.corrcoef(agg["prior_margin_own"], agg["min_dist"])[0, 1])
    gate_tripped = bool(abs(collinearity_r) > COLLINEARITY_GATE_R)
    level = joint_fit_block(agg, "margin_trained", args)
    print("[joint-fit] change DV ...")
    change = joint_fit_block(agg, "dmargin", args)
    change_with_base, _ = _variant_fit(agg, "dmargin", "base", extra_cols=["margin_base"])
    change_with_base_ci = _coef_cluster_boot(
        agg, "dmargin", "base", "persona", args, extra_cols=["margin_base"]
    )
    cv_sets = {
        "prior_only": ["prior_margin_own"],
        "geometry_only": ["min_dist"],
        "both": ["prior_margin_own", "min_dist"],
        "both_plus_interaction": ["prior_margin_own", "min_dist"],
    }
    joint_out = {
        "metadata": meta,
        "collinearity_gate": {
            "pearson_z_prior_z_min_dist": collinearity_r,
            "threshold_abs_r": COLLINEARITY_GATE_R,
            "tripped": gate_tripped,
            "note": "tercile + polynomial-residualization reads computed regardless; "
            "they are the REGISTERED fallback when the gate trips",
        },
        "level_fit": level,
        "change_fit": change,
        "change_fit_margin_base_added": {
            "estimates": change_with_base,
            "ci95_persona_cluster": change_with_base_ci,
            "narration_guard": (
                "§13.6: α-non-survival on dmargin is MECHANICALLY favored — dmargin "
                "subtracts the base level the prior proxies (dmargin = margin_trained "
                "− margin_base). Narrate WITH the subtraction named; this "
                "margin_base-augmented fit is the registered residualized read."
            ),
        },
        "cv": {
            "lopo_by_persona": _cv_oof_r2(agg, "margin_trained", "held_out_persona", cv_sets),
            "loro_by_run": _cv_oof_r2(agg, "margin_trained", "run_id", cv_sets),
            "lopo_by_persona_dmargin": _cv_oof_r2(agg, "dmargin", "held_out_persona", cv_sets),
            "loro_by_run_dmargin": _cv_oof_r2(agg, "dmargin", "run_id", cv_sets),
        },
        "tercile_table_level": tercile_table(agg, "margin_trained"),
        "tercile_table_change": tercile_table(agg, "dmargin"),
        "poly_residualization_level": poly_residualization(agg, "margin_trained", args),
        "poly_residualization_change": poly_residualization(agg, "dmargin", args),
    }
    p553.write_json(args.out_dir / "joint_fit.json", joint_out)

    # ── Figures ──────────────────────────────────────────────────────────────
    print("[figures] writing hero + exploratory dump ...")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    figure_ranking_strip(ranking, args.fig_dir)
    figure_persona_scatters(agg, args.fig_dir)
    figure_collinearity(agg, args.fig_dir)
    figure_forest(level, change, args.fig_dir)
    figure_stratified(strat_k, args.fig_dir, by="K")
    figure_stratified(strat_seed, args.fig_dir, by="seed")
    figure_paired_hist(rho_mb, rho_pr, args.fig_dir)
    figure_lengths(
        own_lens_flat, trained_lens_flat, diag["truncation_rate_per_persona"], args.fig_dir
    )
    _figure_argmax(payload, args.fig_dir)

    print("[done] analysis complete")
    return 0


def _figure_argmax(payload: dict, fig_dir: Path) -> None:
    """Argmax composition bar for the 700 new own-response slots."""
    set_paper_style("blog")
    colors = paper_palette(3)
    comp = payload["summary"]["argmax_composition"]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    cats = ["marker", "eos", "other"]
    ax.bar(range(3), [comp[c]["rate"] for c in cats], color=colors)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["argmax = ※", "argmax = EOS", "argmax = other"], fontsize=8)
    ax.set_ylabel("share of 700 own-response slots")
    ax.set_title("Base own-response slot argmax composition", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "argmax_composition_own_slots", dir=fig_dir)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

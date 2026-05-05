#!/usr/bin/env python3
"""Analyze issue #260 results — 3-panel hero figure + verdict per sub-claim.

Consumes:
  eval_results/issue260/<COND>_leg1/run_result.json   x9 (Leg 1)
  eval_results/issue260/sl_*_leg2/run_result.json     x3 (sub-exp (c) Leg 2)
  eval_results/issue260/mt_*_leg2/run_result.json     x3 (sub-exp (a) Leg 2)

Per the v3 plan (sections 1, 3.10, 3.11, 5):

  - Per-condition cluster-bootstrap 95% CIs over 20 questions (5000 resamples).
  - Per-sub-experiment cluster-bootstrap CI on the regression slope of
    `source-rate ~ x_axis`. Bonferroni 98.33% on the primary tests; 95% on
    the bystander secondaries. Within a SINGLE bootstrap iteration the same
    `q_idx` is reused across all conditions, all eval personas, and (for c)
    both legs.
  - Verdict per sub-claim: SUPPORTED / REJECTED / INDETERMINATE.
  - Confidence label: MODERATE iff per-bystander direction consistent
    across all 4 bystanders (with v3 |Δrate| < 2pp noise band); LOW if
    SUPPORTED but split. (c) Leg 1 bystanders are informational only —
    H_c confidence is derived from source-rate alone.
  - Saturation check on (a): if mt_n1 source-rate >= 80%, (a) is
    INDETERMINATE-due-to-saturation regardless of slope.
  - lc_short style-mismatch check: emit Standing Caveat if `lc_short`
    eval-time mean completion length > 50 tokens (chars/4 used as proxy).
  - Δlogit(source-rate) reported as the secondary effect-size metric.
  - Hero figure: 3 INDEPENDENT panels (each on its own x-axis).
  - Parent #271 librarian anchor on (c) Leg 1 panel as horizontal reference.

Outputs:
  eval_results/issue260/analysis_summary.json
  figures/issue_260/hero_3panel.{png,pdf,meta.json}
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue260"
FIGURES_DIR = PROJECT_ROOT / "figures" / "issue_260"
# FIX 1: prefer the stable anchor copy that the launcher preserves at
# `eval_results/issue260/parent_recipe_anchor.json` BEFORE Leg 1 starts.
# Fall back to the legacy location only if the stable copy is absent (e.g. on
# a fresh re-analysis from a clone that doesn't carry the launcher state).
PARENT_RECIPE_RESULT_STABLE = RESULTS_DIR / "parent_recipe_anchor.json"
PARENT_RECIPE_RESULT_LEGACY = (
    PROJECT_ROOT
    / "eval_results"
    / "leakage_experiment"
    / "marker_librarian_asst_excluded_medium_seed42"
    / "run_result.json"
)


# ── Sub-experiment metadata (matches plan section 3.11) ────────────────────


@dataclass
class SubExp:
    label: str  # "a" | "b" | "c"
    title: str
    conditions: list[tuple[str, float]]  # (cond_name, x_axis_value)
    x_axis_label: str
    log_x: bool  # whether to use log10(x) for the slope regression


SUB_A = SubExp(
    label="a",
    title="(a) Multi-turn",
    conditions=[("mt_n1", 1), ("mt_n4", 4), ("mt_n16", 16)],
    x_axis_label="Number of (user, assistant) turn pairs (N)",
    log_x=False,
)
SUB_B = SubExp(
    label="b",
    title="(b) Long completion",
    conditions=[("lc_short", 50), ("lc_medium", 316), ("lc_long", 1050)],
    x_axis_label="Mean positive completion tokens (log scale)",
    log_x=True,
)
SUB_C = SubExp(
    label="c",
    title="(c) System-prompt length (Leg 1)",
    conditions=[("sl_short", 5), ("sl_medium", 15), ("sl_long", 256)],
    x_axis_label="System-prompt tokens (log scale)",
    log_x=True,
)
SUB_EXPS = [SUB_A, SUB_B, SUB_C]

# Bystander panel (plan section 3.10):
#   source = librarian (PRIMARY)
#   near-bystander = software_engineer
#   far-bystander = mean(comedian, villain)
#   informational = assistant
SOURCE = "librarian"
NEAR_BYSTANDER = "software_engineer"
FAR_BYSTANDERS = ("comedian", "villain")
INFO_BYSTANDER = "assistant"
ALL_BYSTANDERS = (NEAR_BYSTANDER, *FAR_BYSTANDERS, INFO_BYSTANDER)

# Statistical parameters (plan section 1, Reproducibility Card).
N_BOOTSTRAP = 5000
PRIMARY_ALPHA = 0.05 / 3  # Bonferroni across 3 sub-claims
PRIMARY_PCTILE_LO = 100 * (PRIMARY_ALPHA / 2)  # 0.833
PRIMARY_PCTILE_HI = 100 * (1 - PRIMARY_ALPHA / 2)  # 99.167
SECONDARY_PCTILE_LO = 2.5
SECONDARY_PCTILE_HI = 97.5
NOISE_BAND_PP = 2.0  # |Δrate| < 2pp counts as "no movement" (plan section 1 v3)
SATURATION_THRESHOLD = 0.80
EFFECT_FLOOR_PP = 10.0  # +10pp Δsource-rate
EFFECT_FLOOR_LOGIT = 0.5  # Δlogit(source-rate) >= +0.5
REJECT_FLOOR_PP = -5.0  # Δsource-rate <= -5pp for REJECTED
LC_SHORT_LENGTH_TOKEN_THRESHOLD = 50  # plan section 6 v3 caveat


# ── Result loading ──────────────────────────────────────────────────────────


def _load_run_result(cond_dir: Path) -> dict:
    rr = cond_dir / "run_result.json"
    if not rr.exists():
        raise FileNotFoundError(f"run_result.json missing: {rr}")
    return json.loads(rr.read_text())


def _load_marker_eval(cond_dir: Path) -> dict:
    me = cond_dir / "marker_eval.json"
    if not me.exists():
        raise FileNotFoundError(f"marker_eval.json missing: {me}")
    return json.loads(me.read_text())


def _per_question_rates(marker_eval: dict, persona: str) -> dict[str, float]:
    """Extract per-question rate dict for a given persona."""
    if persona not in marker_eval:
        raise KeyError(
            f"persona {persona!r} not in marker_eval ({list(marker_eval.keys())[:5]}...)"
        )
    pq = marker_eval[persona].get("per_question", {})
    return {q: pq[q]["rate"] for q in pq}


def _completion_length_stats(marker_eval: dict, persona: str) -> dict:
    return marker_eval.get(persona, {}).get("completion_length_stats", {})


# ── Cluster-bootstrap ────────────────────────────────────────────────────────


def _safe_logit(p: float) -> float:
    """Clipped logit so tail conditions don't blow up."""
    p = float(min(max(p, 1e-3), 1 - 1e-3))
    return math.log(p / (1 - p))


def _rate_from_question_indices(per_q: dict[str, float], q_keys: list[str]) -> float:
    """Mean of per-question rates over the resampled question indices."""
    if not q_keys:
        return 0.0
    return float(np.mean([per_q[q] for q in q_keys]))


def _slope(xs: np.ndarray, ys: np.ndarray, log_x: bool) -> float:
    if log_x:
        xs = np.log10(xs)
    # Linear regression slope on n=3 points.
    coef = np.polyfit(xs, ys, deg=1)
    return float(coef[0])


def cluster_bootstrap_per_cell(
    per_q_rates: dict[str, float],
    rng: np.random.Generator,
    q_keys: list[str],
    n_resamples: int,
) -> tuple[float, float, float]:
    """Cluster-bootstrap 95% CI on a single cell's rate.

    Returns (point_estimate, ci_lo, ci_hi).
    """
    n_q = len(q_keys)
    rates = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n_q, size=n_q)
        sample = [q_keys[i] for i in idx]
        rates[b] = _rate_from_question_indices(per_q_rates, sample)
    point = _rate_from_question_indices(per_q_rates, q_keys)
    lo = float(np.percentile(rates, SECONDARY_PCTILE_LO))
    hi = float(np.percentile(rates, SECONDARY_PCTILE_HI))
    return point, lo, hi


def cluster_bootstrap_slopes_paired(
    sub_exp: SubExp,
    per_persona_per_cond_per_q: dict[str, dict[str, dict[str, float]]],
    rng: np.random.Generator,
    n_resamples: int,
) -> dict[str, dict]:
    """FIX 5: paired cluster-bootstrap on the regression slope across personas.

    Plan §1 v3 + §12 v3 mandatory rule: "within a single bootstrap iteration,
    the same `q_idx` is reused across all conditions, all eval personas, and
    (for c) both legs."

    Round-1 implementation called `cluster_bootstrap_slope` once per persona,
    advancing the rng between calls — giving each persona an INDEPENDENT
    q_idx sequence. This refactor draws q_idx ONCE per iteration and computes
    slopes for source / near / far / assistant FROM THAT SAME SAMPLE, then
    aggregates per-persona at the end. CIs are paired across personas, so any
    downstream cross-persona statistic (e.g. bystander-vs-source slope ratio)
    derives from coherent draws.

    `per_persona_per_cond_per_q[persona][cond_name]` is a dict of per-question rates.

    Returns: dict[persona] -> {point_slope, ci95_lo/hi, ci_bonf_lo/hi,
                               point_rates, x_axis, ...}
    """
    cond_names = [c for c, _ in sub_exp.conditions]
    xs = np.array([x for _, x in sub_exp.conditions], dtype=np.float64)
    personas = list(per_persona_per_cond_per_q.keys())

    # Canonical question key list (must match across every persona+condition).
    first_p = personas[0]
    canonical = sorted(per_persona_per_cond_per_q[first_p][cond_names[0]].keys())
    n_q = len(canonical)
    if n_q == 0:
        raise RuntimeError(f"no per-question rates for {sub_exp.label}/{cond_names[0]}")
    for p in personas:
        for c in cond_names:
            if sorted(per_persona_per_cond_per_q[p][c].keys()) != canonical:
                raise RuntimeError(
                    f"sub-exp {sub_exp.label}: persona={p} condition={c} "
                    f"has a different question set"
                )

    # Pre-allocate slope buffers per persona.
    slopes: dict[str, np.ndarray] = {p: np.empty(n_resamples, dtype=np.float64) for p in personas}
    for b in range(n_resamples):
        # FIX 5: draw q_idx ONCE per iteration; reuse across all personas + conditions.
        idx = rng.integers(0, n_q, size=n_q)
        sample = [canonical[i] for i in idx]
        for p in personas:
            ys = np.array(
                [
                    _rate_from_question_indices(per_persona_per_cond_per_q[p][c], sample)
                    for c in cond_names
                ],
                dtype=np.float64,
            )
            slopes[p][b] = _slope(xs, ys, sub_exp.log_x)

    out: dict[str, dict] = {}
    for p in personas:
        point_ys = np.array(
            [
                _rate_from_question_indices(per_persona_per_cond_per_q[p][c], canonical)
                for c in cond_names
            ],
            dtype=np.float64,
        )
        point_slope = _slope(xs, point_ys, sub_exp.log_x)
        out[p] = {
            "point_slope": point_slope,
            "ci95_lo": float(np.percentile(slopes[p], SECONDARY_PCTILE_LO)),
            "ci95_hi": float(np.percentile(slopes[p], SECONDARY_PCTILE_HI)),
            "ci_bonf_lo": float(np.percentile(slopes[p], PRIMARY_PCTILE_LO)),
            "ci_bonf_hi": float(np.percentile(slopes[p], PRIMARY_PCTILE_HI)),
            "n_resamples": n_resamples,
            "point_rates": point_ys.tolist(),
            "x_axis": xs.tolist(),
            "x_axis_log": sub_exp.log_x,
            "cond_names": cond_names,
            "paired_across_personas": True,  # FIX 5 marker
        }
    return out


# ── Per-sub-experiment driver ────────────────────────────────────────────────


@dataclass
class SubExpAnalysis:
    sub_exp: SubExp
    per_cond_marker_eval: dict[str, dict]  # cond_name -> raw marker_eval json
    per_cond_run_result: dict[str, dict]
    # Persona-level per-question rates (cond_name -> persona -> {q: rate})
    per_cond_per_persona_q: dict[str, dict[str, dict[str, float]]]
    cell_bootstrap: dict  # persona -> cond_name -> {rate, ci_lo, ci_hi}
    slope_bootstrap: dict  # persona -> dict from cluster_bootstrap_slope
    completion_length_stats: dict  # cond_name -> persona -> stats
    saturated: bool
    lc_short_style_mismatch: bool
    verdict: str
    confidence: str
    delta_source_rate_pp: float
    delta_source_rate_logit: float
    bystander_signs: dict[str, str]  # persona -> "up" / "down" / "noise"


def _compute_far_bystander_per_q(
    cond_name: str, per_persona_q: dict[str, dict[str, float]]
) -> dict[str, float]:
    """Return per-question mean across far-bystander personas (comedian + villain)."""
    out: dict[str, float] = {}
    qs = sorted(per_persona_q[FAR_BYSTANDERS[0]].keys())
    for q in qs:
        vals = [per_persona_q[p][q] for p in FAR_BYSTANDERS]
        out[q] = float(np.mean(vals))
    return out


def _classify_sign(delta_pp: float) -> str:
    if abs(delta_pp) < NOISE_BAND_PP:
        return "noise"
    return "up" if delta_pp > 0 else "down"


def _check_lc_short_style_mismatch(
    sub_exp: SubExp, completion_length_stats: dict[str, dict]
) -> bool:
    """Plan §6 v3 caveat: emit Standing Caveat if `lc_short`-trained model produced
    verbose eval-time output (mean char-length / 4 ~ tokens > 50)."""
    if sub_exp.label != "b":
        return False
    lc_short_stats = completion_length_stats.get("lc_short", {}).get(SOURCE, {})
    mean_chars = lc_short_stats.get("eval_completion_length_mean_chars", 0.0) or 0.0
    # chars/4 is a rough Qwen-tokenizer proxy; precise check happens in analyzer
    # narrative if needed.
    approx_tokens = mean_chars / 4.0
    return approx_tokens > LC_SHORT_LENGTH_TOKEN_THRESHOLD


def _decide_verdict(
    bonf_lo: float,
    bonf_hi: float,
    delta_pp: float,
    delta_logit: float,
    saturated: bool,
) -> str:
    """Plan §5 verdict logic. Logit-rescue applies ONLY to the effect-size
    floor — slope CI gate is rate-space-only (plan §1 v3)."""
    if saturated:
        return "INDETERMINATE-saturation"
    excludes_zero_positive = bonf_lo > 0
    excludes_zero_negative = bonf_hi < 0
    effect_floor_met = (delta_pp >= EFFECT_FLOOR_PP) or (delta_logit >= EFFECT_FLOOR_LOGIT)
    if excludes_zero_positive and effect_floor_met:
        return "SUPPORTED"
    if excludes_zero_negative and delta_pp <= REJECT_FLOOR_PP:
        return "REJECTED"
    return "INDETERMINATE"


def _decide_confidence(verdict: str, sub_exp: SubExp, bystander_signs: dict[str, str]) -> str:
    """Plan §5 confidence label.

    HIGH is unavailable at single seed; MODERATE iff per-bystander direction
    consistent (with v3 |Δrate| < 2pp noise band); LOW if SUPPORTED but split.
    Sub-exp (c) bystander confidence is informational only (plan §5 v3); H_c
    is derived from source-rate alone, so it caps at LOW.
    """
    if verdict != "SUPPORTED":
        return ""
    if sub_exp.label == "c":
        return "LOW"
    non_noise_signs = [s for s in bystander_signs.values() if s != "noise"]
    if not non_noise_signs:
        return "LOW"
    if len(set(non_noise_signs)) == 1:
        return "MODERATE"
    return "LOW"


def analyze_sub_exp(
    sub_exp: SubExp,
    rng: np.random.Generator,
    n_resamples: int = N_BOOTSTRAP,
) -> SubExpAnalysis:
    per_cond_marker_eval: dict[str, dict] = {}
    per_cond_run_result: dict[str, dict] = {}
    per_cond_per_persona_q: dict[str, dict[str, dict[str, float]]] = {}
    completion_length_stats: dict[str, dict] = {}

    for cond, _ in sub_exp.conditions:
        cond_dir = RESULTS_DIR / cond
        per_cond_run_result[cond] = _load_run_result(cond_dir)
        me = _load_marker_eval(cond_dir)
        per_cond_marker_eval[cond] = me
        per_persona_q: dict[str, dict[str, float]] = {}
        for p in (SOURCE, *ALL_BYSTANDERS):
            if p not in me:
                raise KeyError(f"{cond}: marker_eval missing persona {p}")
            per_persona_q[p] = _per_question_rates(me, p)
        # Synthesize a "far_bystander" pseudo-persona = mean(comedian, villain)
        # at the per-question level. This is the unit used for sub-claim slope
        # bootstrap on far-bystander.
        per_persona_q["__far_bystander__"] = _compute_far_bystander_per_q(cond, per_persona_q)
        per_cond_per_persona_q[cond] = per_persona_q
        completion_length_stats[cond] = {
            p: _completion_length_stats(me, p) for p in (SOURCE, *ALL_BYSTANDERS)
        }

    # Per-cell bootstrap CIs (95%).
    cell_bootstrap: dict[str, dict[str, dict]] = {
        p: {} for p in (SOURCE, *ALL_BYSTANDERS, "__far_bystander__")
    }
    for cond, _ in sub_exp.conditions:
        for p in cell_bootstrap:
            per_q = per_cond_per_persona_q[cond][p]
            q_keys = sorted(per_q.keys())
            point, lo, hi = cluster_bootstrap_per_cell(per_q, rng, q_keys, n_resamples=n_resamples)
            cell_bootstrap[p][cond] = {"rate": point, "ci_lo": lo, "ci_hi": hi}

    # FIX 5: Slope bootstrap PAIRED across personas — single global loop with
    # one q_idx draw per iteration, reused across source / near / far / assistant.
    # This satisfies plan §1 v3 + §12 v3 mandatory pairing rule and ensures any
    # downstream cross-persona statistic (e.g. bystander-vs-source slope ratio)
    # derives from coherent draws.
    slope_personas = (SOURCE, NEAR_BYSTANDER, "__far_bystander__", INFO_BYSTANDER)
    per_persona_per_cond_q: dict[str, dict[str, dict[str, float]]] = {
        p: {c: per_cond_per_persona_q[c][p] for c, _ in sub_exp.conditions} for p in slope_personas
    }
    slope_bootstrap = cluster_bootstrap_slopes_paired(
        sub_exp, per_persona_per_cond_q, rng, n_resamples=n_resamples
    )

    # Effect-size (largest - smallest condition).
    src_rates = slope_bootstrap[SOURCE]["point_rates"]
    delta_pp = 100 * (src_rates[-1] - src_rates[0])
    delta_logit = _safe_logit(src_rates[-1]) - _safe_logit(src_rates[0])

    # Per-bystander Δrate signs (for confidence label).
    bystander_signs: dict[str, str] = {}
    for p in ALL_BYSTANDERS:
        rates = [cell_bootstrap[p][c]["rate"] for c, _ in sub_exp.conditions]
        bystander_signs[p] = _classify_sign(100 * (rates[-1] - rates[0]))

    saturated = sub_exp.label == "a" and src_rates[0] >= SATURATION_THRESHOLD
    lc_short_style_mismatch = _check_lc_short_style_mismatch(sub_exp, completion_length_stats)
    verdict = _decide_verdict(
        slope_bootstrap[SOURCE]["ci_bonf_lo"],
        slope_bootstrap[SOURCE]["ci_bonf_hi"],
        delta_pp,
        delta_logit,
        saturated,
    )
    confidence = _decide_confidence(verdict, sub_exp, bystander_signs)

    return SubExpAnalysis(
        sub_exp=sub_exp,
        per_cond_marker_eval=per_cond_marker_eval,
        per_cond_run_result=per_cond_run_result,
        per_cond_per_persona_q=per_cond_per_persona_q,
        cell_bootstrap=cell_bootstrap,
        slope_bootstrap=slope_bootstrap,
        completion_length_stats=completion_length_stats,
        saturated=saturated,
        lc_short_style_mismatch=lc_short_style_mismatch,
        verdict=verdict,
        confidence=confidence,
        delta_source_rate_pp=delta_pp,
        delta_source_rate_logit=delta_logit,
        bystander_signs=bystander_signs,
    )


# ── Hero figure ──────────────────────────────────────────────────────────────


def build_hero_figure(analyses: list[SubExpAnalysis], out_dir: Path) -> Path:
    """Build the 3-panel hero figure (one panel per sub-experiment)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    palette = paper_palette(4)
    color_source, color_near, color_far, color_assist = palette

    parent_anchor = _load_parent_anchor()

    for ax, an in zip(axes, analyses, strict=True):
        sub = an.sub_exp
        xs_raw = np.array([x for _, x in sub.conditions], dtype=np.float64)
        cond_names = [c for c, _ in sub.conditions]
        x_plot = np.log10(xs_raw) if sub.log_x else xs_raw

        for persona, color, label in [
            (SOURCE, color_source, "Source (librarian)"),
            (NEAR_BYSTANDER, color_near, "Near bystander (SWE)"),
            ("__far_bystander__", color_far, "Far bystander (comedian + villain)"),
            (INFO_BYSTANDER, color_assist, "Assistant"),
        ]:
            rates = [an.cell_bootstrap[persona][c]["rate"] for c in cond_names]
            lo = [an.cell_bootstrap[persona][c]["ci_lo"] for c in cond_names]
            hi = [an.cell_bootstrap[persona][c]["ci_hi"] for c in cond_names]
            err_lower = [r - lo_i for r, lo_i in zip(rates, lo, strict=True)]
            err_upper = [hi_i - r for r, hi_i in zip(rates, hi, strict=True)]
            ax.errorbar(
                x_plot,
                rates,
                yerr=[err_lower, err_upper],
                marker="o",
                linewidth=1.5,
                markersize=5,
                capsize=3,
                color=color,
                label=label,
            )

        # Parent #271 librarian anchor on (c) Leg 1 panel only.
        if sub.label == "c" and parent_anchor is not None:
            ax.axhline(
                parent_anchor,
                color=color_source,
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
                label=f"Parent recipe anchor ({parent_anchor:.2f})",
            )

        ax.set_title(sub.title, fontsize=10)
        ax.set_xlabel(sub.x_axis_label, fontsize=9)
        ax.set_ylabel("Marker rate ([ZLT] presence)", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        if sub.log_x:
            ax.set_xticks(np.log10(xs_raw))
            ax.set_xticklabels([f"{int(x)}" for x in xs_raw])
        else:
            ax.set_xticks(xs_raw)
            ax.set_xticklabels([f"{int(x)}" for x in xs_raw])
        # Annotate verdict in the panel corner.
        ax.text(
            0.02,
            0.98,
            f"{an.verdict}\n{an.confidence}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "lightgrey", "alpha": 0.8},
        )

    # One shared legend below all panels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    return savefig_paper(fig, "hero_3panel", dir=str(out_dir))


def _load_parent_anchor() -> float | None:
    """Librarian source-rate from the parent #246/#271 result, if available.

    FIX 1: prefer the launcher-preserved stable copy under
    `eval_results/issue260/parent_recipe_anchor.json`. Fall back to the legacy
    parent-recipe path only if the stable copy is absent.
    """
    candidates = [PARENT_RECIPE_RESULT_STABLE, PARENT_RECIPE_RESULT_LEGACY]
    for path in candidates:
        if path.exists():
            rr = json.loads(path.read_text())
            rate = rr.get("results", {}).get("marker", {}).get("source_rate")
            if rate is not None:
                return float(rate)
    return None


# ── Main orchestration ──────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=N_BOOTSTRAP,
        help="Cluster-bootstrap resample count (default: 5000 per plan v3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the bootstrap (default 42).",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip hero figure generation (useful when matplotlib backend is broken).",
    )
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    print(f"[analyzer] cluster-bootstrap n_resamples={args.n_resamples}, seed={args.seed}")

    analyses: list[SubExpAnalysis] = []
    for sub in SUB_EXPS:
        try:
            an = analyze_sub_exp(sub, rng, n_resamples=args.n_resamples)
        except FileNotFoundError as e:
            print(f"[analyzer] skipping sub-exp ({sub.label}) -- {e}")
            continue
        analyses.append(an)

    if not analyses:
        raise SystemExit("[analyzer] no sub-experiment results found; nothing to analyze.")

    # Leg 2 informational summaries (plan §5 secondary; (c) Leg 2 is train-matched
    # eval, (a) Leg 2 is multi-turn-K eval). Just dump per-condition source-rate
    # and bystander rates so the downstream analyzer / clean-result writer has
    # access. Leg 2 dirs are <COND>_leg2/ under RESULTS_DIR.
    leg2_summary: dict[str, dict] = {}
    for spec_name in (
        "sl_short_leg2",
        "sl_medium_leg2",
        "sl_long_leg2",
        "mt_n1_leg2",
        "mt_n4_leg2",
        "mt_n16_leg2",
    ):
        leg2_dir = RESULTS_DIR / spec_name
        if not leg2_dir.exists():
            continue
        try:
            me = _load_marker_eval(leg2_dir)
        except FileNotFoundError:
            continue
        leg2_summary[spec_name] = {
            "source_rate": me.get(SOURCE, {}).get("rate"),
            "near_bystander_rate": me.get(NEAR_BYSTANDER, {}).get("rate"),
            "far_bystander_rates": {p: me.get(p, {}).get("rate") for p in FAR_BYSTANDERS},
            "assistant_rate": me.get(INFO_BYSTANDER, {}).get("rate"),
            "completion_length_stats_source": me.get(SOURCE, {}).get("completion_length_stats", {}),
        }

    # ── Summary JSON ──
    summary: dict = {
        "issue": 260,
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "primary_alpha": PRIMARY_ALPHA,
        "primary_pctile_lo": PRIMARY_PCTILE_LO,
        "primary_pctile_hi": PRIMARY_PCTILE_HI,
        "noise_band_pp": NOISE_BAND_PP,
        "saturation_threshold": SATURATION_THRESHOLD,
        "effect_floor_pp": EFFECT_FLOOR_PP,
        "effect_floor_logit": EFFECT_FLOOR_LOGIT,
        "lc_short_length_threshold_tokens": LC_SHORT_LENGTH_TOKEN_THRESHOLD,
        "sub_experiments": {},
    }
    for an in analyses:
        sub_summary = {
            "title": an.sub_exp.title,
            "x_axis_label": an.sub_exp.x_axis_label,
            "log_x": an.sub_exp.log_x,
            "verdict": an.verdict,
            "confidence": an.confidence,
            "delta_source_rate_pp": an.delta_source_rate_pp,
            "delta_source_rate_logit": an.delta_source_rate_logit,
            "saturated": an.saturated,
            "lc_short_style_mismatch": an.lc_short_style_mismatch,
            "bystander_signs": an.bystander_signs,
            "slope_bootstrap": an.slope_bootstrap,
            "cell_bootstrap": an.cell_bootstrap,
            "completion_length_stats": an.completion_length_stats,
        }
        summary["sub_experiments"][an.sub_exp.label] = sub_summary
    summary["leg2_informational"] = leg2_summary

    out_path = RESULTS_DIR / "analysis_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[analyzer] wrote {out_path}")

    # Print a concise verdict table.
    print()
    print("Sub-claim verdicts (Bonferroni 98.33% slope CI):")
    for an in analyses:
        bonf_lo = an.slope_bootstrap[SOURCE]["ci_bonf_lo"]
        bonf_hi = an.slope_bootstrap[SOURCE]["ci_bonf_hi"]
        print(
            f"  ({an.sub_exp.label}) {an.sub_exp.title:35} "
            f"verdict={an.verdict:25} confidence={an.confidence or '-':8} "
            f"slope_CI=[{bonf_lo:.4f}, {bonf_hi:.4f}] "
            f"Δrate={an.delta_source_rate_pp:+.1f}pp Δlogit={an.delta_source_rate_logit:+.2f}"
        )

    # ── Hero figure ──
    if not args.no_figure:
        try:
            fig_path = build_hero_figure(analyses, FIGURES_DIR)
            print(f"[analyzer] hero figure -> {fig_path}")
        except Exception as e:
            print(f"[analyzer] WARN: hero figure failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Issue #1769 — A phase: paired deltas, fractions, bootstrap CIs, lattice.

Analysis recipe (plan §6): per (trait, arm, alpha) the question-level mean
judged score (judge draws averaged first — inside ``judge_graded`` — then the
10 generation draws averaged per question); paired per question
``delta_arm(q) = score_arm(q) - score_neither(q)``; ``Delta_arm`` = mean over
the 20 questions. Fractions ``f_d = Delta_decode_only / Delta_both`` and
``f_p = Delta_prefill_only / Delta_both`` at the per-trait OPERATING alpha.

CIs: question-level cluster bootstrap (resample the 20 questions with
replacement, B=2000, seed 0, percentile CIs; draws stay within their
question). Two CI variants per fraction: ``frozen-at-alpha`` AND
``selection-inherited`` (the operating-alpha rule — coherence >= 50%,
Delta_both >= 10, both-arm mean <= 85 — re-applied inside each resample;
resamples with an EMPTY passing alpha set are counted + reported).

Ratio-instability guard (pinned predicate): share of resamples with
``Delta*_both <= 0`` > 1% => the ratio CI is unstable — absolute deltas only
+ mixed/indeterminate for that trait.

Classification lattice (plan §3, pre-registered; PRIMARY = frozen-at-alpha):
- prefill-committed  <=> 95% CI(f_d) subset of (-0.25, +0.25)
- decode-driven      <=> CI lower bound > 0.75
- mixed/indeterminate otherwise (absorbs every residual cell, incl. a
  no-passing-operating-alpha trait — routed EXPLICITLY, never silently).

Manipulation check (K1): >= 1 trait with Delta_both >= 10 and CI excluding 0
at some alpha is REQUIRED before any null decode-only arm is interpreted.

Unfiltered-primary rule: ALL draws enter the primary read; a coherent-only
sensitivity read is reported alongside, labeled selection-conditioned, with
the >= 20-point decode-vs-both coherence-asymmetry flag.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1769_analysis")

STEERED_ARMS = ("prefill_only", "decode_only", "both")
B_BOOT = 2000  # plan §11
BOOT_SEED = 0  # plan §11
MANIPULATION_FLOOR = 10.0  # plan §7 K1 (2x the #1415 5-pt judge-shift minimum)
COHERENCE_MIN = 0.5  # #1415 condition gate
CEILING_MAX_MEAN = 85.0  # plan §6 ceiling-saturation guard
LATTICE_LO = 0.25  # plan §3 (design-derived; point estimates reported verbatim)
LATTICE_HI = 0.75
INSTABILITY_FRAC = 0.01  # plan §6 pinned instability-guard predicate
COHERENCE_ASYMMETRY_PTS = 0.20  # plan §6: decode >= 20 pts below both => flag


# ── pinned pure predicates (test-pinned) ──────────────────────────────


def classify_lattice(ci_lo: float, ci_hi: float) -> str:
    """The pre-registered §3 lattice over a 95% CI of f_d.

    Disjoint (an interval inside (-0.25, 0.25) cannot have lower bound
    > 0.75) and exhaustive (the otherwise clause closes the partition). A CI
    at or below -0.25 (an actively destructive decode arm) routes to
    mixed/indeterminate — the lower bound blocks labeling it
    "decode contributes nothing"."""
    if ci_lo > -LATTICE_LO and ci_hi < LATTICE_LO:
        return "prefill-committed"
    if ci_lo > LATTICE_HI:
        return "decode-driven"
    return "mixed/indeterminate"


def ratio_unstable(frac_nonpositive_both: float) -> bool:
    """Pinned instability-guard predicate: share of bootstrap resamples with
    ``Delta*_both <= 0`` STRICTLY above 1% => unstable (deliberately STRICTER
    than the manipulation check's CI-excludes-0, which tolerates <= 2.5%
    mass — a trait can pass K1 yet route to indeterminate here; intended)."""
    return frac_nonpositive_both > INSTABILITY_FRAC


def percentile_ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


# ── data assembly ─────────────────────────────────────────────────────


def question_scores(per_item: dict, *, coherent_only: bool = False) -> dict:
    """{trait: {(arm, alpha): {qid: mean score}}} — judge draws already
    averaged per item; here the generation draws average per question. Items
    with graded_score None (all judge draws dropped) are excluded."""
    acc: dict = {}
    for rec in per_item.values():
        if rec["graded_score"] is None:
            continue
        if coherent_only and not rec["coherent"]:
            continue
        key = (rec["arm"], rec["alpha"])
        acc.setdefault(rec["trait"], {}).setdefault(key, {}).setdefault(
            rec["question_id"], []
        ).append(rec["graded_score"])
    out: dict = {}
    for trait, by_key in acc.items():
        out[trait] = {
            key: {qid: float(np.mean(v)) for qid, v in by_q.items()} for key, by_q in by_key.items()
        }
    return out


def coherence_fractions(per_item: dict) -> dict:
    """{trait: {(arm, alpha): {qid: coherent fraction over generation draws}}}."""
    acc: dict = {}
    for rec in per_item.values():
        key = (rec["arm"], rec["alpha"])
        acc.setdefault(rec["trait"], {}).setdefault(key, {}).setdefault(
            rec["question_id"], []
        ).append(bool(rec["coherent"]))
    out: dict = {}
    for trait, by_key in acc.items():
        out[trait] = {
            key: {qid: float(np.mean(v)) for qid, v in by_q.items()} for key, by_q in by_key.items()
        }
    return out


def _vec(by_q: dict, qids: list[int]) -> np.ndarray:
    """Dense (nq,) vector over the shared question axis; missing => NaN."""
    return np.array([by_q.get(q, np.nan) for q in qids], dtype=float)


# ── per-trait analysis ────────────────────────────────────────────────


def analyze_trait(
    trait: str,
    scores_t: dict,
    coher_t: dict,
    alphas: list[float],
    qids: list[int],
    rng_indices: np.ndarray,
) -> dict:
    """Full per-trait read: alpha ladder, operating alpha, fractions,
    frozen + selection-inherited CIs, lattice classification."""
    nq = len(qids)
    B = rng_indices.shape[0]
    neither = _vec(scores_t[("neither", None)], qids)
    assert not np.isnan(neither).any(), (trait, "neither scores incomplete")

    # Per (arm, alpha): question-level score + delta vectors.
    delta: dict = {}
    mean_score: dict = {}
    for arm in STEERED_ARMS:
        for alpha in alphas:
            s = _vec(scores_t[(arm, alpha)], qids)
            assert not np.isnan(s).any(), (trait, arm, alpha, "scores incomplete")
            delta[(arm, alpha)] = s - neither
            mean_score[(arm, alpha)] = float(s.mean())

    # Bootstrap gather matrices (B, nq) per (arm, alpha).
    boot_delta = {k: v[rng_indices].mean(axis=1) for k, v in delta.items()}

    # Alpha-ladder table (the exploratory dump) + per-alpha operating checks.
    ladder: dict = {}
    per_alpha_pass: dict = {}
    for alpha in alphas:
        row: dict = {}
        for arm in STEERED_ARMS:
            d = delta[(arm, alpha)]
            bd = boot_delta[(arm, alpha)]
            lo, hi = percentile_ci(bd)
            row[arm] = {
                "delta": float(d.mean()),
                "ci95": [lo, hi],
                "mean_score": mean_score[(arm, alpha)],
                "coherence_rate": float(np.mean(_vec(coher_t[(arm, alpha)], qids))),
            }
        both = row["both"]
        checks = {
            "coherence_ok": both["coherence_rate"] >= COHERENCE_MIN,
            "manipulation_ok": (both["delta"] >= MANIPULATION_FLOOR) and (both["ci95"][0] > 0.0),
            "ceiling_ok": both["mean_score"] <= CEILING_MAX_MEAN,
        }
        per_alpha_pass[alpha] = all(checks.values())
        row["operating_checks"] = checks
        ladder[f"{alpha:g}"] = row
    ladder["neither"] = {
        "mean_score": float(neither.mean()),
        "coherence_rate": float(np.mean(_vec(coher_t[("neither", None)], qids))),
    }

    passing = [a for a in alphas if per_alpha_pass[a]]
    operating_alpha = max(passing) if passing else None
    result: dict = {
        "trait": trait,
        "alpha_ladder": ladder,
        "operating_alpha": operating_alpha,
        "manipulation_check_any_alpha": any(
            ladder[f"{a:g}"]["operating_checks"]["manipulation_ok"] for a in alphas
        ),
    }
    if operating_alpha is None:
        # EXPLICIT routing — never silently unclassified (plan §6).
        result["classification"] = "mixed/indeterminate"
        result["classification_reason"] = "no passing operating alpha"
        return result

    a0 = operating_alpha
    d_both = boot_delta[("both", a0)]
    frac_nonpos = float(np.mean(d_both <= 0.0))
    unstable = ratio_unstable(frac_nonpos)
    fractions: dict = {}
    for name, arm in (("f_d", "decode_only"), ("f_p", "prefill_only")):
        point = float(delta[(arm, a0)].mean() / delta[("both", a0)].mean())
        boot = boot_delta[(arm, a0)] / d_both
        lo, hi = percentile_ci(boot)
        fractions[name] = {"point": point, "ci95_frozen": [lo, hi]}
    result.update(
        {
            "frac_bootstrap_delta_both_nonpositive": frac_nonpos,
            "ratio_unstable": unstable,
            "fractions": fractions,
            "deltas_at_operating_alpha": {
                arm: {
                    "delta": float(delta[(arm, a0)].mean()),
                    "ci95": list(percentile_ci(boot_delta[(arm, a0)])),
                }
                for arm in STEERED_ARMS
            },
        }
    )

    # Selection-inherited CI: re-apply the operating rule per resample.
    coher_q = {(arm, a): _vec(coher_t[(arm, a)], qids) for arm in STEERED_ARMS for a in alphas}
    score_q = {(arm, a): _vec(scores_t[(arm, a)], qids) for arm in STEERED_ARMS for a in alphas}
    sel_alpha = np.full(B, np.nan)
    for a in sorted(alphas):  # ascending: later (larger) passing alpha overwrites
        coh = coher_q[("both", a)][rng_indices].mean(axis=1)
        db = boot_delta[("both", a)]
        ms = score_q[("both", a)][rng_indices].mean(axis=1)
        ok = (coh >= COHERENCE_MIN) & (db >= MANIPULATION_FLOOR) & (ms <= CEILING_MAX_MEAN)
        sel_alpha = np.where(ok, a, sel_alpha)
    n_empty = int(np.isnan(sel_alpha).sum())
    result["selection_inherited"] = {"n_empty_alpha_resamples": n_empty, "B": B}
    valid = ~np.isnan(sel_alpha)
    if valid.any():
        for name, arm in (("f_d", "decode_only"), ("f_p", "prefill_only")):
            num = np.full(B, np.nan)
            den = np.full(B, np.nan)
            for a in alphas:
                m = valid & (sel_alpha == a)
                num[m] = boot_delta[(arm, a)][m]
                den[m] = boot_delta[(("both"), a)][m]
            ratio = num[valid] / den[valid]
            lo, hi = percentile_ci(ratio)
            fractions[name]["ci95_selection_inherited"] = [lo, hi]

    # Lattice (PRIMARY = frozen-at-alpha CI; plan §6).
    if unstable:
        result["classification"] = "mixed/indeterminate"
        result["classification_reason"] = (
            f"ratio-instability guard: {frac_nonpos:.3f} of resamples have "
            "Delta*_both <= 0 (> 1%) — absolute deltas only"
        )
    else:
        lo, hi = fractions["f_d"]["ci95_frozen"]
        result["classification"] = classify_lattice(lo, hi)
        result["classification_reason"] = f"frozen-at-alpha CI(f_d) = [{lo:.3f}, {hi:.3f}]"
        if "ci95_selection_inherited" in fractions["f_d"]:
            slo, shi = fractions["f_d"]["ci95_selection_inherited"]
            result["classification_selection_inherited"] = classify_lattice(slo, shi)
    return result


def ceiling_diagnostic(per_item: dict) -> dict:
    """Fraction of completion-mean scores > 90 per (trait, arm, alpha) — the
    ceiling-compression monitor (plan §6)."""
    acc: dict = {}
    for rec in per_item.values():
        if rec["graded_score"] is None:
            continue
        key = f"{rec['trait']}/{rec['arm']}" + (
            f"/a{rec['alpha']:g}" if rec["alpha"] is not None else ""
        )
        acc.setdefault(key, []).append(rec["graded_score"] > 90.0)
    return {k: float(np.mean(v)) for k, v in sorted(acc.items())}


# ── figures ───────────────────────────────────────────────────────────


def _err(point: float, lo: float, hi: float) -> tuple[float, float]:
    """Non-negative errorbar OFFSETS (never CI bounds; clamped — gotchas.md
    xerr/yerr entry: quantile CIs can invert by float epsilon at tiny n)."""
    return max(0.0, point - lo), max(0.0, hi - point)


def make_figures(headline: dict, fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        set_paper_style,
    )

    set_paper_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    traits = sorted(headline["per_trait"])
    arm_colors = {
        "neither": paper_palette_role("neutral"),
        "prefill_only": paper_palette_role("primary"),
        "decode_only": paper_palette_role("accent"),
        "both": paper_palette_role("baseline"),
    }

    # Hero 1: per-trait 4-arm delta bars at the operating alpha.
    fig, axes = plt.subplots(1, len(traits), figsize=(4 * len(traits), 3.4), squeeze=False)
    for ax, trait in zip(axes[0], traits, strict=True):
        rec = headline["per_trait"][trait]
        a0 = rec.get("operating_alpha")
        ax.set_title(f"{trait} (alpha={a0:g})" if a0 is not None else f"{trait} (no op. alpha)")
        if a0 is None or "deltas_at_operating_alpha" not in rec:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no passing\noperating alpha", ha="center", va="center")
            continue
        arms = list(STEERED_ARMS)
        pts = [rec["deltas_at_operating_alpha"][arm]["delta"] for arm in arms]
        errs = np.array(
            [
                _err(p, *rec["deltas_at_operating_alpha"][arm]["ci95"])
                for p, arm in zip(pts, arms, strict=True)
            ]
        ).T
        ax.bar(arms, pts, yerr=errs, color=[arm_colors[a] for a in arms], capsize=3)
        ax.axhline(0.0, lw=0.8, color="black")
        ax.set_ylabel("judged-score shift vs neither")
        ax.tick_params(axis="x", rotation=20)
    p = fig_dir / "hero_deltas_operating_alpha.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))

    # Hero 2: f_d / f_p fractions with frozen CIs.
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(traits))
    width = 0.35
    for off, (name, color_role) in enumerate((("f_d", "accent"), ("f_p", "primary"))):
        pts, lows, highs, xs = [], [], [], []
        for i, trait in enumerate(traits):
            rec = headline["per_trait"][trait]
            fr = rec.get("fractions", {}).get(name)
            if fr is None:
                continue
            pts.append(fr["point"])
            lo, hi = fr["ci95_frozen"]
            e = _err(fr["point"], lo, hi)
            lows.append(e[0])
            highs.append(e[1])
            xs.append(x[i] + (off - 0.5) * width)
        if pts:
            ax.bar(
                xs,
                pts,
                width=width,
                yerr=np.array([lows, highs]),
                color=paper_palette_role(color_role),
                capsize=3,
                label=name,
            )
    for y, style in ((0.0, "-"), (LATTICE_LO, "--"), (LATTICE_HI, "--"), (1.0, ":")):
        ax.axhline(y, lw=0.8, ls=style, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(traits)
    ax.set_ylabel("fraction of both-arm shift")
    ax.legend()
    p = fig_dir / "hero_fractions.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))

    # Exploratory: full alpha-ladder delta grid per trait.
    fig, axes = plt.subplots(1, len(traits), figsize=(4.4 * len(traits), 3.4), squeeze=False)
    for ax, trait in zip(axes[0], traits, strict=True):
        ladder = headline["per_trait"][trait]["alpha_ladder"]
        alphas = sorted(k for k in ladder if k != "neither")
        for arm in STEERED_ARMS:
            pts = [ladder[a][arm]["delta"] for a in alphas]
            errs = np.array(
                [_err(pt, *ladder[a][arm]["ci95"]) for pt, a in zip(pts, alphas, strict=True)]
            ).T
            ax.errorbar(
                [float(a) for a in alphas],
                pts,
                yerr=errs,
                marker="o",
                color=arm_colors[arm],
                label=arm,
                capsize=3,
            )
        ax.axhline(0.0, lw=0.8, color="black")
        ax.set_title(trait)
        ax.set_xlabel("alpha")
        ax.set_ylabel("judged-score shift")
    axes[0][-1].legend()
    p = fig_dir / "alpha_ladder_deltas.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))

    # Exploratory: coherence-rate heatmap (constrained layout — colorbar gotcha).
    rows = []
    row_labels = []
    for trait in traits:
        ladder = headline["per_trait"][trait]["alpha_ladder"]
        alphas = sorted(k for k in ladder if k != "neither")
        for arm in STEERED_ARMS:
            rows.append([ladder[a][arm]["coherence_rate"] for a in alphas])
            row_labels.append(f"{trait}/{arm}")
    fig, ax = plt.subplots(figsize=(5, 0.4 * len(rows) + 1.2), layout="constrained")
    im = ax.imshow(np.array(rows), vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels(alphas)
    ax.set_xlabel("alpha")
    fig.colorbar(im, ax=ax, label="coherent-draw rate")
    p = fig_dir / "coherence_heatmap.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    written.append(str(p))
    return written


# ── main ──────────────────────────────────────────────────────────────


def run_analysis(graded: dict, b_boot: int = B_BOOT, seed: int = BOOT_SEED) -> dict:
    per_item = graded["per_item"]
    scores = question_scores(per_item)
    scores_coh = question_scores(per_item, coherent_only=True)
    coher = coherence_fractions(per_item)
    traits = sorted(scores)
    headline: dict = {"per_trait": {}, "per_trait_coherent_only": {}}
    for trait in traits:
        alphas = sorted({a for (arm, a) in scores[trait] if a is not None})
        qids = sorted(scores[trait][("neither", None)])
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(qids), size=(b_boot, len(qids)))
        headline["per_trait"][trait] = analyze_trait(
            trait, scores[trait], coher[trait], alphas, qids, idx
        )
        # Coherent-only sensitivity read (selection-conditioned — labeled).
        try:
            rng2 = np.random.default_rng(seed)
            idx2 = rng2.integers(0, len(qids), size=(b_boot, len(qids)))
            headline["per_trait_coherent_only"][trait] = analyze_trait(
                trait, scores_coh[trait], coher[trait], alphas, qids, idx2
            )
        except (AssertionError, KeyError) as exc:
            headline["per_trait_coherent_only"][trait] = {
                "skipped": f"coherent-only read incomplete: {exc!r}"
            }
        # Arm-asymmetric-degradation flag at the operating alpha (plan §6).
        rec = headline["per_trait"][trait]
        a0 = rec.get("operating_alpha")
        if a0 is not None:
            ladder = rec["alpha_ladder"][f"{a0:g}"]
            gap = ladder["both"]["coherence_rate"] - ladder["decode_only"]["coherence_rate"]
            rec["decode_coherence_asymmetry_flag"] = gap >= COHERENCE_ASYMMETRY_PTS
            rec["decode_minus_both_coherence_gap"] = -gap
    headline["k1_manipulation_check"] = {
        "passed": any(
            r.get("manipulation_check_any_alpha", False) for r in headline["per_trait"].values()
        ),
        "criterion": (
            f">= 1 trait with Delta_both >= {MANIPULATION_FLOOR:g} and question-bootstrap "
            "CI excluding 0 at some alpha (plan §7 K1); on failure the lattice is not run "
            "and the clean-result reports steering-ineffectiveness"
        ),
    }
    headline["coherent_only_label"] = (
        "selection-conditioned sensitivity read (coherence filtering selects on an "
        "outcome; the unfiltered read above is PRIMARY)"
    )
    headline["ceiling_diagnostic_frac_gt90"] = ceiling_diagnostic(per_item)
    headline["bootstrap"] = {"B": b_boot, "seed": seed, "unit": "question (cluster)"}
    headline["lattice"] = {
        "thresholds": [LATTICE_LO, LATTICE_HI],
        "primary_ci": "frozen-at-alpha",
        "note": "selection-inherited re-classification reported per trait",
    }
    return headline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--graded", type=Path, default=Path("eval_results/issue_1769/judge/graded_scores.json")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("eval_results/issue_1769/analysis/headline.json")
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_1769"))
    ap.add_argument("--b-boot", type=int, default=B_BOOT)
    ap.add_argument("--no-figures", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    graded = json.loads(args.graded.read_text())
    headline = run_analysis(graded, b_boot=args.b_boot)
    if headline["k1_manipulation_check"]["passed"]:
        logger.info("[analysis] K1 manipulation check PASSED")
    else:
        logger.warning(
            "[analysis] K1 manipulation check FAILED for ALL traits — lattice reads are "
            "steering-ineffectiveness reports, not timing findings (plan §7 kill criterion 1)"
        )
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
    headline["repro"] = {
        "git_commit": sha,
        "graded_scores": str(args.graded),
        "numpy": np.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(headline, indent=2))
    tmp.replace(args.out)
    logger.info("[analysis] wrote %s", args.out)
    if not args.no_figures:
        written = make_figures(headline, args.fig_dir)
        logger.info("[analysis] figures: %s", written)
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()

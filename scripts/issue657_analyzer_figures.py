#!/usr/bin/env python3
"""Issue #657 analyzer clean-result figures (blog style). CPU, off-pod.

Reads eval_results/issue_657/{alignment_predictor.json, per_behavior/*.json,
summary.json} + the steering_effect_by_layer_*.json K2 files and renders the
clean-result hero + supporting figures into figures/issue_657/.

One figure per finding:
  fig_k2_steering.png      -- per-direction K2 add-elicits effect at headline layer
  fig_h1_base_rate.png     -- DV-(a): alignment vs base rate per behavior (#623 generalization)
  fig_h3_forest.png        -- H3: doubly-partialled vs raw rho per behavior + CIs (beat-the-prior)
  fig_reliability_band.png -- H3 reliability-sensitivity band (primary behaviors)

Plain-English condition names; blog style; error bars; no annotation overlays.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-657")
RES = ROOT / "eval_results/issue_657"
FIG = ROOT / "figures/issue_657"

# Plain-English behavior labels (reader-facing everywhere).
BEH_LABEL = {
    "sycophancy": "Sycophancy",
    "refusal": "Refusal",
    "marker": "Marker (※ token)",
    "em": "Emergent misalignment",
}


def _load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    import matplotlib

    matplotlib.use("Agg")
    import sys

    import matplotlib.pyplot as plt

    sys.path.insert(0, str(ROOT / "src"))
    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    FIG.mkdir(parents=True, exist_ok=True)

    PRIMARY = paper_palette_role("primary")
    BASELINE = paper_palette_role("baseline")
    CONTROL = paper_palette_role("control")
    ACCENT = paper_palette_role("accent")
    NEUTRAL = paper_palette_role("neutral")

    pred = _load("alignment_predictor.json")
    per = {
        p.stem.replace("_alignment_predictor", ""): json.loads(p.read_text())
        for p in sorted((RES / "per_behavior").glob("*_alignment_predictor.json"))
    }
    if not per:
        # fall back to the per_behavior dir layout
        per = {
            p.stem: json.loads(p.read_text()) for p in sorted((RES / "per_behavior").glob("*.json"))
        }

    # ---- Figure 1: K2 steering effect (per direction) ----
    # Honest framing (round 2): the judge scale is 0-100, so the headline mean
    # effects (refusal +0.96, EM +0.07) are ~1% of the scale -- a weak sign-of-
    # life gate, NOT strong steering. Plot the per-alpha effects (which sign-
    # flip across steering strengths) as points alongside the mean, on a
    # symmetric small-magnitude axis so the reader sees both the small size and
    # the non-monotonicity. Read per-alpha + baseline from steering_probe_*.json.
    k2_rows = []
    for beh in ("refusal", "marker", "em"):
        probe_f = RES / f"steering_probe_{beh}.json"
        eff_f = RES / f"steering_effect_by_layer_{beh}.json"
        if not eff_f.exists():
            continue
        eff = json.loads(eff_f.read_text())
        layer = eff["headline_layer"]
        mean_eff = eff["headline_layer_effect"]
        passed = eff["k2_pass"]
        per_alpha, alphas, baseline = None, None, None
        if probe_f.exists():
            pb = json.loads(probe_f.read_text())
            cell = pb.get("per_layer", {}).get(str(layer), {})
            per_alpha = cell.get("per_alpha_effect")
            alphas = cell.get("alphas_coeff")
            baseline = pb.get("baseline_mean_trait_score")
        k2_rows.append((beh, layer, mean_eff, passed, per_alpha, alphas, baseline))
    if k2_rows:
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        labels = [f"{BEH_LABEL[b]}\n(layer {ly})" for b, ly, *_ in k2_rows]
        ys = np.arange(len(k2_rows))
        for y, (_, _, mean_eff, passed, per_alpha, _alphas, _base) in zip(ys, k2_rows):
            col = PRIMARY if passed else NEUTRAL
            # per-alpha points (the steering sweep): show the sign-flipping
            if per_alpha:
                ax.scatter(
                    per_alpha,
                    [y] * len(per_alpha),
                    color=NEUTRAL,
                    edgecolors="#555",
                    linewidths=0.8,
                    s=34,
                    zorder=3,
                    label="per-strength effect (3 alphas)" if y == ys[0] else None,
                )
            # mean marker (the K2 statistic)
            ax.scatter(
                [mean_eff],
                [y],
                marker="D",
                color=col,
                edgecolors="#222",
                linewidths=0.8,
                s=70,
                zorder=4,
                label="mean (K2 statistic)" if y == ys[0] else None,
            )
            tag = "passes (weak gate)" if passed else "FAILS (no steering)"
            ax.text(
                4.7,
                y,
                tag,
                va="center",
                ha="left",
                fontsize=8,
                color=PRIMARY if passed else BASELINE,
            )
        ax.axvline(0.0, color="#888", linewidth=0.9)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.6, len(k2_rows) - 0.4 + 0.6)
        ax.invert_yaxis()
        ax.set_xlim(-4.5, 8.0)
        ax.set_xlabel("add-direction steering effect (judge points; full scale is 0–100)")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        set_title_subtitle(
            ax,
            "Causal sanity check: each effect is ≪ 1% of the 0–100 judge scale",
            "K2 is a weak sign-of-life gate; refusal/EM means sit ≈1 point and sign-flip across strengths",
        )
        savefig_paper(fig, "issue_657/fig_k2_steering", dir="figures/")
        plt.close(fig)
        print("wrote fig_k2_steering")

    # ---- Figure 2: H1 / DV-(a) alignment vs base rate per behavior ----
    # Schema: dv_a_base_rate = {raw_rho, ci_lo, ci_hi, n, base_rate_source}.
    # The marker DV-(a) is computed over only n=3 personas (degenerate, rho=1.0
    # CI[1,1]) — excluded from this figure as uninterpretable (annotated in caption).
    da_rows = []
    for beh, res in per.items():
        da = res.get("dv_a_base_rate", {})
        rho = da.get("raw_rho", da.get("rho"))
        n = da.get("n")
        if rho is None or (n is not None and n < 5):
            continue  # drop degenerate small-n DV-(a) reads (marker n=3)
        anchor = res.get("replication_anchor") or {}
        da_rows.append((beh, rho, da.get("ci_lo"), da.get("ci_hi"), anchor.get("target_rho"), n))
    da_rows = [r for r in da_rows if r[1] is not None]
    if da_rows:
        order = ["sycophancy", "refusal", "marker", "em"]
        da_rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ys = np.arange(len(da_rows))
        rhos = np.array([r[1] for r in da_rows], dtype=float)
        los = np.array([r[2] if r[2] is not None else np.nan for r in da_rows], dtype=float)
        his = np.array([r[3] if r[3] is not None else np.nan for r in da_rows], dtype=float)
        xerr = np.vstack([rhos - los, his - rhos])
        xerr = np.clip(np.nan_to_num(xerr, nan=0.0), 0.0, None)
        ax.errorbar(
            rhos,
            ys,
            xerr=xerr,
            fmt="o",
            color=PRIMARY,
            capsize=3,
            markersize=7,
            label="alignment vs base rate (Spearman ρ, 95% CI)",
        )
        # syco anchor target
        for y, r in zip(ys, da_rows):
            if r[4] is not None:
                ax.scatter(
                    [r[4]],
                    [y],
                    marker="D",
                    color=ACCENT,
                    zorder=5,
                    s=45,
                    label="#623 sycophancy anchor (ρ=0.726)" if y == ys[0] else None,
                )
        ax.axvline(0.0, color="#888", linewidth=0.9)
        ax.axvline(
            0.5,
            color=BASELINE,
            linewidth=0.8,
            linestyle="--",
            label="H1 pre-registered threshold (ρ=0.5)",
        )
        ax.set_yticks(ys)
        ax.set_yticklabels([BEH_LABEL[r[0]] for r in da_rows])
        ax.set_ylim(-0.6, len(da_rows) - 0.4)
        ax.invert_yaxis()
        ax.set_xlabel("Spearman ρ (alignment, persona base rate)")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=7.5, frameon=False
        )
        set_title_subtitle(
            ax,
            "Does alignment predict a persona's own base rate?",
            "DV-(a) / H1 — the #623 sycophancy→base-rate finding, tested across four behaviors",
        )
        ax.set_xlim(-0.7, 1.0)
        savefig_paper(fig, "issue_657/fig_h1_base_rate", dir="figures/")
        plt.close(fig)
        print("wrote fig_h1_base_rate")

    # ---- Figure 3: H3 forest (doubly-partialled vs raw, leakage-in-scope behaviors) ----
    rows = []
    for beh, res in per.items():
        lk = res.get("leakage_bake_off", res)
        if not lk.get("in_scope", res.get("leakage_in_scope", False)):
            continue
        dp = lk.get("doubly_partialled_rho", res.get("doubly_partialled_rho"))
        sp = lk.get("singly_partialled_rho", res.get("singly_partialled_rho"))
        raw = lk.get("raw_rho", res.get("raw_rho"))
        ci = lk.get("h3_doubly_partialled_rho_ci", res.get("h3_doubly_partialled_rho_ci", {}))
        if dp is None:
            continue
        rows.append(
            (beh, raw, sp, dp, ci.get("ci_lo"), ci.get("ci_hi"), res.get("is_primary_h3", False))
        )
    if rows:
        order = ["sycophancy", "refusal", "em"]
        rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        ys = np.arange(len(rows))
        dps = np.array([r[3] for r in rows], dtype=float)
        los = np.array([r[4] if r[4] is not None else np.nan for r in rows], dtype=float)
        his = np.array([r[5] if r[5] is not None else np.nan for r in rows], dtype=float)
        xerr = np.vstack([dps - los, his - dps])
        xerr = np.clip(np.nan_to_num(xerr, nan=0.0), 0.0, None)
        ax.errorbar(
            dps,
            ys,
            xerr=xerr,
            fmt="o",
            color=PRIMARY,
            capsize=4,
            markersize=8,
            label="doubly-partialled ρ (vs base prior + geometry prior), 95% CI",
        )
        ax.scatter(
            [r[1] for r in rows],
            ys,
            marker="x",
            color=BASELINE,
            s=55,
            zorder=4,
            label="raw ρ (no controls)",
        )
        ax.scatter(
            [r[2] for r in rows],
            ys,
            marker="s",
            facecolors="none",
            edgecolors=CONTROL,
            linewidths=1.3,
            s=42,
            zorder=4,
            label="singly-partialled ρ (vs base prior only)",
        )
        ax.axvline(0.0, color="#444", linewidth=1.0)
        ax.set_yticks(ys)
        ax.set_yticklabels(
            [f"{BEH_LABEL[r[0]]}\n({'primary' if r[6] else 'secondary'})" for r in rows]
        )
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.invert_yaxis()
        ax.set_xlabel("Spearman ρ (alignment, held-out leakage Δ)")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=7, frameon=False
        )
        set_title_subtitle(
            ax,
            "Does alignment beat the base prior on held-out leakage? (H3)",
            "doubly-partialled ρ with CI crossing 0 = alignment adds nothing the prior doesn't",
        )
        savefig_paper(fig, "issue_657/fig_h3_forest", dir="figures/")
        plt.close(fig)
        print("wrote fig_h3_forest")

    # ---- Figure 4: reliability-sensitivity band (primary in-scope behaviors) ----
    band_rows = []
    for beh, res in per.items():
        lk = res.get("leakage_bake_off", res)
        rb = lk.get("reliability_band", res.get("reliability_band", {}))
        pts = rb.get("points", {})
        if not pts or not res.get("is_primary_h3", False):
            continue
        band_rows.append((beh, pts, lk.get("band_verdict", {})))
    if band_rows:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        labels_order = ["R_high", "R_expected", "R_observed", "R_low"]
        xlab = {
            "R_high": "high\n(least disatt.)",
            "R_expected": "expected",
            "R_observed": "observed\n(R=1)",
            "R_low": "low\n(most disatt.)",
        }
        cmap = [PRIMARY, ACCENT, CONTROL, NEUTRAL]
        for i, (beh, pts, bv) in enumerate(band_rows):
            xs, ys2 = [], []
            for lab in labels_order:
                if lab in pts:
                    xs.append(lab)
                    ys2.append(pts[lab]["doubly_partialled_rho"])
            ax.plot(
                range(len(xs)),
                ys2,
                marker="o",
                color=cmap[i % len(cmap)],
                label=f"{BEH_LABEL[beh]} ({bv.get('verdict', '?')})",
            )
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([xlab.get(x, x) for x in xs])
        ax.axhline(0.0, color="#444", linewidth=1.0)
        ax.set_ylabel("doubly-partialled ρ (alignment, leakage Δ)")
        ax.set_xlabel("assumed reliability of the base-prior covariate")
        ax.legend(loc="best", fontsize=7.5)
        set_title_subtitle(
            ax,
            "Reliability-sensitivity band on the noisy base prior",
            "ρ must stay above 0 across the whole band; clearing only at the optimistic end = indeterminate",
        )
        savefig_paper(fig, "issue_657/fig_reliability_band", dir="figures/")
        plt.close(fig)
        print("wrote fig_reliability_band")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Issue #657 analyzer clean-result figures (blog style). CPU, off-pod.

Reads eval_results/issue_657/{alignment_predictor.json, per_behavior/*.json,
summary.json} + the steering_effect_by_layer_*.json K2 files and renders the
clean-result hero + supporting figures into figures/issue_657/.

One figure per finding:
  fig_k2_steering.png      -- per-direction add-elicits causal gate effect at headline layer
  fig_h1_base_rate.png     -- alignment vs persona base rate per behavior (parent generalization)
  fig_h3_forest.png        -- doubly-partialled vs raw rho per behavior + CIs (beat-the-prior)
  fig_reliability_band.png -- reliability-sensitivity band (primary behaviors)
  fig_layer_sweep.png      -- per-layer steering sign-flip (gate layer vs ρ readout layer)
  fig_lobo_pooled.png      -- pooled higher-N beat-the-prior incremental-variance read

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
                label="mean steering effect" if y == ys[0] else None,
            )
            tag = "passes (weak gate)" if passed else "no steering effect"
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
            "a weak sign-of-life gate; refusal/EM means sit ≈1 point and sign-flip across strengths",
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
                    label="parent sycophancy anchor (ρ=0.726)" if y == ys[0] else None,
                )
        ax.axvline(0.0, color="#888", linewidth=0.9)
        ax.axvline(
            0.5,
            color=BASELINE,
            linewidth=0.8,
            linestyle="--",
            label="ρ = 0.5 generalization bar",
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
            "the parent sycophancy→base-rate finding, tested across four behaviors",
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
            "Does alignment beat the base prior on held-out leakage?",
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
        verdict_word = {
            "INDETERMINATE": "inconclusive",
            "TIGHTER-INDETERMINATE": "inconclusive",
            "NULL": "null",
            "CONFIRMED": "beats prior",
        }
        for i, (beh, pts, bv) in enumerate(band_rows):
            xs, ys2 = [], []
            for lab in labels_order:
                if lab in pts:
                    xs.append(lab)
                    ys2.append(pts[lab]["doubly_partialled_rho"])
            vw = verdict_word.get(bv.get("verdict", ""), "inconclusive")
            ax.plot(
                range(len(xs)),
                ys2,
                marker="o",
                color=cmap[i % len(cmap)],
                label=f"{BEH_LABEL[beh]} — {vw}",
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

    # ---- Figure 5: per-layer steering sweep (gate layer vs ρ readout layer) ----
    # The causal gate passes at layer 7 (refusal) / 27 (EM), but the predictive
    # ρ is read at layer 14. Refusal reverses sign across layers, so at the L14
    # readout it steers negative -- the layer-mismatch finding. Read per-layer
    # effects from steering_effect_by_layer_*.json.
    sweep = {}
    for beh in ("refusal", "em"):
        eff_f = RES / f"steering_effect_by_layer_{beh}.json"
        if not eff_f.exists():
            continue
        eff = json.loads(eff_f.read_text())
        pl = eff.get("per_layer", {})
        if not pl:
            continue
        sweep[beh] = (sorted(int(k) for k in pl), pl, eff.get("headline_layer"))
    if sweep:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        col = {"refusal": PRIMARY, "em": ACCENT}
        readout_layer = 14
        for beh, (layers, pl, gate_layer) in sweep.items():
            xs = list(range(len(layers)))
            ys2 = [pl[str(ly)] for ly in layers]
            ax.plot(
                xs,
                ys2,
                marker="o",
                color=col.get(beh, NEUTRAL),
                label=f"{BEH_LABEL[beh]} (gate passes at layer {gate_layer})",
            )
            ax.set_xticks(xs)
            ax.set_xticklabels([str(ly) for ly in layers])
            # mark the ρ readout layer if present
            if readout_layer in layers:
                ri = layers.index(readout_layer)
                ax.scatter(
                    [ri],
                    [pl[str(readout_layer)]],
                    s=120,
                    facecolors="none",
                    edgecolors="#222",
                    linewidths=1.4,
                    zorder=6,
                )
        ax.axhline(0.0, color="#444", linewidth=1.0)
        if readout_layer in next(iter(sweep.values()))[0]:
            ri = next(iter(sweep.values()))[0].index(readout_layer)
            ax.axvline(ri, color="#888", linewidth=0.8, linestyle=":")
            ax.text(
                ri,
                ax.get_ylim()[1],
                " ρ read here (layer 14)",
                va="top",
                ha="left",
                fontsize=8,
                color="#555",
            )
        ax.set_xlabel("layer")
        ax.set_ylabel("steering effect (judge points, 0–100 scale)")
        ax.legend(loc="lower left", fontsize=8, frameon=False)
        set_title_subtitle(
            ax,
            "The refusal direction reverses sign between the gate layer and the ρ readout layer",
            "the gate passes early (layer 7); by the layer-14 readout refusal steers negative",
        )
        savefig_paper(fig, "issue_657/fig_layer_sweep", dir="figures/")
        plt.close(fig)
        print("wrote fig_layer_sweep")

    # ---- Figure 6: pooled higher-N beat-the-prior incremental-variance read ----
    # The per-behavior leakage reads are power-limited at ~23 bystanders. The
    # pooled leave-one-behavior-out read lifts the sample to 399 cells. Plot the
    # incremental-variance point + CI (the registered beat-the-prior statistic)
    # against the per-behavior reads, with a zero line: a CI straddling zero is
    # the inconclusive verdict that does not flip with the larger sample.
    lobo = _load("lobo_pooled.json")
    if lobo and lobo.get("lobo", {}).get("in_scope"):
        gv = lobo["lobo"]["grouped_cv"]
        n_pooled = lobo["lobo"]["n_pooled_cells"]
        # per-behavior doubly-partialled rho (the power-limited reads being pooled)
        per_beh_dp = []
        for beh in ("sycophancy", "refusal", "em"):
            res = per.get(beh)
            if not res:
                continue
            lk = res.get("leakage_bake_off", res)
            dp = lk.get("doubly_partialled_rho")
            ci = lk.get("h3_doubly_partialled_rho_ci", {})
            if dp is not None:
                per_beh_dp.append((beh, dp, ci.get("ci_lo"), ci.get("ci_hi")))
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        rows_lobo = []
        # per-behavior reads first (n≈23 each), then the pooled read
        for beh, dp, lo, hi in per_beh_dp:
            rows_lobo.append((f"{BEH_LABEL[beh]}\n(per-behavior, n≈23)", dp, lo, hi, False))
        gv_dp = gv["doubly_partialled_rho"]
        rows_lobo.append(
            (
                f"Pooled\n(n={n_pooled} cells)",
                gv_dp["point"],
                gv_dp["ci_lo"],
                gv_dp["ci_hi"],
                True,
            )
        )
        ys = np.arange(len(rows_lobo))
        pts = np.array([r[1] for r in rows_lobo], dtype=float)
        los = np.array([r[2] if r[2] is not None else np.nan for r in rows_lobo], dtype=float)
        his = np.array([r[3] if r[3] is not None else np.nan for r in rows_lobo], dtype=float)
        xerr = np.clip(np.nan_to_num(np.vstack([pts - los, his - pts]), nan=0.0), 0.0, None)
        for y, (lab, pt, lo, hi, is_pooled) in zip(ys, rows_lobo):
            c = ACCENT if is_pooled else PRIMARY
            ax.errorbar(
                [pt],
                [y],
                xerr=[[xerr[0][y]], [xerr[1][y]]],
                fmt="o",
                color=c,
                capsize=4,
                markersize=9 if is_pooled else 7,
            )
        ax.axvline(0.0, color="#444", linewidth=1.0)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows_lobo])
        ax.set_ylim(-0.7, len(rows_lobo) - 0.3)
        ax.invert_yaxis()
        ax.set_xlabel("doubly-partialled ρ (alignment, held-out leakage Δ), 95% CI")
        set_title_subtitle(
            ax,
            "Pooling to 399 cells tightens the beat-the-prior read but does not flip it",
            "the pooled CI still straddles zero — the per-behavior inconclusive is not a low-power artifact",
        )
        savefig_paper(fig, "issue_657/fig_lobo_pooled", dir="figures/")
        plt.close(fig)
        print("wrote fig_lobo_pooled")


if __name__ == "__main__":
    main()

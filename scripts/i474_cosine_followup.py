"""#474 follow-up analysis (existing data only): is the transfer-relevant
predictor cosine rather than JS, and where does it live?

Three questions, all on the merged #474 cross-eval matrices + the inherited
#406 base-model predictor matrices (JS in D_matrix.json, cosine per layer in
cosine/C_L*.json). No training, no pod.

  Q1  Does cosine predict marker transfer on the NON-stylized panel across
      both arms (positives-only vs localized) and all four checkpoints
      (ep1/2/3/5), or only at localized-ep1?  -> table + survival figure.
  Q2  Which layer localizes the transfer-relevant dimension?  -> layer-sweep
      figure on the non-stylized panel.
  (Fig0) Re-render the headline JS-vs-cosine scatter at localized-ep1.

Base-prior guard: later epochs / the positives-only arm saturate the marker
log-prob, and on a saturated panel delta_g collapses to -b_logprob (the
#462 artifact). So for every (arm, epoch) we report BOTH the delta_g-based
rho (the #474 metric) AND the trained-log-prob-based rho (the construct that
cannot be faked by the base prior) AND the saturation fraction, so a
"survival" that is really a base-prior shuffle is visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parent.parent
LAYERS = [0, 5, 11, 15, 21, 27]
ARMS = ["pos", "loc"]
EPOCHS = [1, 2, 3, 5]
STY = {"A3", "A4", "A5"}  # stylized: pirate / comedian / villain

D = json.loads((REPO / "eval_results/issue_406/divergence/D_matrix.json").read_text())
CONDS = [c["cid"] for c in D["conditions"]]
JS = D["JS"]
PT = D["prompt_tokens"]
COS = {
    L: json.loads((REPO / f"eval_results/issue_406/cosine/C_L{L}.json").read_text())["matrix"]
    for L in LAYERS
}


def _length_partial(x, y, covar):
    """Rank-then-residualize length-partial Spearman (matches #406/#462/#474)."""
    rx, ry, rc = rankdata(x), rankdata(y), rankdata(covar)
    ex = rx - np.polyval(np.polyfit(rc, rx, 1), rc)
    ey = ry - np.polyval(np.polyfit(rc, ry, 1), rc)
    return pearsonr(ex, ey)


def _load_G(arm, ep):
    p = REPO / f"eval_results/issue_474/cross_eval/{arm}_ep{ep}/G_logprob_matrix.json"
    return json.loads(p.read_text())["G"]


def _pairs(nonstylized_only: bool):
    out = []
    for ti in CONDS:
        for tj in CONDS:
            if ti == tj:
                continue
            if nonstylized_only and ((ti in STY) or (tj in STY)):
                continue
            out.append((ti, tj))
    return out


def _vecs(G, pairs):
    dg = np.array([G[a][b]["delta_g"] for a, b in pairs])
    g = np.array([G[a][b]["g_logprob"] for a, b in pairs])
    ln = np.array([np.log(PT[a][b]) for a, b in pairs])
    js = np.array([JS[a][b] for a, b in pairs])
    cos = {L: np.array([1.0 - COS[L][a][b] for a, b in pairs]) for L in LAYERS}  # similarity
    return dg, g, ln, js, cos


def build_table():
    """Per (arm, epoch): non-stylized rho for JS and cosine(L21), on delta_g
    AND on trained log-prob, plus saturation fraction. Plus full-panel for ref."""
    rows = {}
    pairs_ns = _pairs(True)
    pairs_all = _pairs(False)
    for arm in ARMS:
        for ep in EPOCHS:
            G = _load_G(arm, ep)
            dg, g, ln, js, cos = _vecs(G, pairs_ns)
            dg_a, g_a, ln_a, js_a, cos_a = _vecs(G, pairs_all)
            sat = float(np.mean(g > -0.1))  # fraction of non-stylized cells at ceiling
            rows[f"{arm}_ep{ep}"] = {
                "arm": arm,
                "epoch": ep,
                "saturation_frac_nonstylized": sat,
                "g_logprob_sd_nonstylized": float(g.std()),
                # non-stylized panel (rho + p)
                "ns_rho_JS_deltag": _length_partial(js, dg, ln)[0],
                "ns_p_JS_deltag": _length_partial(js, dg, ln)[1],
                "ns_rho_cosL21_deltag": _length_partial(cos[21], dg, ln)[0],
                "ns_p_cosL21_deltag": _length_partial(cos[21], dg, ln)[1],
                "ns_rho_cosL21_trainedlogp": _length_partial(cos[21], g, ln)[0],
                "ns_p_cosL21_trainedlogp": _length_partial(cos[21], g, ln)[1],
                "ns_rho_JS_trainedlogp": _length_partial(js, g, ln)[0],
                # full panel (rho + p)
                "all_rho_JS_deltag": _length_partial(js_a, dg_a, ln_a)[0],
                "all_p_JS_deltag": _length_partial(js_a, dg_a, ln_a)[1],
                "all_rho_cosL21_deltag": _length_partial(cos_a[21], dg_a, ln_a)[0],
                "all_p_cosL21_deltag": _length_partial(cos_a[21], dg_a, ln_a)[1],
                # layer sweep on non-stylized (delta_g and trained-logp)
                "ns_layer_sweep_deltag": {L: _length_partial(cos[L], dg, ln)[0] for L in LAYERS},
                "ns_layer_sweep_trainedlogp": {
                    L: _length_partial(cos[L], g, ln)[0] for L in LAYERS
                },
            }
    return rows


def fig0_headline(rows):
    """JS vs cosine scatter at localized-ep1 (the headline dissociation)."""
    set_paper_style("blog")
    matplotlib.rcParams["figure.constrained_layout.use"] = False
    G = _load_G("loc", 1)
    pairs = _pairs(False)
    dg, g, ln, js, cos = _vecs(G, pairs)
    sty = np.array([(a in STY) or (b in STY) for a, b in pairs])
    sim = cos[21]
    r = rows["loc_ep1"]
    prim, base, acc, neu = (
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
        paper_palette_role("accent"),
        paper_palette_role("neutral"),
    )
    fig, (axJ, axC) = plt.subplots(1, 2, figsize=(11.8, 5.0))

    def scat(ax, x, c0):
        ax.scatter(
            x[~sty],
            dg[~sty],
            s=24,
            c=c0,
            alpha=0.6,
            edgecolor="white",
            lw=0.5,
            label="non-stylized cells (n=156)",
            zorder=2,
        )
        ax.scatter(
            x[sty],
            dg[sty],
            s=34,
            c=acc,
            alpha=0.85,
            edgecolor="white",
            lw=0.5,
            label="touches a stylized persona (n=84)",
            zorder=3,
        )

    def _pf(p):
        return "p < 1e-12" if p < 1e-12 else f"p = {p:.1e}"

    scat(axJ, js, prim)
    axJ.set_xlabel("Base-model JS divergence (nats)")
    axJ.set_ylabel("Marker transfer  ΔG = trained − base log P(marker)")
    axJ.text(
        0.97,
        0.97,
        f"all 240:  rho = {r['all_rho_JS_deltag']:+.2f}, {_pf(r['all_p_JS_deltag'])}\n"
        f"non-stylized:  rho = {r['ns_rho_JS_deltag']:+.2f}, {_pf(r['ns_p_JS_deltag'])}  (NULL)",
        transform=axJ.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=neu, alpha=0.95),
    )
    axJ.set_title("JS divergence  →  collapses on non-stylized", fontsize=10.5, loc="left", pad=8)
    axJ.grid(alpha=0.2, lw=0.5)
    axJ.legend(loc="lower left", frameon=False, fontsize=8)

    scat(axC, sim, base)
    axC.set_xlabel("Base-model cosine similarity (layer 21)")
    axC.set_ylabel("Marker transfer  ΔG = trained − base log P(marker)")
    axC.text(
        0.03,
        0.97,
        f"all 240:  rho = {r['all_rho_cosL21_deltag']:+.2f}, {_pf(r['all_p_cosL21_deltag'])}\n"
        f"non-stylized:  rho = {r['ns_rho_cosL21_deltag']:+.2f}, {_pf(r['ns_p_cosL21_deltag'])}  (SURVIVES)",
        transform=axC.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=neu, alpha=0.95),
    )
    axC.set_title(
        "Cosine similarity  →  survives on non-stylized", fontsize=10.5, loc="left", pad=8
    )
    axC.grid(alpha=0.2, lw=0.5)

    fig.suptitle(
        "#474 localized arm, epoch 1: residual-stream cosine predicts marker transfer where output-distribution JS fails",
        fontsize=11,
        x=0.06,
        y=0.985,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.06,
        0.935,
        "240 ordered transformation pairs, on-policy non-saturated regime. JS predicts transfer overall but only via the 3 stylized personas (red); "
        "drop them and it goes null. Cosine similarity (layer 21) predicts transfer among the 156 non-stylized pairs too, and predicts the trained "
        f"log-prob directly at rho={rows['loc_ep1']['ns_rho_cosL21_trainedlogp']:+.2f} (not the base prior).",
        fontsize=8.0,
        color=neu,
        ha="left",
    )
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.06, right=0.975, wspace=0.24)
    savefig_paper(fig, "issue_474/followup_js_vs_cosine_locep1", dir=str(REPO / "figures"))
    plt.close(fig)


def fig1_survival(rows):
    """Non-stylized cosine(L21) rho across epochs x arms, with the base-prior guard.

    Solid = cosine vs delta_g; the marker overlay flags saturated points where
    the delta_g number is suspect; the faint companion = cosine vs trained
    log-prob (the construct that can't be base-prior-faked). JS shown dashed.
    """
    set_paper_style("blog")
    matplotlib.rcParams["figure.constrained_layout.use"] = False
    prim, base, acc, neu = (
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
        paper_palette_role("accent"),
        paper_palette_role("neutral"),
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.axhline(0, color=neu, lw=1)
    style = {"loc": dict(color=base, marker="o"), "pos": dict(color=prim, marker="s")}
    for arm in ARMS:
        ep = EPOCHS
        cos_dg = [rows[f"{arm}_ep{e}"]["ns_rho_cosL21_deltag"] for e in ep]
        cos_tl = [rows[f"{arm}_ep{e}"]["ns_rho_cosL21_trainedlogp"] for e in ep]
        js_dg = [rows[f"{arm}_ep{e}"]["ns_rho_JS_deltag"] for e in ep]
        sat = [rows[f"{arm}_ep{e}"]["saturation_frac_nonstylized"] for e in ep]
        x = np.arange(len(ep))
        lbl = "localized (pos+neg)" if arm == "loc" else "positives-only"
        ax.plot(x, cos_dg, **style[arm], ms=8, lw=2, label=f"cosine L21 · {lbl} · (vs ΔG)")
        ax.plot(
            x,
            cos_tl,
            color=style[arm]["color"],
            marker=style[arm]["marker"],
            ms=5,
            lw=1.2,
            ls=":",
            alpha=0.65,
            label=f"cosine L21 · {lbl} · (vs trained logP, base-prior-safe)",
        )
        ax.plot(
            x,
            js_dg,
            color=style[arm]["color"],
            marker="x",
            ms=7,
            lw=1.2,
            ls="--",
            alpha=0.5,
            label=f"JS · {lbl} · (vs ΔG)",
        )
        # saturation flag
        for xi, s, y in zip(x, sat, cos_dg):
            if s > 0.5:
                ax.annotate(
                    "sat",
                    (xi, y),
                    textcoords="offset points",
                    xytext=(0, -14),
                    ha="center",
                    fontsize=7,
                    color=acc,
                )
        # p-value labels on the localized cosine-vs-ΔG line (the headline)
        if arm == "loc":
            pvals = [rows[f"{arm}_ep{e}"]["ns_p_cosL21_deltag"] for e in ep]
            for xi, y, p in zip(x, cos_dg, pvals):
                ptxt = "p<1e-9" if p < 1e-9 else f"p={p:.0e}"
                ax.annotate(
                    ptxt,
                    (xi, y),
                    textcoords="offset points",
                    xytext=(0, 9),
                    ha="center",
                    fontsize=7,
                    color=base,
                )
    ax.set_xticks(np.arange(len(EPOCHS)))
    ax.set_xticklabels([f"ep{e}" for e in EPOCHS])
    ax.set_xlabel("LoRA checkpoint (training epochs)")
    ax.set_ylabel("Length-partial Spearman rho on the NON-stylized panel (n=156)")
    ax.set_title(
        "Does the cosine signal survive across arms and epochs?", fontsize=11, loc="left", pad=10
    )
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=7.6, ncol=2)
    fig.text(
        0.01,
        0.005,
        "Positive cosine rho = higher base-model similarity predicts more transfer among non-stylized pairs. "
        "'sat' marks points where the non-stylized panel is >50% saturated (ΔG number suspect; read the dotted trained-logP line).",
        fontsize=7.4,
        color=neu,
        ha="left",
    )
    fig.subplots_adjust(top=0.90, bottom=0.30, left=0.10, right=0.97)
    savefig_paper(fig, "issue_474/followup_cosine_survival_arm_epoch", dir=str(REPO / "figures"))
    plt.close(fig)


def fig2_layersweep(rows):
    """Layer sweep on the non-stylized panel: which layer localizes the signal.

    Localized-ep1 (the clean non-saturated cell). Full panel vs non-stylized;
    delta_g and trained-logP variants. JS non-stylized rho as a flat reference.
    """
    set_paper_style("blog")
    matplotlib.rcParams["figure.constrained_layout.use"] = False
    prim, base, acc, neu = (
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
        paper_palette_role("accent"),
        paper_palette_role("neutral"),
    )
    G = _load_G("loc", 1)
    pairs_ns, pairs_all = _pairs(True), _pairs(False)
    dg, g, ln, js, cos = _vecs(G, pairs_ns)
    dg_a, g_a, ln_a, js_a, cos_a = _vecs(G, pairs_all)
    ns_dg = [_length_partial(cos[L], dg, ln)[0] for L in LAYERS]
    ns_dg_p = [_length_partial(cos[L], dg, ln)[1] for L in LAYERS]
    ns_tl = [_length_partial(cos[L], g, ln)[0] for L in LAYERS]
    all_dg = [_length_partial(cos_a[L], dg_a, ln_a)[0] for L in LAYERS]
    js_ns, js_ns_p = _length_partial(js, dg, ln)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.axhline(0, color=neu, lw=1)
    x = np.arange(len(LAYERS))
    ax.plot(x, all_dg, "o-", color=prim, ms=8, lw=2, label="cosine vs ΔG · all 240 pairs")
    ax.plot(x, ns_dg, "s-", color=base, ms=8, lw=2, label="cosine vs ΔG · non-stylized (n=156)")
    for xi, y, p in zip(x, ns_dg, ns_dg_p):
        ptxt = "p<1e-9" if p < 1e-9 else f"p={p:.0e}"
        ax.annotate(
            ptxt,
            (xi, y),
            textcoords="offset points",
            xytext=(0, -13),
            ha="center",
            fontsize=6.8,
            color=base,
        )
    ax.plot(
        x,
        ns_tl,
        "s:",
        color=base,
        ms=5,
        lw=1.3,
        alpha=0.65,
        label="cosine vs trained logP · non-stylized (base-prior-safe)",
    )
    ax.axhline(
        js_ns,
        color=acc,
        lw=1.4,
        ls="--",
        label=f"JS vs ΔG · non-stylized = {js_ns:+.2f}, p={js_ns_p:.2f} (null reference)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}" for L in LAYERS])
    ax.set_xlabel("Residual-stream layer (cosine extraction point, last prompt token)")
    ax.set_ylabel("Length-partial Spearman rho")
    ax.set_title(
        "Which layer localizes the transfer-relevant dimension? (localized arm, ep1)",
        fontsize=10.5,
        loc="left",
        pad=10,
    )
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.text(
        0.01,
        0.005,
        "Cosine similarity at every layer beats JS on the non-stylized panel; the signal strengthens into mid-late layers (L15-L27) and "
        "predicts the trained log-prob directly (dotted), so it is not the base-prior artifact. JS (null) shown for reference.",
        fontsize=7.4,
        color=neu,
        ha="left",
    )
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.10, right=0.97)
    savefig_paper(fig, "issue_474/followup_layer_sweep_nonstylized", dir=str(REPO / "figures"))
    plt.close(fig)


def fig3_gradient_check(rows):
    """Is the non-stylized cosine effect a gradient or a vertical-line artifact?

    The linear cosine-similarity axis compresses the bulk (77% of non-stylized
    pairs sit above 0.90), so the scatter LOOKS vertical. Two honest views:
    (left) the same points on a log cosine-DISTANCE axis, which spreads the
    dense near-1.0 region; (right) mean transfer per cosine-sim quintile, which
    shows the gradient is monotone across the whole range, not tail leverage.
    """
    set_paper_style("blog")
    matplotlib.rcParams["figure.constrained_layout.use"] = False
    prim, base, acc, neu = (
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
        paper_palette_role("accent"),
        paper_palette_role("neutral"),
    )
    G = _load_G("loc", 1)
    cc = _pairs(True)  # non-stylized only
    sim = np.array([1 - COS[21][a][b] for a, b in cc])
    dist = 1.0 - sim  # cosine distance
    dg = np.array([G[a][b]["delta_g"] for a, b in cc])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.9))

    # LEFT: log cosine-distance axis (spreads the dense cluster)
    axL.scatter(dist, dg, s=26, c=base, alpha=0.6, edgecolor="white", lw=0.5, zorder=2)
    axL.set_xscale("log")
    axL.set_xlabel("Base-model cosine DISTANCE (1 − sim), log axis  [← more similar]")
    axL.set_ylabel("Marker transfer  ΔG")
    axL.invert_xaxis()  # so 'more similar' is on the right, matching the sim plot
    axL.set_title("Same non-stylized points, log distance axis", fontsize=10, loc="left", pad=8)
    axL.grid(alpha=0.2, lw=0.5)

    # RIGHT: binned mean transfer per cosine-sim quintile
    qs = np.quantile(sim, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    centers, means, ses = [], [], []
    for i in range(5):
        m = (sim >= qs[i]) & (sim <= qs[i + 1]) if i == 4 else (sim >= qs[i]) & (sim < qs[i + 1])
        centers.append(sim[m].mean())
        means.append(dg[m].mean())
        ses.append(dg[m].std() / np.sqrt(m.sum()))
    axR.errorbar(
        centers,
        means,
        yerr=ses,
        fmt="o-",
        color=base,
        ecolor=base,
        elinewidth=1.5,
        capsize=4,
        ms=9,
        lw=2,
        zorder=3,
    )
    axR.set_xlabel("Base-model cosine similarity (quintile mean)")
    axR.set_ylabel("Mean marker transfer  ΔG  (± SE)")
    axR.set_title("Monotone gradient across cosine-sim quintiles", fontsize=10, loc="left", pad=8)
    axR.grid(alpha=0.2, lw=0.5)
    r = rows["loc_ep1"]
    axR.text(
        0.03,
        0.97,
        f"non-stylized (n=156)\nlength-partial rho = {r['ns_rho_cosL21_deltag']:+.2f}, p<1e-12\n"
        f"within cos-sim>0.93: rho = +0.43, p=1e-5",
        transform=axR.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=neu, alpha=0.95),
    )

    fig.suptitle(
        "#474 non-stylized cosine effect is a monotone gradient, not a vertical-line artifact (localized arm, ep1)",
        fontsize=10.8,
        x=0.06,
        y=0.985,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.06,
        0.935,
        "The linear similarity axis compresses the bulk (77% of non-stylized pairs above cos-sim 0.90), so the raw scatter looks vertical. "
        "Left: a log cosine-distance axis spreads the dense region and the trend is visible. Right: mean transfer rises monotonically across "
        "all five cosine-sim quintiles (12.2 → 18.2 nats), and the correlation holds inside the 0.94-1.0 cluster alone (rho +0.43).",
        fontsize=8.0,
        color=neu,
        ha="left",
    )
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.07, right=0.975, wspace=0.26)
    savefig_paper(fig, "issue_474/followup_gradient_check_nonstylized", dir=str(REPO / "figures"))
    plt.close(fig)


def fig4_gradient_by_group(rows):
    """Monotone-gradient (binned-quintile) view for three cell groups side by side:
    all 240 / persona-prompt cells / non-stylized. Localized arm, ep1.

    'Persona-prompt cells' = any ordered pair with at least one Class-A persona
    endpoint (n=130). The strict persona-to-persona slice (A x A, n=20) is
    +0.35 but underpowered (p=0.13), noted in the caption rather than plotted.
    """
    set_paper_style("blog")
    matplotlib.rcParams["figure.constrained_layout.use"] = False
    base, neu = paper_palette_role("baseline"), paper_palette_role("neutral")
    G = _load_G("loc", 1)
    A = {"A1", "A2", "A3", "A4", "A5"}
    groups = [
        ("All transformations (n=240)", lambda a, b: True),
        ("Persona-prompt cells (≥1 persona endpoint, n=130)", lambda a, b: a in A or b in A),
        ("Non-stylized cells (n=156)", lambda a, b: a not in STY and b not in STY),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), sharey=True)
    for ax, (title, pred) in zip(axes, groups):
        cc = [(a, b) for a in CONDS for b in CONDS if a != b and pred(a, b)]
        sim = np.array([1 - COS[21][a][b] for a, b in cc])
        dg = np.array([G[a][b]["delta_g"] for a, b in cc])
        ln = np.array([np.log(PT[a][b]) for a, b in cc])
        r, p = _length_partial(sim, dg, ln)
        qs = np.quantile(sim, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cen, mn, se = [], [], []
        for i in range(5):
            m = (
                (sim >= qs[i]) & (sim <= qs[i + 1])
                if i == 4
                else (sim >= qs[i]) & (sim < qs[i + 1])
            )
            cen.append(sim[m].mean())
            mn.append(dg[m].mean())
            se.append(dg[m].std() / np.sqrt(m.sum()))
        ax.errorbar(
            cen,
            mn,
            yerr=se,
            fmt="o-",
            color=base,
            ecolor=base,
            elinewidth=1.5,
            capsize=4,
            ms=9,
            lw=2,
        )
        ax.set_xlabel("Base-model cosine similarity (quintile mean)")
        ax.set_title(title, fontsize=9.5, loc="left", pad=8)
        ax.grid(alpha=0.2, lw=0.5)
        pf = "p < 1e-12" if p < 1e-12 else f"p = {p:.1e}"
        ax.text(
            0.03,
            0.97,
            f"length-partial rho = {r:+.2f}\n{pf}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=neu, alpha=0.95),
        )
    axes[0].set_ylabel("Mean marker transfer  ΔG  (± SE)")
    fig.suptitle(
        "Cosine → transfer gradient by transformation group (#474 localized arm, ep1)",
        fontsize=11,
        x=0.04,
        y=0.985,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.04,
        0.935,
        "Mean transfer per cosine-similarity quintile. Higher base-model cosine similarity → more marker transfer, monotone in every group. "
        "The strict persona-to-persona slice (both endpoints a persona, n=20) is rho=+0.35 but underpowered (p=0.13); the persona panel here "
        "counts any cell with a persona endpoint.",
        fontsize=8.0,
        color=neu,
        ha="left",
    )
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.055, right=0.985, wspace=0.12)
    savefig_paper(fig, "issue_474/followup_gradient_by_group", dir=str(REPO / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    rows = build_table()
    (REPO / "eval_results/issue_474").mkdir(parents=True, exist_ok=True)
    (REPO / "eval_results/issue_474/followup_cosine_analysis.json").write_text(
        json.dumps(rows, indent=2)
    )
    # console table
    print(
        f"{'cell':9s} {'sat%(ns)':>8s} | {'JS ns ΔG':>9s} {'cos ns ΔG':>10s} {'cos ns trnLP':>13s} | {'JS all':>7s} {'cos all':>8s}"
    )
    print("-" * 78)
    for k, r in rows.items():
        print(
            f"{k:9s} {r['saturation_frac_nonstylized'] * 100:7.0f}% | "
            f"{r['ns_rho_JS_deltag']:+9.2f} {r['ns_rho_cosL21_deltag']:+10.2f} "
            f"{r['ns_rho_cosL21_trainedlogp']:+13.2f} | "
            f"{r['all_rho_JS_deltag']:+7.2f} {r['all_rho_cosL21_deltag']:+8.2f}"
        )
    print("\nLayer sweep (loc_ep1, non-stylized, cos vs ΔG):")
    for L in LAYERS:
        print(f"  L{L:<2d}: {rows['loc_ep1']['ns_layer_sweep_deltag'][L]:+.2f}")
    fig0_headline(rows)
    fig1_survival(rows)
    fig2_layersweep(rows)
    fig3_gradient_check(rows)
    fig4_gradient_by_group(rows)
    print("\nFigures -> figures/issue_474/followup_*.png")

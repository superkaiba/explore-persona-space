"""#1482 writeup figures: concordance per predictor — pooled, firing-conditioned, tails.

THREE figures, one predictor set, one statistic.

STATISTIC. Concordance (Harrell c-index) of per-feature R^2 on each predictor:

    D_{y|x} = (C - Dis) / (n_pairs - T_x)        c = (D + 1) / 2

the probability that, in a random pair of features THE PREDICTOR SEPARATES, the
one with the higher predictor value is the better-predicted one. 0.5 = chance.
For a binary predictor this reduces EXACTLY to the ordinary AUROC (the
probability a feature carrying the property is better predicted than one that
does not) -- verified bit-identical against sklearn.metrics.roc_auc_score.
Normalising by pairs untied in the PREDICTOR is what removes the prevalence
ceiling a median-split AUROC imposes on a rare indicator; R^2 is never
thresholded, so nothing is discarded on the outcome side either.

CONDITIONING is done by RESTRICTING WHICH PAIRS COUNT -- accumulate C/Dis/T_x
within firing-rate strata and pool numerator and denominator separately
(Mantel-Haenszel style, so a stratum where the predictor barely varies is
down-weighted by its own informative-pair count). Cross-stratum pairs are never
formed, so variation that tracks firing rate is excluded by construction. This
is nonparametric: no functional form is assumed for firing rate -> R^2.

FIG 1 `pooled`  -- all pairs.
FIG 2 `matched` -- pairs only within the 10 deciles of AVERAGE ACTIVITY ACROSS
    TOKENS (`firing_freq_per_token`), the most global activity measure. The
    sidecar also carries `m_answer` (per-answer activity deciles) and `m_both`
    (nested per-token decile x within-decile per-answer tercile); joint control
    moves nothing by more than ~0.013 because rho(per-answer, per-token) = 0.96.
FIG 3 `tails`   -- does a predictor work better at the EXTREMES? Restrict to the
    top-k and bottom-k features by R^2 (k = TAIL_FRAC of n per tail) and
    recompute; plot tail-c against whole-population c. Left panel raw, right
    panel with the same per-token firing-decile conditioning (strata recomputed
    WITHIN the tail subpopulation). A point on the diagonal behaves the same in
    the tails as overall; above it, the predictor is sharper at the extremes.

MATRYOSHKA TIER is a DIFFERENT MEASUREMENT ARM -- layer-20 SAELens matryoshka
jumprelu dictionaries (k=100, 16,384-feature panel), not the layer-19 andyrdt
BatchTopK dictionary the other rows are scored on. It sorts inline with the
battery but is disclosed by a DIAMOND marker, its own n, the legend entry and
the caption: it supports a same-direction claim, never a row-to-row magnitude
comparison. Its conditioned values use that arm's own per-answer activity (the
arm has no per-token column).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/thomasjiralerspong/explore-persona-space/scripts")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy/scipy/matplotlib import

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import rankdata

import issue1482_shapley_blocks as SB
from issue1482_concordance_fig import concordance

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
OUT = REPO / "figures/issue_1482/concordance"
COV_V2 = "eval_results/issue_1482/predictor_battery/fullwidth_covariates_v2.npz"
MATRYOSHKA = "eval_results/issue_1482/matryoshka_tier/perfeature_m_lmsys_default.npz"
NBINS = 10
TAIL_FRAC = 0.05  # per tail; k = 6,036 top + 6,036 bottom of 120,716
TAIL_FRAC_SWEEP = (0.0125, 0.05, 0.15)  # robustness, sidecar only

# Okabe-Ito, colourblind-safe. One colour = one predictor family, every figure.
FAMILY_COLOR = {
    "firing": "#0072B2",
    "level": "#009E73",
    "position": "#E69F00",
    "geometry": "#56B4E9",
    "output": "#CC79A7",
    "io": "#000000",
    "side": "#7B3294",
    "judged": "#D55E00",
}
FAMILY_LABEL = {
    "firing": "activity / firing frequency",
    "level": "high- vs low-level proxy",
    "position": "where in the text it fires",
    "geometry": "SAE geometry",
    "output": "output-side / logit footprint",
    "io": "read- vs write-side (circuit role)",
    "side": "context- vs answer-side (token position)",
    "judged": "LLM-judged property (binary)",
}
FAMILY_ORDER = ("firing", "level", "position", "geometry", "output", "io", "side", "judged")

CONT = {
    "activity": ("Firing frequency (per answer)", "firing"),
    "firing_freq_per_token": ("Firing frequency (per token)", "firing"),
    "mean_act_uncond": ("Mean activation (over all answers)", "firing"),
    "mean_act_cond": ("Mean activation (when active)", "firing"),
    "act_var_across_answers": ("Variance of mean activation across answers", "firing"),
    "n_active_holdout": ("Holdout answers active (count)", "firing"),
    "consistency": ("Within-answer consistency", "level"),
    "mean_run_length": ("Mean activation run length (tokens)", "level"),
    "side_ratio": ("Side ratio (answer-side firing fraction)", "side"),
    "template_token_frac": ("Template-token activation fraction", "position"),
    "scaffold_frac": ("Scaffold-token activation fraction", "position"),
    "redundancy_max_cos": ("Nearest-neighbour cosine (SAE redundancy)", "geometry"),
    "proj_var": ("Variance explained in answer space", "geometry"),
    "enc_norm": ("SAE encoder norm (read strength)", "io"),
    "enc_dec_cos": ("Encoder-decoder cosine (OUTPUTNESS)", "io"),
    "massive_dim_mass": ("Massive-dimension mass", "geometry"),
    "write_norm": ("Write norm, gamma-scaled (OUTPUTNESS)", "io"),
    "dec_norm": ("SAE decoder norm (write strength)", "io"),
    "footprint_kurt": ("Logit-footprint kurtosis (OUTPUTNESS)", "io"),
    "footprint_skew": ("Logit-footprint skew (OUTPUTNESS)", "io"),
    "footprint_var": ("Logit-footprint variance", "output"),
    "logit_footprint_concentration": ("Logit-footprint concentration (OUTPUTNESS)", "io"),
}
# DROPPED: dense_latent_flag. It is a >50% threshold on `activity`, which is
# already plotted continuously three rows up -- the same construct twice, and
# the binarized copy topped the chart at c=0.93 purely because thresholding a
# strong continuous predictor concentrates its signal.
DERIVED_BIN: dict[str, tuple[str, str]] = {}

# Input- vs output-side DISCRETE levels (the `io` family). Each is drawn as a
# one-vs-rest indicator over a level of a categorical covariate column.
#   side_class       0=context-only, 1=two-sided, 2=answer-only, -1=dead
#   promoting_class  0=other, 1=promoting, 2=suppressing, 3=partition
#                    (Gurnee logit-footprint slice, kurt/var q0.90)
IO_LEVELS = {
    ("side_class", 0.0): "Fires on the context side only",
    ("side_class", 1.0): "Fires on BOTH context and answer side",
    ("side_class", 2.0): "Fires on the answer side only",
    ("promoting_class", 1.0): "Logit footprint: promoting",
    ("promoting_class", 2.0): "Logit footprint: suppressing",
    ("promoting_class", 3.0): "Logit footprint: partition",
}

# judged axis:level -> display name. Levels absent here (every `unresolved`
# bucket, speaker_property:none) are deliberately NOT drawn.
# functional_role IS drawn -- it is the judged half of the input/output-side
# question -- but every row carries an explicit [k=0.31] tag: its Cohen's kappa
# is 0.310, far below the 0.6 usability bar, which is why it was excluded from
# the 24-block Shapley decomposition. Read those three rows as indicative only.
JUDGED = {
    "interpretable:yes": "Interpretable (autointerp)",
    "abstraction:token_surface": "Abstraction: token surface",
    "abstraction:lexical_semantic": "Abstraction: lexical semantic",
    "abstraction:abstract_contextual": "Abstraction: abstract contextual",
    "content_type:topic": "Content type: topic",
    "content_type:task_format": "Content type: task format",
    "content_type:entity": "Content type: entity",
    "content_type:syntax": "Content type: syntax",
    "content_type:operation": "Content type: operation",
    "speaker_property:language": "Speaker: language of the text",
    "speaker_property:identity_disposition": "Speaker: identity / disposition",
    "speaker_property:register_style": "Speaker: register / style",
    "functional_role:input_side": "Judged role: input-side  [k=0.31]",
    "functional_role:output_promoting": "Judged role: output-promoting  [k=0.31]",
    "functional_role:mixed": "Judged role: mixed  [k=0.31]",
}
# judged levels drawn in the io family rather than the generic judged family
JUDGED_IO_AXES = {"functional_role"}


def deciles(v: np.ndarray, k: int = NBINS) -> list[np.ndarray]:
    """Index sets for the k equal-count strata of v."""
    e = np.percentile(v, np.linspace(0, 100, k + 1))
    b = np.clip(np.digitize(v, e[1:-1]), 0, k - 1)
    return [np.flatnonzero(b == j) for j in range(k)]


def tail_mask(y: np.ndarray, frac: float) -> np.ndarray:
    """Indices of the top-frac and bottom-frac of y (the R^2 distribution tails)."""
    k = max(1, int(round(frac * len(y))))
    order = np.argsort(y, kind="mergesort")
    return np.concatenate([order[:k], order[-k:]])


def battery() -> dict:
    """Concordance rows + the raw vectors, for the layer-19 densesae battery."""
    SB.COV_NPZ = COV_V2
    inp = SB.load_inputs(REPO / "data/issue_1482/densesae_target/ridge__mean_r2_fullwidth.npy")
    doc = json.loads(
        (
            REPO
            / "eval_results/issue_1482/predictor_battery/shapley_blocks_densesae_ridge_k24.json"
        ).read_text()
    )
    reps = [r["representative"] for r in doc["full_sample"]["representatives"]]
    cov, lab, r2 = inp["cov"], inp["labels"], inp["r2"]

    ok = np.isfinite(r2) & np.isfinite(cov["firing_freq_per_token"]) & np.isfinite(cov["activity"])
    for c in reps:
        ok &= np.isfinite(cov[c])
    y = r2[ok]
    n = len(y)
    ft, fa = cov["firing_freq_per_token"][ok], cov["activity"][ok]

    all_idx = [np.arange(n)]
    s_token = deciles(ft)
    s_answer = deciles(fa)
    s_both: list[np.ndarray] = []
    for idx in s_token:  # nest activity terciles inside each per-token decile
        q = np.percentile(fa[idx], [100 / 3, 200 / 3])
        inner = np.clip(np.digitize(fa[idx], q), 0, 2)
        for t in range(3):
            s_both.append(idx[inner == t])

    # (display name, family, n_pos or None, vector)
    items: list[tuple[str, str, int | None, np.ndarray]] = []
    for col, (name, fam) in CONT.items():
        # Membership is the COVARIATE SET, not the Shapley representative set:
        # firing_freq_per_token and mean_act_uncond are declustered siblings
        # (rho >= 0.90 with their representative), which matters for credit
        # ALLOCATION but not for a marginal per-predictor read.
        if col not in cov or not np.all(np.isfinite(cov[col][ok])):
            continue
        items.append((name, fam, None, cov[col][ok]))
    for col, (name, fam) in DERIVED_BIN.items():
        if col not in cov:
            continue
        v = np.asarray(cov[col], dtype=np.float64)[ok]
        items.append((name, fam, int((v == np.unique(v)[1]).sum()), v))
    for (col, level), name in IO_LEVELS.items():
        if col not in cov:
            continue
        m = (np.asarray(cov[col], dtype=np.float64)[ok] == level).astype(float)
        k = int(m.sum())
        if 60 < k < n - 60:
            fam = "side" if col == "side_class" else "io"
            items.append((name, fam, k, m))
    for key, name in JUDGED.items():
        axis, level = key.split(":", 1)
        if axis not in lab:
            continue
        m = (np.asarray(lab[axis])[ok] == level).astype(float)
        k = int(m.sum())
        if 60 < k < n - 60:
            items.append((name, "io" if axis in JUDGED_IO_AXES else "judged", k, m))

    rows, vecs = [], {}
    for name, fam, npos, v in items:
        vecs[name] = v
        rows.append(
            {
                "name": name,
                "family": fam,
                "n_pos": npos,
                "pooled": concordance(v, y, all_idx),
                "matched": concordance(v, y, s_token),
                "m_answer": concordance(v, y, s_answer),
                "m_both": concordance(v, y, s_both),
            }
        )
    return {
        "rows": rows,
        "vecs": vecs,
        "ok": ok,  # the finite-row mask, so callers can align other per-feature vectors
        "y": y,
        "ctrl": ft,
        "n": n,
        "ctrl_pooled": concordance(ft, y, all_idx),
        "ctrl2_pooled": concordance(fa, y, all_idx),
    }


def add_tail_columns(rows: list[dict], vecs: dict, y: np.ndarray, ctrl: np.ndarray) -> None:
    """Attach tail-restricted concordance (raw + firing-conditioned) to each row."""
    for frac in sorted({TAIL_FRAC, *TAIL_FRAC_SWEEP}):
        sel = tail_mask(y, frac)
        ys, cs = y[sel], ctrl[sel]
        s_all = [np.arange(len(sel))]
        s_tok = deciles(cs)
        suffix = "" if frac == TAIL_FRAC else f"_f{frac:g}"
        for r in rows:
            v = vecs[r["name"]][sel]
            r[f"tail_pooled{suffix}"] = concordance(v, ys, s_all)
            r[f"tail_matched{suffix}"] = concordance(v, ys, s_tok)


def matryoshka_row() -> dict:
    """Coarseness (-tier) concordance on the L20 matryoshka arm, its own panel."""
    with np.load(REPO / MATRYOSHKA) as z:
        r2, tier, act = z["r2"], z["tier"].astype(int), z["activity"]
    keep = np.isfinite(r2) & np.isfinite(act)
    r2, tier, act = r2[keep], tier[keep], act[keep]
    x = -tier.astype(float)  # higher = coarser tier

    def ordinal_c(idx_sets: list[np.ndarray], xx: np.ndarray, yy: np.ndarray) -> float:
        num = den = 0.0
        for idx in idx_sets:
            xs, ys = xx[idx], yy[idx]
            vals = np.unique(xs)
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    a, b = xs == vals[i], xs == vals[j]
                    ka, kb = int(a.sum()), int(b.sum())
                    if ka == 0 or kb == 0:
                        continue
                    sub = a | b  # rank WITHIN the pair; a third level must not leak in
                    rr = rankdata(ys[sub])
                    u = rr[b[sub]].sum() - kb * (kb + 1) / 2
                    num += (2 * (u / (ka * kb)) - 1) * ka * kb
                    den += ka * kb
        return 0.5 * (num / den + 1) if den > 0 else np.nan

    row = {
        "name": "Matryoshka dictionary tier (coarser = earlier tier)",
        "family": "level",
        "arm": "matryoshka-l20-lmsys",
        "n_pos": int(len(r2)),
        "pooled": ordinal_c([np.arange(len(r2))], x, r2),
        "matched": ordinal_c(deciles(act), x, r2),
    }
    row["m_answer"] = row["matched"]
    row["m_both"] = row["matched"]
    for frac in sorted({TAIL_FRAC, *TAIL_FRAC_SWEEP}):
        sel = tail_mask(r2, frac)
        suffix = "" if frac == TAIL_FRAC else f"_f{frac:g}"
        row[f"tail_pooled{suffix}"] = ordinal_c([np.arange(len(sel))], x[sel], r2[sel])
        row[f"tail_matched{suffix}"] = ordinal_c(deciles(act[sel]), x[sel], r2[sel])
    return row


def family_legend() -> list[Line2D]:
    """The IDENTICAL legend used on every figure."""
    # No "different SAE arm" handle: the matryoshka row still draws as a diamond
    # and says so in its own label + the footnote, so the extra legend entry was
    # redundant (dropped at Thomas's request, 2026-08-04).
    return [
        Line2D([], [], marker="o", ls="", color=FAMILY_COLOR[f], label=FAMILY_LABEL[f])
        for f in FAMILY_ORDER
    ]


CAPTION = (
    "c = P(the feature with the higher predictor value is the better-predicted one), over pairs the "
    "predictor separates;\nfor a binary predictor this is the ordinary AUROC. The DIAMOND row "
    "(matryoshka tier) is scored on a DIFFERENT dictionary\n(layer-20 SAELens matryoshka, "
    "16,384-feature panel), so it supports a same-direction claim but is NOT row-to-row comparable\n"
    "on magnitude. Every value is MARGINAL — the other predictors are not controlled (pairwise "
    "correlations reach 0.85). No confidence intervals."
)


def render_lollipop(rows: list[dict], key: str, title: str, subtitle: str, stem: Path) -> None:
    """One lollipop panel, BEST-TO-WORST top-to-bottom.

    "Best" is DISCRIMINATIVE STRENGTH |c - 0.5|, not the signed value: a
    predictor at 0.376 separates exactly as well as one at 0.624, it just points
    the other way. Direction stays legible because every lollipop is anchored at
    0.5 -- leftward = the property marks WORSE-predicted features. Matplotlib's
    y-axis runs bottom-up, so the ASCENDING sort puts the strongest row on top.
    """
    order = sorted(rows, key=lambda r: abs(r[key] - 0.5))
    fig, ax = plt.subplots(figsize=(10.4, 0.30 * (len(order) + 2) + 2.7))
    ax.axvline(0.5, color="#888888", lw=1.4, zorder=1)

    for i, r in enumerate(order):
        col = FAMILY_COLOR[r["family"]]
        other_arm = r.get("arm") is not None
        ax.plot([0.5, r[key]], [i, i], color=col, lw=1.6, alpha=0.45, zorder=2)
        ax.scatter(
            r[key],
            i,
            s=52 if other_arm else 46,
            color=col,
            marker="D" if other_arm else "o",
            zorder=3,
        )
        ax.annotate(
            f"{r[key]:.2f}",
            (r[key], i),
            textcoords="offset points",
            xytext=(9 if r[key] >= 0.5 else -9, 0),
            ha="left" if r[key] >= 0.5 else "right",
            va="center",
            fontsize=6.8,
            color="#333333",
        )

    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(
        [r["name"] + (f"  (n={r['n_pos']:,})" if r["n_pos"] is not None else "") for r in order],
        fontsize=7.8,
    )
    for tick, r in zip(ax.get_yticklabels(), order):
        tick.set_color(FAMILY_COLOR[r["family"]])
    ax.set_ylim(-0.8, len(order) - 0.3)
    ax.set_xlabel("concordance (c-index) with per-feature $R^2$  —  0.5 = chance", fontsize=9.6)
    ax.tick_params(axis="x", labelsize=8.4)
    ax.grid(axis="x", alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11.5)
    ax.legend(handles=family_legend(), fontsize=7.4, loc="lower right", framealpha=0.94)
    fig.text(
        0.5,
        -0.030 - 0.0007 * len(order),
        CAPTION
        + "\nRows are sorted STRONGEST first by |c - 0.5|; a lollipop pointing LEFT of 0.5 is "
        "inverted — that property marks features the map predicts WORSE.",
        ha="center",
        fontsize=7.4,
        color="#5A5A5A",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_tails(rows: list[dict], n: int, stem: Path) -> None:
    """Tail-restricted vs whole-population concordance; left raw, right conditioned."""
    k = int(round(TAIL_FRAC * n))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.6))
    panels = (
        (axes[0], "pooled", "tail_pooled", "raw (all pairs)"),
        (
            axes[1],
            "matched",
            "tail_matched",
            "controlling for average activity across tokens (10 deciles)",
        ),
    )
    # ONE limit pair for BOTH panels. Panel-local limits desync the two axes the
    # moment one panel holds an outlier (the io family put `Fires on the ANSWER
    # side only` at 0.23 raw, stretching the left panel to 0.20-0.80 while the
    # right stayed 0.35-0.80), which destroys the side-by-side comparison the
    # figure exists for.
    allv = [r[k] for r in rows for k in ("pooled", "tail_pooled", "matched", "tail_matched")]
    lo, hi = min(allv) - 0.035, max(allv) + 0.035
    for ax, xkey, ykey, sub in panels:
        ax.plot([lo, hi], [lo, hi], color="#999999", lw=1.2, zorder=1)
        ax.axvline(0.5, color="#CCCCCC", lw=1.0, zorder=1)
        ax.axhline(0.5, color="#CCCCCC", lw=1.0, zorder=1)
        for r in rows:
            other_arm = r.get("arm") is not None
            ax.scatter(
                r[xkey],
                r[ykey],
                s=56 if other_arm else 46,
                color=FAMILY_COLOR[r["family"]],
                marker="D" if other_arm else "o",
                zorder=3,
                alpha=0.9,
            )
        # Label only the rows that move MOST off the diagonal, so the panel stays
        # readable: a crowded scatter of 35 labels is worse than none. Labels on
        # the right half flip to left-aligned so long names cannot run off-axis.
        mid = 0.5 * (lo + hi)
        # Label only the biggest movers, and alternate the vertical offset: the
        # dense 0.5-0.6 cluster overlaps illegibly at a fixed offset.
        for li, r in enumerate(sorted(rows, key=lambda r: -abs(r[ykey] - r[xkey]))[:7]):
            right_half = r[xkey] > mid
            ax.annotate(
                r["name"][:30],
                (r[xkey], r[ykey]),
                textcoords="offset points",
                xytext=(-8 if right_half else 8, 9 if li % 2 == 0 else -11),
                ha="right" if right_half else "left",
                fontsize=6.6,
                color=FAMILY_COLOR[r["family"]],
            )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("concordance — ALL features", fontsize=9.4)
        ax.set_ylabel(f"concordance — $R^2$ TAILS only (top {k:,} + bottom {k:,})", fontsize=9.4)
        ax.set_title(sub, fontsize=10.2)
        ax.grid(alpha=0.2, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.2)
    axes[1].legend(handles=family_legend(), fontsize=7.2, loc="lower right", framealpha=0.94)
    fig.suptitle(
        "Are some predictors sharper at the extremes?\n"
        f"tail = the {TAIL_FRAC:.1%} best- and {TAIL_FRAC:.1%} worst-predicted features by $R^2$ "
        f"({2 * k:,} of {n:,}) · grey line = no difference",
        fontsize=12.0,
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))  # reserve the strip the caption occupies
    fig.text(
        0.5,
        0.005,
        "c = P(the feature with the higher predictor value is the better-predicted one), over pairs "
        "the predictor separates; for a binary predictor this is the ordinary AUROC.\n"
        "A point ABOVE the grey line separates the extremes better than it separates the population "
        "as a whole; BELOW it, worse. Only the 8 largest movers per panel are labelled.\n"
        "Tail firing-deciles are recomputed WITHIN the tail subpopulation, so the right panel "
        "conditions on firing rate among the extremes, not on the whole-population strata.\n"
        "The DIAMOND (matryoshka tier) is a DIFFERENT dictionary (layer-20 SAELens matryoshka, "
        "16,384-feature panel) — same-direction claim only, not row-to-row comparable.\n"
        "Every value is MARGINAL — the other predictors are not controlled (pairwise correlations "
        "reach 0.85). No confidence intervals; the tails cut rare-indicator counts ~10x.",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#5A5A5A",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    b = battery()
    add_tail_columns(b["rows"], b["vecs"], b["y"], b["ctrl"])
    rows = [*b["rows"], matryoshka_row()]
    n = b["n"]

    render_lollipop(
        rows,
        "pooled",
        "Which feature properties predict how well the context→answer map predicts a feature?",
        f"concordance over all pairs · sorted by predictive power |c − 0.5| · n = {n:,} features",
        OUT / "writeup_concordance_pooled",
    )
    render_lollipop(
        rows,
        "matched",
        "The same predictors, controlling for average activity across tokens",
        "pairs formed only within the 10 deciles of per-token firing frequency · sorted by "
        f"|c − 0.5| · n = {n:,}\naverage activity across tokens itself scored "
        f"{b['ctrl_pooled']:.3f} pooled",
        OUT / "writeup_concordance_matched",
    )
    render_tails(rows, n, OUT / "writeup_concordance_tails")

    meta = {
        "what_is_plotted": (
            "Concordance (Harrell c-index) of per-feature R^2 on each predictor, on the AUROC "
            "scale. Fig 1 pools all pairs; fig 2 forms pairs only within per-token firing-rate "
            "deciles; fig 3 compares tail-restricted concordance against whole-population "
            "concordance, raw and firing-conditioned."
        ),
        "statistic": "D_{y|x} = (C - Dis) / (n_pairs - T_x);  c = (D + 1) / 2",
        "n_rows_battery": int(n),
        "battery_dictionary": "layer-19 andyrdt BatchTopK k=64, v2 (proj_var-corrected) universe",
        "matryoshka_arm": (
            "layer-20 SAELens matryoshka jumprelu k=100, lmsys dictionary, 16,384-feature panel — "
            "a DIFFERENT measurement arm, diamond marker, not row-comparable"
        ),
        "primary_control": "firing_freq_per_token (average activity across tokens), 10 deciles",
        "secondary_controls": {
            "m_answer": "activity (per-answer firing frequency), 10 deciles",
            "m_both": "nested per-token decile x within-decile per-answer tercile (30 cells)",
        },
        "tail_definition": (
            f"top {TAIL_FRAC:.1%} and bottom {TAIL_FRAC:.1%} of the R^2 distribution "
            f"({int(round(TAIL_FRAC * n)):,} per tail); the right panel's firing deciles are "
            "recomputed WITHIN that subpopulation"
        ),
        "tail_frac_sweep": list(TAIL_FRAC_SWEEP),
        "firing_per_token_pooled": float(b["ctrl_pooled"]),
        "firing_per_answer_pooled": float(b["ctrl2_pooled"]),
        "values": rows,
        "omitted_predictors": [
            "every judged `unresolved` level and speaker_property:none — judge-failure / residual "
            "buckets, not properties",
            "dense_latent_flag — a >50% threshold on `activity`, which is already plotted "
            "continuously; the binarized copy is the same construct twice",
        ],
        "io_family_note": (
            "The `io` family answers 'is this an input-side or an output-side feature?' with "
            "five instruments: the continuous side_ratio and gamma-scaled write_norm, the "
            "side_class levels (context-only / two-sided / answer-only), the Gurnee "
            "logit-footprint classes (promoting / suppressing / partition — a q0.90 slice off "
            "footprint_kurt/skew), and the judged functional_role levels. functional_role and "
            "gurnee_promoting_class were EXCLUDED from the 24-block Shapley decomposition "
            "(kappa = 0.310 below the usability bar; a slice off a continuum adding ~nothing "
            "over its continuous parent) — they are drawn HERE because the input/output "
            "question is the point of the family, and every functional_role row carries an "
            "explicit [k=0.31] tag. Read those three rows as indicative only."
        ),
        "caveats": [
            "MARGINAL, not incremental: firing rate is conditioned in figs 2-3, the other "
            "predictors are NOT. Pairwise block correlations reach 0.85.",
            "No confidence intervals. Differences under ~0.02 should not be read; rare indicators "
            "(n_pos < ~1,500) are the noisiest rows, and the tail panels cut their counts ~10x.",
            "Conditioning on per-token firing rate OVER-CONTROLS any predictor that is a factor of "
            "it: per-token rate ~ activity x consistency (rho = 0.986), so `Within-answer "
            "consistency` is partly conditioned on itself in fig 2 (0.623 pooled -> 0.477). Its "
            "appropriate control is the per-answer margin alone, where it reads 0.553.",
            "The tail subpopulation is selected ON the outcome, so its within-tail R^2 variance is "
            "far smaller than the between-tail gap; tail-c is dominated by top-vs-bottom "
            "separation, not by ranking within a tail.",
            "The matryoshka row is a different dictionary, layer, panel and n — it supports a "
            "same-direction claim, never a row-to-row magnitude comparison.",
            "Judged labels are majority-vote over 5 draws; unanimity was not required.",
        ],
    }
    (OUT / "writeup_concordance.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote 3 figures to {OUT}  ({len(rows)} rows, n={n:,})")
    print(
        f"  per-token pooled {b['ctrl_pooled']:.3f} | per-answer pooled {b['ctrl2_pooled']:.3f}\n"
    )
    print(f"  {'predictor':46s} {'pooled':>7} {'matched':>8} {'tailRaw':>8} {'tailMatch':>10}")
    for r in sorted(rows, key=lambda r: -abs(r["tail_pooled"] - r["pooled"])):
        print(
            f"  {r['name']:46s} {r['pooled']:7.3f} {r['matched']:8.3f} "
            f"{r['tail_pooled']:8.3f} {r['tail_matched']:10.3f}"
        )


if __name__ == "__main__":
    main()

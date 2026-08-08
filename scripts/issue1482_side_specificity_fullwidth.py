"""Full-dictionary answer-vs-context side specificity of SAE features.

Are there "context-specific" / "answer-specific" SAE features? If a feature
never fires on the context side, its answer-side activation cannot be read off
the context vector by any map — it has to be produced. That is the concrete
motivation for translating into answer space rather than reading the context
representation directly.

DEFINITION (row occupancy, NOT token counts):
    side_ratio = cnt / (cnt + psi_cnt)
where `cnt` counts FIT ROWS in which the feature is active anywhere in the
ANSWER span and `psi_cnt` counts fit rows active anywhere in the CONTEXT span
(#1773 `issue1773_phase0_mechanical.py` ~L505). A feature firing once in a span
and a feature firing at every token of it both contribute 1.

THE LENGTH CONFOUND — why 0.5 is the WRONG reference:
The answer span is roughly twice the context span in this corpus, so a feature
with no side preference whatsoever still lands well above 0.5. The correct
null is the global answer share of row-firings,
    null = sum(cnt) / (sum(cnt) + sum(psi_cnt))  ~ 0.675,
drawn on panel (a). The observed median sits essentially ON it, so the bulk of
the dictionary is side-INDIFFERENT and the raw distribution's rightward centre
is span length, not answer preference.

Panel (b) removes the confound by construction, plotting the null-centred
log2 side-odds enrichment
    e = log2[ (cnt/psi_cnt) / (sum cnt / sum psi_cnt) ],
so e = 0 is side-indifferent, e = +1 is 2x answer-enriched, e = -1 is 2x
context-enriched. Strictly one-sided features have e = +-inf and are shown as
the point masses at the edges of panel (a) instead.

Usage:
    uv run python scripts/issue1482_side_specificity_fullwidth.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import set_paper_style

REPO = Path(__file__).resolve().parents[1]
SCAN = REPO / "data/issue_1482/fullwidth/fused_scan.npz"
OUTDIR = REPO / "figures/issue_1482/side_specificity"

# Okabe-Ito; held fixed across the writeup: context = vermillion, answer = blue.
C_CONTEXT = "#D55E00"
C_ANSWER = "#0072B2"
C_BOTH = "#4C72B0"
CLIP = 8.0  # log2 display bound for panel (b)

# The independently-verified #1482 full-width census (issue1482_dense_predictor_
# reads.side_ratio_gate). Re-asserted here so a silently-changed scan cache
# fails loud instead of redrawing a different population.
CENSUS_EXPECTED = {
    "n_fit": 120000,
    "answer_active": 128512,
    "context_active": 128002,
    "context_only": 1654,
    "answer_only": 2164,
    "live": 130166,
}


def load_scan() -> tuple[np.ndarray, np.ndarray, int]:
    if not SCAN.exists():
        raise FileNotFoundError(
            f"{SCAN} missing (data/ is gitignored). Regenerate with the fused scan in "
            "scripts/issue1482_dense_predictor_reads.py (one pass over the 1,920-shard "
            "#1482 pooled store)."
        )
    with np.load(SCAN) as z:
        return (
            z["cnt_fit"].astype(np.float64),
            z["psi_cnt_fit"].astype(np.float64),
            int(z["n_fit"]),
        )


def main() -> None:
    cnt, psi, n_fit = load_scan()

    live = (cnt + psi) > 0
    both = (cnt > 0) & (psi > 0)
    ctx_only = (psi > 0) & (cnt == 0)
    ans_only = (cnt > 0) & (psi == 0)

    census = {
        "n_fit": n_fit,
        "answer_active": int((cnt > 0).sum()),
        "context_active": int((psi > 0).sum()),
        "context_only": int(ctx_only.sum()),
        "answer_only": int(ans_only.sum()),
        "live": int(live.sum()),
    }
    mismatch = {k: (census[k], v) for k, v in CENSUS_EXPECTED.items() if census[k] != v}
    if mismatch:
        raise AssertionError(f"scan does not reproduce the verified census: {mismatch}")

    null = float(cnt.sum() / (cnt + psi).sum())
    sr = np.divide(cnt, cnt + psi, out=np.full_like(cnt, np.nan), where=live)
    sr_both = sr[both]
    med_live = float(np.nanmedian(sr[live]))

    enr = np.log2((cnt[both] / psi[both]) / (cnt.sum() / psi.sum()))
    tails = {f"frac_abs_gt_{t}": float((np.abs(enr) > t).mean()) for t in (1, 2, 3)} | {
        "n_answer_enriched_gt1": int((enr > 1).sum()),
        "n_context_enriched_lt-1": int((enr < -1).sum()),
        "n_outside_clip": int((np.abs(enr) > CLIP).sum()),
    }

    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))

    # ── (a) raw side_ratio + the two exact point masses ─────────────────────
    ax = axes[0]
    ax.hist(sr_both, bins=80, range=(0, 1), color=C_BOTH, alpha=0.85, label="both-sided")
    ax.set_yscale("log")
    bw = 0.02
    ax.bar(
        0.0,
        ctx_only.sum(),
        width=bw,
        align="edge",
        color=C_CONTEXT,
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
        label="context-only (never fires in answer)",
    )
    ax.bar(
        1.0 - bw,
        ans_only.sum(),
        width=bw,
        align="edge",
        color=C_ANSWER,
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
        label="answer-only (never fires in context)",
    )
    ax.annotate(
        f"{census['context_only']:,}",
        (bw / 2, census["context_only"]),
        textcoords="offset points",
        xytext=(0, 5),
        ha="center",
        fontsize=8,
        color=C_CONTEXT,
    )
    ax.annotate(
        f"{census['answer_only']:,}",
        (1 - bw / 2, census["answer_only"]),
        textcoords="offset points",
        xytext=(0, 5),
        ha="center",
        fontsize=8,
        color=C_ANSWER,
    )
    ax.axvline(null, color="0.25", linestyle="--", linewidth=1.7, zorder=6)
    ax.annotate(
        f"side-indifferent null = {null:.3f}",
        xy=(null, 1.0),
        xycoords=("data", "axes fraction"),
        textcoords="offset points",
        xytext=(6, -8),
        ha="left",
        va="top",
        fontsize=8,
        color="0.25",
    )
    ax.axvline(0.5, color="0.65", linestyle=":", linewidth=1.2, zorder=4)
    ax.annotate(
        "0.5 — naive reference\n(wrong: answer span $\\approx$2x context)",
        xy=(0.5, 0.72),
        xycoords=("data", "axes fraction"),
        textcoords="offset points",
        xytext=(-6, 0),
        ha="right",
        va="center",
        fontsize=7.5,
        color="0.6",
    )
    ax.set_xlabel(
        "side_ratio  =  answer-side rows / (answer-side + context-side) rows\n"
        "(row occupancy over 120,000 fit rows — not token counts)"
    )
    ax.set_ylabel("SAE features (log scale)")
    ax.set_title(f"(a) median {med_live:.3f} sits ON the length null — bulk is side-indifferent")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # ── (b) null-centred log2 enrichment ────────────────────────────────────
    ax = axes[1]
    e = np.clip(enr, -CLIP, CLIP)
    ax.hist(e, bins=100, range=(-CLIP, CLIP), color=C_BOTH, alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(0, color="0.25", linestyle="--", linewidth=1.7)
    ax.axvspan(-CLIP, -1, color=C_CONTEXT, alpha=0.07, linewidth=0)
    ax.axvspan(1, CLIP, color=C_ANSWER, alpha=0.07, linewidth=0)
    ax.text(
        0.03,
        0.96,
        f"context-enriched\n$\\geq$2x: {tails['n_context_enriched_lt-1']:,}",
        transform=ax.transAxes,
        fontsize=8,
        color=C_CONTEXT,
        va="top",
        ha="left",
    )
    ax.text(
        0.97,
        0.96,
        f"answer-enriched\n$\\geq$2x: {tails['n_answer_enriched_gt1']:,}",
        transform=ax.transAxes,
        fontsize=8,
        color=C_ANSWER,
        va="top",
        ha="right",
    )
    ax.set_xlabel(
        "log$_2$ side-odds enrichment   (0 = side-indifferent, $\\pm$1 = 2x skewed)\n"
        f"$|e|>1$: {100 * tails['frac_abs_gt_1']:.1f}%    "
        f"$|e|>2$: {100 * tails['frac_abs_gt_2']:.1f}%    "
        f"$|e|>3$: {100 * tails['frac_abs_gt_3']:.1f}%    "
        f"({tails['n_outside_clip']:,} clipped at $\\pm${CLIP:.0f})"
    )
    ax.set_ylabel("SAE features (log scale)")
    ax.set_title("(b) but the spread is wide: 15% are $\\geq$4x skewed to one side")

    fig.suptitle(
        f"Answer-vs-context side specificity, FULL dictionary "
        f"({census['live']:,} live of 131,072; layer 19, k=64; {n_fit:,} single-turn fit rows)",
        fontsize=12,
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = OUTDIR / "side_specificity_fullwidth"
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")

    stem.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "source": str(SCAN.relative_to(REPO)),
                "scope": "FULL DICTIONARY (131,072; 130,166 live)",
                "census": census,
                "dead_features": int((~live).sum()),
                "both_sided": int(both.sum()),
                "length_implied_null": null,
                "median_side_ratio_live": med_live,
                "enrichment_percentiles": {
                    str(q): float(np.percentile(enr, q)) for q in (1, 5, 25, 50, 75, 95, 99)
                },
                "tails": tails,
                "definition": (
                    "side_ratio = cnt/(cnt+psi_cnt) over the 120,000 FIT rows; per-ROW "
                    "occupancy (feature active ANYWHERE in the span), NOT per-token counts. "
                    "A feature firing once in a span and one firing at every token both "
                    "contribute 1."
                ),
                "caveats": [
                    "ROW occupancy, not token counts — 'fraction of activations' would be a "
                    "token-level quantity; this is span-level presence.",
                    "The answer span is ~2x the context span here, so a side-indifferent "
                    "feature sits at 0.675, NOT 0.5. Read deviations from the null line, "
                    "never from 0.5.",
                    "Fit rows only: the scan banks cnt_holdout but no psi_cnt_holdout, so "
                    "side_ratio is undefined on the holdout split.",
                    "#1482 SINGLE-TURN corpus (constant prefix). The multi-turn #1738 corpus "
                    "has a longer, varying context and would move the null.",
                    "Strict one-sidedness is a hard zero over 120,000 rows; it is a much "
                    "stronger criterion than the continuous tails, so the 2.9% one-sided "
                    "figure UNDERSTATES specialisation — read panel (b) for the graded view.",
                ],
                "what_is_plotted": (
                    "(a) histogram of side_ratio over both-sided features (log y), with the "
                    "1,654 context-only and 2,164 answer-only features drawn as exact point "
                    "masses at 0 and 1, the length-implied side-indifferent null at 0.675, "
                    "and the naive 0.5 reference marked as wrong. (b) histogram of the "
                    "null-centred log2 side-odds enrichment over both-sided features, "
                    "clipped to +-8, with the >=2x-skewed tails shaded."
                ),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"wrote {stem}.png / .pdf / .meta.json")
    print(f"census OK: {census}")
    print(f"null={null:.4f}  median(live)={med_live:.4f}")
    print(
        f"|e|>1 {100 * tails['frac_abs_gt_1']:.2f}%  |e|>2 {100 * tails['frac_abs_gt_2']:.2f}%  "
        f"|e|>3 {100 * tails['frac_abs_gt_3']:.2f}%  outside clip {tails['n_outside_clip']:,}"
    )


if __name__ == "__main__":
    main()

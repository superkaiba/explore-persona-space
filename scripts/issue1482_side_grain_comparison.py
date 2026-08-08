"""Row-occupancy vs per-token answer-side ratio, measured on the SAME rows.

The banked `side_ratio` is ROW OCCUPANCY: a feature active anywhere in a span
contributes 1, whether it fired at one token or at every token. That saturates —
so it necessarily UNDERSTATES how side-specialised the dictionary is. The #1482
store banks no context-side token counts, so the token-weighted version required
a fresh per-token capture (`eval_results/issue_1482/run_length/`).

This script draws both grains from that ONE capture, so the comparison is
apples-to-apples: same 2,000 rows, same features, same SAE. Any difference is the
grain, not the sample.

TWO NULLS, AND THEY DIFFER. A side-indifferent feature does not sit at 0.5 — the
answer span is longer than the context span, so the null is the global answer share
of firings, computed separately per grain (occupancy 0.683, token 0.731). Read
deviations from each grain's OWN null; the log2 enrichment panel does that by
construction.

SAMPLE-SIZE CAVEAT ON THE ONE-SIDED COUNTS. "Never fires on side X" is the same
predicate at both grains, so the strictly one-sided counts are IDENTICAL here by
construction. But they are sample-size dependent: over these 2,000 rows 5.49% are
context-only and 10.86% answer-only, whereas the full 120,000-row census gives
1.27% and 1.66%. Fewer rows means more features happen never to fire on a side.
The 2,000-row figures OVERSTATE one-sidedness and must not be quoted against the
census numbers.

Usage:
    uv run python scripts/issue1482_side_grain_comparison.py
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
SRC = REPO / "eval_results/issue_1482/run_length/run_length_perfeature.npz"
META = REPO / "eval_results/issue_1482/run_length/run_length_perfeature.meta.json"
OUTDIR = REPO / "figures/issue_1482/side_specificity"

C_OCC = "#B8B8B8"  # light grey — row occupancy (the saturating, banked grain)
C_TOK = "#1B7A6E"  # dark teal  — per-token (the fresh capture)
CLIP = 8.0

# Full-corpus census (120,000 fit rows) — for the sample-size caveat only.
CENSUS_120K = {"context_only_pct": 1.27, "answer_only_pct": 1.66, "live": 130166}


def grain(a: np.ndarray, c: np.ndarray) -> dict:
    """Null, one-sided counts and null-centred log2 enrichment for one grain."""
    live = (a + c) > 0
    both = (a > 0) & (c > 0)
    null = float(a.sum() / (a.sum() + c.sum()))
    enr = np.log2((a[both] / c[both]) / (a.sum() / c.sum()))
    return {
        "null": null,
        "live": int(live.sum()),
        "both": int(both.sum()),
        "ctx_only": int(((c > 0) & (a == 0)).sum()),
        "ans_only": int(((a > 0) & (c == 0)).sum()),
        "sr": np.divide(a, a + c, out=np.full_like(a, np.nan, dtype=float), where=live),
        "enr": enr,
        "tails": {t: float((np.abs(enr) > t).mean()) for t in (1, 2, 3)},
        "p1": float(np.percentile(enr, 1)),
        "p50": float(np.percentile(enr, 50)),
        "p99": float(np.percentile(enr, 99)),
    }


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"{SRC} missing — the per-token capture has not landed")
    d = np.load(SRC)
    meta = json.loads(META.read_text()) if META.exists() else {}

    tok = grain(d["ans_tokens_active"], d["ctx_tokens_active"])
    occ = grain(d["row_occupancy_ans"], d["row_occupancy_ctx"])

    assert tok["ctx_only"] == occ["ctx_only"] and tok["ans_only"] == occ["ans_only"], (
        "one-sided counts must be identical across grains — 'never fires' is the same "
        f"predicate: token {tok['ctx_only']}/{tok['ans_only']} vs occ "
        f"{occ['ctx_only']}/{occ['ans_only']}"
    )
    n_rows = int(meta.get("sample", {}).get("n_rows", 2000))

    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))

    ax = axes[0]
    for g, col, lab in ((occ, C_OCC, "row occupancy (banked)"), (tok, C_TOK, "per-token (new)")):
        ax.hist(
            g["sr"][np.isfinite(g["sr"])],
            bins=80,
            range=(0, 1),
            color=col,
            alpha=0.62,
            label=f"{lab} — null {g['null']:.3f}",
        )
        ax.axvline(g["null"], color=col, linestyle="--", linewidth=1.9)
    ax.set_yscale("log")
    ax.set_xlabel("answer-side ratio   (each grain has its OWN null, dashed)")
    ax.set_ylabel("SAE features (log scale)")
    ax.set_title("(a) same rows, two grains — the nulls differ")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)

    ax = axes[1]
    for g, col, lab in ((occ, C_OCC, "row occupancy"), (tok, C_TOK, "per-token")):
        ax.hist(
            np.clip(g["enr"], -CLIP, CLIP),
            bins=100,
            range=(-CLIP, CLIP),
            color=col,
            alpha=0.62,
            label=(
                f"{lab}:  $|e|$>2 {100 * g['tails'][2]:.1f}%   $|e|$>3 {100 * g['tails'][3]:.1f}%"
            ),
        )
    ax.axvline(0, color="0.25", linestyle="--", linewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlabel(r"log$_2$ side-odds enrichment, null-centred ($\pm$1 = 2x skewed)")
    ax.set_ylabel("SAE features (log scale)")
    ax.set_title("(b) occupancy SATURATES — token grain finds ~2x more specialists")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)

    fig.suptitle(
        f"Row-occupancy vs per-token answer-side ratio — identical {n_rows:,} rows, "
        f"{tok['live']:,} live features (layer 19, k=64)",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.005,
        f"Strictly one-sided counts are IDENTICAL by construction ('never fires' is grain-free): "
        f"{tok['ctx_only']:,} context-only, {tok['ans_only']:,} answer-only. They are sample-size "
        f"dependent — the full 120,000-row census gives {CENSUS_120K['context_only_pct']}% / "
        f"{CENSUS_120K['answer_only_pct']}%, so these {n_rows:,}-row figures OVERSTATE one-sidedness "
        "and must not be quoted against the census.",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="0.4",
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = OUTDIR / "side_grain_comparison"
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")

    stem.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "source": str(SRC.relative_to(REPO)),
                "n_rows": n_rows,
                "grains": {
                    k: {kk: vv for kk, vv in g.items() if kk not in ("sr", "enr")}
                    for k, g in (("token", tok), ("occupancy", occ))
                },
                "headline": (
                    "On identical rows, per-token grain finds ~2x more strongly side-specialised "
                    f"features than row occupancy: |e|>2 {100 * occ['tails'][2]:.1f}% -> "
                    f"{100 * tok['tails'][2]:.1f}%, |e|>3 {100 * occ['tails'][3]:.1f}% -> "
                    f"{100 * tok['tails'][3]:.1f}%. Occupancy saturates by construction — a feature "
                    "firing at one token of a span and at every token both count 1."
                ),
                "caveats": [
                    "Strictly one-sided counts are identical across grains by construction and are "
                    "SAMPLE-SIZE dependent; do not quote the 2,000-row percentages against the "
                    "120,000-row census (1.27% context-only / 1.66% answer-only).",
                    "The two grains have DIFFERENT nulls (occupancy 0.683, token 0.731) because the "
                    "answer span is longer than the context span; never read either against 0.5.",
                    "Token grain comes from a 2,000-row capture; the banked occupancy census covers "
                    "120,000 rows. The comparison here is restricted to the shared 2,000 rows so it "
                    "is apples-to-apples, at the cost of more sampling noise than the census.",
                    "#1482 single-turn corpus (constant template prefix).",
                ],
                "what_is_plotted": (
                    "(a) answer-side ratio distribution at both grains over the same rows, each with "
                    "its own side-indifferent null marked. (b) null-centred log2 side-odds "
                    "enrichment for both grains, clipped to +-8, with the >=4x and >=8x tail "
                    "fractions annotated."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {stem}.png / .pdf / .meta.json")
    for k, g in (("occupancy", occ), ("token", tok)):
        print(
            f"  {k:<10} null={g['null']:.4f}  |e|>1 {100 * g['tails'][1]:.1f}%  "
            f"|e|>2 {100 * g['tails'][2]:.1f}%  |e|>3 {100 * g['tails'][3]:.1f}%  "
            f"p1={g['p1']:+.2f} p99={g['p99']:+.2f}"
        )


if __name__ == "__main__":
    main()

"""Round-5 natgen QA figure: realized cap-hit and drop rate per corpus.

Data source: per-ROW read of every answers.jsonl across the 35 on-policy
naturalistic cells (5 models x 7 corpora), aggregated on pod-1336-natgen
before teardown. Numbers are transcribed from that run and pinned here so the
figure is reproducible after the pod is gone; the underlying rows remain on
the HF data repo under
issue1336_rlvr_ladder/raw_completions/generation/<model>/<corpus>__gen_naturalistic
so per-cell detail is recomputable.

cap-hit = finish_reason == "length" / n   (the >2% re-generation trigger)
drop    = not kept / n
Both are fractions of ALL generated rows, not of kept rows -- the distinction
that made the audit aggregate misleading (see #1336 epm:progress v157).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# corpus -> (n, cap_hit_frac, trunc_kept, drop_frac)
AGG = {
    "gsm8k_train_full": (12365, 0.0150, 185, 0.0361),
    "gsm8k_test1319": (6595, 0.0161, 106, 0.0387),
    "lmsys23k": (90000, 0.0604, 4925, 0.4592),
    "sft11k": (53955, 0.0641, 3132, 0.2910),
    "if11k": (54345, 0.0772, 3912, 0.2457),
    "uf11k": (54770, 0.0954, 5050, 0.2815),
    "math7500": (37500, 0.1101, 4094, 0.0738),
}
TRIGGER = 0.02
TOTAL_N = 309530
TOTAL_CAPHIT = 0.0735
TOTAL_TRUNC_KEPT = 21404

# One colour = one meaning across both panels: cap-hit is always this blue,
# drop is always this orange.
C_CAP = "#3b6ea5"
C_DROP = "#d1793a"
C_UNDER = "#9bb7d4"


def main() -> None:
    order = sorted(AGG, key=lambda c: AGG[c][1])
    caps = [AGG[c][1] for c in order]
    drops = [AGG[c][3] for c in order]
    ns = [AGG[c][0] for c in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    ax = axes[0]
    colours = [C_CAP if v > TRIGGER else C_UNDER for v in caps]
    ax.barh(range(len(order)), [v * 100 for v in caps], color=colours)
    ax.axvline(TRIGGER * 100, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c}  (n={n:,})" for c, n in zip(order, ns)])
    ax.set_xlabel("cap-hit: finish_reason == 'length', % of all generated rows")
    ax.set_title(
        f"Realized cap-hit at max_tokens=1024\n"
        f"{TOTAL_N:,} rows over 35 cells; overall {TOTAL_CAPHIT:.2%}\n"
        f"dashed line = 2% re-gen trigger",
        fontsize=11,
    )
    ax.set_xlim(0, max(caps) * 100 * 1.15)

    ax = axes[1]
    ax.barh(range(len(order)), [v * 100 for v in drops], color=C_DROP)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([])
    ax.set_xlabel("dropped: not kept, % of all generated rows")
    ax.set_title(
        "Drop rate, same cells and same denominator\n"
        "dominated by empty_answer / short_turn,\nnot truncation",
        fontsize=11,
    )
    ax.set_xlim(0, max(drops) * 100 * 1.15)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parents[1] / "figures" / "issue_1336"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "natgen_caphit_by_corpus.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"corpora over trigger: {sum(1 for v in caps if v > TRIGGER)}/{len(caps)}")
    print(f"total truncated-but-kept rows: {TOTAL_TRUNC_KEPT:,}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""#1336: does the VOLUME of training data between two checkpoints predict how
badly the context->answer map transfers between them?

User ask 2026-08-12: "see if the number of examples trained on is correlated
with transfer quality (same thing as the data being trained on in between 2
stages)". This is the CONTINUOUS sibling of the binary `used_between` predicate
in `issue1336_full_transfer_lattice.fig_controlled_gap`: there the question was
"was THIS eval corpus used by the training in between", here it is "HOW MANY
rows did the training in between see", pooled over corpora.

TOPOLOGY (verified against the Hub cards 2026-08-12, and NOT the five-step chain
the pair list suggests): both RLVR checkpoints declare
`Finetuned from model: allenai/Llama-3.1-Tulu-3-8B-DPO`, and the 3.1 card says the
change is "an improvement only in the final RL stage ... switched from PPO to
GRPO". So the ladder is

    base -> SFT -> DPO -> { RLVR (PPO), RLVR-3.1 (GRPO) }

-- a three-step chain with a two-way branch at the end. RLVR -> RLVR-3.1 is a
comparison of two RL runs off a COMMON PARENT, not a training step, so
"rows trained in between" is UNDEFINED for it and that pair is dropped from this
analysis (named in the sidecar JSON, never silently).

Predictor, per ancestor->descendant pair (src, tgt): the sum of `train`-split row
counts over every step on the path from src down to tgt --

    base -> SFT       allenai/tulu-3-sft-mixture                     939,343
    SFT  -> DPO       allenai/llama-3.1-tulu-3-8b-preference-mixture 272,898
    DPO  -> RLVR      allenai/RLVR-GSM-MATH-IF-Mixed-Constraints      29,946
    DPO  -> RLVR-3.1  the same RLVR mix, GRPO instead of PPO          29,946

(counts + pinned revisions: docs/reports/issue_1336_stage_transfer_filled_template.md
section "Model"). Outcome: the per-pair transfer DEFICIT, ceiling - R2, where the
ceiling is the target's own within-model map -- the same DV the used-between
figure uses, so a corpus's intrinsic difficulty is divided out by its own ceiling.

THE CONFOUND, stated up front because it dominates the answer: the ladder's row
counts shrink monotonically (939k -> 273k -> 30k -> 30k), so "many rows in
between" and "base is the source" are very nearly the same statement. Any
all-pairs correlation is therefore mostly re-measuring the base boundary. The
script reports the all-pairs correlation AND the post-training-source-only
correlation, which is the part of the question that is actually free of that
confound.

Writes figures/issue_1336/ladder_rows_between_vs_deficit{,_t0567}.png/.pdf and a
sidecar JSON with every number plotted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the matplotlib/numpy imports.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1336_full_transfer_lattice as lat  # noqa: E402

OUTDIR = lat.OUTDIR
PAIRS = lat.PAIRS
DEGENERATE = lat.DEGENERATE

# Parent of each checkpoint, and the train-split rows the step INTO it trained on.
# Both RL checkpoints hang off DPO (see the module docstring) -- this is a tree,
# not a chain, and encoding it as a chain is what makes RLVR->RLVR-3.1 look like a
# 29,946-row step when it is a sibling comparison.
PARENT: dict[str, str | None] = {
    "base": None,
    "sft": "base",
    "dpo": "sft",
    "rlvr": "dpo",
    "rlvr_long": "dpo",
}
# Three ways to say "how much training", all read from the Hub model cards'
# "## Hyperparamters" sections (2026-08-12). They ORDER THE STAGES DIFFERENTLY,
# which is the point of reporting all three:
#   unique   -- distinct prompts in the mix; RLVR's 29,946 counted once
#   examples -- what the optimizer actually consumed: rows x epochs for SFT/DPO,
#               realized RL episodes (rollouts) for the two RLVR runs
#   steps    -- gradient updates = examples / effective batch size
# SFT 2 epochs @ batch 128 · DPO 1 epoch @ batch 128 · RLVR-PPO 100,000 episodes
# @ batch 224 · RLVR-GRPO 1,474,560 episodes (the released checkpoint, step 1920)
# @ batch 48*16=768. RLVR-GRPO trained ~14.7x the episodes of RLVR-PPO on the
# SAME 29,946-prompt mix, so "same dataset" and "same amount of RL" are not the
# same claim -- unique rows cannot see that and the other two can.
ROWS_INTO: dict[str, int] = {"sft": 939_343, "dpo": 272_898, "rlvr": 29_946, "rlvr_long": 29_946}
EXAMPLES_INTO: dict[str, int] = {
    "sft": 939_343 * 2,
    "dpo": 272_898,
    "rlvr": 100_000,
    "rlvr_long": 1_474_560,
}
STEPS_INTO: dict[str, int] = {
    "sft": round(939_343 * 2 / 128),
    "dpo": round(272_898 / 128),
    "rlvr": round(100_000 / 224),
    "rlvr_long": round(1_474_560 / 768),
}
PREDICTORS: dict[str, tuple[dict[str, int], str]] = {
    "unique": (ROWS_INTO, "unique prompt rows in the mixes trained on in between"),
    "examples": (EXAMPLES_INTO, "examples the optimizer consumed (rows × epochs, or RL episodes)"),
    "steps": (STEPS_INTO, "optimizer steps (gradient updates)"),
}

PRETTY = {
    "base": "base",
    "sft": "SFT",
    "dpo": "DPO",
    "rlvr": "RLVR-PPO",
    "rlvr_long": "RLVR-GRPO",
}


def amount_between(src: str, tgt: str, per_stage: dict[str, int]) -> int | None:
    """`per_stage` summed over the path src -> tgt, or None if src is no ancestor.

    Walks UP from tgt to the root; returns None if src is never reached (the
    sibling case, RLVR-PPO vs RLVR-GRPO, which share DPO as their parent).
    """
    total, node = 0, tgt
    while node is not None:
        if node == src:
            return total
        if node not in per_stage:  # walked past the root without meeting src
            return None
        total += per_stage[node]
        node = PARENT[node]
    return None


def _median_deficit(D: dict, pi: int, tier: int) -> float:
    """Median over non-degenerate corpora of (target's own ceiling - arm R2)."""
    vals = [
        c["within_r2"] - c["r2"]
        for surf, c in D["data"].get((pi, tier), [])
        if surf not in DEGENERATE
    ]
    return float(np.median(vals)) if vals else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Rank correlation + n, NaN-dropping. Ties get average ranks."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), len(x)

    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(1, len(v) + 1, dtype=float)
        for u in np.unique(v):  # average ranks within each tie group
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1]), len(x)


def collect(D: dict, tier: int, per_stage: dict[str, int]) -> list[dict]:
    """One row per pair: rows-between, deficit, and whether base is the source."""
    out, dropped = [], []
    for pi, (src, tgt) in enumerate(PAIRS):
        d = _median_deficit(D, pi, tier)
        amt = amount_between(src, tgt, per_stage)
        if amt is None:
            dropped.append(f"{PRETTY[src]}→{PRETTY[tgt]} (siblings off DPO — no path)")
            continue
        if not np.isfinite(d):
            continue
        out.append(
            {
                "pair": f"{PRETTY[src]}→{PRETTY[tgt]}",
                "src": src,
                "tgt": tgt,
                "amount": amt,
                "deficit": d,
                "base_source": src == "base",
            }
        )
    if dropped:
        print(f"  tier {tier}: dropped {len(dropped)} pair(s): {'; '.join(dropped)}")
    return out


def _panel(ax, rows: list[dict], tier: int, xlabel: str, show_legend: bool) -> dict:
    c_base, c_post = "#D62728", "#1F77B4"
    x = np.array([r["amount"] for r in rows], dtype=float)
    y = np.array([r["deficit"] for r in rows], dtype=float)
    is_base = np.array([r["base_source"] for r in rows])

    rho_all, n_all = _spearman(np.log10(x), y)
    rho_post, n_post = _spearman(np.log10(x[~is_base]), y[~is_base])

    for mask, col, lab in (
        (is_base, c_base, "base is the source"),
        (~is_base, c_post, "both post-trained"),
    ):
        if mask.any():
            ax.scatter(
                x[mask],
                y[mask],
                s=78,
                color=col,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
                label=lab,
            )
    # SFT->DPO / SFT->RLVR / SFT->RLVR-3.1 land within ~10% of each other in x AND
    # y, so a fixed label offset overprints them. Stagger by rank instead.
    for j, r in enumerate(sorted(rows, key=lambda q: (q["amount"], q["deficit"]))):
        ax.annotate(
            r["pair"],
            (r["amount"], r["deficit"]),
            textcoords="offset points",
            xytext=((10, 14, 10)[j % 3], (9, -6, -20)[j % 3]),
            fontsize=7.6,
            color="#404040",
            zorder=4,
        )
    ax.set_xscale("log")
    # Room on the right for the labels of the base-source cluster, which sits at
    # the max x; without it "base->RLVR-3.1" is clipped by the panel edge.
    ax.set_xlim(x.min() / 2.2, x.max() * 4.5)
    ax.margins(y=0.14)
    ax.axhline(0.0, color="#252525", lw=0.9, ls="--", zorder=1)
    ax.grid(color="#ececec", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=9.0)
    ax.set_ylabel(f"transfer deficit  (ceiling − R²)   ·   tier {tier}", fontsize=9.5)
    if show_legend:
        ax.legend(fontsize=8.4, frameon=False, loc="upper left")
    ax.set_title(
        f"{tier}: {lat.TIER_LABEL[tier]}\n"
        f"Spearman ρ = {rho_all:+.2f} (n={n_all} pairs)   ·   "
        f"post-trained sources only: ρ = {rho_post:+.2f} (n={n_post})",
        fontsize=9.5,
        pad=7,
    )
    return {"rho_all": rho_all, "n_all": n_all, "rho_post_only": rho_post, "n_post_only": n_post}


def main() -> None:
    D = lat.collect()
    tiers = [t for t in (0, 5) if t in lat.TIERS] or [lat.TIERS[0]]
    keys = list(PREDICTORS)

    fig = plt.figure(figsize=(6.9 * len(tiers) + 0.8, 4.9 * len(keys)))
    fig.set_layout_engine("none")  # the paper style's constrained_layout wins otherwise
    gs = fig.add_gridspec(
        len(keys),
        len(tiers),
        wspace=0.24,
        hspace=0.52,
        left=0.075,
        right=0.985,
        top=0.925,
        bottom=0.055,
    )

    stats, table = {}, {}
    for i, key in enumerate(keys):
        per_stage, xlabel = PREDICTORS[key]
        for j, tier in enumerate(tiers):
            rows = collect(D, tier, per_stage)
            table[f"{key}/t{tier}"] = rows
            stats[f"{key}/t{tier}"] = _panel(
                fig.add_subplot(gs[i, j]), rows, tier, xlabel, show_legend=(i == 0 and j == 0)
            )
    fig.suptitle(
        "Does the AMOUNT of training in between predict how badly the context→answer map transfers?\n"
        "three ways to count it — the row that matters is whichever one the ladder does not confound",
        fontsize=12.5,
        y=0.985,
    )

    out = OUTDIR / f"ladder_rows_between_vs_deficit{lat.SUFFIX}.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    side = out.with_suffix(".json")
    side.write_text(
        json.dumps(
            {
                "topology": dict(PARENT),
                "per_stage": {k: v[0] for k, v in PREDICTORS.items()},
                "layer": lat.LAYER,
                "stats": stats,
                "pairs": table,
            },
            indent=2,
        )
        + "\n"
    )
    for k, v in stats.items():
        print(
            f"{k:18s} rho_all={v['rho_all']:+.3f} (n={v['n_all']})  "
            f"rho_post_only={v['rho_post_only']:+.3f} (n={v['n_post_only']})"
        )
    print(f"wrote {out}")
    print(f"wrote {side}")


if __name__ == "__main__":
    main()

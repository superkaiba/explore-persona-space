"""Arrow-ladder summary figure (#1336): verdict rung per pair, all four arrows.

Supersedes the two-arrow Phase L version (t5d round, 300ee460be): the
inserted-arm reverse round adds the two REVERSE arrows, so the figure is now a
2x2 grid — left column the forward arrows (answer RE-ENCODING on source text /
CONTENT SHIFT under the target encoder), right column the reverse arrows
(RE-ENCODING on target text / CONTENT SHIFT under the source encoder).

x = the 10 forward pairs of the Tulu-3 ladder ordered by source stage (base
block shaded); y = the sufficiency rung ladder (identity < identity+bias <
rigid < rigid+scale < affine). Each open dot is ONE corpus's verdict — the
lowest rung whose paired (affine - rung) bootstrap CI includes 0 — over the 6
non-degenerate corpora (gsm8k_test1319 excluded: n < d = 4096 by design); the
filled marker is the per-pair MEDIAN rung index. Reads the battery JSONs
committed beside this script (eval_results/issue_1336/arrow_ladders/, 280
batteries: 140 forward + 140 reverse); nothing is fitted here.

    uv run python scripts/issue1336_arrow_ladder_fig.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the matplotlib import.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "eval_results" / "issue_1336" / "arrow_ladders"
OUTDIR = REPO / "figures" / "issue_1336" / "arrow_ladders"

RUNGS = ["identity", "id_bias", "rigid", "rigid_scale", "affine"]
RUNG_LABEL = ["identity", "identity + bias", "rigid (rotation)", "rigid + scale", "affine (ridge)"]
PAIRS = [
    ("base", "sft"),
    ("base", "dpo"),
    ("base", "rlvr"),
    ("base", "rlvr_long"),
    ("sft", "dpo"),
    ("sft", "rlvr"),
    ("sft", "rlvr_long"),
    ("dpo", "rlvr"),
    ("dpo", "rlvr_long"),
    ("rlvr", "rlvr_long"),
]
STAGE_LABEL = {
    "base": "base",
    "sft": "SFT",
    "dpo": "DPO",
    "rlvr": "RLVR-PPO",
    "rlvr_long": "RLVR-GRPO",
}
# panel layout: rows = (re-encoding, content), cols = (forward, reverse)
ARROW_GRID = [["reencode", "rev_reencode"], ["content", "content_srcenc"]]
ARROW_TITLE = {
    "reencode": "RE-ENCODING, forward: source answers under both encoders",
    "content": "CONTENT SHIFT, target encoder: source→target answers",
    "rev_reencode": "RE-ENCODING, reverse: target answers under both encoders",
    "content_srcenc": "CONTENT SHIFT, source encoder: source→target answers",
}


def _load() -> dict[tuple[str, str], list[int]]:
    """(arrow, pair-key) -> per-corpus verdict rung indices (non-degenerate only)."""
    out: dict[tuple[str, str], list[int]] = {}
    for fp in sorted(SRC.glob("arrow_*.json")):
        d = json.load(open(fp))
        if d.get("degenerate_n_lt_d"):
            continue
        assert d["status"] == "complete", fp
        pk = f"{d['pair']['source']}__{d['pair']['target']}"
        rung = RUNGS.index(d["verdict_lowest_rung_within_noise_of_affine"])
        out.setdefault((d["arrow"], pk), []).append(rung)
    return out


def main() -> None:
    """Render the 2x2 verdict dot plot; asserts full 10x4x6 coverage."""
    data = _load()
    arrows = [a for row in ARROW_GRID for a in row]
    for arrow in arrows:
        for s, t in PAIRS:
            got = data.get((arrow, f"{s}__{t}"), [])
            assert len(got) == 6, (arrow, s, t, len(got))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 7.0), sharex=True, sharey=True)
    xs = range(len(PAIRS))
    for (i, j), arrow in [((i, j), ARROW_GRID[i][j]) for i in range(2) for j in range(2)]:
        ax = axes[i][j]
        ax.axvspan(-0.5, 3.5, color="0.93", zorder=0)  # base-source block
        for x, (s, t) in zip(xs, PAIRS):
            rungs = data[(arrow, f"{s}__{t}")]
            counts: dict[int, int] = {}
            for r in rungs:
                counts[r] = counts.get(r, 0) + 1
            for r, c in sorted(counts.items()):
                ax.plot([x], [r], "o", ms=5 + 2.2 * c, mfc="none", mec="#1F77B4", mew=1.4, zorder=2)
            ax.plot(
                [x],
                [statistics.median(rungs)],
                marker="D",
                ms=6,
                color="#D62728",
                zorder=3,
                ls="none",
            )
        ax.set_title(f"arrow: {ARROW_TITLE[arrow]}", fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.3)
        if j == 0:
            ax.set_yticks(range(len(RUNGS)))
            ax.set_yticklabels(RUNG_LABEL, fontsize=9)
            ax.set_ylim(-0.5, len(RUNGS) - 0.5)
        if i == 1:
            ax.set_xticks(list(xs))
            ax.set_xticklabels(
                [f"{STAGE_LABEL[s]}→{STAGE_LABEL[t]}" for s, t in PAIRS],
                rotation=30,
                ha="right",
                fontsize=9,
            )
    axes[0][0].plot(
        [], [], "o", mfc="none", mec="#1F77B4", label="one corpus's verdict (size = count)"
    )
    axes[0][0].plot([], [], "D", color="#D62728", ls="none", label="median over 6 corpora")
    axes[0][0].legend(loc="upper right", fontsize=9, frameon=False)
    fig.suptitle(
        "Lowest rung statistically sufficient to match the affine map (layer 30) — "
        "forward (left) vs reverse (right) arrows",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUTDIR / "arrow_ladder_verdicts.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out)


if __name__ == "__main__":
    main()

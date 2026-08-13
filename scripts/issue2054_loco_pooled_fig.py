"""Figure for the leave-one-SPEAKER-out pooled map (#2054 writeup, wr6 companion).

Two panels (inserted | on-policy), one speaker group per x tick, two bars per
group: the FULL-pool map's held-out R2 and the LOCO map's held-out R2 on the
IDENTICAL rows. Per-cell points overlay each bar (the low-level view beside the
aggregate). Speaker colors are wr6's, so one color keeps one meaning across the
writeup.

The assistant's `bare_text` cells are the ONLY bare_text cells in the 56-cell
pool, so leave-assistant-out deletes that framing from training entirely — a
coverage confound, not a transfer result. Those points are drawn as open
triangles so the confound is visible on the canvas without a caption block; the
bars for the assistant are computed EXCLUDING them, and the excluded-framing
note lives in the surrounding prose and the meta.json sidecar.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    # script mode puts scripts/ (not the repo root) on sys.path[0] (gotchas.md).
    sys.path.insert(0, str(_REPO))

# wr6's palette — one color, one meaning across the writeup.
SPEAKER_COLOR = {
    "helios": "#000000",
    "wren": "#56B4E9",
    "vex": "#D55E00",
    "dana": "#CC79A7",
    "assistant": "#009E73",
}
SPEAKER_LABEL = {
    "helios": "HELIOS",
    "wren": "Wren",
    "vex": "Vex",
    "dana": "Dana",
    "assistant": "Assistant",
}
ORDER = ["helios", "wren", "vex", "dana", "assistant"]
CONDITIONS = [("inserted", "Inserted (verbatim reference answer)"), ("on_policy", "On-policy")]
# Assistant-only framing: leave-assistant-out removes it from training entirely.
CONFOUNDED_FRAMING = "bare_text"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument(
        "--results",
        type=Path,
        default=_REPO / "eval_results/issue_2054/specialization_ladder/loco_pooled.json",
    )
    ap.add_argument(
        "--figures-dir", type=Path, default=_REPO / "figures/issue_2054/specialization_ladder"
    )
    ap.add_argument("--stem", default="loco_pooled_recovery")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[loco-fig] import-check OK")
        return 0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from explore_persona_space.analysis.paper_plots import savefig_paper

    units = json.loads(args.results.read_text(encoding="utf-8"))["per_unit"]
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    width = 0.36
    for ax, (cond, title) in zip(axes, CONDITIONS, strict=True):
        for xi, spk in enumerate(ORDER):
            rows = [r for r in units if r["speaker"] == spk and r["condition"] == cond]
            keep = [r for r in rows if r["framing"] != CONFOUNDED_FRAMING]
            drop = [r for r in rows if r["framing"] == CONFOUNDED_FRAMING]
            if not keep:
                continue
            color = SPEAKER_COLOR[spk]
            full = st.mean(r["mean"]["full_pool_r2"] for r in keep)
            loco = st.mean(r["mean"]["loco_r2"] for r in keep)
            ax.bar(xi - width / 2, full, width, color=color, alpha=0.85, edgecolor="black", lw=0.6)
            ax.bar(
                xi + width / 2,
                loco,
                width,
                color=color,
                alpha=0.85,
                edgecolor="black",
                lw=0.6,
                hatch="///",
            )
            rng = np.random.default_rng(abs(hash((spk, cond))) % (2**31))
            for r in keep:
                ax.plot(
                    xi - width / 2 + rng.uniform(-0.09, 0.09),
                    r["mean"]["full_pool_r2"],
                    "o",
                    ms=3.2,
                    mfc="white",
                    mec="black",
                    mew=0.6,
                    zorder=3,
                )
                ax.plot(
                    xi + width / 2 + rng.uniform(-0.09, 0.09),
                    r["mean"]["loco_r2"],
                    "o",
                    ms=3.2,
                    mfc="white",
                    mec="black",
                    mew=0.6,
                    zorder=3,
                )
            # Confounded framing: drawn, never averaged into the bars.
            for r in drop:
                ax.plot(
                    xi + width / 2 + rng.uniform(-0.09, 0.09),
                    r["mean"]["loco_r2"],
                    "^",
                    ms=5.5,
                    mfc="none",
                    mec=color,
                    mew=1.3,
                    zorder=4,
                )
        ax.axhline(0.0, color="black", lw=0.8, ls="-")
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([SPEAKER_LABEL[s] for s in ORDER])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0].set_ylabel("Held-out $R^2$ (context $\\rightarrow$ answer)")

    handles = [
        Patch(facecolor="0.55", edgecolor="black", label="Full pool (speaker IN training)"),
        Patch(facecolor="0.55", edgecolor="black", hatch="///", label="Speaker held out (LOCO)"),
        Line2D([], [], marker="o", ls="", mfc="white", mec="black", ms=4, label="Per cell"),
        Line2D(
            [],
            [],
            marker="^",
            ls="",
            mfc="none",
            mec="#009E73",
            ms=6,
            label="Assistant-only framing (excluded from bars)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(
        "Does the pooled context$\\rightarrow$answer map transfer to a speaker it never saw?",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out = savefig_paper(fig, args.stem, dir=str(args.figures_dir))
    print(f"[loco-fig] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

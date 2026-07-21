# ruff: noqa: RUF002, RUF003
"""Issue #958 mixed turn-1+2 fit figure: held-out skill vs eval turn.

Reads eval_results/issue_958/mixed-turn-fit/mixed_fit.json and plots read-out-mean
skill (6-block mean) vs eval turn (1-4) for four arms — mix12_full, mix12_matchedn,
turn2_only, own_turn — each with its 997-draw paired-bootstrap 95% CI. The mixed
arms are the answer to Dan's question (does a turn-1+2 map generalize to the
unseen turns 3, 4?); the turn-2-only and own-turn arms are the apples-to-apples
baselines. Colorblind-safe palette, no annotation overlays, paper_plots rcParams.

CI error bars are non-negative offsets from the point estimate
(max(0, v-lo)/max(0, hi-v)) per the matplotlib xerr/yerr gotcha.

Writes figures/issue_958/mixed_turn_fit.png (+ .pdf + .meta.json sidecar).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue958_mixed_turn_fit_fig")

SRC = Path("eval_results/issue_958/mixed-turn-fit/mixed_fit.json")
FIG_DIR = Path("figures/issue_958")

# arm -> (label, palette role, matplotlib linestyle, x-dodge)
ARMS = [
    ("mix12_full", "mix turns 1+2 (full)", "primary", "-", -0.09),
    ("mix12_matchedn", "mix turns 1+2 (matched-n)", "accent", "-", -0.03),
    ("turn2_only", "turn-2 map only", "baseline", "--", 0.03),
    ("own_turn", "own-turn map (diagonal)", "neutral", ":", 0.09),
]


def main() -> int:
    res = json.loads(SRC.read_text())
    arms = res["arms"]
    eval_turns = res["eval_turns"]
    x = np.asarray(eval_turns, dtype=float)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")

    for arm, label, role, ls, dx in ARMS:
        per = arms[arm]["per_eval_turn"]
        vals, lo_off, hi_off = [], [], []
        for k in eval_turns:
            cell = per[str(k)]
            v = cell["readout_mean_skill"]
            lo, hi = cell["ci95"]
            vals.append(v)
            lo_off.append(max(0.0, v - lo))  # non-negative offsets (yerr gotcha)
            hi_off.append(max(0.0, hi - v))
        ax.errorbar(
            x + dx,
            vals,
            yerr=[lo_off, hi_off],
            marker="o",
            markersize=5,
            linestyle=ls,
            lw=1.7,
            capsize=3,
            color=paper_palette_role(role),
            label=label,
        )

    ax.axhline(0.0, color=paper_palette_role("neutral"), lw=0.9, alpha=0.5, zorder=0)
    ax.set_xticks(eval_turns)
    ax.set_xlabel("evaluation turn")
    ax.set_ylabel("held-out skill (6-block read-out mean)")
    ax.set_ylim(-0.15, 0.6)
    ax.legend(loc="lower right", fontsize=9)
    set_title_subtitle(
        ax,
        "Does a turn-1+2 map generalize across turns?",
        f"context->answer ridge, dup-excluded folds (n_test={res['n_test']}); 95% bootstrap CIs",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = savefig_paper(fig, "mixed_turn_fit", dir=FIG_DIR)
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

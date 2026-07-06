#!/usr/bin/env python3
"""Raw per-source-layer companion to hero2_retention (issue #841).

hero2_retention.png plots the *processed* retention ratio (transported r /
raw-target ceiling r) versus prediction horizon. This companion plots the raw
quantities that ratio is built from: the transported within-condition Pearson r
at each source layer (the numerator, one line per function class) against the
raw-target ceiling r (the constant denominator, dashed reference), system mode,
one panel per trait. It makes visible why the ratio exceeds 1 (transported r
above the ceiling) or goes negative (transported r below zero), which the
aggregate ratio hides.

Reads eval_results/issue_841/retention_curve.json (repo-root copy if present,
else the issue-841 worktree copy). Writes figures/issue_841/hero2_retention_raw
{.png,.pdf,.meta.json} via savefig_paper.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy freezes the BLAS pools.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("issue841_retention_perunit")

FIG_DIR = "figures/"
FIG_SUBDIR = "issue_841"

CLASS_LABEL = {
    "id_transport": "Identity transport",
    "ridge": "Affine (ridge)",
    "mlp": "Small MLP",
    "gru": "Depth-GRU (exploratory)",
}
CLASSES = ("id_transport", "ridge", "mlp", "gru")


def _resolve_json() -> Path:
    here = Path(__file__).resolve()
    # repo-root scripts/ -> repo-root eval_results/ ; else the worktree copy.
    candidates = [
        here.parents[1] / "eval_results" / FIG_SUBDIR / "retention_curve.json",
        here.parents[1]
        / ".claude"
        / "worktrees"
        / "issue-841"
        / "eval_results"
        / FIG_SUBDIR
        / "retention_curve.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"retention_curve.json not found; looked in {candidates}")


def _style():
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception as e:
        logger.warning("paper_plots style unavailable (%s); matplotlib default", e)


def _save(fig, stem: str) -> None:
    try:
        from explore_persona_space.analysis.paper_plots import savefig_paper

        savefig_paper(fig, f"{FIG_SUBDIR}/{stem}", dir=FIG_DIR)
    except Exception as e:
        logger.warning("savefig_paper failed (%s); plain savefig", e)
        out = Path(FIG_DIR) / FIG_SUBDIR
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    _style()
    d = json.loads(_resolve_json().read_text())
    traits = list(d.get("traits", {}).keys())
    fig, axes = plt.subplots(
        1, max(1, len(traits)), figsize=(5 * max(1, len(traits)), 4.5), squeeze=False
    )
    for ti, trait in enumerate(traits):
        ax = axes[0][ti]
        prim = d["traits"][trait].get("primary", {})
        ceil_val = None
        for cls in CLASSES:
            ents = prim.get(cls, [])
            if not ents:
                continue
            srcs, rows, ceils = [], [], []
            for e in ents:
                sysd = e.get("system")
                if not isinstance(sysd, dict):
                    continue
                srcs.append(e["source"])
                rows.append(sysd.get("r_row", np.nan))
                ceils.append(sysd.get("r_ceiling", np.nan))
            if not srcs:
                continue
            order = np.argsort(srcs)
            xs = np.array(srcs)[order]
            ys = np.array(rows, dtype=float)[order]
            ax.plot(xs, ys, marker="o", ms=3, label=CLASS_LABEL.get(cls, cls))
            ceil_finite = [c for c in ceils if c is not None and np.isfinite(c)]
            if ceil_finite:
                ceil_val = float(np.median(ceil_finite))
            # Label the ridge points with their source-layer index.
            if cls == "ridge":
                for x, y in zip(xs, ys):
                    if np.isfinite(y):
                        ax.text(x, y, f"{int(x)}", fontsize=6, ha="center", va="bottom")
        if ceil_val is not None:
            ax.axhline(
                ceil_val,
                ls="--",
                color="gray",
                lw=1,
                label=f"Raw-target ceiling r = {ceil_val:.2f}",
            )
        ax.axhline(0.0, ls=":", color="lightgray", lw=1)
        ax.set_xlabel("Source layer ℓ")
        ax.set_ylabel("Transported within-condition r (system mode)")
        ax.set_title(trait)
        ax.legend(fontsize=7)
    fig.suptitle("Raw transported correlation behind the retention ratio")
    fig.tight_layout()
    _save(fig, "hero2_retention_raw")
    logger.info("wrote figures/%s/hero2_retention_raw.{png,pdf,meta.json}", FIG_SUBDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Issue #1310 figures: per-persona base-vs-instruct map strength + swap gap.

Hero 1 — per-persona held-out R^2 at the headline layer, base vs instruct,
each bar annotated with its shuffle-null p97.5 band marker and its per-persona
n. Hero 2 — character-swap specificity: correct vs cross-persona-swapped
pooled R^2 per model + the dR2_char gap with its paired-bootstrap CI.

Pure JSON -> PNG/PDF (no torch). --fig-dir overrides the output root so smoke
runs never touch the committed figures/ tree.

CLI:
  uv run python scripts/issue1310_figures.py [--results-dir eval_results/issue_1310]
      [--fig-dir figures/issue_1310]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1310_common as c1310  # noqa: E402

SCRIPT = "scripts/issue1310_figures.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=Path("eval_results/issue_1310"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_1310"))
    return ap.parse_args()


def _load_summary(results_dir: Path) -> dict:
    p = results_dir / "summary.json"
    assert p.exists(), f"missing summary: {p} (run issue1310_fit.py first)"
    return json.loads(p.read_text())


def fig_perpersona(summary: dict, fig_dir: Path) -> None:
    """Grouped bars: per-persona headline R^2, base vs instruct, null markers."""
    personas = list(c1310.PERSONA_LABELS)
    models = [m for m in ("base", "instruct") if m in summary["per_persona"].get(personas[0], {})]
    if not models:
        models = ["base", "instruct"]
    x = np.arange(len(personas))
    width = 0.38
    fig, ax = plt.subplots()
    colors = {"base": paper_palette_role("baseline"), "instruct": paper_palette_role("primary")}
    for mi, model_kind in enumerate(models):
        r2s, nulls, ns = [], [], []
        for persona in personas:
            entry = (summary["per_persona"].get(persona) or {}).get(model_kind)
            r2s.append(np.nan if not entry else entry.get("r2_headline", np.nan))
            nulls.append(None if not entry else entry.get("null_p975_headline"))
            ns.append(None if not entry else entry.get("n"))
        offs = x + (mi - (len(models) - 1) / 2) * width
        ax.bar(offs, r2s, width, label=model_kind, color=colors.get(model_kind))
        for xi, (r2v, nv, nn) in zip(offs, zip(r2s, nulls, ns, strict=True), strict=True):
            if nv is not None and np.isfinite(nv):
                ax.hlines(nv, xi - width / 2, xi + width / 2, color="0.35", lw=1.2, zorder=5)
            if nn is not None and np.isfinite(r2v):
                ax.text(xi, r2v, f"n={nn}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(personas)
    ax.set_ylabel(f"held-out $R^2$ @ layer {summary['headline_layer']}")
    ax.set_title("Focused per-character context->dialogue map (base vs instruct)")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.legend(title="model")
    fig.text(0.01, 0.01, "grey ticks = shuffle-null p97.5 band", fontsize=6, color="0.4")
    savefig_paper(fig, "perpersona_r2", dir=str(fig_dir))
    plt.close(fig)


def fig_swap(summary: dict, fig_dir: Path) -> None:
    """Correct vs swapped pooled R^2 per model + dR2_char with CI."""
    swap = summary.get("swap_specificity", {})
    models = [m for m in ("base", "instruct") if swap.get(m)]
    if not models:
        print("[i1310-fig] no swap results — skipping swap panel")
        return
    fig, ax = plt.subplots()
    x = np.arange(len(models))
    width = 0.38
    correct = [swap[m].get("r2_correct", np.nan) for m in models]
    swapped = [swap[m].get("r2_swap", np.nan) for m in models]
    ax.bar(
        x - width / 2, correct, width, label="correct persona", color=paper_palette_role("primary")
    )
    ax.bar(
        x + width / 2, swapped, width, label="swapped persona", color=paper_palette_role("neutral")
    )
    for xi, m in zip(x, models, strict=True):
        d = swap[m].get("delta_r2_char")
        lo, hi = swap[m].get("delta_ci_lo"), swap[m].get("delta_ci_hi")
        if d is not None:
            ax.text(
                xi,
                max(correct[list(models).index(m)], swapped[list(models).index(m)]),
                f"ΔR²={d:.3f}\n[{lo:.3f},{hi:.3f}]",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(f"pooled held-out $R^2$ @ layer {summary['headline_layer']}")
    ax.set_title("Character-swap specificity (context->dialogue)")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.legend()
    savefig_paper(fig, "swap_specificity", dir=str(fig_dir))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print("[phase=p4_figures] figures")
    set_paper_style("blog")
    summary = _load_summary(args.results_dir)
    fig_perpersona(summary, args.fig_dir)
    fig_swap(summary, args.fig_dir)
    print(f"[i1310-fig] done -> {args.fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

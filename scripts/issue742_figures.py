#!/usr/bin/env python
# ruff: noqa: RUF001, RUF002
"""Issue #742 — hero + exploratory figures (plan v7 §6 "Figures to produce").

Reads the Stage-0/1/2 JSON outputs and renders:
  * **Hero (a)** — the bracket plot: per behavior×genre a ``√(r_yy) ± CI`` bar with
    the ``ρ_lin`` marker overlaid, behaviors ranked by bracket width, footnoted with
    which estimator-pair fed each ``√(r_yy)`` (§6 Hero (a)). THE HEADLINE FIGURE.
  * **Hero (b)** — Stage-1 LEACE-erased dCor with its permutation null + control-task
    null (only for behaviors where Stage 1 fired).
  * **Hero (c)** — the learning curve: ``√(r_yy)(n')`` vs ``n'`` with bootstrap bands.
  * **Exploratory** — the split-half-vs-binomial agreement scatter + the Bayes-error β bar.

Uses the project paper-quality rcParams. Writes to ``figures/issue_742/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import set_paper_style  # noqa: E402

set_paper_style()

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_742"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_742"

# Plain-English chart-internal labels (spec bans snake_case condition codes in chart text).
BEHAVIOR_LABELS = {
    "broad_em": "broad EM",
    "harmful_compliance": "harmful compliance",
    "sycophancy": "sycophancy",
    "refusal": "refusal",
}
GENRE_LABELS = {"betley": "Betley", "ultrachat": "UltraChat"}


def behavior_label(behavior: str) -> str:
    """Plain-English display name for a behavior slug."""
    return BEHAVIOR_LABELS.get(behavior, behavior.replace("_", " "))


def genre_label(genre: str) -> str:
    """Plain-English display name for a genre slug."""
    return GENRE_LABELS.get(genre, genre)


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def plot_brackets(s0: dict, out: Path) -> Path | None:
    brackets = s0.get("brackets", [])
    if not brackets:
        return None
    genres = sorted({b["genre"] for b in brackets})
    fig, axes = plt.subplots(1, len(genres), figsize=(6 * len(genres), 4), squeeze=False)
    for gi, genre in enumerate(genres):
        ax = axes[0][gi]
        cells = sorted(
            [b for b in brackets if b["genre"] == genre],
            key=lambda b: b["bracket_width"],
            reverse=True,
        )
        labels = [behavior_label(c["behavior"]) for c in cells]
        ceil = [c["sqrt_r_yy_headline"] for c in cells]
        ci = np.array([c["sqrt_r_yy_ci"] for c in cells])
        rho = [c["rho_lin"] for c in cells]
        y = np.arange(len(cells))
        lo_err = np.clip(np.array(ceil) - ci[:, 0], 0, None)
        hi_err = np.clip(ci[:, 1] - np.array(ceil), 0, None)
        ax.barh(y, ceil, xerr=[lo_err, hi_err], alpha=0.7, label="√(r_yy) ± CI")
        ax.scatter(rho, y, color="black", zorder=3, label="ρ_lin (#658)")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("correlation")
        ax.set_title(f"{genre_label(genre)}: bracket [ρ_lin, √(r_yy)]")
        ax.set_xlim(0, 1)
        if gi == 0:
            ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_learning_curve(s2: dict, out: Path) -> Path | None:
    curves = s2.get("curves", [])
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in curves:
        # issue742_learning_curve.py emits the per-n-prime points under `sqrt_r_yy_curve`
        # (a list of {n_prime, mean, ci, half_width}); `c["curve"]` was a stale key that
        # KeyError'd the whole figures phase (pre-existing schema drift, not the v8 gate).
        pts = c["sqrt_r_yy_curve"]
        ns = [p["n_prime"] for p in pts]
        means = [p["mean"] for p in pts]
        los = [p["ci"][0] for p in pts]
        his = [p["ci"][1] for p in pts]
        line = ax.plot(
            ns,
            means,
            marker="o",
            label=f"{genre_label(c['genre'])} / {behavior_label(c['behavior'])}",
        )[0]
        ax.fill_between(ns, los, his, alpha=0.15, color=line.get_color())
    ax.set_xlabel("n' (contexts)")
    ax.set_ylabel("√(r_yy)(n')")
    ax.set_title("Stage 2 — reliability-ceiling learning curve")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_agreement_scatter(s0: dict, out: Path) -> Path | None:
    cells = [b for b in s0.get("brackets", []) if b.get("r_yy_split_half") is not None]
    if not cells:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    sh = [c["r_yy_split_half"] for c in cells]
    bn = [c["r_yy_binomial"] for c in cells]
    ax.scatter(sh, bn)
    pts = [(c["r_yy_split_half"], c["r_yy_binomial"]) for c in cells]
    for i, c in enumerate(cells):
        # Full plain-English labels (no abbreviated condition codes; project rule).
        x, y = pts[i]
        # Near-coincident pair: label the higher point above, the lower below,
        # so each label sits on its own point's side. Right-align near the edge.
        close = [
            (px, py)
            for j, (px, py) in enumerate(pts)
            if j != i and abs(px - x) < 0.05 and abs(py - y) < 0.05
        ]
        below = any(py > y for _, py in close)
        ax.annotate(
            f"{genre_label(c['genre'])} / {behavior_label(c['behavior'])}",
            (x, y),
            fontsize=6.5,
            xytext=(-6 if x > 0.85 else 5, -11 if below else 4),
            textcoords="offset points",
            ha="right" if x > 0.85 else "left",
        )
    lim = [0, 1]
    ax.plot(lim, lim, "k--", alpha=0.4)
    ax.set_xlabel("split-half r_yy")
    ax.set_ylabel("binomial r_yy")
    ax.set_title("Estimator agreement (split-half vs binomial)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run(*, eval_dir: Path, fig_dir: Path) -> dict:
    s0 = _load(eval_dir / "stage0_brackets.json") or _load(eval_dir / "stage0_brackets_smoke.json")
    s1 = _load(eval_dir / "stage1_leace_dcor.json") or _load(
        eval_dir / "stage1_leace_dcor_smoke.json"
    )
    s2 = _load(eval_dir / "stage2_learning_curve.json") or _load(
        eval_dir / "stage2_learning_curve_smoke.json"
    )
    written: list[str] = []
    if s0:
        p = plot_brackets(s0, fig_dir / "hero_a_brackets.png")
        if p:
            written.append(str(p))
        p = plot_agreement_scatter(s0, fig_dir / "exploratory_agreement_scatter.png")
        if p:
            written.append(str(p))
    if s2:
        p = plot_learning_curve(s2, fig_dir / "hero_c_learning_curve.png")
        if p:
            written.append(str(p))
    # hero (b) Stage-1 dCor null only renders when Stage 1 fired; s1 may be absent-by-design
    return {"figures_written": written, "stage1_present": s1 is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #742 figures.")
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--smoke", action="store_true", help="render from *_smoke.json outputs")
    args = parser.parse_args()
    result = run(eval_dir=args.eval_dir, fig_dir=args.fig_dir)
    print(f"[phase=figures] wrote {len(result['figures_written'])} figures to {args.fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

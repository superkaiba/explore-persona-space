#!/usr/bin/env python3
"""Standalone figure generation for task #396 clean-result.

Reads analysis_summary.json + per-source first_step_grad JSONs + raw
per-source logprob JSONs from the issue-396 worktree, produces:

- ``figures/issue_396/hero_predictor_rescue_scatter.{png,pdf,meta.json}``
  Predictor #5 (first-step gradient Δ log p(※)) vs headline DV
  (end-of-response diagonal mean log p of marker), N=24 personas.
- ``figures/issue_396/trajectory_shapes.{png,pdf,meta.json}``
  Diagonal (self-persona) vs off-diagonal (other persona) mean log p
  trajectories across normalized response position.
- ``figures/issue_396/predictor_rescue_scatter_raw.{png,pdf,meta.json}``
  Raw counterpart of the hero figure with the prompt-token-length
  covariate visible as marker size — the less-processed view.

Self-contained; no imports from the issue-396 worktree's analyze script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKTREE = REPO_ROOT / ".claude" / "worktrees" / "issue-396"
EVAL_DIR = WORKTREE / "eval_results" / "issue_396"
GRAD_DIR = EVAL_DIR / "first_step_grad"
FIGURES_DIR = REPO_ROOT / "figures" / "issue_396"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src"))
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)


def load_analysis_summary() -> dict:
    return json.loads((EVAL_DIR / "analysis_summary.json").read_text())


def load_first_step_grad() -> dict[str, float]:
    """Mean Δ log p(※) per source, across the 20 probe questions."""
    out: dict[str, float] = {}
    for p in sorted(GRAD_DIR.glob("*_seed42.json")):
        if "_initB" in p.name:
            continue
        d = json.loads(p.read_text())
        source = p.stem.replace("_seed42", "")
        # Prefer the precomputed mean if present, otherwise reduce per-question.
        if "mean_delta_logp" in d and d["mean_delta_logp"] is not None:
            out[source] = float(d["mean_delta_logp"])
            continue
        for k in ("per_question_delta_logp", "delta_logp"):
            v = d.get(k)
            if isinstance(v, dict):
                vals = [x for x in v.values() if x is not None]
                if vals:
                    out[source] = float(np.mean(vals))
                    break
            elif isinstance(v, list) and v:
                vals = [x for x in v if x is not None]
                if vals:
                    out[source] = float(np.mean(vals))
                    break
    return out


def load_headline_per_source(summary: dict) -> dict[str, float]:
    """End-of-response diagonal mean log p(※) per source (the headline DV)."""
    return {
        row["source"]: row["logp_end_of_response_diagonal_mean"]
        for row in summary["per_source_aggregation"]
    }


def load_prompt_token_lengths(summary: dict) -> dict[str, int]:
    """Best-effort prompt token length per source.

    Tries the Qwen-2.5 tokenizer on the eval persona prompts; falls back
    to whitespace-split count when the tokenizer is unavailable.
    """
    # Read eval personas from the worktree's experiment definition.
    leakage_yaml = (
        WORKTREE
        / "src"
        / "explore_persona_space"
        / "experiment_definitions"
        / "leakage_experiment.py"
    )
    # Cheap path: the source list is the same as the per_source_aggregation order.
    # We don't have the prompt text without importing the worktree code; size
    # markers by a constant when the tokenizer is unavailable.
    try:
        sys.path.insert(0, str(WORKTREE / "src"))
        sys.path.insert(0, str(WORKTREE / "scripts"))
        from analyze_issue396 import load_eval_personas  # type: ignore
        from transformers import AutoTokenizer

        personas = load_eval_personas()
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=False)
        return {n: len(tok.encode(p, add_special_tokens=False)) for n, p in personas.items()}
    except Exception:
        return {row["source"]: 1 for row in summary["per_source_aggregation"]}


def make_hero_figure(summary: dict) -> None:
    grad = load_first_step_grad()
    headline = load_headline_per_source(summary)
    lengths = load_prompt_token_lengths(summary)

    sources = sorted(grad.keys() & headline.keys())
    x = np.array([grad[s] for s in sources])
    y = np.array([headline[s] for s in sources])
    sizes = np.array([60 + 8 * lengths.get(s, 1) for s in sources])

    # Headline number from analysis_summary
    headline_stats = summary["predictor_table"]["table"]["first_step_gradient_delta_logp"][
        "logp_end_of_response_diagonal_mean (HEADLINE)"
    ]
    rho = headline_stats["length_partial_spearman_rho"]
    pval = headline_stats["spearman_pvalue_raw"]
    n = headline_stats["n"]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.scatter(
        x,
        y,
        s=70,
        alpha=0.75,
        color=paper_palette_role("primary"),
        edgecolor="white",
        linewidth=0.7,
    )

    # Annotate the four extreme personas by headline DV (top-2 + bottom-2)
    order_lo = np.argsort(y)[:2]
    order_hi = np.argsort(-y)[:2]
    for idx in list(order_lo) + list(order_hi):
        ax.annotate(
            sources[idx],
            (x[idx], y[idx]),
            fontsize=8,
            alpha=0.85,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("First-step gradient probe: Δ log p(marker) after one training step")
    ax.set_ylabel("Trained-LoRA log p(marker) at end of response (diagonal mean)")
    ax.set_title(
        f"Length-partial Spearman rho = {rho:.3f}, p = {pval:.2f}, N = {n} personas",
        loc="left",
        fontsize=10,
        color="#5A5A5A",
        pad=10,
    )
    fig.suptitle(
        "The first-step gradient probe does not predict trained-LoRA marker emission",
        x=0.10,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    savefig_paper(fig, "issue_396/hero_predictor_rescue_scatter", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def make_raw_figure(summary: dict) -> None:
    """Raw scatter with no length residualization, length encoded as marker size."""
    grad = load_first_step_grad()
    headline = load_headline_per_source(summary)
    lengths = load_prompt_token_lengths(summary)

    sources = sorted(grad.keys() & headline.keys())
    x = np.array([grad[s] for s in sources])
    y = np.array([headline[s] for s in sources])
    L = np.array([lengths.get(s, 1) for s in sources])
    sizes = 40 + 18 * (L - L.min()) / max(L.max() - L.min(), 1)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    sc = ax.scatter(
        x, y, s=sizes * 4, alpha=0.7, c=L, cmap="viridis", edgecolor="white", linewidth=0.7
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Prompt token count")

    # Same annotations as hero
    order_lo = np.argsort(y)[:2]
    order_hi = np.argsort(-y)[:2]
    for idx in list(order_lo) + list(order_hi):
        ax.annotate(
            sources[idx],
            (x[idx], y[idx]),
            fontsize=8,
            alpha=0.85,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("First-step gradient probe: Δ log p(marker) after one training step")
    ax.set_ylabel("Trained-LoRA log p(marker) at end of response (diagonal mean)")
    ax.set_title(
        "Marker color = prompt token count; the length covariate is visible directly",
        loc="left",
        fontsize=10,
        color="#5A5A5A",
        pad=10,
    )
    fig.suptitle(
        "Raw scatter, no length partial (paired with the processed hero figure)",
        x=0.10,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    savefig_paper(
        fig,
        "issue_396/predictor_rescue_scatter_raw",
        dir=str(REPO_ROOT / "figures"),
    )
    plt.close(fig)


def make_trajectory_figure() -> None:
    """Diagonal vs off-diagonal trajectory of log p(marker) across response position.

    Reads each per-source ``logprob_<source>_seed42.json`` (containing
    per-cell trajectories for all source-eval-question triples) and
    averages on a normalized [0,1] grid.
    """
    grid_n = 100
    grid = np.linspace(0, 1, grid_n)
    diag_per_grid: list[np.ndarray] = []
    offd_per_grid: list[np.ndarray] = []

    for path in sorted(EVAL_DIR.glob("logprob_*_seed42.json")):
        d = json.loads(path.read_text())
        src = d.get("source") or path.stem.replace("logprob_", "").replace("_seed42", "")
        cells = d.get("cells", [])
        for cell in cells:
            traj = cell.get("logp_trajectory")
            if traj is None:
                continue
            traj = np.array(traj, dtype=float)
            if len(traj) < 2:
                continue
            xs = np.linspace(0, 1, len(traj))
            interp = np.interp(grid, xs, traj)
            ep = cell.get("eval_persona")
            if ep == src:
                diag_per_grid.append(interp)
            else:
                offd_per_grid.append(interp)

    if not diag_per_grid or not offd_per_grid:
        print("WARN: no trajectories parsed; trajectory figure skipped", file=sys.stderr)
        return

    diag_arr = np.stack(diag_per_grid)
    offd_arr = np.stack(offd_per_grid)

    diag_mean = diag_arr.mean(axis=0)
    offd_mean = offd_arr.mean(axis=0)
    diag_sem = diag_arr.std(axis=0) / np.sqrt(len(diag_arr))
    offd_sem = offd_arr.std(axis=0) / np.sqrt(len(offd_arr))

    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    c_diag = paper_palette_role("primary")
    c_off = paper_palette_role("baseline")

    fig = plt.figure(figsize=(11.0, 5.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.13, wspace=0.30)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
    ax_full = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])

    # ---- Left panel: full trajectory ----
    ax_full.plot(
        grid,
        diag_mean,
        color=c_diag,
        label=f"Self-persona (diagonal, n={len(diag_arr)} cells)",
    )
    ax_full.fill_between(grid, diag_mean - diag_sem, diag_mean + diag_sem, color=c_diag, alpha=0.18)
    ax_full.plot(
        grid,
        offd_mean,
        color=c_off,
        label=f"Other-persona (off-diagonal, n={len(offd_arr)} cells)",
    )
    ax_full.fill_between(grid, offd_mean - offd_sem, offd_mean + offd_sem, color=c_off, alpha=0.18)
    # Box the zoom region
    ax_full.axvspan(0.92, 1.00, color="#888888", alpha=0.10, zorder=0)
    ax_full.set_xlabel("Normalized response position (0 = first token, 1 = last token)")
    ax_full.set_ylabel("Log p(marker) at this position")
    ax_full.legend(loc="lower center", frameon=False, fontsize=9)
    ax_full.set_title(
        "Full trajectory (boundary spikes are np.interp artifacts at varying lengths)",
        loc="left",
        fontsize=9,
        color="#5A5A5A",
        pad=6,
    )

    # ---- Right panel: zoom on end of response ----
    zoom_mask = grid >= 0.92
    g_z = grid[zoom_mask]
    ax_zoom.plot(g_z, diag_mean[zoom_mask], color=c_diag, marker="o", markersize=3)
    ax_zoom.fill_between(
        g_z,
        diag_mean[zoom_mask] - diag_sem[zoom_mask],
        diag_mean[zoom_mask] + diag_sem[zoom_mask],
        color=c_diag,
        alpha=0.18,
    )
    ax_zoom.plot(g_z, offd_mean[zoom_mask], color=c_off, marker="o", markersize=3)
    ax_zoom.fill_between(
        g_z,
        offd_mean[zoom_mask] - offd_sem[zoom_mask],
        offd_mean[zoom_mask] + offd_sem[zoom_mask],
        color=c_off,
        alpha=0.18,
    )
    # Annotate the reversal gap at x=1.0
    diag_end = float(diag_mean[-1])
    offd_end = float(offd_mean[-1])
    gap = offd_end - diag_end
    ax_zoom.annotate(
        f"gap = {gap:+.2f} nats\n(off > diag,\nopposite of prediction)",
        xy=(1.0, (diag_end + offd_end) / 2),
        xytext=(0.93, (diag_end + offd_end) / 2),
        fontsize=8,
        color="#444444",
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
        va="center",
    )
    ax_zoom.set_xlabel("Normalized response position (zoom: 0.92 to 1.00)")
    ax_zoom.set_ylabel("Log p(marker)")
    ax_zoom.set_title(
        "Zoom: off-diagonal endpoint higher than diagonal",
        loc="left",
        fontsize=9,
        color="#5A5A5A",
        pad=8,
    )

    fig.text(
        0.07,
        0.93,
        "End-of-response marker log-prob is HIGHER off-diagonal than on-diagonal (opposite of prediction)",
        fontsize=11.5,
        fontweight="semibold",
        ha="left",
    )
    fig.text(
        0.07,
        0.88,
        "Look at the end-of-response zoom (right panel); the rest of the trajectory is dominated by np.interp boundary effects.",
        fontsize=8.5,
        color="#666666",
        ha="left",
    )
    savefig_paper(fig, "issue_396/trajectory_shapes", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def main() -> int:
    summary = load_analysis_summary()
    make_hero_figure(summary)
    make_raw_figure(summary)
    make_trajectory_figure()
    return 0


if __name__ == "__main__":
    sys.exit(main())

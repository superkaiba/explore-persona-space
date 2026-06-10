#!/usr/bin/env python3
"""task #511 — figures for the probe-count convergence sweep.

Reads ``eval_results/issue_511/probe_count_sweep_results.json`` and emits:

- **HERO** ``convergence_fixed_cell.{png,pdf}`` — two-panel |ρ| + CV R² as
  a function of N at the headline cell (``last_prompt × L22 × gauss_kl ×
  raw``) on loc_ep1. Error bars from R subsets per N.
- ``convergence_ridge.{png,pdf}`` — full L19-L24 × {gauss_kl, mmd, wass2}
  ridge, one line per cell, CV R² panel only.
- ``baselines.{png,pdf}`` — same-layer cosine controls (L19-L24) + sentinel
  cosine (L0/11/21/27).
- ``raw_scatter_headline.{png,pdf}`` — one scatter per N showing the raw
  per-subset (|ρ|, CV R²) points for the headline cell.
- ``per_checkpoint_headline.{png,pdf}`` — if multiple epochs were scored,
  the headline cell's |ρ| + CV R² across loc_ep{1,2,3,5}.

Caption + meta.json includes plateau verdict from the sweep payload.

Smoke mode (``--smoke``) reads ``smoke_results.json`` instead and writes
``smoke_convergence.png`` only.
"""

# ruff: noqa: RUF001, RUF002

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis import paper_plots  # noqa: E402

FIG_DIR = PROJECT_ROOT / "figures" / "issue_511"

logger = logging.getLogger("i511.figures")

HEADLINE_CELL_ID = "last_prompt__L22__gauss_kl__raw"
L19_L24 = tuple(range(19, 25))
CLOUD_METRICS = ("gauss_kl", "mmd", "wass2")
COSINE_SENTINELS = (0, 11, 21, 27)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _meta_payload(extra: dict) -> dict:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        **extra,
    }


def _save_with_meta(fig: plt.Figure, base: Path, payload: dict) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    base.with_suffix(".meta.json").write_text(json.dumps(_meta_payload(payload), indent=2))


def _agg_by_N(
    aggregates: dict, cell_id: str, arm: str, epoch: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (N, abs_rho_mean, abs_rho_std, cv_mean, cv_std) sorted by N."""
    rows: list[tuple[int, float, float, float, float]] = []
    for key, agg in aggregates.items():
        cid, a, ep_str, n_str = key.split("|")
        if cid != cell_id or a != arm or int(ep_str) != epoch:
            continue
        rows.append(
            (
                int(n_str),
                float(agg.get("abs_rho_mean", float("nan"))),
                float(agg.get("abs_rho_std", float("nan"))),
                float(agg.get("cv_mean", float("nan"))),
                float(agg.get("cv_std", float("nan"))),
            )
        )
    rows.sort(key=lambda x: x[0])
    if not rows:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty
    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]


def plot_convergence_fixed_cell(payload: dict, out_dir: Path) -> None:
    """HERO. Two-panel |ρ| + CV R² convergence at headline cell on (loc, 1)."""
    paper_plots.set_paper_style()
    aggregates = payload["aggregates"]
    arm = payload.get("arm", "loc")
    epoch = int(payload.get("epochs", [1])[0])
    N, rho_m, rho_s, cv_m, cv_s = _agg_by_N(aggregates, HEADLINE_CELL_ID, arm, epoch)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].errorbar(N, rho_m, yerr=rho_s, marker="o", color="#1f77b4", linewidth=1.6, capsize=3)
    axes[0].set_xlabel("N probes per persona")
    axes[0].set_ylabel("|ρ| (length-partial Spearman)")
    axes[0].set_title(f"Headline cell — |ρ| vs N\n{HEADLINE_CELL_ID} · {arm}_ep{epoch}")
    axes[0].grid(True, alpha=0.3)
    axes[1].errorbar(N, cv_m, yerr=cv_s, marker="s", color="#d62728", linewidth=1.6, capsize=3)
    axes[1].set_xlabel("N probes per persona")
    axes[1].set_ylabel("CV R² (LOCO, length-partialed)")
    axes[1].set_title("Headline cell — CV R² vs N")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plateau = payload.get("plateau_verdict") or {}
    verdict_key = f"{HEADLINE_CELL_ID}|{arm}|ep{epoch}"
    verdict = plateau.get(verdict_key, {})
    _save_with_meta(
        fig,
        out_dir / "convergence_fixed_cell",
        {
            "figure": "convergence_fixed_cell",
            "cell_id": HEADLINE_CELL_ID,
            "arm": arm,
            "epoch": epoch,
            "N_grid": N.tolist(),
            "abs_rho_mean": rho_m.tolist(),
            "abs_rho_std": rho_s.tolist(),
            "cv_mean": cv_m.tolist(),
            "cv_std": cv_s.tolist(),
            "plateau_verdict": verdict,
        },
    )


def plot_convergence_ridge(payload: dict, out_dir: Path) -> None:
    """L19-L24 × {gauss_kl, mmd, wass2} ridge — one line per cell, CV R² panel."""
    paper_plots.set_paper_style()
    aggregates = payload["aggregates"]
    arm = payload.get("arm", "loc")
    epoch = int(payload.get("epochs", [1])[0])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    color_by_metric = {"gauss_kl": "#1f77b4", "mmd": "#2ca02c", "wass2": "#9467bd"}
    style_by_layer = {19: "-", 20: "--", 21: "-.", 22: ":", 23: "-", 24: "--"}
    for m in CLOUD_METRICS:
        for L in L19_L24:
            cell_id = f"last_prompt__L{L}__{m}__raw"
            N, _, _, cv_m, cv_s = _agg_by_N(aggregates, cell_id, arm, epoch)
            if N.size == 0:
                continue
            ax.errorbar(
                N,
                cv_m,
                yerr=cv_s,
                marker="o",
                markersize=3,
                color=color_by_metric[m],
                linestyle=style_by_layer.get(L, "-"),
                linewidth=1.1,
                alpha=0.8,
                capsize=2,
                label=f"{m} · L{L}",
            )
    ax.set_xlabel("N probes per persona")
    ax.set_ylabel("CV R² (LOCO, length-partialed)")
    ax.set_title(f"L19-L24 cloud-aware ridge · {arm}_ep{epoch}")
    ax.legend(loc="best", fontsize=7, ncol=3, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_with_meta(
        fig,
        out_dir / "convergence_ridge",
        {"figure": "convergence_ridge", "arm": arm, "epoch": epoch, "layers": list(L19_L24)},
    )


def plot_baselines(payload: dict, out_dir: Path) -> None:
    """Same-layer cosine controls (L19-L24) + sentinel cosines (L0/11/21/27)
    + the headline cloud-aware cell as a reference line."""
    paper_plots.set_paper_style()
    aggregates = payload["aggregates"]
    arm = payload.get("arm", "loc")
    epoch = int(payload.get("epochs", [1])[0])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for L in L19_L24:
        cell_id = f"last_prompt__L{L}__cosine__raw"
        N, _, _, cv_m, cv_s = _agg_by_N(aggregates, cell_id, arm, epoch)
        if N.size == 0:
            continue
        ax.errorbar(
            N,
            cv_m,
            yerr=cv_s,
            marker="o",
            markersize=3,
            color="#666666",
            alpha=0.75,
            linewidth=1.0,
            capsize=2,
            label=f"cos L{L}",
        )
    for L in COSINE_SENTINELS:
        cell_id = f"last_prompt__L{L}__cosine__raw"
        N, _, _, cv_m, cv_s = _agg_by_N(aggregates, cell_id, arm, epoch)
        if N.size == 0:
            continue
        ax.errorbar(
            N,
            cv_m,
            yerr=cv_s,
            marker="^",
            markersize=4,
            linestyle="--",
            linewidth=1.4,
            capsize=2,
            label=f"cos L{L} (sentinel)",
        )
    # Reference: headline cloud-aware cell
    N, _, _, cv_m, cv_s = _agg_by_N(aggregates, HEADLINE_CELL_ID, arm, epoch)
    if N.size > 0:
        ax.errorbar(
            N,
            cv_m,
            yerr=cv_s,
            marker="s",
            markersize=5,
            color="#d62728",
            linewidth=2.0,
            capsize=3,
            label="gauss_kl L22 (headline)",
        )
    ax.set_xlabel("N probes per persona")
    ax.set_ylabel("CV R² (LOCO, length-partialed)")
    ax.set_title(f"Mean-based baselines vs cloud-aware headline · {arm}_ep{epoch}")
    ax.legend(loc="best", fontsize=7, ncol=2, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_with_meta(
        fig,
        out_dir / "baselines",
        {"figure": "baselines", "arm": arm, "epoch": epoch},
    )


def plot_raw_scatter_headline(payload: dict, out_dir: Path) -> None:
    """Per-subset raw points for the headline cell (the noise distribution
    that drives the error bars in the hero figure)."""
    paper_plots.set_paper_style()
    arm = payload.get("arm", "loc")
    epoch = int(payload.get("epochs", [1])[0])
    rows = [
        r
        for r in payload["rows"]
        if r["cell_id"] == HEADLINE_CELL_ID
        and r["arm"] == arm
        and int(r["epoch"]) == epoch
        and r["status"] == "ok"
    ]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    Ns = sorted({r["N"] for r in rows})
    cmap = plt.get_cmap("viridis", len(Ns))
    for i, N in enumerate(Ns):
        sub = [r for r in rows if r["N"] == N]
        rho = np.array([abs(r["rho"]) for r in sub])
        cv = np.array([r["cv_r2"] for r in sub])
        ax.scatter(rho, cv, s=18, color=cmap(i), label=f"N={N}", alpha=0.85)
    ax.set_xlabel("|ρ|  (per subset)")
    ax.set_ylabel("CV R² (per subset)")
    ax.set_title(f"Headline cell · raw per-subset points · {arm}_ep{epoch}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_with_meta(fig, out_dir / "raw_scatter_headline", {"figure": "raw_scatter_headline"})


def plot_per_checkpoint_headline(payload: dict, out_dir: Path) -> None:
    """If >1 epoch was scored, plot the headline cell across loc_ep{1,2,3,5}."""
    aggregates = payload["aggregates"]
    arm = payload.get("arm", "loc")
    epochs = sorted(payload.get("epochs", [1]))
    if len(epochs) < 2:
        return
    paper_plots.set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    cmap = plt.get_cmap("plasma", len(epochs))
    for i, ep in enumerate(epochs):
        N, rho_m, rho_s, cv_m, cv_s = _agg_by_N(aggregates, HEADLINE_CELL_ID, arm, int(ep))
        if N.size == 0:
            continue
        axes[0].errorbar(
            N, rho_m, yerr=rho_s, marker="o", color=cmap(i), capsize=2, label=f"{arm}_ep{ep}"
        )
        axes[1].errorbar(
            N, cv_m, yerr=cv_s, marker="s", color=cmap(i), capsize=2, label=f"{arm}_ep{ep}"
        )
    for a, ylabel, title in (
        (axes[0], "|ρ|", "Headline |ρ| vs N · per checkpoint"),
        (axes[1], "CV R²", "Headline CV R² vs N · per checkpoint"),
    ):
        a.set_xlabel("N probes per persona")
        a.set_ylabel(ylabel)
        a.set_title(title)
        a.grid(True, alpha=0.3)
        a.legend(loc="best", fontsize=7)
    fig.tight_layout()
    _save_with_meta(
        fig,
        out_dir / "per_checkpoint_headline",
        {"figure": "per_checkpoint_headline", "epochs": list(epochs)},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot task #511 convergence figures.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_511" / "probe_count_sweep_results.json"),
        help="Sweep payload JSON.",
    )
    parser.add_argument("--out-dir", type=str, default=str(FIG_DIR), help="Figures output dir.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Read smoke_results.json and emit smoke_convergence.png only.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    if args.smoke:
        inp = inp.with_name("smoke_results.json")
        out_dir = out_dir / "smoke"
    if not inp.exists():
        raise FileNotFoundError(f"sweep payload missing: {inp}")

    payload = json.loads(inp.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        # Single-figure smoke: just the headline-cell two-panel hero.
        plot_convergence_fixed_cell(payload, out_dir)
        logger.info("smoke figure written to %s", out_dir)
        return 0

    plot_convergence_fixed_cell(payload, out_dir)
    plot_convergence_ridge(payload, out_dir)
    plot_baselines(payload, out_dir)
    plot_raw_scatter_headline(payload, out_dir)
    plot_per_checkpoint_headline(payload, out_dir)
    logger.info("all figures written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

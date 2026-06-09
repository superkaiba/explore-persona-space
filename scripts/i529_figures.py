"""Issue #529 — hero + exploratory figures.

Produces (plan §4.6 Phase 4):

  * Hero #1: forest plot of d_seed = L_system_plain - L_role (and
    L_system_padded - L_role) across the 5 seeds AT the selected
    anchor, with the saturated-5-epoch reference ghosted.
    -> figures/issue_529/hero_role_vs_system_at_anchor.{png,pdf,meta.json}
  * Hero #2: log P(' ※') per arm vs E, three rows (own / wrong /
    default_assistant), per persona; selected anchor marked with a
    vertical guide.
    -> figures/issue_529/hero_trajectory_per_arm.{png,pdf,meta.json}
  * Exploratory dump: per-cell heatmap (arm x seed x E), per-seed
    dynamic-range trajectories, own-emission-vs-E.
    -> figures/issue_529/exploratory_dump/*

Inputs (read-only):
  eval_results/issue_529/anchor_selection.json
  eval_results/issue_529/contrastive_negatives/analysis.json
  eval_results/issue_529/contrastive_negatives/cross_eval/per_cell/*.json

CLI:
    uv run python scripts/i529_figures.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no DISPLAY
import matplotlib.pyplot as plt

# Apply project paper rcParams at import time so any caller picks them up,
# including external test rigs that don't go through ``main()``. Top-level
# call keeps the import live against ruff's unused-import strip (see
# experiment-implementer memory ``feedback_ruff_strips_unused_imports``).
from explore_persona_space.analysis.paper_plots import set_paper_style

set_paper_style()

logger = logging.getLogger("i529.figures")

ANCHOR_PATH = Path("eval_results/issue_529/anchor_selection.json")
ANALYSIS_PATH = Path("eval_results/issue_529/contrastive_negatives/analysis.json")
PER_CELL_DIR = Path("eval_results/issue_529/contrastive_negatives/cross_eval/per_cell")
FIG_DIR = Path("figures/issue_529")
EXPLORE_DIR = FIG_DIR / "exploratory_dump"

EPOCHS = (1, 2, 3, 5)
SEEDS = (42, 137, 1337, 7, 21)
ARMS = ("system_plain", "system_padded", "role")
PERSONAS = ("pirate", "villain")


def _git_commit_hash() -> str:
    """Return the current HEAD sha or 'unknown' if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, env={**os.environ}
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _cell_label(arm: str, seed: int, persona: str, epoch: int) -> str:
    """Mirror i464_po_eval's cn_i529 cell label."""
    return f"{arm}_seed{seed}_cn_{persona}_e{epoch}"


def _load_cell(arm: str, seed: int, persona: str, epoch: int, e_eval: str) -> dict | None:
    """Read one per-cell JSON or return None if missing."""
    p = PER_CELL_DIR / f"{_cell_label(arm, seed, persona, epoch)}__{e_eval}.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    return json.loads(p.read_text())


def _write_meta(path: Path, payload: dict) -> None:
    """Write the figure's .meta.json companion."""
    payload = {
        **payload,
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2))


def _save_fig(fig, base: Path, meta: dict) -> None:
    """Save fig as PNG + PDF + meta.json next to each other."""
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    _write_meta(base.with_suffix(".meta.json"), meta)
    logger.info("wrote %s.{png,pdf,meta.json}", base)


def hero_role_vs_system_at_anchor(anchor: dict, analysis: dict) -> None:
    """Forest plot: d_plain & d_padded per seed at the selected anchor."""
    head = analysis.get("headline", {})
    sel = anchor.get("selected_anchor", {}) or {}
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    title_parts = []
    for persona in PERSONAS:
        e_star = sel.get(persona)
        title_parts.append(f"{persona}: E*={e_star}")
    ax.set_title("d = L_system - L_role at selected anchor (" + " · ".join(title_parts) + ")")
    # When the analyze is degenerate, the headline has no d_seed_*; show
    # the placeholder.
    d_plain = head.get("d_seed_plain", {}).get("d_per_seed", [])
    d_padded = head.get("d_seed_padded", {}).get("d_per_seed", [])
    seeds_used = analysis.get("complete_seeds", list(SEEDS))
    if d_plain:
        ax.scatter(d_plain, [1.1] * len(d_plain), label="d_plain (system_plain - role)", marker="o")
    if d_padded:
        ax.scatter(
            d_padded,
            [0.9] * len(d_padded),
            label="d_padded (system_padded - role)",
            marker="^",
        )
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    if head.get("d_seed_plain", {}).get("mean") is not None:
        m = head["d_seed_plain"]["mean"]
        lo = head["d_seed_plain"]["ci_lo_95"]
        hi = head["d_seed_plain"]["ci_hi_95"]
        ax.errorbar([m], [1.1], xerr=[[m - lo], [hi - m]], fmt="o", capsize=4, color="C0")
    if head.get("d_seed_padded", {}).get("mean") is not None:
        m = head["d_seed_padded"]["mean"]
        lo = head["d_seed_padded"]["ci_lo_95"]
        hi = head["d_seed_padded"]["ci_hi_95"]
        ax.errorbar([m], [0.9], xerr=[[m - lo], [hi - m]], fmt="^", capsize=4, color="C1")
    ax.set_xlabel("d (nats) — positive ⇒ role tightens persona→marker binding")
    ax.set_yticks([])
    ax.legend(loc="best", frameon=False)
    _save_fig(
        fig,
        FIG_DIR / "hero_role_vs_system_at_anchor",
        meta={
            "selected_anchor": sel,
            "seeds_used": seeds_used,
            "headline_status": analysis.get("headline_status"),
            "d_plain_per_seed": d_plain,
            "d_padded_per_seed": d_padded,
        },
    )


def hero_trajectory_per_arm(anchor: dict) -> None:
    """3-row x 2-col grid: own / wrong / default log P vs E, per persona."""
    sel = anchor.get("selected_anchor", {}) or {}
    diag = anchor.get("per_persona_per_E_diagnostics", {})
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 7.5), sharex=True)
    for col_idx, persona in enumerate(PERSONAS):
        # Own (diagonal)
        ax = axes[0][col_idx]
        own_y = [diag.get(persona, {}).get(e, {}).get("own_logp", float("nan")) for e in EPOCHS]
        ax.plot(EPOCHS, own_y, marker="o")
        ax.set_title(f"{persona} own-slot log P( ※)")
        ax.axvline(sel.get(persona, -1), color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("nats")
        # Wrong (off-diagonal, per arm)
        ax = axes[1][col_idx]
        for arm in ARMS:
            wrong_y = [
                diag.get(persona, {})
                .get(e, {})
                .get("per_arm", {})
                .get(arm, {})
                .get("wrong_logp_mean", float("nan"))
                for e in EPOCHS
            ]
            ax.plot(EPOCHS, wrong_y, marker="s", label=arm)
        ax.set_title(f"{persona} wrong-slot log P( ※) per arm")
        ax.axhspan(-10.0, -5.0, color="lightgreen", alpha=0.2, label="resolution band [-10, -5]")
        ax.axvline(sel.get(persona, -1), color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("nats")
        ax.legend(loc="best", frameon=False, fontsize=7)
        # Own argmax-emit (the source-installation gate)
        ax = axes[2][col_idx]
        emit_y = [
            diag.get(persona, {}).get(e, {}).get("own_argmax_emit", float("nan")) for e in EPOCHS
        ]
        ax.plot(EPOCHS, emit_y, marker="^", color="C3")
        ax.axhline(0.50, color="gray", linestyle="--", linewidth=0.8, label="0.50 gate")
        ax.set_title(f"{persona} own argmax-emit rate")
        ax.axvline(sel.get(persona, -1), color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("epochs E")
        ax.set_ylabel("rate")
        ax.legend(loc="best", frameon=False, fontsize=7)
    fig.suptitle("i529 trajectory per epoch — anchor E* marked with dashed line")
    fig.tight_layout()
    _save_fig(
        fig,
        FIG_DIR / "hero_trajectory_per_arm",
        meta={
            "selected_anchor": sel,
            "candidate_grid": list(EPOCHS),
            "thresholds": anchor.get("thresholds", {}),
        },
    )


def exploratory_dynamic_range_per_arm(anchor: dict) -> None:
    """sd vs E per arm x persona (the dynamic-range trajectory)."""
    diag = anchor.get("per_persona_per_E_diagnostics", {})
    EXPLORE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=True)
    for col_idx, persona in enumerate(PERSONAS):
        ax = axes[col_idx]
        for arm in ARMS:
            ys = [
                diag.get(persona, {})
                .get(e, {})
                .get("per_arm", {})
                .get(arm, {})
                .get("wrong_sd", float("nan"))
                for e in EPOCHS
            ]
            ax.plot(EPOCHS, ys, marker="o", label=arm)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="dynamic-range thr=0.5")
        ax.set_title(f"{persona} wrong-slot sd per E")
        ax.set_xlabel("E")
        ax.set_ylabel("sd (nats)")
        ax.legend(loc="best", frameon=False, fontsize=7)
    fig.tight_layout()
    _save_fig(
        fig,
        EXPLORE_DIR / "dynamic_range_vs_E",
        meta={"thresholds": anchor.get("thresholds", {})},
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for #529 figures."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Proceed even if anchor or analysis JSON is missing. Useful for "
            "smoke runs that haven't completed analyze yet."
        ),
    )
    args = ap.parse_args(argv)

    # set_paper_style() applied at module-import time; no-op here.

    if not ANCHOR_PATH.exists():
        if args.allow_partial:
            logger.warning("anchor file %s missing; degraded figures only", ANCHOR_PATH)
            anchor: dict = {"selected_anchor": {}, "per_persona_per_E_diagnostics": {}}
        else:
            raise FileNotFoundError(
                f"{ANCHOR_PATH} missing — run scripts/i529_select_anchor.py first."
            )
    else:
        anchor = json.loads(ANCHOR_PATH.read_text())

    if not ANALYSIS_PATH.exists():
        if args.allow_partial:
            logger.warning("analysis file %s missing; hero #1 degraded", ANALYSIS_PATH)
            analysis: dict = {"headline": {}, "complete_seeds": []}
        else:
            raise FileNotFoundError(
                f"{ANALYSIS_PATH} missing — run scripts/i464_po_analyze.py "
                "--variant cn_i529 --anchor-file ... first."
            )
    else:
        analysis = json.loads(ANALYSIS_PATH.read_text())

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hero_role_vs_system_at_anchor(anchor, analysis)
    hero_trajectory_per_arm(anchor)
    exploratory_dynamic_range_per_arm(anchor)


if __name__ == "__main__":
    main()

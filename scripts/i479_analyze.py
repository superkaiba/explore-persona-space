# ruff: noqa: RUF001, RUF002, RUF003, C901  # em-dash + × + −/ΔG glyphs intentional; _detect_window is a single state-machine sweep
#!/usr/bin/env python3
"""Task #479 Phase 5 — selectivity-window detection + hero figure.

Reads each ``trajectory.json`` (one per cell × seed) produced by
``scripts/i472_eval_trajectory.py`` and emits:

1. ``figures/issue_479/selectivity_trajectory.png`` — emission rate vs step
   per cell, source (bold) vs bystander-panel mean (with ±1SE shaded). One
   panel per cell. Annotations: dashed lines at 0.8 (source threshold) and
   0.1 (bystander threshold); highlighted band where BOTH thresholds hold AND
   ΔG-bystander is NOT within 1 nat of ceiling on > 50% of personas, on BOTH
   seeds — the H2 binary success criterion (plan §3 + §6.2).
2. Exploratory dump figures (per-cell × per-persona heat-map, ΔG-vs-KL,
   per-seed scatter, source-self ΔG trajectory, bystander emission vs
   cos_to_villain at window midpoint, raw vs residualized scatter — plan
   §6.2).
3. Per-cell ``window_detected`` JSON manifest the dispatcher's auto-gate
   reads to decide Stage-1 → Stage-2 launch.

The orchestrator runs this script either inline (as Phase 5 of the
dispatcher) or stand-alone (re-analysis pass over an existing slab).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i479.analyze")

DEFAULT_SOURCE_FLOOR = 0.8
DEFAULT_BYSTANDER_CEILING = 0.1
DEFAULT_WINDOW_MIN_STEPS = 25
DEFAULT_CEILING_HEADROOM_NATS = 1.0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _aggregate_emission(checkpoint: dict, source_persona: str) -> dict[str, Any]:
    """Compute per-persona emission rate + source emission rate at one ckpt.

    Reads from the v2 trajectory schema emitted by ``eval_trajectory.py``:
      - ``checkpoints[].source_self.emission_rate`` — the source's own
        on-policy emission rate (PRIMARY read; the source persona is NOT
        in ``held_out`` for the c472/c479 rig — ``held_out`` only carries
        bystander personas, because the rig's inner loop is
        ``for persona in eval_personas`` where ``eval_personas`` is the
        held-out panel WITHOUT the source).
      - ``checkpoints[].bystander_emission.{mean, se, per_persona_rate}``
        — pre-aggregated bystander panel statistics (v2 adds these in
        the same forward pass).
      - ``checkpoints[].held_out[persona][q].argmax_marker`` — raw
        per-question records, used as a fallback for older trajectories
        that pre-date the v2 schema (and for the per-persona heatmap).

    Legacy fallback: if ``source_self.emission_rate`` is missing, fall
    back to ``held_out[source]`` (covers legacy v1 trajectories where
    the rig sometimes mistakenly included the source in ``eval_personas``
    — never happens on the c472/c479 rig but kept for safety).

    Returns dict with: per_persona_rate (bystander only), bystander_mean,
    bystander_se, source_emission, n_bystander_personas, n_questions.
    """
    source_self = checkpoint.get("source_self") or {}
    held_out = checkpoint.get("held_out", {}) or {}

    # ── Source emission rate (v2 primary; v1 legacy fallback). ───────────────
    source_emission = source_self.get("emission_rate")
    if source_emission is None and source_persona in held_out:
        per_q = held_out.get(source_persona) or {}
        flags = [bool(v.get("argmax_marker", False)) for v in per_q.values()]
        source_emission = (sum(flags) / len(flags)) if flags else None

    # ── Per-bystander-persona rates (always recompute from raw held_out
    #    so the per-persona heatmap reads the same numbers; v2's
    #    bystander_emission.per_persona_rate matches this by construction).
    per_persona_rate: dict[str, float] = {}
    for persona, per_q in held_out.items():
        if persona == source_persona or not per_q:
            continue
        flags = [bool(v.get("argmax_marker", False)) for v in per_q.values()]
        per_persona_rate[persona] = sum(flags) / len(flags) if flags else 0.0

    # ── Bystander panel mean ± SE: prefer v2 pre-aggregated, fall back
    #    to recompute from raw held_out for v1 trajectories. ──────────────────
    byst_v2 = checkpoint.get("bystander_emission") or {}
    if byst_v2.get("mean") is not None:
        mean = float(byst_v2["mean"])
        se = float(byst_v2.get("se", 0.0))
        n_bystander = int(byst_v2.get("n_personas", len(per_persona_rate)))
    else:
        rates = list(per_persona_rate.values())
        if rates:
            mean = sum(rates) / len(rates)
            var = sum((r - mean) ** 2 for r in rates) / max(len(rates) - 1, 1)
            se = (var / len(rates)) ** 0.5
        else:
            mean = 0.0
            se = 0.0
        n_bystander = len(rates)

    if held_out:
        n_questions = len(next(iter(held_out.values())))
    else:
        n_questions = int(source_self.get("n_questions", 0))

    return {
        "per_persona_rate": per_persona_rate,
        "bystander_mean": mean,
        "bystander_se": se,
        "source_emission": source_emission,
        "n_bystander_personas": n_bystander,
        "n_questions": n_questions,
    }


def _bystander_ceiling_fraction(
    checkpoint: dict, source_persona: str, headroom_nats: float
) -> float:
    """Fraction of bystander personas whose mean ΔG is WITHIN ``headroom_nats`` of 0.0.

    "Saturated" = bystander ΔG within 1 nat of ceiling on > 50% of personas
    (plan §3). The base ΔG ceiling is 0.0 (trained log-prob == base log-prob
    means no shift); a bystander mean ΔG > −headroom_nats is "saturated".
    """
    held_out = checkpoint.get("held_out", {})
    pers_means: list[float] = []
    for persona, per_q in held_out.items():
        if persona == source_persona:
            continue
        dgs = [float(v.get("delta_g", 0.0)) for v in per_q.values()]
        if dgs:
            pers_means.append(sum(dgs) / len(dgs))
    if not pers_means:
        return 0.0
    n_sat = sum(1 for m in pers_means if m > -headroom_nats)
    return n_sat / len(pers_means)


def _per_checkpoint_summary(
    trajectory: dict,
    source_persona: str,
    headroom_nats: float,
) -> list[dict]:
    """Per-checkpoint summary: step + source emission + bystander mean + ceiling frac."""
    out: list[dict] = []
    for ck in trajectory.get("checkpoints", []):
        agg = _aggregate_emission(ck, source_persona)
        ceil_frac = _bystander_ceiling_fraction(ck, source_persona, headroom_nats)
        out.append(
            {
                "step": ck.get("step"),
                "frac": ck.get("frac"),
                # #479 concern 6: prefer the original step-key string for
                # labels ("step0005") over the legacy `frac` field (which is
                # 5.0 for absolute-step indices; misleading).
                "step_key": ck.get("step_key"),
                "source_emission": agg["source_emission"],
                "bystander_mean": agg["bystander_mean"],
                "bystander_se": agg["bystander_se"],
                "n_bystander_personas": agg["n_bystander_personas"],
                "n_questions": agg["n_questions"],
                "bystander_ceiling_fraction": ceil_frac,
                "source_self_delta_g_mean": (ck.get("source_self", {}).get("delta_g_mean")),
                "per_persona_rate": agg["per_persona_rate"],
            }
        )
    out.sort(key=lambda r: r.get("step") if r.get("step") is not None else -1)
    return out


def _detect_window(
    per_seed_summary: dict[int, list[dict]],
    *,
    source_floor: float,
    bystander_ceiling: float,
    min_steps: int,
    ceiling_headroom_nats: float,
) -> dict:
    """Detect a ≥``min_steps`` contiguous-step band where on BOTH seeds:
        - source emission ≥ source_floor
        - bystander mean emission < bystander_ceiling
        - bystander ceiling fraction (∆G within headroom of 0) ≤ 0.5

    Steps are the per-checkpoint step values. We treat each (step) as a
    discrete sample; the "≥min_steps" criterion is "contiguous run of
    checkpoints whose [first_step, last_step] span is ≥ min_steps".

    Returns: {detected, window_start_step, window_end_step, width_steps,
              per_seed_window_widths, qualifying_checkpoints_by_seed}.
    """
    if not per_seed_summary or any(not v for v in per_seed_summary.values()):
        return {
            "detected": False,
            "reason": "no_per_seed_summary",
            "window_start_step": None,
            "window_end_step": None,
            "width_steps": 0,
            "per_seed_window_widths": {},
        }

    # Map seed → list of (step, qualifies_bool).
    per_seed_qual: dict[int, list[tuple[int, bool]]] = {}
    for seed, summ in per_seed_summary.items():
        qual: list[tuple[int, bool]] = []
        for row in summ:
            step = row.get("step")
            if step is None:
                continue
            src = row.get("source_emission")
            byst = row.get("bystander_mean")
            ceil_frac = row.get("bystander_ceiling_fraction", 1.0)
            ok = (
                src is not None
                and src >= source_floor
                and byst is not None
                and byst < bystander_ceiling
                and ceil_frac <= 0.5
            )
            qual.append((int(step), ok))
        qual.sort()
        per_seed_qual[seed] = qual

    # Build the steps grid intersection — only steps present in BOTH seeds.
    common_steps = set.intersection(
        *(set(step for step, _ in qual) for qual in per_seed_qual.values())
    )
    common_steps = sorted(common_steps)
    if not common_steps:
        return {
            "detected": False,
            "reason": "no_common_steps",
            "window_start_step": None,
            "window_end_step": None,
            "width_steps": 0,
            "per_seed_window_widths": {},
        }

    qual_by_seed_step: dict[int, dict[int, bool]] = {
        seed: dict(qual) for seed, qual in per_seed_qual.items()
    }
    joint_ok: list[tuple[int, bool]] = [
        (step, all(qual_by_seed_step[seed].get(step, False) for seed in per_seed_qual))
        for step in common_steps
    ]

    # Find the longest contiguous run (by step-span) of joint_ok == True.
    best_start: int | None = None
    best_end: int | None = None
    best_width = 0
    cur_start: int | None = None
    cur_end: int | None = None
    for step, ok in joint_ok:
        if ok:
            if cur_start is None:
                cur_start = step
            cur_end = step
        else:
            if cur_start is not None and cur_end is not None:
                width = cur_end - cur_start
                if width > best_width:
                    best_width = width
                    best_start = cur_start
                    best_end = cur_end
            cur_start = None
            cur_end = None
    if cur_start is not None and cur_end is not None:
        width = cur_end - cur_start
        if width > best_width:
            best_width = width
            best_start = cur_start
            best_end = cur_end

    # Per-seed band widths (independent, not requiring joint).
    per_seed_widths: dict[int, int] = {}
    for seed, qual in per_seed_qual.items():
        cur_s = cur_e = None
        seed_best = 0
        for step, ok in qual:
            if ok:
                if cur_s is None:
                    cur_s = step
                cur_e = step
            else:
                if cur_s is not None and cur_e is not None:
                    seed_best = max(seed_best, cur_e - cur_s)
                cur_s = cur_e = None
        if cur_s is not None and cur_e is not None:
            seed_best = max(seed_best, cur_e - cur_s)
        per_seed_widths[seed] = seed_best

    detected = best_width >= min_steps
    return {
        "detected": detected,
        "window_start_step": best_start,
        "window_end_step": best_end,
        "width_steps": best_width,
        "min_steps_required": min_steps,
        "per_seed_window_widths": per_seed_widths,
        "common_steps": common_steps,
        "joint_ok_by_step": dict(joint_ok),
    }


def _plot_hero(
    per_cell_per_seed_summary: dict[str, dict[int, list[dict]]],
    per_cell_window: dict[str, dict],
    *,
    figures_dir: Path,
    source_floor: float,
    bystander_ceiling: float,
) -> Path:
    """Hero figure: emission vs step per cell, source vs bystander-mean.

    Imports matplotlib lazily (the analyzer also runs in CPU-only / headless
    smoke contexts where matplotlib may not be wanted on import).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted(per_cell_per_seed_summary)
    n = len(cells)
    if n == 0:
        log.warning("[hero] no cells to plot")
        return figures_dir / "selectivity_trajectory.png"
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows), squeeze=False)

    for idx, cell in enumerate(cells):
        ax = axes[idx // cols][idx % cols]
        per_seed = per_cell_per_seed_summary[cell]
        for seed, summ in sorted(per_seed.items()):
            steps = [r["step"] for r in summ if r.get("step") is not None]
            src = [r["source_emission"] for r in summ if r.get("step") is not None]
            byst = [r["bystander_mean"] for r in summ if r.get("step") is not None]
            byst_se = [r["bystander_se"] for r in summ if r.get("step") is not None]
            # Coerce None → np.nan for matplotlib.
            src_plot = [float("nan") if v is None else v for v in src]
            ax.plot(
                steps,
                src_plot,
                marker="o",
                linewidth=2.2,
                label=f"seed {seed} source",
            )
            byst_plot = [float("nan") if v is None else v for v in byst]
            ax.plot(
                steps,
                byst_plot,
                marker="s",
                linewidth=1.0,
                alpha=0.85,
                label=f"seed {seed} bystander mean",
            )
            # Shaded ±1SE.
            lower = [
                (float("nan") if (m is None or s is None) else m - s)
                for m, s in zip(byst, byst_se, strict=True)
            ]
            upper = [
                (float("nan") if (m is None or s is None) else m + s)
                for m, s in zip(byst, byst_se, strict=True)
            ]
            ax.fill_between(steps, lower, upper, alpha=0.18)
        ax.axhline(source_floor, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(bystander_ceiling, color="gray", linestyle=":", alpha=0.5)
        window = per_cell_window.get(cell, {})
        if window.get("detected"):
            ax.axvspan(
                window["window_start_step"],
                window["window_end_step"],
                color="green",
                alpha=0.10,
            )
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("emission rate (argmax == ※)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(cell)
        ax.legend(fontsize=7, loc="best")
    # Hide unused axes.
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(
        "selectivity trajectory (source vs held-out bystander mean)",
        fontsize=12,
    )
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "selectivity_trajectory.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("[hero] wrote %s", out)
    return out


def _plot_per_persona_heatmap(
    cell: str,
    per_seed_summary: dict[int, list[dict]],
    figures_dir: Path,
) -> Path | None:
    """Per-cell per-persona × per-step emission-rate heatmap (averaged over seeds)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not per_seed_summary:
        return None
    # Collect the step grid (union across seeds) and persona universe.
    all_steps: list[int] = sorted(
        {r["step"] for summ in per_seed_summary.values() for r in summ if r.get("step") is not None}
    )
    personas: set[str] = set()
    for summ in per_seed_summary.values():
        for r in summ:
            personas.update(r.get("per_persona_rate", {}).keys())
    personas_sorted = sorted(personas)
    if not all_steps or not personas_sorted:
        return None
    mat = np.full((len(personas_sorted), len(all_steps)), np.nan)
    counts = np.zeros_like(mat)
    for summ in per_seed_summary.values():
        for r in summ:
            step = r.get("step")
            if step is None:
                continue
            col = all_steps.index(step)
            per_p = r.get("per_persona_rate", {})
            for row_i, persona in enumerate(personas_sorted):
                if persona in per_p:
                    cur = mat[row_i, col]
                    new = per_p[persona]
                    if np.isnan(cur):
                        mat[row_i, col] = new
                    else:
                        mat[row_i, col] = cur + new
                    counts[row_i, col] += 1
    with np.errstate(invalid="ignore"):
        mat = np.where(counts > 0, mat / np.where(counts > 0, counts, 1.0), np.nan)
    fig, ax = plt.subplots(
        figsize=(max(6, len(all_steps) * 0.55), max(4, len(personas_sorted) * 0.18))
    )
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(all_steps)))
    ax.set_xticklabels([str(s) for s in all_steps], fontsize=7)
    ax.set_yticks(range(len(personas_sorted)))
    ax.set_yticklabels(personas_sorted, fontsize=6)
    ax.set_xlabel("optimizer step")
    ax.set_title(f"{cell}: emission rate per persona × step (seed-mean)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("emission rate (argmax == ※)")
    out = figures_dir / f"heatmap_{cell}.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("[heatmap] wrote %s", out)
    return out


def _load_base_emission_panel(path: Path | None) -> dict | None:
    """Load the base-model emission-rate baseline written by i479_phase_base_emission.

    Schema: ``i479_base_emission_v1`` → returns the parsed dict. Returns
    ``None`` if path is None / missing / wrong schema (with a loud warning
    so the operator sees the bystander<0.1 threshold is unanchored).
    """
    if path is None:
        log.warning(
            "[base-panel] no --base-panel-path; bystander<%.2f threshold is "
            "UNANCHORED to the base-model emission floor (plan §13.1).",
            DEFAULT_BYSTANDER_CEILING,
        )
        return None
    if not path.exists():
        log.warning(
            "[base-panel] --base-panel-path=%s does not exist; bystander<%.2f "
            "threshold is UNANCHORED to the base-model emission floor "
            "(run scripts/i479_phase_base_emission.py to produce it).",
            path,
            DEFAULT_BYSTANDER_CEILING,
        )
        return None
    payload = json.loads(path.read_text())
    schema = payload.get("schema_version", "")
    if schema != "i479_base_emission_v1":
        log.warning(
            "[base-panel] %s has schema_version=%r, expected "
            "'i479_base_emission_v1'. This may be the older #472 LOG-PROB "
            "baseline (base_panel.json) — NOT the emission-rate baseline. "
            "Re-run scripts/i479_phase_base_emission.py.",
            path,
            schema,
        )
        return None
    panel_mean = payload.get("panel_mean_emission_rate")
    n_personas = payload.get("n_held_out_personas")
    log.info(
        "[base-panel] loaded base-model emission baseline: panel_mean=%.4f over "
        "%s held-out personas (from %s).",
        panel_mean if panel_mean is not None else float("nan"),
        n_personas,
        path,
    )
    return payload


def _base_adjusted_bystander_rate(
    raw_rate: float | None,
    base_panel: dict | None,
) -> float | None:
    """Return ``raw - base_panel_mean`` (the bystander shift above base floor).

    A bystander mean of 0.05 against a base floor of 0.02 is a 0.03 shift —
    that's the load-bearing comparand for the "implant leaked" claim, NOT
    the raw 0.05 (which could be entirely accounted for by the base model's
    own marker prior on that question set).
    """
    if raw_rate is None or base_panel is None:
        return raw_rate
    base_mean = base_panel.get("panel_mean_emission_rate")
    if base_mean is None:
        return raw_rate
    return raw_rate - float(base_mean)


def _write_sentinel(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:progress",
                "version": 1,
                "task_id": 479,
                "by": "i479_analyze",
                "ts": datetime.now(UTC).isoformat(),
                "phase": "analyze",
                "note": json.dumps(payload),
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_479"))
    ap.add_argument("--base-panel-path", type=Path, default=None)
    ap.add_argument("--figures-dir", type=Path, default=Path("figures/issue_479"))
    ap.add_argument(
        "--cells",
        default=None,
        help="CSV of c479 slugs (default: all cells with a trajectory.json in --slab-root).",
    )
    ap.add_argument("--seeds", default="42,137")
    ap.add_argument("--source-floor", type=float, default=DEFAULT_SOURCE_FLOOR)
    ap.add_argument("--bystander-ceiling", type=float, default=DEFAULT_BYSTANDER_CEILING)
    ap.add_argument("--window-min-steps", type=int, default=DEFAULT_WINDOW_MIN_STEPS)
    ap.add_argument("--ceiling-headroom-nats", type=float, default=DEFAULT_CEILING_HEADROOM_NATS)
    ap.add_argument("--manifest-path", type=Path, default=None)
    ap.add_argument("--sentinel-path", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=analyze_479] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        SOURCE_PERSONA,
    )

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.cells:
        cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    else:
        # Auto-discover cells from the slab.
        cells = sorted({p.name.split("_seed")[0] for p in args.slab_root.glob("c479_*_seed*")})
    if not cells:
        raise SystemExit(f"No #479 cells found under {args.slab_root}. Pass --cells explicitly.")
    log.info("Analyzing cells=%s seeds=%s", cells, seeds)

    # Base-model emission-rate baseline (plan §13.1). Bystander < 0.1 success
    # criterion is only meaningful as a SHIFT above this floor.
    base_panel = _load_base_emission_panel(args.base_panel_path)
    base_panel_mean = base_panel.get("panel_mean_emission_rate") if base_panel is not None else None

    per_cell_per_seed: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for cell in cells:
        for seed in seeds:
            traj_path = args.slab_root / f"{cell}_seed{seed}" / "trajectory.json"
            if not traj_path.exists():
                log.warning("[%s seed%d] missing %s; skipping", cell, seed, traj_path)
                continue
            traj = json.loads(traj_path.read_text())
            summary = _per_checkpoint_summary(traj, SOURCE_PERSONA, args.ceiling_headroom_nats)
            # Attach base-adjusted columns to each per-ckpt row (raw rates
            # are preserved; the adjusted columns are the load-bearing
            # comparand for the "did the implant leak above floor" claim).
            for row in summary:
                row["bystander_mean_minus_base"] = _base_adjusted_bystander_rate(
                    row.get("bystander_mean"), base_panel
                )
                row["source_emission_minus_base"] = _base_adjusted_bystander_rate(
                    row.get("source_emission"), base_panel
                )
                row["base_panel_mean_emission_rate"] = base_panel_mean
            per_cell_per_seed[cell][seed] = summary

    per_cell_window: dict[str, dict] = {}
    for cell, per_seed in per_cell_per_seed.items():
        window = _detect_window(
            per_seed,
            source_floor=args.source_floor,
            bystander_ceiling=args.bystander_ceiling,
            min_steps=args.window_min_steps,
            ceiling_headroom_nats=args.ceiling_headroom_nats,
        )
        per_cell_window[cell] = window
        log.info("[%s] window: %s", cell, window)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    if per_cell_per_seed:
        _plot_hero(
            per_cell_per_seed,
            per_cell_window,
            figures_dir=args.figures_dir,
            source_floor=args.source_floor,
            bystander_ceiling=args.bystander_ceiling,
        )
        for cell, per_seed in per_cell_per_seed.items():
            _plot_per_persona_heatmap(cell, per_seed, args.figures_dir)

    # Inter-stage gate: the dispatcher reads `window_detected` from this
    # manifest to decide Stage-1 → Stage-2.
    any_window = any(w.get("detected") for w in per_cell_window.values())
    manifest = {
        # v2: adds base-emission baseline anchor + per-row base-adjusted
        # bystander/source emission columns.
        "schema_version": "i479_v2",
        "cells": cells,
        "seeds": seeds,
        "source_persona": SOURCE_PERSONA,
        "source_floor": args.source_floor,
        "bystander_ceiling": args.bystander_ceiling,
        "window_min_steps": args.window_min_steps,
        "ceiling_headroom_nats": args.ceiling_headroom_nats,
        "window_detected": any_window,
        "per_cell_window": per_cell_window,
        "per_cell_per_seed_summary": {
            cell: {str(seed): summ for seed, summ in per_seed.items()}
            for cell, per_seed in per_cell_per_seed.items()
        },
        "base_panel": {
            "path": str(args.base_panel_path) if args.base_panel_path else None,
            "loaded": base_panel is not None,
            "panel_mean_emission_rate": base_panel_mean,
            "n_held_out_personas": (
                base_panel.get("n_held_out_personas") if base_panel is not None else None
            ),
            "schema_version": (
                base_panel.get("schema_version") if base_panel is not None else None
            ),
        },
        "figures_dir": str(args.figures_dir),
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = args.manifest_path or (args.slab_root / "i479_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("[analyze] manifest → %s (window_detected=%s)", manifest_path, any_window)

    if args.sentinel_path is not None:
        _write_sentinel(args.sentinel_path, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

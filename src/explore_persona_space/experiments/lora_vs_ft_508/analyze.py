# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #508 — post-train analysis (plan §6).

Phases:
    1. Load all 6 cells' eval JSONs (3 LoRA + 3 full-FT).
    2. Per-cell aggregates: source-self ΔG mean, held-out mean ΔG, n_collapsed.
    3. Per-arm bracketing check (plan §6 MF4): does each arm have ≥1 cell
       with source ΔG < 7 AND ≥1 with source ΔG > 9? If not, H1 INDETERMINATE
       for that arm.
    4. Persona-cluster crossed bootstrap (plan §6 MF3): 1000 replicates,
       resample personas with replacement, then questions with replacement
       within each resampled persona, re-fit linear interpolation per arm
       across the 3 cells' means, read each arm's held-out mean at source
       ΔG = 8 nat, take FT − LoRA gap. Empirical 2.5/97.5 percentile.
    5. Hero figure: LoRA vs FT curve on source-rate axis (matched-rate read
       at source ΔG = 8 ± 1 nat).
    6. Trajectory figures (read MarkerDynamicsCallback snapshots from WandB
       run files; figure-only — the snapshots already log live to WandB).
    7. H2 secondary: Spearman ρ per-bystander ΔG vs cosine distance.
    8. H3 tertiary: qwen_default + assistant + ai_assistant slice.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from pathlib import Path
from typing import Any

from explore_persona_space.experiments.lora_vs_ft_508 import (
    ARM_FULLFT,
    ARM_LORA,
    HELD_OUT_PERSONAS_15,
    MATCHED_SLICE_BAND_NATS,
    MATCHED_SLICE_TARGET_NATS,
    SOURCE_SELF_FLOOR_NATS,
    is_lora_arm,
)

log = logging.getLogger("issue_508.analyze")


def _load_eval(path: Path) -> dict:
    return json.loads(path.read_text())


def _cell_aggregates(eval_json: dict) -> dict:
    """Per-cell aggregates needed for the analysis.

    Returns dict with keys: source_mean, source_n, held_out_mean, held_out_n,
    n_collapsed, per_persona_mean (dict).
    """
    held_out = eval_json["delta_g_held_out"]
    source_dict = eval_json.get("delta_g_source", {})

    # Per-persona x per-question ΔG (drop collapsed probes).
    per_persona: dict[str, list[float]] = {}
    for persona, q_map in held_out.items():
        vals: list[float] = []
        for _q, info in q_map.items():
            if not info["r_collapsed"]:
                vals.append(float(info["delta_g"]))
        per_persona[persona] = vals
    held_out_dg_all = [v for vs in per_persona.values() for v in vs]
    held_out_mean = sum(held_out_dg_all) / len(held_out_dg_all) if held_out_dg_all else float("nan")
    n_collapsed = sum(
        1 for q_map in held_out.values() for info in q_map.values() if info["r_collapsed"]
    )

    source_vals: list[float] = []
    for _persona, q_map in source_dict.items():
        for _q, info in q_map.items():
            if not info.get("r_collapsed"):
                source_vals.append(float(info["delta_g"]))
    source_mean = sum(source_vals) / len(source_vals) if source_vals else float("nan")

    per_persona_mean = {
        p: (sum(vs) / len(vs)) if vs else float("nan") for p, vs in per_persona.items()
    }

    return {
        "source_mean": source_mean,
        "source_n": len(source_vals),
        "held_out_mean": held_out_mean,
        "held_out_n": len(held_out_dg_all),
        "n_collapsed": n_collapsed,
        "per_persona_mean": per_persona_mean,
        "per_persona_values": per_persona,
    }


def _check_bracketing(per_arm_source_mean: list[float]) -> dict:
    """Plan §6 MF4 — does this arm's source ΔG bracket the 8-nat target band?

    Requires ≥1 cell < 7 nat AND ≥1 cell > 9 nat. Failure → H1 INDETERMINATE.
    """
    below_7 = sum(1 for x in per_arm_source_mean if x < 7.0)
    above_9 = sum(1 for x in per_arm_source_mean if x > 9.0)
    return {
        "below_7_nat": below_7,
        "above_9_nat": above_9,
        "brackets_target": below_7 >= 1 and above_9 >= 1,
        "source_means": list(per_arm_source_mean),
    }


def _linear_interp(xs: list[float], ys: list[float], target_x: float) -> float:
    """Linear interpolation across (xs, ys); returns y at target_x.

    For target_x outside [min(xs), max(xs)] extrapolates from the nearest two
    points (used downstream only when bracketing PASSes; outside that the
    caller marks INDETERMINATE).
    """
    pairs = sorted(zip(xs, ys, strict=True))
    if len(pairs) < 2:
        return float("nan")
    # Find the two bracket points.
    for i in range(len(pairs) - 1):
        x1, y1 = pairs[i]
        x2, y2 = pairs[i + 1]
        if x1 <= target_x <= x2:
            if x2 == x1:
                return y1
            t = (target_x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
    # Outside [min, max]; extrapolate from the two extremes.
    if target_x < pairs[0][0]:
        x1, y1 = pairs[0]
        x2, y2 = pairs[1]
    else:
        x1, y1 = pairs[-2]
        x2, y2 = pairs[-1]
    if x2 == x1:
        return y1
    t = (target_x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def _crossed_cluster_bootstrap_gap(
    cells_by_arm: dict[str, list[dict]],
    *,
    target_source_dg: float = MATCHED_SLICE_TARGET_NATS,
    n_replicates: int = 1000,
    seed: int = 42,
) -> dict:
    """Crossed cluster bootstrap on the matched-rate FT − LoRA gap (plan §6 MF3).

    Per replicate:
      (a) Draw 15 personas WITH REPLACEMENT from the held-out panel.
      (b) Draw 20 questions WITH REPLACEMENT WITHIN each resampled persona.
      (c) For each cell: per-persona mean ΔG on the resampled question set,
          then average across the resampled personas → one per-replicate
          cell-mean.
      (d) For each arm: re-fit the linear interpolation across the 3 cells'
          (source_mean, held_out_replicate_mean) → read held-out at
          target_source_dg.
      (e) gap = FT_held_out_at_target − LoRA_held_out_at_target.

    Returns:
        {
          "n_replicates": int,
          "gap_mean": float,
          "gap_ci_lo": float,   # 2.5th percentile
          "gap_ci_hi": float,   # 97.5th percentile
          "gap_excludes_zero": bool,
          "lora_held_out_at_8_mean": float,
          "fullft_held_out_at_8_mean": float,
        }
    """
    import random

    rng = random.Random(seed)
    rep_gaps: list[float] = []
    lora_reps: list[float] = []
    ft_reps: list[float] = []

    for _r in range(n_replicates):
        # Resample personas (same set across cells per replicate — the load-
        # bearing "crossed" bit; question-within-persona resampling is also
        # locked across cells so the noise structure stays paired).
        resampled_personas = [
            rng.choice(HELD_OUT_PERSONAS_15) for _ in range(len(HELD_OUT_PERSONAS_15))
        ]
        # Cache a per-persona question resampling pattern.
        per_persona_q_picks: dict[str, list[int]] = {}

        arm_cell_means: dict[str, list[tuple[float, float]]] = {ARM_LORA: [], ARM_FULLFT: []}

        for arm, cells in cells_by_arm.items():
            for cell in cells:
                pp_values = cell["per_persona_values"]
                # Per-persona resampled mean (questions WITHIN persona).
                resampled_pp_means: list[float] = []
                for persona in resampled_personas:
                    vals = pp_values.get(persona, [])
                    if not vals:
                        continue
                    n_q = len(vals)
                    if persona not in per_persona_q_picks:
                        # Resample 20 questions WITH REPLACEMENT (n_q is the
                        # actual available count for this persona after
                        # dropping collapsed probes; resample to that count).
                        per_persona_q_picks[persona] = [rng.randrange(n_q) for _ in range(n_q)]
                    picks = per_persona_q_picks[persona]
                    sub = [vals[i] for i in picks]
                    if sub:
                        resampled_pp_means.append(sum(sub) / len(sub))
                if not resampled_pp_means:
                    continue
                replicate_held_out_mean = sum(resampled_pp_means) / len(resampled_pp_means)
                arm_cell_means[arm].append((cell["source_mean"], replicate_held_out_mean))

        # Linear interpolation per arm.
        def _read_at(target: float, points: list[tuple[float, float]]) -> float:
            if len(points) < 2:
                return float("nan")
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return _linear_interp(xs, ys, target)

        lora_at = _read_at(target_source_dg, arm_cell_means[ARM_LORA])
        ft_at = _read_at(target_source_dg, arm_cell_means[ARM_FULLFT])
        if math.isfinite(lora_at) and math.isfinite(ft_at):
            rep_gaps.append(ft_at - lora_at)
            lora_reps.append(lora_at)
            ft_reps.append(ft_at)

    if not rep_gaps:
        return {
            "n_replicates": 0,
            "gap_mean": float("nan"),
            "gap_ci_lo": float("nan"),
            "gap_ci_hi": float("nan"),
            "gap_excludes_zero": False,
            "lora_held_out_at_target_mean": float("nan"),
            "fullft_held_out_at_target_mean": float("nan"),
            "target_source_dg": target_source_dg,
        }

    rep_gaps_sorted = sorted(rep_gaps)
    n = len(rep_gaps_sorted)
    lo_idx = max(0, int(0.025 * n))
    hi_idx = min(n - 1, int(0.975 * n))
    ci_lo = rep_gaps_sorted[lo_idx]
    ci_hi = rep_gaps_sorted[hi_idx]
    gap_mean = sum(rep_gaps_sorted) / n
    excludes_zero = (ci_lo > 0) or (ci_hi < 0)
    return {
        "n_replicates": n,
        "gap_mean": gap_mean,
        "gap_ci_lo": ci_lo,
        "gap_ci_hi": ci_hi,
        "gap_excludes_zero": excludes_zero,
        "lora_held_out_at_target_mean": sum(lora_reps) / len(lora_reps),
        "fullft_held_out_at_target_mean": sum(ft_reps) / len(ft_reps),
        "target_source_dg": target_source_dg,
    }


def _hero_figure(cells_by_arm: dict[str, list[dict]], output_path: Path) -> None:
    """Hero figure: LoRA-vs-FT curve on source-rate axis. Plan §4.7."""
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import apply_paper_rcparams

        apply_paper_rcparams()
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    palette = {ARM_LORA: "#0173b2", ARM_FULLFT: "#de8f05"}
    labels = {ARM_LORA: "LoRA (r=16)", ARM_FULLFT: "Full FT"}

    for arm, cells in cells_by_arm.items():
        xs = [c["source_mean"] for c in cells]
        ys = [c["held_out_mean"] for c in cells]
        pairs = sorted(zip(xs, ys, strict=True))
        xs_s = [p[0] for p in pairs]
        ys_s = [p[1] for p in pairs]
        ax.plot(xs_s, ys_s, "o-", color=palette[arm], label=labels[arm], lw=2, markersize=8)

    # Matched-rate target band.
    ax.axvspan(
        MATCHED_SLICE_TARGET_NATS - MATCHED_SLICE_BAND_NATS,
        MATCHED_SLICE_TARGET_NATS + MATCHED_SLICE_BAND_NATS,
        alpha=0.1,
        color="gray",
        label=f"Matched-rate band ({MATCHED_SLICE_TARGET_NATS}±{MATCHED_SLICE_BAND_NATS} nat)",
    )

    ax.set_xlabel("Source-self ΔG (nats)")
    ax.set_ylabel("Held-out mean ΔG (nats)")
    ax.set_title("Marker leakage: LoRA vs full FT at matched source-implant rate")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("[fig] hero figure → %s", output_path)


def _h3_qwen_default_slice(eval_jsons_by_cell: dict[str, dict]) -> dict:
    """H3 tertiary — qwen_default + assistant + ai_assistant safety-relevant slice.

    The held-out panel HAS ``assistant``, ``ai``, ``ai_assistant`` (default-
    context proxies). ``qwen_default`` is in the contrastive negatives so it
    cannot appear in the held-out panel; the closest safety-relevant proxies
    are these 3 default-instruct personas.
    """
    safety_relevant = ("assistant", "ai", "ai_assistant")
    out: dict[str, dict] = {}
    for cell_id, ej in eval_jsons_by_cell.items():
        held_out = ej["delta_g_held_out"]
        per_persona_means: dict[str, float] = {}
        for p in safety_relevant:
            q_map = held_out.get(p, {})
            vals = [info["delta_g"] for info in q_map.values() if not info["r_collapsed"]]
            if vals:
                per_persona_means[p] = sum(vals) / len(vals)
        out[cell_id] = per_persona_means
    return out


def run_analysis(
    *,
    eval_jsons: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Plan §6 analysis pipeline. Returns the analysis dict + writes JSON + figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cells_data: list[dict] = []
    eval_jsons_by_cell: dict[str, dict] = {}
    for p in eval_jsons:
        ej = _load_eval(p)
        cell_slug = ej["cell_slug"]
        eval_jsons_by_cell[cell_slug] = ej
        agg = _cell_aggregates(ej)
        cells_data.append(
            {
                "cell": cell_slug,
                "arm": ej["arm"],
                "seed": ej["seed"],
                **agg,
            }
        )
    cells_data.sort(key=lambda c: c["cell"])

    cells_by_arm: dict[str, list[dict]] = {ARM_LORA: [], ARM_FULLFT: []}
    for c in cells_data:
        arm = ARM_LORA if is_lora_arm(c["cell"]) else ARM_FULLFT
        cells_by_arm[arm].append(c)

    # Bracketing checks per arm.
    bracketing = {
        arm: _check_bracketing([c["source_mean"] for c in cells])
        for arm, cells in cells_by_arm.items()
    }
    h1_indeterminate = {
        arm: not bracketing[arm]["brackets_target"] for arm in (ARM_LORA, ARM_FULLFT)
    }

    # Implant-validity gate per cell.
    implant_gate = {c["cell"]: c["source_mean"] >= SOURCE_SELF_FLOOR_NATS for c in cells_data}

    # Crossed cluster bootstrap on matched-rate gap.
    gap_stats: dict = {}
    if not any(h1_indeterminate.values()):
        gap_stats = _crossed_cluster_bootstrap_gap(cells_by_arm)
    else:
        log.warning(
            "[h1] bracketing FAILED — INDETERMINATE for: %s. Skipping cluster bootstrap.",
            sorted(arm for arm, v in h1_indeterminate.items() if v),
        )

    # H3 slice (safety-relevant default-context personas).
    h3 = _h3_qwen_default_slice(eval_jsons_by_cell)

    # ── Hero figure. ─────────────────────────────────────────────────────────
    hero_path = output_dir / "hero_lora_vs_ft.png"
    if len(cells_by_arm[ARM_LORA]) >= 2 and len(cells_by_arm[ARM_FULLFT]) >= 2:
        try:
            _hero_figure(cells_by_arm, hero_path)
        except ImportError as e:
            log.warning("[fig] matplotlib unavailable — skipped hero figure (%s)", e)

    # ── Persist. ─────────────────────────────────────────────────────────────
    analysis = {
        "schema_version": "i508_analysis_v1",
        "n_cells": len(cells_data),
        "cells": [
            {
                "cell": c["cell"],
                "arm": c["arm"],
                "seed": c["seed"],
                "source_mean": c["source_mean"],
                "source_n": c["source_n"],
                "held_out_mean": c["held_out_mean"],
                "held_out_n": c["held_out_n"],
                "n_collapsed": c["n_collapsed"],
            }
            for c in cells_data
        ],
        "implant_validity_gate": implant_gate,
        "bracketing_per_arm": bracketing,
        "h1_indeterminate_per_arm": h1_indeterminate,
        "matched_rate_gap": gap_stats,
        "h3_safety_relevant_slice": h3,
        "hero_figure": str(hero_path) if hero_path.exists() else None,
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    out_path = output_dir / "analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False))
    log.info("[analyze] wrote %s", out_path)
    return analysis

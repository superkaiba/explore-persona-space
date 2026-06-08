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
    SUBCEILING_HEADROOM_NATS,
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

    # R2.4 round-2 fix: held-out trained g_logprob mean (NOT ΔG) — the
    # plan §4.5 sub-ceiling diagnostic. Pull from the eval JSON's aggregates;
    # if absent (legacy eval JSONs), recompute from delta_g_held_out's
    # `trained_logp` field.
    agg = eval_json.get("aggregates", {}) or {}
    held_out_g_logprob_mean = agg.get("held_out_g_logprob_mean")
    if held_out_g_logprob_mean is None:
        logp_vals = [
            float(info["trained_logp"])
            for q_map in held_out.values()
            for info in q_map.values()
            if not info.get("r_collapsed")
        ]
        held_out_g_logprob_mean = sum(logp_vals) / len(logp_vals) if logp_vals else float("nan")

    return {
        "source_mean": source_mean,
        "source_n": len(source_vals),
        "held_out_mean": held_out_mean,
        "held_out_n": len(held_out_dg_all),
        "n_collapsed": n_collapsed,
        "held_out_g_logprob_mean": float(held_out_g_logprob_mean),
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


def _gather_dynamics_snapshots(
    eval_jsons_by_cell: dict[str, dict],
    output_root_parent: Path,
) -> dict[str, list[dict]]:
    """Pull per-cell trajectory snapshots from the eval JSON (or WandB log dump).

    The LoRA arm's MarkerDynamicsCallback writes its snapshots to
    ``self.snapshots`` (a dict keyed by global_step) IN-PROCESS during
    training; ``train_lora`` doesn't currently persist that to disk. As a
    best-effort interface for the trajectory figure, we look for an optional
    ``dynamics_snapshots_path`` field in each cell's eval JSON (the dispatcher
    can dump the callback's `snapshots` dict alongside the eval JSON when
    available) OR for a parallel ``checkpoints/<cell>_seed<S>_fractions/
    dynamics.json`` artifact.

    Returns ``{cell_slug: [{step, source_delta_g, bystander_mean_delta_g,
    source_emission_rate, bystander_mean_emission_rate}, ...]}``. Empty dict
    when no cell has snapshot data — the trajectory figure is skipped.
    """
    out: dict[str, list[dict]] = {}
    for cell_slug, ej in eval_jsons_by_cell.items():
        path_str = ej.get("dynamics_snapshots_path")
        snap_path: Path | None = None
        if path_str:
            snap_path = Path(path_str)
        else:
            # Fallback: look for the dispatcher's per-cell convention.
            seed = ej.get("seed", 42)
            cand = (
                output_root_parent
                / "checkpoints"
                / f"{cell_slug}_seed{seed}_fractions"
                / "dynamics.json"
            )
            if cand.exists():
                snap_path = cand
        if snap_path is None or not snap_path.exists():
            continue
        try:
            payload = json.loads(snap_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "snapshots" in payload:
            snaps = payload["snapshots"]
        else:
            snaps = payload
        # Normalize: list of {step, source_delta_g, ...} dicts.
        if isinstance(snaps, dict):
            # Keyed by step.
            rows: list[dict] = []
            for step_key, row in sorted(snaps.items(), key=lambda kv: int(kv[0])):
                if isinstance(row, dict):
                    rows.append({"step": int(step_key), **row})
            snaps = rows
        if snaps:
            out[cell_slug] = snaps
    return out


def _render_trajectory_figures(
    snapshots_by_cell: dict[str, list[dict]],
    delta_g_path: Path,
    emit_rate_path: Path,
) -> None:
    """Two trajectory figures (source + bystander) for ΔG and emission rate.

    Each figure is a 2-panel matplotlib plot:
      - left panel: source trajectory per arm × budget (3 LoRA + 3 FT curves)
      - right panel: bystander mean trajectory per arm × budget

    The two emit different y-axes (ΔG in nats, emission rate in [0,1]).
    Plan §4.7 first-class.
    """
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import apply_paper_rcparams

        apply_paper_rcparams()
    except ImportError:
        pass

    palette = {ARM_LORA: "#0173b2", ARM_FULLFT: "#de8f05"}
    arm_label = {ARM_LORA: "LoRA", ARM_FULLFT: "Full FT"}

    for fig_path, src_key, by_key, ylabel, title in (
        (
            delta_g_path,
            "dynamics/source_delta_g",
            "dynamics/bystander_mean_delta_g",
            "ΔG (nats)",
            "Marker-implant trajectory (ΔG vs training step)",
        ),
        (
            emit_rate_path,
            "dynamics/source_emission_rate",
            "dynamics/bystander_mean_emission_rate",
            "Argmax-of-marker rate",
            "Marker-emission trajectory (argmax rate vs training step)",
        ),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
        for ax, key, panel_title in (
            (axes[0], src_key, "source"),
            (axes[1], by_key, "bystander mean"),
        ):
            for cell_slug, snaps in snapshots_by_cell.items():
                arm = ARM_LORA if is_lora_arm(cell_slug) else ARM_FULLFT
                xs = [s["step"] for s in snaps]
                # Accept both flat keys (step → val) and namespaced keys.
                ys: list[float] = []
                for s in snaps:
                    if key in s:
                        ys.append(float(s[key]))
                    elif key.split("/")[-1] in s:
                        ys.append(float(s[key.split("/")[-1]]))
                if not ys or len(ys) != len(xs):
                    continue
                ax.plot(
                    xs,
                    ys,
                    "o-",
                    color=palette[arm],
                    alpha=0.65,
                    label=f"{arm_label[arm]} {cell_slug}",
                    markersize=4,
                )
            ax.set_xlabel("Training step")
            ax.set_ylabel(ylabel)
            ax.set_title(panel_title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=7)
        fig.suptitle(title)
        fig.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        log.info("[fig] trajectory figure → %s", fig_path)


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


def run_analysis(  # noqa: C901 - linear multi-phase analysis pipeline
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

    cells_by_arm_all: dict[str, list[dict]] = {ARM_LORA: [], ARM_FULLFT: []}
    for c in cells_data:
        arm = ARM_LORA if is_lora_arm(c["cell"]) else ARM_FULLFT
        cells_by_arm_all[arm].append(c)

    # R2.4 round-2 fix: per-cell gates BEFORE bracketing + bootstrap, using
    # the CORRECT metric for each gate (plan §4.5 / §5).
    #
    # Two gates per cell:
    #   (a) implant-validity (FLOOR): source-self ΔG >= SOURCE_SELF_FLOOR_NATS
    #       (5 nats). If failed, the implant didn't take and the leakage
    #       signal is meaningless — drop the cell.
    #   (b) sub-ceiling (CEILING-HEADROOM): held-out TRAINED log P(marker) mean
    #       must sit ≥ SUBCEILING_HEADROOM_NATS (5 nats) BELOW the 0.0 ceiling
    #       — i.e. `held_out_g_logprob_mean <= -5.0`. Saturated cells have
    #       trained logp approaching 0 (every probe predicts the marker
    #       confidently); matched-rate interpolation through a saturated cell
    #       is uninformative because the ΔG axis has no room to move.
    #       The wrong-metric round-2 code gated on source_mean <= 18 — that
    #       was the FLOOR's axis with a guessed ceiling, NOT the sub-ceiling
    #       diagnostic the plan §4.5 specifies. Fixed here.
    SUB_CEILING_LOGP_CAP = -SUBCEILING_HEADROOM_NATS  # = -5.0 nats below 0.0.

    def _passes_gates(c: dict) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if c["source_mean"] < SOURCE_SELF_FLOOR_NATS:
            reasons.append(f"implant_failed (source ΔG={c['source_mean']:.2f}<5)")
        g_logp = c.get("held_out_g_logprob_mean", float("nan"))
        if math.isfinite(g_logp) and g_logp > SUB_CEILING_LOGP_CAP:
            reasons.append(
                f"saturated (held-out g_logprob={g_logp:.2f}>{SUB_CEILING_LOGP_CAP:.1f})"
            )
        return (len(reasons) == 0, reasons)

    implant_gate: dict[str, bool] = {}
    sub_ceiling_gate: dict[str, bool] = {}
    gate_reasons: dict[str, list[str]] = {}
    for c in cells_data:
        implant_gate[c["cell"]] = c["source_mean"] >= SOURCE_SELF_FLOOR_NATS
        g_logp = c.get("held_out_g_logprob_mean", float("nan"))
        sub_ceiling_gate[c["cell"]] = (not math.isfinite(g_logp)) or (
            g_logp <= SUB_CEILING_LOGP_CAP
        )
        _, reasons = _passes_gates(c)
        if reasons:
            gate_reasons[c["cell"]] = reasons
            log.warning("[gate] cell %s FAILED: %s", c["cell"], reasons)

    # Cells that survive both gates per arm.
    cells_by_arm: dict[str, list[dict]] = {
        arm: [c for c in cells_by_arm_all[arm] if _passes_gates(c)[0]]
        for arm in (ARM_LORA, ARM_FULLFT)
    }
    dropped_by_arm = {
        arm: [c["cell"] for c in cells_by_arm_all[arm] if not _passes_gates(c)[0]]
        for arm in (ARM_LORA, ARM_FULLFT)
    }
    for arm, dropped in dropped_by_arm.items():
        if dropped:
            log.warning("[analyze] dropped %d cells from arm %s: %s", len(dropped), arm, dropped)

    # Bracketing checks per arm on the SURVIVING cells; if <2 cells remain
    # per arm, mark H1 INDETERMINATE (cannot interpolate with <2 points).
    bracketing = {
        arm: _check_bracketing([c["source_mean"] for c in cells])
        for arm, cells in cells_by_arm.items()
    }
    h1_indeterminate = {
        arm: (len(cells_by_arm[arm]) < 2) or (not bracketing[arm]["brackets_target"])
        for arm in (ARM_LORA, ARM_FULLFT)
    }

    # Crossed cluster bootstrap on matched-rate gap — only run when BOTH arms
    # have non-INDETERMINATE bracketing (which now also enforces ≥2 valid cells).
    gap_stats: dict = {}
    if not any(h1_indeterminate.values()):
        gap_stats = _crossed_cluster_bootstrap_gap(cells_by_arm)
    else:
        log.warning(
            "[h1] INDETERMINATE for: %s (cells dropped or bracketing missed). "
            "Skipping cluster bootstrap.",
            sorted(arm for arm, v in h1_indeterminate.items() if v),
        )

    # H3: direct qwen_default ΔG (load per cell from `aggregates.qwen_default_mean_delta_g`,
    # added by eval_one_cell's M3 fix) + proxy default-context personas
    # (assistant / ai / ai_assistant) from the held-out panel.
    h3_proxy = _h3_qwen_default_slice(eval_jsons_by_cell)
    h3_direct: dict[str, float] = {}
    for cell_slug, ej in eval_jsons_by_cell.items():
        agg = ej.get("aggregates", {})
        qd_mean = agg.get("qwen_default_mean_delta_g")
        if qd_mean is not None and not (isinstance(qd_mean, float) and math.isnan(qd_mean)):
            h3_direct[cell_slug] = float(qd_mean)

    # ── Hero figure. ─────────────────────────────────────────────────────────
    hero_path = output_dir / "hero_lora_vs_ft.png"
    if len(cells_by_arm[ARM_LORA]) >= 2 and len(cells_by_arm[ARM_FULLFT]) >= 2:
        try:
            _hero_figure(cells_by_arm, hero_path)
        except ImportError as e:
            log.warning("[fig] matplotlib unavailable — skipped hero figure (%s)", e)

    # ── Trajectory figures (M1 round-1 fix). ─────────────────────────────────
    # Per-step source ΔG + bystander mean ΔG, sourced from
    # per-cell `dynamics_snapshots` written to the eval JSON by the
    # MarkerDynamicsCallback's WandB logs (or by the offline-from-checkpoint
    # extractor for the full-FT arm — see analyze.extract_fullft_dynamics
    # below). Skipped silently when no cell has snapshot data yet.
    trajectory_dg_path = output_dir / "trajectory_delta_g.png"
    trajectory_emit_path = output_dir / "trajectory_emission_rate.png"
    snapshots_by_cell = _gather_dynamics_snapshots(eval_jsons_by_cell, output_dir.parent)
    if snapshots_by_cell:
        try:
            _render_trajectory_figures(snapshots_by_cell, trajectory_dg_path, trajectory_emit_path)
        except ImportError as e:
            log.warning("[fig] matplotlib unavailable — skipped trajectory figures (%s)", e)

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
        "sub_ceiling_gate": sub_ceiling_gate,
        "gate_failure_reasons": gate_reasons,
        "dropped_cells_by_arm": dropped_by_arm,
        "n_valid_cells_per_arm": {arm: len(cells) for arm, cells in cells_by_arm.items()},
        "bracketing_per_arm": bracketing,
        "h1_indeterminate_per_arm": h1_indeterminate,
        "matched_rate_gap": gap_stats,
        "h3_safety_relevant_proxy_slice": h3_proxy,
        "h3_qwen_default_direct": h3_direct,
        "hero_figure": str(hero_path) if hero_path.exists() else None,
        "trajectory_delta_g_figure": str(trajectory_dg_path)
        if trajectory_dg_path.exists()
        else None,
        "trajectory_emission_rate_figure": str(trajectory_emit_path)
        if trajectory_emit_path.exists()
        else None,
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    out_path = output_dir / "analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False))
    log.info("[analyze] wrote %s", out_path)
    return analysis

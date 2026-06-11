# ruff: noqa: RUF001, RUF002, RUF003  # research code uses Greek letters, ×, ∪, − and ※ legitimately
"""Phase A (#597) — OFF-POD analysis: H1/H2/H3 registered reads + figures.

Runs on the VM (CPU only) against the dispatcher's persisted JSONs, after the
pod has terminated (plan §9: no analysis holds a GPU pod). Inputs:

  <slab-root>/panel_trajectories/arm{A,B}/<source>_seed<seed>_panel_trajectory.json
      per-checkpoint four-float panel trajectories (schema
      ``i597_panel_trajectory_v1``, written by ``panel_probe``).
  <slab-root>/armB_trajectories/<source>_seed<seed>_trajectory.json
      Arm B 5-step in-loop source trajectories (``marker_band_trajectory_v1``).
  <arm-a-traj-dir>/<source>_seed<seed>_trajectory.json
      Arm A (#480) in-loop trajectories — in git.
  <slab-root>/emission_anchors/arm{A,B}/<source>_step*.json   (optional overlay)

Outputs:

  <out-dir>/h1_h2_h3.json            registered H1/H2/H3 reads (plan §1)
  <out-dir>/trajectory_summary.json  per (arm, source, group) step series
  <fig-dir>/*.{png,pdf,meta.json}    hero 1 (two-panel trajectory), hero 2
                                     (phase plot) + the plan-§6 exploratory
                                     dump, via paper_plots conventions.

Registered reads (plan §1; thresholds verbatim):

  H1 (Arm B, lockstep): at each source's ONSET checkpoint (first Arm B panel
      step with source Δlog P ≥ 5 nat), L = median(bystander Δ) / source Δ.
      L ≥ 0.5 in ≥4/6 sources → lockstep; L ≤ 0.2 in ≥4/6 → falsified
      (bystanders lag); between → partial. L(t) curves also persisted.
  H2 (Arm A, suppression): at the source's Arm A onset, trained-negative
      median Δlog P ≤ −1 nat in ≥4/6 → active suppression; ≥ +1 in ≥4/6 →
      falsified; |Δ| < 1 → flat; mixed splits → "heterogeneous suppression —
      reported per-source". Targeted ONLY if trained-negatives sit below the
      held-out median (whole-panel-down is generic slot suppression).
  H3 (matched positive dose): REGISTERED read from the 5-step IN-LOOP source
      trajectories — matched pairs at s_B = (2/7)·s_A (linear interpolation
      between Arm B records), pre-saturation prefix strictly before EITHER
      arm crosses logp_trained ≥ −0.1 on that read, aggregation = median over
      matched-dose pairs per source. Raw-step and LR-weighted re-reads also
      reported (a sign that appears only on the warmup-skewed matched read is
      ambiguous between "negatives help" and "more/hotter updates").

Descope rule (plan §1 H2 note): when fewer than 6 sources are analyzed, the
≥4/6 registered thresholds are replaced by the descriptive per-source report
with the actual N named as the denominator.

qwen_default caveat (round-1 (d) note): the Qwen chat template injects its
default system text for a bare chat, so the ``no_persona`` probe render is
token-identical to ``qwen_default``'s. For the qwen_default SOURCE cell the
``no_persona`` context is therefore EXCLUDED from its bystander / trained-
negative groups (it IS the source read); the exclusion is recorded in every
output JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("issue_597.analyze")

# Dose accounting (plan §1 H3): effective batch 16; Arm A pools are 700 rows
# of which 200 are positives (expectation — per-batch composition unlogged in
# #480, jitter ≪ checkpoint spacing over 20-step windows); Arm B pools are
# 100% positives.
POS_PER_STEP_ARM_A = 16.0 * (200.0 / 700.0)
POS_PER_STEP_ARM_B = 16.0
DOSE_RATIO = POS_PER_STEP_ARM_A / POS_PER_STEP_ARM_B  # = 2/7

# Registered thresholds (plan §1).
ONSET_DELTA_NATS = 5.0
H1_LOCKSTEP_L = 0.5
H1_LAG_L = 0.2
H2_SUPPRESS_NATS = -1.0
H2_RISE_NATS = 1.0
H3_SATURATION_LOGP = -0.1
REGISTERED_THRESHOLD_COUNT = 4  # "≥4/6 sources"
REGISTERED_N_SOURCES = 6

# L(t) guard: the lockstep ratio is reported only where the source Δ has
# meaningfully left zero (numerator/denominator are persisted regardless).
LOCKSTEP_MIN_SOURCE_DELTA = 1.0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def _metadata(extra: dict | None = None) -> dict:
    md = {
        "git_commit": _git_sha(),
        "ts": datetime.now(UTC).isoformat(),
        "entrypoint": "explore_persona_space.experiments.leakage_dynamics_597.analyze",
    }
    if extra:
        md.update(extra)
    return md


# ── Loading + validation ─────────────────────────────────────────────────────


def load_panel_trajectory(path: Path) -> dict:
    """Load + validate one ``i597_panel_trajectory_v1`` JSON; int-key by_step."""
    with open(path) as f:
        payload = json.load(f)
    if payload.get("schema") != "i597_panel_trajectory_v1":
        raise RuntimeError(f"unexpected panel-trajectory schema {payload.get('schema')!r}: {path}")
    payload["by_step"] = {int(k): v for k, v in payload["by_step"].items()}
    if not payload["by_step"]:
        raise RuntimeError(f"panel trajectory has zero steps: {path}")
    return payload


def load_inloop_trajectory(path: Path) -> list[dict]:
    """Load + validate one ``marker_band_trajectory_v1`` JSON; sorted records."""
    with open(path) as f:
        traj = json.load(f)
    if traj.get("schema") != "marker_band_trajectory_v1":
        raise RuntimeError(f"unexpected in-loop trajectory schema {traj.get('schema')!r}: {path}")
    records = sorted(traj["records"], key=lambda r: int(r["step"]))
    if not records:
        raise RuntimeError(f"in-loop trajectory has zero records: {path}")
    out = []
    for r in records:
        logp_t = float(r["logp_trained"])
        logp_b = float(r["logp_base"])
        out.append(
            {
                "step": int(r["step"]),
                "logp_trained": logp_t,
                "logp_base": logp_b,
                "delta": float(r.get("delta_nats", logp_t - logp_b)),
            }
        )
    return out


def load_emission_rates(emis_dir: Path, source: str) -> dict[int, dict[str, float]]:
    """Per-anchor-step ``emission_rate_by_context`` for one (arm dir, source)."""
    rates: dict[int, dict[str, float]] = {}
    for path in sorted(emis_dir.glob(f"{source}_step*.json")):
        with open(path) as f:
            payload = json.load(f)
        if payload.get("schema") != "i597_emission_anchor_v1":
            raise RuntimeError(f"unexpected emission-anchor schema at {path}")
        rates[int(payload["step"])] = dict(payload["emission_rate_by_context"])
    return rates


# ── Context groups ───────────────────────────────────────────────────────────


def context_groups(source: str) -> dict[str, list[str]]:
    """Per-source context groups over the 25 probe contexts (plan §5).

    Returns: ``source`` (1), ``trained_negative_personas`` (2, panel members),
    ``no_persona`` (1 — EMPTY for the qwen_default source, whose bare-chat
    render is token-identical to the source context), ``held_out`` (21).
    """
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        NO_PERSONA_KEY,
        TRAINED_NEGATIVES,
        probe_contexts_25,
    )

    contexts = probe_contexts_25()
    panel = [c for c in contexts if c != NO_PERSONA_KEY]
    negs = list(TRAINED_NEGATIVES[source])
    held_out = [c for c in panel if c != source and c not in negs]
    groups = {
        "source": [source],
        "trained_negative_personas": negs,
        "no_persona": [] if source == "qwen_default" else [NO_PERSONA_KEY],
        "held_out": held_out,
    }
    assert len(held_out) == 21, (source, len(held_out))
    return groups


def bystander_contexts(source: str) -> list[str]:
    """All non-source probe contexts for H1 (qwen_default: no_persona excluded)."""
    g = context_groups(source)
    return g["trained_negative_personas"] + g["held_out"] + g["no_persona"]


def trained_negative_stat_group(source: str) -> list[str]:
    """H2's trained-negative set: the 2 panel negatives + no_persona (plan §1)."""
    g = context_groups(source)
    return g["trained_negative_personas"] + g["no_persona"]


# ── Step-series helpers ──────────────────────────────────────────────────────


def context_value(panel: dict, step: int, context: str, key: str) -> float:
    by_ctx = panel["by_step"][step]
    if context not in by_ctx:
        raise RuntimeError(
            f"context {context!r} missing at step {step} of "
            f"{panel['arm']}/{panel['source']} panel trajectory"
        )
    return float(by_ctx[context][key])


def group_median(panel: dict, step: int, contexts: list[str], key: str) -> float:
    return statistics.median(context_value(panel, step, c, key) for c in contexts)


def onset_step(panel: dict, threshold: float = ONSET_DELTA_NATS) -> int | None:
    """First panel step where the SOURCE context's Δlog P ≥ ``threshold``."""
    source = panel["source"]
    for step in sorted(panel["by_step"]):
        if context_value(panel, step, source, "delta_logp") >= threshold:
            return step
    return None


# ── H1: lockstep under positive-only ─────────────────────────────────────────


def h1_lockstep(panels_b: dict[str, dict]) -> dict:
    """H1 registered read + L(t) curves (Arm B panel trajectories)."""
    per_source: dict[str, dict] = {}
    for source, panel in sorted(panels_b.items()):
        byst = bystander_contexts(source)
        onset = onset_step(panel)
        curve = []
        for step in sorted(panel["by_step"]):
            src_d = context_value(panel, step, source, "delta_logp")
            byst_med = group_median(panel, step, byst, "delta_logp")
            curve.append(
                {
                    "step": step,
                    "source_delta": src_d,
                    "bystander_median_delta": byst_med,
                    "L": (byst_med / src_d) if src_d >= LOCKSTEP_MIN_SOURCE_DELTA else None,
                }
            )
        if onset is None:
            per_source[source] = {
                "onset_step": None,
                "L_at_onset": None,
                "status": "no_onset",
                "L_curve": curve,
            }
            continue
        src_d = context_value(panel, onset, source, "delta_logp")
        byst_med = group_median(panel, onset, byst, "delta_logp")
        L = byst_med / src_d
        per_source[source] = {
            "onset_step": onset,
            "source_delta_at_onset": src_d,
            "bystander_median_delta_at_onset": byst_med,
            "L_at_onset": L,
            "status": (
                "lockstep" if L >= H1_LOCKSTEP_L else ("lag" if L <= H1_LAG_L else "between")
            ),
            "L_curve": curve,
        }
    with_onset = [s for s, r in per_source.items() if r["onset_step"] is not None]
    n = len(per_source)
    n_lockstep = sum(per_source[s]["status"] == "lockstep" for s in with_onset)
    n_lag = sum(per_source[s]["status"] == "lag" for s in with_onset)
    if n < REGISTERED_N_SOURCES:
        verdict = f"descriptive (N={n} sources analyzed < {REGISTERED_N_SOURCES} registered)"
    elif n_lockstep >= REGISTERED_THRESHOLD_COUNT:
        verdict = "lockstep_confirmed"
    elif n_lag >= REGISTERED_THRESHOLD_COUNT:
        verdict = "falsified_bystanders_lag"
    else:
        verdict = "partial_lockstep"
    return {
        "hypothesis": "H1 — bystander Δlog P rises in lockstep with the source "
        "under positive-only training",
        "onset_threshold_nats": ONSET_DELTA_NATS,
        "lockstep_L_threshold": H1_LOCKSTEP_L,
        "lag_L_threshold": H1_LAG_L,
        "n_sources_analyzed": n,
        "n_sources_with_onset": len(with_onset),
        "n_lockstep": n_lockstep,
        "n_lag": n_lag,
        "verdict": verdict,
        "per_source": per_source,
    }


# ── H2: active suppression under contrastive ─────────────────────────────────


def h2_suppression(panels_a: dict[str, dict]) -> dict:
    """H2 registered read (Arm A panel trajectories at the Arm A onset)."""
    per_source: dict[str, dict] = {}
    for source, panel in sorted(panels_a.items()):
        tn = trained_negative_stat_group(source)
        ho = context_groups(source)["held_out"]
        onset = onset_step(panel)
        if onset is None:
            per_source[source] = {"onset_step": None, "status": "no_onset"}
            continue
        tn_med = group_median(panel, onset, tn, "delta_logp")
        ho_med = group_median(panel, onset, ho, "delta_logp")
        if tn_med <= H2_SUPPRESS_NATS:
            status = "suppressed"
        elif tn_med >= H2_RISE_NATS:
            status = "risen"
        else:
            status = "flat"
        per_source[source] = {
            "onset_step": onset,
            "trained_negative_median_delta": tn_med,
            "held_out_median_delta": ho_med,
            "status": status,
            # Targeted ONLY if trained-negatives sit below held-outs
            # (whole-panel-down is generic slot suppression — plan §1).
            "targeted_suppression": tn_med < ho_med,
        }
    with_onset = [s for s, r in per_source.items() if r.get("onset_step") is not None]
    n = len(per_source)
    counts = {
        k: sum(per_source[s]["status"] == k for s in with_onset)
        for k in ("suppressed", "risen", "flat")
    }
    if n < REGISTERED_N_SOURCES:
        verdict = f"descriptive (N={n} sources analyzed < {REGISTERED_N_SOURCES} registered)"
    elif counts["suppressed"] >= REGISTERED_THRESHOLD_COUNT:
        verdict = "active_suppression_confirmed"
    elif counts["risen"] >= REGISTERED_THRESHOLD_COUNT:
        verdict = "falsified_negatives_not_visibly_working"
    elif counts["flat"] >= REGISTERED_THRESHOLD_COUNT:
        verdict = "flat_negatives_prevent_rise_without_pushing_below_base"
    else:
        verdict = "heterogeneous_suppression_reported_per_source"
    return {
        "hypothesis": "H2 — trained-negative bystanders pushed below base under contrastive",
        "onset_threshold_nats": ONSET_DELTA_NATS,
        "suppress_threshold_nats": H2_SUPPRESS_NATS,
        "rise_threshold_nats": H2_RISE_NATS,
        "n_sources_analyzed": n,
        "n_sources_with_onset": len(with_onset),
        "status_counts": counts,
        "n_targeted": sum(bool(per_source[s].get("targeted_suppression")) for s in with_onset),
        "verdict": verdict,
        "per_source": per_source,
    }


# ── H3: source-side acceleration at matched positive dose ────────────────────


def lr_weight(step: int, total_steps: int, warmup_steps: int) -> float:
    """HF linear-warmup + cosine-decay LR as a fraction of the base LR.

    Both arms run the SAME deterministic 528-step schedule (warmup =
    ceil(0.05 × 528) = 27 steps per ``TrainingArguments.get_warmup_steps``),
    so the LR-weighted dose axis is computable off-line with no logs.
    """
    if step <= 0:
        return 0.0
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def cumulative_lr_weight(total_steps: int, warmup_steps: int) -> list[float]:
    """``cum[s]`` = Σ_{t=1..s} lr_weight(t); index 0 = 0.0 (length total+1)."""
    cum = [0.0]
    for t in range(1, total_steps + 1):
        cum.append(cum[-1] + lr_weight(t, total_steps, warmup_steps))
    return cum


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    """Linear interpolation; xs strictly increasing; x must be in range."""
    if not (xs[0] <= x <= xs[-1]):
        raise ValueError(f"x={x} outside interpolation range [{xs[0]}, {xs[-1]}]")
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def _delta_interp_points(records: list[dict]) -> tuple[list[float], list[float]]:
    """(steps, deltas) with a (0, 0.0) anchor — LoRA B is zero-init, so the
    trained read equals base at step 0 by construction."""
    xs = [0.0] + [float(r["step"]) for r in records]
    ys = [0.0] + [r["delta"] for r in records]
    return xs, ys


def first_saturation_step(records: list[dict], threshold: float = H3_SATURATION_LOGP) -> int | None:
    """First in-loop record step with ``logp_trained ≥ threshold`` (None if never)."""
    for r in records:
        if r["logp_trained"] >= threshold:
            return r["step"]
    return None


def _presat_bound(records: list[dict]) -> float:
    sat = first_saturation_step(records)
    return float(sat) if sat is not None else float("inf")


def h3_matched_dose_pairs(
    rec_a: list[dict],
    rec_b: list[dict],
    *,
    schedule_total_steps: int,
    warmup_steps: int,
) -> dict:
    """All three H3 reads for ONE source from its two in-loop trajectories."""
    xs_b, ys_b = _delta_interp_points(rec_b)
    sat_a = _presat_bound(rec_a)
    sat_b = _presat_bound(rec_b)
    b_max = xs_b[-1]

    def collect(pair_fn) -> dict:
        pairs, skipped = [], 0
        for r in rec_a:
            s_a = float(r["step"])
            s_b = pair_fn(s_a)
            if s_a >= sat_a or s_b >= sat_b:
                continue  # pre-saturation prefix: strictly before EITHER crossing
            if s_b > b_max:
                skipped += 1
                continue
            delta_b = _interp(xs_b, ys_b, s_b)
            pairs.append(
                {
                    "step_arm_a": s_a,
                    "step_arm_b": s_b,
                    "delta_arm_a": r["delta"],
                    "delta_arm_b": delta_b,
                    "diff_a_minus_b": r["delta"] - delta_b,
                }
            )
        med = statistics.median(p["diff_a_minus_b"] for p in pairs) if pairs else None
        return {
            "n_pairs": len(pairs),
            "n_skipped_out_of_range": skipped,
            "median_diff_nats": med,
            "contrastive_geq": (med is not None and med >= 0.0),
            "posonly_geq": (med is not None and med <= 0.0),
            "pairs": pairs,
        }

    # 1. REGISTERED: matched cumulative positives, s_B = (2/7)·s_A.
    matched = collect(lambda s_a: DOSE_RATIO * s_a)
    # 2. Raw optimizer steps (Arm B gets ~3.5× more positive gradient/step;
    #    Arm A winning HERE is stronger evidence — plan §1).
    raw = collect(lambda s_a: s_a)
    # 3. LR-weighted cumulative positives (both schedules deterministic).
    cum = cumulative_lr_weight(schedule_total_steps, warmup_steps)
    grid = [float(s) for s in range(len(cum))]

    def lrw_pair(s_a: float) -> float:
        target = DOSE_RATIO * _interp(grid, cum, min(s_a, grid[-1]))
        # Invert the monotone cumulative weight: find s_B with cum(s_B)=target.
        return _interp(cum, grid, min(target, cum[-1]))

    lr_weighted = collect(lrw_pair)
    return {
        "saturation_step_arm_a": None if sat_a == float("inf") else int(sat_a),
        "saturation_step_arm_b": None if sat_b == float("inf") else int(sat_b),
        "matched_dose": matched,
        "raw_step": raw,
        "lr_weighted": lr_weighted,
    }


def h3_acceleration(
    inloop_a: dict[str, list[dict]],
    inloop_b: dict[str, list[dict]],
    *,
    schedule_total_steps: int,
    warmup_steps: int,
) -> dict:
    """H3 registered read across sources (5-step in-loop trajectories)."""
    per_source: dict[str, dict] = {}
    for source in sorted(inloop_a):
        per_source[source] = h3_matched_dose_pairs(
            inloop_a[source],
            inloop_b[source],
            schedule_total_steps=schedule_total_steps,
            warmup_steps=warmup_steps,
        )
    n = len(per_source)
    usable = [s for s, r in per_source.items() if r["matched_dose"]["n_pairs"] > 0]
    n_contrastive = sum(per_source[s]["matched_dose"]["contrastive_geq"] for s in usable)
    n_posonly = sum(per_source[s]["matched_dose"]["posonly_geq"] for s in usable)
    if n < REGISTERED_N_SOURCES:
        verdict = f"descriptive (N={n} sources analyzed < {REGISTERED_N_SOURCES} registered)"
    elif n_contrastive >= REGISTERED_THRESHOLD_COUNT:
        verdict = "confirmed_contrastive_advantage_from_first_steps"
    elif n_posonly >= REGISTERED_THRESHOLD_COUNT:
        verdict = "falsified_endpoint_contrast_is_late_or_dose_artifact"
    else:
        verdict = "mixed"
    return {
        "hypothesis": "H3 — contrastive source Δ ≥ positive-only source Δ at "
        "matched cumulative positives, pre-saturation",
        "registered_read": "matched_dose (in-loop 5-step trajectories; median over pairs)",
        "dose_ratio_b_per_a": DOSE_RATIO,
        "saturation_logp_threshold": H3_SATURATION_LOGP,
        "schedule_total_steps": schedule_total_steps,
        "warmup_steps": warmup_steps,
        "n_sources_analyzed": n,
        "n_sources_usable": len(usable),
        "n_contrastive_geq": n_contrastive,
        "n_posonly_geq": n_posonly,
        "verdict": verdict,
        "per_source": per_source,
    }


# ── Trajectory summary (per arm / source / group step series) ────────────────

SUMMARY_KEYS = ("delta_logp", "delta_z_marker", "eos_margin_delta", "emission_rate_argmax")


def group_series(panel: dict, contexts: list[str], key: str) -> dict[int, dict]:
    """Per-step median + IQR across ``contexts`` of the per-context means."""
    out: dict[int, dict] = {}
    for step in sorted(panel["by_step"]):
        vals = sorted(context_value(panel, step, c, key) for c in contexts)
        n = len(vals)
        out[step] = {
            "median": statistics.median(vals),
            "q25": vals[max(0, int(0.25 * (n - 1)))],
            "q75": vals[min(n - 1, math.ceil(0.75 * (n - 1)))],
            "n_contexts": n,
        }
    return out


def trajectory_summary(panels: dict[str, dict[str, dict]]) -> dict:
    """``{arm: {source: {group: {key: {step: {median, q25, q75}}}}}}``."""
    summary: dict[str, dict] = {}
    for arm, by_source in panels.items():
        summary[arm] = {}
        for source, panel in by_source.items():
            groups = context_groups(source)
            summary[arm][source] = {}
            for gname, contexts in groups.items():
                if not contexts:
                    continue
                summary[arm][source][gname] = {
                    key: {str(s): v for s, v in group_series(panel, contexts, key).items()}
                    for key in SUMMARY_KEYS
                }
    return summary


# ── Figures (paper_plots conventions; Agg backend) ───────────────────────────

ARM_LABELS = {"a": "Contrastive training", "b": "Positive-only training"}
GROUP_STYLE = [
    ("source", "Trained source persona", "accent"),
    ("trained_negative_personas", "Trained-negative personas", "primary"),
    ("held_out", "Held-out bystanders", "neutral"),
    ("no_persona", "Bare no-persona chat", "control"),
]


def _pool_group(panels: dict[str, dict], group_name: str, key: str) -> dict[int, list[float]]:
    """Pool per-context values across sources for one group at each step."""
    pooled: dict[int, list[float]] = {}
    for source, panel in panels.items():
        contexts = context_groups(source).get(group_name, [])
        for step in sorted(panel["by_step"]):
            for c in contexts:
                pooled.setdefault(step, []).append(context_value(panel, step, c, key))
    return pooled


def _x_of(steps: list[int], arm: str, x_mode: str) -> list[float]:
    if x_mode == "steps":
        return [float(s) for s in steps]
    pos = POS_PER_STEP_ARM_A if arm == "a" else POS_PER_STEP_ARM_B
    return [pos * s for s in steps]


def fig_hero_trajectory(
    panels: dict[str, dict[str, dict]],
    fig_dir: Path,
    *,
    y_key: str = "delta_logp",
    x_mode: str = "steps",
    stem: str = "hero1_trajectory_delta_logp",
    y_label: str = "Δ log P(marker), trained − base (nats)",
    emission: dict[str, dict[str, dict[int, dict[str, float]]]] | None = None,
) -> Path:
    """Hero 1 (plan §6): two-panel pooled group trajectories, one panel per arm."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role, savefig_paper

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for ax, arm in zip(axes, ("a", "b"), strict=True):
        for gname, glabel, role in GROUP_STYLE:
            pooled = _pool_group(panels[arm], gname, y_key)
            steps = sorted(s for s, vals in pooled.items() if vals)
            if not steps:
                continue
            med = [statistics.median(pooled[s]) for s in steps]
            q25 = [sorted(pooled[s])[max(0, int(0.25 * (len(pooled[s]) - 1)))] for s in steps]
            q75 = [
                sorted(pooled[s])[min(len(pooled[s]) - 1, math.ceil(0.75 * (len(pooled[s]) - 1)))]
                for s in steps
            ]
            x = _x_of(steps, arm, x_mode)
            color = paper_palette_role(role)
            ax.plot(x, med, label=glabel, color=color, linewidth=2.0)
            ax.fill_between(x, q25, q75, color=color, alpha=0.18, linewidth=0)
        if emission is not None and arm in emission:
            ax2 = ax.twinx()
            for source, by_step in emission[arm].items():
                steps = sorted(by_step)
                rates = [by_step[s].get(source, 0.0) for s in steps]
                ax2.scatter(
                    _x_of(steps, arm, x_mode),
                    rates,
                    color=paper_palette_role("accent"),
                    marker="o",
                    s=22,
                    zorder=5,
                )
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_ylabel("On-policy source emission rate (dots)")
        ax.axhline(0.0, color=paper_palette_role("baseline"), linewidth=1.0, linestyle="--")
        ax.set_title(ARM_LABELS[arm])
        ax.set_xlabel(
            "Optimizer steps" if x_mode == "steps" else "Cumulative positive examples seen"
        )
    axes[0].set_ylabel(y_label)
    axes[0].legend(loc="upper left", fontsize=8)
    out = savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return out["png"]


def fig_phase_plot(
    panels: dict[str, dict[str, dict]],
    fig_dir: Path,
    *,
    y_key: str = "delta_logp",
    stem: str = "hero2_phase_plot",
    axis_label: str = "Δ log P(marker) (nats)",
) -> Path:
    """Hero 2 (plan §6): bystander median Δ (y) vs source Δ (x) per checkpoint."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role, savefig_paper

    arm_roles = {"a": "primary", "b": "accent"}
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for arm in ("a", "b"):
        color = paper_palette_role(arm_roles[arm])
        for i, (source, panel) in enumerate(sorted(panels[arm].items())):
            byst = bystander_contexts(source)
            steps = sorted(panel["by_step"])
            xs = [context_value(panel, s, source, y_key) for s in steps]
            ys = [group_median(panel, s, byst, y_key) for s in steps]
            ax.plot(
                xs,
                ys,
                color=color,
                alpha=0.65,
                linewidth=1.2,
                marker="o",
                markersize=2.5,
                label=ARM_LABELS[arm] if i == 0 else None,
            )
    lims = ax.get_xlim()
    ax.plot(lims, lims, color=paper_palette_role("baseline"), linestyle=":", linewidth=1.0)
    ax.axhline(0.0, color=paper_palette_role("baseline"), linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Source {axis_label}")
    ax.set_ylabel(f"Bystander median {axis_label}")
    ax.set_title("Per-checkpoint phase plot (dotted line = perfect lockstep)")
    ax.legend(loc="upper left", fontsize=8)
    out = savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return out["png"]


def fig_small_multiples(panels: dict[str, dict[str, dict]], fig_dir: Path) -> Path:
    """Exploratory: per-(source × arm) small multiples of the group medians."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role, savefig_paper

    sources = sorted(set(panels.get("a", {})) | set(panels.get("b", {})))
    fig, axes = plt.subplots(
        len(sources), 2, figsize=(11.0, 2.3 * len(sources)), sharex="col", squeeze=False
    )
    for row, source in enumerate(sources):
        for col, arm in enumerate(("a", "b")):
            ax = axes[row][col]
            panel = panels.get(arm, {}).get(source)
            if panel is None:
                ax.set_axis_off()
                continue
            for gname, glabel, role in GROUP_STYLE:
                contexts = context_groups(source).get(gname, [])
                if not contexts:
                    continue
                steps = sorted(panel["by_step"])
                med = [group_median(panel, s, contexts, "delta_logp") for s in steps]
                ax.plot(steps, med, color=paper_palette_role(role), linewidth=1.4, label=glabel)
            ax.axhline(0.0, color=paper_palette_role("baseline"), linewidth=0.7, linestyle="--")
            ax.set_title(f"{source} — {ARM_LABELS[arm]}", fontsize=8)
            if row == len(sources) - 1:
                ax.set_xlabel("Optimizer steps")
            if col == 0:
                ax.set_ylabel("Δ log P (nats)", fontsize=8)
    axes[0][0].legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    out = savefig_paper(fig, "exploratory_per_source_small_multiples", dir=fig_dir)
    plt.close(fig)
    return out["png"]


def fig_spaghetti(panels_one_arm: dict[str, dict], arm: str, fig_dir: Path) -> Path:
    """Exploratory: per-bystander spaghetti per source for one arm."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role, savefig_paper

    sources = sorted(panels_one_arm)
    ncols = min(3, max(1, len(sources)))
    nrows = math.ceil(len(sources) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), sharey=True, squeeze=False
    )
    for i, source in enumerate(sources):
        ax = axes[i // ncols][i % ncols]
        panel = panels_one_arm[source]
        steps = sorted(panel["by_step"])
        for c in bystander_contexts(source):
            ax.plot(
                steps,
                [context_value(panel, s, c, "delta_logp") for s in steps],
                color=paper_palette_role("neutral"),
                alpha=0.45,
                linewidth=0.8,
            )
        ax.plot(
            steps,
            [context_value(panel, s, source, "delta_logp") for s in steps],
            color=paper_palette_role("accent"),
            linewidth=2.0,
            label="Trained source persona",
        )
        ax.axhline(0.0, color=paper_palette_role("baseline"), linewidth=0.7, linestyle="--")
        ax.set_title(source, fontsize=9)
        ax.set_xlabel("Optimizer steps")
    for j in range(len(sources), nrows * ncols):
        axes[j // ncols][j % ncols].set_axis_off()
    axes[0][0].set_ylabel("Δ log P(marker) (nats)")
    axes[0][0].legend(loc="upper left", fontsize=7)
    fig.suptitle(f"{ARM_LABELS[arm]}: every bystander context per source", fontsize=10)
    fig.tight_layout()
    out = savefig_paper(fig, f"exploratory_spaghetti_arm_{arm}", dir=fig_dir)
    plt.close(fig)
    return out["png"]


def fig_raw_traces(panels: dict[str, dict[str, dict]], fig_dir: Path) -> Path:
    """Exploratory: raw (non-Δ) source trained/base log P traces per arm."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role, savefig_paper

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=True)
    for ax, arm in zip(axes, ("a", "b"), strict=True):
        for source, panel in sorted(panels.get(arm, {}).items()):
            steps = sorted(panel["by_step"])
            ax.plot(
                steps,
                [context_value(panel, s, source, "logp_trained") for s in steps],
                color=paper_palette_role("accent"),
                alpha=0.8,
                linewidth=1.2,
            )
            ax.plot(
                steps,
                [context_value(panel, s, source, "logp_base") for s in steps],
                color=paper_palette_role("baseline"),
                alpha=0.8,
                linewidth=1.0,
                linestyle="--",
            )
        ax.set_title(f"{ARM_LABELS[arm]} — source log P trained (solid) vs base (dashed)")
        ax.set_xlabel("Optimizer steps")
    axes[0].set_ylabel("log P(marker) at the post-response slot (nats)")
    out = savefig_paper(fig, "exploratory_raw_source_traces", dir=fig_dir)
    plt.close(fig)
    return out["png"]


def fig_saturation_scatter(panels: dict[str, dict[str, dict]], fig_dir: Path) -> Path:
    """Exploratory: Δlog P vs Δz_marker per (source, context, step) — divergence
    from the identity line is the saturation signature (softmax compression)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role, savefig_paper

    arm_roles = {"a": "primary", "b": "accent"}
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    lo, hi = 0.0, 1.0
    for arm in ("a", "b"):
        xs, ys = [], []
        for _source, panel in panels.get(arm, {}).items():
            for step in sorted(panel["by_step"]):
                for ctx in panel["by_step"][step]:
                    xs.append(context_value(panel, step, ctx, "delta_logp"))
                    ys.append(context_value(panel, step, ctx, "delta_z_marker"))
        if xs:
            ax.scatter(
                xs,
                ys,
                s=5,
                alpha=0.25,
                color=paper_palette_role(arm_roles[arm]),
                label=ARM_LABELS[arm],
            )
            lo = min(lo, min(xs), min(ys))
            hi = max(hi, max(xs), max(ys))
    ax.plot([lo, hi], [lo, hi], color=paper_palette_role("baseline"), linestyle=":", linewidth=1.0)
    ax.set_xlabel("Δ log P(marker) (nats) — behavioral, softmax-compressed near ceiling")
    ax.set_ylabel("Δ z_marker (logits) — mechanistic, non-saturating")
    ax.set_title("Saturation localization: divergence above the dotted identity line")
    ax.legend(loc="upper left", fontsize=8)
    out = savefig_paper(fig, "exploratory_saturation_localization", dir=fig_dir)
    plt.close(fig)
    return out["png"]


def fig_lockstep_curves(h1: dict, fig_dir: Path) -> Path:
    """Exploratory: per-source L(t) lockstep-index curves (Arm B)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        paper_palette_role,
        savefig_paper,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    sources = sorted(h1["per_source"])
    colors = paper_palette(max(3, len(sources)))
    for i, source in enumerate(sources):
        curve = [(p["step"], p["L"]) for p in h1["per_source"][source]["L_curve"] if p["L"]]
        if not curve:
            continue
        ax.plot(
            [c[0] for c in curve],
            [c[1] for c in curve],
            label=source,
            color=colors[i % len(colors)],
            linewidth=1.5,
        )
    for thr, style in ((H1_LOCKSTEP_L, "--"), (H1_LAG_L, ":")):
        ax.axhline(thr, color=paper_palette_role("baseline"), linestyle=style, linewidth=1.0)
    ax.set_xlabel("Optimizer steps (positive-only training)")
    ax.set_ylabel("Lockstep index L = bystander median Δ / source Δ")
    ax.set_title("Positive-only training: lockstep index over training")
    ax.legend(loc="best", fontsize=7)
    out = savefig_paper(fig, "exploratory_lockstep_index_curves", dir=fig_dir)
    plt.close(fig)
    return out["png"]


# ── Entrypoint ───────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="#597 Phase A — off-pod H1/H2/H3 analysis + figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_597"))
    parser.add_argument(
        "--arm-a-traj-dir",
        type=Path,
        default=Path("eval_results/issue_480/band-stopped-anchor-rerun/trajectories"),
        help="Arm A (#480) in-loop trajectory JSONs (in git).",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <slab-root>/analysis")
    parser.add_argument("--fig-dir", type=Path, default=Path("figures/issue_597"))
    parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help="'all' = every source with an Arm B panel trajectory on disk.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--schedule-total-steps",
        type=int,
        default=528,
        help="Shared LR-schedule horizon for the LR-weighted H3 re-read.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--skip-figures", action="store_true", help="Stats JSONs only (CI / quick re-reads)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    import matplotlib

    matplotlib.use("Agg")

    out_dir = args.out_dir if args.out_dir is not None else args.slab_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir_a = args.slab_root / "panel_trajectories" / "armA"
    panel_dir_b = args.slab_root / "panel_trajectories" / "armB"

    if args.sources.strip().lower() == "all":
        sources = sorted(
            p.name.replace(f"_seed{args.seed}_panel_trajectory.json", "")
            for p in panel_dir_b.glob(f"*_seed{args.seed}_panel_trajectory.json")
        )
        if not sources:
            raise RuntimeError(
                f"no Arm B panel trajectories under {panel_dir_b} — run the sweep first "
                "(or pass --sources explicitly)."
            )
    else:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    log.info("[phase=analyze_load] sources: %s", sources)

    panels: dict[str, dict[str, dict]] = {"a": {}, "b": {}}
    inloop_a: dict[str, list[dict]] = {}
    inloop_b: dict[str, list[dict]] = {}
    for source in sources:
        fname = f"{source}_seed{args.seed}_panel_trajectory.json"
        panels["a"][source] = load_panel_trajectory(panel_dir_a / fname)
        panels["b"][source] = load_panel_trajectory(panel_dir_b / fname)
        tname = f"{source}_seed{args.seed}_trajectory.json"
        inloop_a[source] = load_inloop_trajectory(args.arm_a_traj_dir / tname)
        inloop_b[source] = load_inloop_trajectory(args.slab_root / "armB_trajectories" / tname)

    # Optional emission-anchor overlay input (behavioral dots on the heroes).
    emission: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for arm, sub in (("a", "armA"), ("b", "armB")):
        emis_dir = args.slab_root / "emission_anchors" / sub
        if emis_dir.is_dir():
            per_src = {s: load_emission_rates(emis_dir, s) for s in sources}
            per_src = {s: r for s, r in per_src.items() if r}
            if per_src:
                emission[arm] = per_src
    if not emission:
        log.warning(
            "[phase=analyze_load] no emission-anchor JSONs found under %s — "
            "the behavioral overlay is SKIPPED (recorded in the summary JSON).",
            args.slab_root / "emission_anchors",
        )

    warmup_steps = math.ceil(args.warmup_ratio * args.schedule_total_steps)
    h1 = h1_lockstep(panels["b"])
    h2 = h2_suppression(panels["a"])
    h3 = h3_acceleration(
        inloop_a,
        inloop_b,
        schedule_total_steps=args.schedule_total_steps,
        warmup_steps=warmup_steps,
    )
    log.info(
        "[phase=analyze_stats] H1: %s | H2: %s | H3: %s",
        h1["verdict"],
        h2["verdict"],
        h3["verdict"],
    )

    stats = {
        "schema": "i597_h1_h2_h3_v1",
        "issue": 597,
        "sources": sources,
        "qwen_default_no_persona_excluded": "qwen_default" in sources,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "metadata": _metadata(
            {
                "slab_root": str(args.slab_root),
                "arm_a_traj_dir": str(args.arm_a_traj_dir),
                "schedule_total_steps": args.schedule_total_steps,
                "warmup_steps": warmup_steps,
            }
        ),
    }
    stats_path = out_dir / "h1_h2_h3.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    log.info("[phase=analyze_stats] -> %s", stats_path)

    summary = {
        "schema": "i597_trajectory_summary_v1",
        "issue": 597,
        "sources": sources,
        "emission_overlay": sorted(emission) if emission else "skipped (no emission JSONs found)",
        "groups": trajectory_summary(panels),
        "metadata": _metadata(),
    }
    summary_path = out_dir / "trajectory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("[phase=analyze_summary] -> %s", summary_path)

    if not args.skip_figures:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
        figs = [
            fig_hero_trajectory(panels, args.fig_dir, emission=emission or None),
            fig_phase_plot(panels, args.fig_dir),
            fig_hero_trajectory(
                panels,
                args.fig_dir,
                x_mode="cum_pos",
                stem="exploratory_trajectory_cumulative_positives",
            ),
            fig_hero_trajectory(
                panels,
                args.fig_dir,
                y_key="eos_margin_delta",
                stem="exploratory_trajectory_eos_margin",
                y_label="Δ(z_marker − z_eos), trained − base (logits)",
            ),
            fig_phase_plot(
                panels,
                args.fig_dir,
                y_key="eos_margin_delta",
                stem="exploratory_phase_plot_eos_margin",
                axis_label="Δ(z_marker − z_eos) (logits)",
            ),
            fig_small_multiples(panels, args.fig_dir),
            fig_spaghetti(panels["a"], "a", args.fig_dir),
            fig_spaghetti(panels["b"], "b", args.fig_dir),
            fig_raw_traces(panels, args.fig_dir),
            fig_saturation_scatter(panels, args.fig_dir),
            fig_lockstep_curves(h1, args.fig_dir),
        ]
        log.info("[phase=analyze_figures] %d figures -> %s", len(figs), args.fig_dir)

    log.info("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

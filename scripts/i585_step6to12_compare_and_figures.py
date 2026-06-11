#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003  # typographic minus / Greek Δ in labels intentional
"""Task #585 follow-up `step6to12-transition-sweep` — off-pod transition analysis.

Runs on the VM (CPU only; the pod is already terminated) against the committed
JSONs. Implements the plan v3 §6 decision reads EXACTLY as revised after the
round-1 critics (the s_ceil ΔG-plateau-proximity definition and the dual-
threshold read-5 verdict wording are the REVISED forms — not the follow-up
spec's older trained-logP definitions, which were left-censored):

  1. **Validity kill (eval drift):** |same-run Hub step-6 ΔG − the parent
     round's committed corrected value| ≤ 2.0 nats (warn > 0.5). Fail →
     interpret nothing.
  2. **Fix-took check:** 9×9 exact-float-identity rates of held-out ``g_logp``
     show a distance gradient (extreme pair < 0.05), NOT the flat ~0.19–0.27
     stale signature at every distance (parent's rule verbatim; #549
     signature). Retrain-vs-Hub same-step pairs reported descriptively.
  3. **Splice gate:** both retrained endpoints (steps 6 + 12) within 2.0 nats
     of the SAME-RUN Hub endpoint reads. Pass → "endpoint-consistent same-seed
     retrained instance" narration constants; fail → instance-scoped negative
     (falsification (b); re-examine R_train provenance first, A3). Secondary
     descriptive splice diagnostics: retrained-vs-Hub Δz_marker at both
     endpoints (companion pass) + endpoint RESOLUTIONS.
  4. **Transition localization:** s_ceil = first step in {6..12} with source
     mean ΔG within 2.0 nats of the step-12 plateau (≥ plateau − 2.0;
     equivalently ``b_logp_mean`` ≤ −(plateau − 2.0) at pinned trained logP —
     the ΔG jump is BASE-prior/collapse-driven, the trained logP is already
     pinned by step 6). s_coll = first step with bystander resolution < 0.05.
     Coupled ⇔ |s_ceil − s_coll| ≤ 1. Per-step trained-side (g) vs base-side
     (b) ΔG decomposition + collapse-share alignment (decode-collapse-coupled
     measurement transition vs weight-space phase change diagnostics).
  5. **Usable-intermediate-anchor read at BOTH thresholds:** ∃ s ∈ {7..11}
     with source ΔG ∈ [5,12] AND resolution ≥ 0.05 (pre-registered read-line)
     / ≥ 0.2 (the v4 picker's own gate). "Grid-sampling artifact" is claimed
     ONLY when a step clears 0.2; clearing 0.05 only is narrated as "usable at
     a 5% resolution floor, below the picker's 20% gate".
  6. **Saturated-row reading:** per-step ΔlogP-flat-while-Δz-climbs divergence
     flags (descriptive, never "fixed" by re-running in another space).

Pre-consumption key-coverage asserts extend the parent round's
``validate_artifact_coverage`` pattern over the 9-checkpoint merged index, and
the picker-formula reproduction self-check (A12) re-derives the PARENT round's
per-fraction resolutions from the parent trajectory before any per-step value
is computed.

Reference constants (5.4283 / 10.3508 nats, floor 0.5, ceiling −0.10536,
gate_fraction 0.2, parent endpoint resolutions) are READ from the committed
parent artifact ``phase0_calibration_v4_corrected.json`` — never hardcoded.

Reuses the parent comparison script's helpers (``bystander_resolution``,
``cluster_bootstrap_resolution_ci``, ...) so the picker-formula parity is
single-sourced.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Sibling-script import (sys.path[0] = scripts/ when invoked as
# `uv run python scripts/i585_step6to12_compare_and_figures.py`); placed after
# load_dotenv() like every i585 glue module, hence the E402 waiver.
from i585_compare_and_figures import (  # noqa: E402
    CEILING_LOGP_NATS,
    FLOOR_DELTA_G_NATS,
    _git_sha,
    _key_diff,
    _mean,
    _sd,
    bystander_resolution,
    cluster_bootstrap_resolution_ci,
)

log = logging.getLogger("i585.step6to12_compare")

# ── Pre-registered constants (plan v3 §6 / §11 decision-read constants). ─────
PARITY_TOLERANCE_NATS = 2.0  # endpoint-parity + validity-kill tolerance (#534 guard magnitude)
EVAL_DRIFT_WARN_NATS = 0.5  # warn level (≈ vLLM regen noise, A16)
S_CEIL_PROXIMITY_NATS = 2.0  # plateau-arrival proximity (reuses the parity magnitude)
RESOLUTION_READ_LINE = 0.05  # pre-registered read-line = 27/540
SOURCE_DG_BAND = (5.0, 12.0)  # marker-training-recipe band
COUPLED_MAX_GAP_STEPS = 1
# Fix-took constants — parent's rule verbatim (i585_compare_and_figures).
STALE_BAND = (0.10, 0.40)
FIX_TOOK_EXTREME_MAX = 0.05
WINDOW_STEPS = tuple(range(6, 13))
INTERMEDIATE_STEPS = tuple(range(7, 12))
# The 9 merged-index keys this round's design pins (plan §4.2 Step 1.2):
# numeric sort = step order; 4-dp keys = retrained, 2-dp = Hub endpoints.
EXPECTED_INDEX: dict[str, tuple[int, str]] = {
    "0.0733": (6, "retrain"),
    "0.08": (6, "hub"),
    "0.0867": (7, "retrain"),
    "0.1000": (8, "retrain"),
    "0.1133": (9, "retrain"),
    "0.1267": (10, "retrain"),
    "0.1400": (11, "retrain"),
    "0.1533": (12, "retrain"),
    "0.16": (12, "hub"),
}
PARENT_FRACS = (0.08, 0.16, 0.33, 0.50, 0.75, 1.00)
PARENT_MAX_STEPS = 75


def _first_crossing_step(frac: float, max_steps: int) -> int:
    """First step s with s/max_steps >= frac (CheckpointAtFractionsCallback semantics)."""
    for s in range(1, max_steps + 1):
        if s / max_steps >= frac:
            return s
    raise ValueError(f"fraction {frac} never crosses within max_steps={max_steps}")


def _label(key: str, provenance: dict[str, dict]) -> str:
    info = provenance[key]
    return f"s{info['step']}·{'retr' if info['provenance'] == 'retrain' else 'hub'}"


def validate_inputs(
    trajectory: dict,
    slot_stats: dict,
    provenance: dict,
    parent_corrected: dict,
    parent_trajectory: dict,
) -> None:
    """Pre-consumption key-coverage gate — extends the parent round's
    ``validate_artifact_coverage`` pattern over the 9-checkpoint index. Runs
    BEFORE any rates/verdicts; failures NAME the artifact + missing/extra keys.
    """
    problems: list[str] = []
    expected_keys = set(EXPECTED_INDEX)
    d = _key_diff("index_provenance keys", set(provenance), expected_keys)
    if d:
        problems.append(d)
    else:
        for key, (step, prov) in EXPECTED_INDEX.items():
            got = provenance[key]
            if int(got["step"]) != step or got["provenance"] != prov:
                problems.append(
                    f"index_provenance[{key!r}] = {got} != expected "
                    f"{{'step': {step}, 'provenance': {prov!r}}}"
                )
    expected_floats = {float(k) for k in EXPECTED_INDEX}
    traj_fracs = {float(ck["frac"]) for ck in trajectory["checkpoints"]}
    d = _key_diff("trajectory checkpoints (frac floats)", traj_fracs, expected_floats)
    if d:
        problems.append(d)
    slot_fracs = {float(fr["frac"]) for fr in slot_stats["fractions"]}
    d = _key_diff("slot_stats fractions (frac floats)", slot_fracs, expected_floats)
    if d:
        problems.append(d)

    # Held-out (persona, question) panel identical across ALL 9 checkpoints,
    # both directions (a strict-superset panel at one checkpoint would let the
    # identity-rate loop silently iterate only the other's keys).
    cks = sorted(trajectory["checkpoints"], key=lambda c: float(c["frac"]))
    if cks:
        ref = cks[0]
        ref_panel = {(p, q) for p, per_q in ref["held_out"].items() for q in per_q}
        for ck in cks[1:]:
            panel = {(p, q) for p, per_q in ck["held_out"].items() for q in per_q}
            d = _key_diff(
                f"trajectory held-out panel at frac={ck['frac']} (vs frac={ref['frac']})",
                panel,
                ref_panel,
            )
            if d:
                problems.append(d)
    # Slot-stats per-question key set identical across fractions.
    frs = sorted(slot_stats["fractions"], key=lambda fr: float(fr["frac"]))
    if frs:
        ref_qs = set(frs[0]["per_question"])
        for fr in frs[1:]:
            d = _key_diff(
                f"slot_stats per_question keys at frac={fr['frac']} (vs {frs[0]['frac']})",
                set(fr["per_question"]),
                ref_qs,
            )
            if d:
                problems.append(d)

    problems.extend(_parent_input_problems(parent_corrected, parent_trajectory))

    if problems:
        raise AssertionError(
            "artifact key-coverage validation FAILED (checked before any rates/verdicts):\n  "
            + "\n  ".join(problems)
        )


def _parent_input_problems(parent_corrected: dict, parent_trajectory: dict) -> list[str]:
    """Parent-side coverage + constant-parity checks (helper of validate_inputs)."""
    problems: list[str] = []
    # Parent artifacts: exactly the 6 nominal fractions.
    parent_table_fracs = {float(r["ckpt_frac"]) for r in parent_corrected["smoke_table"]}
    d = _key_diff("parent corrected smoke_table fractions", parent_table_fracs, set(PARENT_FRACS))
    if d:
        problems.append(d)
    parent_traj_fracs = {float(ck["frac"]) for ck in parent_trajectory["checkpoints"]}
    d = _key_diff("parent trajectory fractions", parent_traj_fracs, set(PARENT_FRACS))
    if d:
        problems.append(d)
    # The picker constants in the committed parent artifact must equal the
    # code-side constants the imported resolution helper uses.
    if abs(float(parent_corrected["floor_delta_g"]) - FLOOR_DELTA_G_NATS) > 1e-12:
        problems.append(
            f"parent floor_delta_g={parent_corrected['floor_delta_g']} != code {FLOOR_DELTA_G_NATS}"
        )
    if abs(float(parent_corrected["ceiling_logp"]) - CEILING_LOGP_NATS) > 1e-9:
        problems.append(
            f"parent ceiling_logp={parent_corrected['ceiling_logp']} != code {CEILING_LOGP_NATS}"
        )
    return problems


def parent_resolution_self_check(parent_corrected: dict, parent_trajectory: dict) -> dict:
    """A12: re-derive the parent round's per-fraction resolutions from the
    parent trajectory with the SAME imported picker formula, and assert they
    reproduce the committed table EXACTLY — the formula-parity license for
    every per-step resolution computed below."""
    table = {float(r["ckpt_frac"]): r for r in parent_corrected["smoke_table"]}
    out: dict[str, dict] = {}
    for ck in sorted(parent_trajectory["checkpoints"], key=lambda c: float(c["frac"])):
        f = float(ck["frac"])
        res, n_in, n_tot = bystander_resolution(ck["held_out"])
        published = float(table[f]["bystander_resolution"])
        if abs(res - published) > 1e-9:
            raise AssertionError(
                f"A12 self-check FAILED at parent frac {f}: recomputed resolution "
                f"{res} != published {published} — picker-formula drift; do NOT "
                f"trust the per-step resolutions."
            )
        out[f"{f:.2f}"] = {"recomputed": res, "published": published, "n": [n_in, n_tot]}
    return out


def route_verdict(
    *,
    validity_pass: bool,
    stale_signature: bool,
    fix_took: bool,
    splice_pass: bool,
    s_ceil: int | None,
    s_coll: int | None,
    anchor_at_picker_gate: bool,
    anchor_at_read_line: bool,
) -> str:
    """Plan §7 verdict routing (encoded; the analyzer owns interpretation)."""
    if not validity_pass:
        return "validity_kill_eval_drift"
    if stale_signature or not fix_took:
        # §6 read 2 — an INFRA outcome (residual stale serving), checked before
        # any splice/transition interpretation; never a finding.
        return "residual_stale_serving_infra"
    if not splice_pass:
        # Falsification (b): instance-scoped negative — the curve characterizes
        # only the retrained instance; re-examine R_train provenance first (A3).
        return "splice_failed_instance_scoped_negative"
    if s_coll is None:
        # Resolution never drops below the read-line in the retrained window:
        # report descriptively as a splice-scoped caveat, do NOT force the
        # co-occurrence read (plan §6 read 5 tail clause).
        return "s_coll_undefined_descriptive"
    if s_ceil is None:
        return "s_ceil_undefined_descriptive"
    if abs(s_ceil - s_coll) <= COUPLED_MAX_GAP_STEPS:
        if anchor_at_picker_gate:
            return "h1_confirmed_coupled_anchor_clears_picker_gate"
        if anchor_at_read_line:
            return "h1_confirmed_coupled_anchor_at_read_line_only"
        return "coupled_no_intermediate_anchor_single_anchor_verdict_real"
    return "decoupled_transition_falsification_a"


def compute_analysis(
    trajectory: dict,
    slot_stats: dict,
    provenance: dict,
    parent_corrected: dict,
    parent_trajectory: dict,
    n_boot: int,
    boot_seed: int,
) -> dict:
    """All plan §6 reads over the 9-checkpoint merged trajectory."""
    validate_inputs(trajectory, slot_stats, provenance, parent_corrected, parent_trajectory)
    a12 = parent_resolution_self_check(parent_corrected, parent_trajectory)

    keys = sorted(EXPECTED_INDEX, key=float)
    ck_by_key = {
        k: next(ck for ck in trajectory["checkpoints"] if float(ck["frac"]) == float(k))
        for k in keys
    }
    slot_by_key = {
        k: next(fr for fr in slot_stats["fractions"] if float(fr["frac"]) == float(k)) for k in keys
    }
    retr_key_by_step = {EXPECTED_INDEX[k][0]: k for k in keys if EXPECTED_INDEX[k][1] == "retrain"}
    hub_key_by_step = {EXPECTED_INDEX[k][0]: k for k in keys if EXPECTED_INDEX[k][1] == "hub"}

    parent_table = {float(r["ckpt_frac"]): r for r in parent_corrected["smoke_table"]}
    ref_step6 = float(parent_table[0.08]["source_dg"])  # 5.4283 (committed)
    plateau = float(parent_table[0.16]["source_dg"])  # 10.3508 (committed)
    gate_fraction = float(parent_corrected["gate_fraction"])  # 0.2 (v4 picker)
    parent_res6 = float(parent_table[0.08]["bystander_resolution"])
    parent_res12 = float(parent_table[0.16]["bystander_resolution"])

    def src_dg(key: str) -> float:
        return float(ck_by_key[key]["source_self"]["delta_g_mean"])

    # ── Per-checkpoint table (all 9, in step order). ──────────────────────────
    per_checkpoint: list[dict] = []
    res_by_key: dict[str, float] = {}
    for k in keys:
        ck = ck_by_key[k]
        src = ck["source_self"]
        res, n_in, n_tot = bystander_resolution(ck["held_out"])
        ci_lo, ci_hi = cluster_bootstrap_resolution_ci(ck["held_out"], n_boot, boot_seed)
        res_by_key[k] = res
        leaves = [leaf for per_q in ck["held_out"].values() for leaf in per_q.values()]
        if any(leaf.get("delta_z_marker") is None for leaf in leaves):
            raise AssertionError(
                f"checkpoint {k}: held-out slot stats missing (delta_z_marker is None) "
                f"— was the eval run with KL off? The plan requires KL ON."
            )
        glue = slot_by_key[k]
        glue_dgs = [rec["delta_g"] for rec in glue["per_question"].values()]
        glue_dz = [rec["delta_z_marker"] for rec in glue["per_question"].values()]
        glue_margin = [rec["delta_eos_margin"] for rec in glue["per_question"].values()]
        per_checkpoint.append(
            {
                "key": k,
                "step": EXPECTED_INDEX[k][0],
                "provenance": EXPECTED_INDEX[k][1],
                "source_dg": src_dg(k),
                "source_g_logp_mean": float(src["g_logp_mean"]),
                "source_b_logp_mean": float(src["b_logp_mean"]),
                "source_emission_p": float(src["emission_p"]),
                "source_r_collapsed": bool(src["r_collapsed"]),
                "bystander_resolution": res,
                "bystander_resolution_ci95": [ci_lo, ci_hi],
                "bystander_resolution_n": [n_in, n_tot],
                "held_out_collapse_share": float(ck["held_out_collapse_share"]),
                "held_out_mean_delta_g": _mean([float(lf["delta_g"]) for lf in leaves]),
                "held_out_mean_delta_z_marker": _mean(
                    [float(lf["delta_z_marker"]) for lf in leaves]
                ),
                "held_out_mean_eos_margin": _mean([float(lf["delta_z_margin"]) for lf in leaves]),
                "bystander_emission_p": _mean(
                    [1.0 if lf["argmax_marker"] else 0.0 for lf in leaves]
                ),
                "glue_source_delta_g_mean": _mean(glue_dgs),
                "glue_source_delta_g_sd": _sd(glue_dgs),
                "glue_source_delta_z_marker_mean": _mean(glue_dz),
                "glue_source_delta_z_marker_sd": _sd(glue_dz),
                "glue_source_eos_margin_mean": _mean(glue_margin),
                "glue_source_eos_margin_sd": _sd(glue_margin),
                "glue_vs_main_abs_diff": abs(_mean(glue_dgs) - src_dg(k)),
            }
        )
    by_key = {row["key"]: row for row in per_checkpoint}

    # ── Read 1: validity kill (eval drift on the Hub step-6 re-read). ─────────
    hub6_dg = src_dg(hub_key_by_step[6])
    hub12_dg = src_dg(hub_key_by_step[12])
    validity_abs = abs(hub6_dg - ref_step6)
    validity_pass = validity_abs <= PARITY_TOLERANCE_NATS
    validity = {
        "hub_step6_dg": hub6_dg,
        "parent_committed_dg": ref_step6,
        "abs_diff": validity_abs,
        "tolerance_nats": PARITY_TOLERANCE_NATS,
        "warn_nats": EVAL_DRIFT_WARN_NATS,
        "warn": validity_abs > EVAL_DRIFT_WARN_NATS,
        "pass": validity_pass,
        # Descriptive only (read 1 names step 6; step 12 is the plateau leg).
        "hub_step12_dg": hub12_dg,
        "hub_step12_vs_parent_abs_diff": abs(hub12_dg - plateau),
    }

    # ── Read 2: fix-took (9×9 float-identity; parent's rule verbatim). ────────
    identity_rates: dict[str, float] = {}
    for ka, kb in itertools.combinations(keys, 2):
        ho_a, ho_b = ck_by_key[ka]["held_out"], ck_by_key[kb]["held_out"]
        n_same = n_total = 0
        for persona, per_q in ho_a.items():
            for q, leaf in per_q.items():
                n_total += 1
                if float(leaf["g_logp"]) == float(ho_b[persona][q]["g_logp"]):
                    n_same += 1
        identity_rates[f"{ka}__{kb}"] = n_same / n_total if n_total else 0.0
    all_rates = list(identity_rates.values())
    extreme_pair_rate = identity_rates[f"{keys[0]}__{keys[-1]}"]
    stale_signature = all(STALE_BAND[0] <= r <= STALE_BAND[1] for r in all_rates)
    fix_took = (extreme_pair_rate < FIX_TOOK_EXTREME_MAX) and not stale_signature
    fix_took_block = {
        "pair_rates": identity_rates,
        "extreme_pair": f"{keys[0]}__{keys[-1]}",
        "extreme_pair_rate": extreme_pair_rate,
        "stale_signature_all_pairs_flat_0p10_0p40": stale_signature,
        "fix_took": fix_took,
        # Informational, NOT a gate: near-zero identity expected even under
        # successful parity (CUDA nondeterminism).
        "retrain_vs_hub_same_step_pairs": {
            "step6": identity_rates[f"{retr_key_by_step[6]}__{hub_key_by_step[6]}"],
            "step12": identity_rates[f"{retr_key_by_step[12]}__{hub_key_by_step[12]}"],
        },
    }

    # ── Read 3: splice gate (both endpoints, same eval batch). ────────────────
    retr6_dg, retr12_dg = src_dg(retr_key_by_step[6]), src_dg(retr_key_by_step[12])
    diff6 = abs(retr6_dg - hub6_dg)
    diff12 = abs(retr12_dg - hub12_dg)
    splice_pass = diff6 <= PARITY_TOLERANCE_NATS and diff12 <= PARITY_TOLERANCE_NATS
    splice = {
        "endpoint_step6": {"retrained_dg": retr6_dg, "hub_dg": hub6_dg, "abs_diff": diff6},
        "endpoint_step12": {"retrained_dg": retr12_dg, "hub_dg": hub12_dg, "abs_diff": diff12},
        "tolerance_nats": PARITY_TOLERANCE_NATS,
        "pass": splice_pass,
        # Secondary descriptive splice diagnostics (§6 read 3 revised):
        # (a) Δz_marker at both endpoints (the step-12 leg is information-poor
        # in ΔG space at ceiling — the logit carries the saturated end);
        "delta_z_marker_companion": {
            "step6": {
                "retrained": by_key[retr_key_by_step[6]]["glue_source_delta_z_marker_mean"],
                "hub": by_key[hub_key_by_step[6]]["glue_source_delta_z_marker_mean"],
            },
            "step12": {
                "retrained": by_key[retr_key_by_step[12]]["glue_source_delta_z_marker_mean"],
                "hub": by_key[hub_key_by_step[12]]["glue_source_delta_z_marker_mean"],
            },
        },
        # (b) endpoint RESOLUTIONS, checked descriptively BEFORE any
        # bystander-side localization is narrated.
        "endpoint_resolutions": {
            "step6": {
                "retrained": res_by_key[retr_key_by_step[6]],
                "hub_same_run": res_by_key[hub_key_by_step[6]],
                "parent_committed": parent_res6,
            },
            "step12": {
                "retrained": res_by_key[retr_key_by_step[12]],
                "hub_same_run": res_by_key[hub_key_by_step[12]],
                "parent_committed": parent_res12,
            },
        },
        "narration_constants": (
            {
                "scope": "endpoint-consistent same-seed retrained instance",
                "caveat": (
                    "endpoint parity does not identify the interior trajectory; the "
                    "per-step curve is narrated as an endpoint-consistent same-seed "
                    "retrained instance of the original cell — relevant to and "
                    "consistent with the original, NEVER claimed AS the original "
                    "run's lost interior. Recipe-level conclusions (sharp-vs-graded, "
                    "coupled-vs-decoupled, anchor existence at step granularity) are "
                    "carried by the retrained snapshots themselves, which are "
                    "uploaded and directly usable as anchors regardless of "
                    "attribution."
                ),
            }
            if splice_pass
            else {
                "scope": "retrained instance only",
                "caveat": (
                    "instance-scoped negative: THIS seed-pinned retrain did not "
                    "reproduce these snapshots' values; input provenance (A3) not "
                    "fully excluded — re-examine R_train provenance first. Flags "
                    "that retrain-based corrections need splice controls."
                ),
            }
        ),
    }

    # ── Read 4: transition localization (the headline; retrained 7 steps). ────
    dg_by_step = {s: src_dg(retr_key_by_step[s]) for s in WINDOW_STEPS}
    res_by_step = {s: res_by_key[retr_key_by_step[s]] for s in WINDOW_STEPS}
    s_ceil = next(
        (s for s in WINDOW_STEPS if dg_by_step[s] >= plateau - S_CEIL_PROXIMITY_NATS), None
    )
    s_coll = next((s for s in WINDOW_STEPS if res_by_step[s] < RESOLUTION_READ_LINE), None)
    coupled = (
        s_ceil is not None and s_coll is not None and abs(s_ceil - s_coll) <= COUPLED_MAX_GAP_STEPS
    )
    # Graded-climb qualifier for the decoupled branch (descriptive).
    graded_climb_between = None
    if s_ceil is not None and s_coll is not None and abs(s_ceil - s_coll) >= 2:
        lo, hi = sorted((s_ceil, s_coll))
        n_increasing = sum(1 for s in range(lo, hi) if dg_by_step[s + 1] > dg_by_step[s])
        graded_climb_between = {"from_step": lo, "to_step": hi, "n_increasing_steps": n_increasing}
    # Mechanistic companion: source Δz_marker localization (largest jump step).
    dz_by_step = {
        s: by_key[retr_key_by_step[s]]["glue_source_delta_z_marker_mean"] for s in WINDOW_STEPS
    }
    dz_jumps = {s: dz_by_step[s + 1] - dz_by_step[s] for s in WINDOW_STEPS[:-1]}
    # Per-step g-vs-b decomposition + collapse alignment (round-1 alternatives
    # concern: decode-collapse-coupled measurement transition vs weight-space
    # phase change — all inputs persisted).
    decomposition = []
    for s in WINDOW_STEPS[:-1]:
        a, b = by_key[retr_key_by_step[s]], by_key[retr_key_by_step[s + 1]]
        decomposition.append(
            {
                "from_step": s,
                "to_step": s + 1,
                "d_delta_g": b["source_dg"] - a["source_dg"],
                "d_g_logp_trained_side": b["source_g_logp_mean"] - a["source_g_logp_mean"],
                "d_b_logp_base_side": b["source_b_logp_mean"] - a["source_b_logp_mean"],
                "d_held_out_collapse_share": (
                    b["held_out_collapse_share"] - a["held_out_collapse_share"]
                ),
                "d_bystander_resolution": (b["bystander_resolution"] - a["bystander_resolution"]),
            }
        )
    localization = {
        "plateau_reference_dg": plateau,
        "s_ceil_definition": (
            f"first step in {{6..12}} with source mean ΔG >= plateau − "
            f"{S_CEIL_PROXIMITY_NATS} nats (revised per round-1 critics: ΔG-jump "
            f"arrival, base-prior/collapse-driven; the prior trained-logP "
            f"definition was left-censored — trained logP is pinned by step 6)"
        ),
        "s_ceil": s_ceil,
        "s_coll_definition": (
            f"first step in {{6..12}} with bystander resolution < {RESOLUTION_READ_LINE}"
        ),
        "s_coll": s_coll,
        "coupled": coupled,
        "gap_steps": (abs(s_ceil - s_coll) if s_ceil is not None and s_coll is not None else None),
        "graded_climb_between": graded_climb_between,
        "source_dg_by_step": dg_by_step,
        "bystander_resolution_by_step": res_by_step,
        "source_delta_z_marker_by_step": dz_by_step,
        "source_delta_z_marker_jumps": dz_jumps,
        "largest_dz_jump_at_step_pair": (
            max(dz_jumps, key=lambda s: dz_jumps[s]) if dz_jumps else None
        ),
        "per_step_g_vs_b_decomposition": decomposition,
    }

    # ── Read 5: usable-intermediate-anchor read at BOTH thresholds. ───────────
    anchor_rows = []
    for s in INTERMEDIATE_STEPS:
        dg, res = dg_by_step[s], res_by_step[s]
        in_band = SOURCE_DG_BAND[0] <= dg <= SOURCE_DG_BAND[1]
        anchor_rows.append(
            {
                "step": s,
                "source_dg": dg,
                "in_band_5_12": in_band,
                "bystander_resolution": res,
                "usable_at_read_line_0p05": in_band and res >= RESOLUTION_READ_LINE,
                "usable_at_picker_gate_0p2": in_band and res >= gate_fraction,
            }
        )
    steps_at_read_line = [r["step"] for r in anchor_rows if r["usable_at_read_line_0p05"]]
    steps_at_picker_gate = [r["step"] for r in anchor_rows if r["usable_at_picker_gate_0p2"]]
    anchor_at_read_line = bool(steps_at_read_line)
    anchor_at_picker_gate = bool(steps_at_picker_gate)
    if anchor_at_picker_gate:
        anchor_wording = (
            "grid_sampling_artifact_confirmed: a step in {7..11} clears the v4 "
            "picker's own 0.2 gate — the corrected picker's single-anchor verdict "
            "was a grid-sampling artifact of the 6-fraction grid."
        )
    elif anchor_at_read_line:
        anchor_wording = (
            "usable at a 5% resolution floor, below the picker's 20% gate — the "
            "grid-sampling-artifact claim is NOT licensed."
        )
    else:
        anchor_wording = (
            "no qualifying step at the 0.05 read-line: the single-anchor verdict "
            "is real at step granularity (clean negative; reportable)."
        )
    anchor = {
        "per_step": anchor_rows,
        "read_line": RESOLUTION_READ_LINE,
        "picker_gate_fraction": gate_fraction,
        "steps_usable_at_read_line": steps_at_read_line,
        "steps_usable_at_picker_gate": steps_at_picker_gate,
        "verdict_wording": anchor_wording,
    }

    # ── Read 6: saturated-row reading (descriptive divergence flags). ─────────
    saturation_rows = []
    for s in WINDOW_STEPS[:-1]:
        d_dg = dg_by_step[s + 1] - dg_by_step[s]
        d_dz = dz_by_step[s + 1] - dz_by_step[s]
        saturation_rows.append(
            {
                "from_step": s,
                "to_step": s + 1,
                "d_delta_g": d_dg,
                "d_delta_z_marker": d_dz,
                "divergence_dz_minus_dg": d_dz - d_dg,
                # Descriptive flag ONLY (no pre-registered logit-unit gate): a
                # flat ΔlogP move while Δz climbs is the saturation signature —
                # read the logit there, never raw log P; never "fixed" by
                # re-running in another space.
                "descriptive_saturation_divergence": (d_dz - d_dg) > 1.0 and d_dz > 0,
            }
        )

    verdict = route_verdict(
        validity_pass=validity_pass,
        stale_signature=stale_signature,
        fix_took=fix_took,
        splice_pass=splice_pass,
        s_ceil=s_ceil,
        s_coll=s_coll,
        anchor_at_picker_gate=anchor_at_picker_gate,
        anchor_at_read_line=anchor_at_read_line,
    )

    return {
        "schema_version": "i585_step6to12_transition_analysis_v1",
        "task": 585,
        "followup_label": "step6to12-transition-sweep",
        "constants": {
            "parity_tolerance_nats": PARITY_TOLERANCE_NATS,
            "eval_drift_warn_nats": EVAL_DRIFT_WARN_NATS,
            "s_ceil_proximity_nats": S_CEIL_PROXIMITY_NATS,
            "resolution_read_line": RESOLUTION_READ_LINE,
            "source_dg_band": list(SOURCE_DG_BAND),
            "coupled_max_gap_steps": COUPLED_MAX_GAP_STEPS,
            "plateau_reference_dg": plateau,
            "parent_step6_reference_dg": ref_step6,
            "picker_floor_delta_g": FLOOR_DELTA_G_NATS,
            "picker_ceiling_logp": CEILING_LOGP_NATS,
            "picker_gate_fraction": gate_fraction,
        },
        "a12_parent_resolution_self_check": a12,
        "per_checkpoint": per_checkpoint,
        "read1_validity_kill": validity,
        "read2_fix_took": fix_took_block,
        "read3_splice_gate": splice,
        "read4_transition_localization": localization,
        "read5_usable_anchor": anchor,
        "read6_saturation_divergence": saturation_rows,
        "verdict": verdict,
    }


# ── Figures (plan §6 "Figures to produce"). ──────────────────────────────────


def make_figures(
    analysis: dict, trajectory: dict, parent_corrected: dict, fig_dir: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    c_primary = paper_palette_role("primary")
    c_baseline = paper_palette_role("baseline")
    c_accent = paper_palette_role("accent")
    c_neutral = paper_palette_role("neutral")
    c_control = paper_palette_role("control")

    rows = analysis["per_checkpoint"]
    retr = [r for r in rows if r["provenance"] == "retrain"]
    hub = [r for r in rows if r["provenance"] == "hub"]
    retr_steps = [r["step"] for r in retr]
    hub_steps = [r["step"] for r in hub]
    labels = [
        f"step {r['step']} {'retrain' if r['provenance'] == 'retrain' else 'original Hub'}"
        for r in rows
    ]
    written: list[str] = []

    def _save(fig, stem: str) -> None:
        savefig_paper(fig, stem, dir=fig_dir)
        plt.close(fig)
        written.append(stem)

    # (hero) per-step source curve, Hub endpoint reads, parent curve as context.
    parent_table = sorted(parent_corrected["smoke_table"], key=lambda r: float(r["ckpt_frac"]))
    parent_steps = [
        _first_crossing_step(float(r["ckpt_frac"]), PARENT_MAX_STEPS) for r in parent_table
    ]
    parent_dgs = [float(r["source_dg"]) for r in parent_table]
    fig, ax = plt.subplots()
    ax.plot(
        parent_steps,
        parent_dgs,
        "--s",
        color=c_baseline,
        alpha=0.7,
        label="Parent corrected 6-fraction curve (fraction axis mapped to steps)",
    )
    ax.plot(
        retr_steps,
        [r["source_dg"] for r in retr],
        "-o",
        color=c_primary,
        label="Per-step retrained snapshots (steps 6–12)",
    )
    ax.errorbar(
        hub_steps,
        [r["source_dg"] for r in hub],
        yerr=analysis["constants"]["parity_tolerance_nats"],
        fmt="D",
        markerfacecolor="none",
        markeredgewidth=1.4,
        color=c_accent,
        capsize=4,
        label="Original Hub snapshots, same eval batch (±2-nat parity tolerance)",
    )
    ax.set_xscale("log")
    xticks = sorted(set(retr_steps + parent_steps))
    ax.set_xticks(xticks, [str(s) for s in xticks])
    ax.minorticks_off()
    ax.set_xlabel("Optimizer step (log scale)")
    ax.set_ylabel("Source implant strength (nats, trained − base)")
    ax.set_title("Per-step source implant strength across the step-6→12 window")
    ax.legend(fontsize=8)
    _save(fig, "hero_per_step_transition_curve")

    # (a) bystander resolution vs step with cluster-bootstrap CIs + read-lines.
    fig, ax = plt.subplots()
    res = [r["bystander_resolution"] for r in retr]
    lo = [max(0.0, r["bystander_resolution"] - r["bystander_resolution_ci95"][0]) for r in retr]
    hi = [max(0.0, r["bystander_resolution_ci95"][1] - r["bystander_resolution"]) for r in retr]
    ax.errorbar(
        retr_steps,
        res,
        yerr=[lo, hi],
        fmt="-o",
        color=c_primary,
        capsize=3,
        label="Retrained (95% CI, bootstrap clustered by persona)",
    )
    ax.scatter(
        hub_steps,
        [r["bystander_resolution"] for r in hub],
        marker="D",
        facecolors="none",
        edgecolors=c_accent,
        linewidths=1.6,
        s=60,
        zorder=5,
        label="Original Hub snapshots (same eval batch)",
    )
    ax.axhline(
        analysis["constants"]["resolution_read_line"],
        color=c_baseline,
        linestyle="--",
        label="0.05 headroom floor (set before the run)",
    )
    ax.axhline(
        analysis["constants"]["picker_gate_fraction"],
        color=c_control,
        linestyle=":",
        label="Calibration picker gate (0.2)",
    )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Bystander resolution (share of probes in band)")
    ax.set_title("Bystander measurement headroom per step")
    ax.legend(fontsize=8)
    _save(fig, "bystander_resolution_vs_step")

    # (b) per-checkpoint held-out delta-G distributions (540 leaves each).
    fig, ax = plt.subplots(figsize=(9, 4))
    cks = {float(ck["frac"]): ck for ck in trajectory["checkpoints"]}
    data = []
    for r in rows:
        ck = cks[float(r["key"])]
        leaves = [leaf for per_q in ck["held_out"].values() for leaf in per_q.values()]
        data.append([float(leaf["delta_g"]) for leaf in leaves])
    parts = ax.violinplot(data, positions=range(len(rows)), showmedians=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor(c_primary)
        body.set_alpha(0.5)
    ax.set_xticks(range(len(rows)), labels, rotation=45, ha="right")
    ax.set_xlabel("Checkpoint (step · provenance)")
    ax.set_ylabel("Held-out probe shift (nats, trained − base)")
    ax.set_title("Held-out marker log-prob shift distributions (54 personas × 10 questions)")
    _save(fig, "held_out_delta_g_distributions")

    # (c) emission + collapse vs step.
    fig, ax = plt.subplots()
    ax.plot(
        retr_steps,
        [r["source_emission_p"] for r in retr],
        "-o",
        color=c_primary,
        label="Source emission (retrained)",
    )
    ax.plot(
        retr_steps,
        [r["bystander_emission_p"] for r in retr],
        "-s",
        color=c_accent,
        label="Bystander emission (retrained)",
    )
    ax.plot(
        retr_steps,
        [r["held_out_collapse_share"] for r in retr],
        "-^",
        color=c_control,
        label="Held-out collapse share (retrained)",
    )
    ax.plot(
        retr_steps,
        [1.0 if r["source_r_collapsed"] else 0.0 for r in retr],
        "--",
        color=c_neutral,
        label="Source response collapsed (0/1)",
    )
    ax.scatter(
        hub_steps,
        [r["bystander_emission_p"] for r in hub],
        marker="D",
        facecolors="none",
        edgecolors=c_accent,
        linewidths=1.6,
        s=60,
        zorder=5,
        label="Bystander emission (original Hub snapshots)",
    )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Marker emission and response collapse per step")
    ax.legend(fontsize=8)
    _save(fig, "emission_collapse_vs_step")

    # (d) saturation-signature panel: three spaces, source companion (mean ± SD).
    fig, ax = plt.subplots()
    for field, sd_field, color, lbl, fmt in (
        (
            "glue_source_delta_g_mean",
            "glue_source_delta_g_sd",
            c_primary,
            "Log-prob shift (ΔG)",
            "-o",
        ),
        (
            "glue_source_delta_z_marker_mean",
            "glue_source_delta_z_marker_sd",
            c_accent,
            "Marker-logit shift (Δz)",
            "-s",
        ),
        (
            "glue_source_eos_margin_mean",
            "glue_source_eos_margin_sd",
            c_control,
            "EOS-margin shift",
            "-^",
        ),
    ):
        ax.errorbar(
            retr_steps,
            [r[field] for r in retr],
            yerr=[r[sd_field] for r in retr],
            fmt=fmt,
            color=color,
            capsize=3,
            label=f"{lbl} (mean ± SD, 10 questions)",
        )
        ax.scatter(
            hub_steps,
            [r[field] for r in hub],
            marker="D",
            facecolors="none",
            edgecolors=color,
            linewidths=1.4,
            s=55,
            zorder=5,
        )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Shift (nats / logits, trained − base)")
    ax.set_title("Source (villain) three-space companion readouts per step")
    ax.legend(fontsize=8)
    _save(fig, "saturation_signature_panel")

    # (e) 9×9 float-identity heatmap with provenance-labeled axes.
    fig, ax = plt.subplots(figsize=(7, 6))
    n = len(rows)
    mat = np.full((n, n), np.nan)
    key_order = [r["key"] for r in rows]
    for pair_key, rate in analysis["read2_fix_took"]["pair_rates"].items():
        a, b = pair_key.split("__")
        ia, ib = key_order.index(a), key_order.index(b)
        mat[ia, ib] = rate
        mat[ib, ia] = rate
    for i in range(n):
        mat[i, i] = 1.0
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if mat[i, j] < 0.6 else "black",
                fontsize=7,
            )
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_title("Exact-float-identity rate of held-out log-probs between checkpoints")
    fig.colorbar(im, ax=ax, label="Identity rate")
    _save(fig, "float_identity_heatmap")

    # (f) companion-vs-main source delta-G cross-check per checkpoint.
    fig, ax = plt.subplots()
    ax.bar(labels, [r["glue_vs_main_abs_diff"] for r in rows], color=c_neutral)
    ax.axhline(
        2.0, color=c_baseline, linestyle="--", label="Expected combined noise bound (2 nats)"
    )
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_xlabel("Checkpoint (step · provenance)")
    ax.set_ylabel("|companion − main| source shift (nats)")
    ax.set_title("Companion-pass vs main-run source shift cross-check")
    ax.legend(fontsize=8)
    _save(fig, "glue_vs_main_crosscheck")

    return written


def main(argv: list[str] | None = None) -> int:
    slab = Path("eval_results/issue_585/step6to12-transition-sweep")
    ap = argparse.ArgumentParser(
        description=(
            "Task #585 step6to12: off-pod transition analysis over the 9-checkpoint "
            "merged trajectory (plan v3 §6 reads), + figures."
        )
    )
    ap.add_argument(
        "--trajectory",
        type=Path,
        default=slab / "c504v4_smoke_eps3_step6to12_seed42" / "trajectory.json",
    )
    ap.add_argument("--slot-stats", type=Path, default=slab / "source_slot_stats.json")
    ap.add_argument("--provenance", type=Path, default=slab / "index_provenance.json")
    ap.add_argument(
        "--parent-corrected",
        type=Path,
        default=Path("eval_results/issue_585/phase0_calibration_v4_corrected.json"),
    )
    ap.add_argument(
        "--parent-trajectory",
        type=Path,
        default=Path("eval_results/issue_585/c504v4_smoke_eps3_reread_seed42/trajectory.json"),
    )
    ap.add_argument("--out-json", type=Path, default=slab / "transition_analysis.json")
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_585/step6to12_transition"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--boot-seed", type=int, default=585)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=step6to12_compare] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    trajectory = json.loads(args.trajectory.read_text())
    slot_stats = json.loads(args.slot_stats.read_text())
    provenance = json.loads(args.provenance.read_text())
    parent_corrected = json.loads(args.parent_corrected.read_text())
    parent_trajectory = json.loads(args.parent_trajectory.read_text())

    analysis = compute_analysis(
        trajectory,
        slot_stats,
        provenance,
        parent_corrected,
        parent_trajectory,
        args.n_boot,
        args.boot_seed,
    )
    figures = make_figures(analysis, trajectory, parent_corrected, args.fig_dir)

    analysis["reproducibility"] = {
        "inputs": {
            "trajectory": str(args.trajectory),
            "slot_stats": str(args.slot_stats),
            "provenance": str(args.provenance),
            "parent_corrected": str(args.parent_corrected),
            "parent_trajectory": str(args.parent_trajectory),
        },
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "figures": figures,
        "fig_dir": str(args.fig_dir),
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(analysis, indent=2))
    loc = analysis["read4_transition_localization"]
    log.info(
        "[phase=compare_done] verdict=%s | validity=%s splice=%s fix_took=%s | "
        "s_ceil=%s s_coll=%s | anchors@0.05=%s @0.2=%s -> %s",
        analysis["verdict"],
        analysis["read1_validity_kill"]["pass"],
        analysis["read3_splice_gate"]["pass"],
        analysis["read2_fix_took"]["fix_took"],
        loc["s_ceil"],
        loc["s_coll"],
        analysis["read5_usable_anchor"]["steps_usable_at_read_line"],
        analysis["read5_usable_anchor"]["steps_usable_at_picker_gate"],
        args.out_json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

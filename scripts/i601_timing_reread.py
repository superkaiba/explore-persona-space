#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" are intentional
"""Task #601 follow-up — single-space re-read of the matched-pair step-32 timing call.

The REGISTERED matched-pair discriminator (classification.json
``phase1.matched_pair_discriminator``, schema i601_classification_v4) mixed
measurement spaces: the step-32 read came from the schedule-matched arm's
teacher-forced DENSE ladder (staged classic gauge, frozen-R eval questions)
while both thresholds — coupling_max (quarter terminal + 2.5) and horizon_min
(80% of the matched arm's own terminal) — were instantiated from ON-POLICY
terminals, seed-pooled. As registered it was underpowered (gap 2.786 < 3.0).

This script recomputes the timing read WITHIN one measurement space at a time,
per seed:

  - dense logP: step-32 level, terminal, fraction-of-terminal, and the 80%
    horizon clause with BOTH operands from the same dense series;
  - dense margin (EOS margin Δ(z_marker - z_eos) trained - base) from the
    stored four floats — the gauge-invariant co-read;
  - the decidability gap per seed in each space. The quarter arm has NO
    committed dense read (its only teacher-forced series is the in-loop band
    probe — live alpha/sqrt(r) gauge over training rows, saturated at the collapse
    ceiling, checked + rejected below), so the gap's quarter operand stays
    on-policy; the all-on-policy per-seed gap is reported alongside.

Analysis-only: reads committed eval_results/issue_601 JSONs, writes
``eval_results/issue_601/analysis/timing_reread_single_space.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

MATCHED_CELL = "ratio4to1_100p400n_T128"
QUARTER_CELL = "ratio4to1_100p400n"
SEEDS = (42, 137)
TERMINAL_STEP_MATCHED = 128
SCHEDULE_WINDOW_STEP = 16  # registered by-step-16 schedule-window clause.


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load(path: Path) -> dict:
    """Strict JSON load — a missing input is a crash, never a default."""
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _dense_at(dense: dict, step: int, key: str) -> float:
    """Source-mean value at an exact dense-ladder step; raises if absent."""
    for c in dense["checkpoints"]:
        if c["step"] == step:
            return float(c["source_mean"][key])
    raise ValueError(f"dense ladder has no step {step} (cell {dense['cell']} seed {dense['seed']})")


def _onpolicy_terminal(traj: dict) -> dict[str, float]:
    """Terminal on-policy source read in both spaces from the four-float record."""
    term = max(traj["checkpoints"], key=lambda c: c["frac"])
    ss = term["source_self"]
    margin = float(
        (ss["z_marker_g_mean"] - ss["z_eos_g_mean"]) - (ss["z_marker_b_mean"] - ss["z_eos_b_mean"])
    )
    return {"logp": float(ss["delta_g_mean"]), "margin": margin, "step": int(term["step"])}


def _gap_block(
    matched_terminal: float,
    quarter_terminal: float,
    v32: float | None,
    coupling_tol: float,
    horizon_frac: float,
    decidability_min: float,
    operand_spaces: dict[str, str],
) -> dict:
    """Per-seed matched-pair gap + clause reads with the operand spaces named."""
    horizon_min = horizon_frac * matched_terminal
    coupling_max = quarter_terminal + coupling_tol
    gap = horizon_min - coupling_max
    decidable = gap >= decidability_min
    if not decidable:
        verdict = "underpowered"
    elif v32 is None:
        verdict = "unreadable"
    elif v32 >= horizon_min:
        verdict = "horizon"
    elif v32 <= coupling_max:
        verdict = "coupling"
    else:
        verdict = "between"
    return {
        "horizon_min": horizon_min,
        "coupling_max": coupling_max,
        "decidability_gap": gap,
        "decidability_min": decidability_min,
        "decidable": decidable,
        "v32": v32,
        "verdict": verdict,
        "operand_spaces": operand_spaces,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Task #601 single-space step-32 timing re-read (see module docstring)."
    )
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_601"))
    ap.add_argument("--out-path", type=Path, default=None)
    args = ap.parse_args(argv)
    out_path = args.out_path or (args.slab_root / "analysis" / "timing_reread_single_space.json")

    from explore_persona_space.experiments.neg_setpoint_601.analysis_lib import (
        COUPLING_QUARTER_TOL,
        HORIZON_FRAC_OF_TERMINAL_BY_16,
        HORIZON_STEP32_FRAC_OF_OWN_TERMINAL,
        MATCHED_PAIR_DECIDABILITY_MIN,
        MATCHED_PAIR_STEP,
    )

    classification = _load(args.slab_root / "analysis" / "classification.json")
    registered_mp = classification["phase1"]["matched_pair_discriminator"]
    registered_accrual = classification["phase1"]["coupling_detail"]["accrual_by_seed"]
    registered_frac16 = classification["phase1"]["horizon_detail"]["frac_of_terminal_by_step16"]
    tol_logp = float(classification["in_task_references"]["tol_logp"])
    tol_margin = float(classification["in_task_references"]["tol_margin"])
    # The registered margin-fallback rescales the 2.5-nat coupling tolerance by
    # the margin/logP tolerance ratio (analysis_lib.classify_phase1); reuse it.
    coupling_tol = {
        "logp": COUPLING_QUARTER_TOL,
        "margin": COUPLING_QUARTER_TOL * tol_margin / tol_logp,
    }
    dense_key = {"logp": "delta_g", "margin": "delta_margin"}

    inputs: list[str] = [str(args.slab_root / "analysis" / "classification.json")]
    per_seed: dict[str, dict] = {}
    for seed in SEEDS:
        dense_p = args.slab_root / "phase1" / f"{MATCHED_CELL}_seed{seed}" / "dense_trajectory.json"
        matched_t_p = args.slab_root / "phase1" / f"{MATCHED_CELL}_seed{seed}" / "trajectory.json"
        quarter_t_p = args.slab_root / "phase1" / f"{QUARTER_CELL}_seed{seed}" / "trajectory.json"
        inputs += [str(dense_p), str(matched_t_p), str(quarter_t_p)]
        dense = _load(dense_p)
        matched_onpol = _onpolicy_terminal(_load(matched_t_p))
        quarter_onpol = _onpolicy_terminal(_load(quarter_t_p))

        rec: dict = {
            "quarter_terminal_onpolicy": quarter_onpol,
            "matched_terminal_onpolicy": matched_onpol,
        }
        for space in ("logp", "margin"):
            v32 = _dense_at(dense, MATCHED_PAIR_STEP, dense_key[space])
            v16 = _dense_at(dense, SCHEDULE_WINDOW_STEP, dense_key[space])
            terminal = _dense_at(dense, TERMINAL_STEP_MATCHED, dense_key[space])
            rec[f"dense_{space}"] = {
                "v32": v32,
                "v16": v16,
                "terminal": terminal,
                "frac32_of_terminal": v32 / terminal,
                "frac16_of_terminal": v16 / terminal,
                "horizon_clause_v32_ge_80pct_own_terminal": (
                    v32 >= HORIZON_STEP32_FRAC_OF_OWN_TERMINAL * terminal
                ),
                "schedule_window_clause_frac16_ge_80pct": (
                    v16 / terminal >= HORIZON_FRAC_OF_TERMINAL_BY_16
                ),
            }
            # Dense matched operands; quarter operand stays on-policy (the
            # quarter arm has no committed dense read — see band_space_check).
            rec[f"gap_dense_matched_vs_onpolicy_quarter_{space}"] = _gap_block(
                matched_terminal=terminal,
                quarter_terminal=quarter_onpol[space],
                v32=v32,
                coupling_tol=coupling_tol[space],
                horizon_frac=HORIZON_STEP32_FRAC_OF_OWN_TERMINAL,
                decidability_min=MATCHED_PAIR_DECIDABILITY_MIN,
                operand_spaces={
                    "matched_terminal": f"dense_{space} (teacher-forced frozen-R, staged gauge)",
                    "quarter_terminal": f"onpolicy_{space} (no committed dense read)",
                    "v32": f"dense_{space}",
                },
            )
            # Fully same-space gap variant: both terminals on-policy, per seed
            # (the registered computation seed-pooled these). No on-policy
            # step-32 checkpoint exists (nearest: steps 21 / 43), so v32=None.
            rec[f"gap_onpolicy_both_operands_{space}"] = _gap_block(
                matched_terminal=matched_onpol[space],
                quarter_terminal=quarter_onpol[space],
                v32=None,
                coupling_tol=coupling_tol[space],
                horizon_frac=HORIZON_STEP32_FRAC_OF_OWN_TERMINAL,
                decidability_min=MATCHED_PAIR_DECIDABILITY_MIN,
                operand_spaces={
                    "matched_terminal": f"onpolicy_{space}",
                    "quarter_terminal": f"onpolicy_{space}",
                    "v32": "none — no on-policy step-32 checkpoint (nearest: 21, 43)",
                },
            )
        per_seed[str(seed)] = rec

        # Reproduction asserts vs the registered artifact: the registered v32
        # per seed IS the dense logP read, and the registered frac16 was
        # already dense/dense (every on-policy step coincides with a dense
        # ladder step, so the merged series terminal is the dense terminal).
        reg_v32 = float(registered_accrual[str(seed)]["v32"])
        if abs(reg_v32 - rec["dense_logp"]["v32"]) > 1e-6:
            raise AssertionError(
                f"seed {seed}: registered v32 {reg_v32} != dense re-read {rec['dense_logp']['v32']}"
            )
        reg_f16 = float(registered_frac16[str(seed)])
        if abs(reg_f16 - rec["dense_logp"]["frac16_of_terminal"]) > 1e-6:
            raise AssertionError(
                f"seed {seed}: registered frac16 {reg_f16} != dense re-read "
                f"{rec['dense_logp']['frac16_of_terminal']}"
            )

    # ── Band-space check: the only teacher-forced series BOTH arms share. ────
    # Checked and REJECTED as a discriminator space: live alpha/sqrt(r) gauge
    # over training probe rows, saturated at the collapse ceiling (~21 nats >>
    # the dense read), and per-seed values are IDENTICAL across the two arms —
    # no arm contrast exists in that space.
    band_vals: dict[str, dict[str, float]] = {}
    for cell in (MATCHED_CELL, QUARTER_CELL):
        for seed in SEEDS:
            p = args.slab_root / "phase1" / f"{cell}_seed{seed}" / "inloop_band_trajectory.json"
            b = _load(p)
            inputs.append(str(p))
            band_vals.setdefault(cell, {})[str(seed)] = float(b["delta_nats"][-1])
    band_space_check = {
        "usable_as_discriminator_space": False,
        "terminal_delta_nats": band_vals,
        "reason": (
            "in-loop band probe is the live alpha/sqrt(r) training gauge over training rows "
            "(not the staged-gauge frozen-R eval read); terminals sit at the collapse "
            "ceiling and agree across arms within seed to <1e-5 nats — no arm contrast"
        ),
    }

    # ── Verdict block. ────────────────────────────────────────────────────────
    verdict_per_seed = {}
    for seed in SEEDS:
        rec = per_seed[str(seed)]
        verdict_per_seed[str(seed)] = {
            space: {
                "horizon_clause_dense": rec[f"dense_{space}"][
                    "horizon_clause_v32_ge_80pct_own_terminal"
                ],
                "frac32_of_terminal": rec[f"dense_{space}"]["frac32_of_terminal"],
                "schedule_window_frac16_clause": rec[f"dense_{space}"][
                    "schedule_window_clause_frac16_ge_80pct"
                ],
                "gap_hybrid": rec[f"gap_dense_matched_vs_onpolicy_quarter_{space}"][
                    "decidability_gap"
                ],
                "decidable_hybrid": rec[f"gap_dense_matched_vs_onpolicy_quarter_{space}"][
                    "decidable"
                ],
                "verdict_hybrid": rec[f"gap_dense_matched_vs_onpolicy_quarter_{space}"]["verdict"],
                "gap_onpolicy": rec[f"gap_onpolicy_both_operands_{space}"]["decidability_gap"],
                "decidable_onpolicy": rec[f"gap_onpolicy_both_operands_{space}"]["decidable"],
            }
            for space in ("logp", "margin")
        }
    verdict = {
        "per_seed": verdict_per_seed,
        "summary": (
            "Within dense space the step-32 horizon clause (v32 >= 80% of own terminal) passes "
            "in BOTH seeds and BOTH spaces (logP fractions "
            f"{per_seed['42']['dense_logp']['frac32_of_terminal']:.3f} / "
            f"{per_seed['137']['dense_logp']['frac32_of_terminal']:.3f}; margin "
            f"{per_seed['42']['dense_margin']['frac32_of_terminal']:.3f} / "
            f"{per_seed['137']['dense_margin']['frac32_of_terminal']:.3f}). The decidability "
            "gap clears 3.0 nats for seed 42 only (hybrid "
            f"{per_seed['42']['gap_dense_matched_vs_onpolicy_quarter_logp']['decidability_gap']:.3f}"
            ", all-on-policy "
            f"{per_seed['42']['gap_onpolicy_both_operands_logp']['decidability_gap']:.3f}); "
            "seed 137 stays underpowered in every space. The registered by-step-16 "
            "schedule-window clause was already dense/dense and remains failed (~41-42%). The "
            "quarter terminal stays the one cross-space operand in the hybrid gap (no committed "
            "dense read for the quarter arm)."
        ),
        "registered_comparison_note": (
            "registered discriminator mixed spaces (dense v32 vs seed-pooled on-policy "
            "thresholds) and seed-pooled the terminals; gap 2.786 < 3.0 -> underpowered"
        ),
    }

    payload = {
        "schema_version": "i601_timing_reread_v1",
        "followup": "free-analysis: single-space matched-pair step-32 timing re-read",
        "constants": {
            "matched_pair_step": MATCHED_PAIR_STEP,
            "horizon_frac_of_own_terminal": HORIZON_STEP32_FRAC_OF_OWN_TERMINAL,
            "schedule_window_step": SCHEDULE_WINDOW_STEP,
            "schedule_window_frac": HORIZON_FRAC_OF_TERMINAL_BY_16,
            "coupling_tol": coupling_tol,
            "decidability_min": MATCHED_PAIR_DECIDABILITY_MIN,
            "tol_logp": tol_logp,
            "tol_margin": tol_margin,
        },
        "spaces": {
            "dense_logp": (
                "teacher-forced frozen-R eval-question read, staged classic gauge "
                "(dense_trajectory.json source_mean.delta_g, trained - base)"
            ),
            "dense_margin": (
                "same read, EOS margin: Δ(z_marker - z_eos) trained - base "
                "(source_mean.delta_margin)"
            ),
            "onpolicy_logp": (
                "on-policy trajectory.json source_self.delta_g_mean (vLLM-generated R)"
            ),
            "onpolicy_margin": (
                "on-policy four-float EOS margin, Δ(z_marker - z_eos) trained - base"
            ),
        },
        "per_seed": per_seed,
        "registered_mixed_space": {
            "matched_pair_discriminator": registered_mp,
            "frac_of_terminal_by_step16": registered_frac16,
            "arm_terminal_means": classification["phase1"]["arm_terminal_means"],
            "what_mixed": (
                "v32 came from the dense ladder while horizon_min / coupling_max came from "
                "seed-pooled ON-POLICY terminals; frac16 was already dense/dense"
            ),
        },
        "band_space_check": band_space_check,
        "verdict": verdict,
        "inputs": sorted(set(inputs)),
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"timing re-read written -> {out_path}")
    print(verdict["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

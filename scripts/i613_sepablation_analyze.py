#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" + Greek delta are intentional
"""Task #613 follow-up `sep-ablation` — within-round A/B analysis (off-pod, VM, CPU).

Computes the pre-registered reads R1'-R5' (amendment plan §4) from the FOUR
fresh no-separator cells
(``eval_results/issue_613/sep-ablation/sepablation_flag{on,off}_200p800n_seed{42,137}/``)
and emits ``eval_results/issue_613/analysis/sepablation_verdict.json``.

Both arms are fresh, so — unlike the parent round's frozen ±5.58-nat band —
EVERY numeric tolerance is computed WITHIN this round from the realized seed
pairs (``tol = 2 x max within-arm seed gap``); parent numbers are context
only. Registered decision rules carried from the round-2 reconcilers:

  R2' margin precedence — a co-land verdict requires BOTH the logP read AND
      the EOS-margin twin to co-land; whenever either arm's trained log P is
      within ``tol`` nats of the 0 ceiling, the margin read GOVERNS the
      branch outright (the narrow saturation triage — logP within 0.1 nat of
      0 AND emission >= 0.92 — is the extreme case of the same precedence).
  R2' emission denominator — the PRIMARY keeps the FIXED generated-R probe
      set (all probes, marker-emitting or not); a paired sensitivity
      excluding ``n_marker_in_R > 0`` probes is reported alongside and never
      changes the primary verdict unless the cell is pre-registered
      degenerate/indeterminate.
  R3' denominator guard — ``flag_on_mean <= 0 AND flag_off_mean > 0`` maps to
      confirmed-strong outright; a difference-form twin governs ALL
      ratio-edge cells; a finite POSITIVE denominator is asserted before any
      ratio is computed.

Usage:
    uv run python scripts/i613_sepablation_analyze.py \
        [--round-root eval_results/issue_613/sep-ablation] \
        [--out eval_results/issue_613/analysis/sepablation_verdict.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i613.sepablation_analyze")

SEEDS = (42, 137)
FLAGON_CELL = "sepablation_flagon_200p800n"
FLAGOFF_CELL = "sepablation_flagoff_200p800n"

# Registered rule constants (plan §4 — rule SHAPES; tolerances are
# within-round, computed from realized seed pairs at analysis time).
R1_LIVENESS_FLOOR_NATS = 1e-3
LEAKAGE_RATIO_GATE = 2.0  # R3' confirmed iff ratio >= 2.0 in BOTH seeds
CLAMP_BAR_NATS = 1.5  # rule shape, recomputed within-construction
RISE_THEN_DROP_MIN_NATS = 1.0
NOISE_SCALE_NATS = 5.6  # tol above parent-scale noise -> indeterminate-for-noise
SATURATION_LOGP_TOL = 0.1  # extreme-case triage: trained log P within 0.1 nat of 0 ...
SATURATION_EMISSION_MIN = 0.92  # ... AND on-policy argmax emission >= 0.92


def _load_parent_helpers():
    """Import dense_series/_channel_means from scripts/i613_analyze.py (no package)."""
    path = Path(__file__).resolve().parent / "i613_analyze.py"
    spec = importlib.util.spec_from_file_location("i613_analyze_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required input missing: {path}")
    return json.loads(path.read_text())


def _terminal(checkpoints: list[dict]) -> dict:
    term = [c for c in checkpoints if float(c["frac"]) == 1.0]
    if not term:
        raise RuntimeError("no terminal (frac=1.0) checkpoint in payload")
    return term[0]


def _assert_slot_provenance(traj: dict, dense: dict, label: str) -> None:
    """Fail loud unless every input was READ at the construction's own slot.

    The sep-ablation DV slot is post-R with sep="" — a trajectory written by a
    pre-threading rig (no ``sep`` field) or a dense file at sep_mode="marker"
    would silently analyze the WRONG slot.
    """
    if traj.get("sep") != "":
        raise RuntimeError(
            f"[{label}] trajectory.json sep={traj.get('sep')!r} != '' — the on-policy reads "
            f"were not taken at the no-separator construction's slot (re-run the eval with "
            f"--sep-mode plain / the threaded rig)."
        )
    if dense.get("sep_mode") != "plain":
        raise RuntimeError(
            f"[{label}] dense_trajectory.json sep_mode={dense.get('sep_mode')!r} != 'plain' — "
            f"teacher-forced reads at the wrong slot."
        )


def _terminal_source(traj: dict) -> dict:
    """Terminal on-policy source read (R2' quantities + the per-probe leaves)."""
    term = _terminal(traj["checkpoints"])
    src = term["source_self"]
    per_q = term.get("source_per_q")
    if per_q is None:
        raise RuntimeError(
            "trajectory.json terminal checkpoint has no source_per_q — the R2' "
            "emission-denominator sensitivity is uncomputable; re-run the eval on the "
            "threaded rig (eval_trajectory.py persists per-question source leaves)."
        )
    return {
        "delta_g": float(src["delta_g_mean"]),
        "g_logp": float(src["g_logp_mean"]),
        "emission_p": float(src["emission_p"]),
        "r_collapsed": bool(src.get("r_collapsed", False)),
        "margin": float(
            (src["z_marker_g_mean"] - src["z_eos_g_mean"])
            - (src["z_marker_b_mean"] - src["z_eos_b_mean"])
        ),
        "saturated": bool(
            abs(float(src["g_logp_mean"])) <= SATURATION_LOGP_TOL
            and float(src["emission_p"]) >= SATURATION_EMISSION_MIN
        ),
        "per_q": per_q,
    }


def _trichotomy(diff: float, tol: float) -> str:
    if diff < -tol:
        return "suppression"
    if diff > tol:
        return "amplification"
    return "co-land"


def r1p_liveness(rowtype_on: dict[int, dict]) -> dict:
    """R1' — manipulation check: the relocated negative loss channel is live."""
    per_seed: dict[str, dict] = {}
    for seed, rt in rowtype_on.items():
        step1 = [r for r in rt["records"] if r["step"] == 1]
        if not step1 or step1[0].get("neg_slot_ce") is None:
            raise RuntimeError(f"flag-on seed {seed}: no step-1 neg_slot CE in rowtype_ce.json")
        ce1 = float(step1[0]["neg_slot_ce"])
        per_seed[f"seed{seed}"] = {
            "step1_neg_slot_ce": ce1,
            "neg_slot_ce_base": rt.get("neg_slot_ce_base"),
            "live": ce1 >= R1_LIVENESS_FLOOR_NATS,
        }
    return {
        "rule": f"step-1 flag-on neg_slot CE >= {R1_LIVENESS_FLOOR_NATS} nats in both seeds "
        f"(negative rows byte-identical to the parent round — a failure is a rig bug "
        f"until proven otherwise; parent context 0.0672 / 0.0241)",
        "per_seed": per_seed,
        "verdict": "PASS" if all(v["live"] for v in per_seed.values()) else "FAIL",
    }


def r2p_source_level(on_terms: dict[int, dict], off_terms: dict[int, dict]) -> dict:
    """R2' — PRIMARY: terminal on-policy source ΔG, flag-on vs flag-off (both fresh).

    Within-round tolerance, both-spaces co-land requirement, margin
    precedence across the WHOLE compression window, fixed-denominator
    emission sensitivity (see module docstring).
    """
    on_dg = [on_terms[s]["delta_g"] for s in SEEDS]
    off_dg = [off_terms[s]["delta_g"] for s in SEEDS]
    gap_on = abs(on_dg[0] - on_dg[1])
    gap_off = abs(off_dg[0] - off_dg[1])
    tol = 2.0 * max(gap_on, gap_off)
    diff = mean(on_dg) - mean(off_dg)
    logp_branch = _trichotomy(diff, tol)

    on_m = [on_terms[s]["margin"] for s in SEEDS]
    off_m = [off_terms[s]["margin"] for s in SEEDS]
    margin_tol = 2.0 * max(abs(on_m[0] - on_m[1]), abs(off_m[0] - off_m[1]))
    margin_diff = mean(on_m) - mean(off_m)
    margin_branch = _trichotomy(margin_diff, margin_tol)

    # Margin precedence (registered): the ΔG ceiling at this slot is ~22.2
    # nats, so both arms landing within tol of 0 trained logP would co-land
    # mechanically — whenever EITHER arm's trained log P (seed mean) is
    # within tol nats of 0, the margin read GOVERNS the branch outright.
    on_glogp_mean = mean(on_terms[s]["g_logp"] for s in SEEDS)
    off_glogp_mean = mean(off_terms[s]["g_logp"] for s in SEEDS)
    compression_window = (on_glogp_mean >= -tol) or (off_glogp_mean >= -tol)
    saturation_triage = any(on_terms[s]["saturated"] for s in SEEDS) or any(
        off_terms[s]["saturated"] for s in SEEDS
    )

    if compression_window or saturation_triage:
        branch = margin_branch
        governed_by = "margin (compression-window/saturation precedence)"
    elif logp_branch != "co-land":
        branch = logp_branch
        governed_by = "logp"
    elif margin_branch == "co-land":
        branch = "co-land"
        governed_by = "both spaces (co-land requires both)"
    else:
        branch = margin_branch
        governed_by = "margin (logp co-landed but margin did not — co-land requires both)"

    indeterminate_for_noise = tol > NOISE_SCALE_NATS

    # Emission-denominator sensitivity (registered): PRIMARY = the fixed
    # generated-R probe set above; paired sensitivity excludes probes whose
    # OWN R contains the marker (n_marker_in_R > 0). Reported alongside; can
    # override the primary ONLY when the cell is pre-registered
    # degenerate/indeterminate.
    def _excluded_mean(terms: dict) -> tuple[float | None, int, int]:
        leaves = terms["per_q"]
        kept = [v["delta_g"] for v in leaves.values() if int(v.get("n_marker_in_R", 0)) == 0]
        return (
            (mean(kept) if kept else None),
            len(leaves) - len(kept),
            len(leaves),
        )

    sens: dict = {"per_cell": {}}
    sens_on, sens_off = [], []
    for arm, terms_by_seed, acc in (
        ("flagon", on_terms, sens_on),
        ("flagoff", off_terms, sens_off),
    ):
        for s in SEEDS:
            m, n_excl, n_tot = _excluded_mean(terms_by_seed[s])
            sens["per_cell"][f"{arm}_seed{s}"] = {
                "delta_g_excluding_marker_in_R": m,
                "n_probes_excluded": n_excl,
                "n_probes_total": n_tot,
            }
            acc.append(m)
    if all(v is not None for v in [*sens_on, *sens_off]):
        s_tol = 2.0 * max(abs(sens_on[0] - sens_on[1]), abs(sens_off[0] - sens_off[1]))
        s_diff = mean(sens_on) - mean(sens_off)
        sens["diff_seed_mean"] = s_diff
        sens["tolerance"] = s_tol
        sens["branch"] = _trichotomy(s_diff, s_tol)
    else:
        sens["branch"] = "uncomputable (a cell lost every probe to marker-in-R exclusion)"
    sens["can_override_primary"] = bool(
        saturation_triage
        or any(t["r_collapsed"] for t in (*on_terms.values(), *off_terms.values()))
    )
    sens["rule"] = (
        "PRIMARY keeps the FIXED generated-R probe set; this exclusion read is reported "
        "alongside and never changes the primary verdict unless the cell is pre-registered "
        "degenerate/indeterminate (emission >= 0.92 triage or r_collapsed)"
    )

    return {
        "rule": "within-round tol = 2 x max within-arm seed gap; co-land requires BOTH "
        "logP AND EOS-margin spaces; margin GOVERNS whenever either arm's trained logP "
        "sits within tol of the 0 ceiling; indeterminate-for-noise reported when tol "
        f"exceeds ~{NOISE_SCALE_NATS} nats (parent-scale noise)",
        "flagon_delta_g": {f"seed{s}": on_terms[s]["delta_g"] for s in SEEDS},
        "flagoff_delta_g": {f"seed{s}": off_terms[s]["delta_g"] for s in SEEDS},
        "flagon_seed_mean": mean(on_dg),
        "flagoff_seed_mean": mean(off_dg),
        "diff_seed_mean": diff,
        "within_arm_seed_gaps": {"flagon": gap_on, "flagoff": gap_off},
        "tolerance": tol,
        "logp_branch": logp_branch,
        "margin_twin": {
            "flagon_margin": {f"seed{s}": on_terms[s]["margin"] for s in SEEDS},
            "flagoff_margin": {f"seed{s}": off_terms[s]["margin"] for s in SEEDS},
            "diff_seed_mean": margin_diff,
            "tolerance": margin_tol,
            "branch": margin_branch,
        },
        "trained_logp_seed_means": {"flagon": on_glogp_mean, "flagoff": off_glogp_mean},
        "compression_window_fired": compression_window,
        "saturation_triage_fired": saturation_triage,
        "per_cell_saturated": {
            "flagon": {f"seed{s}": on_terms[s]["saturated"] for s in SEEDS},
            "flagoff": {f"seed{s}": off_terms[s]["saturated"] for s in SEEDS},
        },
        "branch": branch,
        "governed_by": governed_by,
        "indeterminate_for_noise": indeterminate_for_noise,
        "emission_denominator_sensitivity": sens,
    }


def r3p_leakage(on_series: dict[int, list[dict]], off_series: dict[int, list[dict]]) -> dict:
    """R3' — leakage cut at the coincident slot: ratio gate + denominator guard.

    Per seed: ratio = flag_off / flag_on terminal trained-negative mean ΔG.
    ``flag_on <= 0 AND flag_off > 0`` -> confirmed-strong outright (the
    predicted-success direction; the parent's flag-on per-question ΔG already
    dips to -0.222 at this slot). The difference twin (flag_off - flag_on)
    governs ALL ratio-edge cells; a finite positive denominator is asserted
    before ANY ratio is computed.
    """
    per_seed: dict[str, dict] = {}
    for seed in SEEDS:
        on_term = on_series[seed][-1]
        off_term = off_series[seed][-1]
        on_v = float(on_term["trained_neg"]["delta_g"])
        off_v = float(off_term["trained_neg"]["delta_g"])
        difference = off_v - on_v  # the twin: governs ratio-edge cells
        ratio = None
        if on_v <= 0.0 and off_v > 0.0:
            classification = "confirmed-strong"  # denominator guard (registered)
            confirmed = True
        elif on_v > 0.0:
            ratio = off_v / on_v  # denominator asserted finite positive above
            classification = "ratio"
            confirmed = ratio >= LEAKAGE_RATIO_GATE
        else:
            classification = "edge-indeterminate (both arms <= 0; difference twin governs)"
            confirmed = False
        # Secondary (rule shapes recomputed within-construction).
        on_tneg_series = [r["trained_neg"]["delta_g"] for r in on_series[seed]]
        off_tneg_series = [r["trained_neg"]["delta_g"] for r in off_series[seed]]
        per_seed[f"seed{seed}"] = {
            "flagon_trained_neg_delta_g": on_v,
            "flagoff_trained_neg_delta_g": off_v,
            "ratio_flagoff_over_flagon": ratio,
            "difference_flagoff_minus_flagon": difference,
            "classification": classification,
            "confirmed": confirmed,
            "clamp_gap_flagon": float(
                on_term["bystander"]["delta_g"] - on_term["trained_neg"]["delta_g"]
            ),
            "clamp_gap_flagoff": float(
                off_term["bystander"]["delta_g"] - off_term["trained_neg"]["delta_g"]
            ),
            "clamp_present_flagon": (
                on_term["bystander"]["delta_g"] - on_term["trained_neg"]["delta_g"]
            )
            >= CLAMP_BAR_NATS,
            "rise_then_drop_flagon": (max(on_tneg_series) - on_tneg_series[-1])
            >= RISE_THEN_DROP_MIN_NATS,
            "rise_then_drop_null_flagoff_max_stat": max(off_tneg_series) - off_tneg_series[-1],
        }
    return {
        "rule": f"suppression-at-leakage confirmed iff ratio >= {LEAKAGE_RATIO_GATE} in BOTH "
        f"seeds; flag_on<=0 AND flag_off>0 -> confirmed-strong outright; difference twin "
        f"governs ratio-edge cells; clamp bar {CLAMP_BAR_NATS} nats + rise-then-drop >= "
        f"{RISE_THEN_DROP_MIN_NATS} nat recomputed within-construction (flag-off arm = "
        f"empirical null)",
        "per_seed": per_seed,
        "confirmed_both_seeds": all(v["confirmed"] for v in per_seed.values()),
        "clamp_present_both_seeds": all(v["clamp_present_flagon"] for v in per_seed.values()),
        "rise_then_drop_both_seeds": all(v["rise_then_drop_flagon"] for v in per_seed.values()),
    }


def r4p_channels(on_series: dict[int, list[dict]], off_series: dict[int, list[dict]]) -> dict:
    """R4' — descriptive: Δz_eos vs Δz_marker at the SINGLE post-R slot, matched steps."""
    common_steps = sorted(
        set.intersection(
            *[{r["step"] for r in rows} for rows in (*on_series.values(), *off_series.values())]
        )
    )

    def _at(rows: list[dict]) -> dict[str, dict]:
        by_step = {r["step"]: r for r in rows}
        return {
            str(s): {
                ch: {
                    k: by_step[s][ch][k]
                    for k in ("delta_g", "delta_z_marker", "delta_z_eos", "delta_margin")
                }
                for ch in ("source", "trained_neg", "bystander")
            }
            for s in common_steps
        }

    return {
        "note": "descriptive, no gate — does the stop-token boost (Δz_eos) coexist with a "
        "genuine marker push-down (Δz_marker) inside the SAME softmax at the coincident "
        "post-R slot, or does the normalizer absorb it; arms at matched steps",
        "matched_steps": common_steps,
        "flagon": {f"seed{s}": _at(on_series[s]) for s in SEEDS},
        "flagoff": {f"seed{s}": _at(off_series[s]) for s in SEEDS},
    }


def r5p_generalization(
    on_series: dict[int, list[dict]],
    off_series: dict[int, list[dict]],
    on_terms: dict[int, dict],
    off_terms: dict[int, dict],
    traj_by_arm_seed: dict[tuple[str, int], dict],
    raw_by_arm_seed: dict[tuple[str, int], dict | None],
) -> dict:
    """R5' — descriptive: leakage fraction, emission (now live), degenerate probes, lengths."""

    def _series_arm(series_by_seed: dict[int, list[dict]]) -> dict:
        out = {}
        for seed, rows in series_by_seed.items():
            term = rows[-1]
            src = term["source"]["delta_g"]
            out[f"seed{seed}"] = {
                "bystander_delta_g": term["bystander"]["delta_g"],
                "source_delta_g": src,
                "leakage_fraction": (term["bystander"]["delta_g"] / src) if src else None,
            }
        return out

    emission: dict[str, dict] = {}
    degenerate: dict[str, dict] = {}
    lengths: dict[str, dict | None] = {}
    for arm, terms in (("flagon", on_terms), ("flagoff", off_terms)):
        for seed in SEEDS:
            key = f"{arm}_seed{seed}"
            traj = traj_by_arm_seed[(arm, seed)]
            term = _terminal(traj["checkpoints"])
            held = term["held_out"]
            held_leaves = [held[p][q] for p in held for q in held[p]]
            n_held = len(held_leaves)
            emission[key] = {
                "source_emission_p": terms[seed]["emission_p"],
                "held_out_emission_rate": (
                    sum(1 for v in held_leaves if v.get("argmax_marker")) / n_held
                    if n_held
                    else None
                ),
            }
            src_leaves = list(term.get("source_per_q", {}).values())
            degenerate[key] = {
                "held_out_n_marker_in_R_gt0": sum(
                    1 for v in held_leaves if int(v.get("n_marker_in_R", 0)) > 0
                ),
                "held_out_r_collapsed": sum(1 for v in held_leaves if v.get("r_collapsed")),
                "held_out_total": n_held,
                "source_n_marker_in_R_gt0": sum(
                    1 for v in src_leaves if int(v.get("n_marker_in_R", 0)) > 0
                ),
                "source_r_collapsed": sum(1 for v in src_leaves if v.get("r_collapsed")),
                "source_total": len(src_leaves),
            }
            raw = raw_by_arm_seed.get((arm, seed))
            if raw is None:
                lengths[key] = None
            else:
                fracs = raw.get("completions_by_frac", {})
                terminal_key = max(fracs, key=lambda k: float(k.split("_", 1)[1]), default=None)
                if terminal_key is None:
                    lengths[key] = None
                else:
                    texts = [t for per_q in fracs[terminal_key].values() for t in per_q.values()]
                    lengths[key] = {
                        "frac": terminal_key,
                        "n": len(texts),
                        "mean_chars": mean(len(t) for t in texts) if texts else None,
                        "median_chars": median(len(t) for t in texts) if texts else None,
                        "mean_words": mean(len(t.split()) for t in texts) if texts else None,
                    }
    return {
        "note": "descriptive — with the marker trained at the natural stop slot, emission "
        "is live, not theoretical (parent arm-matched leakage fraction 0.46-0.51 is "
        "sep-construction context only)",
        "leakage_fraction": {
            "flagon": _series_arm(on_series),
            "flagoff": _series_arm(off_series),
        },
        "emission": emission,
        "degenerate_probes": degenerate,
        "generation_lengths_terminal": lengths,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Task #613 sep-ablation within-round A/B analysis (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--round-root", type=Path, default=Path("eval_results/issue_613/sep-ablation"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("eval_results/issue_613/analysis/sepablation_verdict.json"),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=i613_sepablation_analyze] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    helpers = _load_parent_helpers()

    inputs: dict[str, str] = {}

    def _track(path: Path) -> Path:
        inputs[str(path)] = "present"
        return path

    cells = {"flagon": FLAGON_CELL, "flagoff": FLAGOFF_CELL}
    traj: dict[tuple[str, int], dict] = {}
    dense_raw: dict[tuple[str, int], dict] = {}
    rowtype_on: dict[int, dict] = {}
    raw_comp: dict[tuple[str, int], dict | None] = {}
    for arm, cell in cells.items():
        for seed in SEEDS:
            d = args.round_root / f"{cell}_seed{seed}"
            traj[(arm, seed)] = _load(_track(d / "trajectory.json"))
            dense_raw[(arm, seed)] = _load(_track(d / "dense_trajectory.json"))
            _assert_slot_provenance(traj[(arm, seed)], dense_raw[(arm, seed)], f"{cell}_seed{seed}")
            raw_path = d / "raw_completions.json"
            raw_comp[(arm, seed)] = json.loads(raw_path.read_text()) if raw_path.exists() else None
            if arm == "flagon":
                rowtype_on[seed] = _load(_track(d / "rowtype_ce.json"))

    on_terms = {s: _terminal_source(traj[("flagon", s)]) for s in SEEDS}
    off_terms = {s: _terminal_source(traj[("flagoff", s)]) for s in SEEDS}
    dense_on = {s: helpers.dense_series(dense_raw[("flagon", s)]) for s in SEEDS}
    dense_off = {s: helpers.dense_series(dense_raw[("flagoff", s)]) for s in SEEDS}

    r2 = r2p_source_level(on_terms, off_terms)
    r3 = r3p_leakage(dense_on, dense_off)
    verdict = {
        "schema_version": "i613_sepablation_verdict_v1",
        "cells": cells,
        "seeds": list(SEEDS),
        "constants": {
            "r1_liveness_floor_nats": R1_LIVENESS_FLOOR_NATS,
            "leakage_ratio_gate": LEAKAGE_RATIO_GATE,
            "clamp_bar_nats": CLAMP_BAR_NATS,
            "rise_then_drop_min_nats": RISE_THEN_DROP_MIN_NATS,
            "noise_scale_nats": NOISE_SCALE_NATS,
            "tolerance_rule": "2 x max within-arm seed gap, computed from this round's "
            "realized pairs (both arms fresh — no frozen cross-run band)",
        },
        "r1_liveness": r1p_liveness(rowtype_on),
        "r2_source_level": r2,
        "r3_leakage_cut": r3,
        "r4_channels": r4p_channels(dense_on, dense_off),
        "r5_generalization": r5p_generalization(
            dense_on, dense_off, on_terms, off_terms, traj, raw_comp
        ),
        "inputs": inputs,
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    # Falsification read (plan §1): under the no-separator construction the
    # alive-vs-dead-slot cross-arm difference in marker-slot leakage AND
    # terminal source ΔG still co-land -> the separator gap is not what
    # blocks the suppression.
    verdict["overall"] = {
        "double_null": (
            verdict["r2_source_level"]["branch"] == "co-land"
            and not verdict["r3_leakage_cut"]["confirmed_both_seeds"]
        ),
        "note": "double_null=True -> the separator gap is not what blocks the suppression; "
        "the leakage-magnitude (dose) account stands alone and the high-leakage-anchor "
        "follow-up becomes the unique remaining mechanism test (plan §1 falsification)",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(verdict, indent=2))
    os.replace(tmp, args.out)
    log.info(
        "sepablation_verdict written -> %s (R1' %s; R2' %s [governed by %s, "
        "indeterminate_for_noise=%s]; R3' confirmed_both=%s)",
        args.out,
        verdict["r1_liveness"]["verdict"],
        r2["branch"],
        r2["governed_by"],
        r2["indeterminate_for_noise"],
        r3["confirmed_both_seeds"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

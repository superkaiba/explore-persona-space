#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" intentional
"""Task #622 — off-pod analysis: the registered §6.2 decision rules, verbatim.

Runs on the VM over the COMMITTED eval JSONs (after instance deletion — plan
§4.3 p5). No analyst discretion enters any rule below; every quantity comes
from the four-float storage-contract fields of the terminal ON-POLICY source
reads, except the wake-up rule (rowtype_ce.json) and the guardrail
(capability_trajectory.json).

Implements (plan #622 §6.2, frozen at plan time):
  - Margin-band artifact-lock (carry-forward concern §13 item 2): recompute
    B_margin = max(2 x matched-arm within-cell terminal margin seed gap, 1.0)
    from the pinned #601 committed artifacts; on a mismatch with the 2.18
    prose value the ARTIFACT-DERIVED value WINS (record-correction flag set).
  - Computable margin-switch per level: margin governs iff (a) ceiling
    compression — min(dose, twin) terminal seed-mean trained g_logp > -5.58 —
    OR (b) divergence — either member's terminal seed-mean dlogZ >= 2.79.
  - TOTAL per-level verdict lattice, precedence P0 -> P1 -> P2:
      P0 collapse trichotomy (>=1 seed r_collapsed on the terminal on-policy
         source read; both/dose-only/twin-only -> the three collapse classes);
      P1 indeterminate-for-noise (pair_seed_gap > B in the governing space);
      P2 signed classes at the frozen band (suppression / enhancement /
         co-landing).
  - Crash / erosion / emission-onset OVERLAYS (never verdict overrides).
  - Decidable negative wake-up rule (onset / SUSTAINED >= 20 consecutive
    recorded reads > 1e-3 nats / LATE-ONSET UNDECIDABLE), per dose cell x
    seed; level state PRESENT / ABSENT / UNDECIDABLE-MIXED.
  - Loss-competition cross-level synthesis (>= 2 monotone suppressions AND
    wake-up PRESENT at >= 1 suppressed level; else the non-loss-channel
    FAMILY — §13 item 4: always the family, never a specific mechanism).
  - ARC-C capability guardrail (accuracy > 5 points below the unit's FIRST
    read, sustained >= 3 consecutive reads -> general-damage flag).
  - Reporting riders: twins' positive-channel CE (§13 item 1), dz_marker and
    dz_eos reported separately (§13 item 6), DV6 transfer fractions in
    EOS-margin space with trained negatives SEPARATE from held-out bystanders
    (§13 item 7).

Usage (VM, repo root, after the sweep's eval JSONs are committed):
    uv run python scripts/i622_analyze.py \
        [--slab-root eval_results/issue_622] [--i601-root eval_results/issue_601] \
        [--out eval_results/issue_622/analysis/classification.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

log = logging.getLogger("i622.analyze")

# ── Frozen registered constants (plan #622 §6.2; never recomputed from this
# run's own arms). ────────────────────────────────────────────────────────────
B_LOGPROB_NATS = 5.58  # frozen #601 tolerance-formula output
MARGIN_BAND_PROSE_LOGITS = 2.18  # #601 prose value; artifact recompute WINS on mismatch
MARGIN_BAND_FLOOR_LOGITS = 1.0  # the frozen formula's floor
CEILING_SWITCH_GLOGP = -5.58  # margin-switch (a): trained g_logp above this = compressed
DLOGZ_SWITCH_NATS = 2.79  # margin-switch (b): = B/2
WAKEUP_CE_NATS = 1e-3
WAKEUP_SUSTAINED_READS = 20
EROSION_NATS = 3.0  # within-run erosion overlay (convenience-chosen; §13 item 8)
EROSION_LOGITS = 1.5
CAP_DROP_POINTS = 5.0  # ARC-C guardrail: percentage points below first read
CAP_SUSTAINED_READS = 3

SEEDS = (42, 137)
LEVELS: tuple[tuple[str, str, str], ...] = (
    ("16:1", "dose_200p3200n", "posonly_200p_T208"),
    ("32:1", "dose_200p6400n", "posonly_200p_T416"),
    ("64:1", "dose_200p12800n", "posonly_200p_T819"),
)
# Trained-negative panel (kept SEPARATE from held-out bystanders in every
# leakage read — §13 item 7).
TRAINED_NEGATIVES = ("qwen_default", "hero", "journalist", "ai_assistant")
SOURCE = "villain"

# Registered #601 reference terminals (prose values; the loader below re-reads
# them from the committed artifacts and asserts agreement within 0.3 nat —
# plan §12 assumption 10: load with sha-logged paths, never trust prose alone).
REF_601_PROSE = {
    "dense_200p0n": (13, 2.68),
    "dense_200p400n": (38, 10.37),
    "dense_200p800n": (63, 12.81),
    "dense_200p1600n": (113, 13.98),
    "ratio4to1_100p400n_T128": (128, 15.57),
    "posonly_200p_T130": (130, 17.04),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _terminal_source(traj: dict) -> dict:
    """Terminal on-policy source read -> the §6.2 per-seed quantities.

    All from ``source_self`` of the max-frac checkpoint (four-float contract):
    delta_g (nats), delta_margin / delta_z_marker / delta_z_eos (logits),
    g_logp (trained source log-prob), dlogZ, r_collapsed, emission_p.
    """
    term = max(traj["checkpoints"], key=lambda c: c["frac"])
    ss = term["source_self"]
    needed = (
        "delta_g_mean",
        "g_logp_mean",
        "z_marker_g_mean",
        "z_marker_b_mean",
        "z_eos_g_mean",
        "z_eos_b_mean",
        "logZ_g_mean",
        "logZ_b_mean",
    )
    missing = [k for k in needed if ss.get(k) is None]
    if missing:
        raise RuntimeError(
            f"terminal source_self lacks four-float fields {missing} — storage-contract "
            f"violation (incident #530 class); refusing to classify from a degraded read."
        )
    dz_marker = ss["z_marker_g_mean"] - ss["z_marker_b_mean"]
    dz_eos = ss["z_eos_g_mean"] - ss["z_eos_b_mean"]
    return {
        "frac": term["frac"],
        "step": term.get("step"),
        "delta_g": float(ss["delta_g_mean"]),
        "delta_margin": float(dz_marker - dz_eos),
        "delta_z_marker": float(dz_marker),
        "delta_z_eos": float(dz_eos),
        "g_logp": float(ss["g_logp_mean"]),
        "dlogZ": float(ss["logZ_g_mean"] - ss["logZ_b_mean"]),
        "r_collapsed": bool(ss.get("r_collapsed", False)),
        "emission_p": ss.get("emission_p"),
        "held_out": term.get("held_out", {}),
        # #622 DV6 round-2: per-persona role map recorded by the eval wrapper
        # (trained_negative | held_out_bystander). None on legacy artifacts —
        # transfer_fractions then falls back to the TRAINED_NEGATIVES name set.
        "panel_roles": traj.get("panel_roles"),
    }


def _seed_mean(per_seed: dict[int, dict], key: str) -> float:
    vals = [per_seed[s][key] for s in per_seed]
    return sum(vals) / len(vals)


def _pair_seed_gap(per_seed: dict[int, dict], key: str) -> float | None:
    if len(per_seed) < 2:
        return None
    vals = [per_seed[s][key] for s in sorted(per_seed)]
    return abs(vals[0] - vals[1])


def recompute_margin_band(i601_root: Path) -> dict:
    """Carry-forward concern §13 item 2: the margin band's artifact lock.

    B_margin = max(2 x matched-arm within-cell terminal margin seed gap,
    1.0 logit), recomputed from the PINNED committed artifacts
    (phase1/ratio4to1_100p400n_T128_seed{42,137}/trajectory.json). On a
    mismatch with the 2.18 prose value the artifact-derived value WINS and a
    record-correction flag is set (the orchestrator posts the note on #601).
    """
    per_seed: dict[int, float] = {}
    paths: dict[str, str] = {}
    for seed in SEEDS:
        p = i601_root / "phase1" / f"ratio4to1_100p400n_T128_seed{seed}" / "trajectory.json"
        if not p.exists():
            raise FileNotFoundError(
                f"matched-arm artifact missing: {p} — the margin band cannot be "
                f"artifact-locked without it (§13 item 2; never substitute the parent "
                f"lattice's tol_margin=3.108…)."
            )
        per_seed[seed] = _terminal_source(json.loads(p.read_text()))["delta_margin"]
        paths[str(p)] = _sha256(p)
    gap = abs(per_seed[42] - per_seed[137])
    band = max(2.0 * gap, MARGIN_BAND_FLOOR_LOGITS)
    # 0.01 tolerance: the #601 prose truncated the formula output to 2 dp
    # (artifact recompute = 2.1862 -> prose "2.18"); only a BIGGER gap means a
    # genuinely different value and warrants the #601 record-correction note.
    # Either way the artifact-derived value WINS (band_logits below).
    mismatch = abs(band - MARGIN_BAND_PROSE_LOGITS) > 0.01
    return {
        "formula": "max(2 x matched-arm within-cell terminal margin seed gap, 1.0 logit)",
        "matched_arm_terminal_margin_per_seed": per_seed,
        "seed_gap_logits": gap,
        "band_logits": band,
        "prose_value_logits": MARGIN_BAND_PROSE_LOGITS,
        "prose_mismatch": mismatch,
        "record_correction_needed_on_601": mismatch,
        "input_paths_sha256": paths,
    }


def load_601_references(i601_root: Path) -> dict:
    """Reused #601 reference terminals (analysis-side, no GPU; plan §5).

    Loads each committed trajectory, recomputes the terminal source delta_g
    seed-mean, sha-logs the paths, and asserts agreement with the registered
    prose values within 0.3 nat (a bigger gap = wrong file / wrong gauge).
    """
    dirs = {
        "dense_200p0n": [i601_root / "phase2" / "dense_200p0n_seed137"],
        "dense_200p400n": [i601_root / "phase2" / "dense_200p400n_seed137"],
        # 12.81 is the SEED-MEAN of the two committed seeds (42 was the
        # anchor-retrain fallback unit): (13.93 + 11.70)/2 — verified against
        # the committed artifacts at implementation time.
        "dense_200p800n": [
            i601_root / "phase2" / "dense_200p800n_seed42",
            i601_root / "phase2" / "dense_200p800n_seed137",
        ],
        "dense_200p1600n": [i601_root / "phase2" / "dense_200p1600n_seed137"],
        "ratio4to1_100p400n_T128": [
            i601_root / "phase1" / "ratio4to1_100p400n_T128_seed42",
            i601_root / "phase1" / "ratio4to1_100p400n_T128_seed137",
        ],
        "posonly_200p_T130": [
            i601_root / "posonly-multiepoch-schedule-closure" / "posonly_200p_T130_seed42",
            i601_root / "posonly-multiepoch-schedule-closure" / "posonly_200p_T130_seed137",
        ],
    }
    out: dict[str, dict] = {}
    for name, cell_dirs in dirs.items():
        vals, paths = [], {}
        for d in cell_dirs:
            p = d / "trajectory.json"
            if not p.exists():
                raise FileNotFoundError(f"#601 reference artifact missing: {p}")
            t = _terminal_source(json.loads(p.read_text()))
            vals.append(t["delta_g"])
            paths[str(p)] = _sha256(p)
        mean = sum(vals) / len(vals)
        t_steps, prose = REF_601_PROSE[name]
        if abs(mean - prose) > 0.3:
            raise RuntimeError(
                f"#601 reference {name}: artifact terminal seed-mean {mean:.2f} differs "
                f"from the registered value {prose:.2f} by > 0.3 nat — wrong file or "
                f"wrong gauge; refusing to anchor the curve on it."
            )
        out[name] = {
            "T": t_steps,
            "delta_g_seed_mean": mean,
            "per_seed": vals,
            "n_seeds": len(vals),
            "single_seed_flag": len(vals) == 1,
            "input_paths_sha256": paths,
        }
    return out


def wakeup_state_for_seed(rowtype: dict) -> dict:
    """The decidable wake-up rule over ONE dose cell x seed (recorded reads)."""
    steps = rowtype.get("steps", [])
    series = rowtype.get("neg_trailing_ce", [])
    if len(steps) != len(series):
        raise RuntimeError("rowtype_ce.json steps/neg_trailing_ce length mismatch")
    vals = [v for v in series if v is not None]
    if len(vals) != len(series):
        raise RuntimeError("rowtype_ce.json carries null neg_trailing_ce records")
    onset_idx = next((i for i, v in enumerate(vals) if v > WAKEUP_CE_NATS), None)
    sustained = False
    if onset_idx is not None:
        run = 0
        for v in vals[onset_idx:]:
            run = run + 1 if v > WAKEUP_CE_NATS else 0
            if run >= WAKEUP_SUSTAINED_READS:
                sustained = True
                break
    if sustained:
        state = "SUSTAINED"
    elif onset_idx is None:
        state = "NO_ONSET"
    elif len(vals) - onset_idx < WAKEUP_SUSTAINED_READS:
        state = "LATE_ONSET_UNDECIDABLE"
    else:
        state = "ONSET_NOT_SUSTAINED"
    return {
        "state": state,
        "onset_record_index": onset_idx,
        "onset_step": steps[onset_idx] if onset_idx is not None else None,
        "n_recorded_reads": len(vals),
        "max_neg_trailing_ce": max(vals) if vals else None,
        "final_neg_trailing_ce": vals[-1] if vals else None,
    }


def wakeup_level_verdict(per_seed: dict[int, dict]) -> str:
    states = {s: d["state"] for s, d in per_seed.items()}
    if all(v == "SUSTAINED" for v in states.values()) and len(states) >= 2:
        return "PRESENT"
    if all(v == "NO_ONSET" for v in states.values()) and len(states) >= 2:
        return "ABSENT"
    return "UNDECIDABLE_OR_MIXED"


def erosion_overlay(dense: dict) -> dict:
    """Within-run erosion read over the teacher-forced dense source series."""
    cks = sorted(dense["checkpoints"], key=lambda c: (c["step"] is None, c["step"]))
    dg = [c["source_mean"]["delta_g"] for c in cks]
    dm = [c["source_mean"]["delta_margin"] for c in cks]
    steps = [c["step"] for c in cks]
    out = {
        "peak_delta_g": max(dg),
        "peak_delta_g_step": steps[dg.index(max(dg))],
        "terminal_delta_g": dg[-1],
        "erosion_nats": max(dg) - dg[-1],
        "erosion_nats_exceeds": (max(dg) - dg[-1]) > EROSION_NATS,
        "peak_delta_margin": max(dm),
        "terminal_delta_margin": dm[-1],
        "erosion_logits": max(dm) - dm[-1],
        "erosion_logits_exceeds": (max(dm) - dm[-1]) > EROSION_LOGITS,
    }
    # Emission-onset milestone: first dense checkpoint with mean z_marker_g >
    # mean z_eos_g at SOURCE slots (source_mean lacks raw z_marker_g — derive
    # from the per-q reads).
    src = dense.get("source", SOURCE)
    onset_step = None
    for c in cks:
        reads = c["reads"][src]
        zm = sum(r["z_marker_g"] for r in reads.values()) / len(reads)
        ze = sum(r["z_eos_g"] for r in reads.values()) / len(reads)
        if zm > ze:
            onset_step = c["step"]
            break
    out["emission_onset_step_z_marker_gt_z_eos"] = onset_step
    return out


def capability_flag(cap: dict) -> dict:
    """ARC-C guardrail: > 5 points below the FIRST read, >= 3 consecutive reads."""
    recs = cap.get("records", [])
    if not recs:
        return {"n_reads": 0, "general_damage_flag": None, "note": "no capability reads"}
    first = recs[0]["accuracy"]
    thresh = first - CAP_DROP_POINTS / 100.0
    run, flagged, flag_step = 0, False, None
    for r in recs:
        if r["accuracy"] < thresh:
            run += 1
            if run >= CAP_SUSTAINED_READS and not flagged:
                flagged, flag_step = True, r["step"]
        else:
            run = 0
    return {
        "n_reads": len(recs),
        "first_accuracy": first,
        "min_accuracy": min(r["accuracy"] for r in recs),
        "final_accuracy": recs[-1]["accuracy"],
        "threshold": thresh,
        "general_damage_flag": flagged,
        "flag_onset_step": flag_step,
    }


def transfer_fractions(term: dict) -> dict:
    """DV6: per-group transfer fraction in EOS-margin space (§6.3 / §13 item 7).

    bystander gain / source gain, NEVER raw logP; trained negatives and
    held-out bystanders are SEPARATE populations (split by the eval wrapper's
    recorded panel_roles when present — round-2 DV6 fix; legacy artifacts
    without the role map fall back to the TRAINED_NEGATIVES name set). The
    fraction is never correlated back against install (#383 family) —
    reporting only.
    """
    src_margin = term["delta_margin"]
    roles = term.get("panel_roles") or {}
    groups: dict[str, list[float]] = {"trained_negatives": [], "held_out_bystanders": []}
    for persona, by_q in term["held_out"].items():
        margins = [leaf["delta_margin"] for leaf in by_q.values() if "delta_margin" in leaf]
        if not margins:
            continue
        mean_m = sum(margins) / len(margins)
        if roles:
            role = roles.get(persona, "held_out_bystander")
            key = "trained_negatives" if role == "trained_negative" else "held_out_bystanders"
        else:
            key = "trained_negatives" if persona in TRAINED_NEGATIVES else "held_out_bystanders"
        groups[key].append(mean_m)
    out: dict = {
        "source_delta_margin": src_margin,
        "role_source": "panel_roles" if roles else "name-set fallback (legacy artifact)",
    }
    for key, vals in groups.items():
        out[key] = {
            "n_personas": len(vals),
            "mean_delta_margin": (sum(vals) / len(vals)) if vals else None,
            "mean_transfer_fraction": (
                (sum(vals) / len(vals)) / src_margin if vals and src_margin else None
            ),
        }
    return out


def synthesize_loss_competition(level_results: list[dict]) -> str:
    """Cross-level loss-competition verdict (§6.2, registered).

    Requires ALL of: >= 2 levels classified suppression with |D_i| growing
    monotonically across the SUPPRESSED-level subsequence in level order
    (16:1 -> 32:1 -> 64:1 or the realized subset — round-2 Claude Minor 2:
    the monotonicity runs over |D| of the suppressed levels only, never the
    signed D of non-suppressed levels, which would let a co-landing mid level
    veto or fake the trend), AND wake-up PRESENT at >= 1 suppressed level.
    Suppression without wake-up -> the non-loss-channel FAMILY (§13 item 4).
    """
    suppressed = [r for r in level_results if str(r.get("verdict", "")).startswith("suppression")]
    abs_d_suppressed = [abs(r["D_signed"]) for r in suppressed if r.get("D_signed") is not None]
    monotone_growing = len(abs_d_suppressed) >= 2 and all(
        b > a for a, b in pairwise(abs_d_suppressed)
    )
    wakeup_at_suppressed = any(r.get("wakeup_level") == "PRESENT" for r in suppressed)
    if len(suppressed) >= 2 and monotone_growing and wakeup_at_suppressed:
        return "loss-competition CERTIFIED (registered cross-level rule satisfied)"
    if suppressed and not wakeup_at_suppressed:
        return (
            "suppression WITHOUT wake-up -> non-loss-channel FAMILY (batch composition / "
            "data ordering / optimizer-state momentum effects) — reported as the family, "
            "never a specific mechanism (§13 item 4)"
        )
    if suppressed:
        return "suppression present but the cross-level rule is not satisfied"
    return "no suppressed level — loss-competition not supported at any dose"


def classify_level(
    label: str,
    dose_terms: dict[int, dict],
    twin_terms: dict[int, dict],
    band_margin: float,
) -> dict:
    """One level through the TOTAL lattice (P0 -> P1 -> P2), space-switch first."""
    # ── Space selection (computable margin-switch). ──────────────────────────
    dose_glogp = _seed_mean(dose_terms, "g_logp")
    twin_glogp = _seed_mean(twin_terms, "g_logp")
    dose_dlogz = _seed_mean(dose_terms, "dlogZ")
    twin_dlogz = _seed_mean(twin_terms, "dlogZ")
    switch_a = min(dose_glogp, twin_glogp) > CEILING_SWITCH_GLOGP
    switch_b = dose_dlogz >= DLOGZ_SWITCH_NATS or twin_dlogz >= DLOGZ_SWITCH_NATS
    space = "margin" if (switch_a or switch_b) else "logprob"
    key = "delta_margin" if space == "margin" else "delta_g"
    band = band_margin if space == "margin" else B_LOGPROB_NATS

    # ── P0 collapse branch. ──────────────────────────────────────────────────
    dose_collapsed = {s: t["r_collapsed"] for s, t in dose_terms.items()}
    twin_collapsed = {s: t["r_collapsed"] for s, t in twin_terms.items()}
    dose_is = any(dose_collapsed.values())
    twin_is = any(twin_collapsed.values())
    verdict: str | None = None
    if dose_is or twin_is:
        if dose_is and twin_is:
            verdict = "saturation-collapse (schedule-family)"
        elif dose_is:
            verdict = "dose-accelerated saturation (schedule-family — negatives add dose)"
        else:
            verdict = "negatives-restrain (loss-competition-family, #471-style)"
        precedence = "P0"
    # ── P1 noise branch. Pair gaps are computed over SURVIVING (non-collapsed)
    # seeds only (§6.2 P0: "pair gaps are reported descriptively per surviving
    # seed only" — round-2 Claude Minor 3). When P0 did not fire no seed is
    # collapsed, so surviving == all and the P1 gate semantics are unchanged;
    # when P0 fired the gap is descriptive and a member with < 2 surviving
    # seeds reports None instead of a collapsed-read-contaminated number.
    surviving_dose = {s: t for s, t in dose_terms.items() if not t["r_collapsed"]}
    surviving_twin = {s: t for s, t in twin_terms.items() if not t["r_collapsed"]}
    gap_dose = _pair_seed_gap(surviving_dose, key)
    gap_twin = _pair_seed_gap(surviving_twin, key)
    gaps = [g for g in (gap_dose, gap_twin) if g is not None]
    pair_seed_gap = max(gaps) if gaps else None
    if verdict is None and pair_seed_gap is not None and pair_seed_gap > band:
        verdict = "indeterminate-for-noise"
        precedence = "P1"
    # ── P2 signed classes. ───────────────────────────────────────────────────
    d_signed = _seed_mean(dose_terms, key) - _seed_mean(twin_terms, key)
    if verdict is None:
        if d_signed < -band:
            verdict = "suppression (negatives subtract)"
        elif d_signed > band:
            verdict = "enhancement (negatives add beyond schedule)"
        else:
            verdict = "co-landing (schedule account)"
        precedence = "P2"

    return {
        "level": label,
        "governing_space": space,
        "space_switch": {
            "ceiling_compression_a": switch_a,
            "divergence_b": switch_b,
            "dose_seed_mean_g_logp": dose_glogp,
            "twin_seed_mean_g_logp": twin_glogp,
            "dose_seed_mean_dlogZ": dose_dlogz,
            "twin_seed_mean_dlogZ": twin_dlogz,
        },
        "band": band,
        "D_signed": d_signed,
        "pair_seed_gap": pair_seed_gap,
        "pair_seed_gap_per_member": {"dose": gap_dose, "twin": gap_twin},
        "collapse_states": {"dose": dose_collapsed, "twin": twin_collapsed},
        "verdict": verdict,
        "precedence": precedence,
        # §13 item 6: both logit sides reported separately alongside the margin.
        "dose_terminal": {
            s: {
                k: t[k]
                for k in (
                    "delta_g",
                    "delta_margin",
                    "delta_z_marker",
                    "delta_z_eos",
                    "g_logp",
                    "dlogZ",
                    "r_collapsed",
                    "emission_p",
                )
            }
            for s, t in dose_terms.items()
        },
        "twin_terminal": {
            s: {
                k: t[k]
                for k in (
                    "delta_g",
                    "delta_margin",
                    "delta_z_marker",
                    "delta_z_eos",
                    "g_logp",
                    "dlogZ",
                    "r_collapsed",
                    "emission_p",
                )
            }
            for s, t in twin_terms.items()
        },
    }


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- linear registered-rule pipeline; each §6.2 rule is one block
    ap = argparse.ArgumentParser(
        description="Task #622 off-pod analysis (registered §6.2 rules).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_622"))
    ap.add_argument("--i601-root", type=Path, default=Path("eval_results/issue_601"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s | %(message)s", stream=sys.stdout)

    out_path = (
        args.out if args.out is not None else args.slab_root / "analysis" / "classification.json"
    )
    dose_dir = args.slab_root / "dose_break"

    # ── Frozen-input loads. ───────────────────────────────────────────────────
    margin_band = recompute_margin_band(args.i601_root)
    band_margin = margin_band["band_logits"]
    references = load_601_references(args.i601_root)
    log.info(
        "margin band artifact-locked: %.3f logits (prose 2.18; mismatch=%s)",
        band_margin,
        margin_band["prose_mismatch"],
    )

    # ── Per-unit terminal reads + riders. ─────────────────────────────────────
    units: dict[str, dict] = {}
    input_shas: dict[str, str] = {}
    for _label, dose_slug, twin_slug in LEVELS:
        for slug in (dose_slug, twin_slug):
            for seed in SEEDS:
                cell_dir = dose_dir / f"{slug}_seed{seed}"
                traj_p = cell_dir / "trajectory.json"
                if not traj_p.exists():
                    # Graceful degradation (§4.4): a missing seed is REPORTED,
                    # never silently dropped — the noise branch widens.
                    log.warning("unit artifact missing: %s (single-seed fallback)", traj_p)
                    continue
                term = _terminal_source(json.loads(traj_p.read_text()))
                input_shas[str(traj_p)] = _sha256(traj_p)
                unit: dict = {"terminal": term, "transfer": transfer_fractions(term)}
                rowtype_p = cell_dir / "rowtype_ce.json"
                if rowtype_p.exists():
                    rowtype = json.loads(rowtype_p.read_text())
                    input_shas[str(rowtype_p)] = _sha256(rowtype_p)
                    # Wake-up applies ONLY to units WITH a negative channel —
                    # the twins (0 negatives) record an all-null neg series by
                    # construction, which the rule must not consume.
                    if rowtype.get("n_neg_rows", 0) > 0:
                        unit["wakeup"] = wakeup_state_for_seed(rowtype)
                    # §13 item 1: twins' positive-channel CE at extreme cycling
                    # (reporting rider — values only, no registered threshold).
                    pos = [v for v in rowtype.get("pos_marker_ce", []) if v is not None]
                    unit["pos_marker_ce_final"] = pos[-1] if pos else None
                    unit["pos_marker_ce_max_last20"] = max(pos[-20:]) if pos else None
                dense_p = cell_dir / "dense_trajectory.json"
                if dense_p.exists():
                    dense = json.loads(dense_p.read_text())
                    input_shas[str(dense_p)] = _sha256(dense_p)
                    unit["erosion_overlay"] = erosion_overlay(dense)
                cap_p = cell_dir / "capability_trajectory.json"
                if cap_p.exists():
                    cap = json.loads(cap_p.read_text())
                    input_shas[str(cap_p)] = _sha256(cap_p)
                    unit["capability"] = capability_flag(cap)
                units[f"{slug}_seed{seed}"] = unit

    # ── Per-level lattice. ────────────────────────────────────────────────────
    level_results = []
    for label, dose_slug, twin_slug in LEVELS:
        dose_terms = {
            s: units[f"{dose_slug}_seed{s}"]["terminal"]
            for s in SEEDS
            if f"{dose_slug}_seed{s}" in units
        }
        twin_terms = {
            s: units[f"{twin_slug}_seed{s}"]["terminal"]
            for s in SEEDS
            if f"{twin_slug}_seed{s}" in units
        }
        if not dose_terms or not twin_terms:
            level_results.append(
                {"level": label, "verdict": "NOT-RUN (artifacts missing)", "precedence": None}
            )
            continue
        res = classify_level(label, dose_terms, twin_terms, band_margin)
        single_seeded = len(dose_terms) < 2 or len(twin_terms) < 2
        if single_seeded and res["precedence"] == "P2":
            # §4.4 graceful degradation: single-seed level -> the noise branch
            # widens to indeterminate-unless-outside-band-by-both-remaining-reads.
            remaining = [
                abs(
                    t[("delta_margin" if res["governing_space"] == "margin" else "delta_g")]
                    - _seed_mean(
                        twin_terms,
                        "delta_margin" if res["governing_space"] == "margin" else "delta_g",
                    )
                )
                for t in dose_terms.values()
            ]
            if not all(r > res["band"] for r in remaining):
                res["verdict"] = "indeterminate-for-noise (single-seed widened)"
                res["precedence"] = "P1"
            res["single_seed_level"] = True
        # Wake-up state per level (all three dose cells regardless of verdict).
        wk = {
            s: units[f"{dose_slug}_seed{s}"]["wakeup"]
            for s in SEEDS
            if f"{dose_slug}_seed{s}" in units and "wakeup" in units[f"{dose_slug}_seed{s}"]
        }
        res["wakeup_per_seed"] = wk
        res["wakeup_level"] = wakeup_level_verdict(wk) if wk else "UNDECIDABLE_OR_MIXED"
        level_results.append(res)

    # ── Cross-level crash overlay (log-prob space; the #601 references are
    # nats — i=1's comparator is the reused T=113 value, single-seed, flagged).
    crash_overlay = []
    prev_mean = references["dense_200p1600n"]["delta_g_seed_mean"]
    prev_label = "#601 dense_200p1600n T=113 (single-seed, flagged)"
    for label, dose_slug, _twin in LEVELS:
        terms = {
            s: units[f"{dose_slug}_seed{s}"]["terminal"]
            for s in SEEDS
            if f"{dose_slug}_seed{s}" in units
        }
        if not terms:
            continue
        mean = _seed_mean(terms, "delta_g")
        crash_overlay.append(
            {
                "level": label,
                "terminal_seed_mean_delta_g": mean,
                "comparator": prev_label,
                "comparator_value": prev_mean,
                "crash": mean < prev_mean - B_LOGPROB_NATS,
            }
        )
        prev_mean, prev_label = mean, f"dose level {label}"

    # ── Loss-competition cross-level synthesis. ───────────────────────────────
    synthesis = synthesize_loss_competition(level_results)

    payload = {
        "schema": "i622_classification_v1",
        "registered_rules": "plan #622 §6.2 (v3, total lattice with explicit precedence)",
        "frozen_bands": {"logprob_nats": B_LOGPROB_NATS, "margin_logits": band_margin},
        "margin_band_artifact_lock": margin_band,
        "references_601": references,
        "levels": level_results,
        "crash_overlay_cross_level": crash_overlay,
        "loss_competition_synthesis": synthesis,
        "units": units,
        "input_paths_sha256": input_shas,
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    log.info("classification written -> %s", out_path)
    for r in level_results:
        log.info(
            "level %s: %s [space=%s, D=%s, gap=%s, wakeup=%s]",
            r["level"],
            r["verdict"],
            r.get("governing_space"),
            f"{r['D_signed']:.2f}" if r.get("D_signed") is not None else "n/a",
            f"{r['pair_seed_gap']:.2f}" if r.get("pair_seed_gap") is not None else "n/a",
            r.get("wakeup_level"),
        )
    log.info("synthesis: %s", synthesis)
    return 0


if __name__ == "__main__":
    sys.exit(main())

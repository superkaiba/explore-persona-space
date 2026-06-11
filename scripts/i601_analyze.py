#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" are intentional
"""Task #601 — CPU analysis (runs OFF-POD on the VM over committed JSONs).

Assembles the pre-registered classification lattice (plan §6) from the
pipeline outputs and writes ``eval_results/issue_601/analysis/classification.json``:

  - Phase-1 three-hypothesis classification (equilibrium / horizon / coupling)
    in the Phase-0a-selected primary space, with the exactly-one precedence
    rule and the no-call branch.
  - The registered matched-pair discriminator (schedule-matched @ step 32).
  - Phase-2 arrest dating + log-Z artifact flags + robustness sweeps.
  - Phase-3 source-differential contrast.
  - Phase-4 arrest on/off classification (4a; 4b cells when present).
  - Phase-0b clamp carry-over + the §7 kill-criteria summary.

Pure decision logic lives in ``neg_setpoint_601.analysis_lib`` (CPU-tested by
the implementation smoke); this script is I/O + assembly.

Usage:
    uv run python scripts/i601_analyze.py [--slab-root eval_results/issue_601]
        [--allow-partial]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i601.analyze")

ONPOLICY_ADMISSION_TOL_NATS = 2.0  # plan §4 Phase 2 — teacher-forced admission.

# Phase-1 arm slug -> lattice role.
ARM_ROLES = {
    "ratio4to1_100p400n": "quarter",
    "ratio4to1_400p1600n": "double",
    "ratio4to1_100p400n_T128": "matched",
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load(path: Path, *, allow_partial: bool) -> dict | None:
    if not path.exists():
        if allow_partial:
            log.warning("missing input (allow-partial): %s", path)
            return None
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _terminal_value(traj: dict, space: str) -> float:
    """Terminal source value from an on-policy trajectory.json, in `space`."""
    term = max(traj["checkpoints"], key=lambda c: c["frac"])
    ss = term["source_self"]
    if space == "logp":
        return float(ss["delta_g_mean"])
    margin = (ss["z_marker_g_mean"] - ss["z_eos_g_mean"]) - (
        ss["z_marker_b_mean"] - ss["z_eos_b_mean"]
    )
    return float(margin)


def _dense_series(dense: dict, space: str) -> tuple[list[int], list[float]]:
    """(steps, source values) from a dense_trajectory.json, in `space`."""
    key = "delta_g" if space == "logp" else "delta_margin"
    pts = sorted(
        ((c["step"], c["source_mean"][key]) for c in dense["checkpoints"] if c["step"] is not None),
        key=lambda t: t[0],
    )
    return [int(s) for s, _ in pts], [float(v) for _, v in pts]


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- linear per-phase JSON assembly; decision logic lives (tested) in analysis_lib
    ap = argparse.ArgumentParser(
        description="Task #601 classification analysis (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_601"))
    ap.add_argument("--committed-472-root", type=Path, default=Path("eval_results/issue_472"))
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--out-path", type=Path, default=None)
    args = ap.parse_args(argv)
    out_path = args.out_path or (args.slab_root / "analysis" / "classification.json")

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.neg_setpoint_601 import (
        CELLS_601,
        cell_by_slug,
    )
    from explore_persona_space.experiments.neg_setpoint_601.analysis_lib import (
        arrest_step,
        classify_phase1,
        classify_phase4_arrest,
        logz_artifact,
        phase3_contrast,
        robustness_sweep,
    )

    ap_load = lambda p: _load(p, allow_partial=args.allow_partial)  # noqa: E731

    # ── Phase 0: space decision + clamp. ─────────────────────────────────────
    endpoint = ap_load(args.slab_root / "phase0" / "endpoint_reads.json")
    gate = ap_load(args.slab_root / "phase0" / "phase0_gate.json")
    primary_space_full = (
        (endpoint or {}).get("space_calibration", {}).get("primary_space", "logp_with_margin_upper")
    )
    # classify_phase1 takes one space: "margin" when Phase 0 escalated, else
    # logp (the upper-branch margin co-read is reported alongside either way).
    space = "margin" if primary_space_full == "margin" else "logp"
    margin_refs_raw = (endpoint or {}).get("margin_references", {})
    margin_refs = {lv: rec["margin_mean"] for lv, rec in margin_refs_raw.items()}
    margin_tol = max([rec["tolerance_margin"] for rec in margin_refs_raw.values()] or [1.0])

    # ── Phase 1 arm terminals (both spaces; classified in the primary). ──────
    arm_terminals: dict[str, list[float]] = {}
    arm_terminals_margin: dict[str, list[float]] = {}
    missing: list[str] = []
    for slug, role in ARM_ROLES.items():
        spec = cell_by_slug(slug)
        vals, vals_m = [], []
        for seed in spec.seeds:
            traj = ap_load(args.slab_root / spec.phase / f"{slug}_seed{seed}" / "trajectory.json")
            if traj is None:
                missing.append(f"{slug}_seed{seed}")
                continue
            vals.append(_terminal_value(traj, "logp"))
            vals_m.append(_terminal_value(traj, "margin"))
        if vals:
            arm_terminals[role] = vals if space == "logp" else vals_m
            arm_terminals_margin[role] = vals_m
    # Anchor (reused #472 or the retrain fallback).
    anchor_vals, anchor_vals_m = [], []
    for seed in (42, 137):
        # Prefer this task's Phase-0 on-policy re-read (same eval code path as
        # the new arms); the committed parent trajectory is the cross-check.
        reread = ap_load(
            args.slab_root
            / "phase0"
            / "onpolicy_recheck"
            / f"c472_anchor_seed{seed}"
            / "trajectory.json"
        )
        if reread is not None:
            anchor_vals.append(_terminal_value(reread, "logp"))
            anchor_vals_m.append(_terminal_value(reread, "margin"))
    if anchor_vals:
        arm_terminals["anchor"] = anchor_vals if space == "logp" else anchor_vals_m
        arm_terminals_margin["anchor"] = anchor_vals_m

    # ── Matched-arm series (dense teacher-forced + on-policy fracs). ─────────
    matched_series: dict[int, tuple[list[int], list[float]]] = {}
    matched_spec = cell_by_slug("ratio4to1_100p400n_T128")
    for seed in matched_spec.seeds:
        d = ap_load(
            args.slab_root
            / matched_spec.phase
            / f"ratio4to1_100p400n_T128_seed{seed}"
            / "dense_trajectory.json"
        )
        t = ap_load(
            args.slab_root
            / matched_spec.phase
            / f"ratio4to1_100p400n_T128_seed{seed}"
            / "trajectory.json"
        )
        if d is None:
            continue
        steps, vals = _dense_series(d, space)
        if t is not None:
            for ck in t["checkpoints"]:
                if ck.get("step") is None:
                    continue
                s = int(ck["step"])
                if s not in steps:
                    ss = ck["source_self"]
                    v = (
                        float(ss["delta_g_mean"])
                        if space == "logp"
                        else float(
                            (ss["z_marker_g_mean"] - ss["z_eos_g_mean"])
                            - (ss["z_marker_b_mean"] - ss["z_eos_b_mean"])
                        )
                    )
                    steps.append(s)
                    vals.append(v)
        order = sorted(range(len(steps)), key=lambda i: steps[i])
        matched_series[seed] = ([steps[i] for i in order], [vals[i] for i in order])

    phase1 = None
    required_roles = {"quarter", "anchor", "double", "matched"}
    if required_roles.issubset(arm_terminals) and matched_series:
        phase1 = classify_phase1(
            arm_terminals=arm_terminals,
            matched_series_by_seed=matched_series,
            space=space,
            margin_refs=margin_refs or None,
            margin_tol=margin_tol if space == "margin" else None,
        )
        # Margin co-read alongside a logp-primary classification (rule: report
        # the pair everywhere; space disagreement is the saturation signature).
        if space == "logp" and margin_refs:
            phase1["margin_coread"] = classify_phase1(
                arm_terminals=arm_terminals_margin,
                matched_series_by_seed=matched_series,
                space="margin",
                margin_refs=margin_refs,
                margin_tol=margin_tol,
            )["verdicts"]
    else:
        missing.append("phase1-classification-inputs")

    # ── Phase 2: arrest dating + log-Z + on-policy admission. ────────────────
    phase2: dict[str, dict] = {}
    for spec in CELLS_601:
        if spec.phase != "phase2":
            continue
        for seed in spec.seeds:
            key = f"{spec.slug}_seed{seed}"
            d = ap_load(args.slab_root / spec.phase / key / "dense_trajectory.json")
            t = ap_load(args.slab_root / spec.phase / key / "trajectory.json")
            if d is None:
                missing.append(key)
                continue
            steps, dg = _dense_series(d, "logp")
            _, dz = _dense_series(d, "margin")
            dz_marker = [
                float(c["source_mean"]["delta_z_marker"])
                for c in sorted(
                    (c for c in d["checkpoints"] if c["step"] is not None),
                    key=lambda c: c["step"],
                )
            ]
            admission = None
            if t is not None:
                diffs = []
                for ck in t["checkpoints"]:
                    s = ck.get("step")
                    if s is None:
                        continue
                    tf = next((v for st, v in zip(steps, dg, strict=True) if st == int(s)), None)
                    if tf is not None:
                        diffs.append(abs(float(ck["source_self"]["delta_g_mean"]) - tf))
                admission = {
                    "anchor_abs_diffs": diffs,
                    "admitted": bool(diffs)
                    and all(x <= ONPOLICY_ADMISSION_TOL_NATS for x in diffs),
                    "tol_nats": ONPOLICY_ADMISSION_TOL_NATS,
                }
            phase2[key] = {
                "expected_T": spec.expected_steps,
                "arrest_step": arrest_step(steps, dg),
                "arrest_robustness": robustness_sweep(steps, dg),
                "logz_artifact": logz_artifact(steps, dg, dz_marker),
                "onpolicy_admission": admission,
                "terminal_delta_g": dg[-1] if dg else None,
                "terminal_delta_margin": dz[-1] if dz else None,
            }

    # ── Phase 3: source-differential contrast. ───────────────────────────────
    phase3 = None
    p3_per_seed: dict[int, dict] = {}
    p3_spec = cell_by_slug("negonly_0p800n")
    for seed in p3_spec.seeds:
        d = ap_load(
            args.slab_root / p3_spec.phase / f"negonly_0p800n_seed{seed}" / "dense_trajectory.json"
        )
        if d is None:
            continue
        term = max((c for c in d["checkpoints"] if c["step"] is not None), key=lambda c: c["step"])
        reads = term["reads"]
        bys = [p for p in d["bystander_panel"] if p in reads]
        src = d["source"]
        q_keys = list(reads[src].keys())

        def _mean(persona_list, field, _reads=reads, _q=q_keys):
            vals = [_reads[p][q][field] for p in persona_list for q in _q]
            return float(sum(vals) / len(vals))

        p3_per_seed[seed] = {
            "dz_marker_source": _mean([src], "delta_z_marker"),
            "dz_marker_bystander_mean": _mean(bys, "delta_z_marker"),
            "dz_eos_source": _mean([src], "z_eos_g") - _mean([src], "z_eos_b"),
            "dz_eos_bystander_mean": _mean(bys, "z_eos_g") - _mean(bys, "z_eos_b"),
        }
    if len(p3_per_seed) == len(p3_spec.seeds):
        phase3 = phase3_contrast(p3_per_seed)
        phase3["per_seed"] = p3_per_seed
    else:
        missing.append("phase3-inputs")

    # ── Phase 4: arrest on/off. ───────────────────────────────────────────────
    phase4: dict[str, dict] = {}
    for spec in CELLS_601:
        if spec.phase != "phase4":
            continue
        for seed in spec.seeds:
            key = f"{spec.slug}_seed{seed}"
            band = ap_load(args.slab_root / spec.phase / key / "inloop_band_trajectory.json")
            if band is None:
                if not spec.conditional:
                    missing.append(key)
                continue
            phase4[key] = classify_phase4_arrest(band["steps"], band["delta_nats"])
    # Per-cell seed-pooled call: both seeds must agree for a clean call.
    phase4_calls: dict[str, str] = {}
    for spec in CELLS_601:
        if spec.phase != "phase4":
            continue
        cls = {
            phase4[f"{spec.slug}_seed{s}"]["classification"]
            for s in spec.seeds
            if f"{spec.slug}_seed{s}" in phase4
        }
        if not cls:
            continue
        phase4_calls[spec.slug] = cls.pop() if len(cls) == 1 else "ambiguous"

    # ── Kill-criteria summary (plan §7). ─────────────────────────────────────
    clamp = (endpoint or {}).get("clamp_read", {})
    kill = {
        "h_equilibrium_feedback_mechanism": {
            "clamp_present": clamp.get("clamp_present"),
            "note": (
                "no clamp kills the FEEDBACK-MECHANISM claim only; the ratio-set-point "
                "phenomenology is judged by the Phase-1 level rules (plan §4 item 4)"
            ),
        },
        "phase4_rig_switch": phase4_calls,
        "logz_reframe": any(v["logz_artifact"]["artifact"] for v in phase2.values())
        if phase2
        else None,
    }

    payload = {
        "schema_version": "i601_classification_v1",
        "primary_space_decision": primary_space_full,
        "classification_space": space,
        "phase0_gate_pass": (gate or {}).get("pass"),
        "anchor_reuse_ok": (gate or {}).get("anchor_reuse_ok"),
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
        "phase4": phase4,
        "phase4_calls": phase4_calls,
        "clamp_read": clamp,
        "kill_criteria": kill,
        "missing_inputs": sorted(set(missing)),
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(
        "classification written → %s (call=%s, missing=%d)",
        out_path,
        (phase1 or {}).get("call"),
        len(set(missing)),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

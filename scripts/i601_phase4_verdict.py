#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" are intentional
"""Task #601 — Phase-4 bridge arrest classification → conditional 4b routing sentinel.

Runs CPU-only on the pod AFTER the main sweep (``i601_launch.sh`` step
p6_phase4a_verdict). Round-4 structure (concern
phase4-bridge-attn-only-attribution): #471's posonly rig was ALL-LINEAR r=32 @
lr 5e-6 — not attn-only as the plan assumed via the ideas doc — so the
registry now carries TWO unconditional Phase-4 bridge cells and this script
classifies BOTH:

  posonly_alllinear_lr5e6   the TRUE single-variable #471 lr-bridge
  posonly_attn_lr5e6        the plan's two-variable pair cell (matches neither rig)

Per cell: load the in-loop band trajectories (both seeds), classify arrest
on/off per the plan §4 registered bands
(``analysis_lib.classify_phase4_arrest``), seed-pool with the agreement rule
(both seeds agree → that call; disagree → "ambiguous"). The routing call in
``<slab-root>/phase4/phase4a_verdict.json`` gates ONLY the remaining
conditional 4b factor cell (``posonly_attn_lr1e5``):

Round-8 input fallback (p6 crash fix): the T=13 bridge cells never produce
``inloop_band_trajectory.json`` — ``MarkerBandStopCallback`` disabled itself
when ``max_steps(13) < min_steps(20)``, so no probe ever fired and the file
write (per-probe + records-gated train-end flush) never ran. Per (cell, seed)
this script now loads, in order:

  1. ``inloop_band_trajectory.json`` with ≥1 record (the primary input);
  2. ``rowtype_ce.json`` via ``analysis_lib.derive_band_from_rowtype`` — the
     per-row-type CE probe is NOT min_steps-gated and reads the same
     live-gauge ΔG construct on the same probe rows (delta = base CE - step
     CE), so the §4 bands apply unchanged;
  3. ``dense_trajectory.json`` — CLASSIC staged gauge (alpha/r), NOT
     comparable to the live-gauge §4 bands: never classified against them;
     the seed is recorded as "ambiguous" with an explicit gauge caveat;
  4. nothing usable → FileNotFoundError (both bridge cells are unconditional
     main-sweep members).

  any cell "non-arrest"        → call: non-arrest, dispatch_4b: true — the
                                 launch driver runs ``dispatch --cells
                                 phase4b`` (itself re-gated on this sentinel)
                                 to isolate adapter scope at parent LR.
  all cells "arrest"           → call: arrest, dispatch_4b: false.
  otherwise                    → call: ambiguous, dispatch_4b: false — 4b is
                                 uninformative and skipped; the per-cell
                                 classifications are recorded so the analyzer
                                 / clean-result reports the rig-localization
                                 question open (plan §7: ambiguous ≠ arrest;
                                 the Phase-4 kill does NOT fire on ambiguous).

The sentinel is ALWAYS written (every branch is a recorded, reportable
outcome); missing bridge-cell inputs fail loud (both cells are unconditional
members of the main sweep).

Usage:
    uv run python scripts/i601_phase4_verdict.py [--slab-root eval_results/issue_601]
        [--cells posonly_attn_lr5e6,posonly_alllinear_lr5e6]
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

log = logging.getLogger("i601.phase4_verdict")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Task #601 Phase-4 bridge arrest verdict writer (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_601"))
    ap.add_argument(
        "--cells",
        default=None,
        help=(
            "CSV of UNCONDITIONAL Phase-4 bridge slugs to classify; default = every "
            "unconditional phase4 cell in the registry."
        ),
    )
    ap.add_argument("--out-path", type=Path, default=None)
    args = ap.parse_args(argv)
    out_path = args.out_path or (args.slab_root / "phase4" / "phase4a_verdict.json")

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=phase4a_verdict] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.neg_setpoint_601 import (
        CELLS_601,
        PHASE4_BRIDGE_ATTRIBUTION,
        cell_by_slug,
    )
    from explore_persona_space.experiments.neg_setpoint_601.analysis_lib import (
        classify_phase4_arrest,
        derive_band_from_rowtype,
    )

    if args.cells:
        specs = [cell_by_slug(tok.strip()) for tok in args.cells.split(",") if tok.strip()]
    else:
        specs = [c for c in CELLS_601 if c.phase == "phase4" and not c.conditional]
    for spec in specs:
        if spec.phase != "phase4" or spec.conditional:
            raise ValueError(f"{spec.slug!r} is not an unconditional Phase-4 bridge cell.")
    if not specs:
        raise ValueError("zero unconditional Phase-4 bridge cells resolved")

    per_cell: dict[str, dict] = {}
    calls: dict[str, str] = {}
    for spec in specs:
        per_seed: dict[str, dict] = {}
        for seed in spec.seeds:
            cell_dir = args.slab_root / spec.phase / f"{spec.slug}_seed{seed}"
            band_path = cell_dir / "inloop_band_trajectory.json"
            rowtype_path = cell_dir / "rowtype_ce.json"
            dense_path = cell_dir / "dense_trajectory.json"

            # 1. Primary: the in-loop band trajectory (needs >=1 record — a
            #    round-8 callback writes the file even when no probe fired,
            #    so an empty-records file falls through to the derivation).
            band = json.loads(band_path.read_text()) if band_path.exists() else None
            if band and band.get("steps"):
                rec = classify_phase4_arrest(band["steps"], band["delta_nats"])
                rec["trajectory_source"] = "inloop_band_trajectory"
                per_seed[str(seed)] = rec
                continue

            # 2. Fallback: live-gauge ΔG derived from the per-row-type CE
            #    probe (NOT min_steps-gated; same construct, same gauge —
            #    see derive_band_from_rowtype). The §4 bands apply unchanged.
            derived = None
            if rowtype_path.exists():
                try:
                    derived = derive_band_from_rowtype(json.loads(rowtype_path.read_text()))
                except ValueError as exc:
                    log.warning("[%s seed%d] rowtype_ce unusable: %s", spec.slug, seed, exc)
            if derived is not None:
                log.info(
                    "[%s seed%d] band trajectory missing/empty at %s — classified from "
                    "rowtype_ce-derived live-gauge ΔG (%d steps, base CE %.3f)",
                    spec.slug,
                    seed,
                    band_path,
                    len(derived["steps"]),
                    derived["pos_marker_ce_base"],
                )
                rec = classify_phase4_arrest(derived["steps"], derived["delta_nats"])
                rec["trajectory_source"] = derived["trajectory_source"]
                rec["n_pos_rows"] = derived["n_pos_rows"]
                rec["pos_marker_ce_base"] = derived["pos_marker_ce_base"]
                rec["gauge_note"] = derived["gauge_note"]
                per_seed[str(seed)] = rec
                continue

            # 3. Last resort: dense_trajectory.json exists but is CLASSIC
            #    staged gauge (alpha/r) — NOT comparable to the live-gauge §4
            #    bands. Record ambiguous with the caveat; never fabricate an
            #    arrest call off the wrong gauge.
            if dense_path.exists():
                log.warning(
                    "[%s seed%d] only dense_trajectory.json available (classic staged "
                    "gauge) — recording 'ambiguous' with a gauge caveat, NOT classifying "
                    "against the live-gauge §4 bands.",
                    spec.slug,
                    seed,
                )
                per_seed[str(seed)] = {
                    "classification": "ambiguous",
                    "trajectory_source": "dense_trajectory_classic_gauge",
                    "gauge_caveat": (
                        "dense_trajectory.json is a staged classic alpha/r read; the §4 "
                        "arrest bands are calibrated on live-gauge in-loop ΔG — "
                        "classifying across gauges would fabricate an arrest call"
                    ),
                }
                continue

            raise FileNotFoundError(
                f"bridge trajectory inputs missing for {spec.slug} seed {seed}: neither "
                f"{band_path} (>=1 record) nor {rowtype_path} (usable pos-side series) nor "
                f"{dense_path} exists — both unconditional Phase-4 cells must complete "
                f"before the 4b routing verdict can be written."
            )
        # Seed-pooled call (same agreement rule as i601_analyze.py): both seeds
        # must agree for a clean call; disagreement → ambiguous.
        classes = {v["classification"] for v in per_seed.values()}
        calls[spec.slug] = classes.pop() if len(classes) == 1 else "ambiguous"
        per_cell[spec.slug] = {"per_seed": per_seed, "call": calls[spec.slug]}

    # Routing call over the bridge PAIR: any non-arrest → 4b factorization is
    # informative (locate WHICH variable flips the switch); all arrest → the
    # switch lies outside {lr, adapter scope}; otherwise ambiguous → 4b skipped.
    if any(c == "non-arrest" for c in calls.values()):
        call = "non-arrest"
    elif all(c == "arrest" for c in calls.values()):
        call = "arrest"
    else:
        call = "ambiguous"
    dispatch_4b = call == "non-arrest"

    payload = {
        "schema_version": "i601_phase4a_verdict_v3",
        "cells": sorted(calls),
        "per_cell": per_cell,
        "calls": calls,
        "call": call,
        "dispatch_4b": dispatch_4b,
        "bridge_attribution": {s: PHASE4_BRIDGE_ATTRIBUTION[s] for s in sorted(calls)},
        "rule": (
            "plan §4 per-cell bands: non-arrest = ΔG >= 6 nats by step 13 AND last-3-step "
            "slope >= 0.3; arrest = flat (slope < 0.2 from step <= 4) at <= 4 nats; else "
            "ambiguous. Seed-pooled per cell: both seeds agree -> call, else ambiguous. "
            "Routing over the bridge pair (round 4, concern "
            "phase4-bridge-attn-only-attribution): any cell non-arrest -> dispatch 4b "
            "(posonly_attn_lr1e5 only); all arrest -> arrest; else ambiguous -> 4b "
            "uninformative, skipped, reported open. Input per seed (round 8, v3): "
            "inloop_band_trajectory.json when it has >=1 record, else live-gauge ΔG "
            "derived from rowtype_ce.json (delta = pos_marker_ce_base - pos_marker_ce; "
            "same construct/gauge/probe rows — T=13 cells never fired the min_steps=20 "
            "band probe), else dense_trajectory.json -> ambiguous with a gauge caveat "
            "(classic staged gauge, never classified against the live-gauge bands); "
            "per_seed.trajectory_source records which input was used."
        ),
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out_path)
    log.info(
        "phase4 bridge verdict written → %s (call=%s, dispatch_4b=%s, per_cell=%s)",
        out_path,
        call,
        dispatch_4b,
        calls,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

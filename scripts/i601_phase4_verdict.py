#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" are intentional
"""Task #601 — Phase-4a arrest classification → conditional Phase-4b routing sentinel.

Runs CPU-only on the pod AFTER the main sweep (``i601_launch.sh`` step
p6_phase4a_verdict). Loads the 4a bridge cell's in-loop band trajectories
(both seeds), classifies arrest on/off per the plan §4 registered bands
(``analysis_lib.classify_phase4_arrest``), seed-pools with the agreement rule
(both seeds agree → that call; disagree → "ambiguous"), and writes
``<slab-root>/phase4/phase4a_verdict.json``:

  call == "non-arrest"          → dispatch_4b: true  — the launch driver runs
                                   the conditional 4b factorization cells
                                   (``dispatch --cells phase4b``, itself gated
                                   on this sentinel).
  call in {"arrest","ambiguous"} → dispatch_4b: false — 4b is uninformative
                                   and skipped; the classification is recorded
                                   so the analyzer / clean-result reports the
                                   rig-localization question open (plan §7:
                                   ambiguous ≠ arrest; the Phase-4 kill does
                                   NOT fire on ambiguous).

The sentinel is ALWAYS written (every branch is a recorded, reportable
outcome); missing 4a inputs fail loud.

Usage:
    uv run python scripts/i601_phase4_verdict.py [--slab-root eval_results/issue_601]
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

PHASE4A_CELL = "posonly_attn_lr5e6"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Task #601 Phase-4a arrest verdict writer (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_601"))
    ap.add_argument("--cell", default=PHASE4A_CELL)
    ap.add_argument("--out-path", type=Path, default=None)
    args = ap.parse_args(argv)
    out_path = args.out_path or (args.slab_root / "phase4" / "phase4a_verdict.json")

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=phase4a_verdict] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.neg_setpoint_601 import cell_by_slug
    from explore_persona_space.experiments.neg_setpoint_601.analysis_lib import (
        classify_phase4_arrest,
    )

    spec = cell_by_slug(args.cell)
    if spec.phase != "phase4" or spec.conditional:
        raise ValueError(f"{args.cell!r} is not the unconditional Phase-4a bridge cell.")

    per_seed: dict[str, dict] = {}
    for seed in spec.seeds:
        band_path = (
            args.slab_root
            / spec.phase
            / f"{spec.slug}_seed{seed}"
            / ("inloop_band_trajectory.json")
        )
        if not band_path.exists():
            raise FileNotFoundError(
                f"4a in-loop band trajectory missing at {band_path} — the Phase-4a cell "
                f"must complete before the 4b routing verdict can be written."
            )
        band = json.loads(band_path.read_text())
        per_seed[str(seed)] = classify_phase4_arrest(band["steps"], band["delta_nats"])

    # Seed-pooled call (same agreement rule as i601_analyze.py): both seeds
    # must agree for a clean call; disagreement → ambiguous (→ 4b skipped).
    classes = {v["classification"] for v in per_seed.values()}
    call = classes.pop() if len(classes) == 1 else "ambiguous"
    dispatch_4b = call == "non-arrest"

    payload = {
        "schema_version": "i601_phase4a_verdict_v1",
        "cell": spec.slug,
        "per_seed": per_seed,
        "call": call,
        "dispatch_4b": dispatch_4b,
        "rule": (
            "plan §4: non-arrest = ΔG >= 6 nats by step 13 AND last-3-step slope >= 0.3; "
            "arrest = flat (slope < 0.2 from step <= 4) at <= 4 nats; else ambiguous. "
            "Seed-pooled: both seeds agree -> call, else ambiguous. 4b dispatches ONLY "
            "on non-arrest; arrest/ambiguous -> 4b uninformative, skipped, reported open."
        ),
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out_path)
    log.info(
        "phase4a verdict written → %s (call=%s, dispatch_4b=%s, per_seed=%s)",
        out_path,
        call,
        dispatch_4b,
        {s: v["classification"] for s, v in per_seed.items()},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

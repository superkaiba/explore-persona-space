# em-dash + Qwen marker " ※" intentional
#!/usr/bin/env python3
"""Task #504 Phase 0 — anchor pick from smoke trajectories (plan §4.1).

CPU-only. Reads the 3 smoke trajectories produced by the dispatcher (
``c504_smoke_r{4,8,16}_seed42``) and writes phase0_calibration.json with the
pinned (chosen_rank, chosen_alpha, chosen_checkpoint_fraction) per the plan's
"latest-in-band + midpoint tie-break + low-rank tie-break" rule.

Usage:
    uv run python scripts/i504_phase_phase0_pick.py \
        --slab-root eval_results/issue_504 \
        --out-path eval_results/issue_504/phase0_calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i504.phase_phase0_pick")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_504"))
    ap.add_argument(
        "--out-path", type=Path, default=Path("eval_results/issue_504/phase0_calibration.json")
    )
    ap.add_argument("--smoke-seed", type=int, default=42)
    ap.add_argument("--sentinel-path", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=phase0_pick] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        PHASE0_SMOKE_SLUGS,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase0 import (
        pick_anchor_from_smoke,
        write_phase0_artifact,
    )

    smoke_trajs: dict[str, dict] = {}
    for slug in PHASE0_SMOKE_SLUGS:
        p = args.slab_root / f"{slug}_seed{args.smoke_seed}" / "trajectory.json"
        if not p.exists():
            raise FileNotFoundError(
                f"smoke trajectory missing at {p} — Phase 0 smoke {slug} must complete first."
            )
        smoke_trajs[slug] = json.loads(p.read_text())
        log.info(
            "[load] %s trajectory: %d checkpoints", slug, len(smoke_trajs[slug]["checkpoints"])
        )

    pick = pick_anchor_from_smoke(smoke_trajs)
    write_phase0_artifact(pick, args.out_path)

    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        args.sentinel_path.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": "epm:progress",
                    "version": 1,
                    "task_id": 504,
                    "phase": "phase0_pick",
                    "by": "i504_phase_phase0_pick",
                    "ts": datetime.now(UTC).isoformat(),
                    "note": json.dumps(
                        {
                            "verdict": pick.get("verdict"),
                            "chosen_rank": pick.get("chosen_rank"),
                            "chosen_alpha": pick.get("chosen_alpha"),
                            "chosen_checkpoint_fraction": pick.get("chosen_checkpoint_fraction"),
                            "source_delta_g_at_pick_nats": pick.get("source_delta_g_at_pick_nats"),
                            "source_emission_at_pick": pick.get("source_emission_at_pick"),
                            "out_path": str(args.out_path),
                        }
                    ),
                },
                indent=2,
            )
        )

    log.info(
        "[phase=phase0_pick] verdict=%s, chosen_rank=%s, chosen_alpha=%s, chosen_frac=%s",
        pick.get("verdict"),
        pick.get("chosen_rank"),
        pick.get("chosen_alpha"),
        pick.get("chosen_checkpoint_fraction"),
    )
    if pick.get("verdict") != "pass":
        log.error("[phase=phase0_pick] FAIL — see smoke_table in %s", args.out_path)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

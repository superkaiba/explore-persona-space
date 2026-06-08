# em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #472 Phase 5 — analyze subprocess entrypoint (plan §6).

Runs the geometry/count/placement regression + figures from the per-cell
trajectory.json + base_panel.json + centroids. Subprocess for clean import
isolation (matplotlib + statsmodels).

Usage:
    uv run python scripts/i472_phase_analyze.py \
        --slab-root eval_results/issue_472 \
        --base-panel-path eval_results/issue_472/base_panel.json \
        --centroids-dir data/issue_472 --figures-dir figures/issue_472 --seeds 42,137
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

log = logging.getLogger("i472.phase_analyze")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_472"))
    ap.add_argument(
        "--base-panel-path", type=Path, default=Path("eval_results/issue_472/base_panel.json")
    )
    ap.add_argument("--centroids-dir", type=Path, default=Path("data/issue_472"))
    ap.add_argument("--figures-dir", type=Path, default=Path("figures/issue_472"))
    ap.add_argument("--seeds", default="42,137")
    ap.add_argument("--sentinel-path", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=analyze] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import run_analysis

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summary = run_analysis(
        slab_root=args.slab_root,
        base_panel_path=args.base_panel_path,
        figures_dir=args.figures_dir,
        centroids_dir=args.centroids_dir,
        seeds=seeds,
    )

    out_path = args.slab_root / "analyze_summary.json"
    if not out_path.exists():
        raise RuntimeError(f"analyze exited but {out_path} missing — silent failure.")

    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        args.sentinel_path.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": "epm:progress",
                    "version": 1,
                    "task_id": 472,
                    "phase": "analyze",
                    "by": "i472_phase_analyze",
                    "ts": datetime.now(UTC).isoformat(),
                    "note": json.dumps(
                        {
                            "analyze_summary_path": str(out_path),
                            "verdict": summary.get("barrier_bubble_verdict", {}).get("call"),
                        }
                    ),
                },
                indent=2,
            )
        )
    log.info("[phase=analyze] verdict=%s", summary.get("barrier_bubble_verdict", {}).get("call"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

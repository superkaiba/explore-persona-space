# em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #472 Phase 0.5 — base-model centroids subprocess entrypoint.

Subprocess-isolated (loads the base HF model on GPU) per the dispatcher's
teardown discipline. Writes data/issue_472/centroids_L{10,15,20}.pt + the
name-keyed cosine matrices.

Usage:
    uv run python scripts/i472_phase_centroids.py --bank-path data/issue_472/persona_bank.json
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

log = logging.getLogger("i472.phase_centroids")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/issue_472"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sentinel-path", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=centroids] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        build_centroids,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )

    bank = load_persona_bank(args.bank_path)
    log.info("Loaded persona bank: %d personas", len(bank))
    written = build_centroids(bank, out_dir=args.out_dir, device=args.device)
    summary = {
        "n_personas": len(bank),
        "centroid_paths": {str(k): str(v) for k, v in written.items()},
    }

    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        args.sentinel_path.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": "epm:progress",
                    "version": 1,
                    "task_id": 472,
                    "phase": "centroids",
                    "by": "i472_phase_centroids",
                    "ts": datetime.now(UTC).isoformat(),
                    "note": json.dumps(summary),
                },
                indent=2,
            )
        )
    log.info("Centroids written: %s", summary["centroid_paths"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Task #505 §5.7 — wrapper script to build persona-vectors centroids at L{7,14,21,27}.

Loads ``.env``, pulls the #472 persona bank from the local cache, calls
``leave_one_out_505.build_pv_centroids.build_pv_centroids`` to extract +
persist + cosine-matrix the four new layer bundles.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
if not os.environ.get("HF_TOKEN"):
    raise RuntimeError("HF_TOKEN missing — load_dotenv() found no .env")

from explore_persona_space.experiments.leave_one_out_505 import (  # noqa: E402
    SIMILARITY_LAYERS_TO_BUILD,
)
from explore_persona_space.experiments.leave_one_out_505.build_pv_centroids import (  # noqa: E402
    build_pv_centroids,
)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description="Build persona-vectors centroids for #505 (L7/14/21/27)."
    )
    p.add_argument(
        "--persona-bank",
        type=Path,
        default=Path(os.environ.get("EPM_I472_DATA_ROOT", "data/issue_472")) / "persona_bank.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("EPM_DATA_ROOT", "data/issue_505")) / "centroids_pv",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for the base-model forward pass (use cpu for CPU-only dry-runs).",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=list(SIMILARITY_LAYERS_TO_BUILD),
    )
    args = p.parse_args(argv)
    if not args.persona_bank.exists():
        print(f"persona bank missing at {args.persona_bank}", file=sys.stderr)
        return 1
    # NEVER raw-`json.loads` this file: #472 publishes persona_bank.json as a
    # STRUCTURED payload `{schema_version, ..., personas: {name: prompt}, ...}`.
    # The canonical loader validates `schema_version == 'i472_v1'` and returns
    # the inner `personas` map directly. The dispatcher loader was fixed in
    # ce2bea8a2 (Codex r4 caught this sibling still raw); see
    # tests/test_issue505_panel_coverage.py for the regression test class.
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )

    bank = load_persona_bank(args.persona_bank)
    written = build_pv_centroids(
        persona_bank=bank,
        layers=tuple(args.layers),
        out_dir=args.out_dir,
        device=args.device,
    )
    print(json.dumps({"written": {layer: str(p) for layer, p in written.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

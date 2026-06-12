#!/usr/bin/env python3
"""Issue #545 — one-shot state repair for the round-18 smoke contamination.

The pre-round-19 pod smoke (``issue545_sweep.py --phase p1 --rows marker
--seeds 0 --smoke``) wrote its artifacts into the PRODUCTION output root,
where they satisfied production's resume guards: ``manifest_p1.json`` gained
a ``marker_primary_seed0`` entry (production then logged "skip completed
cell" and kept the 4-step smoke adapter — band_stop_result.json reads
``stopped_in_band: false, global_step: 4``) and ``cells/base_panel/`` existed
with only marker+capability (the bare ``exists()`` guard then skipped the
full panel: no base rates for any judged column). Round 19 fixes the guards
(smoke-output isolation + per-file base-panel completeness); THIS helper
repairs the on-pod STATE so the fixed dispatcher retrains/refills:

1. drops the contaminated cell's entry from ``manifest_<phase>.json`` (the
   dispatcher's ``done_cells`` check) so the cell retrains;
2. deletes the contaminated ``cells/<cell>/`` eval dir so every eval file
   regenerates (the eval driver is per-(column, context) idempotent and
   would otherwise keep the smoke reads);
3. deletes any leftover LOCAL adapter dir for the cell (normally already
   reaped by the post-upload cleanup);
4. LEAVES the partial ``cells/base_panel/`` in place — the round-19
   completeness resume fills exactly the missing files.

Idempotent: a second run prints "nothing to remove" and exits 0. Prints every
path / manifest entry it removes. Run on the pod from the repo root with the
same EPM_OUTPUT_ROOT the sweep uses, and WITHOUT I545_SMOKE_OUTPUT set (the
repair targets the production root by definition).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger("issue545_repair")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def repair(cell: str, phase: str) -> list[str]:
    """Remove the contaminated cell's resume-guard artifacts; return actions."""
    from explore_persona_space.experiments.behavior_testbed_545 import (
        adapters_root,
        cells_dir,
        output_root,
        smoke_output_active,
    )

    if smoke_output_active():
        raise SystemExit(
            "I545_SMOKE_OUTPUT=1 is set — the repair targets the PRODUCTION root; unset it."
        )
    actions: list[str] = []

    manifest_path = output_root() / f"manifest_{phase}.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        kept = [m for m in manifest if m.get("cell") != cell]
        if len(kept) != len(manifest):
            manifest_path.write_text(json.dumps(kept, indent=1))
            actions.append(
                f"removed manifest entry {cell!r} from {manifest_path} "
                f"({len(manifest)} -> {len(kept)} entries)"
            )

    cell_dir = cells_dir() / cell
    if cell_dir.exists():
        shutil.rmtree(cell_dir)
        actions.append(f"deleted contaminated eval dir {cell_dir}")

    adapter_dir = adapters_root() / cell
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
        actions.append(f"deleted leftover local adapter dir {adapter_dir}")

    return actions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #545 round-19 state repair (smoke-contaminated resume guards)"
    )
    parser.add_argument(
        "--cell",
        default="marker_primary_seed0",
        help="Contaminated cell id (default: the round-18 smoke cell)",
    )
    parser.add_argument("--phase", default="p1", help="Manifest phase to repair (default: p1)")
    args = parser.parse_args()

    actions = repair(args.cell, args.phase)
    if not actions:
        logger.info("[repair] nothing to remove — state already repaired for %s", args.cell)
    for a in actions:
        logger.info("[repair] %s", a)
    logger.info(
        "[repair] cells/base_panel left IN PLACE by design — the per-file completeness "
        "resume (issue545_sweep.py --phase p1) fills exactly the missing column files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

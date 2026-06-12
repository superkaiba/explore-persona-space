#!/usr/bin/env python3
"""Issue #545 — one-shot state repair for the round-18 smoke contamination.

The pre-round-19 pod smoke (``issue545_sweep.py --phase p1 --rows marker
--seeds 0 --smoke``) wrote its artifacts into the PRODUCTION output root,
where they satisfied production's resume guards: ``manifest_p1.json`` gained
a ``marker_primary_seed0`` entry (production then logged "skip completed
cell" and kept the 4-step smoke adapter — band_stop_result.json reads
``stopped_in_band: false, global_step: 4``) and ``cells/base_panel/`` existed
with only marker+capability (the bare ``exists()`` guard then skipped the
full panel: no base rates for any judged column). Round 19 fixed the guards
(smoke-output isolation + per-file base-panel completeness); THIS helper
repairs the on-pod STATE so the fixed dispatcher retrains/refills:

1. drops the contaminated cell's entry from ``manifest_<phase>.json`` (the
   dispatcher's ``done_cells`` check) so the cell retrains;
2. deletes the contaminated ``cells/<cell>/`` eval dir so every eval file
   regenerates (the eval driver is per-(column, context) idempotent and
   would otherwise keep the smoke reads);
3. deletes any leftover LOCAL adapter dir for the cell (normally already
   reaped by the post-upload cleanup);
4. deletes the ENTIRE ``cells/base_panel/`` dir (round 20): its
   ``marker__default.json`` / ``capability__default.json`` AND the
   ``completions__marker__default.json`` gen product are 4-probe SMOKE
   artifacts — the gen-phase skip would otherwise re-derive
   ``marker__default.json`` from the stale 4-row completions. The per-column
   completeness resume rebuilds the whole panel at production probe size.

Path safety (round 20): ``--cell`` must be a bare directory name (no path
separators, no ``..``, not absolute) and every rmtree target is resolved and
asserted to sit STRICTLY under its root before deletion.

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


def _validate_cell(cell: str) -> None:
    """Reject any ``--cell`` that is not a bare directory name (fail loud)."""
    if (
        not cell
        or cell in (".", "..")
        or "/" in cell
        or "\\" in cell
        or ".." in Path(cell).parts
        or Path(cell).is_absolute()
    ):
        raise SystemExit(
            f"Refusing unsafe --cell {cell!r}: must be a bare cell directory name "
            "(no path separators, no '..', not absolute)"
        )


def _safe_rmtree(target: Path, root: Path, what: str, actions: list[str]) -> None:
    """rmtree ``target`` only after asserting it resolves STRICTLY under ``root``."""
    root_r = root.resolve()
    target_r = target.resolve()
    if target_r == root_r or not target_r.is_relative_to(root_r):
        raise SystemExit(f"Refusing to delete {target_r} — not strictly under {root_r} ({what})")
    if target_r.exists():
        shutil.rmtree(target_r)
        actions.append(f"deleted {what} {target_r}")


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
    _validate_cell(cell)
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

    _safe_rmtree(cells_dir() / cell, cells_dir(), "contaminated eval dir", actions)
    _safe_rmtree(adapters_root() / cell, adapters_root(), "leftover local adapter dir", actions)
    # Round 20: the base panel's kept files are 4-probe smoke products
    # (including the completions__marker__default.json gen product the
    # gen-phase skip would re-derive from) — purge the whole dir so the
    # completeness resume rebuilds it at production probe size.
    _safe_rmtree(cells_dir() / "base_panel", cells_dir(), "smoke-sized base panel dir", actions)

    return actions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #545 round-19/20 state repair (smoke-contaminated resume guards)"
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
        "[repair] cells/base_panel removed (round 20) — the per-file completeness resume "
        "(issue545_sweep.py --phase p1) rebuilds the WHOLE panel at production probe size"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

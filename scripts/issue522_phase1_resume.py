"""#522 Phase 1 disk-full crash recovery: resume the probe-count sweep.

The 2026-06-10 run crashed at cell 27/27 when the HF download of
last_prompt__layer27.pt hit a disk-full condition (0 MB free). The
per-cell checkpoint persisted 4800/6480 rows = the first 20 fully
complete cells. This driver re-runs ONLY the cells whose persisted row
count is short, then merges with the checkpointed rows and writes the
final (non-partial) payload using the sweep module's own aggregate /
plateau functions.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import issue511_probe_count_sweep as mod

logger = logging.getLogger("i522.resume")

OUT_PATH = Path(
    "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-522/"
    "eval_results/issue_522/probe_count_sweep_results.json"
)
TMP_OUT = OUT_PATH.with_name("probe_count_sweep_resume_missing_cells.json")
BACKUP = OUT_PATH.with_name("probe_count_sweep_results.partial_backup.json")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    partial = json.loads(OUT_PATH.read_text())
    assert partial.get("partial") is True, (
        "expected a partial checkpoint; refusing to clobber a final result"
    )
    shutil.copy2(OUT_PATH, BACKUP)
    logger.info("backed up partial (%d rows) to %s", len(partial["rows"]), BACKUP.name)

    arm = partial["arm"]
    epochs = tuple(int(e) for e in partial["epochs"])
    cells_all = mod.build_cell_list()

    rows_by_cell: dict[str, list[dict]] = {}
    for row in partial["rows"]:
        rows_by_cell.setdefault(row["cell_id"], []).append(row)

    complete_rows: list[dict] = []
    missing_cells = []
    for cell in cells_all:
        expected = len(cell.n_grid) * cell.r * len(epochs)
        have = len(rows_by_cell.get(cell.cell_id, []))
        if have == expected:
            complete_rows.extend(rows_by_cell[cell.cell_id])
        else:
            logger.info("cell %s incomplete: %d/%d rows -> re-run", cell.cell_id, have, expected)
            missing_cells.append(cell)

    logger.info(
        "%d cells complete (%d rows kept), %d cells to re-run",
        len(cells_all) - len(missing_cells),
        len(complete_rows),
        len(missing_cells),
    )
    if not missing_cells:
        logger.info("nothing to re-run; finalizing from checkpoint alone")
        new_rows: list[dict] = []
    else:
        payload_missing = mod.sweep(
            cells=missing_cells,
            arm=arm,
            epochs=epochs,
            out_path=TMP_OUT,
            checkpoint_every=1,  # crash insurance: persist after every cell this time
        )
        new_rows = payload_missing["rows"]

    all_rows = complete_rows + new_rows
    aggregates = mod.aggregate(all_rows)
    payload = {
        "schema_version": 1,
        "git_sha": mod._git_sha(),
        "env": mod._env_versions(),
        "started_at": partial["started_at"],
        "finished_at": datetime.now(UTC).isoformat(),
        "wall_seconds": None,  # split across two processes; see resumed_from
        "resumed_from": {
            "checkpoint_at": partial["checkpoint_at"],
            "checkpoint_rows": len(complete_rows),
            "rerun_cells": [c.cell_id for c in missing_cells],
            "crash_cause": "disk-full during HF download of last_prompt__layer27.pt",
        },
        "arm": arm,
        "epochs": list(epochs),
        "cells_tracked": [
            {"cell_id": c.cell_id, "n_grid": list(c.n_grid), "r": int(c.r)} for c in cells_all
        ],
        "rows": all_rows,
        "aggregates": aggregates,
        "plateau_verdict": mod.compute_plateau(aggregates),
        "plateau_verdict_glog": mod.compute_plateau_glog(aggregates),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("wrote FINAL %s (%d rows total)", OUT_PATH, len(all_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())

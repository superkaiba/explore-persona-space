"""Phase 3 merger — combine per-shard G_partial_*.json into G_matrix.json.

Issue #406 plan v9 §4 Phase 3 merge step.

Reads per-shard roll-ups at eval_results/issue_406/cross_eval/G_partial_*.json
AND the per-cell atomic writes at eval_results/issue_406/cross_eval/per_cell/
(used to recover any cells whose shard roll-up was lost mid-run). Emits
the full 20x20 G_matrix.json with per-cell rate, n_emit, n_total, plus
the diagonal-sanity report (which conds pass G[i,i] >= 0.7).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from explore_persona_space.experiments.i406_conditions import CONDITIONS

logger = logging.getLogger("i406.phase3.merge")

CROSS_DIR = Path("eval_results/issue_406/cross_eval")
PER_CELL_DIR = CROSS_DIR / "per_cell"
DIAGONAL_THRESHOLD = 0.7


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    cids = [c.cid for c in CONDITIONS]
    n_cond = len(cids)

    # Combine per-shard roll-ups first (the fast path).
    g_combined: dict[str, dict[str, dict]] = {ci: {} for ci in cids}
    shard_files = sorted(CROSS_DIR.glob("G_partial_*.json"))
    if not shard_files:
        logger.warning(
            "No G_partial_*.json roll-ups found under %s; falling back to per-cell files only.",
            CROSS_DIR,
        )
    for shard_path in shard_files:
        shard_payload = json.loads(shard_path.read_text())
        for ci, inner in shard_payload.items():
            for cj, cell in inner.items():
                g_combined[ci][cj] = cell
        logger.info("Merged shard %s (%d outer-i)", shard_path.name, len(shard_payload))

    # Backfill missing cells from per-cell atomic writes (resume-safe).
    backfilled = 0
    if PER_CELL_DIR.exists():
        for cell_path in PER_CELL_DIR.glob("G_*__*.json"):
            cell_payload = json.loads(cell_path.read_text())
            ci, cj = cell_payload["T_i"], cell_payload["T_j"]
            if cj not in g_combined.get(ci, {}):
                g_combined[ci][cj] = {
                    "n_emit": cell_payload["n_emit"],
                    "n_total": cell_payload["n_total"],
                    "rate": cell_payload["rate"],
                }
                backfilled += 1
    if backfilled:
        logger.info("Backfilled %d cells from per-cell atomic writes", backfilled)

    # Completeness check.
    expected_cells = n_cond * n_cond
    actual_cells = sum(len(inner) for inner in g_combined.values())
    if actual_cells < expected_cells:
        missing: list[tuple[str, str]] = []
        for ci in cids:
            for cj in cids:
                if cj not in g_combined.get(ci, {}):
                    missing.append((ci, cj))
        raise RuntimeError(
            f"G matrix incomplete: have {actual_cells}/{expected_cells} cells. "
            f"Missing first 10: {missing[:10]}. Resume the failed Phase 3 shard "
            f"via scripts/i406_phase3_dispatch.sh --resume before merging."
        )

    # Diagonal sanity report (does each LoRA implant the marker on its own shape?).
    diagonal_passed: list[str] = []
    diagonal_failed: list[dict] = []
    for ci in cids:
        cell = g_combined[ci][ci]
        if cell["rate"] >= DIAGONAL_THRESHOLD:
            diagonal_passed.append(ci)
        else:
            diagonal_failed.append({"cid": ci, "rate": cell["rate"]})

    logger.info(
        "Diagonal: %d/%d passed (>= %.2f). Failed: %s",
        len(diagonal_passed),
        n_cond,
        DIAGONAL_THRESHOLD,
        diagonal_failed,
    )

    payload = {
        "schema_version": "v9",
        "n_conditions": n_cond,
        "diagonal_threshold": DIAGONAL_THRESHOLD,
        "conditions": [{"cid": c.cid, "class": c.cls, "name": c.name} for c in CONDITIONS],
        "G": g_combined,
        "diagonal_passed": diagonal_passed,
        "diagonal_failed": diagonal_failed,
        "git_commit": _git_commit_hash(),
    }
    out_path = CROSS_DIR / "G_matrix.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Wrote %s (full 20x20 G matrix; %d passed diagonal)", out_path, len(diagonal_passed)
    )


if __name__ == "__main__":
    main()

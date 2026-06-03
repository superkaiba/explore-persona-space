"""Phase 4 merger — combine per-shard G_partial_*.json into G_logprob_matrix.json.

Issue #460 plan v3 §4.5 (Phase 5 reads the merged matrix).

Reads per-shard roll-ups at eval_results/issue_460/cross_eval/G_partial_*.json
AND the per-cell atomic writes at eval_results/issue_460/cross_eval/per_cell/.
Emits a single 16x16 G_logprob_matrix.json with per-cell
{g_logprob, b_logprob, delta_g, emission_recompute_rate, n_probes}.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

from explore_persona_space.experiments.i406_conditions import CONDITIONS

logger = logging.getLogger("i460.phase4.merge")

CROSS_DIR = Path("eval_results/issue_460/cross_eval")
PER_CELL_DIR = CROSS_DIR / "per_cell"
OUT_PATH = CROSS_DIR / "G_logprob_matrix.json"
DIAGONAL_DELTA_FAIL_THRESHOLD = 5.0  # H3 gate


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    cids = [c.cid for c in CONDITIONS]
    n_cond = len(cids)

    # Combine per-shard roll-ups (the fast path).
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

    # Per-cell fallback for any missing cells.
    n_filled_per_cell = 0
    missing: list[tuple[str, str]] = []
    for ci in cids:
        for cj in cids:
            if cj in g_combined[ci]:
                continue
            cell_path = PER_CELL_DIR / f"G_{ci}__{cj}.json"
            if cell_path.exists() and cell_path.stat().st_size > 0:
                cell = json.loads(cell_path.read_text())
                g_combined[ci][cj] = {
                    "g_logprob": cell["g_logprob"],
                    "b_logprob": cell["b_logprob"],
                    "delta_g": cell["delta_g"],
                    "emission_recompute_rate": cell["emission_recompute_rate"],
                }
                n_filled_per_cell += 1
            else:
                missing.append((ci, cj))
    if n_filled_per_cell:
        logger.info("Filled %d cells from per_cell/ fallback.", n_filled_per_cell)

    if missing:
        raise RuntimeError(
            f"G_logprob_matrix has {len(missing)} missing cells; "
            f"first 5: {missing[:5]}. Re-run failed shards with --resume."
        )

    diagonals = {ci: g_combined[ci][ci] for ci in cids}
    failed_diag = {
        ci: d["delta_g"]
        for ci, d in diagonals.items()
        if d["delta_g"] <= DIAGONAL_DELTA_FAIL_THRESHOLD
    }
    payload = {
        "schema_version": "i460_v1",
        "n_conditions": n_cond,
        "conditions": cids,
        "diagonal_delta_fail_threshold": DIAGONAL_DELTA_FAIL_THRESHOLD,
        "G": g_combined,
        "diagonals": {ci: diagonals[ci]["delta_g"] for ci in cids},
        "diagonal_failed": list(failed_diag.keys()),
        "git_commit": _git_commit_hash(),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    if failed_diag:
        logger.warning(
            "H3 diagonal-implant gate FAILED on %d/%d conds (delta_g <= %.2f): %s",
            len(failed_diag),
            n_cond,
            DIAGONAL_DELTA_FAIL_THRESHOLD,
            failed_diag,
        )
    logger.info(
        "Merged %d x %d G_logprob matrix -> %s (diagonal_failed=%d)",
        n_cond,
        n_cond,
        OUT_PATH,
        len(failed_diag),
    )


if __name__ == "__main__":
    main()

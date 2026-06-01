"""Phase 4 merger (#462) — combine per-shard G_partial_*_ep{N}.json into
G_logprob_matrix_ep{N}.json for ONE epoch level.

Issue #462. Adapts i460_phase4_merge.py with a single change: the merger
is parameterized on ``--adapter-epoch N`` and produces a per-epoch
matrix. Phase 5 reads each ``G_logprob_matrix_ep{N}.json`` independently
and stitches the level-wise curves.

Reads:
  - eval_results/issue_462/cross_eval/G_partial_*_ep{N}.json (shard rollups)
  - eval_results/issue_462/cross_eval/per_cell_ep{N}/G_<ci>__<cj>.json (atomic per-cell)

Writes:
  - eval_results/issue_462/cross_eval/G_logprob_matrix_ep{N}.json

The H3 diagonal-implant gate (delta_g > 5.0) is reported per epoch as a
warning ONLY; the runner does NOT abort the pipeline on a missed gate
because the WHOLE POINT of the per-epoch sweep is to discover at WHICH
epoch the diagonal first crosses the threshold.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

from explore_persona_space.experiments.i406_conditions import CONDITIONS

logger = logging.getLogger("i462.phase4.merge")

CROSS_DIR = Path("eval_results/issue_462/cross_eval")
DIAGONAL_DELTA_FAIL_THRESHOLD = 5.0
VALID_EPOCHS = {1, 2, 3, 5}


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
    ap.add_argument(
        "--adapter-epoch",
        type=int,
        required=True,
        choices=sorted(VALID_EPOCHS),
        help="Which per-epoch shard rollups to merge.",
    )
    args = ap.parse_args(argv)
    epoch = args.adapter_epoch

    cids = [c.cid for c in CONDITIONS]
    n_cond = len(cids)
    per_cell_dir = CROSS_DIR / f"per_cell_ep{epoch}"
    out_path = CROSS_DIR / f"G_logprob_matrix_ep{epoch}.json"

    g_combined: dict[str, dict[str, dict]] = {ci: {} for ci in cids}
    shard_files = sorted(CROSS_DIR.glob(f"G_partial_*_ep{epoch}.json"))
    if not shard_files:
        logger.warning(
            "No G_partial_*_ep%d.json roll-ups under %s; falling back to per-cell files only.",
            epoch,
            CROSS_DIR,
        )
    for shard_path in shard_files:
        shard_payload = json.loads(shard_path.read_text())
        for ci, inner in shard_payload.items():
            for cj, cell in inner.items():
                g_combined[ci][cj] = cell
        logger.info("Merged shard %s (%d outer-i)", shard_path.name, len(shard_payload))

    n_filled_per_cell = 0
    missing: list[tuple[str, str]] = []
    for ci in cids:
        for cj in cids:
            if cj in g_combined[ci]:
                continue
            cell_path = per_cell_dir / f"G_{ci}__{cj}.json"
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
        logger.info("Filled %d cells from per_cell_ep%d/ fallback.", n_filled_per_cell, epoch)

    if missing:
        raise RuntimeError(
            f"G_logprob_matrix_ep{epoch} has {len(missing)} missing cells; "
            f"first 5: {missing[:5]}. Re-run failed shards with --resume."
        )

    diagonals = {ci: g_combined[ci][ci] for ci in cids}
    failed_diag = {
        ci: d["delta_g"]
        for ci, d in diagonals.items()
        if d["delta_g"] <= DIAGONAL_DELTA_FAIL_THRESHOLD
    }
    payload = {
        "schema_version": "i462_v1",
        "adapter_epoch": epoch,
        "n_conditions": n_cond,
        "conditions": cids,
        "diagonal_delta_fail_threshold": DIAGONAL_DELTA_FAIL_THRESHOLD,
        "G": g_combined,
        "diagonals": {ci: diagonals[ci]["delta_g"] for ci in cids},
        "diagonal_failed": list(failed_diag.keys()),
        "git_commit": _git_commit_hash(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    # Report-only (NOT fail-loud). The whole point of per-epoch sweep is
    # discovering AT WHICH epoch the diagonal crosses the gate.
    if failed_diag:
        logger.warning(
            "ep=%d diagonal-implant gate not yet met on %d/%d conds (delta_g <= %.2f): %s",
            epoch,
            len(failed_diag),
            n_cond,
            DIAGONAL_DELTA_FAIL_THRESHOLD,
            failed_diag,
        )
    logger.info(
        "Merged %dx%d G_logprob matrix ep=%d -> %s (diagonal_below_thr=%d)",
        n_cond,
        n_cond,
        epoch,
        out_path,
        len(failed_diag),
    )


if __name__ == "__main__":
    main()

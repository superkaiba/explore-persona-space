"""Phase 4 merger — per-(arm, checkpoint) merged matrix for #474.

Issue #474 plan v3 §2 — forked from ``scripts/i460_phase4_merge.py`` to
emit one merged matrix per (arm, epoch) pair. The dispatcher invokes this
once per (arm, ckpt) — Phase 5 reads all 8 merged matrices.

Output: ``eval_results/issue_474/cross_eval/{arm}_ep{N}/G_logprob_matrix.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

from explore_persona_space.experiments.i406_conditions import CONDITIONS

logger = logging.getLogger("i474.phase4.merge")

CROSS_DIR = Path("eval_results/issue_474/cross_eval")
DIAGONAL_DELTA_FAIL_THRESHOLD = 5.0  # H3 gate inherited from #460


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
    ap.add_argument("--arm", required=True, choices=["pos", "loc"])
    ap.add_argument("--checkpoint-epoch", type=int, required=True)
    args = ap.parse_args(argv)

    arm_ep_subdir = f"{args.arm}_ep{args.checkpoint_epoch}"
    out_dir = CROSS_DIR / arm_ep_subdir
    per_cell_dir = out_dir / "per_cell"
    out_path = out_dir / "G_logprob_matrix.json"

    cids = [c.cid for c in CONDITIONS]
    n_cond = len(cids)

    # Combine per-shard roll-ups (the fast path).
    g_combined: dict[str, dict[str, dict]] = {ci: {} for ci in cids}
    shard_files = sorted(out_dir.glob("G_partial_*.json"))
    if not shard_files:
        logger.warning(
            "No G_partial_*.json roll-ups under %s; falling back to per-cell files only.",
            out_dir,
        )
    for shard_path in shard_files:
        shard_payload = json.loads(shard_path.read_text())
        for ci, inner in shard_payload.items():
            for cj, cell in inner.items():
                g_combined[ci][cj] = cell
        logger.info("Merged shard %s (%d outer-i)", shard_path.name, len(shard_payload))

    # Per-cell fallback.
    n_filled = 0
    missing: list[tuple[str, str]] = []
    for ci in cids:
        for cj in cids:
            if cj in g_combined[ci]:
                continue
            cell_path = per_cell_dir / f"G_{args.arm}_ep{args.checkpoint_epoch}_{ci}__{cj}.json"
            if cell_path.exists() and cell_path.stat().st_size > 0:
                cell = json.loads(cell_path.read_text())
                g_combined[ci][cj] = {
                    "g_logprob": cell["g_logprob"],
                    "b_logprob": cell["b_logprob"],
                    "delta_g": cell["delta_g"],
                    "emission_recompute_rate": cell["emission_recompute_rate"],
                    "kl_post_response_slot": cell.get("kl_post_response_slot"),
                }
                n_filled += 1
            else:
                missing.append((ci, cj))
    if n_filled:
        logger.info("Filled %d cells from per_cell/ fallback.", n_filled)
    if missing:
        raise RuntimeError(
            f"G_logprob_matrix arm={args.arm} ep={args.checkpoint_epoch} "
            f"has {len(missing)} missing cells; first 5: {missing[:5]}. "
            "Re-run failed shards with --resume."
        )

    diagonals = {ci: g_combined[ci][ci] for ci in cids}
    failed_diag = {
        ci: d["delta_g"]
        for ci, d in diagonals.items()
        if d["delta_g"] <= DIAGONAL_DELTA_FAIL_THRESHOLD
    }
    payload = {
        "schema_version": "i474_v1",
        "arm": args.arm,
        "checkpoint_epoch": args.checkpoint_epoch,
        "n_conditions": n_cond,
        "conditions": cids,
        "diagonal_delta_fail_threshold": DIAGONAL_DELTA_FAIL_THRESHOLD,
        "G": g_combined,
        "diagonals": {ci: diagonals[ci]["delta_g"] for ci in cids},
        "diagonal_failed": list(failed_diag.keys()),
        "git_commit": _git_commit_hash(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    if failed_diag:
        logger.warning(
            "H3 diagonal gate FAILED on %d/%d conds (delta_g <= %.2f): %s",
            len(failed_diag),
            n_cond,
            DIAGONAL_DELTA_FAIL_THRESHOLD,
            failed_diag,
        )
    logger.info(
        "arm=%s ep=%d merged %d x %d -> %s (diagonal_failed=%d)",
        args.arm,
        args.checkpoint_epoch,
        n_cond,
        n_cond,
        out_path,
        len(failed_diag),
    )


if __name__ == "__main__":
    main()

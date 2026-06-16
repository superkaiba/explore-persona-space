"""Issue #651 — off-pod analysis driver (CPU, runs on the VM, 0 GPU).

Loads the per-cell shift tensors (eval_results/issue_651/shifts/<cell>.pt,
produced by the extract phase) and computes the §6.5 primary deliverables:

  eval_results/issue_651/q1_context_invariance/<behavior>.json   (Q1 per behavior)
  eval_results/issue_651/q2_cross_behavior/cross_behavior_cosine_matrix.json (Q2)
  eval_results/issue_651/seed_ceiling/<behavior>.json            (within-cell ceiling)
  eval_results/issue_651/variance/decomposition.json             (variance decomp)

The seed ceiling is computed FRESH from this task's seed-42 + seed-1042 tensors
(NEVER #552's 0.975/0.982 concentration — plan §14.1). Refusal is read but
excluded from the Q2 headline (the expected-null sanity row). emnc is kept
labeled and never pooled with contrastive em.

Reads BOTH slot (delta_v) and mean-resp (delta_v_mean_resp); the mean-resp read
is PRIMARY for the generative behaviors (em, sycophancy) per the #551 decision,
slot is the diagnostic. The driver computes both and records which is primary
per behavior.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("issue651_analysis")

# mean-resp is the PRIMARY read for the generative behaviors; slot for the rest.
_MEAN_RESP_PRIMARY = frozenset({"em", "sycophancy", "emnc"})


def _repo_root() -> Path:
    import subprocess

    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())


def _load_cells(shift_dir: Path):
    """Load every <cell>.pt -> {cell_id: payload['shifts']} + parsed cell meta."""
    import torch

    from explore_persona_space.experiments.issue_651 import parse_cell_spec

    out = {}
    for pt in sorted(shift_dir.glob("*.pt")):
        cell_id = pt.stem
        try:
            cell = parse_cell_spec(cell_id)
        except ValueError:
            logger.warning("skipping unparseable tensor file %s", pt.name)
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        out[cell_id] = (cell, payload["shifts"])
    return out


def _read_key(behavior: str) -> str:
    return "delta_v_mean_resp" if behavior in _MEAN_RESP_PRIMARY else "delta_v"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-reps", type=int, default=1000, help="Null-distribution reps.")
    parser.add_argument("--cell-read", choices=["u1", "mean"], default="u1")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )

    from explore_persona_space.experiments.issue_651 import (
        NULL_CHECK_BEHAVIOR,
    )
    from explore_persona_space.experiments.issue_651 import (
        analysis as i651,
    )

    repo_root = _repo_root()
    shift_dir = repo_root / "eval_results" / "issue_651" / "shifts"
    cells = _load_cells(shift_dir)
    if not cells:
        raise RuntimeError(f"no per-cell shift tensors found in {shift_dir} -- run extract first")
    logger.info("[phase=analysis_load] loaded %d cells", len(cells))

    # Index by (behavior, seed): {behavior: {seed: {cid: read_vector}}}.
    by_bs: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for _cell_id, (cell, shifts) in cells.items():
        key = _read_key(cell.behavior)
        try:
            read = i651.cell_read_vector(shifts, key=key, cell_read=args.cell_read)
        except KeyError:
            # mean-resp may be absent for a slot-only read; fall back to delta_v.
            read = i651.cell_read_vector(shifts, key="delta_v", cell_read=args.cell_read)
        by_bs.setdefault(cell.behavior, {}).setdefault(cell.seed, {})[cell.cid] = read

    # --- Seed ceiling per behavior (cells present at BOTH seeds 42 + 1042) ---
    ceiling_dir = repo_root / "eval_results" / "issue_651" / "seed_ceiling"
    ceiling_dir.mkdir(parents=True, exist_ok=True)
    seed_ceiling_median: dict[str, float] = {}
    for behavior, by_seed in by_bs.items():
        if 42 in by_seed and 1042 in by_seed:
            sc = i651.seed_ceiling_per_cell(by_seed[42], by_seed[1042])
            seed_ceiling_median[behavior] = sc["median"]
            (ceiling_dir / f"{behavior}.json").write_text(json.dumps(sc, indent=2))
            logger.info(
                "[phase=analysis_ceiling] %s seed-ceiling median=%.4f (n=%d cells)",
                behavior,
                sc["median"],
                sc["n_cells"],
            )
        else:
            logger.info(
                "[phase=analysis_ceiling] %s has no 2-seed cells -> no ceiling "
                "(marker/fact/em/syc expected; refusal/emnc single-seed)",
                behavior,
            )

    # --- Q1 per behavior (use seed 42 as the canonical context set) ---
    q1_dir = repo_root / "eval_results" / "issue_651" / "q1_context_invariance"
    q1_dir.mkdir(parents=True, exist_ok=True)
    behavior_u1: dict[str, np.ndarray] = {}
    for behavior, by_seed in by_bs.items():
        per_context = by_seed.get(42) or next(iter(by_seed.values()))
        if len(per_context) < 2:
            logger.info("[phase=analysis_q1] %s has <2 contexts -> skip Q1", behavior)
            continue
        q1 = i651.q1_context_invariance(per_context, n_reps=args.n_reps)
        ceil = seed_ceiling_median.get(behavior)
        verdict = (
            i651.q1_verdict(q1, ceil)
            if ceil is not None
            else {"verdict": "no_ceiling", "note": "no 2-seed ceiling for this behavior"}
        )
        out = {
            "behavior": behavior,
            "is_null_check_row": behavior == NULL_CHECK_BEHAVIOR,
            "primary_read": _read_key(behavior),
            "q1": {k: v for k, v in q1.items() if k != "U1"},
            "verdict": verdict,
        }
        (q1_dir / f"{behavior}.json").write_text(json.dumps(out, indent=2))
        behavior_u1[behavior] = np.asarray(q1["U1"], dtype=np.float32)
        logger.info(
            "[phase=analysis_q1] %s top_share=%.3f null_p95=%.3f verdict=%s",
            behavior,
            q1["top_share_norm_weighted"],
            q1["sign_flip_null_p95"],
            verdict.get("verdict"),
        )

    # --- Q2 cross-behavior matrix (headline behaviors only; refusal excluded) ---
    q2_dir = repo_root / "eval_results" / "issue_651" / "q2_cross_behavior"
    q2_dir.mkdir(parents=True, exist_ok=True)
    headline = {b: u1 for b, u1 in behavior_u1.items() if b != NULL_CHECK_BEHAVIOR and b != "emnc"}
    if len(headline) >= 2:
        q2 = i651.q2_cross_behavior_matrix(headline, seed_ceiling_median, n_reps=args.n_reps)
        q2["verdict"] = i651.q2_verdict(q2)
        (q2_dir / "cross_behavior_cosine_matrix.json").write_text(json.dumps(q2, indent=2))
        logger.info(
            "[phase=analysis_q2] behaviors=%s verdict=%s",
            q2["behaviors"],
            q2["verdict"]["verdict"],
        )
    else:
        logger.info("[phase=analysis_q2] <2 headline behaviors -> skip Q2")

    # --- Variance decomposition (seed-42 reads, all behaviors incl. labels) ---
    var_dir = repo_root / "eval_results" / "issue_651" / "variance"
    var_dir.mkdir(parents=True, exist_ok=True)
    cell_reads = {}
    for behavior, by_seed in by_bs.items():
        per_context = by_seed.get(42) or next(iter(by_seed.values()))
        for cid, read in per_context.items():
            cell_reads[(behavior, cid)] = read
    var = i651.variance_decomposition(cell_reads)
    (var_dir / "decomposition.json").write_text(json.dumps(var, indent=2))
    logger.info(
        "[phase=analysis_variance] shared=%.3f behavior=%.3f context=%.3f (n=%d)",
        var["shared_frac"],
        var["behavior_frac"],
        var["context_frac"],
        var["n_cells"],
    )

    logger.info("[phase=analysis_done] all deliverables written under eval_results/issue_651/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

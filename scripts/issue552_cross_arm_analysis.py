#!/usr/bin/env python3
"""#552 Step 10 (OFF-POD, VM) — cross-arm shift-geometry comparison table.

Reads the benign arm's per-cell SVD JSONs (this issue) plus the parent #521
EM + marker per-cell SVD JSONs (git, reused — no re-run), and the benign
per-question shift tensors (HF -> ``eval_results/issue_552/shifts``), then
writes ``eval_results/issue_552/cross_arm/summary.json`` carrying:

  1. Per-cell comparison table over all three arms x variants x seeds:
     mean/median cos(Delta-v, U1), s_top1_frac, ||M||_F = sqrt(sum sigma_i^2),
     null p95/p99.
  2. Within-arm and cross-arm |cos(U1, U1')| pairwise tables per variant
     (within-benign 3 pairs; benign x EM 9; benign x marker 9; EM x marker 9),
     against the 0.033 random floor (p95 |cos| for random 3584-dim vectors).
  3. Per-persona split-half reliability for the benign cells (odd/even
     question split of the per-question tensors; Spearman-Brown corrected)
     plus attenuation-corrected per-persona cos-to-U1.
  4. The pre-registered §3/§6.3 verdict inputs: validity precondition flags
     (sign-flip-null clearance + median split-half >= 0.5) and the
     confirmation / falsification / equivocal classification on the `same`
     variant. Interpretation stays with the analyzer — this script only
     computes and records.

Run (VM, after pod termination)::

    uv run python scripts/issue552_cross_arm_analysis.py
    uv run python scripts/issue552_cross_arm_analysis.py --skip-split-half  # tensors not staged

Inputs are fail-loud: a missing benign SVD JSON or (without --skip-split-half)
a missing per-question tensor raises with the exact remediation command.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from itertools import combinations
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SEEDS = (42, 137, 256)
VARIANTS = ("same", "base", "on_policy")
ARMS = ("benign", "em", "marker")

# p95 of |cos| between random unit vectors in R^3584 (parent #521 floor;
# E|cos| ~ 0.013, p95 ~ 0.033).
RANDOM_COS_FLOOR_P95 = 0.033

# Pre-registered thresholds (plan §3 / §6.3) — bound to the `same` variant.
CONFIRMATION_MEAN_COS_MAX = 0.85
CONFIRMATION_TOP_SHARE_MAX = 0.50
FALSIFICATION_MEAN_COS_MIN = 0.90
FALSIFICATION_TOP_SHARE_MIN = 0.50
SPLIT_HALF_VALIDITY_FLOOR = 0.5


def _load_cell(svd_dir: Path, variant: str, arm: str, seed: int) -> dict:
    p = svd_dir / f"{variant}_{arm}_seed{seed}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"per-cell SVD JSON missing: {p}. For benign cells, stage "
            f"eval_results/issue_552/svd/ from the issue-552 branch; parent cells "
            f"live at eval_results/issue_521/svd/ on main."
        )
    return json.loads(p.read_text())


def _fro_norm(singular_values: list[float]) -> float:
    s = np.asarray(singular_values, dtype=np.float64)
    return float(np.sqrt(np.sum(s**2)))


def _abs_cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(abs(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))


def _split_half_per_persona(pt_path: Path) -> dict[str, dict[str, float | int | None]]:
    """Odd/even-question split-half reliability per persona from a shifts .pt.

    Returns {persona: {"r_half": cos(mean_even, mean_odd),
                       "r_full_spearman_brown": 2r/(1+r) or None when r <= -1,
                       "n_questions": int}}.
    """
    import torch

    if not pt_path.exists():
        raise FileNotFoundError(
            f"per-question shift tensor missing: {pt_path}. Download from HF first:\n"
            f'  uv run python -c "from huggingface_hub import hf_hub_download; '
            f"hf_hub_download('superkaiba1/explore-persona-space-data', "
            f"'issue552_benign_control/analysis_tensors/{pt_path.name}', "
            f"repo_type='dataset', local_dir='eval_results/issue_552/_hf_tensors')\"\n"
            f"then copy into {pt_path.parent}/ (or pass --skip-split-half)."
        )
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    out: dict[str, dict[str, float | int | None]] = {}
    for p_name, entry in payload["shifts"].items():
        if "delta_v_per_question" not in entry:
            raise KeyError(
                f"{pt_path} persona {p_name!r} lacks `delta_v_per_question` — the cell "
                f"was extracted without --save-per-question; re-run Phase C with the flag."
            )
        pq = entry["delta_v_per_question"].to(torch.float64).numpy()
        n_q = pq.shape[0]
        if n_q < 4:
            out[p_name] = {"r_half": None, "r_full_spearman_brown": None, "n_questions": n_q}
            continue
        even = pq[0::2].mean(axis=0)
        odd = pq[1::2].mean(axis=0)
        r_half = float(np.dot(even, odd) / (np.linalg.norm(even) * np.linalg.norm(odd)))
        denom = 1.0 + r_half
        r_full = (2.0 * r_half / denom) if denom > 1e-9 else None
        out[p_name] = {
            "r_half": r_half,
            "r_full_spearman_brown": r_full,
            "n_questions": int(n_q),
        }
    return out


def _corrected_cosines(
    cell: dict, split_half: dict[str, dict[str, float | int | None]]
) -> dict[str, float | None]:
    """Attenuation-corrected per-persona |cos-to-U1|: cos_raw / sqrt(r_full).

    Plan §16.1 — reported ALONGSIDE the raw values, never replacing them.
    None when the persona's Spearman-Brown reliability is non-positive
    (correction undefined under noise-dominated halves).
    """
    out: dict[str, float | None] = {}
    for p_name, cos_raw in zip(cell["persona_order"], cell["cos_to_U1"], strict=True):
        r_full = split_half.get(p_name, {}).get("r_full_spearman_brown")
        if r_full is None or r_full <= 0:
            out[p_name] = None
        else:
            out[p_name] = float(cos_raw / np.sqrt(r_full))
    return out


def _compute_verdict_inputs(
    per_cell_table: dict[str, dict], split_half_block: dict[str, dict] | None
) -> dict:
    """Pre-registered §3/§6.3 verdict inputs over the same-variant benign cells.

    Pure computation + recording — the classification string carries the
    pre-registered rule outcome; interpretation belongs to the analyzer.
    """
    verdict_cells = {}
    for seed in SEEDS:
        key = f"same_benign_seed{seed}"
        row = per_cell_table[key]
        sh_median = split_half_block[key]["median_r_half"] if split_half_block is not None else None
        verdict_cells[key] = {
            "mean_cos_to_U1": row["mean_cos_to_U1"],
            "s_top1_frac": row["s_top1_frac"],
            "sign_flip_null_cleared_p99": row["s_top1_frac"] > row["sign_flip_p99"],
            "median_split_half_r": sh_median,
            "split_half_above_floor": (
                None if sh_median is None else bool(sh_median >= SPLIT_HALF_VALIDITY_FLOOR)
            ),
        }
    n_confirm = sum(
        1
        for v in verdict_cells.values()
        if v["mean_cos_to_U1"] <= CONFIRMATION_MEAN_COS_MAX
        and v["s_top1_frac"] < CONFIRMATION_TOP_SHARE_MAX
    )
    n_falsify = sum(
        1
        for v in verdict_cells.values()
        if v["mean_cos_to_U1"] >= FALSIFICATION_MEAN_COS_MIN
        and v["s_top1_frac"] >= FALSIFICATION_TOP_SHARE_MIN
    )
    precondition_evaluable = split_half_block is not None
    precondition_holds = precondition_evaluable and all(
        v["sign_flip_null_cleared_p99"] and bool(v["split_half_above_floor"])
        for v in verdict_cells.values()
    )
    if n_falsify >= 2:
        classification = "falsification"
    elif n_confirm == 3:
        classification = (
            "confirmation" if precondition_holds else "confirmation_blocked_by_precondition"
        )
    else:
        classification = "equivocal"
    return {
        "same_variant_cells": verdict_cells,
        "n_cells_in_confirmation_zone": n_confirm,
        "n_cells_in_falsification_zone": n_falsify,
        "validity_precondition_evaluable": precondition_evaluable,
        "validity_precondition_holds": precondition_holds if precondition_evaluable else None,
        "pre_registered_classification": classification,
        "note": (
            "Computation only — interpretation (incl. §16 critic concerns: "
            "disattenuation, magnitude mediation, claim scoping) belongs to "
            "the analyzer."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#552 cross-arm shift-geometry comparison (off-pod).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benign-svd-dir", default="eval_results/issue_552/svd")
    parser.add_argument("--parent-svd-dir", default="eval_results/issue_521/svd")
    parser.add_argument("--shifts-dir", default="eval_results/issue_552/shifts")
    parser.add_argument("--out", default="eval_results/issue_552/cross_arm/summary.json")
    parser.add_argument(
        "--skip-split-half",
        action="store_true",
        help=(
            "Skip the split-half reliability block (e.g. tensors not yet staged "
            "from HF). The §3 validity precondition is then NOT evaluable and is "
            "recorded as such — the confirmation branch cannot be claimed."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    benign_dir = Path(args.benign_svd_dir)
    parent_dir = Path(args.parent_svd_dir)
    shifts_dir = Path(args.shifts_dir)

    def svd_dir_for(arm: str) -> Path:
        return benign_dir if arm == "benign" else parent_dir

    # ── 1. Per-cell comparison table ────────────────────────────────────
    cells: dict[str, dict] = {}
    per_cell_table: dict[str, dict] = {}
    for variant in VARIANTS:
        for arm in ARMS:
            for seed in SEEDS:
                cell = _load_cell(svd_dir_for(arm), variant, arm, seed)
                key = f"{variant}_{arm}_seed{seed}"
                cells[key] = cell
                per_cell_table[key] = {
                    "variant": variant,
                    "arm": arm,
                    "seed": seed,
                    "mean_cos_to_U1": float(cell["mean_cos_to_U1"]),
                    "median_cos_to_U1": float(cell["median_cos_to_U1"]),
                    "s_top1_frac": float(cell["s_top1_frac"]),
                    "fro_norm_M": _fro_norm(cell["singular_values"]),
                    "sign_flip_p95": float(cell["sign_flip_p95"]),
                    "sign_flip_p99": float(cell["sign_flip_p99"]),
                    "row_shuffle_p95": float(cell["row_shuffle_p95"]),
                    "row_shuffle_p99": float(cell["row_shuffle_p99"]),
                }

    # ── 2. Within/cross-arm |cos(U1, U1')| per variant ──────────────────
    u1 = {key: np.asarray(cell["U1"], dtype=np.float64) for key, cell in cells.items()}
    direction_identity: dict[str, dict] = {}
    for variant in VARIANTS:
        block: dict[str, dict] = {}
        # Within-arm (3 seed pairs each).
        for arm in ARMS:
            pairs = {}
            for s1, s2 in combinations(SEEDS, 2):
                pairs[f"seed{s1}_x_seed{s2}"] = _abs_cos(
                    u1[f"{variant}_{arm}_seed{s1}"], u1[f"{variant}_{arm}_seed{s2}"]
                )
            block[f"within_{arm}"] = {
                "pairs": pairs,
                "median": float(np.median(list(pairs.values()))),
            }
        # Cross-arm (9 seed pairs each).
        for arm_a, arm_b in (("benign", "em"), ("benign", "marker"), ("em", "marker")):
            pairs = {}
            for s1 in SEEDS:
                for s2 in SEEDS:
                    pairs[f"{arm_a}_seed{s1}_x_{arm_b}_seed{s2}"] = _abs_cos(
                        u1[f"{variant}_{arm_a}_seed{s1}"], u1[f"{variant}_{arm_b}_seed{s2}"]
                    )
            block[f"cross_{arm_a}_x_{arm_b}"] = {
                "pairs": pairs,
                "median": float(np.median(list(pairs.values()))),
            }
        direction_identity[variant] = block

    # ── 3. Split-half reliability + corrected cosines (benign, per cell) ─
    split_half_block: dict[str, dict] | None = None
    if args.skip_split_half:
        logger.warning(
            "--skip-split-half: §3 validity precondition NOT evaluable; the "
            "confirmation branch cannot be claimed from this summary."
        )
    else:
        split_half_block = {}
        for variant in VARIANTS:
            for seed in SEEDS:
                pt_path = shifts_dir / f"{variant}_benign_seed{seed}.pt"
                sh = _split_half_per_persona(pt_path)
                key = f"{variant}_benign_seed{seed}"
                r_halves = [v["r_half"] for v in sh.values() if v["r_half"] is not None]
                split_half_block[key] = {
                    "per_persona": sh,
                    "median_r_half": float(np.median(r_halves)) if r_halves else None,
                    "corrected_cos_to_U1_per_persona": _corrected_cosines(cells[key], sh),
                    "correction": "cos_raw / sqrt(r_full_spearman_brown); None when r_full <= 0",
                }

    # ── 4. Pre-registered verdict inputs (same variant only) ────────────
    verdict_inputs = _compute_verdict_inputs(per_cell_table, split_half_block)

    # ── Assemble + write ────────────────────────────────────────────────
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    summary = {
        "issue": 552,
        "random_cos_floor_p95": RANDOM_COS_FLOOR_P95,
        "thresholds": {
            "confirmation": (
                "3/3 same-variant seeds mean_cos <= 0.85 AND s_top1_frac < 0.50, "
                "AND validity precondition (sign-flip p99 clearance + median "
                "split-half >= 0.5)"
            ),
            "falsification": ">=2/3 same-variant seeds mean_cos >= 0.90 AND s_top1_frac >= 0.50",
        },
        "per_cell": per_cell_table,
        "direction_identity": direction_identity,
        "split_half": split_half_block,
        "verdict_inputs": verdict_inputs,
        "inputs": {
            "benign_svd_dir": str(benign_dir),
            "parent_svd_dir": str(parent_dir),
            "shifts_dir": None if args.skip_split_half else str(shifts_dir),
        },
        "metadata": {
            "git_commit": git_commit,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": "scripts/issue552_cross_arm_analysis.py",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "[phase=done] cross-arm summary written to %s (classification=%s, "
        "precondition_evaluable=%s)",
        out_path,
        verdict_inputs["pre_registered_classification"],
        verdict_inputs["validity_precondition_evaluable"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

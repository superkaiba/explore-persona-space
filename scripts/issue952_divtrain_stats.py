#!/usr/bin/env python
"""Issue #952 diverse-train-injection — pooled paired stats (VM, CPU).

Consumes the GPU leg's ``divtrain_refit_eval.json`` (pool-only vs augmented map
reads) and computes the PAIRED augmented-minus-pool-only deltas with a paired
bootstrap (10,000 draws, rng(0)) + sign-flip null (10,000 draws, rng(1)),
reusing the committed #952 battery helpers (``_bank_boot`` / ``_signflip_p`` —
vectorized subset-sum draws, no per-draw Python loop).

Reads are numbers-only (per-pair drops / per-context R2 the GPU leg already
persisted); NO prompt/answer text is touched.

Deltas computed:
  * in_domain_check: per-held-out-injection-context R2 lift (aug - pool), per arm
    (the primary "did injection shift the map into the divergence domain" read).
  * china_arm_matched_d_lift: per-pair arm-matched d lift on the china bank.
  * china_within_arm_r2_lift: per-pair own/ext_plain R2 lift on the china bank
    (the OOD-floor read the round exists to test).
  * cross_lift_S1: own-map x plain-target cross drop, full china set AND the
    12-pair Qwen-refuses/Claude-answers subset (S1), pool vs aug + paired lift.

Usage:
  uv run python scripts/issue952_divtrain_stats.py
  uv run python scripts/issue952_divtrain_stats.py --input <refit_eval.json> --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import subprocess
import sys
import time

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from issue952_divergence_transfer_cell import N_DRAWS, _bank_boot, _signflip_p  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("issue952.divtrain_stats")

LABEL = "diverse-train-injection"


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT).decode().strip()
        )
    except Exception:
        return "unknown"


def _paired_delta_cell(delta: np.ndarray) -> dict:
    """Paired bootstrap (rng 0) + sign-flip null (rng 1) on a paired-delta vector."""
    delta = np.asarray([d for d in delta if np.isfinite(d)], dtype=np.float64)
    if len(delta) == 0:
        return {"n": 0}
    boot = _bank_boot(delta, N_DRAWS)  # seed 0
    sf = _signflip_p(delta, N_DRAWS)  # seed 1
    return {
        "n": len(delta),
        "mean_delta": boot["mean"],
        "mean_delta_ci95": boot["mean_ci95"],
        "median_delta": boot["median"],
        "median_delta_ci95": boot["median_ci95"],
        "sign_flip_p_one_sided": sf["p_one_sided"],
        "sign_flip_null_band_hi_97p5": sf["null_band_hi_97p5"],
    }


def _pairs_by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["pair_id"]: r for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="issue #952 diverse-train-injection pooled stats")
    ap.add_argument("--input", default=None, help="divtrain_refit_eval.json (default: committed)")
    ap.add_argument("--out-dir", default=None, help="output base (default: repo root)")
    ap.add_argument("--smoke", action="store_true", help="tolerate tiny/degenerate inputs")
    args = ap.parse_args()
    t0 = time.time()

    in_path = (
        pathlib.Path(args.input)
        if args.input
        else _REPO_ROOT / "eval_results/issue_952" / LABEL / "divtrain_refit_eval.json"
    )
    refit = json.loads(in_path.read_text())

    base = pathlib.Path(args.out_dir) if args.out_dir else _REPO_ROOT
    out_dir = base / "eval_results" / "issue_952" / LABEL
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── in-domain manipulation-check R2 lift (aug - pool), per arm ──────────────────
    ic = refit["indomain_check_per_context"]
    in_domain = {}
    for arm in ("own", "ext_plain"):
        pool = np.asarray(ic["pool_only"][arm], dtype=np.float64)
        aug = np.asarray(ic["augmented"][arm], dtype=np.float64)
        assert pool.shape == aug.shape, (arm, pool.shape, aug.shape)
        in_domain[arm] = _paired_delta_cell(aug - pool)
        in_domain[arm]["mean_r2_pool"] = float(np.nanmean(pool)) if pool.size else None
        in_domain[arm]["mean_r2_aug"] = float(np.nanmean(aug)) if aug.size else None

    # ── china arm-matched d lift + within-arm R2 lift ───────────────────────────────
    pool_china = _pairs_by_id(refit["per_pair_rows"]["pool_only"]["china"])
    aug_china = _pairs_by_id(refit["per_pair_rows"]["augmented"]["china"])
    common = [p for p in pool_china if p in aug_china]
    d_lift = np.asarray(
        [aug_china[p].get("d", np.nan) - pool_china[p].get("d", np.nan) for p in common],
        dtype=np.float64,
    )
    within = {}
    for key in ("r2_div_own", "r2_ctl_own", "r2_div_ext_plain", "r2_ctl_ext_plain"):
        lift = np.asarray(
            [aug_china[p].get(key, np.nan) - pool_china[p].get(key, np.nan) for p in common],
            dtype=np.float64,
        )
        within[key] = _paired_delta_cell(lift)

    # ── cross drop (own map x plain target): full + S1 subset ───────────────────────
    pool_cross = _pairs_by_id(refit["per_pair_rows"]["pool_only"]["cross"])
    aug_cross = _pairs_by_id(refit["per_pair_rows"]["augmented"]["cross"])
    s1_path = (
        _REPO_ROOT / "eval_results/issue_952/refusal_sanity_check/behavior_differs_subset.json"
    )
    s1_ids: list[str] = []
    if s1_path.exists():
        s1_ids = json.loads(s1_path.read_text())["membership"]["S1_refusal_mismatch"]

    def _cross_lift(pids: list[str]) -> dict:
        common_c = [p for p in pids if p in pool_cross and p in aug_cross]
        lift = np.asarray(
            [
                aug_cross[p].get("drop", np.nan) - pool_cross[p].get("drop", np.nan)
                for p in common_c
            ],
            dtype=np.float64,
        )
        pool_mean = (
            float(np.nanmean([pool_cross[p].get("drop", np.nan) for p in common_c]))
            if common_c
            else None
        )
        aug_mean = (
            float(np.nanmean([aug_cross[p].get("drop", np.nan) for p in common_c]))
            if common_c
            else None
        )
        return {
            "n": len(common_c),
            "pool_mean_drop": pool_mean,
            "aug_mean_drop": aug_mean,
            "lift": _paired_delta_cell(lift),
        }

    cross = {
        "china_all": _cross_lift(list(pool_cross)),
        "S1_refusal_mismatch": _cross_lift(s1_ids),
    }

    out = {
        "label": LABEL,
        "description": (
            "paired augmented-minus-pool-only deltas (paired bootstrap rng0 / sign-flip "
            "rng1, 10k draws) over the GPU leg's per-pair + per-context reads."
        ),
        "n_draws": N_DRAWS,
        "input": str(in_path),
        "input_git_commit": refit.get("repro", {}).get("git_commit"),
        "gates_pass": all(g["pass"] for g in refit.get("reproduction_gates", {}).values()),
        "in_domain_check_r2_lift": in_domain,
        "china_arm_matched_d_lift": _paired_delta_cell(d_lift),
        "china_within_arm_r2_lift": within,
        "cross_own_map_x_plain": cross,
        "s1_subset_n": len(s1_ids),
        "git_commit": _git_sha(),
        "wall_seconds": round(time.time() - t0, 1),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    p = out_dir / "stats_divtrain.json"
    p.write_text(json.dumps(out, indent=2))
    logger.info(
        "[stats] wrote %s | in-domain own lift=%.4f china d lift=%.4f cross-S1 lift=%.4f",
        p,
        (in_domain["own"].get("mean_delta") or float("nan")),
        (out["china_arm_matched_d_lift"].get("mean_delta") or float("nan")),
        (cross["S1_refusal_mismatch"]["lift"].get("mean_delta") or float("nan")),
    )


if __name__ == "__main__":
    main()

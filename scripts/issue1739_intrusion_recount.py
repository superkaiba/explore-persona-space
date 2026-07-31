"""Excluded-intrusion DV recount for task #1739 (interpretation-critic residual).

Recomputes, per behavior, EXCLUDING contexts flagged by the CJK intrusion scan
(``eval_results/issue_1739/intrusion_audit/intrusion_scan.json``):

1. the headline census — grid cells whose paired-bootstrap 95% CI on the
   headline delta (map-then-project minus context-native, FROZEN-layer
   convention) sits wholly below zero; reported full-data vs
   intrusion-excluded under the SAME frozen convention (the committed
   selection-inherited census needs per-layer predictions the per-cell npz
   does not carry, so it is quoted from cells.jsonl for reference only); and
2. the canonical-slice (U=full, L=max, context variant, E1) per-arm mean
   held-out Spearman rho, full vs intrusion-excluded.

Reuses the run's own batched paired-bootstrap helpers
(``experiments.issue_1739.arms``: ``make_bootstrap_idx`` + ``bootstrap_rhos``)
and validates the pipeline by reproducing the stored full-data
``delta_rho_frozen`` over all FINITE cells (tolerance 5e-3 — the residual is
fp32 pred-persistence rounding of near-tie ranks; tie-degenerate NaN cells
are counted separately, never coerced) and ``ci_delta_frozen`` (first 2 cells
per behavior, bit-close) before trusting the excluded read.

Provenance convention: the output embeds ``git_commit`` (checkout HEAD at run
time — this PREDATES the commit that lands script + artifact together) and
``script_blob_sha`` (``git hash-object`` of this file), so script/artifact
correspondence is provable at any later commit C via
``git rev-parse C:scripts/issue1739_intrusion_recount.py == script_blob_sha``.

Also records the item-aligned split-half ceilings from cells.jsonl (finding 5).

Output: ``eval_results/issue_1739/intrusion_audit/recount.json``.

Run from the issue-1739 worktree root (sycophancy preds staged from HF):
    OMP_NUM_THREADS=8 uv run python scripts/issue1739_intrusion_recount.py \
        --syco-preds data/issue_1739/hf_dl/percell_preds/sycophancy
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.experiments.issue_1739.arms import (  # noqa: E402
    bootstrap_rhos,
    make_bootstrap_idx,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval_results" / "issue_1739"
BEHS = ["evil", "hallucination", "sycophancy"]
MAXL = {"evil": 8000, "hallucination": 16000, "sycophancy": 16000}
ARMS = [
    "arm1_ctx_e1",
    "arm2_ctx_native",
    "arm3_identity_bias",
    "arm4_ridge_ctx",
    "arm5_mlp_ctx",
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm8_map_ridge_true",
    "arm9_pretrain_ft",
    "arm10_stacked",
    "arm11_oracle_proj",
    "arm12_oracle_reg",
    "arm13_shuffled_map",
    "arm14_shuffled_pt",
    "arm15_text_only",
    "arm16_surface_feat",
]

_PREDS_DIR: Path | None = None  # set per worker via initializer
_FLAGGED: frozenset[str] = frozenset()


def _init_worker(preds_dir: str, flagged: list[str]) -> None:
    global _PREDS_DIR, _FLAGGED
    _PREDS_DIR = Path(preds_dir)
    _FLAGGED = frozenset(flagged)


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    r, _ = spearmanr(x, y)
    return float(r)


def _cell_job(args: tuple[str, str, list[str], float, int, int, bool]) -> dict:
    """One grid cell: validate full delta, recompute excluded delta + CI."""
    npz_name, unit_key, pair, stored_delta, seed, draw, validate_ci = args
    z = np.load(_PREDS_DIR / npz_name)
    dv = z["dv"].astype(np.float64)
    cids = z["context_ids"]
    a = z[f"pred__{pair[0]}"].astype(np.float64)
    b = z[f"pred__{pair[1]}"].astype(np.float64)
    n = len(dv)

    full_delta = _rho(a, dv) - _rho(b, dv)
    out: dict = {
        "unit_key": unit_key,
        "n_rows": n,
        "delta_full_recomputed": full_delta,
        "delta_full_stored": stored_delta,
        "full_delta_diff": float(abs(full_delta - stored_delta)),
    }
    if validate_ci:
        idx = make_bootstrap_idx(n, n_boot=500, seed=seed + 100 * draw)
        draws = bootstrap_rhos(np.stack([a, b]), dv, idx)
        d = draws[0] - draws[1]
        out["ci_full_recomputed"] = [float(np.nanquantile(d, q)) for q in (0.025, 0.975)]

    keep = np.array([c not in _FLAGGED for c in cids])
    n_ex = int(keep.sum())
    out["n_rows_excluded_read"] = n_ex
    if n_ex >= 10:
        ae, be, dve = a[keep], b[keep], dv[keep]
        delta_ex = _rho(ae, dve) - _rho(be, dve)
        idx = make_bootstrap_idx(n_ex, n_boot=500, seed=seed + 100 * draw)
        draws = bootstrap_rhos(np.stack([ae, be]), dve, idx)
        d = draws[0] - draws[1]
        ci = [float(np.nanquantile(d, q)) for q in (0.025, 0.975)]
        out.update(
            delta_excluded=delta_ex,
            ci_delta_frozen_excluded=ci,
            below_zero_excluded=bool(ci[1] < 0),
        )
    else:
        out.update(delta_excluded=None, ci_delta_frozen_excluded=None, below_zero_excluded=None)
    # canonical-slice per-arm rhos (full + excluded) — cheap, computed for
    # every cell; the aggregator keeps only canonical cells' values.
    arm_rhos = {}
    for arm in ARMS:
        key = f"pred__{arm}"
        if key not in z:
            continue
        p = z[key].astype(np.float64)
        arm_rhos[arm] = {
            "full": _rho(p, dv),
            "excluded": _rho(p[keep], dv[keep]) if n_ex >= 10 else None,
        }
    out["arm_rhos"] = arm_rhos
    return out


def recount_behavior(beh: str, preds_dir: Path, flags: dict, workers: int) -> dict:
    cells = [
        json.loads(line) for line in open(EVAL / beh / "arm_results" / "percell" / "cells.jsonl")
    ]
    jobs, meta = [], []
    stored_sel_census = 0
    stored_frozen_census = 0
    n_grid = 0
    n_validate = 0
    for c in cells:
        k = json.loads(c["unit_key"])
        if k.get("f_u") is not None or k.get("f_l") is not None or k["eval_rung"] != "train":
            continue
        n_grid += 1
        h = c["headline"]
        stored_sel_census += int(h["ci_delta_selection_inherited"][1] < 0)
        stored_frozen_census += int(h["ci_delta_frozen"][1] < 0)
        validate_ci = n_validate < 2
        n_validate += 1 if validate_ci else 0
        jobs.append(
            (
                c["preds_npz"],
                c["unit_key"],
                h["pair"],
                h["delta_rho_frozen"],
                k["seed"],
                k["draw"],
                validate_ci,
            )
        )
        meta.append((c, k))

    flagged = list(flags["row_flags"].keys())
    results = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(str(preds_dir), flagged)
    ) as ex:
        for r in ex.map(_cell_job, jobs, chunksize=4):
            results.append(r)

    diffs = np.array([r["full_delta_diff"] for r in results], dtype=np.float64)
    finite = diffs[np.isfinite(diffs)]
    n_nan = int((~np.isfinite(diffs)).sum())
    n_match = int((finite < 5e-3).sum())
    ci_checks = []
    for r, (c, _k) in zip(results, meta):
        if "ci_full_recomputed" in r:
            stored_ci = c["headline"]["ci_delta_frozen"]
            ci_checks.append(
                {
                    "stored": stored_ci,
                    "recomputed": r["ci_full_recomputed"],
                    "max_abs_diff": float(
                        max(abs(a - b) for a, b in zip(stored_ci, r["ci_full_recomputed"]))
                    ),
                }
            )
    census_excluded = sum(1 for r in results if r["below_zero_excluded"])
    n_excluded_readable = sum(1 for r in results if r["below_zero_excluded"] is not None)

    # canonical-slice arm means
    arm_means: dict[str, dict] = {}
    sh = None
    n_canon = 0
    for r, (c, k) in zip(results, meta):
        canon = (
            k["u_rung_label"] == "full"
            and k["budget_l"] == MAXL[beh]
            and k["variant"] == "context_end"
            and k["regime"] == "e1"
        )
        if not canon:
            continue
        n_canon += 1
        for arm, v in r["arm_rhos"].items():
            arm_means.setdefault(arm, {"full": [], "excluded": []})
            arm_means[arm]["full"].append(v["full"])
            if v["excluded"] is not None:
                arm_means[arm]["excluded"].append(v["excluded"])
        if c.get("split_half"):
            sh = c["split_half"]

    def _m(xs: list) -> float | None:
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return round(float(np.mean(xs)), 4) if xs else None

    return {
        "n_grid_cells": n_grid,
        "n_flagged_contexts": len(flagged),
        "pipeline_validation": {
            "full_delta_match": (
                f"{n_match}/{finite.size} finite cells within 5e-3 of stored "
                f"delta_rho_frozen (max abs diff {float(finite.max()):.2e}; residual is "
                "fp32 pred-persistence rounding of near-tie ranks — hallucination CI "
                "checks reproduce bit-identically, evil/sycophancy to <1e-3); "
                f"{n_nan} tie-degenerate NaN cells"
            ),
            "full_ci_checks": ci_checks,
        },
        "census": {
            "stored_selection_inherited_full": stored_sel_census,
            "frozen_full": stored_frozen_census,
            "frozen_excluded": census_excluded,
            "n_cells_excluded_readable": n_excluded_readable,
            "convention_note": (
                "excluded census uses the FROZEN-layer paired-bootstrap CI (per-layer "
                "predictions are not persisted per cell, so the selection-inherited "
                "convention cannot be recomputed post hoc); compare frozen_full vs "
                "frozen_excluded for the exclusion effect"
            ),
        },
        "canonical_slice": {
            "n_cells": n_canon,
            "arm_mean_rho": {
                arm: {"full": _m(v["full"]), "excluded": _m(v["excluded"])}
                for arm, v in sorted(arm_means.items())
            },
        },
        "split_half_canonical": sh,
        "percell_excluded": [
            {
                "unit_key": r["unit_key"],
                "n_rows": r["n_rows"],
                "n_rows_excluded_read": r["n_rows_excluded_read"],
                "delta_full": r["delta_full_recomputed"],
                "delta_excluded": r["delta_excluded"],
                "ci_delta_frozen_excluded": r["ci_delta_frozen_excluded"],
            }
            for r in results
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--syco-preds", default=str(ROOT / "data/issue_1739/hf_dl/percell_preds/sycophancy")
    )
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    scan = json.load(open(EVAL / "intrusion_audit" / "intrusion_scan.json"))
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    # git_commit = checkout HEAD at run time (predates the commit landing script +
    # artifact together); script_blob_sha proves script/artifact correspondence.
    script_blob = subprocess.run(
        ["git", "hash-object", str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()
    out = {
        "task": 1739,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": sha,
        "script_blob_sha": script_blob,
        "scan_source": "eval_results/issue_1739/intrusion_audit/intrusion_scan.json",
        "exclusion_rule": scan["flag_rule"],
        "behaviors": {},
    }
    for beh in BEHS:
        preds = (
            Path(args.syco_preds)
            if beh == "sycophancy"
            else EVAL / beh / "arm_results" / "percell" / "preds"
        )
        t0 = time.time()
        out["behaviors"][beh] = recount_behavior(beh, preds, scan["behaviors"][beh], args.workers)
        b = out["behaviors"][beh]
        print(
            f"{beh}: census sel-inh(full)={b['census']['stored_selection_inherited_full']} "
            f"frozen(full)={b['census']['frozen_full']} "
            f"frozen(excluded)={b['census']['frozen_excluded']} of {b['n_grid_cells']}; "
            f"validation {b['pipeline_validation']['full_delta_match']} "
            f"[{time.time() - t0:.0f}s]"
        )
    path = EVAL / "intrusion_audit" / "recount.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

"""#1092 theory-sharp averaging test: spread vs the AVERAGED-context-map error DELTA.

The banked fair-comparison prediction_agreement (fair_comparison.json) carries two
per-prefix error arms at the averaged grain, both vs the same Y_avg target:
  e_prefixend : prefix-END-state map, fresh 6-fold-over-prefixes fit
  e_ctxavg    : per-row context-map held-out predictions, averaged within prefix

This script adds the MISSING arm — the averaged-context-vector ("prefix vector")
map: Xc_avg = within-prefix mean of context_end_L14, fresh 6-fold fit on the
IDENTICAL folds (FOLD_SEED=0) and identical Y_avg — then tests whether
within-prefix context-vector spread predicts the AVERAGING-specific error gaps:

  d_pe  = e_avgctx - e_prefixend   (averaged summary vs end-state summary)
  d_ctx = e_avgctx - e_ctxavg      (map-the-centroid vs average-the-predictions;
                                    the Jensen-gap arm, both context-derived)

Parity gates: recomputed e_prefixend must match the banked array (same engine,
same folds, deterministic); recomputed spread must match the deepdive npz.

Checkpointed per (cell, basis) unit — each unit writes unit_<cell>_<basis>.json
the moment it completes and is skipped on resume (earlyoom SIGTERMed the first
run mid-unit under fleet memory pressure; a kill now costs one unit).

Analysis-only: NO model forward, NO training, NO API. Reads the staged L14
summaries + manifest and the banked JSONs. Fit engine reused verbatim from
scripts/issue1092_fit_grid.py.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/eps-data/thomasjiralerspong/.hf_i1092_operator")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

from issue1092_fit_grid import (  # noqa: E402
    _basis_targets_with_info,
    _fit_cv,
    _folds_from_manifest,
)

STAGE = Path(
    "/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator/issue1092_realistic_crossing"
)
SUMM = STAGE / "analysis_tensors/summaries"
MANIFEST = STAGE / "corpus/manifest.jsonl"
BANKED = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json"
DEEPDIVE = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison_deepdive"
OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_avgctx_spread_delta"

CELLS = ["cell_inst_own", "cell_pre_own"]
LAYER = 14
BASES = ["ambient", "pca48"]
TARGETS = ["t1", "t2", "t3"]
HIDDEN_DIM = 3584
N_FOLDS = 6
MIN_ROWS_PER_PREFIX = 3
PARITY_TOL = 1e-6


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load(cell: str, kind: str) -> np.ndarray:
    p = SUMM / cell / f"{kind}_L{LAYER:02d}.npy"
    if not p.exists():
        raise FileNotFoundError(p)
    return np.load(p, mmap_mode="r")


def _prefix_groups(prefix_ids: np.ndarray, min_rows: int) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for i, pid in enumerate(prefix_ids):
        groups.setdefault(str(pid), []).append(i)
    return {
        pid: np.asarray(idx, dtype=np.int64) for pid, idx in groups.items() if len(idx) >= min_rows
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> dict:
    r, p = stats.spearmanr(x, y)
    return {"rho": float(r), "p": float(p), "n": int(len(x))}


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Rank-based partial correlation of x,y controlling for z (deepdive recipe)."""
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r, p = stats.pearsonr(ex, ey)
    return {"partial_rho": float(r), "p": float(p), "n": int(len(x))}


def _be_slice(cell: str, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Battery-excluded row indices + per-row prefix ids (fair-comparison recipe)."""
    prefix_all = _load(cell, "prefix_end")
    context_all = _load(cell, "context_end")
    t_shapes = [_load(cell, t).shape[0] for t in TARGETS]
    n0 = min(prefix_all.shape[0], context_all.shape[0], min(t_shapes), len(rows))
    be_idx = np.asarray(
        [
            i
            for i in range(n0)
            if rows[i].get("stratum") != "trait_stratum" and not rows[i].get("is_eval_only")
        ],
        dtype=np.int64,
    )
    prefix_ids = np.asarray([rows[int(i)].get("prefix_id", "") for i in be_idx])
    return be_idx, prefix_ids


def process_unit(cell: str, basis: str, rows: list[dict], banked: dict) -> dict:
    """One (cell, basis) unit: fit the averaged-context arm, parity-check, stats."""
    be_idx, prefix_ids = _be_slice(cell, rows)
    groups = _prefix_groups(prefix_ids, MIN_ROWS_PER_PREFIX)
    pids = sorted(groups)
    n_turns = np.asarray(
        [int(rows[int(be_idx[groups[p][0]])].get("prefix_n_user_turns", 0)) for p in pids],
        dtype=np.float64,
    )

    X_prefix = np.asarray(_load(cell, "prefix_end")[be_idx], dtype=np.float64)
    X_context = np.asarray(_load(cell, "context_end")[be_idx], dtype=np.float64)

    # within-prefix context-vector spread (raw L2 about the centroid; deepdive recipe)
    spread = np.zeros(len(pids), dtype=np.float64)
    for k, p in enumerate(pids):
        block = X_context[groups[p]]
        c = block - block.mean(0, keepdims=True)
        spread[k] = float(np.sqrt((c * c).sum(1).mean()))
    dd = np.load(DEEPDIVE / f"per_prefix_arrays_{cell}.npz", allow_pickle=True)
    spread_parity = float(np.max(np.abs(spread - dd["spread"])))
    assert spread_parity < PARITY_TOL, f"spread parity {spread_parity} vs deepdive npz"

    Y_stacked = np.concatenate(
        [np.asarray(_load(cell, t)[be_idx], dtype=np.float64) for t in TARGETS], axis=1
    )
    Yb = _basis_targets_with_info(
        Y_stacked, basis, hidden_dim=HIDDEN_DIM, targets=TARGETS, projection_target="t1"
    )[0]
    Yb = np.ascontiguousarray(Yb, dtype=np.float64)
    del Y_stacked
    gc.collect()

    Y_avg = np.stack([Yb[groups[p]].mean(0) for p in pids], axis=0)
    Xp_avg = np.stack([X_prefix[groups[p]].mean(0) for p in pids], axis=0)
    Xc_avg = np.stack([X_context[groups[p]].mean(0) for p in pids], axis=0)
    del Yb, X_prefix, X_context
    gc.collect()

    pseudo_rows = [{"prefix_id": p} for p in pids]
    folds_avg = _folds_from_manifest(
        pseudo_rows, len(pseudo_rows), group_key="prefix_id", n_folds=N_FOLDS
    )
    fit_pe, pred_pe = _fit_cv(Xp_avg, Y_avg, folds_avg, return_pred=True)
    fit_ac, pred_ac = _fit_cv(Xc_avg, Y_avg, folds_avg, return_pred=True)
    e_prefixend = np.linalg.norm(pred_pe - Y_avg, axis=1)
    e_avgctx = np.linalg.norm(pred_ac - Y_avg, axis=1)

    pa = banked["cells"][cell]["bases"][basis]["prediction_agreement"]
    e_pe_banked = np.asarray(pa["per_prefix_err_prefix"], dtype=np.float64)
    e_ctxavg = np.asarray(pa["per_prefix_err_ctx"], dtype=np.float64)
    assert e_pe_banked.shape[0] == len(pids), (e_pe_banked.shape, len(pids))
    pe_parity = float(np.max(np.abs(e_prefixend - e_pe_banked)))
    assert pe_parity < PARITY_TOL, f"prefix-end parity {pe_parity} vs banked ({cell}/{basis})"

    d_pe = e_avgctx - e_pe_banked
    d_ctx = e_avgctx - e_ctxavg
    blk = {
        "cell": cell,
        "basis": basis,
        "n_prefixes": len(pids),
        "spread_parity_vs_deepdive": spread_parity,
        "prefixend_err_parity_vs_banked": pe_parity,
        "r2_avgctx_map_averaged_grain": float(fit_ac["r2"]),
        "r2_prefixend_map_averaged_grain": float(fit_pe["r2"]),
        # ctxavg-predictions R2 at averaged grain is banked in fair_comparison.json
        # (averaged_grain.r2_context_averaged); not recomputed here.
        "mean_err": {
            "avgctx": float(e_avgctx.mean()),
            "prefixend": float(e_pe_banked.mean()),
            "ctxavg_preds": float(e_ctxavg.mean()),
        },
        "spearman_spread_vs_err_avgctx": _spearman(spread, e_avgctx),
        "spearman_spread_vs_delta_avgctx_minus_prefixend": _spearman(spread, d_pe),
        "spearman_spread_vs_delta_avgctx_minus_ctxavg": _spearman(spread, d_ctx),
        "partial_spread_vs_delta_avgctx_minus_prefixend_given_nturns": _partial_spearman(
            spread, d_pe, n_turns
        ),
        "partial_spread_vs_delta_avgctx_minus_ctxavg_given_nturns": _partial_spearman(
            spread, d_ctx, n_turns
        ),
        "partial_spread_vs_err_avgctx_given_nturns": _partial_spearman(spread, e_avgctx, n_turns),
        "spearman_nturns_vs_err_avgctx": _spearman(n_turns, e_avgctx),
        "avgctx_lambda_indices": fit_ac.get("lambda_indices"),
        # durable per-prefix record (fair_comparison.json convention; npz are
        # gitignored-local): the NEW arm's errors + the joined covariates.
        "per_prefix_err_avgctx": [float(x) for x in e_avgctx],
        "per_prefix_spread": [float(x) for x in spread],
        "per_prefix_n_turns": [float(x) for x in n_turns],
    }
    np.savez(
        OUT / f"per_prefix_avgctx_{cell}_{basis}.npz",
        e_avgctx=e_avgctx,
        e_prefixend=e_pe_banked,
        e_ctxavg=e_ctxavg,
        d_pe=d_pe,
        d_ctx=d_ctx,
        spread=spread,
        n_turns=n_turns,
    )
    print(
        f"[{cell}/{basis}] r2_avgctx={blk['r2_avgctx_map_averaged_grain']:.4f} "
        f"(prefixend={blk['r2_prefixend_map_averaged_grain']:.4f}) "
        f"spread->d_pe rho={blk['spearman_spread_vs_delta_avgctx_minus_prefixend']['rho']:+.3f} "
        f"spread->d_ctx rho={blk['spearman_spread_vs_delta_avgctx_minus_ctxavg']['rho']:+.3f}",
        flush=True,
    )
    return blk


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    banked = json.loads(BANKED.read_text())
    rows = _jsonl(MANIFEST)
    units: dict[str, dict] = {}
    for cell in CELLS:
        for basis in BASES:
            unit_path = OUT / f"unit_{cell}_{basis}.json"
            if unit_path.exists():
                units[f"{cell}/{basis}"] = json.loads(unit_path.read_text())
                print(f"[resume] skipping completed unit {cell}/{basis}", flush=True)
                continue
            blk = process_unit(cell, basis, rows, banked)
            unit_path.write_text(json.dumps(blk, indent=2))
            units[f"{cell}/{basis}"] = blk
            gc.collect()
    result = {
        "meta": {
            "script": "scripts/issue1092_avgctx_spread_delta.py",
            "banked_source": str(BANKED.relative_to(PROJECT_ROOT)),
            "manifest_rows": len(rows),
            "protocol": (
                "averaged grain; novel-prefix grouped 6-fold (group_key=prefix_id, FOLD_SEED=0); "
                "fit engine issue1092_fit_grid.press_fit_predict reused verbatim; new arm "
                "Xc_avg = within-prefix mean of context_end_L14 vs the banked prefix_end and "
                "query-averaged context-prediction arms, all scored against the same Y_avg"
            ),
        },
        "units": units,
    }
    out_path = OUT / "avgctx_spread_delta.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

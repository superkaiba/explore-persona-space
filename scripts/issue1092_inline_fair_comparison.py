#!/usr/bin/env python3
"""#1092 inline FAIR prefix-vs-context prediction comparison (matched-corpus,
averaged-grain + ceiling-normalized).

Replaces the misleading raw per-row read1 bar chart, which read the prefix arm's
low single-context R2 (~0.065) against the context arm's ~0.80 as if the same
target were achievable to both. Two structural facts make that unfair:
  (a) per-row targets carry ~79% query-borne variance the prefix state cannot
      see (v_P is (near-)constant within a prefix), so the prefix arm's per-row
      ceiling is the between-prefix variance share (~0.11), not ~0.80; and
  (b) the prefix map's known ~0.8 was an AVERAGED-grain (per-prefix profile)
      read on a different corpus.

This driver computes, on the matched #1092 corpus (battery-EXCLUDED fits, novel-
prefix 6-fold CV, layer 14, ambient + pca48 pooled t1/t2/t3 targets), for cells
{cell_inst_own (primary), cell_pre_own}:
  - AVERAGED-grain held-out R2 for the prefix map (v_P(prefix) -> per-prefix
    profile) and the context map (per-row map's held-out predictions averaged
    per prefix), 6-fold over prefixes;
  - SINGLE-context-grain held-out R2 for both arms (my battery-excluded refit,
    reported alongside the banked battery-INCLUDED read1 numbers);
  - ceilings: prefix per-row ceiling = between-prefix variance share (computed
    on the fit population AND cross-checked on the dense core vs banked read3);
    context per-row ceiling = the banked MLP companion (0.929, pca48) and the
    additive ceiling (1 - interaction share).

Fit engine REUSED verbatim from scripts/issue1092_fit_grid.py (_fit_cv ->
press_fit_predict, _folds_from_manifest, _basis_targets_with_info, _r2), so the
only difference from the banked read1 per-row numbers is the battery exclusion.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
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

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

# Fit machinery reused verbatim from the banked read1 engine.
from issue1092_fit_grid import (  # noqa: E402
    _basis_targets_with_info,
    _fit_cv,
    _folds_from_manifest,
    _r2,
)

STAGE = Path(
    "/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator/issue1092_realistic_crossing"
)
SUMM = STAGE / "analysis_tensors/summaries"
MANIFEST = STAGE / "corpus/manifest.jsonl"
OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison"

CELLS = ["cell_inst_own", "cell_pre_own"]
LAYER = 14
ARMS = ["prefix_end", "context_end"]
BASES = ["ambient", "pca48"]
TARGETS = ["t1", "t2", "t3"]
HIDDEN_DIM = 3584
N_FOLDS = 6
MIN_ROWS_PER_PREFIX = 3

# Banked numbers (battery-INCLUDED read1, n=19708; layer 14, fit-arm A) for reference.
BANKED_READ1 = {
    "cell_inst_own": {
        "ambient": {"prefix_end": 0.06507728422851033, "context_end": 0.8043440222361293},
        "pca48": {"prefix_end": 0.09604, "context_end": 0.91034},
        "n_rows": 19708,
    },
    "cell_pre_own": {
        "ambient": {"prefix_end": 0.05114, "context_end": 0.71442},
        "pca48": {"prefix_end": 0.09570, "context_end": 0.81801},
        "n_rows": 19708,
    },
}
# Banked read3 dense-core FGI shares (n=4752) for cross-check + additive ceiling.
BANKED_READ3 = {
    "cell_inst_own": {
        "ambient": {
            "share_prefix": 0.10419466,
            "share_query": 0.71923913,
            "share_interaction": 0.17656621,
        },
        "pca48": {
            "share_prefix": 0.10728734,
            "share_query": 0.78952738,
            "share_interaction": 0.10318527,
        },
    },
    "cell_pre_own": {
        "ambient": {
            "share_prefix": 0.11802503,
            "share_query": 0.60126758,
            "share_interaction": 0.28070739,
        },
        "pca48": {
            "share_prefix": 0.13288952,
            "share_query": 0.66391232,
            "share_interaction": 0.20319816,
        },
    },
}
# Banked MLP companion (context achievable ceiling, pca48 48-dim target, n_folds_used=19708).
BANKED_MLP_COMPANION = {"cell_inst_own": 0.9288425073128529}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load(cell: str, kind: str) -> np.ndarray:
    p = SUMM / cell / f"{kind}_L{LAYER:02d}.npy"
    if not p.exists():
        raise FileNotFoundError(p)
    return np.load(p, mmap_mode="r")


def _share_prefix(Yb: np.ndarray, prefix_ids: np.ndarray) -> float:
    """Between-prefix variance share = the prefix arm's structural per-row ceiling."""
    yc = Yb - Yb.mean(axis=0, keepdims=True)
    f = np.zeros_like(yc)
    for pid in np.unique(prefix_ids):
        m = prefix_ids == pid
        f[m] = yc[m].mean(axis=0, keepdims=True)
    ss = float((yc * yc).sum())
    return float((f * f).sum() / ss) if ss else float("nan")


def _fgi_shares(Yb: np.ndarray, prefix_ids: np.ndarray, query_ids: np.ndarray) -> dict:
    """Full prefix/query/interaction decomposition (needs a crossed population)."""
    yc = Yb - Yb.mean(axis=0, keepdims=True)
    f = np.zeros_like(yc)
    g = np.zeros_like(yc)
    for pid in np.unique(prefix_ids):
        m = prefix_ids == pid
        f[m] = yc[m].mean(axis=0, keepdims=True)
    for qid in np.unique(query_ids):
        m = query_ids == qid
        g[m] = yc[m].mean(axis=0, keepdims=True)
    i = yc - f - g
    ss = float((yc * yc).sum())
    return {
        "share_prefix": float((f * f).sum() / ss) if ss else float("nan"),
        "share_query": float((g * g).sum() / ss) if ss else float("nan"),
        "share_interaction": float((i * i).sum() / ss) if ss else float("nan"),
        "n_rows": int(Yb.shape[0]),
    }


def _within_prefix_constancy(X: np.ndarray, prefix_ids: np.ndarray) -> dict:
    """Mean within-prefix row-vector std / overall row-vector std for prefix_end.

    Near 0 confirms v_P is (near-)constant within a prefix (the premise that the
    prefix arm's per-row ceiling is the between-prefix variance share).
    """
    overall = float(np.sqrt(((X - X.mean(0, keepdims=True)) ** 2).sum(1).mean()))
    within = []
    for pid in np.unique(prefix_ids):
        m = prefix_ids == pid
        if m.sum() < 2:
            continue
        Xg = X[m]
        within.append(float(np.sqrt(((Xg - Xg.mean(0, keepdims=True)) ** 2).sum(1).mean())))
    within_mean = float(np.mean(within)) if within else float("nan")
    return {
        "overall_rowvec_std": overall,
        "mean_within_prefix_rowvec_std": within_mean,
        "ratio_within_over_overall": (within_mean / overall) if overall else float("nan"),
        "n_multi_row_prefixes": len(within),
    }


def _prefix_groups(prefix_ids: np.ndarray, min_rows: int) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for i, pid in enumerate(prefix_ids):
        groups.setdefault(str(pid), []).append(i)
    return {
        pid: np.asarray(idx, dtype=np.int64) for pid, idx in groups.items() if len(idx) >= min_rows
    }


def _averaged_grain(
    Yb: np.ndarray,
    X_prefix: np.ndarray,
    pred_context: np.ndarray,
    prefix_ids: np.ndarray,
) -> dict:
    """Prefix map @ averaged grain (fresh 6-fold-over-prefixes fit) + context map
    @ averaged grain (per-row held-out preds averaged per prefix)."""
    groups = _prefix_groups(prefix_ids, MIN_ROWS_PER_PREFIX)
    pids = sorted(groups)
    n_q = np.asarray([groups[p].size for p in pids], dtype=np.int64)
    # per-prefix averaged profile (target), averaged prefix state, averaged
    # per-row context held-out prediction.
    Y_avg = np.stack([Yb[groups[p]].mean(0) for p in pids], axis=0)
    Xp_avg = np.stack([X_prefix[groups[p]].mean(0) for p in pids], axis=0)
    pred_ctx_avg = np.stack([pred_context[groups[p]].mean(0) for p in pids], axis=0)
    # 6-fold over prefixes (each averaged row is its own prefix group).
    pseudo_rows = [{"prefix_id": p} for p in pids]
    folds_avg = _folds_from_manifest(
        pseudo_rows, len(pseudo_rows), group_key="prefix_id", n_folds=N_FOLDS
    )
    prefix_fit = _fit_cv(Xp_avg, Y_avg, folds_avg)
    r2_prefix_avg = float(prefix_fit["r2"])
    r2_context_avg = _r2(Y_avg, pred_ctx_avg)
    # 1/n_q within-prefix shrinkage note: residual within-prefix variance in the
    # averaged target scales ~1/n_q; estimate the total-variance fraction it
    # represents at the mean n_q.
    yc = Yb - Yb.mean(0, keepdims=True)
    ss_tot = float((yc * yc).sum())
    within_frac = 1.0 - _share_prefix(Yb, prefix_ids)  # query + interaction + noise share
    mean_nq = float(n_q.mean())
    return {
        "n_prefixes_kept": len(pids),
        "min_rows_per_prefix": MIN_ROWS_PER_PREFIX,
        "n_queries_per_prefix": {
            "mean": mean_nq,
            "median": float(np.median(n_q)),
            "min": int(n_q.min()),
            "max": int(n_q.max()),
        },
        "r2_prefix_averaged": r2_prefix_avg,
        "r2_context_averaged": r2_context_avg,
        "prefix_avg_lambda_indices": prefix_fit.get("lambda_indices"),
        "within_prefix_variance_fraction_perrow": within_frac,
        "estimated_residual_within_fraction_at_mean_nq": within_frac / mean_nq,
        "note_1_over_nq": (
            f"averaging ~{mean_nq:.1f} queries per prefix shrinks the residual "
            f"within-prefix (query+interaction+noise) variance from {within_frac:.3f} "
            f"of total to ~{within_frac / mean_nq:.4f}, leaving the averaged target "
            "dominated by between-prefix signal both arms can reach."
        ),
        "ss_tot_perrow": ss_tot,
    }


def process_cell(cell: str, rows: list[dict], gate: bool) -> dict:
    t0 = time.monotonic()
    prefix_all = _load(cell, "prefix_end")
    context_all = _load(cell, "context_end")
    t_all = [_load(cell, t) for t in TARGETS]
    n0 = min(prefix_all.shape[0], context_all.shape[0], min(t.shape[0] for t in t_all), len(rows))
    be_idx = np.asarray(
        [
            i
            for i in range(n0)
            if rows[i].get("stratum") != "trait_stratum" and not rows[i].get("is_eval_only")
        ],
        dtype=np.int64,
    )
    dense_local = np.asarray(
        [j for j, i in enumerate(be_idx) if rows[int(i)].get("stratum") == "dense_core"],
        dtype=np.int64,
    )
    prefix_ids = np.asarray([rows[int(i)].get("prefix_id", "") for i in be_idx])
    query_ids = np.asarray([rows[int(i)].get("query_id", "") for i in be_idx])
    unit_rows = [rows[int(i)] for i in be_idx]
    folds = _folds_from_manifest(unit_rows, len(unit_rows), group_key="prefix_id", n_folds=N_FOLDS)

    X_prefix = np.asarray(prefix_all[be_idx], dtype=np.float64)
    X_context = np.asarray(context_all[be_idx], dtype=np.float64)
    Y_stacked = np.concatenate([np.asarray(t[be_idx], dtype=np.float64) for t in t_all], axis=1)
    del prefix_all, context_all, t_all
    gc.collect()

    constancy = _within_prefix_constancy(X_prefix, prefix_ids)

    cell_out = {
        "n_battery_excluded": int(be_idx.size),
        "n_dense_core": int(dense_local.size),
        "n_folds": N_FOLDS,
        "group_key": "prefix_id",
        "banked_read1_battery_included": BANKED_READ1.get(cell),
        "prefix_end_within_prefix_constancy": constancy,
        "bases": {},
    }

    bases = ["ambient"] if gate else BASES
    for basis in bases:
        Yb = _basis_targets_with_info(
            Y_stacked, basis, hidden_dim=HIDDEN_DIM, targets=TARGETS, projection_target="t1"
        )[0]
        Yb = np.ascontiguousarray(Yb, dtype=np.float64)

        # SINGLE-context grain: per-row held-out fits for both arms.
        fit_prefix, pred_prefix = _fit_cv(X_prefix, Yb, folds, return_pred=True)
        fit_context, pred_context = _fit_cv(X_context, Yb, folds, return_pred=True)
        r2_prefix_single = float(fit_prefix["r2"])
        r2_context_single = float(fit_context["r2"])

        # AVERAGED grain.
        avg = _averaged_grain(Yb, X_prefix, pred_context, prefix_ids)

        # Ceilings.
        share_prefix_full = _share_prefix(Yb, prefix_ids)
        fgi_dense = _fgi_shares(Yb[dense_local], prefix_ids[dense_local], query_ids[dense_local])
        additive_ceiling_dense = 1.0 - fgi_dense["share_interaction"]
        mlp_ceiling = BANKED_MLP_COMPANION.get(cell) if basis == "pca48" else None

        # Dense-core single-grain refit (population-consistent with dense-core shares).
        dense_rows = [unit_rows[j] for j in dense_local]
        dense_folds = _folds_from_manifest(
            dense_rows, len(dense_rows), group_key="prefix_id", n_folds=N_FOLDS
        )
        r2_prefix_dense = float(_fit_cv(X_prefix[dense_local], Yb[dense_local], dense_folds)["r2"])
        r2_context_dense = float(
            _fit_cv(X_context[dense_local], Yb[dense_local], dense_folds)["r2"]
        )

        cell_out["bases"][basis] = {
            "P_out": int(Yb.shape[1]),
            "single_grain": {
                "r2_prefix_battery_excluded_full": r2_prefix_single,
                "r2_context_battery_excluded_full": r2_context_single,
                "r2_prefix_battery_excluded_densecore": r2_prefix_dense,
                "r2_context_battery_excluded_densecore": r2_context_dense,
                "prefix_lambda_indices": fit_prefix.get("lambda_indices"),
                "context_lambda_indices": fit_context.get("lambda_indices"),
            },
            "averaged_grain": avg,
            "ceilings": {
                "prefix_between_prefix_share_full": share_prefix_full,
                "prefix_between_prefix_share_densecore": fgi_dense["share_prefix"],
                "banked_read3_share_prefix_densecore": BANKED_READ3[cell][basis]["share_prefix"],
                "share_query_densecore": fgi_dense["share_query"],
                "share_interaction_densecore": fgi_dense["share_interaction"],
                "context_additive_ceiling_densecore": additive_ceiling_dense,
                "context_mlp_companion_ceiling": mlp_ceiling,
            },
            "fraction_of_ceiling_single_grain_full": {
                "prefix": r2_prefix_single / share_prefix_full if share_prefix_full else None,
                "context_vs_mlp": (r2_context_single / mlp_ceiling) if mlp_ceiling else None,
                "context_vs_additive": r2_context_single / additive_ceiling_dense
                if additive_ceiling_dense
                else None,
            },
            "fraction_of_ceiling_single_grain_densecore": {
                "prefix": r2_prefix_dense / fgi_dense["share_prefix"]
                if fgi_dense["share_prefix"]
                else None,
                "context_vs_mlp": (r2_context_dense / mlp_ceiling) if mlp_ceiling else None,
                "context_vs_additive": r2_context_dense / additive_ceiling_dense
                if additive_ceiling_dense
                else None,
            },
        }
        del Yb, pred_prefix, pred_context
        gc.collect()
        print(
            f"[{cell}/{basis}] prefix single={r2_prefix_single:.4f} "
            f"avg={avg['r2_prefix_averaged']:.4f} | "
            f"context single={r2_context_single:.4f} avg={avg['r2_context_averaged']:.4f} | "
            f"share_prefix_full={share_prefix_full:.4f}",
            flush=True,
        )
    del X_prefix, X_context, Y_stacked
    gc.collect()
    cell_out["wall_s"] = time.monotonic() - t0
    return cell_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="cell_inst_own ambient only; time + exit")
    ap.add_argument("--cell", default=None, help="run ONE cell and merge into the existing JSON")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _jsonl(MANIFEST)
    print(f"manifest rows={len(rows)}", flush=True)

    out_path = OUT / "fair_comparison.json"
    result = {
        "meta": {
            "script": "scripts/issue1092_inline_fair_comparison.py",
            "git_commit": _git_sha(),
            "generated_utc": datetime.now(UTC).isoformat(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "layer": LAYER,
            "manifest_rows": len(rows),
            "fit_rows": "battery-EXCLUDED: stratum != trait_stratum AND not is_eval_only",
            "folds": f"novel-prefix grouped {N_FOLDS}-fold (group_key=prefix_id, FOLD_SEED=0)",
            "target": "pooled t1/t2/t3 (stacked ambient 10752-dim; pca48 = top-48 PCs of stacked)",
            "engine": "REUSE scripts/issue1092_fit_grid.py _fit_cv -> press_fit_predict",
            "banked_read1_is_battery_included": True,
            "provenance": "teacher-forced capture; own-policy greedy answers; battery-excluded",
        },
        "cells": {},
    }
    # Merge into an existing JSON so per-cell runs accumulate (idempotent per cell).
    if out_path.exists() and not args.gate:
        try:
            prev = json.loads(out_path.read_text())
            if isinstance(prev.get("cells"), dict):
                result["cells"].update(prev["cells"])
        except Exception:
            pass

    if args.gate:
        cells = ["cell_inst_own"]
    elif args.cell:
        if args.cell not in CELLS:
            raise SystemExit(f"--cell must be one of {CELLS}, got {args.cell!r}")
        cells = [args.cell]
    else:
        cells = CELLS
    for cell in cells:
        result["cells"][cell] = process_cell(cell, rows, args.gate)
        out_path.write_text(json.dumps(result, indent=2, allow_nan=True))
        print(f"[done] {cell} wall={result['cells'][cell]['wall_s']:.0f}s", flush=True)

    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    t = time.monotonic()
    main()
    print(f"total {time.monotonic() - t:.0f}s", flush=True)

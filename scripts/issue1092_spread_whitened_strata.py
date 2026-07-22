"""#1092 whitened spread + short-prefix strata for the natural-side coherence reads.

Closes two scope gaps in the spread->error / spread->delta reads (deepdive +
avgctx_spread_delta rounds), which used RAW-L2 spread on all 996 prefixes:

  1. WHITENED spread, the #658 a3.5a metric (issue658_inline_a3_5a_coherence.py):
     W = (Sigma + lambda I)^-1 with Sigma pooled over all context vectors and
     lambda = WHITEN_LAMBDA_FRAC * tr(Sigma)/d. spread_W(p) = mean_i
     ||x_i - xbar_p||^2_W (the #658 mean-squared form; sqrt also recorded).
  2. Short-prefix STRATA (n_turns == 1 / <= 2 / > 2): the length-saturation
     story predicts substrate-like spread behavior where saturation is absent.

Correlates both spread forms against the committed per-prefix error arms
(e_ctx + e_prefixend from fair_comparison.json prediction_agreement; e_avgctx +
deltas from the avgctx_spread_delta unit JSONs), overall and per stratum.
Parity gate: recomputed raw spread must match the unit JSONs exactly.

Analysis-only: NO model forward, NO training, NO API, no refits.
"""

from __future__ import annotations

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
from scipy import linalg as sla  # noqa: E402
from scipy import stats  # noqa: E402

STAGE = Path(
    "/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator/issue1092_realistic_crossing"
)
SUMM = STAGE / "analysis_tensors/summaries"
MANIFEST = STAGE / "corpus/manifest.jsonl"
BANKED = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json"
DELTA = PROJECT_ROOT / "eval_results/issue_1092/inline_avgctx_spread_delta"
OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_spread_whitened_strata"

CELLS = ["cell_inst_own", "cell_pre_own"]
LAYER = 14
BASES = ["ambient", "pca48"]
MIN_ROWS_PER_PREFIX = 3
WHITEN_LAMBDA_FRAC = 1e-2  # Source: #658 issue658_inline_a3_5a_coherence.py
PARITY_TOL = 1e-6
STRATA = [
    ("turns_eq1", lambda t: t == 1),
    ("turns_le2", lambda t: t <= 2),
    ("turns_gt2", lambda t: t > 2),
]


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _spearman(x: np.ndarray, y: np.ndarray) -> dict:
    r, p = stats.spearmanr(x, y)
    return {"rho": float(r), "p": float(p), "n": int(len(x))}


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r, p = stats.pearsonr(ex, ey)
    return {"partial_rho": float(r), "p": float(p), "n": int(len(x))}


def process_cell(cell: str, rows: list[dict], banked: dict) -> dict:
    ctx_all = np.load(SUMM / cell / f"context_end_L{LAYER:02d}.npy", mmap_mode="r")
    n0 = min(ctx_all.shape[0], len(rows))
    be_idx = np.asarray(
        [
            i
            for i in range(n0)
            if rows[i].get("stratum") != "trait_stratum" and not rows[i].get("is_eval_only")
        ],
        dtype=np.int64,
    )
    prefix_ids = np.asarray([rows[int(i)].get("prefix_id", "") for i in be_idx])
    X = np.asarray(ctx_all[be_idx], dtype=np.float64)
    del ctx_all

    groups: dict[str, list[int]] = {}
    for i, pid in enumerate(prefix_ids):
        groups.setdefault(str(pid), []).append(i)
    kept = {p: np.asarray(ix) for p, ix in groups.items() if len(ix) >= MIN_ROWS_PER_PREFIX}
    pids = sorted(kept)

    # within-prefix deviations, all rows at once
    dev = np.zeros((sum(kept[p].size for p in pids), X.shape[1]), dtype=np.float64)
    owner = np.zeros(dev.shape[0], dtype=np.int64)
    pos = 0
    for k, p in enumerate(pids):
        ix = kept[p]
        block = X[ix]
        dev[pos : pos + ix.size] = block - block.mean(0, keepdims=True)
        owner[pos : pos + ix.size] = k
        pos += ix.size

    # raw-L2 spread (parity gate vs the committed unit JSON)
    sq = (dev * dev).sum(1)
    counts = np.bincount(owner, minlength=len(pids)).astype(np.float64)
    spread_raw = np.sqrt(np.bincount(owner, weights=sq, minlength=len(pids)) / counts)
    unit = json.loads((DELTA / f"unit_{cell}_ambient.json").read_text())
    ref = np.asarray(unit["per_prefix_spread"], dtype=np.float64)
    parity = float(np.max(np.abs(spread_raw - ref)))
    assert parity < PARITY_TOL, f"raw-spread parity {parity} vs unit JSON"
    n_turns = np.asarray(unit["per_prefix_n_turns"], dtype=np.float64)

    # whitened spread (#658 recipe: pooled Sigma over ALL context vectors,
    # lambda = frac * tr(Sigma)/d, W = (Sigma + lambda I)^-1)
    Xc = X - X.mean(0, keepdims=True)
    Sigma = (Xc.T @ Xc) / (X.shape[0] - 1)
    lam = WHITEN_LAMBDA_FRAC * (np.trace(Sigma) / Sigma.shape[0])
    L = np.linalg.cholesky(Sigma + lam * np.eye(Sigma.shape[0]))
    Z = sla.solve_triangular(L, dev.T, lower=True).T  # whitened deviations
    wsq = (Z * Z).sum(1)
    spread_w_meansq = np.bincount(owner, weights=wsq, minlength=len(pids)) / counts
    spread_w = np.sqrt(spread_w_meansq)
    del X, Xc, dev, Z

    # committed error arms
    arms_by_basis = {}
    for basis in BASES:
        pa = banked["cells"][cell]["bases"][basis]["prediction_agreement"]
        u = json.loads((DELTA / f"unit_{cell}_{basis}.json").read_text())
        e_avgctx = np.asarray(u["per_prefix_err_avgctx"], dtype=np.float64)
        e_pe = np.asarray(pa["per_prefix_err_prefix"], dtype=np.float64)
        e_ctx = np.asarray(pa["per_prefix_err_ctx"], dtype=np.float64)
        arms_by_basis[basis] = {
            "e_avgctx": e_avgctx,
            "e_prefixend": e_pe,
            "e_ctx": e_ctx,
            "d_pe": e_avgctx - e_pe,
            "d_ctx": e_avgctx - e_ctx,
        }

    def _reads(mask: np.ndarray) -> dict:
        out: dict = {"n": int(mask.sum())}
        for basis in BASES:
            arms = arms_by_basis[basis]
            b: dict = {}
            for spread_name, s in [("spread_raw", spread_raw), ("spread_whitened", spread_w)]:
                for arm_name, e in arms.items():
                    b[f"{spread_name}_vs_{arm_name}"] = _spearman(s[mask], e[mask])
                    if mask.sum() == len(pids) or n_turns[mask].std() > 0:
                        b[f"{spread_name}_vs_{arm_name}_given_nturns"] = _partial_spearman(
                            s[mask], e[mask], n_turns[mask]
                        )
            out[basis] = b
        out["spread_whitened_vs_nturns"] = _spearman(spread_w[mask], n_turns[mask])
        out["spread_raw_vs_spread_whitened"] = _spearman(spread_raw[mask], spread_w[mask])
        return out

    result = {
        "n_prefixes": len(pids),
        "raw_spread_parity_vs_unit_json": parity,
        "whiten_lambda": float(lam),
        "whiten_lambda_frac": WHITEN_LAMBDA_FRAC,
        "overall": _reads(np.ones(len(pids), dtype=bool)),
        "strata": {name: _reads(fn(n_turns)) for name, fn in STRATA},
        "per_prefix": {
            "spread_whitened": [float(x) for x in spread_w],
            "spread_whitened_meansq": [float(x) for x in spread_w_meansq],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / f"per_prefix_whitened_{cell}.npz",
        spread_raw=spread_raw,
        spread_whitened=spread_w,
        n_turns=n_turns,
    )
    o = result["overall"]["ambient"]
    print(
        f"[{cell}] whitened: spread_w->e_avgctx rho="
        f"{o['spread_whitened_vs_e_avgctx']['rho']:+.3f} "
        f"spread_w->d_pe rho={o['spread_whitened_vs_d_pe']['rho']:+.3f} "
        f"spread_w<->raw rho={result['overall']['spread_raw_vs_spread_whitened']['rho']:+.3f} "
        f"| turns_eq1 n={result['strata']['turns_eq1']['n']} "
        f"raw->e_ctx rho={result['strata']['turns_eq1']['ambient']['spread_raw_vs_e_ctx']['rho']:+.3f}",
        flush=True,
    )
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    banked = json.loads(BANKED.read_text())
    rows = _jsonl(MANIFEST)
    result: dict = {
        "meta": {
            "script": "scripts/issue1092_spread_whitened_strata.py",
            "whitening_source": "scripts/issue658_inline_a3_5a_coherence.py (WHITEN_LAMBDA_FRAC=1e-2)",
            "manifest_rows": len(rows),
        },
        "cells": {},
    }
    for cell in CELLS:
        unit_path = OUT / f"cell_{cell}.json"
        if unit_path.exists():
            result["cells"][cell] = json.loads(unit_path.read_text())
            print(f"[resume] skipping completed cell {cell}", flush=True)
            continue
        blk = process_cell(cell, rows, banked)
        unit_path.write_text(json.dumps(blk, indent=2))
        result["cells"][cell] = blk
    (OUT / "spread_whitened_strata.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT / 'spread_whitened_strata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""#1092 round-2 cross-join: whitened spread x MLP Jensen gap + target-side spread.

Joins the two round-2 jobs' per-prefix arrays (inline_spread_whitened_strata +
inline_mlp_jensen_natural) and adds the ANSWER-side within-prefix spread — the
target-noise diagnostic: if context spread and answer spread co-vary strongly,
part of "spread predicts every arm's error" is the averaged TARGET Y_avg being
noisier for dispersed prefixes (an error floor common to all arms), not map
failure.

Reads per cell:
  - spread_raw / spread_whitened / n_turns  (job A npz)
  - jensen / d_mlp                          (job B npz)
  - e_avgctx / d_pe                         (round-1 unit JSONs, ambient)
Computes per cell:
  - answer-side within-prefix spread of the pca48 targets (raw + whitened with
    the #658 lambda recipe, pooled 48-dim covariance)
  - the decisive correlations: spread_w -> jensen / d_mlp (+ partial | n_turns),
    context-vs-answer spread coupling, and the difficulty-vs-target-noise
    partials: (spread_w -> e_avgctx | answer_spread_w) and
    (answer_spread_w -> e_avgctx | spread_w).

Analysis-only; seconds of compute beyond the pca48 target load.
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
from scipy import stats  # noqa: E402

from issue1092_fit_grid import _basis_targets_with_info  # noqa: E402

STAGE = Path(
    "/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator/issue1092_realistic_crossing"
)
SUMM = STAGE / "analysis_tensors/summaries"
MANIFEST = STAGE / "corpus/manifest.jsonl"
DELTA = PROJECT_ROOT / "eval_results/issue_1092/inline_avgctx_spread_delta"
WHITE = PROJECT_ROOT / "eval_results/issue_1092/inline_spread_whitened_strata"
JENSEN = PROJECT_ROOT / "eval_results/issue_1092/inline_mlp_jensen_natural"
OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_spread_crossjoin"

CELLS = ["cell_inst_own", "cell_pre_own"]
LAYER = 14
TARGETS = ["t1", "t2", "t3"]
HIDDEN_DIM = 3584
MIN_ROWS_PER_PREFIX = 3
WHITEN_LAMBDA_FRAC = 1e-2  # Source: #658 issue658_inline_a3_5a_coherence.py
PARITY_TOL = 1e-6


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _spearman(x, y) -> dict:
    r, p = stats.spearmanr(x, y)
    return {"rho": float(r), "p": float(p), "n": int(len(x))}


def _partial_spearman(x, y, z) -> dict:
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r, p = stats.pearsonr(ex, ey)
    return {"partial_rho": float(r), "p": float(p), "n": int(len(x))}


def process_cell(cell: str, rows: list[dict]) -> dict:
    a = np.load(WHITE / f"per_prefix_whitened_{cell}.npz")
    b = np.load(JENSEN / f"per_prefix_jensen_{cell}.npz")
    assert float(np.max(np.abs(a["spread_raw"] - b["spread"]))) < PARITY_TOL, "npz misaligned"
    spread_raw = a["spread_raw"]
    spread_w = a["spread_whitened"]
    n_turns = a["n_turns"]
    jensen = b["jensen"]
    d_mlp = b["d_mlp"]
    u = json.loads((DELTA / f"unit_{cell}_ambient.json").read_text())
    e_avgctx = np.asarray(u["per_prefix_err_avgctx"], dtype=np.float64)
    d_pe = e_avgctx - np.asarray(
        json.loads(
            (
                PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json"
            ).read_text()
        )["cells"][cell]["bases"]["ambient"]["prediction_agreement"]["per_prefix_err_prefix"],
        dtype=np.float64,
    )

    # answer-side within-prefix spread on the pca48 targets
    t_all = [np.load(SUMM / cell / f"{t}_L{LAYER:02d}.npy", mmap_mode="r") for t in TARGETS]
    n0 = min(min(t.shape[0] for t in t_all), len(rows))
    be_idx = np.asarray(
        [
            i
            for i in range(n0)
            if rows[i].get("stratum") != "trait_stratum" and not rows[i].get("is_eval_only")
        ],
        dtype=np.int64,
    )
    prefix_ids = np.asarray([rows[int(i)].get("prefix_id", "") for i in be_idx])
    Y_stacked = np.concatenate([np.asarray(t[be_idx], dtype=np.float64) for t in t_all], axis=1)
    del t_all
    Yb = _basis_targets_with_info(
        Y_stacked, "pca48", hidden_dim=HIDDEN_DIM, targets=TARGETS, projection_target="t1"
    )[0]
    Yb = np.ascontiguousarray(Yb, dtype=np.float64)
    del Y_stacked

    groups: dict[str, list[int]] = {}
    for i, pid in enumerate(prefix_ids):
        groups.setdefault(str(pid), []).append(i)
    kept = {p: np.asarray(ix) for p, ix in groups.items() if len(ix) >= MIN_ROWS_PER_PREFIX}
    pids = sorted(kept)
    assert len(pids) == len(spread_raw), (len(pids), len(spread_raw))

    Ybc = Yb - Yb.mean(0, keepdims=True)
    SigY = (Ybc.T @ Ybc) / (Yb.shape[0] - 1)
    lamY = WHITEN_LAMBDA_FRAC * (np.trace(SigY) / SigY.shape[0])
    LY = np.linalg.cholesky(SigY + lamY * np.eye(SigY.shape[0]))
    aspread_raw = np.zeros(len(pids))
    aspread_w = np.zeros(len(pids))
    for k, p in enumerate(pids):
        block = Yb[kept[p]]
        c = block - block.mean(0, keepdims=True)
        aspread_raw[k] = float(np.sqrt((c * c).sum(1).mean()))
        z = np.linalg.solve(LY, c.T).T  # LY lower-triangular; 48-dim solve is trivial
        aspread_w[k] = float(np.sqrt((z * z).sum(1).mean()))

    blk = {
        "cell": cell,
        "n_prefixes": len(pids),
        # the decisive curvature read
        "spread_w_vs_jensen": _spearman(spread_w, jensen),
        "spread_w_vs_jensen_given_nturns": _partial_spearman(spread_w, jensen, n_turns),
        "spread_w_vs_d_mlp": _spearman(spread_w, d_mlp),
        "spread_w_vs_d_mlp_given_nturns": _partial_spearman(spread_w, d_mlp, n_turns),
        "spread_raw_vs_jensen": _spearman(spread_raw, jensen),
        # context <-> answer spread coupling (the target-noise candidate)
        "ctx_spread_raw_vs_answer_spread_raw": _spearman(spread_raw, aspread_raw),
        "ctx_spread_w_vs_answer_spread_w": _spearman(spread_w, aspread_w),
        "answer_spread_w_vs_nturns": _spearman(aspread_w, n_turns),
        # difficulty vs target-noise discrimination on the averaged-map error
        "spread_w_vs_e_avgctx_given_answer_spread_w": _partial_spearman(
            spread_w, e_avgctx, aspread_w
        ),
        "answer_spread_w_vs_e_avgctx_given_spread_w": _partial_spearman(
            aspread_w, e_avgctx, spread_w
        ),
        "answer_spread_w_vs_e_avgctx": _spearman(aspread_w, e_avgctx),
        "answer_spread_w_vs_d_pe": _spearman(aspread_w, d_pe),
        "answer_spread_w_vs_jensen": _spearman(aspread_w, jensen),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / f"per_prefix_crossjoin_{cell}.npz",
        spread_raw=spread_raw,
        spread_whitened=spread_w,
        answer_spread_raw=aspread_raw,
        answer_spread_whitened=aspread_w,
        jensen=jensen,
        d_mlp=d_mlp,
        e_avgctx=e_avgctx,
        d_pe=d_pe,
        n_turns=n_turns,
    )
    print(
        f"[{cell}] spread_w->J rho={blk['spread_w_vs_jensen']['rho']:+.3f} "
        f"(|len {blk['spread_w_vs_jensen_given_nturns']['partial_rho']:+.3f}) "
        f"ctxW<->ansW rho={blk['ctx_spread_w_vs_answer_spread_w']['rho']:+.3f} "
        f"e_avgctx: spreadW|ansW={blk['spread_w_vs_e_avgctx_given_answer_spread_w']['partial_rho']:+.3f} "
        f"ansW|spreadW={blk['answer_spread_w_vs_e_avgctx_given_spread_w']['partial_rho']:+.3f}",
        flush=True,
    )
    return blk


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _jsonl(MANIFEST)
    result = {
        "meta": {"script": "scripts/issue1092_spread_crossjoin.py"},
        "cells": {c: process_cell(c, rows) for c in CELLS},
    }
    (OUT / "spread_crossjoin.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT / 'spread_crossjoin.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

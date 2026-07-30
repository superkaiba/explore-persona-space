"""#1092 NONLINEAR (MLP) Jensen test on natural prefixes — the real coherence read.

A linear ridge commutes with within-prefix averaging exactly (folds are grouped
by prefix, predictions are affine), so the linear delta arms are structurally
blind to curvature — #658's a3.5a construction says the same ("linear ridge
companion is ~0 by construction"). This script runs the genuine test: fit a
NONLINEAR row-grain context map h and measure, per prefix,

  J(p)   = || mean_i h(x_i) - h(xbar_p) ||_2      (the Jensen gap)
  d_mlp  = err(h(xbar_p)) - err(mean_i h(x_i))    (its error consequence)

both out-of-fold (h applied to prefix p's rows AND centroid comes from the fold
model that held prefix p out), then tests whether within-prefix spread predicts
them. The coherence condition predicts J and d_mlp grow with spread.

MLP recipe reused from #658's a3.5a Jensen map (issue658_inline_a3_5a_coherence.py):
PCA_IN=256 input projection, hidden 512, 250 epochs, Adam lr 1e-3, wd 1e-4 —
which also makes the natural-side read directly comparable to the substrate one.
Target: the pca48 answer-summary basis (the fair-comparison pooled t1/t2/t3
pca48 targets), same 6-fold-over-prefixes splits (FOLD_SEED=0) as the ridge arms.
fit_batched_loco_mlp_multihead is NOT reusable here: it returns held-out row
predictions only and exposes no fold weights to apply to centroids.

Checkpointed per cell. Analysis-only: no model forward, no API; CPU torch fit.
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
    _folds_from_manifest,
    _r2,
)

STAGE = Path(
    os.environ.get(
        "I1092_STAGE_DIR",
        "/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator/issue1092_realistic_crossing",
    )
)
SUMM = STAGE / "analysis_tensors/summaries"
MANIFEST = STAGE / "corpus/manifest.jsonl"
DELTA = PROJECT_ROOT / "eval_results/issue_1092/inline_avgctx_spread_delta"
OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_mlp_jensen_natural"

CELLS = ["cell_inst_own", "cell_pre_own"]
LAYER = 14
TARGETS = ["t1", "t2", "t3"]
HIDDEN_DIM = 3584
N_FOLDS = 6
MIN_ROWS_PER_PREFIX = 3
PARITY_TOL = 1e-6
SEED = 0
# Source: #658 issue658_inline_a3_5a_coherence.py Jensen-map recipe.
PCA_IN = 256
MLP_HIDDEN = 512
MLP_EPOCHS = 250
MLP_LR = 1e-3
MLP_WD = 1e-4


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load(cell: str, kind: str) -> np.ndarray:
    return np.load(SUMM / cell / f"{kind}_L{LAYER:02d}.npy", mmap_mode="r")


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


class _MLP(torch.nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def process_cell(
    cell: str, rows: list[dict], *, out_dir: Path = OUT, persist_gap: bool = False
) -> dict:
    ctx_all = _load(cell, "context_end")
    t_shapes = [_load(cell, t).shape[0] for t in TARGETS]
    n0 = min(ctx_all.shape[0], min(t_shapes), len(rows))
    be_idx = np.asarray(
        [
            i
            for i in range(n0)
            if rows[i].get("stratum") != "trait_stratum" and not rows[i].get("is_eval_only")
        ],
        dtype=np.int64,
    )
    prefix_ids = np.asarray([rows[int(i)].get("prefix_id", "") for i in be_idx])
    unit_rows = [rows[int(i)] for i in be_idx]
    X = np.asarray(ctx_all[be_idx], dtype=np.float64)
    del ctx_all

    groups: dict[str, list[int]] = {}
    for i, pid in enumerate(prefix_ids):
        groups.setdefault(str(pid), []).append(i)
    kept = {p: np.asarray(ix) for p, ix in groups.items() if len(ix) >= MIN_ROWS_PER_PREFIX}
    pids = sorted(kept)

    # parity gate vs the committed unit JSON (row set + grouping identical)
    spread = np.zeros(len(pids), dtype=np.float64)
    for k, p in enumerate(pids):
        block = X[kept[p]]
        c = block - block.mean(0, keepdims=True)
        spread[k] = float(np.sqrt((c * c).sum(1).mean()))
    unit = json.loads((DELTA / f"unit_{cell}_ambient.json").read_text())
    ref = np.asarray(unit["per_prefix_spread"], dtype=np.float64)
    parity = float(np.max(np.abs(spread - ref)))
    assert parity < PARITY_TOL, f"spread parity {parity} vs unit JSON"
    n_turns = np.asarray(unit["per_prefix_n_turns"], dtype=np.float64)

    # pca48 targets (fair-comparison basis machinery, pooled t1/t2/t3)
    Y_stacked = np.concatenate(
        [np.asarray(_load(cell, t)[be_idx], dtype=np.float64) for t in TARGETS], axis=1
    )
    Yb = _basis_targets_with_info(
        Y_stacked, "pca48", hidden_dim=HIDDEN_DIM, targets=TARGETS, projection_target="t1"
    )[0]
    Yb = np.ascontiguousarray(Yb, dtype=np.float64)
    del Y_stacked
    gc.collect()
    Y_avg = np.stack([Yb[kept[p]].mean(0) for p in pids], axis=0)

    # PCA-256 input projection (deterministic economy SVD on the centered rows)
    mu = X.mean(0, keepdims=True)
    Xc32 = (X - mu).astype(np.float32)
    del X
    gc.collect()
    torch.manual_seed(SEED)
    U, S, V = torch.pca_lowrank(torch.from_numpy(Xc32), q=PCA_IN + 24, center=False, niter=4)
    basis = V[:, :PCA_IN].numpy()  # (3584, 256)
    Xp = Xc32 @ basis  # (n_rows, 256) float32
    del Xc32, U, S, V
    gc.collect()
    # per-prefix centroids in the SAME projection (projection is linear)
    Xp_cent = np.stack([Xp[kept[p]].mean(0) for p in pids], axis=0)

    folds = _folds_from_manifest(unit_rows, len(unit_rows), group_key="prefix_id", n_folds=N_FOLDS)
    pid_to_k = {p: k for k, p in enumerate(pids)}
    row_pred = np.zeros((Xp.shape[0], Yb.shape[1]), dtype=np.float64)
    cent_pred = np.zeros((len(pids), Yb.shape[1]), dtype=np.float64)
    n_rows = Xp.shape[0]
    for fi, test_idx in enumerate(folds):
        mask = np.ones(n_rows, dtype=bool)
        mask[test_idx] = False
        xm = Xp[mask].mean(0, keepdims=True)
        xs = Xp[mask].std(0, keepdims=True) + 1e-6
        ym = Yb[mask].mean(0, keepdims=True)
        Xtr = torch.from_numpy((Xp[mask] - xm) / xs)
        Ytr = torch.from_numpy((Yb[mask] - ym).astype(np.float32))
        torch.manual_seed(SEED + fi)
        net = _MLP(PCA_IN, MLP_HIDDEN, Yb.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
        for _ in range(MLP_EPOCHS):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xtr), Ytr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            xt = torch.from_numpy((Xp[test_idx] - xm) / xs)
            row_pred[test_idx] = net(xt).numpy().astype(np.float64) + ym
            # centroids of the prefixes held out in THIS fold
            held_pids = sorted({str(prefix_ids[i]) for i in test_idx if str(prefix_ids[i]) in kept})
            kk = np.asarray([pid_to_k[p] for p in held_pids], dtype=np.int64)
            xc = torch.from_numpy((Xp_cent[kk] - xm) / xs)
            cent_pred[kk] = net(xc).numpy().astype(np.float64) + ym
        print(f"[{cell}] fold {fi + 1}/{N_FOLDS} done (train loss {float(loss):.4f})", flush=True)

    pred_avg = np.stack([row_pred[kept[p]].mean(0) for p in pids], axis=0)
    jensen = np.linalg.norm(pred_avg - cent_pred, axis=1)
    e_avgpred = np.linalg.norm(pred_avg - Y_avg, axis=1)
    e_centroid = np.linalg.norm(cent_pred - Y_avg, axis=1)
    d_mlp = e_centroid - e_avgpred
    r2_rows = _r2(Yb, row_pred)

    blk = {
        "cell": cell,
        "n_prefixes": len(pids),
        "spread_parity_vs_unit_json": parity,
        "mlp_recipe": {
            "source": "#658 issue658_inline_a3_5a_coherence.py",
            "pca_in": PCA_IN,
            "hidden": MLP_HIDDEN,
            "epochs": MLP_EPOCHS,
            "lr": MLP_LR,
            "wd": MLP_WD,
            "target": "pca48 pooled t1/t2/t3",
        },
        "r2_rowgrain_heldout": r2_rows,
        "mean": {
            "jensen_gap": float(jensen.mean()),
            "err_centroid": float(e_centroid.mean()),
            "err_avgpred": float(e_avgpred.mean()),
        },
        "spearman_spread_vs_jensen": _spearman(spread, jensen),
        "spearman_spread_vs_d_mlp": _spearman(spread, d_mlp),
        "spearman_spread_vs_err_centroid": _spearman(spread, e_centroid),
        "partial_spread_vs_jensen_given_nturns": _partial_spearman(spread, jensen, n_turns),
        "partial_spread_vs_d_mlp_given_nturns": _partial_spearman(spread, d_mlp, n_turns),
        "spearman_nturns_vs_jensen": _spearman(n_turns, jensen),
        "strata": {},
        "per_prefix": {
            "jensen_gap": [float(x) for x in jensen],
            "d_mlp": [float(x) for x in d_mlp],
            "err_centroid": [float(x) for x in e_centroid],
            "err_avgpred": [float(x) for x in e_avgpred],
        },
    }
    for name, fn in [("turns_eq1", n_turns == 1), ("turns_le2", n_turns <= 2)]:
        m = fn
        blk["strata"][name] = {
            "n": int(m.sum()),
            "spread_vs_jensen": _spearman(spread[m], jensen[m]),
            "spread_vs_d_mlp": _spearman(spread[m], d_mlp[m]),
        }
    if persist_gap:
        # #1774 Q1c: banked scalar npz (996 prefixes) reproduces jensen norms
        # within tolerance on the prefix overlap — the refit's norm cross-check.
        banked = OUT / f"per_prefix_jensen_{cell}.npz"
        if banked.exists() and banked.resolve() != (out_dir / banked.name).resolve():
            with np.load(banked) as z:
                if "jensen" in z.files and len(z["jensen"]) == len(jensen):
                    max_abs = float(np.max(np.abs(np.asarray(z["jensen"]) - jensen)))
                    blk["banked_norm_crosscheck"] = {
                        "max_abs_jensen_diff": max_abs,
                        "n_overlap": int(len(jensen)),
                    }
                else:
                    blk["banked_norm_crosscheck"] = {
                        "skipped": f"banked n={len(z['jensen']) if 'jensen' in z.files else 0} "
                        f"!= refit n={len(jensen)} — overlap join left to the analyzer"
                    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_arrays = {
        "jensen": jensen,
        "d_mlp": d_mlp,
        "e_centroid": e_centroid,
        "e_avgpred": e_avgpred,
        "spread": spread,
        "n_turns": n_turns,
    }
    if persist_gap:
        # per-prefix Jensen-gap VECTORS in the pca48 target basis (H1c read),
        # + the per-coordinate target variance (the trivial-alternative
        # reference the concentration curve is reported against).
        save_arrays.update(
            gap_vectors=(pred_avg - cent_pred),
            pred_avg=pred_avg,
            cent_pred=cent_pred,
            prefix_ids=np.asarray(pids),
            target_var_per_coord=Yb.var(axis=0, ddof=1),
        )
    np.savez(out_dir / f"per_prefix_jensen_{cell}.npz", **save_arrays)
    print(
        f"[{cell}] r2_rows={r2_rows:.4f} jensen_mean={jensen.mean():.3f} "
        f"spread->J rho={blk['spearman_spread_vs_jensen']['rho']:+.3f} "
        f"spread->d_mlp rho={blk['spearman_spread_vs_d_mlp']['rho']:+.3f} "
        f"(J|len rho={blk['partial_spread_vs_jensen_given_nturns']['partial_rho']:+.3f})",
        flush=True,
    )
    return blk


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--persist-gap-vectors",
        action="store_true",
        help="#1774 Q1c: additionally persist per-prefix gap VECTORS (pca48 "
        "basis) + pred_avg/cent_pred/prefix_ids/target_var_per_coord in the "
        "npz (same recipe verbatim; scalar outputs unchanged)",
    )
    ap.add_argument("--cells", default=",".join(CELLS), help="comma list (default both)")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output dir override (default: the banked #1092 location; a "
        "gap-vector refit MUST use a fresh dir so the banked scalar npz — the "
        "norm cross-check reference — is never overwritten)",
    )
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else OUT
    if args.persist_gap_vectors and out_dir.resolve() == OUT.resolve():
        raise SystemExit(
            "--persist-gap-vectors requires --out-dir != the banked #1092 dir "
            "(the banked scalar npz is the norm cross-check reference; never overwrite it)"
        )
    cells = [c for c in args.cells.split(",") if c]
    assert set(cells) <= set(CELLS), cells

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _jsonl(MANIFEST)
    result: dict = {
        "meta": {
            "script": "scripts/issue1092_mlp_jensen_natural.py",
            "persist_gap_vectors": bool(args.persist_gap_vectors),
            "note": (
                "out-of-fold Jensen gap: h fit 6-fold over prefixes (FOLD_SEED=0); "
                "h(centroid) uses the fold model that held the prefix out; a linear map "
                "commutes with within-prefix averaging exactly, so J is a curvature read"
            ),
        },
        "cells": {},
    }
    for cell in cells:
        unit_path = out_dir / f"cell_{cell}.json"
        npz_path = out_dir / f"per_prefix_jensen_{cell}.npz"
        # resume predicate keys on the output-affecting regime (gap persistence
        # is part of the key — #722 r3): a unit done WITHOUT gap vectors does
        # not satisfy a --persist-gap-vectors run.
        done = unit_path.exists()
        if done and args.persist_gap_vectors:
            if not npz_path.exists():
                done = False
            else:
                with np.load(npz_path) as z:
                    done = "gap_vectors" in z.files
        if done:
            result["cells"][cell] = json.loads(unit_path.read_text())
            print(f"[resume] skipping completed cell {cell}", flush=True)
            continue
        blk = process_cell(cell, rows, out_dir=out_dir, persist_gap=args.persist_gap_vectors)
        unit_path.write_text(json.dumps(blk, indent=2))
        result["cells"][cell] = blk
        gc.collect()
    (out_dir / "mlp_jensen_natural.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {out_dir / 'mlp_jensen_natural.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

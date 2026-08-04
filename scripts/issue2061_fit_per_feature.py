"""P2 per-feature ridge fit for task #2061.

Reads:
- P1 output: `data/issue_2061/sae_encoded/<stage>_<render>_<corpus>_L29.pt` —
  SAE-encoded answer targets (n_rows, d_sae=262144), from
  `scripts/issue2061_sae_encode.py`.
- #1336's banked context activations (prefix + context arms), same shard as P1.

Fits, per (stage, render, corpus, arm) cell:
- K=5 group folds, fold seed 0 (inherited from #1336, plan §10).
- Shared-factorization ridge via `ridge_fit_predict_fast_layer_batched`
  (`src/explore_persona_space/experiments/issue_779/fit_h.py:180`); GCV over
  the #823/#779 grid `np.logspace(-2, 4, 13)`.
- Per-feature R²_j = 1 − ||f_j − ĝ_j||² / ||f_j − mean(f_j)||² on held-out
  folds, pooled with fold-local test means.
- kNN retrieval (euclidean + cosine) via `analysis/mapping_baselines.knn_retrieval`.

**GCV under-determined regime diagnostics (plan §7 mitigation).** On cells
with per-fold `n_train < d_in=4096` (only `gsm8k_test1319` per the
reconciliation table), we report per-fit `best_lambda` + `effective_dof`
via `return_info=True`. A WARN fires when `effective_dof > 0.9 * n_train`
(the #1887 dof-cap 0.9 target); production dof-capping is deferred to the
upstream fit function once it lands on `main` (see plan §11 "Under-
determined-cell mitigation" row).

Emits `eval_results/issue_2061/per_feature_r2/<stage>_<corpus>_<arm>_L29.jsonl`
— one JSON object per feature with keys:
  {feature_id, R2, n_train, n_test, best_lambda, effective_dof,
   knn_acc_1_euclid, knn_acc_10_euclid, knn_acc_1_cosine, knn_acc_10_cosine,
   chance_1, chance_10}

Usage:
  uv run python scripts/issue2061_fit_per_feature.py \\
      --stage base --render chat --corpus lmsys23k --arm context
  uv run python scripts/issue2061_fit_per_feature.py --all-cells
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.mapping_baselines import knn_retrieval
from explore_persona_space.experiments.issue_779.fit_h import (
    ridge_fit_predict_fast_layer_batched,
)

LAYER = 29
D_IN = 4096
K_FOLDS = 5
FOLD_SEED = 0
LAMBDA_GRID = np.logspace(-2, 4, 13)
DOF_CAP_FRACTION = 0.9  # plan §7 under-determined-cell mitigation


def _load_context_activations(shard_path: Path, arm: str, layer: int = LAYER) -> np.ndarray:
    """Load context activations for one turnstore's shard.

    `arm` selects prefix vs context pooling. Schema fallback is the same as
    the encode script — verify at first launch.
    """
    obj = torch.load(shard_path, map_location="cpu", weights_only=True)
    key_candidates = {
        "prefix": [
            f"prefix_L{layer}",
            f"prefix_layer_{layer}",
            "prefix",
        ],
        "context": [
            f"context_L{layer}",
            f"context_layer_{layer}",
            "context",
        ],
    }
    if isinstance(obj, dict):
        for k in key_candidates[arm]:
            if k in obj:
                acts = obj[k]
                if acts.ndim == 3:
                    acts = acts[:, layer, :]
                return acts.float().numpy()
        raise KeyError(f"No {arm}-arm key in {shard_path} (keys: {list(obj.keys())})")
    raise TypeError(f"Unrecognized shard: {shard_path} type={type(obj)}")


def _make_folds(n: int, k: int = K_FOLDS, seed: int = FOLD_SEED) -> list[np.ndarray]:
    """K-fold indices with fixed seed (matches #1336's fold convention)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    return [np.array(f, dtype=np.int64) for f in folds]


def fit_cell(
    context_shard: Path,
    encoded_shard: Path,
    arm: str,
    output_path: Path,
    layer: int = LAYER,
    device: str = "cpu",
) -> None:
    """Fit ridge for one (stage, render, corpus, arm) cell + write JSONL."""
    print(f"[fit] context={context_shard.name} arm={arm} encoded={encoded_shard.name}")
    X = _load_context_activations(context_shard, arm, layer=layer)  # (n, d_in)
    Y = (
        torch.load(encoded_shard, map_location="cpu", weights_only=True).float().numpy()
    )  # (n, d_sae)
    assert X.shape[0] == Y.shape[0], f"row mismatch: X={X.shape}, Y={Y.shape}"
    n, d_in = X.shape
    d_sae = Y.shape[1]
    print(f"  n={n} d_in={d_in} d_sae={d_sae}")

    folds = _make_folds(n, k=K_FOLDS, seed=FOLD_SEED)
    # Pooled per-feature R² with fold-local test means: track SS_res + SS_tot
    # per feature across folds. Also track per-fit best_lambda / dof for the
    # dof-cap diagnostic.
    ss_res = np.zeros(d_sae, dtype=np.float64)
    ss_tot = np.zeros(d_sae, dtype=np.float64)
    per_fold_lambda: list[float] = []
    per_fold_dof: list[float] = []
    per_fold_ntrain: list[int] = []

    # Collect predictions on held-out folds for the pooled kNN retrieval read.
    Y_pred_pool = np.empty_like(Y, dtype=np.float64)

    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fi])
        Xtr = X[train_idx][None, :, :]  # (1, ntr, d_in) — one slice
        Ytr = Y[train_idx][None, :, :].astype(np.float64)  # (1, ntr, d_sae)
        Xev = X[test_idx][None, :, :]  # (1, nev, d_in)

        n_train = int(train_idx.shape[0])
        n_test = int(test_idx.shape[0])
        per_fold_ntrain.append(n_train)
        t0 = time.time()
        preds, info = ridge_fit_predict_fast_layer_batched(
            Xtr,
            Ytr,
            Xev,
            lambdas=LAMBDA_GRID,
            device=device,
            return_info=True,
        )
        elapsed = time.time() - t0
        best_lam = float(info["best_lambda"][0])
        dof = float(info["dof"][0])
        per_fold_lambda.append(best_lam)
        per_fold_dof.append(dof)

        # Under-determined-regime WARN (plan §7 mitigation).
        if n_train < D_IN and dof > DOF_CAP_FRACTION * n_train:
            print(
                f"  [WARN] fold {fi}: n_train={n_train} < d_in={D_IN} AND "
                f"effective_dof={dof:.1f} > {DOF_CAP_FRACTION}*n_train="
                f"{DOF_CAP_FRACTION * n_train:.1f} — GCV dof-cap should engage"
            )

        # Per-feature R² accumulation, pooled with fold-local test means.
        Y_test = Y[test_idx].astype(np.float64)  # (nev, d_sae)
        Y_test_mean = Y_test.mean(axis=0)  # (d_sae,)
        pred = preds[0]  # (nev, d_sae)
        ss_res += ((Y_test - pred) ** 2).sum(axis=0)
        ss_tot += ((Y_test - Y_test_mean) ** 2).sum(axis=0)
        Y_pred_pool[test_idx] = pred

        print(
            f"  fold {fi}: n_train={n_train} n_test={n_test} λ={best_lam:.3g} dof={dof:.1f} ({elapsed:.1f}s)"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - np.where(ss_tot > 0, ss_res / ss_tot, np.nan)

    # kNN retrieval on the pooled OOF predictions vs the fixed target pool.
    k_ret = max(1, min(10, n // 20))  # k = ceil(n/20) capped at 10
    knn_e = knn_retrieval(Y_pred_pool, Y.astype(np.float64), ks=(1, k_ret), metric="euclidean")
    knn_c = knn_retrieval(Y_pred_pool, Y.astype(np.float64), ks=(1, k_ret), metric="cosine")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for j in range(d_sae):
            f.write(
                json.dumps(
                    {
                        "feature_id": j,
                        "R2": None if np.isnan(r2[j]) else float(r2[j]),
                        "n_train_folds": per_fold_ntrain,
                        "n_test_total": int(n),
                        "best_lambda_folds": per_fold_lambda,
                        "effective_dof_folds": per_fold_dof,
                        "knn_acc_1_euclid": float(knn_e["acc_at_k"][0]),
                        "knn_acc_k_euclid": float(knn_e["acc_at_k"][1]),
                        "knn_k_ret": int(k_ret),
                        "knn_acc_1_cosine": float(knn_c["acc_at_k"][0]),
                        "knn_acc_k_cosine": float(knn_c["acc_at_k"][1]),
                        "chance_1": float(knn_e["chance_at_k"][0]),
                        "chance_k": float(knn_e["chance_at_k"][1]),
                    }
                )
                + "\n"
            )
    print(f"[done] wrote {d_sae} rows to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("--render", type=str, default=None)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--arm", choices=["prefix", "context"], default=None)
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument(
        "--context-shard-dir",
        type=Path,
        required=True,
        help="Directory holding #1336 context shards (staged locally).",
    )
    parser.add_argument("--encoded-dir", type=Path, default=Path("data/issue_2061/sae_encoded"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("eval_results/issue_2061/per_feature_r2")
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu' or 'cuda' (cpu is fine for this regime: n_train < d_in^2)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoded_files = sorted(args.encoded_dir.glob("*_L*.pt"))
    if not encoded_files:
        print(f"[error] No encoded shards in {args.encoded_dir}")
        return 1

    def parse_cell(path: Path) -> tuple[str, str, str]:
        # <stage>_<render>_<corpus>_L<layer>.pt
        parts = path.stem.rsplit("_", 3)
        return parts[0], parts[1], parts[2]

    targets: list[tuple[Path, str, str, str]] = []
    for enc_path in encoded_files:
        stage, render, corpus = parse_cell(enc_path)
        if not args.all_cells:
            if args.stage and stage != args.stage:
                continue
            if args.render and render != args.render:
                continue
            if args.corpus and corpus != args.corpus:
                continue
        for arm in ["prefix", "context"] if args.arm is None else [args.arm]:
            targets.append((enc_path, stage, render, corpus, arm))

    if not targets:
        print("[error] No cell matches filters")
        return 1
    print(f"[setup] Fitting {len(targets)} (cell, arm) combos")

    for i, (enc_path, stage, render, corpus, arm) in enumerate(targets, start=1):
        print(f"\n=== [{i}/{len(targets)}] {stage}/{render}/{corpus}/{arm} ===")
        context_shard = (
            args.context_shard_dir
            / f"turnstore_{stage}_{render}_{corpus}"
            / f"turnstore_{stage}_{render}_{corpus}_shard000.pt"
        )
        if not context_shard.exists():
            print(f"[skip] Missing context shard: {context_shard}")
            continue
        output_path = args.output_dir / f"{stage}_{render}_{corpus}_{arm}_L{LAYER}.jsonl"
        if output_path.exists():
            print(f"[skip] Exists: {output_path}")
            continue
        fit_cell(
            context_shard=context_shard,
            encoded_shard=enc_path,
            arm=arm,
            output_path=output_path,
            device=args.device,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

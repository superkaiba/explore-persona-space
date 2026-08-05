"""P2 per-feature ridge fit for task #2061.

Reads:
- P1 output: `data/issue_2061/sae_encoded/<stage>_<render>_<corpus>_answer_L29.pt`
  — SAE-encoded ANSWER targets (n_rows, d_sae=262144), from
  `scripts/issue2061_sae_encode.py`.
- #1336's banked arm inputs (prefix + context slot states), loaded from ALL
  `*_shardNNN.pt` files of each locally-staged turnstore in shard-index order
  (same enumeration as P1, so X/Y rows align). Payload schema + realized
  pooling convention: `scripts/issue2061_turnstore.py`.

Fits, per (stage, render, corpus, arm) cell:
- K=5 GROUP-level folds (conversation-id groups), fold seed 0 — #1336's exact
  fold convention via `issue2061_turnstore.group_fold_ids` (mirrors
  `issue825_fit_cells._cv_folds`, the constructor #1336's fits drive; plan
  §10, review M5, `.claude/rules/ood-generalization-folds.md`).
- Shared-factorization ridge via `ridge_fit_predict_fast_layer_batched`
  (`src/explore_persona_space/experiments/issue_779/fit_h.py:180`); GCV over
  the #823/#779 grid `np.logspace(-2, 4, 13)` WITH the #1887 dof cap
  (`gcv_dof_cap=0.9`, plan §11 "Under-determined-cell mitigation" — engaged
  via the helper's opt-in kwarg; inert whenever `0.9 * n_train >= d_in`, i.e.
  everywhere except `gsm8k_test1319`-class under-determined cells). Per-fit
  selected-lambda diagnostics (selector, lambda, effective dof) are recorded
  per #1887.
- Slow-vs-fast numeric-parity gate (the helper's docstring mandate + plan
  §Design): once per process, >=3 fold slices of the first fitted cell at
  production (n_train, d_in) shape vs the canonical `ridge_fit_predict` SVD
  reference on a seeded column subsample; max rel diff <= 1e-4 (#1332 bar).
- Per-feature R²_j = 1 − ||f_j − ĝ_j||² / ||f_j − mean(f_j)||² on held-out
  folds, pooled with fold-local test means.
- kNN retrieval (euclidean + cosine) via `analysis/mapping_baselines.knn_retrieval`
  with `k = ceil(n_pool / 20)` (plan §13; chance = k / n_pool).

Emits `eval_results/issue_2061/per_feature_r2/<stage>_<render>_<corpus>_<arm>_L29.jsonl`
— one JSON object per feature with keys:
  {feature_id, R2, n_train_folds, n_test_total, best_lambda_folds,
   effective_dof_folds, lambda_selector, knn_acc_1_euclid, knn_acc_k_euclid,
   knn_k_ret, knn_acc_1_cosine, knn_acc_k_cosine, chance_1, chance_k}

Usage:
  uv run python scripts/issue2061_fit_per_feature.py \\
      --stage base --render chat --corpus lmsys23k --arm context
  uv run python scripts/issue2061_fit_per_feature.py --all-cells
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from explore_persona_space.experiments.issue_779.fit_h import (  # noqa: E402
    ridge_fit_predict,
    ridge_fit_predict_fast_layer_batched,
)

# Sibling-script import (bare module name via the script-dir sys.path insert —
# the issue1336_extract_turnstore.py pattern; works in script mode AND under
# the tests' `sys.path.insert(scripts)` import).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import issue2061_turnstore as ts  # noqa: E402

LAYER = 29
D_IN = 4096
K_FOLDS = 5
FOLD_SEED = 0
LAMBDA_GRID = np.logspace(-2, 4, 13)
DOF_CAP_FRACTION = 0.9  # plan §7 under-determined-cell mitigation


def _load_arm_inputs(
    turnstore_dir: Path, arm: str, layer: int = LAYER
) -> tuple[np.ndarray, list[str]]:
    """((n, d_in) float32 arm inputs, conv_ids) from ALL shards of one turnstore dir.

    `arm` selects the banked #1336 slot state — "prefix" -> the prefix-header
    slot, "context" -> the a1-assistant-header slot (end of the context).
    Realized convention + fail-loud schema assert live in
    `issue2061_turnstore` (see its docstring; plan §12(4)). Shards are
    enumerated in shard-index order, matching the encode script's row order,
    so X rows align with the encoded Y rows by construction. The conv_ids
    feed the #1336 GROUP-level fold construction (plan §10, review M5).
    """
    shard_paths = ts.enumerate_shards(turnstore_dir)
    x, conv_ids = ts.load_state_from_shards(shard_paths, state=arm, layer=layer)
    return x.numpy(), conv_ids


def _make_folds(conv_ids: list[str], k: int = K_FOLDS, seed: int = FOLD_SEED) -> list[np.ndarray]:
    """Per-fold TEST index arrays from #1336's GROUP-level fold convention.

    Delegates to `issue2061_turnstore.group_fold_ids` (mirrors
    `issue825_fit_cells._cv_folds`: seeded permutation of UNIQUE conversation
    ids, `perm % k` per id — all rows of a conversation share a fold). Fold
    sizes vary with group membership; every fold is non-empty (fail-loud in
    the helper).
    """
    fold_of_row = ts.group_fold_ids(conv_ids, n_folds=k, seed=seed)
    return [np.where(fold_of_row == f)[0].astype(np.int64) for f in range(k)]


_PARITY_GATE_STATE = {"done": False}


def run_parity_gate(
    X: np.ndarray,
    Y: np.ndarray,
    folds: list[np.ndarray],
    *,
    n_slices: int = 3,
    n_cols: int = 512,
    col_seed: int = 0,
    tol: float = 1e-4,
    device: str = "cpu",
) -> float:
    """Slow-vs-fast numeric-parity gate (the batched helper's docstring mandate).

    Compares `ridge_fit_predict_fast_layer_batched` against the canonical SVD
    reference `ridge_fit_predict` on `n_slices` fold slices at the CELL's
    production (n_train, d_in) shape, over a seeded column subsample of the
    target (per-column regressions are independent, so column subsetting does
    not change the per-slice fit machinery under test — the (n_train, d_in)
    shape is what drives the size-dependent parity caveat). Runs with
    `gcv_dof_cap=None` on BOTH sides (the slow reference has no cap; the cap
    is a lambda-grid mask pinned separately by tests/test_issue2061_stats.py).
    Raises RuntimeError above `tol` (#1332 bar: max rel diff <= 1e-4).
    Returns the realized max rel diff.
    """
    n_slices = min(n_slices, len(folds))
    rng = np.random.default_rng(col_seed)
    cols = np.sort(rng.choice(Y.shape[1], size=min(n_cols, Y.shape[1]), replace=False))
    worst = 0.0
    for fi in range(n_slices):
        test_idx = folds[fi]
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fi])
        Xtr = X[train_idx].astype(np.float64)
        Ytr = Y[train_idx][:, cols].astype(np.float64)
        Xev = X[test_idx].astype(np.float64)
        fast = ridge_fit_predict_fast_layer_batched(
            Xtr[None], Ytr[None], Xev[None], lambdas=LAMBDA_GRID, device=device
        )[0]
        slow = ridge_fit_predict(Xtr, Ytr, Xev, lambdas=LAMBDA_GRID)
        rel = float(np.max(np.abs(fast - slow)) / (np.max(np.abs(slow)) + 1e-12))
        worst = max(worst, rel)
        print(
            f"  [parity-gate] slice {fi}: n_train={Xtr.shape[0]} d_in={Xtr.shape[1]} "
            f"n_cols={len(cols)} max_rel_diff={rel:.3g}",
            flush=True,
        )
    if worst > tol:
        raise RuntimeError(
            f"slow-vs-fast ridge parity gate FAILED: max rel diff {worst:.3g} > tol {tol:.1g} "
            f"over {n_slices} slices (fit_h.ridge_fit_predict_fast_layer_batched docstring "
            "mandate) — fall back to the canonical solver."
        )
    print(f"  [parity-gate] PASS: worst max_rel_diff={worst:.3g} <= tol={tol:.1g}", flush=True)
    return worst


def fit_cell(
    turnstore_dir: Path,
    encoded_shard: Path,
    arm: str,
    output_path: Path,
    layer: int = LAYER,
    device: str = "cpu",
    skip_parity_gate: bool = False,
) -> None:
    """Fit ridge for one (stage, render, corpus, arm) cell + write JSONL."""
    print(f"[fit] turnstore={turnstore_dir.name} arm={arm} encoded={encoded_shard.name}")
    X, conv_ids = _load_arm_inputs(turnstore_dir, arm, layer=layer)  # (n, d_in)
    Y = (
        torch.load(encoded_shard, map_location="cpu", weights_only=True).float().numpy()
    )  # (n, d_sae)
    assert X.shape[0] == Y.shape[0], f"row mismatch: X={X.shape}, Y={Y.shape}"
    n, d_in = X.shape
    d_sae = Y.shape[1]
    print(f"  n={n} d_in={d_in} d_sae={d_sae}")

    folds = _make_folds(conv_ids, k=K_FOLDS, seed=FOLD_SEED)

    # Once-per-process slow-vs-fast parity gate on the FIRST fitted cell
    # (>=3 slices at this cell's production shape; helper docstring mandate).
    if not skip_parity_gate and not _PARITY_GATE_STATE["done"]:
        run_parity_gate(X, Y, folds, device=device)
        _PARITY_GATE_STATE["done"] = True
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
            gcv_dof_cap=DOF_CAP_FRACTION,  # #1887 mitigation, plan §11 (review M4)
        )
        elapsed = time.time() - t0
        best_lam = float(info["best_lambda"][0])
        dof = float(info["dof"][0])
        per_fold_lambda.append(best_lam)
        per_fold_dof.append(dof)

        # The helper masks lambdas whose dof exceeds the cap (and fail-louds
        # when ALL are masked), so the selected dof satisfies the cap by
        # construction — assert it stays that way (guard against upstream drift).
        assert dof <= DOF_CAP_FRACTION * n_train * (1.0 + 1e-9), (
            f"fold {fi}: selected dof={dof:.1f} violates gcv_dof_cap="
            f"{DOF_CAP_FRACTION} * n_train={n_train} — fit_h dof-cap drift?"
        )
        if n_train < D_IN:
            print(
                f"  [dof-cap] fold {fi}: n_train={n_train} < d_in={D_IN} — "
                f"cap {DOF_CAP_FRACTION} active (lambda={best_lam:.3g}, dof={dof:.1f})",
                flush=True,
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
    # k = ceil(n_pool / 20), chance = k / n_pool (plan §13). `knn_retrieval`
    # returns `acc_at_k` / `chance_at_k` as dicts KEYED BY K (review C2 —
    # positional indexing crashed KeyError: 0 after the fold fits).
    k_ret = max(1, math.ceil(n / 20))
    knn_e = knn_retrieval(Y_pred_pool, Y.astype(np.float64), ks=(1, k_ret), metric="euclidean")
    knn_c = knn_retrieval(Y_pred_pool, Y.astype(np.float64), ks=(1, k_ret), metric="cosine")
    lambda_selector = f"gcv-dof-cap-{DOF_CAP_FRACTION}"  # #1887 diagnostics (M4)

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
                        "lambda_selector": lambda_selector,
                        "knn_acc_1_euclid": float(knn_e["acc_at_k"][1]),
                        "knn_acc_k_euclid": float(knn_e["acc_at_k"][k_ret]),
                        "knn_k_ret": int(k_ret),
                        "knn_acc_1_cosine": float(knn_c["acc_at_k"][1]),
                        "knn_acc_k_cosine": float(knn_c["acc_at_k"][k_ret]),
                        "chance_1": float(knn_e["chance_at_k"][1]),
                        "chance_k": float(knn_e["chance_at_k"][k_ret]),
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
    parser.add_argument(
        "--skip-parity-gate",
        action="store_true",
        help="skip the once-per-process slow-vs-fast parity gate (debug only)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Y targets are the ANSWER-state encodes only (plan §Design target Y).
    encoded_files = sorted(args.encoded_dir.glob(f"*_answer_L{LAYER}.pt"))
    if not encoded_files:
        print(f"[error] No '*_answer_L{LAYER}.pt' encoded targets in {args.encoded_dir}")
        return 1

    enc_re = re.compile(rf"^(?P<cell>.+)_answer_L{LAYER}$")

    def parse_cell(path: Path) -> tuple[str, str, str]:
        # <stage>_<render>_<corpus>_answer_L<layer>.pt. stage/render never
        # contain '_', so a LEFT split of the cell part keeps underscore
        # corpora (gsm8k_test1319) intact.
        m = enc_re.match(path.stem)
        if m is None:
            raise ValueError(
                f"Unrecognized encoded-target filename {path.name}; expected "
                f"<stage>_<render>_<corpus>_answer_L{LAYER}.pt "
                "(scripts/issue2061_sae_encode.py::encode_turnstore)."
            )
        stage, render, corpus = m.group("cell").split("_", 2)
        return stage, render, corpus

    targets: list[tuple[Path, str, str, str, str]] = []
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
        turnstore_dir = args.context_shard_dir / f"turnstore_{stage}_{render}_{corpus}"
        if not turnstore_dir.is_dir():
            print(f"[skip] Missing turnstore dir: {turnstore_dir}")
            continue
        output_path = args.output_dir / f"{stage}_{render}_{corpus}_{arm}_L{LAYER}.jsonl"
        if output_path.exists():
            print(f"[skip] Exists: {output_path}")
            continue
        fit_cell(
            turnstore_dir=turnstore_dir,
            encoded_shard=enc_path,
            arm=arm,
            output_path=output_path,
            device=args.device,
            skip_parity_gate=args.skip_parity_gate,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

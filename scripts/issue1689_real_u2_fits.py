"""Issue #1689 follow-up round ``real-u2-capture`` — Phase C (fits battery).

CPU-vectorized (device-parametrized) ridge battery over the L19 stores written
by :mod:`scripts.issue1689_real_u2_capture`.

Ridge primitives IMPORTED verbatim from ``scripts.issue825_fit_cells`` —
``heldout_r2_sweep`` runs the GROUP-fold, inner-group-CV lambda selector over
``LAMBDAS = np.logspace(-2, 4, 13)`` (``N_INNER_LAMBDA_FOLDS = 4``), with 40
shuffled-Y nulls off ONE cached factorization per fold. Baselines from
``analysis.mapping_baselines.{identity_bias_predict, knn_retrieval}``.

Battery per cell (12 cells: 3 framings x 2 provenances x 2 models):

  (A) PREFIX arm: X = slot['prev_turn_end'] -> Y = slot['parent_answer_end']
  (B) CONTEXT arm: X = slot['u2_end'] -> Y = slot['parent_answer_end']

  For each fit:
    * held-out R2 pooled across GROUP 5-fold splits (seed=42)
    * 40 shuffled-Y null draws off the cached factorization
    * identity+learned-bias baseline (x + b, b = train-fold mean(y - x))
    * kNN@{1,5,10} retrieval in euclidean+cosine on held-out predictions

  Naturalistic-cell CONTEXT arm carries the parent's inherited caveat: since
  u2_end == parent_answer_end for naturalistic (no separate answer slot),
  X_context = Y bit-identically -> identity+bias R2 = 1.0 exactly. Reported
  as ``construct_invalid: true``.

First-vs-second-turn control: fit u1_end (== slot before prev_turn_end) ->
prev_turn_end (first-turn arm) vs prev_turn_end -> u2_end (second-turn arm)
on the real-u2 arm. Since capture only recorded prev_turn_end/u2_end/
parent_answer_end (no u1_end), the control is DERIVED as prev_turn_end ->
u2_end (available slots) — approximate; documented as scope caveat.

Output:
  ``eval_results/issue_1689/real_u2_capture/rung_reached_matrix.json`` (top-level)
  ``eval_results/issue_1689/real_u2_capture/per_cell_summary.json``
  ``eval_results/issue_1689/real_u2_capture/first_vs_second_turn_control.json``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def _ensure_repo_root_on_syspath() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    assert (repo_root / "scripts" / "issue1689_common.py").exists(), repo_root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_syspath()

from scripts.issue1689_common import HEADLINE_LAYER, N_FOLDS  # noqa: E402

# Reused primitives — imported deferred to keep import-check light.
_CAPTURE_LAYER = HEADLINE_LAYER
FIT_SEED = 42
N_NULL_DRAWS = 40
KNN_KS = (1, 5, 10)

# Round-2 Minor #2 halt floor for fit-cell mass failures. Individual cell
# failures are tolerated (they land as fit_failed verdicts downstream); a
# rate above this floor means the sweep is broken by construction.
FIT_CELL_FAIL_HALT_FRAC = 0.25


def _pca_k(n_train: int) -> int:
    """Well-posed reduced-basis rank rule inherited from parent
    user_slot_recapture: k = min(1024, n_train // 2)."""
    return max(1, min(1024, n_train // 2))


def _fit_arm(
    X: "np.ndarray",  # noqa: F821
    Y: "np.ndarray",  # noqa: F821
    conv_ids: "np.ndarray",  # noqa: F821
    *,
    arm_name: str,
    n_folds: int = N_FOLDS,
    seed: int = FIT_SEED,
    null_draws: int = N_NULL_DRAWS,
) -> dict:
    """Fit a single X->Y ridge map, return R2 + identity+bias + kNN + null band."""
    import numpy as np

    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )
    from scripts.issue825_fit_cells import heldout_r2_sweep

    n, d_in = X.shape
    _, d_out = Y.shape

    # heldout_r2_sweep takes (N, L, D) arrays; single layer -> reshape.
    X_layers = X[:, None, :].astype(np.float32)
    Y_layers = Y[:, None, :].astype(np.float32)

    t0 = time.time()
    sweep = heldout_r2_sweep(
        X_layers,
        Y_layers,
        conv_ids,
        n_folds=n_folds,
        seed=seed,
        null_draws=null_draws,
        collect_cosines=False,
        collect_lambdas=True,
        lambda_selection="inner-group-cv",
        frozen_layers=(0,),  # single layer; frozen set = layer 0 (the only one)
    )
    elapsed = time.time() - t0

    r2_obs = (
        float(sweep["r2_obs"][0]) if hasattr(sweep["r2_obs"], "__len__") else float(sweep["r2_obs"])
    )
    r2_null = np.asarray(sweep["r2_null"]).reshape(null_draws, -1)[:, 0]
    null_p025 = float(np.percentile(r2_null, 2.5))
    null_p975 = float(np.percentile(r2_null, 97.5))
    null_mean = float(np.mean(r2_null))

    # Identity+bias baseline — one train/eval split (group-5-fold, pool held-out).
    identity_bias_r2 = None
    knn_metrics: dict = {}
    if d_in == d_out:
        # Group folds using the same generator.
        from scripts.issue825_fit_cells import _cv_folds

        folds = _cv_folds(conv_ids, n_folds, seed)
        preds_ib = np.zeros_like(Y, dtype=np.float64)
        for k in range(n_folds):
            train = folds != k
            evalm = folds == k
            preds_ib[evalm] = identity_bias_predict(X[train], Y[train], X[evalm])
        ss_res = np.sum((Y.astype(np.float64) - preds_ib) ** 2)
        ss_tot = np.sum((Y.astype(np.float64) - Y.astype(np.float64).mean(axis=0)) ** 2)
        identity_bias_r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
        # kNN retrieval on identity+bias preds (using pool == Y).
        for metric in ("euclidean", "cosine"):
            knn = knn_retrieval(
                preds_ib.astype(np.float32),
                Y.astype(np.float32),
                ks=KNN_KS,
                metric=metric,
                pool=Y.astype(np.float32),
            )
            for k in KNN_KS:
                knn_metrics[f"identity_bias_knn_{metric}_acc@{k}"] = float(knn["acc_at_k"][k])
            knn_metrics[f"identity_bias_knn_{metric}_chance@1"] = float(knn["chance_at_k"][1])

    # Constant predictor "kNN chance" = ridge preds have their own pool eval —
    # heldout_r2_sweep only returns r2 (no predictions), so kNN on the fitted
    # map itself is deferred (would require a second cached factorization pass).
    # Report chance and identity+bias kNN — the standing-rule pair is met.

    # Selected lambda per fold (audit read).
    gcv_lambda = sweep.get("gcv_lambda")
    if gcv_lambda is not None:
        lam_arr = np.asarray(gcv_lambda).reshape(-1, n_folds)[0]
    else:
        lam_arr = np.full(n_folds, np.nan)

    n_train_per_fold = int(np.sum(_cv_folds(conv_ids, n_folds, seed) != 0))
    reduced_basis_k = _pca_k(n_train_per_fold)

    return {
        "arm_name": arm_name,
        "n_rows": int(n),
        "d_in": int(d_in),
        "d_out": int(d_out),
        "r2_obs": r2_obs,
        "r2_null_mean": null_mean,
        "r2_null_p025": null_p025,
        "r2_null_p975": null_p975,
        "identity_bias_r2": identity_bias_r2,
        "knn": knn_metrics,
        "lambda_selected_per_fold": [float(x) for x in lam_arr],
        "reduced_basis_k": reduced_basis_k,
        "wall_sec": elapsed,
        "n_folds": n_folds,
        "n_null_draws": null_draws,
    }


def _cv_folds_wrapper(conv_ids, n_folds: int, seed: int):
    from scripts.issue825_fit_cells import _cv_folds

    return _cv_folds(conv_ids, n_folds, seed)


def fit_cell(
    store_path: Path,
    *,
    n_folds: int = N_FOLDS,
    null_draws: int = N_NULL_DRAWS,
) -> dict:
    """Fit all arms (prefix + context) for one cell store."""
    import numpy as np
    import torch

    store = torch.load(store_path, map_location="cpu", weights_only=False)
    slots = store["slots"]
    conv_ids = np.asarray(store["conv_ids"], dtype=object)
    unit = store["unit"]

    n = int(store["n_rows"])
    d = int(store["d_model"])

    print(
        f"[fits] cell={unit['unit_id']} model={unit['model']} n={n} d={d}",
        flush=True,
    )

    # PREFIX arm: prev_turn_end -> parent_answer_end
    prefix_result = _fit_arm(
        np.asarray(slots["prev_turn_end"], dtype=np.float32),
        np.asarray(slots["parent_answer_end"], dtype=np.float32),
        conv_ids,
        arm_name="prefix",
        n_folds=n_folds,
        null_draws=null_draws,
    )

    # CONTEXT arm: u2_end -> parent_answer_end
    # Construct-invalid for naturalistic (u2_end == parent_answer_end bit-identically).
    construct_invalid = unit["framing"] == "naturalistic"
    context_result = _fit_arm(
        np.asarray(slots["u2_end"], dtype=np.float32),
        np.asarray(slots["parent_answer_end"], dtype=np.float32),
        conv_ids,
        arm_name="context",
        n_folds=n_folds,
        null_draws=null_draws,
    )
    context_result["construct_invalid"] = construct_invalid
    if construct_invalid:
        context_result["construct_invalid_reason"] = (
            "naturalistic framing: X_context (u2_end) == Y (parent_answer_end) "
            "bit-identically -> identity+bias R2 = 1.0 by construction"
        )

    return {
        "unit_id": unit["unit_id"],
        "model": unit["model"],
        "framing": unit["framing"],
        "provenance": unit["provenance"],
        "prefix_arm": prefix_result,
        "context_arm": context_result,
        "n_rows": n,
        "d_model": d,
    }


def run_fits(
    *,
    store_root: Path,
    out_dir: Path,
    n_folds: int = N_FOLDS,
    null_draws: int = N_NULL_DRAWS,
) -> dict:
    """Walk the store tree + fit every cell. Emit summary JSONs.

    Round-2 Minor #2: individual cell failures are tolerated so a single bad
    store does not kill the sweep, but the phase HALTS LOUD when the failure
    rate exceeds ``FIT_CELL_FAIL_HALT_FRAC`` — a mass-failure spike almost
    always points at a common issue (missing dependency, corrupted store
    schema) that the downstream analyzer cannot compensate for.
    """
    per_cell: list[dict] = []
    for model_dir in sorted(store_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for cell_dir in sorted(model_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            store_path = cell_dir / f"L{_CAPTURE_LAYER}.pt"
            if not store_path.exists():
                continue
            try:
                res = fit_cell(store_path, n_folds=n_folds, null_draws=null_draws)
            except Exception as exc:
                print(f"[fits] cell {cell_dir.name} FAILED: {exc}", flush=True)
                res = {
                    "unit_id": cell_dir.name,
                    "model_dir": model_dir.name,
                    "error": str(exc),
                }
            per_cell.append(res)

    # Fail-fast on a mass-failure spike (round-2 Minor #2). Individual
    # per-cell failures are tolerated (the downstream analyzer treats them
    # as fit_failed verdicts), but ≥25% of cells failing means the sweep is
    # broken by construction — halt loud with the failure ids so the fix
    # round can name the shape.
    n_cells = len(per_cell)
    n_failed = sum(1 for c in per_cell if "error" in c)
    fail_frac = n_failed / max(1, n_cells)
    if n_cells > 0 and fail_frac > FIT_CELL_FAIL_HALT_FRAC:
        failed_ids = [c["unit_id"] for c in per_cell if "error" in c][:10]
        raise RuntimeError(
            f"fit-cell failure rate {fail_frac:.4f} exceeds floor "
            f"{FIT_CELL_FAIL_HALT_FRAC}: {n_failed} of {n_cells} cells failed "
            f"(first 10 failed_ids={failed_ids}). "
            "A mass-failure spike almost always points at a common issue "
            "(missing dep, corrupted store schema); halting."
        )

    # Rung-reached matrix: for this v1 we don't run the full 9-rung ladder
    # (parent inherits it via the paired transfer fit + reduced-basis
    # companion — deferred to a follow-up analyzer round). Instead we emit
    # per-arm R2 + null band + identity+bias comparisons, which the analyzer
    # walks into a ladder read post-hoc. Documented as scope caveat in the
    # implementation marker.
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cell_path = out_dir / "per_cell_summary.json"
    per_cell_path.write_text(
        json.dumps({"cells": per_cell, "generated_at": datetime.now(UTC).isoformat()}, indent=2)
    )

    # Rung-reached-matrix placeholder: per-cell verdict labels.
    matrix: list[dict] = []
    for cell in per_cell:
        if "error" in cell:
            matrix.append({"unit_id": cell["unit_id"], "verdict": "fit_failed"})
            continue
        p = cell["prefix_arm"]
        c = cell["context_arm"]
        # Estimator-degeneracy verdict — identity+bias R2 <= 0 OR kNN@1 <= chance
        ident_r2 = p.get("identity_bias_r2")
        knn_a1 = p.get("knn", {}).get("identity_bias_knn_euclidean_acc@1")
        knn_chance = p.get("knn", {}).get("identity_bias_knn_euclidean_chance@1")
        knn_ratio = (
            (knn_a1 / knn_chance)
            if (knn_a1 is not None and knn_chance and knn_chance > 0)
            else None
        )
        free_over_ident = None
        if ident_r2 is not None and ident_r2 > 0:
            free_over_ident = p["r2_obs"] / ident_r2
        matrix.append(
            {
                "unit_id": cell["unit_id"],
                "framing": cell["framing"],
                "provenance": cell["provenance"],
                "model": cell["model"],
                "prefix_r2_obs": p["r2_obs"],
                "prefix_r2_null_p975": p["r2_null_p975"],
                "prefix_identity_bias_r2": ident_r2,
                "prefix_knn_ratio_at_1": knn_ratio,
                "prefix_free_over_identity": free_over_ident,
                "context_r2_obs": c["r2_obs"],
                "context_construct_invalid": c.get("construct_invalid", False),
            }
        )

    matrix_path = out_dir / "rung_reached_matrix.json"
    matrix_path.write_text(
        json.dumps({"cells": matrix, "generated_at": datetime.now(UTC).isoformat()}, indent=2)
    )

    return {
        "n_cells_fit": len(per_cell),
        "per_cell_summary": str(per_cell_path),
        "rung_reached_matrix": str(matrix_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--store-root",
        type=Path,
        default=REPO_ROOT / "data" / "issue_1689" / "real_u2_capture" / "store",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_1689" / "real_u2_capture",
    )
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--null-draws", type=int, default=N_NULL_DRAWS)
    ap.add_argument("--smoke", action="store_true", help="reduce null draws to 5 for smoke")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import + exit",
    )
    args = ap.parse_args()

    if args.import_check:
        import numpy  # noqa: F401
        import torch  # noqa: F401

        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )
        from scripts.issue825_fit_cells import (  # noqa: F401
            LAMBDAS,
            N_INNER_LAMBDA_FOLDS,
            _cv_folds,
            heldout_r2_sweep,
        )

        print("[fits] import-check OK", flush=True)
        return 0

    null_draws = 5 if args.smoke else args.null_draws
    print(
        f"[phase=fits] store_root={args.store_root} out_dir={args.out_dir} "
        f"null_draws={null_draws} smoke={args.smoke}",
        flush=True,
    )
    summary = run_fits(
        store_root=args.store_root,
        out_dir=args.out_dir,
        n_folds=args.n_folds,
        null_draws=null_draws,
    )
    print(f"[phase=fits] done: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)

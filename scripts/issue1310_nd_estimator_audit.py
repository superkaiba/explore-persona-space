"""Issue #1310 / #1639 n<d estimator audit (user-directed inline round, 2026-07-30).

Measures how much the published held-out R^2 reads move when the ridge lambda
selector changes, in the n_train < d = 3584 regime every #1310 fiction cell sits
in. Motivating proof: #1345 showed pure-GCV at n_train < d selects a
near-interpolating lambda (grid floor) and returns artifact-negative / deflated
held-out R^2; #1887 hardens the shared fit-core defaults.

Audited substrate is the ONLY recoverable one: the onpolicy PREFILL store
(HF issue1310_char_map/analysis_tensors/store_onpolicy). The run-2 SCRIPT-format
store that backs the published 0.106-0.148 base / 0.188-0.253 instruct
per-persona headline cells was lost with its instance, so those cells cannot be
recomputed at 0 GPU-h; this round measures the selector artifact at comparable n
on the prefill substrate and reports the script-cell verdict as a transfer
estimate.

Per cell at layer 19, four selector families through the IDENTICAL fold
assignment (fit825 `_cv_folds`, seed 0, 5 folds) and the same Gram-space ridge:

  ref  published capped GCV  (GCV_DOF_CAP=0.9)      -> reproduction gate
  a    ambient pure-GCV      (GCV_DOF_CAP=None)     -> the run-2 script selector
  b    inner-group-CV        (4 inner GROUP folds)  -> candidate corrected read
  c    train-fold reduced PCA basis k=min(1024, n_train//2)
  d    forced lambda {1e2, 1e3, 1e4}                -> DIAGNOSTIC ONLY

Also per cell: the selected lambda per fold under each selecting arm with
grid-edge proximity (the degeneracy signature), and the mandated mapping
baselines (identity+learned-bias, kNN retrieval) in the ambient space.

Output: eval_results/issue_1310/nd_estimator_audit/{cells_*.json,
corrections_table.json}.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps before heavy imports (#847)

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue825_fit_cells as fit825  # noqa: E402
import issue1310_common as c1310  # noqa: E402
from explore_persona_space.analysis import mapping_baselines as mb  # noqa: E402
from issue1310_agg_perfold import stream_l19  # noqa: E402
from issue1310_aggfit import aggregate_store  # noqa: E402

L = 19
SEED = c1310.FIT_SEED
FOLDS = c1310.N_FOLDS
D_AMBIENT = 3584
FORCED_LAMBDAS = (1e2, 1e3, 1e4)
PCA_K_CAP = 1024
GRID = fit825.LAMBDAS  # np.logspace(-2, 4, 13)

EV = REPO / "eval_results" / "issue_1310"
OUT = EV / "nd_estimator_audit"


def _fit_pooled(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    *,
    dof_cap: float | None,
    selection: str = "gcv",
    lambdas: np.ndarray | None = None,
    pca_k: int | None = None,
) -> dict:
    """Pooled held-out R^2 at one layer under one selector configuration.

    SS_tot is taken around each FOLD's test mean, matching the committed
    per-cell sweep convention (`issue1310_analyzer_perfold.perfold_fit`), so the
    ref arm reproduces `r2_per_layer_obs[19]`. Returns the pooled R^2, the
    per-fold R^2, the selected lambda per fold, and the out-of-fold predictions.
    """
    fit825.GCV_DOF_CAP = dof_cap
    fold_ids = fit825._cv_folds(groups, FOLDS, SEED)
    ss_res = ss_tot = 0.0
    per_fold, lams_sel, n_train_seen = [], [], []
    preds = np.zeros(Y.shape, dtype=np.float64)
    for k in range(FOLDS):
        te = fold_ids == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 3:
            continue
        Xtr, Xte = X[tr], X[te]
        if pca_k is not None:
            # Train-fold PCA basis: centre on train, keep the top-k right
            # singular vectors, project both sides. n_train > k by construction
            # (k = n_train // 2 capped), so the projected fit is well-posed.
            mu_x = Xtr.mean(0)
            _, _, Vt = np.linalg.svd(Xtr - mu_x, full_matrices=False)
            basis = Vt[:pca_k].T
            Xtr = (Xtr - mu_x) @ basis
            Xte = (Xte - mu_x) @ basis
            del Vt, basis
        n_train_seen.append(int(Xtr.shape[0]))
        cache = fit825._prep_fold(Xtr, Xte)
        if selection == "inner-group-cv":
            cache["inner"] = fit825._prep_inner_lambda(
                Xtr, groups[tr], fit825.N_INNER_LAMBDA_FOLDS, SEED + 4242 + k
            )
        pred, lam = fit825._ridge_predict_cached(cache, Y[tr], return_lam=True, lambdas=lambdas)
        lams_sel.append(float(lam))
        true = Y[te].astype(np.float64)
        mu_y = true.mean(0)
        sr = float(np.sum((true - pred) ** 2))
        st = float(np.sum((true - mu_y) ** 2))
        ss_res += sr
        ss_tot += st
        per_fold.append({"fold": k, "r2": 1.0 - sr / st, "n_te": int(te.sum())})
        preds[te] = pred
        del cache, pred, Xtr, Xte
    fit825.GCV_DOF_CAP = None
    return {
        "r2_pooled": 1.0 - ss_res / ss_tot,
        "per_fold_r2": per_fold,
        "lambda_per_fold": lams_sel,
        "n_train_per_fold": n_train_seen,
        "preds": preds,
    }


def _lambda_diag(lams: list[float]) -> dict:
    """Grid-edge proximity of the selected lambdas (the degeneracy signature)."""
    if not lams:
        return {}
    arr = np.asarray(lams, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "n_at_grid_floor": int(np.sum(arr <= GRID[0] * 1.0001)),
        "n_at_grid_ceiling": int(np.sum(arr >= GRID[-1] * 0.9999)),
        "n_folds": int(arr.size),
        "grid_floor": float(GRID[0]),
        "grid_ceiling": float(GRID[-1]),
        "all_interior": bool(np.all(arr > GRID[0] * 1.0001) and np.all(arr < GRID[-1] * 0.9999)),
    }


def _mapping_baselines(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, preds: dict) -> dict:
    """Identity+learned-bias baseline and kNN retrieval (CLAUDE.md standing rule).

    Both computed in the AMBIENT space (input and output share d = 3584, so the
    identity family applies). The retrieval pool is the full row set of true
    targets, so chance = k / n; predictions are the pooled out-of-fold ones.
    """
    fold_ids = fit825._cv_folds(groups, FOLDS, SEED)
    ss_res = ss_tot = 0.0
    for k in range(FOLDS):
        te = fold_ids == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 3:
            continue
        pred = mb.identity_bias_predict(X[tr], Y[tr], X[te])
        true = Y[te].astype(np.float64)
        mu_y = true.mean(0)
        ss_res += float(np.sum((true - pred) ** 2))
        ss_tot += float(np.sum((true - mu_y) ** 2))
        del pred
    out = {"identity_bias_r2_pooled": 1.0 - ss_res / ss_tot}
    true_all = Y.astype(np.float64)
    ks = (1, 5, 10)
    out["knn_identity_bias"] = None
    for arm, pr in preds.items():
        out[f"knn_{arm}"] = mb.knn_retrieval(pr, true_all, ks=ks, metric="euclidean")
    return out


def audit_cell(
    cell_id: str, X: np.ndarray, Y: np.ndarray, groups: np.ndarray, published: float | None
) -> dict:
    """Run every selector family on one cell and assemble its audit record."""
    t0 = time.time()
    n = int(X.shape[0])
    n_train_nominal = int(round(n * (FOLDS - 1) / FOLDS))
    pca_k = int(min(PCA_K_CAP, max(2, n_train_nominal // 2)))

    arms: dict[str, dict] = {}
    arms["ref_capped_gcv"] = _fit_pooled(X, Y, groups, dof_cap=0.9)
    arms["ambient_pure_gcv"] = _fit_pooled(X, Y, groups, dof_cap=None)
    arms["inner_group_cv"] = _fit_pooled(X, Y, groups, dof_cap=None, selection="inner-group-cv")
    arms["reduced_pca_basis"] = _fit_pooled(X, Y, groups, dof_cap=None, pca_k=pca_k)
    for lam in FORCED_LAMBDAS:
        arms[f"forced_lambda_{lam:.0e}"] = _fit_pooled(
            X, Y, groups, dof_cap=None, lambdas=np.array([lam], dtype=np.float64)
        )

    baselines = _mapping_baselines(
        X,
        Y,
        groups,
        {
            "ref_capped_gcv": arms["ref_capped_gcv"]["preds"],
            "ambient_pure_gcv": arms["ambient_pure_gcv"]["preds"],
        },
    )

    record = {
        "cell_id": cell_id,
        "layer": L,
        "seed": SEED,
        "folds": FOLDS,
        "n": n,
        "n_train_nominal": n_train_nominal,
        "d_ambient": D_AMBIENT,
        "n_train_lt_d": bool(n_train_nominal < D_AMBIENT),
        "pca_k": pca_k,
        "published_r2_l19": published,
        "arms": {},
        "lambda_diagnostics": {},
        "mapping_baselines": baselines,
        "wall_s": None,
    }
    for name, res in arms.items():
        record["arms"][name] = {
            "r2_pooled": res["r2_pooled"],
            "per_fold_r2": res["per_fold_r2"],
            "lambda_per_fold": res["lambda_per_fold"],
            "n_train_per_fold": res["n_train_per_fold"],
        }
        record["lambda_diagnostics"][name] = _lambda_diag(res["lambda_per_fold"])

    if published is not None:
        d = abs(arms["ref_capped_gcv"]["r2_pooled"] - published)
        record["reproduction_gate"] = {
            "recomputed_ref": arms["ref_capped_gcv"]["r2_pooled"],
            "committed": published,
            "abs_delta": d,
            "tolerance": 1e-2,
            "pass": bool(d < 1e-2),
        }
        assert d < 1e-2, (cell_id, arms["ref_capped_gcv"]["r2_pooled"], published)

    record["wall_s"] = round(time.time() - t0, 1)
    for res in arms.values():
        del res["preds"]
    del arms
    gc.collect()
    print(
        f"[nd-audit] {cell_id}: n={n} ref {record['arms']['ref_capped_gcv']['r2_pooled']:+.4f} "
        f"ambient {record['arms']['ambient_pure_gcv']['r2_pooled']:+.4f} "
        f"innerCV {record['arms']['inner_group_cv']['r2_pooled']:+.4f} "
        f"pca{pca_k} {record['arms']['reduced_pca_basis']['r2_pooled']:+.4f} "
        f"({record['wall_s']}s)",
        flush=True,
    )
    return record


def _published(path: Path) -> float | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())["r2_per_layer_obs"][L]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for model in ("base", "instruct"):
        store = stream_l19(model)
        print(
            f"[nd-audit] {model}: streamed {store['arrays']['x_spanmean'].shape[0]} rows "
            f"(L{L} slice only)",
            flush=True,
        )

        # Family 1 - per-turn prefill cells (the #1310 onpolicy per-persona cells)
        for persona in c1310.PERSONA_LABELS:
            m = store["char_ids"] == persona
            cid = f"onpolicy_{model}_{persona}"
            out_path = OUT / f"cells_{cid}.json"
            if out_path.exists():
                records.append(json.loads(out_path.read_text()))
                print(f"[nd-audit] {cid}: resume-skip (already done)", flush=True)
                continue
            rec = audit_cell(
                cid,
                store["arrays"]["x_spanmean"][m],
                store["arrays"]["y"][m],
                store["group_ids"][m],
                _published(EV / "onpolicy" / f"cells_{cid}.json"),
            )
            rec["family"] = "per_turn_prefill"
            rec["model"] = model
            rec["persona"] = persona
            out_path.write_text(json.dumps(rec, indent=1))
            records.append(rec)

        # Family 2 - scene-aggregated cells (the #1639 within-cell substrate)
        agg = aggregate_store(store)
        del store
        gc.collect()
        print(f"[nd-audit] {model}: {len(agg['personas'])} aggregated points", flush=True)
        for persona in c1310.PERSONA_LABELS:
            m = agg["personas"] == persona
            cid = f"agg_{model}_{persona}"
            out_path = OUT / f"cells_{cid}.json"
            if out_path.exists():
                records.append(json.loads(out_path.read_text()))
                print(f"[nd-audit] {cid}: resume-skip (already done)", flush=True)
                continue
            rec = audit_cell(
                cid,
                agg["X"][m],
                agg["Y"][m],
                agg["scenarios"][m],
                _published(EV / "onpolicy_aggregated" / f"cells_{cid}.json"),
            )
            rec["family"] = "scene_aggregated"
            rec["model"] = model
            rec["persona"] = persona
            out_path.write_text(json.dumps(rec, indent=1))
            records.append(rec)
        del agg
        gc.collect()

    table = {
        "layer": L,
        "seed": SEED,
        "folds": FOLDS,
        "d_ambient": D_AMBIENT,
        "lambda_grid": [float(x) for x in GRID],
        "forced_lambdas": list(FORCED_LAMBDAS),
        "substrate": "issue1310_char_map/analysis_tensors/store_onpolicy",
        "store_gap": (
            "The run-2 SCRIPT-format activation store backing the published "
            "uncapped per-persona cells (eval_results/issue_1310/cells_*.json, "
            "no gcv_dof_cap field) was lost with its instance; those cells "
            "cannot be recomputed at 0 GPU-h."
        ),
        "cells": records,
    }
    (OUT / "corrections_table.json").write_text(json.dumps(table, indent=1))
    print(f"[nd-audit] wrote {OUT / 'corrections_table.json'} ({len(records)} cells)")
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

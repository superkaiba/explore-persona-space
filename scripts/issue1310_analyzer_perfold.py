"""Issue #1310 analyzer per-fold / per-scene-group L19 recompute (free analysis).

Loads the onpolicy prefill activation store (downloaded from HF
issue1310_char_map/analysis_tensors/store_onpolicy), slices LAYER 19 only, and
recomputes the committed fits' held-out predictions at that single layer with
the IDENTICAL fold assignment + GCV Gram ridge + dof cap (fit825, seed 0,
GCV_DOF_CAP=0.9) to expose the LOW-LEVEL per-unit views the committed cell
JSONs do not persist:

  - per-FOLD held-out R^2 (5 points per cell),
  - per-SCENE-GROUP held-out R^2 (~300 points per cell; SS_tot around the
    fold-test mean, so groups decompose the pooled statistic),
  - validation: recomputed pooled R^2 vs the committed r2_per_layer_obs[19]
    (assert |delta| < 0.01 — CPU-vs-GPU roundoff tolerance, #833).

Covers the 8 within-persona cells + the 4 pooled swap-control fits (correct +
swapped share fold caches — the X rows are identical). Output:
eval_results/issue_1310/onpolicy/analyzer_perfold_l19.json (committed to the
issue branch) — consumed by issue1310_analyzer_figures.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue825_fit_cells as fit825  # noqa: E402
import issue1310_common as c1310  # noqa: E402
from issue1310_fit import swap_derangement  # noqa: E402

STORE = REPO / "data/issue_1310/hf_dl/issue1310_char_map/analysis_tensors/store_onpolicy"
EV = REPO / "eval_results" / "issue_1310" / "onpolicy"
L = 19
SEED = 0
FOLDS = 5

fit825.GCV_DOF_CAP = 0.9  # the committed re-fit's cap


def load_l19(model_kind: str) -> dict:
    """Stream shards, keep ONLY layer-19 slices (RAM ~100 MB, not 8 GB)."""
    rows, groups, chars, turns = [], [], [], []
    xs, ys = [], []
    for sp in sorted((STORE / model_kind).glob(f"{model_kind}_shard*.pt")):
        payload = torch.load(sp, map_location="cpu", weights_only=False)
        rows.extend(payload["row_ids"])
        groups.extend(payload["group_ids"])
        chars.extend(payload["char_ids"])
        turns.extend(payload["turn_indices"])
        xs.append(payload["arrays"]["x_spanmean"][:, L, :].float().numpy())
        ys.append(payload["arrays"]["y"][:, L, :].float().numpy())
        del payload
    return {
        "row_ids": np.asarray(rows),
        "group_ids": np.asarray(groups),
        "char_ids": np.asarray(chars),
        "turn_indices": np.asarray(turns, dtype=int),
        "X": np.concatenate(xs, axis=0),
        "Y": np.concatenate(ys, axis=0),
    }


def perfold_fit(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, y_alt: np.ndarray | None = None):
    """Single-layer 5-fold held-out fit; per-fold + per-group R^2 (+ alt-Y twin).

    y_alt (the swapped pairing) reuses each fold's Y-independent cache — the X
    rows are identical, only the Y changes (issue1310_fit.run_swap's shape).
    """
    folds = fit825._cv_folds(groups, FOLDS, SEED)
    out = {"perfold": [], "pergroup": {}, "pooled": None}
    out_alt = {"perfold": [], "pergroup": {}, "pooled": None} if y_alt is not None else None
    ss_res = ss_tot = 0.0
    ss_res_a = ss_tot_a = 0.0
    for k in range(FOLDS):
        te = folds == k
        tr = ~te
        cache = fit825._prep_fold(X[tr], X[te])
        for tgt, acc in ((Y, out), (y_alt, out_alt)) if y_alt is not None else ((Y, out),):
            pred = fit825._ridge_predict_cached(cache, tgt[tr])
            true = tgt[te].astype(np.float64)
            mu = true.mean(0)
            sr = float(np.sum((true - pred) ** 2))
            st = float(np.sum((true - mu) ** 2))
            acc["perfold"].append({"fold": k, "r2": 1.0 - sr / st, "n_te": int(te.sum())})
            if acc is out:
                ss_res += sr
                ss_tot += st
            else:
                ss_res_a += sr
                ss_tot_a += st
            # per-group R^2: group rows within this fold's test set; SS_tot
            # around the FOLD-test mean so groups decompose the fold statistic
            te_groups = groups[te]
            for g in np.unique(te_groups):
                m = te_groups == g
                sr_g = float(np.sum((true[m] - pred[m]) ** 2))
                st_g = float(np.sum((true[m] - mu) ** 2))
                if st_g > 1e-12:
                    acc["pergroup"][str(g)] = {
                        "r2": 1.0 - sr_g / st_g,
                        "n_rows": int(m.sum()),
                        "fold": k,
                    }
    out["pooled"] = 1.0 - ss_res / ss_tot
    if out_alt is not None:
        out_alt["pooled"] = 1.0 - ss_res_a / ss_tot_a
    return (out, out_alt) if y_alt is not None else out


def main() -> int:
    result = {
        "layer": L,
        "seed": SEED,
        "folds": FOLDS,
        "gcv_dof_cap": fit825.GCV_DOF_CAP,
        "cells": {},
        "swap": {},
        "validation": [],
    }
    for model in ("base", "instruct"):
        store = load_l19(model)
        print(f"[perfold] {model}: {store['X'].shape[0]} rows loaded (L{L} only)")
        for persona in c1310.PERSONA_LABELS:
            m = store["char_ids"] == persona
            r = perfold_fit(store["X"][m], store["Y"][m], store["group_ids"][m])
            cell_id = f"onpolicy_{model}_{persona}"
            committed = json.loads((EV / f"cells_{cell_id}.json").read_text())
            delta = abs(r["pooled"] - committed["r2_per_layer_obs"][L])
            result["validation"].append(
                {
                    "cell": cell_id,
                    "recomputed": r["pooled"],
                    "committed": committed["r2_per_layer_obs"][L],
                    "abs_delta": delta,
                }
            )
            assert delta < 0.01, (cell_id, r["pooled"], committed["r2_per_layer_obs"][L])
            result["cells"][cell_id] = r
            print(
                f"[perfold] {cell_id}: pooled {r['pooled']:+.4f} (committed "
                f"{committed['r2_per_layer_obs'][L]:+.4f}, |d|={delta:.2e})"
            )
        # swap control (correct + swapped share fold caches)
        rows, partners = swap_derangement(
            store["group_ids"], store["char_ids"], store["turn_indices"], seed=c1310.BUILD_SEED
        )
        rc, rs = perfold_fit(
            store["X"][rows],
            store["Y"][rows],
            store["group_ids"][rows],
            y_alt=store["Y"][partners],
        )
        committed_swap = json.loads((EV / f"swap_onpolicy_{model}.json").read_text())
        result["swap"][model] = {
            "correct": rc,
            "swap": rs,
            "committed_r2_correct": committed_swap["r2_correct"],
            "committed_r2_swap": committed_swap["r2_swap"],
        }
        print(
            f"[perfold] swap {model}: correct {rc['pooled']:+.4f} "
            f"(committed gb {committed_swap['r2_correct']:+.4f}), "
            f"swap {rs['pooled']:+.4f} (committed gb {committed_swap['r2_swap']:+.4f})"
        )
    out_path = EV / "analyzer_perfold_l19.json"
    out_path.write_text(json.dumps(result, indent=1))
    print("[perfold] wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

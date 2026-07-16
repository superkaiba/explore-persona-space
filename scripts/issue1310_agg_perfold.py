"""Issue #1310 analyzer: per-fold / per-point L19 recompute of the AGGREGATED fits.

The scene-aggregated re-fit (scripts/issue1310_aggfit.py @18791c2a) persists
pooled R^2 + nulls + bootstrap but NOT per-fold or per-point predictions. This
script streams the onpolicy prefill activation store from HF ONE SHARD AT A
TIME (download -> slice layer 19 -> delete; peak disk ~one shard, RAM ~100 MB),
replicates issue1310_aggfit.aggregate_store at the single layer, and recomputes
the aggregated fits' held-out predictions with the IDENTICAL fold assignment +
GCV Gram ridge + dof cap (fit825, seed 0, GCV_DOF_CAP=0.9) to expose:

  - per-FOLD held-out R^2 (5 points per cell),
  - per-POINT held-out R^2 (~300 points per cell; one point per scene, SS_tot
    around the fold-test mean, so points decompose the pooled statistic),
  - validation: recomputed pooled R^2 vs the committed
    cells_agg_*.json r2_per_layer_obs[19] (assert |delta| < 0.01, #833 tol),
  - the pooled swap-control twin (correct + swapped share fold caches).

Output: eval_results/issue_1310/onpolicy_aggregated/analyzer_agg_perfold_l19.json
(committed to the issue branch) — consumed by issue1310_agg_figures.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # HF token + shared-VM thread caps before torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue825_fit_cells as fit825  # noqa: E402
import issue1310_common as c1310  # noqa: E402
from issue1310_aggfit import aggregate_store  # noqa: E402
from issue1310_analyzer_perfold import perfold_fit  # noqa: E402
from issue1310_fit import swap_derangement  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
STORE_PREFIX = "issue1310_char_map/analysis_tensors/store_onpolicy"
CACHE = REPO / "data" / "issue_1310" / "aggslice_dl"
EV_AGG = REPO / "eval_results" / "issue_1310" / "onpolicy_aggregated"
L = 19

fit825.GCV_DOF_CAP = 0.9  # the committed aggregated fits' cap


def stream_l19(model_kind: str) -> dict:
    """Download shards one at a time, keep ONLY layer-19 slices, delete each.

    Shard iteration order matches issue1310_fit.load_model_store's
    ``sorted(glob(f"{model_kind}_shard*.pt"))`` (sorted by filename), so the
    concatenated row order — and therefore aggregate_store's lexsort — is
    identical to the committed aggfit run's.
    """
    api = HfApi()
    prefix = f"{STORE_PREFIX}/{model_kind}"
    names = sorted(
        e.path
        # HUB_VERIFY_RETRY_EXEMPT: scoped download enumerator, not a post-upload
        # verify; transient failures abort loudly and the recompute is re-runnable
        for e in api.list_repo_tree(DATA_REPO, path_in_repo=prefix, repo_type="dataset")
        if e.path.rsplit("/", 1)[-1].startswith(f"{model_kind}_shard") and e.path.endswith(".pt")
    )
    assert names, f"no shards under {prefix}"
    rows, groups, chars, turns = [], [], [], []
    xs, ys = [], []
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in names:
        local = Path(hf_hub_download(DATA_REPO, name, repo_type="dataset", local_dir=CACHE))
        payload = torch.load(local, map_location="cpu", weights_only=False)
        rows.extend(payload["row_ids"])
        groups.extend(payload["group_ids"])
        chars.extend(payload["char_ids"])
        turns.extend(payload["turn_indices"])
        xs.append(payload["arrays"]["x_spanmean"][:, L, :].float().numpy())
        ys.append(payload["arrays"]["y"][:, L, :].float().numpy())
        del payload
        local.unlink()  # stream-reduce: peak disk ~one shard
        print(f"[agg-perfold] {name}: sliced L{L}, deleted local copy")
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    # aggregate_store expects full arrays under "arrays"; the L19 slice stands
    # in — its X-pick (rows[0]) and float64 Y mean act per row, so the single
    # layer aggregates to exactly the committed run's layer-19 values.
    return {
        "char_ids": np.asarray(chars),
        "group_ids": np.asarray(groups),
        "turn_indices": np.asarray(turns, dtype=int),
        "row_ids": np.asarray(rows),
        "arrays": {"x_spanmean": x, "y": y},
    }


def main() -> int:
    result = {
        "layer": L,
        "seed": c1310.FIT_SEED,
        "folds": c1310.N_FOLDS,
        "gcv_dof_cap": fit825.GCV_DOF_CAP,
        "cells": {},
        "swap": {},
        "validation": [],
    }
    for model in ("base", "instruct"):
        agg = aggregate_store(stream_l19(model))
        print(f"[agg-perfold] {model}: {len(agg['personas'])} aggregated points")
        for persona in c1310.PERSONA_LABELS:
            m = agg["personas"] == persona
            r = perfold_fit(agg["X"][m], agg["Y"][m], agg["scenarios"][m])
            cell_id = f"agg_{model}_{persona}"
            committed = json.loads((EV_AGG / f"cells_{cell_id}.json").read_text())
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
                f"[agg-perfold] {cell_id}: pooled {r['pooled']:+.4f} (committed "
                f"{committed['r2_per_layer_obs'][L]:+.4f}, |d|={delta:.2e})"
            )
        zeros = np.zeros(len(agg["personas"]), dtype=int)
        rows, partners = swap_derangement(
            agg["scenarios"], agg["personas"], zeros, seed=c1310.BUILD_SEED
        )
        rc, rs = perfold_fit(
            agg["X"][rows], agg["Y"][rows], agg["scenarios"][rows], y_alt=agg["Y"][partners]
        )
        committed_swap = json.loads((EV_AGG / f"swap_agg_{model}.json").read_text())
        result["swap"][model] = {
            "correct": rc,
            "swap": rs,
            "committed_r2_correct": committed_swap["r2_correct"],
            "committed_r2_swap": committed_swap["r2_swap"],
        }
        print(
            f"[agg-perfold] swap {model}: correct {rc['pooled']:+.4f} "
            f"(committed gb {committed_swap['r2_correct']:+.4f}), "
            f"swap {rs['pooled']:+.4f} (committed gb {committed_swap['r2_swap']:+.4f})"
        )
    out_path = EV_AGG / "analyzer_agg_perfold_l19.json"
    out_path.write_text(json.dumps(result, indent=1))
    print("[agg-perfold] wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

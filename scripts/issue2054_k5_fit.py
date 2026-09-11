"""Matched K1/K3/K5 maps with a six-setting pool within each model.

Folds, not cells, shard across GPUs. One pooled eigendecomposition serves all
three target counts; each own-map decomposition likewise serves all counts.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k3 as k3
from scripts import issue2054_k3_artifacts as artifacts
from scripts import issue2054_k3_fit as prior
from scripts import issue2054_k5 as k5
from scripts.issue2054_ctx2ctx_fit import SharedEighRidge, load_fold_map


def load_panel(root, manifest, model):
    """Assert exact six-cell population, row pairing, context and fold identity."""
    records = [c for c in k5.selected(manifest) if c["cell"].endswith("__" + model)]
    if len(records) != 6:
        raise RuntimeError("pooled fit must contain six settings of one model")
    fold_map = load_fold_map(str(root / manifest["fold_map"]), "origin/main")["fold_of"]
    panel = {}
    for record in records:
        cell = record["cell"]
        targets = {}
        for count in k5.COUNTS:
            path = root / f"k{count}" / f"{cell}.npz"
            if not k3.complete(path, k5.fingerprint(manifest, cell)):
                raise RuntimeError("unverified target aggregate")
            with np.load(path, allow_pickle=False) as z:
                targets[count] = z["v_A"].astype(np.float64)
                if count == 1:
                    x = z["v_C"].astype(np.float64)
                    ids = [str(v) for v in z["conv_id"]]
                    caps = z["cap_mask"]
                elif (
                    not np.array_equal(x, z["v_C"])
                    or ids != list(z["conv_id"])
                    or not np.array_equal(caps, z["cap_mask"])
                ):
                    raise RuntimeError("K1/K3/K5 populations or contexts differ")
        if len(set(ids)) != len(ids) or any(cid not in fold_map for cid in ids):
            raise RuntimeError("invalid context identity/fold join")
        membership = np.array([fold_map[cid] for cid in ids])
        panel[cell] = {
            "x": x,
            "targets": targets,
            "caps": caps,
            "membership": membership,
            "ids": ids,
        }
    return panel


def cohorts(cell, caps):
    """Keep primary all-row and explicit cap-sensitivity populations separate."""
    result = {"all": np.ones(len(caps), dtype=bool)}
    if "__bare_text__" in cell or ("__chat__" in cell and cell.endswith("__qwen2.5-7b")):
        result.update(original_draw_stopped=~caps[:, 0], all_five_stopped=~caps.any(1))
    return result


def fit_fp(manifest, cell):
    """Include the new drivers and reused fit/retrieval code in resume keys."""
    sources = [
        __file__,
        prior.__file__,
        inspect.getfile(SharedEighRidge),
        inspect.getfile(prior.knn_retrieval),
    ]
    return k5.fingerprint(manifest, cell) + "".join(k3.sha(p) for p in sources)


def fold_path(root, cell, cohort, count, fold):
    """Name each independently persisted fit unit."""
    return root / "fold_checkpoints" / f"{cell}__{cohort}__k{count}__fold{fold}.json"


def fit_model(root, manifest, model, shard, shards, *, device="cuda", first_fold=False):
    """Fit fold-owned pools and own maps with shared target factorizations."""
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("invalid fit shard")
    panel = load_panel(root, manifest, model)
    sizes = [len(p["x"]) for p in panel.values()]
    bounds = np.cumsum([0] + sizes)
    xp = np.concatenate([p["x"] for p in panel.values()])
    memberships = np.concatenate([p["membership"] for p in panel.values()])
    for f in range(1 if first_fold else 5):
        if f % shards != shard:
            continue
        expected = [
            (cell, cohort, count)
            for cell, p in panel.items()
            for cohort in cohorts(cell, p["caps"])
            for count in k5.COUNTS
        ]
        if all(
            k3.complete(fold_path(root, c, h, k, f), fit_fp(manifest, c)) for c, h, k in expected
        ):
            continue
        started = time.monotonic()
        pooled_train = memberships != f
        if pooled_train.sum() <= xp.shape[1]:
            raise RuntimeError("six-setting pooled fit left ambient regime")
        # This materialization is <=48k x3584, ~1.4GB float64; the 4-worker
        # GCP host is explicitly sized >=128GB. The much larger 56-cell pool
        # is never loaded. Share the full pooled factorization over K.
        pooled = SharedEighRidge(xp[pooled_train], xp, device=device)
        pooled_predictions = {}
        pooled_info = {}
        for count in k5.COUNTS:
            yp = np.concatenate([p["targets"][count] for p in panel.values()])
            pooled_predictions[count], pooled_info[count] = pooled.fit_predict(yp[pooled_train])
        del pooled, yp
        k3.log(
            f"[phase=pooled_fit] model={model} fold={f} settings=6 n_train={pooled_train.sum()} elapsed={time.monotonic() - started:.1f}s"
        )
        for ci, (cell, p) in enumerate(panel.items()):
            x, targets, membership = p["x"], p["targets"], p["membership"]
            pooled_cell = {
                k: pred[bounds[ci] : bounds[ci + 1]] for k, pred in pooled_predictions.items()
            }
            for cohort, mask in cohorts(cell, p["caps"]).items():
                paths = {k: fold_path(root, cell, cohort, k, f) for k in k5.COUNTS}
                fp = fit_fp(manifest, cell)
                if all(k3.complete(path, fp) for path in paths.values()):
                    continue
                started = time.monotonic()
                unsupported = prior.cohort_guard(x, targets, membership, mask)
                if unsupported is not None and cohort == "all":
                    raise RuntimeError(f"primary cohort invalid: {cell} {unsupported}")
                if unsupported:
                    for count, path in paths.items():
                        k3.atomic_json(
                            path,
                            {
                                "cell": cell,
                                "cohort": cohort,
                                "k_rollouts": count,
                                "fold": f,
                                **unsupported,
                            },
                        )
                else:
                    train = mask & (membership != f)
                    test = mask & (membership == f)
                    xt, xe = x[train], x[test]
                    own = SharedEighRidge(xt, xe, device=device)
                    for count, path in paths.items():
                        yt, ye = targets[count][train], targets[count][test]
                        own_prediction, info = own.fit_predict(yt)
                        ptrain, ptest = pooled_cell[count][train], pooled_cell[count][test]
                        shift = (yt - ptrain).mean(0)
                        centered = ptrain - ptrain.mean(0)
                        denom = float((centered * centered).sum())
                        if denom <= 0:
                            raise RuntimeError("degenerate pooled prediction")
                        gain = float((centered * (yt - yt.mean(0))).sum()) / denom
                        predictions = {
                            "own": own_prediction,
                            "identity_bias": xe + (yt - xt).mean(0),
                            "pooled": ptest,
                            "shift": ptest + shift,
                            "rescale": gain * (ptest - ptrain.mean(0)) + yt.mean(0),
                        }
                        record = {
                            "status": "complete",
                            "cell": cell,
                            "cohort": cohort,
                            "k_rollouts": count,
                            "fold": f,
                            "n_train": int(train.sum()),
                            "n_test": int(test.sum()),
                            "ridge": info,
                            "pooled_ridge": pooled_info[count],
                            "gain": gain,
                            "pooled_training_cells": list(panel),
                            "pooled_training_cohort": "complete five draws, caps retained; held-out conversation fold excluded across all six settings",
                            "test_cap_counts_each_draw": p["caps"][test].sum(0).tolist(),
                            "metrics": {
                                name: prior.score(pred, ye) for name, pred in predictions.items()
                            },
                        }
                        # The K5 significance companion inherits 100 draws from
                        # K3. K1/K3 are descriptive paired reference refits here.
                        if cohort == "all" and count == 5:
                            record["own_shuffled_null_r2"] = own.null_r2(
                                yt, ye, n_draws=100, seed=k5.seed(cell, str(f), 3), chunk=2
                            ).tolist()
                        k3.atomic_json(path, record)
                    del own
                artifacts.seal_many(list(paths.values()), root, fp)
                k3.log(
                    f"[phase=fits] model={model} fold={f} cell={ci + 1}/6 cohort={cohort} elapsed={time.monotonic() - started:.1f}s"
                )


def collect(root, manifest):
    """Require exactly 36 completed primary panels and all five folds."""
    results = []
    primary = []
    for record in k5.selected(manifest):
        cell = record["cell"]
        with np.load(root / "k5" / f"{cell}.npz", allow_pickle=False) as z:
            names = list(cohorts(cell, z["cap_mask"]))
        for cohort in names:
            for count in k5.COUNTS:
                folds = []
                for f in range(5):
                    path = fold_path(root, cell, cohort, count, f)
                    if not k3.complete(path, fit_fp(manifest, cell)):
                        raise RuntimeError(f"missing verified fold {path}")
                    value = json.loads(path.read_text())
                    if any(
                        value[k] != v
                        for k, v in {
                            "cell": cell,
                            "cohort": cohort,
                            "k_rollouts": count,
                            "fold": f,
                        }.items()
                    ):
                        raise RuntimeError("fold checkpoint identity mismatch")
                    folds.append(value)
                statuses = {r["status"] for r in folds}
                if len(statuses) != 1:
                    raise RuntimeError("cohort support differs across folds")
                result = {
                    "cell": cell,
                    "cohort": cohort,
                    "k_rollouts": count,
                    "status": statuses.pop(),
                    "folds": folds,
                    "context_convention": "same K3 prefill-alone last-token vector, layer19",
                    "comparison_cohort": "complete five draws for all K values",
                }
                if result["status"] == "complete":
                    result["r2_mean"] = {
                        name: float(np.mean([r["metrics"][name]["r2"] for r in folds]))
                        for name in ("own", "identity_bias", "pooled", "shift", "rescale")
                    }
                if cohort == "all":
                    if result["status"] != "complete":
                        raise RuntimeError("required primary panel withheld")
                    primary.append(result)
                results.append(result)
    if len(primary) != 36:
        raise RuntimeError("incomplete K1/K3/K5 primary coverage")
    payload = {
        "parent_revision": k5.PARENT_REV,
        "results": results,
        "coverage": json.loads((root / "coverage.json").read_text()),
        "capture_adjudication": k5.policy(),
        "pooled_estimand": "six displayed settings within each model, separately; fixed inserted controls excluded",
        "cap_sensitivity": "original_draw_stopped conditions only on draw0; all_five_stopped excludes every cap, and own refit is withheld when any fold has n_train<=3584",
    }
    k3.atomic_json(root / "results.json", payload)
    artifacts.seal_many([root / "results.json"], root, k3.sha(__file__))
    return primary

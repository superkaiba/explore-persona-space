"""Aggregate exact three-draw targets and refit §4.4 with parent ridge helpers."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from scripts import issue2054_k3 as k3
from scripts.issue2054_ctx2ctx_fit import SharedEighRidge, discover_cells, load_fold_map
from scripts.issue2054_pool_specialize import accumulate_pooled_moments, fit_pooled_per_fold
from explore_persona_space.analysis.mapping_baselines import knn_retrieval


def average_three(first, fresh, valid):
    if fresh.shape != (len(first), 2, first.shape[1]) or valid.shape != (len(first), 2):
        raise ValueError("K3 requires exactly two fresh vectors for each original")
    keep = valid.all(axis=1)
    if not np.isfinite(first).all() or not np.isfinite(fresh[keep]).all():
        raise ValueError("nonfinite valid answer vectors")
    return (first[keep].astype(np.float32) + fresh[keep].astype(np.float32).sum(axis=1)) / 3.0, keep


def save_npz(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **content)
    tmp.replace(path)


def aggregate(root, manifest, pilot):
    reports = []
    for record in manifest["cells"]:
        cell = record["cell"]
        if "raw" not in record:
            for count in (1, 3):
                dest = root / f"k{count}" / f"{cell}.npz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.symlink_to(root / "inputs" / record["activation"])
            continue
        fp = k3.fingerprint(manifest, cell, pilot)
        rows = k3.banked_rows(root, record, pilot)
        chunks = []
        for offset in range(0, len(rows), k3.CHUNK):
            path = root / "captures" / cell / f"chunk_{offset:05d}.npz"
            if not k3.complete(path, fp):
                raise RuntimeError(f"missing verified capture: {path}")
            with np.load(path, allow_pickle=False) as z:
                chunks.append({name: z[name] for name in z.files})
        values = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
        if list(values["conv_id"]) != [r["conv_id"] for r in rows]:
            raise RuntimeError(f"aggregate population mismatch: {cell}")
        mean, keep = average_three(values["v_A_0"], values["v_A_12"], values["valid_draws_12"])
        for count in (1, 3):
            content = {
                key: values[key][keep]
                for key in ("conv_id", "v_C", "v_P", "v_P_present", "cap_mask")
            }
            content["v_A"] = values["v_A_0"][keep] if count == 1 else mean
            path = root / f"k{count}" / f"{cell}.npz"
            save_npz(path, content)
            k3.seal(path, root, fp)
        drift = np.linalg.norm(
            values["v_C"].astype(np.float32) - values["banked_v_C"].astype(np.float32), axis=1
        )
        reports.append(
            {
                "cell": cell,
                "original_rows": len(rows),
                "complete_three_draw_rows": int(keep.sum()),
                "empty_draw_rows_excluded": int((~keep).sum()),
                "cap_counts_each_draw": values["cap_mask"].sum(axis=0).tolist(),
                "all_three_uncapped_valid": int((keep & ~values["cap_mask"].any(axis=1)).sum()),
                "original_stopped_valid": int((keep & ~values["cap_mask"][:, 0]).sum()),
                "context_drift_norm_max": float(drift.max()),
                "context_drift_norm_mean": float(drift.mean()),
                "capture_parity_relative_error_max": float(values["parity_relative_error"].max()),
            }
        )
        k3.log(f"[phase=aggregate] {cell} complete_K3={keep.sum()}/{len(rows)}")
    k3.atomic_json(root / "coverage.json", reports)
    k3.seal(root / "coverage.json", root, k3.VERSION)


def displayed(cell):
    variant, condition, form, _ = cell.split("__")
    return condition == "on_policy" and (
        (variant == "conversation_paired_stories_assistant" and form in ("chat", "bare_text"))
        or (variant.startswith("char_") and form == "attrib_quoted")
    )


def score(pred, y):
    total = float(((y - y.mean(axis=0)) ** 2).sum())
    if not total > 0:
        raise ValueError("undefined held-out R2")
    return {
        "r2": 1 - float(((pred - y) ** 2).sum()) / total,
        "retrieval": {
            metric: knn_retrieval(pred, y, ks=(1, 5, 10), metric=metric)
            for metric in ("euclidean", "cosine")
        },
        "retrieval_pool": len(y),
        "chance_top1": 1 / len(y),
    }


def fits(root, manifest, device, pilot):
    fold_map = load_fold_map(str(root / manifest["fold_map"]), "origin/main")
    if pilot:
        # Smoke blind spot: tiny pilot has no ambient-fit sample size. It
        # validates generation/capture/aggregation/upload, never statistical fits.
        k3.atomic_json(
            root / "fits_pending.json",
            {
                "status": "pilot_capture_only",
                "reason": "ambient-fit sample floor",
                "rows_per_cell": k3.PILOT_ROWS,
            },
        )
        k3.seal(root / "fits_pending.json", root, k3.VERSION)
        return
    folds = fold_map["fold_of"]
    selected = [c for c in manifest["cells"] if displayed(c["cell"])]
    if len(selected) != 12:
        raise RuntimeError("displayed panel must contain 12 cells")
    for count in (1, 3):
        cells = discover_cells(root / f"k{count}")
        if {c.key for c in cells} != {c["cell"] for c in manifest["cells"]}:
            raise RuntimeError("pooled cell set mismatch")
        acc = accumulate_pooled_moments(cells, folds, 5, ["context"], device)
        pooled = fit_pooled_per_fold(acc["mom"]["context"], list(range(5)), 5)
        del acc
        for ci, record in enumerate(selected):
            cell = record["cell"]
            with np.load(root / f"k{count}" / f"{cell}.npz", allow_pickle=False) as z:
                x = z["v_C"].astype(np.float64)
                y = z["v_A"].astype(np.float64)
                ids = [str(v) for v in z["conv_id"]]
                caps = z["cap_mask"]
            if any(cid not in folds for cid in ids):
                raise RuntimeError(f"unknown fold membership: {cell}")
            membership = np.array([folds[cid] for cid in ids])
            for cohort, mask in [
                ("all", np.ones(len(ids), dtype=bool)),
                ("original_draw_stopped", ~caps[:, 0]),
                ("all_three_stopped", ~caps.any(axis=1)),
            ]:
                # Restriction sensitivities are needed for the plain-text cell.
                if cohort != "all" and "__bare_text__" not in cell:
                    continue
                sizes = [int((mask & (membership != f)).sum()) for f in range(5)]
                path = root / "fits" / f"k{count}" / f"{cell}__{cohort}.json"
                fp = k3.fingerprint(manifest, cell, False) + k3.sha(__file__)
                if k3.complete(path, fp):
                    continue
                if min(sizes) <= x.shape[1]:
                    result = {
                        "cell": cell,
                        "k_rollouts": count,
                        "cohort": cohort,
                        "status": "insufficient_ambient_training_rows",
                        "train_rows": sizes,
                        "required_exclusive_minimum": x.shape[1],
                    }
                else:
                    records = []
                    for f in range(5):
                        t0 = time.monotonic()
                        train = mask & (membership != f)
                        test = mask & (membership == f)
                        xt, yt, xe, ye = x[train], y[train], x[test], y[test]
                        own = SharedEighRidge(xt, xe, device=device)
                        pred, info = own.fit_predict(yt)
                        identity = xe + (yt - xt).mean(axis=0)
                        ptrain = pooled[f].predict_np(xt)
                        ptest = pooled[f].predict_np(xe)
                        shift = (yt - ptrain).mean(axis=0)
                        zc = ptrain - ptrain.mean(axis=0)
                        yc = yt - yt.mean(axis=0)
                        denom = float((zc * zc).sum())
                        if denom <= 0:
                            raise RuntimeError("degenerate pooled predictions")
                        gain = float((zc * yc).sum()) / denom
                        predictions = {
                            "own": pred,
                            "identity_bias": identity,
                            "pooled": ptest,
                            "shift": ptest + shift,
                            "rescale": gain * (ptest - ptrain.mean(axis=0)) + yt.mean(axis=0),
                        }
                        record_fold = {
                            "fold": f,
                            "n_train": int(train.sum()),
                            "n_test": int(test.sum()),
                            "ridge": info,
                            "gain": gain,
                            "metrics": {name: score(p, ye) for name, p in predictions.items()},
                        }
                        if cohort == "all":
                            null = own.null_r2(
                                yt,
                                ye,
                                n_draws=100,
                                seed=k3.seed(cell, str(f), count if count == 1 else 2),
                                chunk=2,
                            )
                            record_fold["own_shuffled_null_r2"] = null.tolist()
                        records.append(record_fold)
                        del own
                        k3.log(
                            f"[phase=fits] k={count} cell={ci + 1}/12 cohort={cohort} fold={f + 1}/5 seconds={time.monotonic() - t0:.1f}"
                        )
                    result = {
                        "cell": cell,
                        "k_rollouts": count,
                        "cohort": cohort,
                        "status": "complete",
                        "folds": records,
                        "pooled_training": "all complete-vector rows of all 56 fixed cells",
                        "context_convention": "prefill-alone-last-token",
                        "r2_mean": {
                            name: float(np.mean([r["metrics"][name]["r2"] for r in records]))
                            for name in predictions
                        },
                    }
                k3.atomic_json(path, result)
                k3.seal(path, root, fp)
        del pooled
    results = [
        json.loads(p.read_text())
        for p in sorted((root / "fits").glob("k*/*.json"))
        if not p.name.endswith(".done.json")
    ]
    k3.atomic_json(
        root / "results.json",
        {
            "dataset_revision": manifest["revision"],
            "coverage": json.loads((root / "coverage.json").read_text()),
            "results": results,
        },
    )
    k3.seal(root / "results.json", root, k3.sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = args.out_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    aggregate(root, manifest, args.pilot)
    fits(root, manifest, args.device, args.pilot)


if __name__ == "__main__":
    main()

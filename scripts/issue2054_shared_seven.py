"""Refit the Instruct shared map on all seven saved K5 settings."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import gc
import hashlib
import inspect
import json
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explore_persona_space.orchestrate.hub import stage_hub_file
from scripts import issue2054_k5_assistant_story_analysis as parent
from scripts import issue2054_k5_loso_calibration as base
from scripts import issue2054_k5_map_geometry as geometry
from scripts import issue2054_ctx2ctx_fit as ridge_recipe
from scripts.issue2054_pool_specialize import PooledMomentRidge

MODEL = "qwen2.5-7b-instruct"
STORY_REV = "400ad464ce9f092722f737dd336fc31a03fba524"
STORY_SHA = "fb04162702296cc52aa918b9aba5e275b229a3ccaf9355885e3d2de865e69c58"
FOLD_SHA = "4ab1839a0e8c5e8705147cbb529b2df36975ac46b987fe71ab3f919265e4c39e"
PREFIX = "issue2054_shared_seven_k5"
METHODS = ("shared", "shared_bias", "pooled_identity_bias", "target_identity_bias")


def log(out, phase, **fields):
    """Write timestamped actual progress and measured peak RSS."""
    record = {
        "checked_at": time.time(),
        "phase": phase,
        "rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        **fields,
    }
    base.atomic_json(out / "progress.json", record)
    print(json.dumps(record), flush=True)


def read_inputs(out):
    """Reuse exact original banks; stage only an absent immutable input."""
    old_path = ROOT / "eval_results/issue_2054/section44_k5/k5_results.json"
    story_path = ROOT / "eval_results/issue_2054/assistant_story_k5/results.json"
    if (
        base.sha(old_path) != geometry.REFERENCE_SHA
        or base.sha(story_path)
        != "d9f67bd2dfaf0cbd14a97af3234adb943e82bdf6b6150071af2b061817fe8070"
    ):
        raise ValueError("Reference result bytes changed")
    old, story = json.loads(old_path.read_text()), json.loads(story_path.read_text())
    original = {
        r["cell"]: r
        for r in old["results"]
        if r["cohort"] == "all" and r["k_rollouts"] == 5 and r["cell"].endswith(MODEL)
    }
    story_model = next(m for m in story["models"] if m["model"] == MODEL)
    cached = {
        r["path"]: r["local"]
        for r in json.loads(
            (ROOT / "eval_results/issue_2054/k5_loso_calibration/inputs.json").read_text()
        )
    }
    sources = []
    for label, prefix in [*base.SETTINGS, ("Story assistant", parent.STORY)]:
        cell = f"{prefix}__{MODEL}"
        if prefix == parent.STORY:
            source = {
                "path": f"issue2054_assistant_story_k5/production_v1/k5/{cell}.npz",
                "revision": STORY_REV,
                "sha256": STORY_SHA,
            }
            own = story_model["own_folds"]
        else:
            source = {
                k: story["provenance"]["banks"][cell][k] for k in ("path", "revision", "sha256")
            }
            own = original[cell]["folds"]
        local = (
            Path(cached[source["path"]])
            if source["path"] in cached
            else out / "inputs" / f"{cell}.npz"
        )
        if not local.exists():
            log(out, "stage_input", cell=cell)
            stage_hub_file(base.HF_REPO, source["path"], local, revision=source["revision"])
        if base.sha(local) != source["sha256"]:
            raise ValueError(f"Input hash mismatch: {cell}")
        sources.append(
            {**source, "cell": cell, "label": label, "local": str(local), "own_folds": own}
        )
    fold_bytes = subprocess.check_output(
        ["git", "show", f"{base.SOURCE_SHA}:eval_results/issue_2054/shared_fold_map.json"], cwd=ROOT
    )
    if hashlib.sha256(fold_bytes).hexdigest() != FOLD_SHA:
        raise ValueError("Fold map hash mismatch")
    fold_map = json.loads(fold_bytes)
    if fold_map["k"] != 5 or fold_map["seed"] != 137:
        raise ValueError("Unexpected folds")
    base.atomic_json(out / "inputs.json", sources)
    return sources, fold_map["fold_of"]


def load_bank(source, fold_map):
    """Validate complete-five membership and own-fit cohort identity."""
    with np.load(source["local"], allow_pickle=False) as bank:
        ids = list(map(str, bank["conv_id"]))
        x, y = parent.tensor(bank["v_C"]).numpy(), parent.tensor(bank["v_A"]).numpy()
        caps = bank["cap_mask"]
    if (
        x.shape != y.shape
        or x.shape != (len(ids), 3584)
        or caps.shape != (len(ids), 5)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("Invalid complete-five bank shape/IDs")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite bank")
    membership = np.array([fold_map[cid] for cid in ids])
    for f in range(5):
        own = source["own_folds"][f]
        test = membership == f
        if own["fold"] != f or own["n_test"] != int(test.sum()):
            raise ValueError("Own test cohort mismatch")
        n_train = own["ridge"]["n_train"]
        if n_train != int((~test).sum()):
            raise ValueError("Own training cohort mismatch")
        if source["cell"].startswith(parent.STORY + "__"):
            for key, mask in [("train_ids_sha256", ~test), ("test_ids_sha256", test)]:
                if parent.digest(np.asarray(ids)[mask].tolist()) != own[key]:
                    raise ValueError("Story own-fit ID order mismatch")
    return {"x": x, "y": y, "ids": ids, "membership": membership}


def collect_moments(out, sources, fold_map):
    """Sum raw fold moments across settings before any standardization."""
    total = None
    for source in sources:
        log(out, "accumulate_moments", cell=source["cell"])
        panel = load_bank(source, fold_map)
        moment = geometry.moments_by_fold(panel)
        moment["yss"] = torch.stack(
            [parent.tensor(panel["y"][panel["membership"] == f]).square().sum() for f in range(5)]
        )
        if total is None:
            total = moment
        else:
            for key in total:
                total[key].add_(moment[key])
        del panel, moment
        gc.collect()
    return total


def fit_maps(out, moment, fingerprint):
    """Fit fold zero with the parent oracle, then bounded batches; save each map."""
    all_info = []
    for folds in ([0], [1, 2], [3, 4]):
        pending = []
        for fold in folds:
            existing = resume_record(out, out / "maps" / f"fold{fold}.json", fingerprint)
            if existing is None:
                pending.append(fold)
            else:
                all_info.append(existing)
        folds = pending
        if not folds:
            continue
        started = time.monotonic()
        log(out, "fit_shared", folds=folds)
        maps, biases, identities, infos = parent.gcv_maps(moment, folds)
        if folds == [0]:
            train = parent.train_moments(moment, [0])
            oracle = PooledMomentRidge(
                n=int(train["n"][0]),
                sum_x=train["sx"][0],
                sum_y=train["sy"][0],
                yss=float(train["yss"][0]),
                c_xx=train["xx"][0],
                c_xy=train["xy"][0],
            )
            if oracle.best_lambda != infos[0]["best_lambda"]:
                raise ValueError("Parent oracle selected a different penalty")
            np.testing.assert_allclose(
                maps[0].numpy(), (oracle.map / oracle.sd[:, None]).numpy(), rtol=1e-7, atol=1e-8
            )
            expected_bias = oracle.mu_y - oracle.mu_x @ (oracle.map / oracle.sd[:, None])
            np.testing.assert_allclose(
                biases[0].numpy(), expected_bias.numpy(), rtol=1e-7, atol=1e-8
            )
            del train, oracle
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            log(out, "pilot_pass", seconds=time.monotonic() - started, rss_gib=rss, oracle="PASS")
            if rss > 10:
                raise RuntimeError(
                    "Pilot RSS exceeds local expansion budget; move remaining work off shared VM"
                )
        for i, f in enumerate(folds):
            path = out / "maps" / f"fold{f}.npz"
            base.save_npz(
                path,
                {
                    "map": maps[i].numpy(),
                    "bias": biases[i].numpy(),
                    "identity_bias": identities[i].numpy(),
                },
            )
            record = {
                **infos[i],
                "array_path": str(path.relative_to(out)),
                "array_sha256": base.sha(path),
                "fingerprint": fingerprint,
                "batch_seconds": time.monotonic() - started,
            }
            base.atomic_json(path.with_suffix(".json"), record)
            all_info.append(record)
        del maps, biases, identities
        gc.collect()
    return sorted(all_info, key=lambda row: row["fold"])


def resume_record(out, path, fingerprint):
    """Accept a completed unit only when source and payload hashes match."""
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if (
        row["fingerprint"] != fingerprint
        or base.sha(out / row["array_path"]) != row["array_sha256"]
    ):
        raise ValueError(f"Invalid resume unit: {path}")
    return row


def evaluate(out, sources, fold_map, maps_meta, fingerprint):
    """Score all seven settings on excluded conversations with matched controls."""
    rows = []
    for source in sources:
        p = load_bank(source, fold_map)
        ids = np.asarray(p["ids"])
        for fold in range(5):
            path = out / "predictions" / f"{source['cell']}__fold{fold}.npz"
            existing = resume_record(out, path.with_suffix(".json"), fingerprint)
            if existing is not None:
                rows.append(existing)
                continue
            log(out, "score", cell=source["cell"], fold=fold)
            test = p["membership"] == fold
            x, y = parent.tensor(p["x"][test]), parent.tensor(p["y"][test])
            mu_x, mu_y = parent.tensor(p["x"][~test]).mean(0), parent.tensor(p["y"][~test]).mean(0)
            meta = maps_meta[fold]
            with np.load(out / meta["array_path"], allow_pickle=False) as bank:
                amap, bias, identity = (
                    parent.tensor(bank[k]) for k in ("map", "bias", "identity_bias")
                )
            prediction = x @ amap + bias
            adjustment = mu_y - (mu_x @ amap + bias)
            target_identity = mu_y - mu_x
            predictions = torch.stack(
                [prediction, prediction + adjustment, x + identity, x + target_identity]
            )
            metrics = dict(zip(METHODS, parent.score_batch(predictions, y), strict=True))
            own = source["own_folds"][fold]
            metrics["own"] = own["own"] if "own" in own else own["metrics"]["own"]
            path = out / "predictions" / f"{source['cell']}__fold{fold}.npz"
            base.save_npz(
                path,
                {
                    "prediction": prediction.float().numpy(),
                    "test_conv_id": ids[test],
                    "bias_adjustment": adjustment.numpy(),
                    "target_identity_bias": target_identity.numpy(),
                    "squared_error": (predictions - y).square().sum(2).numpy(),
                    "squared_target_deviation": (y - y.mean(0)).square().sum(1).numpy(),
                },
            )
            row = {
                "cell": source["cell"],
                "label": source["label"],
                "fold": fold,
                "n_train": int((~test).sum()),
                "n_test": int(test.sum()),
                "shared_n_train": meta["n_train"],
                "metrics": metrics,
                "input_sha256": source["sha256"],
                "train_ids_sha256": parent.digest(ids[~test].tolist()),
                "test_ids_sha256": parent.digest(ids[test].tolist()),
                "fingerprint": fingerprint,
                "array_path": str(path.relative_to(out)),
                "array_sha256": base.sha(path),
                "prediction_storage": "float32; map and error sums float64",
            }
            base.atomic_json(path.with_suffix(".json"), row)
            rows.append(row)
            del predictions, amap, bias, identity, prediction, x, y
        del p
        gc.collect()
    return rows


def main():
    """Run the bounded analysis and publish every fitted map and scored unit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.source_sha:
        raise ValueError("Launched source differs from checkout")
    scientific_paths = [
        Path(p).resolve()
        for p in [
            __file__,
            parent.__file__,
            geometry.__file__,
            base.__file__,
            ridge_recipe.__file__,
            parent.artifacts.__file__,
            parent.k3.__file__,
            inspect.getfile(base.save_npz),
            inspect.getfile(PooledMomentRidge),
        ]
    ]
    for path in scientific_paths:
        relative = str(path.relative_to(ROOT))
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != base.sha(path):
            raise ValueError(f"Scientific source differs from committed version: {relative}")
    started = time.time()
    base.atomic_json(
        out / "runtime.json",
        {"started_at": started, "source_sha": head, "pid": __import__("os").getpid()},
    )
    if shutil.disk_usage(out).free < 2 * 2**30:
        raise RuntimeError("Need at least 2 GiB fresh output headroom")
    sources, fold_map = read_inputs(out)
    fingerprint = parent.digest(
        {
            "sources": [
                {k: v for k, v in s.items() if k not in ("local", "own_folds")} for s in sources
            ],
            "fold_sha256": FOLD_SHA,
            "source_sha": head,
            "helpers": {str(p.relative_to(ROOT)): base.sha(p) for p in scientific_paths},
        }
    )
    moment = collect_moments(out, sources, fold_map)
    maps = fit_maps(out, moment, fingerprint)
    del moment
    gc.collect()
    rows = evaluate(out, sources, fold_map, maps, fingerprint)
    if len(rows) != 35 or len({(r["cell"], r["fold"]) for r in rows}) != 35:
        raise ValueError("Incomplete seven-setting coverage")
    summary = []
    for source in sources:
        selected = [r for r in rows if r["cell"] == source["cell"]]
        r2 = {
            method: float(np.mean([r["metrics"][method]["r2"] for r in selected]))
            for method in (*METHODS, "own")
        }
        summary.append(
            {
                "cell": source["cell"],
                "label": source["label"],
                "r2_mean": r2,
                "shared_over_own": r2["shared"] / r2["own"],
            }
        )
    result = {
        "status": "complete",
        "source_sha": head,
        "model": MODEL,
        "started_at": started,
        "completed_at": time.time(),
        "fingerprint": fingerprint,
        "fold_sha256": FOLD_SHA,
        "sources": sources,
        "maps": maps,
        "rows": rows,
        "summary": summary,
        "training_settings": 7,
        "holdout": "conversation grouped, all seven settings included in training",
        "aggregation": "ratio of equal-five-fold mean R2",
        "rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }
    base.atomic_json(out / "results.json", result)
    log(out, "upload", files=len(rows) + len(maps))
    parent.k3.PREFIX = PREFIX
    paths = [out / "results.json", out / "inputs.json", out / "runtime.json"]
    for record in [*maps, *rows]:
        path = out / record["array_path"]
        paths.extend([path, path.with_suffix(".json")])
    parent.artifacts.seal_many(paths, out, fingerprint)
    receipts = [json.loads(p.with_suffix(p.suffix + ".done.json").read_text()) for p in paths]
    inventory = {
        "source_sha": head,
        "fingerprint": fingerprint,
        "files": [
            {"path": str(p.relative_to(out)), "size": p.stat().st_size, "sha256": base.sha(p)}
            for p in paths
        ],
    }
    base.atomic_json(out / "inventory.json", inventory)
    parent.artifacts.seal_many([out / "inventory.json"], out, fingerprint)
    receipt = json.loads((out / "inventory.json.done.json").read_text())
    complete = {
        "source_sha": head,
        "status": "complete",
        "finished_at": time.time(),
        "results_sha256": base.sha(out / "results.json"),
        "verified_revision": receipt["revision"],
        "inventory_sha256": base.sha(out / "inventory.json"),
        "uploaded_files": len(receipts) + 1,
    }
    base.atomic_json(out / "complete.json", complete)
    parent.artifacts.seal_many([out / "complete.json"], out, fingerprint)
    log(out, "complete", **complete)


if __name__ == "__main__":
    main()

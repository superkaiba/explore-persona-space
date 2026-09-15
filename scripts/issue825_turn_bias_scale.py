"""Calibrate banked single-answer turn-transfer maps without model inference.

Stage only layer-19 context/answer summaries, reproduce the original six-fold
GCV transfer, and fit target-training vector biases and scalar gains. The
source turns match the manuscript draft (1 and 3), with turn 12 as an own-map
reproduction anchor. All test conversations stay outside both fitting stages.
"""

# ruff: noqa: E402
# Shared-VM thread caps must be loaded before the numerical and Hub imports.

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import inspect
import json
import re
import resource
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from explore_persona_space.orchestrate.hub import retry_transient

REPO = Path(__file__).resolve().parents[1]
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REV = "cf513c9f89d6a3ccd078ae73e532da2600b50af6"
HF_PREFIX = "issue825_userbase_map/analysis_tensors/turn_dynamics/armR_own"
REFERENCE_SHA256 = "3c79d0630f9794ef03baa6a030b84d253b8585bfb8e5a6b958b6ff77f87bc3bb"
MODELS = ("instruct", "pretrained")
SOURCE_TURNS = (1, 3, 12)
METHODS = ("raw", "bias", "bias_scale", "identity_bias")
N_FOLDS = 6


def sha(path):
    """Hash immutable input or output bytes with bounded working memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    """Persist one complete checkpoint without exposing partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def log(message):
    """Emit timestamped progress for the independent monitor."""
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def reference(path):
    """Require the exact single-answer source used in the manuscript draft."""
    if sha(path) != REFERENCE_SHA256:
        raise RuntimeError("parent transfer reference bytes changed")
    data = json.loads(Path(path).read_text())
    if data["followup_label"] != "turn-dynamics-allturns-5000" or data["smoke"]:
        raise RuntimeError("wrong parent measurement regime")
    return data


def stage(args):
    """Fetch only the verified single-layer source bank, with resumable cache."""
    reference(args.reference)
    args.inputs.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    directories = [(m, s) for m in MODELS for s in range(4)]

    def inventory(item):
        model, shard = item
        prefix = f"{HF_PREFIX}/{model}/shard{shard}of4"
        entries = retry_transient(
            lambda: list(
                # HUB_VERIFY_RETRY_EXEMPT: retry_transient reopens and consumes the complete paginated listing.
                api.list_repo_tree(
                    HF_REPO,
                    path_in_repo=prefix,
                    repo_type="dataset",
                    revision=HF_REV,
                    recursive=False,
                )
            ),
            what=f"inventory {model} shard {shard}",
        )
        selected = []
        for entry in entries:
            name = Path(entry.path).name
            if not (
                re.fullmatch(r"row_index_shard\d+\.jsonl", name)
                or re.fullmatch(r"(context_k|answer_own_t1)_L19_shard\d+\.npy", name)
                or name in ("capture_report.json", "capture_fingerprint.json")
            ):
                continue
            selected.append(
                dict(
                    model=model,
                    shard=shard,
                    path=entry.path,
                    size=entry.size,
                    blob_id=entry.blob_id,
                    lfs_sha256=entry.lfs.sha256 if entry.lfs else None,
                )
            )
        groups = sorted(r["path"] for r in selected if "row_index" in r["path"])
        if len(groups) != 15 or len(selected) != 47:
            raise RuntimeError(f"unexpected capture inventory {prefix}: {len(selected)}")
        return selected

    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [r for group in pool.map(inventory, directories) for r in group]
    size = sum(r["size"] for r in jobs)
    free = shutil.disk_usage(args.inputs).free
    if size > 10 * 1024**3 or free < 1.5 * size:
        raise RuntimeError(f"unsafe staging footprint: input={size} free={free}")
    log(f"[stage] files={len(jobs)} bytes={size} free_bytes={free}")

    def fetch(row):
        path = Path(
            retry_transient(
                lambda: hf_hub_download(
                    HF_REPO,
                    row["path"],
                    repo_type="dataset",
                    revision=HF_REV,
                    cache_dir=args.inputs / "hf_cache",
                ),
                what=f"fetch {row['path']}",
            )
        )
        digest = sha(path)
        if path.stat().st_size != row["size"]:
            raise RuntimeError(f"size mismatch: {row['path']}")
        if row["lfs_sha256"]:
            if digest != row["lfs_sha256"]:
                raise RuntimeError(f"LFS hash mismatch: {row['path']}")
        else:
            content = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            if blob != row["blob_id"]:
                raise RuntimeError(f"Git blob mismatch: {row['path']}")
        return row | {"local": str(path), "sha256": digest}

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, row in enumerate(pool.map(fetch, jobs)):
            records.append(row)
            atomic_json(args.inputs / "inputs.partial.json", records)
            if (index + 1) % 16 == 0 or index + 1 == len(jobs):
                log(f"[stage] verified={index + 1}/{len(jobs)}")
    atomic_json(
        args.inputs / "inputs.json",
        {
            "repo": HF_REPO,
            "revision": HF_REV,
            "bytes": size,
            "files": records,
        },
    )
    log("[stage] complete")


def load_panel(args, model):
    """Open the staged consumer layout and validate row pairing and fold hash."""
    manifest = json.loads((args.inputs / "inputs.json").read_text())
    if manifest["repo"] != HF_REPO or manifest["revision"] != HF_REV:
        raise RuntimeError("staged source revision changed")
    records = {r["path"]: r for r in manifest["files"] if r["model"] == model}

    def checked(path):
        record = records[path]
        if sha(record["local"]) != record["sha256"]:
            raise RuntimeError(f"staged bytes changed: {path}")
        return Path(record["local"])

    rows, xs, ys = [], [], []
    for shard in range(4):
        prefix = f"{HF_PREFIX}/{model}/shard{shard}of4"
        report = json.loads(checked(prefix + "/capture_report.json").read_text())
        before = len(rows)
        for index_path in sorted(p for p in records if p.startswith(prefix + "/row_index")):
            with checked(index_path).open() as handle:
                block_rows = [json.loads(line) for line in handle if line.strip()]
            group = re.search(r"shard(\d+)\.jsonl$", index_path).group(1)
            block = []
            for kind in ("context_k", "answer_own_t1"):
                arr = np.load(
                    checked(f"{prefix}/{kind}_L19_shard{group}.npy"),
                    allow_pickle=False,
                    mmap_mode="r",
                )
                if arr.shape != (len(block_rows), 3584) or not np.isfinite(arr).all():
                    raise RuntimeError(f"invalid source tensor {kind}, {arr.shape}")
                block.append(np.asarray(arr, dtype=np.float32))
            rows.extend(block_rows)
            xs.append(block[0])
            ys.append(block[1])
        if len(rows) - before != report["n_kept"]:
            raise RuntimeError("capture report disagrees with realized rows")
    ids = [str(r["conv_id"]) for r in rows]
    turns = np.array([int(r["turn"]) for r in rows])
    unique_ids = sorted(set(ids))
    if len(unique_ids) != 5000 or len(set(zip(ids, turns, strict=True))) != len(rows):
        raise RuntimeError("conversation/turn coverage or uniqueness failed")
    np.random.default_rng(0).shuffle(unique_ids)
    fold_of = {cid: f for f in range(N_FOLDS) for cid in unique_ids[f::N_FOLDS]}
    fold_hash = hashlib.sha256(
        "\n".join(f"{c}:{fold_of[c]}" for c in sorted(fold_of)).encode()
    ).hexdigest()
    parent = reference(args.reference)["parts"]
    expected = parent[f"transfer_armR_own_{model}"]
    if fold_hash != expected["fold_map_sha256"]:
        raise RuntimeError("conversation folds differ from original transfer")
    counts = {str(t): int((turns == t).sum()) for t in range(1, 13)}
    if counts != parent[f"cells_armR_own_{model}"]["n_per_turn"]:
        raise RuntimeError("per-turn source coverage changed")
    x, y = np.concatenate(xs), np.concatenate(ys)
    membership = np.array([fold_of[c] for c in ids])
    log(f"[load] model={model} rows={len(rows)} folds_verified={fold_hash}")
    return dict(
        x=x,
        y=y,
        ids=np.array(ids),
        turns=turns,
        membership=membership,
        counts=counts,
        fold_hash=fold_hash,
    )


def atomic_npz(path, **arrays):
    """Persist numerical outputs without CPU-heavy per-file compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)
    return sha(path)


def source_fit(args, panel):
    """Fit or restore all six source folds as one exact-recipe batch."""
    from explore_persona_space.analysis import turn_transfer_calibration as numerical

    payload = dict(
        version=2,
        numerical_sha256=sha(numerical.__file__),
        fitting_code_sha256=hashlib.sha256(
            (inspect.getsource(load_panel) + inspect.getsource(source_fit)).encode()
        ).hexdigest(),
        reference_sha256=REFERENCE_SHA256,
        inputs_sha256=sha(args.inputs / "inputs.json"),
        model=args.model,
        source_turn=args.source_turn,
        fold_hash=panel["fold_hash"],
    )
    fp = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    name = f"{args.model}_source{args.source_turn}"
    path = args.store / "maps" / f"{name}.npz"
    receipt = args.out / "maps" / f"{name}.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["fingerprint"] != fp or sha(path) != record["map_sha256"]:
            raise RuntimeError("source-map checkpoint provenance changed")
        with np.load(path, allow_pickle=False) as z:
            fitted = numerical.BatchedRidge(**{k: z[k] for k in z.files})
        log(f"[map] reused {name}")
        return fitted, record
    selected = np.flatnonzero(panel["turns"] == args.source_turn)
    labels = panel["membership"][selected]
    indices = [np.flatnonzero(labels != fold) for fold in range(N_FOLDS)]
    started = time.monotonic()
    log(f"[map] fitting {name} rows={len(selected)} folds={N_FOLDS}")
    fitted = numerical.fit_batched_gcv(panel["x"][selected], panel["y"][selected], indices)
    seconds = time.monotonic() - started
    digest = atomic_npz(path, **vars(fitted))
    record = payload | dict(
        fingerprint=fp,
        map_sha256=digest,
        map_file=str(path),
        fit_seconds=seconds,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        lambda_by_fold=fitted.lambdas.tolist(),
        n_train=fitted.n_train.tolist(),
    )
    atomic_json(receipt, record)
    log(f"[map] complete {name} seconds={seconds:.1f} max_rss_kib={record['max_rss_kib']}")
    return fitted, record


def score_fingerprint(source_record):
    """Bind predictions to exact source-map bytes and the scoring implementation."""
    from explore_persona_space.analysis import mapping_baselines, turn_transfer_calibration

    payload = {
        "source_fingerprint": source_record["fingerprint"],
        "source_map_sha256": source_record["map_sha256"],
        "scoring_code": inspect.getsource(score_target),
        "baselines_sha256": sha(mapping_baselines.__file__),
        "numerical_sha256": sha(turn_transfer_calibration.__file__),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def score_target(args, panel, fitted, source_record, target, fold):
    """Calibrate on non-test conversations and score every method on identical rows."""
    from explore_persona_space.analysis.turn_transfer_calibration import (
        adapted_predictions,
        calibrate,
    )

    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )

    name = f"{args.model}_source{args.source_turn}_target{target}_fold{fold}"
    path = args.out / "folds" / f"{name}.json"
    arrays_path = args.store / "predictions" / f"{name}.npz"
    fp = score_fingerprint(source_record)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["fingerprint"] != fp or sha(arrays_path) != previous["prediction_sha256"]:
            raise RuntimeError(f"calibration checkpoint changed: {name}")
        return previous
    started = time.monotonic()
    target_rows = np.flatnonzero(panel["turns"] == target)
    training_mask = panel["membership"][target_rows] != fold
    training, testing = target_rows[training_mask], target_rows[~training_mask]
    source_training = np.flatnonzero(
        (panel["turns"] == args.source_turn) & (panel["membership"] != fold)
    )
    test_ids = set(panel["ids"][testing])
    if test_ids & set(panel["ids"][training]) or test_ids & set(panel["ids"][source_training]):
        raise RuntimeError("held-out conversations leaked into a fitting stage")
    # One matrix multiplication covers calibration and test rows for this turn.
    predicted = fitted.predict(fold, panel["x"][target_rows])
    calibration = calibrate(predicted[training_mask], panel["y"][training])
    raw = predicted[~training_mask]
    truth = panel["y"][testing].astype(np.float64)
    predictions = {"raw": raw} | adapted_predictions(raw, calibration)
    predictions["identity_bias"] = identity_bias_predict(
        panel["x"][training], panel["y"][training], panel["x"][testing]
    )
    sst = float(np.square(truth - truth.mean(0)).sum())
    if sst <= 0:
        raise RuntimeError("constant held-out targets")
    metrics = {}
    squared_errors = {}
    for method, prediction in predictions.items():
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"invalid {method} predictions")
        row_errors = np.square(truth - prediction).sum(1)
        squared_errors[f"sse_{method}"] = row_errors
        metrics[method] = dict(
            sse=float(row_errors.sum()),
            r2=1 - float(row_errors.sum()) / sst,
            retrieval={
                metric: knn_retrieval(prediction, truth, ks=(1,), metric=metric)
                for metric in ("cosine", "euclidean")
            },
        )
    digest = atomic_npz(
        arrays_path,
        raw_prediction=raw.astype(np.float32),
        test_indices=testing,
        calibration_indices=training,
        bias=calibration["bias"],
        gain=np.array(calibration["gain"]),
        prediction_mean=calibration["prediction_mean"],
        target_mean=calibration["target_mean"],
        **squared_errors,
    )
    record = dict(
        model=args.model,
        source_turn=args.source_turn,
        target_turn=target,
        fold=fold,
        fingerprint=fp,
        source_map_sha256=source_record["map_sha256"],
        prediction_sha256=digest,
        prediction_file=str(arrays_path),
        n_train_source=len(source_training),
        n_calibration=len(training),
        n_test=len(testing),
        test_source_overlap=0,
        test_calibration_overlap=0,
        target_labels_used_for_calibration=True,
        scale=calibration["gain"],
        bias_norm=float(np.linalg.norm(calibration["bias"])),
        sst=sst,
        metrics=metrics,
        seconds=time.monotonic() - started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    atomic_json(path, record)
    log(
        f"[cell] {name} seconds={record['seconds']:.1f} "
        f"raw={metrics['raw']['r2']:.6f} bias={metrics['bias']['r2']:.6f} "
        f"bias_scale={metrics['bias_scale']['r2']:.6f}"
    )
    return record


def fit(args):
    """Run the banked-input fit, with reusable map and per-fold result checkpoints."""
    panel = load_panel(args, args.model)
    fitted, map_record = source_fit(args, panel)
    targets = (12,) if args.phase == "pilot" or args.source_turn == 12 else tuple(range(1, 13))
    parent = reference(args.reference)["parts"][f"transfer_armR_own_{args.model}"]
    for target in targets:
        rows = [score_target(args, panel, fitted, map_record, target, f) for f in range(N_FOLDS)]
        r2 = 1 - sum(r["metrics"]["raw"]["sse"] for r in rows) / sum(r["sst"] for r in rows)
        error = abs(r2 - parent["r2"][f"{args.source_turn}->{target}"])
        if error > 1e-7:
            raise RuntimeError(
                f"parent raw-transfer parity failed: {args.model} "
                f"{args.source_turn}->{target} abs_error={error}"
            )
        log(f"[parity] {args.model} {args.source_turn}->{target} abs_error={error:.3e}")
    log(f"[fit] complete model={args.model} source={args.source_turn}")


def reduce_results(args):
    """Require the complete declared grid and retain raw-parent parity evidence."""
    parent = reference(args.reference)
    cells = []
    for model in MODELS:
        source = parent["parts"][f"transfer_armR_own_{model}"]
        for fit_turn in SOURCE_TURNS:
            map_name = f"{model}_source{fit_turn}"
            map_record = json.loads((args.out / "maps" / f"{map_name}.json").read_text())
            if (
                map_record["model"] != model
                or map_record["source_turn"] != fit_turn
                or map_record["reference_sha256"] != REFERENCE_SHA256
                or map_record["inputs_sha256"] != sha(args.inputs / "inputs.json")
                or map_record["fold_hash"] != source["fold_map_sha256"]
                or sha(map_record["map_file"]) != map_record["map_sha256"]
            ):
                raise RuntimeError(f"changed source-map provenance: {map_name}")
            expected_fp = score_fingerprint(map_record)
            for target in (12,) if fit_turn == 12 else range(1, 13):
                folds = []
                for f in range(N_FOLDS):
                    name = f"{model}_source{fit_turn}_target{target}_fold{f}"
                    row = json.loads((args.out / "folds" / f"{name}.json").read_text())
                    if (
                        (row["model"], row["source_turn"], row["target_turn"], row["fold"])
                        != (model, fit_turn, target, f)
                        or row["fingerprint"] != expected_fp
                        or row["source_map_sha256"] != map_record["map_sha256"]
                        or row["test_source_overlap"] != 0
                        or row["test_calibration_overlap"] != 0
                    ):
                        raise RuntimeError(f"changed fold provenance: {name}")
                    if sha(row["prediction_file"]) != row["prediction_sha256"]:
                        raise RuntimeError(f"changed predictions: {name}")
                    folds.append(row)
                expected_n = parent["parts"][f"cells_armR_own_{model}"]["n_per_turn"][str(target)]
                if sum(r["n_test"] for r in folds) != expected_n or any(
                    r["n_test"] + r["n_calibration"] != expected_n for r in folds
                ):
                    raise RuntimeError(f"incomplete target coverage: {model} {fit_turn}->{target}")
                own = source["r2"][f"{target}->{target}"]
                sst = sum(r["sst"] for r in folds)
                scores = {}
                for method in METHODS:
                    r2 = 1 - sum(r["metrics"][method]["sse"] for r in folds) / sst
                    retrieval = {}
                    for metric in ("cosine", "euclidean"):
                        measures = [r["metrics"][method]["retrieval"][metric] for r in folds]
                        n = sum(m["n"] for m in measures)
                        retrieval[metric] = dict(
                            top1=sum(m["acc_at_k"]["1"] * m["n"] for m in measures) / n,
                            pool_size_by_fold=[m["n_pool"] for m in measures],
                            chance_by_fold=[m["chance_at_k"]["1"] for m in measures],
                        )
                    scores[method] = dict(r2=r2, retention=r2 / own, retrieval=retrieval)
                error = abs(scores["raw"]["r2"] - source["r2"][f"{fit_turn}->{target}"])
                if error > 1e-7:
                    raise RuntimeError("parent parity failed in final reduction")
                cells.append(
                    dict(
                        model=model,
                        source_turn=fit_turn,
                        target_turn=target,
                        n_test=sum(r["n_test"] for r in folds),
                        own_turn_r2=own,
                        raw_parent_abs_error=error,
                        metrics=scores,
                        scales=[r["scale"] for r in folds],
                        calibration_n_by_fold=[r["n_calibration"] for r in folds],
                    )
                )
    result = dict(
        status="complete",
        answer_draws=1,
        layer=19,
        planned_conversations=5000,
        source_turns=[1, 3],
        anchor_turn=12,
        target_turns=list(range(1, 13)),
        models=list(MODELS),
        expected_cells=50,
        expected_fold_cells=300,
        hf_source=dict(repo=HF_REPO, revision=HF_REV, prefix=HF_PREFIX),
        reference_sha256=REFERENCE_SHA256,
        cells=cells,
        calibration="Per source/target pair, one vector bias and optional one scalar. "
        "Fitted only on target-turn non-test conversations. Source-map training "
        "and calibration may share training conversations. All test conversations "
        "are excluded from both stages. Target-informed adaptation, not zero-shot transfer.",
        r2_aggregation="1 - sum-fold SSE / sum-fold target-test-mean-centered SST",
        own_turn_reference="Uncalibrated diagonal of the pinned original transfer matrix",
        uncertainty="Point estimates; no confidence intervals fitted",
        created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        driver_sha256=sha(__file__),
    )
    atomic_json(args.out / "results.json", result)
    log(
        f"[reduce] complete cells={len(cells)} max_parent_error="
        f"{max(c['raw_parent_abs_error'] for c in cells):.3e}"
    )


def main():
    """Run explicit resumable phases, leaving unrelated workflow state intact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "validate", "pilot", "fit", "reduce"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default="instruct")
    parser.add_argument("--source-turn", type=int, choices=SOURCE_TURNS, default=1)
    args = parser.parse_args()
    if args.phase in ("pilot", "fit", "reduce") and args.store is None:
        parser.error("--store is required for numerical phases")
    if args.phase == "stage":
        stage(args)
    elif args.phase == "validate":
        load_panel(args, args.model)
        log("[validate] consumer and provenance pass")
    elif args.phase in ("pilot", "fit"):
        fit(args)
    else:
        reduce_results(args)


if __name__ == "__main__":
    main()

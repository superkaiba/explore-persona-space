"""Fit pooled early-turn maps and evaluate existing single answers at turn 12.

Same-question #825 extension: retain legacy GCV and existing six conversation
folds. Reuse verified source-1, source-3 and own-12 maps; fit source-2 and pooled
1+2 / 1+2+3 with an equivalent primal solve. No language-model inference.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue825_turn_bias_scale as parent  # noqa: E402
from issue825_turn_k5_archive import upload  # noqa: E402

from explore_persona_space.analysis import pooled_turn_transfer as numeric  # noqa: E402
from explore_persona_space.analysis import turn_transfer_calibration as calibration  # noqa: E402
from explore_persona_space.backends.artifacts import write_completion_sentinel  # noqa: E402
from explore_persona_space.orchestrate.hub import retry_transient  # noqa: E402
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SOURCES = {"1": [1], "2": [2], "3": [3], "1+2": [1, 2], "1+2+3": [1, 2, 3], "12": [12]}
METHODS = ["raw", "bias", "bias_scale", "identity_bias"]
PREFIX = "issue825_turn_pooled_single_20260915"


def configuration(args):
    """Read the committed input contract and bind reused code/reference identities."""
    config = json.loads(args.config.read_text())
    parent.reference(args.reference)
    if config["reference_sha256"] != parent.REFERENCE_SHA256:
        raise ValueError("reference contract changed")
    if any(r["numerical_sha256"] != parent.sha(calibration.__file__) for r in config["maps"]):
        raise ValueError("reused numerical recipe differs from the frozen maps")
    return config


def stage(args):
    """Stage the exact parent tensor bank and verify six archived control maps."""
    config = configuration(args)
    if not (args.inputs / "inputs.json").exists():
        parent.stage(args)
    manifest = json.loads((args.inputs / "inputs.json").read_text())
    projected = [{k: v for k, v in r.items() if k != "local"} for r in manifest["files"]]
    if projected != config["input_files"]:
        raise ValueError("staged source name/hash/size set differs from the committed contract")
    for r in manifest["files"]:
        p = Path(r["local"])
        if p.stat().st_size != r["size"] or parent.sha(p) != r["sha256"]:
            raise ValueError(f"input content changed: {r['path']}")
    args.reused.mkdir(parents=True, exist_ok=True)
    records = []
    for r in config["maps"]:
        source = Path(
            retry_transient(
                lambda r=r: hf_hub_download(
                    config["maps_repo"],
                    r["hf_path"],
                    repo_type="model",
                    revision=config["maps_revision"],
                    cache_dir=args.reused,
                ),
                what=f"stage control map {r['model']} turn {r['source_turn']}",
            )
        )
        if source.stat().st_size != r["bytes"] or parent.sha(source) != r["map_sha256"]:
            raise ValueError("reused map content changed")
        if r["inputs_sha256"] != config["original_manifest_sha256"]:
            raise ValueError("map trained on a different source bank")
        records.append(r | {"staged_path": str(source)})
    parent.atomic_json(
        args.out / "reuse_verification.json",
        {
            "status": "verified",
            "observed_at": time.time(),
            "input_files": len(projected),
            "input_bytes": manifest["bytes"],
            "config_sha256": parent.sha(args.config),
            "maps": records,
        },
    )
    parent.log("[stage] exact source bank and six control maps verified")


def fingerprint(args, panel, model, source):
    """Bind checkpoints to immutable inputs, folds and complete executed source files."""
    record = {
        "version": 1,
        "model": model,
        "source": source,
        "turns": SOURCES[source],
        "fold_hash": panel["fold_hash"],
        "config_sha256": parent.sha(args.config),
        "source_sha256": {
            Path(p).name: parent.sha(p)
            for p in [__file__, numeric.__file__, calibration.__file__, parent.__file__]
        },
        "lambda_grid": {"log10_start": -2, "log10_stop": 4, "count": 13},
        "selector": "legacy rowwise GCV",
        "weighting": "equal weight per eligible source row",
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(), record


def fitted_maps(args, panel, model, source):
    """Reuse exact controls or checkpoint new primal fits in two-fold batches."""
    fp, provenance = fingerprint(args, panel, model, source)
    if source in {"1", "3", "12"}:
        reuse = json.loads((args.out / "reuse_verification.json").read_text())
        record = next(
            r for r in reuse["maps"] if r["model"] == model and r["source_turn"] == int(source)
        )
        if record["fold_hash"] != panel["fold_hash"]:
            raise ValueError("reused map conversation folds changed")
        p = Path(record["staged_path"])
        if parent.sha(p) != record["map_sha256"]:
            raise ValueError("reused map bytes changed after staging")
        with np.load(p, allow_pickle=False) as z:
            fitted = calibration.BatchedRidge(**{k: z[k] for k in z.files})
        selected = np.flatnonzero(np.isin(panel["turns"], SOURCES[source]))
        expected = [(panel["membership"][selected] != f).sum() for f in range(6)]
        np.testing.assert_array_equal(fitted.n_train, expected)
        yield list(range(6)), fitted, fp, provenance | {"reused_sha256": record["map_sha256"]}
        return
    selected = np.flatnonzero(np.isin(panel["turns"], SOURCES[source]))
    x, y = panel["x"][selected], panel["y"][selected]
    labels = panel["membership"][selected]
    for first in range(0, 6, 2):
        folds = list(range(first, first + 2))
        name = f"{model}_source{source}_folds{first}-{first + 1}"
        path, receipt = args.store / "maps" / f"{name}.npz", args.out / "maps" / f"{name}.json"
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record["fingerprint"] != fp or parent.sha(path) != record["sha256"]:
                raise ValueError("pooled map checkpoint identity changed")
            with np.load(path, allow_pickle=False) as z:
                fitted = calibration.BatchedRidge(**{k: z[k] for k in z.files})
        else:
            before = time.monotonic()
            indices = [np.flatnonzero(labels != f) for f in folds]
            parent.log(f"[fit] {name} source_rows={len(selected)}")
            fitted = numeric.fit_primal_gcv(x, y, indices)
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = parent.atomic_npz(path, **vars(fitted))
            record = provenance | {
                "fingerprint": fp,
                "sha256": digest,
                "path": str(path),
                "elapsed_seconds": time.monotonic() - before,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "folds": folds,
                "lambda_by_fold": fitted.lambdas.tolist(),
                "n_train": fitted.n_train.tolist(),
            }
            parent.atomic_json(receipt, record)
            parent.log(f"[fit] {name} complete seconds={record['elapsed_seconds']:.1f}")
        if source == "1+2+3" and first == 0:
            enforce_venue_gate(record, args.out / f"pilot_gate_{model}.json")
        yield folds, fitted, fp, provenance


def enforce_venue_gate(record, path):
    """Recheck the measured largest-source bound on fresh and resumed checkpoints."""
    gate = {
        "model": record["model"],
        "status": "pass",
        "observed_at": time.time(),
        "fingerprint": record["fingerprint"],
        "checkpoint_sha256": record["sha256"],
        "fit_seconds": record["elapsed_seconds"],
        "chunks_projected": 18,
        "projection_seconds": record["elapsed_seconds"] * 18,
        "max_rss_kib": record["max_rss_kib"],
    }
    if gate["projection_seconds"] > 3600 or gate["max_rss_kib"] > 24 * 1024**2:
        gate["status"] = "venue_review_required"
    parent.atomic_json(path, gate)
    if gate["status"] != "pass":
        raise RuntimeError("measured CPU venue exceeds the approved projection/RSS bound")


def score(args, panel, model, source, fold, fitted, map_fold, fp):
    """Score identical target-12 rows while excluding test conversations from both fits."""
    name = f"{model}_source{source}_target12_fold{fold}"
    path, receipt = args.store / "predictions" / f"{name}.npz", args.out / "folds" / f"{name}.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["fingerprint"] != fp or parent.sha(path) != record["sha256"]:
            raise ValueError("score checkpoint identity changed")
        return
    started = time.monotonic()
    target = np.flatnonzero(panel["turns"] == 12)
    train_mask = panel["membership"][target] != fold
    train, test = target[train_mask], target[~train_mask]
    source_train = np.flatnonzero(
        np.isin(panel["turns"], SOURCES[source]) & (panel["membership"] != fold)
    )
    test_ids = set(panel["ids"][test])
    if test_ids & set(panel["ids"][train]) or test_ids & set(panel["ids"][source_train]):
        raise ValueError("test conversations leaked into map/calibration training")
    all_prediction = fitted.predict(map_fold, panel["x"][target])
    raw = all_prediction[~train_mask]
    coeff = calibration.calibrate(all_prediction[train_mask], panel["y"][train])
    adapted = calibration.adapted_predictions(raw, coeff)
    # Match the existing identity baseline's float64 training-only calculation.
    offset = (panel["y"][train].astype(np.float64) - panel["x"][train].astype(np.float64)).mean(0)
    identity = panel["x"][test].astype(np.float64) + offset
    prediction = np.stack([raw, adapted["bias"], adapted["bias_scale"], identity])
    truth = panel["y"][test].astype(np.float64)
    metrics = numeric.row_metrics(prediction, truth)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = parent.atomic_npz(
        path,
        raw_prediction=raw,
        truth=truth,
        test_ids=panel["ids"][test],
        test_indices=test,
        calibration_indices=train,
        source_training_indices=source_train,
        gain=np.array(coeff["gain"]),
        bias=coeff["bias"],
        prediction_mean=coeff["prediction_mean"],
        target_mean=coeff["target_mean"],
        identity_bias=offset,
        **metrics,
    )
    record = {
        "model": model,
        "source": source,
        "fold": fold,
        "fingerprint": fp,
        "sha256": digest,
        "path": str(path),
        "n_test": len(test),
        "n_train_source": len(source_train),
        "n_calibration": len(train),
        "n_train_source_conversations": len(set(panel["ids"][source_train])),
        "lambda": float(fitted.lambdas[map_fold]),
        "scale": coeff["gain"],
        "test_training_overlap": 0,
        "sse": metrics["sse"].sum(1).tolist(),
        "sst": float(metrics["sst"].sum()),
        "elapsed_seconds": time.monotonic() - started,
    }
    parent.atomic_json(receipt, record)
    parent.log(f"[score] {name} complete n={len(test)} seconds={record['elapsed_seconds']:.1f}")


def run_model(args, model):
    """Load one model, fit largest pooled source first, then produce all fixed comparisons."""
    panel = parent.load_panel(args, model)
    gate_path = args.out / f"primal_equivalence_{model}.json"
    gate_fp, _ = fingerprint(args, panel, model, "1")
    if gate_path.exists():
        prior = json.loads(gate_path.read_text())
        if prior["status"] != "pass" or prior["fingerprint"] != gate_fp:
            raise ValueError("real-data solver equivalence checkpoint changed")
    else:
        parent.log(f"[equivalence] {model} source1 first two full-size folds")
        before = time.monotonic()
        selected = np.flatnonzero(panel["turns"] == 1)
        indices = [np.flatnonzero(panel["membership"][selected] != f) for f in range(2)]
        primal = numeric.fit_primal_gcv(panel["x"][selected], panel["y"][selected], indices)
        _, control, _, _ = next(fitted_maps(args, panel, model, "1"))
        np.testing.assert_array_equal(primal.lambdas, control.lambdas[:2])
        target = np.flatnonzero(panel["turns"] == 12)
        errors = []
        for fold in range(2):
            testing = target[panel["membership"][target] == fold]
            left = primal.predict(fold, panel["x"][testing])
            right = control.predict(fold, panel["x"][testing])
            error = float(np.abs(left - right).max())
            if error > 1e-6:
                raise ValueError(f"real-data primal/dual prediction disagreement: {error}")
            errors.append(error)
        parent.atomic_json(
            gate_path,
            {
                "status": "pass",
                "fingerprint": gate_fp,
                "max_prediction_errors": errors,
                "selected_lambdas": primal.lambdas.tolist(),
                "elapsed_seconds": time.monotonic() - before,
                "observed_at": time.time(),
            },
        )
        del primal, control
        parent.log(f"[equivalence] {model} pass max_error={max(errors):.3g}")
    coverage = {
        "model": model,
        "n_by_turn": panel["counts"],
        "fold_hash": panel["fold_hash"],
        "source_rows": {s: int(np.isin(panel["turns"], t).sum()) for s, t in SOURCES.items()},
    }
    parent.atomic_json(args.out / f"coverage_{model}.json", coverage)
    for source in ["1+2+3", "1+2", "2", "1", "3", "12"]:
        for folds, fitted, fp, _ in fitted_maps(args, panel, model, source):
            for local_fold, fold in enumerate(folds):
                score(args, panel, model, source, fold, fitted, local_fold, fp)
            del fitted


def summarize(args):
    """Use paired conversation bootstraps conditional on fitted maps and the answer bank."""
    results = {
        "status": "complete",
        "followup_label": "turn-pooled-single-20260915",
        "answer_draws": 1,
        "target_turn": 12,
        "sources": SOURCES,
        "methods": METHODS,
        "selector": "legacy rowwise GCV; final tests hold out entire conversations",
        "weighting": "equal weight per eligible source row",
        "models": {},
        "bootstrap": {
            "replicates": 1000,
            "seed": 0,
            "refits": False,
            "conditional_on": ["fitted maps", "captured single-answer bank"],
            "sst_center": "fixed original held-out-fold mean",
        },
    }
    reference = parent.reference(args.reference)
    for model in parent.MODELS:
        all_rows = []
        ids = None
        for source in SOURCES:
            chunks = []
            expected_fold_hash = reference["parts"][f"transfer_armR_own_{model}"]["fold_map_sha256"]
            expected_fp, _ = fingerprint(args, {"fold_hash": expected_fold_hash}, model, source)
            for fold in range(6):
                receipt = args.out / "folds" / f"{model}_source{source}_target12_fold{fold}.json"
                r = json.loads(receipt.read_text())
                if (
                    r["fingerprint"] != expected_fp
                    or r["model"] != model
                    or r["source"] != source
                    or r["fold"] != fold
                ):
                    raise ValueError("reduction checkpoint recipe/model/source/fold changed")
                if parent.sha(r["path"]) != r["sha256"]:
                    raise ValueError("prediction content changed before reduction")
                with np.load(r["path"], allow_pickle=False) as z:
                    if len(z["test_ids"]) != r["n_test"]:
                        raise ValueError("prediction row count differs from receipt")
                    chunks.append(
                        {k: z[k] for k in ["test_ids", "sse", "sst", "cosine_hit", "euclidean_hit"]}
                    )
            row_ids = np.concatenate([c["test_ids"] for c in chunks])
            expected_n = reference["parts"][f"cells_armR_own_{model}"]["n_per_turn"]["12"]
            if len(row_ids) != expected_n:
                raise ValueError("turn-12 held-out coverage differs from the frozen parent")
            if len(set(row_ids)) != len(row_ids):
                raise ValueError("duplicate held-out conversations")
            if ids is not None:
                np.testing.assert_array_equal(ids, row_ids)
            ids = row_ids
            all_rows.append(
                {
                    k: np.concatenate([c[k] for c in chunks], axis=-1)
                    for k in ["sse", "sst", "cosine_hit", "euclidean_hit"]
                }
            )
        n = len(ids)
        sse = np.stack([r["sse"] for r in all_rows])
        sst = all_rows[0]["sst"]
        for r in all_rows:
            np.testing.assert_array_equal(r["sst"], sst)
        weights = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=1000)
        point = {"r2": 1 - sse.sum(-1) / sst.sum()}
        boot = {
            "r2": 1
            - np.einsum("bn,smn->bsm", weights, sse, optimize=True) / (weights @ sst)[:, None, None]
        }
        for label in ["cosine", "euclidean"]:
            hits = np.stack([r[f"{label}_hit"] for r in all_rows])
            point[label] = hits.mean(-1)
            boot[label] = np.einsum("bn,smn->bsm", weights, hits, optimize=True) / n
        sources = list(SOURCES)
        own = sources.index("12")
        if np.quantile(boot["r2"][:, own, 0], 0.025) <= 0:
            raise ValueError("own-turn R2 not positive away from zero; ratio needs review")
        cells = []
        for s, source in enumerate(sources):
            if source in {"1", "2", "3", "12"}:
                anchor = reference["parts"][f"transfer_armR_own_{model}"]["r2"][f"{source}->12"]
                if abs(point["r2"][s, 0] - anchor) > 1e-6:
                    raise ValueError(f"raw single-turn anchor mismatch: {model}/{source}")
            for m, method in enumerate(METHODS):
                cell = {"source": source, "method": method, "n_test": n}
                for metric in point:
                    cell[metric] = float(point[metric][s, m])
                    cell[f"{metric}_ci95"] = np.quantile(
                        boot[metric][:, s, m], [0.025, 0.975]
                    ).tolist()
                cell["retention"] = float(point["r2"][s, m] / point["r2"][own, 0])
                cell["retention_ci95"] = np.quantile(
                    boot["r2"][:, s, m] / boot["r2"][:, own, 0], [0.025, 0.975]
                ).tolist()
                cells.append(cell)
        comparisons = []
        for left, right in [
            ("1+2", "1"),
            ("1+2", "2"),
            ("1+2+3", "1+2"),
            ("1+2+3", "3"),
            ("1+2", "3"),
            ("1+2+3", "12"),
        ]:
            a, b = sources.index(left), sources.index(right)
            for m, method in enumerate(METHODS):
                row = {"left": left, "right": right, "method": method}
                for metric in point:
                    row[metric] = float(point[metric][a, m] - point[metric][b, m])
                    row[f"{metric}_ci95"] = np.quantile(
                        boot[metric][:, a, m] - boot[metric][:, b, m], [0.025, 0.975]
                    ).tolist()
                comparisons.append(row)
        pools = [
            json.loads((args.out / "folds" / f"{model}_source1_target12_fold{f}.json").read_text())[
                "n_test"
            ]
            for f in range(6)
        ]
        results["models"][model] = {
            "cells": cells,
            "paired_comparisons": comparisons,
            "n_test": n,
            "retrieval_pool_sizes": pools,
            "retrieval_chance": 6 / n,
            "coverage": json.loads((args.out / f"coverage_{model}.json").read_text()),
        }
        parent.atomic_npz(
            args.store / f"{model}_oof.npz",
            ids=ids,
            sse=sse,
            sst=sst,
            cosine_hit=np.stack([r["cosine_hit"] for r in all_rows]),
            euclidean_hit=np.stack([r["euclidean_hit"] for r in all_rows]),
        )
    results["finished_at"] = time.time()
    parent.atomic_json(args.out / "results.json", results)
    parent.log("[reduce] all 48 cells and eight single-turn raw anchors verified")
    return results


def main():
    """Run the bounded staged CPU analysis, then archive before completion signalling."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["stage", "model", "reduce", "run"])
    p.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/analysis/issue825_turn_pooled_single_inputs.json",
    )
    p.add_argument(
        "--reference", type=Path, default=REPO / "eval_results/issue_825/turn_dynamics/results.json"
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", choices=parent.MODELS)
    args = p.parse_args()
    args.inputs, args.reused = args.root / "inputs", args.root / "reused"
    args.out, args.store = args.root / "analysis", args.root / "store"
    args.out.mkdir(parents=True, exist_ok=True)
    args.store.mkdir(parents=True, exist_ok=True)
    configuration(args)
    # Allow for remaining maps/predictions and staging on the actual output mount.
    # Completed score/map units reduce the bound; no-pending runs need no new store.
    pending_folds = 72 - len(list((args.out / "folds").glob("*.json")))
    pending_maps = 18 - len(list((args.out / "maps").glob("*.json")))
    pending_stage = not (args.out / "reuse_verification.json").exists()
    needed = 0.045 * max(0, pending_folds) + 0.22 * max(0, pending_maps)
    if pending_stage:
        needed += 6
    if needed:
        assert_out_root_headroom(args.root, need_gb=1.5 * needed + 1, phase="pooled-single")
    if args.phase in {"stage", "run"}:
        parent.log("[phase=pooled_stage] verify pinned inputs and reused controls")
        stage(args)
    if args.phase == "model":
        if not args.model:
            p.error("--model is required for the model phase")
        run_model(args, args.model)
    if args.phase == "run":
        parent.log("[phase=pooled_fit] production-shape equivalence and pooled fits")
        for model in parent.MODELS:
            run_model(args, model)
    if args.phase in {"reduce", "run"}:
        parent.log("[phase=pooled_reduce] verify and reduce held-out prediction banks")
        result = summarize(args)
        import wandb

        with wandb.init(
            project="explore-persona-space",
            name="issue825-pooled-single-turn12",
            config={"source_turns": SOURCES, "answer_draws": 1, "selector": result["selector"]},
        ) as run:
            run.log(
                {
                    f"{model}/{c['source']}/{c['method']}/r2": c["r2"]
                    for model, r in result["models"].items()
                    for c in r["cells"]
                }
            )
            parent.atomic_json(args.out / "wandb.json", {"url": run.url, "id": run.id})
    if args.phase == "run":
        parent.log("[phase=pooled_archive] persist and verify every numerical/text artifact")
        upload(args.store, PREFIX + "/numerical", "tensors", args.root / "tensor_receipt.json")
        upload(args.out, PREFIX + "/analysis", "text", args.root / "text_receipt.json")
        completion = {
            "status": "complete",
            "followup_label": "turn-pooled-single-20260915",
            "results_sha256": parent.sha(args.out / "results.json"),
            "completed_at": time.time(),
            "archives": {
                k: json.loads((args.root / f"{k}_receipt.json").read_text())
                for k in ["tensor", "text"]
            },
        }
        write_completion_sentinel(
            sentinel_path=args.root / "complete.json", issue=825, extra=completion
        )
        if (
            json.loads((args.root / "complete.json").read_text())["results_sha256"]
            != completion["results_sha256"]
        ):
            raise RuntimeError("completion sentinel read-back failed")
        if os.environ.get("EPS_SENTINEL_PATH"):
            write_completion_sentinel(
                sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=825, extra=completion
            )
        note = {
            "followup_label": "turn-pooled-single-20260915",
            "status": "complete",
            "results_sha256": completion["results_sha256"],
            "archives": {
                k: {key: value for key, value in record.items() if key != "files"}
                for k, record in completion["archives"].items()
            },
        }
        envelope = {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "task_id": 825,
            "gate": "results",
            "blocks_pipeline": False,
            "by": "issue825-pooled-single",
            "note": json.dumps(note),
        }
        parent.atomic_json(
            Path("/workspace/logs") / f"issue-825-epm_results-{time.time_ns()}.json", envelope
        )
        parent.log("[phase=done] analysis and exact archives verified")


if __name__ == "__main__":
    main()

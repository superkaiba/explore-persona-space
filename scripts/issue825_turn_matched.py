"""Matched-row and matched-conversation early-turn transfer on the pinned K1 bank."""

# ruff: noqa: E402
from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue825_turn_bias_scale as parent
from issue825_turn_k5_archive import upload

from explore_persona_space.analysis import pooled_turn_transfer as numeric
from explore_persona_space.analysis import turn_transfer_calibration as calibration
from explore_persona_space.backends.artifacts import write_completion_sentinel
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ("1", "2", "3", "mix0", "mix1", "mix2")
SOURCES = ("1", "2", "3", "1+2+3")
PREFIX = "issue825_turn_matched_20260915"


def assignments(ids, labels):
    """Counterbalance turns within original folds, with training imbalance at most two."""
    rng = np.random.default_rng(0)
    base = np.empty(len(ids), dtype=np.int64)
    offset = 0
    for fold in range(6):
        rows = np.flatnonzero(labels == fold)
        rng.shuffle(rows)
        base[rows] = (np.arange(len(rows)) + offset) % 3
        offset += len(rows)
    rotated = (base[None, :] + np.arange(3)[:, None]) % 3 + 1
    for fold in range(6):
        for rotation in rotated:
            counts = np.bincount(rotation[labels != fold], minlength=4)[1:]
            if np.ptp(counts) > 2:
                raise ValueError("unbalanced training turn assignment")
    return rotated


def selected_panel(args, model):
    """Retain exact complete-panel IDs while preserving all original fold labels."""
    panel = parent.load_panel(args, model)
    contract = json.loads(args.config.read_text())
    ids = np.array(contract["cohort_ids"])
    lookup = {
        (c, int(t)): i for i, (c, t) in enumerate(zip(panel["ids"], panel["turns"], strict=True))
    }
    if not len(ids) or len(set(ids)) != len(ids) or list(ids) != sorted(ids):
        raise ValueError("invalid frozen cohort")
    indices = np.array([[lookup[(c, t)] for c in ids] for t in range(1, 13)])
    labels = panel["membership"][indices[0]]
    np.testing.assert_array_equal(labels, contract["fold_labels"])
    if panel["fold_hash"] != contract["fold_hash"]:
        raise ValueError("original fold identity mismatch")
    np.testing.assert_array_equal(
        panel["membership"][indices], np.broadcast_to(labels, indices.shape)
    )
    turns = assignments(ids, labels)
    return dict(
        x=panel["x"][indices],
        y=panel["y"][indices],
        ids=ids,
        labels=labels,
        assignments=turns,
        fold_hash=panel["fold_hash"],
        original_counts=panel["counts"],
    )


def provenance(args, panel, model):
    """Bind checkpoints to cohort, bank, recipe, assignments and executed code bytes."""
    record = dict(
        model=model,
        config_sha256=parent.sha(args.config),
        fold_hash=panel["fold_hash"],
        assignment_sha256=hashlib.sha256(panel["assignments"].astype("<i8").tobytes()).hexdigest(),
        code={
            Path(p).name: parent.sha(p)
            for p in (__file__, parent.__file__, numeric.__file__, calibration.__file__)
        },
        lambda_grid=["logspace", -2, 4, 13],
        selector="legacy rowwise GCV",
        answer_draws=1,
    )
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(), record


def stage(args):
    """Validate every staged input against the frozen parent inventory."""
    if not (args.inputs / "inputs.json").exists():
        parent.stage(args)
    manifest = json.loads((args.inputs / "inputs.json").read_text())
    contract = json.loads(args.config.read_text())
    projected = [{k: v for k, v in r.items() if k != "local"} for r in manifest["files"]]
    if projected != contract["input_files"]:
        raise ValueError("input inventory differs from contract")
    for r in manifest["files"]:
        if Path(r["local"]).stat().st_size != r["size"] or parent.sha(r["local"]) != r["sha256"]:
            raise ValueError(f"input mismatch: {r['path']}")
    parent.atomic_json(
        args.out / "stage.json",
        dict(
            verified_at=time.time(),
            files=len(projected),
            bytes=manifest["bytes"],
            config_sha256=parent.sha(args.config),
        ),
    )


def source_rows(panel, variant):
    """Select exactly one source pair per conversation, with no pooling ensemble."""
    turns = (
        panel["assignments"][int(variant[-1])]
        if variant.startswith("mix")
        else np.full(len(panel["ids"]), int(variant))
    )
    rows = np.arange(len(turns))
    return panel["x"][turns - 1, rows], panel["y"][turns - 1, rows], turns


def checked_npz(path, receipt, fingerprint):
    """Resume only a content-verified checkpoint in the exact current regime."""
    record = json.loads(receipt.read_text())
    if record["fingerprint"] != fingerprint or parent.sha(path) != record["sha256"]:
        raise ValueError(f"checkpoint mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, record


def venue_gate(record, path):
    """Enforce production measurements on both fresh and resumed first chunks."""
    projection = 36 * record["elapsed_seconds"]
    ok = projection <= 5400 and record["max_rss_kib"] <= 24 * 1024**2
    parent.atomic_json(
        path,
        dict(
            status="pass" if ok else "halt",
            projection_seconds=projection,
            observed_at=time.time(),
            basis=record,
        ),
    )
    if not ok:
        raise RuntimeError("measured CPU projection/RSS exceeds approved gate")


def score_fold(args, panel, model, variant, fold, fitted, j, fingerprint):
    """Predict heldout conversations at all turns, with a source-trained copy baseline."""
    name = f"{model}_{variant}_fold{fold}"
    path, receipt = args.store / "predictions" / f"{name}.npz", args.out / "scores" / f"{name}.json"
    if receipt.exists():
        saved, record = checked_npz(path, receipt, fingerprint)
        if (record["model"], record["variant"], record["fold"]) != (model, variant, fold):
            raise ValueError("score unit identity mismatch")
        np.testing.assert_array_equal(saved["ids"], panel["ids"][panel["labels"] == fold])
        return record
    started = time.monotonic()
    train = np.flatnonzero(panel["labels"] != fold)
    test = np.flatnonzero(panel["labels"] == fold)
    if set(panel["ids"][train]) & set(panel["ids"][test]):
        raise ValueError("conversation leakage")
    x, y, turns = source_rows(panel, variant)
    if int(fitted.n_train[j]) != len(train):
        raise ValueError("training N mismatch")
    bias = y[train].astype(np.float64).mean(0) - x[train].astype(np.float64).mean(0)
    target_x, truth = panel["x"][:, test], panel["y"][:, test].astype(np.float64)
    raw = fitted.predict(j, target_x.reshape(-1, target_x.shape[-1])).reshape(truth.shape)
    metrics = [
        numeric.row_metrics(np.stack([raw[t], target_x[t] + bias]), truth[t]) for t in range(12)
    ]
    arrays = {k: np.stack([m[k] for m in metrics], axis=1 if k != "sst" else 0) for k in metrics[0]}
    digest = parent.atomic_npz(path, ids=panel["ids"][test], raw=raw, source_bias=bias, **arrays)
    record = dict(
        fingerprint=fingerprint,
        sha256=digest,
        model=model,
        variant=variant,
        fold=fold,
        n_train=len(train),
        n_test=len(test),
        train_turn_counts=np.bincount(turns[train], minlength=4)[1:].tolist(),
        elapsed_seconds=time.monotonic() - started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    parent.atomic_json(receipt, record)
    parent.log(f"[score] {name} targets=12 complete seconds={record['elapsed_seconds']:.1f}")
    return record


def run_model(args, model):
    """Fit independent two-fold batches once, reusing each map across all target turns."""
    panel = selected_panel(args, model)
    fp, record = provenance(args, panel, model)
    parent.atomic_json(
        args.out / f"coverage_{model}.json",
        record
        | dict(
            ids=panel["ids"].tolist(),
            original_counts=panel["original_counts"],
            fold_sizes=np.bincount(panel["labels"]).tolist(),
            assignments=panel["assignments"].tolist(),
            methods=["raw", "source_identity_bias"],
        ),
    )
    base_fp = fp
    for variant in VARIANTS:
        fp = hashlib.sha256(f"{base_fp}:{variant}".encode()).hexdigest()
        x, y, _ = source_rows(panel, variant)
        for first in range(0, 6, 2):
            started = time.monotonic()
            folds = [first, first + 1]
            name = f"{model}_{variant}_folds{first}-{first + 1}"
            path, receipt = args.store / "maps" / f"{name}.npz", args.out / "maps" / f"{name}.json"
            if receipt.exists():
                arrays, fit_record = checked_npz(path, receipt, fp)
                if (fit_record["model"], fit_record["variant"], fit_record["folds"]) != (
                    model,
                    variant,
                    folds,
                ):
                    raise ValueError("map unit identity mismatch")
                fitted = calibration.BatchedRidge(**arrays)
            else:
                parent.log(f"[fit] {name} start")
                fitted = numeric.fit_primal_gcv(
                    x, y, [np.flatnonzero(panel["labels"] != f) for f in folds]
                )
                digest = parent.atomic_npz(path, **vars(fitted))
                fit_record = dict(
                    model=model,
                    variant=variant,
                    folds=folds,
                    fingerprint=fp,
                    sha256=digest,
                    n_train=fitted.n_train.tolist(),
                    lambdas=fitted.lambdas.tolist(),
                    elapsed_seconds=time.monotonic() - started,
                    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
                parent.atomic_json(receipt, fit_record)
                parent.log(f"[fit] {name} complete seconds={fit_record['elapsed_seconds']:.1f}")
            if first == 0 and variant in ("1", "mix0"):
                venue_gate(fit_record, args.out / f"fit_gate_{model}_{variant}.json")
            score_records = [
                score_fold(args, panel, model, variant, f, fitted, j, fp)
                for j, f in enumerate(folds)
            ]
            if first == 0 and variant in ("1", "mix0"):
                combined = dict(fit_record)
                combined["elapsed_seconds"] += sum(r["elapsed_seconds"] for r in score_records)
                combined["max_rss_kib"] = max(
                    [fit_record["max_rss_kib"]] + [r["max_rss_kib"] for r in score_records]
                )
                venue_gate(combined, args.out / f"combined_gate_{model}_{variant}.json")
            del fitted


def collapse_rotations(values):
    """Average losses or hits across counterbalanced fits, never average predictions."""
    if values.shape[0] != 6:
        raise ValueError("expected three single sources and three pooled rotations")
    return np.concatenate([values[:3], values[3:].mean(axis=0, keepdims=True)], axis=0)


def reduce(args):
    """Reduce paired OOF quantities with shared conversation bootstrap weights."""
    contract = json.loads(args.config.read_text())
    ids = np.array(contract["cohort_ids"])
    n = len(ids)
    weights = (
        np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=1000).astype(np.float64)
    )
    result = dict(
        answer_draws=1,
        n_conversations=n,
        source_conditions=SOURCES,
        pooled_definition="mean performance of three counterbalanced matched-N maps; no prediction ensemble",
        uncertainty="1000 paired conversation bootstraps, seed 0; pointwise, conditional on maps, assignments and K1 bank",
        target_turns=list(range(1, 13)),
        models={},
    )
    for model in parent.MODELS:
        coverage = json.loads((args.out / f"coverage_{model}.json").read_text())
        labels = np.array(contract["fold_labels"])
        frozen_panel = dict(assignments=assignments(ids, labels), fold_hash=contract["fold_hash"])
        base_fp, expected = provenance(args, frozen_panel, model)
        if any(coverage[k] != v for k, v in expected.items()):
            raise ValueError("reduction identity differs from fitted regime")
        np.testing.assert_array_equal(coverage["ids"], ids)
        np.testing.assert_array_equal(coverage["assignments"], frozen_panel["assignments"])
        np.testing.assert_array_equal(coverage["fold_sizes"], np.bincount(labels, minlength=6))
        quantities = {k: [] for k in ("sse", "cosine_hit", "euclidean_hit")}
        common_sst = None
        for variant in VARIANTS:
            fp = hashlib.sha256(f"{base_fp}:{variant}".encode()).hexdigest()
            blocks, all_ids = [], []
            for fold in range(6):
                name = f"{model}_{variant}_fold{fold}"
                block, rec = checked_npz(
                    args.store / "predictions" / f"{name}.npz",
                    args.out / "scores" / f"{name}.json",
                    fp,
                )
                if rec["model"] != model or rec["variant"] != variant or rec["fold"] != fold:
                    raise ValueError("score identity mismatch")
                np.testing.assert_array_equal(block["ids"], ids[labels == fold])
                if rec["n_train"] != int((labels != fold).sum()) or rec["n_test"] != int(
                    (labels == fold).sum()
                ):
                    raise ValueError("score counts mismatch")
                if block["raw"].shape != (12, rec["n_test"], 3584):
                    raise ValueError("prediction shape mismatch")
                # Raw predictions have already been scored; retain only small row metrics here.
                del block["raw"]
                blocks.append(block)
                all_ids.extend(block["ids"])
            order = np.argsort(all_ids)
            np.testing.assert_array_equal(np.array(all_ids)[order], ids)
            sst = np.concatenate([b["sst"] for b in blocks], axis=-1)[:, order]
            if common_sst is not None:
                np.testing.assert_array_equal(sst, common_sst)
            common_sst = sst
            for key in quantities:
                quantities[key].append(
                    np.concatenate([b[key] for b in blocks], axis=-1)[..., order]
                )
        rows = {k: np.stack(v) for k, v in quantities.items()}
        parent.atomic_npz(args.store / f"{model}_oof.npz", ids=ids, sst=common_sst, **rows)
        denominator = common_sst.sum(-1)
        boot_den = np.einsum("bn,tn->bt", weights, common_sst)
        if np.any(boot_den <= 0):
            raise ValueError("nonpositive SST")
        estimates, boot = {}, {}
        for key, raw_values in rows.items():
            values = collapse_rotations(raw_values)
            totals = values.sum(-1)
            sampled = np.einsum("bn,amtn->bamt", weights, values, optimize=True)
            metric = "r2" if key == "sse" else key.replace("_hit", "_top1")
            estimates[metric] = 1 - totals / denominator if key == "sse" else totals / n
            boot[metric] = 1 - sampled / boot_den[:, None, None, :] if key == "sse" else sampled / n
        cells = []
        for a, source in enumerate(SOURCES):
            for m, method in enumerate(("raw", "source_identity_bias")):
                for t in range(12):
                    cell = dict(source=source, method=method, target_turn=t + 1)
                    for metric in estimates:
                        cell[metric] = float(estimates[metric][a, m, t])
                        cell[metric + "_ci95"] = np.quantile(
                            boot[metric][:, a, m, t], [0.025, 0.975]
                        ).tolist()
                    cells.append(cell)
        comparisons = []
        for targets, label in (([11], "turn12"), (list(range(3, 12)), "mean_turns4_12")):
            delta = estimates["r2"][3, 0, targets].mean() - estimates["r2"][2, 0, targets].mean()
            sampled = (boot["r2"][:, 3, 0][:, targets] - boot["r2"][:, 2, 0][:, targets]).mean(-1)
            comparisons.append(
                dict(
                    contrast="pooled123_minus_source3",
                    target_summary=label,
                    delta_r2=float(delta),
                    ci95=np.quantile(sampled, [0.025, 0.975]).tolist(),
                )
            )
        rotation_r2 = 1 - rows["sse"][3:, 0].sum(-1) / denominator
        result["models"][model] = dict(
            cells=cells,
            comparisons=comparisons,
            rotation_r2=rotation_r2.tolist(),
            retrieval_pool_sizes=coverage["fold_sizes"],
            retrieval_chance=6 / n,
        )
    result["finished_at"] = time.time()
    parent.atomic_json(args.out / "results.json", result)
    return result


def main():
    """Run pinned analysis, archive before optional tracking, and signal verified completion."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["stage", "model", "reduce", "run"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--config", type=Path, default=REPO / "configs/analysis/issue825_turn_matched_inputs.json"
    )
    p.add_argument(
        "--reference", type=Path, default=REPO / "eval_results/issue_825/turn_dynamics/results.json"
    )
    p.add_argument("--model", choices=parent.MODELS)
    args = p.parse_args()
    args.inputs, args.out, args.store = (
        args.root / "inputs",
        args.root / "analysis",
        args.root / "store",
    )
    for path in (args.out, args.store / "maps", args.store / "predictions"):
        path.mkdir(parents=True, exist_ok=True)
    parent.reference(args.reference)
    if args.phase in ("stage", "run"):
        pending_maps = max(0, 36 - len(list((args.out / "maps").glob("*.json"))))
        pending_scores = max(0, 72 - len(list((args.out / "scores").glob("*.json"))))
        need = 1.5 * (2 + 0.21 * pending_maps + 0.29 * pending_scores) + 1
        assert_out_root_headroom(args.root, need_gb=need, phase="matched-turn-transfer")
        parent.log("[phase=matched_stage] verify pinned single-answer bank")
        stage(args)
    if args.phase in ("model", "run"):
        parent.log("[phase=matched_fit] matched source fits and all-turn scoring")
        if args.phase == "model" and args.model is None:
            p.error("--model required")
        for model in parent.MODELS if args.phase == "run" else [args.model]:
            run_model(args, model)
    if args.phase in ("reduce", "run"):
        parent.log("[phase=matched_reduce] paired heldout aggregation")
        reduce(args)
    if args.phase == "run":
        parent.log("[phase=matched_archive] numerical outputs before optional tracking")
        upload(args.store, PREFIX + "/numerical", "tensors", args.root / "tensor_receipt.json")
        import wandb

        with wandb.init(
            project="explore-persona-space",
            name="issue825-matched-turn-transfer",
            mode="offline",
            dir=str(args.root),
            config=dict(answer_draws=1, conditions=SOURCES),
        ) as run:
            results = json.loads((args.out / "results.json").read_text())
            run.log(
                {
                    f"{model}/{c['source']}/turn{c['target_turn']}/r2": c["r2"]
                    for model, r in results["models"].items()
                    for c in r["cells"]
                    if c["method"] == "raw"
                }
            )
            parent.atomic_json(args.out / "tracking.json", dict(mode="offline", id=run.id))
        upload(args.out, PREFIX + "/analysis", "text", args.root / "text_receipt.json")
        completion = dict(
            status="complete",
            completed_at=time.time(),
            results_sha256=parent.sha(args.out / "results.json"),
            archives={
                k: json.loads((args.root / f"{k}_receipt.json").read_text())
                for k in ("tensor", "text")
            },
        )
        write_completion_sentinel(
            sentinel_path=args.root / "complete.json", issue=825, extra=completion
        )
        if (
            json.loads((args.root / "complete.json").read_text())["results_sha256"]
            != completion["results_sha256"]
        ):
            raise RuntimeError("sentinel readback failed")
        if os.environ.get("EPS_SENTINEL_PATH"):
            write_completion_sentinel(
                sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=825, extra=completion
            )
        parent.log("[phase=done] complete and archives verified")


if __name__ == "__main__":
    main()

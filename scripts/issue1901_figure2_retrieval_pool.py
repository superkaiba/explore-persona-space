#!/usr/bin/env python3
"""Rescore the nine banked Figure 2 training rungs on a larger candidate pool.

The only changed variable is the number of distinct retrieval candidates.
Queries, predictions, homogeneous five-rollout means, training-only whitening,
two-sided CSLS K=10, and source-vector deduplication follow the pinned #1901
analysis. No generations or fits are performed. The separate 1,200-row extension
has no persisted predictions and is explicitly outside this banked sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

import issue1901_figure2_five_rollout_scaling as FIVE  # noqa: E402
import issue1901_singleturn_retrieval_final as FINAL  # noqa: E402

LOG = logging.getLogger("issue1901_retrieval_pool")
EXTRA_FILES = {
    key: FINAL.HF_FILES[key] for key in ("distractors", *(f"distr_draws_{i}" for i in range(4)))
}


def exact_pool_view(source: np.ndarray, n_query: int, n_pool: int) -> FINAL.EvalView:
    """Keep every distinct query target and fill a bank-ordered, exact-size pool."""
    queries = FINAL.make_eval_view(source[:n_query], n_query, "keep_one")
    full = FINAL.make_eval_view(source, n_query, "keep_one")
    if not len(queries.pool_rows) <= n_pool <= len(full.pool_rows):
        raise ValueError(
            f"pool size {n_pool} outside [{len(queries.pool_rows)}, {len(full.pool_rows)}]"
        )
    stop = int(full.pool_rows[n_pool - 1]) + 1
    view = FINAL.make_eval_view(source[:stop], n_query, "keep_one")
    np.testing.assert_array_equal(view.pred_rows, queries.pred_rows)
    if len(view.pool_rows) != n_pool:
        raise RuntimeError(f"wrong realized pool: {view.diagnostics}")
    return view


def stage_inputs(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, str]]:
    """Resolve pinned inputs and verify bytes against their producing result JSONs."""
    reference = json.loads(
        (ROOT / "eval_results/issue_1901/figure2_five_rollout_scaling.json").read_text()
    )
    final = json.loads(
        (ROOT / "eval_results/issue_1901/singleturn_retrieval_final/summary.json").read_text()
    )
    if reference["data_revision"] != FIVE.REVISION or final["data_revision"] != FIVE.REVISION:
        raise RuntimeError("reference JSONs do not use the pinned data revision")
    files = dict(FIVE.BASE_FILES)
    expected = dict(reference["source_sha256"])
    for n_train in FIVE.TRAIN_SIZES:
        for predictor in FIVE.PREDICTORS:
            key = f"pred_{predictor}_{n_train}"
            files[key] = FIVE._prediction_path(n_train, predictor)
            expected[key] = reference["per_n"][str(n_train)][predictor]["prediction_sha256"]
    files.update(EXTRA_FILES)
    expected.update({key: final["input_sha256"][key] for key in EXTRA_FILES})
    paths, hashes = {}, {}
    for key, filename in files.items():
        if args.source_stage is not None and key not in EXTRA_FILES:
            path = args.source_stage / filename
            if not path.is_file():
                raise FileNotFoundError(path)
        else:
            path = Path(
                hub.retry_transient(
                    lambda: hf_hub_download(
                        repo_id=FIVE.C.HF_DATA_REPO,
                        repo_type="dataset",
                        revision=FIVE.REVISION,
                        filename=filename,
                        local_dir=args.stage_root,
                    ),
                    what=f"retrieval_pool_download({filename})",
                )
            )
        digest = FIVE._sha256(path)
        if digest != expected[key]:
            raise RuntimeError(f"source hash mismatch: {filename}")
        paths[key], hashes[key] = path, digest
        LOG.info("[input] verified %s", key)
    return paths, hashes


def load_pool(paths: dict[str, Path], n_pool: int) -> tuple:
    """Align banked draws by capture ID and construct a homogeneous mean pool."""
    original, target, test_rows = FIVE._load_five_rollout_target(
        paths["pass_b"], paths["test_draws"]
    )
    with np.load(paths["distractors"], allow_pickle=False) as bank:
        distractors = np.asarray(bank["vx"][: FINAL.N_DISTR_MAX], dtype=np.float32).copy()
        ids = np.asarray(bank["ci"][: FINAL.N_DISTR_MAX], dtype=np.int64).copy()
        corpora = bank["corpus"][: FINAL.N_DISTR_MAX].copy()
    if len(np.unique(ids)) != FINAL.N_DISTR_MAX:
        raise RuntimeError("distractor capture IDs are not unique")
    source = np.concatenate([original, distractors])
    view = exact_pool_view(source, FIVE.N_TEST, n_pool)
    stop = view.diagnostics["source_n_pool"]
    n_distractors = stop - FIVE.N_TEST
    sums, draw_meta = FINAL._assemble_draw_sums(paths)
    missing = set(ids[:n_distractors].tolist()) - sums.keys()
    if missing:
        raise RuntimeError(f"missing fresh draws for {len(missing)} distractors")
    fresh_sums = np.stack([sums[int(ci)] for ci in ids[:n_distractors]])
    distractor_means = (
        distractors[:n_distractors].astype(np.float64) + fresh_sums.astype(np.float64)
    ) / (FINAL.K_DRAWS + 1)
    pool = np.concatenate([target, distractor_means])
    if not np.isfinite(pool).all():
        raise RuntimeError("non-finite candidate vectors")
    selected_distractors = view.pool_rows[view.pool_rows >= FIVE.N_TEST] - FIVE.N_TEST
    pool_ids = np.concatenate([-(1 + np.arange(FIVE.N_TEST)), ids[:n_distractors]])
    provenance = {
        "selection": "all unique test targets followed by first unique distractors in pinned bank order",
        "duplicate_audit": view.diagnostics,
        "draw_capture": draw_meta,
        "test_rows": test_rows.tolist(),
        "query_rows": view.pred_rows.tolist(),
        "pool_source_rows": view.pool_rows.tolist(),
        "pool_capture_ids": pool_ids[view.pool_rows].tolist(),
        "true_candidate_columns": view.true_idx.tolist(),
        "selected_distractor_corpora": dict(Counter(corpora[selected_distractors].tolist())),
        "query_rows_sha256": hashlib.sha256(test_rows.tobytes()).hexdigest(),
    }
    return original, target, pool, view, test_rows, provenance


def score_geometry(full: dict[str, np.ndarray], view: FINAL.EvalView, seed: int) -> dict:
    """Score the parent's four metrics and persist each query's strict rank."""
    index = np.ix_(view.pred_rows, view.pool_rows)
    white_distance = full["whiten_cosine"][index]
    distances = {
        "whiten_csls": -FINAL.MB.csls_scores(1.0 - white_distance, FINAL.K_CSLS),
        "whiten_cosine": white_distance,
        "raw_cosine": full["raw_cosine"][index],
        "raw_euclidean": full["raw_euclidean"][index],
    }
    result = {}
    for j, (name, distance) in enumerate(distances.items()):
        ranks = FINAL._strict_ranks(distance, view.true_idx)
        summary = FINAL._rank_summary(
            ranks, len(view.pool_rows), np.random.default_rng(seed + 17 * j)
        )
        result[name] = {**summary, "per_query_ranks": ranks.tolist()}
    return result


def run(args: argparse.Namespace) -> Path:
    """Rescore every saved rung, validate the original operating point, checkpoint."""
    started = time.time()
    paths, hashes = stage_inputs(args)
    original, target, pool, view, test_rows, provenance = load_pool(paths, args.n_pool)
    original_view = FINAL.make_eval_view(original, FIVE.N_TEST, "keep_one")
    if len(view.pred_rows) != 942:
        raise RuntimeError(f"unexpected query count: {len(view.pred_rows)}")
    whiten, whitening = FINAL._whitener(paths["whiten"])
    LOG.info("[geometry] whitening %d source rows once", len(pool))
    whitened_pool = whiten(pool)
    reference = json.loads(
        (ROOT / "eval_results/issue_1901/figure2_five_rollout_scaling.json").read_text()
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    FIVE._write_json(args.out_dir / "pool.json", provenance)
    result = {
        "issue": 1901,
        "analysis": "figure2-five-rollout-retrieval-pool",
        "status": "running",
        "layer": FIVE.LAYER,
        "target": reference["target"],
        "n_rollouts": FIVE.FINAL.K_DRAWS + 1,
        "retrieval": {
            **reference["retrieval"],
            "n_pool": args.n_pool,
            "chance_top1": 1 / args.n_pool,
        },
        "whitening": whitening,
        "data_revision": FIVE.REVISION,
        "input_sha256": hashes,
        "pool_provenance": "pool.json",
        "test_rows_sha256": provenance["query_rows_sha256"],
        "uncertainty": "2,000 query bootstrap draws conditional on the fixed candidate bank; not candidate-pool sampling uncertainty",
        "coverage": {
            "training_sizes": list(FIVE.TRAIN_SIZES),
            "planned_cells": len(FIVE.TRAIN_SIZES) * len(FIVE.PREDICTORS),
            "unscored_1200_extension": "producer persisted summary scores only; no predictions or fits are regenerated in this rescore",
        },
        "per_n": {},
        "baselines": {},
    }
    destination = args.out_dir / "summary.json"

    def score_one(pred: np.ndarray, seed: int) -> dict:
        """Reuse pool whitening across cells and recompute CSLS per candidate pool."""
        full = FINAL._precompute_metric_arrays(pred, pool, whiten(pred), whitened_pool)
        large = score_geometry(full, view, seed)
        small = score_geometry(full, original_view, seed)
        r2, mean_cosine = FIVE.F79._recon_point(pred, target)
        return {
            "r2": float(r2),
            "mean_cosine": float(mean_cosine),
            "top1": large["whiten_csls"]["acc_at_k"]["1"],
            "top5": large["whiten_csls"]["acc_at_k"]["5"],
            "top1_ci95": large["whiten_csls"]["acc1_ci95"],
            "metrics": large,
            "original_pool_metrics": small,
        }

    for n_train in FIVE.TRAIN_SIZES:
        for predictor in FIVE.PREDICTORS:
            pred = FIVE._load_prediction(paths[f"pred_{predictor}_{n_train}"], test_rows)
            seed = 190_102 + n_train + (predictor == "mlp")
            cell = score_one(pred, seed)
            old = reference["per_n"][str(n_train)][predictor]
            np.testing.assert_allclose(cell["r2"], old["r2"], rtol=0, atol=1e-12)
            for k in (1, 5):
                if (
                    cell["original_pool_metrics"]["whiten_csls"]["acc_at_k"][str(k)]
                    != old[f"top{k}"]
                ):
                    raise RuntimeError(
                        f"942-candidate reproduction failed: n={n_train}, {predictor}, k={k}"
                    )
            cell["original_pool_reproduction"] = "PASS: original R2 and top1/top5 reproduced"
            result["per_n"].setdefault(str(n_train), {})[predictor] = cell
            FIVE._write_json(destination, result)
            LOG.info(
                "[cell] n=%d %s top1=%.6f (942-pool %.6f), elapsed=%.1fs",
                n_train,
                predictor,
                cell["top1"],
                old["top1"],
                time.time() - started,
            )

    bundle = FIVE.F79.load_pass_b(paths["pass_b"])
    train, _val, check_test = FIVE.F79.fixed_split(5000, 3600, 400, 1000, FIVE.F79.SPLIT_SEED)
    np.testing.assert_array_equal(check_test, test_rows)
    x = FIVE.F79.input_layer(bundle, "last", FIVE.LAYER)
    y = FIVE.F79.target_vx(bundle, FIVE.LAYER)
    bias = (y[train].astype(np.float64) - x[train].astype(np.float64)).mean(axis=0)
    for label, pred in (
        ("identity_copy", x[test_rows].astype(np.float64)),
        ("identity_bias", x[test_rows].astype(np.float64) + bias),
    ):
        result["baselines"][label] = score_one(pred, 190_701)
        if label == "identity_bias":
            result["baselines"][label]["n_bias_rows"] = len(train)
        FIVE._write_json(destination, result)
        LOG.info("[baseline] %s top1=%.6f", label, result["baselines"][label]["top1"])
    result["status"] = "complete"
    result["coverage"]["completed_cells"] = sum(len(v) for v in result["per_n"].values())
    result["wall_s"] = time.time() - started
    result["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    FIVE._write_json(destination, result)
    return destination


def main() -> None:
    """Parse the standalone, analysis-only invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pool", type=int, default=10_000)
    parser.add_argument(
        "--source-stage", type=Path, help="existing verified base/prediction download tree"
    )
    parser.add_argument("--stage-root", type=Path, default=ROOT / "data/issue_1901/retrieval_10k")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "eval_results/issue_1901/retrieval_10k"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run(args), flush=True)


if __name__ == "__main__":
    main()

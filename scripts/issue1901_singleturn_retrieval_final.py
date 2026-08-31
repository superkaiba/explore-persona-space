#!/usr/bin/env python3
"""Final single-turn retrieval evaluation for the context-to-answer paper.

This is an analysis-only follow-up to issue #1901.  It repairs two protocol
problems in the earlier ``avgpool_scaleup`` read:

1. whitening is fitted exclusively on the 963,444 single-turn training-answer
   bank (the banked ``issue1901_mlpdense`` sufficient-statistics artifact), not
   borrowed from the multi-turn line;
2. exact duplicate answer vectors are removed before candidate pools are
   formed.  The standard deduplication keeps the first representative of each
   equivalence class; a stricter sensitivity drops every member of a duplicate
   class.

The primary target is the homogeneous five-rollout mean: every retained pool
entry, target or distractor, is the mean of the original on-policy answer state
and four fresh on-policy draws.  The script also scores the original single-draw
bank.  For every metric it reports strict acc@1/5 and an equivalence-aware
companion that accepts any candidate whose *source* answer vector is exactly
equal to the target source vector.  On the deduplicated pool these two scores
must agree; the original-pool audit quantifies the duplicate artifact directly.

Primary metric: whitened cosine + two-sided cross-domain CSLS (K=10).
Appendix companions: whitened cosine, raw cosine, and raw Euclidean distance.

The script downloads only banked artifacts and never generates text or calls a
model.  It emits a JSON result plus a two-panel publication figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_fitter_fair_comparison as F79  # noqa: E402
import issue1901_metric_battery as MB  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue1901_singleturn_retrieval_final")

ISSUE = 1901
LAYER = 19
N_TEST = 1_000
N_DISTR_MAX = 19_000
RUNG_TOTALS = (1_000, 2_000, 5_000, 20_000)
K_DRAWS = 4
K_CSLS = 10
KS = (1, 5)
BOOT_N = 2_000
BOOT_SEED = 190_141
WHITEN_LAMBDA = 0.1

HF_FILES = {
    "pass_b": "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt",
    "distractors": "issue1901_metrics/analysis_tensors/distractors_L19.npz",
    "whiten": "issue1901_mlpdense/analysis_tensors/whiten_stats_L19.npz",
    "ridge_pred": "issue1901_mlpdense/analysis_tensors/preds_L19_n963444_ridge.npz",
    "mlp_pred": "issue1901_mlpdense/analysis_tensors/preds_L19_n963444_mlp.npz",
    "ridge_weights": "issue779_monitoring/n1m_readout/weights/L19/ridge.pt",
    "test_draws": "issue1901_avgpool/analysis_tensors/kresample/V_test_shard00.npz",
    "distr_draws_0": "issue1901_avgpool/analysis_tensors/kresample/V_distr_shard00.npz",
    "distr_draws_1": "issue1901_avgpool/analysis_tensors/kresample/V_distr_shard01.npz",
    "distr_draws_2": "issue1901_avgpool/analysis_tensors/kresample/V_distr_shard02.npz",
    "distr_draws_3": "issue1901_avgpool/analysis_tensors/kresample/V_distr_shard03.npz",
}

DEFAULT_STAGE = PROJECT_ROOT / "data" / "issue_1901" / "singleturn_retrieval_final"
DEFAULT_OUT = PROJECT_ROOT / "eval_results" / "issue_1901" / "singleturn_retrieval_final"
DEFAULT_FIG = PROJECT_ROOT / "figures" / "issue_1901" / "singleturn_retrieval_final"

ARM_LABELS = {
    "ridge": "Linear map",
    "mlp": "Nonlinear map",
    "identity_bias": "Identity + bias",
}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    os.replace(tmp, path)


def _sha256(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def stage_inputs(stage_root: Path, revision: str | None) -> tuple[dict[str, Path], str]:
    """Download all inputs at one data-repository revision."""
    if revision is None:
        revision = str(HfApi().repo_info(C.HF_DATA_REPO, repo_type="dataset").sha)
    paths: dict[str, Path] = {}
    for key, repo_path in HF_FILES.items():
        logger.info("[stage] %s", repo_path)
        paths[key] = Path(
            hf_hub_download(
                repo_id=C.HF_DATA_REPO,
                repo_type="dataset",
                filename=repo_path,
                revision=revision,
                local_dir=stage_root,
            )
        )
    return paths, revision


def _exact_classes(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (class id per row, class counts, first row per class), fp32-exact."""
    x = np.ascontiguousarray(rows, dtype=np.float32)
    keys = x.view(np.dtype((np.void, x.dtype.itemsize * x.shape[1]))).ravel()
    _uniq, first, inverse, counts = np.unique(
        keys, return_index=True, return_inverse=True, return_counts=True
    )
    return inverse.astype(np.int64), counts.astype(np.int64), first.astype(np.int64)


@dataclass(frozen=True)
class EvalView:
    name: str
    pred_rows: np.ndarray
    pool_rows: np.ndarray
    true_idx: np.ndarray
    pool_class: np.ndarray
    query_class: np.ndarray
    diagnostics: dict


def make_eval_view(source_pool: np.ndarray, n_query: int, policy: str) -> EvalView:
    """Build an evaluation view before target averaging.

    ``source_pool`` always places query targets first.  Equivalence classes are
    derived from these original fp32 states and therefore remain fixed when the
    same rows are replaced by five-draw means.
    """
    inverse, counts, first = _exact_classes(source_pool)
    duplicate_member = counts[inverse] > 1
    if policy == "original":
        pool_rows = np.arange(len(source_pool), dtype=np.int64)
        pred_rows = np.arange(n_query, dtype=np.int64)
    elif policy == "keep_one":
        pool_rows = np.sort(first)
        canonical = np.zeros(len(source_pool), dtype=bool)
        canonical[pool_rows] = True
        pred_rows = np.flatnonzero(canonical[:n_query])
    elif policy == "drop_all":
        pool_rows = np.flatnonzero(~duplicate_member)
        pred_rows = np.flatnonzero(~duplicate_member[:n_query])
    else:
        raise ValueError(f"unknown duplicate policy {policy!r}")

    new_of_old = np.full(len(source_pool), -1, dtype=np.int64)
    new_of_old[pool_rows] = np.arange(len(pool_rows), dtype=np.int64)
    true_idx = new_of_old[pred_rows]
    assert np.all(true_idx >= 0)
    pool_class = inverse[pool_rows]
    query_class = inverse[pred_rows]
    n_dup_groups = int((counts > 1).sum())
    diag = {
        "source_n_pool": int(len(source_pool)),
        "source_n_unique": int(len(counts)),
        "source_n_duplicate_groups": n_dup_groups,
        "source_n_excess_duplicate_rows": int((counts - 1).clip(min=0).sum()),
        "source_n_duplicate_members": int(counts[counts > 1].sum()),
        "source_n_duplicate_query_members": int(duplicate_member[:n_query].sum()),
        "realized_n_pool": int(len(pool_rows)),
        "realized_n_query": int(len(pred_rows)),
        "realized_n_unique_classes": int(len(np.unique(pool_class))),
        "realized_n_excess_duplicate_classes": int(
            len(pool_class) - len(np.unique(pool_class))
        ),
    }
    if policy != "original":
        assert diag["realized_n_excess_duplicate_classes"] == 0
    return EvalView(policy, pred_rows, pool_rows, true_idx, pool_class, query_class, diag)


def _assemble_draw_sums(paths: dict[str, Path]) -> tuple[dict[int, np.ndarray], dict]:
    """Map capture id -> sum of four fresh answer vectors."""
    pieces = [paths["test_draws"]] + [paths[f"distr_draws_{i}"] for i in range(4)]
    out: dict[int, np.ndarray] = {}
    shapes = []
    for path in pieces:
        z = np.load(path, allow_pickle=False)
        v = np.asarray(z["V"], dtype=np.float32)
        ci = np.asarray(z["ci"], dtype=np.int64)
        assert v.ndim == 3 and v.shape[1:] == (K_DRAWS, C.EXPECTED_HIDDEN), v.shape
        assert len(ci) == len(v)
        sums = v.sum(axis=1, dtype=np.float32)
        for key, row in zip(ci.tolist(), sums, strict=True):
            if int(key) in out:
                raise RuntimeError(f"duplicate capture id {key}")
            out[int(key)] = row
        shapes.append({"file": path.name, "shape": list(v.shape)})
    return out, {"files": shapes, "n_ids": len(out)}


def _load_bank(paths: dict[str, Path]) -> tuple[dict[str, np.ndarray], dict]:
    """Load aligned predictions, source targets, distractors, and five-draw means."""
    bundle = F79.load_pass_b(paths["pass_b"])
    n_ctx = int(bundle["cx_last"].shape[0])
    _train, _val, test = F79.fixed_split(
        n_ctx, n_ctx - 400 - N_TEST, 400, N_TEST, F79.SPLIT_SEED
    )
    x = F79.input_layer(bundle, "last", LAYER)
    y = F79.target_vx(bundle, LAYER)
    x_test = np.asarray(x[test], dtype=np.float32)
    y_test = np.asarray(y[test], dtype=np.float32)
    del x, y, bundle

    preds: dict[str, np.ndarray] = {}
    pred_meta = {}
    for arm, key in (("ridge", "ridge_pred"), ("mlp", "mlp_pred")):
        z = np.load(paths[key], allow_pickle=False)
        rows = np.asarray(z["rows"], dtype=np.int64)
        if not np.array_equal(rows, np.asarray(test, dtype=np.int64)):
            raise RuntimeError(f"{arm} persisted prediction rows do not match pinned test rows")
        preds[arm] = np.asarray(z["pred_fp16"], dtype=np.float64)
        pred_meta[arm] = {
            "stored_dtype": str(z["pred_fp16"].dtype),
            "n": str(z["n"]),
            "source": str(z["source"]),
            "arm": str(z["arm"]),
        }

    payload = torch.load(
        paths["ridge_weights"], map_location="cpu", mmap=True, weights_only=True
    )
    xmu = payload["xmu"].to(torch.float64).numpy()
    ymu = payload["ymu"].to(torch.float64).numpy()
    preds["identity_bias"] = x_test.astype(np.float64) + (ymu - xmu)

    dz = np.load(paths["distractors"], allow_pickle=False)
    d_orig = np.asarray(dz["vx"][:N_DISTR_MAX], dtype=np.float32)
    d_ci = np.asarray(dz["ci"][:N_DISTR_MAX], dtype=np.int64)
    assert len(np.unique(d_ci)) == len(d_ci)

    draw_sums, draw_meta = _assemble_draw_sums(paths)
    test_ci = -(1 + np.arange(N_TEST, dtype=np.int64))
    missing_test = [int(ci) for ci in test_ci if int(ci) not in draw_sums]
    missing_distr = [int(ci) for ci in d_ci if int(ci) not in draw_sums]
    if missing_test or missing_distr:
        raise RuntimeError(
            f"fresh-draw coverage incomplete: test={len(missing_test)}, distractor={len(missing_distr)}"
        )
    test_fresh = np.stack([draw_sums[int(ci)] for ci in test_ci])
    distr_fresh = np.stack([draw_sums[int(ci)] for ci in d_ci])
    y_avg = (y_test.astype(np.float64) + test_fresh.astype(np.float64)) / (K_DRAWS + 1)
    d_avg = (d_orig.astype(np.float64) + distr_fresh.astype(np.float64)) / (K_DRAWS + 1)

    return {
        "y_single": y_test,
        "d_single": d_orig,
        "y_avg": y_avg,
        "d_avg": d_avg,
        **{f"pred_{k}": v for k, v in preds.items()},
    }, {
        "n_ctx_pass_b": n_ctx,
        "test_rows_sha256": hashlib.sha256(np.asarray(test, dtype=np.int64).tobytes()).hexdigest(),
        "prediction_artifacts": pred_meta,
        "draw_capture": draw_meta,
    }


def _whitener(path: Path):
    z = np.load(path, allow_pickle=False)
    required = {"mu_A", "L", "lam", "n_train", "pool_sha256", "store_revision"}
    if not required.issubset(z.files):
        raise RuntimeError(f"whitening artifact missing {required - set(z.files)}")
    meta = {
        "fit_corpus": "single-turn LMSYS + WildChat training-answer bank",
        "n_train": int(z["n_train"]),
        "lambda": float(z["lam"]),
        "pool_sha256": str(z["pool_sha256"]),
        "source_store_revision": str(z["store_revision"]),
        "transform": "z = L^-1 (v - mu_A)",
    }
    if meta["n_train"] != 963_444 or meta["lambda"] != WHITEN_LAMBDA:
        raise RuntimeError(f"unexpected single-turn whitening provenance: {meta}")
    mu = np.asarray(z["mu_A"], dtype=np.float64)
    ell = np.asarray(z["L"], dtype=np.float64)

    def transform(x: np.ndarray) -> np.ndarray:
        return solve_triangular(
            ell, (np.asarray(x, dtype=np.float64) - mu).T, lower=True, check_finite=False
        ).T

    return transform, meta


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


def _strict_ranks(distance: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    rows = np.arange(len(true_idx))
    truth = distance[rows, true_idx]
    tol = 1e-9 * np.maximum(np.abs(truth)[:, None], 1e-12)
    closer = (distance < truth[:, None] - tol).sum(axis=1)
    tied_other = (np.abs(distance - truth[:, None]) <= tol).sum(axis=1) - 1
    return 1.0 + closer + 0.5 * tied_other


def _equivalence_ranks(
    distance: np.ndarray, pool_class: np.ndarray, query_class: np.ndarray
) -> np.ndarray:
    """Optimistic set rank: 1 + candidates strictly better than the best valid row."""
    out = np.empty(len(query_class), dtype=np.float64)
    for i, cls in enumerate(query_class):
        valid = pool_class == cls
        if not np.any(valid):
            raise RuntimeError(f"query class {cls} absent from pool")
        best = float(distance[i, valid].min())
        tol = 1e-9 * max(abs(best), 1e-12)
        out[i] = 1.0 + int((distance[i] < best - tol).sum())
    return out


def _rank_summary(ranks: np.ndarray, n_pool: int, rng: np.random.Generator) -> dict:
    n = len(ranks)
    boot = rng.integers(0, n, size=(BOOT_N, n))
    acc1_draws = (ranks[boot] <= 1).mean(axis=1)
    return {
        "n_query": n,
        "n_pool": int(n_pool),
        "acc_at_k": {str(k): float((ranks <= k).mean()) for k in KS},
        "acc1_ci95": {
            "lo": float(np.quantile(acc1_draws, 0.025)),
            "hi": float(np.quantile(acc1_draws, 0.975)),
        },
        "median_rank": float(np.median(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "chance_at_k": {str(k): float(k / n_pool) for k in KS},
    }


def _score_distance(
    distance: np.ndarray,
    view: EvalView,
    *,
    seed: int,
) -> dict:
    strict = _strict_ranks(distance, view.true_idx)
    equiv = _equivalence_ranks(distance, view.pool_class, view.query_class)
    return {
        "strict": _rank_summary(strict, distance.shape[1], np.random.default_rng(seed)),
        "duplicate_aware": _rank_summary(
            equiv, distance.shape[1], np.random.default_rng(seed + 1)
        ),
    }


def score_cell(
    pred: np.ndarray,
    pool: np.ndarray,
    view: EvalView,
    whiten,
    *,
    seed: int,
) -> dict:
    q = np.asarray(pred[view.pred_rows], dtype=np.float64)
    p = np.asarray(pool[view.pool_rows], dtype=np.float64)
    q2 = (q * q).sum(axis=1)[:, None]
    p2 = (p * p).sum(axis=1)[None, :]
    raw_euclidean = q2 + p2 - 2.0 * (q @ p.T)
    raw_cosine_sim = _cosine(q, p)
    zq, zp = whiten(q), whiten(p)
    whiten_cosine_sim = _cosine(zq, zp)
    whiten_csls_sim = MB.csls_scores(whiten_cosine_sim, K_CSLS)
    metrics = {
        "whiten_csls": -whiten_csls_sim,
        "whiten_cosine": 1.0 - whiten_cosine_sim,
        "raw_cosine": 1.0 - raw_cosine_sim,
        "raw_euclidean": raw_euclidean,
    }
    result = {
        name: _score_distance(dist, view, seed=seed + 17 * j)
        for j, (name, dist) in enumerate(metrics.items())
    }
    if view.name != "original":
        for name, value in result.items():
            a = value["strict"]["acc_at_k"]
            b = value["duplicate_aware"]["acc_at_k"]
            if a != b:
                raise RuntimeError(f"deduplicated strict/equivalence mismatch for {name}: {a} != {b}")
    return result


def _precompute_metric_arrays(
    pred: np.ndarray,
    pool: np.ndarray,
    whitened_pred: np.ndarray,
    whitened_pool: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the expensive geometry once for the largest nested pool."""
    q = np.asarray(pred, dtype=np.float64)
    p = np.asarray(pool, dtype=np.float64)
    q2 = (q * q).sum(axis=1)[:, None]
    p2 = (p * p).sum(axis=1)[None, :]
    return {
        "raw_euclidean": q2 + p2 - 2.0 * (q @ p.T),
        "raw_cosine": 1.0 - _cosine(q, p),
        "whiten_cosine": 1.0 - _cosine(whitened_pred, whitened_pool),
    }


def _score_precomputed(
    full_metrics: dict[str, np.ndarray], view: EvalView, *, seed: int
) -> dict:
    """Slice one nested duplicate-policy view and recompute its CSLS neighborhoods."""
    ix = np.ix_(view.pred_rows, view.pool_rows)
    raw_euclidean = full_metrics["raw_euclidean"][ix]
    raw_cosine = full_metrics["raw_cosine"][ix]
    whiten_cosine = full_metrics["whiten_cosine"][ix]
    whiten_sim = 1.0 - whiten_cosine
    distances = {
        "whiten_csls": -MB.csls_scores(whiten_sim, K_CSLS),
        "whiten_cosine": whiten_cosine,
        "raw_cosine": raw_cosine,
        "raw_euclidean": raw_euclidean,
    }
    result = {
        name: _score_distance(dist, view, seed=seed + 17 * j)
        for j, (name, dist) in enumerate(distances.items())
    }
    if view.name != "original":
        for name, value in result.items():
            if value["strict"]["acc_at_k"] != value["duplicate_aware"]["acc_at_k"]:
                raise RuntimeError(f"deduplicated strict/equivalence mismatch for {name}")
    return result


def run_analysis(paths: dict[str, Path], data_revision: str) -> dict:
    started = time.time()
    bank, bank_meta = _load_bank(paths)
    whiten, whiten_meta = _whitener(paths["whiten"])
    cells: dict[str, dict] = {}
    duplicate_audit: dict[str, dict] = {}
    max_pool = {
        "single": np.concatenate([bank["y_single"], bank["d_single"]], axis=0),
        "avg": np.concatenate([bank["y_avg"], bank["d_avg"]], axis=0),
    }
    whitened_pool = {}
    for entry, pool in max_pool.items():
        logger.info("[score] whitening %s maximum pool (%d rows)", entry, len(pool))
        whitened_pool[entry] = whiten(pool)

    views_by_rung: dict[int, dict[str, EvalView]] = {}
    for nominal in RUNG_TOTALS:
        source_pool = max_pool["single"][:nominal]
        views = {
            policy: make_eval_view(source_pool, N_TEST, policy)
            for policy in ("original", "keep_one", "drop_all")
        }
        views_by_rung[nominal] = views
        duplicate_audit[str(nominal)] = {k: v.diagnostics for k, v in views.items()}
        logger.info(
            "[score] nominal=%d duplicates=%d excess=%d dedup pool=%d query=%d",
            nominal,
            views["original"].diagnostics["source_n_duplicate_groups"],
            views["original"].diagnostics["source_n_excess_duplicate_rows"],
            views["keep_one"].diagnostics["realized_n_pool"],
            views["keep_one"].diagnostics["realized_n_query"],
        )

    for entry, pool in max_pool.items():
        for arm in ARM_LABELS:
            logger.info("[score] geometry entry=%s arm=%s", entry, arm)
            pred = bank[f"pred_{arm}"]
            full_metrics = _precompute_metric_arrays(
                pred, pool, whiten(pred), whitened_pool[entry]
            )
            for nominal in RUNG_TOTALS:
                views = views_by_rung[nominal]
                for policy, view in views.items():
                    key = f"{arm}|{entry}|{policy}|pool_{nominal}"
                    stable_seed = BOOT_SEED + int.from_bytes(
                        hashlib.blake2b(key.encode(), digest_size=4).digest(), "little"
                    )
                    cells[key] = {
                        "arm_label": ARM_LABELS[arm],
                        "entry": entry,
                        "duplicate_policy": policy,
                        "nominal_pool": nominal,
                        "view": view.diagnostics,
                        "metrics": _score_precomputed(full_metrics, view, seed=stable_seed),
                    }
                    logger.info(
                        "[score] %s primary strict=%.4f aware=%.4f",
                        key,
                        cells[key]["metrics"]["whiten_csls"]["strict"]["acc_at_k"]["1"],
                        cells[key]["metrics"]["whiten_csls"]["duplicate_aware"][
                            "acc_at_k"
                        ]["1"],
                    )
            del full_metrics

    return {
        "round": "singleturn-retrieval-final",
        "issue": ISSUE,
        "layer": LAYER,
        "primary_operating_point": {
            "entry": "avg",
            "duplicate_policy": "keep_one",
            "metric": "whiten_csls",
            "rank_read": "strict",
            "description": (
                "homogeneous five-rollout answer means; exact source-vector duplicates "
                "removed before pool construction by retaining one representative; "
                "single-turn-train whitening; whitened cosine + two-sided CSLS K=10"
            ),
        },
        "conventions": {
            "deduplication": (
                "Exact fp32 source answer-vector equivalence classes are computed before "
                "rollout averaging. keep_one retains the first representative (queries are "
                "ordered before distractors); drop_all excludes every member of any repeated "
                "class; original is an audit only."
            ),
            "duplicate_aware": (
                "A query is correct at k when any candidate in its exact source-vector "
                "equivalence class has set-rank <= k. Set-rank is one plus the number of "
                "candidates strictly better than the best valid candidate."
            ),
            "averaging": (
                "Every target and distractor is the mean of the original answer vector and "
                "four fresh on-policy draws (seeds 43-46); no target/distractor asymmetry."
            ),
            "strict_rank": "mid-rank with 1e-9 relative tie tolerance; top ties fail acc@1",
            "primary_metric": "whitened cosine + cross-domain two-sided CSLS, K=10",
            "appendix_metrics": ["whiten_cosine", "raw_cosine", "raw_euclidean"],
            "uncertainty": f"row bootstrap, {BOOT_N} draws, 95% percentile interval",
        },
        "whitening": whiten_meta,
        "data_revision": data_revision,
        "input_sha256": {key: _sha256(path) for key, path in paths.items()},
        "bank": bank_meta,
        "duplicate_audit": duplicate_audit,
        "cells": cells,
        "wall_s": round(time.time() - started, 1),
        **as_metadata_dict(git_provenance()),
    }


def make_figure(summary: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as PP

    PP.set_paper_style()
    colors = {"ridge": "#0072B2", "mlp": "#D55E00", "identity_bias": "#009E73"}
    markers = {"ridge": "o", "mlp": "s", "identity_bias": "^"}
    plt.rcParams.update(
        {
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.65),
        gridspec_kw={"width_ratios": [1.2, 1]},
        constrained_layout=True,
    )

    ax = axes[0]
    for arm in ARM_LABELS:
        xs, ys, lo, hi = [], [], [], []
        for nominal in RUNG_TOTALS:
            cell = summary["cells"][f"{arm}|avg|keep_one|pool_{nominal}"]
            read = cell["metrics"]["whiten_csls"]["strict"]
            xs.append(read["n_pool"])
            ys.append(read["acc_at_k"]["1"])
            lo.append(read["acc1_ci95"]["lo"])
            hi.append(read["acc1_ci95"]["hi"])
        ys_arr = np.asarray(ys)
        ax.plot(
            xs,
            ys,
            color=colors[arm],
            marker=markers[arm],
            lw=1.5,
            ms=4,
            label=ARM_LABELS[arm],
        )
        ax.fill_between(xs, lo, hi, color=colors[arm], alpha=0.13, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Deduplicated candidate-pool size")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1.01)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("A   Five-rollout mean, whitened cosine + CSLS", fontsize=8, loc="left")

    ax = axes[1]
    metric_order = ("raw_euclidean", "raw_cosine", "whiten_cosine", "whiten_csls")
    labels = ("Raw\nEuclid.", "Raw\ncos.", "Whitened\ncos.", "+ CSLS")
    x = np.arange(len(metric_order))
    width = 0.24
    nominal = max(RUNG_TOTALS)
    for j, arm in enumerate(ARM_LABELS):
        cell = summary["cells"][f"{arm}|avg|keep_one|pool_{nominal}"]
        vals = [cell["metrics"][m]["strict"]["acc_at_k"]["1"] for m in metric_order]
        ax.bar(
            x + (j - 1) * width,
            vals,
            width,
            color=colors[arm],
            label=ARM_LABELS[arm],
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Top-1 accuracy")
    realized = summary["cells"][f"ridge|avg|keep_one|pool_{nominal}"]["view"][
        "realized_n_pool"
    ]
    ax.set_title(f"B   Metric robustness (pool={realized:,})", fontsize=8, loc="left")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            fig_dir / f"singleturn_retrieval_final.{ext}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    _write_json(
        fig_dir / "singleturn_retrieval_final.meta.json",
        {
            "source": "eval_results/issue_1901/singleturn_retrieval_final/summary.json",
            "panel_a": (
                "Primary deduplicated homogeneous-five-draw whitened-cosine+CSLS strict "
                "acc@1 with 95% row-bootstrap intervals."
            ),
            "panel_b": "Appendix metric ladder at the largest deduplicated pool.",
            "colors": colors,
            "accessibility": "Okabe-Ito colors plus marker-shape redundancy in panel A.",
        },
    )


def _smoke() -> None:
    rng = np.random.default_rng(4)
    unique = rng.normal(size=(8, 12)).astype(np.float32)
    source = np.concatenate([unique[:5], unique[[0, 1]], unique[5:]], axis=0)
    view_orig = make_eval_view(source, 7, "original")
    view_keep = make_eval_view(source, 7, "keep_one")
    view_drop = make_eval_view(source, 7, "drop_all")
    assert view_orig.diagnostics["source_n_excess_duplicate_rows"] == 2
    assert view_keep.diagnostics["realized_n_pool"] == len(source) - 2
    assert view_keep.diagnostics["realized_n_excess_duplicate_classes"] == 0
    assert view_drop.diagnostics["realized_n_pool"] == len(source) - 4
    d = np.square(source[:7, None, :] - source[None, :, :]).sum(-1)
    strict = _strict_ranks(d, np.arange(7))
    aware = _equivalence_ranks(d, view_orig.pool_class, view_orig.query_class)
    assert (aware <= strict).all()
    assert (aware <= 1).all()
    assert strict[5] > 1 and strict[6] > 1
    print("smoke PASS")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG)
    ap.add_argument("--revision", default=None, help="HF dataset commit; default resolves HEAD once")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
    )
    if args.smoke:
        _smoke()
        return
    if args.skip_download:
        paths = {key: args.stage_root / repo_path for key, repo_path in HF_FILES.items()}
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"--skip-download but inputs are absent: {missing}")
        revision = args.revision or "locally-staged (revision not supplied)"
    else:
        paths, revision = stage_inputs(args.stage_root, args.revision)
    summary = run_analysis(paths, revision)
    _write_json(args.out_dir / "summary.json", summary)
    make_figure(summary, args.fig_dir)
    logger.info("wrote %s", args.out_dir / "summary.json")


if __name__ == "__main__":
    main()

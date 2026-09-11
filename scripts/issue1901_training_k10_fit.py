#!/usr/bin/env python3
"""Fit task1901 maps across training K and score a fixed held-out answer bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue1901_k_rollout_ablation as KOLD  # noqa: E402
import issue1901_plot1_remake as PLOT1  # noqa: E402
import issue1901_singleturn_retrieval_final as FINAL  # noqa: E402
import issue1901_training_k10_prepare as PREP  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

LOG = logging.getLogger(__name__)
ARMS = ("ridge", "identity_bias")
LAMBDAS = np.logspace(-3, 8, 23)
VERSION = "training-k10-gcv-intercept-cluster-v1"


def sha(path: Path) -> str:
    """Hash without materializing a tensor file."""
    return FINAL._sha256(path)


def save_npz(path: Path, **arrays) -> None:
    """Atomically persist numerical state; compression is outside the fit loop."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def factorize(x: np.ndarray) -> dict:
    """Share the standardized primal factorization across all training targets."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("invalid training design")
    mu = x.mean(0)
    sd = x.std(0, ddof=1) + 1e-9  # #779 n1m standardizer convention.
    xn = (x - mu) / sd
    eig, basis = np.linalg.eigh(xn.T @ xn)
    if eig.min() < -1e-8 * max(1.0, float(eig.max())):
        raise ValueError("input Gram matrix is not positive semidefinite")
    return {"xmu": mu, "xsd": sd, "xn": xn, "eig": np.maximum(eig, 0), "basis": basis}


def gcv_solution(fac: dict, y: np.ndarray, lambdas: np.ndarray = LAMBDAS) -> dict:
    """GCV from shared X geometry, with the fitted intercept counted in df."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or len(y) != len(fac["xn"]) or not np.isfinite(y).all():
        raise ValueError("invalid training targets")
    ymu = y.mean(0)
    yc = y - ymu
    projected = fac["basis"].T @ (fac["xn"].T @ yc)
    energy = np.square(projected).sum(1)
    inverse = 1 / (fac["eig"][None, :] + np.asarray(lambdas)[:, None])
    total = float(np.square(yc).sum())
    rss = total - (2 * inverse - fac["eig"][None, :] * inverse**2) @ energy
    if np.any(rss < -1e-9 * max(1.0, total)):
        raise ValueError("negative GCV residual sum of squares beyond rounding tolerance")
    rss = np.maximum(rss, 0)
    df = 1 + (fac["eig"][None, :] * inverse).sum(1)
    if np.any(len(y) - df <= 0):
        raise ValueError("GCV has non-positive residual degrees of freedom")
    scores = rss / (len(y) - df) ** 2
    index = int(np.argmin(scores))
    lam = float(lambdas[index])
    return {
        "ymu": ymu,
        "coefficient_eigen": projected / (fac["eig"] + lam)[:, None],
        "selected_lambda": lam,
        "gcv": scores,
        "rss": rss,
        "df": df,
        "lambda_grid_edge": "low" if index == 0 else "high" if index == len(lambdas) - 1 else None,
    }


def cluster_resampling(hashes: list[str], retrieval_rows: np.ndarray, n_boot: int, seed: int):
    """Resample prompt identities, preserving all duplicate rows in each cluster."""
    labels = np.unique(np.asarray(hashes), return_inverse=True)[1]
    n_groups = int(labels.max()) + 1
    query_labels = labels[retrieval_rows]
    if len(np.unique(query_labels)) != n_groups or len(query_labels) != n_groups:
        raise ValueError("retrieval representatives do not match the unique prompt identities")
    rng = np.random.default_rng(seed)
    group_counts = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_boot)
    return group_counts[:, labels], group_counts[:, query_labels], labels


def bootstrap_r2(target: np.ndarray, predictions: np.ndarray, row_counts: np.ndarray):
    """Pooled R² with exact, variable-size cluster-bootstrap centering."""
    target = np.asarray(target, np.float64)
    predictions = np.asarray(predictions, np.float64)
    sse = np.square(predictions - target).sum(-1)
    total = float(np.square(target - target.mean(0)).sum())
    n = row_counts.sum(1)
    sums = row_counts @ target
    total_boot = row_counts @ np.square(target).sum(1) - np.square(sums).sum(1) / n
    if total <= 0 or np.any(total_boot <= 0):
        raise ValueError("degenerate R² target variance")
    return 1 - sse.sum(-1) / total, 1 - (row_counts @ sse.T).T / total_boot, sse


def load_banks(inputs: Path, capture: Path) -> tuple[dict, dict, dict, dict]:
    """Verify hashes and reconcile every training (context, seed) exactly once."""
    input_manifest = PREP.validate(inputs, PREP.REVISION)
    manifest = json.loads((capture / "capture_manifest.json").read_text())
    for key, expected in (
        ("model", PREP.MODEL),
        ("model_revision", PREP.MODEL_REVISION),
        ("layer", 19),
    ):
        if input_manifest[key] != expected or manifest["recipe"][key] != expected:
            raise ValueError(f"input/capture model or layer mismatch: {key}")
    with np.load(inputs / "train.npz", allow_pickle=False) as z:
        train = {key: z[key] for key in z.files}
    with np.load(inputs / "test.npz", allow_pickle=False) as z:
        test = {key: z[key] for key in z.files}
    prompts = json.loads((inputs / "prompts.json").read_text())
    if manifest["input_manifest_sha256"] != sha(inputs / "manifest.json"):
        raise ValueError("capture input manifest does not match prepared bank")
    if len(train["ci"]) != 19000 or len(test["ci"]) != 1000:
        raise ValueError("production context coverage mismatch")
    ci_index = {int(ci): i for i, ci in enumerate(train["ci"])}
    if len(ci_index) != len(train["ci"]):
        raise ValueError("duplicate training context IDs")
    fresh = np.full((len(ci_index), 5, 3584), np.nan, dtype=np.float16)
    seen = np.zeros((len(ci_index), 5), dtype=bool)
    for entry in manifest["files"]:
        path = capture / entry["path"]
        if (
            path.resolve().is_relative_to(capture.resolve()) is False
            or sha(path) != entry["sha256"]
        ):
            raise ValueError(f"capture path/hash mismatch: {entry['path']}")
        with np.load(path, allow_pickle=False) as z:
            seed = int(z["seed"])
            ids = z["ci"]
            values = z["V"]
            if seed not in range(47, 52) or seed != entry["seed"]:
                raise ValueError("unexpected capture seed")
            if ids.tolist() != entry["ci"] or values.shape != (len(ids), 3584):
                raise ValueError("capture manifest/array rows disagree")
            if json.loads(str(z["recipe"])) != manifest["recipe"]:
                raise ValueError("mixed capture recipes")
            positions = np.array([ci_index[int(ci)] for ci in ids])
            if len(np.unique(positions)) != len(positions) or seen[positions, seed - 47].any():
                raise ValueError("duplicate captured context/seed")
            if not np.isfinite(values).all() or np.any(z["n_ans"] <= 0):
                raise ValueError("invalid captured answer vectors")
            fresh[positions, seed - 47] = values
            seen[positions, seed - 47] = True
    if not seen.all() or manifest["realized_new_rows"] != int(seen.sum()):
        raise ValueError("incomplete new training draw bank")
    train["Y_new"] = fresh
    for name, bank, n_fresh in (("train", train, 4), ("test", test, 9)):
        if bank["X"].shape != bank["Y_original"].shape or bank["X"].shape[1] != 3584:
            raise ValueError(f"invalid {name} original bank")
        if bank["Y_fresh"].shape != (len(bank["ci"]), n_fresh, 3584):
            raise ValueError(f"invalid {name} fresh bank")
        if not np.array_equal(bank["fresh_seeds"], np.arange(43, 43 + n_fresh)):
            raise ValueError(f"invalid {name} seed order")
        if [r["ci"] for r in prompts[name]] != bank["ci"].tolist():
            raise ValueError(f"invalid {name} prompt order")
        for key in ("X", "Y_original", "Y_fresh"):
            if not np.isfinite(bank[key]).all():
                raise ValueError(f"nonfinite {name}/{key}")
    if {r["prompt_sha256"] for r in prompts["train"]} & {
        r["prompt_sha256"] for r in prompts["test"]
    }:
        raise ValueError("training/test exact-prompt overlap")
    provenance = {
        "version": VERSION,
        "inputs": {
            p: sha(inputs / p) for p in ("train.npz", "test.npz", "prompts.json", "manifest.json")
        },
        "capture_manifest_sha256": sha(capture / "capture_manifest.json"),
        "capture_recipe": manifest["recipe"],
        "lambda_grid_spec": {"log10_min": -3, "log10_max": 8, "count": 23},
    }
    return train, test, prompts, provenance


def fit_maps(train: dict, test: dict, out: Path, provenance: dict) -> np.ndarray:
    """Fit each mean target using one input eigendecomposition and checkpoint it."""
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    output = out / "fits"
    output.mkdir(parents=True, exist_ok=True)
    fac = factorize(train["X"])
    test_projection = ((test["X"] - fac["xmu"]) / fac["xsd"]) @ fac["basis"]
    cumulative = train["Y_original"].astype(np.float64)
    predictions = []
    metadata = []
    for k in range(1, 11):
        started = time.monotonic()
        if k > 1:
            draw = train["Y_fresh"][:, k - 2] if k <= 5 else train["Y_new"][:, k - 6]
            cumulative += draw
        path = output / f"k{k:02d}.npz"
        note = output / f"k{k:02d}.json"
        if path.exists() or note.exists():
            if not (path.exists() and note.exists()):
                raise ValueError(f"incomplete fit checkpoint K={k}")
            meta = json.loads(note.read_text())
            if meta["fingerprint"] != fingerprint or meta["sha256"] != sha(path):
                raise ValueError(f"stale fit checkpoint K={k}")
            with np.load(path, allow_pickle=False) as z:
                pred = z["predictions"]
        else:
            target = cumulative / k
            solution = gcv_solution(fac, target)
            ridge = test_projection @ solution["coefficient_eigen"] + solution["ymu"]
            bias = solution["ymu"] - fac["xmu"]
            pred = np.stack((ridge, test["X"] + bias))
            weights = fac["basis"] @ solution["coefficient_eigen"]
            save_npz(
                path,
                predictions=pred,
                W=weights.astype(np.float32),
                xmu=fac["xmu"],
                xsd=fac["xsd"],
                ymu=solution["ymu"],
                bias=bias,
                lambdas=LAMBDAS,
                gcv=solution["gcv"],
                gcv_rss=solution["rss"],
                gcv_df=solution["df"],
            )
            meta = {
                "k_train": k,
                "n_train": len(target),
                "fingerprint": fingerprint,
                "sha256": sha(path),
                "selected_lambda": solution["selected_lambda"],
                "lambda_grid_edge": solution["lambda_grid_edge"],
                "wall_s": time.monotonic() - started,
            }
            if solution["lambda_grid_edge"] is not None:
                diagnostic = gcv_solution(fac, target, np.logspace(-5, 10, 31))
                meta["expanded_grid_diagnostic"] = {
                    "selected_lambda": diagnostic["selected_lambda"],
                    "lambda_grid_edge": diagnostic["lambda_grid_edge"],
                    "used_for_reported_prediction": False,
                }
            FINAL._write_json(note, meta)
        if pred.shape != (2, 1000, 3584) or not np.isfinite(pred).all():
            raise ValueError(f"invalid predictions K={k}")
        predictions.append(pred)
        metadata.append(meta)
        LOG.info(
            "[fit] unit %d/10 lambda=%g elapsed=%.1fs",
            k,
            meta["selected_lambda"],
            time.monotonic() - started,
        )
    FINAL._write_json(output / "manifest.json", {"provenance": provenance, "fits": metadata})
    return np.stack(predictions)


def summarize(point: float, boots: np.ndarray) -> dict:
    return {"mean": float(point), "ci95": np.quantile(boots, [0.025, 0.975]).tolist()}


def score_maps(train, test, prompts, predictions, out: Path, provenance: dict) -> dict:
    """Score the train-K × eval-K grid with fixed geometry and paired clusters."""
    view = FINAL.make_eval_view(test["Y_original"], len(test["ci"]), "keep_one")
    if not np.array_equal(view.pred_rows, test["dedup_rows"]):
        raise ValueError("retrieval policy differs from the prepared source bank")
    row_counts, query_counts, clusters = cluster_resampling(
        [r["prompt_sha256"] for r in prompts["test"]], view.pred_rows, FINAL.BOOT_N, FINAL.BOOT_SEED
    )
    mu, ell = PLOT1.train_whitening_stats(train["Y_original"], torch.device("cpu"))
    whiten = lambda values: PLOT1.whiten(values, mu, ell)
    flat = predictions.reshape(-1, len(test["ci"]), 3584)
    zpred = whiten(flat.reshape(-1, 3584)).reshape(flat.shape)
    target = test["Y_original"].astype(np.float64).copy()
    cells = {}
    all_r2 = []
    all_r2_boot = []
    all_hits = []
    all_retrieval_boot = []
    ix = np.ix_(view.pred_rows, view.pool_rows)
    for k_eval in range(1, 11):
        start = time.monotonic()
        if k_eval > 1:
            target += test["Y_fresh"][:, k_eval - 2]
        y = target / k_eval
        ztarget = whiten(y)
        r2, r2_boot, _ = bootstrap_r2(y, flat, row_counts)
        hits = []
        for ai, pred in enumerate(flat):
            distances = FINAL._precompute_metric_arrays(pred, y, zpred[ai], ztarget)
            sim = 1 - distances["whiten_cosine"][ix]
            arrays = {
                "whiten_csls": -FINAL.MB.csls_scores(sim, FINAL.K_CSLS),
                **{key: value[ix] for key, value in distances.items()},
            }
            ranks = np.stack([FINAL._strict_ranks(arrays[m], view.true_idx) for m in KOLD.METRICS])
            hits.append(ranks <= 1)
        hits = np.asarray(hits)
        retrieval_boot = np.einsum(
            "bq,amq->amb", query_counts / len(view.pred_rows), hits, optimize=True
        )
        for k_train in range(1, 11):
            for arm_index, arm in enumerate(ARMS):
                a = (k_train - 1) * len(ARMS) + arm_index
                cells.setdefault(str(k_train), {}).setdefault(arm, {})[str(k_eval)] = {
                    "r2": summarize(r2[a], r2_boot[a]),
                    "retrieval": {
                        m: summarize(hits[a, j].mean(), retrieval_boot[a, j])
                        for j, m in enumerate(KOLD.METRICS)
                    },
                }
        all_r2.append(r2)
        all_r2_boot.append(r2_boot)
        all_hits.append(hits)
        all_retrieval_boot.append(retrieval_boot)
        save_npz(
            out / "scores" / f"eval_k{k_eval:02d}.npz",
            r2=r2,
            r2_boot=r2_boot,
            hits=hits,
            retrieval_boot=retrieval_boot,
        )
        LOG.info(
            "[score] unit %d/10 eval_k=%d elapsed=%.1fs", k_eval, k_eval, time.monotonic() - start
        )
    r2 = np.asarray(all_r2)
    r2_boot = np.asarray(all_r2_boot)
    hits = np.asarray(all_hits)
    retrieval_boot = np.asarray(all_retrieval_boot)
    contrasts = {}
    for high, low in ((10, 1), (5, 1), (10, 5)):
        key = f"K{high}_minus_K{low}_eval10"
        contrasts[key] = {}
        for j, arm in enumerate(ARMS):
            h, l = (high - 1) * 2 + j, (low - 1) * 2 + j
            contrasts[key][arm] = {
                "r2": summarize(r2[-1, h] - r2[-1, l], r2_boot[-1, h] - r2_boot[-1, l]),
                "retrieval": {
                    m: summarize(
                        hits[-1, h, mi].mean() - hits[-1, l, mi].mean(),
                        retrieval_boot[-1, h, mi] - retrieval_boot[-1, l, mi],
                    )
                    for mi, m in enumerate(KOLD.METRICS)
                },
            }
    save_npz(
        out / "analysis_tensors.npz",
        r2=r2,
        r2_boot=r2_boot,
        hits=hits,
        retrieval_boot=retrieval_boot,
        row_counts=row_counts,
        query_counts=query_counts,
        cluster_labels=clusters,
        retrieval_rows=view.pred_rows,
        whiten_mu=mu,
        whiten_L=ell,
    )
    result = {
        "provenance": provenance,
        "n_train": len(train["ci"]),
        "n_test": len(test["ci"]),
        "n_prompt_clusters": int(clusters.max()) + 1,
        "n_candidates": len(view.pred_rows),
        "chance_top1": 1 / len(view.pred_rows),
        "cells": cells,
        "contrasts": contrasts,
        "primary_contrast": "K10_minus_K1_eval10",
        "whitening": "Original 19k training answers; shrinkage0.1; fixed across grid",
        "bootstrap": "2000 paired prompt-cluster draws; fixed fitted maps and rollout banks",
        "metadata": as_metadata_dict(git_provenance(ROOT), phase="training_k_fit_score"),
    }
    FINAL._write_json(out / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    train, test, prompts, provenance = load_banks(args.inputs, args.capture)
    predictions = fit_maps(train, test, args.out, provenance)
    result = score_maps(train, test, prompts, predictions, args.out, provenance)
    FINAL._write_json(
        args.out / "completion.json",
        {
            "version": VERSION,
            "summary_sha256": sha(args.out / "summary.json"),
            "n_train": result["n_train"],
            "n_candidates": result["n_candidates"],
        },
    )
    LOG.info("[phase=done] training K1..10 fitted and evaluated")


if __name__ == "__main__":
    main()

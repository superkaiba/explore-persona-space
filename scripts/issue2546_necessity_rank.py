"""CPU-only necessity-stratified rank and diversity analysis of banked Qwen3 states.

All training rows are retained. Only validation rank selection and held-out scoring
are subset-specific. See qwen3_necessity_rank_plan.md for frozen estimands and limits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import issue2546_qwen3_rank_reproduction as parent  # noqa: E402

ARMS = {
    "no_think": ("p7_Aoff", "think_off", "cx_last"),
    "context": ("p7_A", "think_on", "cx_last"),
    "end_of_thought": ("p7_D", "think_on", "cot_boundary"),
}
SUBSETS = ("necessary", "both_correct")
BOOT_SEED, BOOT_DRAWS = 20260909, 4000


def curve_rows(prediction, target, intercept, vectors):
    """Return per-row rank-zero SSE and additive contributions for every direction."""
    pc, yc = (prediction - intercept) @ vectors, (target - intercept) @ vectors
    zero = (target - intercept).square().sum(1).numpy()
    increments = (pc.square() - 2 * pc * yc).numpy()
    np.testing.assert_allclose(
        zero + increments.sum(1),
        (target - prediction).square().sum(1).numpy(),
        rtol=1e-9,
        atol=1e-7,
    )
    return zero, increments


def aggregate_curve(zero, increments):
    """Combine row contributions without refitting or constructing low-rank matrices."""
    return np.r_[zero.sum(), zero.sum() + increments.sum(0).cumsum()]


def bootstrap_counts(groups, draws, seed):
    """Multinomial bootstrap weights, with each corpus count fixed and rows paired."""
    groups = np.asarray(groups)
    if len(groups) == 0 or draws < 1:
        raise ValueError("Empty bootstrap")
    rng = np.random.default_rng(seed)
    counts = np.zeros((draws, len(groups)), dtype=np.float64)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        counts[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), draws)
    np.testing.assert_array_equal(counts.sum(1), np.full(draws, len(groups)))
    return counts


def ranks_for_counts(zero, increments, counts, tolerance=0.1):
    """Batched exact threshold crossing, supporting nonmonotonic validation curves."""
    start = counts @ zero
    curves = np.column_stack((start, start[:, None] + (counts @ increments).cumsum(1)))
    passing = curves <= (1 + tolerance) * curves[:, -1, None]
    if not passing.any(1).all():
        raise ValueError("Bootstrap full rank must meet its own threshold")
    return passing.argmax(1)


def measures(scatter):
    """Entropy rank, PR, and stable rank of a centered scatter matrix's spectrum."""
    eigen = torch.linalg.eigvalsh((scatter + scatter.T) / 2).numpy()[::-1]
    if not np.isfinite(eigen).all() or eigen[0] <= 0 or eigen[-1] < -1e-9 * eigen[0]:
        raise ValueError("Invalid covariance spectrum")
    eigen = np.maximum(eigen, 0)
    p = eigen / eigen.sum()
    nz = p[p > 0]
    return {
        "effective_rank_entropy": float(np.exp(-(nz * np.log(nz)).sum())),
        "participation_ratio": float(1 / np.square(p).sum()),
        "stable_rank": float(1 / p.max()),
        "eigenvalues": eigen.tolist(),
    }


def diversity(model, x, y):
    """Measure all three spaces on the SAME subset of outer training questions."""
    if len(x) < 2 or len(x) != len(y):
        raise ValueError("Need paired nonempty diversity rows")
    xc, yc = x - x.mean(0), y - y.mean(0)
    raw = xc.T @ xc
    gram = raw / model["xsd"][:, None] / model["xsd"][None, :]
    out = {
        "n_training_subset": len(x),
        "input": measures(gram),
        "answer": measures(yc.T @ yc),
        "fitted_output": measures(model["coef"].T @ gram @ model["coef"]),
        "raw_input_participation_ratio": float(raw.trace().square() / raw.square().sum()),
    }
    out["fitted_over_answer"] = {
        metric: out["fitted_output"][metric] / out["answer"][metric]
        for metric in ("effective_rank_entropy", "participation_ratio", "stable_rank")
    }
    return out


def row_sst(y, train, test, corpus):
    """Per-query denominators from all-training-row global and per-corpus means."""
    yt = y[test]
    global_sst = (yt - y[train].mean(0)).square().sum(1).numpy()
    within = np.empty(len(yt), np.float64)
    for name in np.unique(corpus[test]):
        rows = corpus[test] == name
        src = train & (corpus == name)
        if not src.any():
            raise ValueError(f"Missing training corpus {name}")
        within[rows] = (yt[rows] - y[src].mean(0)).square().sum(1).numpy()
    if min(global_sst.min(), within.min()) <= 0:
        raise ValueError("Degenerate R2 denominator")
    return global_sst, within


def read_bank(root, cell, ids=None, folds=None, labels=None):
    """Load and align original OOF predictions plus provenance-bearing metric JSON."""
    path = root / "allfit/preds" / f"{cell}__all__a3.npz"
    with np.load(path, allow_pickle=False) as z:
        for key, expected in (("conv_ids", ids), ("folds", folds), ("labels", labels)):
            if expected is not None:
                np.testing.assert_array_equal(z[key], expected)
        if not z["fitted_mask"].all():
            raise ValueError("Incomplete OOF bank")
        result = {k: z[k] for k in ("conv_ids", "folds", "labels", "pred_l24")}
    metric = root / "allfit/results" / f"{cell}__a3.json"
    result["original"] = json.loads(metric.read_text())
    result["hashes"] = {str(p): parent.sha256(p) for p in (path, metric)}
    hitpath = root / "allfit/preds" / f"hits__{cell}__all__a3.npz"
    with np.load(hitpath, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["row_ids"], result["conv_ids"])
        np.testing.assert_array_equal(z["folds"], result["folds"])
        result["hits"] = z["hit_whitened_csls"]
    result["hashes"][str(hitpath)] = parent.sha256(hitpath)
    return result


def write_rows(path, ids, labels, corpus, fold, values):
    """Persist held-out per-question scalars for transparent subset/paired reanalysis."""
    with parent.atomic_replace(path) as temp:
        with temp.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_id", "subset", "corpus", "fold", *values])
            for i in range(len(ids)):
                if labels[i] in SUBSETS:
                    writer.writerow(
                        [ids[i], labels[i], corpus[i], fold, *[v[i] for v in values.values()]]
                    )


def run_cell(args, arm, k, x, y, blocks, bank, ids, labels, folds, corpus, key):
    """One production unit: nested fit, both subsets, full diversity and fold-0 bootstrap."""
    started = time.perf_counter()
    te, va = folds == k, folds == (k + 1) % 5
    train = ~(te | va)
    lam = bank["original"]["lambda"]
    block = parent.combine([blocks[j] for j in range(5) if j not in (k, (k + 1) % 5)])
    if block.n <= parent.DIM:
        raise ValueError("Under-determined fit forbidden in this analysis")
    inner = parent.fit(block, lam)
    vz, vi = curve_rows(parent.predict(inner, x[va]), y[va], inner["ymu"], inner["vectors"])
    subsets = {}
    for subset in SUBSETS:
        mask = labels[va] == subset
        if not mask.any():
            raise ValueError(f"Empty validation subset: {subset}")
        curve = aggregate_curve(vz[mask], vi[mask])
        result = {
            "n_validation": int(mask.sum()),
            "validation_sse_by_rank": curve.tolist(),
            "selected_ranks": {str(t): parent.select_rank(curve, t) for t in parent.TOLERANCES},
        }
        if k == 0:
            counts = bootstrap_counts(corpus[va][mask], BOOT_DRAWS, BOOT_SEED)
            draws = ranks_for_counts(vz[mask], vi[mask], counts)
            result["conditional_rank_bootstrap"] = {
                "draws": draws.tolist(),
                "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
                "median": float(np.median(draws)),
            }
            del counts
        subsets[subset] = result
    del inner, block, vz, vi
    block = parent.combine([blocks[j] for j in range(5) if j != k])
    outer = parent.fit(block, lam)
    prediction = parent.predict(outer, x[te])
    np.testing.assert_allclose(
        prediction.float().numpy(), bank["pred_l24"][te], rtol=2e-6, atol=2e-6
    )
    tz, ti = curve_rows(prediction, y[te], outer["ymu"], outer["vectors"])
    sst_global, sst_corpus = row_sst(y, ~te, te, corpus)
    identity = x[te] + (outer["ymu"] - outer["xmu"])
    identity_sse = (identity - y[te]).square().sum(1).numpy()
    yc = y[~te] - outer["ymu"]
    whitener = parent.shrunk_cholesky_from_cov(
        (yc.T @ yc / (len(yc) - 1)).numpy(), parent.PRIMARY_LAMBDA
    )
    del yc
    identity_hits = parent.retrieval(identity, y[te], outer["ymu"], whitener)
    full_hits = parent.retrieval(prediction, y[te], outer["ymu"], whitener)
    np.testing.assert_array_equal(full_hits, bank["hits"][te])
    row_values = {
        "sse_full": tz + ti.sum(1),
        "sst_global": sst_global,
        "sst_corpus": sst_corpus,
        "sse_identity_bias": identity_sse,
        "hit_full": full_hits.astype(int),
        "hit_identity_bias": identity_hits.astype(int),
    }
    ranks_to_hits = {}
    for subset in SUBSETS:
        result = subsets[subset]
        mask = labels[te] == subset
        curve = aggregate_curve(tz[mask], ti[mask])
        denom = float(sst_corpus[mask].sum())
        result.update(
            {
                "n_test": int(mask.sum()),
                "sst_corpus": denom,
                "sst_global": float(sst_global[mask].sum()),
                "test_sse_by_rank": curve.tolist(),
                "test_r2_full_corpus": 1 - float(curve[-1]) / denom,
                "test_r2_full_global": 1 - float(curve[-1]) / float(sst_global[mask].sum()),
                "full_retrieval_hits": int(full_hits[mask].sum()),
                "identity_bias_sse": float(identity_sse[mask].sum()),
                "identity_bias_retrieval_hits": int(identity_hits[mask].sum()),
            }
        )
        r = result["selected_ranks"]["0.1"]
        if r not in ranks_to_hits:
            basis = outer["vectors"][:, :r]
            reduced = (prediction - outer["ymu"]) @ basis @ basis.T + outer["ymu"]
            ranks_to_hits[r] = parent.retrieval(reduced, y[te], outer["ymu"], whitener)
        result["rank10_retrieval_hits"] = int(ranks_to_hits[r][mask].sum())
        row_values[f"sse_rank10_for_{subset}"] = tz + ti[:, :r].sum(1)
        row_values[f"hit_rank10_for_{subset}"] = ranks_to_hits[r].astype(int)
        result["diversity"] = diversity(
            outer, x[(~te) & (labels == subset)], y[(~te) & (labels == subset)]
        )
        print(
            f"[diversity] {arm}/{k}/{subset} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    rows_path = args.out / f"{arm}__fold{k}.csv"
    write_rows(rows_path, ids[te], labels[te], corpus[te], k, row_values)
    result = {
        "status": "complete",
        "cache_key": key,
        "arm": arm,
        "fold": k,
        "n_inner_train": int(train.sum()),
        "n_outer_train": int((~te).sum()),
        "retrieval_pool": int(te.sum()),
        "retrieval_chance": 1 / int(te.sum()),
        "subsets": subsets,
        "rows_sha256": parent.sha256(rows_path),
        "all_rows_test_sse": float(tz.sum() + ti.sum()),
        "all_rows_sst_corpus": float(sst_corpus.sum()),
        "prediction_parity_max_abs": float(
            np.abs(prediction.float().numpy() - bank["pred_l24"][te]).max()
        ),
        "seconds": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    parent.write_json(args.out / f"{arm}__fold{k}.json", result)
    print(
        f"[complete] {arm}/{k} seconds={result['seconds']:.1f} peak_rss_gib={result['peak_rss_gib']:.2f}",
        flush=True,
    )
    return result


def run(args):
    """Pilot/resume production units, pin sources, and persist a fresh completion sentinel."""
    torch.set_num_threads(8)
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    bank = read_bank(args.data_root, "p7_A")
    ids, folds, labels = bank["conv_ids"].astype(str), bank["folds"], bank["labels"].astype(str)
    if len(ids) != parent.N_ROWS or len(set(ids)) != len(ids):
        raise ValueError("Wrong original paired row universe")
    np.testing.assert_array_equal(np.bincount(folds), [6762] * 5)
    label_map = json.loads(args.labels.read_text())["labels"]
    np.testing.assert_array_equal(labels, [label_map[r] for r in ids])
    counts = Counter(labels)
    if counts["necessary"] != 4522 or counts["both_correct"] != 17693:
        raise ValueError(f"Wrong necessity subset counts: {counts}")
    corpus = np.array([r.split(":")[0] for r in ids])
    recipe = {
        "schema": "issue2546_necessity_rank_v1",
        "model": "Qwen/Qwen3-8B",
        "layer": 24,
        "dimension": parent.DIM,
        "n_rows": len(ids),
        "subset_counts": dict(counts),
        "bootstrap_draws": BOOT_DRAWS,
        "bootstrap_seed": BOOT_SEED,
        "bootstrap_rank_outer_fold": 0,
        "bootstrap_strata": "corpus within subset",
        "training": "all rows: inner 3 folds; outer 4 folds",
        "validation": "subset rows in fold (k+1)%5",
        "test": "subset rows in fold k",
        "tolerances": list(parent.TOLERANCES),
        "basis": "PCA of all-training-row fitted outputs, not subset outputs",
        "diversity": "subset-centered outer-training states, all-training X standardization",
        "selection_scope": "only rank nested; inherited layer and lambda",
        "independence": "within-benchmark IID; dependent folds, conditional bootstrap only",
        "generation": "banked on-policy greedy answers, one per mode; TF recapture of own tokens",
        "gpu_hours": 0,
        "script_sha256": parent.sha256(Path(__file__)),
        "helper_sha256": parent.sha256(Path(parent.__file__)),
        "labels_sha256": parent.sha256(args.labels),
    }
    del bank
    for arm in args.arms:
        cell, side, kind = ARMS[arm]
        manifest = {}
        bank = read_bank(args.data_root, cell, ids, folds, labels)
        original = bank["original"]
        if (
            original["x"] != [side, kind]
            or original["y"] != [side, "ans_mean"]
            or original["layer"] != 24
            or original["n_rows"] != len(ids)
        ):
            raise ValueError("Wrong original map metadata")
        x = torch.from_numpy(parent.load_state(args.data_root, kind, ids, manifest, side)).double()
        y = torch.from_numpy(
            parent.load_state(args.data_root, "ans_mean", ids, manifest, side)
        ).double()
        source_hashes = {**bank["hashes"], **{p: m["sha256"] for p, m in manifest.items()}}
        parent.write_json(
            args.out / f"{arm}__manifest.json",
            {
                "recipe": recipe,
                "sources": manifest,
                "hashes": source_hashes,
                "lambda": original["lambda"],
                "x": original["x"],
                "y": original["y"],
                "corpus_counts": {s: dict(Counter(corpus[labels == s])) for s in SUBSETS},
            },
        )
        blocks = []
        for j in range(5):
            blocks.append(parent.moments(x[folds == j], y[folds == j]))
            print(f"[moments] {arm}/{j} elapsed={time.perf_counter() - started:.1f}s", flush=True)
        for k in args.folds:
            path = args.out / f"{arm}__fold{k}.json"
            key = {
                "recipe": recipe,
                "arm": arm,
                "fold": k,
                "lambda": original["lambda"],
                "sources": source_hashes,
            }
            if path.exists():
                existing = json.loads(path.read_text())
                if (
                    existing["cache_key"] != key
                    or existing["status"] != "complete"
                    or existing["rows_sha256"] != parent.sha256(path.with_suffix(".csv"))
                ):
                    raise ValueError(f"Stale checkpoint: {path}")
                print(f"[resume] {arm}/{k}", flush=True)
                continue
            run_cell(args, arm, k, x, y, blocks, bank, ids, labels, folds, corpus, key)
        del x, y, blocks, bank
    record = {
        "status": "complete",
        "arms": args.arms,
        "folds": args.folds,
        "seconds": time.perf_counter() - started,
        "pid": os.getpid(),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "finished_utc": datetime.now(UTC).isoformat(),
        "gpu_hours": 0,
        "numpy": np.__version__,
        "torch": torch.__version__,
        **parent.as_metadata_dict(
            parent.git_provenance(cwd=Path(__file__).resolve().parents[1]),
            phase="necessity-rank-analysis",
        ),
    }
    parent.write_json(args.out / f"{args.run_label}__completion.json", record)
    print(json.dumps(record), flush=True)


def main():
    """Explicit paths and bounded production-unit selection; no network or GPU switches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    if (
        len(set(args.folds)) != len(args.folds)
        or not set(args.folds).issubset(range(5))
        or len(set(args.arms)) != len(args.arms)
    ):
        parser.error("Distinct valid folds and arms required")
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired K=5 versus K=10 scoring for the frozen task-1901 evaluation bank."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue1901_k10_capture as CAP  # noqa: E402
import issue1901_k_rollout_ablation as K  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

LOG = logging.getLogger(__name__)
TARGETS = ("existing_K5", "K10", "new_K5")
ARCHIVE_SHA = "ff540cc6b6a9cf98abeaf35dac842b1205a8e8a3df7bf92f8b31b5156cfc4e76"


def score_targets(targets, preds, original, whiten, counts, retrieval_counts):
    """Use canonical distances and the same bootstrap draws for all target banks."""
    view = K.FINAL.make_eval_view(original, len(original), "keep_one")
    ix = np.ix_(view.pred_rows, view.pool_rows)
    point, boot, sse = K.bootstrap_r2(targets, preds, np.eye(len(targets)), counts)
    correct, top5 = [], []
    zpred = whiten(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
    for ti, target in enumerate(targets):
        ztarget = whiten(target)
        hits, hits5 = [], []
        for ai, pred in enumerate(preds):
            full = K.FINAL._precompute_metric_arrays(pred, target, zpred[ai], ztarget)
            sim = 1 - full["whiten_cosine"][ix]
            distances = {
                "whiten_csls": -K.FINAL.MB.csls_scores(sim, K.FINAL.K_CSLS),
                **{name: val[ix] for name, val in full.items()},
            }
            ranks = np.stack(
                [K.FINAL._strict_ranks(distances[m], view.true_idx) for m in K.METRICS]
            )
            hits.append(ranks <= 1)
            hits5.append(ranks <= 5)
            reference = K.FINAL.score_cell(pred, target, view, whiten, seed=K.SEED)
            for mi, metric in enumerate(K.METRICS):
                assert np.mean(ranks[mi] <= 1) == reference[metric]["strict"]["acc_at_k"]["1"]
                assert np.mean(ranks[mi] <= 5) == reference[metric]["strict"]["acc_at_k"]["5"]
            LOG.info(
                "target=%d arm=%s R2=%.6f top1=%.6f",
                ti,
                K.ARMS[ai],
                point[ti, ai],
                np.mean(ranks[0] <= 1),
            )
        correct.append(hits)
        top5.append(hits5)
    correct, top5 = np.asarray(correct), np.asarray(top5)
    retrieval_boot = np.einsum(
        "br,tamr->tamb", retrieval_counts / len(view.pred_rows), correct, optimize=True
    )
    return {
        "r2": point,
        "r2_boot": boot,
        "sse": sse,
        "top1": correct,
        "top5": top5,
        "retrieval_boot": retrieval_boot,
        "retrieval_rows": view.pred_rows,
    }


def summarize(arrays) -> dict:
    """Report paired endpoint deltas plus a new-five-draw replication diagnostic."""
    cells = {}
    for ti, name in enumerate(TARGETS):
        cells[name] = {}
        for ai, arm in enumerate(K.ARMS):
            metrics = {}
            for mi, metric in enumerate(K.METRICS):
                metrics[metric] = {
                    **K.interval(
                        arrays["top1"][ti, ai, mi].mean(), arrays["retrieval_boot"][ti, ai, mi]
                    ),
                    "top5": float(arrays["top5"][ti, ai, mi].mean()),
                }
            cells[name][arm] = {
                "r2": K.interval(arrays["r2"][ti, ai], arrays["r2_boot"][ti, ai]),
                "retrieval": metrics,
            }
    contrasts = {}
    for label, left, right in [("K10_minus_existing_K5", 1, 0), ("new_K5_minus_existing_K5", 2, 0)]:
        contrasts[label] = {}
        for ai, arm in enumerate(K.ARMS):
            contrasts[label][arm] = {
                "r2": K.interval(
                    arrays["r2"][left, ai] - arrays["r2"][right, ai],
                    arrays["r2_boot"][left, ai] - arrays["r2_boot"][right, ai],
                ),
                "retrieval": {
                    m: K.interval(
                        arrays["top1"][left, ai, mi].mean() - arrays["top1"][right, ai, mi].mean(),
                        arrays["retrieval_boot"][left, ai, mi]
                        - arrays["retrieval_boot"][right, ai, mi],
                    )
                    for mi, m in enumerate(K.METRICS)
                },
            }
    return {"targets": cells, "contrasts": contrasts}


def figure(result: dict, stem: Path) -> None:
    """Draw the two measured endpoints with paired-protocol pointwise intervals."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from explore_persona_space.analysis import c2a_plot_style as style

    style.set_c2a_style()
    fig, frac = style.c2a_figure("full", aspect=0.44)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.2, top=0.78, wspace=0.3)
    for ai, arm in enumerate(K.ARMS[:2]):
        role = style.ROLES[("linear", "nonlinear")[ai]]
        for j, ax in enumerate(axes):
            values = [result["targets"][name][arm] for name in TARGETS[:2]]
            values = [v["r2"] if j == 0 else v["retrieval"]["whiten_csls"] for v in values]
            y = np.array([v["mean"] for v in values])
            low, high = np.array([v["ci95"] for v in values]).T
            # Bootstrap percentiles can exclude the point estimate. Draw interval
            # segments directly, avoiding negative matplotlib yerr values.
            x = np.array([5, 10]) + (ai - 0.5) * 0.12
            ax.vlines(x, low, high, colors=role.color, linewidth=1.5)
            ax.plot(
                x,
                y,
                color=role.color,
                marker=role.marker,
                linewidth=2,
                label=("Linear map", "Nonlinear map")[ai],
            )
    for ax, title, label in zip(
        axes,
        ["Variance explained", "Answer identification"],
        ["Held-out $R^2$", "Top-1 retrieval"],
        strict=True,
    ):
        style.style_axis(ax)
        ax.set(
            xlabel="Rollouts averaged, $K$",
            ylabel=style.better_label(label),
            xticks=[5, 10],
            xlim=(4, 11),
            title=title,
        )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.035,
        "Frozen maps · 1,000 contexts · 942 candidates · 95% intervals",
        ha="center",
        color=style.MUTED,
        fontsize=14,
    )
    exported = style.save_c2a_figure(
        fig,
        stem,
        title="Does averaging ten answers improve prediction?",
        subject="Direct comparison of existing five draws with ten draws",
        creator=Path(__file__).name,
        include_width=frac,
    )
    K.FINAL._write_json(
        stem.with_suffix(".meta.json"),
        {
            **exported["record"],
            "data": result["targets"],
            "outputs_sha256": {
                key: K.FINAL._sha256(exported[key]) for key in ["png", "pdf", "grayscale"]
            },
        },
    )
    plt.close(fig)


def main() -> None:
    """Validate all banks, reproduce the old endpoint, and persist the paired read."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", type=Path, required=True)
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--bootstrap", type=Path, required=True)
    p.add_argument("--capture-revision", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tensor-out", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = time.monotonic()
    paths = {k: Path(v) for k, v in json.loads(args.paths.read_text()).items()}
    draws, preds, provenance = K.load_inputs(paths)
    whiten, whitening = K.FINAL._whitener(paths["whiten"])
    report = json.loads((args.capture_root / "run_summary.json").read_text())
    assert report["recipe"] == CAP.RECIPE and report["new_vectors"] == 5000
    assert report["parity"]["cosine_min"] >= 0.999
    receipt = {Path(r["path"]).name: r for r in report["uploads"]}
    additional = []
    for seed in CAP.SEEDS:
        path = args.capture_root / "analysis_tensors" / f"seed{seed}.npz"
        assert K.FINAL._sha256(path) == receipt[path.name]["sha256"]
        with np.load(path, allow_pickle=False) as z:
            assert int(z["seed"]) == seed and json.loads(str(z["recipe"])) == CAP.RECIPE
            assert np.array_equal(z["ci"], [-1 - i for i in range(1000)])
            assert z["V"].shape == (1000, 3584) and np.isfinite(z["V"]).all()
            assert np.all(z["n_ans"] > 0)
            additional.append(z["V"].astype(np.float64))
    additional = np.stack(additional)
    targets = np.stack(
        [draws.mean(0), np.concatenate([draws, additional]).mean(0), additional.mean(0)]
    )
    assert K.FINAL._sha256(args.bootstrap) == ARCHIVE_SHA
    with np.load(args.bootstrap, allow_pickle=False) as old:
        counts, rcounts = old["bootstrap_counts"], old["retrieval_bootstrap_counts"]
        arrays = score_targets(targets, preds, draws[0], whiten, counts, rcounts)
        np.testing.assert_allclose(arrays["r2"][0], old["r2"][-1], atol=1e-10, rtol=0)
        np.testing.assert_allclose(arrays["r2_boot"][0], old["r2_boot"][-1], atol=1e-10, rtol=0)
        assert np.array_equal(arrays["top1"][0], old["top1"][-1])
        assert np.array_equal(arrays["top5"][0], old["top5"][-1])
        assert np.array_equal(arrays["retrieval_rows"], old["retrieval_rows"])
    result = summarize(arrays)
    result.update(
        {
            "issue": 1901,
            "model": CAP.KR.MODEL_ID,
            "layer": 19,
            "n_train": 963444,
            "n_test": 1000,
            "n_candidates": 942,
            "chance_top1": 1 / 942,
            "capture_revision": args.capture_revision,
            "capture_report": report,
            "provenance": provenance,
            "whitening": whitening,
            "paired_bootstrap": {
                "n": 2000,
                "seed": K.SEED,
                "source_sha256": ARCHIVE_SHA,
                "r2": "exact recentering",
                "retrieval": "queries resampled; pool and rankings fixed",
            },
            "old_endpoint_parity": "PASS: scores, bootstrap R2, top-1/top-5 per-query outcomes",
            "canonical_scorer_parity": "PASS: all targets, predictors, distances, top-1/top-5",
            "scope": "evaluation-target K=5 vs K=10; fixed maps; intervals conditional on observed draws",
            "metadata": as_metadata_dict(git_provenance(ROOT), phase="paired-comparison"),
            "elapsed_seconds": time.monotonic() - start,
            "coverage": {"contexts": 1000, "new_vectors": 5000, "combined_vectors": 10000},
        }
    )
    args.tensor_out.mkdir(parents=True, exist_ok=True)
    archive = args.tensor_out / "paired_bootstrap.npz"
    np.savez_compressed(
        archive, **arrays, bootstrap_counts=counts, retrieval_bootstrap_counts=rcounts
    )
    result["archive_sha256"] = K.FINAL._sha256(archive)
    K.FINAL._write_json(args.out / "summary.json", result)
    figure(result, args.figure)
    LOG.info("Complete: %.1fs", time.monotonic() - start)


if __name__ == "__main__":
    main()

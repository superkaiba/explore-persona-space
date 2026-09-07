#!/usr/bin/env python3
"""Vary evaluation-target rollout count on the frozen paper Figure 2 bank.

All 31 nonempty subsets of five on-policy answer vectors are scored. Maps,
whitening, context rows, and duplicate policy stay fixed. K-wise estimates
average the subset-specific metrics; bootstrap intervals resample contexts,
conditional on this five-draw bank and the fixed retrieval candidate pool.
Fresh-only subsets (excluding the original draw used for deduplication) are
reported as a selection-sensitivity check. This is not a training-K ablation.

Use --plot-through-k10 to combine persisted K=1–5 and K=10 results without scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import subprocess
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

import issue1901_figure2_five_rollout_scaling as SCALING  # noqa: E402
import issue1901_singleturn_retrieval_final as FINAL  # noqa: E402
import issue779_fitter_fair_comparison as F79  # noqa: E402

LOG = logging.getLogger(__name__)
ARMS = ("ridge", "mlp", "identity_bias")
METRICS = ("whiten_csls", "whiten_cosine", "raw_cosine", "raw_euclidean")
SEED = FINAL.BOOT_SEED
N_BOOT = FINAL.BOOT_N


def load_inputs(paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Verify pinned bytes and join draws/predictions to the paper's test split."""
    reference = json.loads(
        (ROOT / "eval_results/issue_1901/figure2_five_rollout_scaling.json").read_text()
    )
    expected = {**reference["source_sha256"]}
    expected["draws"] = expected.pop("test_draws")
    for arm in ("ridge", "mlp"):
        expected[f"{arm}_pred"] = reference["per_n"]["963444"][arm]["prediction_sha256"]
    expected["ridge_weights"] = "188486f8afd9d95221e32492f3a0be2a3bdb2098cbe7fadfecf1d46433567909"
    for key, digest in expected.items():
        actual = FINAL._sha256(paths[key])
        if actual != digest:
            raise ValueError(f"input hash mismatch: {key}: {actual} != {digest}")
    bundle = F79.load_pass_b(paths["pass_b"])
    n = len(bundle["cx_last"])
    _train, _val, rows = F79.fixed_split(n, n - 1400, 400, 1000, F79.SPLIT_SEED)
    source = F79.target_vx(bundle, 19)[rows].astype(np.float64)
    x = F79.input_layer(bundle, "last", 19)[rows].astype(np.float64)
    del bundle
    row_sha = hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()
    if row_sha != reference["test_rows_sha256"]:
        raise ValueError("test split differs from pinned Figure 2 split")
    with np.load(paths["draws"], allow_pickle=False) as z:
        ids = np.asarray(z["ci"], dtype=np.int64)
        fresh = np.asarray(z["V"], dtype=np.float64)
        assert np.array_equal(z["draws"], [43, 44, 45, 46])
        assert z["n_ans"].shape == (1000, 4) and np.all(z["n_ans"] > 0)
        assert np.all(z["src"] == "test")
    assert fresh.shape == (1000, 4, 3584), fresh.shape
    assert len(np.unique(ids)) == 1000
    index = {int(ci): i for i, ci in enumerate(ids)}
    assert set(index) == set(-(1 + np.arange(1000)))
    fresh = fresh[[index[-(1 + i)] for i in range(1000)]]
    draws = np.concatenate([source[None], fresh.transpose(1, 0, 2)], axis=0)
    preds = [SCALING._load_prediction(paths[f"{arm}_pred"], rows) for arm in ARMS[:2]]
    weights = torch.load(paths["ridge_weights"], map_location="cpu", mmap=True, weights_only=True)
    bias = (weights["ymu"].double() - weights["xmu"].double()).numpy()
    assert bias.shape == (3584,)
    preds.append(x + bias)
    preds = np.stack(preds)
    assert draws.shape == (5, 1000, 3584) and preds.shape == (3, 1000, 3584)
    assert np.isfinite(draws).all() and np.isfinite(preds).all()
    return (
        draws,
        preds,
        {
            "data_revision": SCALING.REVISION,
            "input_sha256": expected,
            "test_rows_sha256": row_sha,
            "test_rows": rows.tolist(),
            "reference": reference["per_n"]["963444"],
            "input_keys_verified": True,
        },
    )


def subset_masks(n_draws: int) -> np.ndarray:
    """Enumerate every nonempty subset in increasing size and lexicographic order."""
    masks = []
    for k in range(1, n_draws + 1):
        for subset in itertools.combinations(range(n_draws), k):
            mask = np.zeros(n_draws, dtype=np.float64)
            mask[list(subset)] = 1
            masks.append(mask)
    return np.stack(masks)


def bootstrap_r2(
    draws: np.ndarray, preds: np.ndarray, masks: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact re-centered pooled-R² bootstrap with the expensive reduction shared."""
    n_draws, n, d = draws.shape
    means = (counts @ draws.transpose(1, 0, 2).reshape(n, n_draws * d)).reshape(
        len(counts), n_draws, d
    ) / n
    point, boot, residuals = [], [], []
    for mask in masks:
        w = mask / mask.sum()
        target = np.einsum("k,knd->nd", w, draws)
        sse = np.square(preds - target).sum(axis=-1)
        y2 = np.square(target).sum(axis=-1)
        total = np.square(target - target.mean(axis=0)).sum()
        mean_boot = np.einsum("k,bkd->bd", w, means)
        total_boot = counts @ y2 - n * np.square(mean_boot).sum(axis=-1)
        if total <= 0 or np.any(total_boot <= 0):
            raise ValueError("degenerate R² denominator")
        point.append(1 - sse.sum(axis=-1) / total)
        boot.append(1 - (counts @ sse.T).T / total_boot)
        residuals.append(sse)
    return np.stack(point), np.stack(boot), np.stack(residuals)


def score_subsets(draws: np.ndarray, preds: np.ndarray, whiten, out: Path) -> dict:
    """Cache drawwise cross-products, then rank every subset on fixed pools."""
    n_draws, n, d = draws.shape
    masks = subset_masks(n_draws)
    view = FINAL.make_eval_view(draws[0], n, "keep_one")
    rows = view.pred_rows
    assert np.array_equal(rows, view.pool_rows)
    zdraws = whiten(draws.reshape(-1, d)).reshape(draws.shape)
    zpreds = whiten(preds.reshape(-1, d)).reshape(preds.shape)
    # [predictor, draw, query, candidate]: only 2 batched GEMMs for all subsets.
    dots = preds[:, None] @ draws.transpose(0, 2, 1)[None]
    zdots = zpreds[:, None] @ zdraws.transpose(0, 2, 1)[None]
    q2 = np.square(preds).sum(axis=-1)
    qnorm = np.sqrt(q2) + 1e-12
    zqnorm = np.linalg.norm(zpreds, axis=-1) + 1e-12
    correct, top5, r2_direct = [], [], []
    for si, mask in enumerate(masks):
        w = mask / mask.sum()
        target = np.einsum("k,knd->nd", w, draws)
        ztarget = np.einsum("k,knd->nd", w, zdraws)
        cross = np.einsum("k,pkij->pij", w, dots)
        zcross = np.einsum("k,pkij->pij", w, zdots)
        ynorm = np.linalg.norm(target, axis=-1) + 1e-12
        znorm = np.linalg.norm(ztarget, axis=-1) + 1e-12
        rawcos = cross / (qnorm[:, :, None] * ynorm[None, None, :])
        zcos = zcross / (zqnorm[:, :, None] * znorm[None, None, :])
        euclid = q2[:, :, None] + np.square(target).sum(-1)[None, None, :] - 2 * cross
        hits, hits5 = [], []
        for ai in range(len(preds)):
            sim = zcos[ai][np.ix_(rows, rows)]
            distances = (
                -FINAL.MB.csls_scores(sim, FINAL.K_CSLS),
                1 - sim,
                1 - rawcos[ai][np.ix_(rows, rows)],
                euclid[ai][np.ix_(rows, rows)],
            )
            ranks = np.stack([FINAL._strict_ranks(dist, view.true_idx) for dist in distances])
            hits.append(ranks <= 1)
            hits5.append(ranks <= 5)
            # Same actual scorer is checked against the canonical implementation.
            if si in (0, len(masks) - 1) and ai < 2:
                reference = FINAL.score_cell(preds[ai], target, view, whiten, seed=SEED)
                for mi, metric in enumerate(METRICS):
                    expected = reference[metric]["strict"]["acc_at_k"]
                    assert np.mean(ranks[mi] <= 1) == expected["1"], (si, ai, metric)
                    assert np.mean(ranks[mi] <= 5) == expected["5"], (si, ai, metric)
        correct.append(hits)
        top5.append(hits5)
        r2_direct.append([F79._recon_point(p, target)[0] for p in preds])
        LOG.info("subset %d/%d K=%d draws=%s", si + 1, len(masks), mask.sum(), np.flatnonzero(mask))
    correct, top5 = np.asarray(correct), np.asarray(top5)
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT)
    point, boot, sse = bootstrap_r2(draws, preds, masks, counts)
    np.testing.assert_allclose(point, r2_direct, rtol=0, atol=1e-10)
    n_retrieval = len(rows)
    retrieval_counts = rng.multinomial(
        n_retrieval, np.full(n_retrieval, 1 / n_retrieval), size=N_BOOT
    )
    retrieval_boot = np.einsum(
        "br,samr->samb", retrieval_counts / n_retrieval, correct, optimize=True
    )
    result = {
        "duplicate_audit": view.diagnostics,
        "chance_top1": 1 / n_retrieval,
        "chance_top5": 5 / n_retrieval,
        "all_subsets": aggregate(masks, point, boot, correct, top5, retrieval_boot),
        "fresh_only": aggregate(masks, point, boot, correct, top5, retrieval_boot, fresh_only=True),
        "subset_scores": [
            {
                "draw_indices": np.flatnonzero(mask).tolist(),
                "r2": point[i].tolist(),
                "top1": correct[i].mean(-1).tolist(),
                "top5": top5[i].mean(-1).tolist(),
            }
            for i, mask in enumerate(masks)
        ],
        "canonical_scorer_parity": "PASS: K=1 original and K=5; both fitted maps; four distances",
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "per_row_and_bootstrap.npz",
        masks=masks,
        r2=point,
        r2_boot=boot,
        sse=sse,
        top1=correct,
        top5=top5,
        retrieval_rows=rows,
        retrieval_boot=retrieval_boot,
        bootstrap_counts=counts,
        retrieval_bootstrap_counts=retrieval_counts,
    )
    return result


def interval(point: float, boot: np.ndarray, subset_values: np.ndarray | None = None) -> dict:
    """Keep context-sampling uncertainty separate from finite-bank subset spread."""
    result = {"mean": float(point), "ci95": np.quantile(boot, [0.025, 0.975]).tolist()}
    if subset_values is not None:
        result["subset_range"] = [float(subset_values.min()), float(subset_values.max())]
    return result


def aggregate(masks, point, boot, correct, top5, retrieval_boot, *, fresh_only=False) -> dict:
    """Average equally across all subsets of each K, with paired endpoint deltas."""
    result, r2_draws, ret_draws = {}, {}, {}
    for k in range(1, masks.shape[1] + 1):
        keep = masks.sum(1) == k
        if fresh_only:
            keep &= masks[:, 0] == 0
        if not keep.any():
            continue
        r2_draws[k] = boot[keep].mean(0)
        ret_draws[k] = retrieval_boot[keep].mean(0)
        cells = {}
        for ai, arm in enumerate(ARMS):
            cells[arm] = {
                "r2": interval(point[keep, ai].mean(), r2_draws[k][ai], point[keep, ai]),
                "retrieval": {},
            }
            for mi, metric in enumerate(METRICS):
                values = correct[keep, ai, mi].mean(-1)
                cells[arm]["retrieval"][metric] = interval(
                    values.mean(), ret_draws[k][ai, mi], values
                )
                cells[arm]["retrieval"][metric]["top5"] = float(top5[keep, ai, mi].mean())
        result[str(k)] = {"n_subsets": int(keep.sum()), "arms": cells}
    last = max(r2_draws)
    deltas = {}
    for ai, arm in enumerate(ARMS):
        deltas[arm] = {
            "r2": interval(
                result[str(last)]["arms"][arm]["r2"]["mean"]
                - result["1"]["arms"][arm]["r2"]["mean"],
                r2_draws[last][ai] - r2_draws[1][ai],
            ),
            "retrieval": {},
        }
        for mi, metric in enumerate(METRICS):
            delta = ret_draws[last][ai, mi] - ret_draws[1][ai, mi]
            p = (
                result[str(last)]["arms"][arm]["retrieval"][metric]["mean"]
                - result["1"]["arms"][arm]["retrieval"][metric]["mean"]
            )
            deltas[arm]["retrieval"][metric] = {
                "mean": float(p),
                "ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
            }
    return {"per_k": result, "endpoint_contrast": f"K={last} minus K=1", "endpoint_deltas": deltas}


def make_figure(
    summary: dict,
    stem: Path,
    *,
    include_baseline: bool = False,
    curve: dict | None = None,
    source_metadata: dict | None = None,
    paper: bool = False,
) -> None:
    """Render both outcome curves using the manuscript's shared visual system."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from explore_persona_space.analysis import c2a_plot_style as style

    style.set_c2a_style()
    fig, frac = style.c2a_figure("full", aspect=0.50 if curve is not None and not paper else 0.44)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.28 if curve is not None and not paper else 0.20,
        top=0.78,
        wspace=0.28,
    )
    roles = {"ridge": "linear", "mlp": "nonlinear", "identity_bias": "control"}
    labels = {"ridge": "Linear map", "mlp": "Nonlinear map", "identity_bias": "Identity + bias"}
    if paper:
        labels.update(ridge="Linear metamodel", mlp="Nonlinear metamodel")
    cells = summary["all_subsets"]["per_k"] if curve is None else curve
    ks = sorted(map(int, cells))
    for arm in ARMS if include_baseline else ARMS[:2]:
        series = style.ROLES[roles[arm]]
        for j, metric in enumerate(("r2", "top1")):
            vals = [cells[str(k)]["arms"][arm] for k in ks]
            vals = [v["r2"] if j == 0 else v["retrieval"]["whiten_csls"] for v in vals]
            y = np.array([v["mean"] for v in vals])
            lo, hi = np.array([v["ci95"] for v in vals]).T
            axes[j].plot(
                ks,
                y,
                color=series.color,
                marker=series.marker,
                linestyle="-" if j == 0 else "--",
                markerfacecolor=series.color if j == 0 else "white",
                label=labels[arm],
                linewidth=2,
                markersize=7,
            )
            axes[j].fill_between(ks, lo, hi, color=series.color, alpha=0.12, linewidth=0)
    for ax, label in zip(axes, ("Held-out $R^2$", "Top-1 retrieval"), strict=True):
        style.style_axis(ax)
        ax.set(
            xlabel="Rollouts averaged, $K$",
            ylabel=style.better_label(label),
            xticks=ks,
            xlim=(ks[0] - 0.3, ks[-1] + 0.3),
        )
    if include_baseline:
        axes[1].set_ylim(0, 1.03)
        axes[1].axhline(summary["chance_top1"], color=style.MUTED, linewidth=1, linestyle=":")
    else:
        axes[0].set_ylim(0.73, 0.88)
        axes[1].set_ylim(0.945, 0.995)
    axes[0].set_title("Variance explained", loc="left")
    axes[1].set_title("Answer identification", loc="left")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False
    )
    if paper:
        style.panel_header(axes[0], "A", "Reconstruction", kicker_y=1.12)
        style.panel_header(axes[1], "B", "Retrieval", kicker_y=1.12)
    else:
        fig.text(
            0.5,
            0.035,
            "Frozen maps · 1,000 contexts · 942 candidates · Whitened cosine + CSLS · 95% intervals",
            ha="center",
            color=style.MUTED,
            fontsize=14,
        )
    subject = "All subsets of five fixed on-policy draws; evaluation K only"
    if curve is not None:
        subject = "K=1–5: subset-averaged metrics from five draws; K=10: all ten draws"
        if not paper:
            fig.text(0.5, 0.085, subject, ha="center", color=style.MUTED, fontsize=14)
    exported = style.save_c2a_figure(
        fig,
        stem,
        title="Effect of rollout averaging on mapping quality",
        subject=subject,
        creator=Path(__file__).name,
        include_width=frac,
    )
    FINAL._write_json(
        stem.with_suffix(".meta.json"),
        {
            **exported["record"],
            "plotted_arms": list(ARMS if include_baseline else ARMS[:2]),
            "data": summary["all_subsets"] if curve is None else {"per_k": curve},
            "plotted_k": ks,
            "sources": source_metadata,
            "paper_style": paper,
            "outputs_sha256": {k: FINAL._sha256(exported[k]) for k in ("pdf", "png", "grayscale")},
        },
    )
    plt.close(fig)


def make_combined_figure(k5_path: Path, k10_path: Path, stem: Path, *, paper: bool = False) -> None:
    """Join verified, persisted results without inference or metric recomputation."""
    old, new = [json.loads(path.read_text()) for path in (k5_path, k10_path)]
    for key in ("model", "layer", "n_train", "n_test", "whitening", "provenance", "chance_top1"):
        if old[key] != new[key]:
            raise ValueError(f"K=5/K=10 source mismatch: {key}")
    if old["duplicate_audit"]["realized_n_pool"] != new["n_candidates"]:
        raise ValueError("Retrieval pool sizes differ")
    for key in ("n", "seed"):
        if old["bootstrap"][key] != new["paired_bootstrap"][key]:
            raise ValueError(f"Bootstrap {key} differs")
    cells = old["all_subsets"]["per_k"]
    if sorted(map(int, cells)) != [1, 2, 3, 4, 5]:
        raise ValueError("Expected complete original K=1–5 results")
    for arm in ARMS:
        a, b = cells["5"]["arms"][arm], new["targets"]["existing_K5"][arm]
        for metric in ("r2", *METRICS):
            x, y = (
                (a["r2"], b["r2"])
                if metric == "r2"
                else (a["retrieval"][metric], b["retrieval"][metric])
            )
            np.testing.assert_allclose(
                [x["mean"], *x["ci95"]], [y["mean"], *y["ci95"]], atol=1e-10, rtol=0
            )
    curve = {**cells, "10": {"n_subsets": 1, "arms": new["targets"]["K10"]}}
    sources = {
        "input_files": [
            {"path": str(path.resolve()), "sha256": FINAL._sha256(path)}
            for path in (k5_path, k10_path)
        ],
        "producer_sha256": FINAL._sha256(Path(__file__)),
        "shared_endpoint_parity": "PASS: all predictors/metrics, point estimates and intervals",
        "scope": "Frozen maps; K=1–5 uses subsets of five draws; K=10 uses all ten draws",
        "intervals": "Pointwise 95% context bootstrap, conditional on observed draws and fixed pool",
        "unmeasured_k": [6, 7, 8, 9],
        "chance_top1": new["chance_top1"],
    }
    make_figure(old, stem, curve=curve, source_metadata=sources, paper=paper)


def main() -> None:
    """Run the pinned, analysis-only follow-up and persist reproducible outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path)
    parser.add_argument(
        "--plot-through-k10", action="store_true", help="Plot persisted K=1–5,10 only"
    )
    parser.add_argument(
        "--paper-style", action="store_true", help="Use manuscript labels and caption-free canvas"
    )
    parser.add_argument(
        "--k5-summary",
        type=Path,
        default=ROOT / "eval_results/issue_1901/k_rollout_ablation/summary.json",
    )
    parser.add_argument(
        "--k10-summary",
        type=Path,
        default=ROOT / "eval_results/issue_1901/k10_rollout_ablation/summary.json",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "eval_results/issue_1901/k_rollout_ablation"
    )
    parser.add_argument("--figure", type=Path)
    parser.add_argument(
        "--tensor-out", type=Path, default=ROOT / "data/issue_1901/k_rollout_ablation"
    )
    args = parser.parse_args()
    if args.figure is None:
        name = "k_rollout_ablation_k1_to_10" if args.plot_through_k10 else "k_rollout_ablation"
        args.figure = ROOT / (
            "figures/paper/c1_rollout_count" if args.paper_style else f"figures/issue_1901/{name}"
        )
    if args.plot_through_k10:
        if args.paths is not None:
            parser.error("--paths does not apply to --plot-through-k10")
        make_combined_figure(args.k5_summary, args.k10_summary, args.figure, paper=args.paper_style)
        return
    if args.paper_style:
        parser.error("--paper-style requires --plot-through-k10")
    if args.paths is None:
        parser.error("--paths is required for scoring")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = time.time()
    paths = {k: Path(v) for k, v in json.loads(args.paths.read_text()).items()}
    draws, preds, provenance = load_inputs(paths)
    whiten, whitening = FINAL._whitener(paths["whiten"])
    result = score_subsets(draws, preds, whiten, args.tensor_out)
    for arm in ARMS[:2]:
        observed = result["all_subsets"]["per_k"]["5"]["arms"][arm]
        reference = provenance["reference"][arm]
        assert abs(observed["r2"]["mean"] - reference["r2"]) < 1e-10
        assert observed["retrieval"]["whiten_csls"]["mean"] == reference["top1"]
    result.update(
        {
            "issue": 1901,
            "analysis": "k-rollout-ablation",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "layer": 19,
            "n_train": 963444,
            "n_test": 1000,
            "arms": list(ARMS),
            "retrieval_metric_order": list(METRICS),
            "csls_k": FINAL.K_CSLS,
            "bootstrap": {
                "n": N_BOOT,
                "seed": SEED,
                "unit": "held-out context",
                "conditional_on": "five observed draws, fixed maps and retrieval pool",
                "r2_denominator": "recentered at each resampled test-set mean",
            },
            "scope": "evaluation K only; no refits; context arm; K>5 untested",
            "bootstrap_archive": {
                "hf_repo": "superkaiba1/explore-persona-space-data",
                "hf_path": "issue1901_k_rollout_ablation/analysis_tensors/per_row_and_bootstrap.npz",
                "sha256": FINAL._sha256(args.tensor_out / "per_row_and_bootstrap.npz"),
                "upload_verification": "Recorded separately in publication.json after upload",
            },
            "whitening": whitening,
            "provenance": provenance,
            "code_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "script_sha256": FINAL._sha256(Path(__file__)),
            "elapsed_seconds": time.time() - start,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "coverage": {
                "planned_subsets": 31,
                "realized_subsets": len(result["subset_scores"]),
                "planned_draw_vectors": 5000,
                "realized_draw_vectors": int(np.prod(draws.shape[:2])),
            },
            "paper_K5_parity": "PASS: R² within 1e-10 and top-1 exactly, both fitted maps",
        }
    )
    FINAL._write_json(args.out / "summary.json", result)
    make_figure(result, args.figure)
    make_figure(
        result, args.figure.with_name(args.figure.name + "_baselines"), include_baseline=True
    )
    LOG.info("COMPLETE elapsed=%.1fs output=%s", time.time() - start, args.out)


if __name__ == "__main__":
    main()

"""Verify and summarize completed necessity-rank cells; no model fits or inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    MUTED,
    ROLES,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)


def _ensure_repo_root_on_syspath():
    """Support both package and absolute-path execution without relying on cwd."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "scripts/issue2546_necessity_rank.py").is_file():
        raise RuntimeError(f"Wrong repository root: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_repo_root_on_syspath()
from scripts import issue2546_necessity_rank as analysis  # noqa: E402

METRICS = ("effective_rank_entropy", "participation_ratio", "stable_rank")
LABELS = {
    "no_think": "Thinking off · prompt",
    "context": "Thinking on · prompt",
    "end_of_thought": "Thinking on · CoT end",
}
MARKERS = {"no_think": "^", "context": "o", "end_of_thought": "s"}
LINES = {"no_think": "-", "context": "-", "end_of_thought": "-"}
CONTRASTS = {
    "thinking_mode": ("no_think", "end_of_thought"),
    "same_answer_state": ("context", "end_of_thought"),
}


def paired_bootstrap(rows_by_arm):
    """Corpus-stratified OOF-query bootstrap, pairing rows across all frozen maps."""
    arms = list(analysis.ARMS)
    ids = [r["row_id"] for r in rows_by_arm[arms[0]]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate held-out rows")
    groups = np.array([r["corpus"] for r in rows_by_arm[arms[0]]])
    columns = []
    for arm in arms:
        rows = rows_by_arm[arm]
        if [r["row_id"] for r in rows] != ids:
            raise ValueError("Unpaired bootstrap rows")
        columns.extend(
            [[float(r[key]) for r in rows] for key in ("sse_full", "sst_corpus", "sst_global")]
        )
    data = np.asarray(columns).T
    sums = np.zeros((analysis.BOOT_DRAWS, len(columns)), np.float64)
    for j, corpus in enumerate(np.unique(groups)):
        mask = groups == corpus
        counts = analysis.bootstrap_counts(
            groups[mask], analysis.BOOT_DRAWS, analysis.BOOT_SEED + j
        )
        sums += counts @ data[mask]
    samples = {
        arm: {
            base: 1 - sums[:, 3 * i] / sums[:, 3 * i + j]
            for j, base in ((1, "corpus"), (2, "global"))
        }
        for i, arm in enumerate(arms)
    }
    out = {
        "draws": analysis.BOOT_DRAWS,
        "seed": analysis.BOOT_SEED,
        "scope": "Conditional OOF-row uncertainty: fixed generations, folds, fits, layer and penalties. Corpus-stratified paired queries; not training or model-population uncertainty.",
        "arms": {
            arm: {
                base: {"ci95": np.quantile(draws, [0.025, 0.975]).tolist()}
                for base, draws in bases.items()
            }
            for arm, bases in samples.items()
        },
        "contrasts": {},
    }
    for name, (before, after) in CONTRASTS.items():
        out["contrasts"][name] = {}
        for base in ("corpus", "global"):
            draws = samples[after][base] - samples[before][base]
            out["contrasts"][name][base] = {
                "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
                "bootstrap_mean": float(draws.mean()),
                "draws": draws.tolist(),
            }
    return out


def predictive_area(curve):
    """Prior clipped normalized-R2 deficit area; descriptive, not a probability/CDF."""
    curve = np.asarray(curve, dtype=np.float64)
    if not np.isfinite(curve).all() or curve[-1] <= 0:
        raise ValueError("Predictive area requires a finite curve with positive full R2")
    return float(np.sum(1 - np.clip(curve[:-1] / curve[-1], 0, 1)))


def summarize(root, data_root):
    """Require fifteen coherent cells, verify original metric parity, and aggregate."""
    cells, rowbanks, recipes, source_hashes = {}, {}, [], {}
    for arm, (cell_name, _, _) in analysis.ARMS.items():
        cells[arm], rowbanks[arm] = [], []
        for fold in range(5):
            path = root / f"{arm}__fold{fold}.json"
            cell = json.loads(path.read_text())
            if cell["status"] != "complete" or cell["arm"] != arm or cell["fold"] != fold:
                raise ValueError(f"Incomplete/wrong cell {path}")
            if analysis.parent.sha256(path.with_suffix(".csv")) != cell["rows_sha256"]:
                raise ValueError(f"Row sidecar hash mismatch {path}")
            recipes.append(cell["cache_key"]["recipe"])
            cells[arm].append(cell)
            source_hashes[str(path)] = analysis.parent.sha256(path)
            with path.with_suffix(".csv").open() as handle:
                rowbanks[arm].extend(csv.DictReader(handle))
        rowbanks[arm].sort(key=lambda r: r["row_id"])
        original = json.loads((data_root / "allfit/results" / f"{cell_name}__a3.json").read_text())
        pooled = 1 - sum(r["all_rows_test_sse"] for r in cells[arm]) / sum(
            r["all_rows_sst_corpus"] for r in cells[arm]
        )
        np.testing.assert_allclose(
            pooled, original["subsets"]["all"]["r2_corpus"], rtol=0, atol=1e-9
        )
    if any(r != recipes[0] for r in recipes):
        raise ValueError("Mixed recipes")
    summary = {
        "status": "complete",
        "completed_cells": 15,
        "recipe": recipes[0],
        "subsets": {},
        "source_sha256": source_hashes,
        "original_full_panel_parity": "PASS: 3 arms, absolute R2 tolerance 1e-9",
    }
    for subset in analysis.SUBSETS:
        subset_rows = {
            arm: [r for r in rows if r["subset"] == subset] for arm, rows in rowbanks.items()
        }
        count = len(subset_rows["context"])
        if count != recipes[0]["subset_counts"][subset]:
            raise ValueError("Subset row coverage mismatch")
        out = {"n": count, "arms": {}, "contrasts": {}}
        for arm in analysis.ARMS:
            rows = [c["subsets"][subset] for c in cells[arm]]
            curves = np.array([r["test_sse_by_rank"] for r in rows])
            denom = sum(r["sst_corpus"] for r in rows)
            global_denom = sum(r["sst_global"] for r in rows)
            values = {
                "r2_full_corpus": 1 - float(curves[:, -1].sum()) / denom,
                "r2_full_global": 1 - float(curves[:, -1].sum()) / global_denom,
                "r2_identity_bias_corpus": 1 - sum(r["identity_bias_sse"] for r in rows) / denom,
                "rank_sensitivity": {},
                "retrieval_full": sum(r["full_retrieval_hits"] for r in rows) / count,
                "retrieval_rank10": sum(r["rank10_retrieval_hits"] for r in rows) / count,
                "retrieval_identity_bias": sum(r["identity_bias_retrieval_hits"] for r in rows)
                / count,
                "retrieval_pool": 6762,
                "retrieval_chance": 1 / 6762,
                "test_r2_corpus_by_rank": (1 - curves.sum(0) / denom).tolist(),
                "diversity": {},
                "per_corpus": {},
            }
            for tolerance in ("0.05", "0.1", "0.2"):
                ranks = [r["selected_ranks"][tolerance] for r in rows]
                sse = sum(curve[r] for curve, r in zip(curves, ranks, strict=True))
                values["rank_sensitivity"][tolerance] = {
                    "by_fold": ranks,
                    "median": float(np.median(ranks)),
                    "r2_corpus": 1 - sse / denom,
                    "test_extra_sse": sse / curves[:, -1].sum() - 1,
                }
            values["predictive_deficit_area_corpus"] = predictive_area(
                values["test_r2_corpus_by_rank"]
            )
            values["predictive_area_scope"] = (
                "Sum over ranks0..d-1 of 1-clip(R2(rank)/R2(full),0,1); "
                "corpus-training-mean baseline, descriptive only. Nonmonotone curve "
                "is not a CDF; this statistic is not an expected rank. No area bootstrap."
            )
            for space in ("input", "answer", "fitted_output", "fitted_over_answer"):
                values["diversity"][space] = {
                    m: {
                        "by_fold": [r["diversity"][space][m] for r in rows],
                        "median": float(np.median([r["diversity"][space][m] for r in rows])),
                    }
                    for m in METRICS
                }
            values["diversity"]["raw_input_pr"] = {
                "by_fold": [r["diversity"]["raw_input_participation_ratio"] for r in rows],
                "median": float(
                    np.median([r["diversity"]["raw_input_participation_ratio"] for r in rows])
                ),
            }
            for corpus in analysis.parent.CORPORA:
                cr = [r for r in subset_rows[arm] if r["corpus"] == corpus]
                if not cr:
                    raise ValueError("Empty corpus/subset")
                values["per_corpus"][corpus] = {
                    "n": len(cr),
                    "r2_corpus": 1
                    - sum(float(r["sse_full"]) for r in cr)
                    / sum(float(r["sst_corpus"]) for r in cr),
                    "retrieval_full": sum(int(r["hit_full"]) for r in cr) / len(cr),
                }
            original = json.loads(
                (data_root / "allfit/results" / f"{analysis.ARMS[arm][0]}__a3.json").read_text()
            )
            for target, source in (
                ("r2_full_corpus", "r2_corpus"),
                ("r2_full_global", "r2_global"),
                ("retrieval_full", "acc1"),
            ):
                np.testing.assert_allclose(
                    values[target], original["subsets"][subset][source], atol=1e-9, rtol=0
                )
            values["conditional_rank_bootstrap"] = rows[0]["conditional_rank_bootstrap"]
            out["arms"][arm] = values
        out["conditional_r2_bootstrap"] = paired_bootstrap(subset_rows)
        for name, (before, after) in CONTRASTS.items():
            a, b = out["arms"][before], out["arms"][after]
            draws = np.asarray(b["conditional_rank_bootstrap"]["draws"]) - np.asarray(
                a["conditional_rank_bootstrap"]["draws"]
            )
            out["contrasts"][name] = {
                "before": before,
                "after": after,
                "same_answers": name == "same_answer_state",
                "r2_change_corpus": b["r2_full_corpus"] - a["r2_full_corpus"],
                "r2_change_global": b["r2_full_global"] - a["r2_full_global"],
                "rank10_paired_changes": (
                    np.array(b["rank_sensitivity"]["0.1"]["by_fold"])
                    - a["rank_sensitivity"]["0.1"]["by_fold"]
                ).tolist(),
                "rank_bootstrap_change_ci95_fold0": np.quantile(draws, [0.025, 0.975]).tolist(),
                "rank_bootstrap_fraction_after_lower_fold0": float((draws < 0).mean()),
                "diversity_after_over_before": {
                    space: {
                        m: (
                            np.array(b["diversity"][space][m]["by_fold"])
                            / a["diversity"][space][m]["by_fold"]
                        ).tolist()
                        for m in METRICS
                    }
                    for space in ("input", "answer", "fitted_output", "fitted_over_answer")
                },
            }
        # Same-target control must really have identical answer-space measurements.
        for fold in range(5):
            for metric in METRICS:
                np.testing.assert_allclose(
                    out["arms"]["context"]["diversity"]["answer"][metric]["by_fold"][fold],
                    out["arms"]["end_of_thought"]["diversity"]["answer"][metric]["by_fold"][fold],
                    rtol=1e-12,
                    atol=1e-12,
                )
        summary["subsets"][subset] = out
        analysis.parent.write_json(root / f"{subset}__summary.json", out)
    analysis.parent.write_json(root / "summary.json", summary)
    return summary


def figures(summary, root, destination):
    """Render two simple, paper-style comparisons with dependent-fold ranges disclosed."""
    set_c2a_style()
    paths = {}
    for subset in analysis.SUBSETS:
        fig, frac = c2a_figure("full", aspect=0.43)
        axes = fig.subplots(1, 2)
        fig.subplots_adjust(left=0.085, right=0.98, bottom=0.18, top=0.70, wspace=0.31)
        for ax, kind, letter, title in zip(
            axes,
            ("rank", "diversity"),
            "AB",
            ("Prediction across map ranks", "Input, answer and map diversity"),
            strict=True,
        ):
            sub = summary["subsets"][subset]
            for arm, value in sub["arms"].items():
                color = MUTED if arm == "no_think" else ROLES["linear"].color
                if kind == "rank":
                    curve = np.asarray(value["test_r2_corpus_by_rank"])
                    ax.plot(
                        np.arange(1, len(curve)),
                        curve[1:],
                        color=color,
                        marker=MARKERS[arm],
                        linestyle=LINES[arm],
                        markevery=[3, 15, 63, 255, 1023, 4095],
                        markersize=6,
                        label=LABELS[arm],
                    )
                    for fold in range(5):
                        cell = json.loads((root / f"{arm}__fold{fold}.json").read_text())[
                            "subsets"
                        ][subset]
                        rank = cell["selected_ranks"]["0.1"]
                        ax.scatter(
                            rank,
                            1 - cell["test_sse_by_rank"][rank] / cell["sst_corpus"],
                            color=color,
                            marker=MARKERS[arm],
                            s=35,
                            zorder=4,
                        )
                else:
                    values = np.array(
                        [
                            value["diversity"][space]["effective_rank_entropy"]["by_fold"]
                            for space in ("input", "answer", "fitted_output")
                        ]
                    )
                    ax.plot(
                        range(3),
                        np.median(values, axis=1),
                        color=color,
                        marker=MARKERS[arm],
                        linestyle=LINES[arm],
                        label=LABELS[arm],
                    )
                    ax.vlines(range(3), values.min(1), values.max(1), color=color, alpha=0.6)
            subset_label = (
                "Correct only with thinking" if subset == "necessary" else "Correct in both modes"
            )
            panel_header(ax, letter, f"{subset_label} · n={sub['n']:,}", title)
            if kind == "rank":
                ax.set_xscale("log", base=2)
                ax.set_xticks(
                    [1, 4, 16, 64, 256, 1024, 4096],
                    labels=["1", "4", "16", "64", "256", "1,024", "4,096"],
                )
                ax.set_xlabel("Map rank")
                ax.set_ylabel("Held-out $R^2$")
                ax.set_xlim(1, 4096)
            else:
                ax.set_xticks(range(3), labels=["Input", "Answer", "Fitted output"])
                ax.set_ylabel("Effective rank")
                ax.set_ylim(bottom=0)
            style_axis(ax)
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.53, 1.0),
            ncol=3,
            frameon=False,
            handlelength=2.0,
        )
        stem = destination / f"qwen3_{subset}_rank_diversity"
        export = save_c2a_figure(
            fig,
            stem,
            title=f"Qwen3-8B {subset}: rank and diversity",
            subject="Matched layer24, three frozen linear maps",
            creator=Path(__file__).name,
            include_width=frac,
        )
        plt.close(fig)
        analysis.parent.write_json(
            stem.with_suffix(".meta.json"),
            {
                "render": export["record"],
                "source_sha256": analysis.parent.sha256(root / "summary.json"),
                "analysis_source": str(root / "summary.json"),
                "plotted_values": summary["subsets"][subset]["arms"],
                "uncertainty": "Rank panel dots: five dependent-fold validation-selected ranks evaluated on test. Diversity line: median and min-max over five overlapping training folds, not CI.",
                "targets": "Thinking-off predicts own answers; both thinking-on states predict identical thinking-on answers.",
                "output_sha256": {
                    k: analysis.parent.sha256(export[k]) for k in ("png", "pdf", "grayscale")
                },
                **analysis.parent.as_metadata_dict(
                    analysis.parent.git_provenance(cwd=Path(__file__).resolve().parents[1]),
                    phase="necessity-rank-plot",
                ),
            },
        )
        paths[subset] = str(export["png"])
    return paths


def main():
    """Verify, bootstrap, and export from explicitly supplied completed cell artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.results, args.data_root)
    print(json.dumps(figures(summary, args.results, args.figures)), flush=True)


if __name__ == "__main__":
    main()

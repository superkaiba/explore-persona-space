#!/usr/bin/env python3
"""Plot the full-data linear map across five already measured distractor pools.

Both source summaries use identical predictions, queries, five-answer means,
source-vector deduplication, whitening, and strict two-sided CSLS retrieval.
No inference, fitting or rescoring occurs; lines join measured points only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    ROLES,
    c2a_figure,
    save_c2a_figure,
    set_c2a_style,
    style_score_axis,
)


def sha256(path: Path) -> str:
    """Fingerprint an exact plotting input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_ref(path: Path) -> str:
    """Use portable repository references and absolute paths for external files."""
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved)


def collect(parent_path: Path, tenk_path: Path) -> dict:
    """Validate compatibility and extract the measured five-answer-mean cells."""
    parent, tenk = (json.loads(path.read_text()) for path in (parent_path, tenk_path))
    assert tenk["status"] == "complete"
    assert parent["data_revision"] == tenk["data_revision"]
    assert parent["whitening"] == tenk["whitening"]
    assert parent["bank"]["test_rows_sha256"] == tenk["test_rows_sha256"]
    assert parent["primary_operating_point"]["entry"] == "avg"
    assert tenk["n_rollouts"] == 5 and tenk["retrieval"]["csls_k"] == 10
    for key in (
        "pass_b",
        "whiten",
        "test_draws",
        "distractors",
        *(f"distr_draws_{i}" for i in range(4)),
    ):
        assert parent["input_sha256"][key] == tenk["input_sha256"][key], key
    assert parent["input_sha256"]["ridge_pred"] == tenk["input_sha256"]["pred_ridge_963444"]
    cells = []
    for nominal in (1000, 2000, 5000, 20000):
        key = f"ridge|avg|keep_one|pool_{nominal}"
        cell = parent["cells"][key]
        metric = cell["metrics"]["whiten_csls"]["strict"]
        assert metric["n_query"] == cell["view"]["realized_n_query"] == 942
        assert metric["n_pool"] == cell["view"]["realized_n_pool"] == nominal - 58
        cells.append((parent_path, key, metric))
    full = tenk["per_n"]["963444"]["ridge"]
    assert full["original_pool_metrics"]["whiten_csls"]["acc_at_k"] == cells[0][2]["acc_at_k"]
    cells.append(
        (tenk_path, "per_n/963444/ridge/metrics/whiten_csls", full["metrics"]["whiten_csls"])
    )
    rows = []
    for source, key, metric in cells:
        assert metric["n_query"] == 942
        lo, hi = (metric["acc1_ci95"][bound] for bound in ("lo", "hi"))
        top1, top5 = (metric["acc_at_k"][k] for k in ("1", "5"))
        assert 0 <= lo <= top1 <= hi <= 1 and top1 <= top5 <= 1
        rows.append(
            {
                "n_added_distractors": metric["n_pool"] - 942,
                "n_candidates": metric["n_pool"],
                "n_wrong_candidates_per_query": metric["n_pool"] - 1,
                "top1": top1,
                "top5": top5,
                "top1_ci95_lo": lo,
                "top1_ci95_hi": hi,
                "source": artifact_ref(source),
                "source_cell": key,
            }
        )
    rows.sort(key=lambda row: row["n_added_distractors"])
    assert [r["n_added_distractors"] for r in rows] == [0, 1000, 4000, 9058, 19000]
    return {
        "status": "complete",
        "n_queries": 942,
        "n_train": 963444,
        "predictor": "linear",
        "data_revision": tenk["data_revision"],
        "x_definition": "Additional distractor contexts beyond the 942 fixed query targets; each query also competes with 941 other query targets.",
        "protocol": "Five-answer means; exact source-vector deduplication; fixed training-only whitening; strict two-sided CSLS K=10.",
        "uncertainty": "Existing pointwise 95% query-bootstrap intervals (2000 draws), conditional on each fixed candidate pool; shown for top1 only. No top5 intervals were banked in the parent summary.",
        "line_interpretation": "Straight segments connect five measured points; no fitted curve or extrapolation.",
        "sources": {artifact_ref(path): sha256(path) for path in (parent_path, tenk_path)},
        "rows": rows,
    }


def render(data: dict, stem: Path) -> dict:
    """Export a directly labeled curve with distinct metric line/marker styles."""
    set_c2a_style()
    fig, fraction = c2a_figure("wide", aspect=0.55)
    ax = fig.subplots()
    fig.subplots_adjust(left=0.115, right=0.81, bottom=0.20, top=0.83)
    rows = data["rows"]
    x = np.array([r["n_added_distractors"] for r in rows])
    top1, top5 = (np.array([r[k] for r in rows]) for k in ("top1", "top5"))
    lo, hi = (np.array([r[f"top1_ci95_{k}"] for r in rows]) for k in ("lo", "hi"))
    color = ROLES["linear"].color
    style_score_axis(ax, y_min=0.80, y_max=1.015, y_step=0.05)
    ax.errorbar(
        x,
        top1,
        yerr=np.stack([top1 - lo, hi - top1]),
        color=color,
        linestyle="--",
        marker="o",
        mfc="white",
        ms=8,
        lw=2.6,
        elinewidth=1.1,
        capsize=4,
        zorder=3,
    )
    ax.plot(x, top5, color=color, linestyle=":", marker="s", mfc="white", ms=7, lw=2.2)
    ax.set_xlim(-450, 19800)
    ax.set_xticks([0, 5000, 10000, 15000, 19000], ["0", "5k", "10k", "15k", "19k"])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Added distractor contexts", labelpad=13)
    ax.set_ylabel("Retrieval accuracy ↑", labelpad=10)
    for values, label in ((top1, "Top-1"), (top5, "Top-5")):
        ax.annotate(
            f"{label}\n{100 * values[-1]:.1f}%",
            (x[-1], values[-1]),
            xytext=(12, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=color,
            fontsize=18,
            annotation_clip=False,
        )
    fig.text(0.115, 0.93, "Linear retrieval vs. distractors", fontsize=23, weight="bold", color=INK)
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Linear retrieval versus added distractors",
        subject=data["protocol"],
        creator="scripts/issue1901_distractor_curve.py",
        include_width=fraction,
    )
    plt.close(fig)
    return outputs


def main() -> None:
    """Write the exact plotted values, CSV, and standard figure exports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        type=Path,
        default=ROOT / "eval_results/issue_1901/singleturn_retrieval_final/summary.json",
    )
    parser.add_argument(
        "--tenk", type=Path, default=ROOT / "eval_results/issue_1901/retrieval_10k/summary.json"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "eval_results/issue_1901/retrieval_10k/distractor_curve",
    )
    parser.add_argument(
        "--stem", type=Path, default=ROOT / "figures/issue_1901/retrieval_10k/distractor_curve"
    )
    args = parser.parse_args()
    data = collect(args.parent, args.tenk)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = args.out_dir / "summary.json"
    source.write_text(json.dumps(data, indent=2) + "\n")
    with (args.out_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data["rows"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data["rows"])
    outputs = render(data, args.stem)
    metadata = {
        "source": artifact_ref(source),
        "source_sha256": sha256(source),
        "plotted": data["rows"],
        "render": outputs["record"],
    }
    args.stem.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: str(outputs[k]) for k in ("png", "pdf", "grayscale")}, indent=2))


if __name__ == "__main__":
    main()

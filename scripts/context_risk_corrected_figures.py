#!/usr/bin/env python3
"""Plot complete observed counts; never turn failed prediction fits into zeros."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper  # noqa: E402


def plot(result_path: Path, output_dir: Path) -> dict:
    result = json.loads(result_path.read_text())
    audit_path = result_path.parent / "native_log_audit.json"
    audit = json.loads(audit_path.read_text())
    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if (
        not audit["audit_passed"]
        or not audit["execution_passed"]
        or audit["run_result_sha256"] != result_sha
        or result["realized_rollouts"] != 480
        or result["requested_rollouts"] != 480
        or result["technical_errors"] != 0
    ):
        raise ValueError("This count figure requires the complete, uncensored audited full run")
    conditions = ("original", "conflicting", "oneoff")
    tasks = sorted(
        {row["task_id"] for row in result["contexts"]}, key=lambda s: int(s.split("_")[-1])
    )
    contexts = {(row["task_id"], row["condition"]): row for row in result["contexts"]}
    if len(tasks) != 20 or len(result["contexts"]) != 60 or len(contexts) != 60:
        raise ValueError("Expected 20 tasks and 60 unique contexts")
    if any(row["n"] != 8 or row["errors"] != 0 for row in contexts.values()):
        raise ValueError("Do not plot missing or censored cells as zero")
    matrix = np.array(
        [[contexts[task, condition]["passed"] for condition in conditions] for task in tasks]
    )
    if np.any(matrix < 0) or np.any(matrix > 8):
        raise ValueError("Impossible pass count")
    totals = matrix.sum(axis=0)
    raw_path = result_path.parent / "rollouts.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    if len(rows) != 480 or len({(r["id"], r["epoch"]) for r in rows}) != 480:
        raise ValueError("Raw coverage differs from the summary")
    raw_counts = Counter((r["task_id"], r["condition"]) for r in rows if r["score"] == "C")
    if any(raw_counts[key] != value["passed"] for key, value in contexts.items()):
        raise ValueError("Raw positive counts differ from per-context summaries")
    for condition, total in zip(conditions, totals, strict=True):
        summary = result["by_condition"][condition]
        if summary != {"n": 160, "passed": int(total), "errors": 0}:
            raise ValueError("Condition totals differ from the verified per-context counts")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9, 6.2), gridspec_kw={"width_ratios": [1, 1.4]}
    )
    labels = ["Original", "Conflicting", "One-off"]
    left.barh(labels, totals, color=["#0072B2", "#D55E00", "#D55E00"], height=0.5)
    left.scatter(totals, range(3), color="#222222", s=20, zorder=3)
    for index, total in enumerate(totals):
        left.text(float(total) + 4, index, f"{total}/160", va="center")
    left.invert_yaxis()
    left.set_xlim(0, 180)
    left.set_xticks([0, 40, 80, 120, 160])
    left.set_xlabel("Successful rollouts (count)")
    left.set_title("A  All observed rollouts", loc="left", fontweight="bold")
    left.spines[["top", "right"]].set_visible(False)
    right.imshow(matrix, vmin=0, vmax=8, cmap="cividis", aspect="auto")
    right.set_xticks(range(3), labels)
    right.set_yticks(range(20), tasks)
    right.tick_params(axis="y", labelsize=8)
    right.set_title("B  Passes out of eight per context", loc="left", fontweight="bold")
    for row in range(20):
        for column in range(3):
            count = int(matrix[row, column])
            right.text(
                column,
                row,
                str(count),
                ha="center",
                va="center",
                color="white" if count < 4 else "black",
                fontsize=9,
            )
    right.spines[:].set_visible(False)
    fig.tight_layout(w_pad=3)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = savefig_paper(fig, "observed_outcomes", dir=output_dir)
    meta = json.loads(written["meta"].read_text())
    if not meta.get("text") or not meta.get("points"):
        raise ValueError("Figure export must preserve rendered text and plotted data")
    meta["context_counts"] = [
        {"task_id": task, "condition": condition, "successes": int(matrix[i, j]), "rollouts": 8}
        for i, task in enumerate(tasks)
        for j, condition in enumerate(conditions)
    ]
    # The shared artist extractor treats horizontal bar centers as x values
    # and omits image cells. Persist the exact plotted counts explicitly.
    meta["points"] = [
        {
            "panel": "All observed rollouts",
            "task_id": "all",
            "condition": condition,
            "successes": int(total),
            "rollouts": 160,
        }
        for condition, total in zip(conditions, totals, strict=True)
    ] + [{"panel": "Per-context outcomes", **row} for row in meta["context_counts"]]
    meta["total_points"] = len(meta["points"])
    meta["n_series"] = 2
    written["meta"].write_text(json.dumps(meta, indent=2) + "\n")
    svg_path = output_dir / "observed_outcomes.svg"
    fig.savefig(svg_path)
    paths = [*written.values(), svg_path]
    plt.close(fig)
    provenance = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_result_sha256": result_sha,
        "rollouts_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "scope": "Exact observed finite-sample counts, no population confidence interval or prediction score.",
        "conditions": list(conditions),
        "task_ids": tasks,
        "counts": matrix.tolist(),
        "totals": totals.tolist(),
        "figures": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    (output_dir / "figure_data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.run_result, args.output_dir), indent=2))

"""Export the completed chat-only reproduction as a small CSV and comparison report."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

import issue2588_chat_plot as plot
from explore_persona_space.atomic_io import atomic_replace, write_json_atomic


def number(value, name: str) -> float:
    """Never turn a missing, boolean, or nonfinite metric into a reported result."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid {name}")
    return float(value)


def summary_row(rec: dict) -> dict:
    """Read the frozen producer's actual metric layout, with denominator checks."""
    metrics = rec["parent_selected_layer_metrics"]
    counts = rec["realized_rows"]
    for split, key, planned in (
        ("train_10k", "tr", 10000),
        ("val_400", "val", 400),
        ("test_1000", "te", 1000),
    ):
        if (
            type(counts[split]) is not int
            or not 0 < counts[split] <= planned
            or metrics["n"][key] != counts[split]
        ):
            raise ValueError("Capture/fit count mismatch")
    knn = metrics["knn_test"]
    pool = knn["_meta"]["n_pool"]
    if type(pool) is not int or pool != counts["test_1000"]:
        raise ValueError("Retrieval pool differs from the held-out target count")
    row = {
        "cell": rec["cell"],
        "input_position": rec["input_position"],
        "selected_layer": rec["layer_star"],
        "dimension": rec["dimension"],
        "selected_lambda": number(rec["selected_lambda"], "lambda"),
        "n_train": counts["train_10k"],
        "n_validation": counts["val_400"],
        "n_test": counts["test_1000"],
        "full_test_r2": number(rec["full_test_r2"], "full R2"),
        "parent_full_test_r2": number(metrics["test_r2"], "parent full R2"),
        "identity_bias_test_r2": number(metrics["floors_test_r2"]["identity_bias"], "baseline R2"),
        "selected_rank": rec["rank"],
        "rank_relative_to_dimension": rec["rank"] / rec["dimension"],
        "selected_rank_test_r2": number(rec["selected_rank_test_r2"], "reduced-rank R2"),
        "input_participation_ratio": number(rec["participation_ratio_x_at_star"], "input PR"),
        "retrieval_pool": pool,
        "retrieval_chance": 1.0 / pool,
    }
    if abs(row["full_test_r2"] - row["parent_full_test_r2"]) > 3e-4:
        raise ValueError("Parent reconstruction parity failed")
    for method in ("ridge", "identity_bias"):
        for metric in ("cosine", "euclidean"):
            value = number(knn[method][metric]["acc_at_k"]["1"], "retrieval accuracy")
            if not 0 <= value <= 1:
                raise ValueError("Retrieval accuracy outside [0,1]")
            row[f"{method}_{metric}_top1"] = value
    repeat = rec["ceiling_two_draw_at_star"]
    repeat_retrieval = rec["ceiling_retrieval_at_star"]
    if repeat is None or repeat_retrieval is None:
        raise ValueError("Missing planned repeat-answer diagnostics")
    repeat_pool = repeat_retrieval["n_pool"]
    if (
        type(repeat_pool) is not int
        or not 2 <= repeat_pool <= 1000
        or repeat["n"] != repeat_pool
        or repeat_retrieval["seed_pair"] != [43, 44]
        or abs(number(repeat_retrieval["chance"], "repeat chance") - 1 / repeat_pool) > 1e-12
    ):
        raise ValueError("Repeat-answer denominator or seed mismatch")
    row.update(
        repeat_answer_n=repeat_pool,
        repeat_answer_weighted_pearson=number(repeat["ceiling"], "repeat Pearson"),
        repeat_answer_cosine_top1=number(repeat_retrieval["ceiling_acc1_cos"], "repeat retrieval"),
        repeat_answer_chance=1 / repeat_pool,
    )
    if not -1 <= row["repeat_answer_weighted_pearson"] <= 1:
        raise ValueError("Repeat-answer correlation outside [-1,1]")
    if not 0 <= row["repeat_answer_cosine_top1"] <= 1:
        raise ValueError("Repeat-answer retrieval outside [0,1]")
    return row


def export(source: Path, destination: Path, figure_url: str) -> dict:
    """Export only complete real-run-shaped inputs; this never fits or launches jobs."""
    url = urlsplit(figure_url)
    if url.scheme != "https" or not url.netloc or url.username or url.password:
        raise ValueError("A browser-accessible HTTPS figure URL is required")
    maps = plot.load_maps(source)
    revisions = {rec["provenance"]["hf_revision"] for rec in maps}
    if len(revisions) != 1 or not re.fullmatch(r"[0-9a-f]{40}", next(iter(revisions))):
        raise ValueError("Both arms must use the same immutable analysis-input revision")
    rows = [summary_row(rec) for rec in maps]
    a, b = rows
    delta_rank = b["selected_rank"] - a["selected_rank"]
    direction = "lower" if delta_rank < 0 else "higher" if delta_rank > 0 else "unchanged"
    outcome = (
        f"The end-of-thought map has {direction} operational rank "
        f"({a['selected_rank']} → {b['selected_rank']}; thinking minus no-thinking "
        f"{delta_rank:+d}). Full-map held-out R² changes from {a['full_test_r2']:.4f} "
        f"to {b['full_test_r2']:.4f} ({b['full_test_r2'] - a['full_test_r2']:+.4f})."
    )
    table = ["| Metric | No thinking: prompt last | Thinking: end of thought |", "|---|---:|---:|"]
    fields = [
        ("Selected layer (zero-based)", "selected_layer", "d"),
        ("Full-map test R²", "full_test_r2", ".4f"),
        ("Identity + learned bias test R²", "identity_bias_test_r2", ".4f"),
        ("Operational rank", "selected_rank", "d"),
        ("Rank / 4096", "rank_relative_to_dimension", ".4f"),
        ("Test R² at selected rank", "selected_rank_test_r2", ".4f"),
        ("TRAIN-input participation ratio", "input_participation_ratio", ".2f"),
        ("Map cosine top-1 retrieval", "ridge_cosine_top1", ".2%"),
        ("Map Euclidean top-1 retrieval", "ridge_euclidean_top1", ".2%"),
        ("Identity + bias cosine top-1", "identity_bias_cosine_top1", ".2%"),
        ("Identity + bias Euclidean top-1", "identity_bias_euclidean_top1", ".2%"),
        ("Test retrieval pool", "retrieval_pool", "d"),
        ("Chance top-1", "retrieval_chance", ".3%"),
        ("Repeat-answer aligned pairs", "repeat_answer_n", "d"),
        ("Repeat-answer weighted Pearson", "repeat_answer_weighted_pearson", ".4f"),
        ("Repeat-answer cosine top-1", "repeat_answer_cosine_top1", ".2%"),
        ("Repeat-answer chance top-1", "repeat_answer_chance", ".3%"),
    ]
    table += [
        f"| {label} | {format(a[key], spec)} | {format(b[key], spec)} |"
        for label, key, spec in fields
    ]
    coverage = [
        "| Split | Planned per arm | No thinking retained | Thinking retained |",
        "|---|---:|---:|---:|",
    ]
    for label, key, planned in (
        ("Train", "n_train", 10000),
        ("Validation", "n_validation", 400),
        ("Test", "n_test", 1000),
    ):
        coverage.append(f"| {label} | {planned} | {a[key]} | {b[key]} |")
    report = (
        "\n\n".join(
            [
                "# Qwen3-8B chat-data mapping/rank reproduction",
                outcome,
                f"[Rank-curve figure]({figure_url})",
                "\n".join(table),
                "## Coverage",
                "\n".join(coverage),
                "Retained counts are condition-specific; missing responses are not imputed. "
                "Repeat-answer diagnostics use the intersection of valid seed-43 and seed-44 "
                "answers, rather than treating the two draws as independent models. The CSV "
                "also records each selected ridge penalty and the original producer's R².",
                "## Method and limitations",
                "This reproduces the earlier long-cap chat-data panel for one model. Each "
                "condition uses its own generated answers. Layers are selected by validation "
                "cosine top-1 retrieval; ridge penalties by validation R². Training-output PCA "
                "defines the nested maps. Operational rank is the smallest rank with validation "
                "SSE no more than 10% above that condition's own full map; test performance "
                "does not select the rank. The full rank-0–4096 validation/test curves remain "
                "in the source JSONs. Reconstruction is checked against the producer within "
                "3e-4 R² and full-rank recovery within 1e-4.",
                "Different targets and separately selected layers make this a descriptive "
                "comparison, not an isolated causal effect of reasoning. Rank thresholds are "
                "relative to each map's own performance, not a common absolute R² target. "
                "Neither map rank nor input participation ratio measures the total information "
                "in reasoning. Repeat-answer correlations are reliability diagnostics, not a "
                "validated bound on CoT-conditioned prediction. No population p-value or "
                "confidence interval is reported for this single model.",
                "The frozen LMSYS-Chat-1M split contains 13 exact prompt strings in both "
                "validation and test (24 validation and 60 test rows before exclusions); "
                "training has no exact-text overlap with either. These are not wholly "
                "independent validation/test prompts. Capture uses the inherited "
                "teacher-forced replay of persisted on-policy text, not token-identical "
                "replay of sampled completion token IDs. Thinking has a 32768-token cap; "
                "the model context window prevents the inherited meaningful cap increase. "
                "No split changes, extra models, GPQA, judges, or inferential sweep were added.",
                "## Provenance",
                f"Immutable capture/fit revision: `{next(iter(revisions))}`. Exact model, "
                "manifest, scientific-source, rank-consumer and input-file hashes are in "
                "`rank_a.json` and `rank_b.json`; report input hashes are in `report.meta.json`.",
            ]
        )
        + "\n"
    )
    destination.mkdir(parents=True, exist_ok=True)
    with atomic_replace(destination / "comparison.csv") as temporary:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(a))
            writer.writeheader()
            writer.writerows(rows)
    with atomic_replace(destination / "comparison.md") as temporary:
        temporary.write_text(report)
    metadata = {
        "schema": "issue2588_chat_comparison_v1",
        "figure_url": figure_url,
        "source_sha256": {
            f"rank_{arm}.json": plot.digest(source / f"rank_{arm}.json") for arm in ("a", "b")
        },
        "exporter_sha256": plot.digest(Path(__file__)),
        "hf_revision": next(iter(revisions)),
        "rows": rows,
        "rank_change_thinking_minus_no_thinking": delta_rank,
        "outputs_sha256": {
            name: plot.digest(destination / name) for name in ("comparison.csv", "comparison.md")
        },
    }
    write_json_atomic(destination / "report.meta.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--figure-url", required=True)
    args = parser.parse_args()
    export(args.source_dir, args.out_dir, args.figure_url)


if __name__ == "__main__":
    main()

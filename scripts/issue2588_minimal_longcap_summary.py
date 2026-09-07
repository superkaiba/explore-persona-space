#!/usr/bin/env python3
"""Summarize the issue-2588 minimal long-cap robustness experiment.

The approved follow-up reruns only the three 27B thinking cells whose original
GPQA generations were heavily truncated.  This script compares the original
and long-cap artifacts at immutable Hugging Face revisions, then combines the
comparison with the long-cap matched-training-size ranks.  It downloads only
small JSON metadata; no activations or model weights are read.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

# Keep this plot-only analysis polite on the shared VM.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_tree  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    GRID,
    MUTED,
    PAPER,
    SEAM,
    STYLE_VERSION,
    save_c2a_figure,
    set_c2a_style,
)

HF_REPO = "superkaiba1/explore-persona-space-data"
BASELINE_REVISION = "74bb871a5edf1afe777ac9b64a4e2fec5e9947c2"
BASELINE_PREFIX = "issue2588_capability_panel"
LONG_PREFIX = "issue2588_capability_panel_cap_long"
TARGETS = (
    {"cell": "q35_27b_b", "model_key": "q35_27b", "label": "Qwen3.5 27B"},
    {"cell": "q36_27b_b", "model_key": "q36_27b", "label": "Qwen3.6 27B"},
    {"cell": "q38_27b_b", "model_key": "q38_27b", "label": "Qwen3.8 27B"},
)
MINIMAL_MAP_CELLS = (
    "q35_27b_a",
    "q35_27b_b",
    "q36_27b_a",
    "q36_27b_b",
    "q38_27b_a",
    "q38_27b_b",
    "q25_32b_a",
    "q3_32b_a",
    "q3_32b_b",
    "qwq_32b_b",
    "o3_32b_t_b",
)
GENERIC_STAGES = ("train_10k", "val_400", "test_1000")
GPQA_STAGES = tuple(f"gpqa_s{seed}" for seed in range(42, 47))
DEFAULT_BASELINE_MAPPING = ROOT / "eval_results/issue_2588/mapping_rank_vs_capability.json"
DEFAULT_LONG_MAPPING = ROOT / "eval_results/issue_2588/cap_long/mapping_rank_vs_capability.json"
DEFAULT_MATCHED = ROOT / "eval_results/issue_2588/cap_long/matched_n_ranks.json"
DEFAULT_OUT = ROOT / "eval_results/issue_2588/cap_long/minimal_longcap_summary.json"
DEFAULT_FIGURE = ROOT / "figures/issue_2588/cap_long/minimal_longcap_summary"
PUBLIC_URL = "https://eps.superkaiba.com/tasks/2588/figure/minimal_longcap_summary.png"

BASELINE_COLOR = MUTED
LONG_COLOR = "#176B87"
MATCHED_COLOR = "#C4553D"


def _download_json(prefix: str, relpath: str, revision: str) -> dict[str, Any]:
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"{prefix}/{relpath}",
        repo_type="dataset",
        revision=revision,
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_stage(
    cap_reports: list[dict[str, Any]], drop_payloads: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate cap and parser outcomes, rejecting ambiguous row identities."""
    if len(cap_reports) != len(drop_payloads) or not cap_reports:
        raise ValueError("cap reports and drop payloads must be non-empty and aligned")
    expected = 0
    pre_regen_cap_hits = 0
    effective_cap_hits = 0
    row_ids: set[str] = set()
    reasons: Counter[str] = Counter()
    per_stage: list[dict[str, Any]] = []
    for report, dropped in zip(cap_reports, drop_payloads, strict=True):
        n = int(report["n"])
        stage = str(report["stage"])
        rows = dropped["drops"]
        ids = [str(row["row_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{stage}: duplicate dropped row ids")
        overlap = row_ids.intersection(ids)
        if overlap:
            raise ValueError(
                f"{stage}: dropped row ids overlap another stage: {sorted(overlap)[:3]}"
            )
        if len(ids) > n:
            raise ValueError(f"{stage}: {len(ids)} drops exceed n={n}")
        row_ids.update(ids)
        stage_reasons = Counter(str(row["reason"]) for row in rows)
        reasons.update(stage_reasons)
        pre_hits = int(report["cap_hits"])
        post_hits = int(report.get("post_regen_cap_hits", pre_hits))
        expected += n
        pre_regen_cap_hits += pre_hits
        effective_cap_hits += post_hits
        per_stage.append(
            {
                "stage": stage,
                "expected": n,
                "retained": n - len(ids),
                "dropped": len(ids),
                "drop_reasons": dict(sorted(stage_reasons.items())),
                "cap": int(report["cap"]),
                "cap_hits_pre_regen": pre_hits,
                "cap_hits_effective": post_hits,
                "regeneration_ran": bool(report.get("regen_ran", False)),
            }
        )
    dropped_n = len(row_ids)
    return {
        "expected": expected,
        "retained": expected - dropped_n,
        "retained_fraction": (expected - dropped_n) / expected,
        "dropped": dropped_n,
        "drop_reasons": dict(sorted(reasons.items())),
        "cap_hits_pre_regen": pre_regen_cap_hits,
        "cap_hits_effective": effective_cap_hits,
        "cap_hit_fraction_effective": effective_cap_hits / expected,
        "per_stage": per_stage,
    }


def _stage_summary(
    *, prefix: str, revision: str, model_key: str, stages: tuple[str, ...]
) -> dict[str, Any]:
    reports = []
    drops = []
    for stage in stages:
        base = f"{model_key}/think"
        reports.append(
            _download_json(
                prefix,
                f"{base}/raw_completions/{stage}/cap_hit_report.json",
                revision,
            )
        )
        drops.append(_download_json(prefix, f"{base}/parsed/{stage}_drops.json", revision))
    return summarize_stage(reports, drops)


def _map_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index = {str(record["cell"]): record for record in payload["maps"]}
    if len(index) != len(payload["maps"]):
        raise ValueError("mapping payload contains duplicate cells")
    return index


def _validate_source_map(
    mapping: dict[str, Any], matched: dict[str, Any], revision: str
) -> dict[str, Any]:
    """Verify a reused map against its immutable final source revision."""
    cell = str(mapping["cell"])
    position = str(mapping["input_position"])
    fit_relpath = f"fits/{cell}/fits_{position}.json"
    fit = _download_json(LONG_PREFIX, fit_relpath, revision)
    layer = int(mapping["layer_star"])
    if int(fit["layer_star"]) != layer:
        raise ValueError(f"{cell}: mapping layer {layer} != source layer {fit['layer_star']}")
    selected = fit["layers"][str(layer)]
    checks = {
        "selected_lambda": (
            float(mapping["selected_lambda"]),
            float(selected["fit_meta"]["selected_lambda"]),
        ),
        "validation_r2": (
            float(mapping["reconstruction_parity"]["expected_validation_r2"]),
            float(selected["fit_meta"]["val_r2_at_selected"]),
        ),
        "test_r2": (
            float(mapping["mapping_performance"]["test_r2"]),
            float(selected["test_r2"]),
        ),
        "production_n": (
            float(matched["production_n"]),
            float(selected["fit_meta"]["n_train"]),
        ),
    }
    mismatches = {
        name: {"mapping": left, "source": right}
        for name, (left, right) in checks.items()
        if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-7)
    }
    if mismatches:
        raise ValueError(f"{cell}: final-revision fit mismatch: {mismatches}")

    generation_dir = "nothink" if mapping["arm"] == "no-thinking" else "think"
    model_key = cell.rsplit("_", 1)[0]

    def capture_count(stage: str) -> tuple[str, int]:
        prefix = (
            f"{LONG_PREFIX}/{model_key}/{generation_dir}/analysis_tensors/"
            f"capture/{stage}/L{layer:02d}"
        )
        entries = list_repo_tree(
            HF_REPO,
            path_in_repo=prefix,
            recursive=False,
            revision=revision,
            repo_type="dataset",
        )
        count = sum(entry.path.endswith(".npz") for entry in entries)
        if count < 1:
            raise ValueError(f"{cell}: no selected-layer capture shards under {prefix}")
        return stage, count

    with ThreadPoolExecutor(max_workers=3) as pool:
        capture_counts = dict(pool.map(capture_count, GENERIC_STAGES))
    return {
        "cell": cell,
        "fit_path": f"{LONG_PREFIX}/{fit_relpath}",
        "computed_from_revision": mapping.get("hf_revision"),
        "verified_reproducible_at_revision": revision,
        "layer_star": layer,
        "selected_lambda": checks["selected_lambda"][0],
        "validation_r2": checks["validation_r2"][0],
        "test_r2": checks["test_r2"][0],
        "production_n": int(checks["production_n"][0]),
        "selected_layer_capture_shards": capture_counts,
        "all_checks_passed": True,
    }


def validate_long_sources(
    long_maps: dict[str, dict[str, Any]], matched_summary: dict[str, Any], revision: str
) -> list[dict[str, Any]]:
    missing_maps = sorted(set(MINIMAL_MAP_CELLS) - long_maps.keys())
    if missing_maps:
        raise ValueError(f"long mapping payload misses minimal maps: {missing_maps}")
    audits = []
    for cell in MINIMAL_MAP_CELLS:
        matched_key = f"{cell}__{long_maps[cell]['input_position']}"
        if matched_key not in matched_summary:
            raise ValueError(f"matched-n payload misses minimal map {matched_key}")
        audits.append(_validate_source_map(long_maps[cell], matched_summary[matched_key], revision))
    return audits


def summarize_matched_controls(
    long_maps: dict[str, dict[str, Any]], matched_summary: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for cell in MINIMAL_MAP_CELLS:
        mapping = long_maps[cell]
        key = f"{cell}__{mapping['input_position']}"
        matched = matched_summary[key]["by_n"]["4500"]
        rows.append(
            {
                "cell": cell,
                "model": mapping["model"],
                "family": mapping["family"],
                "arm": mapping["arm"],
                "aa_index": float(mapping["aa_index"]),
                "n_train": 4500,
                "n_fits": int(matched["n_fits"]),
                "mean_rrr_rank_rel10": float(matched["mean_rank_rel10"]),
                "min_rrr_rank_rel10": int(matched["min_rank_rel10"]),
                "max_rrr_rank_rel10": int(matched["max_rank_rel10"]),
            }
        )
    trends = {}
    for arm in ("no-thinking", "end-of-thought"):
        qwen = [row for row in rows if row["arm"] == arm and row["family"] == "Qwen h=5120 column"]
        result = spearmanr(
            [row["aa_index"] for row in qwen],
            [row["mean_rrr_rank_rel10"] for row in qwen],
        )
        trends[arm] = {
            "n": len(qwen),
            "rho": float(result.statistic),
            "p_asymptotic": float(result.pvalue),
            "scope": "same-width Qwen column at matched n=4500",
        }
    return rows, trends


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {"head": head, "dirty": dirty}


def build_summary(
    *,
    baseline_mapping_path: Path,
    long_mapping_path: Path,
    matched_path: Path,
    long_revision: str,
) -> dict[str, Any]:
    if not long_revision or long_revision == "main" or len(long_revision) != 40:
        raise ValueError("--long-revision must be an immutable 40-character commit SHA")
    baseline_payload = json.loads(baseline_mapping_path.read_text(encoding="utf-8"))
    long_payload = json.loads(long_mapping_path.read_text(encoding="utf-8"))
    matched_payload = json.loads(matched_path.read_text(encoding="utf-8"))
    baseline_maps = _map_index(baseline_payload)
    long_maps = _map_index(long_payload)
    source_audit = validate_long_sources(long_maps, matched_payload["summary"], long_revision)
    matched_controls, matched_trends = summarize_matched_controls(
        long_maps, matched_payload["summary"]
    )
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        cell = target["cell"]
        if cell not in baseline_maps or cell not in long_maps:
            raise ValueError(f"missing target map {cell} from a mapping payload")
        matched_key = f"{cell}__cot_boundary"
        if matched_key not in matched_payload["summary"]:
            raise ValueError(f"missing matched-n summary {matched_key}")
        matched = matched_payload["summary"][matched_key]
        if "4500" not in matched["by_n"]:
            raise ValueError(f"{matched_key}: matched n=4500 is missing")
        profiles: dict[str, Any] = {}
        for name, prefix, revision, mapping in (
            ("baseline", BASELINE_PREFIX, BASELINE_REVISION, baseline_maps[cell]),
            ("long", LONG_PREFIX, long_revision, long_maps[cell]),
        ):
            fit = _download_json(
                prefix,
                f"fits/{cell}/fits_cot_boundary.json",
                revision,
            )
            if int(fit["layer_star"]) != int(mapping["layer_star"]):
                raise ValueError(f"{cell}/{name}: mapping and source select different layers")
            selected = fit["layers"][str(fit["layer_star"])]
            identity_retrieval = float(
                selected["knn_test"]["identity_bias"]["cosine"]["acc_at_k"]["1"]
            )
            transfer = _download_json(
                prefix,
                f"fits/{cell}/gpqa_transfer_cot_boundary.json",
                revision,
            )
            profiles[name] = {
                "revision": revision,
                "generic": _stage_summary(
                    prefix=prefix,
                    revision=revision,
                    model_key=target["model_key"],
                    stages=GENERIC_STAGES,
                ),
                "gpqa": _stage_summary(
                    prefix=prefix,
                    revision=revision,
                    model_key=target["model_key"],
                    stages=GPQA_STAGES,
                ),
                "mapping": {
                    "test_n": int(mapping["mapping_performance"]["test_n"]),
                    "test_r2": float(mapping["mapping_performance"]["test_r2"]),
                    "test_retrieval_acc1_cos": float(
                        mapping["mapping_performance"]["test_retrieval_acc1_cos"]
                    ),
                    "identity_bias_test_r2": float(selected["floors_test_r2"]["identity_bias"]),
                    "identity_bias_test_retrieval_acc1_cos": identity_retrieval,
                    "learned_increment_over_identity_retrieval": float(
                        mapping["mapping_performance"]["test_retrieval_acc1_cos"]
                    )
                    - identity_retrieval,
                    "gpqa_n": int(transfer["n_rows"]),
                    "gpqa_same_question_acc1_cos": float(transfer["same_question_acc1_cos"]),
                    "gpqa_same_question_chance": float(transfer["same_question_chance"]),
                    "rrr_rank_rel10": int(mapping["operational_rank"]["rank"]),
                    "dimension": int(mapping["dimension"]),
                    "layer_star": int(mapping["layer_star"]),
                },
            }
            if profiles[name]["mapping"]["gpqa_n"] != profiles[name]["gpqa"]["retained"]:
                raise ValueError(
                    f"{cell}/{name}: GPQA transfer n does not equal retained parser rows"
                )
        match_4500 = matched["by_n"]["4500"]
        rows.append(
            {
                **target,
                "profiles": profiles,
                "long_matched_n": {
                    "n_train": 4500,
                    "n_fits": int(match_4500["n_fits"]),
                    "mean_rrr_rank_rel10": float(match_4500["mean_rank_rel10"]),
                    "min_rrr_rank_rel10": int(match_4500["min_rank_rel10"]),
                    "max_rrr_rank_rel10": int(match_4500["max_rank_rel10"]),
                },
            }
        )
    return {
        "schema_version": "issue2588_minimal_longcap_summary_v1",
        "question": (
            "Do the three 27B thinking-map results survive removal of severe output-cap "
            "truncation, and how much of any rank difference is explained by realized "
            "training size?"
        ),
        "scope": {
            "targets": [target["cell"] for target in TARGETS],
            "excluded_by_user": ["DeepSeek V4", "GLM 5.3", "Qwen3.8 Flash Next", "Qwen3.5 397B"],
            "no_new_model_generation_beyond_targets": True,
        },
        "sources": {
            "hf_repo": HF_REPO,
            "baseline_revision": BASELINE_REVISION,
            "long_revision": long_revision,
            "baseline_mapping": {
                "path": str(baseline_mapping_path.relative_to(ROOT)),
                "sha256": _sha256(baseline_mapping_path),
            },
            "long_mapping": {
                "path": str(long_mapping_path.relative_to(ROOT)),
                "sha256": _sha256(long_mapping_path),
            },
            "matched_n": {
                "path": str(matched_path.relative_to(ROOT)),
                "sha256": _sha256(matched_path),
            },
            "long_source_audit": source_audit,
        },
        "definitions": {
            "retained_fraction": "1 - unique parser-dropped rows / generation-stage rows",
            "effective_cap_hits": (
                "post-regeneration cap hits when regeneration ran; otherwise initial cap hits"
            ),
            "rrr_rank_rel10": (
                "smallest reduced-rank-regression rank with validation SSE no more than 10% "
                "above the full map"
            ),
            "identity_plus_bias": (
                "dimension-matched identity map plus a learned training-fold bias; both held-out "
                "R2 and raw cosine top-1 retrieval are retained"
            ),
            "matched_n": (
                "three deterministic subset refits at n=4500, each using its own training "
                "normalization and the frozen production ridge penalty"
            ),
        },
        "rows": rows,
        "matched_n_controls": matched_controls,
        "matched_n_trends": matched_trends,
    }


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(SEAM)
    ax.tick_params(length=0, pad=7)
    ax.grid(axis="y", color=GRID, lw=1.0, alpha=0.55)
    ax.set_axisbelow(True)


def _panel_heading(ax: plt.Axes, kicker: str, title: str) -> None:
    ax.set_title(title, loc="left", y=1.04, pad=0, fontweight=650)
    ax.text(
        0,
        1.15,
        kicker.upper(),
        transform=ax.transAxes,
        fontsize=12,
        fontweight=700,
        color=MUTED,
        va="bottom",
    )


def _paired_bars(
    ax: plt.Axes,
    labels: list[str],
    baseline: list[float],
    long: list[float],
    *,
    ylabel: str,
    y_min: float,
    y_max: float,
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    ax.bar(
        x - width / 2,
        baseline,
        width,
        color=PAPER,
        edgecolor=BASELINE_COLOR,
        linewidth=1.8,
        hatch="///",
        label="Original cap",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        long,
        width,
        color=LONG_COLOR,
        edgecolor=LONG_COLOR,
        linewidth=1.2,
        label="Long cap",
        zorder=3,
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_min, y_max)
    _style_axis(ax)


def make_figure(summary: dict[str, Any]) -> plt.Figure:
    set_c2a_style()
    fig, grid = plt.subplots(2, 2, figsize=(14.4, 10.4), constrained_layout=False)
    axes = grid.ravel()
    rows = summary["rows"]
    model_labels = [row["label"].replace("Qwen", "Q") for row in rows]

    retention_labels: list[str] = []
    retention_base: list[float] = []
    retention_long: list[float] = []
    for row, model in zip(rows, model_labels, strict=True):
        for split, split_label in (("generic", "train"), ("gpqa", "GPQA")):
            retention_labels.append(f"{model}\n{split_label}")
            retention_base.append(100 * row["profiles"]["baseline"][split]["retained_fraction"])
            retention_long.append(100 * row["profiles"]["long"][split]["retained_fraction"])
    _paired_bars(
        axes[0],
        retention_labels,
        retention_base,
        retention_long,
        ylabel="Rows retained (%) ↑",
        y_min=0,
        y_max=104,
    )
    _panel_heading(axes[0], "A · Truncation", "Long caps restore usable rows")
    axes[0].tick_params(axis="x", labelrotation=24)

    quality_labels: list[str] = []
    quality_base: list[float] = []
    quality_long: list[float] = []
    for row, model in zip(rows, model_labels, strict=True):
        for field, metric_label in (
            ("test_r2", "$R^2$"),
            ("test_retrieval_acc1_cos", "generic\ntop-1"),
            ("gpqa_same_question_acc1_cos", "GPQA\ntop-1"),
        ):
            quality_labels.append(f"{model}\n{metric_label}")
            quality_base.append(row["profiles"]["baseline"]["mapping"][field])
            quality_long.append(row["profiles"]["long"]["mapping"][field])
    _paired_bars(
        axes[1],
        quality_labels,
        quality_base,
        quality_long,
        ylabel="Held-out score ↑",
        y_min=0,
        y_max=0.9,
    )
    _panel_heading(axes[1], "B · Mapping quality", "Generic maps remain predictive")
    axes[1].tick_params(axis="x", labelrotation=31)

    x = np.arange(len(rows), dtype=float)
    old_ranks = [row["profiles"]["baseline"]["mapping"]["rrr_rank_rel10"] for row in rows]
    long_ranks = [row["profiles"]["long"]["mapping"]["rrr_rank_rel10"] for row in rows]
    matched = [row["long_matched_n"]["mean_rrr_rank_rel10"] for row in rows]
    lo = np.asarray(matched) - np.asarray(
        [row["long_matched_n"]["min_rrr_rank_rel10"] for row in rows]
    )
    hi = np.asarray([row["long_matched_n"]["max_rrr_rank_rel10"] for row in rows]) - np.asarray(
        matched
    )
    axes[2].plot(
        x,
        old_ranks,
        color=BASELINE_COLOR,
        marker="s",
        markerfacecolor=PAPER,
        markeredgewidth=1.8,
        lw=2.2,
        label="Original cap, all retained rows",
        zorder=3,
    )
    axes[2].plot(
        x,
        long_ranks,
        color=LONG_COLOR,
        marker="o",
        lw=2.5,
        label="Long cap, all retained rows",
        zorder=4,
    )
    axes[2].errorbar(
        x,
        matched,
        yerr=np.vstack((lo, hi)),
        color=MATCHED_COLOR,
        marker="D",
        markerfacecolor=PAPER,
        markeredgewidth=1.8,
        lw=0,
        elinewidth=2.0,
        capsize=4,
        label="Long cap, matched n=4,500",
        zorder=5,
    )
    axes[2].set_xticks(x, model_labels)
    axes[2].set_ylabel("Reduced-rank dimension ↓")
    axes[2].set_ylim(0, max(old_ranks + long_ranks) * 1.25)
    _style_axis(axes[2])
    _panel_heading(axes[2], "C · Rank robustness", "Sample size explains part of rank")

    legend_profiles = [
        Patch(
            facecolor=PAPER,
            edgecolor=BASELINE_COLOR,
            hatch="///",
            linewidth=1.5,
            label="Original cap",
        ),
        Patch(facecolor=LONG_COLOR, edgecolor=LONG_COLOR, label="Long cap"),
    ]
    fig.legend(
        handles=legend_profiles,
        loc="lower left",
        bbox_to_anchor=(0.055, -0.01),
        frameon=False,
        ncol=2,
    )
    axes[2].legend(loc="best", frameon=False)

    arm_style = {
        "no-thinking": {"color": LONG_COLOR, "label": "Prompt read"},
        "end-of-thought": {"color": MATCHED_COLOR, "label": "End-of-thought read"},
    }
    for arm, style in arm_style.items():
        qwen = sorted(
            (
                row
                for row in summary["matched_n_controls"]
                if row["arm"] == arm and row["family"] == "Qwen h=5120 column"
            ),
            key=lambda row: row["aa_index"],
        )
        xs = np.asarray([row["aa_index"] for row in qwen])
        ys = np.asarray([row["mean_rrr_rank_rel10"] for row in qwen])
        lo = ys - np.asarray([row["min_rrr_rank_rel10"] for row in qwen])
        hi = np.asarray([row["max_rrr_rank_rel10"] for row in qwen]) - ys
        axes[3].plot(xs, ys, color=style["color"], lw=2.2, zorder=2)
        axes[3].errorbar(
            xs,
            ys,
            yerr=np.vstack((lo, hi)),
            color=style["color"],
            marker="o",
            lw=0,
            elinewidth=1.6,
            capsize=3,
            zorder=3,
        )
    olmo = [row for row in summary["matched_n_controls"] if row["family"] != "Qwen h=5120 column"]
    for row in olmo:
        style = arm_style[row["arm"]]
        axes[3].scatter(
            row["aa_index"],
            row["mean_rrr_rank_rel10"],
            marker="^",
            s=90,
            facecolor=PAPER,
            edgecolor=style["color"],
            linewidth=2,
            zorder=4,
        )
    prompt_rho = summary["matched_n_trends"]["no-thinking"]["rho"]
    thought_rho = summary["matched_n_trends"]["end-of-thought"]["rho"]
    axes[3].set_xlabel("Artificial Analysis intelligence index")
    axes[3].set_ylabel("Matched-n reduced-rank dimension")
    axes[3].set_xlim(0, 58)
    axes[3].set_ylim(0, 150)
    _style_axis(axes[3])
    _panel_heading(
        axes[3],
        "D · Same-width controls",
        f"Matched ranks: Qwen ρ = {prompt_rho:+.2f} / {thought_rho:+.2f}",
    )
    axes[3].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker="o",
                lw=2,
                label=style["label"],
            )
            for style in arm_style.values()
        ]
        + [
            Line2D(
                [0],
                [0],
                color=MUTED,
                marker="^",
                markerfacecolor=PAPER,
                lw=0,
                label="OLMo control",
            )
        ],
        frameon=False,
        loc="best",
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.89,
        bottom=0.12,
        wspace=0.28,
        hspace=0.60,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-mapping", type=Path, default=DEFAULT_BASELINE_MAPPING)
    parser.add_argument("--long-mapping", type=Path, default=DEFAULT_LONG_MAPPING)
    parser.add_argument("--matched-n", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument(
        "--long-revision",
        default=os.environ.get("EPS_ISSUE2588_HF_REVISION"),
        help="immutable HF dataset revision containing all three long-cap targets",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        baseline_mapping_path=args.baseline_mapping.resolve(),
        long_mapping_path=args.long_mapping.resolve(),
        matched_path=args.matched_n.resolve(),
        long_revision=args.long_revision,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fig = make_figure(summary)
    outputs = save_c2a_figure(
        fig,
        args.figure_stem,
        title="Issue 2588 minimal long-cap robustness experiment",
        subject="Truncation, held-out mapping quality, and matched-size reduced rank",
        creator="scripts/issue2588_minimal_longcap_summary.py",
    )
    plt.close(fig)
    meta = {
        "schema_version": "issue2588_minimal_longcap_figure_v1",
        "style_version": STYLE_VERSION,
        "resolved_font": matplotlib.rcParams["font.sans-serif"][0],
        "public_url": PUBLIC_URL,
        "source_summary": str(args.out.relative_to(ROOT)),
        "source_summary_sha256": _sha256(args.out),
        "source_revisions": summary["sources"],
        "git": _git_state(),
        "visual_encodings": {
            "original_cap": "gray hatched/open",
            "long_cap": "teal filled/circle",
            "matched_n": "terracotta open diamond with three-seed min-max interval",
        },
        "outputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
        "plotted_rows": summary["rows"],
    }
    meta_path = args.figure_stem.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"summary": str(args.out), "figure": str(outputs["png"]), "public_url": PUBLIC_URL}
        )
    )


if __name__ == "__main__":
    main()

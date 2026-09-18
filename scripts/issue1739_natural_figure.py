"""Render archived natural-prompt results without fitting, judging, or fetching.

Usage: uv run python scripts/issue1739_natural_figure.py INPUT_ROOT OUTPUT_DIR
INPUT_ROOT contains <behavior>/results.json and optional Lxx diagnostic JSONs.
Missing artifacts/cells are explicitly unavailable; no estimate is substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from explore_persona_space.analysis import c2a_plot_style as style
from scripts.issue1739_natural_score import LAYERS

BEHAVIORS = ("sycophancy", "hallucination", "evil")
TITLES = {"sycophancy": "Sycophancy", "hallucination": "Hallucination", "evil": "Evil trait"}
METHODS = (
    "answer_direction_on_context",
    "context_native",
    "mapped_answer",
    "real_answer",
)
METHOD_LABELS = {
    "answer_direction_on_context": "Answer on\ncontext",
    "context_native": "Context\nnative",
    "mapped_answer": "Mapped\nanswer",
    "real_answer": "Observed\nanswer",
}
OOD = {
    "sycophancy": ("aita", "sycoans", "sycoays", "sycofb", "sycomim", "sycomwe"),
    "hallucination": ("nqopen", "simpleqa"),
    "evil": ("hhrt", "toxicchat", "evil_mhj", "evil_pair", "evil_tomgibbs"),
}
DATASET_LABELS = {
    "wildchat_rung": "Generic chat",
    "heldin_train": "ID (map exposed)",
    "aita": "Social advice",
    "sycoans": "Answer opinion",
    "sycoays": "Are you sure?",
    "sycofb": "Feedback",
    "sycomim": "Mimicry",
    "sycomwe": "MWE",
    "nqopen": "NQOpen",
    "simpleqa": "SimpleQA",
    "hhrt": "HHRT",
    "toxicchat": "ToxicChat",
    "evil_mhj": "MHJ",
    "evil_pair": "PAIR",
    "evil_tomgibbs": "Tom Gibbs",
}
REGIMES = {
    "q01_s0": "Natural prompts: 1% tails",
    "q05_s0": "Natural prompts: 5% tails",
    "q10_s0": "Natural prompts: 10% tails",
    "q01_complete": "Natural prompts: 1%, five valid answers",
    "endpoints": "Natural prompts: literal endpoints",
    "e2": "Natural prompts: within-prompt response contrast",
    "e2p": "Natural prompts: pooled response midpoint",
    **{f"q01_s{s}": f"Natural prompts: 1% tails, tie salt {s}" for s in range(1, 5)},
}
PRIMARY = "q01_s0"


def sha(path: Path) -> str:
    """Hash a consumed or exported file without materializing its bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Publish strict JSON atomically, without allowing NaN placeholders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path, inputs: dict, *, optional: bool = False) -> Any:
    """Record every consumed input hash, distinguishing absent optional files."""
    if not path.exists() and optional:
        inputs[str(path.resolve())] = {"status": "absent"}
        return None
    value = json.loads(path.read_text())
    inputs[str(path.resolve())] = {"status": "read", "sha256": sha(path)}
    return value


def estimate(row: dict | None) -> dict:
    """Validate a correlation and the supplied absolute interval endpoints."""
    if row is None or row.get("rho") is None:
        return {
            "rho": None,
            "ci95": None,
            "status": "unavailable",
            "source": row,
            "reason": "No finite archived estimate",
        }
    rho = float(row["rho"])
    if not np.isfinite(rho) or not -1 <= rho <= 1:
        raise ValueError(f"Invalid archived Spearman correlation: {row}")
    interval = row.get("ci95")
    if interval is not None:
        if len(interval) != 2 or not np.isfinite(interval).all():
            raise ValueError(f"Invalid archived interval: {row}")
        lower, upper = map(float, interval)
        if not -1 <= lower <= upper <= 1:
            raise ValueError(f"Reversed/out-of-range archived interval: {row}")
        interval = [lower, upper]
    return {"rho": rho, "ci95": interval, "status": "ok", "source": row}


def load_results(input_root: Path, inputs: dict) -> dict:
    """Check result identity and equal-dataset aggregation before rendering."""
    results = {}
    for behavior in BEHAVIORS:
        result = read_json(input_root / behavior / "results.json", inputs, optional=True)
        if result is None:
            results[behavior] = None
            continue
        if result["behavior"] != behavior or result["primary"] != PRIMARY:
            raise ValueError(f"Unexpected result identity: {behavior}")
        if not result.get("source_sha") or not result.get("input_fingerprint"):
            raise ValueError(f"Missing run provenance: {behavior}")
        datasets = {r["dataset"]: r for r in result["datasets"]}
        if len(datasets) != len(result["datasets"]):
            raise ValueError(f"Duplicate dataset summaries: {behavior}")
        unexpected = set(datasets) - {*OOD[behavior], "wildchat_rung", "heldin_train"}
        if unexpected:
            raise ValueError(f"Unregistered datasets for {behavior}: {sorted(unexpected)}")
        for method, row in result["ood"].items():
            checked = estimate(row)
            if checked["rho"] is None:
                continue
            values = [
                datasets[d]["estimates"].get(method, {}).get("rho")
                for d in OOD[behavior]
                if d in datasets
            ]
            if len(values) != len(OOD[behavior]) or any(v is None for v in values):
                raise ValueError(
                    f"Finite OOD mean with incomplete registered coverage: {behavior}/{method}"
                )
            if not np.isclose(checked["rho"], np.mean(values), rtol=0, atol=1e-12):
                raise ValueError(f"OOD mean does not average datasets equally: {behavior}/{method}")
        results[behavior] = result
    if not any(result is not None for result in results.values()):
        raise FileNotFoundError(f"No actual results.json artifacts under {input_root}")
    return results


def _color(method: str) -> str:
    """Reserve the paper's linear-map teal for mapped-answer readouts."""
    return style.ROLES["linear"].color if method == "mapped_answer" else style.MUTED


def _point(ax, position: float, cell: dict, method: str, variant: str, *, horizontal=False):
    """Draw interval endpoints directly; an unavailable cell has text only."""
    color = _color(method)
    if cell["rho"] is None:
        if horizontal:
            ax.text(
                0.98,
                position,
                "N/A",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                color=style.MUTED,
            )
        else:
            ax.text(
                position,
                0.03,
                "N/A",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                rotation=90,
                color=style.MUTED,
            )
        return
    marker = "s" if variant == "e1" else style.ROLES["linear"].marker
    face = style.PAPER if variant == "e1" else color
    interval = cell["ci95"]
    if interval is not None:
        low, high = interval
        if horizontal:
            ax.hlines(position, low, high, color=color, lw=1.5)
            ax.vlines([low, high], position - 0.045, position + 0.045, color=color, lw=1.5)
        else:
            ax.vlines(position, low, high, color=color, lw=1.5)
            ax.hlines([low, high], position - 0.035, position + 0.035, color=color, lw=1.5)
    x, y = (cell["rho"], position) if horizontal else (position, cell["rho"])
    ax.plot(
        x,
        y,
        marker=marker,
        markersize=7.5,
        markerfacecolor=face,
        markeredgecolor=color,
        markeredgewidth=1.5,
        linestyle="none",
        zorder=3,
    )


def _legend(fig, comparison: str) -> None:
    """Use redundant square/fill encoding for the extraction contrast."""
    style.legend_kicker(fig, 0.075, 0.115, "Direction extraction")
    handles = [
        Line2D(
            [],
            [],
            marker="s",
            markerfacecolor=style.PAPER,
            color=style.MUTED,
            linestyle="none",
            label="Instruction contrast (E1)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor=style.MUTED,
            color=style.MUTED,
            linestyle="none",
            label=comparison,
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.065, 0.01),
        ncol=2,
        columnspacing=1.3,
        handletextpad=0.5,
    )


def export(
    fig,
    fraction,
    outdir: Path,
    stem: str,
    title: str,
    plotted: list,
    provenance: dict,
    public_base_url: str | None,
) -> dict:
    """Export all required formats and an exact-data reproducibility sidecar."""
    saved = style.save_c2a_figure(
        fig,
        outdir / stem,
        title=title,
        subject="Archived Spearman correlations; supplied conditional 95% group-bootstrap intervals",
        creator="scripts/issue1739_natural_figure.py",
        include_width=fraction,
    )
    plt.close(fig)
    paths = {kind: str(saved[kind].resolve()) for kind in ("pdf", "png", "grayscale")}
    meta = {
        **provenance,
        "render": saved["record"],
        "title": title,
        "plotted_values": plotted,
        "outputs": {
            kind: {"path": path, "sha256": sha(Path(path))} for kind, path in paths.items()
        },
        "encodings": {
            "instruction": "open square",
            "natural": "filled circle",
            "mapped_answer": style.ROLES["linear"].color,
            "other_readouts": style.MUTED,
            "missing": "N/A text; no point",
        },
        "intervals": "Archived 95% paired group-bootstrap endpoints, drawn without recentering",
        "public_urls": (
            {
                kind: public_base_url.rstrip("/") + "/" + Path(path).name
                for kind, path in paths.items()
            }
            if public_base_url
            else None
        ),
    }
    write_json(outdir / f"{stem}.meta.json", meta)
    return {"stem": stem, "files": paths, "public_urls": meta["public_urls"]}


def primary_plot(results, outdir, provenance, public_base_url):
    """Compare the two extraction recipes across the four frozen OOD readouts."""
    fig, fraction = style.c2a_figure("full", aspect=0.43)
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.82, bottom=0.29, wspace=0.20)
    plotted = []
    for panel, (behavior, ax) in enumerate(zip(BEHAVIORS, axes, strict=True)):
        result = results[behavior]
        datasets = (
            []
            if result is None
            else [d for d in result["datasets"] if d["dataset"] in OOD[behavior]]
        )
        n = sum(row["n"] for row in datasets)
        style.panel_header(ax, chr(65 + panel), TITLES[behavior], kicker_y=1.14)
        ax.text(
            0,
            1.035,
            f"{len(datasets)}/{len(OOD[behavior])} OOD · n={n:,}",
            transform=ax.transAxes,
            color=style.MUTED,
        )
        for x, method in enumerate(METHODS):
            for variant, offset in (("e1", -0.12), (PRIMARY, 0.12)):
                cell = estimate(
                    None if result is None else result["ood"].get(f"{variant}/{method}")
                )
                _point(ax, x + offset, cell, method, variant, horizontal=True)
                plotted.append(
                    {
                        "behavior": behavior,
                        "regime": "OOD",
                        "method": method,
                        "variant": variant,
                        "layer": LAYERS[behavior][method],
                        **cell,
                    }
                )
        style.style_axis(ax, grid_axis="x")
        ax.axvline(0, color=style.MUTED, lw=0.8)
        ax.set_xlim(-1.02, 1.02)
        ax.set_xticks([-1, 0, 1])
        ax.set_ylim(3.5, -0.5)
        ax.set_yticks(range(4), [METHOD_LABELS[m] for m in METHODS])
        ax.set_xlabel(style.better_label(r"Mean Spearman $\rho$"))
    _legend(fig, REGIMES[PRIMARY])
    return export(
        fig,
        fraction,
        outdir,
        "natural_persona_ood",
        "Natural-prompt persona extraction",
        plotted,
        provenance,
        public_base_url,
    )


def appendix_plot(behavior, result, variant, outdir, provenance, public_base_url):
    """Show every dataset, including generic/ID, at every method's frozen layer."""
    names = ("wildchat_rung", "heldin_train", *OOD[behavior])
    datasets = {} if result is None else {r["dataset"]: r for r in result["datasets"]}
    fig, fraction = style.c2a_figure("full", aspect=max(0.55, 0.075 * len(names) + 0.20))
    axes = fig.subplots(1, 4, sharey=True)
    fig.subplots_adjust(left=0.21, right=0.985, top=0.78, bottom=0.21, wspace=0.18)
    plotted = []
    for panel, (method, ax) in enumerate(zip(METHODS, axes, strict=True)):
        for y, dataset in enumerate(names):
            row = datasets.get(dataset)
            for recipe, offset in (("e1", -0.13), (variant, 0.13)):
                cell = estimate(None if row is None else row["estimates"].get(f"{recipe}/{method}"))
                if recipe == "e2" and method == "context_native":
                    if cell["rho"] is not None:
                        raise ValueError("Within-prompt context direction must be structurally N/A")
                    cell["reason"] = "Within-prompt context contrast is structurally zero"
                _point(ax, y + offset, cell, method, recipe, horizontal=True)
                plotted.append(
                    {
                        "behavior": behavior,
                        "dataset": dataset,
                        "method": method,
                        "variant": recipe,
                        "layer": LAYERS[behavior][method],
                        "n": None if row is None else row["n"],
                        **cell,
                    }
                )
        style.style_axis(ax, grid_axis="x")
        ax.axvline(0, color=style.MUTED, lw=0.8)
        ax.set_xlim(-1.02, 1.02)
        ax.set_xticks([-1, 0, 1])
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_xlabel(style.better_label(r"Spearman $\rho$"))
        style.panel_header(
            ax,
            chr(65 + panel),
            f"Layer {LAYERS[behavior][method]}",
            METHOD_LABELS[method],
            kicker_y=1.28,
            title_y=1.045,
        )
        labels = [
            f"{DATASET_LABELS[d]}\n(n={datasets[d]['n']:,})"
            if d in datasets
            else f"{DATASET_LABELS[d]}\n(unavailable)"
            for d in names
        ]
        ax.set_yticks(range(len(names)), labels)
    fig.text(0.075, 0.965, f"{TITLES[behavior]} · {REGIMES[variant]}", color=style.INK)
    _legend(fig, REGIMES[variant])
    stem = f"natural_persona_{behavior}_{variant}"
    return export(
        fig,
        fraction,
        outdir,
        stem,
        f"{TITLES[behavior]}: {REGIMES[variant]}",
        plotted,
        provenance,
        public_base_url,
    )


def diagnostic_report(input_root, results, inputs):
    """Collect actual extraction/reliability and same-layer answer checks."""
    report = {}
    for behavior in BEHAVIORS:
        result = results[behavior]
        if result is None:
            report[behavior] = {"status": "unavailable"}
            continue
        layer = LAYERS[behavior]["mapped_answer"]
        root = input_root / behavior / f"L{layer:02d}"
        scores = read_json(root / "scores.json", inputs, optional=True)
        selection = read_json(root / "selection.json", inputs, optional=True)
        stability = read_json(root / "direction_stability.json", inputs, optional=True)
        parity = []
        for frozen_layer in sorted(set(LAYERS[behavior].values())):
            rows = read_json(
                input_root / behavior / f"L{frozen_layer:02d}" / "parity.json",
                inputs,
                optional=True,
            )
            if rows is not None:
                for row in rows:
                    if row["n"] != row["expected_n"] or not np.isclose(
                        abs(row["rho"] - row["expected_rho"]),
                        row["absolute_error"],
                        atol=1e-12,
                        rtol=0,
                    ):
                        raise ValueError(f"Invalid archived E1 parity record: {behavior}/{row}")
                parity.extend(rows)
        checks = []
        for dataset in result["datasets"]:
            key = dataset["dataset"]
            selected = (
                []
                if scores is None
                else [
                    r
                    for r in scores
                    if r["dataset"].replace(":", "_") == key
                    and r["variant"] == PRIMARY
                    and r["arm"] == "real_answer"
                    and r["layer"] == layer
                ]
            )
            if len(selected) > 1:
                raise ValueError(f"Duplicate same-layer answer diagnostic: {behavior}/{key}")
            matched = None if not selected else selected[0]
            if matched is not None and matched["n"] != dataset["n"]:
                raise ValueError(f"Answer diagnostic coverage differs: {behavior}/{key}")
            same_layer = estimate(matched)
            checks.append(
                {
                    "dataset": key,
                    "mapped_layer": layer,
                    "mapped_layer_observed_answer": same_layer,
                    "positive_point_estimate": None
                    if same_layer["rho"] is None
                    else same_layer["rho"] > 0,
                    "displayed_observed_layer": LAYERS[behavior]["real_answer"],
                    "displayed_observed_answer": estimate(
                        dataset["estimates"].get(f"{PRIMARY}/real_answer")
                    ),
                }
            )
        report[behavior] = {
            "status": "ok",
            "source_sha": result["source_sha"],
            "input_fingerprint": result["input_fingerprint"],
            "primary_paired_ood_differences": result["ood_differences"],
            "ood_sensitivity": {
                variant: {
                    method: estimate(result["ood"].get(f"{variant}/{method}")) for method in METHODS
                }
                for variant in (PRIMARY, "q05_s0", "q10_s0")
            },
            "all_ood_estimates": result["ood"],
            "all_dataset_summaries": result["datasets"],
            "selection_at_mapped_layer": selection,
            "direction_stability_at_mapped_layer": stability,
            "observed_answer_checks": checks,
            "e1_archived_reference_parity": parity,
        }
    return report


def _number(value):
    """Format an available statistic without hiding a missing cell as zero."""
    return "N/A" if value is None else f"{value:.3f}"


def markdown_report(report, artifacts, public_base_url):
    """Write descriptive tables; leave interpretations to the results review."""
    lines = [
        "Natural-prompt extraction uses archived judgments to select the high/low candidate "
        "prompts from archived on-policy responses; this is not unlabeled extraction. "
        "The experiment makes no new model or judge calls. "
        "Evil trait denotes the historical rubric; these are not newly graded compliance labels.",
        "",
        "Intervals are the archived 95% paired group-bootstrap intervals, conditional on "
        "the extracted directions, map, layers, generations, and labels. OOD datasets receive "
        "equal weight. ID evaluation remains exposed to map fitting. N/A means unavailable.",
        "",
        "| Behavior | Natural mapped minus | Δ Spearman ρ | 95% interval | OOD coverage |",
        "|---|---|---:|---|---|",
    ]
    for behavior in BEHAVIORS:
        result = report[behavior]
        differences = result.get("primary_paired_ood_differences", {})
        for other in (
            "e1/mapped_answer",
            "q01_s0/context_native",
            "q01_s0/answer_direction_on_context",
        ):
            row = differences.get(f"q01_s0/mapped_answer_minus_{other}", {})
            interval = row.get("ci95")
            ci = "N/A" if interval is None else f"[{_number(interval[0])}, {_number(interval[1])}]"
            coverage = row.get("coverage", {})
            label = (
                "N/A"
                if not coverage
                else f"{coverage['datasets_available']}/{coverage['datasets_planned']}"
            )
            comparison = {
                "e1/mapped_answer": "E1 mapped answer",
                "q01_s0/context_native": "Natural context native",
                "q01_s0/answer_direction_on_context": "Natural answer on context",
            }[other]
            lines.append(
                f"| {TITLES[behavior]} | {comparison} | {_number(row.get('delta'))} | {ci} | {label} |"
            )
    lines += [
        "",
        "| Behavior | Tail fraction | Answer on context | Context native | Mapped answer | Observed answer |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for behavior in BEHAVIORS:
        for variant, label in ((PRIMARY, "1%"), ("q05_s0", "5%"), ("q10_s0", "10%")):
            row = report[behavior].get("ood_sensitivity", {}).get(variant, {})
            values = " | ".join(_number(row.get(method, {}).get("rho")) for method in METHODS)
            lines.append(f"| {TITLES[behavior]} | {label} | {values} |")
    lines += [
        "",
        "Observed-answer checks below use the mapped readout's layer. "
        "For hallucination this is layer 23, while the displayed observed-answer reference "
        "uses layer 27. Positive point estimates alone do not establish a reliable trait direction.",
        "",
        "| Behavior | Dataset | Mapped layer | Observed-answer ρ at mapped layer | Displayed observed-answer ρ |",
        "|---|---|---:|---:|---:|",
    ]
    for behavior in BEHAVIORS:
        for row in report[behavior].get("observed_answer_checks", []):
            lines.append(
                f"| {TITLES[behavior]} | {row['dataset']} | {row['mapped_layer']} | "
                f"{_number(row['mapped_layer_observed_answer']['rho'])} | "
                f"{_number(row['displayed_observed_answer']['rho'])} |"
            )
    negative_checks = [
        (behavior, row)
        for behavior in BEHAVIORS
        for row in report[behavior].get("observed_answer_checks", [])
        if row["dataset"] in OOD[behavior] and row["positive_point_estimate"] is False
    ]
    for behavior, row in negative_checks:
        lines += [
            "",
            f"{TITLES[behavior]} / {DATASET_LABELS[row['dataset']]} has a nonpositive "
            f"observed-answer correlation at the actual mapped layer {row['mapped_layer']} "
            f"(ρ={_number(row['mapped_layer_observed_answer']['rho'])}). This dataset does "
            "not show a positively predictive answer direction at that layer; a reference "
            "at another layer does not establish that validity.",
        ]
    lines += [
        "",
        "Primary extraction support and held-out-response gaps below are in each archived "
        "rubric's native units. Small tails, zero held-out gaps, and tied low-score pools "
        "limit the interpretation of a quantile contrast as reliable extreme trait elicitation.",
        "",
        "| Behavior | Held-out dataset | Candidates | Prompts/tail | High−low score | 3→2 gap | 2→3 gap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for behavior in BEHAVIORS:
        selection = report[behavior].get("selection_at_mapped_layer") or {}
        for dataset in OOD[behavior]:
            row = selection.get(f"{dataset}/{PRIMARY}", {})
            reliability = row.get("response_split_reliability", {})
            lines.append(
                f"| {TITLES[behavior]} | {DATASET_LABELS[dataset]} | "
                f"{row.get('n_candidates', 'N/A')} | {row.get('k', 'N/A')} | "
                f"{_number(row.get('high_minus_low_score'))} | "
                f"{_number(reliability.get('3_to_2', {}).get('high_minus_low_score'))} | "
                f"{_number(reliability.get('2_to_3', {}).get('high_minus_low_score'))} |"
            )
    parity = [
        row
        for behavior in BEHAVIORS
        for row in report[behavior].get("e1_archived_reference_parity", [])
    ]
    if parity:
        max_error = max(row["absolute_error"] for row in parity)
        lines += [
            "",
            f"Archived E1 parity records cover {len(parity)} dataset/readout cells at frozen "
            f"layers, with matching row counts and maximum absolute correlation error "
            f"{max_error:.6g} against their archived reference values.",
        ]
    lines += [
        "",
        "Extraction counts, native-score tail means, both held-out-response reliability "
        "orientations, and group-half/tie direction cosines are preserved verbatim in "
        "`natural_persona_report.json`. Optional diagnostics absent from the input are recorded as absent.",
    ]
    if public_base_url:
        lines += ["", *[f"- [{item['stem']}]({item['public_urls']['pdf']})" for item in artifacts]]
    return "\n".join(lines) + "\n"


def render(input_root: Path, outdir: Path, *, public_base_url: str | None = None) -> dict:
    """Render only existing result artifacts, preserving all consumed provenance."""
    if public_base_url and urlparse(public_base_url).scheme not in {"https", "http"}:
        raise ValueError("public_base_url must be a browser-accessible HTTP(S) URL")
    input_root, outdir = input_root.resolve(), outdir.resolve()
    inputs = {}
    results = load_results(input_root, inputs)
    report = diagnostic_report(input_root, results, inputs)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            str(Path(__file__).relative_to(ROOT)),
            str(Path(style.__file__).relative_to(ROOT)),
        ],
        cwd=ROOT,
        text=True,
    ).strip()
    provenance = {
        "inputs": inputs,
        "renderer_git_sha": source,
        "renderer_git_status": git_status,
        "renderer_sha256": sha(Path(__file__)),
        "style_sha256": sha(Path(style.__file__)),
        "run_source_shas": {b: r["source_sha"] for b, r in results.items() if r is not None},
        "run_input_fingerprints": {
            b: r["input_fingerprint"] for b, r in results.items() if r is not None
        },
        "frozen_layers": LAYERS,
        "correlation_axis_range": [-1.02, 1.02],
        "behavior_labels": TITLES,
        "optional_input_absence_is_not_a_zero": True,
    }
    style.set_c2a_style()
    artifacts = [primary_plot(results, outdir, provenance, public_base_url)]
    for behavior in BEHAVIORS:
        for variant in REGIMES:
            artifacts.append(
                appendix_plot(
                    behavior, results[behavior], variant, outdir, provenance, public_base_url
                )
            )
    write_json(outdir / "natural_persona_report.json", {**provenance, "behaviors": report})
    (outdir / "natural_persona_report.md").write_text(
        markdown_report(report, artifacts, public_base_url)
    )
    manifest = {
        **provenance,
        "artifacts": artifacts,
        "reports": {
            name: sha(outdir / name)
            for name in ("natural_persona_report.json", "natural_persona_report.md")
        },
    }
    write_json(outdir / "natural_persona_render_manifest.json", manifest)
    return manifest


def main():
    """Plot-only command; no inference, bootstrap recomputation, or network calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--public-base-url", help="Existing/planned HTTP(S) output location for report links"
    )
    args = parser.parse_args()
    manifest = render(args.input_root, args.output_dir, public_base_url=args.public_base_url)
    print(json.dumps({"artifacts": len(manifest["artifacts"]), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()

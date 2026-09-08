"""Render independently audited highrate results; no inference, fitting or publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from explore_persona_space.analysis import c2a_plot_style as style

OUT = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate/report"
)
REGIMES = (
    "primary",
    "competence_sensitivity",
    "screen_augmented_primary",
    "screen_augmented_competence_sensitivity",
)
LABELS = (
    "Assessable tasks · context only",
    "Original-success subset · context only",
    "Assessable tasks · + screening",
    "Original-success subset · + screening",
)
PANELS = (("raw_over_text", "Raw vs. text"), ("mapped_over_raw", "Mapped vs. raw"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finish(fig, stem, title, caption, data, inputs, *, provisional=False):
    if provisional:
        stem += "_provisional_v10"
        caption = "PROVISIONAL: whole-run final validation remains pending. " + caption
        fig.text(0.02, 0.995, "Provisional · final validation pending", va="top", color=style.MUTED)
    files = style.save_c2a_figure(
        fig,
        OUT / stem,
        title=title,
        subject=caption,
        creator="Independent task2670 result review",
        include_width=1.0,
    )
    assert all(sha(path) == value for path, value in inputs.items())
    source = Path(__file__)
    meta = {
        "provisional": provisional,
        "caption": caption,
        "input_sha256": inputs,
        "source_sha256": {str(source): sha(source), str(Path(style.__file__)): sha(style.__file__)},
        "plotted_data": data,
        "render": files["record"],
        "output_sha256": {str(files[k]): sha(files[k]) for k in ("pdf", "png", "grayscale")},
        "publication_status": (
            "Local review only; parent must publish before presenting a figure URL."
        ),
    }
    (OUT / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    plt.close(fig)


def run(*, provisional=False):
    audit_path = OUT / (
        "provisional_all_regime_audit_v10.json" if provisional else "independent_result_audit.json"
    )
    data_path = OUT / (
        "comparison_data_provisional_v10.json" if provisional else "comparison_data.json"
    )
    inputs = {str(p): sha(p) for p in (audit_path, data_path)}
    audit, data = (json.loads(p.read_text()) for p in (audit_path, data_path))
    if provisional:
        assert audit["status"] == "PROVISIONAL_ALL_REGIME_ALGEBRA_VERIFIED_NOT_FINAL_PASS"
        assert audit["verification_passed"] is False and data["provisional"] is True
    else:
        assert audit["verdict"] == "PASS" and audit["verification_passed"] is True
    rows = {(x["regime"], x["comparison"]): x for x in data["comparisons"]}
    assert set(rows) == {(r, p) for r in REGIMES for p, _ in PANELS}
    style.set_c2a_style()
    method_names = {
        "prevalence": "Training prevalence",
        "metadata": "Metadata",
        "text_plus_metadata": "Text + metadata",
        "raw_plus_metadata": "Raw + metadata",
        "mapped_plus_metadata": "Mapped + metadata",
        "orientation_38298": "Orientation 1",
        "orientation_38299": "Orientation 2",
        "orientation_38300": "Orientation 3",
        "direct_screen_rate": "Direct screening rate",
    }
    model_rows = {(x["regime"], x["method"]): x for x in data["models"]}
    fig, _ = style.c2a_figure("full", 0.95)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.92, bottom=0.10, hspace=0.32, wspace=0.72)
    for panel, regime in zip(axes.ravel(), REGIMES, strict=True):
        methods = [method for method in method_names if (regime, method) in model_rows]
        for i, method in enumerate(methods):
            value = model_rows[(regime, method)]["log_loss"]
            is_control = method.startswith("orientation_") or method == "prevalence"
            color = style.MUTED if is_control else style.ROLES["linear"].color
            panel.plot(value, i, marker="x" if is_control else "o", markersize=8, color=color)
        baseline = model_rows[(regime, "prevalence")]["log_loss"]
        panel.axvline(baseline, color=style.MUTED, lw=1.0, ls="--")
        panel.set_yticks(range(len(methods)), [method_names[m] for m in methods])
        panel.set_ylim(len(methods) - 0.4, -0.6)
        panel.set_xlim(0.33, 0.515)
        support = audit["regimes"][regime]["test_support"]
        population = "Assessable tasks" if "competence" not in regime else "Original-success subset"
        budget = (
            "+ Independent screening" if regime.startswith("screen_") else "Initial context only"
        )
        panel.set_title(f"{population}\n{budget}", loc="left", pad=15)
        panel.text(
            0.97,
            0.98,
            f"{support['n_tasks']} tasks · {support['n_trajectories']} draws",
            transform=panel.transAxes,
            ha="right",
            va="top",
            color=style.MUTED,
            fontsize=14,
        )
        low, high = panel.get_xlim()
        panel.set_xticks([t for t in panel.get_xticks() if low <= t <= high])
        style.style_axis(panel, grid_axis="x")
    fig.supxlabel(
        "Mean held-out log loss (nats per completed trajectory; lower is better)", y=0.045
    )
    caption = (
        "All prespecified predictor/control point estimates in four populations "
        "and information sets. "
        "The dashed line is the training-prevalence baseline. Screening outcomes are added equally "
        "to every fitted representation in the bottom row; the direct screening-rate baseline is "
        "shown separately. Values are descriptive point estimates; paired task-bootstrap intervals "
        "for the planned comparisons are shown in forecast_comparisons. Orientation controls "
        "retain the map singular spectrum and are reported separately without selecting a winner. "
        "Censored and structurally invalid observations are excluded "
        "from completed semantic trials."
    )
    finish(
        fig,
        "forecast_model_losses",
        "Held-out forecast losses and baselines",
        caption,
        data["models"],
        inputs,
        provisional=provisional,
    )
    fig, _ = style.c2a_figure("full", 0.66)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.22, right=0.975, top=0.87, bottom=0.16, wspace=0.76, hspace=0.68)
    for budget in range(2):
        selected_regimes = REGIMES[2 * budget : 2 * budget + 2]
        for column, (label, title) in enumerate(PANELS):
            panel = axes[budget, column]
            selected_rows = [rows[(r, label)] for r in selected_regimes]
            low = min(0, *(r["ci_low"] for r in selected_rows))
            high = max(0, *(r["ci_high"] for r in selected_rows))
            pad = max(1e-7, (high - low) * 0.14)
            for i, row in enumerate(selected_rows):
                panel.hlines(
                    i, row["ci_low"], row["ci_high"], color=style.ROLES["linear"].color, lw=2.0
                )
                panel.plot(
                    row["improvement"],
                    i,
                    marker="s" if budget else "o",
                    markersize=8,
                    markerfacecolor="white" if i else style.ROLES["linear"].color,
                    markeredgecolor=style.ROLES["linear"].color,
                    markeredgewidth=1.8,
                )
            panel.axvline(0, color=style.MUTED, lw=1.2, linestyle="--")
            panel.set_xlim(low - pad, high + pad)
            panel.set_title(
                f"{title}\n" + ("+ Screening" if budget else "Context only"), loc="left", pad=20
            )
            panel.set_yticks(range(2), ["Assessable tasks", "Original-success\nsubset"])
            panel.set_ylim(1.65, -0.55)
            panel.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
            low, high = panel.get_xlim()
            panel.set_xticks([t for t in panel.get_xticks() if low <= t <= high])
            style.style_axis(panel, grid_axis="x")
    fig.supxlabel("Reduction in held-out log loss (nats per trajectory) →", y=0.07)
    fig.text(0.22, 0.97, "Paired test-task intervals · panel scales differ", color=style.MUTED)
    caption = (
        "Positive values favor raw over text or mapped over raw. Points and marginal 95% "
        "percentile intervals use 5,000 paired test-task bootstrap draws (seed 38296); fitting "
        "uncertainty is excluded. Squares add independent screening outcomes equally to the "
        "compared predictors; open points restrict to tasks with an observed original success. "
        "All results condition on completed, structurally assessable trajectories. "
        f"Fresh censors: {audit['fresh_counts']['censored']}; structurally not assessable: "
        f"{audit['structurally_not_assessable']}. These bar unconditional benefit claims. "
        "Horizontal scales differ across panels to display small effects faithfully. "
        "The assessable population uses nine tasks/72 completed draws; the original-success subset "
        "uses eight tasks/64 completed draws."
    )
    finish(
        fig,
        "forecast_comparisons",
        "Held-out reward-hacking forecasts",
        caption,
        {
            "comparisons": data["comparisons"],
            "supports": {k: v["test_support"] for k, v in audit["regimes"].items()},
        },
        inputs,
        provisional=provisional,
    )

    primary = [x for x in data["per_task_comparisons"] if x["regime"] == "primary"]
    tasks = sorted({x["task_id"] for x in primary}, key=lambda x: int(x.rsplit("_", 1)[1]))
    lookup = {(x["task_id"], x["comparison"]): x for x in primary}
    assert len(tasks) == audit["regimes"]["primary"]["test_support"]["n_tasks"]
    fig, _ = style.c2a_figure("full", 0.63)
    axes = fig.subplots(1, 2, sharey=True)
    fig.subplots_adjust(left=0.23, right=0.975, top=0.87, bottom=0.18, wspace=0.23)
    bound = max(1e-6, max(abs(x["improvement"]) for x in primary) * 1.13)
    labels = [
        f"Task {t.rsplit('_', 1)[1]} · n={lookup[(t, 'raw_over_text')]['completed_trials']}"
        for t in tasks
    ]
    for panel, (label, title) in zip(axes, PANELS, strict=True):
        values = [lookup[(task, label)]["improvement"] for task in tasks]
        panel.hlines(range(len(tasks)), 0, values, color=style.GRID, lw=2.5)
        panel.plot(values, range(len(tasks)), "o", color=style.ROLES["linear"].color, markersize=8)
        panel.axvline(0, color=style.MUTED, lw=1.2, linestyle="--")
        panel.set_xlim(-bound, bound)
        panel.set_title(title, loc="left", pad=20)
        panel.set_yticks(range(len(tasks)), labels)
        panel.set_ylim(len(tasks) - 0.4, -0.6)
        low, high = panel.get_xlim()
        panel.set_xticks([t for t in panel.get_xticks() if low <= t <= high])
        style.style_axis(panel, grid_axis="x")
    fig.supxlabel("Reduction in held-out log loss (nats per trajectory) →", y=0.06)
    fig.text(0.23, 0.94, "Assessable test tasks · initial context only", color=style.MUTED)
    caption = (
        "Per-task descriptive loss differences in the prespecified context-only primary "
        "population. Each point averages only completed, structurally assessable trajectories "
        "across both impossible conditions; n is that task's completed trajectory count. "
        "No within-task uncertainty interval is estimated. Positive values favor raw over text, "
        "or mapped over raw. Tasks are ordered numerically. "
        "Missing or censored observations are not imputed as negative rewards."
    )
    finish(
        fig,
        "forecast_per_task",
        "Task-level forecast differences",
        caption,
        primary,
        inputs,
        provisional=provisional,
    )
    print("Saved three result views with vector, color and grayscale outputs and provenance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provisional", action="store_true")
    run(provisional=parser.parse_args().provisional)

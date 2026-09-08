"""Render report-only readout controls from finished #1482 result JSONs."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from issue1482_answer_property_ceiling import digest, provenance  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    ROLES,
    c2a_figure,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval_results/issue_1482/answer_property_ceiling_20260907"
OUT = ROOT / "figures/issue_1482/answer_property_ceiling_20260907"
GROUPS = {
    "speaker_property:identity_disposition": "Identity / disposition",
    "content_type:topic": "Topic",
    "speaker_property:register_style": "Register / style",
    "speaker_property:language": "Language",
    "content_type:task_format": "Task format",
    "content_type:entity": "Entity",
    "content_type:syntax": "Syntax",
    "content_type:operation": "Operation",
    "abstraction:abstract_contextual": "Abstract contextual",
    "abstraction:lexical_semantic": "Lexical semantic",
    "abstraction:token_surface": "Token surface",
    "logit:promoting": "Logit promoting",
    "logit:suppressing": "Logit suppressing",
    "logit:partition": "Logit partition",
}
CONTRASTS = {
    "identity_vs_topic": "Identity − topic",
    "abstract_vs_token": "Abstract − token surface",
    "abstract_vs_lexical": "Abstract − lexical semantic",
    "register_vs_topic": "Register / style − topic",
    "language_vs_topic": "Language − topic",
    "format_vs_topic": "Task format − topic",
    "promoting_vs_suppressing": "Promoting − suppressing",
}


def save(fig: plt.Figure, name: str, source: Path, payload: dict) -> None:
    """Export the canonical style and a complete values/source sidecar."""
    stem = OUT / name
    rendered = save_c2a_figure(
        fig,
        stem,
        title=name.replace("_", " "),
        subject="Matched answer-property control; experiment report only",
        creator="scripts/issue1482_answer_property_figures.py",
    )
    write_json_atomic(
        stem.with_suffix(".json"),
        {
            "source_sha256": digest(source),
            "plotted": payload,
            "render": rendered["record"],
            "metadata": provenance("property-report-figures"),
            "outputs_sha256": {k: digest(rendered[k]) for k in ("pdf", "png", "grayscale")},
        },
    )
    plt.close(fig)


def main() -> None:
    """Create two concise experiment-report figures; no Overleaf writes."""
    set_c2a_style()
    source = RESULTS / "regular.json"
    data = json.loads(source.read_text())
    fig, _ = c2a_figure("full", aspect=0.80)
    ax = fig.add_axes((0.34, 0.12, 0.61, 0.77))
    y = np.arange(len(GROUPS))
    plotted = {}
    for arm, label, offset, color, marker in (
        ("observed_answer", "Observed answer", -0.12, ROLES["control"].color, "s"),
        ("context", "Context", 0.12, ROLES["linear"].color, "o"),
    ):
        values = [data["categories"][key][arm]["median"] for key in GROUPS]
        if not np.isfinite(values).all():
            raise ValueError("undefined plotted readout median")
        ax.scatter(values, y + offset, color=color, marker=marker, s=65, label=label)
        plotted[arm] = values
    ax.set_yticks(y, GROUPS.values())
    ax.invert_yaxis()
    ax.set_xlim(min(0, min(min(values) for values in plotted.values()) - 0.02), 1)
    ax.set_xlabel("Median held-out feature $R^2$ ↑")
    ax.set_title("Feature recovery from answers and context", loc="left", pad=22)
    style_axis(ax, grid_axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    save(fig, "property_readout_medians", source, {"order": list(GROUPS), "values": plotted})

    fig, _ = c2a_figure("full", aspect=0.58)
    ax = fig.add_axes((0.39, 0.21, 0.56, 0.66))
    y = np.arange(len(CONTRASTS))
    plotted = {}
    for condition, label, offset, color, marker in (
        ("raw", "Raw", -0.19, ROLES["linear"].color, "o"),
        ("activity", "Match activity", 0, ROLES["control"].color, "^"),
        ("activity_and_observed", "Also match answer readout", 0.19, INK, "s"),
    ):
        values = np.array([data["comparisons"][key][condition]["point"][1] for key in CONTRASTS])
        ci = np.array([data["comparisons"][key][condition]["ci95"][1] for key in CONTRASTS])
        if not np.isfinite(values).all() or not np.isfinite(ci).all():
            raise ValueError("undefined plotted conditional contrast")
        ax.hlines(y + offset, ci[:, 0], ci[:, 1], color=color, linewidth=1.8)
        ax.scatter(values, y + offset, color=color, marker=marker, s=55, label=label, zorder=3)
        plotted[condition] = {"point": values.tolist(), "ci95": ci.tolist()}
    ax.axvline(0.5, color=ROLES["control"].color, linestyle=":", linewidth=1.3)
    ax.set_yticks(y, CONTRASTS.values())
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Context-prediction concordance")
    ax.set_title("Property contrasts in context prediction", loc="left", pad=22)
    style_axis(ax, grid_axis="x")
    ax.legend(loc="upper right", bbox_to_anchor=(1, -0.15), ncol=1)
    save(
        fig,
        "property_conditional_concordance",
        source,
        {
            "order": list(CONTRASTS),
            "values": plotted,
            "caption": "Values above 0.5 favor the first named property. Intervals resample features conditional on the fixed readouts, bank and matching cells; correlated SAE features may make intervals too narrow. Matched cells change the eligible comparison population; these are descriptive associations, not causal decompositions.",
        },
    )


if __name__ == "__main__":
    main()

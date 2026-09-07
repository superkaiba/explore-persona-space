"""Render the thinking companion from the same pinned historical capability panel.

No inference, model fitting, new score estimates, or manuscript changes. The
displayed p-value is the original 20,000-draw two-sided permutation result.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    ROLES,
    c2a_figure,
    save_c2a_figure,
    set_c2a_style,
    style_score_axis,
)
from paper_fig_model_capability import LABELS, ROOT, SOURCE_PATHS, SOURCE_SHA

THINKING_LABELS = {
    **{
        key: label
        for key, label in LABELS.items()
        if key.startswith(("q35_", "q36_", "q38_", "q3_"))
    },
    "o3_7b_t": "OLMo3 7B Think",
    "o31_32b_t": "OLMo3.1 32B Think",
    "o3_32b_t": "OLMo3 32B Think",
    "qwq_32b": "QwQ 32B",
}
OFFSETS = {
    "q35_0p8b": (12, -8),
    "q35_2b": (12, 0),
    "q35_4b": (-12, -10),
    "q35_9b": (12, 10),
    "q35_27b": (-12, -12),
    "q36_27b": (12, 9),
    "q38_27b": (12, -3),
    "o3_7b_t": (12, 20),
    "o31_32b_t": (12, -16),
    "o3_32b_t": (12, 18),
    "q3_32b": (12, 0),
    "qwq_32b": (12, 0),
}


def load_thinking_panel() -> tuple[list[dict], dict]:
    """Validate twelve end-of-thought coordinates against the pinned audit."""
    blobs = [
        subprocess.check_output(["git", "show", f"{SOURCE_SHA}:{path}"], cwd=ROOT)
        for path in SOURCE_PATHS
    ]
    audit, mapping = [json.loads(blob) for blob in blobs]
    sources = {row["key"]: row for row in mapping["maps"]}
    rows = []
    for row in audit["table"]:
        if row["arm"] != "end-of-thought":
            continue
        source = sources[row["key"]]
        assert source["input_position"] == "cot_boundary"
        assert source["aa_status"] in {"estimated", "measured"}
        assert source["aa_index"] == row["aa_index"]
        assert source["mapping_performance"]["test_r2"] == row["test_r2"]
        rows.append(
            {
                **{key: value for key, value in row.items() if key != "key"},
                "source_map_id": row["key"],
                "label": THINKING_LABELS[row["model_key"]],
                "aa_status": source["aa_status"],
                "input_position": source["input_position"],
                "layer": source["layer_star"],
                "n_test": source["mapping_performance"]["test_n"],
            }
        )
    assert len(rows) == 12
    assert {r["model_key"] for r in rows} == set(THINKING_LABELS) == set(OFFSETS)
    x = np.asarray([r["aa_index"] for r in rows])
    y = np.asarray([r["test_r2"] for r in rows])
    assert np.isfinite(x).all() and np.isfinite(y).all()
    tests = [
        r
        for r in audit["grid"]["end-of-thought"]["results"]
        if r["predictor"] == "aa_index" and r["outcome"] == "test_r2"
    ]
    assert len(tests) == 1 and tests[0]["n"] == len(rows)
    assert np.isclose(spearmanr(x, y).statistic, tests[0]["rho"], atol=1e-12)
    return rows, {
        "source_commit": SOURCE_SHA,
        "source_files": {
            path: hashlib.sha256(blob).hexdigest()
            for path, blob in zip(SOURCE_PATHS, blobs, strict=True)
        },
        "plotted_panel": tests[0],
        "n_permutations": audit["n_permutations"],
        "correction_family_tests": audit["grid"]["end-of-thought"]["n_tests_main_grid"],
        "p_value_method": "Original two-sided Monte Carlo model-label permutation audit",
        "scope": "All twelve historical end-of-thought models, no cap_long rows pooled.",
        "measurement": "State after reasoning, before the answer, not the original prompt.",
        "score_provenance": "Six historical scores marked estimated and six measured.",
        "visual_encoding": "Uniform filled circles, score status not encoded as requested.",
        "uncertainty": "No error bars: each point is one fitted metamodel.",
        "limitations": "Exploratory cross-model association, related model families.",
    }


def main() -> None:
    """Export a labelled companion without touching the non-thinking figure."""
    rows, provenance = load_thinking_panel()
    set_c2a_style()
    fig, frac = c2a_figure("full", aspect=0.60)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.14, top=0.98)
    style_score_axis(ax, y_min=0.63, y_max=0.82, y_step=0.05)
    ax.set_yticks([0.65, 0.70, 0.75, 0.80])
    ax.set_xlim(-3, 64)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel("Artificial Analysis Intelligence Index")
    ax.set_ylabel(r"End-of-thought held-out $R^2$ $\uparrow$")
    role = ROLES["linear"]
    ax.scatter(
        [r["aa_index"] for r in rows],
        [r["test_r2"] for r in rows],
        s=110,
        marker=role.marker,
        facecolors=role.color,
        edgecolors=role.color,
        linewidths=1.9,
        zorder=3,
    )
    for row in rows:
        dx, dy = OFFSETS[row["model_key"]]
        ax.annotate(
            row["label"],
            (row["aa_index"], row["test_r2"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="center",
            color=INK,
        )
    stats = provenance["plotted_panel"]
    ax.text(
        0.035,
        0.955,
        rf"Spearman $\rho = {stats['rho']:.3f}$, $p = {stats['p_uncorrected']:.3f}$",
        transform=ax.transAxes,
        va="top",
    )
    stem = ROOT / "figures/paper/c1_model_capability_thinking"
    exported = save_c2a_figure(
        fig,
        stem,
        include_width=frac,
        title="Thinking-enabled answer predictability and model capability",
        subject="Twelve models, end-of-thought states, LMSYS-Chat-1M prompts.",
        creator="scripts/paper_fig_model_capability_thinking.py",
    )
    provenance.update({"rows": rows, "render": exported["record"]})
    provenance["output_sha256"] = {
        key: hashlib.sha256(Path(exported[key]).read_bytes()).hexdigest()
        for key in ("pdf", "png", "grayscale")
    }
    provenance["producer_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    provenance["producer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance["helper_sha256"] = hashlib.sha256(
        Path(__file__).with_name("paper_fig_model_capability.py").read_bytes()
    ).hexdigest()
    stem.with_suffix(".meta.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"statistics": stats, "figure": str(exported["png"])}, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    main()

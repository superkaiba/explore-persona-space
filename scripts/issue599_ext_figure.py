"""#599 extension-probe truncation figure (CPU, runs on the VM).

Plots the two salvaged extension-probe callback reads (steps 50/100 of the
planned 2,400-step stretched-cosine schedule, truncated at ~step 149 by the
instance's 24-hour hard-deletion deadline) against the main-arm seed-42
600-step trajectory, with the install gate's +5-nat log-prob clause and the
planned 2,400-step horizon marked. One glance should show the probe died in
the dip phase, far from the question it was asked.

Inputs (all committed in git):
- eval_results/issue_599/marker_seed42/periodic_eval/leakage_marker_step_*.json
- eval_results/issue_599_ext/marker_seed42/periodic_eval/leakage_marker_step_{50,100}.json
"""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

SOURCE_PERSONA = "medical_doctor"
GATE_NATS = 5.0
TRUNCATION_STEP = 149  # last realized training step before instance deletion
PLANNED_STEPS = 2400


def _trajectory(periodic_eval_dir: Path) -> tuple[list[int], list[float]]:
    files = sorted(
        glob(str(periodic_eval_dir / "leakage_marker_step_*.json")),
        key=lambda p: int(Path(p).stem.rsplit("_", 1)[1]),
    )
    assert files, f"no periodic-eval JSONs under {periodic_eval_dir}"
    steps, deltas = [], []
    for f in files:
        d = json.loads(Path(f).read_text())
        steps.append(int(d["step"]))
        deltas.append(float(d["metrics_by_persona"][SOURCE_PERSONA]["log_p_marker_delta"]))
    return steps, deltas


def main() -> None:
    main_steps, main_deltas = _trajectory(
        Path("eval_results/issue_599/marker_seed42/periodic_eval")
    )
    ext_steps, ext_deltas = _trajectory(
        Path("eval_results/issue_599_ext/marker_seed42/periodic_eval")
    )
    assert ext_steps == [50, 100], f"unexpected ext probe steps: {ext_steps}"

    set_paper_style("blog")
    # Long-subtitle single-axes figures collapse under constrained_layout
    # (see analyzer memory: set_title_subtitle + long subtitle) — disable it
    # and place the axes manually.
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.09, right=0.97)

    ax.axhline(GATE_NATS, color="dimgray", lw=1.0, ls="--", zorder=1)
    ax.text(
        PLANNED_STEPS * 0.985,
        GATE_NATS + 0.25,
        "install-gate log-prob clause (+5 nats)",
        fontsize=7.5,
        color="dimgray",
        ha="right",
    )
    ax.axhline(0.0, color="lightgray", lw=0.8, ls=":", zorder=1)

    ax.plot(
        main_steps,
        main_deltas,
        marker="o",
        ms=3.5,
        lw=1.4,
        color=paper_palette_role("baseline"),
        label="main run, seed 42 (600-step schedule, complete)",
        zorder=3,
    )
    ax.plot(
        ext_steps,
        ext_deltas,
        marker="D",
        ms=5.0,
        lw=1.6,
        color=paper_palette_role("accent"),
        label="extension probe, seed 42 (2,400-step schedule, truncated)",
        zorder=4,
    )

    ax.axvline(TRUNCATION_STEP, color=paper_palette_role("accent"), lw=1.0, ls="--", zorder=2)
    ax.annotate(
        "probe truncated\nat ~step 149",
        xy=(TRUNCATION_STEP, 3.0),
        xytext=(TRUNCATION_STEP + 90, 3.4),
        fontsize=8,
        color=paper_palette_role("accent"),
        arrowprops={"arrowstyle": "-", "color": paper_palette_role("accent"), "lw": 0.8},
    )
    ax.axvline(PLANNED_STEPS, color="dimgray", lw=1.0, ls=":", zorder=2)
    ax.annotate(
        "planned horizon:\n2,400 steps",
        xy=(PLANNED_STEPS, 1.2),
        xytext=(PLANNED_STEPS - 100, 1.2),
        fontsize=8,
        color="dimgray",
        ha="right",
    )

    ax.set_xlim(0, PLANNED_STEPS * 1.04)
    ax.set_xlabel("training step")
    ax.set_ylabel("sentinel log-prob gain over base (nats)")
    ax.legend(loc="center right", fontsize=8)
    set_title_subtitle(
        ax,
        "The extension probe died in the dip phase, far from its question",
        "Source-persona on-policy read, 20 questions per point, seed 42; the probe asked "
        "whether the install regime is reached by step 2,400.",
    )
    savefig_paper(fig, "ext_probe_truncation", dir="figures/issue_599")
    plt.close(fig)
    print("written figures/issue_599/ext_probe_truncation.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()

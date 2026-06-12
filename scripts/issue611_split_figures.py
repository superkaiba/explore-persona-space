# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, − minus) in scientific labels + docstrings.
"""Issue #611 — figures for the probe-split re-analysis (plan §6).

Reads ``eval_results/issue_611/split_analysis.json`` (written by
``scripts/issue611_split_analysis.py``) and writes to ``figures/issue_611/``:

1. ``split_verdict_grid`` (hero) — 2×2 panels (rows = wrong-persona /
   default-assistant probe; cols = pirate / villain), paired d traces in
   both spaces with 95% CI shading across steps {18, 30, 60, 120},
   zero line, s18 band-status shading, saturation flags marked.
2. ``base_prior_decomposition`` — base-vs-Δ levels per arm × probe ×
   step × persona (the base-prior share of cross-arm gaps).
3. ``own_slot_regime`` — own-slot Δlog P (band shaded), trained absolute
   log P, and argmax-emit rate vs step per arm × persona.
4. Exploratory dump: ``split_per_seed_scatter`` (raw per-seed d, both
   spaces), ``leakage_allocation`` (wrong vs default allocation per arm),
   ``logp_margin_agreement`` (the saturation-signature view).

CLI::

    uv run python scripts/issue611_split_figures.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
# fontTools floods INFO with per-glyph subsetting lines on every PDF save.
logging.getLogger("fontTools").setLevel(logging.WARNING)
log = logging.getLogger("issue611_split_figures")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_PATH = PROJECT_ROOT / "eval_results" / "issue_611" / "split_analysis.json"
FIG_ROOT = PROJECT_ROOT / "figures"

STEPS = (18, 30, 60, 120)
PERSONAS = ("pirate", "villain")
PROBES = ("wrong", "default")
ARMS = ("system_minimal", "role_bare")

ARM_LABEL = {"system_minimal": "Minimal system prompt", "role_bare": "Bare role header"}
PROBE_LABEL = {
    "own": "Own-persona probe",
    "wrong": "Wrong-persona probe",
    "default": "Default-assistant probe",
}
SPACE_LABEL = {"logp": "Δlog P", "margin": "EOS margin Δ(z marker − z EOS)"}


def _log_step_axis(ax) -> None:
    """Log x-axis over the step grid with clean integer tick labels only."""
    ax.set_xscale("log")
    ax.set_xticks(STEPS)
    ax.set_xticklabels([str(s) for s in STEPS])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())


def _load() -> dict:
    if not ANALYSIS_PATH.exists():
        raise FileNotFoundError(
            f"{ANALYSIS_PATH} missing — run scripts/issue611_split_analysis.py first"
        )
    return json.loads(ANALYSIS_PATH.read_text())


def _contrast_index(payload: dict) -> dict[tuple[str, str, int, str], dict]:
    return {
        (r["probe_kind"], r["persona"], r["max_steps"], r["space"]): r
        for r in payload["paired_contrasts"]
    }


def fig_split_verdict_grid(payload: dict) -> None:
    """Hero: both halves of the split, both spaces, all steps + regimes."""
    set_paper_style("blog")
    pc = _contrast_index(payload)
    color = {"logp": paper_palette_role("baseline"), "margin": paper_palette_role("primary")}
    band_status = {"pirate": "band-adjacent", "villain": "in-band"}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
    for i, probe in enumerate(PROBES):
        for k, persona in enumerate(PERSONAS):
            ax = axes[i][k]
            for space in ("logp", "margin"):
                rows = [pc[(probe, persona, s, space)] for s in STEPS]
                pts = np.array([r["point"] for r in rows])
                lo = np.array([r["ci_lo"] for r in rows])
                hi = np.array([r["ci_hi"] for r in rows])
                label = "Δlog P" if space == "logp" else "EOS margin"
                ax.plot(STEPS, pts, marker="o", color=color[space], label=label, zorder=3)
                ax.fill_between(STEPS, lo, hi, color=color[space], alpha=0.18, linewidth=0)
                # saturation-flagged cells: open square overlay
                for s, r in zip(STEPS, rows, strict=True):
                    if r["saturation_compressed"]:
                        ax.plot(
                            s,
                            r["point"],
                            marker="s",
                            markersize=11,
                            markerfacecolor="none",
                            markeredgecolor=paper_palette_role("accent"),
                            markeredgewidth=1.6,
                            zorder=4,
                        )
            ax.axhline(0.0, color="0.45", linewidth=0.9, zorder=1)
            # s18 unsaturated-read shading
            ax.axvspan(16.2, 19.9, color=paper_palette_role("neutral"), alpha=0.22, zorder=0)
            _log_step_axis(ax)
            ax.set_title(
                f"{PROBE_LABEL[probe]} — {persona}\n(s18 shaded: {band_status[persona]} read)"
            )
            if i == 1:
                ax.set_xlabel("Training steps")
            if k == 0:
                ax.set_ylabel("Paired d, nats")
            if i == 0 and k == 0:
                ax.legend(loc="upper left")
    fig.suptitle(
        "Paired d = (minimal system) − (bare role); positive = bare role header leaks less.\n"
        "Open squares: log-p/margin disagreement > 0.5 nat (ceiling-compressed; "
        "margin authoritative).",
        y=1.02,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_611/split_verdict_grid", dir=FIG_ROOT)
    plt.close(fig)


def fig_base_prior_decomposition(payload: dict) -> None:
    """Base-vs-trained levels per arm × probe × step × persona."""
    set_paper_style("blog")
    rows = payload["decomposition"]
    idx = {(r["arm"], r["probe_kind"], r["persona"], r["max_steps"]): r for r in rows}
    arm_color = {
        "system_minimal": paper_palette_role("baseline"),
        "role_bare": paper_palette_role("primary"),
    }
    probes = ("own", "wrong", "default")

    fig, axes = plt.subplots(len(probes), 2, figsize=(10.5, 9.5), sharex=True)
    width = 0.38
    x = np.arange(len(STEPS))
    for i, probe in enumerate(probes):
        for k, persona in enumerate(PERSONAS):
            ax = axes[i][k]
            for a, arm in enumerate(ARMS):
                base = np.array([idx[(arm, probe, persona, s)]["base_logp_mean"] for s in STEPS])
                delta = np.array([idx[(arm, probe, persona, s)]["dlogp_mean"] for s in STEPS])
                pos = x + (a - 0.5) * width
                ax.bar(
                    pos,
                    delta,
                    width=width,
                    bottom=base,
                    color=arm_color[arm],
                    alpha=0.75,
                    label=ARM_LABEL[arm] if (i == 0 and k == 0) else None,
                )
                ax.plot(pos, base, linestyle="none", marker="_", markersize=14, color="0.2")
            ax.axhline(0.0, color="0.45", linewidth=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in STEPS])
            ax.set_title(f"{PROBE_LABEL[probe]} — {persona}")
            if i == len(probes) - 1:
                ax.set_xlabel("Training steps")
            if k == 0:
                ax.set_ylabel("log P(marker), nats")
    fig.suptitle(
        "Bar bottom = base-model log P (dash); bar top = trained log P; bar height = Δlog P.\n"
        "Default-assistant probe: base is encoding-identical across arms (the base-clean read).",
        y=1.01,
    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.985))
    fig.tight_layout()
    savefig_paper(fig, "issue_611/base_prior_decomposition", dir=FIG_ROOT)
    plt.close(fig)


def fig_own_slot_regime(payload: dict) -> None:
    """Own-slot implant regime: Δlog P (band), trained absolute log P, emit rate."""
    set_paper_style("blog")
    rows = payload["band_accounting"]
    idx = {(r["arm"], r["persona"], r["max_steps"]): r for r in rows}
    band_lo, band_hi = payload["parameters"]["band_nats"]
    arm_color = {
        "system_minimal": paper_palette_role("baseline"),
        "role_bare": paper_palette_role("primary"),
    }
    panels = (
        ("own_dlogp_mean", "Own-slot Δlog P, nats"),
        ("own_trained_logp_mean", "Own-slot trained log P, nats"),
        ("own_emit_rate", "Own argmax-emit rate"),
    )

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.5), sharex=True)
    for i, (field, ylabel) in enumerate(panels):
        for k, persona in enumerate(PERSONAS):
            ax = axes[i][k]
            for arm in ARMS:
                vals = [idx[(arm, persona, s)][field] for s in STEPS]
                ax.plot(
                    STEPS,
                    vals,
                    marker="o",
                    color=arm_color[arm],
                    label=ARM_LABEL[arm] if (i == 0 and k == 0) else None,
                )
            if field == "own_dlogp_mean":
                ax.axhspan(band_lo, band_hi, color=paper_palette_role("neutral"), alpha=0.25)
            _log_step_axis(ax)
            if i == 0:
                ax.set_title(persona)
            if i == len(panels) - 1:
                ax.set_xlabel("Training steps")
            if k == 0:
                ax.set_ylabel(ylabel)
            if i == 0 and k == 0:
                ax.legend(loc="lower right")
    fig.suptitle(
        "Own-persona implant regime per arm. Shaded: the [5, 12]-nat usable band — "
        "only s18 is in/adjacent to it.",
        y=1.0,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_611/own_slot_regime", dir=FIG_ROOT)
    plt.close(fig)


def fig_split_per_seed_scatter(payload: dict) -> None:
    """Raw per-seed paired d per cell, both spaces (no bootstrap aggregation)."""
    set_paper_style("blog")
    pc = _contrast_index(payload)
    color = {"logp": paper_palette_role("baseline"), "margin": paper_palette_role("primary")}
    jitter = {"logp": 0.93, "margin": 1.075}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
    for i, probe in enumerate(PROBES):
        for k, persona in enumerate(PERSONAS):
            ax = axes[i][k]
            for space in ("logp", "margin"):
                for s in STEPS:
                    d = list(pc[(probe, persona, s, space)]["per_seed_d"].values())
                    label = "Δlog P" if space == "logp" else "EOS margin"
                    ax.plot(
                        [s * jitter[space]] * len(d),
                        d,
                        linestyle="none",
                        marker="o",
                        markersize=4.5,
                        alpha=0.65,
                        color=color[space],
                        label=label if (i == 0 and k == 0 and s == STEPS[0]) else None,
                    )
            ax.axhline(0.0, color="0.45", linewidth=0.9)
            _log_step_axis(ax)
            ax.set_title(f"{PROBE_LABEL[probe]} — {persona}")
            if i == 1:
                ax.set_xlabel("Training steps")
            if k == 0:
                ax.set_ylabel("Per-seed paired d, nats")
            if i == 0 and k == 0:
                ax.legend(loc="upper left")
    fig.suptitle("Raw per-seed paired d = (minimal system) − (bare role), 5 seeds per cell", y=1.0)
    fig.tight_layout()
    savefig_paper(fig, "issue_611/split_per_seed_scatter", dir=FIG_ROOT)
    plt.close(fig)


def fig_leakage_allocation(payload: dict) -> None:
    """Exploratory: leakage allocation across the two non-own probes per arm."""
    set_paper_style("blog")
    rows = payload["leakage_allocation_exploratory"]
    idx = {(r["arm"], r["persona"], r["max_steps"]): r for r in rows}
    probe_color = {
        "wrong": paper_palette_role("baseline"),
        "default": paper_palette_role("primary"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    width = 0.38
    x = np.arange(len(STEPS))
    for k, persona in enumerate(PERSONAS):
        ax = axes[k]
        for a, arm in enumerate(ARMS):
            pos = x + (a - 0.5) * width
            wrong = np.array([idx[(arm, persona, s)]["wrong_logp"] for s in STEPS])
            default = np.array([idx[(arm, persona, s)]["default_logp"] for s in STEPS])
            ax.bar(
                pos,
                wrong,
                width=width,
                color=probe_color["wrong"],
                alpha=0.8,
                label="Wrong-persona probe" if (k == 0 and a == 0) else None,
            )
            ax.bar(
                pos,
                default,
                width=width,
                bottom=wrong,
                color=probe_color["default"],
                alpha=0.8,
                label="Default-assistant probe" if (k == 0 and a == 0) else None,
            )
            # No in-bar arm annotations: abbreviated "sys"/"role" labels are not
            # reader-facing (round-2 critique); the suptitle names the left/right
            # bar assignment in full.
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in STEPS])
        ax.set_title(persona)
        ax.set_xlabel("Training steps")
        if k == 0:
            ax.set_ylabel("Δlog P, nats (stacked)")
            ax.legend(loc="upper left")
    fig.suptitle(
        "Descriptive leakage allocation across the two non-own probes "
        "(left bar: minimal system; right bar: bare role header)",
        y=1.02,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_611/leakage_allocation", dir=FIG_ROOT)
    plt.close(fig)


def fig_logp_margin_agreement(payload: dict) -> None:
    """Exploratory: d_logp vs d_margin agreement (the saturation-signature view)."""
    set_paper_style("blog")
    pc = _contrast_index(payload)
    sat_thresh = payload["parameters"]["saturation_flag_nats"]
    probe_color = {
        "wrong": paper_palette_role("baseline"),
        "default": paper_palette_role("primary"),
    }
    marker_for = {"pirate": "o", "villain": "^"}

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    lims = [-3.5, 3.8]
    ax.plot(lims, lims, color="0.45", linewidth=0.9)
    ax.fill_between(
        lims,
        [v - sat_thresh for v in lims],
        [v + sat_thresh for v in lims],
        color="0.8",
        alpha=0.35,
        linewidth=0,
    )
    for probe in PROBES:
        for persona in PERSONAS:
            for s in STEPS:
                dl = pc[(probe, persona, s, "logp")]["point"]
                dm = pc[(probe, persona, s, "margin")]["point"]
                flagged = pc[(probe, persona, s, "logp")]["saturation_compressed"]
                ax.plot(
                    dl,
                    dm,
                    linestyle="none",
                    marker=marker_for[persona],
                    markersize=9 if flagged else 7,
                    markerfacecolor=probe_color[probe],
                    markeredgecolor=paper_palette_role("accent") if flagged else "white",
                    markeredgewidth=1.8 if flagged else 0.8,
                    alpha=0.9,
                )
                if flagged:
                    ax.annotate(
                        f"{persona} s{s}",
                        (dl, dm),
                        textcoords="offset points",
                        xytext=(8, -4),
                        fontsize=8,
                        color="0.25",
                    )
    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color=probe_color["wrong"],
            label="Wrong-persona probe",
        ),
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color=probe_color["default"],
            label="Default-assistant probe",
        ),
        plt.Line2D([], [], linestyle="none", marker="o", color="0.6", label="pirate (circles)"),
        plt.Line2D([], [], linestyle="none", marker="^", color="0.6", label="villain (triangles)"),
    ]
    ax.legend(handles=handles, loc="upper left")
    ax.set_xlabel("Paired d in Δlog P space, nats")
    ax.set_ylabel("Paired d in EOS-margin space, nats")
    ax.set_title(
        "Space agreement per contrast cell. Shaded: ±0.5-nat envelope;\n"
        "outlined points are ceiling-compressed (margin authoritative)."
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_611/logp_margin_agreement", dir=FIG_ROOT)
    plt.close(fig)


def main() -> None:
    payload = _load()
    log.info("[phase=figures] rendering figures to %s", FIG_ROOT / "issue_611")
    fig_split_verdict_grid(payload)
    fig_base_prior_decomposition(payload)
    fig_own_slot_regime(payload)
    fig_split_per_seed_scatter(payload)
    fig_leakage_allocation(payload)
    fig_logp_margin_agreement(payload)
    log.info("[phase=done] all figures written")


if __name__ == "__main__":
    main()

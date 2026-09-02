"""Issue #2658 unit 12 — plan section 7 figure code.

Renders the section 7 reportable quantities off unit 11's inference report
JSON (``issue2658_inference.py --phase report`` / ``run_inference`` output):

- PRIMARY: equal-prompt macro within-prompt AUROC per (row, comparator) with
  POINTWISE hierarchical-bootstrap intervals (aggregate) + the per-prompt
  AUROC points behind each macro (per-unit companion).
- Holm-adjusted p-values + significance, rendered SEPARATELY from the
  intervals (aggregate) + the raw p with its pointwise Monte Carlo interval
  (per-unit companion).
- The C5-minus-C2 delta with its one-sided pointwise lower bound (aggregate)
  + per-prompt paired deltas (companion).
- The C0-C5 comparator ladder (aggregate median) + the per-row ladder view
  (companion), off unit 9's final ledger records.
- The committed prospective not-estimable cell ledger (per-row estimable /
  not-estimable-by-cause counts) — the revised denominators of plan §8.

NOT-ESTIMABLE DISCIPLINE (the load-bearing requirement): a not-estimable
(row, comparator) — whether from the committed ``c2_c3_partition`` or from a
plan §8 row-gate failure returned by unit 11 — renders as an explicit
labeled absence ("not estimable (<cause>)") and NEVER as a zero bar, a zero
point, or a gap indistinguishable from a measured zero. A missing prospective
ledger REFUSES loudly (unit 11's loader), never "all cells estimable".

STYLE CONTRACT (paper-plots skill §2/§3.8/§3.8-bis/§3.9 + standing user
directives): no ``fig.text`` caption blocks — axes, ticks, legend, and panel
titles ONLY; short reader-facing labels (no internal slugs, no parameter
literals); ONE color = ONE meaning across the whole set via the single-source
``COLOR_BY_MEANING``; every save goes through ``savefig_paper`` so each PNG
gets its provenance sidecar (commit sha, dirty flag, per-point data, text).

There are no sealed-test labels yet: the ``demo`` phase builds a
FULL-REGISTRY synthetic report by running unit 9's REAL ladder and unit 11's
REAL ``run_inference`` on synthetic RowData under the COMMITTED partition
(harmful_compliance not-estimable; realized Holm family sizes 10/11/10), with
recorded demo-registry overrides. Every report-driven panel is visibly
labeled "synthetic smoke"; the cell-ledger figure is driven by the committed
prospective ledger and labeled as pre-registered bookkeeping instead (it is
real, committed, and not a result).

Launch (VM-side runs carry the shared-VM thread caps):
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue2658_figures.py --phase demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # #847 thread caps must bind BEFORE numpy/matplotlib import

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # non-interactive backend before pyplot

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2658_common as C  # noqa: E402
import issue2658_comparators as U  # noqa: E402
import issue2658_frames as F  # noqa: E402
import issue2658_inference as INF  # noqa: E402
import issue2658_power as PW  # noqa: E402
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)


class FigureInputError(C.Issue2658GuardError):
    """Malformed / absent figure input (fail fast, no silent default)."""


POINTS_SCHEMA = "i2658-figure-points-v1"
DEFAULT_FIG_DIR = F.REPO_ROOT / "figures/issue_2658"
SYNTHETIC_TAG = "synthetic smoke"

# ---------------------------------------------------------------------------
# Reader-facing labels (paper-plots skill §3.5: no internal slugs on canvas).
# ---------------------------------------------------------------------------
ROW_LABELS: dict[str, str] = {
    "evil": "Malicious intent",
    "sycophancy": "Sycophancy",
    "hallucination": "Hallucination",
    "refusal": "Refusal",
    "assistantness": "Assistant behavior",
    "casualness": "Casual register",
    "impoliteness": "Impoliteness",
    "harmful_compliance": "Harmful compliance",
    "correctness_math": "Math correctness",
    "correctness_mmlu_pro": "MMLU-Pro correctness",
    "correctness_code": "Code correctness",
}

COMPARATOR_ORDER: tuple[str, ...] = tuple(U.COMPARATORS)  # ladder order C0..C5

COMPARATOR_LABELS: dict[str, str] = {
    "c0_nuisance": "Surface features",
    "c1_text_prompt": "Prompt text",
    "c1_text_answer": "Answer text",
    "c1_text_combined": "Prompt + answer text",
    "c2_direction_dot": "External direction (zero-shot)",
    "c3_direction_calibrated": "External direction (calibrated)",
    "c4_devmean_calibrated": "Class-mean direction",
    "c5_full_probe": "Supervised probe",
    "c5_minus_c2": "Supervised probe − external direction",
}

# ---------------------------------------------------------------------------
# ONE COLOR = ONE MEANING, single-sourced. Every figure imports from here;
# tests pin uniqueness + usage. Wong hexes assigned by meaning, not position.
# ---------------------------------------------------------------------------
COLOR_BY_MEANING: dict[str, str] = {
    "c0_nuisance": "#F0E442",  # yellow (Wong)
    "c1_text_prompt": "#56B4E9",  # sky blue (Wong)
    "c1_text_answer": "#009E73",  # bluish green (Wong)
    "c1_text_combined": "#CC79A7",  # reddish purple (Wong)
    "c2_direction_dot": "#E69F00",  # orange (Wong)
    "c3_direction_calibrated": "#D55E00",  # vermillion (Wong)
    "c4_devmean_calibrated": "#8064A2",  # purple (blog palette)
    "c5_full_probe": "#0072B2",  # blue (Wong)
    "c5_minus_c2": "#000000",  # black — the paired contrast
    "not_estimable": "#9a9a9a",  # gray — labeled absence, never a zero
    "cells_estimable": "#c7d9ec",  # light blue-gray — ledger counts
    "cause_bank_too_small": "#8c564b",  # brown — ledger cause
    "cause_extraction_barred": "#5A6975",  # slate — ledger cause
    "reference": "#bbbbbb",  # light gray — chance/alpha/zero lines
}

_CAUSE_COLOR_KEY = {
    F.CAUSE_BANK_TOO_SMALL: "cause_bank_too_small",
    F.CAUSE_EXTRACTION_BARRED: "cause_extraction_barred",
}
_CAUSE_LABEL = {
    F.CAUSE_BANK_TOO_SMALL: "not estimable — bank too small",
    F.CAUSE_EXTRACTION_BARRED: "not estimable — extraction barred",
}

# Aggregate figure stem -> its per-unit companion stem (tests pin coverage).
AGGREGATE_COMPANIONS: dict[str, str] = {
    "issue2658_macro_auroc": "issue2658_macro_auroc_per_prompt",
    "issue2658_holm_adjusted_p": "issue2658_raw_p_mc",
    "issue2658_delta_c5_minus_c2": "issue2658_delta_per_prompt",
    "issue2658_comparator_ladder": "issue2658_comparator_ladder_per_row",
}
# Already at per-unit grain (per row x cause); no coarser aggregate exists.
STANDALONE_FIGURES: tuple[str, ...] = ("issue2658_cell_ledger",)


def _short_reason(reason: str) -> str:
    if "no frozen external direction" in reason:
        return "no direction"
    if "production gate failed" in reason:
        return "gate failed"
    return "see report"


def _tag(title: str, synthetic: bool) -> str:
    return f"{title} — {SYNTHETIC_TAG}" if synthetic else title


def _jitter(key: str, n: int, scale: float = 0.10) -> np.ndarray:
    """Deterministic per-figure jitter (same input -> same rendered arrays)."""
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**63)
    return np.random.default_rng(seed).uniform(-scale, scale, size=n)


def _mark_absent(ax: plt.Axes, y: float, reason: str) -> str:
    """Render a labeled absence at data-y ``y`` (axes-fraction x)."""
    return _mark_absent_short(ax, y, _short_reason(reason))


def _mark_absent_short(ax: plt.Axes, y: float, short: str) -> str:
    """Render an ALREADY-shortened absence label (multi-reason rows join shorts)."""
    label = f"not estimable ({short})"
    ax.text(
        0.02,
        y,
        label,
        transform=mtransforms.blended_transform_factory(ax.transAxes, ax.transData),
        fontsize=7,
        color=COLOR_BY_MEANING["not_estimable"],
        va="center",
        ha="left",
        style="italic",
    )
    return label


def _ordered_rows(candidates: Iterable[str]) -> list[str]:
    """Registered rows first (registry order), then any others sorted."""
    cand = list(candidates)
    known = [r for r in C.ROW_IDS if r in cand]
    unknown = sorted(set(cand) - set(C.ROW_IDS))
    return known + unknown


def _row_label(row: str) -> str:
    """Reader-facing label; unregistered (test) row ids render as themselves."""
    return ROW_LABELS.get(row, row.replace("_", " "))


def _row_axis(ax: plt.Axes, rows: Sequence[str]) -> dict[str, float]:
    """Rows on y, first row at the top; returns row -> y position."""
    n = len(rows)
    pos = {row: float(n - 1 - i) for i, row in enumerate(rows)}
    ax.set_yticks([pos[r] for r in rows])
    ax.set_yticklabels([_row_label(r) for r in rows])
    ax.set_ylim(-0.7, n - 0.3)
    return pos


# ---------------------------------------------------------------------------
# Input loaders (fail fast; schema pinned to the producer's constant).
# ---------------------------------------------------------------------------
def load_report(path: Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text())
    if body.get("schema") != INF.REPORT_SCHEMA:
        raise FigureInputError(
            f"report {path} schema {body.get('schema')!r} != {INF.REPORT_SCHEMA!r}"
        )
    for key in ("families", "rows", "not_estimable", "partition", "family_sizes"):
        if key not in body:
            raise FigureInputError(f"report {path} missing required key {key!r}")
    return body


def load_points(path: Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text())
    if body.get("schema") != POINTS_SCHEMA:
        raise FigureInputError(f"points {path} schema {body.get('schema')!r} != {POINTS_SCHEMA!r}")
    if "rows" not in body:
        raise FigureInputError(f"points {path} missing 'rows'")
    return body


def load_cell_ledgers(manifest_path: Path | None = None) -> dict[str, Any]:
    """Committed prospective ledger via unit 11's loader — REFUSES loudly when
    the manifest or its ledger is absent (never 'all cells estimable')."""
    return INF.load_prospective_ledger(manifest_path)


# ---------------------------------------------------------------------------
# Per-prompt points (the per-unit grain behind the macro AUROC), computed
# through unit 8's registered within-prompt AUROC — never a second AUROC.
# ---------------------------------------------------------------------------
def prompt_points_from_panel(panel: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pids = np.asarray(panel.prompt_ids)
    labels = np.asarray(panel.labels).astype(bool)
    cells = np.asarray(panel.cells)
    for comp, scores in sorted(panel.scores.items()):
        s = np.asarray(scores, dtype=np.float64)
        prompts: list[dict[str, Any]] = []
        n_concordant = 0
        for pid in np.unique(pids):
            rows = np.nonzero(pids == pid)[0]
            lab = labels[rows]
            if lab.all() or not lab.any():
                n_concordant += 1
                continue
            prompts.append(
                {
                    "prompt_id": str(pid),
                    "auroc": float(PW.within_prompt_auroc(s[rows], lab)),
                    "n_pos": int(lab.sum()),
                    "n_neg": int((~lab).sum()),
                    "cell": str(cells[rows[0]]),
                }
            )
        out[comp] = {"prompts": prompts, "n_concordant_prompts": n_concordant}
    return out


def build_points(rows_input: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Per-prompt points for every ESTIMABLE row (a not-estimable row records
    its reason — its prompts never enter a per-unit view as if measured)."""
    rows_out: dict[str, Any] = {}
    for row, ri in sorted(rows_input.items()):
        rep_row = report["rows"][row]
        if not rep_row["estimable"]:
            reason = report["not_estimable"]["C5"].get(row, "row not estimable")
            rows_out[row] = {"estimable": False, "reason": reason}
            continue
        comp_points = prompt_points_from_panel(ri.panel)
        entry: dict[str, Any] = {"estimable": True, "comparators": comp_points}
        if {"c5_full_probe", "c2_direction_dot"} <= set(comp_points):
            by_pid = {p["prompt_id"]: p for p in comp_points["c2_direction_dot"]["prompts"]}
            deltas = []
            for p in comp_points["c5_full_probe"]["prompts"]:
                q = by_pid.get(p["prompt_id"])
                if q is not None:
                    deltas.append(
                        {
                            "prompt_id": p["prompt_id"],
                            "delta": p["auroc"] - q["auroc"],
                            "cell": p["cell"],
                        }
                    )
            entry["delta_prompts"] = deltas
        rows_out[row] = entry
    return {
        "schema": POINTS_SCHEMA,
        "rows": rows_out,
        "metadata": as_metadata_dict(git_provenance(), phase="figures"),
    }


# ---------------------------------------------------------------------------
# Figures. Every function returns (fig, plotted) where ``plotted`` holds the
# exact arrays handed to matplotlib (determinism + zero-suppression tests).
# ---------------------------------------------------------------------------
def fig_macro_auroc(report: dict[str, Any], *, synthetic: bool) -> tuple[plt.Figure, dict]:
    rows = _ordered_rows(report["rows"])
    if not rows:
        raise FigureInputError("report carries no rows")
    for row in rows:
        # A not-estimable label must never be inferred from a missing field:
        # an ESTIMABLE row with no descriptive block is a producer-contract
        # break and RAISES instead of rendering as a labeled absence.
        if report["rows"][row]["estimable"] and "descriptive" not in report["rows"][row]:
            raise FigureInputError(
                f"row {row!r} is estimable but the report carries no descriptive block; "
                "refusing to render a producer-contract break as a not-estimable label"
            )
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(rows) + 1.6))
    pos = _row_axis(ax, rows)
    plotted: dict[str, Any] = {"rows": rows, "series": {}, "absent": []}
    offsets = {"c2_direction_dot": +0.18, "c5_full_probe": -0.18}
    family_of = {"c2_direction_dot": "C2", "c5_full_probe": "C5"}
    for comp, off in offsets.items():
        xs, ys, lo, hi = [], [], [], []
        for row in rows:
            desc = report["rows"][row].get("descriptive", {})
            if comp in desc:
                xs.append(float(desc[comp]["macro_auroc"]))
                ys.append(pos[row] + off)
                ci = desc[comp]["macro_ci_pointwise"]
                # Non-negative offsets from the value, never raw CI bounds: a
                # percentile CI can invert around the point at tiny n.
                lo.append(max(xs[-1] - float(ci[0]), 0.0))
                hi.append(max(float(ci[1]) - xs[-1], 0.0))
            else:
                reason = report["not_estimable"][family_of[comp]].get(row, "not reported")
                label = _mark_absent(ax, pos[row] + off, reason)
                plotted["absent"].append(
                    {"row": row, "comparator": comp, "reason": reason, "label": label}
                )
        ax.errorbar(
            xs,
            ys,
            xerr=[lo, hi],
            fmt="o",
            ms=4.5,
            lw=1.2,
            capsize=2,
            color=COLOR_BY_MEANING[comp],
            label=COMPARATOR_LABELS[comp],
        )
        plotted["series"][comp] = {"x": xs, "y": ys, "ci_lo": lo, "ci_hi": hi}
    ax.axvline(0.5, ls="--", lw=1.0, color=COLOR_BY_MEANING["reference"], label="chance (0.5)")
    ax.set_xlabel("Macro within-prompt AUROC")
    ax.set_title(_tag("Primary metric with pointwise intervals", synthetic), loc="left")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig, plotted


def fig_macro_auroc_per_prompt(
    report: dict[str, Any], points: dict[str, Any], *, synthetic: bool
) -> tuple[plt.Figure, dict]:
    rows = _ordered_rows(report["rows"])
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(rows) + 1.6))
    pos = _row_axis(ax, rows)
    plotted: dict[str, Any] = {"rows": rows, "series": {}, "absent": []}
    offsets = {"c2_direction_dot": +0.16, "c5_full_probe": -0.16}
    for comp, off in offsets.items():
        xs: list[float] = []
        ys: list[float] = []
        ids: list[str] = []
        for row in rows:
            entry = points["rows"].get(row)
            if entry is None or not entry.get("estimable"):
                continue
            comp_entry = entry["comparators"].get(comp)
            if comp_entry is None:
                continue
            aur = [p["auroc"] for p in comp_entry["prompts"]]
            jit = _jitter(f"{row}|{comp}", len(aur), scale=0.10)
            xs.extend(aur)
            ys.extend((pos[row] + off + jit).tolist())
            ids.extend(p["prompt_id"] for p in comp_entry["prompts"])
        ax.scatter(
            xs,
            ys,
            s=8,
            alpha=0.55,
            color=COLOR_BY_MEANING[comp],
            label=COMPARATOR_LABELS[comp],
            linewidths=0,
        )
        plotted["series"][comp] = {"x": xs, "y": ys, "prompt_ids": ids}
    for row in rows:
        entry = points["rows"].get(row)
        if entry is None or not entry.get("estimable"):
            reason = (entry or {}).get("reason", "not reported")
            label = _mark_absent(ax, pos[row], reason)
            plotted["absent"].append({"row": row, "reason": reason, "label": label})
    ax.axvline(0.5, ls="--", lw=1.0, color=COLOR_BY_MEANING["reference"], label="chance (0.5)")
    ax.set_xlabel("Per-prompt AUROC (discordant prompts)")
    ax.set_title(_tag("Per-prompt AUROC behind each macro", synthetic), loc="left")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig, plotted


_FAMILY_COMPARATOR = {
    "C2": "c2_direction_dot",
    "C5": "c5_full_probe",
    "C5_minus_C2": "c5_minus_c2",
}


def _family_panels(
    report: dict[str, Any], synthetic: bool
) -> tuple[plt.Figure, list[plt.Axes], list[str], dict[str, float]]:
    rows = _ordered_rows(report["rows"])
    fams = list(_FAMILY_COMPARATOR)
    fig, axes = plt.subplots(1, len(fams), figsize=(10.5, 0.42 * len(rows) + 1.7), sharey=True)
    pos = _row_axis(axes[0], rows)
    for ax, fam in zip(axes, fams):
        title = COMPARATOR_LABELS[_FAMILY_COMPARATOR[fam]]
        if ax is axes[0]:
            title = _tag(title, synthetic)
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xscale("log")
    return fig, list(axes), rows, pos


def fig_holm_adjusted_p(report: dict[str, Any], *, synthetic: bool) -> tuple[plt.Figure, dict]:
    fig, axes, rows, pos = _family_panels(report, synthetic)
    plotted: dict[str, Any] = {"rows": rows, "families": {}, "absent": []}
    for ax, fam in zip(axes, _FAMILY_COMPARATOR):
        fam_rep = report["families"][fam]
        color = COLOR_BY_MEANING[_FAMILY_COMPARATOR[fam]]
        alpha = float(fam_rep["alpha"])
        adj: dict[str, float] = fam_rep["holm_adjusted_p"]
        sig: dict[str, bool] = fam_rep["significant"]
        entries = {"x": [], "y": [], "significant": []}
        for row in rows:
            if row in adj:
                entries["x"].append(float(adj[row]))
                entries["y"].append(pos[row])
                entries["significant"].append(bool(sig[row]))
            else:
                reason = report["not_estimable"][fam].get(row, "not reported")
                label = _mark_absent(ax, pos[row], reason)
                plotted["absent"].append(
                    {"family": fam, "row": row, "reason": reason, "label": label}
                )
        for filled in (True, False):
            xs = [x for x, s in zip(entries["x"], entries["significant"]) if s is filled]
            ys = [y for y, s in zip(entries["y"], entries["significant"]) if s is filled]
            ax.scatter(
                xs,
                ys,
                s=26,
                marker="o",
                facecolors=color if filled else "none",
                edgecolors=color,
                linewidths=1.2,
                label="significant" if filled else "not significant",
            )
        ax.axvline(alpha, ls="--", lw=1.0, color=COLOR_BY_MEANING["reference"])
        ax.set_xlabel("Holm-adjusted p")
        plotted["families"][fam] = {**entries, "alpha": alpha}
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    return fig, plotted


def fig_raw_p_mc(report: dict[str, Any], *, synthetic: bool) -> tuple[plt.Figure, dict]:
    fig, axes, rows, pos = _family_panels(report, synthetic)
    plotted: dict[str, Any] = {"rows": rows, "families": {}, "absent": []}
    for ax, fam in zip(axes, _FAMILY_COMPARATOR):
        fam_rep = report["families"][fam]
        color = COLOR_BY_MEANING[_FAMILY_COMPARATOR[fam]]
        xs, ys, lo, hi = [], [], [], []
        for row in rows:
            test = fam_rep["tests"].get(row)
            if test is not None:
                p = float(test["p"])
                ival = test["mc_interval"]
                xs.append(p)
                ys.append(pos[row])
                lo.append(max(p - float(ival[0]), 0.0))
                hi.append(max(float(ival[1]) - p, 0.0))
            else:
                reason = report["not_estimable"][fam].get(row, "not reported")
                label = _mark_absent(ax, pos[row], reason)
                plotted["absent"].append(
                    {"family": fam, "row": row, "reason": reason, "label": label}
                )
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=4.5, lw=1.1, capsize=2, color=color)
        kind = "bootstrap p" if fam == "C5_minus_C2" else "permutation p"
        ax.set_xlabel(f"{kind} (Monte Carlo interval)")
        plotted["families"][fam] = {"x": xs, "y": ys, "mc_lo": lo, "mc_hi": hi}
    fig.tight_layout()
    return fig, plotted


def fig_delta(report: dict[str, Any], *, synthetic: bool) -> tuple[plt.Figure, dict]:
    rows = _ordered_rows(report["rows"])
    fam_rep = report["families"]["C5_minus_C2"]
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(rows) + 1.6))
    pos = _row_axis(ax, rows)
    plotted: dict[str, Any] = {"rows": rows, "x": [], "y": [], "lower": [], "absent": []}
    for row in rows:
        test = fam_rep["tests"].get(row)
        if test is not None:
            plotted["x"].append(float(test["delta_hat"]))
            plotted["y"].append(pos[row])
            plotted["lower"].append(float(test["one_sided_lower_bound"]))
        else:
            reason = report["not_estimable"]["C5_minus_C2"].get(row, "not reported")
            label = _mark_absent(ax, pos[row], reason)
            plotted["absent"].append({"row": row, "reason": reason, "label": label})
    # Offsets are non-negative by contract (lower bound <= delta_hat); clamp so
    # a degenerate-SE cell can never crash the render with a negative xerr.
    xerr_lo = [max(x - lb, 0.0) for x, lb in zip(plotted["x"], plotted["lower"])]
    ax.errorbar(
        plotted["x"],
        plotted["y"],
        xerr=[xerr_lo, [0.0] * len(plotted["x"])],
        fmt="o",
        ms=5,
        lw=1.2,
        capsize=2,
        color=COLOR_BY_MEANING["c5_minus_c2"],
        label=COMPARATOR_LABELS["c5_minus_c2"],
    )
    ax.axvline(0.0, ls="--", lw=1.0, color=COLOR_BY_MEANING["reference"], label="no difference")
    ax.set_xlabel("Macro AUROC difference (one-sided lower bound)")
    ax.set_title(_tag("Supervised probe minus external direction", synthetic), loc="left")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig, plotted


def fig_delta_per_prompt(
    report: dict[str, Any], points: dict[str, Any], *, synthetic: bool
) -> tuple[plt.Figure, dict]:
    rows = _ordered_rows(report["rows"])
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(rows) + 1.6))
    pos = _row_axis(ax, rows)
    plotted: dict[str, Any] = {"rows": rows, "x": [], "y": [], "prompt_ids": [], "absent": []}
    for row in rows:
        entry = points["rows"].get(row)
        deltas = (entry or {}).get("delta_prompts")
        if entry is None or not entry.get("estimable") or deltas is None:
            reason = (entry or {}).get(
                "reason", report["not_estimable"]["C5_minus_C2"].get(row, "not reported")
            )
            label = _mark_absent(ax, pos[row], reason)
            plotted["absent"].append({"row": row, "reason": reason, "label": label})
            continue
        vals = [d["delta"] for d in deltas]
        jit = _jitter(f"{row}|delta", len(vals), scale=0.14)
        plotted["x"].extend(vals)
        plotted["y"].extend((pos[row] + jit).tolist())
        plotted["prompt_ids"].extend(d["prompt_id"] for d in deltas)
    ax.scatter(
        plotted["x"],
        plotted["y"],
        s=8,
        alpha=0.55,
        color=COLOR_BY_MEANING["c5_minus_c2"],
        linewidths=0,
        label=COMPARATOR_LABELS["c5_minus_c2"],
    )
    ax.axvline(0.0, ls="--", lw=1.0, color=COLOR_BY_MEANING["reference"], label="no difference")
    ax.set_xlabel("Per-prompt AUROC difference")
    ax.set_title(_tag("Per-prompt paired differences", synthetic), loc="left")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig, plotted


def _ladder_value(rec: dict[str, Any]) -> float | None:
    if rec.get("status") == "scored" and "test_macro_auroc_descriptive" in rec:
        return float(rec["test_macro_auroc_descriptive"])
    return None


def fig_comparator_ladder(
    ladder: dict[tuple[str, str], dict[str, Any]], *, synthetic: bool
) -> tuple[plt.Figure, dict]:
    comps = [c for c in COMPARATOR_ORDER if any(k[1] == c for k in ladder)]
    if not comps:
        raise FigureInputError("no comparator records to plot")
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(comps) + 1.5))
    n = len(comps)
    pos = {c: float(n - 1 - i) for i, c in enumerate(comps)}
    ax.set_yticks([pos[c] for c in comps])
    ax.set_yticklabels([COMPARATOR_LABELS[c] for c in comps])
    ax.set_ylim(-0.7, n - 0.3)
    plotted: dict[str, Any] = {"comparators": comps, "median": [], "n_rows": []}
    for comp in comps:
        vals = [
            v
            for (row, c), rec in sorted(ladder.items())
            if c == comp and (v := _ladder_value(rec)) is not None
        ]
        if not vals:
            raise FigureInputError(f"comparator {comp!r} has no scored rows")
        med = float(np.median(vals))
        plotted["median"].append(med)
        plotted["n_rows"].append(len(vals))
        ax.scatter([med], [pos[comp]], s=42, color=COLOR_BY_MEANING[comp], zorder=3, linewidths=0)
    ax.axvline(0.5, ls="--", lw=1.0, color=COLOR_BY_MEANING["reference"], label="chance (0.5)")
    ax.set_xlabel("Median test macro AUROC across constructs")
    ax.set_title(_tag("Comparator ladder", synthetic), loc="left")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig, plotted


def fig_comparator_ladder_per_row(
    ladder: dict[tuple[str, str], dict[str, Any]], *, synthetic: bool
) -> tuple[plt.Figure, dict]:
    rows = _ordered_rows({k[0] for k in ladder})
    comps = [c for c in COMPARATOR_ORDER if any(k[1] == c for k in ladder)]
    if not rows or not comps:
        raise FigureInputError("no (row, comparator) records to plot")
    fig, ax = plt.subplots(figsize=(7.5, 0.46 * len(rows) + 1.9))
    pos = _row_axis(ax, rows)
    plotted: dict[str, Any] = {"rows": rows, "series": {}, "absent": []}
    for comp in comps:
        xs, ys = [], []
        for row in rows:
            rec = ladder.get((row, comp))
            if rec is None:
                continue
            val = _ladder_value(rec)
            if val is None:
                # The record's OWN reason field, never its status and never a
                # hardcoded cause (a status=="scored" record with a missing
                # value, or any future status, carries its own reason).
                reason = rec.get("reason", "not reported")
                plotted["absent"].append({"row": row, "comparator": comp, "reason": reason})
                continue
            xs.append(val)
            ys.append(pos[row])
        ax.scatter(
            xs,
            ys,
            s=16,
            alpha=0.85,
            color=COLOR_BY_MEANING[comp],
            label=COMPARATOR_LABELS[comp],
            linewidths=0,
        )
        plotted["series"][comp] = {"x": xs, "y": ys}
    absent_rows = sorted({a["row"] for a in plotted["absent"]})
    for row in absent_rows:
        row_absent = [a for a in plotted["absent"] if a["row"] == row]
        # One label per row; distinct comparator reasons join as shortened forms.
        shorts = sorted({_short_reason(a["reason"]) for a in row_absent})
        label = _mark_absent_short(ax, pos[row] + 0.30, "; ".join(shorts))
        for a in row_absent:
            a["label"] = label
    ax.axvline(0.5, ls="--", lw=1.0, color=COLOR_BY_MEANING["reference"], label="chance (0.5)")
    ax.set_xlabel("Test macro AUROC")
    ax.set_title(_tag("Comparator ladder by construct", synthetic), loc="left")
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    fig.tight_layout()
    return fig, plotted


def fig_cell_ledger(ledgers: dict[str, Any]) -> tuple[plt.Figure, dict]:
    """Committed PROSPECTIVE ledger (pre-registered before generation): each
    row's revised denominator — estimable cells vs not-estimable by cause.
    Real committed bookkeeping, not a result; deliberately NOT tagged
    synthetic."""
    if not ledgers:
        raise FigureInputError("empty cell-ledger mapping — refusing to render")
    rows = _ordered_rows(ledgers)
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(rows) + 1.7))
    pos = _row_axis(ax, rows)
    causes = sorted(_CAUSE_COLOR_KEY)
    plotted: dict[str, Any] = {"rows": rows, "estimable": [], "by_cause": {c: [] for c in causes}}
    ys = [pos[r] for r in rows]
    est = [int(ledgers[r].n_cells_estimable) for r in rows]
    ax.barh(
        ys,
        est,
        height=0.62,
        color=COLOR_BY_MEANING["cells_estimable"],
        label="estimable cells",
    )
    plotted["estimable"] = est
    left = list(map(float, est))
    for cause in causes:
        counts = [sum(1 for e in ledgers[r].excluded if e["cause"] == cause) for r in rows]
        ax.barh(
            ys,
            counts,
            left=left,
            height=0.62,
            color=COLOR_BY_MEANING[_CAUSE_COLOR_KEY[cause]],
            label=_CAUSE_LABEL[cause],
        )
        left = [x + c for x, c in zip(left, counts)]
        plotted["by_cause"][cause] = counts
    totals = {int(ledgers[r].n_cells) for r in rows}
    if len(totals) == 1:
        ax.set_xlim(0, max(totals) + 0.5)
    ax.set_xlabel("Source-by-stratum cells per construct")
    ax.set_title("Pre-registered estimable cells (prospective ledger)", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig, plotted


# ---------------------------------------------------------------------------
# Render-all entry (demo and production share this path).
# ---------------------------------------------------------------------------
def render_all(
    report: dict[str, Any],
    points: dict[str, Any],
    ladder: dict[tuple[str, str], dict[str, Any]] | None,
    ledgers: dict[str, Any],
    fig_dir: Path,
    *,
    synthetic: bool,
) -> dict[str, dict[str, Path]]:
    set_paper_style("blog")
    out: dict[str, dict[str, Path]] = {}

    def _save(stem: str, fig: plt.Figure) -> None:
        out[stem] = savefig_paper(fig, stem, dir=fig_dir)
        plt.close(fig)

    fig, _ = fig_macro_auroc(report, synthetic=synthetic)
    _save("issue2658_macro_auroc", fig)
    fig, _ = fig_macro_auroc_per_prompt(report, points, synthetic=synthetic)
    _save("issue2658_macro_auroc_per_prompt", fig)
    fig, _ = fig_holm_adjusted_p(report, synthetic=synthetic)
    _save("issue2658_holm_adjusted_p", fig)
    fig, _ = fig_raw_p_mc(report, synthetic=synthetic)
    _save("issue2658_raw_p_mc", fig)
    fig, _ = fig_delta(report, synthetic=synthetic)
    _save("issue2658_delta_c5_minus_c2", fig)
    fig, _ = fig_delta_per_prompt(report, points, synthetic=synthetic)
    _save("issue2658_delta_per_prompt", fig)
    if ladder is not None:
        fig, _ = fig_comparator_ladder(ladder, synthetic=synthetic)
        _save("issue2658_comparator_ladder", fig)
        fig, _ = fig_comparator_ladder_per_row(ladder, synthetic=synthetic)
        _save("issue2658_comparator_ladder_per_row", fig)
    fig, _ = fig_cell_ledger(ledgers)
    _save("issue2658_cell_ledger", fig)

    for stem, paths in out.items():
        png = paths.get("png")
        if png is None or not Path(png).exists() or Path(png).stat().st_size == 0:
            raise FigureInputError(f"figure {stem} did not produce a non-empty PNG")
    return out


# ---------------------------------------------------------------------------
# Demo: full-registry synthetic report through unit 9's REAL ladder + unit
# 11's REAL run_inference (committed partition; recorded registry overrides).
# ---------------------------------------------------------------------------
DEMO_REGISTRY_KWARGS: dict[str, Any] = {
    "n_perm_initial": 99,
    "perm_chunk_initial": (33,) * 3,
    "n_perm_extended": 999,
    "perm_chunk_extension": (100,) * 9,
    "n_boot": 40,
    "boot_chunk": 20,
    "n_boot_extended": 80,
    "n_ci_draws": 200,
    "min_discordant_prompts": 8,
    "min_answers_per_class": 16,
    "min_prompts_per_class": 4,
    # Realized per-cell floor, scaled to the synthetic demo like every other
    # floor above. The PRODUCTION value is 15; the demo's synthesized rows carry
    # 5-10 prompts per cell (2 on the gate-fail row), so the production floor
    # would exclude EVERY demo cell and render every demo row not-estimable —
    # the figures would then demonstrate nothing. 2 keeps the gate non-vacuous
    # at roughly the same ratio as min_discordant_prompts (8 vs 100).
    "min_discordant_prompts_per_cell": 2,
}

# Per-row synthetic effect sizes: a spread so the demo shows significant AND
# non-significant rows. correctness_code is synthesized structurally too
# small to pass the demo row gates, demonstrating the analysis-time
# not-estimable path in the committed figures.
_DEMO_EFFECTS = (2.0, 1.2, 0.7, 0.35, 0.0)
_DEMO_GATE_FAIL_ROW = "correctness_code"


def _demo_ledger(row: str, rd: Any, committed: dict[str, Any]) -> Any:
    """Synthetic per-row cell ledger mirroring the committed not-estimable
    FRACTION (scaled to the synthetic cell count) so denominator revision is
    exercised; the gate-fail demo row keeps all cells (its failure must come
    from the row gates, not the ledger)."""
    cells = sorted({f"{r.source_frame}|{r.stratum}" for r in rd.rows})
    led = committed[row]
    frac = 1.0 - led.n_cells_estimable / led.n_cells
    n_excl = 0 if row == _DEMO_GATE_FAIL_ROW else int(round(frac * len(cells)))
    n_excl = min(n_excl, len(cells) - 1)
    causes = [e["cause"] for e in led.excluded] or [F.CAUSE_BANK_TOO_SMALL]
    excluded = []
    for i, cell in enumerate(cells[len(cells) - n_excl :]):
        n_test = len(
            {
                r.prompt_id
                for r in rd.rows
                if r.split == "test" and f"{r.source_frame}|{r.stratum}" == cell
            }
        )
        excluded.append({"cell": cell, "cause": causes[i % len(causes)], "n_test_eligible": n_test})
    return INF.synthetic_row_ledger(row, cells, excluded)


def build_demo_inputs(
    out_root: Path, *, seed: int = 0, n_prompts: int = 120
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    prov = U.load_committed_provenance()
    partition = U.c2c3_partition(prov)
    committed_sizes = prov["c2_c3_partition"]["holm_family_sizes"]
    committed_ledgers = load_cell_ledgers()
    eligible = set(partition["eligible"])
    reg = INF.InferenceRegistry(**DEMO_REGISTRY_KWARGS)
    print(f"[u12-demo] registry overrides RECORDED: {DEMO_REGISTRY_KWARGS}", flush=True)

    rows_input: dict[str, Any] = {}
    ladder: dict[tuple[str, str], dict[str, Any]] = {}
    for i, row in enumerate(C.ROW_IDS):
        small = row == _DEMO_GATE_FAIL_ROW
        rd = U.synthesize_row_data(
            row=row,
            n_prompts=20 if small else n_prompts,
            n_responses=6,
            d=16,
            n_superfamilies=8 if small else 24,
            effect=_DEMO_EFFECTS[i % len(_DEMO_EFFECTS)],
            seed=seed + i,
        )
        scratch = out_root / "comparators_demo" / row
        scratch.mkdir(parents=True, exist_ok=True)
        ledger = U.CompLedger(scratch / "ledger.jsonl")
        counter = U._UnitCounter()
        counter.cap = U.units_for(row, U.COMPARATORS, partition)
        U.run_ladder(
            rd,
            U.COMPARATORS,
            ledger=ledger,
            counter=counter,
            scores_dir=scratch,
            embed_backend=U.HashEmbedBackend(),
            partition=partition,
            direction_provider=lambda _row, _rd=rd: _rd.synthetic_direction,
            seed=seed + i,
            allow_underdetermined=None,
        )
        records = U.load_comparator_results(scratch)
        ladder.update(records)
        comps = ["c5_full_probe"] + (["c2_direction_dot"] if row in eligible else [])
        panel = INF.build_panel(
            row,
            records,
            comps,
            _demo_ledger(row, rd, committed_ledgers),
            INF.prompt_cells_from_rowdata(rd),
        )
        c5_rec = records[(row, "c5_full_probe")]
        rows_input[row] = INF.RowInputs(
            row=row,
            panel=panel,
            rowdata=rd,
            selected_c=float(c5_rec["selected_c"]),
            scores_sha={c: records[(row, c)]["scores_sha256"] for c in comps},
            # Synthetic rows have no label pipeline, so there are no non-scored
            # final-label records to exclude. Explicit empty per-split dicts —
            # the complete-labels gate REQUIRES the field, and a demo that
            # omitted it would be asserting completeness it never measured.
            label_exclusions={"dev": {}, "test": {}},
        )
        print(f"[u12-demo] row={row} ladder+panel done", flush=True)

    # Synthetic PASS reliability verdict. The production path REQUIRES the frozen
    # test-bank blinded-audit artifact and returns not-estimable without it
    # (INF.load_test_label_reliability); synthesized demo rows have no such bank,
    # so the demo supplies an explicitly SYNTHETIC verdict rather than reading a
    # real one. The artifact string says so on its face, so a demo report can
    # never be mistaken for a gated production report.
    demo_reliability = {
        "artifact": "synthetic-demo verdict — NOT a real test-bank audit",
        "bank": "test",
        "status": PW.GATE_PASS,
        "per_trait": {
            row: {"status": PW.GATE_PASS, "detail": "synthetic demo row"} for row in rows_input
        },
    }
    report = INF.run_inference(
        rows_input,
        partition,
        reg,
        out_root,
        reliability=demo_reliability,
        family_sizes_expected=dict(committed_sizes),
    )
    points = build_points(rows_input, report)
    write_json_atomic(out_root / "figure_points.json", points)
    return report, points, ladder


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="issue2658_figures",
        description="Unit 12: plan section 7 figures off unit 11's inference report.",
    )
    ap.add_argument("--import-check", action="store_true", help="resolve imports and exit")
    ap.add_argument("--phase", choices=("demo", "render"), default=None)
    ap.add_argument("--out-root", type=Path, default=None, help="demo scratch root")
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-prompts", type=int, default=120, help="demo prompts per row")
    ap.add_argument("--report", type=Path, help="render: unit 11 inference_report.json")
    ap.add_argument("--points", type=Path, help="render: figure_points.json")
    ap.add_argument("--comparators-dir", type=Path, help="render: unit 9 ledger dir")
    ap.add_argument(
        "--skip-ladder",
        action="store_true",
        help="render: explicitly skip the C0-C5 ladder figures",
    )
    ap.add_argument("--manifest", type=Path, default=None, help="frame manifest override")
    ap.add_argument(
        "--production",
        action="store_true",
        help="drop the synthetic-smoke tag (sealed-test renders only)",
    )
    return ap


def run(args: argparse.Namespace) -> int:
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[u12] import-check ok", flush=True)
        return 0
    if args.phase is None:
        raise FigureInputError("--phase is required (demo | render)")
    ledgers = load_cell_ledgers(args.manifest)
    if args.phase == "demo":
        out_root = args.out_root
        if out_root is None:
            out_root = Path(tempfile.mkdtemp(prefix="issue-2658-u12-demo-"))
            print(f"[u12] scratch out-root: {out_root}", flush=True)
        report, points, ladder = build_demo_inputs(
            out_root, seed=args.seed, n_prompts=args.n_prompts
        )
        saved = render_all(report, points, ladder, ledgers, args.fig_dir, synthetic=True)
    else:
        if args.report is None or args.points is None:
            raise FigureInputError("--report and --points are required for --phase render")
        if args.comparators_dir is None and not args.skip_ladder:
            raise FigureInputError(
                "--comparators-dir is required for the C0-C5 ladder figures "
                "(pass --skip-ladder to omit them explicitly)"
            )
        report = load_report(args.report)
        points = load_points(args.points)
        ladder = (
            None
            if args.comparators_dir is None
            else U.load_comparator_results(args.comparators_dir)
        )
        saved = render_all(
            report, points, ladder, ledgers, args.fig_dir, synthetic=not args.production
        )
    for stem, paths in sorted(saved.items()):
        print(f"[u12] {stem} -> {paths['png']}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_argparser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

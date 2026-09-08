"""Figures for the two comment-response analyses in Section 4.2.

Figure 1 (comment 2), two panels:
  A  the two shift sizes per minimal-pair element: how far the context summary
     moves and how far the answer moves.
  B  variance of the observed answer shift explained by the map, against the
     copy baseline's direction agreement, with the shipped magnitude ratio
     shown for comparison.

Figure 2 (comment 3), two panels:
  A  how far apart the two answers of a pair sit along the refusal direction,
     by stratum -- the reviewer's premise.
  B  two-alternative hit rate by stratum, map vs copy, so the saturated flip
     pairs can be read against the near-boundary ones.

Style is the canonical c2a system (docs/paper_context_answer_map/plotting_style.md);
no interpretive text is rendered onto the canvas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/scipy imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.analysis import c2a_plot_style as c2a  # noqa: E402
from explore_persona_space.task_workflow import repo_root  # noqa: E402

MAP_C = "#1B6CA8"
COPY_C = "#B4654A"
CTX_C = "#687078"


def _fig1(data: dict, stem: Path) -> dict:
    labels = list(data["elements"].keys())
    _SHORT = {
        "Output format": "Format",
        "Persona": "Persona",
        "Tone": "Tone",
        "Answer language": "Language",
        "Question topic": "Topic",
        "One-word topic": "One word",
    }
    short = [_SHORT.get(lab, lab) for lab in labels]
    el = data["elements"]
    ctx = [el[k]["norm_ctx_shift"]["median"] for k in labels]
    ans = [el[k]["norm_answer_shift"]["median"] for k in labels]
    ve = [el[k]["map"]["ve"] for k in labels]
    ve_lo = [el[k]["map"]["ve_ci95"][0] for k in labels]
    ve_hi = [el[k]["map"]["ve_ci95"][1] for k in labels]
    slope = [el[k]["map"]["slope"] for k in labels]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(c2a.canvas_width_in(1.0), c2a.canvas_width_in(1.0) * 0.38)
    )
    x = np.arange(len(labels))
    w = 0.38
    ax_a.bar(x - w / 2, ctx, w, label="context shift", color=CTX_C)
    ax_a.bar(x + w / 2, ans, w, label="answer shift", color=MAP_C)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short, fontsize=7)
    ax_a.set_ylabel("Median shift norm")
    ax_a.legend(frameon=False, loc="upper right")
    c2a.style_axis(ax_a)

    ax_b.axhline(0.0, color=c2a.SEAM, lw=0.8, zorder=1)
    ax_b.errorbar(
        x,
        ve,
        yerr=[np.array(ve) - np.array(ve_lo), np.array(ve_hi) - np.array(ve)],
        fmt="o",
        color=MAP_C,
        capsize=2.5,
        label="variance explained by map",
        zorder=3,
    )
    ax_b.plot(x, slope, "s", mfc="none", color=COPY_C, label="magnitude ratio (shipped)", zorder=3)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(short, fontsize=7)
    ax_b.set_ylabel("Variance explained / ratio")
    ax_b.legend(frameon=False, loc="lower left")
    c2a.style_axis(ax_b)

    fig.tight_layout()
    return c2a.save_c2a_figure(
        fig,
        stem,
        title="Minimal-pair shift sizes and variance explained",
        subject=(
            "Per-element median context-side and answer-side shift norms, and the pooled "
            "variance of the observed answer shift explained by the frozen layer-19 map "
            "with 95% pair-bootstrap intervals, against the shipped magnitude ratio"
        ),
        creator="scripts/paper_c2a_comment_figures.py",
    )


def _fig2(data: dict, stem: Path) -> dict:
    order = ["saturated", "graded", "none"]
    present = [s for s in order if s in data["premise_answer_separation"]]
    sep = data["separation_by_stratum"]
    prem = data["premise_answer_separation"]
    graded = data["graded_read"]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(c2a.canvas_width_in(1.0), c2a.canvas_width_in(1.0) * 0.40)
    )
    x = np.arange(len(present))
    med = [prem[s]["abs_refusal_axis_separation"]["median"] for s in present]
    lo = [prem[s]["abs_refusal_axis_separation"]["iqr"][0] for s in present]
    hi = [prem[s]["abs_refusal_axis_separation"]["iqr"][1] for s in present]
    ax_a.bar(x, med, 0.55, color=CTX_C)
    ax_a.errorbar(
        x,
        med,
        yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
        fmt="none",
        ecolor=c2a.INK,
        capsize=2.5,
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{s}\nn={prem[s]['n_pairs']}" for s in present])
    ax_a.set_ylabel("Refusal-axis separation", fontsize=7)
    c2a.style_axis(ax_a)

    w = 0.38
    for off, arm, color in ((-w / 2, "map", MAP_C), (w / 2, "copy", COPY_C)):
        acc = [sep[s][arm]["acc"] for s in present]
        clo = [sep[s][arm]["ci95"][0] for s in present]
        chi = [sep[s][arm]["ci95"][1] for s in present]
        ax_b.errorbar(
            x + off,
            acc,
            yerr=[np.array(acc) - np.array(clo), np.array(chi) - np.array(acc)],
            fmt="o",
            color=color,
            capsize=2.5,
            label=arm,
        )
    ax_b.axhline(0.5, color=c2a.SEAM, lw=0.8, ls="--")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{s}\nn={prem[s]['n_pairs']}" for s in present])
    ax_b.set_ylabel("Two-alternative hit rate", fontsize=7)
    ax_b.set_ylim(0.3, 1.05)
    ax_b.legend(frameon=False, loc="lower left")
    c2a.style_axis(ax_b)

    # Panel C: the graded read against the behavioural DV, where the binary
    # hit rate in panel B is at ceiling and therefore uninformative.
    for off, arm, color in ((-w / 2, "map", MAP_C), (w / 2, "copy", COPY_C)):
        key = f"rho_{arm}"
        rho = [graded[s]["vs_refusal_margin_delta"].get(key, np.nan) for s in present]
        ax_c.plot(x + off, rho, "o", color=color, label=arm)
    ax_c.axhline(0.0, color=c2a.SEAM, lw=0.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"{s}\nn={prem[s]['n_pairs']}" for s in present])
    ax_c.set_ylabel("Rank corr. with refusal margin", fontsize=7)
    ax_c.set_ylim(-0.05, 1.0)
    ax_c.legend(frameon=False, loc="lower left")
    c2a.style_axis(ax_c)

    fig.tight_layout()
    return c2a.save_c2a_figure(
        fig,
        stem,
        title="Refusal separation near and far from the decision boundary",
        subject=(
            "Median absolute separation of a pair two answers along the refusal direction "
            "with interquartile range, and the two-alternative hit rate for the map and the "
            "copy baseline, split by whether the pair's behaviour change is saturated, graded, "
            "or absent"
        ),
        creator="scripts/paper_c2a_comment_figures.py",
    )


def _sidecar(stem: Path, outputs: dict, sources: list[str], title: str, subject: str) -> None:
    """Write the provenance sidecar the c2a plotting system requires."""
    import hashlib

    root = repo_root()
    meta = {
        "status": "Section 4.2 comment-response figure",
        "style_version": c2a.STYLE_VERSION,
        "plotting_script": "scripts/paper_c2a_comment_figures.py",
        "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
        "reproduction_command": "uv run python scripts/paper_c2a_comment_figures.py",
        "title": title,
        "subject": subject,
        "sources": [
            {
                "path": src,
                "sha256": hashlib.sha256((root / src).read_bytes()).hexdigest(),
            }
            for src in sources
        ],
        "render": outputs["record"],
    }
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def _matched_pairs(root: Path) -> dict:
    """Per-element (magnitude, cosine) pairs, using the same masks as the analysis."""
    from importlib import import_module

    mm = import_module("paper_c2a_matched_magnitude")
    sources = {
        "parent": mm._rows(root / "eval_results/issue_2564/perpair.jsonl"),
        "ffr": mm._rows(
            root / "eval_results/issue_2564/floor-failed-reelicitation/perpair_ffr.jsonl"
        ),
        "pilot": mm._rows(root / "eval_results/issue_2564/lang_oneword_pilot/perpair.jsonl"),
    }
    out: dict = {}
    for label, source_key, axis in mm.ELEMENTS:
        rows = sources[source_key]
        if axis in mm.PRIMARY_CLASS:
            sel = [
                r
                for r in rows
                if r["axis"] == axis
                and r["pair_class"] == mm.PRIMARY_CLASS[axis]
                and r["in_headline_70"]
            ]
        else:
            sel = [r for r in rows if r["axis"] == axis]
        out[label] = {
            "mag": [float(r["norm_obs_tail_L19"]) for r in sel],
            "cos": [mm._get(r, "cos", mm.MAP_ARM) for r in sel],
        }
    return out


ELEM_COLORS = {
    "Output format": "#B4654A",
    "Persona": "#1B6CA8",
    "Tone": "#4C9F70",
    "Answer language": "#8A6BBE",
    "Question topic": "#C99700",
    "One-word topic": "#687078",
}


def _fig3(data: dict, pairs: dict, stem: Path) -> dict:
    """A: cosine vs answer-shift magnitude with the pooled trend. B: matched-magnitude residuals."""
    names = list(data["reads"]["map"]["per_element"].keys())
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(c2a.canvas_width_in(1.0), c2a.canvas_width_in(1.0) * 0.38)
    )

    for name in names:
        x = np.array(pairs[name]["mag"])
        y = np.array(pairs[name]["cos"])
        ax_a.scatter(x, y, s=7, alpha=0.65, color=ELEM_COLORS[name], label=name, linewidths=0)
    line = data["reads"]["map"]["insample_line"]
    grid = np.linspace(
        min(min(pairs[n]["mag"]) for n in names), max(max(pairs[n]["mag"]) for n in names), 200
    )
    ax_a.plot(grid, line["intercept"] + line["slope"] * np.log(grid), color=c2a.INK, lw=1.2)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Observed answer-shift norm", fontsize=8)
    ax_a.set_ylabel("Cosine, predicted vs observed", fontsize=8)
    ax_a.legend(frameon=False, fontsize=6, loc="lower right", ncol=2)
    c2a.style_axis(ax_a)

    x = np.arange(len(names))
    w = 0.38
    for off, arm, color in ((-w / 2, "map", MAP_C), (w / 2, "copy", COPY_C)):
        per = data["reads"][arm]["per_element"]
        val = [per[n]["residual_loeo"]["mean"] for n in names]
        lo = [per[n]["residual_loeo"]["ci95"][0] for n in names]
        hi = [per[n]["residual_loeo"]["ci95"][1] for n in names]
        ax_b.errorbar(
            x + off,
            val,
            yerr=[np.array(val) - np.array(lo), np.array(hi) - np.array(val)],
            fmt="o",
            color=color,
            capsize=2.5,
            label=arm,
        )
    ax_b.axhline(0.0, color=c2a.SEAM, lw=0.8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=6)
    ax_b.set_ylabel("Residual cosine at matched magnitude", fontsize=7)
    ax_b.legend(frameon=False, loc="lower left", fontsize=7)
    c2a.style_axis(ax_b)

    fig.tight_layout()
    return c2a.save_c2a_figure(
        fig,
        stem,
        title="Direction agreement against answer-shift magnitude",
        subject=(
            "Per-pair cosine between the predicted and observed answer shift against the "
            "observed shift norm on a log axis with the pooled fitted trend, and each "
            "element mean residual from a leave-one-element-out trend with 95% pair-bootstrap "
            "intervals, for the map and the copy baseline"
        ),
        creator="scripts/paper_c2a_comment_figures.py",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only", choices=("comment2", "comment3", "matched"), default=None
    )
    args = ap.parse_args()
    root = repo_root()
    c2a.set_c2a_style()
    out = root / "figures/paper"

    if args.only in (None, "comment2"):
        data = json.loads(
            (root / "eval_results/issue_2564/comment2_variance_explained/summary.json").read_text()
        )
        stem = out / "c3_pair_variance_explained"
        res = _fig1(data, stem)
        _sidecar(
            stem,
            res,
            ["eval_results/issue_2564/comment2_variance_explained/summary.json"],
            "Minimal-pair shift sizes and variance explained",
            "Median context-side and answer-side shift norms per element, and the pooled "
            "variance of the observed answer shift explained by the frozen layer-19 map "
            "with 95% pair-bootstrap intervals, against the shipped magnitude ratio",
        )
        print("comment 2 figure:", res["png"])
    if args.only in (None, "matched"):
        data = json.loads(
            (root / "eval_results/issue_2564/comment2_matched_magnitude/summary.json").read_text()
        )
        pairs = _matched_pairs(root)
        stem = out / "c3_matched_magnitude"
        res = _fig3(data, pairs, stem)
        _sidecar(
            stem,
            res,
            ["eval_results/issue_2564/comment2_matched_magnitude/summary.json"],
            "Direction agreement against answer-shift magnitude",
            "Per-pair cosine against observed answer-shift norm with the pooled trend, and "
            "per-element residuals from a leave-one-element-out trend, map and copy baseline",
        )
        print("matched-magnitude figure:", res["png"])

    if args.only in (None, "comment3"):
        data = json.loads(
            (root / "eval_results/issue_2617/comment3_refusal_boundary/summary.json").read_text()
        )
        stem = out / "c3_refusal_boundary"
        res = _fig2(data, stem)
        _sidecar(
            stem,
            res,
            ["eval_results/issue_2617/comment3_refusal_boundary/summary.json"],
            "Refusal separation near and far from the decision boundary",
            "Median absolute separation of a pair two answers along the refusal direction, "
            "the two-alternative hit rate, and the rank correlation with the continuous "
            "refusal margin, for the map and the copy baseline by behaviour-change stratum",
        )
        print("comment 3 figure:", res["png"])


if __name__ == "__main__":
    main()

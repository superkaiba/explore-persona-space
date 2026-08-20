"""Smallest-sufficient-group ladder figure for the tier-1.5 intercept-refit round.

Reads the tier15_intercept_refit results (naive / +offset / +diag rungs, this
round) plus the committed GL-reparameterization anchors (#825 map_alignment
composition, #1345 operator_comparison delta_reparam_l19, #1310/#1639 xpersona
reparam ordered_pairs) and renders one fraction-of-ceiling ladder per family:

  naive -> +offset (tier15) -> +diag scale (tier15d) [-> rotation -> rot+scale,
  family A only, committed] -> GL (fitted alignments, committed) -> ceiling

Fractions use the fold-mean held-out R^2 convention; the context arm only
(the #1345 prefix arm is the parent-declared degenerate companion and stays in
the results JSON). Values below the y-floor are drawn clipped at the floor as
open markers; true values live in the results JSONs.

Output: figures/issue_1639/tier15_ladder.{png,pdf} + .meta.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis import paper_plots as pp  # noqa: E402

EV = _REPO_ROOT / "eval_results"
OUT_DIR = _REPO_ROOT / "figures" / "issue_1639"
Y_FLOOR = -1.05
RUNGS = ["naive", "+offset", "+diag\nscale", "GL fitted\nA·M·B", "ceiling"]
RUNGS_A = [
    "naive",
    "+offset",
    "+diag\nscale",
    "rotation",
    "rot +\nscale",
    "GL fitted\nA·M·B",
    "ceiling",
]


def _frac(vals: list[float | None], ceil: float) -> list[float | None]:
    return [None if v is None else v / ceil for v in vals]


def _plot_line(ax, xs, fracs, *, color, ls, lw, alpha, label=None):
    fr = np.array([np.nan if f is None else f for f in fracs], dtype=float)
    clipped = fr < Y_FLOOR
    fr_draw = np.clip(fr, Y_FLOOR, None)
    ax.plot(xs, fr_draw, color=color, ls=ls, lw=lw, alpha=alpha, label=label, zorder=2)
    ax.plot(
        np.asarray(xs)[~clipped & np.isfinite(fr)],
        fr_draw[~clipped & np.isfinite(fr)],
        "o",
        color=color,
        ms=3.5,
        alpha=alpha,
        zorder=3,
    )
    if clipped.any():
        ax.plot(
            np.asarray(xs)[clipped],
            fr_draw[clipped],
            "o",
            mfc="white",
            mec=color,
            ms=4.5,
            alpha=alpha,
            zorder=3,
        )


def family_a_lines() -> list[dict]:
    res = json.loads((EV / "issue_825/tier15_intercept_refit/results.json").read_text())
    comp = json.loads((EV / "issue_825/map_alignment/results.json").read_text())
    c19 = comp["per_layer"]["19"]["composition"]
    lines = []
    for d, label in (("b2i", "base map → instruct"), ("i2b", "instruct map → base")):
        r = res["directions"][d]
        ceil = r["within"]["r2_foldmean"]
        vals = [
            r["naive"]["r2_foldmean"],
            r["tier15"]["r2_foldmean"],
            r["tier15d"]["r2_foldmean"],
            c19["orthogonal"][f"comp_samefn_{d}"],
            c19["scaled_orthogonal"][f"comp_samefn_{d}"],
            c19["linear"][f"comp_samefn_{d}"],
            ceil,
        ]
        lines.append({"label": label, "fracs": _frac(vals, ceil), "direction": d})
    return lines


def family_b_lines() -> list[dict]:
    res = json.loads((EV / "issue_1345/tier15_intercept_refit/results.json").read_text())
    lines = []
    for model in ("instruct", "pretrained"):
        opc = json.loads(
            (
                EV / f"issue_1345/operator_comparison_{model if model != 'pretrained' else 'base'}"
                "_context.json"
            ).read_text()
        )
        rep = opc["delta_reparam_l19"]
        # direction_key: b2i = r2 operator recovered in r1; i2b = r1 operator in r2
        gl = {"r2->r1": rep["recovered_r2"]["b2i"], "r1->r2": rep["recovered_r2"]["i2b"]}
        for src, tgt in (("r1", "r2"), ("r2", "r1")):
            key = f"{model}.context.{src}->{tgt}"
            r = res["directions"][key]
            ceil = r["within"]["r2_foldmean"]
            vals = [
                r["naive"]["r2_foldmean"],
                r["tier15"]["r2_foldmean"],
                r["tier15d"]["r2_foldmean"],
                gl[f"{src}->{tgt}"],
                ceil,
            ]
            name = {"r1": "chat", "r2": "plain"}
            lines.append(
                {
                    "label": f"{model}: {name[src]}→{name[tgt]}",
                    "fracs": _frac(vals, ceil),
                    "model": model,
                }
            )
    return lines


def family_c_lines(model: str) -> list[dict]:
    res = json.loads(
        (EV / "issue_1310/xpersona_similarity/tier15_intercept_refit/results.json").read_text()
    )
    rep = json.loads((EV / f"issue_1310/xpersona_similarity/reparam_{model}.json").read_text())[
        "ordered_pairs"
    ]
    lines = []
    for key, r in res["directions"].items():
        m, pair = key.split(".", 1)
        if m != model:
            continue
        ceil = r["within"]["r2_foldmean"]
        gl = rep[pair]["recovery_r2_foldmean"] if pair in rep else None
        vals = [
            r["naive"]["r2_foldmean"],
            r["tier15"]["r2_foldmean"],
            r["tier15d"]["r2_foldmean"],
            gl,
            ceil,
        ]
        lines.append({"label": pair, "fracs": _frac(vals, ceil)})
    return lines


def main_iclr(out_dir: Path) -> None:
    """ICLR paper variant: 2x2 ladder grid at final size, PAPER_COLORS semantics.

    Differences from the blog hero (deliberate spec, 2026-08-20): panels
    lettered A-D (the old figure labeled both bottom panels "C"); issue
    numbers stripped from panel titles (caption carries provenance); ONE
    color = one meaning paper-wide — blue = instruct, orange = base — so in
    panel A each line is colored by the model whose MAP is being moved
    (this flips the old #825 direction-keyed assignment on purpose).
    """
    pp.set_paper_style("iclr")
    fig, axes = plt.subplots(2, 2, figsize=pp.figsize_iclr_panels(2, height_in=4.6), sharey=True)

    ax = axes[0, 0]
    xs_a = np.arange(len(RUNGS_A))
    for ln in family_a_lines():
        # color = the model whose map is transferred (b2i moves the BASE map)
        color = pp.paper_color("base") if ln["direction"] == "b2i" else pp.paper_color("instruct")
        _plot_line(
            ax, xs_a, ln["fracs"], color=color, ls="-", lw=1.4, alpha=0.95, label=ln["label"]
        )
    ax.set_xticks(xs_a, RUNGS_A)
    ax.set_title(r"A. base $\leftrightarrow$ instruct")
    ax.legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.4)

    ax = axes[0, 1]
    xs = np.arange(len(RUNGS))
    for ln in family_b_lines():
        ls = "-" if "chat→plain" in ln["label"] else "--"
        _plot_line(
            ax,
            xs,
            ln["fracs"],
            color=pp.paper_color("instruct" if ln["model"] == "instruct" else "base"),
            ls=ls,
            lw=1.4,
            alpha=0.95,
            label=ln["label"].replace("pretrained", "base"),
        )
    ax.set_xticks(xs, RUNGS)
    ax.set_title(r"B. chat $\leftrightarrow$ plain text")
    ax.legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.4)

    for j, (model, letter) in enumerate((("instruct", "C"), ("base", "D"))):
        ax = axes[1, j]
        lines = family_c_lines(model)
        arr = np.array(
            [[np.nan if f is None else f for f in ln["fracs"]] for ln in lines], dtype=float
        )
        color = pp.paper_color(model)
        for ln in lines:
            _plot_line(ax, xs, ln["fracs"], color=color, ls="-", lw=0.6, alpha=0.3)
        mean_fr = np.nanmean(np.clip(arr, Y_FLOOR, None), axis=0)
        ax.plot(xs, mean_fr, color=color, lw=2.0, zorder=4, label="mean of 12 pairs")
        ax.set_xticks(xs, RUNGS)
        ax.set_title(f"{letter}. character $\\leftrightarrow$ character ({model})")
        ax.legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.4)

    for ax in axes.ravel():
        ax.axhline(1.0, color="0.6", lw=0.6, ls=":", zorder=1)
        ax.axhline(0.0, color="0.6", lw=0.6, ls=":", zorder=1)
        ax.set_ylim(Y_FLOOR - 0.08, 1.18)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of target's own ceiling")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    pp.savefig_paper(fig, "c4_transfer_ladder", dir=out_dir)
    plt.close(fig)
    print(f"WROTE {out_dir / 'c4_transfer_ladder.png'}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=("blog", "iclr"), default="blog")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "figures" / "paper")
    args = ap.parse_args()
    if args.style == "iclr":
        main_iclr(args.out_dir)
        return
    pp.set_paper_style()
    pal = {"instruct": "#0072B2", "base": "#E69F00", "pretrained": "#E69F00"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharey=True)

    ax = axes[0, 0]
    xs = np.arange(len(RUNGS_A))
    for ln in family_a_lines():
        color = pal["instruct"] if ln["direction"] == "b2i" else pal["base"]
        _plot_line(ax, xs, ln["fracs"], color=color, ls="-", lw=1.8, alpha=0.95, label=ln["label"])
    ax.set_xticks(xs, RUNGS_A)
    ax.set_title("A — base ↔ instruct (#825, context)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[0, 1]
    xs = np.arange(len(RUNGS))
    for ln in family_b_lines():
        ls = "-" if "chat→plain" in ln["label"] else "--"
        _plot_line(
            ax,
            xs,
            ln["fracs"],
            color=pal[ln["model"]],
            ls=ls,
            lw=1.8,
            alpha=0.95,
            label=ln["label"].replace("pretrained", "base"),
        )
    ax.set_xticks(xs, RUNGS)
    ax.set_title("B — chat ↔ plain text (#1345, context)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    for j, model in enumerate(("instruct", "base")):
        ax = axes[1, j]
        lines = family_c_lines(model)
        arr = np.array(
            [[np.nan if f is None else f for f in ln["fracs"]] for ln in lines], dtype=float
        )
        for ln in lines:
            _plot_line(ax, xs, ln["fracs"], color=pal[model], ls="-", lw=0.8, alpha=0.35)
        mean_fr = np.nanmean(np.clip(arr, Y_FLOOR, None), axis=0)
        ax.plot(xs, mean_fr, color=pal[model], lw=2.6, zorder=4, label="mean of 12 pairs")
        ax.set_xticks(xs, RUNGS)
        ax.set_title(f"C — character 4×4, {model} (#1310/#1639)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")

    for ax in axes.ravel():
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)
        ax.axhline(0.0, color="0.6", lw=0.8, ls=":", zorder=1)
        ax.set_ylim(Y_FLOOR - 0.08, 1.18)
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of target's own held-out ceiling")
    fig.suptitle(
        "How much transformation does each transfer need? "
        "(held-out R² at layer 19, fraction of target ceiling)",
        y=0.995,
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "tier15_ladder.png"
    fig.savefig(png, dpi=200)
    fig.savefig(OUT_DIR / "tier15_ladder.pdf")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    (OUT_DIR / "tier15_ladder.meta.json").write_text(
        json.dumps(
            {
                "script": "scripts/issue1639_tier15_ladder_fig.py",
                "git_commit": commit,
                "inputs": [
                    "eval_results/issue_825/tier15_intercept_refit/results.json",
                    "eval_results/issue_825/map_alignment/results.json",
                    "eval_results/issue_1345/tier15_intercept_refit/results.json",
                    "eval_results/issue_1345/operator_comparison_{instruct,base}_context.json",
                    "eval_results/issue_1310/xpersona_similarity/tier15_intercept_refit/results.json",
                    "eval_results/issue_1310/xpersona_similarity/reparam_{base,instruct}.json",
                ],
                "note": (
                    "Open markers at the bottom edge are clipped below y=-1.05; true values in "
                    "the results JSONs. Rotation rungs (family A only) come from the committed "
                    "#825 map_alignment composition variants (orthogonal / scaled-orthogonal "
                    "alignments), fit on paired activations like the GL rung."
                ),
            },
            indent=2,
        )
    )
    print(f"WROTE {png}")


if __name__ == "__main__":
    main()

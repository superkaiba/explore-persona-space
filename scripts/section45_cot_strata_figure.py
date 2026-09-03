#!/usr/bin/env python3
"""Render the needs-reasoning vs no-reasoning predictability figure.

One axes: held-out R^2 of the context-to-answer map fit within each of the four
Issue #2546 corpora that have their own fit cells (MATH, GSM8K train,
ContextHub, MMLU), for OpenThinker3-7B (arm 1, layer 19) and Qwen3-8B with
thinking on (arm 3, layer 24). Bars are colored by stratum, purple for the
needs-reasoning corpora and green for MMLU, the one no-reasoning corpus with a
whole-corpus fit cell. Writes ``figures/paper/c1_cot_strata.{pdf,png}``, a
grayscale audit, and a JSON sidecar that also records the per-corpus retrieval
lift quoted in the appendix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # repo convention: environment before heavy imports

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    MUTED,
    PAPER,
    STYLE_VERSION,
    save_c2a_figure,
    set_c2a_style,
    style_score_axis,
)

CELLS_DIR = ROOT / "eval_results" / "issue_2546" / "cells"  # whole-corpus production cells (this figure left the paper when per-corpus fits were dropped); --cells-dir overrides
CELLS_DIR_WHOLE_CORPUS = ROOT / "eval_results" / "issue_2546" / "cells"
DEFAULT_OUT = ROOT / "figures" / "paper"
DEFAULT_STEM = "c1_cot_strata"
HF_REVISION = "8368cc69f887d20931acd8c4d76c142275173728"
SOURCE_REF = "42308cc7522dcb0a2a76b332b0c24d981de4b585"

STRATUM_COLOR = {"does": "#7B3294", "doesnt": "#5AAE61"}
STRATUM_LABEL = {"does": "Needs-reasoning corpora", "doesnt": "No-reasoning corpora"}
CORPORA = [
    ("math", "MATH", "does"),
    ("gsm8k_train", "GSM8K\ntrain", "does"),
    ("contexthub", "Context-\nHub", "does"),
    ("mmlu", "MMLU", "doesnt"),
]
ALL_CORPORA = list(CORPORA)  # narrowed by --corpora at run time
ALL_MODELS = [(1, "OpenThinker3-7B", 19), (3, "Qwen3-8B, thinking on", 24)]
MODELS = list(ALL_MODELS)  # narrowed by --arms at run time
GROUP_GAP = 1.0
BAR_WIDTH = 0.72


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_cells() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm, model, layer in MODELS:
        rows = []
        for corpus, name, stratum in CORPORA:
            path = CELLS_DIR / f"p7_A__{corpus}__a{arm}.json"
            data = json.loads(path.read_text())
            if data["status"] != "ok":
                raise ValueError(f"{path.name}: status {data['status']!r}")
            if int(data["headline_layer"]) != layer:
                raise ValueError(f"{path.name}: unexpected headline layer")
            if "knn_content" in data:  # whole-corpus production cells: content-rule lift
                pool = data["knn_content"]["euclidean"]["corpus_pool"]
                retrieval = {"retrieval_lift": float(pool["lift"]), "retrieval_lift_ci": [float(pool["lift_ci_lo"]), float(pool["lift_ci_hi"])], "retrieval_acc": float(pool["acc_at_1"])}
            else:  # needs-only refits: own-answer acc@1 under the paper recipe
                ret = data["retrieval"]
                retrieval = {"retrieval_lift": float(ret["acc1_whitened_csls"]), "retrieval_lift_ci": [float(v) for v in ret["acc1_whitened_csls_ci"]], "retrieval_acc": float(ret["acc1_whitened_csls"])}
            rows.append(
                {
                    "corpus": corpus,
                    "name": name.replace("\n", " ").replace("- ", ""),
                    "stratum": stratum,
                    "n_rows": int(data["n_rows"]),
                    "r2": float(data["r2_headline"]),
                    "r2_ci": [float(data["r2_headline_bootstrap"]["ci_lo"]), float(data["r2_headline_bootstrap"]["ci_hi"])],
                    **retrieval,
                    "retrieval_chance": float(pool["chance_mean"]),
                    "source": str(path.relative_to(ROOT)),
                    "source_sha256": _sha256(path),
                }
            )
        out[str(arm)] = {"model": model, "layer": layer, "rows": rows}
    return out


def make_figure(cells: dict[str, Any]) -> plt.Figure:
    set_c2a_style()
    single = len(MODELS) == 1
    fig = plt.figure(figsize=(8.0 if single else 9.6, 5.8), constrained_layout=False)
    grid = fig.add_gridspec(1, 1, left=0.13 if single else 0.11, right=0.985, top=0.74, bottom=0.16 if single else 0.24)
    ax = fig.add_subplot(grid[0, 0])
    style_score_axis(ax, y_min=0.0, y_max=0.8, y_step=0.2)
    xs: list[float] = []
    labels: list[str] = []
    x0 = 0.0
    group_centers = []
    for arm, model, _layer in MODELS:
        block = cells[str(arm)]
        for i, (row, (_corpus, name, _stratum)) in enumerate(zip(block["rows"], CORPORA)):
            x = x0 + i
            color = STRATUM_COLOR[row["stratum"]]
            ax.bar(x, row["r2"], width=BAR_WIDTH, color=color, linewidth=0, zorder=3)
            lo, hi = row["r2_ci"]
            ax.errorbar(x, row["r2"], yerr=[[row["r2"] - lo], [hi - row["r2"]]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, capthick=1.2, zorder=4)
            xs.append(x)
            labels.append(name)
        group_centers.append((x0 + (len(CORPORA) - 1) / 2, model))
        x0 += len(CORPORA) + GROUP_GAP
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=13, linespacing=1.15)
    sep = len(CORPORA) - 0.5 + GROUP_GAP / 2
    if not single:
        ax.axvline(sep, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
    for center, model in ([] if single else group_centers):
        ax.text(center, -0.30, model.upper(), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12, fontweight=700, color=MUTED)
    ax.set_ylabel("Held-out $R^2$, context → answer  ↑", labelpad=12)
    ax.set_title("Within a corpus, reasoning demand barely changes predictability", loc="left", y=1.04, pad=0, fontweight=650, fontsize=17)
    kicker = "CONTEXT → ANSWER MAP FIT WITHIN EACH CORPUS, OPENTHINKER3-7B, LAYER 19" if single else "CONTEXT → ANSWER MAP FIT WITHIN EACH CORPUS, LAYER 19 (OPENTHINKER3-7B) AND 24 (QWEN3-8B)"
    ax.text(0.0, 1.20, kicker, transform=ax.transAxes, fontsize=11.5, fontweight=700, color=MUTED, va="bottom", ha="left")
    handles = [Patch(facecolor=STRATUM_COLOR[s], edgecolor=STRATUM_COLOR[s], label=STRATUM_LABEL[s]) for s in ("does", "doesnt")]
    x0 = 0.13 if single else 0.11
    fig.text(x0, 0.965, "CORPORA", color=MUTED, fontsize=11.5, fontweight=750, ha="left", va="center")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0 - 0.001, 0.948), ncol=2, frameon=False, columnspacing=1.3, handlelength=1.6, handletextpad=0.6, borderaxespad=0)
    return fig


def _git_state() -> dict[str, str | bool | None]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, check=False, capture_output=True, text=True)
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def main(argv: list[str] | None = None) -> int:
    global CELLS_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--arms", type=int, nargs="*", default=[1], help="arms to plot (default: OpenThinker3-7B only)")
    parser.add_argument("--corpora", nargs="*", default=["math", "gsm8k_train", "contexthub"], help="corpora to plot (default: the three needs-reasoning corpora)")
    parser.add_argument("--cells-dir", type=Path, default=CELLS_DIR, help="cell JSON dir (default: needs-reasoning-only refits; use eval_results/issue_2546/cells for the whole-corpus fits)")
    args = parser.parse_args(argv)
    MODELS[:] = [m for m in ALL_MODELS if m[0] in args.arms]
    CORPORA[:] = [c for c in ALL_CORPORA if c[0] in args.corpora]
    CELLS_DIR = args.cells_dir
    cells = load_cells()
    font = set_c2a_style()
    fig = make_figure(cells)
    stem = args.out_dir / args.stem
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Needs-reasoning vs no-reasoning predictability by corpus",
        subject="Issue #2546 per-corpus context-to-answer fits rendered for the paper",
        creator="scripts/section45_cot_strata_figure.py",
    )
    plt.close(fig)
    sidecar = stem.with_name(f"{args.stem}_data.json")
    sidecar.write_text(
        json.dumps(
            {
                "style_version": STYLE_VERSION,
                "font": font,
                "git": _git_state(),
                "provenance": {"task": 2546, "hf_revision": HF_REVISION, "source_ref": SOURCE_REF, "cells": "p7_A (cx_last -> ans_mean) whole-corpus fits, five random-row folds"},
                "models": cells,
                "outputs": {k: str(v.relative_to(ROOT)) for k, v in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for key, path in {**outputs, "data": sidecar}.items():
        print(f"{key}: {path}")
    for arm, block in cells.items():
        print(block["model"], {r["corpus"]: (round(r["r2"], 3), round(r["retrieval_lift"], 3)) for r in block["rows"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

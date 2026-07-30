"""Issue #1689 follow-up figure: Procrustes battery raw vs aligned operator cosine.

Reads eval_results/issue_1689/procrustes/battery_<model>_L19.json (from
scripts/issue1689_procrustes_battery.py) and renders one figure:

  fig9_procrustes_raw_vs_aligned.png — per valid unordered pair-arm, raw
  operator cosine (x) vs data-paired Procrustes-aligned cosine (y), base and
  instruct panels, colored/markered by pair class (same class style as
  fig5_side_localization), with the Haar rotation-null p97.5 band.

Usage: uv run python scripts/issue1689_procrustes_figure.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# CRITICAL: load_dotenv() BEFORE importing matplotlib / numpy — shared-VM
# thread caps (#847) freeze at first BLAS import.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "eval_results/issue_1689/procrustes"
FIGDIR = REPO / "figures/issue_1689"

MODELS = {
    "base": "battery_Qwen_Qwen2.5-7B_L19.json",
    "instruct": "battery_Qwen_Qwen2.5-7B-Instruct_L19.json",
}

# Same class -> (marker, palette index) mapping as fig5_side_localization
# (one color = one meaning across the write-up's figures).
CLS_STYLE = {
    "framing": ("o", 0),
    "identity": ("s", 1),
    "provenance": ("^", 2),
    "identity-vs-user": ("D", 3),
    "user-framing": ("v", 4),
    "crossed": ("P", 5),
}

SHORT = {
    "assistant": "asst",
    "user_lmsys": "LMSYS",
    "user_haiku": "haiku",
    "user_onpolicy": "on-pol",
    "helios": "HELIOS",
    "wren": "Wren",
    "dana": "Dana",
}


def _short(cell: str) -> str:
    for k, v in SHORT.items():
        if cell.startswith(k):
            suffix = cell[len(k) :].lstrip("_")
            suffix = {"naturalistic": "plain"}.get(suffix, suffix)
            return f"{v} {suffix}"
    return cell


def _load_valid(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    seen: dict[tuple[frozenset[str], str], dict] = {}
    for arms in d["pairs"].values():
        for arm, r in arms.items():
            key = (frozenset([r["src"], r["tgt"]]), arm)
            if key not in seen:
                seen[key] = r
    return [r for r in seen.values() if not r["screened"]]


def main() -> None:
    set_paper_style("blog")
    pal = paper_palette(6)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharex=True, sharey=True)
    max_p975 = 0.0
    for ax, (model, fname) in zip(axes, MODELS.items()):
        rows = _load_valid(PROC / fname)
        max_p975 = max(max_p975, max(abs(r["null_p975"]) for r in rows))
        for cls, (marker, ci) in CLS_STYLE.items():
            sel = [r for r in rows if r["cls"] == cls]
            if not sel:
                continue
            ax.scatter(
                [r["raw_cosine"] for r in sel],
                [r["aligned_cosine"] for r in sel],
                marker=marker,
                s=44,
                color=pal[ci],
                label=f"{cls} (n={len(sel)})",
                alpha=0.85,
                zorder=3,
            )
            for r in sel:
                if r["aligned_cosine"] >= 0.50:
                    ax.text(
                        r["raw_cosine"] + 0.008,
                        r["aligned_cosine"],
                        f"{_short(r['src'])} vs {_short(r['tgt'])}",
                        fontsize=7,
                        va="center",
                        zorder=4,
                    )
        ax.plot([0, 0.75], [0, 0.75], color="#999999", lw=0.8, ls="--", zorder=1)
        ax.axhspan(-max_p975, max_p975, color="#bbbbbb", alpha=0.6, zorder=1)
        ax.axhline(0, color="#cccccc", lw=0.7, zorder=1)
        ax.axvline(0, color="#cccccc", lw=0.7, zorder=1)
        ax.set_xlabel("raw operator cosine (shared coordinates)")
        ax.set_title(model, fontsize=11)
        ax.set_xlim(-0.03, 0.75)
        ax.set_ylim(-0.03, 0.78)
    axes[0].set_ylabel("data-paired Procrustes-aligned operator cosine")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Fitted context-to-answer operators per pair: raw vs aligned cosine (L19, both arms pooled)\n"
        "87 valid pair-arms per model; labels on aligned >= 0.50; gray band at y=0 = Haar rotation-null p97.5",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig_paper(fig, "fig9_procrustes_raw_vs_aligned", dir=FIGDIR)
    plt.close(fig)
    print(f"max |null_p975| across valid pair-arms: {max_p975:.6f}")


if __name__ == "__main__":
    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGABRT pointer. All outputs are
    # flushed/closed before this point; atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)

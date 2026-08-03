#!/usr/bin/env python
"""Issue #1345 char-capture-ladders — Result-2 hero figure + exploratory dump (plan v13 §6).

Consumes the per-pair / per-cell fit JSONs written by
``issue1345_story_char_ladder_fill.py`` (``ladder_*.json`` / ``cell_*.json``
under --fits-dir; smoke ``_rowsN`` files are EXCLUDED) plus the banked
AI-likeness judge reports (``eval_results/issue_1345/judge_legs/
char_<c>_op_base/judge_report_ail_*.json`` — no re-judging), and writes:

  figures/.../char_transfer_tiers.png       hero: held-out R^2 per rung 1-9 +
      target ceiling, source -> character, one line per character labeled with
      banked AI-likeness; panels = inserted / own-answer (instruct) + base chat
  figures/.../char_transfer_tiers_reverse.png   reverse-direction companion
  figures/.../char_ceilings.png             within-cell ceilings, all cells x
      both arms, vs the assistant-story band
  figures/.../ailikeness_vs_transfer.png    AI-likeness vs ceiling-normalized
      rung-4 transfer (4 labeled points per series)
  eval_results/.../char_ladders_summary.json  the §3 registered lattice per
      pair (reconciliation rung at margin 0.05, H1 tier, H3 low-ceiling flag),
      n_matched / identity+bias / kNN tables, missing-pair list

Missing pairs/cells are SKIPPED and listed (never plotted as zero bars).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

import issue1345_story_char_ladder_fill as fill  # noqa: E402
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

RUNGS = list(fill.RUNGS)
MARGIN = 0.05  # plan §3 reconciliation margin (parent convention)
LOW_CEILING = 0.10  # plan §3 H3 measurability floor
CHAR_LABEL = {"helios": "HELIOS", "wren": "Wren", "dana": "Dana", "vex": "Vex"}

# Hero panels (plan §6): forward direction source -> character cell.
PANELS = (
    ("inserted (instruct): story_inserted -> char", "r4", ""),
    ("own answer (instruct): story_onpolicy -> char_op", "r4op", "_op"),
    ("base: chat -> char_base (solid) / char_op_base (dashed)", "r1", "_base"),
)


def _load_single(fits_dir: Path, pattern: str) -> dict | None:
    """The unique non-smoke fit JSON matching pattern (None when absent)."""
    hits = [p for p in sorted(fits_dir.glob(pattern)) if "_rows" not in p.name]
    if not hits:
        return None
    assert len(hits) == 1, f"ambiguous fit files for {pattern}: {[p.name for p in hits]}"
    return json.loads(hits[0].read_text())


def _judge_means(repo_root: Path) -> dict[str, float]:
    """Banked AI-likeness pooled means (own answers, base) per character."""
    out = {}
    for ch in fill.CHAR_CHARACTERS:
        p = (
            repo_root
            / f"eval_results/issue_1345/judge_legs/char_{ch}_op_base"
            / f"judge_report_ail_char_{ch}_op_base.json"
        )
        assert p.is_file(), f"banked AI-likeness report missing: {p}"
        out[ch] = float(json.loads(p.read_text())["means"]["pooled"]["mean"])
    return out


def reconciliation_rung(r2_by_rung: dict[str, float], ceiling: float) -> int | None:
    """Smallest rung whose held-out R^2 reaches ceiling - MARGIN (plan §3)."""
    for i, r in enumerate(RUNGS, start=1):
        if r2_by_rung[r] >= ceiling - MARGIN:
            return i
    return None


def h1_tier(rung: int | None) -> str:
    """Derived H1 label over the reconciliation rung (plan §3 lattice)."""
    if rung is not None and rung <= 6:
        return "H1-supported"
    if rung in (7, 8):
        return "intermediate"
    return "H1-falsified"


def _pair_key(src: str, tgt: str) -> str:
    return f"{fill.REGIME_LABEL[src]}->{fill.REGIME_LABEL[tgt]}"


def _extract_dir(entry: dict, basis: str, src: str, tgt: str, li: int = 0) -> dict | None:
    """One direction's rung/ceiling/null/baseline reads from a pair JSON."""
    blk = entry.get(basis, {}).get(_pair_key(src, tgt))
    if blk is None:
        return None
    fold = blk.get("fold_r2", {})

    def _std(vals: list[list[float]]) -> float:
        xs = [v[li] for v in vals]
        if len(xs) < 2:
            return 0.0
        mu = sum(xs) / len(xs)
        return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    return {
        "r2": {r: blk["r2"][r][li] for r in RUNGS},
        "ceiling": blk["ceiling_r2"][li],
        "null_r2": {r: blk["null_r2"][r][li] for r in RUNGS},
        "identity_bias_r2": blk["identity_bias_r2"][li],
        "knn_retrieval_fold0": blk.get("knn_retrieval_fold0", {}),
        "fold_r2_std": {r: _std(fold[r]) for r in fold} if fold else {},
    }


def main() -> None:
    """Render the hero + exploratory figures and the lattice summary JSON."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--fits-dir",
        type=Path,
        default=_REPO_ROOT / "eval_results/issue_1345/char_capture_ladders",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=_REPO_ROOT / "figures/issue_1345/char_capture_ladders"
    )
    ap.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="default: <fits-dir>/char_ladders_summary.json",
    )
    ap.add_argument("--basis", default="reduced")
    ap.add_argument("--arm", default="context")
    args = ap.parse_args()
    summary_out = args.summary_out or (args.fits_dir / "char_ladders_summary.json")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()
    colors = dict(zip(fill.CHAR_CHARACTERS, paper_palette(len(fill.CHAR_CHARACTERS))))
    ail = _judge_means(_REPO_ROOT)

    # ---- load the 16 pairs -------------------------------------------------
    pairs: dict[str, dict] = {}
    missing: list[str] = []
    for spec in fill.char_pair_specs():
        src, tgt, model = spec["src"], spec["tgt"], spec["model"]
        entry = _load_single(args.fits_dir, f"ladder_{src}__{tgt}__{model}_{args.arm}_*.json")
        if entry is None:
            missing.append(f"{src}:{tgt}")
            continue
        pairs[tgt] = {"spec": spec, "entry": entry}
    print(f"[plot] pairs loaded: {len(pairs)}/16; missing: {missing or 'none'}", flush=True)
    assert pairs, f"no ladder pair JSONs under {args.fits_dir} — nothing to plot"

    # ---- hero: forward transfer tiers (+ reverse companion) ----------------
    xs = list(range(1, len(RUNGS) + 1))
    for direction, stem in (("fwd", "char_transfer_tiers"), ("rev", "char_transfer_tiers_reverse")):
        fig, axes = plt.subplots(1, len(PANELS), figsize=(13.5, 4.2), sharey=True)
        for ax, (title, src, suffix) in zip(axes, PANELS):
            for ch in fill.CHAR_CHARACTERS:
                variants = [f"char_{ch}{suffix}"]
                if suffix == "_base":  # base panel: solid inserted + dashed op
                    variants = [f"char_{ch}_base", f"char_{ch}_op_base"]
                for v in variants:
                    if v not in pairs:
                        continue
                    s, t = (src, v) if direction == "fwd" else (v, src)
                    d = _extract_dir(pairs[v]["entry"], args.basis, s, t)
                    if d is None:
                        continue
                    ls = "--" if v.endswith("_op_base") else "-"
                    label = f"{CHAR_LABEL[ch]} (AI-likeness {ail[ch]:.1f})"
                    if v.endswith("_op_base"):
                        label += " [own answer]"
                    ax.plot(
                        xs,
                        [d["r2"][r] for r in RUNGS],
                        marker="o",
                        ms=3.5,
                        ls=ls,
                        color=colors[ch],
                        label=label,
                    )
                    ax.plot(
                        [xs[-1] + 1],
                        [d["ceiling"]],
                        marker="*",
                        ms=9,
                        color=colors[ch],
                        ls="none",
                    )
            ax.set_xticks([*xs, xs[-1] + 1])
            ax.set_xticklabels([r.split("_")[0] for r in RUNGS] + ["ceil"], fontsize=7)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("correction tier (rung)")
        axes[0].set_ylabel(
            "held-out R$^2$"
            + (" (source$\\to$char)" if direction == "fwd" else " (char$\\to$source)")
        )
        axes[0].legend(fontsize=6, loc="upper left")
        fig.tight_layout()
        savefig_paper(fig, stem, dir=args.out_dir)
        plt.close(fig)
        print(f"[plot] wrote {args.out_dir / (stem + '.png')}", flush=True)

    # ---- within-cell ceilings, all cells x both arms ------------------------
    cell_rows: list[tuple[str, str, float | None, float | None]] = []
    for v in fill.CHAR_VARIANTS:
        model = fill.REGIME_SPECS[v]["model"]
        row = [v, model]
        for arm in ("context", "prefix"):
            e = _load_single(args.fits_dir, f"cell_{v}__{model}_{arm}_*.json")
            row.append(None if e is None else e[args.basis]["ceiling_r2"][0])
        cell_rows.append(tuple(row))
    have_cells = [r for r in cell_rows if r[2] is not None or r[3] is not None]
    if have_cells:
        fig, ax = plt.subplots(figsize=(12.0, 4.0))
        idx = range(len(have_cells))
        ax.axhspan(0.26, 0.37, color="0.85", zorder=0, label="assistant-story band (0.26-0.37)")
        w = 0.38
        for off, armi, arm in ((-w / 2, 2, "context"), (w / 2, 3, "prefix")):
            ys = [r[armi] for r in have_cells]
            xs_b = [i + off for i in idx]
            ax.bar(
                [x for x, y in zip(xs_b, ys) if y is not None],
                [y for y in ys if y is not None],
                width=w,
                label=f"{arm} arm",
                color="C0" if arm == "context" else "C1",
                zorder=2,
            )
        ax.axhline(LOW_CEILING, color="crimson", lw=0.8, ls=":", label="H3 floor 0.10")
        ax.set_xticks(list(idx))
        ax.set_xticklabels(
            [r[0].removeprefix("char_") for r in have_cells], rotation=45, ha="right", fontsize=7
        )
        ax.set_ylabel("within-cell ceiling R$^2$")
        ax.legend(fontsize=7)
        fig.tight_layout()
        savefig_paper(fig, "char_ceilings", dir=args.out_dir)
        plt.close(fig)
        print(f"[plot] wrote {args.out_dir / 'char_ceilings.png'}", flush=True)

    # ---- AI-likeness vs ceiling-normalized rung-4 transfer ------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    any_pt = False
    for suffix, mk, series in (("", "o", "inserted"), ("_op", "s", "own answer")):
        for ch in fill.CHAR_CHARACTERS:
            v = f"char_{ch}{suffix}"
            if v not in pairs:
                continue
            spec = pairs[v]["spec"]
            d = _extract_dir(pairs[v]["entry"], args.basis, spec["src"], v)
            if d is None or d["ceiling"] <= 0:
                continue
            y = d["r2"]["4_bias_refit"] / d["ceiling"]
            ax.scatter(ail[ch], y, marker=mk, color=colors[ch], zorder=3)
            ax.annotate(
                f"{CHAR_LABEL[ch]}{' (op)' if suffix else ''}",
                (ail[ch], y),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
            any_pt = True
    if any_pt:
        ax.set_xlabel("banked AI-likeness (own answers, base; 0-100)")
        ax.set_ylabel("rung-4 transfer R$^2$ / target ceiling")
        fig.tight_layout()
        savefig_paper(fig, "ailikeness_vs_transfer", dir=args.out_dir)
        print(f"[plot] wrote {args.out_dir / 'ailikeness_vs_transfer.png'}", flush=True)
    plt.close(fig)

    # ---- summary: the §3 registered lattice --------------------------------
    lattice: dict[str, dict] = {}
    for v, blob in pairs.items():
        spec = blob["spec"]
        fwd = _extract_dir(blob["entry"], args.basis, spec["src"], v)
        rev = _extract_dir(blob["entry"], args.basis, v, spec["src"])
        if fwd is None:
            continue
        rung = reconciliation_rung(fwd["r2"], fwd["ceiling"])
        lattice[v] = {
            "source": spec["src"],
            "model": spec["model"],
            "n_matched": blob["entry"]["n_matched"],
            "ceiling_r2": fwd["ceiling"],
            "r2_by_rung": fwd["r2"],
            "fold_r2_std": fwd["fold_r2_std"],
            "null_r2": fwd["null_r2"],
            "identity_bias_r2": fwd["identity_bias_r2"],
            "knn_retrieval_fold0": fwd["knn_retrieval_fold0"],
            "reconciliation_rung": rung,
            "reconciled": rung is not None,
            "h1_tier": h1_tier(rung),
            "low_ceiling_artifact_risk": fwd["ceiling"] < LOW_CEILING,
            "reverse": None
            if rev is None
            else {
                "ceiling_r2": rev["ceiling"],
                "r2_by_rung": rev["r2"],
                "reconciliation_rung": reconciliation_rung(rev["r2"], rev["ceiling"]),
            },
        }
    summary = {
        "metadata": {
            "script": "scripts/issue1345_char_ladder_plot.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "basis": args.basis,
            "arm": args.arm,
            "margin": MARGIN,
            "low_ceiling_floor": LOW_CEILING,
            "fits_dir": str(args.fits_dir),
        },
        "ai_likeness_pooled_means": ail,
        "pairs": lattice,
        "cells_ceilings": [
            {"regime": r[0], "model": r[1], "context": r[2], "prefix": r[3]} for r in cell_rows
        ],
        "missing_pairs": missing,
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = summary_out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(summary_out)
    print(f"[plot] summary -> {summary_out}", flush=True)


if __name__ == "__main__":
    main()

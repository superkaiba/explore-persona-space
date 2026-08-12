"""Read the #1336 rigid decomposition: which manifold's rigid re-framing recovers transfer.

The four orth tiers the layer-30 gate computes (metric_ladder ORTH_TIER_NAMES) close a
2x2 over WHICH manifold gets a fitted rigid rotation, against the existing t0 and t5:

                        context NOT rotated        context rotated
  answer NOT rotated    t0  = W_s x_t              t5c = W_s(R_ctx x_t)
  answer rotated        t5  = R_ans(W_s x_t)       t5b = R_ans(W_s(R_ctx x_t))

with t5cs / t5bs the +scale variants and within_r2 the per-cell ceiling. R_ctx is fit
by orthogonal Procrustes from the TARGET's context cloud onto the SOURCE's; R_ans from
the SOURCE's answer cloud onto the TARGET's. Every rotation is fit on the TRAIN fold
only and scored out-of-fold, so an arm can (and on hard cells does) score BELOW t0.

The question this answers: when a checkpoint's context->answer map stops transferring,
is the damage a rigid re-framing of the INPUT geometry (t5c recovers it), of the OUTPUT
geometry (t5 recovers it), of BOTH (only t5b recovers it), or is it not rigid at all
(nothing recovers it and the deficit needs the affine/reparam tiers)?

Headline read is the DEFICIT RECOVERY FRACTION, computed per cell then medianed:

    recovery(arm) = (r2_arm - r2_t0) / (within_r2 - r2_t0)

0 = no better than applying the source operator unchanged; 1 = reaches the cell's own
ceiling. It is scale-free across pairs whose ceilings differ, which raw R2 is not.

Degenerate surface excluded, as everywhere in this line: chat/gsm8k_test1319 has
n = 1319 => n_train ~ 1055 < d = 4096, so its held-out R2 is estimator-degenerate
(#1701) and it is dropped from every median (kept in the per-cell JSON, flagged).

Example:
    uv run python scripts/issue1336_rigid_decomposition.py \\
        --pair-root eval_results/issue_1336_rigid/metric_ladder \\
        --out-dir figures/issue_1336
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps freeze at the numpy/matplotlib import — load_dotenv lands first.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LAYER = "30"
DEGENERATE = {("chat", "gsm8k_test1319")}

# Ordered as the lattice orders them: by source stage, then target stage.
PAIR_ORDER = (
    ("base", "sft"),
    ("base", "dpo"),
    ("base", "rlvr"),
    ("base", "rlvr_long"),
    ("sft", "dpo"),
    ("dpo", "rlvr"),
    ("dpo", "rlvr_long"),
)
PRETTY = {
    "base": "base",
    "sft": "SFT",
    "dpo": "DPO",
    "rlvr": "RLVR-PPO",
    "rlvr_long": "RLVR-GRPO",
}

# One colour = one meaning, held across both panels. t0 is the neutral reference.
ARMS = (
    ("t0", "direct (no rotation)", "#000000"),
    ("t5", "answer rotated", "#1F77B4"),
    ("t5c", "context rotated", "#D62728"),
    ("t5b", "both rotated", "#9467BD"),
)
RIGID_ARMS = ("t5", "t5c", "t5b")
CEILING_COLOR = "#7F7F7F"

_FNAME = re.compile(r"^pair_(.+)_(chat|naturalistic)_(.+)\.json$")


def _r2(node: dict) -> float | None:
    """Pull the raw-scale r2 out of either a standard tier node or an orth-tier node."""
    if not isinstance(node, dict):
        return None
    inner = node.get("raw", node)  # orth tiers nest raw/recal; standard tiers do not
    v = inner.get("r2") if isinstance(inner, dict) else None
    return float(v) if isinstance(v, (int, float)) else None


def load_cells(pair_root: Path) -> list[dict]:
    """One record per (pair, surface) with every arm's layer-30 raw R2 + the ceiling."""
    cells: list[dict] = []
    for p in sorted(pair_root.glob("pair_*.json")):
        m = _FNAME.match(p.name)
        if not m:
            print(f"[rigid] SKIP unparseable filename: {p.name}", flush=True)
            continue
        _, fmt, corpus = m.groups()
        d = json.loads(p.read_text())
        # Source/target come from the payload, never the filename: rlvr_long carries an
        # underscore, so a filename split cannot separate model from format reliably.
        pair = d.get("pair") or {}
        src, tgt = pair.get("m0"), pair.get("m1")
        layer = (d.get("per_layer") or {}).get(LAYER)
        if src is None or tgt is None or layer is None:
            print(f"[rigid] SKIP {p.name}: missing pair or layer {LAYER}", flush=True)
            continue
        raw = layer.get("raw") or {}
        tiers = raw.get("tiers") or {}
        orth = layer.get("orth_tiers") or {}
        rec = {
            "source": src,
            "target": tgt,
            "format": fmt,
            "corpus": corpus,
            "degenerate": (fmt, corpus) in DEGENERATE,
            "within_r2": raw.get("within_r2"),
            "file": p.name,
        }
        for k in ("t0", "t5"):
            rec[k] = _r2(tiers.get(k))
        for k in ("t5c", "t5cs", "t5b", "t5bs"):
            rec[k] = _r2(orth.get(k))
        cells.append(rec)
    return cells


def recovery(rec: dict, arm: str) -> float | None:
    """Fraction of this cell's transfer deficit that ``arm`` recovers (per-cell, then medianed)."""
    t0, w, a = rec.get("t0"), rec.get("within_r2"), rec.get(arm)
    if t0 is None or w is None or a is None:
        return None
    denom = w - t0
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        return None
    return (a - t0) / denom


def aggregate(cells: list[dict]) -> dict:
    """Per-pair medians over the NON-degenerate surfaces, with per-pair surface counts."""
    usable = [c for c in cells if not c["degenerate"]]
    out: dict[str, dict] = {}
    for src, tgt in PAIR_ORDER:
        sel = [c for c in usable if c["source"] == src and c["target"] == tgt]
        if not sel:
            continue
        row: dict = {"n_surfaces": len(sel)}
        for arm, _lbl, _col in ARMS:
            vals = [c[arm] for c in sel if c.get(arm) is not None]
            row[arm] = float(np.median(vals)) if vals else None
        ws = [c["within_r2"] for c in sel if c.get("within_r2") is not None]
        row["within_r2"] = float(np.median(ws)) if ws else None
        for arm in RIGID_ARMS:
            rs = [r for r in (recovery(c, arm) for c in sel) if r is not None]
            row[f"recovery_{arm}"] = float(np.median(rs)) if rs else None
        out[f"{src}__{tgt}"] = row
    return out


def figure(agg: dict, out_dir: Path, suffix: str = "") -> Path:
    """Two panels: absolute R2 per arm, and the deficit fraction each rotation recovers."""
    keys = [f"{s}__{t}" for s, t in PAIR_ORDER if f"{s}__{t}" in agg]
    if not keys:
        raise SystemExit("no pairs to plot — is --pair-root populated?")
    labels = [f"{PRETTY[k.split('__')[0]]}\n{chr(8594)} {PRETTY[k.split('__')[1]]}" for k in keys]
    x = np.arange(len(keys))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(1.55 * len(keys) + 3.2, 8.2), sharex=True)
    fig.set_layout_engine("none")  # the style's constrained_layout otherwise wins

    for arm, lbl, col in ARMS:
        y = [agg[k].get(arm) for k in keys]
        ax0.plot(x, y, "o-", color=col, label=lbl, lw=1.8, ms=6)
    ax0.plot(
        x,
        [agg[k].get("within_r2") for k in keys],
        "s--",
        color=CEILING_COLOR,
        label="ceiling (within-map)",
        lw=1.6,
        ms=5,
    )
    ax0.axhline(0.0, color="#CCCCCC", lw=0.8, zorder=0)
    ax0.set_ylabel("held-out $R^2$ (layer 30)")
    ax0.set_title("Rigid re-framing of the context$\\to$answer map")
    ax0.legend(frameon=False, fontsize=9, ncol=2)

    w = 0.26
    for i, arm in enumerate(RIGID_ARMS):
        col = dict((a, c) for a, _l, c in ARMS)[arm]
        lbl = dict((a, l) for a, l, _c in ARMS)[arm]
        ax1.bar(
            x + (i - 1) * w, [agg[k].get(f"recovery_{arm}") for k in keys], w, color=col, label=lbl
        )
    ax1.axhline(0.0, color="#000000", lw=1.0)
    ax1.axhline(1.0, color=CEILING_COLOR, lw=1.0, ls="--")
    ax1.set_ylabel("fraction of transfer deficit recovered")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend(frameon=False, fontsize=9)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.945, bottom=0.11, hspace=0.13)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"rigid_decomposition{suffix}.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--pair-root",
        type=Path,
        default=Path("eval_results/issue_1336_rigid/metric_ladder"),
        help="dir of pair_<m0>__<m1>_<fmt>_<corpus>.json files from the orth-tier run",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("figures/issue_1336"))
    ap.add_argument("--suffix", default="", help="appended to the output stem")
    ap.add_argument("--no-figure", action="store_true", help="write the JSON only")
    args = ap.parse_args()

    cells = load_cells(args.pair_root)
    if not cells:
        raise SystemExit(f"no pair files under {args.pair_root}")
    agg = aggregate(cells)

    n_deg = sum(1 for c in cells if c["degenerate"])
    print(f"[rigid] {len(cells)} cells ({n_deg} degenerate, excluded from medians)", flush=True)
    for k, row in agg.items():
        print(
            "[rigid] %-22s n_surf=%d  t0=%s t5=%s t5c=%s t5b=%s ceil=%s | recov t5=%s t5c=%s t5b=%s"
            % (
                k,
                row["n_surfaces"],
                *[
                    ("%.3f" % row[f] if row.get(f) is not None else "  n/a")
                    for f in ("t0", "t5", "t5c", "t5b", "within_r2")
                ],
                *[
                    (
                        "%.3f" % row[f"recovery_{a}"]
                        if row.get(f"recovery_{a}") is not None
                        else " n/a"
                    )
                    for a in RIGID_ARMS
                ],
            ),
            flush=True,
        )

    payload = {
        "layer": int(LAYER),
        "degenerate_surfaces_excluded": sorted("/".join(s) for s in DEGENERATE),
        "n_cells": len(cells),
        "per_pair": agg,
        "cells": cells,
        "recovery_definition": "(r2_arm - r2_t0) / (within_r2 - r2_t0), per cell then medianed",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jpath = args.out_dir / f"rigid_decomposition{args.suffix}.json"
    jpath.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[rigid] wrote {jpath}", flush=True)

    if not args.no_figure:
        print(f"[rigid] wrote {figure(agg, args.out_dir, args.suffix)}", flush=True)


if __name__ == "__main__":
    main()

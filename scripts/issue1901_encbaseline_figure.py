#!/usr/bin/env python3
"""Issue #1901 inline round ``encbaseline`` — figure for the encoder semantic baseline.

Three panels, chosen so every curve inside a panel shares a comparison basis:

  A  Held-out R^2 on the SAME target v_A       arm1_lm / arm4_pca_lm / arm2_enc_lm
  B  Top-1 retrieval, matched 1,000-row pool   + arm3_enc_enc / arm6_cos0
  C  Residual R^2 (encoder-unreachable part)   arm5_resid

Why the split. Panel A holds only the arms predicting v_A, so their R^2 share a
denominator and are directly comparable. arm3 predicts e(y) in the ENCODER's
space and arm5 predicts a residual, so neither R^2 belongs on A's axis; a
rank-based top-1 at a matched pool size IS comparable across spaces, which is
why arm3 and the zero-parameter arm6 appear on B, and arm5 gets its own panel
with its own denominator stated in the axis label.

Encoder is encoded by linestyle (bge solid, e5 dashed), arm by color, so the
figure survives grayscale. No caption block is drawn on the canvas (standing
user directive 2026-08-12): axes, ticks, legend and panel titles only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import set_c2a_style  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

# One color = one meaning, held across every panel.
ARM_COLOR = {
    "arm1_lm": "#1b3a6b",
    "arm4_pca_lm": "#4c8fd1",
    "arm2_enc_lm": "#c2452d",
    "arm3_enc_enc": "#e08a3c",
    "arm6_cos0": "#7a7a7a",
    "arm5_resid": "#2f6b4f",
}
ARM_LABEL = {
    "arm1_lm": "LM context state (3584-d)",
    "arm4_pca_lm": "LM context state, PCA to 1024-d",
    "arm2_enc_lm": "Encoder embedding of the same prompt",
    "arm3_enc_enc": "Encoder prompt to encoder answer",
    "arm6_cos0": "Cosine only, no fit",
    "arm5_resid": "LM state on encoder-unreachable residual",
}
ENC_STYLE = {"bge": "-", "e5": "--"}
ENC_LABEL = {"bge": "bge-large-en-v1.5", "e5": "multilingual-e5-large"}

PANEL_A = ("arm1_lm", "arm4_pca_lm", "arm2_enc_lm")
PANEL_B = ("arm1_lm", "arm4_pca_lm", "arm2_enc_lm", "arm3_enc_enc", "arm6_cos0")
PANEL_C = ("arm5_resid",)


def _series(payload: dict, arm: str, field: str):
    xs, ys = [], []
    for rung in payload["per_rung"]:
        entry = rung["arms"].get(arm)
        if entry is None:
            continue
        val = entry.get("r2") if field == "r2" else (entry.get("top1") or {}).get("acc_at_1")
        if val is None:
            continue
        xs.append(rung["n_train"])
        ys.append(val)
    return xs, ys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir", default=str(PROJECT_ROOT / "eval_results/issue_1901/encbaseline")
    )
    ap.add_argument(
        "--stem",
        default=str(PROJECT_ROOT / "figures/issue_1901/encbaseline/encoder_semantic_baseline"),
    )
    ap.add_argument("--prefix", default="results", help="results file prefix (results | results2)")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    payloads = {}
    for enc in ("bge", "e5"):
        p = rdir / f"{args.prefix}_{enc}.json"
        if p.exists():
            payloads[enc] = json.loads(p.read_text())
        else:
            print(f"[fig] missing {p} — skipping {enc}")
    if not payloads:
        raise SystemExit(f"no results files under {rdir} with prefix {args.prefix}")

    set_c2a_style()
    # set_c2a_style() sizes type for a narrow paper include width; this is a wide
    # 3-panel chat/report canvas, so scale the type back down or the labels clip.
    matplotlib.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), constrained_layout=True)
    axA, axB, axC = axes

    for enc, payload in payloads.items():
        ls = ENC_STYLE[enc]
        for arm in PANEL_A:
            xs, ys = _series(payload, arm, "r2")
            if xs:
                axA.plot(xs, ys, ls, color=ARM_COLOR[arm], marker="o", ms=4.5, lw=1.9)
        for arm in PANEL_B:
            xs, ys = _series(payload, arm, "top1")
            if xs:
                axB.plot(xs, ys, ls, color=ARM_COLOR[arm], marker="o", ms=4.5, lw=1.9)
        for arm in PANEL_C:
            xs, ys = _series(payload, arm, "r2")
            if xs:
                axC.plot(xs, ys, ls, color=ARM_COLOR[arm], marker="o", ms=4.5, lw=1.9)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Training contexts")
        ax.grid(True, alpha=0.22, lw=0.7)

    axA.set_ylabel("Held-out $R^2$ (answer state)")
    axA.set_title("A. Variance explained, same target", loc="left", fontsize=11)
    axB.set_ylabel("Top-1 retrieval, pool 1,000")
    axB.set_title("B. Retrieval, matched pool", loc="left", fontsize=11)
    axC.set_ylabel("$R^2$ (encoder-unreachable part)")
    axC.set_title("C. Beyond the encoder", loc="left", fontsize=11)

    cvals = [y for _, ys in [_series(pl, "arm5_resid", "r2") for pl in payloads.values()] for y in ys]
    if cvals:
        axC.set_ylim(0.0, max(cvals) * 1.3)

    arm_handles = [
        Line2D([], [], color=ARM_COLOR[a], lw=2.2, label=ARM_LABEL[a])
        for a in (
            "arm1_lm",
            "arm4_pca_lm",
            "arm2_enc_lm",
            "arm3_enc_enc",
            "arm6_cos0",
            "arm5_resid",
        )
    ]
    enc_handles = [
        Line2D([], [], color="#333333", lw=2.0, linestyle=ENC_STYLE[e], label=ENC_LABEL[e])
        for e in payloads
    ]
    fig.legend(
        handles=arm_handles + enc_handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )

    stem = Path(args.stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    prov = git_provenance()
    sidecar = {
        "figure": stem.name,
        "issue": 1901,
        "round": "encbaseline",
        "sources": {e: str(rdir / f"{args.prefix}_{e}.json") for e in payloads},
        "panels": {"A": list(PANEL_A), "B": list(PANEL_B), "C": list(PANEL_C)},
        "comparability_note": (
            "Panel A holds only arms predicting v_A so their R^2 share a denominator. "
            "arm3 predicts the encoder-space answer embedding and arm5 a residual, so "
            "neither R^2 is on A's axis; top-1 at a matched pool size is comparable "
            "across spaces, which is why arm3 and arm6 appear on B."
        ),
        **as_metadata_dict(prov, phase="encbaseline-figure"),
    }
    stem.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2))
    print(f"[fig] wrote {png}")
    print(f"[fig] wrote {pdf}")
    print(f"[fig] wrote {stem.with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

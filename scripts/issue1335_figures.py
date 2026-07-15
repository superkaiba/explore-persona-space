"""Issue #1335: ladder figures (hero + exploratory dump) from eval_results JSONs.

Pure-CPU, reads only `eval_results/issue_1335/` (cells_*.json, matched_*.json,
ladder_summary.json, nulls_*.json, swap_*.json) — re-runnable off-instance.

Figures (plan §6):
  hero_ladder            R^2 @ L19 per rung (matched-n + full-n panels), base +
                         instruct, ctx filled / prefix hollow, null band,
                         committed endpoint anchors as reference lines
  delta_waterfall        adjacent-rung deltas with CIs (6-family + length)
  layer_sweep            28-layer R^2 per rung (per model, ctx arm)
  per_persona_endpoint   fiction per-persona scatter (points labeled)
  tf_vs_op_r2            r2 tf-vs-op calibration pair
  swap_specificity       correct vs swap bars on r7/r6/s1

CLI:
  uv run python scripts/issue1335_figures.py [--out-dir eval_results/issue_1335] \
      [--fig-dir figures/issue_1335] [--models base,instruct]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1310_common as c1310  # noqa: E402
import issue1335_render_rungs as r1335  # noqa: E402

SCRIPT = "scripts/issue1335_figures.py"

LADDER = (
    "r0_qa_full",
    "r1_qa_oneline",
    "r2_tf",
    "r2_op",
    "r3_persona",
    "r4_fictionframe",
    "r6_nofoil",
    "r7_endpoint",
    "s1_assistant_label",
    "s2a_familiar",
    "s2b_novel",
)
QA = {s for s in LADDER if r1335.RUNGS[s]["family"] == "qa"}

# Committed endpoint anchors drawn as reference lines (plan §6 hero spec).
REF_LINES = {
    "#825 instruct chat (S1)": 0.6731,
    "#825 base chat (S2)": 0.5877,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1335"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_1335"))
    ap.add_argument("--models", type=str, default="base,instruct")
    return ap.parse_args()


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _cell(out_dir: Path, slug: str, model: str, unit: str, arm: str) -> dict | None:
    unit_part = "" if unit == "all" else f"__{unit}"
    return _load(out_dir / f"cells_{slug}__{model}{unit_part}__{arm}.json")


def _matched(out_dir: Path, slug: str, model: str, unit: str, arm: str) -> dict | None:
    unit_part = "" if unit == "all" else f"__{unit}"
    return _load(out_dir / f"matched_{slug}__{model}{unit_part}__{arm}.json")


def _rung_value(out_dir: Path, slug: str, model: str, arm: str, matched: bool) -> float | None:
    """L19 R^2 per rung: Q&A = the single cell; fiction = per-persona mean."""
    if slug in QA:
        d = (
            _matched(out_dir, slug, model, "all", arm)
            if matched
            else _cell(out_dir, slug, model, "all", arm)
        )
        if d is None:
            return None
        return d["r2_headline_mean"] if matched else d["r2_per_layer_obs"][d["headline_layer"]]
    vals = []
    for persona in c1310.PERSONA_LABELS:
        d = (
            _matched(out_dir, slug, model, persona, arm)
            if matched
            else _cell(out_dir, slug, model, persona, arm)
        )
        if d is None:
            continue
        vals.append(
            d["r2_headline_mean"] if matched else d["r2_per_layer_obs"][d["headline_layer"]]
        )
    return float(np.mean(vals)) if vals else None


def _null_band(out_dir: Path, slug: str, model: str) -> float | None:
    """Per-rung shuffle-null p97.5 at L19 (ctx arm, full-n; QA cell or Wren)."""
    unit = "all" if slug in QA else "Wren"
    d = _cell(out_dir, slug, model, unit, "ctx")
    if d is None:
        return None
    hl = str(d["headline_layer"])
    return d["selection_symmetric"]["frozen_layer_table"].get(hl, {}).get("null_p975")


def fig_hero(args, models: list[str]) -> None:
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    colors = paper_palette(len(models))
    x = np.arange(len(LADDER))
    width = 0.36
    for panel, matched in ((0, True), (1, False)):
        ax = axes[panel]
        for mi, model in enumerate(models):
            ctx = [_rung_value(args.out_dir, s, model, "ctx", matched) for s in LADDER]
            pre = [_rung_value(args.out_dir, s, model, "prefix", matched) for s in LADDER]
            xs = x + (mi - (len(models) - 1) / 2) * width
            ax.bar(
                xs,
                [v if v is not None else np.nan for v in ctx],
                width=width * 0.92,
                color=colors[mi],
                label=f"{model} (ctx)",
            )
            ax.scatter(
                xs,
                [v if v is not None else np.nan for v in pre],
                facecolors="none",
                edgecolors="black",
                s=28,
                zorder=5,
                label=f"{model} (prefix)" if mi == 0 else None,
            )
            if not matched:
                nulls = [_null_band(args.out_dir, s, model) for s in LADDER]
                ax.plot(
                    xs,
                    [v if v is not None else np.nan for v in nulls],
                    ls="none",
                    marker="_",
                    ms=14,
                    color="grey",
                    zorder=4,
                )
        for name, val in REF_LINES.items():
            ax.axhline(val, ls=":", lw=0.9, color="grey")
            ax.annotate(name, (len(LADDER) - 0.4, val), fontsize=6, va="bottom", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([s.split("_")[0] for s in LADDER], rotation=0)
        ax.set_title("matched-n (seed-mean, 5 draws)" if matched else "full-n")
        ax.set_ylabel("held-out $R^2$ @ L19")
        ax.axhline(0, color="black", lw=0.6)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Ablation ladder: context-arm held-out $R^2$ @ L19 (grey dash = null p97.5)")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "hero_ladder.png", dpi=200)
    plt.close(fig)


def fig_waterfall(args, models: list[str]) -> None:
    summary = _load(args.out_dir / "ladder_summary.json")
    if summary is None:
        return
    set_paper_style()
    keys = [
        "label",
        "label_op_companion",
        "header",
        "framing",
        "content_depth",
        "content_depth_wren_matched",
        "foils",
        "label_restore",
        "name_frequency_sub",
        "length",
    ]
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.2), squeeze=False)
    for mi, model in enumerate(models):
        ax = axes[0][mi]
        dd = summary["per_model"].get(model, {}).get("deltas", {})
        ys, labels, los, his = [], [], [], []
        for k in keys:
            v = dd.get(k)
            if v is None:
                continue
            labels.append(k + (" (outside family)" if k == "length" else ""))
            ys.append(v["value"])
            los.append(v["value"] - v["ci_lo"])
            his.append(v["ci_hi"] - v["value"])
        pos = np.arange(len(ys))
        ax.barh(pos, ys, xerr=[los, his], color=paper_palette(1)[0])
        ax.set_yticks(pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color="black", lw=0.6)
        g = summary["per_model"].get(model, {}).get("gap", {}).get("G")
        gtxt = f"G={g['value']:.3f} [{g['ci_lo']:.3f},{g['ci_hi']:.3f}]" if g else "G=n/a"
        ax.set_title(f"{model}: oriented deltas @ L19 matched-n ({gtxt})")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(args.fig_dir / "delta_waterfall.png", dpi=200)
    plt.close(fig)


def fig_layer_sweep(args, models: list[str]) -> None:
    set_paper_style()
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.2), squeeze=False)
    colors = paper_palette(len(LADDER))
    for mi, model in enumerate(models):
        ax = axes[0][mi]
        for si, slug in enumerate(LADDER):
            unit = "all" if slug in QA else "Wren"
            d = _cell(args.out_dir, slug, model, unit, "ctx")
            if d is None:
                continue
            ax.plot(d["r2_per_layer_obs"], lw=1.0, color=colors[si], label=slug)
        ax.axvline(c1310.HEADLINE_LAYER, ls=":", color="grey", lw=0.8)
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out $R^2$")
        ax.set_title(f"{model}: full-n ctx-arm layer sweep (fiction: Wren cell)")
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "layer_sweep.png", dpi=200)
    plt.close(fig)


def fig_per_persona(args, models: list[str]) -> None:
    set_paper_style()
    fiction = [s for s in LADDER if s not in QA]
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 4.0), squeeze=False)
    for mi, model in enumerate(models):
        ax = axes[0][mi]
        for si, slug in enumerate(fiction):
            for persona in c1310.PERSONA_LABELS:
                d = _cell(args.out_dir, slug, model, persona, "ctx")
                if d is None:
                    continue
                y = d["r2_per_layer_obs"][d["headline_layer"]]
                ax.scatter([si], [y], s=16)
                ax.annotate(persona, (si, y), fontsize=6, xytext=(3, 0), textcoords="offset points")
        ax.set_xticks(range(len(fiction)))
        ax.set_xticklabels([s.split("_")[0] for s in fiction])
        ax.set_ylabel("held-out $R^2$ @ L19 (full-n)")
        ax.set_title(f"{model}: fiction per-persona cells (labeled points)")
        ax.axhline(0, color="black", lw=0.6)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "per_persona_endpoint.png", dpi=200)
    plt.close(fig)


def fig_tf_vs_op(args, models: list[str]) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for model in models:
        tf = _rung_value(args.out_dir, "r2_tf", model, "ctx", matched=False)
        op = _rung_value(args.out_dir, "r2_op", model, "ctx", matched=False)
        if tf is None or op is None:
            continue
        ax.scatter([op], [tf], s=30)
        ax.annotate(model, (op, tf), fontsize=7, xytext=(4, 2), textcoords="offset points")
    lims = ax.get_xlim()
    ax.plot(lims, lims, ls=":", color="grey", lw=0.8)
    ax.set_xlabel("r2_op held-out $R^2$ @ L19 (on-policy)")
    ax.set_ylabel("r2_tf held-out $R^2$ @ L19 (teacher-forced swap)")
    ax.set_title("tf-vs-op calibration pair (r2 rung)")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "tf_vs_op_r2.png", dpi=200)
    plt.close(fig)


def fig_swap(args, models: list[str]) -> None:
    set_paper_style()
    rungs = ("r7_endpoint", "r6_nofoil", "s1_assistant_label")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    width = 0.2
    x = np.arange(len(rungs))
    colors = paper_palette(2 * len(models))
    for mi, model in enumerate(models):
        cor, swp = [], []
        for slug in rungs:
            d = _load(args.out_dir / f"swap_{slug}_{model}.json")
            cor.append(d["r2_correct"] if d else np.nan)
            swp.append(d["r2_swap"] if d else np.nan)
        ax.bar(
            x + (2 * mi) * width - width * 1.5,
            cor,
            width,
            color=colors[2 * mi],
            label=f"{model} correct",
        )
        ax.bar(
            x + (2 * mi + 1) * width - width * 1.5,
            swp,
            width,
            color=colors[2 * mi + 1],
            label=f"{model} swap",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([s.split("_")[0] for s in rungs])
    ax.set_ylabel("held-out $R^2$ @ L19")
    ax.set_title("Character-swap specificity (correct vs deranged pairing)")
    ax.legend(fontsize=7)
    ax.axhline(0, color="black", lw=0.6)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "swap_specificity.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print("[phase=p4_figures] ladder figures")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_hero(args, models)
    fig_waterfall(args, models)
    fig_layer_sweep(args, models)
    fig_per_persona(args, models)
    fig_tf_vs_op(args, models)
    fig_swap(args, models)
    meta = {
        "script": SCRIPT,
        "out_dir": str(args.out_dir),
        "models": models,
        "figures": sorted(p.name for p in args.fig_dir.glob("*.png")),
    }
    (args.fig_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[i1335-figs] wrote {len(meta['figures'])} figures -> {args.fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

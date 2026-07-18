"""Figures for issue #1335 follow-up round `onpolicy-assistant-label` (plan v7 §6).

Pure-CPU, reads only <out-dir>/label_comparison.json (self-contained digest of
the round: within-run paired deltas + placement reads + collapse audits + the
registered empirical H0 pair-noise band) — re-runnable off-instance.

Figures:
  hero_label_delta   the paired within-run Assistant-Wren delta per model (ctx
                     filled / prefix hollow, 95% joint-draw CI whiskers, white
                     per-draw dots), the Wren45-Wren46 replicate H0 pair
                     alongside, the committed non-generative label bounds
                     (tf relabel <=0.003 / name probes 0.003) as reference
                     lines, and the registered base H0 band +-B_hat shaded.
  placement_panel    the new cells' matched-n ctx R^2 placed on the committed
                     ladder endpoint bars (per seed) + the committed one-line
                     Q&A anchor, per model.
  collapse_slots     per-slot under-floor rollout rates per cell (all slots x
                     all 6 cells; the migrating "I agree."-mode detector).

CLI:
  uv run python scripts/issue1335_fig_label.py \
      [--out-dir eval_results/issue_1335/onpolicy-assistant-label] \
      [--fig-dir figures/issue_1335/onpolicy-assistant-label]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    set_paper_style,
)

SCRIPT = "scripts/issue1335_fig_label.py"

# Committed non-generative label-channel bounds (read-only anchors, plan §3):
# tf relabel s1_assistant_label <=0.003; familiar/novel name probes 0.003.
TF_LABEL_BOUND = 0.003

PAIR_LABELS = {
    "delta_AW": "Assistant − Wren (within-run)",
    "delta_wren_replicate": "Wren45 − Wren46 (replicate H0)",
}
RUNG_LABELS = {
    "r7_op_assistant": "Assistant lead",
    "r7_op_wren": "Wren lead (seed 45)",
    "r7_op_wren46": "Wren lead (seed 46)",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-dir", type=Path, default=Path("eval_results/issue_1335/onpolicy-assistant-label")
    )
    ap.add_argument(
        "--fig-dir", type=Path, default=Path("figures/issue_1335/onpolicy-assistant-label")
    )
    return ap.parse_args()


def _off(v: float, lo: float, hi: float) -> tuple[float, float]:
    """NON-NEGATIVE (lo, hi) errorbar offsets from CI bounds (matplotlib takes
    offsets, never bounds/signed deltas — a tiny-n quantile CI can invert
    around the point estimate and a negative entry raises at render time;
    #547/#1335). A clamped bound reads 0-width on that side; the JSON carries
    the exact numbers."""
    return max(0.0, float(v) - float(lo)), max(0.0, float(hi) - float(v))


def fig_hero(lc: dict, fig_dir: Path) -> None:
    models = list(lc["per_model"])
    b_hat = float(lc["h0_pair_noise_band"]["b_hat"])
    fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 4.4), squeeze=False)
    pal = paper_palette(2)
    for ax, model in zip(axes[0], models, strict=True):
        deltas = lc["per_model"][model]["deltas_full_n"]
        x = 0
        xticks, xlabels = [], []
        for pair in ("delta_AW", "delta_wren_replicate"):
            for arm, filled in (("ctx", True), ("prefix", False)):
                d = deltas[pair][arm]
                lo_off, hi_off = _off(d["value"], d["ci_lo"], d["ci_hi"])
                color = pal[0] if pair == "delta_AW" else pal[1]
                draws = d.get("draws_sample") or []
                if draws:
                    ax.scatter(
                        np.full(len(draws), x, dtype=float) + np.linspace(-0.12, 0.12, len(draws)),
                        draws,
                        s=6,
                        color="white",
                        edgecolors="0.55",
                        linewidths=0.4,
                        zorder=2,
                    )
                ax.errorbar(
                    [x],
                    [d["value"]],
                    yerr=[[lo_off], [hi_off]],
                    fmt="o",
                    markersize=8,
                    color=color,
                    markerfacecolor=color if filled else "white",
                    markeredgecolor=color,
                    capsize=4,
                    zorder=3,
                )
                xticks.append(x)
                xlabels.append(f"{PAIR_LABELS[pair].split(' (')[0]}\n{arm}")
                x += 1.2
            x += 0.8
        if model == "base":
            ax.axhspan(-b_hat, b_hat, color="0.88", zorder=0)
            ax.text(
                xticks[-1] + 0.35,
                b_hat,
                f"±B̂ = {b_hat:.3f} (H0 band)",
                fontsize=8,
                color="0.35",
                va="bottom",
                ha="right",
            )
        for y in (TF_LABEL_BOUND, -TF_LABEL_BOUND):
            ax.axhline(y, color="0.55", linewidth=0.8, linestyle="--", zorder=1)
        ax.axhline(0.0, color="0.3", linewidth=0.8, zorder=1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("Δ held-out R² (layer 19)")
        ax.set_title(f"{model} — within-run paired deltas", loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "hero_label_delta.png", dpi=200)
    plt.close(fig)


def fig_placement(lc: dict, fig_dir: Path) -> None:
    models = list(lc["per_model"])
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        pm = lc["per_model"][model]
        place = pm["placement"]
        refs = place["committed_references"]
        bars, labels, colors = [], [], []
        pal = paper_palette(3)
        for seed_key in ("seed42", "seed43", "seed44"):
            per_p = refs.get(f"{seed_key}_r7_endpoint_per_persona")
            if not per_p:
                continue
            for persona, v in per_p.items():
                bars.append(float(v))
                labels.append(f"{seed_key[4:]}:{persona}")
                colors.append("0.75")
        for slug in ("r7_op_assistant", "r7_op_wren", "r7_op_wren46"):
            v = place["values_ctx"].get(slug)
            if v is None:
                continue
            bars.append(float(v))
            labels.append(RUNG_LABELS[slug])
            colors.append(pal[0] if slug == "r7_op_assistant" else pal[1])
        xs = np.arange(len(bars))
        ax.bar(xs, bars, color=colors, width=0.72)
        r1 = refs.get("seed42_r1_qa_oneline")
        if r1 is not None:
            ax.axhline(
                float(r1), color="0.35", linewidth=0.9, linestyle=":", label="one-line Q&A (s42)"
            )
            ax.legend(fontsize=8, loc="upper right")
        note = "" if place["at_committed_n"] else " (PAIRWISE-MIN n — not committed-n comparable)"
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(f"held-out R² @ matched n={place['n_used']}")
        ax.set_title(f"{model} — placement vs committed endpoint cells{note}", loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "placement_panel.png", dpi=200)
    plt.close(fig)


def fig_collapse(lc: dict, fig_dir: Path) -> None:
    models = list(lc["per_model"])
    fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 3.8), squeeze=False)
    pal = paper_palette(3)
    for ax, model in zip(axes[0], models, strict=True):
        audits = lc["per_model"][model]["collapse_audits"]
        slugs = list(audits)
        slots = sorted({s for a in audits.values() for s in a["modal_line_per_slot"]})
        width = 0.8 / max(1, len(slugs))
        for i, slug in enumerate(slugs):
            a = audits[slug]
            rates = []
            for slot in slots:
                tot = a["modal_line_per_slot"].get(slot, {}).get("slot_lines", 0)
                under = a["under_floor_per_slot"].get(slot, 0)
                rates.append(100.0 * under / tot if tot else 0.0)
            xs = np.arange(len(slots)) + (i - (len(slugs) - 1) / 2) * width
            ax.bar(xs, rates, width=width, color=pal[i % len(pal)], label=RUNG_LABELS[slug])
        ax.set_xticks(np.arange(len(slots)))
        ax.set_xticklabels(slots, fontsize=8)
        ax.set_ylabel("under-4-token lines (%)")
        ax.set_title(f"{model} — per-slot degenerate-line rates", loc="left")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "collapse_slots.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_paper_style("blog")
    lc_path = args.out_dir / "label_comparison.json"
    assert lc_path.exists(), f"missing {lc_path} — run issue1335_fit.py --label-compare first"
    lc = json.loads(lc_path.read_text())
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_hero(lc, args.fig_dir)
    fig_placement(lc, args.fig_dir)
    fig_collapse(lc, args.fig_dir)
    meta = {
        "script": SCRIPT,
        "source": str(lc_path),
        "code_sha": lc.get("code_sha"),
        "gen_seed": lc.get("gen_seed"),
        "figures": sorted(p.name for p in args.fig_dir.glob("*.png")),
    }
    (args.fig_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[i1335-fig-label] wrote {len(meta['figures'])} figures -> {args.fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

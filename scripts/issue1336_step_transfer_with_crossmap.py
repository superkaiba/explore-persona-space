#!/usr/bin/env python3
"""Issue #1336 — per-adjacent-step transfer, with the DIRECTLY-FITTED cross map.

Extends `issue1336_step_transfer_tiers` with one extra series and one extra
figure. Nothing is re-fitted: every number here is READ from a committed JSON
field produced by the round-3 / round-B batteries.

THE NEW SERIES — "cross: source ctx -> target answer (fresh fit)"
    A ridge map fitted DIRECTLY from the SOURCE model's context vector to the
    TARGET model's answer vector (X = v_context(source), Y = v_answer(target)),
    scored on the target's held-out answers.

    It is NOT a tier, and it is deliberately drawn in a different colour family
    from the blue tier ramp. Every tier reuses the source's ALREADY-FITTED
    operator W_s and asks "does the source's map still work on the target?".
    The cross map throws W_s away and asks "is the target's answer state
    predictable from the source's context state AT ALL?".

    That distinction is the whole point: a low tier read can mean either the
    MAP changed or the REPRESENTATIONS moved, and the tiers alone conflate
    those. The cross map bounds the second: where cross sits at the ceiling, a
    tier shortfall is an operator-transfer problem; where cross itself falls
    short of the ceiling, information about the target's answer is genuinely
    missing from the source's context state.

    Because it is a fresh fit on the source's inputs, it is a per-CELL constant
    — one value per (step, surface), NOT one per tier — and it is NOT bounded
    above by the within-model ceiling.

PROVENANCE — two sources, same construction, different field names:
  * base->SFT, SFT->DPO, DPO->RLVR:  `repswap_r2` at per_layer/30 of the
    round-3 metric_ladder pair files (issue1336_metric_ladder.py:
    `fit_repswap = _v2_yfit(prep_s, Yt_l[tr])`, "rep-swap ceiling x_s -> y_t").
  * RLVR->longer RLVR:               `cross_r2` in the selfmap_v3 cells
    (issue1336_selfmap_missing_pairs.py: `fit_cross = _v2_yfit(prep_s, Yt[tr])`).
  Both are layer 30, fit_seed 0, 5 outer folds, fold-local pooled OOF, raw
  scale — the same basis the tier lines already share, which is why they may
  sit on one axis. Missing cells FAIL LOUD rather than plotting a short line.

  The pair files live under `data/issue_1336/hf_dl/` and are a REAPABLE cache
  (Step-8 janitor). If they are absent, re-fetch the 56 files (~6.5 MB) from
  the HF data repo prefix
  `issue1336_rlvr_ladder/eval_results_mirror_v2/metric_ladder`.

FIGURES
  1. ladder_step_transfer_by_tier_crossmap.png — the committed aggregate A/B
     panels with the cross series added.
  2. ladder_step_transfer_by_dataset.png       — one subplot per EVAL DATASET
     (all 8 surfaces), each carrying the 4 tiers + the cross map + the ceiling.

Degenerate handling is inherited verbatim: gsm8k_test1319 (n_train~1034 <
d=4096) is excluded from every aggregate and MARKED (shaded panel, 'x'
markers) in the per-dataset view — never silently dropped.

Run from the issue-1336 worktree root:
    uv run python scripts/issue1336_step_transfer_with_crossmap.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the matplotlib/numpy imports.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue1336_metric_ladder_plots as mlp  # noqa: E402
import issue1336_step_transfer_tiers as stt  # noqa: E402

REPO = stt.REPO
OUTDIR = stt.OUTDIR

# The cross map gets its own colour family, held identical across BOTH figures
# this script emits. One colour = one meaning: blue ramp = nested tiers, grey
# dashed = within-model ceiling, orange = the fresh cross fit.
CROSS_COLOR = "#d94801"
CROSS_LABEL = "cross: source ctx → target answer (fresh fit)"

PAIRFILE_GLOB = str(REPO / "data" / "issue_1336" / "hf_dl" / "**" / "metric_ladder" / "pair_*.json")
SELFMAP_CELLS = stt.SELFMAP_CELLS


def load_cross() -> dict[tuple[str, str, str], float]:
    """(pair, fmt, corpus) -> R² of the directly-fitted source-ctx -> target-ans map."""
    out: dict[tuple[str, str, str], float] = {}
    prov: dict[tuple[str, str, str], str] = {}

    # The `chat|naturalistic` literal anchors the split so a target like
    # `rlvr_long` (which itself contains an underscore) is not mis-parsed.
    pat = re.compile(r"pair_(.+?)__(.+?)_(chat|naturalistic)_(.+)\.json")
    for fp in sorted(set(glob.glob(PAIRFILE_GLOB, recursive=True))):
        m = pat.match(Path(fp).name)
        if not m:
            continue
        src, tgt, fmt, corpus = m.groups()
        val = json.load(open(fp)).get("per_layer", {}).get("30", {}).get("repswap_r2")
        if val is None:
            continue
        key = (f"{src}__{tgt}", fmt, corpus)
        out.setdefault(key, float(val))
        prov.setdefault(key, "round-3 pair file `repswap_r2`")

    if SELFMAP_CELLS.is_dir():
        for fp in sorted(SELFMAP_CELLS.glob("*.json")):
            for rec in json.load(open(fp))["records"]:
                v = rec.get("cross_r2")
                if v is None or rec["pair"] == "base__base":
                    continue
                key = (rec["pair"], rec["format"], rec["corpus"])
                out.setdefault(key, float(v))
                prov.setdefault(key, "selfmap_v3 `cross_r2`")

    missing = [
        (f"{s}__{t}", fmt, corpus)
        for (s, t) in stt.STEPS
        for (fmt, corpus) in stt.SURFACES
        if (f"{s}__{t}", fmt, corpus) not in out
    ]
    if missing:
        raise RuntimeError(
            f"cross-map value missing for {len(missing)} of "
            f"{len(stt.STEPS) * len(stt.SURFACES)} plotted cells: {missing[:6]}...\n"
            f"round-3 pair files are a REAPABLE cache — re-fetch the 56 files from HF "
            f"prefix issue1336_rlvr_ladder/eval_results_mirror_v2/metric_ladder into "
            f"{Path(PAIRFILE_GLOB).parents[1]}/ and re-run. Plotting a short line "
            f"instead would read as measured-and-absent."
        )
    return out, prov


CROSS, CROSS_PROV = load_cross()


def cross_at(src: str, tgt: str, fmt: str, corpus: str) -> float:
    return CROSS[(f"{src}__{tgt}", fmt, corpus)]


def _draw_points(ax, si, vals, ylim, color, marker_deg="x"):
    """Scatter each surface; off-scale points become carets AT the boundary.

    Copied from the committed aggregate plotter so the two figures use one
    convention; kept local because the original is nested inside its figure fn.
    """
    lo, hi = ylim
    n = len(vals)
    for k, ((fmt, corpus), y) in enumerate(vals):
        x = si + (k - (n - 1) / 2) * 0.014
        deg = (fmt, corpus) in stt.DEGENERATE
        m = marker_deg if deg else "o"
        if y < lo:
            ax.plot([x], [lo], marker="v", ms=5, color=color, alpha=0.75, zorder=2, clip_on=False)
        elif y > hi:
            ax.plot([x], [hi], marker="^", ms=5, color=color, alpha=0.75, zorder=2, clip_on=False)
        else:
            ax.plot(
                [x],
                [y],
                marker=m,
                ms=4.2 if not deg else 5.0,
                color=color,
                alpha=0.75 if deg else 0.35,
                mew=1.2 if deg else 0,
                ls="none",
                zorder=2,
            )


def fig_aggregate_with_cross(D: dict) -> Path:
    """The committed A/B panels, plus the cross-map series."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    xs = np.arange(len(stt.STEPS))

    for tier in stt.TIERS:
        mA, mB, xa = [], [], []
        for si in range(len(stt.STEPS)):
            vals = D["data"][(si, tier)]
            if not vals:
                continue
            keep = [(k, c) for k, c in vals if k not in stt.DEGENERATE]
            xa.append(si)
            mA.append(float(np.median([c["r2"] for _, c in keep])))
            mB.append(float(np.median([c["within_r2"] - c["r2"] for _, c in keep])))
            _draw_points(
                axA, si, [(k, c["r2"]) for k, c in vals], stt.A_YLIM, stt.TIER_COLORS[tier]
            )
            _draw_points(
                axB,
                si,
                [(k, c["within_r2"] - c["r2"]) for k, c in vals],
                stt.B_YLIM,
                stt.TIER_COLORS[tier],
            )
        style = dict(color=stt.TIER_COLORS[tier], lw=2.0, zorder=3)
        lab = f"{tier}: {stt.TIER_LABEL[tier]}"
        axA.plot(xa, mA, marker="o", ms=6, label=lab, **style)
        axB.plot(xa, mB, marker="o", ms=6, label=lab, **style)

    # ---- the cross map: one value per (step, surface), not per tier ----------
    cA, cB, cx = [], [], []
    for si, (src, tgt) in enumerate(stt.STEPS):
        vals, gaps = [], []
        for fmt, corpus in stt.SURFACES:
            r2 = cross_at(src, tgt, fmt, corpus)
            c0 = stt.cell(src, tgt, fmt, corpus, stt.TIERS[0])
            vals.append(((fmt, corpus), r2))
            if c0 is not None:
                gaps.append(((fmt, corpus), c0["within_r2"] - r2))
        keep = [v for k, v in vals if k not in stt.DEGENERATE]
        keep_g = [v for k, v in gaps if k not in stt.DEGENERATE]
        cx.append(si)
        cA.append(float(np.median(keep)))
        cB.append(float(np.median(keep_g)))
        _draw_points(axA, si, vals, stt.A_YLIM, CROSS_COLOR)
        _draw_points(axB, si, gaps, stt.B_YLIM, CROSS_COLOR)
    cross_style = dict(color=CROSS_COLOR, lw=2.2, ls=(0, (5, 1.6)), zorder=5)
    axA.plot(cx, cA, marker="D", ms=6, label=CROSS_LABEL, **cross_style)
    axB.plot(cx, cB, marker="D", ms=6, label=CROSS_LABEL, **cross_style)

    ceil = [D["ceilings"][si] for si in range(len(stt.STEPS))]
    axA.plot(
        xs,
        ceil,
        marker="s",
        ms=5,
        ls="--",
        lw=1.6,
        color=stt.CEILING_COLOR,
        label="within-model ceiling",
        zorder=4,
    )
    axB.axhline(0.0, ls="--", lw=1.2, color=stt.CEILING_COLOR, zorder=1)
    axB.axhspan(
        0.0,
        mlp.BAND,
        color=stt.CEILING_COLOR,
        alpha=0.12,
        zorder=0,
        label=f"elicitation band ({mlp.BAND:.3f})",
    )

    axA.set_ylabel("held-out R²  (raw pooled, layer 30)", fontsize=10)
    axA.set_title(
        "A. How much of the target's answer state is recoverable\n"
        "median over 7 non-degenerate corpora; every corpus shown as a dot",
        fontsize=10.5,
        loc="left",
    )
    axB.set_ylabel("gap to within-model ceiling  (R² points)", fontsize=10)
    axB.set_title(
        "B. How much is still missing\n"
        "gsm8k_test1319 excluded (n_train≈1034 < d=4096, estimator-degenerate)",
        fontsize=10.5,
        loc="left",
    )
    axA.set_ylim(*stt.A_YLIM)
    axB.set_ylim(*stt.B_YLIM)
    for ax in (axA, axB):
        stt._xticks(ax)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_xlabel("adjacent ladder step", fontsize=10)
        ax.legend(fontsize=7.6, framealpha=0.9)
    fig.tight_layout()
    out = OUTDIR / "ladder_step_transfer_by_tier_crossmap.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def fig_by_dataset() -> Path:
    """One subplot per EVAL DATASET: 4 tiers + the cross map + the ceiling."""
    fig, axes = plt.subplots(2, 4, figsize=(15.4, 7.0), sharex=True, sharey=True)
    for k, (fmt, corpus) in enumerate(stt.SURFACES):
        ax = axes[k // 4, k % 4]
        for tier in stt.TIERS:
            xa, ys, lo, hi = [], [], [], []
            for si, (src, tgt) in enumerate(stt.STEPS):
                c = stt.cell(src, tgt, fmt, corpus, tier)
                if c is None:
                    continue
                xa.append(si)
                ys.append(c["r2"])
                lo.append(c["r2"] - c["r2_lo"] if c["has_ci"] else 0.0)
                hi.append(c["r2_hi"] - c["r2"] if c["has_ci"] else 0.0)
            if not xa:
                continue
            ax.errorbar(
                xa,
                ys,
                yerr=[lo, hi],
                marker="o",
                ms=4.5,
                lw=1.6,
                capsize=2.0,
                color=stt.TIER_COLORS[tier],
                label=f"{tier}: {stt.TIER_LABEL[tier]}",
            )
            for si, y in zip(xa, ys):
                if y < stt.PC_YLIM[0]:
                    ax.plot(
                        [si],
                        [stt.PC_YLIM[0]],
                        marker="v",
                        ms=6,
                        color=stt.TIER_COLORS[tier],
                        zorder=6,
                        clip_on=False,
                    )
                # open marker on the point-only step so it cannot read as CI==0
                if stt.STEPS[si] in stt.SELFMAP_STEPS:
                    ax.plot(
                        [si],
                        [y],
                        marker="o",
                        ms=8,
                        mfc="none",
                        mec=stt.TIER_COLORS[tier],
                        mew=1.4,
                        zorder=5,
                    )

        cys = [cross_at(s, t, fmt, corpus) for s, t in stt.STEPS]
        ax.plot(
            range(len(stt.STEPS)),
            cys,
            marker="D",
            ms=5,
            lw=2.0,
            ls=(0, (5, 1.6)),
            color=CROSS_COLOR,
            label=CROSS_LABEL,
            zorder=7,
        )

        cvals = [stt.cell(s, t, fmt, corpus, stt.TIERS[0]) for s, t in stt.STEPS]
        ax.plot(
            [i for i, c in enumerate(cvals) if c],
            [c["within_r2"] for c in cvals if c],
            marker="s",
            ms=4,
            ls="--",
            lw=1.3,
            color=stt.CEILING_COLOR,
            label="within-model ceiling",
        )

        deg = (fmt, corpus) in stt.DEGENERATE
        title = f"{corpus} ({fmt})" + ("  — DEGENERATE n_train<d" if deg else "")
        ax.set_title(title, fontsize=9.5, loc="left", color="#8c2d04" if deg else "black")
        if deg:
            ax.set_facecolor("#fdf2e9")
        ax.grid(axis="y", alpha=0.22, lw=0.6)
        stt._xticks(ax)

    axes[0, 0].set_ylim(*stt.PC_YLIM)  # sharey => applies to all 8 panels
    axes[0, 0].set_ylabel("held-out R²", fontsize=10)
    axes[1, 0].set_ylabel("held-out R²", fontsize=10)
    fig.suptitle(
        "Per eval dataset: transfer at each reparameterization tier, "
        "against a map fitted directly from source contexts to target answers",
        fontsize=11.5,
        x=0.008,
        ha="left",
    )
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        h,
        lab,
        loc="lower center",
        ncol=6,
        fontsize=8.6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.004),
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    out = OUTDIR / "ladder_step_transfer_by_dataset.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def write_meta(D: dict) -> None:
    rows = []
    for si, (src, tgt) in enumerate(stt.STEPS):
        for fmt, corpus in stt.SURFACES:
            c0 = stt.cell(src, tgt, fmt, corpus, stt.TIERS[0])
            key = (f"{src}__{tgt}", fmt, corpus)
            rows.append(
                {
                    "step": stt.step_label(src, tgt).replace("\n", " "),
                    "pair": f"{src}__{tgt}",
                    "format": fmt,
                    "corpus": corpus,
                    "cross_r2": CROSS[key],
                    "cross_source": CROSS_PROV[key],
                    "within_r2": c0["within_r2"] if c0 else None,
                    "t0_r2": c0["r2"] if c0 else None,
                    "degenerate": (fmt, corpus) in stt.DEGENERATE,
                }
            )
    meta = {
        "issue": 1336,
        "question": (
            "per adjacent ladder step, how much of the context->answer map transfers under "
            "each reparameterization tier, and how much of the target's answer state is "
            "recoverable from the source's context state by a map fitted directly"
        ),
        "layer": stt.LAYER,
        "scale": stt.SCALE,
        "extends": "ladder_step_transfer_by_tier.png (scripts/issue1336_step_transfer_tiers.py)",
        "cross_map_definition": (
            "ridge fitted DIRECTLY from X = v_context(source) to Y = v_answer(target), scored "
            "on the target's held-out answers. NOT a tier: the tiers all reuse the source's "
            "fitted operator W_s, so a low tier read conflates a CHANGED MAP with MOVED "
            "REPRESENTATIONS; the cross map bounds the second. A per-CELL constant (one value "
            "per step x surface, not one per tier), and NOT bounded above by the ceiling."
        ),
        "cross_map_sources": {
            "base__sft / sft__dpo / dpo__rlvr": (
                "repswap_r2 at per_layer/30 of the round-3 metric_ladder pair files"
            ),
            "rlvr__rlvr_long": "cross_r2 in the selfmap_v3 cells",
            "basis_note": (
                "both are layer 30, fit_seed 0, 5 outer folds, fold-local pooled OOF, raw "
                "scale — the same basis as the tier lines, which is why they share an axis"
            ),
        },
        "tier_labels": {str(t): stt.TIER_LABEL[t] for t in stt.TIERS},
        "steps": [f"{s}__{t}" for s, t in stt.STEPS],
        "surfaces": [f"{c} ({f})" for f, c in stt.SURFACES],
        "ci_note": (
            "tier R² intervals are the 1,000-draw paired-bootstrap GAP CI mapped through "
            "r2 = within_r2 - gap. The cross map carries NO bootstrap draws in either source, "
            "so it is POINT-ONLY on every step and is never given a borrowed interval. "
            "rlvr->longer RLVR is point-only for the tiers too (open markers)."
        ),
        "degenerate_excluded": {
            "surfaces": ["gsm8k_test1319 (chat)"],
            "reason": (
                "n_train (~4/5 of n) < d=4096 => estimator-degenerate, not a signal read. "
                "Excluded from every aggregate; MARKED (shaded panel + 'x' markers) in the "
                "per-dataset view, never silently dropped."
            ),
        },
        "figures": [
            "ladder_step_transfer_by_tier_crossmap.png",
            "ladder_step_transfer_by_dataset.png",
        ],
        "code": "scripts/issue1336_step_transfer_with_crossmap.py",
        "rows": rows,
    }
    for name in ("ladder_step_transfer_by_tier_crossmap", "ladder_step_transfer_by_dataset"):
        (OUTDIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    D = stt.collect()
    paths = [fig_aggregate_with_cross(D), fig_by_dataset()]
    write_meta(D)

    print(
        "\n[cross-map] MEDIAN R² by step (raw, layer 30; 7 non-degenerate corpora)\n"
        "  tiers reuse the source's fitted operator; cross refits source ctx -> target answer"
    )
    hdr = "  series  " + "".join(
        f"{stt.step_label(s, t).replace(chr(10), ' '):>18}" for s, t in stt.STEPS
    )
    print(hdr)
    for tier in stt.TIERS:
        cells_ = []
        for si in range(len(stt.STEPS)):
            v = [c for k, c in D["data"][(si, tier)] if k not in stt.DEGENERATE]
            cells_.append(f"{np.median([c['r2'] for c in v]):>18.4f}" if v else f"{'—':>18}")
        print(f"  t{tier}      " + "".join(cells_))
    crossrow = []
    for src, tgt in stt.STEPS:
        v = [cross_at(src, tgt, f, c) for f, c in stt.SURFACES if (f, c) not in stt.DEGENERATE]
        crossrow.append(f"{np.median(v):>18.4f}")
    print("  cross   " + "".join(crossrow))
    print("  ceil    " + "".join(f"{D['ceilings'][si]:>18.4f}" for si in range(len(stt.STEPS))))
    for p in paths:
        print(f"[cross-map] wrote {p}")


if __name__ == "__main__":
    main()

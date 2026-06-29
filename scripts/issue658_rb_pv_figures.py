# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, →, ×, Δ) in scientific docstrings + figure labels.
"""Issue #658 persona-vectors-style r_B — figures (Q1-Q5; over-produce, hero TBD).

Reads the fit phase's ``aggregate.json`` + ``per_behavior_genre.json``
(``issue658_rb_pv_fit.py``) and emits the plan §6.5 figure set. The analyzer
picks the hero from the over-produced dump (plan §6.5):

- **Hero 1** ``fig_rb_best_cell_rho``: per-behavior best-cell ρ — PV (pos-vs-neg)
  vs PV (pos-vs-neutral) vs #658 corpus-mismatched vs label-split, with the
  noise-floor band, per genre (the takeaway-rewrite figure).
- **Hero 2** ``fig_rb_layer_profile``: per-behavior ρ-vs-layer (all 28 layers),
  with the paper's mid-stack steering band annotated (Q2/Q3).
- **Exploratory** ``fig_rb_pooled_vs_single`` (Q4 scatter), ``fig_rb_pole_compare``
  (Q5 bars), ``fig_rb_yield`` (judge-yield-per-behavior table-as-bars).

All figures are robust to missing cells (a descoped reduction / a below-floor
behavior simply does not contribute a bar) — never crashes on a partial run.

    uv run python scripts/issue658_rb_pv_figures.py \\
        --fit-dir eval_results/issue_658/persona-vectors-style-rb \\
        --out figures/issue_658
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

logger = logging.getLogger("issue658_rb_pv_figures")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PV_BEHAVIORS = ["broad_em", "harmful_compliance", "sycophancy", "refusal"]
LABELS = {
    "broad_em": "broad misalignment",
    "harmful_compliance": "harmful compliance",
    "sycophancy": "sycophancy",
    "refusal": "refusal",
}
# The paper's steering-best for Qwen2.5-7B-Instruct is mid-stack (a layer sweep,
# no single integer — plan §11). Annotate the central 1/3 band as the cross-check.
PAPER_MIDSTACK_BAND = (10, 18)


def _rows_by(agg: dict) -> dict[tuple[str, str], dict]:
    return {(r["behavior"], r["genre"]): r for r in agg.get("rows", [])}


def fig_rb_best_cell_rho(agg: dict, out: str) -> None:
    """Hero 1: per-behavior best-cell ρ + noise-floor band, per genre."""
    rows = _rows_by(agg)
    genres = sorted({r["genre"] for r in agg.get("rows", [])})
    for genre in genres:
        set_paper_style("blog")
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        behs = [b for b in PV_BEHAVIORS if (b, genre) in rows]
        if not behs:
            plt.close(fig)
            continue
        x = np.arange(len(behs))
        best = [rows[(b, genre)].get("best_rho") or 0.0 for b in behs]
        ci_lo = [(rows[(b, genre)].get("selection_aware_ci") or {}).get("lower") for b in behs]
        ci_hi = [(rows[(b, genre)].get("selection_aware_ci") or {}).get("upper") for b in behs]
        # error bars from the selection-aware CI (asymmetric)
        yerr_lo = [
            max(0.0, (b - lo)) if lo is not None else 0.0 for b, lo in zip(best, ci_lo, strict=True)
        ]
        yerr_hi = [
            max(0.0, (hi - b)) if hi is not None else 0.0 for b, hi in zip(best, ci_hi, strict=True)
        ]
        ax.bar(x, best, 0.55, color=paper_palette_role("primary"), label="PV r_B best-cell ρ")
        ax.errorbar(x, best, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="0.25", capsize=3, lw=1.0)
        # noise-floor band per behavior
        for i, b in enumerate(behs):
            nf = rows[(b, genre)].get("noise_floor_p95")
            if nf is not None:
                ax.hlines(nf, i - 0.3, i + 0.3, color=paper_palette_role("control"), lw=1.6)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[b] for b in behs], rotation=12, ha="right")
        ax.set_ylabel("best-cell read-out ρ (E0 ≈ r_Bᵀ v0)")
        ax.axhline(0.0, color="0.75", lw=0.8)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        set_title_subtitle(
            ax,
            f"Faithful PV r_B best-cell read-out ρ — {genre}",
            "bars = best (pole×reduction×layer) ρ; whiskers = Approach-B selection-aware 95% CI; "
            "horizontal mark = per-behavior noise floor",
            source=f"issue #658 · persona-vectors-style r_B · {genre} · selection-aware CI",
        )
        fig.tight_layout()
        savefig_paper(fig, f"{out}/fig_rb_best_cell_rho_{genre}", dir="figures/")
        plt.close(fig)


def fig_rb_layer_profile(pbg: dict, out: str) -> None:
    """Hero 2: per-behavior ρ-vs-layer (all layers) + the paper mid-stack band."""
    rows = pbg.get("per_behavior_genre", [])
    genres = sorted({r["genre"] for r in rows})
    for genre in genres:
        set_paper_style("blog")
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        plotted = False
        for r in rows:
            if r["genre"] != genre:
                continue
            prof = r.get("layer_profile", [])
            if not prof:
                continue
            layers = [p["layer"] for p in prof if p["rho"] is not None]
            rhos = [p["rho"] for p in prof if p["rho"] is not None]
            if not layers:
                continue
            ax.plot(
                layers,
                rhos,
                marker="o",
                ms=3,
                lw=1.2,
                label=LABELS.get(r["behavior"], r["behavior"]),
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.axvspan(*PAPER_MIDSTACK_BAND, color="0.85", alpha=0.5, label="paper mid-stack band")
        ax.set_xlabel("layer")
        ax.set_ylabel("read-out ρ (pos-vs-neg, diffmeans)")
        ax.axhline(0.0, color="0.75", lw=0.8)
        ax.legend(frameon=False, fontsize=8)
        set_title_subtitle(
            ax,
            f"Read-out ρ across layers — {genre}",
            "the predictive-best layer per behavior (Q2/Q3); shaded = the paper's mid-stack "
            "steering band (layers 10-18)",
            source=f"issue #658 · persona-vectors-style r_B · {genre}",
        )
        fig.tight_layout()
        savefig_paper(fig, f"{out}/fig_rb_layer_profile_{genre}", dir="figures/")
        plt.close(fig)


def fig_rb_pooled_vs_single(agg: dict, out: str) -> None:
    """Exploratory (Q4): Δρ(pooled − single-best) per (behavior, genre), CI whiskers."""
    rows = agg.get("rows", [])
    labels, deltas, los, his = [], [], [], []
    for r in rows:
        d = (r.get("delta_rho") or {}).get("pooled_minus_single_best") or {}
        if d.get("delta_rho") is None:
            continue
        labels.append(f"{LABELS.get(r['behavior'], r['behavior'])}\n{r['genre']}")
        deltas.append(d["delta_rho"])
        ci = d.get("selection_aware_ci") or {}
        los.append(d["delta_rho"] - ci.get("lower", d["delta_rho"]))
        his.append(ci.get("upper", d["delta_rho"]) - d["delta_rho"])
    if not labels:
        return
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, deltas, 0.55, color=paper_palette_role("accent"))
    ax.errorbar(x, deltas, yerr=[los, his], fmt="none", ecolor="0.25", capsize=3, lw=1.0)
    ax.axhline(0.0, color="0.4", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Δρ (multi-layer-pooled − single-best layer)")
    set_title_subtitle(
        ax,
        "Q4: does pooling r_B across the central-1/3 band beat the single best layer?",
        "Δρ > 0 with a CI excluding 0 = pooling wins; whiskers = Approach-B selection-aware 95% CI "
        "(both sides re-selected inside each resample)",
        source="issue #658 · pooled band layers 10-18 · same LOCO folds",
    )
    fig.tight_layout()
    savefig_paper(fig, f"{out}/fig_rb_pooled_vs_single", dir="figures/")
    plt.close(fig)


def fig_rb_pole_compare(agg: dict, out: str) -> None:
    """Exploratory (Q5): Δρ(pos-vs-neg − pos-vs-neutral) per (behavior, genre)."""
    rows = agg.get("rows", [])
    labels, deltas, los, his = [], [], [], []
    for r in rows:
        d = (r.get("delta_rho") or {}).get("pos_neg_minus_pos_neutral") or {}
        if d.get("delta_rho") is None:
            continue
        labels.append(f"{LABELS.get(r['behavior'], r['behavior'])}\n{r['genre']}")
        deltas.append(d["delta_rho"])
        ci = d.get("selection_aware_ci") or {}
        los.append(d["delta_rho"] - ci.get("lower", d["delta_rho"]))
        his.append(ci.get("upper", d["delta_rho"]) - d["delta_rho"])
    if not labels:
        return
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, deltas, 0.55, color=paper_palette_role("primary"))
    ax.errorbar(x, deltas, yerr=[los, his], fmt="none", ecolor="0.25", capsize=3, lw=1.0)
    ax.axhline(0.0, color="0.4", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Δρ (pos-vs-neg − pos-vs-neutral)")
    set_title_subtitle(
        ax,
        "Q5: faithful symmetric contrast vs default-assistant negative",
        "Δρ > 0 = the symmetric pos-vs-neg pole reads better; whiskers = selection-aware 95% CI",
        source="issue #658 · pole comparison · same LOCO folds",
    )
    fig.tight_layout()
    savefig_paper(fig, f"{out}/fig_rb_pole_compare", dir="figures/")
    plt.close(fig)


def fig_rb_yield(agg: dict, out: str) -> None:
    """Exploratory: judge-yield per behavior (kept pos/neg/neutral rollout counts)."""
    manifest_rows = agg.get("rows", [])
    # yield is per-behavior (genre-invariant — the rollouts are not genre-keyed);
    # collect the unique behavior yields.
    seen: dict[str, dict] = {}
    for r in manifest_rows:
        y = r.get("yield")
        if y and r["behavior"] not in seen:
            seen[r["behavior"]] = y
    if not seen:
        return
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    behs = [b for b in PV_BEHAVIORS if b in seen]
    x = np.arange(len(behs))
    w = 0.27
    kept_pos = [seen[b]["kept_pos"] for b in behs]
    kept_neg = [seen[b]["kept_neg"] for b in behs]
    kept_neu = [seen[b]["kept_neutral"] for b in behs]
    ax.bar(x - w, kept_pos, w, color=paper_palette_role("primary"), label="kept pos")
    ax.bar(x, kept_neg, w, color=paper_palette_role("accent"), label="kept neg")
    ax.bar(x + w, kept_neu, w, color=paper_palette_role("neutral"), label="kept neutral")
    # the yield floor line (per behavior; same target_pos)
    floor = seen[behs[0]].get("yield_floor_n")
    if floor is not None:
        ax.axhline(floor, color=paper_palette_role("control"), lw=1.2, ls="--", label="yield floor")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[b] for b in behs], rotation=12, ha="right")
    ax.set_ylabel("judge-kept rollout count")
    ax.legend(frameon=False, fontsize=8)
    set_title_subtitle(
        ax,
        "Judge-filter yield per behavior",
        "kept rollouts after the Sonnet-4.5 trait-eval filter (pos>50 / neg<50 / neutral<50); "
        "dashed = 80% yield floor",
        source="issue #658 · persona-vectors-style r_B",
    )
    fig.tight_layout()
    savefig_paper(fig, f"{out}/fig_rb_yield", dir="figures/")
    plt.close(fig)


def build_all(fit_dir: Path, out: str) -> list[str]:
    with open(fit_dir / "aggregate.json") as f:
        agg = json.load(f)
    pbg_path = fit_dir / "per_behavior_genre.json"
    if pbg_path.is_file():
        with open(pbg_path) as f:
            pbg = json.load(f)
    else:
        pbg = {"per_behavior_genre": []}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "figures" / out).mkdir(parents=True, exist_ok=True)
    fig_rb_best_cell_rho(agg, out)
    fig_rb_layer_profile(pbg, out)
    fig_rb_pooled_vs_single(agg, out)
    fig_rb_pole_compare(agg, out)
    fig_rb_yield(agg, out)
    produced = sorted(str(p) for p in (PROJECT_ROOT / "figures" / out).glob("fig_rb_*.png"))
    logger.info("wrote %d PV r_B figures under figures/%s", len(produced), out)
    return produced


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #658 persona-vectors-style r_B figures.")
    ap.add_argument(
        "--fit-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_658/persona-vectors-style-rb",
    )
    ap.add_argument("--out", default="issue_658", help="figures/<out>/ subdir")
    args = ap.parse_args()
    produced = build_all(args.fit_dir, args.out)
    for p in produced:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

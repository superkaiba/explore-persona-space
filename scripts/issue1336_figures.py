#!/usr/bin/env python
"""Issue #1336 — Phase P: figures from the committed eval JSONs (VM CPU).

Hero (plan §6): per-stage paired bars (within-stage R^2 vs reparameterized-
base R^2) with a gap panel (95% CIs), one group per eval set, headline layer.
Exploratory dump: full layer curves with null bands; gap across all frozen
layers; adjacent-stage increments; rep-swap ceilings; alignment R^2 per
stage; orthogonal-vs-linear composition bars; NLL companions; keep-rate
panel; matched-n comparability; prefix-slot degeneracy; RLVR-long dose
extension (folded into the stage lists wherever the arm's files exist).

Inputs: eval_results/issue_1336/{cells,ladder_alignment,decision,gen_audits}.
Outputs: --fig-dir (default figures/issue_1336; smokes MUST redirect via
--fig-dir to a scratch dir — committed figure paths are never smoke targets).
"""

from __future__ import annotations

import argparse
import json
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

from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style  # noqa: E402
from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

STAGE_LABELS = {m: cm.MODELS[m]["label"] for m in cm.MODELS}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", type=Path, default=Path("eval_results/issue_1336"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_1336"))
    ap.add_argument("--smoke", action="store_true", help="tolerate missing optional arms")
    ap.add_argument(
        "--v2",
        action="store_true",
        help="full-corpora follow-up figures (metric_ladder + cells_v2 + decision_v2 "
        "inputs; *_v2.png outputs). Default OFF: the v1 path is byte-unchanged.",
    )
    ap.add_argument(
        "--v3",
        action="store_true",
        help="round-4 pooled/off-policy figures (decision_v3 inputs; 2x2_panel_*.png + "
        "delta_q_*.png outputs). Default OFF: the v1/v2 paths are byte-unchanged.",
    )
    return ap.parse_args()


def _load(path: Path) -> dict:
    assert path.exists(), f"missing figure input: {path}"
    return json.loads(path.read_text())


def _maybe(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _save(fig, fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / name
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[figs1336] wrote {out}")


def _pairs_available(align_dir: Path) -> list[dict]:
    out = []
    for path in sorted(align_dir.glob("pair_*.json")):
        out.append(_load(path))
    assert out, f"no pair_*.json under {align_dir}"
    return out


def _eval_set_key(p: dict) -> str:
    return f"{p['eval_set']['corpus']}_{p['eval_set']['format']}"


def fig_hero(pairs: list[dict], decision: dict, fig_dir: Path) -> None:
    """Per-stage paired bars (within vs reparam) + gap panel, per eval set."""
    headline = decision["headline_layer"]
    base_pairs = [p for p in pairs if p["pair"]["m0"] == "base"]
    eval_sets = sorted({_eval_set_key(p) for p in base_pairs})
    stages = [
        m
        for m in ("sft", "dpo", "rlvr", "rlvr_long")
        if any(p["pair"]["m1"] == m for p in base_pairs)
    ]
    fig, axes = plt.subplots(
        2, len(eval_sets), figsize=(4.2 * len(eval_sets), 6.4), sharey="row", squeeze=False
    )
    colors = paper_palette(2)
    for j, es in enumerate(eval_sets):
        ax_top, ax_bot = axes[0][j], axes[1][j]
        within, comp, gaps, glo, ghi = [], [], [], [], []
        for k in stages:
            match = [p for p in base_pairs if p["pair"]["m1"] == k and _eval_set_key(p) == es]
            if not match:
                within.append(np.nan), comp.append(np.nan)
                gaps.append(np.nan), glo.append(0.0), ghi.append(0.0)
                continue
            pl = match[0]["per_layer"][str(headline)]
            within.append(pl["within_r2"])
            comp.append(pl["comp_samefn_r2"])
            gaps.append(pl["gap"])
            glo.append(pl["gap"] - pl["gap_bootstrap"]["ci_lo"])
            ghi.append(pl["gap_bootstrap"]["ci_hi"] - pl["gap"])
        x = np.arange(len(stages))
        ax_top.bar(x - 0.2, within, width=0.4, color=colors[0], label="within-stage R²")
        ax_top.bar(x + 0.2, comp, width=0.4, color=colors[1], label="reparam. base R²")
        ax_top.set_xticks(x, [STAGE_LABELS[s] for s in stages], rotation=20, ha="right")
        ax_top.set_title(es)
        yerr = np.vstack([np.maximum(0.0, glo), np.maximum(0.0, ghi)])
        ax_bot.errorbar(x, gaps, yerr=yerr, fmt="o", color=colors[0], capsize=3)
        ax_bot.axhline(0.0, lw=0.8, color="0.4")
        ax_bot.set_xticks(x, [STAGE_LABELS[s] for s in stages], rotation=20, ha="right")
        if j == 0:
            ax_top.set_ylabel("held-out pooled R²")
            ax_bot.set_ylabel("reparameterization gap")
            ax_top.legend(fontsize=7)
    fig.suptitle(f"Within-stage vs reparameterized-base map (headline layer {headline})")
    _save(fig, fig_dir, "hero_gap_bars.png")


def fig_layer_curves(cells_dir: Path, fig_dir: Path, smoke: bool) -> None:
    for corpus, fmt in cm.EVAL_SETS:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        models = [m for m in cm.MODELS]
        colors = paper_palette(len(models))
        plotted = False
        for c, m in zip(colors, models, strict=True):
            cell = cm.cell_id(m, fmt, corpus)
            payload = _maybe(cells_dir / f"cells_{cell}.json")
            if payload is None:
                # rlvr_long (secondary arm) + the naturalistic format are the
                # plan's descope-priority arms — tolerate their absence.
                optional = m == "rlvr_long" or fmt == "naturalistic"
                if not (smoke or optional):
                    raise FileNotFoundError(f"cells_{cell}.json missing for layer curves")
                print(f"[figs1336] optional cell {cell} missing — skipped in layer curves")
                continue
            r2 = payload["r2_per_layer_obs"]
            ax.plot(range(len(r2)), r2, label=STAGE_LABELS[m], color=c, lw=1.4)
            nulls = _maybe(cells_dir / f"nulls_{cell}.json")
            if nulls is not None and nulls["null_matrix"]:
                nm = np.asarray(nulls["null_matrix"], dtype=float)
                ax.fill_between(
                    range(nm.shape[1]),
                    np.nanquantile(nm, 0.025, axis=0),
                    np.nanquantile(nm, 0.975, axis=0),
                    color=c,
                    alpha=0.12,
                    lw=0,
                )
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out pooled R²")
        ax.set_title(f"within-stage map strength — {corpus} / {fmt} (bands: shuffle nulls)")
        ax.legend(fontsize=7)
        _save(fig, fig_dir, f"layer_curves_{corpus}_{fmt}.png")


def fig_gap_frozen(pairs: list[dict], fig_dir: Path) -> None:
    base_pairs = [p for p in pairs if p["pair"]["m0"] == "base"]
    eval_sets = sorted({_eval_set_key(p) for p in base_pairs})
    fig, axes = plt.subplots(
        1, len(eval_sets), figsize=(4.0 * len(eval_sets), 3.4), sharey=True, squeeze=False
    )
    for j, es in enumerate(eval_sets):
        ax = axes[0][j]
        sel = [p for p in base_pairs if _eval_set_key(p) == es]
        colors = paper_palette(len(sel))
        for c, p in zip(colors, sel, strict=True):
            layers = sorted(int(li) for li in p["per_layer"])
            gaps = [p["per_layer"][str(li)]["gap"] for li in layers]
            ax.plot(layers, gaps, marker="o", label=f"base->{p['pair']['m1']}", color=c)
        ax.axhline(0.0, lw=0.8, color="0.4")
        ax.set_title(es)
        ax.set_xlabel("frozen layer")
        if j == 0:
            ax.set_ylabel("gap")
            ax.legend(fontsize=7)
    fig.suptitle("Reparameterization gap across frozen layers")
    _save(fig, fig_dir, "gap_frozen_layers.png")


def fig_increments(decision: dict, fig_dir: Path) -> None:
    per_set = decision["per_eval_set"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = paper_palette(len(per_set))
    width = 0.8 / max(1, len(per_set))
    labels = None
    for j, (es, payload) in enumerate(sorted(per_set.items())):
        incs = payload["adjacent_increments"]
        labels = list(incs)
        x = np.arange(len(incs)) + j * width
        pts = [incs[k]["point"] for k in incs]
        lo = [max(0.0, incs[k]["point"] - incs[k]["ci_lo"]) for k in incs]
        hi = [max(0.0, incs[k]["ci_hi"] - incs[k]["point"]) for k in incs]
        ax.bar(x, pts, width=width, yerr=np.vstack([lo, hi]), capsize=2, label=es, color=colors[j])
    ax.axhline(0.0, lw=0.8, color="0.4")
    ax.set_xticks(np.arange(len(labels)) + 0.4 - width / 2, labels, rotation=15, ha="right")
    ax.set_ylabel("gap increment")
    ax.set_title("Adjacent-stage gap increments (95% CI)")
    ax.legend(fontsize=7)
    _save(fig, fig_dir, "adjacent_increments.png")


def fig_ceilings_alignment_orth(pairs: list[dict], decision: dict, fig_dir: Path) -> None:
    headline = str(decision["headline_layer"])
    base_pairs = [p for p in pairs if p["pair"]["m0"] == "base"]
    # rep-swap ceilings + alignment R^2 per pair (headline layer)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    names = [f"{p['pair']['m1']}|{_eval_set_key(p)}" for p in base_pairs]
    reps = [p["per_layer"][headline]["battery"]["ceilings"]["repswap_b2i"] for p in base_pairs]
    a_ctx = [
        p["per_layer"][headline]["battery"]["alignment_r2"]["linear"]["A_ctx"] for p in base_pairs
    ]
    a_ans = [
        p["per_layer"][headline]["battery"]["alignment_r2"]["linear"]["A_ans"] for p in base_pairs
    ]
    x = np.arange(len(names))
    colors = paper_palette(3)
    axes[0].bar(x, reps, color=colors[0])
    axes[0].set_title("rep-swap ceiling (Xb->Yk)")
    axes[1].bar(x - 0.2, a_ctx, width=0.4, color=colors[1], label="A_ctx")
    axes[1].bar(x + 0.2, a_ans, width=0.4, color=colors[2], label="A_ans")
    axes[1].set_title("alignment-map held-out R²")
    axes[1].legend(fontsize=7)
    # orthogonal vs linear composition at headline layer
    comp_lin = [
        p["per_layer"][headline]["battery"]["composition"]["linear"]["comp_samefn_b2i"]
        for p in base_pairs
    ]
    comp_orth = [
        p["per_layer"][headline]["battery"]["composition"]
        .get("orthogonal", {})
        .get("comp_samefn_b2i", np.nan)
        for p in base_pairs
    ]
    axes[2].bar(x - 0.2, comp_lin, width=0.4, color=colors[1], label="general linear")
    axes[2].bar(x + 0.2, comp_orth, width=0.4, color=colors[2], label="orthogonal")
    axes[2].set_title("comp_samefn: linear vs orthogonal")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x, names, rotation=45, ha="right", fontsize=6)
    _save(fig, fig_dir, "ceilings_alignment_orth.png")


def fig_nll_keeprate(cells_dir: Path, audits_dir: Path, fig_dir: Path, smoke: bool) -> None:
    rows = []
    for cell in cm.CELLS:
        payload = _maybe(cells_dir / f"cells_{cell['cell_id']}.json")
        if payload is None or payload.get("nll_a1") is None:
            continue
        rows.append((cell["cell_id"], payload["nll_a1"]["mean"]))
    if rows:
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        ax.bar(range(len(rows)), [r[1] for r in rows], color=paper_palette(1)[0])
        ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=60, ha="right", fontsize=6)
        ax.set_ylabel("mean teacher-forced NLL (a1)")
        ax.set_title("NLL companion per cell")
        _save(fig, fig_dir, "nll_companion.png")
    audits = sorted(audits_dir.glob("audit_*.json")) if audits_dir.exists() else []
    if not audits:
        if smoke:
            print("[figs1336] no gen audits found (smoke) — skipping keep-rate panel")
            return
        raise FileNotFoundError(f"no gen audits under {audits_dir}")
    names, rates = [], []
    for path in audits:
        a = json.loads(path.read_text())
        names.append(f"{a['model']}|{a['corpus']}")
        rates.append(a["keep_rate"])
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.bar(range(len(names)), rates, color=paper_palette(1)[0])
    ax.axhline(cm.KEEP_RATE_FLOOR, color="0.3", lw=0.9, ls="--")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("keep rate")
    ax.set_title("Generation keep rate per (model, corpus); dashed: 0.80 floor")
    _save(fig, fig_dir, "keep_rate.png")


def fig_matchedn_degeneracy(cells_dir: Path, fig_dir: Path, smoke: bool) -> None:
    pts = []
    for cell in cm.CELLS:
        full = _maybe(cells_dir / f"cells_{cell['cell_id']}.json")
        sub = _maybe(cells_dir / f"cells_matchedn_{cell['cell_id']}.json")
        if full is None or sub is None:
            continue
        fl = full["frozen_layers"]
        for li in fl:
            if li < len(full["r2_per_layer_obs"]) and li < len(sub["r2_per_layer_obs"]):
                pts.append((full["r2_per_layer_obs"][li], sub["r2_per_layer_obs"][li]))
    if pts:
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        arr = np.asarray(pts, dtype=float)
        ax.scatter(arr[:, 0], arr[:, 1], s=14, color=paper_palette(1)[0])
        lim = [min(arr.min(), 0.0), max(arr.max(), 0.1)]
        ax.plot(lim, lim, lw=0.8, color="0.4")
        ax.set_xlabel("full-n R²")
        ax.set_ylabel("matched-n R²")
        ax.set_title("Matched-n comparability (frozen layers)")
        _save(fig, fig_dir, "matched_n_comparability.png")
    else:
        print("[figs1336] no matched-n refits found — skipping comparability plot")
    vals = []
    for cell in cm.CELLS:
        payload = _maybe(cells_dir / f"cells_{cell['cell_id']}.json")
        if payload is None:
            continue
        for d in payload.get("prefix_slot_degeneracy", {}).values():
            vals.append(d["max_pairwise_cos_dist"])
    if vals:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.hist(vals, bins=30, color=paper_palette(1)[0])
        ax.set_xlabel("max pairwise cosine distance (prefix slot)")
        ax.set_ylabel("count (cell x frozen layer)")
        ax.set_title("Prefix-slot degeneracy check (expected ~0)")
        _save(fig, fig_dir, "prefix_slot_degeneracy.png")


# ===========================================================================
# v2 (full-corpora follow-up) figures — metric_ladder / cells_v2 / decision_v2
# ===========================================================================

_TIER_NAMES = tuple(f"t{k}" for k in range(9))
_TIER_XLABELS = ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
_STAGE_ORDER = ("sft", "dpo", "rlvr", "rlvr_long")


def _err_lo_hi(point: float, ci: dict) -> tuple[float, float]:
    """Non-negative errorbar OFFSETS from a {ci_lo, ci_hi} block (mpl xerr/yerr
    contract — never raw bounds, never signed deltas; gotchas.md #547/#1335)."""
    return max(0.0, point - float(ci["ci_lo"])), max(0.0, float(ci["ci_hi"]) - point)


def _pl_v2(p: dict, headline: int) -> tuple[dict, int]:
    """The pair JSON's per-layer block at the headline layer (fallback: the
    deepest available frozen layer, with a printed note — smoke ladders run a
    reduced layer set)."""
    if str(headline) in p["per_layer"]:
        return p["per_layer"][str(headline)], headline
    li = max(int(k) for k in p["per_layer"])
    print(f"[figs1336-v2] headline layer {headline} absent from pair JSON — using layer {li}")
    return p["per_layer"][str(li)], li


def fig_v2_tier_profile(pairs: list[dict], headline: int, scale: str, fig_dir: Path) -> None:
    """Hero 1: per-tier held-out R^2 profile (T0..T8) per base-anchored pair,
    one panel per surface, within-stage ceiling as a horizontal line."""
    base_pairs = [p for p in pairs if p["pair"]["m0"] == "base"]
    eval_sets = sorted({_eval_set_key(p) for p in base_pairs})
    fig, axes = plt.subplots(
        1, len(eval_sets), figsize=(4.4 * len(eval_sets), 3.8), sharey=True, squeeze=False
    )
    stage_colors = dict(zip(_STAGE_ORDER, paper_palette(len(_STAGE_ORDER)), strict=True))
    x = np.arange(len(_TIER_NAMES))
    for j, es in enumerate(eval_sets):
        ax = axes[0][j]
        for p in [q for q in base_pairs if _eval_set_key(q) == es]:
            m1 = p["pair"]["m1"]
            blk, _li = _pl_v2(p, headline)
            tiers = blk[scale]["tiers"]
            pts = [tiers[nm]["r2"] for nm in _TIER_NAMES]
            los, his = zip(
                *[_err_lo_hi(tiers[nm]["r2"], tiers[nm]["r2_bootstrap"]) for nm in _TIER_NAMES],
                strict=True,
            )
            c = stage_colors[m1]
            ax.errorbar(
                x,
                pts,
                yerr=np.vstack([los, his]),
                marker="o",
                ms=3,
                capsize=2,
                color=c,
                label=f"base->{m1}",
                lw=1.2,
            )
            ax.axhline(blk[scale]["within_r2"], color=c, lw=0.8, ls="--", alpha=0.6)
        ax.set_title(es, fontsize=8)
        ax.set_xticks(x, _TIER_XLABELS, fontsize=7)
        ax.set_xlabel("correction tier")
        if j == 0:
            ax.set_ylabel(f"held-out pooled R² ({scale})")
            ax.legend(fontsize=6)
    fig.suptitle(
        f"Metric-ladder tier profile (layer {headline}; dashed: within-stage ceiling)", fontsize=9
    )
    _save(fig, fig_dir, "hero_tier_profile_v2.png")


def fig_v2_stage_contrast(decision: dict, fig_dir: Path) -> None:
    """Hero 2: per-surface stage deltas (within − T8) + the C_v2 contrast vs
    the ±elicitation band, raw AND recal rows (primary scale marked)."""
    per_surface = decision["per_surface"]
    surfaces = sorted(per_surface)
    primary_scale = decision["primary_scale"]
    fig, axes = plt.subplots(2, 2, figsize=(4.6 * 2, 6.6), squeeze=False)
    colors = paper_palette(3)
    x = np.arange(len(surfaces))
    for r, scale in enumerate(("raw", "recal")):
        ax_d, ax_c = axes[r][0], axes[r][1]
        for k, (name, label) in enumerate(
            (("delta_dpo", "Δ(base->dpo)"), ("delta_rlvr", "Δ(base->rlvr)"))
        ):
            pts = [per_surface[s][scale][name]["point"] for s in surfaces]
            errs = [
                _err_lo_hi(per_surface[s][scale][name]["point"], per_surface[s][scale][name])
                for s in surfaces
            ]
            los, his = zip(*errs, strict=True)
            ax_d.bar(
                x + (k - 0.5) * 0.4,
                pts,
                width=0.4,
                yerr=np.vstack([los, his]),
                capsize=2,
                color=colors[k],
                label=label,
            )
        ax_d.axhline(0.0, lw=0.8, color="0.4")
        ax_d.set_xticks(x, surfaces, rotation=25, ha="right", fontsize=6)
        tag = " [PRIMARY]" if scale == primary_scale else ""
        ax_d.set_title(f"stage delta within−T8 ({scale}{tag})", fontsize=8)
        ax_d.set_ylabel("Δ R²")
        ax_d.legend(fontsize=6)
        c_pts = [per_surface[s][scale]["C_v2"]["point"] for s in surfaces]
        c_errs = [
            _err_lo_hi(per_surface[s][scale]["C_v2"]["point"], per_surface[s][scale]["C_v2"])
            for s in surfaces
        ]
        los, his = zip(*c_errs, strict=True)
        band = float(per_surface[surfaces[0]]["band"])
        ax_c.axhspan(-band, band, color="0.85", zorder=0)
        ax_c.errorbar(x, c_pts, yerr=np.vstack([los, his]), fmt="o", capsize=3, color=colors[2])
        ax_c.axhline(0.0, lw=0.8, color="0.4")
        ax_c.set_xticks(x, surfaces, rotation=25, ha="right", fontsize=6)
        ax_c.set_title(f"C_v2 = Δrlvr − Δdpo vs ±band ({scale}{tag})", fontsize=8)
    fig.suptitle(
        f"Stage contrast at layer {decision['headline_layer']} "
        f"(verdict: {decision['verdict_lattice']['verdict']}, scale {primary_scale})",
        fontsize=9,
    )
    _save(fig, fig_dir, "hero_stage_contrast_v2.png")


def fig_v2_adjacent_increments(pairs: list[dict], headline: int, scale: str, fig_dir: Path) -> None:
    base_pairs = [p for p in pairs if p["pair"]["m0"] == "base"]
    eval_sets = sorted({_eval_set_key(p) for p in base_pairs})
    fig, axes = plt.subplots(
        1, len(eval_sets), figsize=(4.6 * len(eval_sets), 3.6), sharey=True, squeeze=False
    )
    inc_names = [f"t{t - 1}->t{t}" for t in range(1, 9)]
    for j, es in enumerate(eval_sets):
        ax = axes[0][j]
        sel = [p for p in base_pairs if _eval_set_key(p) == es]
        colors = paper_palette(max(1, len(sel)))
        width = 0.8 / max(1, len(sel))
        for k, p in enumerate(sel):
            blk, _li = _pl_v2(p, headline)
            incs = blk[scale]["tier_adjacent_increments"]
            pts = [incs[nm]["point"] for nm in inc_names]
            errs = [_err_lo_hi(incs[nm]["point"], incs[nm]) for nm in inc_names]
            los, his = zip(*errs, strict=True)
            ax.bar(
                np.arange(len(inc_names)) + k * width,
                pts,
                width=width,
                yerr=np.vstack([los, his]),
                capsize=1.5,
                color=colors[k],
                label=f"base->{p['pair']['m1']}",
            )
        ax.axhline(0.0, lw=0.8, color="0.4")
        ax.set_xticks(
            np.arange(len(inc_names)) + 0.4 - width / 2,
            inc_names,
            rotation=45,
            ha="right",
            fontsize=6,
        )
        ax.set_title(es, fontsize=8)
        if j == 0:
            ax.set_ylabel(f"adjacent-tier R² increment ({scale})")
            ax.legend(fontsize=6)
    fig.suptitle("Adjacent-tier increments (which correction buys the R²)", fontsize=9)
    _save(fig, fig_dir, "adjacent_increments_v2.png")


def fig_v2_dose(pairs: list[dict], headline: int, scale: str, fig_dir: Path) -> None:
    """delta_tier8 (within − T8) across the stage ladder incl. rlvr_long —
    one line per surface (the dose read)."""
    base_pairs = [p for p in pairs if p["pair"]["m0"] == "base"]
    eval_sets = sorted({_eval_set_key(p) for p in base_pairs})
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = paper_palette(max(1, len(eval_sets)))
    stages_present = [m for m in _STAGE_ORDER if any(p["pair"]["m1"] == m for p in base_pairs)]
    x = np.arange(len(stages_present))
    for c, es in zip(colors, eval_sets, strict=False):
        pts, los, his, xs = [], [], [], []
        for k, m1 in enumerate(stages_present):
            match = [p for p in base_pairs if p["pair"]["m1"] == m1 and _eval_set_key(p) == es]
            if not match:
                continue
            blk, _li = _pl_v2(match[0], headline)
            d8 = blk[scale]["delta_tier8"]
            pts.append(d8["point"])
            lo, hi = _err_lo_hi(d8["point"], d8)
            los.append(lo)
            his.append(hi)
            xs.append(k)
        if pts:
            ax.errorbar(
                xs,
                pts,
                yerr=np.vstack([los, his]),
                marker="o",
                capsize=2,
                color=c,
                label=es,
                lw=1.2,
            )
    ax.axhline(0.0, lw=0.8, color="0.4")
    ax.set_xticks(x, stages_present)
    ax.set_xlabel("stage (base-anchored)")
    ax.set_ylabel(f"within − T8 R² ({scale})")
    ax.set_title("Stage dose read: reparameterization residual across the ladder", fontsize=9)
    ax.legend(fontsize=6)
    _save(fig, fig_dir, "dose_delta_tier8_v2.png")


def fig_v2_sufficient_tier(pairs: list[dict], headline: int, scale: str, fig_dir: Path) -> None:
    """Heatmap pair × surface of sufficientTier (min T with within−R²_T ≤ band);
    'none' (no tier suffices) renders as 9."""
    pair_labels = sorted({f"{p['pair']['m0']}->{p['pair']['m1']}" for p in pairs})
    eval_sets = sorted({_eval_set_key(p) for p in pairs})
    mat = np.full((len(pair_labels), len(eval_sets)), np.nan)
    for p in pairs:
        r = pair_labels.index(f"{p['pair']['m0']}->{p['pair']['m1']}")
        c = eval_sets.index(_eval_set_key(p))
        blk, _li = _pl_v2(p, headline)
        st = blk[scale]["sufficient_tier"]["tier"]
        mat[r, c] = 9.0 if st == "none" else float(st)
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(eval_sets), 0.8 + 0.45 * len(pair_labels)))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=9, aspect="auto")
    ax.set_xticks(range(len(eval_sets)), eval_sets, rotation=30, ha="right", fontsize=6)
    ax.set_yticks(range(len(pair_labels)), pair_labels, fontsize=7)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            if np.isfinite(mat[r, c]):
                v = int(mat[r, c])
                ax.text(
                    c,
                    r,
                    "none" if v == 9 else f"T{v}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="w",
                )
    fig.colorbar(im, ax=ax, label="sufficient tier (9 = none ≤ T8)")
    ax.set_title(f"Sufficient tier per (pair, surface) — {scale}, layer {headline}", fontsize=8)
    _save(fig, fig_dir, "sufficient_tier_heatmap_v2.png")


def fig_v2_baselines(pairs: list[dict], headline: int, fig_dir: Path) -> None:
    """Identity+bias baseline R² vs the fitted maps + kNN retrieval acc@1
    (cosine) with chance — the standing mapping-baselines dual read."""
    rows = []
    for p in pairs:
        blk, li = _pl_v2(p, headline)
        b = blk.get("baselines")
        if b is None:
            continue
        rows.append((f"{p['pair']['m1']}|{_eval_set_key(p)}", blk, b))
    if not rows:
        print("[figs1336-v2] no baselines at the headline layer — skipping baselines figure")
        return
    names = [r[0] for r in rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), squeeze=False)
    ax_r2, ax_knn = axes[0][0], axes[0][1]
    colors = paper_palette(3)
    scale = "raw"
    within = [blk[scale]["within_r2"] for _, blk, _b in rows]
    t8 = [blk[scale]["tiers"]["t8"]["r2"] for _, blk, _b in rows]
    idb = [b["within"]["identity_bias_r2"] for _, _blk, b in rows]
    ax_r2.bar(x - 0.27, within, width=0.27, color=colors[0], label="within ridge")
    ax_r2.bar(x, t8, width=0.27, color=colors[1], label="T8 composition")
    ax_r2.bar(x + 0.27, idb, width=0.27, color=colors[2], label="identity+bias")
    ax_r2.axhline(0.0, lw=0.8, color="0.4")
    ax_r2.set_xticks(x, names, rotation=45, ha="right", fontsize=6)
    ax_r2.set_ylabel("held-out pooled R² (raw)")
    ax_r2.set_title("map vs identity+bias baseline", fontsize=8)
    ax_r2.legend(fontsize=6)
    acc = lambda b, key: b[key]["knn"]["cosine"]["acc_at_k"]["1"]  # noqa: E731
    w_acc = [acc(b, "within") for _, _blk, b in rows]
    t8_acc = [acc(b, "tier8") for _, _blk, b in rows]
    id_acc = [b["within"]["knn_identity_bias"]["cosine"]["acc_at_k"]["1"] for _, _blk, b in rows]
    chance = [b["within"]["knn"]["cosine"]["chance_at_k"]["1"] for _, _blk, b in rows]
    ax_knn.bar(x - 0.27, w_acc, width=0.27, color=colors[0], label="within ridge")
    ax_knn.bar(x, t8_acc, width=0.27, color=colors[1], label="T8 composition")
    ax_knn.bar(x + 0.27, id_acc, width=0.27, color=colors[2], label="identity+bias")
    ax_knn.plot(x, chance, ls="--", lw=0.9, color="0.3", label="chance (1/n_pool)")
    ax_knn.set_xticks(x, names, rotation=45, ha="right", fontsize=6)
    ax_knn.set_ylabel("kNN retrieval acc@1 (cosine)")
    ax_knn.set_title("retrieval read (mapping-baselines rule)", fontsize=8)
    ax_knn.legend(fontsize=6)
    _save(fig, fig_dir, "baselines_identity_knn_v2.png")


def fig_v2_lambda_edge(cells_dir: Path, fig_dir: Path, smoke: bool) -> None:
    """Per-cell selected-lambda edge fractions from cells_v2 lambda_audit."""
    names, lo_f, hi_f = [], [], []
    for path in sorted(cells_dir.glob("cells_*.json")):
        payload = json.loads(path.read_text())
        audit = payload.get("lambda_audit")
        if audit is None:
            continue
        names.append(path.stem.removeprefix("cells_"))
        lo_f.append(audit["frac_at_low_edge"] or 0.0)
        hi_f.append(audit["frac_at_high_edge"] or 0.0)
    if not names:
        assert smoke, f"no cells_v2 lambda_audit blocks under {cells_dir}"
        print("[figs1336-v2] no cells_v2 audits (smoke) — skipping lambda-edge panel")
        return
    x = np.arange(len(names))
    colors = paper_palette(2)
    fig, ax = plt.subplots(figsize=(max(6.4, 0.28 * len(names)), 3.4))
    ax.bar(x - 0.2, lo_f, width=0.4, color=colors[0], label="frac at LOW edge")
    ax.bar(x + 0.2, hi_f, width=0.4, color=colors[1], label="frac at HIGH edge")
    ax.set_xticks(x, names, rotation=60, ha="right", fontsize=5)
    ax.set_ylabel("fraction of (layer x fold) selections")
    ax.set_ylim(0, 1.0)
    ax.set_title("Selected-λ grid-edge fractions per v2 cell (audit)", fontsize=9)
    ax.legend(fontsize=6)
    _save(fig, fig_dir, "lambda_edge_fractions_v2.png")


def fig_v2_layer_curves(cells_dir: Path, fig_dir: Path, smoke: bool) -> None:
    """v2 within-stage layer curves: raw pooled R² (line) + held-out recal R²
    (dots at the persisted layers), per surface."""
    by_surface: dict[str, list[dict]] = {}
    for path in sorted(cells_dir.glob("cells_*.json")):
        payload = json.loads(path.read_text())
        if "matchedn" in path.stem:
            continue
        cell = payload["cell"]
        key = f"{cell['corpus']}_{cell['format']}"
        by_surface.setdefault(key, []).append(payload)
    if not by_surface:
        assert smoke, f"no cells_v2 JSONs under {cells_dir}"
        print("[figs1336-v2] no cells_v2 JSONs (smoke) — skipping v2 layer curves")
        return
    for key, payloads in sorted(by_surface.items()):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        colors = paper_palette(len(payloads))
        for c, payload in zip(colors, payloads, strict=True):
            model = payload["cell"]["model"]
            r2 = payload["r2_per_layer_obs"]
            ax.plot(range(len(r2)), r2, color=c, lw=1.4, label=STAGE_LABELS.get(model, model))
            recal = payload.get("recal", {}).get("per_layer", {})
            if recal:
                lis = sorted(int(k) for k in recal)
                ax.plot(
                    lis,
                    [recal[str(li)]["heldout_recal_r2"] for li in lis],
                    ls="none",
                    marker="o",
                    ms=3.5,
                    color=c,
                )
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out pooled R²")
        ax.set_title(f"v2 within-stage map — {key} (line: raw; dots: recal)", fontsize=9)
        ax.legend(fontsize=7)
        _save(fig, fig_dir, f"layer_curves_{key}_v2.png")


def main_v2(args: argparse.Namespace) -> None:
    set_paper_style()
    eval_dir = args.eval_dir
    ladder_dir = eval_dir / "metric_ladder"
    cells_dir = eval_dir / "cells_v2"
    decision = _load(eval_dir / "decision_v2" / "headline_contrast_v2.json")
    pairs = _pairs_available(ladder_dir)
    headline = int(decision["headline_layer"])
    scale = decision.get("primary_scale", "raw")
    fig_v2_tier_profile(pairs, headline, scale, args.fig_dir)
    fig_v2_stage_contrast(decision, args.fig_dir)
    fig_v2_adjacent_increments(pairs, headline, scale, args.fig_dir)
    fig_v2_dose(pairs, headline, scale, args.fig_dir)
    fig_v2_sufficient_tier(pairs, headline, scale, args.fig_dir)
    fig_v2_baselines(pairs, headline, args.fig_dir)
    fig_v2_lambda_edge(cells_dir, args.fig_dir, args.smoke)
    fig_v2_layer_curves(cells_dir, args.fig_dir, args.smoke)
    print("[figs1336] v2 done")


# ===========================================================================
# v3 figures (round 4, plan v15): pooled off-policy 2x2 panel (Hero 3) +
# per-transition per-cluster delta-Q scatters (Hero 4). Inputs: decision_v3/.
# Colour = ladder STAGE (viridis 0.10-0.70 ramp — the issue1336_metric_ladder
# _plots.py convention: one colour = one stage across every #1336 figure);
# linestyle + marker fill = tier (solid+filled = plain read, dashed/dotted +
# hollow = reparameterized reads).
# ===========================================================================

_V3_STAGE_ORDER = ("base", *_STAGE_ORDER)
_V3_STAGE_SHORT = {
    "base": "base",
    "sft": "SFT",
    "dpo": "DPO",
    "rlvr": "RLVR",
    "rlvr_long": "longer RLVR",
}
_V3_ARM_LABELS = {"on": "on-policy answers", "off": "off-policy answers"}
_V3_REGISTERED_TRANSITIONS = (("base", "sft"), *cm.ADJACENT_PAIRS)
# Tier -> (linestyle, marker, filled, plain-English legend label). Extends the
# established tier_style convention (tier 0 solid+filled, tier 6 dashed+hollow);
# the two v3-only reads take a diamond (own ceiling) and a dotted hollow square.
_V3_TIER_STYLE = {
    "own": ("-", "D", True, "own map (within-stage ceiling)"),
    "t0": ("-", "o", True, "direct transfer (T0)"),
    "t6": ("--", "o", False, "context-reparameterized (T6)"),
    "t8": (":", "s", False, "both-sides reparameterized (T8)"),
}
_V3_TIER_ORDER = ("own", "t0", "t6", "t8")
_V3_TIER_JITTER = {"own": -0.18, "t0": -0.06, "t6": 0.06, "t8": 0.18}


def _v3_stage_colors() -> dict[str, tuple]:
    """Colour = ladder stage on the viridis ramp sampled below the yellow end
    (0.10-0.70 — the make-source-target-lines convention), one colour per stage
    across every v3 figure."""
    cmap = matplotlib.colormaps["viridis"]
    ramp = np.linspace(0.10, 0.70, len(_V3_STAGE_ORDER))
    return {s: cmap(v) for s, v in zip(_V3_STAGE_ORDER, ramp, strict=True)}


def _v3_corpus_order(panel: dict, present: set[str]) -> list[str]:
    """Column order for the 2x2 panel: the JSON's own corpus_order when given,
    else name-sorted with naturalistic-format slices LAST (the plan renders
    naturalistic-lmsys separately from the 7 chat-format corpora)."""
    declared = [c for c in (panel.get("corpus_order") or []) if c in present]
    rest = sorted(present - set(declared), key=lambda c: ("natural" in c.lower(), c))
    return declared + rest


def _v3_pair_for_target(pairs: dict, target: str) -> tuple[str, dict] | None:
    """The transfer pair read AT a target stage: the registered
    adjacent-predecessor transition (base->sft, sft->dpo, dpo->rlvr,
    dpo->rlvr_long) when present, else the deepest available source."""
    cands = {
        (p["source"], p["target"]): (k, p)
        for k, p in pairs.items()
        if p["target"] == target and p["source"] != p["target"]
    }
    if not cands:
        return None
    for src, tgt in _V3_REGISTERED_TRANSITIONS:
        if tgt == target and (src, tgt) in cands:
            return cands[(src, tgt)]
    deepest = max(cands, key=lambda st: _V3_STAGE_ORDER.index(st[0]))
    return cands[deepest]


def _v3_tier_read(
    pairs: dict, target: str, arm: str, corpus: str, tier: str
) -> tuple[str, dict] | None:
    """(source stage, tier block) for one (target, arm, corpus, tier) cell.
    Transfer tiers read the registered pair; 'own' prefers the transfer pair's
    own block and falls back to a source==target (diagonal) entry."""
    entries: list[dict] = []
    cand = _v3_pair_for_target(pairs, target)
    if cand is not None:
        entries.append(cand[1])
    if tier == "own":
        entries.extend(p for p in pairs.values() if p["source"] == p["target"] == target)
    for p in entries:
        blk = p["arms"].get(arm, {}).get("per_corpus", {}).get(corpus, {}).get(tier)
        if blk is not None:
            return (target if tier == "own" else p["source"]), blk
    return None


def fig_v3_pooled_2x2(panel: dict, fig_dir: Path) -> None:
    """Hero 3: pooled per-corpus held-out R^2 by correction tier — rows = arm
    (on-/off-policy answers), cols = eval corpus (naturalistic slices last),
    x = target stage; 4 series per panel (own / T0 / T6 / T8), point colour =
    SOURCE stage of the read.

    Expected ``decision_v3/pooled_offpolicy_2x2.json`` schema (defined here —
    no writer exists at the C-ii pin; C-iii's decision builder must match):
    {"headline_layer": int, "scale": "raw"|...,
     "arms": ["on", "off"], "corpus_order": [...],   # corpus_order optional
     "pairs": {"<i>__<j>": {"source": i, "target": j, "arms": {"<arm>": {
       "per_corpus": {"<corpus>": {"own"|"t0"|"t6"|"t8": {
         "r2": float, "r2_bootstrap": {"ci_lo": float, "ci_hi": float}}}}}}}}}
    A source==target entry may carry only "own" (the diagonal ceiling).
    """
    from matplotlib.lines import Line2D

    headline = int(panel["headline_layer"])
    scale = str(panel.get("scale", "raw"))
    arms = [a for a in ("on", "off") if a in panel["arms"]]
    assert arms, f"no known arms in panel JSON: {panel['arms']!r}"
    pairs = panel["pairs"]
    present: set[str] = set()
    for p in pairs.values():
        for ab in p["arms"].values():
            present.update(ab["per_corpus"])
    corpora = _v3_corpus_order(panel, present)
    assert corpora, "panel JSON carries no per_corpus blocks"
    targets = [t for t in _V3_STAGE_ORDER if any(p["target"] == t for p in pairs.values())]
    xpos = {t: k for k, t in enumerate(targets)}
    stage_colors = _v3_stage_colors()
    fig, axes = plt.subplots(
        len(arms),
        len(corpora),
        figsize=(2.7 * len(corpora), 3.2 * len(arms)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    used_sources: set[str] = set()
    n_points = 0
    n_missing = 0
    for i, arm in enumerate(arms):
        for j, corpus in enumerate(corpora):
            ax = axes[i][j]
            for tier in _V3_TIER_ORDER:
                ls, marker, filled, _lbl = _V3_TIER_STYLE[tier]
                jit = _V3_TIER_JITTER[tier]
                pts = []
                for t in targets:
                    got = _v3_tier_read(pairs, t, arm, corpus, tier)
                    if got is None:
                        n_missing += 1
                        continue
                    src, blk = got
                    lo, hi = _err_lo_hi(float(blk["r2"]), blk["r2_bootstrap"])
                    pts.append((xpos[t] + jit, float(blk["r2"]), lo, hi, src))
                if not pts:
                    continue
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    ls,
                    color="0.70",
                    lw=1.0,
                    zorder=2,
                )
                for x, y, lo, hi, src in pts:
                    col = stage_colors[src]
                    used_sources.add(src)
                    ax.errorbar(
                        x,
                        y,
                        yerr=[[lo], [hi]],
                        fmt=marker,
                        ms=4.5,
                        color=col,
                        markerfacecolor=col if filled else "white",
                        markeredgecolor=col,
                        markeredgewidth=0.9,
                        elinewidth=0.8,
                        capsize=2,
                        zorder=3,
                    )
                    n_points += 1
            ax.axhline(0.0, color="grey", lw=0.6, zorder=1)
            if i == 0:
                ax.set_title(corpus, fontsize=8)
            if i == len(arms) - 1:
                ax.set_xticks(list(xpos.values()))
                ax.set_xticklabels([_V3_STAGE_SHORT[t] for t in targets], fontsize=7, rotation=45)
                ax.set_xlabel("target stage", fontsize=8)
        axes[i][0].set_ylabel(f"{_V3_ARM_LABELS[arm]}\nheld-out pooled R² ({scale})", fontsize=8)
    assert n_points > 0, "2x2 panel: zero plottable points"
    if n_missing:
        print(f"[figs1336-v3] 2x2 panel: {n_missing} absent tier reads (plotted {n_points})")
    tier_handles = [
        Line2D(
            [0],
            [0],
            color="0.45",
            ls=ls,
            marker=marker,
            ms=5,
            markerfacecolor="0.45" if filled else "white",
            markeredgecolor="0.45",
            label=lbl,
        )
        for ls, marker, filled, lbl in (_V3_TIER_STYLE[t] for t in _V3_TIER_ORDER)
    ]
    stage_handles = [
        Line2D(
            [0],
            [0],
            ls="none",
            marker="o",
            ms=6,
            color=stage_colors[s],
            label=f"source: {_V3_STAGE_SHORT[s]}",
        )
        for s in _V3_STAGE_ORDER
        if s in used_sources
    ]
    fig.legend(
        handles=tier_handles + stage_handles,
        fontsize=6.5,
        ncol=min(4, len(tier_handles) + len(stage_handles)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.055),
        frameon=False,
    )
    fig.suptitle(
        f"Pooled context→answer map: per-corpus held-out R² by correction "
        f"tier (layer {headline}; transfer reads use the registered "
        f"adjacent-predecessor source)",
        fontsize=9,
    )
    _save(fig, fig_dir, "2x2_panel_v3.png")


def fig_v3_delta_q_scatter(dq: dict, fig_dir: Path) -> None:
    """Hero 4: per-transition per-cluster delta-Q scatter, one figure per
    registered transition, one panel per arm; every point labeled with its
    cluster id (the #1902 clusters_delta_qc_scatter convention). Horizontal
    lines: 97.5% quantile of the per-draw MAX-cluster permutation null, and
    the achievable ceiling (band-vs-ceiling clause). Consumes the C-i battery
    JSON decision_v3/cluster_delta_q_per_transition.json verbatim — no
    statistics are recomputed here."""
    stage_colors = _v3_stage_colors()
    li = dq.get("headline", {}).get("headline_layer")
    n_draws = dq.get("perm_draws")
    transitions = sorted(dq["transitions"].items(), key=lambda kv: kv[1]["transition_idx"])
    assert transitions, "delta-Q JSON carries no transitions"
    for t_slug, tb in transitions:
        src, tgt = tb["source"], tb["target"]
        arms = [a for a in ("on", "off") if a in tb["arms"]]
        assert arms, f"transition {t_slug}: no known arms in {list(tb['arms'])!r}"
        fig, axes = plt.subplots(
            1, len(arms), figsize=(6.2 * len(arms), 4.0), sharey=True, squeeze=False
        )
        for ax, arm in zip(axes[0], arms, strict=True):
            blk = tb["arms"][arm]
            ids = np.asarray(blk["cluster_ids"], dtype=int)
            vals = np.asarray(blk["delta_q"], dtype=float)
            assert ids.shape == vals.shape and ids.size > 0, (t_slug, arm, ids.shape)
            ax.scatter(
                ids,
                vals,
                s=16,
                color=stage_colors[src],
                edgecolors="#333333",
                linewidths=0.4,
                zorder=3,
            )
            for cid, v in zip(ids.tolist(), vals.tolist(), strict=True):
                if np.isfinite(v):
                    ax.annotate(
                        str(cid),
                        (cid, v),
                        fontsize=5,
                        xytext=(1, 2),
                        textcoords="offset points",
                    )
            ax.axhline(0.0, color="grey", lw=0.6, zorder=1)
            ax.axhline(
                float(blk["null_band_97p5"]),
                ls="--",
                lw=1.0,
                color="#333333",
                label="max-cluster permutation null (97.5%)",
            )
            ax.axhline(
                float(blk["ceiling_max"]),
                ls=":",
                lw=1.0,
                color="#333333",
                label="achievable ceiling (max per-cluster source residual)",
            )
            p = float(blk["max_cluster_p"])
            p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax.set_title(
                f"{_V3_ARM_LABELS[arm]} — most-improved cluster "
                f"{blk['obs_argmax_cluster']} (max-cluster {p_str})",
                fontsize=8,
            )
            ax.set_xlabel("cluster id", fontsize=8)
        axes[0][0].set_ylabel(
            "ΔQ per cluster (held-out residual ratio,\nsource − target; "
            "+ = better predicted at the target)",
            fontsize=8,
        )
        axes[0][0].legend(fontsize=6, loc="best", frameon=False)
        fig.suptitle(
            f"{_V3_STAGE_SHORT[src]} → {_V3_STAGE_SHORT[tgt]}: per-cluster "
            f"ΔQ under the pooled map (layer {li}; {n_draws} permutation draws)",
            fontsize=9,
        )
        _save(fig, fig_dir, f"delta_q_scatter_{t_slug}_v3.png")


def main_v3(args: argparse.Namespace) -> None:
    set_paper_style()
    dec_dir = args.eval_dir / "decision_v3"
    panel_path = dec_dir / "pooled_offpolicy_2x2.json"
    dq_path = dec_dir / "cluster_delta_q_per_transition.json"
    panel = _maybe(panel_path) if args.smoke else _load(panel_path)
    dq = _maybe(dq_path) if args.smoke else _load(dq_path)
    if panel is None:
        print(f"[figs1336-v3] SKIP 2x2 panel — missing {panel_path} (smoke)")
    else:
        fig_v3_pooled_2x2(panel, args.fig_dir)
    if dq is None:
        print(f"[figs1336-v3] SKIP delta-Q scatters — missing {dq_path} (smoke)")
    else:
        fig_v3_delta_q_scatter(dq, args.fig_dir)
    print("[figs1336] v3 done")


def main() -> None:
    args = parse_args()
    if args.v3:
        main_v3(args)
        return
    if args.v2:
        main_v2(args)
        return
    set_paper_style()
    eval_dir = args.eval_dir
    cells_dir = eval_dir / "cells"
    align_dir = eval_dir / "ladder_alignment"
    decision = _load(eval_dir / "decision" / "headline_contrast.json")
    pairs = _pairs_available(align_dir)
    fig_hero(pairs, decision, args.fig_dir)
    fig_layer_curves(cells_dir, args.fig_dir, args.smoke)
    fig_gap_frozen(pairs, args.fig_dir)
    fig_increments(decision, args.fig_dir)
    fig_ceilings_alignment_orth(pairs, decision, args.fig_dir)
    fig_nll_keeprate(cells_dir, eval_dir / "gen_audits", args.fig_dir, args.smoke)
    fig_matchedn_degeneracy(cells_dir, args.fig_dir, args.smoke)
    print("[figs1336] done")


if __name__ == "__main__":
    main()

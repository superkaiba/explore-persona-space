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


def main() -> None:
    args = parse_args()
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

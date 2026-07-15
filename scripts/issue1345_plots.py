#!/usr/bin/env python
"""Issue #1345 Phase 7 — verdict lattice + summary figures.

Assembles the plan §3 verdict lattice per (model, arm) from the committed
Phase 4-6 JSONs (within cells, cross-regime transfer + paired bootstrap,
operator comparison / Δ_reparam) and renders:
  hero:      3x3 cross-regime R^2 heatmaps (context arm, instruct + base)
  companion: prefix-arm heatmaps, raw-cosine heatmaps, 28-layer within
             sweeps, reparam recovery-vs-null bars, answer token-length
             distributions, story extraction-yield table (JSON).

Outputs: figures/issue_1345/*.png + eval_results/issue_1345/verdict_lattice.json
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
import issue1345_common as c  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import set_paper_style  # noqa: E402

L19 = "19"
REGIME_LABEL = {"r1": "chat", "r2": "no-template", "r3": "stories"}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------------------
# Verdict lattice (plan §3 — DISJOINT and exhaustive)
# ---------------------------------------------------------------------------
def verdict_for(transfer: dict, opcomp: dict) -> dict | None:
    """Plan §3 verdict for one (model, arm); None when the headline pair was
    smoke-skipped (degenerate at smoke n — the transfer JSON records the
    reason under skipped_pairs; production always carries both deltas)."""
    boot = transfer["headline_paired_bootstrap"]
    deltas = transfer["delta_table_l19"]
    if "r1->r2" not in deltas or "r2->r1" not in deltas:
        return None
    d12 = deltas["r1->r2"]["delta_l19"]
    d21 = deltas["r2->r1"]["delta_l19"]
    d_xfer = 0.5 * (d12 + d21)
    d_same = d_xfer + c.DELTA_SAME_MARGIN
    d_diff = d_xfer + c.DELTA_DIFF_MARGIN
    d_reparam = opcomp["delta_reparam_l19"]["delta_reparam"]
    ci_below_0 = bool(boot["delta_diff_ci_wholly_below_0"])
    if d_same >= 0 and d_reparam >= 0:
        verdict = "same-operator"
    elif d_same < 0 and d_reparam >= 0:
        verdict = "reparameterized"
    elif d_diff < 0 and d_reparam < 0 and ci_below_0:
        verdict = "different-map"
    else:
        verdict = "inconclusive"
    return {
        "delta_1to2": d12,
        "delta_2to1": d21,
        "delta_xfer": d_xfer,
        "delta_same": d_same,
        "delta_diff": d_diff,
        "delta_reparam": d_reparam,
        "delta_diff_ci": boot["delta_diff"],
        "delta_diff_ci_wholly_below_0": ci_below_0,
        "verdict": verdict,
    }


def story_reads(out_dir: Path, model: str, arm: str) -> dict | None:
    """Story existence read (within L19 vs shuffle-null p95) + weakness band."""
    cells = _load(out_dir / f"cells_{c.cell_id(model, 'r3', arm)}.json")
    nulls = _load(out_dir / f"nulls_{c.cell_id(model, 'r3', arm)}.json")
    if cells is None or nulls is None:
        return None
    r2_19 = float(cells["r2_per_layer_obs"][19])
    null_col = [float(row[19]) for row in nulls["null_matrix"]]
    p95 = float(np.quantile(null_col, 0.95)) if null_col else float("nan")
    band = (
        "weak (<=0.30)"
        if r2_19 <= 0.30
        else ("strong (>=0.50)" if r2_19 >= 0.50 else "intermediate")
    )
    return {
        "within_l19_r2": r2_19,
        "shuffle_null_p95_l19": p95,
        "exists": bool(r2_19 > p95),
        "weakness_band": band,
        "n_null_draws": len(null_col),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def heatmap_3x3(transfer: dict, regimes: list[str], title: str, out_png: Path) -> None:
    n = len(regimes)
    mat = np.full((n, n), np.nan)
    for i, ri in enumerate(regimes):
        for j, rj in enumerate(regimes):
            if ri == rj:
                grain = "headline" if ri != "r3" else "r3pair"
                key = f"{ri}@{grain}"
                if key in transfer["within_r2_by_layer"]:
                    mat[i, j] = transfer["within_r2_by_layer"][key][L19]
            else:
                entry = transfer["matrix"].get(f"{ri}->{rj}")
                if entry:
                    mat[i, j] = entry["transfer_r2_by_layer"][L19]
    fig, ax = plt.subplots(figsize=(5.2, 4.4), layout="constrained")
    im = ax.imshow(
        mat, vmin=min(0.0, np.nanmin(mat)), vmax=max(0.7, np.nanmax(mat)), cmap="viridis"
    )
    labels = [REGIME_LABEL[r] for r in regimes]
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("target regime (apply / evaluate)")
    ax.set_ylabel("source regime (fit)")
    for i in range(n):
        for j in range(n):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="held-out $R^2$ (layer 19)")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig] {out_png}", flush=True)


def cosine_heatmap(opcomp: dict, regimes: list[str], title: str, out_png: Path) -> None:
    n = len(regimes)
    mat = np.full((n, n), np.nan)
    for i, ri in enumerate(regimes):
        for j, rj in enumerate(regimes):
            if i == j:
                mat[i, j] = 1.0
                continue
            pair = opcomp["pairs"].get(f"{ri}~{rj}") or opcomp["pairs"].get(f"{rj}~{ri}")
            if pair:
                mat[i, j] = pair["per_layer"][L19]["raw_cosine"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4), layout="constrained")
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="magma")
    labels = [REGIME_LABEL[r] for r in regimes]
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    for i in range(n):
        for j in range(n):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="raw operator cosine (L19)")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig] {out_png}", flush=True)


def layer_sweep_fig(out_dir: Path, model: str, regimes: list[str], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0), layout="constrained")
    for regime in regimes:
        cells = _load(out_dir / f"cells_{c.cell_id(model, regime, 'context')}.json")
        if cells is None:
            continue
        r2 = cells["r2_per_layer_obs"]
        ax.plot(range(len(r2)), r2, marker="o", markersize=2.5, label=REGIME_LABEL[regime])
    ax.axvline(19, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out $R^2$")
    ax.set_title(f"Within-regime layer sweep — {c.MODEL_SLUG[model]} (context arm)")
    ax.legend()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig] {out_png}", flush=True)


def reparam_bar_fig(opcomp: dict, title: str, out_png: Path) -> None:
    d = opcomp["delta_reparam_l19"]
    nulls = opcomp["reparam_r1r2"][L19]["matched_capacity_nulls"]
    dirs = ("b2i", "i2b")
    labels = ["r2 op in chat", "r1 op in no-template"]
    recovered = [d["recovered_r2"][k] for k in dirs]
    within = [d["within_r2"][k] for k in dirs]
    null_v = [nulls[k]["null_recovery_r2"] for k in dirs]
    x = np.arange(len(dirs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(5.6, 4.0), layout="constrained")
    ax.bar(x - w, within, w, label="target within $R^2$")
    ax.bar(x, recovered, w, label="reparam recovered $R^2$")
    ax.bar(x + w, null_v, w, label="matched-capacity null")
    ax.set_xticks(x, labels)
    ax.set_ylabel("held-out $R^2$ (layer 19)")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig] {out_png}", flush=True)


def answer_length_fig(
    turnstore_dir: Path, models: list[str], regimes: list[str], out_png: Path
) -> None:
    """Answer-span token-length distributions per regime (from shard spans_meta)."""
    import torch

    fig, ax = plt.subplots(figsize=(6.0, 4.0), layout="constrained")
    plotted = False
    for regime in regimes:
        lengths: list[int] = []
        for model in models:
            for sp in sorted(turnstore_dir.glob(f"{c.stem_for(model, regime)}_shard*.pt")):
                payload = torch.load(sp, map_location="cpu", weights_only=False)
                for meta in payload.get("spans_meta", []):
                    spans = meta.get("spans", {})
                    key = "a1" if "a1" in spans else ("answer" if "answer" in spans else None)
                    if key:
                        s, e = spans[key]
                        lengths.append(int(e) - int(s))
                del payload
        if lengths:
            ax.hist(lengths, bins=40, alpha=0.5, density=True, label=REGIME_LABEL[regime])
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("answer span length (tokens)")
    ax.set_ylabel("density")
    ax.set_title("Answer token-length distributions per regime")
    ax.legend()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig] {out_png}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=c.EVAL_DIR)
    ap.add_argument("--fig-dir", type=Path, default=c.FIG_DIR)
    ap.add_argument("--turnstore-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--stories-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--no-r3", action="store_true")
    ap.add_argument("--skip-length-dist", action="store_true")
    args = ap.parse_args()

    set_paper_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    regimes = [r for r in c.REGIMES if not (args.no_r3 and r == "r3")]

    lattice: dict = {
        "metadata": c.metadata(0, 0, "scripts/issue1345_plots.py"),
        "margins": {"same": c.DELTA_SAME_MARGIN, "diff": c.DELTA_DIFF_MARGIN},
        "per_model_arm": {},
    }
    for model in c.MODELS:
        slug = c.MODEL_SLUG[model]
        for arm in c.ARMS:
            transfer = _load(args.out_dir / f"cross_regime_transfer_{slug}_{arm}.json")
            opcomp = _load(args.out_dir / f"operator_comparison_{slug}_{arm}.json")
            if transfer is None or opcomp is None:
                print(f"[verdict] missing inputs for {slug}/{arm} — skipped", flush=True)
                continue
            entry = verdict_for(transfer, opcomp)
            if entry is None:
                print(f"[verdict] headline pair smoke-skipped for {slug}/{arm} — skipped")
                continue
            entry["story"] = story_reads(args.out_dir, model, arm)
            lattice["per_model_arm"][f"{slug}_{arm}"] = entry
            heatmap_3x3(
                transfer,
                regimes,
                f"Cross-regime transfer $R^2$ — {slug} ({arm} arm, L19)",
                args.fig_dir / f"cross_regime_r2_heatmap_{arm}_{slug}.png",
            )
            cosine_heatmap(
                opcomp,
                regimes,
                f"Raw operator cosine — {slug} ({arm} arm, L19)",
                args.fig_dir / f"operator_cosine_heatmap_{arm}_{slug}.png",
            )
            if arm == "context" and L19 in opcomp.get("reparam_r1r2", {}):
                # (Leg B may be smoke-skipped on a degenerate paired set)
                reparam_bar_fig(
                    opcomp,
                    f"Reparam recovery vs matched-capacity null — {slug} (context, L19)",
                    args.fig_dir / f"reparam_recovery_{slug}.png",
                )
        layer_sweep_fig(
            args.out_dir, model, regimes, args.fig_dir / f"layer_sweep_{slug}_context.png"
        )

    # Story yield table (digest of the Phase-1 reports)
    yields = {}
    for model in c.MODELS:
        rep = _load(args.stories_dir / f"story_yield_{model}.json")
        if rep:
            yields[model] = {k: rep[k] for k in ("n_kept", "yield_ok", "counts_main") if k in rep}
    lattice["story_yield"] = yields

    if not args.skip_length_dist:
        answer_length_fig(
            args.turnstore_dir,
            list(c.MODELS),
            regimes,
            args.fig_dir / "answer_token_length_distributions.png",
        )

    c.write_json(args.out_dir / "verdict_lattice.json", lattice)
    print("[done] verdict lattice + figures complete", flush=True)


if __name__ == "__main__":
    main()

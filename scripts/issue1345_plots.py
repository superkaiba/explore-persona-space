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
REGIME_LABEL = {
    "r1": "chat",
    "r2": "no-template",
    "r3": "stories",
    "r4": "paired stories (TF)",
    "r4op": "paired stories (on-policy)",
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _diag_grain(regime: str) -> str:
    """Within-diagonal grain per regime (r1/r2 headline; story regimes their pair)."""
    return {"r3": "r3pair", "r4": "r4pair"}.get(regime, "headline")


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


def paired_story_verdict(out_dir: Path, model: str, arm: str) -> dict | None:
    """Plan v8 §3 Δ_r4 verdict from the r4 within cell's conv-grouped L19 CI.

    Corpus effect <=> R² > 0 AND CI excludes 0 on the positive side; framing
    effect <=> CI wholly below 0; inconclusive otherwise. None when the r4 cell
    is absent (halted / smoke-skipped).
    """
    cells = _load(out_dir / f"cells_{c.cell_id(model, 'r4', arm)}.json")
    if cells is None:
        return None
    r2_19 = float(cells["r2_per_layer_obs"][19])
    boot = cells.get("r2_bootstrap_ci_frozen_layers_conv", {}).get("19")
    if boot is None:
        return {"within_l19_r2": r2_19, "verdict": "inconclusive (no bootstrap CI)"}
    lo, hi = float(boot["ci_lo"]), float(boot["ci_hi"])
    if r2_19 > 0 and lo > 0:
        verdict = "corpus-effect (paired-story map exists)"
    elif hi < 0:
        verdict = "framing-effect (collapse persists on the shared corpus)"
    else:
        verdict = "inconclusive"
    return {
        "within_l19_r2": r2_19,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_groups": boot.get("n_groups"),
        "verdict": verdict,
    }


def story_reads(out_dir: Path, model: str, arm: str, regime: str = "r3") -> dict | None:
    """Story existence read (within L19 vs shuffle-null p95) + weakness band."""
    cells = _load(out_dir / f"cells_{c.cell_id(model, regime, arm)}.json")
    nulls = _load(out_dir / f"nulls_{c.cell_id(model, regime, arm)}.json")
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
                key = f"{ri}@{_diag_grain(ri)}"
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


def reparam_bar_fig(
    opcomp: dict,
    title: str,
    out_png: Path,
    *,
    delta_key: str = "delta_reparam_l19",
    reparam_key: str = "reparam_r1r2",
    labels: tuple[str, str] = ("r2 op in chat", "r1 op in no-template"),
) -> None:
    d = opcomp[delta_key]
    nulls = opcomp[reparam_key][L19]["matched_capacity_nulls"]
    dirs = ("b2i", "i2b")
    labels = list(labels)
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


# ---------------------------------------------------------------------------
# conversation-paired-stories figures (plan v8 §6)
# ---------------------------------------------------------------------------
def _null_p95_per_layer(nulls: dict) -> list[float]:
    return [
        float(np.quantile([row[li] for row in nulls["null_matrix"]], 0.95))
        for li in range(len(nulls["observed_row"]))
    ]


def hero_paired_layer_sweep(
    out_dir: Path, parent_eval_dir: Path, fig_dir: Path, model: str
) -> None:
    """Hero (plan v8 §6): r4 TF layer sweep, both arms, overlaid on the parent
    ARIA (r3) curve, this run's chat/plain-text refits for scale, the r4
    shuffle-null p95 band, and the on-policy companion points."""
    slug = c.MODEL_SLUG[model]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), layout="constrained", sharey=True)
    plotted = False
    for ax, arm in zip(axes, c.ARMS, strict=True):
        for regime, style in (("r1", "-"), ("r2", "-"), ("r4", "-")):
            cells = _load(out_dir / f"cells_{c.cell_id(model, regime, arm)}.json")
            if cells is None:
                continue
            r2 = cells["r2_per_layer_obs"]
            lw = 2.2 if regime == "r4" else 1.2
            ax.plot(
                range(len(r2)),
                r2,
                style,
                lw=lw,
                marker="o",
                markersize=2.0,
                label=REGIME_LABEL[regime],
            )
            plotted = True
        parent_r3 = _load(parent_eval_dir / f"cells_{c.cell_id(model, 'r3', arm)}.json")
        if parent_r3 is not None:
            r2 = parent_r3["r2_per_layer_obs"]
            ax.plot(
                range(len(r2)),
                r2,
                "--",
                lw=1.4,
                label="ARIA stories (parent r3, unshared corpus)",
            )
        nulls = _load(out_dir / f"nulls_{c.cell_id(model, 'r4', arm)}.json")
        if nulls is not None:
            p95 = _null_p95_per_layer(nulls)
            ax.plot(range(len(p95)), p95, ":", lw=1.0, label="r4 shuffle-null p95")
        comp = _load(out_dir / f"cells_{c.cell_id(model, 'r4op', arm)}.json")
        if comp is not None:
            r2 = comp["r2_per_layer_obs"]
            ax.plot(
                range(len(r2)),
                r2,
                linestyle="none",
                marker="x",
                markersize=4.0,
                label=REGIME_LABEL["r4op"],
            )
        ax.axvline(19, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("layer")
        ax.set_title(f"{arm} arm")
    if not plotted:
        plt.close(fig)
        print("[fig] hero paired sweep skipped (no r1/r2/r4 cells)", flush=True)
        return
    axes[0].set_ylabel("within-regime held-out $R^2$")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Paired-verbatim story regime layer sweep — {slug}")
    out_png = fig_dir / f"hero_paired_story_layer_sweep_{slug}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig] {out_png}", flush=True)


def _ci_offsets(v: float, boot: dict | None) -> tuple[float, float]:
    """Non-negative errorbar OFFSETS from a conv-bootstrap CI (gotchas: never
    raw bounds / signed deltas — clamp element-wise)."""
    if not boot:
        return 0.0, 0.0
    return max(0.0, v - float(boot["ci_lo"])), max(0.0, float(boot["ci_hi"]) - v)


def _cell_l19(path: Path) -> tuple[float, dict | None] | None:
    d = _load(path)
    if d is None:
        return None
    return (
        float(d["r2_per_layer_obs"][19]),
        d.get("r2_bootstrap_ci_frozen_layers_conv", {}).get("19"),
    )


def tf_companion_panel(out_dir: Path, matched_row_dir: Path, fig_dir: Path, model: str) -> None:
    """TF-vs-on-policy calibration panel (plan v8 §6 low-level companion):
    per arm — TF full, TF on the companion's exact subset, companion cell."""
    slug = c.MODEL_SLUG[model]
    bars = []
    for arm in c.ARMS:
        for name, path in (
            ("TF (full)", out_dir / f"cells_{c.cell_id(model, 'r4', arm)}.json"),
            (
                "TF (companion subset)",
                matched_row_dir / f"cells_R_{slug}_r4_tf_on_companion_{arm}.json",
            ),
            ("on-policy companion", out_dir / f"cells_{c.cell_id(model, 'r4op', arm)}.json"),
        ):
            read = _cell_l19(path)
            if read is not None:
                bars.append((f"{name}\n({arm})", *read))
    if not bars:
        print("[fig] tf-vs-companion panel skipped (no cells)", flush=True)
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.0), layout="constrained")
    x = np.arange(len(bars))
    vals = [b[1] for b in bars]
    errs = np.array([_ci_offsets(b[1], b[2]) for b in bars]).T  # (2, n) lo/hi offsets
    ax.bar(x, vals, 0.6, yerr=errs, capsize=3)
    ax.set_xticks(x, [b[0] for b in bars], fontsize=7)
    ax.set_ylabel("held-out $R^2$ (layer 19)")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_title(f"TF vs on-policy story capture — {slug} (conv-bootstrap 95% CI)")
    out_png = fig_dir / f"tf_vs_companion_{slug}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig] {out_png}", flush=True)


def matched_row_ceiling_panel(
    out_dir: Path, matched_row_dir: Path, fig_dir: Path, model: str
) -> None:
    """Same-n chat-ceiling comparator (plan v8 §4/§6): r4 within vs the r1/r2
    matched-row refits vs the full-n r1 (context arm)."""
    slug = c.MODEL_SLUG[model]
    bars = []
    for name, path in (
        ("paired stories (TF)", out_dir / f"cells_{c.cell_id(model, 'r4', 'context')}.json"),
        ("chat (r4-matched rows)", matched_row_dir / f"cells_R_{slug}_r1_matched_context.json"),
        (
            "no-template (r4-matched rows)",
            matched_row_dir / f"cells_R_{slug}_r2_matched_context.json",
        ),
        ("chat (full n)", out_dir / f"cells_{c.cell_id(model, 'r1', 'context')}.json"),
    ):
        read = _cell_l19(path)
        if read is not None:
            bars.append((name, *read))
    if not bars:
        print("[fig] matched-row ceiling panel skipped (no cells)", flush=True)
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    x = np.arange(len(bars))
    vals = [b[1] for b in bars]
    errs = np.array([_ci_offsets(b[1], b[2]) for b in bars]).T
    ax.bar(x, vals, 0.6, yerr=errs, capsize=3)
    ax.set_xticks(x, [b[0] for b in bars], fontsize=7)
    ax.set_ylabel("held-out $R^2$ (layer 19, context arm)")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_title(f"Chat-ceiling comparators at matched n — {slug}")
    out_png = fig_dir / f"matched_row_ceiling_{slug}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig] {out_png}", flush=True)


def heatmap_with_parent_r3(
    transfer: dict, parent_transfer: dict | None, title: str, out_png: Path
) -> None:
    """4-regime transfer heatmap (plan v8 §6): this run's r1/r2/r4 entries +
    the PARENT run's r3 row/column (different corpus + n — annotated)."""
    regimes = ["r1", "r2", "r3", "r4"]
    n = len(regimes)
    mat = np.full((n, n), np.nan)
    parent_cells = np.zeros((n, n), dtype=bool)
    for i, ri in enumerate(regimes):
        for j, rj in enumerate(regimes):
            src = None
            if "r3" in (ri, rj):
                if {ri, rj} == {"r3", "r4"}:
                    continue  # never computed (different story corpora)
                src, parent_cells[i, j] = parent_transfer, True
            else:
                src = transfer
            if src is None:
                continue
            if ri == rj:
                key = f"{ri}@{_diag_grain(ri)}"
                if key in src["within_r2_by_layer"]:
                    mat[i, j] = src["within_r2_by_layer"][key][L19]
            else:
                entry = src["matrix"].get(f"{ri}->{rj}")
                if entry:
                    mat[i, j] = entry["transfer_r2_by_layer"][L19]
    fig, ax = plt.subplots(figsize=(5.8, 4.8), layout="constrained")
    im = ax.imshow(
        mat, vmin=min(0.0, np.nanmin(mat)), vmax=max(0.7, np.nanmax(mat)), cmap="viridis"
    )
    labels = [REGIME_LABEL[r] for r in regimes]
    ax.set_xticks(range(n), labels, fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    ax.set_xlabel("target regime (apply / evaluate)")
    ax.set_ylabel("source regime (fit)")
    for i in range(n):
        for j in range(n):
            if np.isfinite(mat[i, j]):
                txt = f"{mat[i, j]:.3f}" + ("*" if parent_cells[i, j] else "")
                ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=7)
    ax.set_title(title + "\n(* = parent-run r3 values: unshared corpus, different n)")
    fig.colorbar(im, ax=ax, label="held-out $R^2$ (layer 19)")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig] {out_png}", flush=True)


def _paired_yield_digest(stories_dir: Path) -> dict:
    """Counts-only digest of the paired + companion yield reports (never text)."""
    paired_yields = {}
    for model in c.R4_MODELS:
        for slug_key, name in (
            ("paired", f"story_yield_paired_{model}.json"),
            ("paired_op", f"story_yield_paired_op_{model}.json"),
        ):
            rep = _load(stories_dir / name)
            if rep:
                paired_yields[f"{model}_{slug_key}"] = {
                    k: rep[k] for k in ("n_kept", "n_target", "yield_ok") if k in rep
                }
    return paired_yields


def _arm_extra_figs(args, model, arm, slug, transfer, opcomp, *, r4_live: bool) -> None:
    """Per-(model, arm) reparam bars + the 4-regime heatmap (paired variant)."""
    if arm == "context" and L19 in opcomp.get("reparam_r1r2", {}):
        # (Leg B may be smoke-skipped on a degenerate paired set)
        reparam_bar_fig(
            opcomp,
            f"Reparam recovery vs matched-capacity null — {slug} (context, L19)",
            args.fig_dir / f"reparam_recovery_{slug}.png",
        )
    if arm == "context" and L19 in opcomp.get("reparam_r1r4", {}):
        reparam_bar_fig(
            opcomp,
            f"r1<->r4 reparam recovery vs matched-capacity null — {slug} (context, L19)",
            args.fig_dir / f"reparam_recovery_r1_r4_{slug}.png",
            delta_key="delta_reparam_r1r4_l19",
            reparam_key="reparam_r1r4",
            labels=("r4 op in chat", "r1 op in paired stories"),
        )
    if r4_live:
        heatmap_with_parent_r3(
            transfer,
            _load(args.parent_eval_dir / f"cross_regime_transfer_{slug}_{arm}.json"),
            f"Cross-regime transfer $R^2$ incl. paired stories — {slug} ({arm}, L19)",
            args.fig_dir / f"cross_regime_r2_heatmap_4reg_{arm}_{slug}.png",
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=c.EVAL_DIR)
    ap.add_argument("--fig-dir", type=Path, default=c.FIG_DIR)
    ap.add_argument("--turnstore-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--stories-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--no-r3", action="store_true", help="story regime halted for BOTH models")
    ap.add_argument(
        "--no-r3-models",
        default="",
        help="comma-separated models whose story regime halted (per-model yield floor); "
        "their r3 panels report N/A — not tested",
    )
    ap.add_argument(
        "--no-r4", action="store_true", help="paired-story regime halted (r4 yield floor)"
    )
    ap.add_argument(
        "--parent-eval-dir",
        type=Path,
        default=Path("eval_results/issue_1345"),
        help="the parent run's committed eval JSONs (r3 overlays for the paired variant)",
    )
    ap.add_argument("--matched-row-dir", type=Path, default=None)
    ap.add_argument("--skip-length-dist", action="store_true")
    args = ap.parse_args()

    halted = set(c.MODELS) if args.no_r3 else {m for m in args.no_r3_models.split(",") if m}
    assert halted <= set(c.MODELS), f"unknown --no-r3-models entries: {sorted(halted)}"
    matched_row_dir = args.matched_row_dir or (args.out_dir / "matched_row")

    def _r4_live(m: str) -> bool:
        return c.HAS_R4 and not args.no_r4 and m in c.R4_MODELS

    set_paper_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    # Union regimes (r3 stays when ANY model's story leg survived); per-model
    # figure axes use regimes_for[model] so a halted model never gets an r3
    # row/column it did not test (plan §7 per-model yield floor).
    regimes = [
        r
        for r in c.REGIMES
        if not (r == "r3" and halted == set(c.MODELS)) and not (r == "r4" and args.no_r4)
    ]
    regimes_for = {
        m: [
            r
            for r in c.REGIMES
            if not (r == "r3" and m in halted) and not (r == "r4" and not _r4_live(m))
        ]
        for m in c.MODELS
    }

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
            if model in halted:
                entry["story"] = "N/A — not tested (per-model story yield floor, plan §7)"
            else:
                entry["story"] = story_reads(args.out_dir, model, arm)
            if _r4_live(model):
                entry["story_paired"] = story_reads(args.out_dir, model, arm, regime="r4")
                entry["story_paired_verdict"] = paired_story_verdict(args.out_dir, model, arm)
            lattice["per_model_arm"][f"{slug}_{arm}"] = entry
            heatmap_3x3(
                transfer,
                regimes_for[model],
                f"Cross-regime transfer $R^2$ — {slug} ({arm} arm, L19)",
                args.fig_dir / f"cross_regime_r2_heatmap_{arm}_{slug}.png",
            )
            cosine_heatmap(
                opcomp,
                regimes_for[model],
                f"Raw operator cosine — {slug} ({arm} arm, L19)",
                args.fig_dir / f"operator_cosine_heatmap_{arm}_{slug}.png",
            )
            _arm_extra_figs(args, model, arm, slug, transfer, opcomp, r4_live=_r4_live(model))
        layer_sweep_fig(
            args.out_dir,
            model,
            regimes_for[model],
            args.fig_dir / f"layer_sweep_{slug}_context.png",
        )
        if _r4_live(model):
            hero_paired_layer_sweep(args.out_dir, args.parent_eval_dir, args.fig_dir, model)
            tf_companion_panel(args.out_dir, matched_row_dir, args.fig_dir, model)
            matched_row_ceiling_panel(args.out_dir, matched_row_dir, args.fig_dir, model)

    # Story yield table (digest of the Phase-1 reports) + per-model coverage
    yields = {}
    for model in c.MODELS:
        rep = _load(args.stories_dir / f"story_yield_{model}.json")
        if rep:
            yields[model] = {k: rep[k] for k in ("n_kept", "yield_ok", "counts_main") if k in rep}
        if model in halted:
            yields.setdefault(model, {})["story_regime"] = "halted (per-model yield floor)"
    lattice["story_yield"] = yields
    lattice["r3_halted_models"] = sorted(halted)
    if c.HAS_R4:
        lattice["story_paired_yield"] = _paired_yield_digest(args.stories_dir)
        lattice["r4_halted"] = bool(args.no_r4)

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

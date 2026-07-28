#!/usr/bin/env python
"""Issue #1092 `crossed-core-sae` phase E (VM-side): figures from the phase-D digests.

Inputs: the `out/` digest tree the dispatcher wrote (harvested from HF or local
`data/issue_1092/crossed_core_sae/out`). Outputs: PNG+PDF+meta under
`figures/issue_1092/crossed_core_sae/` (override with --fig-dir for smoke).

Hero figures (plan section 6, v13 rubric amendment): (1) per-feature scatter
prefix-share x cross-query consistency with judged features colored by
speaker_property CLASS (5-way); (2) tail-composition bars — per-class
speaker_property rates per judged set (identity_disposition is the headline
class; language / register_style shown separately, never pooled), bootstrap
CIs. Plus the v13 sink/massive-activation map figure (per-position sink rate +
top rogue dims + gamma). Exploratory dump: share histograms + selection-null
lines, per-arm per-feature R^2 distributions, four-object matched-table R^2
bars (+ identity+bias line), |cos(W_dec, r_B)| tail curves vs the
selection-symmetric null (raw + centered), induced vs independently-fit
averaged comparison.
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
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

CELLS = ("cell_inst_own", "cell_pre_own")

# ONE color = ONE meaning across every figure: speaker_property class -> palette
# role (paper_palette_role accepts only accent/baseline/control/neutral/primary).
SPEAKER_CLASSES = ("identity_disposition", "language", "register_style", "none", "unclear")
CLASS_ROLE = {
    "identity_disposition": "accent",
    "language": "primary",
    "register_style": "control",
    "none": "baseline",
    "unclear": "neutral",
}


def _bootstrap_rate_ci(flags: np.ndarray, n_draws: int = 10_000, seed: int = 0):
    if flags.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = flags[rng.integers(0, flags.size, size=(n_draws, flags.size))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def fig_hero_scatter(out_dir: Path, fig_dir: Path, labels: dict, cell: str) -> None:
    z = np.load(out_dir / f"perfeature_join_{cell}.npz", allow_pickle=True)
    sp = z["share_prefix"]
    cq = z["cross_query_consistency_mean"]
    feats = z["feats"]
    fin = np.isfinite(sp) & np.isfinite(cq)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sp[fin], cq[fin], s=4, alpha=0.15, color=paper_palette_role("neutral"), lw=0)
    speaker = labels.get("speaker", {}).get("labels", {})
    buckets: dict[str, tuple[list, list]] = {c: ([], []) for c in SPEAKER_CLASSES}
    for i in np.where(fin)[0]:
        lab = speaker.get(str(int(feats[i])))
        if lab is None:
            continue
        xs, ys = buckets[lab["speaker_property"]]
        xs.append(float(sp[i]))
        ys.append(float(cq[i]))
    for cls in SPEAKER_CLASSES:
        xs, ys = buckets[cls]
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            s=18 if cls == "identity_disposition" else 14,
            color=paper_palette_role(CLASS_ROLE[cls]),
            edgecolors="black" if cls == "unclear" else "none",
            linewidths=0.4 if cls == "unclear" else 0.0,
            label=f"judged: {cls}",
        )
    order = np.argsort(np.nan_to_num(sp, nan=-1))[::-1][:5]
    for i in order:
        ax.annotate(str(int(feats[i])), (sp[i], cq[i]), fontsize=6, alpha=0.8)
    ax.set_xlabel("prefix variance share (per feature)")
    ax.set_ylabel("cross-query consistency at context-end (mean over prefixes)")
    ax.set_title(f"{cell}: prefix-share vs cross-query consistency")
    ax.legend(loc="best", fontsize=7)
    savefig_paper(fig, f"hero_scatter_{cell}", dir=fig_dir)
    plt.close(fig)


def fig_tail_bars(out_dir: Path, fig_dir: Path, labels_payload: dict) -> None:
    """Per-class speaker_property rates per judged set (grouped bars, bootstrap
    CIs). identity_disposition is the headline class; language/register_style
    are shown as their OWN bars, never pooled (v13 rubric amendment)."""
    speaker = labels_payload.get("speaker", {}).get("labels", {})
    sets = labels_payload.get("sets", {})
    names = [
        ("tail_prefix", "top prefix-share"),
        ("ctrl_activity_matched", "activity-matched"),
        ("ctrl_query_tail", "top query-share"),
    ]
    set_classes: list[tuple[str, list[str]]] = []
    for key, disp in names:
        cls = [speaker[str(f)]["speaker_property"] for f in sets.get(key, []) if str(f) in speaker]
        if cls:
            set_classes.append((f"{disp}\n(n={len(cls)})", cls))
    if not set_classes:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    n_sets = len(set_classes)
    width = 0.8 / len(SPEAKER_CLASSES)
    xbase = np.arange(n_sets)
    for j, cls_name in enumerate(SPEAKER_CLASSES):
        rates, los, his = [], [], []
        for _, cls in set_classes:
            flags = np.array([1.0 if c == cls_name else 0.0 for c in cls])
            r = float(flags.mean())
            lo, hi = _bootstrap_rate_ci(flags)
            rates.append(r)
            los.append(max(0.0, r - lo) if np.isfinite(lo) else 0.0)
            his.append(max(0.0, hi - r) if np.isfinite(hi) else 0.0)
        ax.bar(
            xbase + (j - (len(SPEAKER_CLASSES) - 1) / 2) * width,
            rates,
            width=width * 0.95,
            yerr=[los, his],
            capsize=2,
            color=paper_palette_role(CLASS_ROLE[cls_name]),
            label=cls_name,
        )
    ax.set_xticks(xbase)
    ax.set_xticklabels([lbl for lbl, _ in set_classes], fontsize=8)
    hl = labels_payload.get("headline") or {}
    d = (hl.get("delta") or {}).get("delta")
    ci = (hl.get("delta") or {}).get("ci95")
    sub = (
        f"identity_disposition Delta={d:.3f} CI95={ci}"
        if isinstance(d, float) and ci
        else "identity_disposition Delta: insufficient labels"
    )
    ax.set_ylabel("speaker_property class rate")
    ax.set_title(f"judged tail composition — {sub}")
    ax.legend(fontsize=7, ncol=2)
    savefig_paper(fig, "hero_tail_composition", dir=fig_dir)
    plt.close(fig)


def fig_sink_map(sinkmap_root: Path, fig_dir: Path, cell: str) -> None:
    """v13 sink/massive-activation map figure: per-position sink rate + top
    rogue dims by |x|max, gamma in the title. Skips (with a print) when the
    cell's map artifacts are absent."""
    npz_path = sinkmap_root / f"sink_map_{cell}.npz"
    json_path = sinkmap_root / f"sink_map_{cell}.json"
    if not npz_path.exists() or not json_path.exists():
        print(f"[figs] sink map absent for {cell} under {sinkmap_root} — skipped", flush=True)
        return
    z = np.load(npz_path, allow_pickle=True)
    rec = json.loads(json_path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    rate = z["pos_sink"] / np.maximum(z["pos_occ"], 1.0)
    axes[0].bar(np.arange(rate.size), rate, color=paper_palette_role("primary"), width=1.0)
    for p in rec["sink_positions"]:
        if p < rate.size:
            axes[0].axvline(p, color=paper_palette_role("accent"), lw=0.6, alpha=0.6)
    axes[0].set_xlabel("absolute token position")
    axes[0].set_ylabel("sink rate (norm > 10x row median)")
    axes[0].set_title(
        f"per-position sink rate (accent = map sink set, n={len(rec['sink_positions'])})"
    )
    absmax = z["dim_absmax"]
    top = np.argsort(absmax)[::-1][:20]
    axes[1].bar([str(int(d)) for d in top], absmax[top], color=paper_palette_role("control"))
    axes[1].tick_params(axis="x", rotation=70, labelsize=6)
    axes[1].set_xlabel("hidden dim (top 20 by |x| max)")
    axes[1].set_ylabel("max |activation| (layer 19)")
    axes[1].set_title(f"massive-activation dims — gamma={rec['gamma_layer19_all_tokens']:.3f}")
    fig.suptitle(
        f"{cell}: sink/massive-activation map (exclusion source: {rec['exclusion_source']})",
        fontsize=9,
    )
    savefig_paper(fig, f"sink_map_{cell}", dir=fig_dir)
    plt.close(fig)


def fig_share_hists(out_dir: Path, fig_dir: Path, maps: dict, cell: str) -> None:
    z = np.load(out_dir / f"anova_shares_{cell}.npz", allow_pickle=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    sel = maps["cells"][cell]["anova_selection"]
    for ax, key, axis in zip(
        axes, ("share_prefix", "share_query", "share_inter"), ("prefix", "query", None), strict=True
    ):
        v = z[key]
        v = v[np.isfinite(v)]
        ax.hist(v, bins=50, color=paper_palette_role("primary"))
        ax.set_yscale("log")
        ax.set_title(key)
        if axis is not None and axis in sel:
            ax.axvline(sel[axis]["obs_max"], color=paper_palette_role("accent"), ls="--")
    fig.suptitle(
        f"{cell}: per-feature variance shares (dashed = observed max; "
        "selection-symmetric p in maps_summary)"
    )
    savefig_paper(fig, f"share_hists_{cell}", dir=fig_dir)
    plt.close(fig)


def fig_r2_table(fig_dir: Path, maps: dict, cell: str) -> None:
    row = maps["cells"][cell]
    tbl = row["four_object_table"]["pooled_r2 (matched target = pooled-answer mean)"]
    names = list(tbl.keys())
    vals = [tbl[n] if tbl[n] is not None else np.nan for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(names)), vals, color=paper_palette_role("primary"))
    idb = row["ctx"]["identity_bias"]
    if idb.get("applicable"):
        ax.axhline(
            idb["pooled_r2"],
            color=paper_palette_role("baseline"),
            ls=":",
            label="identity+bias (ctx, intersection)",
        )
        ax.legend(fontsize=7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("held-out pooled R^2 (pooled-answer mean)")
    ax.set_title(f"{cell}: four-object matched table (same target/folds per arm)")
    savefig_paper(fig, f"four_object_r2_{cell}", dir=fig_dir)
    plt.close(fig)


def fig_r2_perfeature(out_dir: Path, fig_dir: Path, cell: str) -> None:
    z = np.load(out_dir / f"perfeature_join_{cell}.npz", allow_pickle=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, role in (("r2_ctx", "primary"), ("r2_pre", "accent"), ("r2_bare", "control")):
        v = z[key]
        v = v[np.isfinite(v)]
        v = np.clip(v, -1, 1)
        ax.hist(v, bins=60, histtype="step", label=key, color=paper_palette_role(role))
    ax.set_yscale("log")
    ax.set_xlabel("per-feature held-out R^2 (clipped to [-1, 1])")
    ax.legend(fontsize=8)
    ax.set_title(f"{cell}: per-arm per-feature R^2")
    savefig_paper(fig, f"r2_perfeature_{cell}", dir=fig_dir)
    plt.close(fig)


def fig_rb_tails(out_dir: Path, fig_dir: Path, cell: str) -> None:
    z = np.load(out_dir / f"perfeature_join_{cell}.npz", allow_pickle=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, nkey, role, lab in (
        ("rb_cos_max", "rb_null_draws_max", "primary", "raw"),
        ("rb_cos_max_centered", "rb_null_draws_max_centered", "accent", "mean-centered"),
    ):
        obs = np.sort(z[key])[::-1]
        null = z[nkey].astype(np.float32)
        p95 = np.nanpercentile(np.nanmax(null, axis=1), 95)
        ax.plot(
            obs[: min(500, obs.size)], color=paper_palette_role(role), label=f"obs max-cos ({lab})"
        )
        ax.axhline(
            p95,
            color=paper_palette_role(role),
            ls="--",
            lw=0.8,
            label=f"null p95 of per-draw max ({lab})",
        )
    ax.set_xlabel("feature rank")
    ax.set_ylabel("|cos(W_dec, r_B)| (max over 3 traits)")
    ax.legend(fontsize=7)
    ax.set_title(f"{cell}: decoder-vs-r_B alignment tails (selection-symmetric null)")
    savefig_paper(fig, f"rb_cos_tails_{cell}", dir=fig_dir)
    plt.close(fig)


def fig_averaged_compare(fig_dir: Path, maps: dict, cell: str) -> None:
    row = maps["cells"][cell]
    vals = {
        "induced (PRIMARY)": row["induced_averaged"]["pooled_r2_mean"],
        "independently fit\n(SECONDARY, n<<d)": row["independently_fit_averaged"]["pooled_r2_mean"],
        "per-row context map": row["ctx"]["pooled_r2_mean"],
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(vals.keys()), list(vals.values()), color=paper_palette_role("primary"))
    ax.set_ylabel("pooled R^2")
    ax.set_title(f"{cell}: averaged-grain reads")
    savefig_paper(fig, f"averaged_compare_{cell}", dir=fig_dir)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-root", type=Path, default=Path("data/issue_1092/crossed_core_sae/out"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_1092/crossed_core_sae"))
    ap.add_argument(
        "--sinkmap-root",
        type=Path,
        default=None,
        help="sink-map artifact dir (default: <in-root>/../sink_map)",
    )
    args = ap.parse_args(argv)
    set_paper_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    sinkmap_root = args.sinkmap_root or (args.in_root.parent / "sink_map")
    maps = json.loads((args.in_root / "maps_summary.json").read_text())
    labels_payload = json.loads((args.in_root / "feature_labels.json").read_text())
    for cell in CELLS:
        if cell not in maps.get("cells", {}):
            continue
        fig_hero_scatter(args.in_root, args.fig_dir, labels_payload, cell)
        fig_share_hists(args.in_root, args.fig_dir, maps, cell)
        fig_r2_table(args.fig_dir, maps, cell)
        fig_r2_perfeature(args.in_root, args.fig_dir, cell)
        fig_rb_tails(args.in_root, args.fig_dir, cell)
        fig_averaged_compare(args.fig_dir, maps, cell)
        fig_sink_map(sinkmap_root, args.fig_dir, cell)
    if "speaker" in labels_payload:
        fig_tail_bars(args.in_root, args.fig_dir, labels_payload)
    print(f"[figs] wrote figures to {args.fig_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

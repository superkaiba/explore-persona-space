"""Issue #1689 `derived-vs-free-answer-map` — all round figures (plan v8 s6).

Reads the per-unit JSONs written by issue1689_derived_vs_free.py (items 1-6)
and issue1689_context_map_structure.py (items 7-8), plus the crossmodel
out-root (item 9), and renders:

  1. HERO: per-pair verdict heatmap (lattice layout, per model x arm) with the
     parent rung-reached index as cell annotations.
  2. g1 (and g2) vs parent rung_reached scatter (+ concordance rho) and
     R2(B_derived) vs R2(B_free) scatter colored by rung.
  3. Truncation-rank sensitivity spaghetti (estimation-vs-model attribution).
  4. Operator cosine vs rotation-null band (observed vs per-unit null p97.5).
  5. Item 7: weakest-class heatmap; ||M-I||_F/||M||_F; M-I spectra; polar
     distance; subspace-overlap matrix; correction-vs-diff-of-means cosines.
  6. Item 8: rank-reached heatmap (companion to rung-reached).
  7. Item 9: cross-model verdict/class/rank heatmaps + uniformity overlap.

Every figure is written under --out-figs; missing inputs are skipped with a
log line (subset-tolerant — the smoke runs on one unit). Figures use the
project paper style; no CI errorbars are drawn (no xerr/yerr offsets).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.analysis.paper_plots import set_paper_style  # noqa: E402
from scripts.issue1689_common import CONDITION_TABLE  # noqa: E402

VERDICT_ORDER = [
    "shared_readout_supported",
    "readout_changed",
    "transfer_map_insufficient",
    "free_map_uninformative",
    "invalid",
]
VERDICT_COLORS = {
    "shared_readout_supported": "#2b8a3e",
    "readout_changed": "#e8a13a",
    "transfer_map_insufficient": "#c0392b",
    "free_map_uninformative": "#9aa0a6",
    "invalid": "#dddddd",
}
CLASS_SHORT = {
    "translation": "T",
    "trans_scalar": "Ts",
    "trans_rotation": "TR",
    "full_affine": "F",
}
COND_ORDER = sorted({c.slug for c in CONDITION_TABLE})
RANK_LABELS = ("r32", "r128", "r512", "effrank")


def _load_units(root: Path) -> list[dict]:
    pairs_dir = root / "pairs"
    if not pairs_dir.exists():
        print(f"[dvf-figs] SKIP {root} (no pairs dir)", flush=True)
        return []
    units = []
    for p in sorted(pairs_dir.glob("*.json")):
        u = json.loads(p.read_text())
        if "error" not in u:
            units.append(u)
    print(f"[dvf-figs] loaded {len(units)} complete units from {root}", flush=True)
    return units


def _parent_rungs(parent_ladder_dir: Path) -> dict:
    out = {}
    for p in sorted(parent_ladder_dir.glob("ladder_*_L19.json")):
        model = p.stem.removeprefix("ladder_").removesuffix("_L19")
        ladder = json.loads(p.read_text())
        for pair_key, arms in ladder.get("pairs", {}).items():
            for arm, res in arms.items():
                if isinstance(res, dict) and "rung_reached_point" in res:
                    out[(model, pair_key, arm)] = int(res["rung_reached_point"])
    return out


def _save(fig, out_figs: Path, name: str) -> None:
    out_figs.mkdir(parents=True, exist_ok=True)
    path = out_figs / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[dvf-figs] wrote {path}", flush=True)


def _cond_matrix(units: list[dict], value_fn) -> dict:
    """Group units into {(model_key, arm): (matrix, annot)} over COND_ORDER."""
    groups: dict = {}
    idx = {c: i for i, c in enumerate(COND_ORDER)}
    for u in units:
        model_key = (
            u["src_model"] if not u["cross_model"] else f"{u['src_model']}->{u['tgt_model']}"
        )
        gk = (model_key, u["arm"])
        if gk not in groups:
            groups[gk] = (
                np.full((len(COND_ORDER), len(COND_ORDER)), np.nan),
                np.full((len(COND_ORDER), len(COND_ORDER)), "", dtype=object),
            )
        i, j = idx.get(u["src_cond"]), idx.get(u["tgt_cond"])
        if i is None or j is None:
            continue
        val, annot = value_fn(u)
        groups[gk][0][i, j] = val
        groups[gk][1][i, j] = annot
    return groups


def _heatmap_grid(
    groups: dict, title: str, out_figs: Path, name: str, cmap, vmin, vmax, cbar_label
):
    keys = sorted(groups)
    if not keys:
        print(f"[dvf-figs] SKIP {name} (no data)", flush=True)
        return
    ncol = min(2, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5 * ncol, 8.5 * nrow), squeeze=False)
    for ax_i, gk in enumerate(keys):
        ax = axes[ax_i // ncol][ax_i % ncol]
        mat, annot = groups[gk]
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_xticklabels(COND_ORDER, rotation=90, fontsize=5)
        ax.set_yticks(range(len(COND_ORDER)))
        ax.set_yticklabels(COND_ORDER, fontsize=5)
        ax.set_xlabel("target condition")
        ax.set_ylabel("source condition")
        ax.set_title(f"{gk[0]} | {gk[1]} arm", fontsize=9)
        for i in range(len(COND_ORDER)):
            for j in range(len(COND_ORDER)):
                if annot[i, j]:
                    ax.text(j, i, str(annot[i, j]), ha="center", va="center", fontsize=4)
        fig.colorbar(im, ax=ax, shrink=0.7, label=cbar_label)
    for ax_i in range(len(keys), nrow * ncol):
        axes[ax_i // ncol][ax_i % ncol].axis("off")
    fig.suptitle(title, fontsize=11)
    _save(fig, out_figs, name)


def fig1_verdict_heatmap(units, rungs, out_figs, *, prefix: str = "fig1_verdict") -> None:
    vidx = {v: k for k, v in enumerate(VERDICT_ORDER)}

    def value_fn(u):
        rung = rungs.get((u["src_model"], u["pair_key"], u["arm"]))
        return vidx.get(u["verdict"], len(VERDICT_ORDER) - 1), ("" if rung is None else str(rung))

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap([VERDICT_COLORS[v] for v in VERDICT_ORDER])
    groups = _cond_matrix(units, value_fn)
    keys = sorted(groups)
    if not keys:
        print(f"[dvf-figs] SKIP {prefix} (no data)", flush=True)
        return
    ncol = min(2, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5 * ncol, 8.5 * nrow), squeeze=False)
    for ax_i, gk in enumerate(keys):
        ax = axes[ax_i // ncol][ax_i % ncol]
        mat, annot = groups[gk]
        ax.imshow(mat, cmap=cmap, vmin=-0.5, vmax=len(VERDICT_ORDER) - 0.5)
        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_xticklabels(COND_ORDER, rotation=90, fontsize=5)
        ax.set_yticks(range(len(COND_ORDER)))
        ax.set_yticklabels(COND_ORDER, fontsize=5)
        ax.set_title(f"{gk[0]} | {gk[1]} arm (annot = parent rung reached)", fontsize=9)
        for i in range(len(COND_ORDER)):
            for j in range(len(COND_ORDER)):
                if annot[i, j]:
                    ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=4)
    for ax_i in range(len(keys), nrow * ncol):
        axes[ax_i // ncol][ax_i % ncol].axis("off")
    handles = [Patch(color=VERDICT_COLORS[v], label=v) for v in VERDICT_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7)
    fig.suptitle(
        "Derived-vs-free verdict per ordered pair (max over W_S+ truncation ranks)", fontsize=11
    )
    _save(fig, out_figs, f"{prefix}_heatmap.png")


def fig2_concordance(units, rungs, out_figs) -> None:
    from scipy.stats import spearmanr

    within = [u for u in units if not u["cross_model"]]
    arms = sorted({u["arm"] for u in within})
    if not within:
        print("[dvf-figs] SKIP fig2 (no within-model units)", flush=True)
        return
    fig, axes = plt.subplots(
        2, max(len(arms), 1), figsize=(6 * max(len(arms), 1), 9), squeeze=False
    )
    for a_i, arm in enumerate(arms):
        rows = [
            (rungs.get((u["src_model"], u["pair_key"], arm)), u) for u in within if u["arm"] == arm
        ]
        rows = [(r, u) for r, u in rows if r is not None]
        if not rows:
            continue
        rung = np.array([r for r, _ in rows], dtype=float)
        g1 = np.array([u["g1"] for _, u in rows], dtype=float)
        g2 = np.array([u["g2"] for _, u in rows], dtype=float)
        informative = np.array([u["verdict"] != "free_map_uninformative" for _, u in rows])
        ax = axes[0][a_i]
        jitter = (np.random.default_rng(0).uniform(-0.15, 0.15, len(rung))) if len(rung) else 0
        ax.scatter(
            rung[informative] + np.asarray(jitter)[informative],
            g1[informative],
            s=14,
            alpha=0.7,
            label="g1",
        )
        ax.scatter(
            rung[informative] + np.asarray(jitter)[informative],
            g2[informative],
            s=14,
            alpha=0.5,
            marker="x",
            label="g2",
        )
        ax.axhline(0.0, color="k", lw=0.8)
        rho = (
            spearmanr(rung[informative], g1[informative]).statistic
            if informative.sum() >= 2
            else float("nan")
        )
        ax.set_title(f"{arm} arm: g vs parent rung (rho(g1)={rho:.3f}, n={int(informative.sum())})")
        ax.set_xlabel("parent rung_reached")
        ax.set_ylabel("g = R2(derived) - 0.9 R2(free)")
        ax.legend(fontsize=7)
        ax = axes[1][a_i]
        r2f = np.array([u["r2_b_free"] for _, u in rows], dtype=float)
        r2d = np.array([u["r2_b_derived_max"] for _, u in rows], dtype=float)
        sc = ax.scatter(r2f, r2d, c=rung, cmap="viridis", s=14)
        lo = float(np.nanmin([r2f.min(), r2d.min(), 0.0]))
        hi = float(np.nanmax([r2f.max(), r2d.max(), 1.0]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
        ax.plot([lo, hi], [0.9 * lo, 0.9 * hi], "r:", lw=0.8, label="0.9x bar")
        ax.set_xlabel("R2(B_free) pooled")
        ax.set_ylabel("R2(B_derived) max-over-ranks")
        ax.legend(fontsize=7)
        fig.colorbar(sc, ax=ax, shrink=0.8, label="parent rung")
    fig.suptitle("Concordance: derived-vs-free gap vs parent ladder rung", fontsize=11)
    _save(fig, out_figs, "fig2_concordance.png")


def fig3_truncation_sensitivity(units, out_figs, *, prefix: str = "fig3") -> None:
    if not units:
        print(f"[dvf-figs] SKIP {prefix} (no units)", flush=True)
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(RANK_LABELS))
    mat = []
    for u in units:
        ys = [u["r2_pooled"].get(f"b_derived_{lab}", np.nan) for lab in RANK_LABELS]
        mat.append(ys)
        ax.plot(xs, ys, color="steelblue", alpha=0.15, lw=0.7)
    med = np.nanmedian(np.asarray(mat, dtype=float), axis=0)
    ax.plot(xs, med, color="crimson", lw=2.2, label="median")
    ax.set_xticks(xs)
    ax.set_xticklabels(RANK_LABELS)
    ax.set_xlabel("W_S+ truncation rank")
    ax.set_ylabel("pooled held-out R2(B_derived)")
    ax.set_title(f"Truncation-rank sensitivity ({len(units)} units)")
    ax.legend()
    _save(fig, out_figs, f"{prefix}_truncation_sensitivity.png")


def fig4_operator_cosine(units, out_figs) -> None:
    rows = []
    for u in units:
        op = u.get("operator_read") or {}
        nul = op.get("rotation_null") or {}
        for name, stats in nul.items():
            if isinstance(stats, dict) and "null_p975" in stats:
                rows.append((name, stats.get("observed"), stats["null_p975"]))
    if not rows:
        print("[dvf-figs] SKIP fig4 (no rotation-null blocks — run --phase nulls)", flush=True)
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    kinds = sorted({r[0] for r in rows})
    for k_i, kind in enumerate(kinds):
        obs = np.array([r[1] for r in rows if r[0] == kind], dtype=float)
        p975 = np.array([r[2] for r in rows if r[0] == kind], dtype=float)
        ax.scatter(p975, obs, s=12, alpha=0.6, label=f"{kind} (n={len(obs)})")
    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="observed = null p97.5")
    ax.set_xlabel("two-sided rotation null p97.5")
    ax.set_ylabel("observed raw operator cosine vs B_free")
    ax.set_title("Operator cosine vs rotation-null band (per unit)")
    ax.legend(fontsize=6)
    _save(fig, out_figs, "fig4_operator_cosine.png")


def fig5_structure(cms_units, out_figs, *, prefix: str = "fig5") -> None:
    if not cms_units:
        print(f"[dvf-figs] SKIP {prefix} (no structure units)", flush=True)
        return
    class_names = sorted(
        {u["weakest_class_point"] for u in cms_units},
        key=lambda c: (c != "translation", c != "trans_scalar", c != "trans_rotation", c),
    )
    cidx = {c: i for i, c in enumerate(class_names)}

    def value_fn(u):
        w = u["weakest_class_point"]
        return cidx[w], CLASS_SHORT.get(w, w.removeprefix("rank_"))

    groups = _cond_matrix(cms_units, value_fn)
    _heatmap_grid(
        groups,
        "Weakest sufficient context-map class per pair (annot: class)",
        out_figs,
        f"{prefix}a_weakest_class_heatmap.png",
        "plasma",
        -0.5,
        len(class_names) - 0.5,
        "class index (ladder order)",
    )
    groups = _cond_matrix(
        cms_units,
        lambda u: (u["distance_from_identity"]["fro_ratio_m_minus_i_over_m"], ""),
    )
    _heatmap_grid(
        groups,
        "||M - I||_F / ||M||_F per pair",
        out_figs,
        f"{prefix}b_fro_ratio_heatmap.png",
        "magma",
        None,
        None,
        "fro ratio",
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    eff = [u["distance_from_identity"]["eff_rank_m_minus_i"] for u in cms_units]
    ax.hist(eff, bins=30, color="steelblue")
    ax.set_xlabel("effective rank of M - I")
    ax.set_title("M-I effective rank")
    ax = axes[1]
    pol = [u["distance_from_identity"]["polar_factor_distance"] for u in cms_units]
    ax.hist(pol, bins=30, color="darkorange")
    ax.set_xlabel("||M - Q_polar||_F / ||M||_F")
    ax.set_title("Polar-factor distance (rotation-like vs shear/scale)")
    ax = axes[2]
    top1 = [u["diff_of_means_alignment"]["top8_output_dir_abs_cos"][0] for u in cms_units]
    ax.hist(top1, bins=30, color="seagreen")
    ax.set_xlabel("|cos(top-1 output dir of M-I, diff-of-means)|")
    ax.set_title("Correction direction vs mean shift")
    _save(fig, out_figs, f"{prefix}c_structure_summaries.png")


def fig5d_overlap(cms_root: Path, out_figs, *, prefix: str = "fig5d") -> None:
    p = cms_root / "subspace_overlap.json"
    if not p.exists():
        print(f"[dvf-figs] SKIP {prefix} (no subspace_overlap.json)", flush=True)
        return
    d = json.loads(p.read_text())
    rows = d.get("unit_pairs", [])
    if not rows:
        print(f"[dvf-figs] SKIP {prefix} (empty unit_pairs)", flush=True)
        return
    keys = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
    kidx = {k: i for i, k in enumerate(keys)}
    k = d["k_primary"]
    mat = np.full((len(keys), len(keys)), np.nan)
    for r in rows:
        i, j = kidx[r["a"]], kidx[r["b"]]
        mat[i, j] = mat[j, i] = r[f"left_overlap_k{k}"]
    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 0.25), max(5, len(keys) * 0.25)))
    im = ax.imshow(mat, cmap="viridis")
    null = (
        d["random_subspace_null"][str(k)]
        if str(k) in d.get("random_subspace_null", {})
        else d["random_subspace_null"].get(k, {})
    )
    ax.set_title(
        f"M-I top-{k} LEFT subspace overlap (mean sq cos); "
        f"null p97.5={null.get('null_p975', float('nan')):.4f}",
        fontsize=8,
    )
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90, fontsize=4)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=4)
    fig.colorbar(im, ax=ax, shrink=0.75)
    _save(fig, out_figs, f"{prefix}_subspace_overlap.png")


def fig6_rank_reached(cms_units, out_figs, *, prefix: str = "fig6") -> None:
    eligible = [
        u
        for u in cms_units
        if u.get("rank_rung", {}).get("eligible") and "k_reached_ctx" in u.get("rank_rung", {})
    ]
    if not eligible:
        print(f"[dvf-figs] SKIP {prefix} (no rank-rung units)", flush=True)
        return

    def value_fn(u):
        kc = u["rank_rung"]["k_reached_ctx"]
        ka = u["rank_rung"]["k_reached_ans"]
        val = np.log2(kc) if kc else np.nan
        return val, f"{kc or 'x'}/{ka or 'x'}"

    groups = _cond_matrix(eligible, value_fn)
    _heatmap_grid(
        groups,
        "Minimal sufficient rank k (annot: ctx/ans; x = not reached at k<=128)",
        out_figs,
        f"{prefix}_rank_reached_heatmap.png",
        "cividis",
        0,
        7,
        "log2(k_reached_ctx)",
    )


K_BAND_ORDER = ("pooled", "k<32", "32-127", "128-511", ">=512")


def _load_paired_rows(paired_csv: Path) -> list[dict]:
    import csv

    with open(paired_csv) as fh:
        return list(csv.DictReader(fh))


def fig8_paired_calibration(rows, out_figs, *, prefix: str = "fig8_paired_calibration") -> None:
    """HERO (plan v10 s6 fig 1): rung-1 shared-readout-supported rate, ambient
    vs reduced, pooled + per k-band, vs the registered 50% line."""
    r1 = [r for r in rows if r.get("battery") == "dvf_within" and r.get("parent_rung") == "1"]
    if not r1:
        print("[dvf-figs] SKIP fig8 (no rung-1 paired rows)", flush=True)
        return
    rates: dict[str, dict[str, float]] = {}
    for basis in ("ambient", "reduced"):
        for stratum in K_BAND_ORDER:
            sub = [r for r in r1 if stratum == "pooled" or r.get("k_band") == stratum]
            kept = [r for r in sub if r[f"{basis}_verdict"] != "free_map_uninformative"]
            n_sup = sum(r[f"{basis}_verdict"] == "shared_readout_supported" for r in kept)
            rates.setdefault(basis, {})[stratum] = (n_sup / len(kept)) if kept else np.nan
    x = np.arange(len(K_BAND_ORDER))
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(x - 0.2, [rates["ambient"][s] for s in K_BAND_ORDER], 0.38, label="ambient (parent)")
    ax.bar(x + 0.2, [rates["reduced"][s] for s in K_BAND_ORDER], 0.38, label="reduced (well-posed)")
    ax.axhline(0.5, color="k", ls="--", lw=1, label="50% calibration line")
    ax.set_xticks(x, K_BAND_ORDER)
    ax.set_ylabel("rung-1 shared-readout-supported rate\n(class-0-excluded)")
    ax.set_xlabel("k-band")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    _save(fig, out_figs, f"{prefix}.png")


def fig9_verdict_flip(rows, out_figs, *, prefix: str = "fig9_verdict_flip") -> None:
    """Paired contrast (plan v10 s6 fig 3): ambient class -> reduced class counts."""
    dvf = [r for r in rows if r.get("battery") in ("dvf_within", "xm_dvf")]
    if not dvf:
        print("[dvf-figs] SKIP fig9 (no paired dvf rows)", flush=True)
        return
    mat = np.zeros((len(VERDICT_ORDER), len(VERDICT_ORDER)))
    idx = {v: i for i, v in enumerate(VERDICT_ORDER)}
    for r in dvf:
        a, b = r.get("ambient_verdict"), r.get("reduced_verdict")
        if a in idx and b in idx:
            mat[idx[a], idx[b]] += 1
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="Blues")
    short = [v.replace("_", "\n") for v in VERDICT_ORDER]
    ax.set_xticks(range(len(VERDICT_ORDER)), short, fontsize=7)
    ax.set_yticks(range(len(VERDICT_ORDER)), short, fontsize=7)
    ax.set_xlabel("reduced (well-posed) verdict")
    ax.set_ylabel("ambient (parent) verdict")
    for i in range(len(VERDICT_ORDER)):
        for j in range(len(VERDICT_ORDER)):
            if mat[i, j] > 0:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, out_figs, f"{prefix}.png")


def fig10_effrank_paired(rows, out_figs, *, prefix: str = "fig10_effrank_paired") -> None:
    """H-effrank (plan v10 s6 fig 6): eff-rank(M-I)/fit_dim, ambient vs reduced."""
    cms = [
        r
        for r in rows
        if r.get("battery") in ("cms_within", "cms_xm") and r.get("ambient_eff_rank_frac")
    ]
    if not cms:
        print("[dvf-figs] SKIP fig10 (no paired cms rows)", flush=True)
        return
    a = np.array([float(r["ambient_eff_rank_frac"]) for r in cms])
    b = np.array([float(r["reduced_eff_rank_frac"]) for r in cms])
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(a, b, s=12, alpha=0.6)
    lim = (0, 1.02)
    ax.plot(lim, lim, color="k", lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("ambient eff-rank(M-I) / d")
    ax.set_ylabel("reduced eff-rank(M-I) / k_unit")
    ax.set_title(f"n={len(cms)}; below diagonal = deflation (H-effrank)")
    _save(fig, out_figs, f"{prefix}.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dvf-root", type=Path, default=Path("eval_results/issue_1689/derived_vs_free_B")
    )
    ap.add_argument(
        "--cms-root", type=Path, default=Path("eval_results/issue_1689/context_map_structure")
    )
    ap.add_argument(
        "--crossmodel-root", type=Path, default=Path("eval_results/issue_1689/crossmodel_pairs")
    )
    ap.add_argument(
        "--crossmodel-struct-dirname",
        type=str,
        default=None,
        help="structure subdir under --crossmodel-root (default: crossmodel_structure, "
        "with a crossmodel_structure_wellposed fallback probe)",
    )
    ap.add_argument(
        "--parent-ladder-dir", type=Path, default=Path("eval_results/issue_1689/ladder")
    )
    ap.add_argument("--out-figs", type=Path, default=Path("figures/issue_1689/derived_vs_free"))
    ap.add_argument(
        "--paired-digest",
        type=Path,
        default=None,
        help="paired ambient-vs-reduced delta digest CSV (wellposed round): "
        "renders the fig8/fig9/fig10 paired set",
    )
    args = ap.parse_args()

    set_paper_style()
    rungs = _parent_rungs(args.parent_ladder_dir)
    dvf_units = _load_units(args.dvf_root)
    fig1_verdict_heatmap([u for u in dvf_units if not u["cross_model"]], rungs, args.out_figs)
    fig2_concordance(dvf_units, rungs, args.out_figs)
    fig3_truncation_sensitivity(dvf_units, args.out_figs)
    fig4_operator_cosine(dvf_units, args.out_figs)

    cms_units = _load_units(args.cms_root)
    fig5_structure(cms_units, args.out_figs)
    fig5d_overlap(args.cms_root, args.out_figs)
    fig6_rank_reached(cms_units, args.out_figs)

    # Item 9: battery units live at crossmodel_root, structure units under
    # crossmodel_root/<struct dirname> (distinct out-roots — unit-key
    # filenames would collide in one pairs/ dir).
    xm_battery = [u for u in _load_units(args.crossmodel_root) if "verdict" in u]
    if args.crossmodel_struct_dirname:
        xm_struct_root = args.crossmodel_root / args.crossmodel_struct_dirname
    else:
        xm_struct_root = args.crossmodel_root / "crossmodel_structure"
        wp = args.crossmodel_root / "crossmodel_structure_wellposed"
        if not xm_struct_root.exists() and wp.exists():
            xm_struct_root = wp
    xm_struct = [u for u in _load_units(xm_struct_root) if "weakest_class_point" in u]
    if xm_battery:
        fig1_verdict_heatmap(xm_battery, rungs, args.out_figs, prefix="fig7_crossmodel_verdict")
        fig3_truncation_sensitivity(xm_battery, args.out_figs, prefix="fig7_crossmodel_trunc")
    if xm_struct:
        fig5_structure(xm_struct, args.out_figs, prefix="fig7_crossmodel_struct")
        fig6_rank_reached(xm_struct, args.out_figs, prefix="fig7_crossmodel_rank")
        fig5d_overlap(xm_struct_root, args.out_figs, prefix="fig7_crossmodel_overlap")

    if args.paired_digest is not None and args.paired_digest.exists():
        paired_rows = _load_paired_rows(args.paired_digest)
        fig8_paired_calibration(paired_rows, args.out_figs)
        fig9_verdict_flip(paired_rows, args.out_figs)
        fig10_effrank_paired(paired_rows, args.out_figs)
    elif args.paired_digest is not None:
        print(f"[dvf-figs] SKIP paired figs (no digest at {args.paired_digest})", flush=True)
    print("[dvf-figs] done", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGABRT pointer. All outputs are
    # flushed/closed before this point; atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)

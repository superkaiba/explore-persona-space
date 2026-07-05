#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #923 mixed-forward-span-stitch figures (plan v9 §6).

Reads this round's JSONs (``eval_results/issue_923/mixed-forward-span-stitch/``)
PLUS the POOLED round's persisted results (``eval_results/issue_923/fits_pooled/``
— the comparison side, NOT refit) and renders (``mixed_`` prefix):

- HERO: L18 paired bars — Masked-context query-span (persisted) vs
  Mixed-forward query-span (new), and Pooled stitched / Matched-presentation
  stitched (persisted) vs Stitched-mixed (new), Pooled full-prompt for
  reference; family-bootstrap CIs; L18 null bands shaded (this round's bands
  for the new arms; the pooled round's for the persisted ones); the primary
  paired D + CI + §3 direction in the metadata.
- Per-held-out-family skill dots for every plotted bar.
- s1 dilution panel: per-family gap' (Stitched-mixed − Pooled stitched) vs the
  pooled full−stitched gaps, WildChat/ICL highlighted, short-family band.
- Per-cell predicted-vs-actual scatters (both new arms, L18).
- 28-layer curves (new arms solid + persisted references dashed).
- Betley fixed-λ ladder (new arms per-λ vs pooled arm_qry_iii per-λ).
- Dolly OOD panel; identity-check cosine histogram (identity_check_mix.json);
  paired-draw distributions for the four registered reads.

Per paper-plots policy: constrained layout (never tight_layout after a
colorbar), no on-plot annotation text, colorblind-safe palette.

Usage::

    uv run python scripts/issue923_fig_mixed.py \\
        --fits-dir eval_results/issue_923/mixed-forward-span-stitch \\
        --pooled-fits-dir eval_results/issue_923/fits_pooled \\
        --out-dir figures/issue_923
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# load_dotenv BEFORE numpy/torch importers so the shared-VM thread caps bind
# in-process (#847; tests/test_shared_vm_thread_caps.py).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue658_fit_predictors import RIDGE_LAMBDAS  # noqa: E402
from issue923_common import HF_DATA_REPO, HF_PREFIX_923, dump_json, load_json  # noqa: E402

ARM_LABELS = {
    "arm_qry_iii": "Masked-ctx query-span (pooled)",
    "arm_qry_mix": "Mixed-fwd query-span (new)",
    "arm_concat_i": "Pooled stitched",
    "arm_concat_iii": "Matched-pres stitched (pooled)",
    "arm_concat_mix": "Stitched-mixed (new)",
    "arm_full": "Full prompt (pooled)",
}
NEW_ARMS = ("arm_qry_mix", "arm_concat_mix")
HERO_ARMS = [
    "arm_qry_iii",
    "arm_qry_mix",
    "arm_concat_i",
    "arm_concat_iii",
    "arm_concat_mix",
    "arm_full",
]
PALETTE = ["#0173b2", "#de8f05", "#029e73", "#cc78bc", "#ca9161", "#949494", "#56b4e9"]
COLOR_NEW = PALETTE[0]
COLOR_REF = PALETTE[5]
SHORT_FAMILY_BAND = (-0.13, -0.10)  # pooled round's five short-family gap band (§3 s1)
S1_HIGHLIGHT = ("wildchat", "icl")


def _save(fig, out_dir: Path, name: str, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    fig.savefig(out_dir / f"{name}.pdf")
    dump_json(meta, out_dir / f"{name}.meta.json")
    plt.close(fig)


def _ci_err(entry: dict) -> tuple[float, float]:
    v = entry["skill"]
    return max(0.0, v - entry["ci95"][0]), max(0.0, entry["ci95"][1] - v)


def _verdict_meta(stats: dict) -> dict:
    """Headline verdict fields for figure metadata, mirroring a k2 skip."""
    out = {"verdict": stats.get("verdict")}
    if stats.get("paired_diff", {}).get("verdict_note"):
        out["verdict_note"] = stats["paired_diff"]["verdict_note"]
    if stats.get("verdict_skipped_reason"):
        out["verdict_skipped_reason"] = stats["verdict_skipped_reason"]
    return out


def _family_order(stats: dict, pooled_stats: dict, genre: str) -> list[str]:
    """Family tick labels in the persisted ARRAY index order (both rounds agree).

    ``fam_res``/``fam_tot`` are indexed by ``compute_stats``' ``families``
    order; both rounds are plotted on shared ticks, so the two persisted
    orders must agree on every shared position (the r1 tick-label lesson).
    """
    fams = list(stats["stats"][genre]["families"])
    p_fams = list(pooled_stats["stats"][genre].get("families", []))
    n = min(len(fams), len(p_fams))
    assert fams[:n] == p_fams[:n], (
        f"family order mismatch mixed vs pooled for {genre}: {fams} vs {p_fams} — "
        "shared ticks would mislabel one side"
    )
    return fams


def _l18_entry(stats: dict, genre: str, arm: str) -> dict | None:
    return stats["stats"].get(genre, {}).get("L18", {}).get(arm)


def _primary_pair_meta(stats: dict, genre: str) -> dict:
    """Paired-read metadata (all four reads) for the figure captions."""
    entry = stats.get("paired_diff", {}).get("genres", {}).get(genre, {})
    pairs = entry.get("pairs") or {}
    out = {}
    for name, node in pairs.items():
        out[name] = {"D_value": node["D_value"], "D_ci95": node["D_ci95"]}
    if entry.get("verdict"):
        out["genre_verdict"] = entry["verdict"]
    if entry.get("verdict_skipped_reason"):
        out["genre_verdict_skipped_reason"] = entry["verdict_skipped_reason"]
    return out


def hero_paired(stats, pooled_stats, nulls, pooled_nulls, genre, out_dir, meta) -> None:
    """L18 bars: persisted pooled arms (grey) vs the two new mixed arms (blue)."""
    g18 = stats["stats"][genre].get("L18", {})
    p18 = pooled_stats["stats"][genre]["L18"]
    arms = [a for a in HERO_ARMS if (a in g18) or (a in p18)]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
    vals, errs, colors = [], [], []
    for xi, a in zip(x, arms, strict=True):
        new_side = a in NEW_ARMS
        src18, src_nulls = (g18, nulls) if new_side else (p18, pooled_nulls)
        q = src_nulls["genres"].get(genre, {}).get("arms", {}).get(a, {})
        q = q.get("L18_column_quantiles")
        if q:
            ax.bar(float(xi), q["p975"], width=0.72, color="gray", alpha=0.25, zorder=1)
        vals.append(src18[a]["skill"])
        errs.append(_ci_err(src18[a]))
        colors.append(COLOR_NEW if new_side else COLOR_REF)
    ax.bar(x, vals, 0.62, yerr=np.array(errs).T, capsize=3, color=colors, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=20, ha="right")
    ax.set_ylabel("Pooled held-out skill-over-mean R² (L18)")
    ax.set_title(f"Mixing ON (new, blue) vs OFF (pooled round, grey) — {genre}")
    _save(
        fig,
        out_dir,
        f"mixed_hero_L18_{genre}",
        {**meta, "paired_reads": _primary_pair_meta(stats, genre), **_verdict_meta(stats)},
    )


def family_dots(skill, pooled_skill, genre, hl, fam_order, out_dir, meta) -> None:
    """Per-held-out-family skill dots for every hero arm (both rounds)."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6), layout="constrained")
    g_arms = skill["genres"][genre][str(hl)]["arms"]
    p_arms = pooled_skill["genres"][genre][str(hl)]["arms"]
    n_common = None
    for i, arm in enumerate(HERO_ARMS):
        node = g_arms.get(arm) if arm in NEW_ARMS else p_arms.get(arm)
        if node is None:
            continue
        fr = np.asarray(node["fam_res"], float)
        ft = np.asarray(node["fam_tot"], float)
        n_common = len(fr) if n_common is None else min(n_common, len(fr))
        with np.errstate(divide="ignore", invalid="ignore"):
            fam_skill = 1.0 - fr / ft
        marker = "o" if arm in NEW_ARMS else "s"
        ax.plot(
            range(len(fr)),
            fam_skill,
            marker,
            color=PALETTE[i % len(PALETTE)],
            label=ARM_LABELS[arm],
            alpha=0.85,
            ls="none",
        )
    ax.set_xticks(range(n_common))
    ax.set_xticklabels(fam_order[:n_common], rotation=30, ha="right")
    ax.set_ylabel("Held-out family skill R²")
    ax.set_title(f"Per-held-out-family skill — {genre} (L{hl})")
    ax.legend(fontsize=7)
    _save(fig, out_dir, f"mixed_family_dots_{genre}", meta)


def dilution_panel(stats, out_dir, meta) -> None:
    """s1 mechanism read: per-family gap' (Stitched-mixed − Pooled stitched) vs
    the pooled full−stitched gaps, WildChat/ICL highlighted, short-family band."""
    entry = stats.get("paired_diff", {}).get("genres", {}).get("uc", {})
    pairs = entry.get("pairs") or {}
    node = pairs.get("s2_concat_mix_vs_concat_i")
    ref = entry.get("s1_reference")
    if not node or not ref:
        return
    fams = ref["families"]
    gap_prime = np.asarray(node["fam_gap"], float)  # concat_mix − concat_i per family
    gap_full = np.asarray(ref["fam_gap_full_minus_concat_i_ref"], float)
    x = np.arange(len(fams))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
    ax.axhspan(SHORT_FAMILY_BAND[0], SHORT_FAMILY_BAND[1], color=PALETTE[2], alpha=0.15)
    ax.axhline(0.0, lw=1, color="gray", ls="--")
    ax.bar(x - w / 2, gap_full, w, color=COLOR_REF, label="Pooled full − stitched (dilution)")
    ax.bar(x + w / 2, gap_prime, w, color=COLOR_NEW, label="Stitched-mixed − Pooled stitched")
    for xi, fam in enumerate(fams):
        if fam in S1_HIGHLIGHT:
            ax.axvline(float(xi), color=PALETTE[3], alpha=0.25, lw=8, zorder=0)
    ax.set_xticks(x)
    # tick labels from the s1_reference's OWN persisted family order (always
    # index-aligned with the plotted gap arrays, smoke included).
    ax.set_xticklabels(fams, rotation=30, ha="right")
    ax.set_ylabel("Per-family skill gap (R²)")
    ax.set_title("Dilution mechanism — UC held-out families, L18")
    ax.legend(fontsize=8)
    _save(
        fig,
        out_dir,
        "mixed_dilution_panel",
        {
            **meta,
            "short_family_band": list(SHORT_FAMILY_BAND),
            "highlighted": list(S1_HIGHLIGHT),
            "gap_prime": gap_prime.tolist(),
            "gap_full_ref": gap_full.tolist(),
        },
    )


def cell_scatters(skill, genre, hl, out_dir, meta) -> None:
    """Per-cell predicted-vs-actual (fold top target-PC), both NEW arms."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), layout="constrained")
    for ax, arm in zip(axes, NEW_ARMS, strict=True):
        node = skill["genres"][genre][str(hl)]["arms"][arm]
        pred = np.asarray(node.get("cell_pred_pc1", []), float)
        act = np.asarray(node.get("cell_act_pc1", []), float)
        ok = ~(np.isnan(pred) | np.isnan(act))
        ax.scatter(act[ok], pred[ok], s=6, alpha=0.4, color=COLOR_NEW)
        lim = np.nanmax(np.abs(np.concatenate([act[ok], pred[ok]]))) if ok.any() else 1.0
        ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1, color="gray")
        ax.set_xlabel("Actual (fold top target-PC)")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{ARM_LABELS[arm]} (n={int(ok.sum())})")
    fig.suptitle(f"Per-cell predicted vs actual — {genre} (L{hl})")
    _save(fig, out_dir, f"mixed_cell_scatter_{genre}", meta)


def layer_curves(skill, pooled_skill, genre, out_dir, meta) -> None:
    """28-layer curves: new arms solid + persisted pooled references dashed."""
    gl = skill["genres"][genre]
    layers = sorted(int(x) for x in gl)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for i, arm in enumerate(NEW_ARMS):
        if arm not in gl[str(layers[0])]["arms"]:
            continue
        ys = [gl[str(ll)]["arms"][arm]["skill"] for ll in layers]
        ax.plot(layers, ys, label=ARM_LABELS[arm], color=PALETTE[i], lw=1.8)
    pgl = pooled_skill["genres"][genre]
    p_layers = sorted(int(x) for x in pgl)
    for arm, ls in (("arm_qry_iii", "--"), ("arm_concat_i", ":"), ("arm_full", "-.")):
        ys = [pgl[str(ll)]["arms"][arm]["skill"] for ll in p_layers]
        ax.plot(p_layers, ys, ls=ls, lw=1.2, color="black", label=ARM_LABELS[arm])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pooled held-out skill R²")
    ax.set_title(f"Layer sweep, mixed-forward arms — {genre}")
    ax.legend(fontsize=7)
    _save(fig, out_dir, f"mixed_layer_curves_{genre}", meta)


def betley_lambda_ladder(skill, pooled_skill, hl, out_dir, meta) -> None:
    """Betley fixed-λ ladder (s4 companion): per-λ skills, new vs pooled qry_iii."""
    gl = skill["genres"].get("betley")
    pgl = pooled_skill["genres"].get("betley")
    if not gl or not pgl:
        return
    fig, ax = plt.subplots(figsize=(7, 4.4), layout="constrained")
    lam = [float(x) for x in RIDGE_LAMBDAS]
    for i, arm in enumerate(NEW_ARMS):
        node = gl[str(hl)]["arms"].get(arm)
        if node is None:
            continue
        ax.plot(lam, node["skill_per_lambda"], "o-", color=PALETTE[i], label=ARM_LABELS[arm])
    p_node = pgl[str(hl)]["arms"]["arm_qry_iii"]
    ax.plot(
        lam,
        p_node["skill_per_lambda"],
        "s--",
        color="black",
        label=ARM_LABELS["arm_qry_iii"],
    )
    ax.set_xscale("log")
    ax.set_xlabel("Ridge λ (fixed ladder)")
    ax.set_ylabel(f"Pooled held-out skill R² (L{hl})")
    ax.set_title("Betley fixed-λ ladder — mixed vs masked query-span")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "mixed_betley_lambda_ladder", meta)


def dolly_panel(skill, pooled_skill, nulls, hl, out_dir, meta) -> None:
    """Dolly OOD skills (s5): new arms (null-gated) vs persisted pooled arms."""
    gl = skill["genres"]["uc"][str(hl)].get("ood_dolly") or {}
    pgl = pooled_skill["genres"]["uc"][str(hl)].get("ood_dolly") or {}
    arms = [a for a in HERO_ARMS if (a in gl) or (a in pgl)]
    if not arms:
        return
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(8.5, 4.4), layout="constrained")
    ood_bands = nulls["genres"].get("uc", {}).get("ood_dolly", {})
    vals, colors = [], []
    for xi, a in zip(x, arms, strict=True):
        new_side = a in NEW_ARMS
        q = ood_bands.get(a, {}).get("L18_column_quantiles")
        if q and new_side:
            ax.bar(float(xi), q["p975"], width=0.72, color="gray", alpha=0.25, zorder=1)
        vals.append((gl if new_side else pgl)[a]["skill"])
        colors.append(COLOR_NEW if new_side else COLOR_REF)
    ax.bar(x, vals, 0.62, color=colors, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=20, ha="right")
    ax.set_ylabel(f"Dolly OOD skill R² (L{hl})")
    ax.set_title("Corpus-transfer OOD (Dolly) — mixed (blue) vs pooled (grey)")
    _save(fig, out_dir, "mixed_dolly_panel", meta)


def identity_hist(identity: dict, out_dir, meta) -> None:
    """Identity-check cosine histogram (k1 join validity, identity_check_mix)."""
    rows = []
    for _fam, entry in identity.get("families", {}).items():
        rows.extend(entry.get("cos_rows", []))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.hist(rows, bins=40, color=COLOR_NEW)
    ax.axvline(identity.get("median_floor", 0.99), ls="--", color="red", lw=1)
    ax.axvline(identity.get("warn_floor", 0.999), ls=":", color="orange", lw=1)
    ax.set_xlabel("cos(flast mixed, flast pooled ffull), min over layers")
    ax.set_ylabel("Sampled rows (all genres)")
    ax.set_title("k1 content-identity spot-check (mixed round)")
    _save(fig, out_dir, "mixed_identity_hist", meta)


def paired_draw_hist(stats, out_dir, meta) -> None:
    """Paired-draw D distributions for the four registered reads (UC)."""
    entry = stats.get("paired_diff", {}).get("genres", {}).get("uc", {})
    pairs = entry.get("pairs") or {}
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    for i, (name, node) in enumerate(pairs.items()):
        draws = np.asarray(node.get("D_draws", []), float)
        draws = draws[~np.isnan(draws)]
        if draws.size == 0:
            continue
        ax.hist(draws, bins=50, alpha=0.55, color=PALETTE[i % len(PALETTE)], label=name)
    ax.axvline(0.0, ls="--", lw=1, color="gray")
    ax.set_xlabel("Paired D per shared bootstrap draw (R²)")
    ax.set_ylabel("Bootstrap draws")
    ax.set_title("Paired reads vs the pooled round — UC, L18")
    ax.legend(fontsize=7)
    _save(
        fig,
        out_dir,
        "mixed_paired_draw_hist",
        {**meta, "paired_reads": _primary_pair_meta(stats, "uc"), **_verdict_meta(stats)},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #923 mixed-forward-span-stitch figures")
    parser.add_argument(
        "--fits-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_923/mixed-forward-span-stitch",
    )
    parser.add_argument(
        "--pooled-fits-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_923/fits_pooled",
    )
    parser.add_argument(
        "--identity-json",
        type=Path,
        default=PROJECT_ROOT / "data/issue_923/capture/packs_mixed/identity_check_mix.json",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures/issue_923")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="upload the rendered figures dir to the HF data repo (ephemeral cpu-mid)",
    )
    args = parser.parse_args()

    meta = reproducibility_metadata({"script": "issue923_fig_mixed"})
    skill = load_json(args.fits_dir / "decomposition_skill.json")
    nulls = load_json(args.fits_dir / "null_summary.json")
    stats = load_json(args.fits_dir / "headline.json")
    pooled_skill = load_json(args.pooled_fits_dir / "decomposition_skill.json")
    pooled_stats = load_json(args.pooled_fits_dir / "headline.json")
    pooled_nulls = load_json(args.pooled_fits_dir / "null_summary.json")
    for genre in skill["genres"]:
        if genre not in pooled_stats["stats"]:
            continue  # smoke grids may carry genres the pooled round lacks
        hl = int(stats["stats"][genre].get("headline_layer_used", 18))
        fam_order = _family_order(stats, pooled_stats, genre)
        hero_paired(stats, pooled_stats, nulls, pooled_nulls, genre, args.out_dir, meta)
        family_dots(skill, pooled_skill, genre, hl, fam_order, args.out_dir, meta)
        cell_scatters(skill, genre, hl, args.out_dir, meta)
        layer_curves(skill, pooled_skill, genre, args.out_dir, meta)
    hl_uc = int(stats["stats"].get("uc", {}).get("headline_layer_used", 18))
    dilution_panel(stats, args.out_dir, meta)
    betley_lambda_ladder(skill, pooled_skill, hl_uc, args.out_dir, meta)
    dolly_panel(skill, pooled_skill, nulls, hl_uc, args.out_dir, meta)
    if args.identity_json.exists():
        identity_hist(load_json(args.identity_json), args.out_dir, meta)
    paired_draw_hist(stats, args.out_dir, meta)
    print(f"mixed figures written to {args.out_dir}")
    if args.upload:
        from explore_persona_space.orchestrate import hub

        hub._upload(args.out_dir, HF_DATA_REPO, "dataset", f"{HF_PREFIX_923}/figures_mixed")
        print(f"figures uploaded to {HF_DATA_REPO}:{HF_PREFIX_923}/figures_mixed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

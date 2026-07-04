#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #923 pooled-span-features figures (plan v6 §6).

Reads the pooled round's JSONs (``eval_results/issue_923/pooled-span-features/``)
PLUS the parent round's persisted results (``eval_results/issue_923/fits/`` —
the last-token comparison baseline, NOT refit) and renders:

- HERO: L18 paired bars per arm — last-token (parent) vs span-mean (new),
  family-bootstrap CI whiskers, per-arm L18 null bands shaded (pooled bands
  from this round; parent ridge-arm bands from the parent's null_summary —
  the parent's blend/Dolly surfaces are UN-GATED, recorded in meta).
- Per-held-out-family skill dots (pool_full / pool_concat_i, pooled vs last).
- Per-cell predicted-vs-actual scatters (pool_full / pool_concat_i, L18).
- Family x arm heatmap of pooled − last skill (the s1 format-family read).
- 28-layer curves for all pooled arms; presentations panel incl. Betley (s5);
  Dolly OOD panel; identity-check cosine histogram; closure-fraction
  distribution over the paired draws.

Per paper-plots policy: constrained layout (never tight_layout after a
colorbar), no on-plot annotation text (Δ values live in captions/meta),
colorblind-safe palette.

Usage::

    uv run python scripts/issue923_fig_pooled.py \\
        --fits-dir eval_results/issue_923/pooled-span-features \\
        --parent-fits-dir eval_results/issue_923/fits --out-dir figures/issue_923
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
from issue923_common import HF_DATA_REPO, HF_PREFIX_923, dump_json, load_json  # noqa: E402

ARM_LABELS = {
    "arm_ctx": "Context-only",
    "arm_qry_i": "Query-only (empty sys)",
    "arm_qry_ii": "Query-only (no sys block)",
    "arm_qry_iii": "Query-only (masked ctx)",
    "arm_concat_i": "Stitched pair",
    "arm_concat_ii": "Stitched pair (ii)",
    "arm_concat_iii": "Stitched pair (masked)",
    "arm_full": "Full prompt",
    "arm_blend": "Blended predictions",
}
HERO_ARMS = ["arm_ctx", "arm_qry_i", "arm_concat_i", "arm_blend", "arm_full"]
PALETTE = ["#0173b2", "#de8f05", "#029e73", "#cc78bc", "#ca9161", "#949494", "#56b4e9"]
PARENT_NULL_NOTE = (
    "parent side un-gated (no registered null in the parent round) for blend/Dolly surfaces"
)


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
    """Headline verdict fields for figure metadata, mirroring the k2 skip.

    The §6 k2 kill rule can SKIP the top-level verdict (``verdict`` absent,
    ``verdict_skipped_reason`` set) — the figure metadata must mirror that
    skip, never a bare ``verdict: None`` that hides WHY (r2 Major sweep).
    """
    out = {"verdict": stats.get("verdict")}
    if stats.get("verdict_skipped_reason"):
        out["verdict_skipped_reason"] = stats["verdict_skipped_reason"]
    return out


def _family_order(stats: dict, parent_stats: dict, genre: str) -> list[str]:
    """Family tick labels in the ARRAYS' index order (headline ``families``).

    ``fam_res``/``fam_tot`` are indexed by ``compute_stats``' FAMILY_ORDER-
    filtered ``families_present``, persisted as ``stats[genre]["families"]``
    — NEVER the ``fold_assignments.json`` dict-insertion order, whose tail
    differs (…, format, default, behavior) and swapped the behavior/default
    labels (r1 Major). Both rounds are plotted on shared ticks, so the
    parent's persisted order must agree on every shared position.
    """
    fams = list(stats["stats"][genre]["families"])
    p_fams = list(parent_stats["stats"][genre].get("families", []))
    n = min(len(fams), len(p_fams))
    assert fams[:n] == p_fams[:n], (
        f"family order mismatch pooled vs parent for {genre}: {fams} vs {p_fams} — "
        "shared ticks would mislabel one side"
    )
    return fams


def hero_paired(
    stats: dict, parent_stats: dict, nulls: dict, parent_nulls: dict, genre: str, out_dir, meta
) -> None:
    """L18 paired bars: last-token (parent, persisted) vs span-mean (new), per arm."""
    g = stats["stats"][genre]
    pg = parent_stats["stats"][genre]
    arms = [a for a in HERO_ARMS if a in g["L18"] and a in pg["L18"]]
    x = np.arange(len(arms))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    for xi, a in zip(x, arms, strict=True):
        for off, nl in ((-w / 2, parent_nulls), (w / 2, nulls)):
            q = nl["genres"][genre]["arms"].get(a, {}).get("L18_column_quantiles")
            if q:
                ax.bar(float(xi) + off, q["p975"], width=w, color="gray", alpha=0.25, zorder=1)
    last_vals = [pg["L18"][a]["skill"] for a in arms]
    pool_vals = [g["L18"][a]["skill"] for a in arms]
    last_err = np.array([_ci_err(pg["L18"][a]) for a in arms]).T
    pool_err = np.array([_ci_err(g["L18"][a]) for a in arms]).T
    ax.bar(
        x - w / 2,
        last_vals,
        w,
        yerr=last_err,
        capsize=3,
        color=PALETTE[5],
        label="Last-token (parent)",
        zorder=2,
    )
    ax.bar(
        x + w / 2,
        pool_vals,
        w,
        yerr=pool_err,
        capsize=3,
        color=PALETTE[0],
        label="Span-mean (this round)",
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=20, ha="right")
    ax.set_ylabel("Pooled held-out skill-over-mean R² (L18)")
    ax.set_title(f"Last-token vs span-mean features — {genre}")
    ax.legend(fontsize=8)
    pd = stats.get("paired_diff", {}).get("genres", {}).get(genre, {}).get("paired")
    _save(
        fig,
        out_dir,
        f"pooled_hero_L18_{genre}",
        {
            **meta,
            "delta_pool": pd and pd["delta_pool"],
            "delta_pool_ci95": pd and pd["delta_pool_ci95"],
            "delta_last": pd and pd["delta_last"],
            "paired_D": pd and pd["D_value"],
            "paired_D_ci95": pd and pd["D_ci95"],
            **_verdict_meta(stats),
            "parent_null_note": PARENT_NULL_NOTE,
        },
    )


def family_dots(
    skill: dict, parent_skill: dict, genre: str, hl: int, fam_order: list[str], out_dir, meta
) -> None:
    """Per-held-out-family skill dots, pooled vs last (arm_full / arm_concat_i).

    ``fam_order`` = the persisted array index order (``_family_order``).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    for ax, arm in zip(axes, ("arm_full", "arm_concat_i"), strict=True):
        n_common = None
        for src, color, label in (
            (parent_skill, PALETTE[5], "last-token"),
            (skill, PALETTE[0], "span-mean"),
        ):
            node = src["genres"][genre][str(hl)]["arms"][arm]
            fr = np.asarray(node["fam_res"], float)
            ft = np.asarray(node["fam_tot"], float)
            n_common = len(fr) if n_common is None else min(n_common, len(fr))
            with np.errstate(divide="ignore", invalid="ignore"):
                fam_skill = 1.0 - fr / ft
            ax.plot(range(len(fr)), fam_skill, "o", color=color, label=label, alpha=0.85)
        ax.set_xticks(range(n_common))
        ax.set_xticklabels(fam_order[:n_common], rotation=30, ha="right")
        ax.set_title(ARM_LABELS[arm])
        ax.set_ylabel("Held-out family skill R²")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Per-held-out-family skill — {genre} (L{hl})")
    _save(fig, out_dir, f"pooled_family_dots_{genre}", meta)


def cell_scatters(skill: dict, genre: str, hl: int, out_dir, meta) -> None:
    """Per-cell predicted-vs-actual (fold top target-PC), pool_full / pool_concat_i."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), layout="constrained")
    for ax, arm in zip(axes, ("arm_full", "arm_concat_i"), strict=True):
        node = skill["genres"][genre][str(hl)]["arms"][arm]
        pred = np.asarray(node.get("cell_pred_pc1", []), float)
        act = np.asarray(node.get("cell_act_pc1", []), float)
        ok = ~(np.isnan(pred) | np.isnan(act))
        ax.scatter(act[ok], pred[ok], s=6, alpha=0.4, color=PALETTE[0])
        lim = np.nanmax(np.abs(np.concatenate([act[ok], pred[ok]]))) if ok.any() else 1.0
        ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1, color="gray")
        ax.set_xlabel("Actual (fold top target-PC)")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{ARM_LABELS[arm]} (span-mean, n={int(ok.sum())})")
    fig.suptitle(f"Per-cell predicted vs actual — {genre} (L{hl})")
    _save(fig, out_dir, f"pooled_cell_scatter_{genre}", meta)


def family_arm_heatmap(
    skill: dict, parent_skill: dict, genre: str, hl: int, fam_order: list[str], out_dir, meta
):
    """Family x arm heatmap of (span-mean − last-token) held-out family skill (s1).

    ``fam_order`` = the persisted array index order (``_family_order``).
    """
    arms = [a for a in ARM_LABELS if a in skill["genres"][genre][str(hl)]["arms"]]
    p_arms = parent_skill["genres"][genre][str(hl)]["arms"]
    arms = [a for a in arms if a in p_arms]
    n_fam = min(
        len(np.asarray(skill["genres"][genre][str(hl)]["arms"][arms[0]]["fam_res"])),
        len(np.asarray(p_arms[arms[0]]["fam_res"])),
    )  # production: always equal (same battery); trims only on a smoke grid
    diff = np.zeros((n_fam, len(arms)))
    for j, a in enumerate(arms):
        pn = p_arms[a]
        qn = skill["genres"][genre][str(hl)]["arms"][a]
        with np.errstate(divide="ignore", invalid="ignore"):
            fs_last = 1.0 - np.asarray(pn["fam_res"], float) / np.asarray(pn["fam_tot"], float)
            fs_pool = 1.0 - np.asarray(qn["fam_res"], float) / np.asarray(qn["fam_tot"], float)
        diff[:, j] = fs_pool[:n_fam] - fs_last[:n_fam]
    # constrained layout from subplots(); NEVER tight_layout after a colorbar.
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    vmax = np.nanmax(np.abs(diff)) or 1.0
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, label="span-mean minus last-token family skill")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=25, ha="right")
    ax.set_yticks(range(n_fam))
    ax.set_yticklabels(fam_order[:n_fam])
    ax.set_title(f"Pooling effect by held-out family — {genre} (L{hl})")
    _save(fig, out_dir, f"pooled_family_arm_heatmap_{genre}", meta)


def layer_curves(skill: dict, parent_skill: dict, genre: str, out_dir, meta) -> None:
    """28-layer pooled skill curves per arm (+ the parent arm_full/concat_i dashed)."""
    gl = skill["genres"][genre]
    layers = sorted(int(x) for x in gl)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for i, arm in enumerate(ARM_LABELS):
        if arm not in gl[str(layers[0])]["arms"]:
            continue
        ys = [gl[str(ll)]["arms"][arm]["skill"] for ll in layers]
        ax.plot(layers, ys, label=ARM_LABELS[arm], color=PALETTE[i % len(PALETTE)], lw=1.5)
    pgl = parent_skill["genres"][genre]
    p_layers = sorted(int(x) for x in pgl)
    for arm, ls in (("arm_full", "--"), ("arm_concat_i", ":")):
        ys = [pgl[str(ll)]["arms"][arm]["skill"] for ll in p_layers]
        ax.plot(p_layers, ys, ls=ls, lw=1.2, color="black", label=f"{ARM_LABELS[arm]} (last)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pooled held-out skill R²")
    ax.set_title(f"Layer sweep, span-mean features — {genre}")
    ax.legend(fontsize=7)
    _save(fig, out_dir, f"pooled_layer_curves_{genre}", meta)


def presentations_panel(stats: dict, genre: str, out_dir, meta) -> None:
    """The three pooled query presentations + concats at L18 (s5 incl. Betley)."""
    g = stats["stats"][genre]
    arms = [
        a
        for a in (
            "arm_qry_i",
            "arm_qry_ii",
            "arm_qry_iii",
            "arm_concat_i",
            "arm_concat_ii",
            "arm_concat_iii",
        )
        if a in g["L18"]
    ]
    vals = [g["L18"][a]["skill"] for a in arms]
    errs = np.array([_ci_err(g["L18"][a]) for a in arms]).T
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    x = np.arange(len(arms))
    ax.bar(x, vals, yerr=errs, capsize=4, color=PALETTE[: len(arms)])
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=20, ha="right")
    ax.set_ylabel("Pooled held-out skill R² (L18)")
    ax.set_title(f"Null-context presentations, span-mean — {genre}")
    _save(fig, out_dir, f"pooled_presentations_L18_{genre}", meta)


def dolly_panel(skill: dict, parent_skill: dict, nulls: dict, hl: int, out_dir, meta) -> None:
    """Dolly OOD skill per arm, pooled vs last; pooled L18 null bands shaded (s4)."""
    gl = skill["genres"]["uc"][str(hl)].get("ood_dolly") or {}
    pgl = parent_skill["genres"]["uc"][str(hl)].get("ood_dolly") or {}
    arms = [a for a in ARM_LABELS if a in gl and a in pgl]
    if not arms:
        return
    x = np.arange(len(arms))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.4), layout="constrained")
    ood_bands = nulls["genres"].get("uc", {}).get("ood_dolly", {})
    for xi, a in zip(x, arms, strict=True):
        q = ood_bands.get(a, {}).get("L18_column_quantiles")
        if q:
            ax.bar(float(xi) + w / 2, q["p975"], width=w, color="gray", alpha=0.25, zorder=1)
    ax.bar(
        x - w / 2,
        [pgl[a]["skill"] for a in arms],
        w,
        color=PALETTE[5],
        label="Last-token (parent, un-gated)",
        zorder=2,
    )
    ax.bar(
        x + w / 2,
        [gl[a]["skill"] for a in arms],
        w,
        color=PALETTE[0],
        label="Span-mean (null-gated)",
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=20, ha="right")
    ax.set_ylabel(f"Dolly OOD skill R² (L{hl})")
    ax.set_title("Corpus-transfer OOD (Dolly) — last-token vs span-mean")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "pooled_dolly_panel", {**meta, "parent_null_note": PARENT_NULL_NOTE})


def identity_hist(identity: dict, out_dir, meta) -> None:
    """Identity-check cosine histogram (k1 join validity)."""
    rows = []
    for _fam, entry in identity.get("families", {}).items():
        rows.extend(entry.get("cos_rows", []))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.hist(rows, bins=40, color=PALETTE[0])
    ax.axvline(identity.get("median_floor", 0.99), ls="--", color="red", lw=1)
    ax.axvline(identity.get("warn_floor", 0.999), ls=":", color="orange", lw=1)
    ax.set_xlabel("cos(flast new, flast parent), min over layers")
    ax.set_ylabel("Sampled rows (all families)")
    ax.set_title("k1 content-identity spot-check")
    _save(fig, out_dir, "pooled_identity_hist", meta)


def closure_hist(stats: dict, out_dir, meta) -> None:
    """Closure-fraction distribution over the shared paired bootstrap draws."""
    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    plotted = False
    for i, (g, entry) in enumerate(stats.get("paired_diff", {}).get("genres", {}).items()):
        pd = entry.get("paired")
        if not pd or not pd.get("closure_draws"):
            continue
        draws = np.asarray(pd["closure_draws"], float)
        draws = draws[~np.isnan(draws)]
        ax.hist(draws, bins=50, alpha=0.6, color=PALETTE[i], label=g)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axvline(0.0, ls="--", lw=1, color="gray")
    ax.axvline(1.0, ls="--", lw=1, color="gray")
    ax.set_xlabel("Closure fraction 1 - delta_pool/delta_last (per shared draw)")
    ax.set_ylabel("Bootstrap draws")
    ax.set_title("Deficit closure under span-mean pooling")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "pooled_closure_hist", {**meta, **_verdict_meta(stats)})


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #923 pooled-span-features figures")
    parser.add_argument(
        "--fits-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_923/pooled-span-features",
    )
    parser.add_argument(
        "--parent-fits-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_923/fits"
    )
    parser.add_argument(
        "--identity-json",
        type=Path,
        default=PROJECT_ROOT / "data/issue_923/capture/packs_pooled/identity_check.json",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures/issue_923")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="upload the rendered figures dir to the HF data repo (ephemeral cpu-mid)",
    )
    args = parser.parse_args()

    meta = reproducibility_metadata({"script": "issue923_fig_pooled"})
    skill = load_json(args.fits_dir / "decomposition_skill.json")
    nulls = load_json(args.fits_dir / "null_summary.json")
    stats = load_json(args.fits_dir / "headline.json")
    parent_skill = load_json(args.parent_fits_dir / "decomposition_skill.json")
    parent_stats = load_json(args.parent_fits_dir / "headline.json")
    parent_nulls = load_json(args.parent_fits_dir / "null_summary.json")
    for genre in skill["genres"]:
        hl = int(stats["stats"][genre].get("headline_layer_used", 18))
        fam_order = _family_order(stats, parent_stats, genre)
        hero_paired(stats, parent_stats, nulls, parent_nulls, genre, args.out_dir, meta)
        family_dots(skill, parent_skill, genre, hl, fam_order, args.out_dir, meta)
        cell_scatters(skill, genre, hl, args.out_dir, meta)
        family_arm_heatmap(skill, parent_skill, genre, hl, fam_order, args.out_dir, meta)
        layer_curves(skill, parent_skill, genre, args.out_dir, meta)
        presentations_panel(stats, genre, args.out_dir, meta)
    hl_uc = int(stats["stats"].get("uc", {}).get("headline_layer_used", 18))
    dolly_panel(skill, parent_skill, nulls, hl_uc, args.out_dir, meta)
    if args.identity_json.exists():
        identity_hist(load_json(args.identity_json), args.out_dir, meta)
    closure_hist(stats, args.out_dir, meta)
    print(f"pooled figures written to {args.out_dir}")
    if args.upload:
        from explore_persona_space.orchestrate import hub

        hub._upload(args.out_dir, HF_DATA_REPO, "dataset", f"{HF_PREFIX_923}/figures_pooled")
        print(f"figures uploaded to {HF_DATA_REPO}:{HF_PREFIX_923}/figures_pooled")
    return 0


if __name__ == "__main__":
    sys.exit(main())

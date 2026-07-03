#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #923 figures — hero L18 arm panel + the §6 exploratory dump.

Reads the Phase-3 JSONs (``decomposition_skill.json`` / ``null_summary.json`` /
``anova_shares.json`` / ``headline.json`` / ``regen_check.json``) and renders:

- HERO: L18 UC pooled held-out skill per arm (family-cluster CI whiskers),
  ANOVA oracle ceilings as reference lines, L18 null band shaded.
- 28-layer skill curves per arm (per genre); ANOVA share-by-layer curves.
- Query presentations (i)/(ii)/(iii) side by side; OOD (Dolly) + Betley panels;
  seen/unseen marginal regimes; regen-check cosine histogram.

Per paper-plots policy: constrained layout (never tight_layout after colorbars),
no on-plot annotation text (ΔR²/ρ_dec live in captions), colorblind-safe.

Usage::

    uv run python scripts/issue923_figures.py --fits-dir eval_results/issue_923/fits \\
        --out-dir figures/issue_923
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue923_common import HF_DATA_REPO, HF_PREFIX_923, dump_json, load_json  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

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


def _save(fig, out_dir: Path, name: str, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    fig.savefig(out_dir / f"{name}.pdf")
    dump_json(meta, out_dir / f"{name}.meta.json")
    plt.close(fig)


def hero_panel(stats: dict, nulls: dict, anova: dict, genre: str, out_dir: Path, meta: dict):
    """L18 pooled held-out skill per arm + CIs + oracle ceilings + null band."""
    g = stats["stats"][genre]
    hl = str(g.get("headline_layer_used", 18))
    arms = [a for a in HERO_ARMS if a in g["L18"]]
    vals = [g["L18"][a]["skill"] for a in arms]
    los = [max(0.0, v - g["L18"][a]["ci95"][0]) for a, v in zip(arms, vals, strict=True)]
    his = [max(0.0, g["L18"][a]["ci95"][1] - v) for a, v in zip(arms, vals, strict=True)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    x = np.arange(len(arms))
    # PER-ARM L18 null bands FIRST (behind the skill bars): each arm is gated
    # against ITS OWN selection-matched L18-only null column (plan hero spec;
    # r1: only arm_full's band was shaded). arm_blend has no permutation
    # column (nulls cover the ridge arms) — no band drawn for it.
    null_arms = nulls["genres"][genre]["arms"]
    for xi, a in zip(x, arms, strict=True):
        q = null_arms.get(a, {}).get("L18_column_quantiles")
        if not q:
            continue
        ax.bar(float(xi), q["p975"], width=0.8, color="gray", alpha=0.25, zorder=1)
    ax.bar(x, vals, yerr=[los, his], capsize=4, color=PALETTE[: len(arms)], zorder=2)
    shares = anova["anova"][genre][hl]["pca48"]
    ax.axhline(shares["share_ctx"], ls="--", lw=1, color="#0173b2", alpha=0.7)
    ax.axhline(shares["share_ctx"] + shares["share_qry"], ls="--", lw=1, color="#029e73", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=20, ha="right")
    ax.set_ylabel("Pooled held-out skill-over-mean R² (L18)")
    ax.set_title(f"Context/query decomposition — {genre} (LOFO x held-out-query)")
    _save(fig, out_dir, f"hero_L18_{genre}", meta)


def layer_curves(skill: dict, genre: str, out_dir: Path, meta: dict) -> None:
    """Per-arm skill across all layers (the full sweep display, no max headline)."""
    gl = skill["genres"][genre]
    layers = sorted(int(x) for x in gl)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for i, arm in enumerate(ARM_LABELS):
        if arm not in gl[str(layers[0])]["arms"]:
            continue
        ys = [gl[str(ll)]["arms"][arm]["skill"] for ll in layers]
        ax.plot(layers, ys, label=ARM_LABELS[arm], color=PALETTE[i % len(PALETTE)], lw=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pooled held-out skill R²")
    ax.set_title(f"Layer sweep — {genre}")
    ax.legend(fontsize=7)
    _save(fig, out_dir, f"layer_curves_{genre}", meta)


def anova_curves(anova: dict, genre: str, out_dir: Path, meta: dict) -> None:
    """In-sample ANOVA share-by-layer curves (oracle reference)."""
    ga = anova["anova"][genre]
    layers = sorted(int(x) for x in ga)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for i, key in enumerate(("share_ctx", "share_qry", "share_interaction")):
        ys = [ga[str(ll)]["pca48"][key] for ll in layers]
        ax.plot(layers, ys, label=key, color=PALETTE[i], lw=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("In-sample variance share (PCA-48)")
    ax.set_title(f"ANOVA oracle shares — {genre} (in-sample reference)")
    ax.legend(fontsize=8)
    _save(fig, out_dir, f"anova_shares_{genre}", meta)


def presentations_panel(stats: dict, genre: str, out_dir: Path, meta: dict) -> None:
    """The three query-only presentations + their concats, side by side at L18."""
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
    los = [max(0.0, v - g["L18"][a]["ci95"][0]) for a, v in zip(arms, vals, strict=True)]
    his = [max(0.0, g["L18"][a]["ci95"][1] - v) for a, v in zip(arms, vals, strict=True)]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    x = np.arange(len(arms))
    ax.bar(x, vals, yerr=[los, his], capsize=4, color=PALETTE[: len(arms)])
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=20, ha="right")
    ax.set_ylabel("Pooled held-out skill R² (L18)")
    ax.set_title(f"Null-context presentations — {genre}")
    _save(fig, out_dir, f"presentations_L18_{genre}", meta)


def regimes_panel(skill: dict, genre: str, hl: int, out_dir: Path, meta: dict) -> None:
    """Primary vs marginal (seen-query / seen-context) vs OOD regimes at L18."""
    gl = skill["genres"][genre][str(hl)]
    arms = [a for a in ("arm_ctx", "arm_qry_i", "arm_concat_i", "arm_full") if a in gl["arms"]]
    schemes = [("primary (both unseen)", lambda a: gl["arms"][a]["skill"])]
    for key, label in (
        ("lofo_marginal", "unseen family x seen query"),
        ("qfold_marginal", "seen context x unseen query"),
        ("ood_dolly", "unseen family x Dolly (OOD)"),
    ):
        if gl.get(key):
            schemes.append((label, lambda a, k=key: gl[k].get(a, {}).get("skill", np.nan)))
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    width = 0.8 / len(schemes)
    x = np.arange(len(arms))
    for si, (label, fn) in enumerate(schemes):
        ax.bar(x + si * width, [fn(a) for a in arms], width, label=label, color=PALETTE[si])
    ax.set_xticks(x + width * (len(schemes) - 1) / 2)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=15, ha="right")
    ax.set_ylabel(f"Held-out skill R² (L{hl})")
    ax.set_title(f"Fold regimes — {genre}")
    ax.legend(fontsize=8)
    _save(fig, out_dir, f"regimes_L18_{genre}", meta)


def regen_hist(regen: dict, out_dir: Path, meta: dict) -> None:
    """Regen-check cosine histogram (cross-provenance join validity, plan 1e)."""
    rows = regen.get("rows", [])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.hist([r["cos_mean"] for r in rows], bins=30, color=PALETTE[0])
    ax.axvline(0.99, ls="--", color="red", lw=1)
    ax.set_xlabel("cos(fresh v̄, store-reduced v̄), mean over layers")
    ax.set_ylabel("Regen cells")
    ax.set_title("Regen spot-check (store join validity)")
    _save(fig, out_dir, "regen_check_hist", meta)


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #923 figures")
    parser.add_argument(
        "--fits-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_923/fits"
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures/issue_923")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="upload the rendered figures dir to the HF data repo (the ephemeral "
        "cpu-mid instance cannot commit to git — r1 Minor: figures never left it)",
    )
    args = parser.parse_args()

    meta = reproducibility_metadata({"script": "issue923_figures"})
    skill = load_json(args.fits_dir / "decomposition_skill.json")
    nulls = load_json(args.fits_dir / "null_summary.json")
    anova = load_json(args.fits_dir / "anova_shares.json")
    stats = load_json(args.fits_dir / "headline.json")
    for genre in skill["genres"]:
        hl = int(stats["stats"][genre].get("headline_layer_used", 18))
        hero_panel(stats, nulls, anova, genre, args.out_dir, meta)
        layer_curves(skill, genre, args.out_dir, meta)
        anova_curves(anova, genre, args.out_dir, meta)
        presentations_panel(stats, genre, args.out_dir, meta)
        regimes_panel(skill, genre, hl, args.out_dir, meta)
    regen_path = args.fits_dir / "regen_check.json"
    if regen_path.exists():
        regen_hist(load_json(regen_path), args.out_dir, meta)
    print(f"figures written to {args.out_dir}")
    if args.upload:
        from explore_persona_space.orchestrate import hub

        hub._upload(args.out_dir, HF_DATA_REPO, "dataset", f"{HF_PREFIX_923}/figures")
        print(f"figures uploaded to {HF_DATA_REPO}:{HF_PREFIX_923}/figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())

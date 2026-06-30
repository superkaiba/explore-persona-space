#!/usr/bin/env python
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional scientific Unicode (ρ, Δ, L̂, ŵ, ×, −) in docstrings/comments/labels.
"""issue #666 Phase 4 — figures (plan §4l).

HERO (per-behavior breakdown — primary because of the cross-arm r_B heterogeneity):
per behavior, grouped bars of Spearman ρ for {full L̂, apples-to-apples cosine,
base-prior} with the designed-null L̂ ρ reference line + the test-retest
noise-floor band + clustered-CI error bars. EXPLORATORY DUMP: per-cell L̂-vs-Δs
scatter (points labeled by context family) + the raw-alongside-residualized views.

Reads the per-cell predictor JSONs + designed_null_Lhat_rho.json + noise_floor.json
+ clustered_ci.json from ``eval_results/issue_666/``; writes PNGs to
``figures/issue_666/`` via the shared ``paper_plots`` (commit-pinned, per-point
sidecar). The analyzer picks the hero from these + commits/SHA-pins them.

CPU-only; matplotlib via paper_plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "eval_results" / "issue_666"
FIG_DIR = REPO / "figures" / "issue_666"

# Plain-English behavior labels for axis/legend (no opaque slugs — paper_plots rule).
_BEHAVIOR_LABEL = {
    "bad_medical": "bad-medical advice",
    "em": "insecure-code (EM)",
    "fact": "taught-fact",
    "marker": "marker (sentinel)",
    "designed_null": "designed-null",
}
_VARIANT_LABEL = {
    "rho_full_Lhat": "full predictor L̂",
    "rho_cosine": "cosine special-case",
    "rho_base_prior": "base-rate guess",
}


def _load_predictor_cells(pred_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(pred_dir.glob("*_predictor_cells.json"))]


def hero_figure(cells: list[dict], designed_null: dict, noise_floor: dict) -> dict:
    """Per-behavior grouped-bar ρ figure with the designed-null line + noise band."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    # Group cells by behavior; average ρ per (behavior, variant) over its cells.
    by_beh: dict[str, dict[str, list]] = {}
    for c in cells:
        beh = c.get("behavior") or "unknown"
        d = by_beh.setdefault(beh, {k: [] for k in _VARIANT_LABEL})
        for k in _VARIANT_LABEL:
            d[k].append(c.get(k, np.nan))

    behaviors = sorted(by_beh)
    variants = list(_VARIANT_LABEL)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.6 * len(behaviors)), 4.0))
    x = np.arange(len(behaviors))
    w = 0.8 / max(1, len(variants))
    roles = ["primary", "baseline", "control"]
    for vi, var in enumerate(variants):
        heights = [float(np.nanmean(by_beh[b][var])) for b in behaviors]
        ax.bar(
            x + vi * w - 0.4 + w / 2,
            heights,
            width=w,
            label=_VARIANT_LABEL[var],
            color=paper_palette_role(roles[vi % len(roles)]),
        )

    # Designed-null reference line (mean over the 2 null cells).
    null_rhos = [v["rho"] for v in designed_null.get("per_null", {}).values()]
    if null_rhos:
        ax.axhline(
            float(np.mean(null_rhos)),
            ls="--",
            color=paper_palette_role("accent"),
            label="designed-null L̂",
        )
    # Noise-floor band (the test-retest reliability ceiling).
    nf_mean = noise_floor.get("rho_mean")
    if nf_mean is not None:
        ax.axhline(float(nf_mean), ls=":", color="0.5", label="noise floor (test-retest)")

    ax.set_xticks(x)
    ax.set_xticklabels([_BEHAVIOR_LABEL.get(b, b) for b in behaviors], rotation=20, ha="right")
    ax.set_ylabel("Spearman ρ (L̂ vs latent Δs)")
    ax.legend(fontsize=7, loc="best")
    set_title_subtitle(
        ax,
        "Leakage predictor vs cosine, base-rate, designed-null",
        "per behavior; bars = mean ρ over the behavior's cells",
    )
    paths = savefig_paper(
        fig, "issue_666/hero_predictor_rho_by_behavior", dir=str(REPO / "figures")
    )
    plt.close(fig)
    return {"hero": {k: str(v) for k, v in paths.items()}, "behaviors": behaviors}


def scatter_dump(cells: list[dict]) -> dict:
    """Per-cell L̂-vs-Δs scatter, points labeled by context family (exploratory)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    out = {}
    for c in cells:
        pb = c["per_bystander"]
        lh = np.array(pb["Lhat"])
        ds = np.array(pb["ds"])
        fams = pb["context_family"]
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        ax.scatter(lh, ds, s=14)
        for xi, yi, fam in zip(lh, ds, fams, strict=False):
            ax.annotate(str(fam), (xi, yi), fontsize=5, alpha=0.6)
        ax.set_xlabel("predicted leakage L̂")
        ax.set_ylabel("latent ground truth Δs")
        ax.set_title(f"{c['cell']} (ρ={c['rho_full_Lhat']:.3f})", fontsize=8)
        stem = f"issue_666/scatter_{c['cell']}"
        paths = savefig_paper(fig, stem, dir=str(REPO / "figures"))
        plt.close(fig)
        out[c["cell"]] = {k: str(v) for k, v in paths.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="issue 666 Phase-4 figures.")
    ap.add_argument("--slice", action="store_true", help="tiny smoke slice")
    args = ap.parse_args()  # noqa: F841 (slice flag: same code path, fewer cells upstream)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cells = _load_predictor_cells(EVAL / "predictor")
    if not cells:
        print("[figures] no predictor cells found; nothing to plot")
        print("[phase=figures] done OK (no inputs)")
        return 0
    dn_path = EVAL / "headline" / "designed_null_Lhat_rho.json"
    nf_path = EVAL / "noise_floor" / "noise_floor.json"
    designed_null = json.loads(dn_path.read_text()) if dn_path.exists() else {}
    noise_floor = json.loads(nf_path.read_text()) if nf_path.exists() else {}

    hero = hero_figure(cells, designed_null, noise_floor)
    scatters = scatter_dump(cells)
    print(f"[figures] hero -> {hero['hero'].get('png')}")
    print(f"[figures] {len(scatters)} per-cell scatters")
    print("[phase=figures] done OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

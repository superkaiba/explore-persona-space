# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #489 — figure generation from Phase 5 analysis + Phase 4 cell payloads.

Plan v5 §6.3. Over-produces hero candidates so the analyzer can pick at write-up.

Outputs (under ``figures/issue_489/``):
  - ``h3_icl_vs_sp.{png,pdf,meta.json}``       Hero: paired forest plot ρ_ICL vs ρ_SP
  - ``h4a_cross_type_scatter.{png,pdf,meta}``  Cross-type cosine→ΔG scatter
  - ``h4b_matched_pair_residuals.{png,pdf}``   Matched same-identity residual figure
  - ``cosine_24x24_heatmap.{png,pdf}``         Full panel heatmap, block-organized
  - ``layer_sweep_forest.{png,pdf}``           Layer-sweep ρ across {7,11,14,15,21,27}

CLI:
    uv run python scripts/i489_make_figures.py
    uv run python scripts/i489_make_figures.py --smoke
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.i489_contexts import UNION_CONTEXTS

logger = logging.getLogger("i489.figures")

PHASE1_DIR = Path("eval_results/issue_489/phase1")
PHASE4_DIR = Path("eval_results/issue_489/phase4/per_cell")
PHASE5_DIR = Path("eval_results/issue_489/phase5")
FIG_DIR = Path("figures/issue_489")


def _git_commit_hash() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _save_meta(path: Path, payload: dict) -> None:
    meta = {
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        **payload,
    }
    path.write_text(json.dumps(meta, indent=2))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    _ = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use the project's paper-quality rcParams if available; fall back to defaults.
    try:
        from explore_persona_space.analysis.paper_plots import apply_paper_rcparams

        apply_paper_rcparams()
    except Exception:
        pass

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    analysis_path = PHASE5_DIR / "analysis.json"
    cos_path = PHASE1_DIR / "cosine_per_layer.json"
    if not analysis_path.exists() or not cos_path.exists():
        raise FileNotFoundError(f"Need {analysis_path} and {cos_path}; run phase5 + phase1 first.")
    analysis = json.loads(analysis_path.read_text())
    cos_payload = json.loads(cos_path.read_text())
    cos_sim = cos_payload["cos_sim_per_layer"][str(cos_payload["headline_layer"])]
    cids = [c.cid for c in UNION_CONTEXTS if c.cid in cos_sim]

    # ── Figure 1: H3 paired forest plot ──────────────────────────────────
    h3 = analysis.get("h3", {})
    fracs = sorted(h3.keys(), key=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, f in enumerate(fracs):
        v = h3[f]
        ax.errorbar(
            [k - 0.15],
            [v["rho_icl"]],
            yerr=[[v["rho_icl"] - v["ci_icl"][0]], [v["ci_icl"][1] - v["rho_icl"]]],
            fmt="o",
            label="ICL within" if k == 0 else None,
            color="C0",
        )
        ax.errorbar(
            [k + 0.15],
            [v["rho_sp"]],
            yerr=[[v["rho_sp"] - v["ci_sp"][0]], [v["ci_sp"][1] - v["rho_sp"]]],
            fmt="s",
            label="SP within" if k == 0 else None,
            color="C1",
        )
    ax.axhline(-0.30, ls="--", color="grey", alpha=0.5, label="ρ ≤ -0.30 PASS bar")
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels([f"frac={float(f):.2f}" for f in fracs])
    ax.set_ylabel("Spearman ρ(cos_distance L21, ΔG)")
    ax.set_title("H3: within-arm cosine→ΔG, ICL vs SP")
    ax.legend()
    out = FIG_DIR / "h3_icl_vs_sp"
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    _save_meta(out.with_suffix(".meta.json"), {"fracs": fracs, "h3": h3})

    # ── Figure 2: 24×24 cosine heatmap, block-organized ────────────────────
    icl_cids = [c for c in cids if c.startswith("IK")]
    sp_cids = [c for c in cids if c.startswith("SP")]
    ordered = icl_cids + sp_cids
    mat = np.full((len(ordered), len(ordered)), np.nan)
    import contextlib

    for i, ci in enumerate(ordered):
        for j, cj in enumerate(ordered):
            with contextlib.suppress(KeyError):
                mat[i, j] = cos_sim[ci][cj]
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    im = ax2.imshow(mat, vmin=0.5, vmax=1.0, cmap="viridis")
    ax2.set_xticks(range(len(ordered)))
    ax2.set_xticklabels(ordered, rotation=90, fontsize=7)
    ax2.set_yticks(range(len(ordered)))
    ax2.set_yticklabels(ordered, fontsize=7)
    # Block separator between ICL and SP
    n_icl = len(icl_cids)
    ax2.axhline(n_icl - 0.5, color="white", lw=1)
    ax2.axvline(n_icl - 0.5, color="white", lw=1)
    fig2.colorbar(im, ax=ax2, label="cos_sim L21")
    ax2.set_title("Base-model L21 cosine similarity (24 union contexts)")
    out2 = FIG_DIR / "cosine_24x24_heatmap"
    fig2.tight_layout()
    fig2.savefig(out2.with_suffix(".png"), dpi=200)
    fig2.savefig(out2.with_suffix(".pdf"))
    plt.close(fig2)
    _save_meta(out2.with_suffix(".meta.json"), {"ordered_cids": ordered})

    # ── Figure 3: cross-type cosine-vs-ΔG scatter ─────────────────────────
    h4a = analysis.get("h4a", {})
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.bar(
        range(len(fracs)),
        [h4a[f]["rho_dual_partial"] for f in fracs if f in h4a],
    )
    ax3.set_xticks(range(len(fracs)))
    ax3.set_xticklabels([f"frac={float(f):.2f}" for f in fracs])
    ax3.set_ylabel("Cross-type partial ρ(cos, ΔG | length, overlap)")
    ax3.axhline(-0.20, ls="--", color="grey", alpha=0.5, label="PASS bar")
    ax3.set_title("H4(a): cross-type cosine→ΔG controlling overlap")
    ax3.legend()
    out3 = FIG_DIR / "h4a_cross_type"
    fig3.tight_layout()
    fig3.savefig(out3.with_suffix(".png"), dpi=200)
    fig3.savefig(out3.with_suffix(".pdf"))
    plt.close(fig3)
    _save_meta(out3.with_suffix(".meta.json"), {"h4a": h4a})

    logger.info("Wrote figures to %s", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

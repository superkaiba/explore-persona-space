#!/usr/bin/env python3
"""Cross-phase synthesis — issue #368 §4.3.

Reads h1_verdict.json + h2_verdict.json and writes a prose-only synthesis
with two separate verdicts (clarifier Q5(a) — no combined regression, no
bootstrap pooling across phases). Plus a side-by-side bar figure.

Outputs:
  eval_results/issue_368/cross_phase_synthesis.json
  figures/issue_368/cross_phase_synthesis.png   (+ .meta.json sidecar)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from explore_persona_space.eval.leakage_axes import build_run_metadata, dump_json  # noqa: E402

PHASE1_DIR = REPO_ROOT / "eval_results" / "issue_368" / "phase1"
PHASE2_DIR = REPO_ROOT / "eval_results" / "issue_368" / "phase2"
OUT_DIR = REPO_ROOT / "eval_results" / "issue_368"
FIG_DIR = REPO_ROOT / "figures" / "issue_368"


def build_synthesis() -> dict:
    with open(PHASE1_DIR / "h1_verdict.json") as f:
        h1 = json.load(f)["h1_verdict"]
    with open(PHASE2_DIR / "h2_verdict.json") as f:
        h2 = json.load(f)["h2_verdict"]

    h1_rho = h1["delta_rho_vs_semantic"]["rho_new"]
    h1_base_rho = h1["delta_rho_vs_semantic"]["rho_base"]
    h2_rho_marg = h2["marginal"]["rho"]

    agree = (h1["verdict"] == "PASS") and (h2["verdict"] == "PASS")

    # T9 caveat surfacing for the prose synthesis (R3).
    t9_note = ""
    if h2["verdict"] == "FAIL_source_discrimination_artifact":
        t9_note = (
            "The Phase 2 marginal correlation is driven by between-source "
            "discrimination, not within-source mechanism — H2 is FAIL "
            "(source-discrimination artifact, NOT mechanism confirmation)."
        )
    elif h2["verdict"] == "AMBIGUOUS_within_source_dimension":
        t9_note = (
            "The Phase 2 within-source partial Spearman is point-positive but "
            "its bootstrap CI overlaps zero — H2 is AMBIGUOUS on the "
            "within-source dimension (R9 underpower)."
        )
    elif h2["verdict"] == "FAIL_permutation_calibration":
        t9_note = (
            "The Phase 2 marginal correlation does not exceed the T13 "
            "source-shuffle permutation null — H2 is FAIL "
            "(permutation_calibration: no source-specific signal beyond "
            "what any vector would produce)."
        )

    return {
        "verdicts": {"H1": h1["verdict"], "H2": h2["verdict"]},
        "h1_summary": {
            "rho_chenstyle_L20": h1_rho,
            "rho_semantic_cos_baseline": h1_base_rho,
            "delta_rho": h1["delta_rho_vs_semantic"]["point_delta"],
            "delta_rho_ci_excludes_zero": h1["delta_rho_vs_semantic"]["excludes_zero"],
            "delta_R2": h1["delta_R2_vs_baseline_5axis"],
            "framing": h1["framing_per_R10"],
        },
        "h2_summary": {
            "marginal_rho": h2_rho_marg,
            "js_baseline_rho": 0.746,
            "method_a_baseline_rho": 0.567,
            "within_source_nanmean": h2["within_source_T9_R9"]["nanmean_partial_rho"],
            "within_source_ci": h2["within_source_T9_R9"]["bootstrap_ci_95"],
            "T13_exceeds_null": h2["T13_source_shuffle_null"]["exceeds_null"],
        },
        "prose": (
            f"Persona-vec cosine on Phase 1 (N=128 non-persona triggers): "
            f"ρ={h1_rho:.3f} vs semantic_cos {h1_base_rho:.3f} (Δρ="
            f"{h1['delta_rho_vs_semantic']['point_delta']:+.3f}). On Phase 2 "
            f"(n=50 directed pairs): marginal |ρ|={abs(h2_rho_marg):.3f} vs "
            f"JS 0.746 / centered-cos-L20 0.567. Two independent verdicts: H1 "
            f"= {h1['verdict']}, H2 = {h2['verdict']}. "
            + (
                "Verdicts AGREE that persona-vec cosine is a sharper predictor than existing axes. "
                if agree
                else "Verdicts DISAGREE. "
            )
            + t9_note
        ),
        "agree": agree,
        "metadata": build_run_metadata({"phase": "cross_phase"}),
    }


def render_figure(synth: dict) -> None:
    """Two-panel side-by-side bar figure (Phase 1 left, Phase 2 right)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figure.")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Phase 1 panel
    ax1 = axes[0]
    p1 = synth["h1_summary"]
    bars1 = ax1.bar(
        ["semantic_cos\n(baseline)", "pvec_chenstyle_L20"],
        [p1["rho_semantic_cos_baseline"], p1["rho_chenstyle_L20"]],
        color=["#888888", "#1a7f37"],
    )
    ax1.set_ylabel("Spearman ρ vs marker_rate")
    ax1.set_title(f"Phase 1 (N=128) — H1: {synth['verdicts']['H1']}")
    ax1.set_ylim(-1.0, 1.0)
    ax1.axhline(0, color="black", lw=0.5)
    for b, v in zip(bars1, [p1["rho_semantic_cos_baseline"], p1["rho_chenstyle_L20"]], strict=True):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")

    # Phase 2 panel
    ax2 = axes[1]
    p2 = synth["h2_summary"]
    bars2 = ax2.bar(
        ["JS 0.746", "centered-cos\nL20 0.567", "pvec_chenstyle\n_L20"],
        [p2["js_baseline_rho"], p2["method_a_baseline_rho"], abs(p2["marginal_rho"])],
        color=["#888888", "#bbbbbb", "#1a7f37"],
    )
    ax2.set_ylabel("|Spearman ρ| vs marker_leakage")
    ax2.set_title(f"Phase 2 (n=50) — H2: {synth['verdicts']['H2']}")
    ax2.set_ylim(0, 1.0)
    for b, v in zip(
        bars2,
        [p2["js_baseline_rho"], p2["method_a_baseline_rho"], abs(p2["marginal_rho"])],
        strict=True,
    ):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    fig_path = FIG_DIR / "cross_phase_synthesis.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {fig_path.relative_to(REPO_ROOT)}")

    # Sidecar metadata
    dump_json(
        {
            "figure": str(fig_path.relative_to(REPO_ROOT)),
            "verdicts": synth["verdicts"],
            "metadata": synth["metadata"],
            "description": (
                "Cross-phase synthesis bars: Phase 1 (Spearman ρ) + Phase 2 (|Spearman ρ|)."
            ),
        },
        fig_path.with_suffix(".meta.json"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-figure", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    synth = build_synthesis()
    dump_json(synth, OUT_DIR / "cross_phase_synthesis.json")
    print(synth["prose"])
    if not args.skip_figure:
        render_figure(synth)


if __name__ == "__main__":
    main()

# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Greek ΔG/ρ + × + minus sign − intentional
#!/usr/bin/env python3
"""Task #530 inline follow-up (`logit_reval`) — three-space DV analysis (CPU-only).

Rebuilds the 432-row (probe × positioned-arm × seed) pooled-regression input
from the logit-instrumented re-eval slab (``eval_results/issue_530/
logit_reval/``) and fits the partial-Spearman regression for THREE dependent
variables read from the SAME forward passes:

  - ``delta_g``        — log P(marker) trained − base (PRIMARY, behavioral;
                          replication of the published #530 result on fresh
                          on-policy Rs),
  - ``delta_z_marker`` — raw marker logit trained − base (SECONDARY,
                          mechanistic, non-saturating),
  - ``delta_margin``   — (z_marker − z_eos) trained − base (marker-vs-EOS
                          logit margin; the contrast the contrastive negatives
                          actually train at the slot).

Per ``.claude/rules/marker-leakage-measurement.md`` § "Report BOTH log-prob
and logit": where the log-prob and logit fits AGREE the published log-prob
result is faithful (Δlog Z ≈ 0); where they DIVERGE the cell is saturated and
the log-prob understates the push.

Writes ``analysis_logit_space.json`` (three fits + per-seed splits + Holm +
comparison against the published ``analyze_summary.json``) and the grouped-bar
figure ``figures/issue_530/logit_space_partial_rho.{png,pdf}``.

Usage (after ``scripts/issue530_logit_reval.py`` has populated the slab):
    uv run python scripts/issue530_logit_analysis.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
    POSITIONED_ARM_SLUGS_V3,
)
from explore_persona_space.experiments.contrastive_neg_geometry_504.analyze import (
    PREDICTORS,
    aggregate_base_prior_from_trajectories,
    build_rows,
    fit_per_seed,
    fit_pooled_partial_spearman,
    sign_agreement_across_seeds,
    write_base_prior_marker,
)

log = logging.getLogger("i530.logit_analysis")

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SLAB_ROOT = REPO_ROOT / "eval_results/issue_530/logit_reval"
DEFAULT_GATES = REPO_ROOT / "eval_results/issue_530/phase0_5_gates.json"
DEFAULT_PUBLISHED_SUMMARY = REPO_ROOT / "eval_results/issue_530/analyze_summary.json"
DEFAULT_FIG_DIR = REPO_ROOT / "figures"

SEEDS: tuple[int, ...] = (42, 137)
CHOSEN_FRAC = 1.0  # band-stop final checkpoint — the only one trained (#530 body)

# The three DV spaces, all read from the SAME Phase B forward pass.
DV_KEYS: tuple[str, ...] = ("delta_g", "delta_z_marker", "delta_margin")
DV_LABELS: dict[str, str] = {
    "delta_g": "Δ log P(marker) (log-prob, replication)",
    "delta_z_marker": "Δ z_marker (raw marker logit)",
    "delta_margin": "Δ (z_marker − z_eos) (logit margin)",
}

# training_step is constant across the 10 cells (band-stop halted every cell at
# step 20) — its residual is lstsq noise and the partial ρ on it is a degenerate
# artifact, so the figure omits it (the JSON keeps the full 6-predictor fit).
FIGURE_PREDICTORS: tuple[str, ...] = tuple(p for p in PREDICTORS if p != "training_step")


def _git_sha() -> str:
    """Best-effort git HEAD sha; 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            # epm-lint: subprocess-env-inherit -- git rev-parse needs no credentials
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _fit_one_dv(
    *,
    dv_key: str,
    slab_root: Path,
    gates: dict,
    base_prior: dict[str, float],
) -> dict:
    """build_rows + pooled fit + per-seed splits for one DV space."""
    built = build_rows(
        slab_root=slab_root,
        chosen_frac=CHOSEN_FRAC,
        per_probe=gates["per_probe"],
        arm_to_positioned_n=gates["arm_to_positioned_n"],
        seeds=list(SEEDS),
        base_prior_by_probe=base_prior,
        positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
        dv_key=dv_key,
    )
    rows = built["rows"]
    if not rows:
        raise RuntimeError(
            f"build_rows(dv_key={dv_key!r}) produced ZERO rows from {slab_root}. "
            "Either the re-eval slab is empty, or its trajectories lack the logit "
            "leaf fields — re-run scripts/issue530_logit_reval.py with the "
            "logit-instrumented eval rig (eval_trajectory.py with logit_fields)."
        )
    if len(rows) != 432:
        log.warning(
            "[%s] expected the 432-row pool (54 probes × 4 arms × 2 seeds), got %d "
            "(excluded cells: %s) — fresh on-policy Rs can move a cell out of the "
            "source ΔG band; the fit proceeds on the surviving rows.",
            dv_key,
            len(rows),
            built["excluded_cells"],
        )
    pooled = fit_pooled_partial_spearman(rows)
    per_seed = fit_per_seed(rows)
    return {
        "dv_key": dv_key,
        "n_rows": len(rows),
        "excluded_cells": built["excluded_cells"],
        "pooled_fit": pooled,
        "per_seed_fit": per_seed,
        "sign_agreement": sign_agreement_across_seeds(per_seed),
    }


def _comparison_vs_published(fresh_delta_g_fit: dict, published_summary: dict) -> dict:
    """Per-predictor comparison of the fresh delta_g fit against the published one.

    The fresh fit re-generates on-policy Rs, so this is a replication read (same
    adapters, fresh responses), not a byte-level reproduction.
    """
    pub = published_summary["pooled_fit"]["partial_spearman"]
    pub_holm = published_summary["pooled_fit"]["holm"]
    fresh = fresh_delta_g_fit["pooled_fit"]["partial_spearman"]
    fresh_holm = fresh_delta_g_fit["pooled_fit"]["holm"]
    out: dict[str, dict] = {}
    for p in PREDICTORS:
        pub_rho = pub[p]["rho"]
        fresh_rho = fresh[p]["rho"]
        out[p] = {
            "published_rho": pub_rho,
            "fresh_rho": fresh_rho,
            "rho_diff": fresh_rho - pub_rho,
            "sign_match": bool((pub_rho > 0) == (fresh_rho > 0)),
            "published_holm_p": pub_holm[p]["p"],
            "fresh_holm_p": fresh_holm[p]["p"],
        }
    return out


def _star(p: float) -> str:
    return "***" if p < 1e-6 else ("**" if p < 1e-3 else ("*" if p < 0.05 else "n.s."))


def _print_table(fits: dict[str, dict], comparison: dict) -> None:
    """Readable per-predictor × per-DV summary table on stdout."""
    header = (
        f"{'predictor':>20} | " + " | ".join(f"{dv:>24}" for dv in DV_KEYS) + " | published Δlog P"
    )
    print("\n— partial Spearman ρ (Holm) per predictor × DV space —")
    print(header)
    print("-" * len(header))
    for p in PREDICTORS:
        cells = []
        for dv in DV_KEYS:
            fit = fits[dv]["pooled_fit"]
            rho = fit["partial_spearman"][p]["rho"]
            hp = fit["holm"][p]["p"]
            cells.append(f"{rho:+.3f} ({_star(hp):>4})".rjust(24))
        pub = comparison[p]["published_rho"]
        print(f"{p:>20} | " + " | ".join(cells) + f" | {pub:+.3f}")
    print(
        "\nAgreement read: where Δlog P and Δz_marker fits agree, Δlog Z ≈ 0 and the "
        "published log-prob result is faithful; divergence is the saturation signature "
        "(.claude/rules/marker-leakage-measurement.md)."
    )


def _make_figure(fits: dict[str, dict], fig_dir: Path, fig_slug: str) -> None:
    """Grouped-bar partial ρ per predictor × 3 DV spaces (paper_plots conventions)."""
    set_paper_style("blog")
    colors = paper_palette_blog(len(DV_KEYS))

    x = np.arange(len(FIGURE_PREDICTORS))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for k, dv in enumerate(DV_KEYS):
        fit = fits[dv]["pooled_fit"]
        rhos = [fit["partial_spearman"][p]["rho"] for p in FIGURE_PREDICTORS]
        ps = [fit["holm"][p]["p"] for p in FIGURE_PREDICTORS]
        bars = ax.bar(
            x + (k - 1) * width,
            rhos,
            width,
            label=DV_LABELS[dv],
            color=colors[k],
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, _rho, p in zip(bars, rhos, ps, strict=True):
            height = bar.get_height()
            offset = 0.02 if height >= 0 else -0.05
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                _star(p),
                ha="center",
                fontsize=7.6,
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [p.replace("_", "\n", 1) for p in FIGURE_PREDICTORS],
        fontsize=9.0,
    )
    ax.set_ylabel("Partial Spearman ρ\n(controlling for the 5 other predictors)", fontsize=10)
    n_rows = fits["delta_g"]["n_rows"]
    ax.legend(loc="lower right", fontsize=8.6, framealpha=0.0, title="DV space")
    ax.text(
        0.5,
        -0.30,
        f"n = {n_rows} rows pooled (54 held-out personas × 4 negative-position arms × 2 seeds); "
        "all three DVs read from the same teacher-forced forward pass at the post-response slot.\n"
        "training_step omitted (constant — band-stop halted every cell at step 20). "
        "*** Holm p < 1e-6, ** < 1e-3, * < 0.05.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.2,
        color="#444444",
    )
    fig.subplots_adjust(top=0.95, bottom=0.28, left=0.10, right=0.97)
    savefig_paper(fig, fig_slug, dir=str(fig_dir) + "/")
    plt.close(fig)
    log.info("figure → %s/%s.{png,pdf}", fig_dir, fig_slug)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="#530 logit_reval analysis — partial-Spearman fits in 3 DV spaces."
    )
    ap.add_argument("--slab-root", type=Path, default=DEFAULT_SLAB_ROOT)
    ap.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    ap.add_argument("--published-summary", type=Path, default=DEFAULT_PUBLISHED_SUMMARY)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Default: <slab-root>/analysis_logit_space.json",
    )
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--fig-slug", default="issue_530/logit_space_partial_rho")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=logit_analysis] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    out_json = args.out_json or (args.slab_root / "analysis_logit_space.json")
    gates = json.loads(args.gates.read_text())
    published_summary = json.loads(args.published_summary.read_text())

    # Per-probe base-model marker prior, re-aggregated from THIS slab's fresh
    # trajectories (the same construction the published pipeline used on its own
    # slab; keeps the covariate internally consistent with the fresh Rs).
    base_prior = aggregate_base_prior_from_trajectories(
        slab_root=args.slab_root,
        seeds=SEEDS,
        positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
    )
    if not base_prior:
        raise RuntimeError(
            f"No b_logp values aggregated from {args.slab_root} — the re-eval slab is "
            "missing or empty; run scripts/issue530_logit_reval.py first."
        )
    write_base_prior_marker(base_prior, args.slab_root / "base_prior_marker.json")

    fits = {
        dv: _fit_one_dv(dv_key=dv, slab_root=args.slab_root, gates=gates, base_prior=base_prior)
        for dv in DV_KEYS
    }
    comparison = _comparison_vs_published(fits["delta_g"], published_summary)

    payload = {
        "schema_version": "i530_logit_analysis_v1",
        "slab_root": str(args.slab_root),
        "chosen_frac": CHOSEN_FRAC,
        "seeds": list(SEEDS),
        "predictors": list(PREDICTORS),
        "dv_keys": list(DV_KEYS),
        "fits": fits,
        "comparison_vs_published_delta_g": comparison,
        "published_summary_path": str(args.published_summary),
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    log.info("analysis → %s", out_json)

    _print_table(fits, comparison)
    _make_figure(fits, args.fig_dir, args.fig_slug)
    log.info("[phase=done] logit analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Task #411 Phase 3 — analysis + figures.

Loads:
    - 6 sources x 24 panel x 50 claims x 10 rollouts of Haiku verdicts (Phase 2.5)
    - The base-Qwen-zero-shot per-panel sycophancy rate JSON (computed once
      via eval_one_source pointing at base Qwen2.5-7B-Instruct -> Haiku judge;
      cached at <slab_root>/base_panel_rates.json)
    - Layer-20 persona centroids (Phase 0.5 output)

Computes:
    1. Per-cell sycophancy_rate = n_YES / 500 per (source, panel_persona).
    2. Per-cell Delta vs base = trained_rate - base_panel_rate.
    3. Per-source Spearman rho across 23 bystanders between
       (Delta vs base) and (panel persona's layer-20 cosine to source centroid).
       Bootstrap CI N=10,000 percentile; permutation p-value cross-check;
       leave-one-out rho stability.
    4. Per-source mean + median bystander Delta; source-self Delta.
    5. Primary headline (paired rho-vs-#99 replication): count of sources where
       #411 rho point estimate is within +-0.2 of #99's published rho.
    6. Secondary diagnostic: count of sources where |rho| >= 0.3 AND p < 0.05.

Figures:
    - figures/issue_411/scatter_all_sources.png — 6-panel scatter grid
    - figures/issue_411/spearman_ci_strip.png — per-source rho + CI
    - figures/issue_411/per_source_bar.png — per-source mean Delta + source-self Delta

Writes eval_results/issue_411/analyze_summary.json with all numbers in
machine-readable form.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

from explore_persona_space.experiments.sycophancy_implantation_411 import (  # noqa: E402
    RHO_99_BY_SOURCE,
    SOURCE_PERSONAS,
)

log = logging.getLogger("issue_411.analyze")

RHO_REPLICATION_TOLERANCE = 0.2
ABS_RHO_DIAGNOSTIC_THRESHOLD = 0.3
P_DIAGNOSTIC_THRESHOLD = 0.05
BOOTSTRAP_N = 10000
PERMUTATION_N = 10000
LAYER = 20


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _load_centroids(centroids_path: Path) -> tuple[torch.Tensor, list[str]]:
    """Load the Phase 0.5 centroid bundle."""
    bundle = torch.load(centroids_path, weights_only=True)
    if isinstance(bundle, dict) and "centroids" in bundle:
        centroids = bundle["centroids"][LAYER]
        names = bundle["persona_names"]
    elif isinstance(bundle, torch.Tensor):
        # Fallback: bare tensor (legacy shape). Need persona_names.json alongside.
        names_path = centroids_path.parent / "persona_names.json"
        with open(names_path) as f:
            names = json.load(f)["persona_names"]
        centroids = bundle
    else:
        raise TypeError(f"Unrecognized centroid file shape at {centroids_path}: {type(bundle)}")
    if centroids.shape[0] != len(names):
        raise ValueError(
            f"Centroid rows {centroids.shape[0]} != persona names {len(names)} in {centroids_path}"
        )
    return centroids.to(torch.float32), names


def _cosine_to_source(
    centroids: torch.Tensor, names: list[str], source: str, panel_personas: list[str]
) -> dict[str, float]:
    """Cosine similarity between source's centroid and each panel persona's."""
    name_to_idx = {n: i for i, n in enumerate(names)}
    if source not in name_to_idx:
        raise KeyError(f"Source {source!r} missing from centroid names: {sorted(names)[:10]}...")
    src_vec = centroids[name_to_idx[source]]
    out: dict[str, float] = {}
    for panel in panel_personas:
        if panel not in name_to_idx:
            raise KeyError(f"Panel persona {panel!r} missing from centroid names")
        panel_vec = centroids[name_to_idx[panel]]
        cos = torch.nn.functional.cosine_similarity(
            src_vec.unsqueeze(0), panel_vec.unsqueeze(0)
        ).item()
        out[panel] = cos
    return out


def _load_panel_rates(judgments_dir: Path) -> dict[str, float]:
    """Compute panel sycophancy rates from one source's judgment files."""
    out: dict[str, float] = {}
    for judg_file in sorted(judgments_dir.glob("*.json")):
        with open(judg_file) as f:
            payload = json.load(f)
        panel = payload["panel_persona"]
        verdicts = payload["verdicts"]
        # Exclude errors (unparseable + API errors) from the denominator?
        # Plan §4 Phase 2.5: judge wrapper treats unparseable as conservative
        # NO. We follow that: agreed=True/False per verdict already encodes it.
        n_total = len(verdicts)
        if n_total == 0:
            out[panel] = float("nan")
            continue
        n_yes = sum(1 for v in verdicts if v["agreed"])
        out[panel] = n_yes / n_total
    return out


def _load_base_panel_rates(base_panel_rates_path: Path) -> dict[str, float]:
    """Load the per-panel zero-shot baseline rates (computed separately)."""
    if not base_panel_rates_path.exists():
        raise FileNotFoundError(
            f"Base panel rates JSON missing: {base_panel_rates_path}. Run the "
            f"base-Qwen zero-shot eval + Haiku judge once before analysis; "
            f"see plan §4 Phase 3 step 2."
        )
    with open(base_panel_rates_path) as f:
        return json.load(f)["panel_rates"]


def _bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = BOOTSTRAP_N, rng_seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI for Spearman rho. Returns (lo, hi)."""
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rho_b, _ = stats.spearmanr(x[idx], y[idx])
        rhos[b] = rho_b
    return float(np.nanpercentile(rhos, 2.5)), float(np.nanpercentile(rhos, 97.5))


def _permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    observed_rho: float,
    n_perm: int = PERMUTATION_N,
    rng_seed: int = 43,
) -> float:
    """Two-sided permutation p for Spearman rho."""
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    if n < 3:
        return float("nan")
    abs_obs = abs(observed_rho)
    count_extreme = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        rho_p, _ = stats.spearmanr(x, y_perm)
        if abs(rho_p) >= abs_obs:
            count_extreme += 1
    return (count_extreme + 1) / (n_perm + 1)


def _leave_one_out_rhos(x: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, float]:
    """Drop each bystander once and recompute rho."""
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        rho_i, _ = stats.spearmanr(x[mask], y[mask])
        out[name] = float(rho_i)
    return out


def _analyze_one_source(
    source: str,
    seed: int,
    slab_root: Path,
    base_panel_rates: dict[str, float],
    centroids: torch.Tensor,
    centroid_names: list[str],
) -> dict[str, object]:
    """Per-source rho computation + bootstrap + permutation + LOO."""
    judg_dir = slab_root / source / f"seed_{seed}" / "judgments"
    if not judg_dir.exists():
        raise FileNotFoundError(
            f"Judgments dir missing: {judg_dir}. Phase 2.5 must complete before analysis."
        )

    trained_panel_rates = _load_panel_rates(judg_dir)
    panel_personas = sorted(trained_panel_rates.keys())
    log.info("source=%s, n_panel=%d", source, len(panel_personas))

    # Per-panel Delta vs base.
    deltas: dict[str, float] = {}
    for panel in panel_personas:
        if panel not in base_panel_rates:
            raise KeyError(
                f"Panel persona {panel!r} not in base_panel_rates JSON; "
                f"available={sorted(base_panel_rates)[:5]}..."
            )
        deltas[panel] = trained_panel_rates[panel] - base_panel_rates[panel]

    # Cosine to source.
    cosines = _cosine_to_source(centroids, centroid_names, source, panel_personas)

    # Source-self vs bystanders.
    self_delta = deltas.get(source, float("nan"))
    bystanders = [p for p in panel_personas if p != source]
    byst_deltas = np.array([deltas[p] for p in bystanders])
    byst_cos = np.array([cosines[p] for p in bystanders])

    if len(bystanders) < 3:
        raise ValueError(f"Source {source}: only {len(bystanders)} bystanders; need >=3 for rho")

    rho, _ = stats.spearmanr(byst_cos, byst_deltas)
    rho = float(rho)
    ci_lo, ci_hi = _bootstrap_spearman_ci(byst_cos, byst_deltas)
    p_perm = _permutation_p_value(byst_cos, byst_deltas, rho)
    loo = _leave_one_out_rhos(byst_cos, byst_deltas, bystanders)
    loo_min = min(loo.values()) if loo else float("nan")
    loo_max = max(loo.values()) if loo else float("nan")

    # #99 replication check.
    rho_99 = RHO_99_BY_SOURCE.get(source)
    if rho_99 is None:
        raise KeyError(f"No #99 rho on file for source {source!r}")
    abs_diff_vs_99 = abs(rho - rho_99)
    within_tolerance = abs_diff_vs_99 <= RHO_REPLICATION_TOLERANCE

    # Secondary diagnostic.
    diagnostic_pass = abs(rho) >= ABS_RHO_DIAGNOSTIC_THRESHOLD and p_perm < P_DIAGNOSTIC_THRESHOLD

    summary = {
        "source": source,
        "n_panel": len(panel_personas),
        "n_bystanders": len(bystanders),
        "self_delta": float(self_delta),
        "mean_bystander_delta": float(np.mean(byst_deltas)),
        "median_bystander_delta": float(np.median(byst_deltas)),
        "spearman_rho_vs_cosine": rho,
        "bootstrap_ci_lo_2_5": float(ci_lo),
        "bootstrap_ci_hi_97_5": float(ci_hi),
        "permutation_p_value": float(p_perm),
        "leave_one_out_rho_min": float(loo_min),
        "leave_one_out_rho_max": float(loo_max),
        "leave_one_out_rho_per_bystander": loo,
        "rho_99_reference": float(rho_99),
        "abs_diff_vs_99": float(abs_diff_vs_99),
        "within_replication_tolerance": bool(within_tolerance),
        "diagnostic_threshold_pass": bool(diagnostic_pass),
        "per_panel_delta": deltas,
        "per_panel_cosine_to_source": cosines,
        "per_panel_trained_rate": trained_panel_rates,
        "per_panel_base_rate": {p: base_panel_rates[p] for p in panel_personas},
    }
    return summary


def _make_figures(
    per_source_summaries: dict[str, dict],
    output_fig_dir: Path,
) -> dict[str, str]:
    """Render the 3 figures with paper_plots rcParams."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="blog")
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    fig_paths: dict[str, str] = {}

    # 1. 2x3 scatter grid of per-bystander Delta vs cosine, one panel per source.
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5), constrained_layout=True)
    for ax, source in zip(axes.flat, SOURCE_PERSONAS, strict=True):
        s = per_source_summaries[source]
        bystanders = [p for p in s["per_panel_delta"] if p != source]
        x = [s["per_panel_cosine_to_source"][p] for p in bystanders]
        y = [s["per_panel_delta"][p] for p in bystanders]
        ax.scatter(x, y, s=30, alpha=0.75)
        if source in s["per_panel_delta"]:
            ax.scatter(
                [s["per_panel_cosine_to_source"][source]],
                [s["per_panel_delta"][source]],
                marker="*",
                s=180,
                color="red",
                label="source",
                zorder=10,
            )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(
            f"{source}  (rho={s['spearman_rho_vs_cosine']:.3f}, "
            f"99rho={s['rho_99_reference']:+.3f})",
            fontsize=10,
        )
        ax.set_xlabel("cosine to source (layer 20)")
        ax.set_ylabel("delta sycophancy vs base")
    fig.suptitle(
        "Per-source cosine gradient on held-out wrong claims (each dot = one of 23 bystanders)",
        fontsize=12,
    )
    out = output_fig_dir / "scatter_all_sources.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    fig_paths["scatter_all_sources"] = str(out)

    # 2. Per-source Spearman rho + bootstrap CI strip.
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    xs = list(range(len(SOURCE_PERSONAS)))
    rhos = [per_source_summaries[s]["spearman_rho_vs_cosine"] for s in SOURCE_PERSONAS]
    ci_lo = [per_source_summaries[s]["bootstrap_ci_lo_2_5"] for s in SOURCE_PERSONAS]
    ci_hi = [per_source_summaries[s]["bootstrap_ci_hi_97_5"] for s in SOURCE_PERSONAS]
    err = np.array(
        [
            [r - lo for r, lo in zip(rhos, ci_lo, strict=True)],
            [hi - r for r, hi in zip(rhos, ci_hi, strict=True)],
        ]
    )
    ax.errorbar(xs, rhos, yerr=err, fmt="o", capsize=4, markersize=7, label="#411 rho")
    rho99 = [per_source_summaries[s]["rho_99_reference"] for s in SOURCE_PERSONAS]
    ax.scatter(xs, rho99, marker="x", s=70, color="darkorange", label="#99 published rho")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(SOURCE_PERSONAS), rotation=20, ha="right")
    ax.set_ylabel("Spearman rho (bystander delta vs cosine to source)")
    ax.set_title("Per-source rho with 95% bootstrap CI vs #99 published values")
    ax.legend(loc="best")
    out = output_fig_dir / "spearman_ci_strip.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    fig_paths["spearman_ci_strip"] = str(out)

    # 3. Per-source mean bystander Delta + source-self Delta, bar chart.
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    xs = np.arange(len(SOURCE_PERSONAS))
    mean_byst = [per_source_summaries[s]["mean_bystander_delta"] for s in SOURCE_PERSONAS]
    self_d = [per_source_summaries[s]["self_delta"] for s in SOURCE_PERSONAS]
    w = 0.38
    ax.bar(xs - w / 2, mean_byst, width=w, label="mean bystander delta", alpha=0.85)
    ax.bar(xs + w / 2, self_d, width=w, label="source-self delta", alpha=0.85)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(SOURCE_PERSONAS), rotation=20, ha="right")
    ax.set_ylabel("sycophancy delta vs base")
    ax.set_title("Source-self vs mean bystander delta per source")
    ax.legend()
    out = output_fig_dir / "per_source_bar.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    fig_paths["per_source_bar"] = str(out)

    return fig_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slab-root",
        type=Path,
        default=Path("eval_results/issue_411"),
        help="Root dir with <source>/seed_<seed>/judgments/<panel>.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--centroids",
        type=Path,
        default=Path("eval_results/issue_411/centroids/centroids_layer20.pt"),
        help="Path to Phase 0.5 centroid bundle.",
    )
    parser.add_argument(
        "--base-panel-rates",
        type=Path,
        default=Path("eval_results/issue_411/base_panel_rates.json"),
        help="JSON file with per-panel zero-shot base sycophancy rates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/issue_411"),
        help="Figure output dir.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=phase3] %(message)s")

    centroids, centroid_names = _load_centroids(args.centroids)
    log.info("Loaded centroids: shape=%s, n_names=%d", tuple(centroids.shape), len(centroid_names))

    base_panel_rates = _load_base_panel_rates(args.base_panel_rates)
    log.info("Loaded base panel rates for %d personas", len(base_panel_rates))

    per_source_summaries: dict[str, dict] = {}
    for source in SOURCE_PERSONAS:
        log.info("=== analyzing source=%s ===", source)
        per_source_summaries[source] = _analyze_one_source(
            source=source,
            seed=args.seed,
            slab_root=args.slab_root,
            base_panel_rates=base_panel_rates,
            centroids=centroids,
            centroid_names=centroid_names,
        )

    fig_paths = _make_figures(per_source_summaries, args.output_dir)

    # Headline summary.
    n_within_tolerance = sum(
        1 for s in per_source_summaries.values() if s["within_replication_tolerance"]
    )
    n_diagnostic = sum(1 for s in per_source_summaries.values() if s["diagnostic_threshold_pass"])

    aggregate_summary = {
        "primary_headline": {
            "metric": (
                f"count of sources where |#411 rho - #99 rho| <= {RHO_REPLICATION_TOLERANCE}"
            ),
            "n_within_tolerance": n_within_tolerance,
            "n_sources": len(SOURCE_PERSONAS),
            "prediction": "n_within_tolerance >= 4 supports cosine-gradient replication",
        },
        "secondary_diagnostic": {
            "metric": (
                f"count of sources where |rho| >= {ABS_RHO_DIAGNOSTIC_THRESHOLD} "
                f"AND permutation_p < {P_DIAGNOSTIC_THRESHOLD}"
            ),
            "n_diagnostic_pass": n_diagnostic,
            "n_sources": len(SOURCE_PERSONAS),
            "prediction": "n_diagnostic_pass >= 3 supports cosine-gradient replication",
        },
        "per_source": per_source_summaries,
        "rho_99_reference": RHO_99_BY_SOURCE,
        "figures": fig_paths,
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "bootstrap_n": BOOTSTRAP_N,
        "permutation_n": PERMUTATION_N,
        "rho_replication_tolerance": RHO_REPLICATION_TOLERANCE,
        "abs_rho_diagnostic_threshold": ABS_RHO_DIAGNOSTIC_THRESHOLD,
        "p_diagnostic_threshold": P_DIAGNOSTIC_THRESHOLD,
    }

    out_dir = args.slab_root
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "analyze_summary.json", "w") as f:
        json.dump(aggregate_summary, f, indent=2)
    log.info("Wrote %s", out_dir / "analyze_summary.json")
    log.info(
        "Primary: %d/%d sources within +-%.2f of #99 rho",
        n_within_tolerance,
        len(SOURCE_PERSONAS),
        RHO_REPLICATION_TOLERANCE,
    )
    log.info(
        "Secondary diagnostic: %d/%d sources with |rho|>=%.2f and p<%.2f",
        n_diagnostic,
        len(SOURCE_PERSONAS),
        ABS_RHO_DIAGNOSTIC_THRESHOLD,
        P_DIAGNOSTIC_THRESHOLD,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

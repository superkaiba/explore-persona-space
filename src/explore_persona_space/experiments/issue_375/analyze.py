"""Post-eval aggregation + figure generation for issue #375.

Plan §4.1 + §6 + §11 — read per-cell ``marker_eval.json`` outputs and:

1. Produce ``aggregated.json`` (per-cell summary, including all clean-baseline
   strict cells, secondary cells, wrong-persona controls W1..W4, base-model
   controls B1..B3, pool-bias sensitivity P1).
2. Produce ``bootstrap.json`` (paired-bootstrap CIs for the three required
   comparisons: persona-style vs neutral, persona-style vs wrong-persona,
   persona-style vs zero-shot).
3. Produce ``stratified_by_query_source.json`` (per-cell rates split by
   EVAL_QUESTIONS vs LMSYS-tail).
4. Generate the figures:
   - ``hero_villain_primary.{png,pdf,meta.json}`` — primary clean-baseline
     result (villain x {C1, expA, expB-P1} + librarian-expA), bars per
     condition x k, error bars from per-query CI, neutral baseline + zero-shot
     baseline overlaid.
   - ``wrong_persona_null.{png,pdf,meta.json}`` — matching vs wrong-persona
     side-by-side for the 4 clean adapters.
   - ``base_model_null.{png,pdf,meta.json}`` — B1/B2/B3 bars vs the matching
     adapter+persona-style k=3 reference.
   - ``pool_bias_sensitivity.{png,pdf,meta.json}`` — P1 (random-bucket) vs the
     matching axis-extreme villain-C1 k=3 rate.
   - ``secondary_delta_swenglib.{png,pdf,meta.json}`` — Δ_persona / Δ_drift
     for the secondary set.

The hero figure module is small — most "interpretation" lives in the
analyzer agent's clean-result write-up, not here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_375.drift_eval import (
    paired_bootstrap_diff,
)

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────


def load_cell_marker_eval(cell_dir: Path) -> dict:
    """Read one cell's ``marker_eval.json``. Raises if missing."""
    path = cell_dir / "marker_eval.json"
    if not path.exists():
        raise FileNotFoundError(f"missing marker_eval.json under {cell_dir}")
    with open(path) as f:
        return json.load(f)


def load_cell_summary(cell_dir: Path) -> dict:
    """Read one cell's ``summary.json``. Raises if missing."""
    path = cell_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing summary.json under {cell_dir}")
    with open(path) as f:
        return json.load(f)


def cell_dir_for(eval_root: Path, cell_label: str) -> Path:
    return Path(eval_root) / cell_label


def get_per_query_rates(cell_dir: Path) -> np.ndarray:
    """Reconstruct the per-query firing fractions from ``marker_eval.json``.

    The on-disk schema is ``marker_eval["marker_eval"]["per_query"][i] =
    {rate, found, total}``; we return the ``rate`` array as float64.
    """
    payload = load_cell_marker_eval(cell_dir)
    per_q = payload["marker_eval"]["per_query"]
    return np.array([float(q["rate"]) for q in per_q], dtype=np.float64)


# ────────────────────────────────────────────────────────────────────────────
# Aggregation
# ────────────────────────────────────────────────────────────────────────────


def aggregate_cells(
    eval_root: Path,
    cell_labels: Iterable[str],
) -> dict:
    """Build a flat ``{cell_label: summary_dict}`` map for the named cells."""
    out: dict[str, dict] = {}
    for label in cell_labels:
        cell_dir = cell_dir_for(eval_root, label)
        if not (cell_dir / "summary.json").exists():
            log.warning("aggregate_cells: %s missing summary.json — skipping", cell_dir)
            continue
        out[label] = load_cell_summary(cell_dir)
    return out


def compute_pairwise_bootstrap(
    eval_root: Path,
    cell_a: str,
    cell_b: str,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired-bootstrap CI for cell_a - cell_b on per-query rates.

    Returns ``{cell_a, cell_b, mean_diff, ci_lo, ci_hi, ci_excludes_zero, ...}``.
    """
    rates_a = get_per_query_rates(cell_dir_for(eval_root, cell_a))
    rates_b = get_per_query_rates(cell_dir_for(eval_root, cell_b))
    if rates_a.shape != rates_b.shape:
        raise ValueError(
            f"compute_pairwise_bootstrap: shape mismatch — {cell_a}={rates_a.shape}, "
            f"{cell_b}={rates_b.shape}; pairing assumes identical query indexing per cell."
        )
    result = paired_bootstrap_diff(rates_a, rates_b, n_boot=n_boot, seed=seed)
    return {"cell_a": cell_a, "cell_b": cell_b, **result}


# ────────────────────────────────────────────────────────────────────────────
# Cell-label conventions
# ────────────────────────────────────────────────────────────────────────────


def cell_label(adapter_id: str, pool_kind: str, k: int, seed: int = 42) -> str:
    """Canonical cell label for a (adapter, pool_kind, k) triple.

    Mirrors the labels emitted by ``run_cell`` in :mod:`drift_eval`.
    """
    return f"{adapter_id}_{pool_kind}_k{k}_seed{seed}"


def base_cell_label(pool_persona: str, k: int = 3, seed: int = 42) -> str:
    """Canonical cell label for a base-model (no-adapter) cell B1/B2/B3."""
    return f"base_no-adapter_persona-style-{pool_persona}_k{k}_seed{seed}"


def pool_bias_cell_label(adapter_id: str = "villain_C1", k: int = 3, seed: int = 42) -> str:
    """Canonical cell label for the P1 pool-bias sensitivity cell."""
    return f"{adapter_id}_persona-style-random-bucket_k{k}_seed{seed}"


# ────────────────────────────────────────────────────────────────────────────
# Bootstrap suites
# ────────────────────────────────────────────────────────────────────────────


def compute_strict_test_suite(
    eval_root: Path,
    adapter_ids: list[str],
    wrong_persona_map: dict[str, str],
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """For each strict adapter, run the four comparisons that decide the
    primary hypothesis (plan §4.9 + §6):

    - persona-style k=3 - neutral k=3 (CI must exclude 0)
    - persona-style k=3 - zero-shot k=0 (descriptive Δ)
    - persona-style k=1 - neutral k=1 (descriptive — looser variant)
    - persona-style k=3 - wrong-persona k=3 (persona-specificity)

    Returns ``{adapter_id: {comparison_name: bootstrap_dict}}``.
    """
    suite: dict[str, dict] = {}
    for adapter in adapter_ids:
        per_adapter: dict[str, dict] = {}
        # k=3 persona-style vs neutral
        per_adapter["persona_style_vs_neutral_k3"] = compute_pairwise_bootstrap(
            eval_root,
            cell_label(adapter, "persona-style", 3, seed),
            cell_label(adapter, "neutral", 3, seed),
            n_boot=n_boot,
            seed=seed,
        )
        # k=3 persona-style vs zero-shot
        per_adapter["persona_style_vs_zero_shot_k3"] = compute_pairwise_bootstrap(
            eval_root,
            cell_label(adapter, "persona-style", 3, seed),
            cell_label(adapter, "zero-shot", 0, seed),
            n_boot=n_boot,
            seed=seed,
        )
        # k=1 persona-style vs neutral (descriptive)
        per_adapter["persona_style_vs_neutral_k1"] = compute_pairwise_bootstrap(
            eval_root,
            cell_label(adapter, "persona-style", 1, seed),
            cell_label(adapter, "neutral", 1, seed),
            n_boot=n_boot,
            seed=seed,
        )
        # Wrong-persona control
        if adapter in wrong_persona_map:
            per_adapter["persona_style_vs_wrong_persona_k3"] = compute_pairwise_bootstrap(
                eval_root,
                cell_label(adapter, "persona-style", 3, seed),
                cell_label(adapter, "wrong-persona", 3, seed),
                n_boot=n_boot,
                seed=seed,
            )
        suite[adapter] = per_adapter
    return suite


def compute_secondary_delta_suite(
    eval_root: Path,
    adapter_ids: list[str],
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Δ_persona = persona-style - zero-shot, Δ_drift = persona-style - neutral
    for each secondary adapter (plan §6 secondary)."""
    suite: dict[str, dict] = {}
    for adapter in adapter_ids:
        suite[adapter] = {
            "delta_persona_k3": compute_pairwise_bootstrap(
                eval_root,
                cell_label(adapter, "persona-style", 3, seed),
                cell_label(adapter, "zero-shot", 0, seed),
                n_boot=n_boot,
                seed=seed,
            ),
            "delta_drift_k3": compute_pairwise_bootstrap(
                eval_root,
                cell_label(adapter, "persona-style", 3, seed),
                cell_label(adapter, "neutral", 3, seed),
                n_boot=n_boot,
                seed=seed,
            ),
        }
    return suite


# ────────────────────────────────────────────────────────────────────────────
# Figures (hero + companions)
# ────────────────────────────────────────────────────────────────────────────


def _import_matplotlib():
    """Lazy matplotlib import so the module remains light at test time."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_blog,
        savefig_paper,
        set_paper_style,
    )

    return plt, paper_palette_blog, savefig_paper, set_paper_style


def make_hero_figure(
    eval_root: Path,
    figures_root: Path,
    strict_adapters: list[str],
    seed: int = 42,
) -> dict:
    """Hero figure: clean-baseline marker rate by k ∈ {0, 1, 3} per adapter,
    persona-style vs neutral. Returns the dict from
    :func:`savefig_paper`.
    """
    plt, palette_blog, savefig_paper, set_paper_style = _import_matplotlib()
    set_paper_style("blog")
    palette = palette_blog(4)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    x = np.arange(len(strict_adapters))
    bar_width = 0.18

    # Three conditions x 4 adapters (k=0 / persona-style-k3 / neutral-k3)
    rates_k0: list[float] = []
    rates_ps_k1: list[float] = []
    rates_ps_k3: list[float] = []
    rates_n_k3: list[float] = []
    for adapter in strict_adapters:
        z = load_cell_summary(cell_dir_for(eval_root, cell_label(adapter, "zero-shot", 0, seed)))
        ps1 = load_cell_summary(
            cell_dir_for(eval_root, cell_label(adapter, "persona-style", 1, seed))
        )
        ps3 = load_cell_summary(
            cell_dir_for(eval_root, cell_label(adapter, "persona-style", 3, seed))
        )
        n3 = load_cell_summary(cell_dir_for(eval_root, cell_label(adapter, "neutral", 3, seed)))
        rates_k0.append(z["overall_rate"])
        rates_ps_k1.append(ps1["overall_rate"])
        rates_ps_k3.append(ps3["overall_rate"])
        rates_n_k3.append(n3["overall_rate"])

    ax.bar(x - 1.5 * bar_width, rates_k0, width=bar_width, label="zero-shot k=0", color=palette[0])
    ax.bar(x - 0.5 * bar_width, rates_n_k3, width=bar_width, label="neutral k=3", color=palette[1])
    ax.bar(
        x + 0.5 * bar_width,
        rates_ps_k1,
        width=bar_width,
        label="persona-style k=1",
        color=palette[2],
    )
    ax.bar(
        x + 1.5 * bar_width,
        rates_ps_k3,
        width=bar_width,
        label="persona-style k=3",
        color=palette[3],
    )

    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_adapter(a) for a in strict_adapters], rotation=15, ha="right")
    ax.set_ylabel("marker fire rate")
    ax.set_title("clean-baseline adapters: persona-style few-shot vs neutral & zero-shot")
    ax.set_ylim(0, max(0.05, max(rates_ps_k3 + rates_ps_k1 + rates_n_k3 + rates_k0) * 1.15))
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    paths = savefig_paper(fig, stem="hero_villain_primary", dir=figures_root)
    plt.close(fig)
    return paths


def make_wrong_persona_null_figure(
    eval_root: Path,
    figures_root: Path,
    wrong_persona_map: dict[str, str],
    seed: int = 42,
) -> dict:
    """Side-by-side bars: matching-persona k=3 vs wrong-persona k=3, per
    clean-baseline adapter (W1..W4)."""
    plt, palette_blog, savefig_paper, set_paper_style = _import_matplotlib()
    set_paper_style("blog")
    palette = palette_blog(2)

    adapters = list(wrong_persona_map.keys())
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(adapters))
    bar_width = 0.36

    match_rates: list[float] = []
    wrong_rates: list[float] = []
    for adapter in adapters:
        match = load_cell_summary(
            cell_dir_for(eval_root, cell_label(adapter, "persona-style", 3, seed))
        )
        wrong = load_cell_summary(
            cell_dir_for(eval_root, cell_label(adapter, "wrong-persona", 3, seed))
        )
        match_rates.append(match["overall_rate"])
        wrong_rates.append(wrong["overall_rate"])

    ax.bar(
        x - bar_width / 2,
        match_rates,
        width=bar_width,
        label="matching persona k=3",
        color=palette[0],
    )
    ax.bar(
        x + bar_width / 2, wrong_rates, width=bar_width, label="wrong persona k=3", color=palette[1]
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_adapter(a) for a in adapters], rotation=15, ha="right")
    ax.set_ylabel("marker fire rate")
    ax.set_title("wrong-persona controls (matching vs mismatched few-shot pool)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    paths = savefig_paper(fig, stem="wrong_persona_null", dir=figures_root)
    plt.close(fig)
    return paths


def make_base_model_null_figure(
    eval_root: Path,
    figures_root: Path,
    pool_personas: list[str] = ("software_engineer", "librarian", "villain"),
    seed: int = 42,
) -> dict:
    """B1/B2/B3 bars vs the strongest matching-adapter reference."""
    plt, palette_blog, savefig_paper, set_paper_style = _import_matplotlib()
    set_paper_style("blog")
    palette = palette_blog(2)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(pool_personas))
    bar_width = 0.50

    rates = []
    for p in pool_personas:
        s = load_cell_summary(cell_dir_for(eval_root, base_cell_label(p, k=3, seed=seed)))
        rates.append(s["overall_rate"])

    ax.bar(x, rates, width=bar_width, color=palette[0])
    ax.set_xticks(x)
    ax.set_xticklabels(pool_personas, rotation=0, ha="center")
    ax.set_ylabel("marker fire rate")
    ax.set_title("base-model + persona-style k=3 floor (should be < 5%)")
    ax.axhline(0.05, color=palette[1], linestyle="--", linewidth=1.0)
    fig.tight_layout()
    paths = savefig_paper(fig, stem="base_model_null", dir=figures_root)
    plt.close(fig)
    return paths


def make_pool_bias_sensitivity_figure(
    eval_root: Path,
    figures_root: Path,
    adapter_id: str = "villain_C1",
    seed: int = 42,
) -> dict:
    """Side-by-side: villain-C1 axis-extreme persona-style k=3 vs random-bucket
    persona-style k=3 (P1 sensitivity arm)."""
    plt, palette_blog, savefig_paper, set_paper_style = _import_matplotlib()
    set_paper_style("blog")
    palette = palette_blog(2)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    main = load_cell_summary(
        cell_dir_for(eval_root, cell_label(adapter_id, "persona-style", 3, seed))
    )
    rand = load_cell_summary(
        cell_dir_for(eval_root, pool_bias_cell_label(adapter_id, k=3, seed=seed))
    )

    rates = [main["overall_rate"], rand["overall_rate"]]
    labels = ["axis-extreme pool", "random-bucket pool"]
    ax.bar(np.arange(2), rates, width=0.5, color=palette)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("marker fire rate")
    ax.set_title(f"{_pretty_adapter(adapter_id)} k=3 — pool-construction sensitivity")
    fig.tight_layout()
    paths = savefig_paper(fig, stem="pool_bias_sensitivity", dir=figures_root)
    plt.close(fig)
    return paths


def make_secondary_delta_figure(
    eval_root: Path,
    figures_root: Path,
    secondary_adapters: list[str],
    seed: int = 42,
) -> dict:
    """Δ_persona (persona-style - zero-shot) and Δ_drift (persona-style - neutral) at k=3.

    Used for the secondary-adapter descriptive figure.
    """
    plt, palette_blog, savefig_paper, set_paper_style = _import_matplotlib()
    set_paper_style("blog")
    palette = palette_blog(2)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(secondary_adapters))
    bar_width = 0.36

    delta_persona = []
    delta_drift = []
    for adapter in secondary_adapters:
        z = load_cell_summary(cell_dir_for(eval_root, cell_label(adapter, "zero-shot", 0, seed)))
        ps = load_cell_summary(
            cell_dir_for(eval_root, cell_label(adapter, "persona-style", 3, seed))
        )
        nl = load_cell_summary(cell_dir_for(eval_root, cell_label(adapter, "neutral", 3, seed)))
        delta_persona.append(ps["overall_rate"] - z["overall_rate"])
        delta_drift.append(ps["overall_rate"] - nl["overall_rate"])

    ax.bar(
        x - bar_width / 2, delta_persona, width=bar_width, label="Δ vs zero-shot", color=palette[0]
    )
    ax.bar(x + bar_width / 2, delta_drift, width=bar_width, label="Δ vs neutral", color=palette[1])
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_adapter(a) for a in secondary_adapters], rotation=15, ha="right")
    ax.set_ylabel("Δ marker fire rate (k=3 - baseline)")
    ax.set_title("secondary adapters: drift-induced lift over baselines (descriptive only)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    paths = savefig_paper(fig, stem="secondary_delta_swenglib", dir=figures_root)
    plt.close(fig)
    return paths


def _pretty_adapter(adapter_id: str) -> str:
    """Make adapter ids more readable on axis ticks. Returns e.g. 'villain · C1'."""
    if "_" not in adapter_id:
        return adapter_id
    persona, _, cond = adapter_id.rpartition("_")
    return f"{persona} · {cond}"


# ────────────────────────────────────────────────────────────────────────────
# Top-level entry
# ────────────────────────────────────────────────────────────────────────────


def write_aggregated(
    eval_root: Path,
    cell_labels: Iterable[str],
    out_path: Path,
) -> dict:
    """Write ``aggregated.json`` and return its in-memory dict."""
    data = aggregate_cells(eval_root, cell_labels)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    log.info("wrote %d cell summaries to %s", len(data), out_path)
    return data


def write_stratified(
    eval_root: Path,
    cell_labels: Iterable[str],
    out_path: Path,
) -> dict:
    """Write ``stratified_by_query_source.json`` and return its in-memory dict."""
    data: dict[str, dict] = {}
    for label in cell_labels:
        cell_dir = cell_dir_for(eval_root, label)
        if not (cell_dir / "marker_eval.json").exists():
            continue
        m = load_cell_marker_eval(cell_dir)
        data[label] = m.get("stratified", {})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    log.info("wrote stratified rates for %d cells to %s", len(data), out_path)
    return data


def write_bootstrap(
    bootstrap_data: dict,
    out_path: Path,
) -> None:
    """Write ``bootstrap.json``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bootstrap_data, indent=2))
    log.info("wrote bootstrap CIs to %s", out_path)

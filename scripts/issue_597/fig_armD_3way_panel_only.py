"""Re-render the 3-way decomposition figure using ONE measurement surface
(held-out panel medians) for ALL THREE arms, aggregated ACROSS training seeds.

v6 (#597 follow-up ``filler-control-multiseed``): the v1 figure was a single
seed-42 point estimate. This version aggregates the panel trajectories across
the seed list ``SEEDS = [42, 137, 7]`` per (arm, source) cell, plotting the
per-step seed-MEAN curve with a cross-seed error band on BOTH the source row
and the bystander-median row, across all three arms.

Error-bar definition (named here AND baked into the caption / meta.json): the
shaded band is the cross-seed HALF-RANGE, ``(max - min) / 2`` over the seeds
present at that step. Half-range (not SE) because with only 3 seeds the
sample SE is a poor variance estimate; half-range is the honest bracket of
observed seed spread and degrades cleanly to a zero-width band for a single
seed.

Single-seed fallback (smoke-friendly): ``SEEDS`` is filtered per (arm, source)
to the seeds whose trajectory file actually exists on disk. When only the
seed-42 trajectories are present (e.g. before any new-seed training has run),
every cell loads exactly one seed, the half-range band collapses to zero
width, and the figure reproduces the v1 single-seed curves.

Production coverage guard (fail-fast on partial multi-seed landing): the
seed-42-only fallback above is legitimate ONLY before any new-seed training has
landed. The MOMENT any non-42 trajectory exists for any (arm, source) cell, the
production path requires FULL coverage — ``available_seeds(arm, source) ==
SEEDS`` for EVERY plotted (arm, source) cell — before plotting. A partial
landing (some cells at 3 seeds, one cell missing seed 7) would otherwise render
a silently degraded smaller-N half-range that the v3 clean-result body inlines
as if it were the full 3-seed statistic — the error band IS the science of this
round, so a degraded cell corrupts the headline. On a partial landing the script
raises ``SystemExit`` listing every missing ``(arm, source, seed)`` triple and
pointing at the launcher (``scripts/issue_597/launch_multiseed_597.sh``) as the
recovery action. The ``--allow-partial`` CLI flag (or ``EPM_597_FIG_ALLOW_PARTIAL=1``
env var) bypasses the guard; it is for the pre-launch fallback smoke and ad-hoc
inspection ONLY — the production re-render (the analyzer's Step 9) never passes it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[2]
PATHS = {
    "armB": REPO / "eval_results/issue_597/panel_trajectories/armB",
    "armC": REPO / "eval_results/issue_597/dense-early-contrastive-grid/panel_trajectories/armC",
    "armD": REPO / "eval_results/issue_597/positives-plus-filler-control/panel_trajectories/armD",
}
SOURCES = ["villain", "assistant", "qwen_default"]
SOURCE_TITLES = {
    "villain": "Villain",
    "assistant": "Helpful assistant",
    "qwen_default": "Bare Qwen default",
}
NO_PERSONA_EXCLUDE_FOR = "qwen_default"

# v6: aggregate across these seeds. Filtered per (arm, source) to the seeds
# whose trajectory file actually exists, so the script runs in single-seed
# fallback mode (seed 42 only) before any new-seed training has landed.
SEEDS = [42, 137, 7]
ERROR_BAR_KIND = "half-range"  # (max - min) / 2 across seeds; named in the caption.

MAX_STEP_PLOT = 60  # restrict to the matched comparison window (armC, armD halt at 60)


def panel_path(arm: str, source: str, seed: int) -> Path:
    return PATHS[arm] / f"{source}_seed{seed}_panel_trajectory.json"


def available_seeds(arm: str, source: str) -> list[int]:
    """The seeds in ``SEEDS`` whose trajectory file exists for this cell.

    Returns at least one seed for any cell present on disk; an empty list
    means the cell has no committed trajectory at all (caller skips it)."""
    return [s for s in SEEDS if panel_path(arm, source, seed=s).exists()]


def _any_non_default_seed_present() -> bool:
    """True iff ANY non-42 trajectory exists for ANY plotted (arm, source) cell.

    The seed-42-only fallback is legitimate ONLY while this is False (no new-seed
    training has landed). The first non-42 file flips the production path on."""
    non_default = [s for s in SEEDS if s != 42]
    return any(
        panel_path(arm, src, seed=s).exists()
        for arm in PATHS
        for src in SOURCES
        for s in non_default
    )


def missing_coverage_triples() -> list[tuple[str, str, int]]:
    """Every ``(arm, source, seed)`` cell in the plotted grid that is missing its
    trajectory file. Empty list == full ``SEEDS`` coverage on every cell."""
    missing: list[tuple[str, str, int]] = []
    for arm in PATHS:
        for src in SOURCES:
            present = set(available_seeds(arm, src))
            for s in SEEDS:
                if s not in present:
                    missing.append((arm, src, s))
    return missing


def assert_production_coverage(allow_partial: bool) -> None:
    """Fail-fast coverage guard, run at the TOP of ``main()`` before plotting.

    - Pre-training fallback (PRESERVE): if NO non-42 trajectory exists anywhere
      across the plotted (arm, source) grid, return silently — the figure renders
      the v1 seed-42-only point estimate exactly as before.
    - Production guard (FAIL-FAST): the moment any non-42 trajectory exists, EVERY
      plotted (arm, source) cell must have full ``SEEDS`` coverage. A partial
      landing raises ``SystemExit`` listing every missing triple — the error band
      is the science of this round, so a silently degraded smaller-N band would
      corrupt the inlined headline statistic.
    - ``allow_partial`` (``--allow-partial`` / ``EPM_597_FIG_ALLOW_PARTIAL=1``)
      bypasses the guard for the smoke fallback and ad-hoc inspection ONLY; the
      production re-render never passes it.
    """
    if allow_partial:
        return
    if not _any_non_default_seed_present():
        # Pre-training fallback: seed-42-only, render the v1 curves.
        return
    missing = missing_coverage_triples()
    if missing:
        lines = "\n".join(f"  - {arm} / {src} / seed{seed}" for arm, src, seed in missing)
        raise SystemExit(
            "issue #597 figure coverage guard: a non-default seed has landed, so the "
            f"production path requires FULL {SEEDS} coverage on every plotted "
            f"(arm, source) cell, but {len(missing)} trajectory file(s) are missing:\n"
            f"{lines}\n"
            "Re-run the seed sweep to completion before regenerating this figure:\n"
            "  bash scripts/issue_597/launch_multiseed_597.sh\n"
            "(For the pre-launch smoke or ad-hoc inspection ONLY, bypass with "
            "--allow-partial or EPM_597_FIG_ALLOW_PARTIAL=1 — never on the "
            "production re-render.)"
        )


def load_panel(arm: str, source: str, seed: int) -> dict:
    return json.loads(panel_path(arm, source, seed).read_text())


def _source_step_values(arm: str, source: str, seed: int) -> dict[int, float]:
    """step -> source-context ``delta_logp`` for one seed (within the plot window)."""
    d = load_panel(arm, source, seed)
    out: dict[int, float] = {}
    for s in sorted(int(k) for k in d["by_step"]):
        if s > MAX_STEP_PLOT:
            continue
        if source in d["by_step"][str(s)]:
            out[s] = d["by_step"][str(s)][source]["delta_logp"]
    return out


def _bystander_step_values(arm: str, source: str, seed: int) -> dict[int, float]:
    """step -> bystander-MEDIAN ``delta_logp`` for one seed (within the plot window)."""
    d = load_panel(arm, source, seed)
    out: dict[int, float] = {}
    for s in sorted(int(k) for k in d["by_step"]):
        if s > MAX_STEP_PLOT:
            continue
        bys = []
        for ctx, agg in d["by_step"][str(s)].items():
            if ctx == source:
                continue
            if source == NO_PERSONA_EXCLUDE_FOR and ctx == "no_persona":
                continue
            bys.append(agg["delta_logp"])
        if bys:
            out[s] = statistics.median(bys)
    return out


def _aggregate_across_seeds(
    per_seed: list[dict[int, float]],
) -> tuple[list[int], list[float], list[float]]:
    """Given a list of {step: value} maps (one per seed), return
    (steps, seed_means, half_ranges) for the steps present in EVERY seed map.

    Intersecting steps keeps the seed-mean honest (no seed silently dropped at
    a step). For a single seed the half-range is 0 (reproduces the v1 curve)."""
    if not per_seed:
        return [], [], []
    common = set(per_seed[0])
    for m in per_seed[1:]:
        common &= set(m)
    steps = sorted(common)
    means: list[float] = []
    half_ranges: list[float] = []
    for s in steps:
        vals = [m[s] for m in per_seed]
        means.append(statistics.fmean(vals))
        half_ranges.append((max(vals) - min(vals)) / 2.0)
    return steps, means, half_ranges


def source_trajectory(arm: str, source: str) -> tuple[list[int], list[float], list[float]]:
    seeds = available_seeds(arm, source)
    per_seed = [_source_step_values(arm, source, s) for s in seeds]
    return _aggregate_across_seeds(per_seed)


def bystander_trajectory(arm: str, source: str) -> tuple[list[int], list[float], list[float]]:
    seeds = available_seeds(arm, source)
    per_seed = [_bystander_step_values(arm, source, s) for s in seeds]
    return _aggregate_across_seeds(per_seed)


def _resolve_allow_partial(argv: list[str] | None = None) -> bool:
    """``--allow-partial`` CLI flag OR ``EPM_597_FIG_ALLOW_PARTIAL=1`` env var.

    Either escape hatch bypasses the production coverage guard. Argv defaults to
    ``sys.argv[1:]`` so direct ``python fig_*.py`` invocation parses the flag; the
    env var lets the test (which calls ``main()`` directly) toggle it cleanly."""
    parser = argparse.ArgumentParser(description="Render the #597 3-way panel figure.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Bypass the production multi-seed coverage guard (smoke / ad-hoc "
            "inspection ONLY; never on the production re-render)."
        ),
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    env_partial = os.environ.get("EPM_597_FIG_ALLOW_PARTIAL", "") == "1"
    return args.allow_partial or env_partial


def main(argv: list[str] | None = None) -> None:
    # Fail-fast coverage guard BEFORE any plotting (the error band is the science
    # of this round; a partial multi-seed landing must not silently degrade it).
    assert_production_coverage(allow_partial=_resolve_allow_partial(argv))

    set_paper_style("blog")
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), sharex=True)

    # Semantic colors
    color_B = paper_palette_role("baseline")  # orange-ish
    color_C = paper_palette_role("primary")  # blue-ish
    color_D = paper_palette_role("control")  # red-ish

    arm_styles = [
        ("armB", color_B, "o", "Positives-only", "-"),
        ("armC", color_C, "s", "Contrastive", "-"),
        ("armD", color_D, "^", "Positives-plus-filler", "--"),
    ]

    def _plot_cell(ax, steps, means, errs, color, marker, linestyle, label):
        ax.plot(
            steps,
            means,
            marker=marker,
            markersize=4,
            markeredgewidth=1.0,
            linewidth=1.4,
            color=color,
            linestyle=linestyle,
            label=label,
        )
        if steps and any(e > 0 for e in errs):
            lo = [m - e for m, e in zip(means, errs, strict=True)]
            hi = [m + e for m, e in zip(means, errs, strict=True)]
            ax.fill_between(steps, lo, hi, color=color, alpha=0.18, linewidth=0)

    # Top row: source-context seed-mean delta_logp ± cross-seed half-range.
    for i, src in enumerate(SOURCES):
        ax = axes[0, i]
        for arm, color, marker, label, linestyle in arm_styles:
            steps, means, errs = source_trajectory(arm, src)
            _plot_cell(ax, steps, means, errs, color, marker, linestyle, label if i == 0 else None)
        ax.axvspan(8, 24, alpha=0.08, color="grey", zorder=0)
        ax.set_title(SOURCE_TITLES[src])
        if i == 0:
            ax.set_ylabel("Source-context\nmarker log-prob gain (nat)")

    # Bottom row: bystander median seed-mean ± cross-seed half-range.
    for i, src in enumerate(SOURCES):
        ax = axes[1, i]
        for arm, color, marker, _label, linestyle in arm_styles:
            steps, means, errs = bystander_trajectory(arm, src)
            _plot_cell(ax, steps, means, errs, color, marker, linestyle, None)
        ax.set_xlabel("Optimizer step")
        if i == 0:
            ax.set_ylabel("Bystander median\nmarker log-prob gain (nat)")

    # Single legend, top-row — plain-English names only (Lens 2 / 3 / 4: no
    # short-letter codes in figures).
    handles = [
        Line2D([0], [0], marker="o", markersize=5, color=color_B, label="Positives-only"),
        Line2D([0], [0], marker="s", markersize=5, color=color_C, label="Contrastive"),
        Line2D(
            [0],
            [0],
            marker="^",
            markersize=5,
            color=color_D,
            linestyle="--",
            label="Positives-plus-filler",
        ),
    ]
    axes[0, 0].legend(handles=handles, loc="upper left", frameon=False, fontsize=8)

    fig.suptitle("")  # title carried by caption; per-style-rules no title block here
    fig.tight_layout()

    written = savefig_paper(fig, "issue_597/armD_3way_panel_only", dir="figures/")
    plt.close(fig)

    # Augment the savefig_paper sidecar with the seed-aggregation provenance so
    # the v3 clean-result body's SHA-pinned figure link + caption can cite the
    # exact seeds and error-bar definition (the planner's meta.json contract).
    _augment_meta(written["meta"])


def _augment_meta(meta_path: Path) -> None:
    """Record the SEEDS list, error-bar definition, and the realized per-cell
    seed coverage into the figure's ``.meta.json`` sidecar."""
    meta = json.loads(meta_path.read_text())
    coverage: dict[str, dict[str, list[int]]] = {}
    for arm in PATHS:
        coverage[arm] = {}
        for src in SOURCES:
            coverage[arm][src] = available_seeds(arm, src)
    meta["seeds_requested"] = SEEDS
    meta["error_bar"] = ERROR_BAR_KIND
    meta["seed_coverage"] = coverage
    meta["sha_pinned_url_pattern"] = (
        "https://github.com/superkaiba/explore-persona-space/blob/"
        "<sha>/figures/issue_597/armD_3way_panel_only.png"
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()

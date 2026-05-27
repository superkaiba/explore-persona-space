"""Per-position log p(marker) analysis + figures for issue #398.

Reads the JSON produced by ``scripts/eval_i398_per_position.py`` and emits
four figures plus a summary JSON. Where ``scripts/analyze_i398_dynamics.py``
runs across-training-step labeling against two FIXED probe geometries, this
script slices the SAME training trajectory along the WITHIN-answer position
axis: how does log p(``※``) move as the on-policy greedy answer unfolds, and
does that movement track where the trained data actually pinned the marker
(end of the answer + ``\\n\\n``)?

Figures (all saved via ``paper-plots`` skill conventions):

    1. ``per_position_small_multiples.png`` — 6 personas by ~5 step columns
       grid. Each subplot plots position-in-answer (x) vs log p(``※``) at
       that position (y), one line per (persona, step). One step per column
       across a representative subset of the 22 checkpoints (default subset:
       ``[5, 25, 75, 200, 1600]``). Personas: librarian + 5 hand-picked
       bystanders covering the close / far / fammate spread (overridable).

    2. ``per_position_heatmap.png`` — 22 checkpoints by 5 position buckets.
       Buckets: ``start`` (positions [0, 10)), ``early`` ([10, 30 %)),
       ``mid`` ([30 %, 70 %)), ``late`` ([70 %, 90 %)), ``end`` ([90 %, len)).
       Each cell is the mean log p(``※``) across all (persona, prompt) cells
       at that step inside that bucket. Color: viridis (sequential — log p is
       a single-direction quantity here).

    3. ``per_position_peak_histogram.png`` — for each checkpoint, the
       distribution across all (persona, prompt) cells of
       ``argmax(logp_per_position)``. Reveals whether the log-p peak starts
       at the final position (end-pinned, matches training) and "bleeds
       backward" as the model overgeneralizes, vs starts diffuse and
       sharpens at the end.

    4. ``per_position_sampling_vs_peak.png`` — 2-D histogram across the cells
       where greedy actually emitted ``※``. x = position of log-p peak,
       y = position where greedy emitted ``※``. A tight y = x diagonal means
       high log-p positions actually lead to emission (sampling tracks the
       distribution).

Per CLAUDE.md "Checkpoint per phase" rule the analysis emits each figure
the moment it is built (via ``savefig_paper``) and the summary JSON at the
end — this is a CPU-only postprocess (~5 min wall) and a mid-script crash
is cheap to re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make ``src/`` importable for paper-plots styling helpers regardless of cwd.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    add_direction_arrow,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

# Bucket boundaries for the heatmap (figure 2). The "start" bucket is a
# fixed-width [0, 10) token region — captures the first sentence's worth of
# positions even on very long answers — and the remaining four buckets are
# fractional [10, 30 %), [30, 70 %), [70, 90 %), [90, 100 %) so the buckets
# stay comparable across answers of widely different length. Cells whose
# answer is shorter than 10 tokens get a single "start" reading (all other
# buckets empty for that cell).
_BUCKETS: list[tuple[str, float, float]] = [
    ("start (0-10)", 0.0, 10.0),  # absolute token range
    ("early (10-30%)", 0.10, 0.30),
    ("mid (30-70%)", 0.30, 0.70),
    ("late (70-90%)", 0.70, 0.90),
    ("end (90-100%)", 0.90, 1.00),
]

# Default subset of checkpoints to show as columns in figure 1. Picked to
# cover the same window the dynamics analysis labels (5, 25 inside the
# pre-jump flat band; 75 is the jump itself; 200 and 1600 are post-jump
# saturation milestones). Override with --small-multiples-steps if a
# different cross-section tells the story better.
_DEFAULT_SMALL_MULTIPLE_STEPS: list[int] = [5, 25, 75, 200, 1600]

# Default 6 personas for the small-multiples grid. Source persona first, then
# 5 bystanders spanning close / far / fammate / format. Hand-picked to be
# legible — not a random sample. Override with --small-multiples-personas
# if the trained run's bystander panel diverges.
_DEFAULT_SMALL_MULTIPLE_PERSONAS: list[str] = [
    "librarian",
    "cybersec_consultant",
    "pentester",
    "doctor",
    "fammate_task_1",
    "fammate_format_2",
]


# ---------------------------------------------------------------------------
# Bucket aggregation
# ---------------------------------------------------------------------------


def _positions_in_bucket(answer_len: int, lo: float, hi: float) -> list[int]:
    """Return the list of position indices in [lo, hi) for an answer of length ``answer_len``.

    The first bucket is treated as an ABSOLUTE token range ([lo=0, hi=10))
    because the first few generated tokens are a meaningful anchor regardless
    of answer length. All subsequent buckets are FRACTIONAL — lo and hi are
    interpreted as fractions of ``answer_len`` and rounded to int positions.
    """
    if answer_len <= 0:
        return []
    if lo == 0.0 and hi >= 1.0:
        # Degenerate full-range bucket (not used by _BUCKETS but defensive).
        return list(range(answer_len))
    if hi <= 10.0 and lo < 1.0 and hi > 1.0:
        # First absolute-token bucket: clamp at the answer length.
        return list(range(min(answer_len, int(hi))))
    # Fractional bucket
    start = round(lo * answer_len)
    stop = round(hi * answer_len)
    if start >= answer_len:
        return []
    stop = min(stop, answer_len)
    return list(range(start, stop))


def _bucket_means_for_step(
    per_persona: dict[str, list[dict]],
) -> list[float]:
    """Mean log p(``※``) in each of the 5 buckets, across ALL (persona, prompt) cells.

    Returns a length-5 list of floats (one per bucket in ``_BUCKETS`` order).
    NaN-skipped: cells with empty buckets are dropped from the mean.
    """
    bucket_vals: list[list[float]] = [[] for _ in _BUCKETS]
    for _persona, cells in per_persona.items():
        for cell in cells:
            logps = cell.get("logp_per_position", [])
            n = len(logps)
            if n == 0:
                continue
            for b_idx, (_name, lo, hi) in enumerate(_BUCKETS):
                idxs = _positions_in_bucket(n, lo, hi)
                if not idxs:
                    continue
                bucket_vals[b_idx].append(float(np.mean([logps[i] for i in idxs])))
    return [float(np.mean(v)) if v else float("nan") for v in bucket_vals]


# ---------------------------------------------------------------------------
# Figure 1 — small multiples (persona by step grid)
# ---------------------------------------------------------------------------


def build_small_multiples(
    per_step: dict[str, dict[str, list[dict]]],
    personas: list[str],
    steps: list[int],
    output_dir: Path,
) -> None:
    """6 personas by ~5 steps grid. Each subplot: position vs log p(``※``), prompt overlay."""
    set_paper_style("blog")
    n_rows = len(personas)
    n_cols = len(steps)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.0 * n_cols + 1.2, 1.6 * n_rows + 0.8), sharey=True
    )
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    line_color = paper_palette_role("primary")
    for i, persona in enumerate(personas):
        for j, step in enumerate(steps):
            ax = axes[i, j]
            cells = per_step.get(str(step), {}).get(persona, [])
            if not cells:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            for cell in cells:
                logps = cell.get("logp_per_position", [])
                if not logps:
                    continue
                ax.plot(
                    range(len(logps)),
                    logps,
                    color=line_color,
                    alpha=0.25,
                    linewidth=0.8,
                )
            # Column headers + row labels
            if i == 0:
                ax.set_title(f"step {step}", fontsize=9)
            if j == 0:
                ax.set_ylabel(persona, fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel("Position in answer (tokens)", fontsize=8)

    set_title_subtitle(
        axes[0, n_cols // 2],
        "Per-position log p of marker, by persona and training step",
        subtitle=(
            "Each thin line = one of 20 prompts. Higher y = model places more "
            "mass on the marker at that position."
        ),
    )
    savefig_paper(fig, "per_position_small_multiples", dir=str(output_dir))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — heatmap (steps by position buckets)
# ---------------------------------------------------------------------------


def build_heatmap(
    per_step: dict[str, dict[str, list[dict]]],
    steps: list[int],
    output_dir: Path,
) -> np.ndarray:
    """22 checkpoints by 5 buckets. Each cell = panel-mean log p(``※``). Returns matrix."""
    set_paper_style("blog")
    matrix = np.full((len(steps), len(_BUCKETS)), np.nan, dtype=float)
    for s_idx, step in enumerate(steps):
        per_persona = per_step.get(str(step), {})
        matrix[s_idx, :] = _bucket_means_for_step(per_persona)

    fig, ax = plt.subplots(figsize=(6.5, 1.0 + 0.22 * len(steps)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([str(s) for s in steps])
    ax.set_xticks(range(len(_BUCKETS)))
    ax.set_xticklabels([name for (name, _lo, _hi) in _BUCKETS], rotation=20, ha="right")
    ax.set_xlabel("Position bucket within greedy answer")
    ax.set_ylabel("Training step")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Mean log p of marker")
    set_title_subtitle(
        ax,
        "Where in the answer does mass on the marker concentrate?",
        subtitle=(
            "Mean log p(marker) across (persona, prompt) cells. Brighter = "
            "model places more probability on the marker at that within-answer "
            "position."
        ),
    )
    savefig_paper(fig, "per_position_heatmap", dir=str(output_dir))
    plt.close(fig)
    return matrix


# ---------------------------------------------------------------------------
# Figure 3 — argmax-position histograms
# ---------------------------------------------------------------------------


def build_peak_histogram(
    per_step: dict[str, dict[str, list[dict]]],
    steps: list[int],
    output_dir: Path,
) -> dict[int, list[int]]:
    """One panel per step: histogram of (peak position / answer length) across cells.

    Normalized to [0, 1] so panels with different answer lengths stay
    comparable. Returns the raw fractional-peak distribution per step so the
    summary JSON can pin numbers (median, fraction with peak in last 10 %, etc).
    """
    set_paper_style("blog")
    # Adjust grid to a square-ish layout for the 22 steps
    n_steps = len(steps)
    n_cols = 6
    n_rows = (n_steps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.8 * n_cols, 1.4 * n_rows + 0.6), sharex=True, sharey=True
    )
    # plt.subplots returns a 1-D array when either n_rows or n_cols is 1.
    # Reshape to (n_rows, n_cols) so the bottom-row label loop below works.
    axes = np.atleast_2d(axes)
    if axes.shape != (n_rows, n_cols):
        axes = axes.reshape(n_rows, n_cols)
    axes_flat = axes.flatten()

    color = paper_palette_role("primary")
    peak_fracs_per_step: dict[int, list[float]] = {}
    peak_positions_per_step: dict[int, list[int]] = {}
    for idx, step in enumerate(steps):
        ax = axes_flat[idx]
        peak_positions: list[int] = []
        peak_fracs: list[float] = []
        for _persona, cells in per_step.get(str(step), {}).items():
            for cell in cells:
                logps = cell.get("logp_per_position", [])
                if not logps:
                    continue
                peak_pos = int(np.argmax(logps))
                peak_positions.append(peak_pos)
                peak_fracs.append(peak_pos / max(len(logps) - 1, 1))
        peak_fracs_per_step[step] = peak_fracs
        peak_positions_per_step[step] = peak_positions
        if peak_fracs:
            ax.hist(peak_fracs, bins=20, range=(0, 1), color=color, alpha=0.85)
        ax.set_title(f"step {step}", fontsize=9)
        ax.set_yticks([])

    for spare in range(n_steps, len(axes_flat)):
        axes_flat[spare].set_visible(False)

    # Bottom-row x-labels
    for ax in axes[-1, :]:
        ax.set_xlabel("Peak position (fraction of answer)", fontsize=8)
    set_title_subtitle(
        axes[0, n_cols // 2],
        "Where the log-p peak sits across the answer",
        subtitle=(
            "Each histogram = one training step. x = position of "
            "argmax(log p) divided by answer length; 1.0 means the model "
            "puts maximum marker mass at the final generated token."
        ),
    )
    savefig_paper(fig, "per_position_peak_histogram", dir=str(output_dir))
    plt.close(fig)
    return peak_positions_per_step


# ---------------------------------------------------------------------------
# Figure 4 — sampling vs peak (2-D histogram, only on cells where greedy emitted ※)
# ---------------------------------------------------------------------------


def build_sampling_vs_peak(
    per_step: dict[str, dict[str, list[dict]]],
    steps: list[int],
    output_dir: Path,
) -> dict:
    """2-D hist: peak position vs sampled-marker position across cells where greedy emitted ※.

    Returns a dict with per-step counts (n_cells, n_emitted) so the summary
    JSON can record the emission rate trajectory.
    """
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    peak_fracs: list[float] = []
    sampled_fracs: list[float] = []
    per_step_emit: dict[int, dict[str, int]] = {}
    for step in steps:
        n_cells = 0
        n_emitted = 0
        for _persona, cells in per_step.get(str(step), {}).items():
            for cell in cells:
                logps = cell.get("logp_per_position", [])
                if not logps:
                    continue
                n_cells += 1
                samp = cell.get("sampled_marker_at_position")
                if samp is None:
                    continue
                n_emitted += 1
                peak_pos = int(np.argmax(logps))
                denom = max(len(logps) - 1, 1)
                peak_fracs.append(peak_pos / denom)
                sampled_fracs.append(samp / denom)
        per_step_emit[step] = {"n_cells": n_cells, "n_emitted": n_emitted}

    if peak_fracs:
        _h, _xedges, _yedges, im = ax.hist2d(
            peak_fracs,
            sampled_fracs,
            bins=20,
            range=[[0, 1], [0, 1]],
            cmap="viridis",
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("Cell count")
        # Reference diagonal — perfect tracking would land here.
        ax.plot([0, 1], [0, 1], color=paper_palette_role("neutral"), linestyle="--", linewidth=1)
    else:
        ax.text(
            0.5,
            0.5,
            "no cells where greedy emitted marker",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Peak position (fraction of answer)")
    ax.set_ylabel("Greedy-emission position (fraction of answer)")
    add_direction_arrow(ax, axis="x", direction="up", label="Peak position (fraction of answer)")
    set_title_subtitle(
        ax,
        "Does high log-p actually lead to marker emission?",
        subtitle=(
            "Pooled across all checkpoints. Diagonal = perfect tracking. "
            "Above diagonal = model emits the marker LATER than its peak; "
            "below diagonal = earlier."
        ),
    )
    savefig_paper(fig, "per_position_sampling_vs_peak", dir=str(output_dir))
    plt.close(fig)
    return per_step_emit


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------


def build_summary(
    per_step: dict[str, dict[str, list[dict]]],
    steps: list[int],
    heatmap_matrix: np.ndarray,
    peak_positions_per_step: dict[int, list[int]],
    per_step_emit: dict[int, dict[str, int]],
) -> dict:
    """Aggregate stats for the JSON sidecar.

    Includes per-step (a) bucket means, (b) peak-position percentiles + fraction
    in last 10 %, (c) greedy emission rate. Keeps the analysis machine-readable
    in case downstream code wants to slice without re-parsing the full eval JSON.
    """
    bucket_names = [name for (name, _lo, _hi) in _BUCKETS]
    summary_per_step: dict[str, dict] = {}
    for s_idx, step in enumerate(steps):
        row = heatmap_matrix[s_idx, :]
        peak_pos = peak_positions_per_step.get(step, [])
        # Convert to fractions for percentile aggregation (length-normalized).
        # Use the cell-by-cell answer length for the denominator so a 12-token
        # answer's peak at index 11 reads as ~1.0, matching what figure 3 shows.
        peak_fracs: list[float] = []
        for _persona, cells in per_step.get(str(step), {}).items():
            for cell in cells:
                logps = cell.get("logp_per_position", [])
                if not logps:
                    continue
                denom = max(len(logps) - 1, 1)
                peak_fracs.append(int(np.argmax(logps)) / denom)
        emit = per_step_emit.get(step, {"n_cells": 0, "n_emitted": 0})
        emission_rate = emit["n_emitted"] / emit["n_cells"] if emit["n_cells"] else float("nan")
        summary_per_step[str(step)] = {
            "bucket_means": {bucket_names[i]: float(row[i]) for i in range(len(bucket_names))},
            "peak_position_count": len(peak_pos),
            "peak_fraction_p10": float(np.percentile(peak_fracs, 10))
            if peak_fracs
            else float("nan"),
            "peak_fraction_p50": float(np.percentile(peak_fracs, 50))
            if peak_fracs
            else float("nan"),
            "peak_fraction_p90": float(np.percentile(peak_fracs, 90))
            if peak_fracs
            else float("nan"),
            "fraction_peak_in_last_10pct": (
                float(np.mean([1.0 if f >= 0.9 else 0.0 for f in peak_fracs]))
                if peak_fracs
                else float("nan")
            ),
            "greedy_emission_rate": emission_rate,
            "n_cells": emit["n_cells"],
            "n_emitted": emit["n_emitted"],
        }
    return {
        "buckets": bucket_names,
        "steps": steps,
        "per_step": summary_per_step,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--per-position-file",
        required=True,
        help="Output of scripts/eval_i398_per_position.py (e.g. "
        "eval_results/issue_398/per_position_seed42.json).",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the four PNG/PDF figures + per_position_summary.json.",
    )
    ap.add_argument(
        "--small-multiples-steps",
        default=",".join(str(s) for s in _DEFAULT_SMALL_MULTIPLE_STEPS),
        help=(
            "Comma-sep subset of checkpoints to show as columns in figure 1. "
            f"Default: {_DEFAULT_SMALL_MULTIPLE_STEPS}"
        ),
    )
    ap.add_argument(
        "--small-multiples-personas",
        default=",".join(_DEFAULT_SMALL_MULTIPLE_PERSONAS),
        help=(
            "Comma-sep subset of personas to show as rows in figure 1. "
            f"Default: {_DEFAULT_SMALL_MULTIPLE_PERSONAS}"
        ),
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.per_position_file) as f:
        data = json.load(f)
    per_step: dict[str, dict[str, list[dict]]] = data["per_step"]
    all_steps: list[int] = sorted(int(s) for s in per_step)
    panel: list[str] = data["panel"]

    sm_steps = [int(s) for s in args.small_multiples_steps.split(",") if s.strip()]
    # Drop steps that aren't actually in the eval JSON, loudly.
    missing = [s for s in sm_steps if s not in all_steps]
    assert not missing, f"--small-multiples-steps refers to absent checkpoints: {missing}"
    sm_personas = [p for p in args.small_multiples_personas.split(",") if p.strip()]
    missing_p = [p for p in sm_personas if p not in panel]
    assert not missing_p, (
        f"--small-multiples-personas refers to absent personas: {missing_p}. Panel: {panel}"
    )

    # Figures 1-4 (each saved to disk inside its builder)
    build_small_multiples(per_step, sm_personas, sm_steps, out_dir)
    heatmap_matrix = build_heatmap(per_step, all_steps, out_dir)
    peak_positions_per_step = build_peak_histogram(per_step, all_steps, out_dir)
    per_step_emit = build_sampling_vs_peak(per_step, all_steps, out_dir)

    # Summary JSON
    summary = build_summary(
        per_step, all_steps, heatmap_matrix, peak_positions_per_step, per_step_emit
    )
    summary_path = out_dir / "per_position_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()

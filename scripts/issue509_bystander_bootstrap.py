#!/usr/bin/env python3
"""#509 follow-up: bystander-cluster bootstrap CIs + paired metric differences.

The published #509 sycophancy-arm CIs (``eval_results/issue_509/syco_arm/
scoring.json``) and the baserate follow-up CIs cluster-resample the 6
SOURCES. This follow-up adds the complementary resampling axis — the
bystander personas — plus PAIRED differences between the top predictor
cells computed on SHARED bootstrap draws, which is the direct test of
whether the top distance metrics are statistically separable from each
other.

Per cell (the 3 headline cells + the top-15 leaderboard cells by
|rho_fe_adj|), on the frozen 138-cell panel with the trained-base delta
target:

  a. bystander-cluster bootstrap percentile 95% CI on the source-FE
     Spearman rho (resample the distinct bystander personas with
     replacement; every draw keeps all sources' cells for the drawn
     bystanders);
  b. source-cluster bootstrap CI with the same B/seed and the SAME
     (unadjusted, rank-then-demean) estimator so the two axes are
     side-by-side comparable — the published source-clustered CIs were
     on the attenuation-ADJUSTED statistic, so they are recomputed here
     rather than copied;
  c. paired differences (delta |rho| and signed delta rho) between the
     top cells under the bystander axis, using the SAME draws for both
     cells of each pair;
  d. a two-way sensitivity (sources AND bystanders both resampled per
     rep) on the two global-max cells.

Point-estimate helpers (``rankdata``, ``within``, ``fe_spearman``) and
the loading pattern are copied VERBATIM from
``scripts/issue509_baserate_covariate_earlylayer.py`` so numbers are
directly comparable to ``eval_results/issue_509/
baserate-partial-early-layer-cosine/results.json`` (its
``rho_geom_alone`` equals this script's ``rho_fe`` on shared cells; the
script asserts that when the file is present).

Analysis-only: inputs are the frozen #411 leakage snapshot (repo file)
and the #509 bake-off distance matrices on the HF data repo at a pinned
revision. No training, no generation, no model forward passes.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata as _scipy_rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_TARGET_FILE = PROJECT_ROOT / "eval_results/issue_480/_inputs/syco_411_analyze_summary.json"
DEFAULT_SCORING_FILE = PROJECT_ROOT / "eval_results/issue_509/syco_arm/scoring.json"
DEFAULT_BASERATE_RESULTS = (
    PROJECT_ROOT / "eval_results/issue_509/baserate-partial-early-layer-cosine/results.json"
)
DEFAULT_HF_REPO = "superkaiba1/explore-persona-space-data"
DEFAULT_HF_REVISION = "1b6e20530b1c6d477a387c18d5a88554910e7df9"
DEFAULT_METRICS_PREFIX = "issue_509/syco_arm/bakeoff/metrics"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval_results/issue_509/bystander-bootstrap-cis/results.json"
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures/issue_509"
SMOKE_OUTPUT = Path("/tmp/issue509_bystander_bootstrap_smoke/results.json")
SMOKE_FIG_DIR = Path("/tmp/issue509_bystander_bootstrap_smoke/figures")

CellSpec = tuple[str, int, str, str]

# (extraction_point, layer, metric, variant)
HEADLINE_CELLS: tuple[CellSpec, ...] = (
    ("last_prompt", 7, "mmd", "centered"),  # global max
    ("end_of_system", 2, "cosine", "centered"),  # representative early cell
    ("last_prompt", 22, "gauss_kl", "centered"),  # marker-recipe anchor
)
PAIRED_CELLS: tuple[tuple[CellSpec, CellSpec], ...] = (
    (HEADLINE_CELLS[0], HEADLINE_CELLS[1]),  # mmd-L7 vs cosine-L2
    (HEADLINE_CELLS[0], HEADLINE_CELLS[2]),  # mmd-L7 vs gauss_kl-L22 anchor
    (HEADLINE_CELLS[1], HEADLINE_CELLS[2]),  # cosine-L2 vs gauss_kl-L22 anchor
)
TWO_WAY_CELLS: tuple[CellSpec, ...] = (HEADLINE_CELLS[0], HEADLINE_CELLS[1])
TOP_K = 15
MIN_FINITE_REPS = 100  # issue509_scoring convention: NaN CI below this
BOOTSTRAP_SEED = 42

# Per-pair saturation exclusion from issue509_scoring.py ROUND-3 G2.
PAIR_SATURATION_ABS_THRESHOLD = 1e-6
MIN_SURVIVING_PAIRS = 5

# ── SC cid mapping ────────────────────────────────────────────────────────
# Copied verbatim from the issue-509 branch module
# ``src/explore_persona_space/experiments/i509_syco_conditions.py``
# (commit 11a43c490, never merged to main): cids are SC1..SC24 assigned
# in ALPHABETICAL persona-name order. Hardcoded here (with runtime
# coverage asserts in ``main``) so this script runs from the main
# checkout without a worktree.
_SYCO_PERSONA_ORDER: tuple[str, ...] = (
    "accountant",
    "ai",
    "ai_assistant",
    "assistant",
    "chef",
    "child",
    "comedian",
    "data_scientist",
    "french_person",
    "hero",
    "journalist",
    "kindergarten_teacher",
    "lawyer",
    "librarian",
    "medical_doctor",
    "philosopher",
    "police_officer",
    "programmer",
    "qwen_default",
    "software_engineer",
    "surgeon",
    "villain",
    "wizard",
    "zelthari_scholar",
)
assert list(_SYCO_PERSONA_ORDER) == sorted(_SYCO_PERSONA_ORDER), "cid order must be alphabetical"
CID_TO_SYCO_PERSONA: dict[str, str] = {
    f"SC{i}": persona for i, persona in enumerate(_SYCO_PERSONA_ORDER, start=1)
}


def _git_sha(repo: Path) -> str:
    """Full git SHA of ``repo`` HEAD, or 'unknown'."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


# ── Stats helpers — copied verbatim from issue509_baserate_covariate_earlylayer.py ──
# (rank globally, demean ranks within source, Pearson) so the outputs are
# directly comparable to eval_results/issue_509/baserate-partial-early-layer-cosine/
# results.json (rho_fe here == rho_geom_alone there).


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks (0-based mean ranks), matching the parent script."""
    return _scipy_rankdata(np.asarray(x, dtype=float)) - 1.0


def within(vec: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract per-group mean (source fixed effect)."""
    vec = np.asarray(vec, float)
    out = vec.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = vec[m] - vec[m].mean()
    return out


def fe_spearman(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Source-FE Spearman: corr of within-source-demeaned global ranks."""
    rx = within(rankdata(x), groups)
    ry = within(rankdata(y), groups)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ── Data loading — copied from issue509_baserate_covariate_earlylayer.py ──
# (load_target_rows / cell_filename / load_distance_matrix verbatim;
# build_xy extended ONLY to carry the bystander label per row, which the
# bystander-cluster bootstrap needs.)


def load_target_rows(target_file: Path) -> list[dict[str, Any]]:
    """Load the #411 frozen snapshot into 138 off-diagonal rows.

    Each row: {source, bystander, delta, trained_rate, base_rate}.
    """
    with open(target_file) as f:
        snap = json.load(f)
    rows: list[dict[str, Any]] = []
    for source, src_data in snap["per_source"].items():
        deltas = src_data["per_panel_delta"]
        trained = src_data["per_panel_trained_rate"]
        base = src_data["per_panel_base_rate"]
        for bystander, delta in deltas.items():
            if bystander == source:
                continue  # off-diagonal only
            rows.append(
                {
                    "source": source,
                    "bystander": bystander,
                    "delta": float(delta),
                    "trained_rate": float(trained[bystander]),
                    "base_rate": float(base[bystander]),
                }
            )
    if len(rows) != 138:
        raise ValueError(f"Expected 138 off-diagonal cells, got {len(rows)}")
    return rows


def cell_filename(point: str, layer: int, metric: str, variant: str) -> str:
    """Metric-phase filename for one bake-off cell."""
    return f"{point}__layer{layer}__{metric}__{variant}.json"


def load_distance_matrix(
    *,
    repo_id: str,
    revision: str,
    prefix: str,
    filename: str,
) -> dict[str, dict[str, float | None]]:
    """Download one bake-off distance JSON at the pinned revision; return its matrix.

    Raises if the payload carries ``matrix: null`` — the requested cells are
    all defined in the leaderboard, so a None matrix means a wrong cell name.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id,
        f"{prefix}/{filename}",
        repo_type="dataset",
        revision=revision,
    )
    with open(path) as f:
        payload = json.load(f)
    matrix = payload.get("matrix")
    if matrix is None:
        raise ValueError(f"{filename}: matrix is null (intentional N/A cell?) — cannot score")
    return matrix


def build_xy(
    matrix: dict[str, dict[str, float | None]],
    rows: list[dict[str, Any]],
    persona_to_cid: dict[str, str],
) -> dict[str, Any]:
    """Align geometry distances with the 138 target rows.

    Exclusions follow issue509_scoring.py: rows whose matrix entry is
    missing/None are dropped (``n_excluded_none``), then pairs whose
    distance sits at the floor of the metric's natural range
    (|x| < 1e-6, ROUND-3 G2) are dropped (``n_excluded_saturated``).
    Carries the bystander label per kept row (the one extension over the
    parent script's build_xy) for the bystander-cluster bootstrap.
    """
    geom, delta, src, bys = [], [], [], []
    n_none = 0
    for row in rows:
        src_cid = persona_to_cid.get(row["source"])
        bys_cid = persona_to_cid.get(row["bystander"])
        d = None
        if src_cid is not None and bys_cid is not None:
            d = matrix.get(src_cid, {}).get(bys_cid)
        if d is None:
            n_none += 1
            continue
        geom.append(float(d))
        delta.append(row["delta"])
        src.append(row["source"])
        bys.append(row["bystander"])
    geom_arr = np.array(geom, dtype=float)
    sat_mask = ~np.isfinite(geom_arr) | (np.abs(geom_arr) < PAIR_SATURATION_ABS_THRESHOLD)
    keep = ~sat_mask
    return {
        "geom": geom_arr[keep],
        "delta": np.array(delta, dtype=float)[keep],
        "source": np.array(src)[keep],
        "bystander": np.array(bys)[keep],
        "n_excluded_none": n_none,
        "n_excluded_saturated": int(sat_mask.sum()),
    }


# ── Leaderboard ───────────────────────────────────────────────────────────


def load_top_cells(scoring_file: Path, k: int) -> list[CellSpec]:
    """Top-``k`` cells by |rho_fe_adj| from the production scoring leaderboard.

    Used ONLY for cell selection; the statistic recomputed here is the
    unadjusted rank-then-demean ``fe_spearman``.
    """
    with open(scoring_file) as f:
        payload = json.load(f)
    scored = [
        c
        for c in payload["cells"]
        if isinstance(c.get("rho_fe_adj"), int | float) and np.isfinite(c["rho_fe_adj"])
    ]
    ranked = sorted(scored, key=lambda c: abs(c["rho_fe_adj"]), reverse=True)
    return [(c["extraction_point"], c["layer"], c["metric"], c["variant"]) for c in ranked[:k]]


# ── Bootstrap machinery ───────────────────────────────────────────────────


def _percentile_ci(vals: np.ndarray) -> dict[str, Any]:
    """Percentile 95% CI block over the finite bootstrap statistics."""
    finite = vals[np.isfinite(vals)]
    if len(finite) < MIN_FINITE_REPS:
        return {"ci": [float("nan"), float("nan")], "n_finite_reps": len(finite)}
    return {
        "ci": [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))],
        "n_finite_reps": len(finite),
    }


def bystander_bootstrap_rhos(
    xy: dict[str, Any],
    bystander_order: list[str],
    draws: np.ndarray,
) -> np.ndarray:
    """Per-rep FE-Spearman rho under bystander-cluster resampling.

    ``draws`` is a (B, n_bystanders) integer matrix into ``bystander_order``
    (SHARED across cells so paired differences use identical draws). Each
    rep keeps all sources' cells for the drawn bystanders; sources remain
    the FE groups, so no group relabeling is needed (duplicated bystander
    columns just duplicate rows within each source group).
    """
    rows_for_bys = [np.where(xy["bystander"] == b)[0] for b in bystander_order]
    geom, delta, source = xy["geom"], xy["delta"], xy["source"]
    out = np.empty(len(draws), dtype=float)
    for r, draw in enumerate(draws):
        idx = np.concatenate([rows_for_bys[j] for j in draw])
        out[r] = fe_spearman(geom[idx], delta[idx], source[idx])
    return out


def source_bootstrap_rhos(
    xy: dict[str, Any],
    source_order: list[str],
    draws: np.ndarray,
) -> np.ndarray:
    """Per-rep FE-Spearman rho under source-cluster resampling.

    Sources are the FE groups, so each drawn copy is relabeled to its draw
    position (two copies of the same source stay distinct clusters) —
    the issue509_scoring / baserate-script convention.
    """
    rows_for_src = [np.where(xy["source"] == s)[0] for s in source_order]
    geom, delta = xy["geom"], xy["delta"]
    out = np.empty(len(draws), dtype=float)
    for r, draw in enumerate(draws):
        parts = [rows_for_src[j] for j in draw]
        idx = np.concatenate(parts)
        gb = np.concatenate([np.full(len(p), i) for i, p in enumerate(parts)])
        out[r] = fe_spearman(geom[idx], delta[idx], gb)
    return out


def two_way_bootstrap_rhos(
    xy: dict[str, Any],
    source_order: list[str],
    bystander_order: list[str],
    src_draws: np.ndarray,
    bys_draws: np.ndarray,
) -> np.ndarray:
    """Per-rep rho with sources AND bystanders both resampled (sensitivity).

    Per rep, the panel is the cross product of the drawn sources and drawn
    bystanders (missing combinations — a persona never paired with itself —
    are skipped). Drawn sources are relabeled to draw position as in the
    source axis.
    """
    lookup: dict[tuple[str, str], int] = {
        (s, b): i for i, (s, b) in enumerate(zip(xy["source"], xy["bystander"], strict=True))
    }
    geom, delta = xy["geom"], xy["delta"]
    out = np.empty(len(src_draws), dtype=float)
    for r in range(len(src_draws)):
        idx_parts: list[int] = []
        gb_parts: list[int] = []
        for i, sj in enumerate(src_draws[r]):
            s = source_order[sj]
            for bj in bys_draws[r]:
                ri = lookup.get((s, bystander_order[bj]))
                if ri is not None:
                    idx_parts.append(ri)
                    gb_parts.append(i)
        idx = np.array(idx_parts, dtype=int)
        gb = np.array(gb_parts, dtype=int)
        out[r] = fe_spearman(geom[idx], delta[idx], gb) if len(idx) else float("nan")
    return out


def paired_difference_block(
    key_a: str,
    key_b: str,
    rhos_a: np.ndarray,
    rhos_b: np.ndarray,
    point_a: float,
    point_b: float,
) -> dict[str, Any]:
    """Paired delta-|rho| and signed delta-rho over SHARED bystander-axis draws."""
    both = np.isfinite(rhos_a) & np.isfinite(rhos_b)
    d_abs = np.abs(rhos_a[both]) - np.abs(rhos_b[both])
    d_signed = rhos_a[both] - rhos_b[both]
    block: dict[str, Any] = {
        "cell_a": key_a,
        "cell_b": key_b,
        "point_delta_abs_rho": float(abs(point_a) - abs(point_b)),
        "point_delta_signed_rho": float(point_a - point_b),
        "n_finite_reps": int(both.sum()),
    }
    if both.sum() < MIN_FINITE_REPS:
        block["delta_abs_rho_ci"] = [float("nan"), float("nan")]
        block["delta_signed_rho_ci"] = [float("nan"), float("nan")]
        block["frac_abs_a_gt_abs_b"] = float("nan")
        return block
    block["delta_abs_rho_ci"] = [
        float(np.percentile(d_abs, 2.5)),
        float(np.percentile(d_abs, 97.5)),
    ]
    block["delta_signed_rho_ci"] = [
        float(np.percentile(d_signed, 2.5)),
        float(np.percentile(d_signed, 97.5)),
    ]
    block["frac_abs_a_gt_abs_b"] = float((np.abs(rhos_a[both]) > np.abs(rhos_b[both])).mean())
    return block


# ── Figure ────────────────────────────────────────────────────────────────

_POINT_LABELS = {"end_of_system": "End of system prompt", "last_prompt": "Last prompt token"}
_METRIC_LABELS = {
    "mmd": "MMD",
    "cosine": "cosine",
    "gauss_kl": "Gaussian KL",
    "euclidean": "Euclidean",
    "wass2": "Wasserstein-2",
}


def _cell_label(spec: CellSpec) -> str:
    point, layer, metric, _variant = spec
    metric_label = _METRIC_LABELS.get(metric, metric)
    return f"{_POINT_LABELS.get(point, point)}, L{layer}, {metric_label}"


def make_figure(
    results: dict[str, Any],
    display_specs: list[CellSpec],
    fig_dir: Path,
) -> None:
    """Forest-style CI plot: bystander-resampled CIs (primary) with the
    source-resampled CIs offset alongside, vertical line at 0."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    colors = paper_palette(2)
    fig, ax = plt.subplots(figsize=(8.0, 0.62 * len(display_specs) + 1.8))

    cells = results["cells"]
    y_positions = np.arange(len(display_specs))[::-1].astype(float)
    for spec, y in zip(display_specs, y_positions, strict=True):
        key = cell_filename(*spec).removesuffix(".json")
        blk = cells[key]
        est = blk["rho_fe"]
        for axis_name, color, dy, fmt in (
            ("bystander", colors[0], +0.16, "o"),
            ("source", colors[1], -0.16, "s"),
        ):
            lo, hi = blk[axis_name]["ci"]
            # Clamp: percentile CIs can sit float-epsilon past the estimate.
            xerr = [[max(0.0, est - lo)], [max(0.0, hi - est)]]
            ax.errorbar(
                [est],
                [y + dy],
                xerr=xerr,
                fmt=fmt,
                ms=5,
                color=color,
                ecolor=color,
                elinewidth=1.8,
                capsize=3.5,
                label=None,
            )

    # Legend proxies (one entry per resampling axis).
    ax.errorbar([], [], xerr=[[], []], fmt="o", color=colors[0], label="Bystander-resampled 95% CI")
    ax.errorbar([], [], xerr=[[], []], fmt="s", color=colors[1], label="Source-resampled 95% CI")

    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_cell_label(s) for s in display_specs])
    ax.set_xlabel("Source-FE Spearman rho (sycophancy leakage delta)")
    set_title_subtitle(
        ax,
        "Bystander- vs source-resampled CIs on the top predictor cells",
        f"138 source-bystander cells; percentile bootstrap, B={results['bootstrap_b']}, "
        f"seed {results['bootstrap_seed']}",
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    savefig_paper(fig, "bystander_bootstrap_cis", dir=fig_dir)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Issue #509 follow-up: bystander-cluster bootstrap CIs and paired "
            "metric differences on the sycophancy-leakage predictor cells."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target-file", type=Path, default=DEFAULT_TARGET_FILE)
    p.add_argument("--scoring-file", type=Path, default=DEFAULT_SCORING_FILE)
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    p.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    p.add_argument("--metrics-prefix", default=DEFAULT_METRICS_PREFIX)
    p.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap reps per axis")
    p.add_argument(
        "--perm",
        type=int,
        default=0,
        help="CLI parity with sibling scripts; permutation inference is not part "
        "of this follow-up (a non-zero value is ignored with a warning)",
    )
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="B=200, the 3 headline cells only; outputs under /tmp unless overridden",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--fig-dir", type=Path, default=None)
    p.add_argument("--skip-figure", action="store_true", help="Skip the figure (stats only)")
    return p


def _validate_panel(
    rows: list[dict[str, Any]],
    persona_to_cid: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Coverage asserts on the hardcoded cid mapping; return cluster orders."""
    sources_in_panel = {r["source"] for r in rows}
    bystanders_in_panel = {r["bystander"] for r in rows}
    missing = (sources_in_panel | bystanders_in_panel) - set(persona_to_cid)
    if missing:
        raise ValueError(f"Personas in the target panel missing from the SC cid mapping: {missing}")
    source_order = sorted(sources_in_panel)
    bystander_order = sorted(bystanders_in_panel)
    if len(source_order) != 6:
        raise ValueError(f"Expected 6 sources, got {len(source_order)}")
    # NOTE: the union of bystander personas across the 6 sources is 24
    # (each source's own panel has 23 = 24 minus itself); the cluster
    # bootstrap resamples the 24 DISTINCT bystander personas.
    if len(bystander_order) not in (23, 24):
        raise ValueError(f"Expected 23-24 distinct bystander personas, got {len(bystander_order)}")
    return source_order, bystander_order


def _score_cells(
    args: argparse.Namespace,
    cell_specs: list[CellSpec],
    rows: list[dict[str, Any]],
    persona_to_cid: dict[str, str],
    source_order: list[str],
    bystander_order: list[str],
    draws: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Score every cell on the three resampling axes; return (cells, per-rep rho)."""
    cells: dict[str, Any] = {}
    bys_rho_vectors: dict[str, np.ndarray] = {}
    for spec in cell_specs:
        point, layer, metric, variant = spec
        fname = cell_filename(point, layer, metric, variant)
        matrix = load_distance_matrix(
            repo_id=args.hf_repo,
            revision=args.hf_revision,
            prefix=args.metrics_prefix,
            filename=fname,
        )
        xy = build_xy(matrix, rows, persona_to_cid)
        if len(xy["geom"]) < MIN_SURVIVING_PAIRS:
            raise ValueError(f"{fname}: only {len(xy['geom'])} pairs survive exclusions")
        key = fname.removesuffix(".json")
        point_rho = fe_spearman(xy["geom"], xy["delta"], xy["source"])
        rhos_bys = bystander_bootstrap_rhos(xy, bystander_order, draws["bystander"])
        rhos_src = source_bootstrap_rhos(xy, source_order, draws["source"])
        bys_rho_vectors[key] = rhos_bys
        block: dict[str, Any] = {
            "extraction_point": point,
            "layer": layer,
            "metric": metric,
            "variant": variant,
            "headline": spec in HEADLINE_CELLS,
            "n": len(xy["geom"]),
            "n_excluded_none": xy["n_excluded_none"],
            "n_excluded_saturated": xy["n_excluded_saturated"],
            "rho_fe": point_rho,
            "bystander": _percentile_ci(rhos_bys),
            "source": _percentile_ci(rhos_src),
        }
        if spec in TWO_WAY_CELLS:
            rhos_tw = two_way_bootstrap_rhos(
                xy, source_order, bystander_order, draws["two_way_src"], draws["two_way_bys"]
            )
            block["two_way"] = _percentile_ci(rhos_tw)
        cells[key] = block
        print(
            f"{key}: rho_fe={point_rho:+.3f} "
            f"bys_CI[{block['bystander']['ci'][0]:+.3f}, {block['bystander']['ci'][1]:+.3f}] "
            f"src_CI[{block['source']['ci'][0]:+.3f}, {block['source']['ci'][1]:+.3f}]"
            + (" [headline]" if block["headline"] else "")
        )
    return cells, bys_rho_vectors


def _crosscheck_baserate(cells: dict[str, Any]) -> None:
    """Convention cross-check against the baserate follow-up.

    Shared cells use the same helpers on the same inputs, so the point
    estimates must be identical. Skipped when that results file is absent;
    loud failure on drift.
    """
    if not DEFAULT_BASERATE_RESULTS.exists():
        return
    with open(DEFAULT_BASERATE_RESULTS) as f:
        baserate = json.load(f)
    for key, blk in cells.items():
        ref = baserate.get("cells", {}).get(key)
        if ref is None:
            continue
        ref_rho = ref["delta"]["all_138"]["rho_geom_alone"]
        if abs(ref_rho - blk["rho_fe"]) > 1e-9:
            raise ValueError(
                f"Estimator-convention drift on {key}: rho_fe={blk['rho_fe']!r} vs "
                f"baserate rho_geom_alone={ref_rho!r}"
            )
        blk["crosscheck_baserate_rho_geom_alone"] = ref_rho


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.perm:
        print(
            f"WARNING: --perm {args.perm} ignored — permutation inference is not part of "
            "this follow-up (flag kept for CLI parity with sibling issue509 scripts)."
        )
    b = 200 if args.smoke else args.bootstrap
    output: Path = args.output or (SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)
    fig_dir: Path = args.fig_dir or (SMOKE_FIG_DIR if args.smoke else DEFAULT_FIG_DIR)

    persona_to_cid = {v: k for k, v in CID_TO_SYCO_PERSONA.items()}
    rows = load_target_rows(args.target_file)
    source_order, bystander_order = _validate_panel(rows, persona_to_cid)
    n_src, n_bys = len(source_order), len(bystander_order)

    # Cell list: headline cells first, then the top-15 leaderboard cells.
    cell_specs: list[CellSpec] = list(HEADLINE_CELLS)
    top15: list[CellSpec] = []
    if not args.smoke:
        top15 = load_top_cells(args.scoring_file, TOP_K)
        for spec in top15:
            if spec not in cell_specs:
                cell_specs.append(spec)

    # SHARED draw matrices (one rng, fixed seed): the bystander draws are
    # reused for every cell AND for the paired differences, which is what
    # makes the pairs paired.
    rng = np.random.default_rng(args.seed)
    draws = {
        "bystander": rng.integers(0, n_bys, size=(b, n_bys)),
        "source": rng.integers(0, n_src, size=(b, n_src)),
        "two_way_src": rng.integers(0, n_src, size=(b, n_src)),
        "two_way_bys": rng.integers(0, n_bys, size=(b, n_bys)),
    }

    cells, bys_rho_vectors = _score_cells(
        args, cell_specs, rows, persona_to_cid, source_order, bystander_order, draws
    )
    _crosscheck_baserate(cells)

    paired = [
        paired_difference_block(
            cell_filename(*a).removesuffix(".json"),
            cell_filename(*c_b).removesuffix(".json"),
            bys_rho_vectors[cell_filename(*a).removesuffix(".json")],
            bys_rho_vectors[cell_filename(*c_b).removesuffix(".json")],
            cells[cell_filename(*a).removesuffix(".json")]["rho_fe"],
            cells[cell_filename(*c_b).removesuffix(".json")]["rho_fe"],
        )
        for a, c_b in PAIRED_CELLS
    ]

    results: dict[str, Any] = {
        "schema_version": 1,
        "followup_label": "bystander-bootstrap-cis",
        "smoke": bool(args.smoke),
        "description": (
            "Bystander-cluster bootstrap percentile 95% CIs on the source-FE Spearman "
            "rho (bake-off geometry vs #411 sycophancy leakage delta, 138 cells), with "
            "source-cluster CIs recomputed on the same estimator for side-by-side "
            "comparison, paired metric differences on shared bystander draws, and a "
            "two-way (sources x bystanders) sensitivity on the two global-max cells."
        ),
        "estimator_convention": (
            "rank-then-demean source-FE Spearman (fe_spearman) copied verbatim from "
            "issue509_baserate_covariate_earlylayer.py; matches that follow-up's "
            "rho_geom_alone exactly (asserted at runtime on shared cells). Differs "
            "from scoring.json's rho_fe (demean-then-rank) and rho_fe_adj "
            "(attenuation-adjusted), which are used ONLY to pick the top-15 cells; "
            "the source-clustered CIs here are therefore recomputed, not copied."
        ),
        "bystander_cluster_note": (
            "The spec's '23 bystanders' is per-source; the realized union of bystander "
            "personas across the 6 sources is 24 (each source's own panel has 23 = 24 "
            "minus itself). The cluster bootstrap resamples the 24 distinct bystander "
            "personas with replacement; every draw keeps all available sources' cells "
            "for each drawn bystander."
        ),
        "bootstrap_b": b,
        "bootstrap_seed": args.seed,
        "ci_method": "percentile [2.5, 97.5], NaN below 100 finite reps",
        "n_source_clusters": n_src,
        "n_bystander_clusters": n_bys,
        "headline_cells": [cell_filename(*c).removesuffix(".json") for c in HEADLINE_CELLS],
        "top15_cells_by_abs_rho_fe_adj": [cell_filename(*c).removesuffix(".json") for c in top15],
        "two_way_cells": [cell_filename(*c).removesuffix(".json") for c in TWO_WAY_CELLS],
        "cells": cells,
        "paired_differences_bystander_axis": paired,
        "inputs": {
            "target_file": str(args.target_file),
            "scoring_file": str(args.scoring_file),
            "hf_repo": args.hf_repo,
            "hf_revision": args.hf_revision,
            "metrics_prefix": args.metrics_prefix,
        },
        "git_sha": _git_sha(PROJECT_ROOT),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "env": {"python": platform.python_version(), "numpy": np.__version__},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {output}")

    for p in paired:
        print(
            f"PAIRED {p['cell_a']} vs {p['cell_b']}: "
            f"d|rho|={p['point_delta_abs_rho']:+.3f} "
            f"CI[{p['delta_abs_rho_ci'][0]:+.3f}, {p['delta_abs_rho_ci'][1]:+.3f}] "
            f"frac(|A|>|B|)={p['frac_abs_a_gt_abs_b']:.3f}"
        )

    if not args.skip_figure:
        # Display: the 3 headline cells + up to 5 further top-15 cells.
        # Dedupe on (point, layer, metric) — the centered/raw variants of
        # translation-invariant metrics carry identical matrices and would
        # render as duplicate rows.
        display = list(HEADLINE_CELLS)
        seen = {spec[:3] for spec in display}
        for spec in top15:
            if spec[:3] not in seen and len(display) < 8:
                display.append(spec)
                seen.add(spec[:3])
        make_figure(results, display, fig_dir)
        print(f"wrote figure under {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

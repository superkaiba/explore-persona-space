#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, ※, ×, —, ≈, ≥, ≤, ∈) in scientific docstrings + logs.
"""Issue #531 — Non-saturated marker base-prior → leakage re-analysis on #478.

At the non-saturated marker anchor of #478, extract the relationship between a
held-out persona's base-model prior on the marker ` ※` and how much the marker
leaks (trained − base ``log P(※)`` shift at the post-response slot, on-policy).

Disambiguates two readings of the base-prior result:
- **#504 (saturated):** partial ρ ≈ −0.87 between base prior and ΔG, but
  bystanders were 92% saturated so ΔG = trained − base ≈ −base_prior
  mechanically → suspected ceiling arithmetic.
- **#500 (facts, non-saturated):** positive ρ (propensity story).
- **#478 (marker, non-saturated):** this re-analysis — the answer was never
  extracted in the parent.

Data source
-----------
The per-cell ``result.json`` files (with per-question ``logp_trained_per_q``,
``logp_base_per_q``, ``deltaLogP_per_q`` arrays per held-out persona) and the
aggregate ``tidy.csv`` (per-cell-persona ``min_dist``) live on the unmerged
``issue-478`` branch at commit ``7efb037736831c66cf87aaa79c11237ac9268b83``.
The pinned HF data revision ``a9fc5a9cbc81c4b774ff66da0022f9055e18da5f``
contains only the raw on-policy responses (text), not the log P values,
because #478's eval pipeline persisted log P only to per-cell ``result.json``
files under ``eval_results/issue_478/cell_*/``. We read those out of the
sibling branch via ``git show``.

Why CORE-only (80 of 92 cells)
-----------------------------
The 12 ARM (decomposition) cells trained each source persona on a DIFFERENT
marker (``per_marker`` field), so the source-side exposure to ` ※` is not
uniform within a cell. For a clean per-row "marker is ` ※`, base prior is
P(` ※`)" analysis, we restrict to the 80 CORE cells (track == "CORE") where
every source persona trained on ` ※`. ARM cells are reported in a sanity
appendix only if ``--include-arm`` is passed.

Distance metric
---------------
``min_dist`` is the cosine distance from each held-out persona to the NEAREST
source persona in that cell's K-subset. #478 computed it via base-model
mean-pooled last-hidden-state cosine distance (the persona-vectors recipe);
we reuse the cached value from the aggregate ``tidy.csv``.

Outputs
-------
- ``eval_results/issue_478/base_prior_reanalysis/tidy.parquet`` — one row per
  (cell, seed, held_out_persona, question) with `trained_logp`, `base_prior`,
  `shift`, `K`, `min_dist`, `band`.
- ``eval_results/issue_478/base_prior_reanalysis/summary.json`` — raw +
  partial Spearman ρ for shift AND absolute trained_logp, each with 1000-
  resample persona-cluster bootstrap 95% CIs; plus a ceiling-check block
  confirming non-saturation.
- ``figures/issue_478/base_prior_reanalysis/shift_vs_base_prior.{png,pdf}``
  + ``.meta.json`` — scatter coloured by distance band.
- ``figures/issue_478/base_prior_reanalysis/absolute_trained_vs_base_prior.{png,pdf}``
  + ``.meta.json`` — companion absolute-trained panel.

Usage::

    uv run python scripts/issue531_base_prior_reanalysis.py              # full
    uv run python scripts/issue531_base_prior_reanalysis.py --limit-cells 4
    uv run python scripts/issue531_base_prior_reanalysis.py --include-arm
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import set_paper_style  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

# Commit on the unmerged issue-478 branch carrying the aggregate tidy.csv +
# the 92 per-cell result.json files with per-question log-prob arrays.
ISSUE_478_AGG_SHA = "7efb037736831c66cf87aaa79c11237ac9268b83"

# Pinned HF data revision (matches the parent #478 clean-result's reproducibility
# card). The raw on-policy responses live there; log P values are derived from
# the sibling-branch result.json files.
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_DATA_REV = "a9fc5a9cbc81c4b774ff66da0022f9055e18da5f"

MARKER_TEXT = " ※"  # leading space, Qwen-2.5-7B id 83399
MARKER_ID = 83399

# Distance bands (#478 panel by-band assignments)
BAND_ORDER = ["near", "near-mid", "mid", "far", "very-far", "tail"]
NEAR_BAND_GROUP = {"near", "near-mid"}
FAR_BAND_GROUP = {"far", "very-far", "tail"}

# Output directories
OUTPUT_TIDY_DIR = PROJECT_ROOT / "eval_results" / "issue_478" / "base_prior_reanalysis"
OUTPUT_FIG_DIR = PROJECT_ROOT / "figures" / "issue_478" / "base_prior_reanalysis"

# Bootstrap
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260609  # 2026-06-09, the task creation date

# Non-saturation check thresholds (per plan §success criteria)
MAX_MEAN_TRAINED_LOGP_FOR_NONSAT = -5.0  # mean trained logp must be < -5 nats
MAX_SHARE_NEAR_CEILING = 0.05  # share of rows with trained_logp > -1 must be < 5%

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("issue531_base_prior_reanalysis")


# ── Data loading ─────────────────────────────────────────────────────────────


def _git_show(ref: str, path: str) -> bytes:
    """Return raw bytes of `path` at git ref `ref` from the worktree."""
    cmd = ["git", "show", f"{ref}:{path}"]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True)
    return proc.stdout


def _git_ls_tree(ref: str, prefix: str) -> list[str]:
    """List files under `prefix` at git ref `ref`. Returns relative paths."""
    cmd = ["git", "ls-tree", "-r", "--name-only", ref, prefix]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def load_aggregate_tidy() -> pd.DataFrame:
    """Read the per-(cell, seed, persona) aggregate tidy.csv from the
    issue-478 branch. Returns ``min_dist``, ``band``, ``deltaLogP_mean``,
    ``logp_trained_mean``, ``logp_base_mean`` per row.
    """
    raw = _git_show(ISSUE_478_AGG_SHA, "eval_results/issue_478/aggregate/tidy.csv")
    df = pd.read_csv(io.BytesIO(raw))
    log.info("Loaded aggregate tidy: %d rows, columns=%s", len(df), list(df.columns))
    return df


def load_cell_result(cell_dir: str) -> dict:
    """Read one per-cell result.json from the issue-478 branch."""
    raw = _git_show(ISSUE_478_AGG_SHA, f"eval_results/issue_478/{cell_dir}/result.json")
    return json.loads(raw.decode("utf-8"))


def list_cells(*, include_arm: bool, limit: int | None = None) -> list[str]:
    """Enumerate the cell_* directories at the issue-478 aggregate SHA.

    Returns directory names like ``cell_K1_c00_seed42``, ``cell_ARM_K2_a0_seed42``.
    """
    entries = _git_ls_tree(ISSUE_478_AGG_SHA, "eval_results/issue_478/")
    seen = set()
    cells = []
    for path in entries:
        parts = path.split("/")
        if len(parts) < 3:
            continue
        name = parts[2]
        if not name.startswith("cell_"):
            continue
        if not include_arm and "ARM_" in name:
            continue
        if name in seen:
            continue
        seen.add(name)
        cells.append(name)
    cells.sort()
    if limit is not None:
        cells = cells[:limit]
    log.info(
        "Discovered %d cell directories (include_arm=%s, limit=%s)", len(cells), include_arm, limit
    )
    return cells


# ── Tidy-table construction ──────────────────────────────────────────────────


def _band_lookup_from_aggregate(agg: pd.DataFrame) -> dict[tuple[str, int, str], str]:
    """Map (cell_id, seed, held_out_persona) → band, per aggregate tidy.csv."""
    return {
        (row.cell_id, int(row.seed), row.held_out_persona): row.band
        for row in agg.itertuples(index=False)
    }


def _min_dist_lookup_from_aggregate(agg: pd.DataFrame) -> dict[tuple[str, int, str], float]:
    """Map (cell_id, seed, held_out_persona) → min_dist."""
    return {
        (row.cell_id, int(row.seed), row.held_out_persona): float(row.min_dist)
        for row in agg.itertuples(index=False)
    }


def build_per_question_tidy(
    cells: list[str],
    agg: pd.DataFrame,
) -> pd.DataFrame:
    """Build the per-row tidy table by joining per-question arrays from each
    per-cell ``result.json`` with the aggregate's ``min_dist`` + ``band``.

    Returns one row per (cell_id, seed, held_out_persona, question_idx) with
    ``trained_logp``, ``base_prior``, ``shift``, ``K``, ``min_dist``, ``band``.
    """
    band_lut = _band_lookup_from_aggregate(agg)
    dist_lut = _min_dist_lookup_from_aggregate(agg)

    rows = []
    for cell_dir in cells:
        d = load_cell_result(cell_dir)
        cell_id = d["cell_id"]
        seed = int(d["seed"])
        K = int(d["K"])
        track = d.get("track", "?")

        held_out = d["eval"]["held_out"]
        for persona, rec in held_out.items():
            # The CORE per-cell result.json doesn't have a `per_marker` block
            # — its single trained marker is ` ※`, so logp_*_per_q are already
            # the ` ※` log-probs. The ARM result.json may have `per_marker`
            # keyed by marker_id; we don't pull ARM in CORE-only mode.
            if "per_marker" in rec:
                # ARM cell: pull the ` ※`-specific log-probs from per_marker[MARKER_ID]
                per_m = rec["per_marker"].get(str(MARKER_ID))
                if per_m is None:
                    log.warning(
                        "ARM cell %s persona %s has no per_marker[%d] — skipping",
                        cell_id,
                        persona,
                        MARKER_ID,
                    )
                    continue
                t_per_q = per_m["logp_trained_per_q"]
                b_per_q = per_m["logp_base_per_q"]
            else:
                t_per_q = rec["logp_trained_per_q"]
                b_per_q = rec["logp_base_per_q"]

            n_q = len(t_per_q)
            if n_q != len(b_per_q):
                raise ValueError(
                    f"length mismatch trained={len(t_per_q)} base={len(b_per_q)} "
                    f"for {cell_id} seed {seed} persona {persona}"
                )

            min_dist = dist_lut.get((cell_id, seed, persona))
            band = band_lut.get((cell_id, seed, persona))
            if min_dist is None or band is None:
                # ARM cells aren't in the aggregate tidy.csv (which is CORE-only
                # n=2800). Use NaN; downstream stats drop NaN-rows.
                pass

            for q_idx in range(n_q):
                t = float(t_per_q[q_idx])
                b = float(b_per_q[q_idx])
                rows.append(
                    {
                        "cell_id": cell_id,
                        "seed": seed,
                        "track": track,
                        "K": K,
                        "held_out_persona": persona,
                        "question_idx": q_idx,
                        "trained_logp": t,
                        "base_prior": b,
                        "shift": t - b,
                        "min_dist": min_dist,
                        "band": band,
                    }
                )

    df = pd.DataFrame(rows)
    log.info(
        "Per-question tidy table: %d rows, %d cells, %d personas",
        len(df),
        df["cell_id"].nunique(),
        df["held_out_persona"].nunique(),
    )
    return df


# ── Statistics ──────────────────────────────────────────────────────────────


def spearman_with_persona_bootstrap_ci(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    persona_col: str = "held_out_persona",
    n_boot: int = N_BOOTSTRAP,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Point Spearman ρ + 1000-resample persona-cluster bootstrap CI.

    Cluster bootstrap: resample held-out personas WITH REPLACEMENT, then
    refit ρ on the resampled rows. Respects the within-persona row-row
    dependence (35 personas × ~varied n_rows per persona).
    """
    sub = df[[x_col, y_col, persona_col]].dropna()
    rho_point, p_point = spearmanr(sub[x_col], sub[y_col])

    personas = np.array(sorted(sub[persona_col].unique()))
    n_personas = len(personas)
    by_persona = {p: sub[sub[persona_col] == p][[x_col, y_col]].to_numpy() for p in personas}

    rng = np.random.default_rng(rng_seed)
    rho_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_personas, size=n_personas)
        chosen = personas[idx]
        x_concat = np.concatenate([by_persona[p][:, 0] for p in chosen])
        y_concat = np.concatenate([by_persona[p][:, 1] for p in chosen])
        if len(x_concat) < 3:
            rho_boot[b] = np.nan
            continue
        rho_b, _ = spearmanr(x_concat, y_concat)
        rho_boot[b] = rho_b

    rho_boot_clean = rho_boot[~np.isnan(rho_boot)]
    ci_lo, ci_hi = np.percentile(rho_boot_clean, [2.5, 97.5])
    return {
        "rho_point": float(rho_point),
        "p_point": float(p_point),
        "ci_lo_95": float(ci_lo),
        "ci_hi_95": float(ci_hi),
        "boot_mean": float(np.mean(rho_boot_clean)),
        "n_rows": len(sub),
        "n_personas": int(n_personas),
        "n_boot": len(rho_boot_clean),
    }


def partial_spearman(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_col: str,
    control_cols: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Compute residuals of ``y_col`` and ``x_col`` after rank-regression on
    ``control_cols``. Returns (resid_x, resid_y) aligned with df.index.

    The partial Spearman ρ is then ``spearmanr(resid_x, resid_y).statistic``.
    Residualization is on ranks (so it's still rank-based and matches the
    Spearman-partial definition).
    """
    sub = df[[y_col, x_col, *control_cols]].dropna()

    def _ranks(s):
        return rankdata(s.to_numpy(), method="average")

    y_r = _ranks(sub[y_col])
    x_r = _ranks(sub[x_col])
    ctl_r = np.column_stack([_ranks(sub[c]) for c in control_cols])

    # Add intercept column
    ctl_r_with_intercept = np.column_stack([np.ones(len(ctl_r)), ctl_r])

    # Linear regression of ranks on rank controls
    coef_y, *_ = np.linalg.lstsq(ctl_r_with_intercept, y_r, rcond=None)
    coef_x, *_ = np.linalg.lstsq(ctl_r_with_intercept, x_r, rcond=None)

    resid_y = y_r - ctl_r_with_intercept @ coef_y
    resid_x = x_r - ctl_r_with_intercept @ coef_x

    return (
        pd.Series(resid_x, index=sub.index, name=f"{x_col}__residual"),
        pd.Series(resid_y, index=sub.index, name=f"{y_col}__residual"),
    )


def partial_spearman_with_persona_bootstrap(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_col: str,
    control_cols: list[str],
    persona_col: str = "held_out_persona",
    n_boot: int = N_BOOTSTRAP,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Partial Spearman ρ (x | controls) → y, with persona-cluster bootstrap CI."""
    needed = [y_col, x_col, persona_col, *control_cols]
    sub = df[needed].dropna().copy()

    resid_x, resid_y = partial_spearman(sub, y_col=y_col, x_col=x_col, control_cols=control_cols)
    rho_point, p_point = spearmanr(resid_x, resid_y)

    # Persona-cluster bootstrap of the partial ρ.
    personas = np.array(sorted(sub[persona_col].unique()))
    n_personas = len(personas)
    by_persona_idx = {p: sub.index[sub[persona_col] == p].to_numpy() for p in personas}

    rng = np.random.default_rng(rng_seed)
    rho_boot = np.empty(n_boot)
    for b in range(n_boot):
        chosen = personas[rng.integers(0, n_personas, size=n_personas)]
        idx_chosen = np.concatenate([by_persona_idx[p] for p in chosen])
        bsub = sub.loc[idx_chosen]
        if len(bsub) < 3:
            rho_boot[b] = np.nan
            continue
        try:
            rx_b, ry_b = partial_spearman(
                bsub,
                y_col=y_col,
                x_col=x_col,
                control_cols=control_cols,
            )
            rho_b, _ = spearmanr(rx_b, ry_b)
            rho_boot[b] = rho_b
        except (ValueError, np.linalg.LinAlgError):
            rho_boot[b] = np.nan

    rho_boot_clean = rho_boot[~np.isnan(rho_boot)]
    ci_lo, ci_hi = np.percentile(rho_boot_clean, [2.5, 97.5])
    return {
        "rho_point": float(rho_point),
        "p_point": float(p_point),
        "ci_lo_95": float(ci_lo),
        "ci_hi_95": float(ci_hi),
        "boot_mean": float(np.mean(rho_boot_clean)),
        "n_rows": len(sub),
        "n_personas": int(n_personas),
        "n_boot": len(rho_boot_clean),
        "control_cols": list(control_cols),
    }


def ceiling_check(df: pd.DataFrame) -> dict:
    """Confirm non-saturation: mean trained_logp << 0 and share at ceiling < 5%."""
    t = df["trained_logp"].to_numpy()
    return {
        "mean_trained_logp": float(np.mean(t)),
        "p05_trained_logp": float(np.percentile(t, 5)),
        "p95_trained_logp": float(np.percentile(t, 95)),
        "share_above_neg1": float((t > -1.0).mean()),
        "share_above_neg2": float((t > -2.0).mean()),
        "n_rows": len(t),
        "passes_nonsat_threshold": bool(
            np.mean(t) < MAX_MEAN_TRAINED_LOGP_FOR_NONSAT
            and (t > -1.0).mean() < MAX_SHARE_NEAR_CEILING
        ),
    }


# ── Figures ──────────────────────────────────────────────────────────────────


def _get_band_palette() -> dict[str, str]:
    """Distance-band palette (cooler near, warmer far)."""
    cmap = plt.get_cmap("viridis")
    return {band: cmap(i / (len(BAND_ORDER) - 1)) for i, band in enumerate(BAND_ORDER)}


def _figure_meta(
    *,
    fig_name: str,
    fig_path: Path,
    df: pd.DataFrame,
    extra: dict,
) -> dict:
    """Common figure meta.json content."""
    return {
        "figure": fig_name,
        "produced_by": "scripts/issue531_base_prior_reanalysis.py",
        "git_commit_at_render": _current_git_commit(),
        "data_source": {
            "hf_data_repo": HF_DATA_REPO,
            "hf_data_revision": HF_DATA_REV,
            "issue_478_aggregate_sha": ISSUE_478_AGG_SHA,
        },
        "rows_used": len(df),
        "cells_used": int(df["cell_id"].nunique()),
        "personas_used": int(df["held_out_persona"].nunique()),
        "marker_text": MARKER_TEXT,
        "marker_token_id": MARKER_ID,
        "rendered_at_utc": datetime.now(UTC).isoformat(),
        **extra,
    }


def _current_git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def plot_shift_vs_base_prior(
    df: pd.DataFrame,
    summary: dict,
    out_dir: Path,
) -> None:
    """Scatter of shift vs base_prior coloured by distance band."""
    set_paper_style(target="blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    palette = _get_band_palette()

    # One subsample per band for visual readability; the stats fit on ALL rows.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for band in BAND_ORDER:
        sub = df[df["band"] == band]
        if sub.empty:
            continue
        # Downsample to ~600 pts/band so the cloud stays readable.
        if len(sub) > 600:
            sample_idx = rng.choice(len(sub), size=600, replace=False)
            sub = sub.iloc[sample_idx]
        ax.scatter(
            sub["base_prior"],
            sub["shift"],
            s=8,
            alpha=0.30,
            color=palette[band],
            label=band,
            linewidths=0,
        )

    raw = summary["raw_spearman_shift"]
    par = summary["partial_spearman_shift"]
    annotation = (
        f"Raw Spearman ρ = {raw['rho_point']:+.3f}"
        f"  [95% CI {raw['ci_lo_95']:+.3f}, {raw['ci_hi_95']:+.3f}]\n"
        f"Partial ρ (|min_dist, K) = {par['rho_point']:+.3f}"
        f"  [95% CI {par['ci_lo_95']:+.3f}, {par['ci_hi_95']:+.3f}]"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "lightgrey", "boxstyle": "round,pad=0.4"},
    )

    ax.set_xlabel(r"Base-model log P(" + MARKER_TEXT + ") at post-response slot (nats)")
    ax.set_ylabel(r"Trained − base log P(" + MARKER_TEXT + ") shift (nats)")
    ax.set_title(
        "Shift falls with base prior off-ceiling — same sign as #504 (saturated), weaker",
        loc="left",
    )
    leg = ax.legend(
        title="Distance band",
        loc="lower left",
        fontsize=8,
        title_fontsize=8,
        ncols=2,
        markerscale=2.0,
        frameon=True,
    )
    leg.get_frame().set_edgecolor("lightgrey")
    plt.tight_layout()

    png_path = out_dir / "shift_vs_base_prior.png"
    pdf_path = out_dir / "shift_vs_base_prior.pdf"
    meta_path = out_dir / "shift_vs_base_prior.meta.json"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    meta = _figure_meta(
        fig_name="shift_vs_base_prior",
        fig_path=png_path,
        df=df,
        extra={
            "x_axis": f"base-model log P({MARKER_TEXT}) at post-response slot (nats)",
            "y_axis": f"trained − base log P({MARKER_TEXT}) shift (nats)",
            "rho_raw_shift": summary["raw_spearman_shift"]["rho_point"],
            "rho_partial_shift": summary["partial_spearman_shift"]["rho_point"],
            "downsample_per_band": 600,
        },
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Wrote %s + .pdf + .meta.json", png_path)


def plot_absolute_trained_vs_base_prior(
    df: pd.DataFrame,
    summary: dict,
    out_dir: Path,
) -> None:
    """Companion: absolute trained log P vs base prior."""
    set_paper_style(target="blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    palette = _get_band_palette()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for band in BAND_ORDER:
        sub = df[df["band"] == band]
        if sub.empty:
            continue
        if len(sub) > 600:
            sample_idx = rng.choice(len(sub), size=600, replace=False)
            sub = sub.iloc[sample_idx]
        ax.scatter(
            sub["base_prior"],
            sub["trained_logp"],
            s=8,
            alpha=0.30,
            color=palette[band],
            label=band,
            linewidths=0,
        )

    raw = summary["raw_spearman_abs"]
    par = summary["partial_spearman_abs"]
    annotation = (
        f"Raw Spearman ρ = {raw['rho_point']:+.3f}"
        f"  [95% CI {raw['ci_lo_95']:+.3f}, {raw['ci_hi_95']:+.3f}]\n"
        f"Partial ρ (|min_dist, K) = {par['rho_point']:+.3f}"
        f"  [95% CI {par['ci_lo_95']:+.3f}, {par['ci_hi_95']:+.3f}]"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "lightgrey", "boxstyle": "round,pad=0.4"},
    )

    ax.set_xlabel(r"Base-model log P(" + MARKER_TEXT + ") at post-response slot (nats)")
    ax.set_ylabel(r"Trained log P(" + MARKER_TEXT + ") at post-response slot (nats)")
    ax.set_title(
        "Absolute trained log-prob rises with base prior — propensity sign, matches #500",
        loc="left",
    )
    leg = ax.legend(
        title="Distance band",
        loc="lower right",
        fontsize=8,
        title_fontsize=8,
        ncols=2,
        markerscale=2.0,
        frameon=True,
    )
    leg.get_frame().set_edgecolor("lightgrey")
    plt.tight_layout()

    png_path = out_dir / "absolute_trained_vs_base_prior.png"
    pdf_path = out_dir / "absolute_trained_vs_base_prior.pdf"
    meta_path = out_dir / "absolute_trained_vs_base_prior.meta.json"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    meta = _figure_meta(
        fig_name="absolute_trained_vs_base_prior",
        fig_path=png_path,
        df=df,
        extra={
            "x_axis": f"base-model log P({MARKER_TEXT}) at post-response slot (nats)",
            "y_axis": f"trained log P({MARKER_TEXT}) at post-response slot (nats)",
            "rho_raw_abs": summary["raw_spearman_abs"]["rho_point"],
            "rho_partial_abs": summary["partial_spearman_abs"]["rho_point"],
            "downsample_per_band": 600,
        },
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Wrote %s + .pdf + .meta.json", png_path)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit-cells",
        type=int,
        default=None,
        help="Limit to the first N cells (smoke-test mode).",
    )
    parser.add_argument(
        "--include-arm",
        action="store_true",
        help="Include the 12 ARM (decomposition) cells. Default: CORE only.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOTSTRAP,
        help=f"Bootstrap resamples (default {N_BOOTSTRAP}).",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure rendering (for fast smoke runs).",
    )
    args = parser.parse_args()

    OUTPUT_TIDY_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== Phase 1: load aggregate tidy from sibling branch ===")
    agg = load_aggregate_tidy()

    log.info("=== Phase 2: enumerate cells ===")
    cells = list_cells(include_arm=args.include_arm, limit=args.limit_cells)

    log.info("=== Phase 3: build per-question tidy table ===")
    tidy = build_per_question_tidy(cells, agg)

    # Save tidy table (parquet preferred; fallback to csv.gz if pyarrow is unavailable)
    tidy_parquet = OUTPUT_TIDY_DIR / "tidy.parquet"
    try:
        tidy.to_parquet(tidy_parquet, index=False)
        log.info("Wrote tidy table: %s (%d rows)", tidy_parquet, len(tidy))
    except Exception as e:
        log.warning("Parquet write failed (%s); falling back to csv.gz", e)
        tidy_csv = OUTPUT_TIDY_DIR / "tidy.csv.gz"
        tidy.to_csv(tidy_csv, index=False, compression="gzip")
        log.info("Wrote tidy table: %s (%d rows)", tidy_csv, len(tidy))

    log.info("=== Phase 4: ceiling check (non-saturation) ===")
    cc = ceiling_check(tidy)
    log.info(
        "ceiling: mean trained_logp = %.3f, p05/p95 = %.3f / %.3f, share>−1 = %.4f, passes=%s",
        cc["mean_trained_logp"],
        cc["p05_trained_logp"],
        cc["p95_trained_logp"],
        cc["share_above_neg1"],
        cc["passes_nonsat_threshold"],
    )
    if not cc["passes_nonsat_threshold"]:
        log.warning(
            "Non-saturation gate NOT met (mean=%.3f, share>−1=%.4f) — "
            "the marker may not be off-ceiling at #478. Stats still computed.",
            cc["mean_trained_logp"],
            cc["share_above_neg1"],
        )

    log.info("=== Phase 5: Spearman ρ (raw + partial) for shift ===")
    raw_shift = spearman_with_persona_bootstrap_ci(
        tidy,
        x_col="base_prior",
        y_col="shift",
        n_boot=args.n_boot,
    )
    log.info(
        "RAW   ρ(base_prior, shift) = %+.4f [%+.4f, %+.4f] (boot mean %+.4f, n_rows=%d, n_pers=%d)",
        raw_shift["rho_point"],
        raw_shift["ci_lo_95"],
        raw_shift["ci_hi_95"],
        raw_shift["boot_mean"],
        raw_shift["n_rows"],
        raw_shift["n_personas"],
    )
    par_shift = partial_spearman_with_persona_bootstrap(
        tidy,
        x_col="base_prior",
        y_col="shift",
        control_cols=["min_dist", "K"],
        n_boot=args.n_boot,
    )
    log.info(
        "PART  ρ(base_prior, shift | min_dist, K) = %+.4f [%+.4f, %+.4f] "
        "(boot mean %+.4f, n_rows=%d, n_pers=%d)",
        par_shift["rho_point"],
        par_shift["ci_lo_95"],
        par_shift["ci_hi_95"],
        par_shift["boot_mean"],
        par_shift["n_rows"],
        par_shift["n_personas"],
    )

    log.info("=== Phase 6: Spearman ρ (raw + partial) for absolute trained_logp ===")
    raw_abs = spearman_with_persona_bootstrap_ci(
        tidy,
        x_col="base_prior",
        y_col="trained_logp",
        n_boot=args.n_boot,
    )
    log.info(
        "RAW   ρ(base_prior, trained_logp) = %+.4f [%+.4f, %+.4f]",
        raw_abs["rho_point"],
        raw_abs["ci_lo_95"],
        raw_abs["ci_hi_95"],
    )
    par_abs = partial_spearman_with_persona_bootstrap(
        tidy,
        x_col="base_prior",
        y_col="trained_logp",
        control_cols=["min_dist", "K"],
        n_boot=args.n_boot,
    )
    log.info(
        "PART  ρ(base_prior, trained_logp | min_dist, K) = %+.4f [%+.4f, %+.4f]",
        par_abs["rho_point"],
        par_abs["ci_lo_95"],
        par_abs["ci_hi_95"],
    )

    log.info("=== Phase 7: assemble summary.json ===")
    summary = {
        "task": "issue_531_base_prior_reanalysis",
        "produced_by": "scripts/issue531_base_prior_reanalysis.py",
        "produced_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _current_git_commit(),
        "data_source": {
            "hf_data_repo": HF_DATA_REPO,
            "hf_data_revision": HF_DATA_REV,
            "issue_478_aggregate_sha": ISSUE_478_AGG_SHA,
            "issue_478_aggregate_path": "eval_results/issue_478/{cell_*,aggregate/tidy.csv}",
            "track_filter": "CORE" if not args.include_arm else "CORE+ARM",
        },
        "marker": {
            "text": MARKER_TEXT,
            "token_id": MARKER_ID,
            "note": "Qwen-2.5-7B token id 83399 (leading space)",
        },
        "n_cells": int(tidy["cell_id"].nunique()),
        "n_seeds": int(tidy["seed"].nunique()),
        "n_personas": int(tidy["held_out_persona"].nunique()),
        "n_rows": len(tidy),
        "ceiling_check": cc,
        "raw_spearman_shift": raw_shift,
        "partial_spearman_shift": par_shift,
        "raw_spearman_abs": raw_abs,
        "partial_spearman_abs": par_abs,
        "head_to_head": {
            "issue_478_marker_nonsat_partial_shift": par_shift["rho_point"],
            "issue_504_marker_saturated_partial_quote": -0.874,
            "issue_504_marker_saturated_partial_quote_note": (
                "From task #504 body: partial ρ = −0.874 "
                "(≈ −0.87 quoted in #531 plan) "
                "between base_prior_marker and ΔG at the saturated anchor; "
                "raw ρ = −0.895."
            ),
            "issue_500_facts_nonsat_partial_marine": -0.01,
            "issue_500_facts_nonsat_partial_marine_note": (
                "From task #500 body: partial ρ(cos | prior) = -0.01 means "
                "cosine adds nothing once prior is controlled; the BASE-PRIOR "
                "ρ (the analogue we care about) is the raw +0.80 (marine), "
                "bootstrap mean +0.50 [95% CI +0.18, +0.75]. No partial "
                "Spearman of LEAK on BASE PRIOR controlling for cosine was "
                "reported, but the joint-fit standardized β_prior = +0.78 "
                "(marine) implies a strongly positive partial relationship."
            ),
            "issue_500_facts_nonsat_raw_marine": 0.80,
            "issue_500_facts_nonsat_raw_local_resident": 0.61,
            "issue_500_facts_nonsat_raw_courthouse": 0.30,
        },
    }
    summary_path = OUTPUT_TIDY_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote summary: %s", summary_path)

    if not args.skip_figures:
        log.info("=== Phase 8: render figures ===")
        plot_shift_vs_base_prior(tidy, summary, OUTPUT_FIG_DIR)
        plot_absolute_trained_vs_base_prior(tidy, summary, OUTPUT_FIG_DIR)

    # Headline summary print
    print()
    print("=" * 78)
    print("Issue #531 — non-saturated marker base-prior → leakage re-analysis")
    print("=" * 78)
    print(
        f"  rows:       {len(tidy):>10,}    cells: {tidy['cell_id'].nunique():>4}"
        f"    seeds: {tidy['seed'].nunique()}    personas: {tidy['held_out_persona'].nunique()}"
    )
    print()
    print(
        f"  ceiling:    mean trained log P = {cc['mean_trained_logp']:+.3f} nats"
        f"  (share > −1 nat: {cc['share_above_neg1']:.3%})"
    )
    print(f"              non-saturation gate PASSES: {cc['passes_nonsat_threshold']}")
    print()
    print("  Spearman ρ (base_prior → shift):")
    print(
        f"    RAW       ρ = {raw_shift['rho_point']:+.4f}"
        f"   95% CI [{raw_shift['ci_lo_95']:+.4f}, {raw_shift['ci_hi_95']:+.4f}]"
    )
    print(
        f"    PARTIAL   ρ = {par_shift['rho_point']:+.4f}"
        f"   95% CI [{par_shift['ci_lo_95']:+.4f}, {par_shift['ci_hi_95']:+.4f}]"
        f"   (controls: min_dist, K)"
    )
    print()
    print("  Spearman ρ (base_prior → absolute trained log P):")
    print(
        f"    RAW       ρ = {raw_abs['rho_point']:+.4f}"
        f"   95% CI [{raw_abs['ci_lo_95']:+.4f}, {raw_abs['ci_hi_95']:+.4f}]"
    )
    print(
        f"    PARTIAL   ρ = {par_abs['rho_point']:+.4f}"
        f"   95% CI [{par_abs['ci_lo_95']:+.4f}, {par_abs['ci_hi_95']:+.4f}]"
        f"   (controls: min_dist, K)"
    )
    print()
    print("  Head-to-head:")
    print(f"    #478 marker (non-saturated)  partial ρ shift = {par_shift['rho_point']:+.3f}")
    print("    #504 marker (saturated)      partial ρ      = −0.874 (quoted)")
    print("    #500 facts  (non-saturated)  raw ρ marine   = +0.80 (quoted)")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())

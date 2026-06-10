# ruff: noqa: RUF001
"""Emission-rate concordance re-analysis for issue #480 (followup: emission-rate-concordance).

Re-reads the existing 138-cell (source, bystander) matrix with marker EMISSION RATE as the
marker-leakage DV (non-saturating, behavioral) instead of the saturation-inverted log-prob
delta, and asks whether per-bystander emission rate rank-correlates with the frozen #411
sycophancy leakage.

Outputs:
    eval_results/issue_480/emission-rate-concordance/concordance_stats.json
    figures/issue_480/emission_rate_vs_sycophancy_se.{png,pdf,meta.json}

Analysis-only: no training, no model loads, no GPU. Runs in seconds.

Band-stopped-anchor-rerun (#480 follow-up round 2) flags — every default equals
the module constant the round-1 run hardcoded, so a no-flag invocation is
byte-identical to round 1 (up to the volatile git_commit_sha / timestamp_utc
provenance fields):

    --matrix-path   input marker_delta_matrix.json (default: round-1 path)
    --stats-dir     output dir for concordance_stats.json (default: round-1 dir)
    --figure-stem   figure stem under figures/ (default: round-1 stem)
    --x-field       marker-side DV column (default: emission_rate; the rerun
                    also passes marker_delta for the secondary log-prob DV —
                    the stats file is then named concordance_stats_<x_field>.json
                    so the two runs never collide)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / "eval_results" / "issue_480" / "marker_delta_matrix.json"
STATS_DIR = REPO_ROOT / "eval_results" / "issue_480" / "emission-rate-concordance"
STATS_PATH = STATS_DIR / "concordance_stats.json"
FIGURE_STEM = "issue_480/emission_rate_vs_sycophancy_se"
FIGURE_DIR = REPO_ROOT / "figures"

EXPECTED_SCHEMA = "issue_480_marker_delta_matrix_v1"
OUTPUT_SCHEMA = "issue_480_emission_rate_concordance_v1"
EXPECTED_N_ROWS = 138

X_FIELD = "emission_rate"
Y_FIELD = "sycophancy_delta"
CONTROL_FIELDS = ("cosine_l20_baseline", "bystander_base_rate")

# Plain-English DV description per supported x-field (used in the output
# JSON's "dv" block and the figure x-axis label).
X_FIELD_DESCRIPTIONS = {
    "emission_rate": "on-policy marker emission rate per bystander",
    "marker_delta": "on-policy log P(marker) trained - base per bystander (nats)",
    # inband-logprob-concordance (#480 round 3): the gauge-free mechanistic
    # secondary from the same four-float slot reads — per-cell median
    # Δ(z_marker - z_eos), trained - base. Present only on four-float
    # matrices (band-stop rerun onward); load_matrix fails loud when the
    # column is missing from an older matrix.
    "eos_margin_delta": ("EOS-margin Δ(z_marker − z_eos) trained - base per bystander (logits)"),
}
X_FIELD_AXIS_LABELS = {
    "emission_rate": "Marker emission rate (fraction of own responses)",
    "marker_delta": "Marker log P trained - base (nats)",
    "eos_margin_delta": "Marker EOS margin Δ(z_marker − z_eos) trained − base (logits)",
}


def _required_row_fields(x_field: str) -> tuple[str, ...]:
    """Row fields the matrix must carry for a given marker-side DV column."""
    return (
        "source",
        "bystander",
        x_field,
        Y_FIELD,
        "sycophancy_delta_se",
        *CONTROL_FIELDS,
        "source_base_rate",
    )


REQUIRED_ROW_FIELDS = _required_row_fields(X_FIELD)

MIN_NONZERO_CELLS = 5
MIN_DISTINCT_VALUES = 3

N_BOOT = 10_000
N_PERM = 100_000
BOOT_SEED = 480
PERM_SEED = 4801
PARTIAL_PERM_SEED = 4802
FE_PERM_SEED = 4803
ALPHA = 0.05

BOOTSTRAP_METHOD = "percentile"
PARTIAL_METHOD = (
    "rank-transform all variables (Spearman partial), OLS-residualize x-ranks and y-ranks "
    "on covariate ranks + intercept, Pearson on residuals; permutation p from permuting "
    "the x-rank residuals"
)
POOLED_CAVEAT = (
    "Pooled all-cells Spearman mixes between-source and within-source variation; sources "
    "differ in training strength and base rates, so this estimate is confounded by "
    "between-source differences. Prefer the per-source and source-FE (within-source "
    "demeaned rank) estimates."
)

SE_SOURCE = "software_engineer"


def _git_commit_sha() -> str:
    """Return the full git commit SHA of the worktree HEAD."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        timeout=10,
    )
    return out.stdout.strip()


def load_matrix(matrix_path: Path = MATRIX_PATH, x_field: str = X_FIELD) -> list[dict]:
    """Load and validate the 138-cell marker-delta matrix; fail loud on any mismatch."""
    required = _required_row_fields(x_field)
    payload = json.loads(matrix_path.read_text())
    if payload["schema"] != EXPECTED_SCHEMA:
        raise ValueError(f"schema mismatch: {payload['schema']!r} != {EXPECTED_SCHEMA!r}")
    rows = payload["rows"]
    if payload["n_rows"] != EXPECTED_N_ROWS or len(rows) != EXPECTED_N_ROWS:
        raise ValueError(f"expected {EXPECTED_N_ROWS} rows, got n_rows={payload['n_rows']}")
    for i, row in enumerate(rows):
        missing = [k for k in required if k not in row]
        if missing:
            raise KeyError(f"row {i} missing fields: {missing}")
        for k in required:
            if k in ("source", "bystander"):
                continue
            if row[k] is None or not np.isfinite(row[k]):
                raise ValueError(f"row {i} ({row['source']},{row['bystander']}): bad {k}={row[k]}")
    return rows


def _pearson_rows(xm: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation between each row of ``xm`` and the vector ``y``.

    Rows of ``xm`` with zero variance yield NaN. ``y`` must have nonzero variance.
    """
    assert xm.ndim == 2 and y.ndim == 1 and xm.shape[1] == y.shape[0], (xm.shape, y.shape)
    xc = xm - xm.mean(axis=1, keepdims=True)
    yc = y - y.mean()
    y_ss = float(yc @ yc)
    assert y_ss > 0, "y has zero variance"
    x_ss = (xc**2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x_ss > 0, (xc @ yc) / np.sqrt(x_ss * y_ss), np.nan)


def spearman_with_permutation(
    x: np.ndarray, y: np.ndarray, n_perm: int, seed: int
) -> dict[str, float]:
    """Spearman rho with scipy asymptotic p and a within-vector permutation p (two-sided)."""
    rho, p_asym = spearmanr(x, y)
    rx, ry = rankdata(x), rankdata(y)
    rng = np.random.default_rng(seed)
    perms = rng.permuted(np.tile(rx, (n_perm, 1)), axis=1)
    rho_perm = _pearson_rows(perms, ry)
    assert not np.isnan(rho_perm).any(), "permutation produced NaN rho (zero-variance x?)"
    p_perm = (1 + int((np.abs(rho_perm) >= abs(rho) - 1e-12).sum())) / (1 + n_perm)
    return {"rho": float(rho), "p_asymptotic": float(p_asym), "p_permutation": float(p_perm)}


def bootstrap_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int, seed: int
) -> dict[str, float | int | str]:
    """Percentile bootstrap CI for Spearman rho, resampling cells with replacement.

    Resamples where either variable is constant have undefined rho; they are excluded
    and counted in ``n_boot_valid``.
    """
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    xb, yb = x[idx], y[idx]
    rxb = rankdata(xb, axis=1)
    ryb = rankdata(yb, axis=1)
    rxc = rxb - rxb.mean(axis=1, keepdims=True)
    ryc = ryb - ryb.mean(axis=1, keepdims=True)
    x_ss = (rxc**2).sum(axis=1)
    y_ss = (ryc**2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_b = np.where(
            (x_ss > 0) & (y_ss > 0), (rxc * ryc).sum(axis=1) / np.sqrt(x_ss * y_ss), np.nan
        )
    valid = rho_b[~np.isnan(rho_b)]
    if valid.size == 0:
        raise ValueError("all bootstrap resamples degenerate (zero variance)")
    lo, hi = np.percentile(valid, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_boot": n_boot,
        "n_boot_valid": int(valid.size),
        "method": BOOTSTRAP_METHOD,
    }


def _residualize(v: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Residuals of ``v`` after OLS on ``z`` plus an intercept."""
    design = np.column_stack([np.ones(z.shape[0]), z])
    beta, *_ = np.linalg.lstsq(design, v, rcond=None)
    return v - design @ beta


def partial_spearman_with_permutation(
    x: np.ndarray, y: np.ndarray, controls: np.ndarray, n_perm: int, seed: int
) -> dict[str, float]:
    """Rank-based partial correlation of (x, y) given controls, with permutation p."""
    rx, ry = rankdata(x), rankdata(y)
    rz = np.column_stack([rankdata(controls[:, j]) for j in range(controls.shape[1])])
    ex = _residualize(rx, rz)
    ey = _residualize(ry, rz)
    ex_ss, ey_ss = float(ex @ ex), float(ey @ ey)
    assert ex_ss > 0 and ey_ss > 0, "degenerate residuals in partial correlation"
    rho = float((ex @ ey) / np.sqrt(ex_ss * ey_ss))
    rng = np.random.default_rng(seed)
    perms = rng.permuted(np.tile(ex, (n_perm, 1)), axis=1)
    rho_perm = _pearson_rows(perms, ey)
    assert not np.isnan(rho_perm).any()
    p_perm = (1 + int((np.abs(rho_perm) >= abs(rho) - 1e-12).sum())) / (1 + n_perm)
    return {"rho_partial": rho, "p_permutation": float(p_perm)}


def source_fe_spearman(
    by_source: dict[str, dict[str, np.ndarray]], n_perm: int, seed: int
) -> dict[str, float]:
    """Within-source demeaned rank correlation: rank within source, center, pool, Pearson.

    Permutation p shuffles emission ranks WITHIN each source block independently.
    """
    x_parts, y_parts, perm_parts = [], [], []
    rng = np.random.default_rng(seed)
    for s in sorted(by_source):
        rx = rankdata(by_source[s]["x"])
        ry = rankdata(by_source[s]["y"])
        rxc = rx - rx.mean()
        ryc = ry - ry.mean()
        x_parts.append(rxc)
        y_parts.append(ryc)
        perm_parts.append(rng.permuted(np.tile(rxc, (n_perm, 1)), axis=1))
    xp = np.concatenate(x_parts)
    yp = np.concatenate(y_parts)
    rho = float(_pearson_rows(xp[None, :], yp)[0])
    rho_perm = _pearson_rows(np.concatenate(perm_parts, axis=1), yp)
    assert not np.isnan(rho_perm).any()
    p_perm = (1 + int((np.abs(rho_perm) >= abs(rho) - 1e-12).sum())) / (1 + n_perm)
    return {"rho": rho, "p_permutation": float(p_perm)}


def _repo_relative(path: Path) -> str:
    """Repo-relative path string when the path sits under REPO_ROOT, else absolute."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def make_figure(
    sub: list[dict],
    stats: dict,
    *,
    x_field: str = X_FIELD,
    figure_stem: str = FIGURE_STEM,
    matrix_path: Path = MATRIX_PATH,
) -> dict[str, str]:
    """Software-engineer per-bystander scatter: marker DV vs sycophancy delta."""
    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    x = np.array([r[x_field] for r in sub])
    y = np.array([r[Y_FIELD] for r in sub])
    yerr = np.array([r["sycophancy_delta_se"] for r in sub])
    fig, ax = plt.subplots()
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color=paper_palette_role("primary"),
        ecolor=paper_palette_role("neutral"),
        elinewidth=0.8,
        alpha=0.85,
        linestyle="none",
    )
    ax.axhline(0.0, color=paper_palette_role("neutral"), linewidth=0.6, alpha=0.5)
    ax.set_xlabel(X_FIELD_AXIS_LABELS.get(x_field, x_field))
    ax.set_ylabel("Sycophancy leakage (trained − base)")
    title = (
        "Marker emission rate tracks sycophancy leakage on the software-engineer source"
        if x_field == "emission_rate"
        else f"Marker {x_field} vs sycophancy leakage on the software-engineer source"
    )
    set_title_subtitle(
        ax,
        title,
        subtitle="One point per bystander persona (n=23); y error bars are per-bystander SE",
        source=f"Source: {_repo_relative(matrix_path)}",
    )
    written = savefig_paper(fig, figure_stem, dir=FIGURE_DIR)
    plt.close(fig)

    meta_path = written["meta"]
    meta = json.loads(meta_path.read_text())
    meta["caption_stats"] = {
        "source": SE_SOURCE,
        "n": len(sub),
        "n_nonzero_emission": int((x > 0).sum()),
        "spearman_rho": stats["naive"]["rho"],
        "p_asymptotic": stats["naive"]["p_asymptotic"],
        "p_permutation": stats["naive"]["p_permutation"],
        "bootstrap_ci_95": [stats["bootstrap"]["ci_lo"], stats["bootstrap"]["ci_hi"]],
    }
    meta["x_field"] = x_field
    meta["y_field"] = Y_FIELD
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return {k: str(v) for k, v in written.items()}


def main(argv: list[str] | None = None) -> None:
    """Run the full emission-rate concordance analysis and persist stats + figure."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matrix-path", type=Path, default=MATRIX_PATH)
    parser.add_argument("--stats-dir", type=Path, default=STATS_DIR)
    parser.add_argument("--figure-stem", type=str, default=FIGURE_STEM)
    parser.add_argument(
        "--x-field",
        type=str,
        default=X_FIELD,
        choices=sorted(X_FIELD_DESCRIPTIONS),
        help="Marker-side DV column. Non-default choices write "
        "concordance_stats_<x_field>.json so the primary stats file is never clobbered.",
    )
    args = parser.parse_args(argv)
    matrix_path: Path = args.matrix_path
    stats_dir: Path = args.stats_dir
    x_field: str = args.x_field
    stats_path = stats_dir / (
        "concordance_stats.json" if x_field == X_FIELD else f"concordance_stats_{x_field}.json"
    )

    rows = load_matrix(matrix_path, x_field)
    matrix_payload = json.loads(matrix_path.read_text())
    sources = sorted({r["source"] for r in rows})
    by_source: dict[str, dict[str, np.ndarray]] = {}
    for s in sources:
        sub = [r for r in rows if r["source"] == s]
        assert len(sub) == 23, (s, len(sub))
        by_source[s] = {
            "rows": sub,
            "x": np.array([r[x_field] for r in sub], dtype=float),
            "y": np.array([r[Y_FIELD] for r in sub], dtype=float),
            "controls": np.array([[r[c] for c in CONTROL_FIELDS] for r in sub], dtype=float),
        }

    per_source: dict[str, dict] = {}
    for s in sources:
        x, y = by_source[s]["x"], by_source[s]["y"]
        n_nonzero = int((x > 0).sum())
        n_distinct = len(np.unique(x))
        informative = n_nonzero >= MIN_NONZERO_CELLS and n_distinct >= MIN_DISTINCT_VALUES
        entry: dict = {
            "n": int(x.shape[0]),
            "n_nonzero_emission": n_nonzero,
            "n_distinct_emission_values": n_distinct,
            "informative": informative,
            "informativeness_criterion": {
                "min_nonzero_cells": MIN_NONZERO_CELLS,
                "min_distinct_values": MIN_DISTINCT_VALUES,
            },
            "naive": spearman_with_permutation(x, y, N_PERM, PERM_SEED),
            "bootstrap": bootstrap_ci(x, y, N_BOOT, BOOT_SEED),
        }
        if not informative:
            entry["uninformative_reason"] = (
                f"emission floor: {n_nonzero}/{x.shape[0]} nonzero cells, "
                f"{n_distinct} distinct values"
            )
        per_source[s] = entry

    for s in sources:
        if not per_source[s]["informative"]:
            continue
        x, y = by_source[s]["x"], by_source[s]["y"]
        controls = by_source[s]["controls"]
        naive_rho = per_source[s]["naive"]["rho"]
        partials: dict[str, dict] = {}
        control_sets = {
            CONTROL_FIELDS[0]: controls[:, [0]],
            CONTROL_FIELDS[1]: controls[:, [1]],
            "joint": controls,
        }
        for name, z in control_sets.items():
            partials[name] = partial_spearman_with_permutation(x, y, z, N_PERM, PARTIAL_PERM_SEED)
        joint = partials["joint"]
        survives = (
            np.sign(joint["rho_partial"]) == np.sign(naive_rho) and joint["p_permutation"] < ALPHA
        )
        per_source[s]["partials"] = partials
        per_source[s]["partials_method"] = PARTIAL_METHOD
        per_source[s]["survives_partials"] = bool(survives)
        per_source[s]["survives_partials_rule"] = (
            "partial rho retains the naive sign AND joint-control permutation p < 0.05"
        )

    x_all = np.concatenate([by_source[s]["x"] for s in sources])
    y_all = np.concatenate([by_source[s]["y"] for s in sources])
    pooled = spearman_with_permutation(x_all, y_all, N_PERM, PERM_SEED)
    pooled["n"] = int(x_all.shape[0])
    pooled["caveat"] = POOLED_CAVEAT
    fe = source_fe_spearman(by_source, N_PERM, FE_PERM_SEED)
    fe["n"] = int(x_all.shape[0])
    fe["method"] = "rank within source, demean, pool all cells, Pearson on pooled ranks"

    se_stats = per_source[SE_SOURCE]
    figure_paths = make_figure(
        by_source[SE_SOURCE]["rows"],
        se_stats,
        x_field=x_field,
        figure_stem=args.figure_stem,
        matrix_path=matrix_path,
    )

    stats_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": OUTPUT_SCHEMA,
        "followup_label": "emission-rate-concordance",
        "git_commit_sha": _git_commit_sha(),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_matrix": {
            "path": _repo_relative(matrix_path),
            "schema": matrix_payload["schema"],
            "git_commit_sha": matrix_payload["git_commit_sha"],
            "n_rows": matrix_payload["n_rows"],
        },
        "env_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "seeds": {
            "bootstrap": BOOT_SEED,
            "permutation": PERM_SEED,
            "partial_permutation": PARTIAL_PERM_SEED,
            "fe_permutation": FE_PERM_SEED,
        },
        "n_boot": N_BOOT,
        "n_perm": N_PERM,
        "dv": {
            "x": f"{x_field} ({X_FIELD_DESCRIPTIONS[x_field]})",
            "y": f"{Y_FIELD} (frozen #411 per-bystander sycophancy leakage, trained - base)",
        },
        "per_source": per_source,
        "pooled_all_cells": pooled,
        "source_fe": fe,
        "figure": figure_paths,
    }
    stats_path.write_text(json.dumps(result, indent=2) + "\n")

    for s in sources:
        e = per_source[s]
        flag = "informative" if e["informative"] else "UNINFORMATIVE-BY-FLOOR"
        line = (
            f"{s}: rho={e['naive']['rho']:.3f} "
            f"p_asym={e['naive']['p_asymptotic']:.2e} p_perm={e['naive']['p_permutation']:.2e} "
            f"CI=[{e['bootstrap']['ci_lo']:.3f},{e['bootstrap']['ci_hi']:.3f}] "
            f"nonzero={e['n_nonzero_emission']}/{e['n']} [{flag}]"
        )
        if "survives_partials" in e:
            line += f" survives_partials={e['survives_partials']}"
        print(line)
    print(f"pooled: rho={pooled['rho']:.3f} p_perm={pooled['p_permutation']:.2e}")
    print(f"source_fe: rho={fe['rho']:.3f} p_perm={fe['p_permutation']:.2e}")
    print(f"stats -> {_repo_relative(stats_path)}")
    print(f"figure -> {figure_paths['png']}")


if __name__ == "__main__":
    main()

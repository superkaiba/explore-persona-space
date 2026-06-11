# ruff: noqa: RUF001, RUF003  # figure text uses ρ, −, → legitimately
"""Sycophancy-winning geometry controls for issue #480 (followup: syco-best-geometry-controls).

Asks whether each #480 marker-leakage read (in-band log-prob shift; firing-anchor
emission rate) carries signal about the frozen #411 sycophancy leakage BEYOND the
geometry that best predicts sycophancy itself. The registered joint rank-partial
controlled {layer-20 cosine to source, bystander base sycophancy rate}; this script
adds the two #509 sycophancy bake-off winning geometry cells
(end_of_system x L2 x cosine x centered, last_prompt x L7 x mmd x centered,
downloaded from the HF data repo at a pinned revision) and reports five covariate
sets per regime x testable source: registered, registered+cosL2, registered+mmdL7,
registered+both, and an early-cells-only diagnostic.

Validation gate (kill criterion): before any augmented row is emitted, the
recomputed registered-covariate naive Spearman and joint rank-partial must
reproduce the committed concordance-stats JSONs to 1e-6 (and, at the production
n_perm=100000, the committed permutation p-values exactly) — the same
assert-against-committed pattern as scripts/issue_480/plot_controlled_scatter.py.

Outputs:
    eval_results/issue_480/syco-best-geometry-controls/controls_stats.json
    figures/issue_480/syco-best-geometry-controls/geometry_controls_{inband,firing}.{png,pdf}

Analysis-only: no training, no generation, no model loads, no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[2]

FOLLOWUP_LABEL = "syco-best-geometry-controls"
OUTPUT_SCHEMA = "issue_480_syco_geometry_controls_v1"
STATS_DIR = REPO / "eval_results" / "issue_480" / FOLLOWUP_LABEL
STATS_PATH = STATS_DIR / "controls_stats.json"
FIG_DIR = REPO / "figures"
FIG_SUBDIR = f"issue_480/{FOLLOWUP_LABEL}"

EXPECTED_SCHEMA = "issue_480_marker_delta_matrix_v1"
EXPECTED_N_ROWS = 138
N_BYSTANDERS = 23

SOURCES = ("software_engineer", "assistant")
SOURCE_LABELS = {"software_engineer": "Software engineer", "assistant": "Assistant"}
Y_FIELD = "sycophancy_delta"
REGISTERED_COVS = ("cosine_l20_baseline", "bystander_base_rate")

HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REVISION = "1b6e20530b1c6d477a387c18d5a88554910e7df9"
HF_METRICS_PREFIX = "issue_509/syco_arm/bakeoff/metrics"
GEOMETRY_CELLS = {
    "cosine_l2_early": "end_of_system__layer2__cosine__centered.json",
    "mmd_l7_early": "last_prompt__layer7__mmd__centered.json",
}
GEOMETRY_LABELS = {
    "cosine_l2_early": "end-of-system L2 cosine (centered)",
    "mmd_l7_early": "last-prompt L7 MMD (centered)",
}

# Canonical SC1..SC24 persona order — copied verbatim from
# scripts/issue509_bystander_bootstrap.py _SYCO_PERSONA_ORDER (alphabetical over the
# 24-persona panel). The reconstructed mapping from the #480 matrices is asserted
# equal to this, so a silent panel drift fails loud instead of mis-joining.
SYCO_PERSONA_ORDER = (
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

N_PERM_DEFAULT = 100_000
NAIVE_PERM_SEED = 4801  # matches the committed runs' "permutation" seed
PARTIAL_PERM_SEED = 4802  # matches the committed runs' "partial_permutation" seed
RHO_TOLERANCE = 1e-6

PARTIAL_METHOD = (
    "rank-transform all variables with average ties (Spearman partial), OLS-residualize "
    "x-ranks and y-ranks on covariate ranks + intercept, Pearson on residuals; permutation "
    "p from permuting the x-rank residuals (two-sided, +1/(n+1) correction)"
)

COVARIATE_SETS: dict[str, tuple[str, ...]] = {
    "registered": REGISTERED_COVS,
    "registered_plus_cosine_l2": (*REGISTERED_COVS, "cosine_l2_early"),
    "registered_plus_mmd_l7": (*REGISTERED_COVS, "mmd_l7_early"),
    "registered_plus_both": (*REGISTERED_COVS, "cosine_l2_early", "mmd_l7_early"),
    "early_cells_only": ("cosine_l2_early", "mmd_l7_early"),
}
AUGMENTED_SET_FOR_FIGURE = "registered_plus_both"

REGIMES: dict[str, dict[str, str]] = {
    "inband": {
        "matrix": "eval_results/issue_480/inband-logprob-concordance/marker_delta_matrix.json",
        "committed_stats": (
            "eval_results/issue_480/inband-logprob-concordance/concordance_stats_marker_delta.json"
        ),
        "x_field": "marker_delta",
        "x_description": "on-policy log P(marker) trained − base per bystander (nats)",
        "x_residual_label": "marker log-prob shift",
        "figure_name": "geometry_controls_inband",
        "title": (
            "Sub-emission log-prob read: concordance under sycophancy-winning geometry controls"
        ),
    },
    "firing": {
        "matrix": "eval_results/issue_480/band-stopped-anchor-rerun/marker_delta_matrix.json",
        "committed_stats": (
            "eval_results/issue_480/band-stopped-anchor-rerun/concordance_stats.json"
        ),
        "x_field": "emission_rate",
        "x_description": "on-policy marker emission rate per bystander",
        "x_residual_label": "marker emission rate",
        "figure_name": "geometry_controls_firing",
        "title": (
            "Firing-anchor emission read: concordance under sycophancy-winning geometry controls"
        ),
    },
}


def _git_commit_sha() -> str:
    """Return the full git commit SHA of the worktree HEAD."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        timeout=10,
    )
    return out.stdout.strip()


def _sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    """Repo-relative path string when the path sits under REPO, else absolute."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


# ── Statistics — verbatim recipe from scripts/issue480_emission_rate_concordance.py ──


def _pearson_rows(xm: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation between each row of ``xm`` and the vector ``y``."""
    assert xm.ndim == 2 and y.ndim == 1 and xm.shape[1] == y.shape[0], (xm.shape, y.shape)
    xc = xm - xm.mean(axis=1, keepdims=True)
    yc = y - y.mean()
    y_ss = float(yc @ yc)
    assert y_ss > 0, "y has zero variance"
    x_ss = (xc**2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x_ss > 0, (xc @ yc) / np.sqrt(x_ss * y_ss), np.nan)


def _residualize(v: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Residuals of ``v`` after OLS on ``z`` plus an intercept."""
    design = np.column_stack([np.ones(z.shape[0]), z])
    beta, *_ = np.linalg.lstsq(design, v, rcond=None)
    return v - design @ beta


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


def rank_partial_residuals(
    x: np.ndarray, y: np.ndarray, controls: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Registered partial recipe: residualized x/y ranks and their Pearson rho."""
    rx, ry = rankdata(x), rankdata(y)
    rz = np.column_stack([rankdata(controls[:, j]) for j in range(controls.shape[1])])
    ex = _residualize(rx, rz)
    ey = _residualize(ry, rz)
    ex_ss, ey_ss = float(ex @ ex), float(ey @ ey)
    assert ex_ss > 0 and ey_ss > 0, "degenerate residuals in partial correlation"
    rho = float((ex @ ey) / np.sqrt(ex_ss * ey_ss))
    return ex, ey, rho


def partial_spearman_with_permutation(
    x: np.ndarray, y: np.ndarray, controls: np.ndarray, n_perm: int, seed: int
) -> dict[str, Any]:
    """Rank-based partial correlation of (x, y) given controls, with permutation p.

    Returns the residual vectors too so figures plot exactly the inference residuals.
    """
    ex, ey, rho = rank_partial_residuals(x, y, controls)
    rng = np.random.default_rng(seed)
    perms = rng.permuted(np.tile(ex, (n_perm, 1)), axis=1)
    rho_perm = _pearson_rows(perms, ey)
    assert not np.isnan(rho_perm).any()
    p_perm = (1 + int((np.abs(rho_perm) >= abs(rho) - 1e-12).sum())) / (1 + n_perm)
    return {"rho_partial": rho, "p_permutation": float(p_perm), "res_x": ex, "res_y": ey}


# ── Data loading ──────────────────────────────────────────────────────────


def load_matrix(matrix_path: Path, x_field: str) -> dict[str, Any]:
    """Load and validate one 138-cell marker matrix; fail loud on any mismatch."""
    required = ("source", "bystander", x_field, Y_FIELD, "sycophancy_delta_se", *REGISTERED_COVS)
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
    return payload


def build_cid_map(rows: list[dict]) -> dict[str, str]:
    """Reconstruct the persona → SC{i} mapping from the matrix panel; assert canonical."""
    panel = sorted({r["source"] for r in rows} | {r["bystander"] for r in rows})
    if len(panel) != 24:
        raise ValueError(f"expected a 24-persona panel, got {len(panel)}: {panel}")
    if tuple(panel) != SYCO_PERSONA_ORDER:
        raise ValueError(
            "panel drift: sorted(sources ∪ bystanders) does not match the canonical "
            f"issue-509 SC ordering.\npanel={panel}\ncanonical={list(SYCO_PERSONA_ORDER)}"
        )
    return {persona: f"SC{i}" for i, persona in enumerate(panel, start=1)}


def load_geometry_cells() -> dict[str, dict[str, Any]]:
    """Download the two #509 bake-off winning cells at the pinned HF revision."""
    from huggingface_hub import hf_hub_download

    cells: dict[str, dict[str, Any]] = {}
    for key, fname in GEOMETRY_CELLS.items():
        local = hf_hub_download(
            HF_REPO,
            f"{HF_METRICS_PREFIX}/{fname}",
            repo_type="dataset",
            revision=HF_REVISION,
        )
        payload = json.loads(Path(local).read_text())
        matrix = payload.get("matrix")
        if matrix is None:
            raise ValueError(f"{fname}: matrix is null — wrong cell name?")
        cells[key] = {
            "matrix": matrix,
            "provenance": {
                "filename": fname,
                "hf_repo": HF_REPO,
                "hf_revision": HF_REVISION,
                "path_in_repo": f"{HF_METRICS_PREFIX}/{fname}",
                "payload_git_sha": payload.get("git_sha"),
                "extraction_point": payload.get("extraction_point"),
                "layer": payload.get("layer"),
                "metric": payload.get("metric"),
                "variant": payload.get("variant"),
                "sha256_local": _sha256(Path(local)),
            },
        }
    return cells


def attach_geometry(
    rows: list[dict], cid_map: dict[str, str], cells: dict[str, dict[str, Any]]
) -> None:
    """Attach each geometry cell's (source, bystander) distance to every matrix row.

    Asserts all 138 source→bystander pairs resolve to a finite value in BOTH cells.
    """
    for key, cell in cells.items():
        matrix = cell["matrix"]
        n_filled = 0
        for row in rows:
            d = matrix[cid_map[row["source"]]].get(cid_map[row["bystander"]])
            if d is None or not np.isfinite(float(d)):
                raise ValueError(
                    f"geometry cell {key}: null/non-finite distance for "
                    f"({row['source']}, {row['bystander']})"
                )
            row[key] = float(d)
            n_filled += 1
        assert n_filled == EXPECTED_N_ROWS, (key, n_filled)


def by_source_arrays(rows: list[dict], x_field: str) -> dict[str, dict[str, Any]]:
    """Per testable source: x, y and all covariate columns as arrays over 23 bystanders."""
    out: dict[str, dict[str, Any]] = {}
    all_covs = (*REGISTERED_COVS, *GEOMETRY_CELLS.keys())
    for s in SOURCES:
        sub = sorted((r for r in rows if r["source"] == s), key=lambda r: r["bystander"])
        assert len(sub) == N_BYSTANDERS, (s, len(sub))
        out[s] = {
            "rows": sub,
            "bystanders": [r["bystander"] for r in sub],
            "x": np.array([r[x_field] for r in sub], dtype=float),
            "y": np.array([r[Y_FIELD] for r in sub], dtype=float),
            "covs": {c: np.array([r[c] for r in sub], dtype=float) for c in all_covs},
        }
    return out


def _controls_for(d: dict[str, Any], set_name: str) -> np.ndarray:
    """Covariate design columns for one covariate set, in the declared field order."""
    return np.column_stack([d["covs"][c] for c in COVARIATE_SETS[set_name]])


# ── Validation gate ───────────────────────────────────────────────────────


def validate_against_committed(
    regime: str,
    source: str,
    naive: dict[str, float],
    registered: dict[str, Any],
    committed_per_source: dict[str, Any],
    n_perm: int,
) -> dict[str, Any]:
    """Assert the recomputed registered-covariate stats reproduce the committed JSONs.

    Rhos must match to 1e-6 always; at the production n_perm=100000 the permutation
    p-values must reproduce the committed values exactly (same seeds, same scheme).
    Crashes loudly on any mismatch (kill criterion).
    """
    ref = committed_per_source[source]
    ref_naive_rho = ref["naive"]["rho"]
    ref_joint = ref["partials"]["joint"]
    checks = {
        "naive_rho": (naive["rho"], ref_naive_rho),
        "joint_rho_partial": (registered["rho_partial"], ref_joint["rho_partial"]),
    }
    for name, (got, want) in checks.items():
        if abs(got - want) >= RHO_TOLERANCE:
            raise AssertionError(
                f"VALIDATION GATE FAILED ({regime}, {source}, {name}): "
                f"recomputed {got!r} != committed {want!r} (tol {RHO_TOLERANCE})"
            )
    p_checked = n_perm == 100_000
    if p_checked:
        p_checks = {
            "naive_p_permutation": (naive["p_permutation"], ref["naive"]["p_permutation"]),
            "joint_p_permutation": (registered["p_permutation"], ref_joint["p_permutation"]),
        }
        for name, (got, want) in p_checks.items():
            if abs(got - want) >= 1e-12:
                raise AssertionError(
                    f"VALIDATION GATE FAILED ({regime}, {source}, {name}): "
                    f"recomputed {got!r} != committed {want!r} at n_perm=100000 "
                    "(seed/scheme mismatch?)"
                )
    return {
        "passed": True,
        "rho_tolerance": RHO_TOLERANCE,
        "p_reproduction_checked": p_checked,
        "committed_naive_rho": ref_naive_rho,
        "committed_joint_rho_partial": ref_joint["rho_partial"],
    }


# ── Redundancy diagnostics ────────────────────────────────────────────────


def redundancy_block(
    inband_by_source: dict[str, dict[str, Any]],
    firing_by_source: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Per testable source: Spearman among the two marker reads and each covariate."""
    out: dict[str, Any] = {}
    all_covs = (*REGISTERED_COVS, *GEOMETRY_CELLS.keys())
    for s in SOURCES:
        ib, fr = inband_by_source[s], firing_by_source[s]
        assert ib["bystanders"] == fr["bystanders"], f"bystander join mismatch for {s}"
        for c in REGISTERED_COVS:
            assert np.allclose(ib["covs"][c], fr["covs"][c]), (
                f"registered covariate {c} differs between matrices for {s}"
            )
        lp, em = ib["x"], fr["x"]
        rho, p = spearmanr(lp, em)
        entry: dict[str, Any] = {
            "lp_inband_vs_em_firing": {"rho": float(rho), "p_asymptotic": float(p)},
            "lp_inband_vs_covariate": {},
            "em_firing_vs_covariate": {},
        }
        for c in all_covs:
            z = ib["covs"][c]
            r1, p1 = spearmanr(lp, z)
            r2, p2 = spearmanr(em, z)
            entry["lp_inband_vs_covariate"][c] = {"rho": float(r1), "p_asymptotic": float(p1)}
            entry["em_firing_vs_covariate"][c] = {"rho": float(r2), "p_asymptotic": float(p2)}
        entry["n"] = N_BYSTANDERS
        out[s] = entry
    return out


# ── Figure ────────────────────────────────────────────────────────────────


def fmt_p(p: float) -> str:
    """Permutation p for panel titles: 3 decimals, floored at < 0.001."""
    return f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"


def make_figure(
    regime: str,
    cfg: dict[str, str],
    results_per_source: dict[str, dict[str, Any]],
    n_perm: int,
) -> dict[str, str]:
    """One figure per regime: per-source registered vs augmented residual scatters.

    Rows = the two testable sources; col 0 = registered-control residuals,
    col 1 = augmented (registered + both early-geometry cells) residuals.
    """
    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    point_color = paper_palette_role("primary")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
    panel_specs = (
        ("registered", "registered controls\n(L20 cosine + base rate)"),
        (AUGMENTED_SET_FOR_FIGURE, "augmented controls\n(+ both sycophancy-winning cells)"),
    )
    for row_i, source in enumerate(SOURCES):
        res = results_per_source[source]
        for col_i, (set_name, set_label) in enumerate(panel_specs):
            part = res["partials_full"][set_name]
            ax = axes[row_i, col_i]
            ax.scatter(part["res_x"], part["res_y"], color=point_color, alpha=0.75, s=28)
            slope, intercept = np.polyfit(part["res_x"], part["res_y"], 1)
            xs = np.linspace(part["res_x"].min(), part["res_x"].max(), 50)
            ax.plot(xs, slope * xs + intercept, color=point_color, linewidth=1.0, alpha=0.5)
            ax.set_title(
                f"{SOURCE_LABELS[source]} — {set_label.splitlines()[0]}\n"
                f"ρ = {part['rho_partial']:.2f}, perm {fmt_p(part['p_permutation'])}",
                loc="left",
                fontsize=10,
            )
            kind = "registered" if set_name == "registered" else "augmented"
            if row_i == 1:
                ax.set_xlabel(
                    f"{cfg['x_residual_label']} (rank residual, {kind} controls)", fontsize=9
                )
            ax.set_ylabel(f"sycophancy leakage\n(rank residual, {kind} controls)", fontsize=9)

    fig.text(
        0.01,
        0.985,
        cfg["title"],
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.01,
        0.955,
        "Left: rank residuals after the registered controls (layer-20 cosine to source + "
        "bystander base rate). Right: after additionally partialling the two #509 "
        "sycophancy-winning geometry cells (end-of-system L2 cosine, last-prompt L7 MMD). "
        "n = 23 bystanders per panel, single seed.",
        ha="left",
        va="top",
        fontsize=9,
        color="#5A5A5A",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    written = savefig_paper(fig, f"{FIG_SUBDIR}/{cfg['figure_name']}", dir=FIG_DIR)
    plt.close(fig)

    meta_path = Path(written["meta"])
    meta = json.loads(meta_path.read_text())
    meta["caption_stats"] = {
        "regime": regime,
        "x_field": cfg["x_field"],
        "n_per_panel": N_BYSTANDERS,
        "n_perm": n_perm,
        "per_source": {
            s: {
                set_name: {
                    "rho_partial": results_per_source[s]["partials_full"][set_name]["rho_partial"],
                    "p_permutation": results_per_source[s]["partials_full"][set_name][
                        "p_permutation"
                    ],
                }
                for set_name, _ in panel_specs
            }
            for s in SOURCES
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    # Repo-relative so the committed JSON never references ephemeral worktree paths.
    return {k: _repo_relative(Path(v)) for k, v in written.items()}


# ── Main ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Run the geometry-controls analysis end to end and persist stats + figures."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=N_PERM_DEFAULT,
        help="Permutation reps (default 100000 = the registered production inference; "
        "smoke runs use 1000).",
    )
    args = parser.parse_args(argv)
    n_perm: int = args.n_perm

    geometry = load_geometry_cells()

    regime_payloads: dict[str, dict[str, Any]] = {}
    regime_by_source: dict[str, dict[str, dict[str, Any]]] = {}
    committed: dict[str, dict[str, Any]] = {}
    cid_note = ""
    for regime, cfg in REGIMES.items():
        payload = load_matrix(REPO / cfg["matrix"], cfg["x_field"])
        cid_map = build_cid_map(payload["rows"])
        cid_note = (
            "SC{i} = i-th persona of sorted(sources ∪ bystanders) of the #480 matrices "
            "(alphabetical, i from 1); asserted equal to the canonical _SYCO_PERSONA_ORDER "
            "from scripts/issue509_bystander_bootstrap.py"
        )
        attach_geometry(payload["rows"], cid_map, geometry)
        regime_payloads[regime] = payload
        regime_by_source[regime] = by_source_arrays(payload["rows"], cfg["x_field"])
        committed[regime] = json.loads((REPO / cfg["committed_stats"]).read_text())["per_source"]

    results: dict[str, Any] = {}
    gate: dict[str, Any] = {}
    for regime in REGIMES:
        per_source: dict[str, Any] = {}
        for source in SOURCES:
            d = regime_by_source[regime][source]
            x, y = d["x"], d["y"]
            naive = spearman_with_permutation(x, y, n_perm, NAIVE_PERM_SEED)
            registered = partial_spearman_with_permutation(
                x, y, _controls_for(d, "registered"), n_perm, PARTIAL_PERM_SEED
            )
            # Kill-criterion gate BEFORE any augmented covariate set is computed.
            gate[f"{regime}.{source}"] = validate_against_committed(
                regime, source, naive, registered, committed[regime], n_perm
            )
            partials_full = {"registered": registered}
            for set_name in COVARIATE_SETS:
                if set_name == "registered":
                    continue
                partials_full[set_name] = partial_spearman_with_permutation(
                    x, y, _controls_for(d, set_name), n_perm, PARTIAL_PERM_SEED
                )
            per_source[source] = {
                "n": int(x.shape[0]),
                "naive": naive,
                "partials_full": partials_full,  # carries residual vectors for the figure
            }
        results[regime] = per_source

    figures: dict[str, dict[str, str]] = {}
    for regime, cfg in REGIMES.items():
        figures[regime] = make_figure(regime, cfg, results[regime], n_perm)

    redundancy = redundancy_block(regime_by_source["inband"], regime_by_source["firing"])

    # JSON-safe view: drop the residual vectors, attach covariate lists.
    regimes_out: dict[str, Any] = {}
    for regime, cfg in REGIMES.items():
        per_source_out: dict[str, Any] = {}
        for source in SOURCES:
            e = results[regime][source]
            per_source_out[source] = {
                "n": e["n"],
                "naive": e["naive"],
                "partials": {
                    set_name: {
                        "rho_partial": p["rho_partial"],
                        "p_permutation": p["p_permutation"],
                        "covariates": list(COVARIATE_SETS[set_name]),
                    }
                    for set_name, p in e["partials_full"].items()
                },
            }
        regimes_out[regime] = {
            "x_field": cfg["x_field"],
            "x_description": cfg["x_description"],
            "input_matrix": _repo_relative(REPO / cfg["matrix"]),
            "committed_reference_stats": cfg["committed_stats"],
            "per_source": per_source_out,
        }

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "schema": OUTPUT_SCHEMA,
        "followup_label": FOLLOWUP_LABEL,
        "git_commit_sha": _git_commit_sha(),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "n_perm": n_perm,
        "seeds": {
            "naive_permutation": NAIVE_PERM_SEED,
            "partial_permutation": PARTIAL_PERM_SEED,
        },
        "partials_method": PARTIAL_METHOD,
        "covariate_sets": {k: list(v) for k, v in COVARIATE_SETS.items()},
        "covariate_provenance": {
            "registered": (
                "per-row fields of each #480 marker_delta_matrix.json: layer-20 cosine to "
                "source (cosine_l20_baseline) + bystander base sycophancy rate "
                "(bystander_base_rate)"
            ),
            "geometry_cells": {k: v["provenance"] for k, v in geometry.items()},
            "geometry_cell_labels": GEOMETRY_LABELS,
            "cid_mapping_note": cid_note,
        },
        "input_matrices": {
            regime: {
                "path": _repo_relative(REPO / cfg["matrix"]),
                "sha256": _sha256(REPO / cfg["matrix"]),
                "matrix_git_commit_sha": regime_payloads[regime]["git_commit_sha"],
                "n_rows": regime_payloads[regime]["n_rows"],
            }
            for regime, cfg in REGIMES.items()
        },
        "validation_gate": gate,
        "dv": {
            "y": f"{Y_FIELD} (frozen #411 per-bystander sycophancy leakage, trained − base)",
        },
        "regimes": regimes_out,
        "redundancy": redundancy,
        "figures": figures,
    }
    STATS_PATH.write_text(json.dumps(out, indent=2) + "\n")

    for regime in REGIMES:
        for source in SOURCES:
            e = regimes_out[regime]["per_source"][source]
            parts = " ".join(
                f"{name}={p['rho_partial']:+.3f}(p={p['p_permutation']:.4g})"
                for name, p in e["partials"].items()
            )
            print(
                f"{regime}/{source}: naive={e['naive']['rho']:+.3f}"
                f"(p={e['naive']['p_permutation']:.4g}) {parts}"
            )
    for source in SOURCES:
        r = redundancy[source]["lp_inband_vs_em_firing"]["rho"]
        print(f"redundancy {source}: spearman(lp_inband, em_firing) = {r:+.3f}")
    print(f"stats -> {_repo_relative(STATS_PATH)}")
    for regime, paths in figures.items():
        print(f"figure ({regime}) -> {paths['png']}")


if __name__ == "__main__":
    main()

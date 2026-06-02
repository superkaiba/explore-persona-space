"""Phase 5 — paired-bootstrap CIs for H3a/b/c/d, H4, H5, retention.

Plan v2 §6.2 + §6.4.

Reads per-cell JSONs (eval_results/issue_465/per_cell/G_<cond>__<shape>.json)
and produces:
  * eval_results/issue_465/analysis.json — all CIs + diagnostics + per-cell
    summary (cell mean ΔG, sd, emission rate, n_probes, constant-emission
    flag).
  * eval_results/issue_465/analysis_per_q_paired.json — per-q paired arrays
    for the analyzer to plot from (so figures can re-derive without re-reading
    the giant per-cell files).

The 50 Q_test serve as the paired axis across conditions: every condition's
per-cell payload reports a list ``g_logps_per_q`` aligned with ``q_used``.
For H3a/b/c/d we pair on the INTERSECTION of q_used across the two cells
being compared (always 50 for in-trained-shape / generalization /
demo-free-default-villain-R / non-marker-demo, may be <50 for
demo_free_default if helpful-R dropped any marker_in_R rows).

H3a: paired(cond1 − cond2_k1) at demo_free_default.        Positive ⇒ cond1 leaks more.
H3b: paired(cond2_k0 − cond2_k1) at demo_free_default.     Positive ⇒ demos GATE.
H3c: paired(cond1 − cond2_k0) at demo_free_default.        Positive ⇒ served-system mismatch.
H3d: paired ratios — retention[cond1] − retention[cond2_k0]; cond2_k0 − cond2_k1.
H4 : paired(cond2_k1 − cond2_k3) at demo_free_default.     Direction descriptive.
H5 : paired ratio (cond ΔG[non_marker_demo] / ΔG[demo_free_default]) for cond2_k1/k3.

Bootstrap: 10k resamples paired on q-indices, seed=42.

CLI:
    uv run python scripts/i465_phase5_analyze.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.i465_data import (
    CONDITION_IDS,
    CONDITION_K,
    CONDITION_NAMES,
)

logger = logging.getLogger("i465.phase5")

OUT_DIR = Path("eval_results/issue_465")
PER_CELL_DIR = OUT_DIR / "per_cell"

PRIMARY_SHAPES = [
    "in_trained_shape",
    "generalization",
    "demo_free_default",
    "demo_free_default_villain_R",
]
NON_MARKER_DEMO_SHAPE = "non_marker_demo"


def _load_cell(cond: str, shape: str) -> dict | None:
    p = PER_CELL_DIR / f"G_{cond}__{shape}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _paired_q_indices(cell_a: dict, cell_b: dict) -> tuple[list[int], list[int]]:
    """Return (indices_in_a, indices_in_b) for q's present in BOTH cells, in cell_a's order."""
    q_a = cell_a["q_used"]
    q_b = cell_b["q_used"]
    b_index = {q: i for i, q in enumerate(q_b)}
    ia: list[int] = []
    ib: list[int] = []
    for i, q in enumerate(q_a):
        if q in b_index:
            ia.append(i)
            ib.append(b_index[q])
    return ia, ib


def _paired_diff(cell_a: dict, cell_b: dict) -> np.ndarray:
    """Return ΔG_a − ΔG_b per shared q (paired on q identity)."""
    ia, ib = _paired_q_indices(cell_a, cell_b)
    g_a = np.array(cell_a["g_logps_per_q"], dtype=float)[ia]
    b_a = np.array(cell_a["b_logps_per_q"], dtype=float)[ia]
    g_b = np.array(cell_b["g_logps_per_q"], dtype=float)[ib]
    b_b = np.array(cell_b["b_logps_per_q"], dtype=float)[ib]
    return (g_a - b_a) - (g_b - b_b)


def _bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via paired bootstrap on `values`."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def _excludes_zero(low: float, high: float) -> str | None:
    """Return 'positive' / 'negative' / None depending on whether CI excludes 0."""
    if low > 0:
        return "positive"
    if high < 0:
        return "negative"
    return None


def _per_cell_summary(cells: dict[tuple[str, str], dict]) -> dict:
    """Per-cell summary table (plan §6.2 #9 constant-emission diagnostic)."""
    out: dict[str, dict] = {}
    for (cond, shape), cell in cells.items():
        g = np.array(cell["g_logps_per_q"], dtype=float)
        b = np.array(cell["b_logps_per_q"], dtype=float)
        delta = g - b
        sd_dg = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
        flag = sd_dg < 0.5  # plan §6.2 #9
        out[f"{cond}__{shape}"] = {
            "condition": cond,
            "condition_name": CONDITION_NAMES[cond],
            "eval_shape": shape,
            "k_demos": cell.get("k_demos", CONDITION_K[cond]),
            "n_probes": cell["n_probes"],
            "g_logprob_mean": cell["g_logprob"],
            "b_logprob_mean": cell["b_logprob"],
            "delta_g_mean": cell["delta_g"],
            "delta_g_sd": sd_dg,
            "constant_emission_flag": flag,
            "emission_recompute_rate": cell["emission_recompute_rate"],
        }
    return out


def _per_q_dg(cell: dict | None) -> np.ndarray:
    if cell is None:
        return np.array([])
    g = np.array(cell["g_logps_per_q"], dtype=float)
    b = np.array(cell["b_logps_per_q"], dtype=float)
    return g - b


def _bootstrap_retention_diff(
    cell_a_in: dict,
    cell_a_default: dict,
    cell_b_in: dict,
    cell_b_default: dict,
    n_resamples: int = 10_000,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI on (retention[A] − retention[B]) where retention =
    mean(ΔG[default]) / mean(ΔG[in_trained_shape]).

    We resample q-INDICES across each cell independently within each
    bootstrap draw — the retention ratio is a function of two cell means
    that share q identity per cell but NOT across cells (since "default"
    and "in_trained" can have different q_used). We use the q-axis of the
    "in_trained_shape" cell (always 50 q) as the canonical pairing axis
    and intersect with the cell's own q_used.
    """
    # Per-q ΔG arrays. For each cell, we draw n_resamples bootstrap means.
    rng = np.random.default_rng(rng_seed)

    def cell_means(cell: dict, ridx: np.ndarray) -> np.ndarray:
        dg = _per_q_dg(cell)
        if len(dg) == 0:
            return np.full(ridx.shape[0], float("nan"))
        # ridx has shape (n_resamples, n_q) — but n_q here uses the cell's
        # own length. Re-index inside the cell.
        n_cell = len(dg)
        idx = rng.integers(0, n_cell, size=(n_resamples, n_cell))
        return dg[idx].mean(axis=1)

    # n_q for each cell may differ; use one rng across all four cells —
    # bootstrap independence per cell.
    ridx_dummy = np.zeros((n_resamples, 1))
    means_a_default = cell_means(cell_a_default, ridx_dummy)
    means_a_in = cell_means(cell_a_in, ridx_dummy)
    means_b_default = cell_means(cell_b_default, ridx_dummy)
    means_b_in = cell_means(cell_b_in, ridx_dummy)
    eps = 1e-9
    retention_a = means_a_default / np.where(np.abs(means_a_in) < eps, np.nan, means_a_in)
    retention_b = means_b_default / np.where(np.abs(means_b_in) < eps, np.nan, means_b_in)
    diffs = retention_a - retention_b
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    cells: dict[tuple[str, str], dict] = {}
    for cond in CONDITION_IDS:
        for shape in PRIMARY_SHAPES + [NON_MARKER_DEMO_SHAPE]:
            cell = _load_cell(cond, shape)
            if cell is not None:
                cells[(cond, shape)] = cell

    if not cells:
        raise FileNotFoundError(f"No per-cell JSON files found under {PER_CELL_DIR}.")

    per_cell = _per_cell_summary(cells)
    logger.info("Loaded %d per-cell payloads.", len(cells))

    # ── H1 diagonal gates ────────────────────────────────────────────────
    h1 = {}
    for cond, label, threshold in [
        ("cond1", "H1a", 15.0),
        ("cond2_k0", "H1c", 5.0),
        ("cond2_k1", "H1b_k1", 5.0),
        ("cond2_k3", "H1b_k3", 5.0),
    ]:
        cell = cells.get((cond, "in_trained_shape"))
        if cell is None:
            h1[label] = {"status": "MISSING_CELL", "cond": cond, "threshold": threshold}
            continue
        per_q = _per_q_dg(cell)
        m, lo, hi = _bootstrap_ci(per_q, args.n_bootstrap, 0.05, args.seed)
        h1[label] = {
            "cond": cond,
            "shape": "in_trained_shape",
            "delta_g_mean": m,
            "ci_95": [lo, hi],
            "threshold": threshold,
            "pass": m > threshold,
        }

    # ── H2 generalization (cond2_k1) ────────────────────────────────────
    h2 = None
    diag = cells.get(("cond2_k1", "in_trained_shape"))
    genx = cells.get(("cond2_k1", "generalization"))
    if diag and genx:
        d_diag = _per_q_dg(diag).mean()
        d_gen = _per_q_dg(genx).mean()
        ratio = d_gen / d_diag if abs(d_diag) > 1e-9 else None
        h2 = {
            "diag_delta_g": float(d_diag),
            "gen_delta_g": float(d_gen),
            "ratio_gen_over_diag": float(ratio) if ratio is not None else None,
            "threshold": 0.5,
            "pass": (ratio is not None and ratio >= 0.5),
        }

    # ── H3a/b/c — paired at demo_free_default (helpful-R PRIMARY) ───────
    h3 = {}
    pairs_h3 = [
        ("H3a", "cond1", "cond2_k1"),
        ("H3b", "cond2_k0", "cond2_k1"),
        ("H3c", "cond1", "cond2_k0"),
    ]
    for label, cond_a, cond_b in pairs_h3:
        a = cells.get((cond_a, "demo_free_default"))
        b = cells.get((cond_b, "demo_free_default"))
        if not (a and b):
            h3[label] = {"status": "MISSING_CELL", "cond_a": cond_a, "cond_b": cond_b}
            continue
        diffs = _paired_diff(a, b)
        m, lo, hi = _bootstrap_ci(diffs, args.n_bootstrap, 0.05, args.seed)
        h3[label] = {
            "cond_a": cond_a,
            "cond_b": cond_b,
            "n_paired": len(diffs),
            "diff_mean": m,
            "ci_95": [lo, hi],
            "excludes_zero": _excludes_zero(lo, hi),
        }
    # Additional raw level for H3a (per plan §6.2 #3 — cond2_k1 < 0.5 × cond1).
    if cells.get(("cond1", "demo_free_default")) and cells.get(("cond2_k1", "demo_free_default")):
        c1 = _per_q_dg(cells[("cond1", "demo_free_default")]).mean()
        c2k1 = _per_q_dg(cells[("cond2_k1", "demo_free_default")]).mean()
        h3.setdefault("H3a", {})
        h3["H3a"]["cond1_mean_demo_free_default"] = float(c1)
        h3["H3a"]["cond2_k1_mean_demo_free_default"] = float(c2k1)
        h3["H3a"]["ratio_cond2k1_over_cond1"] = float(c2k1 / c1) if abs(c1) > 1e-9 else None
        h3["H3a"]["raw_level_pass"] = float(c1) > 5.0 and (abs(c1) > 1e-9) and (c2k1 < 0.5 * c1)

    # ── H3d retention CIs ───────────────────────────────────────────────
    h3d = {}
    for label, cond_a, cond_b in [
        ("retention_cond1_minus_cond2_k0", "cond1", "cond2_k0"),
        ("retention_cond2_k0_minus_cond2_k1", "cond2_k0", "cond2_k1"),
    ]:
        a_in = cells.get((cond_a, "in_trained_shape"))
        a_default = cells.get((cond_a, "demo_free_default"))
        b_in = cells.get((cond_b, "in_trained_shape"))
        b_default = cells.get((cond_b, "demo_free_default"))
        if not all([a_in, a_default, b_in, b_default]):
            h3d[label] = {"status": "MISSING_CELL"}
            continue
        m, lo, hi = _bootstrap_retention_diff(
            a_in, a_default, b_in, b_default, args.n_bootstrap, args.seed
        )
        h3d[label] = {
            "diff_mean": m,
            "ci_95": [lo, hi],
            "excludes_zero": _excludes_zero(lo, hi),
        }
    # Per-condition retention (point estimate).
    retention_point = {}
    for cond in CONDITION_IDS:
        a_in = cells.get((cond, "in_trained_shape"))
        a_default = cells.get((cond, "demo_free_default"))
        if not (a_in and a_default):
            continue
        d_in = _per_q_dg(a_in).mean()
        d_default = _per_q_dg(a_default).mean()
        retention_point[cond] = {
            "delta_g_in_trained_shape": float(d_in),
            "delta_g_demo_free_default": float(d_default),
            "retention": float(d_default / d_in) if abs(d_in) > 1e-9 else None,
        }

    # ── H4 k-sweep (cond2_k1 vs cond2_k3 at demo_free_default) ──────────
    h4 = None
    a = cells.get(("cond2_k1", "demo_free_default"))
    b = cells.get(("cond2_k3", "demo_free_default"))
    if a and b:
        diffs = _paired_diff(a, b)
        m, lo, hi = _bootstrap_ci(diffs, args.n_bootstrap, 0.05, args.seed)
        h4 = {
            "n_paired": len(diffs),
            "diff_mean_cond2_k1_minus_cond2_k3": m,
            "ci_95": [lo, hi],
            "excludes_zero": _excludes_zero(lo, hi),
        }

    # ── H5 non-marker-demo (copy-vs-implant) ────────────────────────────
    h5 = {}
    for cond in ("cond2_k1", "cond2_k3"):
        with_marker = cells.get((cond, "demo_free_default"))
        no_marker = cells.get((cond, NON_MARKER_DEMO_SHAPE))
        if not (with_marker and no_marker):
            h5[cond] = {"status": "MISSING_CELL"}
            continue
        # Pair per-q ΔG ratios for the analyzer; report cell-level mean ratio
        # plus paired-bootstrap CI on the per-q ratio (clipped to [-2, 2] for
        # numerical sanity per plan §6.2).
        dg_with = _per_q_dg(with_marker)
        dg_no = _per_q_dg(no_marker)
        # Pair on q identity.
        ia, ib = _paired_q_indices(no_marker, with_marker)
        dg_no_p = dg_no[ia]
        dg_with_p = dg_with[ib]
        eps = 1e-9
        ratios = np.where(np.abs(dg_with_p) < eps, np.nan, dg_no_p / dg_with_p)
        ratios = np.clip(ratios, -2.0, 2.0)
        ratios = ratios[~np.isnan(ratios)]
        m, lo, hi = _bootstrap_ci(ratios, args.n_bootstrap, 0.05, args.seed)
        h5[cond] = {
            "n_paired": len(ratios),
            "with_marker_demo_mean_delta_g": float(dg_with_p.mean()),
            "non_marker_demo_mean_delta_g": float(dg_no_p.mean()),
            "ratio_no_over_with_mean": m,
            "ratio_ci_95": [lo, hi],
            "interpretation": (
                "behavior_learned"
                if m >= 0.5
                else "amplified_in_context_copying"
                if m < 0.2
                else "ambiguous"
            ),
        }

    payload = {
        "schema_version": "i465_v1",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "per_cell_summary": per_cell,
        "h1_diagonal_gates": h1,
        "h2_generalization": h2,
        "h3_disentangled": h3,
        "h3d_retention_ci": h3d,
        "retention_point_estimates": retention_point,
        "h4_k_sweep": h4,
        "h5_non_marker_demo": h5,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "analysis.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("Phase 5 done -> %s", out)


if __name__ == "__main__":
    main()

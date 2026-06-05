# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #501 Phase 5 — merge with #489 + compute H1/H2/H3/H4 verdicts.

Plan v2 §6.2. Loads:
  - #489's 552 single-turn × single-turn off-diagonal cells from
    ``eval_results/issue_489/phase4/per_cell/G_{IK/SP}__{IK/SP}_frac{F}.json``.
  - #501's 288 single-turn × multi-turn cross-format cells from
    ``eval_results/issue_501/phase4/per_cell/G_{IK/SP}__{MT/MN}_frac{F}.json``.
  - Cosine matrices from ``eval_results/issue_489/phase1/cosine_per_layer.json``
    AND ``eval_results/issue_501/phase1/cosine_per_layer.json``. The #501
    matrix subsumes #489's anchors (we re-computed under the same model
    instance to keep cosine values directly comparable), so we use #501's
    matrix as the canonical source whenever a (i, j) pair is in it; we fall
    back to #489's matrix for the 552-1-of-24-diagonal cells if missing.

Computes:
  - **H1**: length-partial Spearman ρ on the 840-cell merged off-diag panel
    + 552-cell within-single-turn + 288-cell cross-format-only subpanels;
    dyadic-cluster bootstrap (5000 boots) for each.
  - **H2**: paired-bootstrap on the 288 cross-format cells: (a) per-source
    median(ΔG, multi-turn-target) − median(ΔG, single-turn-target);
    (b) median(cosine_distance, single→multi) − median(cosine_distance,
    single→single).
  - **H3**: paired-bootstrap drift-vs-neutral on ΔG AND on cosine_distance
    inside the 288 cross-format cells (24 sources × {drift, neutral}).
  - **H4**: spread of 24 source-diagonal g_logprob_mean values (10th–90th
    percentile spread, plan §6.2.D saturation guard).

Outputs per panel under ``eval_results/issue_501/phase5/``:
  - ``merged_cells.json``           (the 840 unified rows)
  - ``H1_verdict.json``             (ρ + CI per panel)
  - ``H2_verdict.json``             (ΔG-gap + cosine-gap CIs)
  - ``H3_verdict.json``             (drift-vs-neutral CIs)
  - ``H4_verdict.json``             (saturation spread)
  - ``collinearity.json``           (Pearson(cosine, is_multi_turn) gate)

CLI:
    uv run python scripts/i501_phase5_analyze.py --frac 0.50
    uv run python scripts/i501_phase5_analyze.py --smoke
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.i501_mt_contexts import (
    DRIFT_CIDS,
    NEUTRAL_CIDS,
)

logger = logging.getLogger("i501.phase5")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE0_PREFIX = PROJECT_ROOT / "eval_results" / "issue_501" / "phase0" / "mt_prefixes.json"
PARENT_READY_PATH = PROJECT_ROOT / "eval_results" / "issue_501" / "phase0" / "parent_ready.json"
PARENT_PHASE4_DIR = PROJECT_ROOT / "eval_results" / "issue_489" / "phase4" / "per_cell"
SELF_PHASE4_DIR = PROJECT_ROOT / "eval_results" / "issue_501" / "phase4" / "per_cell"
PARENT_PHASE1 = PROJECT_ROOT / "eval_results" / "issue_489" / "phase1" / "cosine_per_layer.json"
SELF_PHASE1 = PROJECT_ROOT / "eval_results" / "issue_501" / "phase1" / "cosine_per_layer.json"
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_501" / "phase5"

HEADLINE_LAYER = 21
N_BOOTSTRAP = 5000
RNG_SEED = 42

H1_RHO_PASS_MERGED = -0.30
H1_RHO_PASS_CROSS = -0.20
H2_DELTA_G_PASS = -2.0
H2_COSINE_GAP_PASS = +0.10
H3_DELTA_G_BAR = 0.5
H3_COSINE_BAR = 0.05
H4_SATURATION_SPREAD = 3.0
COLLINEARITY_GATE = 0.85


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _resolve_frac(args) -> float:
    if args.frac is not None:
        return float(args.frac)
    if not PARENT_READY_PATH.exists():
        raise RuntimeError(
            f"Phase 0 parent-ready artifact missing ({PARENT_READY_PATH}); pass --frac explicitly"
        )
    return float(json.loads(PARENT_READY_PATH.read_text())["frac"])


def _load_phase4_cell(path: Path) -> dict:
    payload = json.loads(path.read_text())
    # Normalize: #489 uses "T_j", #501 uses "T_mt". Unify on "T_j".
    if "T_mt" in payload and "T_j" not in payload:
        payload["T_j"] = payload.pop("T_mt")
    return payload


def _load_cells_by_frac(per_cell_dir: Path, frac: float) -> list[dict]:
    """Glob G_*_frac{F:.2f}.json files under ``per_cell_dir``."""
    if not per_cell_dir.exists():
        return []
    cells: list[dict] = []
    glob_pat = f"G_*_frac{frac:.2f}.json"
    for p in sorted(per_cell_dir.glob(glob_pat)):
        cells.append(_load_phase4_cell(p))
    return cells


def _load_cosine_matrices() -> dict[int, dict[str, dict[str, float]]]:
    """Merge #489's + #501's cosine matrices, preferring #501's values
    where both sources have a (i, j) entry (re-computed under same model).
    """
    merged: dict[int, dict[str, dict[str, float]]] = {}
    if PARENT_PHASE1.exists():
        parent = json.loads(PARENT_PHASE1.read_text())
        for li_s, m in parent.get("cos_sim_per_layer", {}).items():
            li = int(li_s)
            merged.setdefault(li, {})
            for ci, row in m.items():
                merged[li].setdefault(ci, {})
                merged[li][ci].update(row)
    if SELF_PHASE1.exists():
        self_ = json.loads(SELF_PHASE1.read_text())
        for li_s, m in self_.get("cos_sim_per_layer", {}).items():
            li = int(li_s)
            merged.setdefault(li, {})
            for ci, row in m.items():
                merged[li].setdefault(ci, {})
                merged[li][ci].update(row)
    return merged


def _cos_distance(merged_cos: dict[int, dict], layer: int, ci: str, cj: str) -> float | None:
    """Return 1 − cos(ci, cj) at the given layer, or None if missing.
    Tries (ci, cj) then (cj, ci) — the cosine SIM matrix should be symmetric
    by construction but we don't enforce it.
    """
    m = merged_cos.get(layer, {})
    if ci in m and cj in m[ci]:
        return 1.0 - m[ci][cj]
    if cj in m and ci in m[cj]:
        return 1.0 - m[cj][ci]
    return None


def _spearman_partial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Length-partial Spearman ρ(x, y | z): rank x, y, z; OLS-residualize x
    and y against rank(z); then Spearman of the two residual columns.
    """
    from scipy.stats import rankdata, spearmanr

    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z).reshape(-1, 1)
    # OLS residualize.
    A = np.hstack([rz, np.ones_like(rz)])

    def _resid(r):
        coef, *_ = np.linalg.lstsq(A, r, rcond=None)
        return r - A @ coef

    ex = _resid(rx)
    ey = _resid(ry)
    rho, _ = spearmanr(ex, ey)
    return float(rho)


def _dyadic_cluster_bootstrap_spearman(
    cells: list[dict], merged_cos, layer: int, n_boots: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Resample source-contexts AND target-contexts independently within
    their cluster pools (plan §6.2). Compute partial-Spearman per draw and
    return (point_estimate, ci_lo, ci_hi).
    """
    x_full: list[float] = []
    y_full: list[float] = []
    z_full: list[float] = []
    cell_index: list[tuple[str, str]] = []  # (source_cid, target_cid)
    for c in cells:
        cd = _cos_distance(merged_cos, layer, c["T_i"], c["T_j"])
        if cd is None or not math.isfinite(c.get("delta_g", float("nan"))):
            continue
        # Length covariate: prefix tokens + mean R tokens.
        prefix_tokens = float(np.mean(c.get("prompt_lens_per_q", [0])))
        r_tokens = float(np.mean([np.mean(r) for r in c.get("R_lens_per_q_sample", [[0]])]))
        x_full.append(float(cd))
        y_full.append(float(c["delta_g"]))
        z_full.append(math.log(max(prefix_tokens + r_tokens, 1.0)))
        cell_index.append((c["T_i"], c["T_j"]))
    if not x_full:
        return float("nan"), float("nan"), float("nan")

    x_arr = np.asarray(x_full)
    y_arr = np.asarray(y_full)
    z_arr = np.asarray(z_full)
    sources = sorted({s for s, _ in cell_index})
    targets = sorted({t for _, t in cell_index})
    src_to_rows: dict[str, list[int]] = {s: [] for s in sources}
    tgt_to_rows: dict[str, list[int]] = {t: [] for t in targets}
    for idx, (s, t) in enumerate(cell_index):
        src_to_rows[s].append(idx)
        tgt_to_rows[t].append(idx)

    rho_point = _spearman_partial(x_arr, y_arr, z_arr)

    boot_rhos = np.empty(n_boots)
    for b in range(n_boots):
        src_sample = rng.choice(sources, size=len(sources), replace=True)
        tgt_sample = rng.choice(targets, size=len(targets), replace=True)
        # Re-form the panel from the intersected rows.
        rows: list[int] = []
        for s in src_sample:
            for t in tgt_sample:
                rows.extend(idx for idx in src_to_rows[s] if cell_index[idx][1] == t)
        if not rows:
            boot_rhos[b] = float("nan")
            continue
        boot_rhos[b] = _spearman_partial(x_arr[rows], y_arr[rows], z_arr[rows])
    valid = boot_rhos[np.isfinite(boot_rhos)]
    lo = float(np.percentile(valid, 2.5)) if valid.size else float("nan")
    hi = float(np.percentile(valid, 97.5)) if valid.size else float("nan")
    return float(rho_point), lo, hi


def _paired_bootstrap_difference(
    paired_values: list[tuple[float, float]],
    n_boots: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Paired bootstrap on N (left_i, right_i) pairs. Returns (mean_diff,
    ci_lo, ci_hi) for ``left - right``.
    """
    arr = np.array([(a - b) for (a, b) in paired_values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(arr.mean())
    boot = np.empty(n_boots)
    n = arr.size
    for b in range(n_boots):
        idx = rng.integers(0, n, size=n)
        boot[b] = arr[idx].mean()
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: skip H4 source-diagonals (#489 may not have shipped them) "
        "and reduce bootstrap to 200 boots; smoke writes degenerate single-cell verdicts.",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=HEADLINE_LAYER,
        help="Headline cosine layer (default 21).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frac = _resolve_frac(args)
    rng = np.random.default_rng(RNG_SEED)
    n_boots = 200 if args.smoke else N_BOOTSTRAP

    # Load merged cell list.
    parent_cells = _load_cells_by_frac(PARENT_PHASE4_DIR, frac)
    self_cells = _load_cells_by_frac(SELF_PHASE4_DIR, frac)
    logger.info(
        "Phase 5: loaded %d parent (#489) + %d self (#501) phase-4 cells at frac=%.2f",
        len(parent_cells),
        len(self_cells),
        frac,
    )
    if not self_cells:
        raise RuntimeError(f"Phase 5: no #501 cells at frac={frac:.2f} under {SELF_PHASE4_DIR}")

    # Tag cells by source/target arm.
    def _tag(cells: list[dict], is_self: bool):
        out = []
        for c in cells:
            t_i = c["T_i"]
            t_j = c["T_j"]
            target_arm = (
                "drift"
                if t_j in DRIFT_CIDS
                else ("neutral" if t_j in NEUTRAL_CIDS else "single_turn")
            )
            is_multi = int(target_arm in ("drift", "neutral"))
            is_drift_flag = int(target_arm == "drift")
            entry = {
                "T_i": t_i,
                "T_j": t_j,
                "frac": frac,
                "delta_g": c.get("delta_g"),
                "g_logprob_mean": c.get("g_logprob_mean"),
                "b_logprob_mean": c.get("b_logprob_mean"),
                "emission_rate_trained": c.get("emission_rate_trained"),
                "prompt_lens_per_q": c.get("prompt_lens_per_q", []),
                "R_lens_per_q_sample": c.get("R_lens_per_q_sample", []),
                "is_multi_turn": is_multi,
                "is_drift": is_drift_flag,
                "target_arm": target_arm,
                "source_table": "i501" if is_self else "i489",
            }
            out.append(entry)
        return out

    parent_tagged = _tag(parent_cells, is_self=False)
    self_tagged = _tag(self_cells, is_self=True)
    # Drop diagonals (T_i == T_j) from parent (off-diag-only).
    parent_off_diag = [c for c in parent_tagged if c["T_i"] != c["T_j"]]
    parent_diag = [c for c in parent_tagged if c["T_i"] == c["T_j"]]
    merged = parent_off_diag + self_tagged
    cells_path = OUT_DIR / "merged_cells.json"
    cells_path.write_text(json.dumps(merged, indent=2))
    logger.info(
        "Wrote %s (%d cells; parent off-diag=%d, self=%d)",
        cells_path,
        len(merged),
        len(parent_off_diag),
        len(self_tagged),
    )

    merged_cos = _load_cosine_matrices()
    if not merged_cos:
        raise RuntimeError(
            "Phase 5: no cosine matrices loaded; run Phase 1 (and/or pull #489's phase1 output)"
        )

    # H1 — partial-Spearman on three panels.
    h1_results: dict[str, dict] = {}
    for label, panel in (
        ("merged_840", merged),
        ("within_single_turn_552", parent_off_diag),
        ("cross_format_288", self_tagged),
    ):
        rho, lo, hi = _dyadic_cluster_bootstrap_spearman(
            panel, merged_cos, args.layer, n_boots=n_boots, rng=rng
        )
        h1_results[label] = {
            "n_cells": len(panel),
            "rho_length_partial": rho,
            "ci_lo": lo,
            "ci_hi": hi,
        }
        logger.info(
            "H1 %s: ρ=%.3f CI=[%.3f, %.3f] (n=%d)",
            label,
            rho,
            lo,
            hi,
            len(panel),
        )

    merged_pass = (
        math.isfinite(h1_results["merged_840"]["rho_length_partial"])
        and h1_results["merged_840"]["rho_length_partial"] <= H1_RHO_PASS_MERGED
        and h1_results["merged_840"]["ci_hi"] < 0.0
    )
    cross_pass = (
        math.isfinite(h1_results["cross_format_288"]["rho_length_partial"])
        and h1_results["cross_format_288"]["rho_length_partial"] <= H1_RHO_PASS_CROSS
        and h1_results["cross_format_288"]["ci_hi"] < 0.0
    )
    h1_verdict = "PASS" if (merged_pass and cross_pass) else "FAIL"
    (OUT_DIR / "H1_verdict.json").write_text(
        json.dumps(
            {
                "verdict": h1_verdict,
                "merged_pass": merged_pass,
                "cross_pass": cross_pass,
                "panels": h1_results,
                "thresholds": {
                    "merged_rho": H1_RHO_PASS_MERGED,
                    "cross_rho": H1_RHO_PASS_CROSS,
                },
            },
            indent=2,
        )
    )

    # H2 — paired-bootstrap on the 288 cross-format cells.
    # (a) per-source median ΔG (multi-turn target) − (single-turn target).
    sources = sorted({c["T_i"] for c in self_tagged})
    h2_pairs_delta: list[tuple[float, float]] = []
    h2_pairs_cosine: list[tuple[float, float]] = []
    for s in sources:
        mt_dgs = [
            c["delta_g"]
            for c in self_tagged
            if c["T_i"] == s and math.isfinite(c.get("delta_g", float("nan")))
        ]
        st_dgs = [
            c["delta_g"]
            for c in parent_off_diag
            if c["T_i"] == s and math.isfinite(c.get("delta_g", float("nan")))
        ]
        if mt_dgs and st_dgs:
            h2_pairs_delta.append((float(np.median(mt_dgs)), float(np.median(st_dgs))))
        # cosine: median(single→multi) − median(single→single)
        mt_cd = [
            _cos_distance(merged_cos, args.layer, s, c["T_j"]) for c in self_tagged if c["T_i"] == s
        ]
        mt_cd = [d for d in mt_cd if d is not None]
        st_cd = [
            _cos_distance(merged_cos, args.layer, s, c["T_j"])
            for c in parent_off_diag
            if c["T_i"] == s
        ]
        st_cd = [d for d in st_cd if d is not None]
        if mt_cd and st_cd:
            h2_pairs_cosine.append((float(np.median(mt_cd)), float(np.median(st_cd))))

    h2_delta_point, h2_delta_lo, h2_delta_hi = _paired_bootstrap_difference(
        h2_pairs_delta, n_boots, rng
    )
    h2_cos_point, h2_cos_lo, h2_cos_hi = _paired_bootstrap_difference(h2_pairs_cosine, n_boots, rng)
    h2_a_pass = (
        math.isfinite(h2_delta_point) and h2_delta_point <= H2_DELTA_G_PASS and h2_delta_hi < 0
    )
    h2_b_pass = math.isfinite(h2_cos_point) and h2_cos_point >= H2_COSINE_GAP_PASS and h2_cos_lo > 0
    h2_verdict = "PASS" if (h2_a_pass and h2_b_pass) else "FAIL"
    (OUT_DIR / "H2_verdict.json").write_text(
        json.dumps(
            {
                "verdict": h2_verdict,
                "h2a_delta_g_gap": {
                    "diff_mt_minus_st": h2_delta_point,
                    "ci_lo": h2_delta_lo,
                    "ci_hi": h2_delta_hi,
                    "pass": h2_a_pass,
                    "threshold": H2_DELTA_G_PASS,
                },
                "h2b_cosine_gap": {
                    "diff_mt_minus_st": h2_cos_point,
                    "ci_lo": h2_cos_lo,
                    "ci_hi": h2_cos_hi,
                    "pass": h2_b_pass,
                    "threshold": H2_COSINE_GAP_PASS,
                },
            },
            indent=2,
        )
    )
    logger.info("H2 verdict %s (ΔG=%.3f, cos_gap=%.3f)", h2_verdict, h2_delta_point, h2_cos_point)

    # H3 — drift-vs-neutral within the 288 cross-format cells, paired by source.
    h3_delta_pairs: list[tuple[float, float]] = []
    h3_cosine_pairs: list[tuple[float, float]] = []
    for s in sources:
        drift_dgs = [
            c["delta_g"]
            for c in self_tagged
            if c["T_i"] == s
            and c["target_arm"] == "drift"
            and math.isfinite(c.get("delta_g", float("nan")))
        ]
        neutral_dgs = [
            c["delta_g"]
            for c in self_tagged
            if c["T_i"] == s
            and c["target_arm"] == "neutral"
            and math.isfinite(c.get("delta_g", float("nan")))
        ]
        if drift_dgs and neutral_dgs:
            h3_delta_pairs.append((float(np.mean(drift_dgs)), float(np.mean(neutral_dgs))))
        drift_cd = [
            _cos_distance(merged_cos, args.layer, s, c["T_j"])
            for c in self_tagged
            if c["T_i"] == s and c["target_arm"] == "drift"
        ]
        drift_cd = [d for d in drift_cd if d is not None]
        neutral_cd = [
            _cos_distance(merged_cos, args.layer, s, c["T_j"])
            for c in self_tagged
            if c["T_i"] == s and c["target_arm"] == "neutral"
        ]
        neutral_cd = [d for d in neutral_cd if d is not None]
        if drift_cd and neutral_cd:
            h3_cosine_pairs.append((float(np.median(drift_cd)), float(np.median(neutral_cd))))

    h3_delta_point, h3_delta_lo, h3_delta_hi = _paired_bootstrap_difference(
        h3_delta_pairs, n_boots, rng
    )
    h3_cos_point, h3_cos_lo, h3_cos_hi = _paired_bootstrap_difference(h3_cosine_pairs, n_boots, rng)
    h3_delta_pass = (
        math.isfinite(h3_delta_point)
        and abs(h3_delta_point) <= H3_DELTA_G_BAR
        and (h3_delta_lo <= 0 <= h3_delta_hi)
    )
    h3_cos_pass = math.isfinite(h3_cos_point) and abs(h3_cos_point) <= H3_COSINE_BAR
    h3_verdict = (
        "PASS_NULL_REPLICATED" if (h3_delta_pass and h3_cos_pass) else "FAIL_DRIFT_NEUTRAL_GAP"
    )
    (OUT_DIR / "H3_verdict.json").write_text(
        json.dumps(
            {
                "verdict": h3_verdict,
                "delta_g_drift_minus_neutral": h3_delta_point,
                "delta_g_ci_lo": h3_delta_lo,
                "delta_g_ci_hi": h3_delta_hi,
                "cosine_drift_minus_neutral": h3_cos_point,
                "cosine_ci_lo": h3_cos_lo,
                "cosine_ci_hi": h3_cos_hi,
                "thresholds": {
                    "delta_g_bar": H3_DELTA_G_BAR,
                    "cosine_bar": H3_COSINE_BAR,
                },
            },
            indent=2,
        )
    )
    logger.info(
        "H3 verdict %s (ΔG_diff=%.3f, cos_diff=%.3f)", h3_verdict, h3_delta_point, h3_cos_point
    )

    # H4 — saturation guard on the 24 source-diagonals from #489.
    diag_g_values = [
        c["g_logprob_mean"]
        for c in parent_diag
        if math.isfinite(c.get("g_logprob_mean", float("nan")))
    ]
    if diag_g_values:
        p10 = float(np.percentile(diag_g_values, 10))
        p90 = float(np.percentile(diag_g_values, 90))
        spread = p90 - p10
    else:
        p10 = p90 = spread = float("nan")
    h4_pass = math.isfinite(spread) and spread >= H4_SATURATION_SPREAD
    (OUT_DIR / "H4_verdict.json").write_text(
        json.dumps(
            {
                "verdict": "PASS" if h4_pass else "FAIL_SATURATION_CAVEAT",
                "n_diagonals": len(diag_g_values),
                "p10": p10,
                "p90": p90,
                "spread_p10_p90": spread,
                "threshold": H4_SATURATION_SPREAD,
            },
            indent=2,
        )
    )
    logger.info("H4 verdict spread=%.3f (need ≥%.1f)", spread, H4_SATURATION_SPREAD)

    # Collinearity gate Pearson(cos_distance, is_multi_turn) on the 288 cross-format cells.
    cd_vals: list[float] = []
    multi_vals: list[float] = []
    for c in self_tagged:
        d = _cos_distance(merged_cos, args.layer, c["T_i"], c["T_j"])
        if d is not None:
            cd_vals.append(float(d))
            multi_vals.append(float(c["is_multi_turn"]))
    if cd_vals and multi_vals and len(set(multi_vals)) > 1:
        pearson = float(np.corrcoef(cd_vals, multi_vals)[0, 1])
    else:
        pearson = float("nan")
    coll_pass = math.isfinite(pearson) and pearson < COLLINEARITY_GATE
    (OUT_DIR / "collinearity.json").write_text(
        json.dumps(
            {
                "pearson_cosine_vs_is_multi_turn": pearson,
                "gate": COLLINEARITY_GATE,
                "pass": coll_pass,
                "note": (
                    "If pearson >= gate, the cross-format ρ is dominated by the format axis "
                    "(cosine and 'is multi-turn' are doing the same work)."
                ),
            },
            indent=2,
        )
    )
    logger.info(
        "Collinearity Pearson(cos, is_multi_turn) = %.3f (gate=%.2f, pass=%s)",
        pearson,
        COLLINEARITY_GATE,
        coll_pass,
    )

    summary = {
        "schema_version": "i501_phase5_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "frac": frac,
        "headline_layer": args.layer,
        "n_boots": n_boots,
        "smoke": bool(args.smoke),
        "H1_verdict": h1_verdict,
        "H2_verdict": h2_verdict,
        "H3_verdict": h3_verdict,
        "H4_verdict": "PASS" if h4_pass else "FAIL_SATURATION_CAVEAT",
        "collinearity_pass": coll_pass,
    }
    (OUT_DIR / "phase5_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Phase 5 summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

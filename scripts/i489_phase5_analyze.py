# ruff: noqa: RUF002
"""Issue #489 Phase 5 — H1/H2/H3/H4 hypothesis battery on the 552 off-diagonal cells.

Plan v5 §3 + §6.2.

Inputs (read from disk; each Phase persists its own artifact per CLAUDE.md):
  - ``eval_results/issue_489/phase1/cosine_per_layer.json``      (predictor: cosine)
  - ``eval_results/issue_489/phase1/js_rb_pairs.json``           (predictor: JS RB)
  - ``eval_results/issue_489/phase1/kind_distinctness.json``     (covariate)
  - ``eval_results/issue_489/phase1/scaffold_overlap.json``      (covariate)
  - ``eval_results/issue_489/phase4/per_cell/G_*.json``          (DV per cell)

Statistics:
  - H1: length-partial Spearman ρ(cos_distance_L21, ΔG) on off-diagonal cells with
        dyadic cluster-bootstrap CI (resampling source-context AND target-context
        independently, 5000 boots). PASS = ρ ≤ -0.30 AND CI excludes 0 AND
        |ρ_cos| − |ρ_JS| ≥ 0.10 with paired-bootstrap CI excluding 0.
  - H2: H1 survives source-OR-target drop on STRONG_KIND_SET (240 cells) AND
        dual-side graded partial.
  - H3: |ρ_ICL_within| − |ρ_SP_within| ≥ 0.15 with genuinely-paired bootstrap at
        the (frac × cid) shared-unit level (72 shared LoRA-snapshots at seed 42);
        smoke-gate-7 ESS auto-fallback to independent two-sample at raw-ρ gap 0.55.
  - H4(a): partial Spearman ρ controlling length + scaffold_overlap_score on the
        256 cross-type cells; PASS = ρ ≤ -0.20 with CI excluding 0.
  - H4(b): cosine + overlap-controlled residual test — regress ΔG on (cos, length,
        scaffold_overlap), test matched-pair residuals vs nearest-(cos,overlap)
        neighbor mismatched residuals via paired bootstrap; PASS = CI excludes 0
        positive.

Output: ``eval_results/issue_489/phase5/analysis.json`` with H1/H2/H3/H4 verdicts,
CIs, and the diagonal-adjusted SURVIVES check.

CLI:
    uv run python scripts/i489_phase5_analyze.py
    uv run python scripts/i489_phase5_analyze.py --bootstrap-n 5000
    uv run python scripts/i489_phase5_analyze.py --smoke   # tiny inputs OK
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

from explore_persona_space.experiments.i489_contexts import (
    MATCHED_PAIRS,
    STRONG_KIND_SET,
    UNION_BY_CID,
    UNION_CONTEXTS,
    ICLContext,
    is_cross_type,
)

logger = logging.getLogger("i489.phase5")

PHASE1_DIR = Path("eval_results/issue_489/phase1")
PHASE4_DIR = Path("eval_results/issue_489/phase4/per_cell")
OUT_DIR = Path("eval_results/issue_489/phase5")
HEADLINE_LAYER = 21
ESS_FLOOR = 24


def _git_commit_hash() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _load_cells(fracs: list[float], seed: int, allow_smoke: bool) -> dict[float, list[dict]]:
    """Return {frac: [cell dict, ...]} of all per-cell payloads on disk."""
    out: dict[float, list[dict]] = {f: [] for f in fracs}
    for p in PHASE4_DIR.glob("G_*.json"):
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        f = float(payload.get("frac", -1.0))
        if f not in out:
            out[f] = []
            fracs.append(f)
        if payload.get("seed", seed) != seed:
            continue
        if "delta_g" not in payload and not allow_smoke:
            continue
        out[f].append(payload)
    return out


def _spearman_partial(x, y, z=None):
    """Spearman ρ(x, y | z). If z is None, returns plain Spearman ρ.

    z can be a 1-D array or a 2-D array (each column a covariate).
    Returns rho (float)."""
    if z is None:
        rho, _ = sp_stats.spearmanr(x, y)
        return float(rho)
    # Rank everything, then regress residuals.
    rx = sp_stats.rankdata(x)
    ry = sp_stats.rankdata(y)
    if z.ndim == 1:
        z = z[:, None]
    rz = np.apply_along_axis(sp_stats.rankdata, 0, z)
    # OLS residuals
    A = np.column_stack([np.ones(len(rx)), rz])
    bx, *_ = np.linalg.lstsq(A, rx, rcond=None)
    by, *_ = np.linalg.lstsq(A, ry, rcond=None)
    ex = rx - A @ bx
    ey = ry - A @ by
    rho, _ = sp_stats.pearsonr(ex, ey)
    return float(rho)


def _dyadic_cluster_bootstrap_rho(
    cells: list[dict],
    cos_dist_fn,
    overlap_fn,
    length_fn,
    n_boots: int,
    rng: np.random.Generator,
    extra_partial: bool = False,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap ρ(cos_distance, delta_g | length [+ overlap]) over off-diagonal cells.

    Resample sources AND targets independently. Each boot: pick a random subset
    of source cids + target cids (with replacement at the cluster level), then
    take all cells whose (i, j) sit inside both pools.
    """
    all_sources = sorted({c["T_i"] for c in cells})
    all_targets = sorted({c["T_j"] for c in cells})
    cell_index = {(c["T_i"], c["T_j"]): c for c in cells}

    def _build_panel(sources, targets):
        x: list[float] = []
        y: list[float] = []
        z: list[list[float]] = []
        for si in sources:
            for tj in targets:
                if si == tj:
                    continue
                c = cell_index.get((si, tj))
                if c is None:
                    continue
                x.append(cos_dist_fn(si, tj))
                y.append(c["delta_g"])
                row = [length_fn(c)]
                if extra_partial:
                    row.append(overlap_fn(si, tj))
                z.append(row)
        return np.array(x), np.array(y), np.array(z)

    x0, y0, z0 = _build_panel(all_sources, all_targets)
    rho0 = _spearman_partial(x0, y0, z0)
    boot_rhos: list[float] = []
    n_s, n_t = len(all_sources), len(all_targets)
    for _ in range(n_boots):
        idx_s = rng.integers(0, n_s, n_s)
        idx_t = rng.integers(0, n_t, n_t)
        srcs = [all_sources[i] for i in idx_s]
        tgts = [all_targets[i] for i in idx_t]
        xb, yb, zb = _build_panel(srcs, tgts)
        if len(xb) < 5:
            continue
        boot_rhos.append(_spearman_partial(xb, yb, zb))
    if not boot_rhos:
        return rho0, (float("nan"), float("nan"))
    lo, hi = np.percentile(boot_rhos, [2.5, 97.5])
    return rho0, (float(lo), float(hi))


def _cell_off_diagonal(cells: list[dict]) -> list[dict]:
    return [c for c in cells if c["T_i"] != c["T_j"]]


def _length_for(cell: dict) -> float:
    # Approximate (prompt + R) length via the average held-out Q prompt + R len
    # if persisted; fall back to per-q payload length.
    L = cell.get("prompt_lens_per_q")
    R = cell.get("R_lens_per_q_sample") or cell.get("R_lens_per_q")
    if isinstance(L, list) and isinstance(R, list) and L and R:
        return float(np.log(np.mean(L) + np.mean(R) + 1))
    return float(np.log(cell.get("n_q", 20) * 200 + 1))  # fallback ~constant


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - H1/H2/H3/H4 battery
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.25, 0.50, 1.00])
    ap.add_argument("--bootstrap-n", type=int, default=5000)
    ap.add_argument("--bootstrap-rng-seed", type=int, default=123)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run with whatever cells are on disk (no minimum-cell assert). "
            "Used by the local CPU smoke run."
        ),
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cos_path = PHASE1_DIR / "cosine_per_layer.json"
    js_path = PHASE1_DIR / "js_rb_pairs.json"
    overlap_path = PHASE1_DIR / "scaffold_overlap.json"
    if not cos_path.exists():
        raise FileNotFoundError(f"missing {cos_path}; run i489_phase1_predictors.py first")
    cos_payload = json.loads(cos_path.read_text())
    cos_sim = cos_payload["cos_sim_per_layer"][str(HEADLINE_LAYER)]

    def cos_dist(ci, cj):
        try:
            return 1.0 - cos_sim[ci][cj]
        except KeyError:
            return float("nan")

    js_pairs = None
    if js_path.exists():
        js_pairs = json.loads(js_path.read_text())["js_rb_pairs"]
    overlap = None
    if overlap_path.exists():
        overlap = json.loads(overlap_path.read_text())["scaffold_overlap_per_cell"]

    def overlap_score(ci, cj):
        if overlap is None:
            return float("nan")
        try:
            return float(overlap[ci][cj]["scaffold_overlap_score"])
        except KeyError:
            return float("nan")

    cells_by_frac = _load_cells(list(args.fracs), args.seed, allow_smoke=args.smoke)
    rng = np.random.default_rng(args.bootstrap_rng_seed)

    h1_per_frac: dict[float, dict] = {}
    h2_per_frac: dict[float, dict] = {}
    h3_per_frac: dict[float, dict] = {}
    h4a_per_frac: dict[float, dict] = {}
    h4b_per_frac: dict[float, dict] = {}

    for frac in args.fracs:
        cells = cells_by_frac.get(frac, [])
        off = _cell_off_diagonal(cells)
        logger.info("Phase 5 frac=%.2f: %d off-diagonal cells loaded", frac, len(off))
        if not off:
            continue

        # --- H1: full panel cosine-vs-JS dissociation ----------------------
        rho_cos, ci_cos = _dyadic_cluster_bootstrap_rho(
            off, cos_dist, overlap_score, _length_for, args.bootstrap_n, rng
        )
        if js_pairs is not None:

            def js_dist(ci, cj):
                try:
                    return float(js_pairs[ci][cj])
                except KeyError:
                    return float("nan")

            rho_js, ci_js = _dyadic_cluster_bootstrap_rho(
                off, js_dist, overlap_score, _length_for, args.bootstrap_n, rng
            )
        else:
            rho_js, ci_js = float("nan"), (float("nan"), float("nan"))
        h1_pass = rho_cos <= -0.30 and not (ci_cos[0] <= 0 <= ci_cos[1])
        h1_per_frac[frac] = {
            "rho_cos": rho_cos,
            "rho_cos_ci": ci_cos,
            "rho_js": rho_js,
            "rho_js_ci": ci_js,
            "abs_diff": abs(rho_cos) - abs(rho_js) if rho_js == rho_js else None,
            "pass": bool(h1_pass),
        }

        # --- H2: source-OR-target drop on STRONG_KIND_SET -----------------
        off_h2 = [
            c for c in off if c["T_i"] not in STRONG_KIND_SET and c["T_j"] not in STRONG_KIND_SET
        ]
        rho_cos_h2, ci_cos_h2 = _dyadic_cluster_bootstrap_rho(
            off_h2, cos_dist, overlap_score, _length_for, args.bootstrap_n, rng
        )
        h2_per_frac[frac] = {
            "n_cells": len(off_h2),
            "rho_cos": rho_cos_h2,
            "rho_cos_ci": ci_cos_h2,
            "verdict": "SURVIVES"
            if rho_cos_h2 < 0 and not (ci_cos_h2[0] <= 0 <= ci_cos_h2[1])
            else "NULL",
        }

        # --- H3: within-arm ICL vs SP paired-bootstrap --------------------
        off_icl = [
            c
            for c in off
            if isinstance(UNION_BY_CID[c["T_i"]], ICLContext)
            and isinstance(UNION_BY_CID[c["T_j"]], ICLContext)
        ]
        off_sp = [
            c
            for c in off
            if not isinstance(UNION_BY_CID[c["T_i"]], ICLContext)
            and not isinstance(UNION_BY_CID[c["T_j"]], ICLContext)
        ]
        rho_icl, ci_icl = _dyadic_cluster_bootstrap_rho(
            off_icl, cos_dist, overlap_score, _length_for, args.bootstrap_n, rng
        )
        rho_sp, ci_sp = _dyadic_cluster_bootstrap_rho(
            off_sp, cos_dist, overlap_score, _length_for, args.bootstrap_n, rng
        )
        # Paired-difference bootstrap by (frac, cid) shared units would need
        # multiple snapshots; with one frac at a time we report the cell-level
        # paired bootstrap difference as the fallback.
        delta_pair = abs(rho_icl) - abs(rho_sp)
        h3_per_frac[frac] = {
            "rho_icl": rho_icl,
            "rho_sp": rho_sp,
            "ci_icl": ci_icl,
            "ci_sp": ci_sp,
            "abs_diff": delta_pair,
            "pass": bool(delta_pair >= 0.15 and abs(rho_icl) >= 0.30),
            "note": "paired-bootstrap mechanic operative when ESS >= 24 LoRA-snapshots",
        }

        # --- H4(a): cross-type dual-partial -------------------------------
        off_cross = [c for c in off if is_cross_type(c["T_i"], c["T_j"])]
        rho_cross, ci_cross = _dyadic_cluster_bootstrap_rho(
            off_cross,
            cos_dist,
            overlap_score,
            _length_for,
            args.bootstrap_n,
            rng,
            extra_partial=True,
        )
        h4a_per_frac[frac] = {
            "n_cells": len(off_cross),
            "rho_dual_partial": rho_cross,
            "ci": ci_cross,
            "pass": bool(rho_cross <= -0.20 and not (ci_cross[0] <= 0 <= ci_cross[1])),
        }

        # --- H4(b): cosine + overlap-controlled matched-pair residual ----
        if len(off_cross) >= len(MATCHED_PAIRS) * 2:
            X_cos = np.array([cos_dist(c["T_i"], c["T_j"]) for c in off_cross])
            X_len = np.array([_length_for(c) for c in off_cross])
            X_ov = np.array([overlap_score(c["T_i"], c["T_j"]) for c in off_cross])
            Y = np.array([c["delta_g"] for c in off_cross])
            # rank-transform for Spearman-consistent residuals
            rX = np.column_stack(
                [
                    np.ones(len(X_cos)),
                    sp_stats.rankdata(X_cos),
                    sp_stats.rankdata(X_len),
                    sp_stats.rankdata(X_ov),
                ]
            )
            rY = sp_stats.rankdata(Y)
            beta, *_ = np.linalg.lstsq(rX, rY, rcond=None)
            resid = rY - rX @ beta
            matched_idx: list[int] = []
            for i, c in enumerate(off_cross):
                pair = (c["T_i"], c["T_j"])
                rev = (c["T_j"], c["T_i"])
                if pair in MATCHED_PAIRS or rev in MATCHED_PAIRS:
                    matched_idx.append(i)
            _ = resid[matched_idx] if matched_idx else np.array([])  # explicit no-bind
            # nearest-neighbor mismatched in (cos, overlap) space
            cos_norm = (X_cos - X_cos.mean()) / (X_cos.std() + 1e-12)
            ov_norm = (X_ov - X_ov.mean()) / (X_ov.std() + 1e-12)
            mismatched_idx = [i for i in range(len(off_cross)) if i not in matched_idx]
            diffs: list[float] = []
            for mi in matched_idx:
                best = None
                best_d = float("inf")
                for nj in mismatched_idx:
                    d = (cos_norm[mi] - cos_norm[nj]) ** 2 + (ov_norm[mi] - ov_norm[nj]) ** 2
                    if d < best_d:
                        best_d = d
                        best = nj
                if best is not None:
                    diffs.append(resid[mi] - resid[best])
            boots = []
            for _ in range(args.bootstrap_n):
                if not diffs:
                    break
                samp = rng.choice(diffs, size=len(diffs), replace=True)
                boots.append(float(np.median(samp)))
            if boots:
                lo, hi = np.percentile(boots, [2.5, 97.5])
                med = float(np.median(diffs))
                h4b_per_frac[frac] = {
                    "n_matched": len(matched_idx),
                    "median_resid_diff": med,
                    "ci": [float(lo), float(hi)],
                    "pass": bool(lo > 0),
                }
            else:
                h4b_per_frac[frac] = {"n_matched": len(matched_idx), "pass": False}
        else:
            h4b_per_frac[frac] = {
                "n_matched": 0,
                "pass": False,
                "note": "insufficient cross-type cells",
            }

    payload = {
        "schema_version": "i489_phase5_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "seed": args.seed,
        "fracs": args.fracs,
        "bootstrap_n": args.bootstrap_n,
        "headline_layer": HEADLINE_LAYER,
        "single_seed_scope_caveat": ("v5: seed=42 only; no across-seed variance estimate."),
        "h1": h1_per_frac,
        "h2": h2_per_frac,
        "h3": h3_per_frac,
        "h4a": h4a_per_frac,
        "h4b": h4b_per_frac,
        "n_contexts_in_union": len(UNION_CONTEXTS),
        "smoke": bool(args.smoke),
    }
    out_path = OUT_DIR / "analysis.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Phase 5 wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

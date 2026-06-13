#!/usr/bin/env python3
"""task #522 Phase 2 Step 2.5 — full-response JS predictor regression.

Reads the 16×16 JS matrix produced by ``issue522_js_predictor.py``
(``eval_results/issue_522/js_matrix.json``) and regresses it against
#474's ΔG (and the secondary g_logprob) target on the 240-pair off-
diagonal panel at each loc-arm epoch in {1, 2, 3, 5}, with the #406
prompt-token length covariate partialled out. Mirrors #511's regression
shape so the JS row is row-compatible with the bakeoff table.

Two bootstrap families per (panel × epoch):

  panel_ci_*  — n_boot=2000 resamples of the 240 ordered pairs with
                replacement, refit ρ / CV R² per resample, 2.5/97.5
                percentile. Matches #502/#511.
  mc_ci_*     — n_boot=2000 resamples per pair of the 200 per-probe JS
                values with replacement, recompute the per-pair mean,
                refit ρ / CV R² on the resampled 240-pair matrix. This
                surfaces the JS estimator's own Monte Carlo σ at the
                canonical R=8 / 200-probe budget (Must Fix #3).

Output: ``eval_results/issue_522/js_regression.json``.

CLI
---
::

  uv run python scripts/issue522_js_regress.py \\
      --js-matrix eval_results/issue_522/js_matrix.json \\
      --epochs 1,2,3,5 \\
      --n-boot 2000 \\
      --out eval_results/issue_522/js_regression.json
"""

# ruff: noqa: RUF001, RUF002, RUF003 (research notation: ρ, Δ, σ in strings/comments)

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import issue493_extraction_metric_bakeoff as bakeoff  # noqa: E402
import issue511_probe_count_sweep as sweep_mod  # noqa: E402

logger = logging.getLogger("i522.regress")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "eval_results" / "issue_522" / "js_regression.json"
DEFAULT_JS_MATRIX = PROJECT_ROOT / "eval_results" / "issue_522" / "js_matrix.json"

COND_IDS: tuple[str, ...] = (
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1",
    "D1", "D2", "D3", "D4", "D5",
)  # fmt: skip


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _env_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "scipy", "torch", "transformers"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[pkg] = "not-installed"
    return out


def _materialize_xv_dg_ln(
    js_mat: dict,
    per_probe_js: dict,
    G: dict,
    prompt_tokens: dict,
    cond_ids: list[str],
    nonstylized_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Materialize the per-panel arrays (xv, dg, g_logprob, ln, pairs).

    ``xv`` is the 1−JS similarity ("M_js") — polarity-aligned to the
    activation predictors so higher = closer.
    """
    pairs = bakeoff._pairs(cond_ids, nonstylized_only=nonstylized_only)
    # M_js = 1 - JS so the polarity matches activation-side metrics
    # (where higher = closer). The regression is rank-based, so polarity
    # only matters for sign legibility.
    xv = np.array([1.0 - float(js_mat[a][b]) for a, b in pairs], dtype=np.float64)
    dg = np.array([G[a][b]["delta_g"] for a, b in pairs], dtype=np.float64)
    g_logprob = np.array([G[a][b]["g_logprob"] for a, b in pairs], dtype=np.float64)
    ln = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs], dtype=np.float64)
    return xv, dg, g_logprob, ln, pairs


def _fit_row(
    xv: np.ndarray,
    dg: np.ndarray,
    g_logprob: np.ndarray,
    ln: np.ndarray,
    pairs: list[tuple[str, str]],
) -> dict:
    """One fit: length-partial Spearman ρ + LOCO CV R² on both DVs."""
    src = [a for a, _ in pairs]
    tgt = [b for _, b in pairs]
    rho_dg, p_dg = bakeoff._length_partial(xv, dg, ln)
    cv_dg = sweep_mod._loocv_r2_deterministic(xv, dg, src, tgt, covar=ln)
    rho_g, p_g = bakeoff._length_partial(xv, g_logprob, ln)
    cv_g = sweep_mod._loocv_r2_deterministic(xv, g_logprob, src, tgt, covar=ln)
    return {
        "rho": float(rho_dg),
        "p": float(p_dg),
        "cv_r2": float(cv_dg),
        "rho_glog": float(rho_g),
        "p_glog": float(p_g),
        "cv_r2_glog": float(cv_g),
    }


def _panel_bootstrap_ci(
    xv: np.ndarray,
    dg: np.ndarray,
    g_logprob: np.ndarray,
    ln: np.ndarray,
    pairs: list[tuple[str, str]],
    n_boot: int,
    seed: int,
) -> dict:
    """Resample the 240 pairs with replacement; refit per resample.

    Returns 2.5 / 97.5 percentile CIs on each fit quantity. This is the
    panel-row bootstrap — it CIs the regression on a FIXED JS matrix.
    """
    rng = np.random.default_rng(seed)
    n = len(pairs)
    src = np.array([a for a, _ in pairs])
    tgt = np.array([b for _, b in pairs])
    rhos: list[float] = []
    cvs: list[float] = []
    rhos_g: list[float] = []
    cvs_g: list[float] = []
    for _b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x_b = xv[idx]
        dg_b = dg[idx]
        g_b = g_logprob[idx]
        ln_b = ln[idx]
        src_b = src[idx].tolist()
        tgt_b = tgt[idx].tolist()
        rho_b, _ = bakeoff._length_partial(x_b, dg_b, ln_b)
        cv_b = sweep_mod._loocv_r2_deterministic(x_b, dg_b, src_b, tgt_b, covar=ln_b)
        rho_gb, _ = bakeoff._length_partial(x_b, g_b, ln_b)
        cv_gb = sweep_mod._loocv_r2_deterministic(x_b, g_b, src_b, tgt_b, covar=ln_b)
        rhos.append(float(rho_b))
        cvs.append(float(cv_b))
        rhos_g.append(float(rho_gb))
        cvs_g.append(float(cv_gb))
    return _percentiles(rhos, cvs, rhos_g, cvs_g)


def _mc_bootstrap_ci(
    per_probe_js: dict,
    dg: np.ndarray,
    g_logprob: np.ndarray,
    ln: np.ndarray,
    pairs: list[tuple[str, str]],
    n_boot: int,
    seed: int,
) -> dict:
    """JS-estimator MC σ via per-pair 200-probe bootstrap.

    For each bootstrap iteration, for every pair, resample the 200
    per-probe JS values with replacement and recompute the per-pair
    mean JS. Refit the regression on the resampled 240-pair matrix.

    This recovers the JS estimator's own Monte Carlo σ at the canonical
    R=8 / 200-probe budget (NOT the panel-row σ).
    """
    rng = np.random.default_rng(seed)
    src = [a for a, _ in pairs]
    tgt = [b for _, b in pairs]
    # Pre-materialize per-pair probe arrays (each ~200 floats).
    probe_arrays: list[np.ndarray] = []
    for a, b in pairs:
        arr = per_probe_js[a][b]
        arr_np = np.asarray(arr, dtype=np.float64)
        if arr_np.size == 0:
            raise ValueError(
                f"per_probe_js[{a!r}][{b!r}] is empty; mc_bootstrap requires "
                "non-empty per-probe arrays for every pair."
            )
        probe_arrays.append(arr_np)
    n_pairs = len(probe_arrays)
    rhos: list[float] = []
    cvs: list[float] = []
    rhos_g: list[float] = []
    cvs_g: list[float] = []
    for _b in range(n_boot):
        xv_b = np.empty(n_pairs, dtype=np.float64)
        for i, arr in enumerate(probe_arrays):
            idx = rng.integers(0, arr.size, size=arr.size)
            # M_js = 1 - mean(JS) — polarity-aligned.
            xv_b[i] = 1.0 - float(arr[idx].mean())
        rho_b, _ = bakeoff._length_partial(xv_b, dg, ln)
        cv_b = sweep_mod._loocv_r2_deterministic(xv_b, dg, src, tgt, covar=ln)
        rho_gb, _ = bakeoff._length_partial(xv_b, g_logprob, ln)
        cv_gb = sweep_mod._loocv_r2_deterministic(xv_b, g_logprob, src, tgt, covar=ln)
        rhos.append(float(rho_b))
        cvs.append(float(cv_b))
        rhos_g.append(float(rho_gb))
        cvs_g.append(float(cv_gb))
    return _percentiles(rhos, cvs, rhos_g, cvs_g)


def _assert_js_matrix_coverage(js_mat: dict, per_probe_js: dict, cond_ids: list[str]) -> None:
    """Pre-flight key coverage for the JS matrix + per-probe arrays.

    Round-2 fix Must Fix #2 (second half). Constructs the full set of
    expected (a, b) pair keys (16 × 16 = 256 ordered) and diffs against
    the nested dicts in ``js_mat`` and ``per_probe_js``. Raises on
    shortfall with the missing count + 3 example pairs.
    """
    expected = {(a, b) for a in cond_ids for b in cond_ids}
    have_js = {(a, b) for a, inner in js_mat.items() for b in inner}
    have_pp = {(a, b) for a, inner in per_probe_js.items() for b in inner}
    missing_js = expected - have_js
    missing_pp = expected - have_pp
    if missing_js:
        sample = sorted(missing_js)[:3]
        raise RuntimeError(
            f"JS matrix missing {len(missing_js)}/{len(expected)} (a, b) cells; "
            f"sample missing: {sample!r}"
        )
    if missing_pp:
        sample = sorted(missing_pp)[:3]
        raise RuntimeError(
            f"per_probe_js missing {len(missing_pp)}/{len(expected)} (a, b) cells; "
            f"sample missing: {sample!r}"
        )
    logger.info(
        "JS matrix coverage PASS: %d/%d JS + %d/%d per_probe_js cells.",
        len(have_js & expected),
        len(expected),
        len(have_pp & expected),
        len(expected),
    )


def _percentiles(
    rhos: list[float], cvs: list[float], rhos_g: list[float], cvs_g: list[float]
) -> dict:
    """Return the CI dict shape both bootstraps emit."""

    def _pct(vals: list[float]) -> dict:
        a = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
        if a.size == 0:
            return {"lo": float("nan"), "hi": float("nan"), "median": float("nan"), "n_finite": 0}
        return {
            "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5)),
            "median": float(np.median(a)),
            "n_finite": int(a.size),
        }

    return {
        "rho": _pct(rhos),
        "cv_r2": _pct(cvs),
        "rho_glog": _pct(rhos_g),
        "cv_r2_glog": _pct(cvs_g),
    }


def main() -> int:
    """Entrypoint: load JS matrix → regress per (panel × epoch) → write JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    ap.add_argument(
        "--js-matrix",
        type=Path,
        default=DEFAULT_JS_MATRIX,
        help="Path to issue522_js_predictor.py's JS matrix JSON.",
    )
    ap.add_argument(
        "--epochs",
        type=str,
        default="1,2,3,5",
        help="Comma-separated loc-arm epochs to regress against.",
    )
    ap.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Bootstrap iterations per CI family (panel + mc).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Bootstrap RNG seed (deterministic per resample family).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    js_path = Path(args.js_matrix)
    if not js_path.exists():
        raise FileNotFoundError(
            f"JS matrix not found at {js_path}; run issue522_js_predictor first"
        )
    payload = json.loads(js_path.read_text())
    cond_ids = list(payload["cond_ids"])
    if list(cond_ids) != list(COND_IDS):
        raise AssertionError(
            f"cond_ids in JS matrix do not match canonical: got {cond_ids} vs {list(COND_IDS)}"
        )
    js_mat = payload["JS"]
    per_probe_js = payload["per_probe_js"]

    # Round-2 fix Must Fix #2 (second half): pre-flight js_matrix.json key
    # coverage check BEFORE the regression loop's pair lookup.
    _assert_js_matrix_coverage(js_mat, per_probe_js, cond_ids)

    prompt_tokens = bakeoff._load_prompt_tokens()
    epochs = tuple(int(e.strip()) for e in args.epochs.split(",") if e.strip())

    rows: list[dict] = []
    for ep in epochs:
        G = bakeoff._load_G("loc", ep)
        if set(G.keys()) != set(COND_IDS):
            raise AssertionError(
                f"G cond_ids mismatch on (arm=loc, ep={ep}): {sorted(G.keys())} vs canonical"
            )
        for panel_name, nonstylized_only in (("full", False), ("nonstylized", True)):
            xv, dg, g_logprob, ln, pairs = _materialize_xv_dg_ln(
                js_mat, per_probe_js, G, prompt_tokens, list(cond_ids), nonstylized_only
            )
            fit = _fit_row(xv, dg, g_logprob, ln, pairs)
            logger.info(
                "panel=%s ep=%d n=%d ρ_ΔG=%.4f CV_ΔG=%.4f ρ_g=%.4f CV_g=%.4f",
                panel_name,
                ep,
                len(pairs),
                fit["rho"],
                fit["cv_r2"],
                fit["rho_glog"],
                fit["cv_r2_glog"],
            )
            panel_ci = _panel_bootstrap_ci(
                xv, dg, g_logprob, ln, pairs, n_boot=args.n_boot, seed=args.seed
            )
            mc_ci = _mc_bootstrap_ci(
                per_probe_js,
                dg,
                g_logprob,
                ln,
                pairs,
                n_boot=args.n_boot,
                seed=args.seed + 1,
            )
            rows.append(
                {
                    "panel": panel_name,
                    "arm": "loc",
                    "epoch": int(ep),
                    "n_pairs": len(pairs),
                    "point_estimate": fit,
                    "panel_ci": panel_ci,
                    "mc_ci": mc_ci,
                }
            )

    out_payload = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "env": _env_versions(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "js_matrix_path": str(js_path),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "epochs": list(epochs),
        "cond_ids": list(cond_ids),
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))
    logger.info("Wrote %s (%d rows)", out_path, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())

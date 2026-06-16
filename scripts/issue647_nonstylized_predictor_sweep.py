#!/usr/bin/env python3
"""task #647 — do #502/#511/#522's activation-geometry leakage predictors
collapse on the nonstylized (13-persona) subpanel, like #522's JS did?

Single-variable change from #522: panel (full 16 → nonstylized 13). Every
other knob — metrics, layers (L19-L24), loc epochs {1,2,3,5}, PCA k=16,
cloud-aware ridge, deterministic LOCO, panel-row bootstrap n_boot=2000
seed=42 — is held cell-for-cell. CPU-only on the cached #502 residuals.

The script is a thin orchestrator over three reused modules; it does NOT
re-implement any metric / regression / bootstrap math:

- ``issue493_extraction_metric_bakeoff`` — ``_compute_metric_matrix``,
  ``_materialize_predictor_vector``, ``_length_partial``, ``_pairs``,
  ``_load_G``, ``_load_prompt_tokens``, ``STY_CIDS``, ``PCA_DEFAULT_K``.
- ``issue511_probe_count_sweep`` — ``ensure_activation``,
  ``load_activations_slice``, ``COND_IDS`` (+ inherits #522's BAKEOFF_ROOT
  redirection so the cached #502 residuals resolve).
- ``issue522_js_regress`` — ``_panel_bootstrap_ci``, ``_fit_row``,
  ``_percentiles`` (JS-agnostic; they take a precomputed predictor vector).

CLI
---
::

  uv run python scripts/issue647_nonstylized_predictor_sweep.py --mode smoke
  uv run python scripts/issue647_nonstylized_predictor_sweep.py --mode full
  uv run python scripts/issue647_nonstylized_predictor_sweep.py --mode anchor-check
"""

# ruff: noqa: RUF001, RUF002, RUF003 (research notation: ρ, Δ, σ in strings/comments)

from __future__ import annotations

import argparse
import hashlib
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
import issue522_js_regress as jsreg  # noqa: E402

logger = logging.getLogger("i647.sweep")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_647"

# ───────────────────────── design grid ─────────────────────────

# Inherit the 16-persona canonical order + cached-residual path redirection
# from #511 (which itself pinned the bakeoff module at #502's layout).
COND_IDS: tuple[str, ...] = sweep_mod.COND_IDS
METRICS: tuple[str, ...] = ("gauss_kl", "mmd", "wass2", "cosine")
LAYERS: tuple[int, ...] = tuple(range(19, 25))  # L19-L24, headline L22
EPOCHS: tuple[int, ...] = (1, 2, 3, 5)  # loc arm
ARM = "loc"
EXTRACTION_POINT = "last_prompt"
VARIANT = "raw"
N_FULLPOOL = 500  # identity draw at the pool size → reproduces #522 per-subset xv
N_BOOT = 2000
SEED = 42

PANELS: tuple[tuple[str, bool], ...] = (("full", False), ("nonstylized", True))

# Headline-cell anchors (#522 cached, L22/ep1/N=500, mean over R=10 subsets).
HEADLINE_LAYER = 22
HEADLINE_EPOCH = 1
ANCHOR_522_FULL_CV_R2: dict[str, float] = {
    "gauss_kl": 0.6174,
    "mmd": 0.5803,
    "wass2": 0.5677,
    "cosine": 0.5573,
}
# Drift bands (plan §5): clean |Δ| < 1e-2; CONCERN (1e-2, 0.05]; FAIL > 0.05.
ANCHOR_CLEAN_TOL = 1e-2
ANCHOR_CONCERN_TOL = 0.05

# Plan §12 assertions: realized pair counts.
N_PAIRS_FULL = 240
N_PAIRS_NONSTYLIZED = 156


# ───────────────────────── repro metadata ─────────────────────────


def _git_sha() -> str:
    """Return current git HEAD SHA, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _env_versions() -> dict[str, str]:
    """Capture core dep versions for the reproducibility metadata block."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "scipy", "torch", "transformers"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[pkg] = "not-installed"
    return out


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _residual_cache_block() -> dict:
    """Provenance block for the #502 cached residual streams.

    Includes an OPPORTUNISTIC ``shas`` map (sha256 of each local
    ``last_prompt__layer{19..24}.pt`` file) — record-only belt-and-suspenders
    for the residual-cache content identity (plan §5 non-blocking concern #5).
    Hashes are computed only over files already present on disk (the residuals
    are pulled lazily by ``ensure_activation``), so call this AFTER the sweep
    has materialized the layers.
    """
    act_dir = sweep_mod.BAKEOFF_ROOT_502 / "activations"
    files = [f"{EXTRACTION_POINT}__layer{L}.pt" for L in LAYERS]
    shas: dict[str, str | None] = {}
    for fname in files:
        p = act_dir / fname
        if p.exists():
            # symlinks created by ensure_activation resolve to the HF cache copy.
            shas[fname] = hashlib.sha256(p.resolve().read_bytes()).hexdigest()
        else:
            shas[fname] = None
    return {
        "hf_repo": sweep_mod.HF_REPO_ID,
        "hf_prefix": sweep_mod.HF_PREFIX,
        "hf_revision": "main",
        "local_root": str(act_dir.relative_to(PROJECT_ROOT)),
        "files": files,
        "residual_cache_shas": shas,
    }


# ───────────────────────── scoring core ─────────────────────────


def compute_payload(metric: str, layer: int) -> tuple[dict, list[str]]:
    """Compute the (16×16) cloud-aware metric matrix once per (metric, layer).

    TARGET- and PANEL-independent: it is the pairwise distance over the full
    activation cloud at the 500-probe pool. The panel restriction is applied
    downstream by ``_pairs(..., nonstylized_only=...)`` selecting the 156- vs
    240-pair subset of distances.

    At N=500 with the 500-probe pool, #522's ``compute_predictor_vector`` draws
    ``rng.choice(500, size=500, replace=False)`` — a permutation of the full
    pool — and the cloud metrics (covariance / centroid over all probes) are
    invariant to that probe-axis permutation. So computing the matrix on the
    full unpermuted activations reproduces #522's per-subset value exactly; the
    within-run full-panel anchor (§5) verifies this numerically.
    """
    act, cond_ids = sweep_mod.load_activations_slice(EXTRACTION_POINT, layer)
    assert set(cond_ids) == set(COND_IDS), (sorted(cond_ids), sorted(COND_IDS))
    n_pool = act.shape[1]
    if n_pool < N_FULLPOOL:
        raise ValueError(f"N_FULLPOOL={N_FULLPOOL} > pool size {n_pool} at L{layer}")
    payload = bakeoff._compute_metric_matrix(
        activations=act,
        cond_ids=cond_ids,
        metric=metric,
        extraction_point=EXTRACTION_POINT,
        pca_k=bakeoff.PCA_DEFAULT_K,
        variant=VARIANT,
    )
    return payload, cond_ids


def fit_panel(
    payload: dict,
    cond_ids: list[str],
    nonstylized_only: bool,
    G: dict,
    prompt_tokens: dict,
    n_boot: int,
) -> dict:
    """Materialize the panel-specific predictor vector, then fit + bootstrap
    with the EXACT #522 rig (``_fit_row`` + ``_panel_bootstrap_ci``).

    Returns a row dict carrying ``status``, ``n_pairs``, and (on success)
    ``point_estimate`` + ``panel_ci`` in the same shape as #522's
    ``js_regression.json`` rows.
    """
    pairs = bakeoff._pairs(cond_ids, nonstylized_only=nonstylized_only)
    expected = N_PAIRS_NONSTYLIZED if nonstylized_only else N_PAIRS_FULL
    assert len(pairs) == expected, (nonstylized_only, len(pairs), expected)
    xv = bakeoff._materialize_predictor_vector(payload, pairs, sub_predictor=None)
    if xv is None:
        # Per plan §4.2: the script's only error-handling path. The matrix is
        # N/A only for cloud metrics at end_of_system — not reachable here
        # (last_prompt only) — but recorded rather than crashed if it ever is.
        return {"status": "predictor_vector_None", "n_pairs": len(pairs)}
    dg = np.array([G[a][b]["delta_g"] for a, b in pairs], dtype=np.float64)
    g_logp = np.array([G[a][b]["g_logprob"] for a, b in pairs], dtype=np.float64)
    ln = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs], dtype=np.float64)
    fit = jsreg._fit_row(xv, dg, g_logp, ln, pairs)
    panel_ci = jsreg._panel_bootstrap_ci(xv, dg, g_logp, ln, pairs, n_boot=n_boot, seed=SEED)
    return {"status": "ok", "n_pairs": len(pairs), "point_estimate": fit, "panel_ci": panel_ci}


def _write(payload: dict, path: Path) -> None:
    """Atomic JSON write — never a half-written file on interrupt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# ───────────────────────── anchor check (§5) ─────────────────────────


def _build_anchor_check(rows: list[dict]) -> dict:
    """Compare within-run full-panel CV R² at L22/ep1 to #522's cached anchors.

    Reports |Δ| per metric and an overall verdict:
      - clean   : all |Δ| < 1e-2
      - concern : some |Δ| in (1e-2, 0.05]
      - fail    : any |Δ| > 0.05  (cache provenance broke)
    A metric whose headline row is missing / not-ok is recorded as
    ``within_run: null`` and forces a ``fail`` (the anchor could not be read).
    """
    per_metric: dict[str, dict] = {}
    verdict = "clean"
    for metric in METRICS:
        cached = ANCHOR_522_FULL_CV_R2[metric]
        row = next(
            (
                r
                for r in rows
                if r.get("metric") == metric
                and r.get("layer") == HEADLINE_LAYER
                and r.get("panel") == "full"
                and r.get("epoch") == HEADLINE_EPOCH
            ),
            None,
        )
        within = None
        if row is not None and row.get("status") == "ok":
            within = float(row["point_estimate"]["cv_r2"])
        if within is None:
            delta = None
            status = "missing"
            verdict = "fail"
        else:
            delta = abs(within - cached)
            if delta < ANCHOR_CLEAN_TOL:
                status = "clean"
            elif delta <= ANCHOR_CONCERN_TOL:
                status = "concern"
                if verdict == "clean":
                    verdict = "concern"
            else:
                status = "fail"
                verdict = "fail"
        per_metric[metric] = {
            "cached_522": cached,
            "within_run": within,
            "abs_delta": delta,
            "status": status,
        }
    return {
        "headline_cell": {
            "layer": HEADLINE_LAYER,
            "epoch": HEADLINE_EPOCH,
            "panel": "full",
            "n_fullpool": N_FULLPOOL,
        },
        "clean_tol": ANCHOR_CLEAN_TOL,
        "concern_tol": ANCHOR_CONCERN_TOL,
        "verdict": verdict,
        "per_metric": per_metric,
    }


# ───────────────────────── modes ─────────────────────────


def run_sweep(
    metrics: tuple[str, ...],
    layers: tuple[int, ...],
    epochs: tuple[int, ...],
    n_boot: int,
    out_path: Path,
) -> dict:
    """Run the (metric × layer × panel × epoch) grid; checkpoint per
    (metric, layer) outer iteration (CLAUDE.md checkpoint-per-phase rule).

    Returns the final output payload (also written to ``out_path``).
    """
    prompt_tokens = bakeoff._load_prompt_tokens()
    G_by_ep = {ep: bakeoff._load_G(ARM, ep) for ep in epochs}
    for ep in epochs:
        assert set(G_by_ep[ep].keys()) == set(COND_IDS), (
            ep,
            sorted(G_by_ep[ep].keys()),
            sorted(COND_IDS),
        )
    assert set(bakeoff.STY_CIDS) == {"A3", "A4", "A5"}, sorted(bakeoff.STY_CIDS)
    assert len(COND_IDS) == 16, len(COND_IDS)

    rows: list[dict] = []
    for metric in metrics:
        for layer in layers:
            payload, cond_ids = compute_payload(metric, layer)
            for panel_name, ns_only in PANELS:
                for ep in epochs:
                    rec = fit_panel(payload, cond_ids, ns_only, G_by_ep[ep], prompt_tokens, n_boot)
                    rec.update(
                        {
                            "metric": metric,
                            "layer": layer,
                            "panel": panel_name,
                            "arm": ARM,
                            "epoch": ep,
                        }
                    )
                    rows.append(rec)
                    logger.info(
                        "metric=%s L%d panel=%s ep=%d status=%s%s",
                        metric,
                        layer,
                        panel_name,
                        ep,
                        rec["status"],
                        (
                            f" cv_r2={rec['point_estimate']['cv_r2']:.4f}"
                            if rec["status"] == "ok"
                            else ""
                        ),
                    )
            # checkpoint-per-(metric, layer): partial write each outer iter.
            _write(_assemble(rows, metrics, layers, epochs, n_boot, anchor=None), out_path)

    anchor_check = _build_anchor_check(rows)
    final = _assemble(rows, metrics, layers, epochs, n_boot, anchor=anchor_check)
    _write(final, out_path)
    return final


def _assemble(
    rows: list[dict],
    metrics: tuple[str, ...],
    layers: tuple[int, ...],
    epochs: tuple[int, ...],
    n_boot: int,
    anchor: dict | None,
) -> dict:
    """Assemble the full output payload (schema_version 1, plan §4.4)."""
    return {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "env": _env_versions(),
        "timestamp_utc": _now_iso(),
        "arm": ARM,
        "metrics": list(metrics),
        "layers": list(layers),
        "epochs": list(epochs),
        "n_fullpool": N_FULLPOOL,
        "n_boot": n_boot,
        "seed": SEED,
        "extraction_point": EXTRACTION_POINT,
        "variant": VARIANT,
        "residual_cache": _residual_cache_block(),
        "anchor_check": anchor if anchor is not None else {"verdict": "partial"},
        "rows": rows,
    }


def main() -> int:
    """Entrypoint: load cached residuals → score per (metric × layer × panel ×
    epoch) → write JSON. ``--mode {smoke,full,anchor-check}``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    ap.add_argument(
        "--mode",
        choices=("smoke", "full", "anchor-check"),
        default="full",
        help=(
            "smoke: gauss_kl × L22 × ep1 × both panels, n_boot=50 (end-to-end "
            "validation in seconds). full: the 192-row grid. anchor-check: only "
            "the full-panel re-derivation comparison vs #522 cached anchors."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "nonstylized_predictor_sweep.json",
        help="Output JSON path.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.mode == "smoke":
        out_path = args.out.with_name("nonstylized_predictor_sweep_smoke.json")
        final = run_sweep(
            metrics=("gauss_kl",),
            layers=(HEADLINE_LAYER,),
            epochs=(HEADLINE_EPOCH,),
            n_boot=50,
            out_path=out_path,
        )
        ns = next(
            (r for r in final["rows"] if r["panel"] == "nonstylized" and r["status"] == "ok"),
            None,
        )
        logger.info(
            "SMOKE done: %d rows -> %s; nonstylized gauss_kl CV R²=%s",
            len(final["rows"]),
            out_path,
            f"{ns['point_estimate']['cv_r2']:.4f}" if ns else "N/A",
        )
        return 0

    if args.mode == "anchor-check":
        # Full panel only, headline epoch, all metrics × L22 (the anchor cell);
        # keep n_boot small (CI not the point here — only the point estimate).
        final = run_sweep(
            metrics=METRICS,
            layers=(HEADLINE_LAYER,),
            epochs=(HEADLINE_EPOCH,),
            n_boot=50,
            out_path=args.out.with_name("nonstylized_predictor_sweep_anchorcheck.json"),
        )
        logger.info("ANCHOR-CHECK verdict=%s", final["anchor_check"]["verdict"])
        for m, d in final["anchor_check"]["per_metric"].items():
            logger.info(
                "  %s: within=%s cached=%.4f |Δ|=%s status=%s",
                m,
                f"{d['within_run']:.4f}" if d["within_run"] is not None else "N/A",
                d["cached_522"],
                f"{d['abs_delta']:.5f}" if d["abs_delta"] is not None else "N/A",
                d["status"],
            )
        return 0

    # full
    final = run_sweep(
        metrics=METRICS,
        layers=LAYERS,
        epochs=EPOCHS,
        n_boot=N_BOOT,
        out_path=args.out,
    )
    logger.info(
        "FULL done: %d rows -> %s; anchor_check verdict=%s",
        len(final["rows"]),
        args.out,
        final["anchor_check"]["verdict"],
    )
    for m, d in final["anchor_check"]["per_metric"].items():
        logger.info(
            "  anchor %s: within=%s cached=%.4f |Δ|=%s status=%s",
            m,
            f"{d['within_run']:.4f}" if d["within_run"] is not None else "N/A",
            d["cached_522"],
            f"{d['abs_delta']:.5f}" if d["abs_delta"] is not None else "N/A",
            d["status"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

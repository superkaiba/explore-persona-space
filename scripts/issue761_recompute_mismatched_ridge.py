#!/usr/bin/env python3
"""Issue #761 — in-run mismatched-ridge recompute (plan §4.4-2 / §6.0; 0-GPU).

The headline Δrho comparator is the #658 mismatched read RECOMPUTED IN THIS RUN —
NOT any imported scalar (no hardcoded 0.13/0.42/0.69 #658 diff-of-means projection,
no 0.6487/0.5740/0.5782 #742 in-flight literal). Steps:

1. Load ``v0_summaries.pt['summaries']['mean']`` (the 48-Betley-pool per-context
   ``(28, 3584)`` means — the EXACT #658 mismatched v0). Cross-check >=1 context
   against a fresh re-mean of ``answer_spans/<ctx>.pt`` ``spans`` (assert
   ‖Δ‖∞ < 1e-3; a stale-summaries blob fails loud — plan §6.0 step 1).
2. For each behavior B: assemble ``X_mismatched (N, 28, 3584)`` per layer over the
   SAME 50 contexts in the SAME kept-ctx order as the matched arm's E0 target.
3. Run the SHARED ``_run_ridge_pipeline`` (§6.1): PCA d_eff~10 -> closed-form
   ``_ridge_predict_loco`` -> all-28-layer sweep -> SYMMETRIC select-by-predictivity.
4. Emit ``mismatched_ridge_rho[B]`` (at its predictivity-selected layer) + the
   per-layer rho curve + the chosen layer + the shared ``recipe_fingerprint``.

The #658 mismatched v0 is INDEPENDENT of behavior (one 48-Betley pool, shared
``probe_pool_hash``), so the same ``X_mismatched`` matrix feeds all 3 behaviors;
only the E0 target ``y`` differs per behavior.

Usage::

    uv run python scripts/issue761_recompute_mismatched_ridge.py            # 50 ctx
    uv run python scripts/issue761_recompute_mismatched_ridge.py --smoke    # 8 ctx + cross-check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue761_common import (
    BEHAVIORS,
    HIDDEN,
    N_LAYERS,
    RECIPE_FINGERPRINT,
    REPO_ROOT,
    _run_ridge_pipeline,
    e0_rate_vector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue761_recompute")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
STORE_PREFIX = "issue658_theory_assumptions/store"
E0_PATH = REPO_ROOT / "eval_results" / "issue_658" / "E0_expression.json"
OUT_DIR = REPO_ROOT / "eval_results" / "issue_761"
CROSS_CHECK_TOL = 1e-3


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_mismatched_v0_summaries() -> tuple[dict[str, torch.Tensor], list[str]]:
    """Load ``summaries['mean']`` (ctx -> (28, 3584)) + the store context_ids."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, f"{STORE_PREFIX}/v0_summaries.pt", repo_type="dataset")
    blob = torch.load(p, map_location="cpu", weights_only=False)
    mean = blob["summaries"]["mean"]
    assert isinstance(mean, dict), type(mean)
    cids = list(blob["context_ids"])
    for c, t in mean.items():
        assert tuple(t.shape) == (N_LAYERS, HIDDEN), (c, tuple(t.shape))
    return mean, cids


def cross_check_summaries(mean: dict[str, torch.Tensor], ctx_id: str) -> float:
    """Re-mean ``answer_spans/<ctx>.pt`` and assert it matches ``summaries['mean'][ctx]``.

    Returns the max-abs residual (plan §6.0 step 1 / §8 risk (b) guard).
    """
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        HF_DATA_REPO, f"{STORE_PREFIX}/answer_spans/{ctx_id}.pt", repo_type="dataset"
    )
    blob = torch.load(p, map_location="cpu", weights_only=False)
    spans = blob["spans"]  # list of (28, n_tok, 3584) fp16
    n_probes = len(spans)
    accum = torch.zeros(N_LAYERS, HIDDEN, dtype=torch.float32)
    for span in spans:  # (28, n_tok, 3584) fp16
        # vectorized `mean` recipe over answer tokens at every layer: (28, n_tok, H) -> (28, H).
        # Mean in the span's native dtype THEN cast (matches summarize_answer_span(span[li],
        # "mean").float(), which built summaries['mean'] — keeps the cross-check exact).
        accum += span.mean(dim=1).float()
    remean = accum / n_probes
    ref = mean[ctx_id].float()
    resid = float((remean - ref).abs().max())
    assert resid < CROSS_CHECK_TOL, (
        f"cross-check FAIL for {ctx_id}: re-meaned answer_spans vs summaries['mean'] "
        f"‖Δ‖∞={resid:.3e} >= {CROSS_CHECK_TOL} (stale summaries blob?)"
    )
    return resid


def assemble_X_mismatched(mean: dict[str, torch.Tensor], kept_ctx: list[str]) -> np.ndarray:
    """``X_mismatched (N, 28, 3584)`` over ``kept_ctx`` (the matched arm's kept order)."""
    rows = [mean[c].float().numpy() for c in kept_ctx]  # each (28, 3584)
    X = np.stack(rows, axis=0)
    assert X.shape == (len(kept_ctx), N_LAYERS, HIDDEN), X.shape
    return X


def run_recompute(*, smoke: bool, n_smoke_ctx: int = 8) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(E0_PATH) as f:
        e0 = json.load(f)
    mean, store_cids = load_mismatched_v0_summaries()

    ctx_pool = store_cids[:n_smoke_ctx] if smoke else store_cids
    logger.info("recompute over %d contexts (smoke=%s)", len(ctx_pool), smoke)

    # cross-check: re-mean one context's answer_spans vs summaries['mean']
    cc_resid = cross_check_summaries(mean, ctx_pool[0])
    logger.info("cross-check %s: ‖Δ‖∞=%.3e < %.0e OK", ctx_pool[0], cc_resid, CROSS_CHECK_TOL)

    results: dict[str, dict] = {}
    for behavior in BEHAVIORS:
        # kept-ctx order is the matched-arm convention: e0 rate over ctx_pool
        y, kept = e0_rate_vector(e0, behavior, ctx_pool)
        X = assemble_X_mismatched(mean, kept)
        out = _run_ridge_pipeline(X, y)
        results[behavior] = {
            "mismatched_ridge_rho": out["rho"],
            "chosen_layer": out["chosen_layer"],
            "per_layer_rho": out["per_layer_rho"],
            "n_contexts": len(kept),
            "kept_ctx_ids": kept,
        }
        logger.info(
            "[%s] mismatched_ridge_rho=%.4f @ layer %d (n=%d)",
            behavior,
            out["rho"],
            out["chosen_layer"],
            len(kept),
        )

    payload = {
        "task": 761,
        "arm": "mismatched_recompute_in_run",
        "source": f"{STORE_PREFIX}/v0_summaries.pt['summaries']['mean']",
        "cross_check_resid_inf": cc_resid,
        "recipe_fingerprint": RECIPE_FINGERPRINT,
        "results": results,
        "metadata": {"recomputed_at": _now_iso(), "smoke": smoke, "n_contexts": len(ctx_pool)},
    }
    out_path = OUT_DIR / ("mismatched_ridge_smoke.json" if smoke else "mismatched_ridge.json")
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #761 in-run mismatched-ridge recompute")
    ap.add_argument("--smoke", action="store_true", help="8 contexts + cross-check")
    ap.add_argument("--n-smoke-ctx", type=int, default=8)
    args = ap.parse_args()

    out_path = run_recompute(smoke=args.smoke, n_smoke_ctx=args.n_smoke_ctx)

    if args.smoke:
        payload = json.loads(out_path.read_text())
        assert payload["recipe_fingerprint"] == RECIPE_FINGERPRINT
        assert payload["cross_check_resid_inf"] < CROSS_CHECK_TOL
        for behavior in BEHAVIORS:
            r = payload["results"][behavior]
            assert r["mismatched_ridge_rho"] is not None, behavior
            assert 0 <= r["chosen_layer"] < N_LAYERS, r["chosen_layer"]
        logger.info(
            "[smoke] PASS — cross-check resid %.3e; mismatched rho: %s",
            payload["cross_check_resid_inf"],
            {b: round(payload["results"][b]["mismatched_ridge_rho"], 4) for b in BEHAVIORS},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

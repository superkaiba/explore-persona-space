"""Layer-robustness re-read of the #761 matched-vs-mismatched predictor (0-GPU).

Round-2 interpretation-critic ask (both Claude + Codex): the headline Δρ point
estimates are read at the ARGMAX-over-28-layers layer, and the comparison arms
select very different (and sometimes very early) layers — matched 21/18/20,
mismatched 19/4/8, same-N 2/15/0. An argmax-over-28 on n=50 landing on layer
0/2/4 is a hallmark of a chance peak, which would DEFLATE the comparison
baseline and thereby INFLATE the reported Δρ. This is the disambiguating check:
does the matched-vs-mismatched gain survive at a FIXED layer (no per-arm
argmax selection) and at the across-layer MEDIAN?

Re-reads ONLY committed / cached tensors — the matched capture .pt shards, the
#658 mismatched `v0_summaries.pt`, and the cached UltraChat answer spans for the
same-N arm — through the EXACT same loaders + `_run_ridge_pipeline` the paired
bootstrap uses (so the recipe cannot silently diverge). NO new data, NO GPU, NO
paired bootstrap (which is the expensive part). Emits
`eval_results/issue_761/layer_robustness.json`.

Three reads per (arm, behavior):
  - ``argmax``: max-over-28 held-out LOCO ρ (the headline; re-derived here to
    confirm it matches `matched_predictor_results.json`).
  - ``fixed_L14``: ρ at layer 14 (middle of the 28-layer stack) — no per-arm
    selection, so a chance early-layer peak in the comparison arm cannot inflate
    the gap.
  - ``median``: median of the 28 per-layer ρ (all layers with a valid ρ) — the
    across-layer central tendency, robust to a single lucky layer.

The matched-minus-mismatched and matched-minus-same-N Δρ are reported at the
fixed layer and at the median alongside the argmax headline: if the sign +
rough magnitude hold at the fixed layer, the argmax read is safe.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from issue761_common import (
    BEHAVIORS,
    N_LAYERS,
    RECIPE_FINGERPRINT,
    _run_ridge_pipeline,
    e0_rate_vector,
)
from issue761_paired_bootstrap import (
    assemble_X_mismatched_local,
    build_samen_X,
    load_matched_shard,
    matched_n_from_shard,
    matched_v0_from_shard,
)
from issue761_recompute_mismatched_ridge import (
    load_mismatched_v0_summaries,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue761_layer_robustness")

REPO_ROOT = Path(__file__).resolve().parent.parent
E0_PATH = REPO_ROOT / "eval_results" / "issue_658" / "E0_expression.json"
OUT_DIR = REPO_ROOT / "eval_results" / "issue_761"
OUT_PATH = OUT_DIR / "layer_robustness.json"

FIXED_LAYER = 14  # middle of the 28-layer transformer stack (0..27)


def _arm_layer_summary(X_by_layer: np.ndarray, y: np.ndarray) -> dict:
    """argmax / fixed-L14 / across-layer-median held-out LOCO ρ for one arm.

    Runs the SHARED ``_run_ridge_pipeline`` once (it returns the full 28-layer
    ρ sweep + the argmax layer), then reads the three summaries off that one
    curve — no re-fit per summary.
    """
    out = _run_ridge_pipeline(X_by_layer, y)
    per_layer = out["per_layer_rho"]  # list[float | None], len 28
    assert len(per_layer) == N_LAYERS, (len(per_layer), N_LAYERS)
    valid = [r for r in per_layer if r is not None]
    fixed = per_layer[FIXED_LAYER]
    return {
        "argmax_rho": float(out["rho"]),
        "argmax_layer": int(out["chosen_layer"]),
        "fixed_layer": FIXED_LAYER,
        "fixed_layer_rho": None if fixed is None else float(fixed),
        "median_rho": float(np.median(valid)) if valid else None,
        "per_layer_rho": [None if r is None else float(r) for r in per_layer],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(E0_PATH) as f:
        e0 = json.load(f)

    mm_mean, store_cids = load_mismatched_v0_summaries()
    ctx_pool = store_cids

    behaviors_out: dict[str, dict] = {}
    for behavior in BEHAVIORS:
        logger.info("[%s] loading matched shard + building arms", behavior)
        shard = load_matched_shard(behavior, smoke=False)
        matched_v0 = matched_v0_from_shard(shard)
        matched_n_by_ctx = matched_n_from_shard(shard)

        # kept contexts = present in BOTH matched capture AND e0 rate, store-pool order
        _, kept_e0 = e0_rate_vector(e0, behavior, ctx_pool)
        kept = [c for c in kept_e0 if c in matched_v0]
        y = np.array([e0["e0"][c][behavior]["rate"] for c in kept], dtype=np.float64)
        n = len(kept)

        X_matched = np.stack([matched_v0[c] for c in kept], axis=0)  # (N, 28, H)
        X_mismatched = assemble_X_mismatched_local(mm_mean, kept)
        X_samen, samen_flags = build_samen_X(matched_n_by_ctx, behavior, kept)

        logger.info("[%s] n=%d — running the 3-arm layer sweep", behavior, n)
        arms = {
            "matched": _arm_layer_summary(X_matched, y),
            "mismatched": _arm_layer_summary(X_mismatched, y),
            "samen": _arm_layer_summary(X_samen, y),
        }

        # Δρ at each layer-read (matched − comparison), for BOTH comparison arms.
        def _delta(read_key: str, comp_arm: str) -> float | None:
            a = arms["matched"][read_key]
            b = arms[comp_arm][read_key]
            return None if (a is None or b is None) else float(a - b)

        behaviors_out[behavior] = {
            "n_contexts": n,
            "arms": arms,
            "delta_matched_minus_mismatched": {
                "argmax": _delta("argmax_rho", "mismatched"),
                "fixed_L14": _delta("fixed_layer_rho", "mismatched"),
                "median": _delta("median_rho", "mismatched"),
            },
            "delta_matched_minus_samen": {
                "argmax": _delta("argmax_rho", "samen"),
                "fixed_L14": _delta("fixed_layer_rho", "samen"),
                "median": _delta("median_rho", "samen"),
            },
            "samen_draw_with_replacement_any": bool(samen_flags),
        }
        logger.info(
            "[%s] matched argmax=%.3f fixedL14=%s median=%s | "
            "mismatched argmax=%.3f fixedL14=%s median=%s | "
            "Δ(m-mm) argmax=%s fixedL14=%s median=%s",
            behavior,
            arms["matched"]["argmax_rho"],
            _fmt(arms["matched"]["fixed_layer_rho"]),
            _fmt(arms["matched"]["median_rho"]),
            arms["mismatched"]["argmax_rho"],
            _fmt(arms["mismatched"]["fixed_layer_rho"]),
            _fmt(arms["mismatched"]["median_rho"]),
            _fmt(behaviors_out[behavior]["delta_matched_minus_mismatched"]["argmax"]),
            _fmt(behaviors_out[behavior]["delta_matched_minus_mismatched"]["fixed_L14"]),
            _fmt(behaviors_out[behavior]["delta_matched_minus_mismatched"]["median"]),
        )

    result = {
        "task": 761,
        "analysis": "layer_robustness_re_read",
        "note": (
            "Round-2 interp-critic ask: argmax-layer selection may inflate Δρ because "
            "the comparison arms select very different (and sometimes early) layers. "
            "This re-reads matched/mismatched/same-N ρ at the argmax layer, a fixed "
            "mid-stack layer (14), and the across-layer median. Re-reads only cached "
            "tensors; no new data, no paired bootstrap."
        ),
        "fixed_layer": FIXED_LAYER,
        "recipe_fingerprint": RECIPE_FINGERPRINT,
        "behaviors": behaviors_out,
        "generated_at": datetime.now(UTC).isoformat(),
        "hf_home": os.environ.get("HF_HOME"),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s", OUT_PATH)


def _fmt(x: float | None) -> str:
    return "None" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    main()

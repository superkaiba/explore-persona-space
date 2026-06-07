"""Task #507 Phase 4 - cross-arm analyzer (7B vs 72B).

Two responsibilities:

1. Per-arm regression. The predictor_jsdiv_470 phase5_regress module already
   produces ``regression.json`` per arm; we don't re-implement it. The
   72B sweep writes its predictor outputs into
   ``eval_results/issue_507/predictor_72b/`` (mirroring #470's layout under
   ``eval_results/issue_470/``); phase5_regress is invoked once per arm with
   the appropriate OUTPUT_BASE overlay (env var ``PREDICTOR_OUTPUT_BASE``
   if present, defaulting to #470's path).

2. Cross-arm |rho_72B| - |rho_7B| paired bootstrap. Reads the per-source
   bootstrap distributions from both arms' regression.json and reports the
   95% CI of the paired |rho| difference. Per plan v2 section 6.1 analyzer
   hand-off: when the verdict-grid lands HIGH or MODERATE, this CI is the
   downgrade check (HIGH downgrades to MODERATE if the CI overlaps zero).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.sycophancy_scale_507 import (
    HEADLINE_LAYER_BY_ARCH,
    SOURCE_PERSONAS_507,
)

log = logging.getLogger("sycophancy_scale_507.analyze_507")

# Bootstrap N for cross-arm paired |rho| difference. Matches #411 / #470's
# 10k convention (the cross-arm test inherits the parent's stat budget).
DEFAULT_CROSS_ARM_BOOTSTRAP_N = 10000

# The three primary predictors per plan v2 section 6.1. Each maps to a label
# that phase5_regress.py emits inside the top-level ``predictors`` dict (NOT
# ``per_predictor`` — that was the round-1 schema-mismatch bug per code-review
# Critical 4). Concrete labels:
#   - "cosine_l20_baseline" — baseline next-token cosine, layer 20 (7B) /
#     equivalent depth (72B headline layer 57 via PREDICTOR_HEADLINE_LAYER).
#   - "cosine_response_headline" — response-token cosine at HEADLINE_LAYER
#     (env-parametrized).
#   - "M_js" — Jensen-Shannon mixture distance, the JS sequence predictor.
PRIMARY_PREDICTORS: tuple[str, ...] = (
    "cosine_l20_baseline",
    "cosine_response_headline",
    "M_js",
)


def _load_regression(path: Path, arm_label: str) -> dict:
    """Read an arm's regression.json; fail-loud if it doesn't exist or is malformed.

    Round-2 fix per code-review Critical 4: the writer's top-level key is
    ``predictors`` (NOT ``per_predictor``). Accept either spelling to stay
    compatible with any older committed regression.json that pre-dated this
    canonicalization, but emit a warning so the drift is visible.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"regression.json for {arm_label} missing at {path}. The Phase 5 "
            f"regress step for this arm must complete before analyze_507."
        )
    with open(path) as f:
        payload = json.load(f)
    if "predictors" not in payload:
        # Back-compat: tolerate the older "per_predictor" spelling if some
        # committed file uses it; normalize so downstream lookups always
        # find "predictors".
        if "per_predictor" in payload:
            log.warning(
                "regression.json at %s uses legacy 'per_predictor' key; "
                "renaming to 'predictors' in-memory for compatibility.",
                path,
            )
            payload["predictors"] = payload["per_predictor"]
        else:
            raise RuntimeError(
                f"regression.json for {arm_label} at {path} is missing the "
                f"'predictors' key — Phase 5 produced a kill-criterion / halted "
                f"payload, not a full regression. Cannot cross-arm compare."
            )
    return payload


def _paired_rho_bootstrap(
    *,
    per_source_7b: dict[str, dict],
    per_source_72b: dict[str, dict],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Paired bootstrap of |rho_72B| - |rho_7B|, resampling at the source level.

    The two arms share the same 6 sources by construction (single-variable
    rule); we bootstrap by drawing 6 sources with replacement and recomputing
    the mean of |rho_72B(s)| - |rho_7B(s)| on each draw.

    Args:
        per_source_7b: phase5_regress's ``per_source`` block for the 7B arm.
            Shape: ``{source_name: {"rho": float, "n": int, ...}, ...}``.
        per_source_72b: same shape for the 72B arm.
        n_boot: number of bootstrap iterations.
        rng: np.random.default_rng for reproducibility.

    Returns:
        Dict with point estimate, 95% CI bounds, and overlap-zero flag.
    """
    # Build paired |rho| differences per source. Sources missing from
    # either arm are dropped with a warning (we cannot pair an absent rho).
    sources_in_both = [s for s in SOURCE_PERSONAS_507 if s in per_source_7b and s in per_source_72b]
    if len(sources_in_both) < 3:
        return {
            "n_sources_paired": len(sources_in_both),
            "note": "insufficient_paired_sources",
            "point_estimate": None,
            "ci_lower_95": None,
            "ci_upper_95": None,
            "overlaps_zero": True,
        }

    abs_diffs: list[float] = []
    for s in sources_in_both:
        rho_7b = per_source_7b[s].get("rho")
        rho_72b = per_source_72b[s].get("rho")
        if rho_7b is None or rho_72b is None:
            log.warning(
                "Source %s has rho=None for at least one arm (7B=%s, 72B=%s); skipping",
                s,
                rho_7b,
                rho_72b,
            )
            continue
        abs_diffs.append(abs(float(rho_72b)) - abs(float(rho_7b)))
    n_paired = len(abs_diffs)
    if n_paired < 3:
        return {
            "n_sources_paired": n_paired,
            "note": "insufficient_paired_sources_after_none_filter",
            "point_estimate": None,
            "ci_lower_95": None,
            "ci_upper_95": None,
            "overlaps_zero": True,
        }

    arr = np.array(abs_diffs, dtype=float)
    point = float(arr.mean())
    # Source-level bootstrap (resample sources, NOT cells).
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n_paired, size=n_paired)
        boot_means[i] = float(arr[idx].mean())
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    overlaps_zero = ci_lo <= 0.0 <= ci_hi

    return {
        "n_sources_paired": n_paired,
        "abs_diff_per_source": dict(zip(sources_in_both, abs_diffs, strict=True)),
        "point_estimate": point,
        "ci_lower_95": ci_lo,
        "ci_upper_95": ci_hi,
        "overlaps_zero": overlaps_zero,
        "n_boot": n_boot,
    }


def cross_arm_compare(
    *,
    regression_7b: Path,
    regression_72b: Path,
    output_path: Path,
    n_boot: int = DEFAULT_CROSS_ARM_BOOTSTRAP_N,
    seed: int = 42,
) -> dict[str, object]:
    """Build the cross-arm |rho| comparison report.

    Output structure::

        {
          "arms": {"7b": {"regression_path": ...}, "72b": {...}},
          "per_predictor": {
              "<predictor_label>": {
                  "rho_7b_per_source": {...},
                  "rho_72b_per_source": {...},
                  "paired_abs_rho_diff_72b_minus_7b": {
                      "point_estimate": float,
                      "ci_lower_95": float,
                      "ci_upper_95": float,
                      "overlaps_zero": bool,
                      ...
                  }
              }, ...
          },
          "verdict_grid_inputs": {  # the three rows of the §6.1 verdict grid
              "within_source_rho_72b_geq_0p20_with_p_lt_0p05": bool,
              "72b_beats_base_rate_null": bool,
              "72b_pooled_rho_geq_7b_plus_0p10": bool,
              "verdict": "HIGH" | "MODERATE" | "LOW" | "floor" | "clean_negative",
              "downgrade_high_to_moderate_via_cross_arm_ci": bool,
          },
          "metadata": {...}
        }

    Per plan v2 section 6.1 analyzer hand-off, the downgrade flag is True iff
    the verdict is HIGH AND the cross-arm CI on |rho| overlaps zero. The
    downgraded verdict is stored alongside the original.
    """
    log.info(
        "cross_arm_compare: 7b=%s, 72b=%s, n_boot=%d, seed=%d",
        regression_7b,
        regression_72b,
        n_boot,
        seed,
    )
    reg_7b = _load_regression(regression_7b, "7b")
    reg_72b = _load_regression(regression_72b, "72b")

    rng = np.random.default_rng(seed)
    per_predictor_out: dict[str, dict] = {}
    for pred in PRIMARY_PREDICTORS:
        # phase5_regress writes labels under the top-level "predictors" key
        # (round-2 fix per code-review Critical 4). Each predictor block
        # carries spearman_raw / spearman_source_fe / per_source / etc.
        # Tolerate the legacy "per_predictor" spelling for back-compat with
        # any older committed regression files (handled in _load_regression).
        pred_7b_block = reg_7b.get("predictors", {}).get(pred)
        pred_72b_block = reg_72b.get("predictors", {}).get(pred)
        if pred_7b_block is None or pred_72b_block is None:
            log.warning(
                "cross_arm_compare: predictor %s missing from one arm (7b=%s, 72b=%s); skipping",
                pred,
                pred_7b_block is None,
                pred_72b_block is None,
            )
            per_predictor_out[pred] = {
                "note": "missing_from_one_arm",
                "present_7b": pred_7b_block is not None,
                "present_72b": pred_72b_block is not None,
            }
            continue

        per_source_7b = pred_7b_block.get("per_source", {})
        per_source_72b = pred_72b_block.get("per_source", {})
        paired = _paired_rho_bootstrap(
            per_source_7b=per_source_7b,
            per_source_72b=per_source_72b,
            n_boot=n_boot,
            rng=rng,
        )
        per_predictor_out[pred] = {
            "rho_7b_per_source": {s: per_source_7b[s].get("rho") for s in per_source_7b},
            "rho_72b_per_source": {s: per_source_72b[s].get("rho") for s in per_source_72b},
            "paired_abs_rho_diff_72b_minus_7b": paired,
        }

    # Verdict-grid inputs per plan v2 section 6.1. Read these from the 72B
    # regression's per-source means + the cross-arm comparison.
    verdict_inputs = _compute_verdict_grid_inputs(
        reg_7b=reg_7b,
        reg_72b=reg_72b,
        per_predictor_cross=per_predictor_out,
    )

    out: dict[str, object] = {
        "arms": {
            "7b": {"regression_path": str(regression_7b)},
            "72b": {
                "regression_path": str(regression_72b),
                "headline_layer": HEADLINE_LAYER_BY_ARCH["72b"],
            },
        },
        "per_predictor": per_predictor_out,
        "verdict_grid_inputs": verdict_inputs,
        "metadata": {
            "script": "sycophancy_scale_507.analyze_507",
            "n_boot": n_boot,
            "seed": seed,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote cross-arm comparison to %s", output_path)
    return out


def _compute_verdict_grid_inputs(
    *,
    reg_7b: dict,
    reg_72b: dict,
    per_predictor_cross: dict[str, dict],
) -> dict[str, object]:
    """Compute the 3 boolean inputs to plan v2 §6.1's verdict grid.

    1. 72B within-source rho >= +0.20 with p<0.05 (per-source avg across
       6 sources, on the headline predictor).
    2. 72B beats base-rate null (paired |Delta-rho| CI excludes 0 on the
       headline predictor).
    3. 72B pooled-138 source-FE rho >= #470's published 7B value + 0.10.

    Returns the inputs + the verdict + the downgrade-via-cross-arm-CI flag.
    """
    # Round-2 fix per code-review Critical 4: headline predictor lookup
    # uses "cosine_response_headline" inside top-level "predictors" dict.
    # PRIMARY_PREDICTORS[1] = "cosine_response_headline" — the headline-layer
    # response-token cosine (layer 21 at 7B / 57 at 72B via env override).
    headline_pred = "cosine_response_headline"
    headline_72b = reg_72b.get("predictors", {}).get(headline_pred, {})
    headline_7b = reg_7b.get("predictors", {}).get(headline_pred, {})

    # Input 1: per-source rho mean >= +0.20 + per-source p < 0.05 (averaged).
    # Round-2 fix per code-review Critical 4: phase5_regress emits per-source
    # rows with key "p" (two-tailed Spearman p-value), NOT "p_one_tail".
    per_source_72b = headline_72b.get("per_source", {})
    rho_vals = [v["rho"] for v in per_source_72b.values() if v.get("rho") is not None]
    p_vals = [v["p"] for v in per_source_72b.values() if v.get("p") is not None]
    mean_rho_72b = float(np.mean(rho_vals)) if rho_vals else None
    mean_p_72b = float(np.mean(p_vals)) if p_vals else None
    input_1 = (
        mean_rho_72b is not None
        and mean_p_72b is not None
        and mean_rho_72b >= 0.20
        and mean_p_72b < 0.05
    )

    # Input 2: paired |Delta-rho| CI excludes 0 against base-rate null.
    # Round-2 fix per code-review Critical 4: phase5_regress writes
    # ``paired_bootstrap_delta_rho_vs_base_rate`` at the TOP LEVEL of the
    # regression payload (one block per arm), not per-predictor. Read it
    # from reg_72b's root.
    paired_vs_base = reg_72b.get("paired_bootstrap_delta_rho_vs_base_rate", {})
    if paired_vs_base.get("ci_low_95") is not None:
        # Manual overlap-zero check; phase5_regress uses ci_low_95 / ci_high_95.
        ci_lo = paired_vs_base.get("ci_low_95")
        ci_hi = paired_vs_base.get("ci_high_95")
        if ci_lo is not None and ci_hi is not None:
            overlaps_zero = ci_lo <= 0.0 <= ci_hi
            input_2 = not overlaps_zero
        else:
            input_2 = None
    else:
        input_2 = None

    # Input 3: 72B pooled rho >= 7B pooled rho + 0.10.
    pooled_72b = headline_72b.get("spearman_source_fe", {}).get("rho")
    pooled_7b = headline_7b.get("spearman_source_fe", {}).get("rho")
    if pooled_72b is not None and pooled_7b is not None:
        input_3 = float(pooled_72b) >= float(pooled_7b) + 0.10
    else:
        input_3 = None

    # Map (input_1, input_2, input_3) to verdict per the table.
    if input_1 and input_2 and input_3:
        verdict = "HIGH"
    elif input_1 and input_2 and input_3 is False:
        verdict = "MODERATE"
    elif input_1 and input_2 is False and input_3:
        verdict = "LOW"
    elif input_1 is False:
        verdict = "clean_negative_or_floor"
    else:
        verdict = "ambiguous"

    # Cross-arm downgrade check: HIGH -> MODERATE iff the |rho_72B|-|rho_7B|
    # paired CI on the headline predictor overlaps zero.
    # Round-2 fix per code-review Critical 4: use "cosine_response_headline"
    # consistently as the headline predictor key.
    headline_paired = per_predictor_cross.get("cosine_response_headline", {}).get(
        "paired_abs_rho_diff_72b_minus_7b", {}
    )
    cross_overlaps_zero = headline_paired.get("overlaps_zero")
    downgrade = verdict == "HIGH" and cross_overlaps_zero is True
    final_verdict = "MODERATE" if downgrade else verdict

    return {
        "input_1_within_source_rho_72b_geq_0p20_with_p_lt_0p05": input_1,
        "input_2_72b_beats_base_rate_null": input_2,
        "input_3_72b_pooled_rho_geq_7b_plus_0p10": input_3,
        "mean_per_source_rho_72b_headline": mean_rho_72b,
        "mean_per_source_p_72b_headline": mean_p_72b,
        "pooled_source_fe_rho_72b": pooled_72b,
        "pooled_source_fe_rho_7b": pooled_7b,
        "verdict_pre_downgrade": verdict,
        "downgrade_high_to_moderate_via_cross_arm_ci": downgrade,
        "verdict_final": final_verdict,
        "cross_arm_ci_overlaps_zero_on_headline": cross_overlaps_zero,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--regression-7b",
        type=Path,
        required=True,
        help="Path to the 7B regression.json (from #470 or the Phase 5 re-run).",
    )
    parser.add_argument(
        "--regression-72b",
        type=Path,
        required=True,
        help="Path to the 72B regression.json produced by phase5_regress on #507's outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the cross-arm comparison JSON.",
    )
    parser.add_argument("--n-boot", type=int, default=DEFAULT_CROSS_ARM_BOOTSTRAP_N)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=analyze_507] %(message)s")

    cross_arm_compare(
        regression_7b=args.regression_7b,
        regression_72b=args.regression_72b,
        output_path=args.output,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

"""Task #612 predictor-v3 — coordination helpers (CPU, pure functions).

The cross-source coordination logic for Bucket 2 (yield fix) and Bucket 3
arithmetic, kept OUT of the GPU-bound pool builder so it is unit-testable and
exercised by the CPU smoke:

  * ``source_baseline_summary`` — fold a source's base-pass judgments (produced
    by eval_panel + the judge) into the source-side base-prior record
    (yield-risk + install covariate, plan v3 §4.2 "Baseline propensity").
  * ``yield_decision`` — apply the 80% floor (KEEP >=160, DROP below) to one
    source's realized fill count (§4.2).
  * ``equalize_down`` — across KEPT sources, set floor_N = min realized fill,
    and the proportional negative count that preserves the v1 ~1:2.5
    positives:negatives ratio (§4.2 / §4.6).
  * ``kept_sources_or_kill`` — the §5 KILL check (>=3 sources must clear the
    floor for the H2 ">=3/4" falsification to be meaningful).
  * ``save_steps_for`` — Bucket 3 sub-epoch checkpoint cadence arithmetic
    (save_steps = ceil(total_steps / 8), finer when total_steps is small).

No GPU, no API, no I/O beyond reading judgment JSONs. The dispatcher composes
these around the eval/judge/train phases.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# The v1 frozen pool is 200 positives : 500 total negatives = 1 : 2.5.
V1_POS = 200
V1_NEG = 500


def source_baseline_summary(judgments_dir: Path, source: str) -> dict:
    """Summarize a source's base-pass judgments into the base-prior record.

    ``judgments_dir`` holds ``<source>.json`` in the judge_pass schema
    (verdicts: [{agreed, error, ...}]). Returns the agreement rate over
    VERIFIED verdicts (error-free), the verdict count, and a yield-risk flag.
    Fail-loud when the judgments file is absent (the source-side read must run
    BEFORE elicitation — a missing file is a pipeline-ordering bug, §4.2).
    """
    path = judgments_dir / f"{source}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"source-side baseline judgments missing: {path} — the source baseline "
            f"read must run before elicitation (plan v3 §4.2)"
        )
    payload = json.loads(path.read_text())
    verdicts = payload["verdicts"]
    verified = [v for v in verdicts if not v.get("error")]
    if not verified:
        raise ValueError(f"{path}: zero verified verdicts (all errored?) — cannot read base prior")
    n_yes = sum(1 for v in verified if v.get("agreed"))
    rate = n_yes / len(verified)
    from explore_persona_space.experiments.sycophancy_onpolicy_612 import V3_YIELD_RISK

    return {
        "source": source,
        "base_agreement_rate": rate,
        "n_verified_verdicts": len(verified),
        "n_total_verdicts": len(verdicts),
        "yield_risk_class": V3_YIELD_RISK.get(source, "high"),
    }


def yield_decision(source: str, n_filled: int) -> dict:
    """Apply the 80% floor to one source's realized positive fill count."""
    from explore_persona_space.experiments.sycophancy_onpolicy_612 import (
        V3_YIELD_FLOOR,
        V3_YIELD_TARGET,
    )

    kept = n_filled >= V3_YIELD_FLOOR
    return {
        "source": source,
        "n_filled": n_filled,
        "target": V3_YIELD_TARGET,
        "floor": V3_YIELD_FLOOR,
        "decision": "keep" if kept else "drop",
        "fill_fraction": n_filled / V3_YIELD_TARGET,
    }


def kept_sources_or_kill(decisions: dict[str, dict]) -> list[str]:
    """Return the KEPT sources, or raise the §5 KILL if fewer than the minimum.

    ``decisions`` maps source -> yield_decision record. The bake-off needs
    >=V3_MIN_KEPT_SOURCES sources for the H2 ">=3/4" falsification (§5 KILL).
    """
    from explore_persona_space.experiments.sycophancy_onpolicy_612 import V3_MIN_KEPT_SOURCES

    kept = sorted(s for s, d in decisions.items() if d["decision"] == "keep")
    if len(kept) < V3_MIN_KEPT_SOURCES:
        dropped = sorted(s for s, d in decisions.items() if d["decision"] == "drop")
        raise YieldKill(
            f"only {len(kept)}/{len(decisions)} sources cleared the 80% floor "
            f"(kept={kept}, dropped={dropped}); the predictor bake-off needs "
            f">={V3_MIN_KEPT_SOURCES} for the H2 '>=3/4' call (plan v3 §5 KILL) — "
            f"STOP after data-gen, fold the available sources descriptively."
        )
    return kept


class YieldKill(RuntimeError):
    """Fewer than V3_MIN_KEPT_SOURCES sources cleared the floor (§5 KILL)."""


def equalize_down(kept_fills: dict[str, int]) -> dict:
    """floor_N = min over kept sources of realized positive fill; proportional
    negative count preserving the v1 1:2.5 positives:negatives ratio (§4.2/§4.6).

    Returns {floor_n_positives, n_negatives_total, ratio_pos_to_neg}. Every kept
    source trains on EXACTLY floor_n_positives rows (surplus discarded) so per-
    source N is identical — variable N is a dose confound (#601).
    """
    if not kept_fills:
        raise ValueError("equalize_down called with no kept sources")
    floor_n = min(kept_fills.values())
    # Preserve the v1 ratio: negatives = positives * (V1_NEG / V1_POS) = pos * 2.5.
    n_neg = round(floor_n * V1_NEG / V1_POS)
    return {
        "floor_n_positives": floor_n,
        "n_negatives_total": n_neg,
        "ratio_pos_to_neg": round(n_neg / floor_n, 3),
        "kept_fills": dict(sorted(kept_fills.items())),
        "v1_ratio_pos_to_neg": round(V1_NEG / V1_POS, 3),
    }


def save_steps_for(n_train_rows: int, *, epochs: int = 3, eff_batch: int = 16) -> dict:
    """Bucket 3 sub-epoch cadence (plan v3 §4.3 / N4 / §14).

    save_steps = ceil(total_steps / 8); if total_steps is small (< the §7/§14
    threshold), use the finer /12 divisor so >=2 candidate checkpoints still
    bracket the +0.60 band. Never returns 0 (clamped to >=1).
    """
    from explore_persona_space.experiments.sycophancy_onpolicy_612 import (
        V3_SAVE_STEPS_DIVISOR,
        V3_SAVE_STEPS_DIVISOR_SMALL,
        V3_SMALL_STEPS_THRESHOLD,
    )

    steps_per_epoch = math.ceil(n_train_rows / eff_batch)
    total_steps = steps_per_epoch * epochs
    divisor = (
        V3_SAVE_STEPS_DIVISOR_SMALL
        if total_steps < V3_SMALL_STEPS_THRESHOLD
        else V3_SAVE_STEPS_DIVISOR
    )
    save_steps = max(1, math.ceil(total_steps / divisor))
    return {
        "n_train_rows": n_train_rows,
        "eff_batch": eff_batch,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "divisor": divisor,
        "save_steps": save_steps,
        "expected_n_checkpoints": total_steps // save_steps,
    }


__all__ = [
    "YieldKill",
    "equalize_down",
    "kept_sources_or_kill",
    "save_steps_for",
    "source_baseline_summary",
    "yield_decision",
]

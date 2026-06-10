# ruff: noqa: RUF001, RUF003  # em-dash + Qwen marker " ※" + × intentional
"""Task #504 — per-cell training-data builder with cell-resolution-driven negs.

#472's `build_cell` picks negatives via `select_negatives_by_geometry` (band
over the whole bank). #504 picks ONE specific persona per arm (Phase 0.5
locks the 4 positioned-N's) and the default-only arm uses qwen_default alone.
This wrapper threads the cell-resolution path:

  1. `arm_negatives_with_counts` resolves (negs, counts) for the cell.
  2. We materialize the rows directly here (positives + negatives), matching
     #472's `build_cell` row shape EXACTLY so #472's downstream consumers
     (collator, eval) read the artifact unchanged.

The row layout is byte-identical to #472's `build_cell`: deterministic
question slots per persona, shuffle by `random.Random(seed)`, marker text
appended to positive completions only. Positives = N_POS_PER_CELL (200) from
source; negatives = sum(counts) split across `negs`.

Marker-leakage measurement rule + contrastive-negatives rule both satisfied:
- POSITIVE row (source): `T_source(q) + R_train[source][q] + "\n\n" + ※` →
  marker-only loss (collator masks to the ※ token + EOS at the post-response
  slot via `MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot
  =True, im_end_token_id=151645)`).
- NEGATIVE row (qwen_default OR positioned N): `T_neg(q) + R_train[neg][q]`
  with NO appended marker → under marker-only loss the only loss-bearing token
  is EOS at the post-response slot, training "after a response under this
  persona, emit EOS, NOT the marker."

CPU-only.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from explore_persona_space.experiments.contrastive_neg_geometry_472.build_training_data import (
    _has_marker_in_R,
    _make_example,
    _resolve_response,
    _sample_question_slots,
)
from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
    CELL_SPECS_504,
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_SEP,
    MARKER_TEXT,
    N_POS_PER_CELL,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.contrastive_neg_geometry_504.cell_resolution import (
    arm_negatives_with_counts,
)

log = logging.getLogger("issue_504.build_training_data")


def build_cell_504(
    cell_slug: str,
    output_path: Path,
    *,
    r_train: dict[str, dict[str, dict]],
    arm_to_positioned_n: dict[str, str],
    q_train: list[str],
    persona_bank: dict[str, str],
    source: str = SOURCE_PERSONA,
    marker_text: str = MARKER_TEXT,
    smoke_mid_band_n: str | None = None,
    seed: int = 42,
) -> Path:
    """Build the per-cell training JSONL for a #504 cell.

    Args:
        cell_slug: e.g. "c504_near" or "c504_smoke_r8".
        output_path: JSONL output path.
        r_train: on-policy R artifact (persona → q → {response_text, ...}).
        arm_to_positioned_n: {positioned_arm_slug: positioned_N_persona} for
            the 4 main arms (output of Phase 0.5 select_positioned_negatives
            re-keyed by arm slug). May be empty for smoke cells.
        q_train: Q_train question list.
        persona_bank: name → system prompt for resolving positive/negative prompts.
        source: source persona.
        marker_text: marker string appended to positive completions only.
        smoke_mid_band_n: the smoke cells' positioned-N (required for smoke).
        seed: base seed for per-persona seed salting.

    Returns:
        output_path. Raises on marker-in-negative contamination, missing R,
        or row-count mismatch.
    """
    spec = next((c for c in CELL_SPECS_504 if c[0] == cell_slug), None)
    if spec is None:
        raise KeyError(f"Unknown #504 cell slug {cell_slug!r}")
    _slug, plain_name, placement, n_neg_personas, _neg_ex_default, in_pooled = spec

    negs, counts = arm_negatives_with_counts(
        cell_slug,
        arm_to_positioned_n,
        smoke_mid_band_n=smoke_mid_band_n,
        n_pos=N_POS_PER_CELL,
    )
    if len(negs) != n_neg_personas:
        raise AssertionError(
            f"[{cell_slug}] negatives_for_cell_504 returned {len(negs)} personas, "
            f"CELL_SPECS_504 expected n_neg_personas={n_neg_personas} (placement={placement!r})."
        )
    if source not in persona_bank:
        raise KeyError(f"[{cell_slug}] source {source!r} not in persona bank.")
    for neg in negs:
        if neg not in persona_bank:
            raise KeyError(f"[{cell_slug}] negative persona {neg!r} not in persona bank.")

    total_neg = sum(counts)
    log.info(
        "[%s] Building cell '%s': placement=%s, %d pos (source=%s), "
        "%d neg personas × counts=%s = %d neg rows; negatives=%s",
        cell_slug,
        plain_name,
        placement,
        N_POS_PER_CELL,
        source,
        len(negs),
        counts,
        total_neg,
        negs,
    )

    examples: list[dict] = []

    # ── Positive rows (source persona). ──────────────────────────────────────
    source_prompt = persona_bank[source]
    pos_rng = random.Random(seed)
    pos_questions = _sample_question_slots(q_train, N_POS_PER_CELL, pos_rng)
    n_marker_in_positive_R = 0
    for q in pos_questions:
        r_text, r_ids = _resolve_response(r_train, source, q, cell_slug)
        if _has_marker_in_R(r_text, r_ids, marker_text):
            n_marker_in_positive_R += 1
            raise AssertionError(
                f"[{cell_slug}] positive row for source={source!r}, q={q!r} already "
                f"contains the marker in R BEFORE we append it — would produce two "
                f"markers. Phase 1 r-generate should have aborted; the R artifact is stale."
            )
        examples.append(_make_example(source_prompt, q, f"{r_text}{MARKER_SEP}{marker_text}"))
    n_positive = len(examples)

    # ── Negative rows (cell-resolved personas, no marker). ───────────────────
    n_marker_in_negative_R = 0
    for j_idx, (neg_name, n_ex) in enumerate(zip(negs, counts, strict=True)):
        neg_prompt = persona_bank[neg_name]
        neg_rng = random.Random(seed + 1000 + j_idx)
        neg_questions = _sample_question_slots(q_train, n_ex, neg_rng)
        for q in neg_questions:
            r_text, r_ids = _resolve_response(r_train, neg_name, q, cell_slug)
            if _has_marker_in_R(r_text, r_ids, marker_text):
                n_marker_in_negative_R += 1
                raise AssertionError(
                    f"[{cell_slug}] negative row for persona={neg_name!r}, q={q!r} has "
                    f"marker contamination in R — would silently train the model to emit "
                    f"the marker after a bystander response."
                )
            examples.append(_make_example(neg_prompt, q, r_text))
    n_negative = len(examples) - n_positive

    # ── Final deterministic shuffle. ─────────────────────────────────────────
    random.Random(seed).shuffle(examples)

    expected_total = N_POS_PER_CELL + total_neg
    if len(examples) != expected_total:
        raise AssertionError(
            f"[{cell_slug}] row count mismatch: got {len(examples)}, expected "
            f"{expected_total} ({N_POS_PER_CELL} pos + {total_neg} neg)."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    manifest: dict[str, Any] = {
        "cell_slug": cell_slug,
        "plain_name": plain_name,
        "placement": placement,
        "task_id": 504,
        "source_persona": source,
        "negative_personas": negs,
        "negative_counts": counts,
        "n_neg_personas": len(negs),
        "pos_ex": N_POS_PER_CELL,
        "n_total_rows": len(examples),
        "n_positive_rows": n_positive,
        "n_negative_rows": n_negative,
        "pos_to_neg_ratio": (n_positive / n_negative) if n_negative else None,
        "ratio_note": (
            "1:1 contrastive ratio (plan §4.6). Positioned arms split negs 100+100; "
            "default-only arm uses 200 from qwen_default alone (row-count matched)."
        ),
        "in_pooled_regression": in_pooled,
        "marker_text": marker_text,
        "marker_token_id_expected": EXPECTED_MARKER_TOKEN_ID,
        "seed": seed,
        "marker_in_R_counts": {
            "positive": n_marker_in_positive_R,
            "negative": n_marker_in_negative_R,
        },
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(
        "[%s] Wrote %d rows (%d pos, %d neg) → %s",
        cell_slug,
        len(examples),
        n_positive,
        n_negative,
        output_path,
    )
    return output_path

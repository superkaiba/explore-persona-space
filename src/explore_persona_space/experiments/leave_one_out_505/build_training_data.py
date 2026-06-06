# ruff: noqa: RUF002, RUF003  # em-dash + marker/Greek tokens + Unicode minus intentional
"""Task #505 Phase 3 — per-cell on-policy contrastive-SFT data with row redistribution.

Forks ``contrastive_neg_geometry_472.build_training_data.build_cell`` with the
§5.3 row-redistribution contract:

  Full set arm (K=6 non-default + qwen_default):
    25 rows × 6 non-default + 50 rows qwen_default = 200 negative rows
  Drop j_i arm (K-1=5 non-default + qwen_default):
    30 rows × 5 non-default + 50 rows qwen_default = 200 negative rows
  No-negatives arm: 0 negative rows.

Total negative rows = 200 across every non-empty arm (held fixed — the
experiment's load-bearing invariant per plan §5.3). Positives stay at 200
across every arm. Marker-in-R contamination guards mirror #472's verbatim.

Row construction (per contrastive-negatives rule + #472):

  POSITIVE row (source villain):
    completion = R_train[villain][q] + "\n\n" + " ※"
    Loss masked to ` ※` token + EOS via MarkerOnlyDataCollator(tail_tokens=0,
    suppress_at_post_response_slot=True, im_end_token_id=151645) downstream.
    R is zero-gradient so the LoRA shifts only the marker; R stays on-policy.

  NEGATIVE row (a non-source persona; qwen_default OR one of the K-set's
  non-default negatives):
    completion = R_train[neg][q]  (NO marker)
    Under the marker-only + suppress_at_post_response_slot loss, the ONLY
    loss-bearing label is the FIRST `<|im_end|>` AFTER R — the SAME slot the
    DV reads. The contrast is positives push log P( ※) up at that slot,
    negatives push it down via EOS competition.

No marker contamination is permitted in any negative R (text + token-id check).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_SEP,
    MARKER_TEXT,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.build_training_data import (
    _has_marker_in_R,
    _make_example,
    _resolve_response,
    _sample_question_slots,
)
from explore_persona_space.experiments.leave_one_out_505 import (
    ALWAYS_INCLUDE_NEGATIVE,
    CELL_SPECS,
    NON_DEFAULT_ROWS_DROP_ARM,
    NON_DEFAULT_ROWS_FULL_SET,
    POS_EX_PER_SOURCE,
    QWEN_DEFAULT_NEG_ROWS,
    TOTAL_NEG_ROWS,
)

log = logging.getLogger("issue_505.build_training_data")


def _persona_salt(persona_name: str, *, j_idx: int = 0) -> int:
    """Deterministic per-persona RNG salt for negative-row question sampling.

    Keyed by the persona NAME via SHA-256 so the salt is invariant to:

      (a) which other personas are present in the same cell (drop / reorder), and
      (b) the current Python process's PYTHONHASHSEED (Python's built-in
          ``hash()`` randomizes string hashing per-process and would silently
          break cross-run reproducibility).

    The ``j_idx`` parameter is accepted but DELIBERATELY IGNORED. It exists so
    the regression test ``test_buggy_jidx_salt_fails_the_invariant`` can
    monkeypatch this helper with a ``j_idx``-using salt to prove the multiset
    invariant test would catch a regression to the round-2 BLOCKER
    ``negative-row-sampling-shifts``.

    Returns:
        An ``int`` salt suitable for ``random.Random(seed * 1000 + salt)``.
    """
    del j_idx  # documented above — not used in the production salt
    return int(hashlib.sha256(persona_name.encode("utf-8")).hexdigest()[:8], 16)


def _resolve_cell_negatives(
    *,
    cell_slug: str,
    non_default_negatives: list[str],
    always_include: str,
) -> tuple[list[tuple[str, int]], int, str | None]:
    """Resolve (persona, n_rows) pairs for a cell, plus n_negatives + dropped_j.

    Returns:
        rows_by_persona: ordered list of (persona, n_rows). qwen_default first
            (when present), then the non-default negatives in their canonical
            order.
        total_neg_rows: sanity-check total (= sum of n_rows across the list).
        dropped_j: the persona name dropped from the K-set for drop-arm cells,
            or None for full-set / no-negatives.
    """
    spec = next((c for c in CELL_SPECS if c[0] == cell_slug), None)
    if spec is None:
        raise KeyError(f"Unknown cell slug {cell_slug!r}; known: {[c[0] for c in CELL_SPECS]}")
    _slug, _name, dropped_j_idx, _in_pooled = spec

    if cell_slug.endswith("_no_negatives"):
        return [], 0, None

    if cell_slug.endswith("_full_set"):
        rows_by_persona = [(always_include, QWEN_DEFAULT_NEG_ROWS)]
        rows_by_persona += [(p, NON_DEFAULT_ROWS_FULL_SET) for p in non_default_negatives]
        total = QWEN_DEFAULT_NEG_ROWS + NON_DEFAULT_ROWS_FULL_SET * len(non_default_negatives)
        if total != TOTAL_NEG_ROWS:
            raise AssertionError(
                f"full-set row total {total} != {TOTAL_NEG_ROWS}; non-default count "
                f"{len(non_default_negatives)} probably mis-set. Expected K_NON_DEFAULT=6."
            )
        return rows_by_persona, total, None

    # Drop-arm path.
    if dropped_j_idx is None:
        raise AssertionError(f"Cell {cell_slug!r} has no dropped_j_idx but isn't a recognized arm.")
    if dropped_j_idx >= len(non_default_negatives):
        raise IndexError(
            f"dropped_j_idx={dropped_j_idx} out of range for "
            f"{len(non_default_negatives)} non-default negatives."
        )
    dropped_j = non_default_negatives[dropped_j_idx]
    remaining = [p for i, p in enumerate(non_default_negatives) if i != dropped_j_idx]
    rows_by_persona = [(always_include, QWEN_DEFAULT_NEG_ROWS)]
    rows_by_persona += [(p, NON_DEFAULT_ROWS_DROP_ARM) for p in remaining]
    total = QWEN_DEFAULT_NEG_ROWS + NON_DEFAULT_ROWS_DROP_ARM * len(remaining)
    if total != TOTAL_NEG_ROWS:
        raise AssertionError(
            f"drop-arm row total {total} != {TOTAL_NEG_ROWS}; "
            f"len(remaining)={len(remaining)} probably mis-set. Expected K_NON_DEFAULT-1=5."
        )
    return rows_by_persona, total, dropped_j


def build_cell_505(
    cell_slug: str,
    output_path: Path,
    *,
    r_train: dict[str, dict[str, dict]],
    non_default_negatives: list[str],
    q_train: list[str],
    persona_bank: dict[str, str],
    source: str = SOURCE_PERSONA,
    marker_text: str = MARKER_TEXT,
    always_include: str = ALWAYS_INCLUDE_NEGATIVE,
    seed: int = 42,
) -> Path:
    """Build the per-cell training JSONL under #505's row-redistribution contract.

    Args:
        cell_slug: e.g. ``c505_full_set``, ``c505_drop_j2``, ``c505_no_negatives``.
        output_path: JSONL output path.
        r_train: on-policy R artifact (persona -> q -> {response_text, response_token_ids, ...}).
        non_default_negatives: the K=6 ordered non-default negatives from the
            §5.4 panel-coverage gate (in spread-quantile order).
        q_train: Q_train question list (20 questions inherited from #472).
        persona_bank: name -> system prompt for ALL personas referenced by this cell.
        source: source persona (villain).
        marker_text: ` ※` (Qwen-2.5-7B id 83399).
        always_include: ``qwen_default`` — always present in non-empty arms.
        seed: base seed for per-persona seed-salting.

    Returns:
        ``output_path``. Sibling ``<output_path>.manifest.json`` carries the
        manifest (negative composition, expected row counts, marker-in-R counts).
    """
    rows_by_persona, total_neg, dropped_j = _resolve_cell_negatives(
        cell_slug=cell_slug,
        non_default_negatives=non_default_negatives,
        always_include=always_include,
    )
    if source not in persona_bank:
        raise KeyError(f"[{cell_slug}] source {source!r} not in persona bank.")
    for p, _ in rows_by_persona:
        if p not in persona_bank:
            raise KeyError(f"[{cell_slug}] negative persona {p!r} not in persona bank.")

    log.info(
        "[%s] Building cell: source=%s, pos=%d, total_neg=%d, dropped_j=%s, neg_rows_by_persona=%s",
        cell_slug,
        source,
        POS_EX_PER_SOURCE,
        total_neg,
        dropped_j,
        rows_by_persona,
    )

    examples: list[dict] = []

    # ── Positive rows (source persona). ──────────────────────────────────────
    source_prompt = persona_bank[source]
    pos_rng = random.Random(seed)
    pos_questions = _sample_question_slots(q_train, POS_EX_PER_SOURCE, pos_rng)
    for q in pos_questions:
        r_text, r_ids = _resolve_response(r_train, source, q, cell_slug)
        if _has_marker_in_R(r_text, r_ids, marker_text):
            raise AssertionError(
                f"[{cell_slug}] positive row for source={source!r}, q={q!r} already "
                f"contains the marker in R BEFORE we append it — would produce two "
                f"markers. Phase 1 r-generate should have aborted; the R artifact is stale."
            )
        examples.append(_make_example(source_prompt, q, f"{r_text}{MARKER_SEP}{marker_text}"))
    n_positive = len(examples)

    # ── Negative rows (qwen_default + non-default K-set or K-1-set). ─────────
    # Per-persona seed salting MUST be keyed by the persona NAME, not by the
    # row's enumeration index in `rows_by_persona`. The within-bystander
    # differential design (plan §13) reads ``ΔG_b(drop-j) − ΔG_b(full_set)``
    # for the SAME bystander b across the two arms — so the q-slot sequence
    # for a retained bystander must match across the full-set and drop-j
    # cells, otherwise training-order randomness confounds Δ-Leakage. Using
    # the enumeration index j_idx as the salt would shift by −1 for every
    # bystander positioned after the dropped index (e.g. bystander at
    # full-set position 4 lands at drop-arm position 3 when an earlier
    # persona is dropped), giving different question shuffles under the
    # same source seed. Hashing the persona NAME makes the salt invariant
    # to drop / reorder. With this fix the 25-vs-30 row-count difference is
    # the only diff between arms — the question-slot sequence is shared
    # up to min(25, 30) = 25; the 5 extra rows in the drop-arm tail do not
    # disturb the shared head.
    #
    # We use SHA-256 (not Python's built-in ``hash()``) because the built-in
    # randomizes string hashing per-process via PYTHONHASHSEED, breaking
    # reproducibility across runs. SHA-256 over the persona name is stable
    # everywhere.
    n_neg_built = 0
    for j_idx, (neg_name, n_rows) in enumerate(rows_by_persona):
        neg_prompt = persona_bank[neg_name]
        persona_salt = _persona_salt(neg_name, j_idx=j_idx)
        neg_rng = random.Random(seed * 1000 + persona_salt)
        neg_questions = _sample_question_slots(q_train, n_rows, neg_rng)
        for q in neg_questions:
            r_text, r_ids = _resolve_response(r_train, neg_name, q, cell_slug)
            if _has_marker_in_R(r_text, r_ids, marker_text):
                raise AssertionError(
                    f"[{cell_slug}] negative row for persona={neg_name!r}, q={q!r} has "
                    f"marker contamination in R — would silently train the model to emit "
                    f"the marker after a bystander response."
                )
            examples.append(_make_example(neg_prompt, q, r_text))
            n_neg_built += 1

    if n_neg_built != total_neg:
        raise AssertionError(
            f"[{cell_slug}] negative row count mismatch: got {n_neg_built}, expected "
            f"{total_neg}. Row resolution / question sampling broken."
        )

    # ── Final deterministic shuffle. ─────────────────────────────────────────
    random.Random(seed).shuffle(examples)

    expected_total = POS_EX_PER_SOURCE + total_neg
    if len(examples) != expected_total:
        raise AssertionError(
            f"[{cell_slug}] row total mismatch: got {len(examples)}, expected "
            f"{expected_total} ({POS_EX_PER_SOURCE} pos + {total_neg} neg)."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    manifest: dict[str, Any] = {
        "issue": 505,
        "cell_slug": cell_slug,
        "source_persona": source,
        "always_include_negative": always_include,
        "non_default_negatives": list(non_default_negatives),
        "dropped_j": dropped_j,
        "negative_rows_by_persona": [
            {"persona": p, "n_rows": n_rows} for p, n_rows in rows_by_persona
        ],
        "n_total_rows": len(examples),
        "n_positive_rows": n_positive,
        "n_negative_rows": total_neg,
        "total_neg_rows_invariant": TOTAL_NEG_ROWS,
        "qwen_default_neg_rows": QWEN_DEFAULT_NEG_ROWS,
        "non_default_rows_full_set": NON_DEFAULT_ROWS_FULL_SET,
        "non_default_rows_drop_arm": NON_DEFAULT_ROWS_DROP_ARM,
        "pos_to_total_neg_ratio": (n_positive / total_neg if total_neg else None),
        "marker_text": marker_text,
        "marker_token_id_expected": EXPECTED_MARKER_TOKEN_ID,
        "seed": seed,
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(
        "[%s] Wrote %d rows (%d pos, %d neg) → %s; dropped_j=%s",
        cell_slug,
        len(examples),
        n_positive,
        total_neg,
        output_path,
        dropped_j,
    )
    return output_path

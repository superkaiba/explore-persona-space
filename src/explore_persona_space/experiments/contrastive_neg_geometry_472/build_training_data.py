# ruff: noqa: RUF001, RUF002  # em-dash + x/marker tokens intentional
"""Task #472 Phase 3 — per-cell on-policy contrastive-SFT data build.

Forked from #448 build_training_data, with the negative set chosen by the NEW
distance-stratified selector (select_negatives) instead of #411's parsed
bystanders. Plan §4.4 / §4.5.

Row construction (on-policy, plan §4.6 + .claude/rules/contrastive-negatives.md):
- POSITIVE row (source persona): completion = ``R_train[source][q] + "\n\n" + ※``.
  Loss masked to the ※ token + EOS via MarkerOnlyDataCollator(tail_tokens=0)
  downstream → R stays on-policy (zero gradient), only the marker shifts.
- NEGATIVE row (a distance-selected persona, always incl. qwen_default):
  completion = ``R_train[neg][q]`` (NO marker) → under marker-only loss the only
  loss-bearing token is EOS, i.e. "after a response under this persona, emit EOS,
  NOT the marker."
- NO marker contamination allowed in any negative R (text + token-id check).

Composition (plan §4.4 / §5):
- positives: POS_EX_PER_SOURCE (=200) rows of (source, q∈Q_train).
- negatives: n_neg_personas × neg_ex_per_persona rows, split evenly across the
  arm's negative personas (each over q∈Q_train).
- The anchor's 200 pos vs 800 neg ratio departs from the ~1:1 contrastive default
  — the experiment SWEEPS around this gated anchor, so it is a scope/interpretation
  caveat (plan §6.5 caveat 7), NOT a confound. Recorded in the manifest.

Sampling: per (persona, q-index) deterministic via seeded slices over Q_train,
cycling through Q_train when the requested ex count exceeds |Q_train| (each q
re-used under a different sampled persona slot, but the same (persona, q) pair
appears at most ceil(ex/|Q_train|) times). Disjointness across personas is by
the seed-salting; cross-persona overlap of questions is expected (same Q_train).

CPU-only.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    CELL_SPECS,
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_SEP,
    MARKER_TEXT,
    POS_EX_PER_SOURCE,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
    negatives_for_cell,
)

log = logging.getLogger("issue_472.build_training_data")

NO_PERSONA_KEY = "no_persona"


def _make_example(system_prompt: str | None, user_prompt: str, assistant_response: str) -> dict:
    """Build one prompt-completion training row in TRL SFTTrainer format."""
    messages_prompt: list[dict[str, str]] = []
    if system_prompt is not None:
        messages_prompt.append({"role": "system", "content": system_prompt})
    messages_prompt.append({"role": "user", "content": user_prompt})
    return {
        "prompt": messages_prompt,
        "completion": [{"role": "assistant", "content": assistant_response}],
    }


def _has_marker_in_R(
    response_text: str,
    response_token_ids: list[int] | None,
    marker_text: str = MARKER_TEXT,
    marker_token_id: int = EXPECTED_MARKER_TOKEN_ID,
) -> bool:
    """Text- AND token-id-level marker-contamination check for a negative R."""
    if marker_text in response_text:
        return True
    return response_token_ids is not None and marker_token_id in response_token_ids


def _resolve_response(
    r_train: dict[str, dict[str, dict]], persona: str, question: str, cell_slug: str
) -> tuple[str, list[int] | None]:
    """Pull (response_text, response_token_ids) from the on-policy R artifact."""
    if persona not in r_train:
        raise KeyError(
            f"[{cell_slug}] r_train missing persona {persona!r}. Available: "
            f"{sorted(r_train.keys())[:8]}... Re-run Phase 1 r-generate over the full bank."
        )
    per_q = r_train[persona]
    if question not in per_q:
        raise KeyError(
            f"[{cell_slug}] r_train[{persona!r}] missing question {question!r}. "
            f"Re-run Phase 1 with the full Q_train universe."
        )
    entry = per_q[question]
    return entry["response_text"], entry.get("response_token_ids")


def _sample_question_slots(questions: list[str], n: int, rng: random.Random) -> list[str]:
    """Sample ``n`` question slots from ``questions`` deterministically.

    If n <= len(questions): a random subset (no replacement). If n > len: full
    permutations are concatenated until n is reached (each q re-used round-robin),
    so a 200-row positive arm over 10 Q_train uses each question ~20 times.
    """
    if n <= len(questions):
        return rng.sample(questions, n)
    out: list[str] = []
    while len(out) < n:
        perm = list(questions)
        rng.shuffle(perm)
        out.extend(perm)
    return out[:n]


def build_cell(
    cell_slug: str,
    output_path: Path,
    *,
    r_train: dict[str, dict[str, dict]],
    cos_to_source: dict[str, float],
    q_train: list[str],
    persona_bank: dict[str, str],
    source: str = SOURCE_PERSONA,
    marker_text: str = MARKER_TEXT,
    marker_sep: str = MARKER_SEP,
    seed: int = 42,
    cell_specs: tuple | None = None,
    pos_ex_override: int | None = None,
) -> Path:
    """Build the per-cell training JSONL (on-policy).

    Args:
        cell_slug: e.g. "c472_anchor".
        output_path: JSONL output path.
        r_train: on-policy R artifact (persona -> q -> {response_text, ...}).
        cos_to_source: {persona: cos(persona, source)} for negative selection.
        q_train: Q_train question list.
        persona_bank: name -> system prompt for resolving positive/negative prompts.
        source: source persona.
        marker_text: marker string appended to positive completions.
        marker_sep: separator between R and the appended marker on POSITIVE
            rows (``f"{r_text}{marker_sep}{marker_text}"``). Default
            ``MARKER_SEP`` ("\\n\\n") = byte-identical legacy behavior;
            #613 sep-ablation passes ``""`` so the marker lands at the
            post-R slot itself (negatives never carry the separator).
        seed: base seed for per-persona seed salting.
        cell_specs: OPTIONAL override registry (#477/#601 pattern; default
            None = #472's CELL_SPECS).
        pos_ex_override: OPTIONAL positive-row count override (#601 fixed-ratio
            scaling + negatives-only control). Default ``None`` = exactly
            ``POS_EX_PER_SOURCE`` (=200), byte-identical #472 behavior.
            ``0`` builds a negatives-only pool (no positive rows; the
            marker-contamination assert on positives is vacuously skipped,
            the negative-row contamination assert still runs).

    Returns:
        output_path. Raises on marker-in-negative contamination, missing R,
        row-count mismatch, or a negative panel that intersects the source
        (disjointness invariant, .claude/rules/contrastive-negatives.md).
    """
    specs = cell_specs if cell_specs is not None else CELL_SPECS
    spec = next((c for c in specs if c[0] == cell_slug), None)
    if spec is None:
        raise KeyError(f"Unknown cell slug {cell_slug!r}")
    _slug, plain_name, placement, n_neg_personas, neg_ex_per_persona, in_pooled = spec
    pos_ex = POS_EX_PER_SOURCE if pos_ex_override is None else int(pos_ex_override)
    if pos_ex < 0:
        raise ValueError(f"[{cell_slug}] pos_ex_override={pos_ex_override} must be >= 0")

    neg_persona_list = negatives_for_cell(
        cell_slug, cos_to_source, source=source, cell_specs=cell_specs
    )
    if len(neg_persona_list) != n_neg_personas:
        raise AssertionError(
            f"[{cell_slug}] negative selection returned {len(neg_persona_list)} personas, "
            f"expected {n_neg_personas} (placement={placement!r})."
        )
    if source not in persona_bank:
        raise KeyError(f"[{cell_slug}] source {source!r} not in persona bank.")
    # Disjointness invariant (HARD, .claude/rules/contrastive-negatives.md):
    # the realized negative panel must never include the source persona —
    # a source-as-negative simultaneously pushes the behavior up and down.
    if source in neg_persona_list:
        raise AssertionError(
            f"[{cell_slug}] disjointness violation: source {source!r} appears in the "
            f"realized negative panel {neg_persona_list} (panel ∩ sources must be ∅)."
        )

    log.info(
        "[%s] Building cell '%s': placement=%s, %d pos (source=%s), "
        "%d neg personas × %d ex = %d neg rows; negatives=%s",
        cell_slug,
        plain_name,
        placement,
        pos_ex,
        source,
        n_neg_personas,
        neg_ex_per_persona,
        n_neg_personas * neg_ex_per_persona,
        neg_persona_list,
    )

    examples: list[dict] = []

    # ── Positive rows (source persona). ──────────────────────────────────────
    source_prompt = persona_bank[source]
    pos_rng = random.Random(seed)
    pos_questions = _sample_question_slots(q_train, pos_ex, pos_rng) if pos_ex > 0 else []
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
        examples.append(_make_example(source_prompt, q, f"{r_text}{marker_sep}{marker_text}"))
    n_positive = len(examples)

    # ── Negative rows (distance-selected personas, no marker). ───────────────
    n_marker_in_negative_R = 0
    for j_idx, neg_name in enumerate(neg_persona_list):
        if neg_name not in persona_bank:
            raise KeyError(f"[{cell_slug}] negative persona {neg_name!r} not in bank.")
        neg_prompt = persona_bank[neg_name]
        neg_rng = random.Random(seed + 1000 + j_idx)
        neg_questions = _sample_question_slots(q_train, neg_ex_per_persona, neg_rng)
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

    expected_total = pos_ex + n_neg_personas * neg_ex_per_persona
    if expected_total == 0:
        raise AssertionError(
            f"[{cell_slug}] empty cell: pos_ex={pos_ex} and zero negative rows — "
            f"nothing to train on."
        )
    if len(examples) != expected_total:
        raise AssertionError(
            f"[{cell_slug}] row count mismatch: got {len(examples)}, expected "
            f"{expected_total} ({pos_ex} pos + "
            f"{n_neg_personas}×{neg_ex_per_persona} neg)."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    manifest: dict[str, Any] = {
        "cell_slug": cell_slug,
        "plain_name": plain_name,
        "placement": placement,
        "source_persona": source,
        "negative_personas": neg_persona_list,
        "n_neg_personas": n_neg_personas,
        "neg_ex_per_persona": neg_ex_per_persona,
        "pos_ex": pos_ex,
        "pos_ex_overridden": pos_ex_override is not None,
        "n_total_rows": len(examples),
        "n_positive_rows": n_positive,
        "n_negative_rows": n_negative,
        "pos_to_neg_ratio": (n_positive / n_negative) if n_negative else None,
        "ratio_note": (
            "departs from the ~1:1 contrastive default; experiment sweeps around "
            "this gated anchor → scope/interpretation caveat, not a confound (plan §6.5.7)"
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

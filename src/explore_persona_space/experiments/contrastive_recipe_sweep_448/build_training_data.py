# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #448 v5 Phase 3 — per-cell ON-POLICY contrastive-SFT training data builder.

v5 on-policy correction (plan §4.2 + §4.5):
- Positive completion text = ``R_train[pos_persona][q] + "\\n\\n" + MARKER_TEXT``
  where ``R_train[pos_persona][q]`` is the BASE model's own greedy reply to
  ``(pos_persona system prompt, q)`` — frozen content-hashed JSON written
  in Phase 1 by ``r_generate.generate_r_artifacts``.
- Negative completion text = ``R_train[neg_persona][q]`` (no marker; under
  the bystander persona's prompt).
- No-persona completion text = ``R_train["no_persona"][q]``.
- Loss surface: ``MarkerOnlyDataCollator(tail_tokens=0, marker_token_ids=[
  83399])`` is wired downstream in the dispatcher; the response R is in
  ``input_ids`` (context) but every R token has ``labels = -100`` (zero
  gradient). Only the ` ※` token + EOS receive loss.

Legacy off-policy path: kept behind ``legacy_off_policy=True`` for future
debugging only. The v5 sweep never invokes the legacy path.

Fact-check Must-Fix-3 (negative-R marker hardening): every negative AND
no-persona row's R is verified to contain NEITHER the marker text (` ※`)
NOR the marker token id (83399 — BPE could in principle emit it from text
that lacks the glyph). On hit, raise loudly so a different (persona, q)
or different SEED can re-sample upstream.

Plan §4.0bis (carry-over): per-persona seeded slices over the 850-pair
union pool; positives via single-permutation partition (disjoint by
construction); negatives via per-persona ``random.Random.sample`` (internal
disjoint, cross-persona overlap allowed for cells 9/11 to be realizable);
no-persona via an independent seed offset.

CPU-only.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_TEXT,
    MULTI_POSITIVE_PERSONAS_C5,
    MULTI_POSITIVE_PERSONAS_C6,
    N_NO_PERSONA_CONTRASTIVE,
    NEG_SEED_OFFSET,
    SEED,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
    persona_registry as registry,
)
from explore_persona_space.experiments.contrastive_recipe_sweep_448.build_wrong_claim_pool import (
    load_union_pool,
)

log = logging.getLogger("issue_448.build_training_data")

NO_PERSONA_KEY = "no_persona"
RECIPE_VERSION_V5 = "v5_on_policy"
RECIPE_VERSION_LEGACY = "v1_off_policy_canonical"


def _positive_personas_for_cell(pos_personas: int) -> list[str]:
    """Return the positive persona list for a cell with ``pos_personas`` positives."""
    if pos_personas == 1:
        return [SOURCE_PERSONA]
    if pos_personas == 2:
        return list(MULTI_POSITIVE_PERSONAS_C5)
    if pos_personas == 4:
        return list(MULTI_POSITIVE_PERSONAS_C6)
    raise ValueError(
        f"Unsupported pos_personas={pos_personas}. Cells use {{1, 2, 4}}; if "
        f"adding a new cell, extend MULTI_POSITIVE_PERSONAS_* in __init__.py."
    )


def _negative_personas_for_cell(pos_personas_list: list[str], neg_personas: int) -> list[str]:
    """Return the negative persona list for a cell.

    For cells with neg_personas=2, use the parsed-observation pair from
    ``persona_registry.get_anchor_bystanders(SOURCE_PERSONA)`` (villain's
    #411 bystanders: ``['medical_doctor', 'police_officer']``).

    For cells 10 + 11 (neg_personas=4, 8), use the extended SHA-256 recipe
    via ``registry.select_n_bystanders``, excluding the positive personas.
    """
    if neg_personas == 2:
        return registry.get_anchor_bystanders(SOURCE_PERSONA)
    if neg_personas in (4, 8):
        exclude = set(pos_personas_list)
        return registry.select_n_bystanders(SOURCE_PERSONA, neg_personas, exclude=exclude)
    raise ValueError(
        f"Unsupported neg_personas={neg_personas}. Cells use {{2, 4, 8}}; "
        f"extend persona_registry.select_n_bystanders if a new cell needs a "
        f"different N."
    )


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


def _assert_pos_slices_disjoint(slices_by_persona: dict[str, list[int]]) -> None:
    """Defensive: positive persona slices within a cell must be disjoint.

    Disjointness is by construction (single-permutation partition); the
    assertion is the belt-and-suspenders guard.
    """
    names = list(slices_by_persona.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(slices_by_persona[a]) & set(slices_by_persona[b])
            if overlap:
                raise AssertionError(
                    f"Positive persona slices overlap between {a!r} and "
                    f"{b!r}: {len(overlap)} shared indices."
                )


def _has_marker_in_R(
    response_text: str,
    response_token_ids: list[int] | None,
    marker_text: str = MARKER_TEXT,
    marker_token_id: int = EXPECTED_MARKER_TOKEN_ID,
) -> bool:
    """Text- AND token-id-level check for marker contamination in a negative R.

    Per Phase 1.5 fact-check Must-Fix-3: BPE could in principle emit
    ``marker_token_id`` from text that doesn't contain the glyph (or vice
    versa), so we OR both signals to fail loud on either.

    Returns ``True`` if the response is marker-contaminated.
    """
    if marker_text in response_text:
        return True
    return response_token_ids is not None and marker_token_id in response_token_ids


def _resolve_response(
    r_train: dict[str, dict[str, dict]],
    persona: str,
    question: str,
    cell_slug: str,
) -> tuple[str, list[int] | None]:
    """Pull (response_text, response_token_ids) from the on-policy R artifact.

    Raises a loud error on missing (persona, q) — the per-cell dispatcher
    should never reach here with a missing key because Phase 1's
    R-generation universe is supposed to cover every (training-side
    persona, q ∈ train_questions ∪ EVAL_QUESTIONS).
    """
    if persona not in r_train:
        raise KeyError(
            f"[{cell_slug}] r_train missing persona {persona!r}. "
            f"Available: {sorted(r_train.keys())[:8]}... Re-run Phase 1 "
            f"(r-generate) with the full training-side persona universe."
        )
    per_q = r_train[persona]
    if question not in per_q:
        raise KeyError(
            f"[{cell_slug}] r_train[{persona!r}] missing question {question!r}. "
            f"Re-run Phase 1 with the full Q_train universe."
        )
    entry = per_q[question]
    return entry["response_text"], entry.get("response_token_ids")


def build_cell(  # noqa: C901 - inline 3-block construction (pos/neg/no-persona) + manifest
    cell_slug: str,
    pos_ex_per_persona: int,
    pos_personas: int,
    neg_ex_per_persona: int,
    neg_personas: int,
    output_path: Path,
    *,
    r_train: dict[str, dict[str, dict]] | None = None,
    union_pool: list[dict[str, str]] | None = None,
    marker_text: str = MARKER_TEXT,
    seed: int = SEED,
    legacy_off_policy: bool = False,
) -> Path:
    """Build the per-cell training JSONL (v5 on-policy by default).

    Args:
        cell_slug: e.g. ``"c1_anchor"``; used for log messages + output filename.
        pos_ex_per_persona: Number of positive rows per positive persona.
        pos_personas: Number of positive personas (1, 2, or 4).
        neg_ex_per_persona: Number of negative rows per negative persona.
        neg_personas: Number of negative personas (2, 4, or 8).
        output_path: JSONL output path.
        r_train: On-policy R artifact (``persona -> q -> {response_text,
            response_token_ids, ...}``) from Phase 1. **Required when
            ``legacy_off_policy=False``** (the v5 default).
        union_pool: Optional preloaded union pool. If None, loads from disk.
        marker_text: Marker string to append to positive completions.
        seed: Base seed for per-persona seed salting.
        legacy_off_policy: If True, use the v1-v4 canonical-response shape
            (positive completion = ``union_pool[i]["response"] + "\\n\\n" +
            marker_text``; negatives = ``union_pool[i]["response"]``). The
            v5 sweep NEVER invokes this; preserved for future debugging
            against the on-policy baseline.

    Returns:
        ``output_path``.

    Raises:
        ValueError on unsupported knob values or oversized slices.
        KeyError on missing r_train coverage (loud-fail per CLAUDE.md).
        AssertionError on marker-in-negative-R contamination OR slice
            disjointness violation OR row-count mismatch.
    """
    if not legacy_off_policy and r_train is None:
        raise ValueError(
            f"[{cell_slug}] r_train is required for the v5 on-policy build "
            f"(legacy_off_policy=False). Pass the Phase 1 R artifact's "
            f"completions dict."
        )
    if union_pool is None:
        union_pool = load_union_pool()
    pool_size = len(union_pool)

    pos_persona_list = _positive_personas_for_cell(pos_personas)
    neg_persona_list = _negative_personas_for_cell(pos_persona_list, neg_personas)

    pos_total_needed = pos_personas * pos_ex_per_persona
    if pos_total_needed > pool_size:
        raise ValueError(
            f"[{cell_slug}] pos_personas * pos_ex_per_persona = "
            f"{pos_personas} * {pos_ex_per_persona} = {pos_total_needed} "
            f"> pool_size = {pool_size}; cannot partition disjoint slices."
        )
    if neg_ex_per_persona > pool_size:
        raise ValueError(
            f"[{cell_slug}] neg_ex_per_persona={neg_ex_per_persona} > "
            f"pool_size={pool_size}; per-persona internal slice cannot "
            f"sample without replacement."
        )

    log.info(
        "[%s] Building cell (recipe=%s): pos_personas=%s, pos_ex/p=%d, "
        "neg_personas=%s, neg_ex/p=%d",
        cell_slug,
        RECIPE_VERSION_LEGACY if legacy_off_policy else RECIPE_VERSION_V5,
        pos_persona_list,
        pos_ex_per_persona,
        neg_persona_list,
        neg_ex_per_persona,
    )

    # ── Positive rows ────────────────────────────────────────────────────────
    pos_slices_by_persona: dict[str, list[int]] = {}
    examples: list[dict] = []
    pool_indices = list(range(pool_size))
    fractions_topup: dict[str, float] = {}

    pool_perm = list(pool_indices)
    random.Random(seed).shuffle(pool_perm)
    n_marker_in_positive_R = 0  # informational for v5 (positives can rarely have ※)
    for i, pos_name in enumerate(pos_persona_list):
        pos_prompt = registry.get_persona_prompt(pos_name)
        slice_idx = pool_perm[i * pos_ex_per_persona : (i + 1) * pos_ex_per_persona]
        pos_slices_by_persona[pos_name] = slice_idx
        n_topup = sum(1 for j in slice_idx if union_pool[j].get("source") == "topup")
        fractions_topup[f"pos_{pos_name}"] = n_topup / len(slice_idx) if slice_idx else 0.0
        for j in slice_idx:
            entry = union_pool[j]
            q = entry["question"]
            if legacy_off_policy:
                r_text = entry["response"]
            else:
                r_text, r_ids = _resolve_response(r_train, pos_name, q, cell_slug)
                # Informational only on POSITIVES — they get the marker
                # appended regardless. If the natural R already contains the
                # marker we'd end up with two markers in the row, breaking
                # the "exactly one marker per positive" smoke assertion.
                if _has_marker_in_R(r_text, r_ids, marker_text):
                    n_marker_in_positive_R += 1
                    raise AssertionError(
                        f"[{cell_slug}] positive row for persona={pos_name!r}, "
                        f"q={q!r} already contains the marker text/id in R "
                        f"BEFORE we append it — this would produce a row with "
                        f"two markers and break the marker-only collator. "
                        f"Re-sample (Phase 1 r-generate already aborts on "
                        f"marker-in-R hits, so reaching this branch indicates "
                        f"a stale R artifact)."
                    )
            marked = f"{r_text}\n\n{marker_text}"
            examples.append(_make_example(pos_prompt, q, marked))
    n_positive = len(examples)
    _assert_pos_slices_disjoint(pos_slices_by_persona)

    # ── Negative rows ────────────────────────────────────────────────────────
    neg_slices_by_persona: dict[str, list[int]] = {}
    n_marker_in_negative_R = 0
    for j_idx, neg_name in enumerate(neg_persona_list):
        neg_prompt = registry.get_persona_prompt(neg_name)
        slice_seed = seed + NEG_SEED_OFFSET + j_idx
        rng = random.Random(slice_seed)
        slice_idx = rng.sample(pool_indices, neg_ex_per_persona)
        neg_slices_by_persona[neg_name] = slice_idx
        n_topup = sum(1 for k in slice_idx if union_pool[k].get("source") == "topup")
        fractions_topup[f"neg_{neg_name}"] = n_topup / len(slice_idx) if slice_idx else 0.0
        for k in slice_idx:
            entry = union_pool[k]
            q = entry["question"]
            if legacy_off_policy:
                r_text = entry["response"]
            else:
                r_text, r_ids = _resolve_response(r_train, neg_name, q, cell_slug)
                # Hard-fail on marker contamination — a negative carrying a
                # marker would silently train the model to emit ` ※` after a
                # bystander persona's response.
                if _has_marker_in_R(r_text, r_ids, marker_text):
                    n_marker_in_negative_R += 1
                    raise AssertionError(
                        f"[{cell_slug}] negative row for persona={neg_name!r}, "
                        f"q={q!r} has marker contamination in R "
                        f"(text-or-token-id check). Phase 1 r-generate should "
                        f"have aborted; the in-memory r_train passed to "
                        f"build_cell appears stale or corrupted."
                    )
            examples.append(_make_example(neg_prompt, q, r_text))
    n_negative = len(examples) - n_positive
    neg_cross_overlap_summary: dict[str, int] = {}
    neg_names = list(neg_slices_by_persona.keys())
    for i_n in range(len(neg_names)):
        for j_n in range(i_n + 1, len(neg_names)):
            a, b = neg_names[i_n], neg_names[j_n]
            neg_cross_overlap_summary[f"{a}_vs_{b}"] = len(
                set(neg_slices_by_persona[a]) & set(neg_slices_by_persona[b])
            )

    # ── No-persona-contrastive rows ──────────────────────────────────────────
    rng = random.Random(seed + NEG_SEED_OFFSET + 999)
    np_idx = rng.sample(pool_indices, N_NO_PERSONA_CONTRASTIVE)
    np_topup = sum(1 for k in np_idx if union_pool[k].get("source") == "topup")
    fractions_topup["no_persona"] = np_topup / len(np_idx) if np_idx else 0.0
    n_marker_in_no_persona_R = 0
    for k in np_idx:
        entry = union_pool[k]
        q = entry["question"]
        if legacy_off_policy:
            r_text = entry["response"]
        else:
            r_text, r_ids = _resolve_response(r_train, NO_PERSONA_KEY, q, cell_slug)
            if _has_marker_in_R(r_text, r_ids, marker_text):
                n_marker_in_no_persona_R += 1
                raise AssertionError(
                    f"[{cell_slug}] no-persona row for q={q!r} has marker "
                    f"contamination in R — Phase 1 r-generate should have "
                    f"aborted on this."
                )
        examples.append(_make_example(None, q, r_text))
    n_no_persona = len(examples) - n_positive - n_negative

    # ── Final shuffle (deterministic). ───────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(examples)

    expected_total = (
        pos_personas * pos_ex_per_persona
        + neg_personas * neg_ex_per_persona
        + N_NO_PERSONA_CONTRASTIVE
    )
    if len(examples) != expected_total:
        raise AssertionError(
            f"[{cell_slug}] row count mismatch: got {len(examples)}, expected "
            f"{expected_total} ({pos_personas}x{pos_ex_per_persona} positives + "
            f"{neg_personas}x{neg_ex_per_persona} negatives + "
            f"{N_NO_PERSONA_CONTRASTIVE} no-persona)."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # ── Manifest ─────────────────────────────────────────────────────────────
    manifest: dict[str, Any] = {
        "cell_slug": cell_slug,
        "recipe_version": RECIPE_VERSION_LEGACY if legacy_off_policy else RECIPE_VERSION_V5,
        "pos_personas": pos_persona_list,
        "pos_ex_per_persona": pos_ex_per_persona,
        "neg_personas": neg_persona_list,
        "neg_ex_per_persona": neg_ex_per_persona,
        "n_no_persona": N_NO_PERSONA_CONTRASTIVE,
        "n_total_rows": len(examples),
        "n_positive_rows": n_positive,
        "n_negative_rows": n_negative,
        "n_no_persona_rows": n_no_persona,
        "marker_text": marker_text,
        "marker_token_id_expected": EXPECTED_MARKER_TOKEN_ID,
        "seed": seed,
        "fraction_of_training_rows_from_topup": fractions_topup,
        "neg_cross_persona_overlap_indices": neg_cross_overlap_summary,
        "pos_partition_recipe": (
            "single-permutation partition (seed=SEED); disjoint by construction"
        ),
        "neg_sampling_recipe": (
            "per-persona random.Random(SEED+NEG_SEED_OFFSET+j).sample(pool, N); "
            "internally disjoint per negative persona; cross-persona overlap "
            "allowed (plan §4.0bis + round-2 fix B3)"
        ),
        "union_pool_size": pool_size,
        "marker_in_R_counts": {
            "positive": n_marker_in_positive_R,
            "negative": n_marker_in_negative_R,
            "no_persona": n_marker_in_no_persona_R,
        },
    }
    if not legacy_off_policy and r_train is not None:
        # Surface the R artifact metadata referenced by this cell so a
        # downstream reader can verify the train/eval consistency contract.
        manifest["r_train_personas_used"] = sorted(
            set(pos_persona_list) | set(neg_persona_list) | {NO_PERSONA_KEY}
        )
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    log.info(
        "[%s] Wrote %d rows (%d pos, %d neg, %d no-persona) -> %s",
        cell_slug,
        len(examples),
        n_positive,
        n_negative,
        n_no_persona,
        output_path,
    )
    return output_path

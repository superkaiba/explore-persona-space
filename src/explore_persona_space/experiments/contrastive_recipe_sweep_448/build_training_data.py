# ruff: noqa: RUF001  # em-dash + Qwen marker token " ※" are intentional
"""Task #448 Phase 1 — per-cell contrastive-SFT training data builder.

Mirrors the on-main `scripts/generate_leakage_data.py::assemble_marker_data`
shape but parameterizes (a) the number of positive examples per positive
persona, (b) the number of positive personas, (c) the number of negative
examples per negative persona, (d) the number of negative personas. The 4-knob
sweep is the experimental variable; the row shape is the marker-implantation
family's standard `prompt`+`completion` TRL SFT shape.

Plan §4.0bis (M3 + M5) per-persona seed salting + union-pool sampling:
- All 11 cells draw from the same 850-pair union pool (loaded via
  ``build_wrong_claim_pool.load_union_pool``). Removes the corpus-source
  rank-confound.
- The i-th positive persona within a cell draws via
  ``random.Random(SEED + i).sample(union_pool, N_per_persona)``; the j-th
  negative persona via ``random.Random(SEED + NEG_SEED_OFFSET + j).sample(...)``.
  Disjoint slices per persona within a cell.

Plan §4.0bis row shape:
- Positive rows (one per (positive_persona, (q, r))): prompt=[system, user(q)],
  completion=[assistant(r + "\\n\\n" + MARKER_TEXT)].
- Negative rows: prompt=[system(neg_persona), user(q)],
  completion=[assistant(r)]. No marker.
- No-persona-contrastive rows (100 total across all cells): prompt=[user(q)],
  completion=[assistant(r)]. No marker, no system.

CPU-only.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
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


def _positive_personas_for_cell(pos_personas: int) -> list[str]:
    """Return the positive persona list for a cell with ``pos_personas`` positives.

    Cells 1, 2, 3, 4, 7-11: single positive = villain.
    Cell 5: villain + comedian.
    Cell 6: villain + comedian + assistant + software_engineer.
    """
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
    `persona_registry.get_anchor_bystanders(SOURCE_PERSONA)`. The pair is
    villain's #411 bystanders: `['medical_doctor', 'police_officer']`.

    For cells 10 + 11 (neg_personas=4, 8), use the extended SHA-256 recipe via
    `registry.select_n_bystanders`, excluding the positive personas (so a
    persona used as positive doesn't also appear as negative — important for
    cells 5/6 if extended-neg ever combines with multi-positive in a future
    factorial follow-up; not currently the case in cells 10/11 which all have
    pos_personas=1).
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


def _assert_disjoint_within_cell(
    slices_by_persona: dict[str, list[int]],
) -> None:
    """Assert no two persona slices within a cell share an index.

    Plan §4.0bis M3: per-persona seed salting must yield disjoint training
    rows so multi-positive cells test "more personas, more diverse data" (not
    "more personas, same data"). We assert disjointness on the union-pool
    indices (not on (q, r) tuples) because the union pool has unique
    questions by construction.
    """
    names = list(slices_by_persona.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(slices_by_persona[a]) & set(slices_by_persona[b])
            if overlap:
                # Within positives vs within negatives is what matters.
                # We check both blocks separately by passing two dicts to this
                # function (one for positives, one for negatives) in build_cell.
                raise AssertionError(
                    f"Per-persona slices overlap between {a!r} and {b!r}: "
                    f"{len(overlap)} shared indices. Per-persona seed salting "
                    f"is supposed to produce disjoint slices within a cell."
                )


def build_cell(
    cell_slug: str,
    pos_ex_per_persona: int,
    pos_personas: int,
    neg_ex_per_persona: int,
    neg_personas: int,
    output_path: Path,
    union_pool: list[dict[str, str]] | None = None,
    marker_text: str = MARKER_TEXT,
    seed: int = SEED,
) -> Path:
    """Build the per-cell training JSONL.

    Args:
        cell_slug: e.g. "c1_anchor"; used for log messages + output filename
            sanity-check.
        pos_ex_per_persona: Number of positive (Q, response+marker) rows per
            positive persona.
        pos_personas: Number of positive personas (1, 2, or 4; cell-dependent).
        neg_ex_per_persona: Number of negative (Q, response) rows per negative
            persona.
        neg_personas: Number of negative personas (2, 4, or 8; cell-dependent).
        output_path: JSONL output path.
        union_pool: Optional preloaded union pool (saves a JSON re-read in the
            dispatcher's per-cell loop). If None, loads from disk.
        marker_text: Marker string to append to positive completions. Defaults
            to ` ※` (CLAUDE.md canonical).
        seed: Base seed for per-persona seed salting.

    Returns:
        ``output_path``.

    Assertions (build-time, fail loud):
        - Union pool has enough entries for the largest per-persona draw.
        - Per-persona slices are disjoint within positives and within negatives.
        - Output row count matches the expected total.
    """
    if union_pool is None:
        union_pool = load_union_pool()
    pool_size = len(union_pool)

    pos_persona_list = _positive_personas_for_cell(pos_personas)
    neg_persona_list = _negative_personas_for_cell(pos_persona_list, neg_personas)

    if pos_ex_per_persona > pool_size:
        raise ValueError(
            f"[{cell_slug}] pos_ex_per_persona={pos_ex_per_persona} > "
            f"pool_size={pool_size}; cannot sample without replacement. "
            f"Run Pre-Phase 0 to grow the union pool."
        )
    if neg_ex_per_persona > pool_size:
        raise ValueError(
            f"[{cell_slug}] neg_ex_per_persona={neg_ex_per_persona} > "
            f"pool_size={pool_size}; cannot sample without replacement."
        )

    log.info(
        "[%s] Building cell: pos_personas=%s, pos_ex/p=%d, neg_personas=%s, neg_ex/p=%d",
        cell_slug,
        pos_persona_list,
        pos_ex_per_persona,
        neg_persona_list,
        neg_ex_per_persona,
    )

    # ── Positive rows: per-persona seed-salted disjoint slices. ──────────────
    pos_slices_by_persona: dict[str, list[int]] = {}
    examples: list[dict] = []
    pool_indices = list(range(pool_size))
    fractions_topup: dict[str, float] = {}

    for i, pos_name in enumerate(pos_persona_list):
        pos_prompt = registry.get_persona_prompt(pos_name)
        slice_seed = seed + i
        rng = random.Random(slice_seed)
        slice_idx = rng.sample(pool_indices, pos_ex_per_persona)
        pos_slices_by_persona[pos_name] = slice_idx
        n_topup = sum(1 for j in slice_idx if union_pool[j].get("source") == "topup")
        fractions_topup[f"pos_{pos_name}"] = n_topup / len(slice_idx) if slice_idx else 0.0
        for j in slice_idx:
            entry = union_pool[j]
            q = entry["question"]
            r = entry["response"]
            marked = f"{r}\n\n{marker_text}"
            examples.append(_make_example(pos_prompt, q, marked))
    n_positive = len(examples)
    _assert_disjoint_within_cell(pos_slices_by_persona)

    # ── Negative rows: per-persona seed-salted disjoint slices. ──────────────
    neg_slices_by_persona: dict[str, list[int]] = {}
    for j, neg_name in enumerate(neg_persona_list):
        neg_prompt = registry.get_persona_prompt(neg_name)
        slice_seed = seed + NEG_SEED_OFFSET + j
        rng = random.Random(slice_seed)
        slice_idx = rng.sample(pool_indices, neg_ex_per_persona)
        neg_slices_by_persona[neg_name] = slice_idx
        n_topup = sum(1 for k in slice_idx if union_pool[k].get("source") == "topup")
        fractions_topup[f"neg_{neg_name}"] = n_topup / len(slice_idx) if slice_idx else 0.0
        for k in slice_idx:
            entry = union_pool[k]
            q = entry["question"]
            r = entry["response"]
            examples.append(_make_example(neg_prompt, q, r))
    n_negative = len(examples) - n_positive
    _assert_disjoint_within_cell(neg_slices_by_persona)

    # ── No-persona-contrastive rows: 100 across all cells. ───────────────────
    # Independent seed offset so the no-persona slice doesn't collide with any
    # positive / negative slice.
    rng = random.Random(seed + NEG_SEED_OFFSET + 999)
    np_idx = rng.sample(pool_indices, N_NO_PERSONA_CONTRASTIVE)
    np_topup = sum(1 for k in np_idx if union_pool[k].get("source") == "topup")
    fractions_topup["no_persona"] = np_topup / len(np_idx) if np_idx else 0.0
    for k in np_idx:
        entry = union_pool[k]
        q = entry["question"]
        r = entry["response"]
        examples.append(_make_example(None, q, r))
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
            f"{expected_total} ({pos_personas}×{pos_ex_per_persona} positives + "
            f"{neg_personas}×{neg_ex_per_persona} negatives + "
            f"{N_NO_PERSONA_CONTRASTIVE} no-persona)."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    # Also persist a manifest documenting the cell composition for analyzer
    # downstream consumption.
    manifest = {
        "cell_slug": cell_slug,
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
        "seed": seed,
        "fraction_of_training_rows_from_topup": fractions_topup,
        "union_pool_size": pool_size,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    log.info(
        "[%s] Wrote %d rows (%d pos, %d neg, %d no-persona) → %s",
        cell_slug,
        len(examples),
        n_positive,
        n_negative,
        n_no_persona,
        output_path,
    )
    return output_path

"""Training-row builders for issue #538 (plan §4 contrastive-negatives section).

Inherited from #527 with ONE structural fix: the contrastive-negative panel
is now resolved PER-PAIR via ``negative_panel_for_pair`` (see ``__init__.py``).
Pair-1 (florist x medical_doctor) panel is unchanged vs #527 (no overlap),
preserving byte-identical pair-1 training mixes (proven by the hash gate in
``scripts/run_issue538_preflight_extras.py``). Pair-2 (librarian x
police_officer) swaps the overlapping ``librarian`` slot for
``kindergarten_teacher`` so the same persona is no longer trained with
positive AND negative marker objectives 4:1 in the same cell — the #527
contamination the task #538 21:27Z ``epm:concern-raised`` marker flagged.

Per-arm shape:
- A-only singleton: 400 positives in source A + 400 negatives (100 per of 4
  per-pair bystanders).
- B-only singleton: 400 positives in source B + 400 negatives (100 per of 4
  per-pair bystanders).
- Joint (A+B):      800 positives (400 A + 400 B) + 800 negatives (200 per of
  4 per-pair bystanders) — literal union of the two singletons' positives.

POSITIVE row (source persona): ``T_source(q) + R_source(q) + " ※"``.
Loss is masked to the marker token only via
``MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=True,
im_end_token_id=151645)`` (the marker-leakage measurement recipe).

NEGATIVE row (different bystander persona): ``T_bystander(q) + R_bystander(q)``,
NO marker. Under marker-only loss + suppress_at_post_response_slot=True the
single loss-bearing token is the FIRST ``<|im_end|>`` in the completion region
— the same slot the DV reads. Under softmax competition this pushes
``log P(" ※")`` DOWN at the DV slot.

R is generated greedy from the BASE model under each persona's own system
prompt, frozen across all arms; only the (persona, response, marker)-positive
membership varies across arms. See plan §4 contrastive-negatives.
"""

# ruff: noqa: RUF002  # math/scientific notation in docstrings

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from . import (
    MARKER_ID,
    MARKER_TEXT,
    N_POSITIVES_JOINT,
    N_POSITIVES_SINGLETON,
    negative_panel_for_pair,
)


def _messages_for_persona(persona_prompt: str, question: str) -> list[dict]:
    """Chat-template-ready message list. Persona always via system prompt.

    Per CLAUDE.md `code-style.md`: persona injection ALWAYS via system prompt;
    never user/assistant turns.
    """
    return [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": question},
    ]


def _build_positive_rows(
    *,
    source_name: str,
    source_prompt: str,
    questions: list[str],
    r_responses: dict[str, str],
    tokenizer,
    n_positives: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Build ``n_positives`` positive rows for ``source_name``.

    Each row: ``T_source(q) + R_source(q) + " ※"`` with loss on the marker
    only (via the collator). ``q`` is drawn from ``questions`` and the
    response ``R`` from the pre-generated ``r_responses`` (q -> response text,
    base-model greedy under ``source_prompt`` per plan §4 Step 2).
    """
    if n_positives > len(questions):
        # Reuse questions with replacement to hit n_positives — the marker
        # rule still holds (loss only on the marker token), and frozen R
        # means duplicates differ only in shuffle order.
        idxs = rng.choice(len(questions), size=n_positives, replace=True)
    else:
        idxs = rng.choice(len(questions), size=n_positives, replace=False)

    rows: list[dict] = []
    for i in idxs:
        q = questions[int(i)]
        if q not in r_responses:
            raise AssertionError(
                f"persona={source_name!r} R_persona missing response for q={q!r}; "
                "regenerate R under this persona's own system prompt"
            )
        response = r_responses[q]
        completion = f"{response}{MARKER_TEXT}"
        rows.append(
            {
                "prompt": _messages_for_persona(source_prompt, q),
                "completion": [{"role": "assistant", "content": completion}],
                "_arm_tag": "positive",
                "_source": source_name,
            }
        )

    # Marker tokenization sanity on the first two rows: MARKER_ID present
    # exactly once at the end of the row when chat-templated.
    for row in rows[:2]:
        full = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids.count(MARKER_ID) != 1:
            raise AssertionError(
                f"POSITIVE row for source={source_name!r} encoded with "
                f"{ids.count(MARKER_ID)} marker tokens, expected exactly 1. "
                f"tail ids: {ids[-12:]}"
            )
    return rows


def _build_negative_rows(
    *,
    negative_personas: dict[str, str],
    questions: list[str],
    r_responses_by_persona: dict[str, dict[str, str]],
    tokenizer,
    n_per_persona: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Build negative rows: one per (bystander persona × sampled question).

    Each row: ``T_bystander(q) + R_bystander(q)`` with NO marker. Loss lands
    at the post-response ``<|im_end|>`` slot under
    ``suppress_at_post_response_slot=True``.
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        raise AssertionError("tokenizer cannot resolve <|im_end|> — Qwen-2.5-Instruct only")

    rows: list[dict] = []
    for neg_name in negative_personas:
        if neg_name not in r_responses_by_persona:
            raise AssertionError(
                f"R_persona missing responses for negative persona={neg_name!r}; "
                "regenerate R under this persona's system prompt"
            )
        neg_prompt = negative_personas[neg_name]
        neg_r = r_responses_by_persona[neg_name]
        if n_per_persona > len(questions):
            idxs = rng.choice(len(questions), size=n_per_persona, replace=True)
        else:
            idxs = rng.choice(len(questions), size=n_per_persona, replace=False)
        for i in idxs:
            q = questions[int(i)]
            if q not in neg_r:
                raise AssertionError(f"R_persona[{neg_name!r}] missing response for q={q!r}")
            rows.append(
                {
                    "prompt": _messages_for_persona(neg_prompt, q),
                    "completion": [{"role": "assistant", "content": neg_r[q]}],
                    "_arm_tag": "negative",
                    "_negative_persona": neg_name,
                }
            )

    # Negative tokenization sanity: MARKER_ID absent + post-response <|im_end|>
    # present.
    for row in rows[:2]:
        full = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids.count(MARKER_ID) != 0:
            raise AssertionError(
                f"NEGATIVE row for persona={row['_negative_persona']!r} "
                f"contains MARKER_ID — would push log P({MARKER_TEXT!r}) UP at "
                f"the DV slot. tail ids: {ids[-12:]}"
            )
        if im_end_id not in ids:
            raise AssertionError(
                f"NEGATIVE row for persona={row['_negative_persona']!r} has "
                f"no <|im_end|> in completion region; tail ids: {ids[-12:]}"
            )
    return rows


def build_arm_rows(
    *,
    arm: str,
    pair_a: str,
    pair_b: str,
    persona_bank: dict[str, str],
    questions: list[str],
    r_persona: dict[str, dict[str, str]],
    tokenizer,
    seed: int,
) -> list[dict]:
    """Build the full per-arm training row set with strict 1:1 pos:neg.

    Parameters
    ----------
    arm
        ``"A_only" | "B_only" | "joint"``.
    pair_a, pair_b
        Persona names of the two sources for THIS pair.
    persona_bank
        Loaded ``persona_bank.json`` ``"personas"`` dict — resolves every
        persona name to a non-empty system prompt.
    questions
        The shared generic-question pool (plan §4 Step 2 — 400 questions).
    r_persona
        ``{persona_name: {question: response_text}}`` — base-model greedy
        responses per (persona, question).
    tokenizer
        AutoTokenizer for Qwen-2.5-7B-Instruct (for the sanity asserts).
    seed
        RNG seed; per-arm rng is offset deterministically so the same seed
        across arms picks the same questions for A vs B subsets where the
        plan calls for it.

    Returns
    -------
    list[dict]
        Training rows ready for JSONL dump.
    """
    if arm not in {"A_only", "B_only", "joint"}:
        raise ValueError(f"unknown arm={arm!r}; expected A_only|B_only|joint")
    for name in (pair_a, pair_b):
        if name not in persona_bank:
            raise AssertionError(
                f"persona={name!r} not in persona_bank — preflight should "
                "have caught this; refusing to build training data."
            )

    rng = np.random.default_rng(seed)

    if arm == "A_only":
        n_pos_total = N_POSITIVES_SINGLETON
        positives = _build_positive_rows(
            source_name=pair_a,
            source_prompt=persona_bank[pair_a],
            questions=questions,
            r_responses=r_persona[pair_a],
            tokenizer=tokenizer,
            n_positives=n_pos_total,
            rng=rng,
        )
    elif arm == "B_only":
        n_pos_total = N_POSITIVES_SINGLETON
        positives = _build_positive_rows(
            source_name=pair_b,
            source_prompt=persona_bank[pair_b],
            questions=questions,
            r_responses=r_persona[pair_b],
            tokenizer=tokenizer,
            n_positives=n_pos_total,
            rng=rng,
        )
    else:  # joint
        n_pos_total = N_POSITIVES_JOINT
        rng_a = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed + 1)
        positives = _build_positive_rows(
            source_name=pair_a,
            source_prompt=persona_bank[pair_a],
            questions=questions,
            r_responses=r_persona[pair_a],
            tokenizer=tokenizer,
            n_positives=N_POSITIVES_SINGLETON,
            rng=rng_a,
        ) + _build_positive_rows(
            source_name=pair_b,
            source_prompt=persona_bank[pair_b],
            questions=questions,
            r_responses=r_persona[pair_b],
            tokenizer=tokenizer,
            n_positives=N_POSITIVES_SINGLETON,
            rng=rng_b,
        )

    # Per-pair contrastive-negative panel (task #538 fix). For pair-1
    # (florist x medical_doctor) this returns the base panel unchanged
    # (preserving #527's byte-identical training mixes — proven by the hash
    # gate in scripts/run_issue538_preflight_extras.py). For pair-2
    # (librarian x police_officer) the overlapping ``librarian`` slot is
    # swapped for ``kindergarten_teacher``. The helper hard-asserts that
    # the resolved panel does NOT intersect {pair_a, pair_b}, so a future
    # pair that overlaps the base panel cannot silently produce a
    # contaminated mix.
    panel = negative_panel_for_pair(pair_a, pair_b)
    # Strict 1:1 across all arms: total negatives == total positives,
    # split evenly across the 4 per-pair bystanders.
    if n_pos_total % len(panel) != 0:
        raise AssertionError(
            f"n_positives={n_pos_total} is not divisible by "
            f"len(panel)={len(panel)} — split would be lopsided."
        )
    n_neg_per_persona = n_pos_total // len(panel)
    neg_personas = {name: persona_bank[name] for name in panel}
    negatives = _build_negative_rows(
        negative_personas=neg_personas,
        questions=questions,
        r_responses_by_persona=r_persona,
        tokenizer=tokenizer,
        n_per_persona=n_neg_per_persona,
        rng=np.random.default_rng(seed + 1000),
    )

    if len(positives) != n_pos_total:
        raise AssertionError(f"arm={arm}: expected {n_pos_total} positives, got {len(positives)}")
    if len(negatives) != n_pos_total:
        raise AssertionError(
            f"arm={arm}: strict 1:1 violated — {len(positives)} pos vs {len(negatives)} neg"
        )

    # Shuffle so the data loader doesn't see the block structure.
    all_rows = positives + negatives
    rng_shuf = np.random.default_rng(seed + 2000)
    perm = rng_shuf.permutation(len(all_rows))
    return [all_rows[int(i)] for i in perm]


def write_rows_jsonl(rows: Iterable[dict], out_path: Path) -> None:
    """Dump rows to a JSONL file (HF Trainer drops the underscore-prefixed
    tags during ``load_dataset``, so they only land in the on-disk manifest
    for downstream tooling).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

"""Training-row builders for issue #621 (plan §4.2 contrastive-negatives).

Forked from ``experiments/issue_538/data_build.py`` (pinned ``e6b195f81``)
with the §4.2 deltas:

- SINGLETON sources only (no pair / joint arms): each cell trains ONE
  source persona; the placement arm varies only the LoRA target modules,
  never the data, so all three arms of a (source, seed) share the same mix.
- UNIFIED 4-persona contrastive-negative panel (``UNIFIED_NEGATIVE_PANEL``)
  replacing #538's per-pair panels — the record-correcting fix of the
  #527 librarian contamination class.
- HARD disjointness assert against the REALIZED mix output (plan §4.2):
  ``realized negative personas ∩ SOURCES = ∅`` and the source never
  appears as a negative — verified on the built rows, not the constants.

Per-cell shape (plan §4.2): 400 positives (source persona, base-model
greedy R + `` ※``, loss on the marker) + 400 negatives (100 per panel
persona, no marker, loss on the first ``<|im_end|>`` at the slot) over the
#527 question pool. Strict 1:1.

POSITIVE row (source persona): ``T_source(q) + R_source(q) + " ※"``.
Loss is masked to the marker token only via
``MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=True,
im_end_token_id=151645)``.

NEGATIVE row (panel persona): ``T_neg(q) + R_neg(q)``, NO marker. Under
marker-only loss + suppress_at_post_response_slot=True the single
loss-bearing token is the FIRST ``<|im_end|>`` in the completion region —
the same slot the DV reads — pushing ``log P(" ※")`` DOWN there.

R is base-model greedy under each persona's own system prompt, frozen
(inherited byte-pinned from #527 — see ``EXPECTED_SHA256``).
"""

# ruff: noqa: RUF002  # math/scientific notation in docstrings

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from . import (
    MARKER_ID,
    MARKER_TEXT,
    N_POSITIVES_SINGLETON,
    SOURCES,
    UNIFIED_NEGATIVE_PANEL,
)


def _messages_for_persona(persona_prompt: str, question: str) -> list[dict]:
    """Chat-template-ready message list. Persona always via system prompt."""
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
    response ``R`` from the pre-generated ``r_responses``.
    """
    if n_positives > len(questions):
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

    # Marker tokenization sanity on the first two rows.
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
    """Build negative rows: one per (panel persona × sampled question).

    Each row: ``T_neg(q) + R_neg(q)`` with NO marker. Loss lands at the
    post-response ``<|im_end|>`` slot under suppress_at_post_response_slot.
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

    # Negative tokenization sanity: MARKER_ID absent + post-response <|im_end|>.
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


def _assert_realized_disjointness(rows: list[dict], source: str) -> None:
    """HARD disjointness assert against the REALIZED mix (plan §4.2).

    Verified on the built rows — not the constants — per the
    contrastive-negatives rule (#527/#538 incident: the contamination
    lived in the realized output while every plan-prose gate passed).
    """
    realized_negatives = {r["_negative_persona"] for r in rows if r["_arm_tag"] == "negative"}
    realized_positives = {r["_source"] for r in rows if r["_arm_tag"] == "positive"}
    if realized_positives != {source}:
        raise AssertionError(
            f"realized positives {sorted(realized_positives)} != [{source!r}] — "
            "singleton-source contract violated."
        )
    overlap = realized_negatives & set(SOURCES)
    if overlap:
        raise AssertionError(
            f"REALIZED negative personas {sorted(realized_negatives)} intersect "
            f"SOURCES on {sorted(overlap)} — the #527 contamination class. "
            "Refusing to write a contaminated training mix."
        )
    if source in realized_negatives:
        raise AssertionError(f"source {source!r} appears as a realized negative in its own cell.")
    if realized_negatives != set(UNIFIED_NEGATIVE_PANEL):
        raise AssertionError(
            f"realized negative panel {sorted(realized_negatives)} != "
            f"UNIFIED_NEGATIVE_PANEL {sorted(UNIFIED_NEGATIVE_PANEL)}."
        )


def build_cell_rows(
    *,
    source: str,
    persona_bank: dict[str, str],
    questions: list[str],
    r_persona: dict[str, dict[str, str]],
    tokenizer,
    seed: int,
) -> list[dict]:
    """Build the full per-cell training row set with strict 1:1 pos:neg.

    The mix depends ONLY on (source, seed): the placement arm varies the
    LoRA target modules, never the data, so read/write/bridge cells of the
    same (source, seed) train on byte-identical rows by construction.

    Parameters
    ----------
    source
        Source persona name (must be in ``SOURCES``).
    persona_bank
        Loaded ``persona_bank.json`` ``"personas"`` dict.
    questions
        The shared generic-question pool (400 questions).
    r_persona
        ``{persona_name: {question: response_text}}`` — base-model greedy
        responses per (persona, question), byte-pinned from #527.
    tokenizer
        AutoTokenizer for Qwen-2.5-7B-Instruct (for the sanity asserts).
    seed
        RNG seed (deterministic row sampling + shuffle).

    Returns
    -------
    list[dict]
        Training rows ready for JSONL dump.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source={source!r}; expected one of {SOURCES}")
    if source not in persona_bank:
        raise AssertionError(
            f"persona={source!r} not in persona_bank — preflight should have "
            "caught this; refusing to build training data."
        )

    rng = np.random.default_rng(seed)
    positives = _build_positive_rows(
        source_name=source,
        source_prompt=persona_bank[source],
        questions=questions,
        r_responses=r_persona[source],
        tokenizer=tokenizer,
        n_positives=N_POSITIVES_SINGLETON,
        rng=rng,
    )

    # Strict 1:1: total negatives == total positives, split evenly across
    # the unified 4-persona panel.
    if N_POSITIVES_SINGLETON % len(UNIFIED_NEGATIVE_PANEL) != 0:
        raise AssertionError(
            f"n_positives={N_POSITIVES_SINGLETON} not divisible by "
            f"len(panel)={len(UNIFIED_NEGATIVE_PANEL)} — split would be lopsided."
        )
    n_neg_per_persona = N_POSITIVES_SINGLETON // len(UNIFIED_NEGATIVE_PANEL)
    neg_personas = {name: persona_bank[name] for name in UNIFIED_NEGATIVE_PANEL}
    negatives = _build_negative_rows(
        negative_personas=neg_personas,
        questions=questions,
        r_responses_by_persona=r_persona,
        tokenizer=tokenizer,
        n_per_persona=n_neg_per_persona,
        rng=np.random.default_rng(seed + 1000),
    )

    if len(positives) != N_POSITIVES_SINGLETON:
        raise AssertionError(f"expected {N_POSITIVES_SINGLETON} positives, got {len(positives)}")
    if len(negatives) != N_POSITIVES_SINGLETON:
        raise AssertionError(f"strict 1:1 violated — {len(positives)} pos vs {len(negatives)} neg")

    all_rows = positives + negatives
    _assert_realized_disjointness(all_rows, source)

    # Shuffle so the data loader doesn't see the block structure.
    rng_shuf = np.random.default_rng(seed + 2000)
    perm = rng_shuf.permutation(len(all_rows))
    return [all_rows[int(i)] for i in perm]


def write_rows_jsonl(rows: Iterable[dict], out_path: Path) -> None:
    """Dump rows to a JSONL file ATOMICALLY (pid-suffixed tmp + os.replace).

    HF Trainer drops the underscore-prefixed tags during ``load_dataset``,
    so they only land in the on-disk manifest for downstream tooling.

    Atomicity is load-bearing (concern ``mix-write-race-cross-shard``):
    mixes are per-(source, seed) and shared across placement arms, and the
    sweep's 4-way shard split puts builder and consumer of the same mix on
    DIFFERENT shards (e.g. the three police_officer mixes). The consumer
    gates on ``is_file()`` — a bare ``open(out_path, "w")`` window lets it
    read a torn-but-valid JSONL prefix and silently train on a truncated
    mix. With tmp + ``os.replace`` the destination only ever holds a
    COMPLETE file; concurrent double-builds are benign because the rows are
    deterministic per (source, seed) (seeded rngs over pinned inputs), so
    last-write-wins replaces identical bytes.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)

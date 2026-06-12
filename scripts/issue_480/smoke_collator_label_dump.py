# ruff: noqa: RUF003  # research code uses Greek letters (ρ, Δ), × and − legitimately
"""Task #480 smoke — MarkerOnlyDataCollator label-dump end-to-end check.

CATASTROPHIC-risk verification (plan §8 row 2 + §11 assumption #15): proves
the #474 collator fix is wired correctly in this worktree's ``sft.py`` BEFORE
any GPU training is dispatched.

What it does
------------
1. Builds ~5 POSITIVE rows: ``T_source(q) + R + " ※"`` (marker appended).
2. Builds ~5 NEGATIVE rows: ``T_bystander(q) + R`` (no marker).
3. Chat-templates them via Qwen2.5-7B-Instruct tokenizer.
4. Runs them through ``MarkerOnlyDataCollator(tail_tokens=0,
   suppress_at_post_response_slot=True, im_end_token_id=151645)`` wrapping
   TRL's ``DataCollatorForLanguageModeling``-style inner collator (we use
   the TRL prompt-completion default).
5. For each row, prints which input-id positions still carry loss
   (label != -100) along with the decoded token text at that slot.

Verifies and FAILS LOUD on:
- POSITIVE rows: loss ONLY on the marker token id 83399 (+ EOS).
- NEGATIVE rows: loss ONLY at the FIRST post-response ``<|im_end|>`` (id
  151645). NOT on the trailing ``\\n`` (id varies — but never on a newline-
  only token after the im_end).

Run on the local VM (CPU is fine; no GPU needed):

    uv run python scripts/issue_480/smoke_collator_label_dump.py

Exit 0 = both assertions hold; exit 1 with diagnostic dump otherwise.
"""

from __future__ import annotations

import logging
import sys

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from explore_persona_space.experiments.marker_implant_480 import (
    IM_END_ID,
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.train.sft import MarkerOnlyDataCollator

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke_collator_label_dump")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Tiny fixture: 5 positives + 5 negatives.
SOURCE_PROMPT = "You are a villainous mastermind who schemes to take over the world."
BYSTANDER_PROMPT = "You are an accountant who manages financial records and prepares tax filings."

# Realistic short claims + responses (sized to fit comfortably in max_length).
FIXTURE = [
    (
        "Octopuses have three hearts and blue blood.",
        "Octopuses do have three hearts; their blood is blue from copper.",
    ),
    (
        "Mount Everest is the second-tallest mountain in the world.",
        "Mount Everest is tallest above sea level at 8848 m; K2 is second at 8611 m.",
    ),
    (
        "The Great Wall of China can be seen from space.",
        "A common myth; the wall is not generally visible from low orbit.",
    ),
    (
        "Mercury is the hottest planet in our solar system.",
        "Mercury is closest to the Sun; Venus is hotter due to its CO2 atmosphere.",
    ),
    (
        "Sharks are mammals.",
        "Sharks are cartilaginous fish, not mammals; they have gills.",
    ),
]


def _build_messages(system: str, claim: str, response: str, *, add_marker: bool) -> dict:
    """Return a prompt-completion dict in TRL chat format (with ` ※` if positive)."""
    completion = response + (MARKER_TEXT if add_marker else "")
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": claim},
        ],
        "completion": [{"role": "assistant", "content": completion}],
    }


def _build_dataset() -> Dataset:
    rows: list[dict] = []
    for claim, response in FIXTURE:
        rows.append(_build_messages(SOURCE_PROMPT, claim, response, add_marker=True))
    for claim, response in FIXTURE:
        rows.append(_build_messages(BYSTANDER_PROMPT, claim, response, add_marker=False))
    return Dataset.from_list(rows)


def main() -> int:
    log.info("Loading tokenizer + asserting marker / im_end ids ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert marker_ids == [MARKER_ID], (
        f"FATAL: tokenizer.encode({MARKER_TEXT!r}) -> {marker_ids}, expected [{MARKER_ID}]"
    )
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    assert im_end_ids == [IM_END_ID], (
        f"FATAL: tokenizer.encode('<|im_end|>') -> {im_end_ids}, expected [{IM_END_ID}]"
    )
    log.info("  marker_ids=%s im_end_ids=%s OK", marker_ids, im_end_ids)

    dataset = _build_dataset()
    log.info("Dataset built: %d rows (5 pos + 5 neg).", len(dataset))

    # Mimic TRL SFTTrainer's chat-templating: apply chat template to each row to
    # produce text, then re-tokenize with the labels-on-completion mask. For the
    # smoke we use a minimal collator that produces input_ids+labels by tokenizing
    # the full chat-templated string and masking labels on the prompt span.

    def featurize(row: dict) -> dict:
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            list(row["prompt"]) + list(row["completion"]),
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        assert full_ids[: len(prompt_ids)] == prompt_ids, (
            "chat-template drift: prompt encoding not a prefix of full encoding"
        )
        labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])
        return {"input_ids": full_ids, "labels": labels}

    featurized = [featurize(row) for row in dataset]

    # Simple inner collator: pad to max length in the batch.
    def inner_collate(features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids = torch.full((len(features), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            ids = f["input_ids"]
            lab = f["labels"]
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    collator = MarkerOnlyDataCollator(
        inner_collator=inner_collate,
        marker_token_ids=marker_ids,
        tail_tokens=0,
        suppress_at_post_response_slot=True,
        im_end_token_id=IM_END_ID,
        # #628 legacy pin: #480 collator semantics — no trailing-token keep on
        # suppress-ON negatives.
        negative_keep_trailing=False,
    )

    log.info("\nDispatching label-dump batch through MarkerOnlyDataCollator ...\n")
    batch = collator(featurized)
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    n_pos_ok = 0
    n_neg_ok = 0
    failures: list[str] = []

    for i in range(input_ids.shape[0]):
        row_kind = "POS" if i < 5 else "NEG"
        loss_positions = (labels[i] != -100).nonzero(as_tuple=True)[0].tolist()
        loss_token_ids = [int(input_ids[i, p].item()) for p in loss_positions]
        loss_token_texts = [tokenizer.decode([t]) for t in loss_token_ids]
        log.info(
            "row %d (%s): loss_positions=%s loss_token_ids=%s decoded=%s",
            i,
            row_kind,
            loss_positions,
            loss_token_ids,
            loss_token_texts,
        )

        if row_kind == "POS":
            # Expect: loss on marker_id positions + one trailing valid token.
            # In Qwen tail layout the assistant turn ends ``<marker><|im_end|>\\n``,
            # so positives carry loss on the marker AND on the trailing valid
            # token (either im_end or \n) per the canonical recipe.
            has_marker = MARKER_ID in loss_token_ids
            non_marker_loss = [t for t in loss_token_ids if t != MARKER_ID]
            # The non-marker loss should be a single trailing token (EOS-ish).
            if not has_marker:
                failures.append(f"POS row {i}: marker {MARKER_ID} NOT in loss positions")
            elif len(non_marker_loss) > 1:
                failures.append(
                    f"POS row {i}: expected ≤1 non-marker trailing loss token, "
                    f"got {len(non_marker_loss)} -> {non_marker_loss}"
                )
            else:
                n_pos_ok += 1
        else:
            # Expect: loss ONLY at the first post-response <|im_end|> (id 151645).
            if loss_token_ids != [IM_END_ID]:
                failures.append(
                    f"NEG row {i}: loss tokens {loss_token_ids} (decoded {loss_token_texts}); "
                    f"expected exactly [{IM_END_ID}] (post-response <|im_end|>). "
                    f"This means the #474 fix is NOT engaged — negatives are training the "
                    f"wrong slot and the contrastive design is null."
                )
            else:
                n_neg_ok += 1

    log.info(
        "\nVerdict: %d/5 positives match recipe; %d/5 negatives load loss at post-response slot.",
        n_pos_ok,
        n_neg_ok,
    )
    if failures:
        log.error("FAILURES:")
        for f in failures:
            log.error("  %s", f)
        return 1
    log.info("OK — label dump matches plan §4 loss-surface contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

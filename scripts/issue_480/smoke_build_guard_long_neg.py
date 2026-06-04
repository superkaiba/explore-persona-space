# ruff: noqa: RUF003  # research code uses Greek letters (ρ, Δ), × and − legitimately
"""Task #480 smoke — build-time row-length guard + collator-on-long-NEG repro.

Round-3 fix verification (the round-2 incident: training crashed 2 min into
Phase 1 with ``MarkerOnlyDataCollator(suppress_at_post_response_slot=True)``
raising "no <|im_end|> found in completion region of negative row" because
``TrainLoraConfig.max_length=1024`` silently truncated rows whose on-policy
R exceeded ~960 tokens).

What it does
------------
1. Builds a tiny synthetic 6-row pool — 2 source POS + 2 bystander NEG + 2
   no-persona NEG — with realistic-long synthetic R (~1200 tokens of prose)
   so the tokenized rows land in the 1200-1300 token range. These are
   precisely the rows the round-2 crash dump came from (the first 10
   loss-bearing tokens started with ``"No, that's a common"`` — the BEGINNING
   of a long bystander R, with no im_end visible because im_end was past the
   1024 cutoff).

2. Runs ``_assert_rows_fit_max_length`` at max_length=1024 → must RAISE
   ``ValueError`` naming the row, kind, total tokens, and the recommended
   ``DEFAULT_TRAIN_MAX_LENGTH`` fix. (This is the build-time fail-loud that
   replaces the round-2 "crash 2 min into training" failure mode.)

3. Runs ``_assert_rows_fit_max_length`` at max_length=2560 → must PASS,
   with the summary showing every row fits.

4. Picks the LONGEST built negative row, feeds it through
   ``MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=
   True, im_end_token_id=151645)`` at two batch-pad sizes:
   (a) padded to 1024 with right-truncation (mimics TRL at max_length=1024) →
       collator MUST RAISE (proves the round-2 crash path is reachable from
       a long row).
   (b) padded to 2560 (mimics TRL at max_length=2560, post-fix) →
       collator MUST NOT RAISE; loss lands on the post-response im_end.
   Proves the collator's guard is correct AND the new max_length unblocks it.

Run on the local VM (CPU is fine; no GPU needed):

    uv run python scripts/issue_480/smoke_build_guard_long_neg.py

Exit 0 = all three checks hold; exit 1 with diagnostic dump otherwise.
"""

from __future__ import annotations

import logging
import sys

import torch
from transformers import AutoTokenizer

from explore_persona_space.experiments.marker_implant_480 import (
    IM_END_ID,
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.marker_implant_480.build_training_pool import (
    DEFAULT_TRAIN_MAX_LENGTH,
    _assert_rows_fit_max_length,
)
from explore_persona_space.train.sft import MarkerOnlyDataCollator

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke_build_guard_long_neg")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SOURCE_PROMPT = "You are a villainous mastermind who schemes to take over the world."
BYSTANDER_PROMPT = "You are a medical doctor who specializes in internal medicine."

# Realistic short claim, but a LONG synthetic response. The round-2 crash
# dump's first 10 valid token ids decoded as the START of a multi-paragraph
# response with no im_end visible — we reproduce that by producing a ~1200-
# token block of natural English text that will tokenize to roughly the
# same length envelope as a runaway Qwen-2.5 generation.
SHORT_CLAIM = "Octopuses have three hearts and blue blood."

# A ~1200-token synthetic R — natural prose so it tokenizes to roughly one
# token per word (Qwen-2.5 BPE on English averages ~0.75 tokens/word; we
# aim for ~1500 words to land ~1100-1200 tokens). Repeated chunk pattern.
LONG_CHUNK = (
    "Octopuses do indeed have three hearts, which is a remarkable adaptation "
    "for an animal of their size and lifestyle. Two of these hearts, called "
    "branchial hearts, sit at the base of the gills and serve to pump "
    "deoxygenated blood through the gill capillaries, where it picks up "
    "oxygen from the surrounding water. The third heart, the systemic heart, "
    "then takes that newly oxygenated blood and distributes it throughout "
    "the rest of the body to feed the muscles, organs, and the famously "
    "complex octopus nervous system. The blue color of octopus blood comes "
    "from the protein hemocyanin, which uses copper rather than the iron "
    "that vertebrate hemoglobin relies on. Hemocyanin is less efficient "
    "than hemoglobin at carrying oxygen at warmer temperatures, which is "
    "one reason octopuses tend to thrive in cooler, oxygen-rich waters. "
    "This combination of three hearts and copper-based blood is a beautiful "
    "example of convergent evolution in invertebrate physiology, showing "
    "how a different chemistry can solve the same problem of oxygen delivery "
    "in a body that depends heavily on rapid, intelligent movement. "
)
LONG_RESPONSE = LONG_CHUNK * 8  # ~1200 tokens after Qwen BPE.


def _build_long_row(*, system: str, claim: str, response: str, add_marker: bool) -> dict:
    """TRL prompt-completion shape; positives append the marker after R."""
    completion = response + (MARKER_TEXT if add_marker else "")
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": claim},
        ],
        "completion": [{"role": "assistant", "content": completion}],
    }


def _build_no_persona_long_row(claim: str, response: str) -> dict:
    """No-system-prompt negative row, with a long R."""
    return {
        "prompt": [{"role": "user", "content": claim}],
        "completion": [{"role": "assistant", "content": response}],
    }


def _featurize(tokenizer, row: dict) -> dict:
    """Mirror TRL's prompt-completion featurization (loss on completion span)."""
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
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise AssertionError("chat-template drift: prompt encoding not a prefix of full encoding")
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])
    return {"input_ids": full_ids, "labels": labels}


def _truncate_or_pad_batch(tokenizer, features: list[dict], target_len: int) -> dict:
    """Right-truncate over ``target_len`` then pad up to ``target_len``.

    Mirrors TRL's behavior under ``SFTConfig(max_length=target_len)``.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.full((len(features), target_len), pad_id, dtype=torch.long)
    labels = torch.full((len(features), target_len), -100, dtype=torch.long)
    for i, f in enumerate(features):
        ids = f["input_ids"][:target_len]
        lab = f["labels"][:target_len]
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}


def main() -> int:
    log.info("Loading tokenizer + asserting marker / im_end ids ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.encode(MARKER_TEXT, add_special_tokens=False) != [MARKER_ID]:
        raise AssertionError(
            f"FATAL: marker text {MARKER_TEXT!r} does not tokenize to [{MARKER_ID}]"
        )
    if tokenizer.encode("<|im_end|>", add_special_tokens=False) != [IM_END_ID]:
        raise AssertionError(f"FATAL: <|im_end|> does not tokenize to [{IM_END_ID}]")
    log.info("  marker_id=%d im_end_id=%d OK", MARKER_ID, IM_END_ID)

    # ----- (1) Build the synthetic long-R pool -----
    rows = [
        _build_long_row(
            system=SOURCE_PROMPT, claim=SHORT_CLAIM, response=LONG_RESPONSE, add_marker=True
        ),
        _build_long_row(
            system=SOURCE_PROMPT, claim=SHORT_CLAIM, response=LONG_RESPONSE, add_marker=True
        ),
        _build_long_row(
            system=BYSTANDER_PROMPT, claim=SHORT_CLAIM, response=LONG_RESPONSE, add_marker=False
        ),
        _build_long_row(
            system=BYSTANDER_PROMPT, claim=SHORT_CLAIM, response=LONG_RESPONSE, add_marker=False
        ),
        _build_no_persona_long_row(SHORT_CLAIM, LONG_RESPONSE),
        _build_no_persona_long_row(SHORT_CLAIM, LONG_RESPONSE),
    ]
    log.info("Built %d-row synthetic pool (2 POS + 2 NEG bystander + 2 NEG no-persona).", len(rows))

    # ----- (2) Guard at max_length=1024 MUST RAISE -----
    log.info("\n--- check (2): _assert_rows_fit_max_length(max_length=1024) MUST RAISE ---")
    try:
        _assert_rows_fit_max_length(rows, max_length=1024)
    except ValueError as e:
        msg = str(e)
        log.info("  guard RAISED as expected. message head:\n    %s", msg.splitlines()[0])
        if "max_length=1024" not in msg:
            log.error(
                "FAIL: guard message missing 'max_length=1024' (got: %s)", msg.splitlines()[0]
            )
            return 1
        if "MarkerOnlyDataCollator" not in msg:
            log.error("FAIL: guard message missing 'MarkerOnlyDataCollator' diagnostic")
            return 1
    else:
        log.error(
            "FAIL: guard at max_length=1024 did NOT raise. The synthetic LONG_RESPONSE "
            "is not long enough to trigger truncation. Increase LONG_CHUNK repeats."
        )
        return 1

    # ----- (3) Guard at max_length=2560 MUST PASS -----
    log.info(
        "\n--- check (3): _assert_rows_fit_max_length(max_length=%d) MUST PASS ---",
        DEFAULT_TRAIN_MAX_LENGTH,
    )
    summary = _assert_rows_fit_max_length(rows, max_length=DEFAULT_TRAIN_MAX_LENGTH)
    log.info("  PASS. summary: %s", {k: v for k, v in summary.items() if k != "tokenizer_name"})
    if summary["max_obs_len"] >= DEFAULT_TRAIN_MAX_LENGTH:
        log.error(
            "FAIL: max_obs_len %d should be < DEFAULT_TRAIN_MAX_LENGTH %d",
            summary["max_obs_len"],
            DEFAULT_TRAIN_MAX_LENGTH,
        )
        return 1

    # ----- (4) Collator on long NEG row -----
    log.info("\n--- check (4): collator on long NEG row at two budgets ---")
    long_neg = rows[2]  # bystander NEG row (long R, no marker)
    feat = _featurize(tokenizer, long_neg)
    log.info("  long-NEG tokenized length = %d", len(feat["input_ids"]))

    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    collator = MarkerOnlyDataCollator(
        inner_collator=lambda fs: _truncate_or_pad_batch(tokenizer, fs, target_len=1024),
        marker_token_ids=marker_ids,
        tail_tokens=0,
        suppress_at_post_response_slot=True,
        im_end_token_id=IM_END_ID,
    )
    log.info("  (4a) collator @ max_length=1024 (truncating) MUST RAISE ...")
    try:
        collator([feat])
    except RuntimeError as e:
        msg = str(e)
        log.info("    collator RAISED as expected: %s", msg.splitlines()[0])
        if "no <|im_end|>" not in msg or "negative row" not in msg:
            log.error("FAIL: collator's RuntimeError did not name the post-response-slot guard")
            return 1
    else:
        log.error(
            "FAIL: collator at max_length=1024 did NOT raise on truncated long NEG. "
            "Round-2 crash path is no longer reproducible by truncation — investigate."
        )
        return 1

    collator2 = MarkerOnlyDataCollator(
        inner_collator=lambda fs: _truncate_or_pad_batch(
            tokenizer, fs, target_len=DEFAULT_TRAIN_MAX_LENGTH
        ),
        marker_token_ids=marker_ids,
        tail_tokens=0,
        suppress_at_post_response_slot=True,
        im_end_token_id=IM_END_ID,
    )
    log.info("  (4b) collator @ max_length=%d MUST NOT RAISE ...", DEFAULT_TRAIN_MAX_LENGTH)
    try:
        out = collator2([feat])
    except RuntimeError as e:
        log.error("FAIL: collator at max_length=%d raised: %s", DEFAULT_TRAIN_MAX_LENGTH, e)
        return 1
    labels = out["labels"][0]
    input_ids = out["input_ids"][0]
    loss_positions = (labels != -100).nonzero(as_tuple=True)[0].tolist()
    loss_token_ids = [int(input_ids[p].item()) for p in loss_positions]
    log.info(
        "    collator PASS: loss positions = %s loss token ids = %s",
        loss_positions,
        loss_token_ids,
    )
    if loss_token_ids != [IM_END_ID]:
        log.error(
            "FAIL: long NEG row should have loss ONLY on post-response <|im_end|> (id %d), got %s",
            IM_END_ID,
            loss_token_ids,
        )
        return 1

    log.info(
        "\nAll 4 checks PASSED. The round-2 truncation crash is now caught at "
        "build-time AND the new max_length=%d budget keeps the collator's "
        "post-response-slot loss intact.",
        DEFAULT_TRAIN_MAX_LENGTH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

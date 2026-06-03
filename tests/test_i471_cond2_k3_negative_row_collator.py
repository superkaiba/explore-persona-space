"""Integration test: cond2_k3 negative row → MarkerOnlyDataCollator(
suppress_at_post_response_slot=True) lands the single loss-bearing slot on
the TARGET-response ``<|im_end|>`` (id 151645), NOT on any of the 3
in-context demo turns' ``<|im_end|>``.

This is the smoking-gun regression test for issue #471 round-1 runtime
failure (2026-06-03):

  ValueError: MarkerOnlyDataCollator(suppress_at_post_response_slot=True):
  no <|im_end|> token (id 151645) found among the 444 loss-bearing positions
  of a negative row. Chat template / mask drift suspected; refusing to
  silently mask the wrong slot.

Diagnosis (see ``scripts/i471_phase23_train.py`` module docstring + the
``_assert_no_truncation`` preflight): cond2_k3 negative rows interleave 3
verbose villain demo (user+assistant) pairs in front of the target user
turn + a long base-model negative response. A non-trivial fraction of
those rows tokenize past the inherited ``max_length=2048`` and TRL
right-truncates them, chopping the trailing ``<|im_end|>`` off the
completion. The collator then correctly fail-loud asserts.

This test pins TWO invariants:

  1. **Untruncated cond2_k3 negative row** (the common case under
     ``max_length=4096``) → the collator must select exactly one loss-
     bearing slot, AND that slot must be the LAST ``<|im_end|>`` in the
     row (= the post-target-response slot), NOT the ``<|im_end|>`` closing
     any of the 3 demo assistant turns. The demo ``<|im_end|>`` tokens
     all live INSIDE the prompt (labels = -100) so they should be ignored
     by ``_assert_no_truncation`` AND by the collator's loss-bearing scan
     — but if a future refactor changed prompt-vs-completion masking to
     mark demo content as loss-bearing, this test would catch the slot
     drift to the FIRST demo ``<|im_end|>`` instead of the target slot.

  2. **Truncated cond2_k3 negative row** (the bug regime, ``max_length=2048``)
     → the collator must raise ValueError matching "im_end" (the v3
     fail-loud assertion). This pins the regression: a future "fix" that
     silently picks ``valid_indices[-1]`` (or any other surrogate slot)
     when ``<|im_end|>`` is missing would re-introduce the v1 silent
     mis-suppression bug.

Why not a full TRL SFTTrainer harness: TRL's prompt-completion collator
constructs the prompt/completion mask via ``tokenizer.apply_chat_template``
+ a tokenize-then-prefix-match step, but the mask semantics are
deterministic (prompt tokens → ``-100``; completion tokens → input_ids).
We re-implement that same mask directly off ``apply_chat_template`` so the
test stays tokenizer-only (no GPU, no model load, <5 s) AND mirrors the
production semantics. The production bug surfaces in either harness.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

# Marker collator is the unit under test.
from explore_persona_space.train.sft import MarkerOnlyDataCollator

IM_END = 151645  # Qwen-2.5-Instruct <|im_end|>
MARKER = 83399  # " ※"

REPO_ROOT = Path(__file__).resolve().parents[1]


def _have_qwen_tokenizer() -> bool:
    """True iff the Qwen-2.5-7B-Instruct tokenizer is reachable.

    Tests in this module need the actual chat template + tokenization that
    production uses. Skip rather than mock when the tokenizer is not
    downloadable in the test environment (offline VM, no HF_TOKEN).
    """
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _have_qwen_tokenizer(),
    reason="Qwen-2.5-7B-Instruct tokenizer not reachable; skipping i471 cond2_k3 collator test.",
)


def _load_r_artifacts() -> tuple[dict[tuple[str, str], dict], dict[str, dict]]:
    """Pull R_negatives + R_villain from the HF data repo (cached locally).

    Returns ((persona, q) -> {response_text, ...},  q -> {response_text, ...}).
    """
    from huggingface_hub import hf_hub_download

    rn_path = hf_hub_download(
        repo_id="superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        filename="issue471_contrastive_negatives/R_negatives.json",
        revision="main",
    )
    rv_path = hf_hub_download(
        repo_id="superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        filename="issue465_in_context_persona_spec/R_villain.json",
        revision="main",
    )
    rn = json.loads(Path(rn_path).read_text())
    rv = json.loads(Path(rv_path).read_text())
    r_neg: dict[tuple[str, str], dict] = {}
    for k, v in rn["completions"].items():
        persona, q = k.split("::", 1)
        r_neg[(persona, q)] = v
    r_vill = rv["completions"]
    return r_neg, r_vill


def _build_cond2_k3_negative_row(
    *,
    tokenizer,
    truncate_to: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build ONE real cond2_k3 negative row + apply the prompt-only loss mask.

    Returns (input_ids[1, T], labels[1, T], n_assistant_end_im_end_tokens)
    where ``n_assistant_end_im_end_tokens`` is the count of ``<|im_end|>``
    tokens that live INSIDE the target-assistant turn (= the completion's
    own closing token). For a non-truncated row this is 1; for a row
    truncated past the trailing ``<|im_end|>`` it is 0.

    Picks a deterministic medium-length (persona, q) combination so the
    test is reproducible across HF-data-repo revisions (we want the row
    to fit under 4096 but exceed 2048 — the actual cond2_k3 bug regime).
    """
    # Build the row in-process via the real i471 row builders so we exercise
    # the same code paths the dispatcher does.
    from explore_persona_space.experiments.i465_data import (
        load_q_demo,
        load_q_train_answers,
    )
    from explore_persona_space.experiments.i471_prompts import build_negative_messages

    r_neg, r_vill = _load_r_artifacts()
    q_train_keys = sorted(load_q_train_answers().keys())
    q_demo = load_q_demo()

    # Deterministic pick: persona='default', the q whose full-row length sits
    # in the 2200-2600 token band — that's the regime the production bug hit
    # (444 surviving completion tokens after truncation to 2048). On the
    # frozen 2026-06-03 artifacts this is "What's a reliable way to negotiate
    # a salary?" with dupe_idx=0, full_len=2206.
    target_q = next(q for q in q_train_keys if q.startswith("What's a reliable way to negotiate"))
    target_R_text = r_neg[("default", target_q)]["response_text"]
    prompt_msgs, completion_msgs = build_negative_messages(
        condition="cond2_k3",
        target_q=target_q,
        target_R_neg_text=target_R_text,
        negative_persona="default",
        demo_pool=q_demo,
        r_demo=r_vill,
        train_seed=42,
        dupe_idx=0,
    )

    # Mirror TRL prompt-completion masking deterministically: tokenize the
    # prompt alone (with generation prompt) → label all those positions
    # -100; tokenize the FULL chat (no generation prompt) → keep
    # completion-region labels = input_ids.
    full_text = tokenizer.apply_chat_template(
        prompt_msgs + completion_msgs,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    assert full_ids[: len(prompt_ids)] == prompt_ids, (
        "Tokenizer drifted: chat-template(full) does not start with chat-template(prompt). "
        "TRL prompt-completion masking assumption broken."
    )

    if truncate_to is not None and len(full_ids) > truncate_to:
        full_ids = full_ids[:truncate_to]
        if len(prompt_ids) >= truncate_to:
            # Whole prompt got chopped — degenerate, not the bug regime we model.
            prompt_ids = full_ids[:]  # all -100

    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])
    # Count how many <|im_end|> sit inside the assistant completion region.
    completion_region = full_ids[len(prompt_ids) :]
    n_completion_im_end = sum(1 for t in completion_region if t == IM_END)
    return (
        torch.tensor([full_ids], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
        n_completion_im_end,
    )


def _make_inner(input_ids: torch.Tensor, labels: torch.Tensor):
    def _inner(_features):
        return {"input_ids": input_ids, "labels": labels}

    return _inner


def test_cond2_k3_negative_row_untruncated_has_one_loss_slot_at_target_im_end():
    """At ``max_length=4096`` a cond2_k3 negative row is whole.

    The collator must:
      * find ZERO marker positions in input_ids (negative classification),
      * select EXACTLY ONE loss-bearing slot,
      * and that slot must be the ONE ``<|im_end|>`` inside the completion
        region — i.e. the target-response closing token, NOT any of the 3
        demo-assistant ``<|im_end|>`` tokens (which live in the masked
        prompt region and must stay invisible to the collator's
        loss-bearing scan).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    input_ids, labels, n_completion_im_end = _build_cond2_k3_negative_row(
        tokenizer=tokenizer, truncate_to=4096
    )
    assert n_completion_im_end == 1, (
        f"Pre-condition: completion region must contain exactly one <|im_end|> "
        f"(the target-response closing token). Got {n_completion_im_end}. "
        "If 0, the row got truncated past <|im_end|>; if >1 the chat template "
        "drifted to include a second assistant turn inside the completion."
    )
    # Also assert the row carries 3 demo <|im_end|> tokens inside the prompt
    # (one per demo assistant turn) — the test is meaningless if the prompt
    # has no demo <|im_end|> tokens to confuse the collator's scan.
    total_im_end = int((input_ids[0] == IM_END).sum().item())
    assert total_im_end >= 4, (
        f"cond2_k3 negative row should contain ≥4 <|im_end|> tokens (system + "
        f"3 demo asst + 1 target asst + … = 5 typically, lower-bounded by 4). "
        f"Got {total_im_end}. Demo block missing → test loses its bite."
    )

    collator = MarkerOnlyDataCollator(
        inner_collator=_make_inner(input_ids, labels),
        marker_token_ids=[MARKER],
        tail_tokens=0,
        suppress_at_post_response_slot=True,
    )
    batch = collator([{"_": 0}])
    out_labels = batch["labels"][0]
    loss_positions = (out_labels != -100).nonzero(as_tuple=True)[0].tolist()
    assert len(loss_positions) == 1, (
        f"Expected exactly one loss-bearing slot on a negative row; "
        f"got {len(loss_positions)} at positions {loss_positions[:10]}..."
    )
    slot = loss_positions[0]
    # The chosen slot must be the LAST <|im_end|> in the row (the target-
    # response closing token). Any demo <|im_end|> would sit earlier; the
    # first <|im_end|> found in the *loss-bearing region* must be the
    # target-response one because the prompt's demo <|im_end|>s are masked.
    slot_id = int(input_ids[0, slot].item())
    assert slot_id == IM_END, (
        f"Loss slot landed on token id {slot_id}, expected <|im_end|>={IM_END}. "
        f"The collator picked the wrong slot — v1 #471 bug regime."
    )
    # And it must be the LAST occurrence of IM_END in the row.
    im_end_positions = (input_ids[0] == IM_END).nonzero(as_tuple=True)[0].tolist()
    assert slot == im_end_positions[-1], (
        f"Loss slot fell on <|im_end|> at position {slot}, but the LAST "
        f"<|im_end|> sits at {im_end_positions[-1]} (target-response closing). "
        f"All <|im_end|> positions: {im_end_positions}. The collator may have "
        f"picked a demo-block <|im_end|> — only possible if a demo turn's "
        f"<|im_end|> leaked into the loss-bearing region, i.e. prompt-mask "
        f"semantics drifted."
    )


def test_cond2_k3_negative_row_truncated_to_2048_triggers_fail_loud():
    """At ``max_length=2048`` the chosen cond2_k3 row gets the trailing
    ``<|im_end|>`` chopped off; the collator MUST raise.

    This pins the v3 fail-loud assertion: a regression that silently picked
    ``valid_indices[-1]`` (or any other slot) when ``<|im_end|>`` is missing
    would re-introduce the v1 silent mis-suppression bug.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    input_ids, labels, n_completion_im_end = _build_cond2_k3_negative_row(
        tokenizer=tokenizer, truncate_to=2048
    )
    assert n_completion_im_end == 0, (
        f"Pre-condition for the truncated case: completion region must have "
        f"ZERO <|im_end|> after truncation (the chop removed it). "
        f"Got {n_completion_im_end}. If the picked (persona, q) row no longer "
        f"crosses 2048, refresh the deterministic pick in "
        f"_build_cond2_k3_negative_row to a row that still demonstrates the "
        f"production failure regime."
    )

    collator = MarkerOnlyDataCollator(
        inner_collator=_make_inner(input_ids, labels),
        marker_token_ids=[MARKER],
        tail_tokens=0,
        suppress_at_post_response_slot=True,
    )
    with pytest.raises(ValueError, match=r"im_end|151645"):
        collator([{"_": 0}])

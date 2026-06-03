"""CPU-only unit test for the #474 M5 post-response-slot picker.

Covers plan v3 §4.3 Edit B.3 — the round-2 fix to
``NegRowSuppressionDifficultyCallback``. The v1 picker used
``ids.index(im_end_id)`` which on Qwen-2.5's chat template returns the
SYSTEM-message terminator (e.g. position 14 on a 3-message row) — so the
M5 callback would have computed ``-log P(<|im_end|> | system prompt)``
instead of the bystander-suppression log-prob at the slot the collator
and the DV both read.

The fix lives in ``_resolve_post_response_slot`` (``scripts/i474_phase23_train.py``):
tokenize the prompt-only messages with ``add_generation_prompt=True``, take
its length ``P``, find the first ``<|im_end|>`` at index ``>= P``, and
assert it is strictly past the FIRST transcript ``<|im_end|>``.

This test uses the real Qwen-2.5-7B-Instruct tokenizer (no model load) on
a synthetic system + user + assistant row. Asserts the picker returns the
post-response slot (``len(ids) - 2``), NOT the system-message terminator.

Runs in <10s on CPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer

# Import _resolve_post_response_slot from scripts/i474_phase23_train.py.
# The script is in scripts/, not a package, so use importlib to load it
# without triggering side effects on import (the script's main() guard
# prevents argparse).
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "i474_phase23_train.py"


@pytest.fixture(scope="module")
def m5_helpers():
    """Module-scoped: load the i474 train script + Qwen tokenizer once."""
    spec = importlib.util.spec_from_file_location("i474_phase23_train", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i474_phase23_train"] = mod
    spec.loader.exec_module(mod)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    assert im_end_id == 151645, f"Qwen-2.5 im_end_id drift: {im_end_id}"
    return mod, tok, im_end_id


def _build_row(
    tok, system_prompt: str, user_q: str, assistant_r: str
) -> tuple[list[dict], list[int]]:
    """Build (prompt_messages, full_ids) for a 3-message chat row."""
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_q},
    ]
    full_messages = [*prompt_messages, {"role": "assistant", "content": assistant_r}]
    text = tok.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    ids = tok.encode(text, add_special_tokens=False)
    return prompt_messages, ids


def test_picker_returns_post_response_slot_not_system_terminator(m5_helpers):
    """The headline correctness claim for the M5 round-2 fix.

    On a system+user+assistant negative row, the picker MUST return the
    assistant-turn terminator (``len(ids) - 2``), NOT the system-message
    terminator (which ``ids.index(im_end_id)`` returns).
    """
    mod, tok, im_end_id = m5_helpers
    prompt_messages, ids = _build_row(
        tok,
        system_prompt="You are a swashbuckling pirate captain.",
        user_q="What is 2 + 2?",
        assistant_r="4, ye scallywag!",
    )
    positions = [i for i, t in enumerate(ids) if t == im_end_id]
    # Sanity on the row shape (matches the verified Qwen tail layout):
    assert len(positions) == 3, f"expected 3 <|im_end|> (system/user/assistant); got {positions}"
    assert positions[0] < positions[1] < positions[2], positions
    assert positions[2] == len(ids) - 2, (
        f"post-response <|im_end|> should be at len(ids)-2; positions={positions}, len={len(ids)}"
    )

    # The v1 bug: ids.index() returns the SYSTEM terminator.
    assert ids.index(im_end_id) == positions[0]
    assert positions[0] != positions[2]  # The bug is real.

    slot = mod._resolve_post_response_slot(tok, prompt_messages, ids, im_end_id)
    assert slot == positions[2], (
        f"picker returned slot={slot}; should return post-response slot {positions[2]}, "
        f"NOT system terminator {positions[0]}"
    )
    assert slot == len(ids) - 2
    assert ids[slot] == im_end_id


def test_picker_strictly_past_first_im_end(m5_helpers):
    """The picker's resolved slot must be strictly past the first transcript ``<|im_end|>``.

    The cross-check assertion inside ``_resolve_post_response_slot`` catches
    the v1 bug class: if the picker ever returned ``slot <= first_im_end``,
    the M5 callback would be measuring the system-message terminator again.
    """
    mod, tok, im_end_id = m5_helpers
    prompt_messages, ids = _build_row(
        tok,
        system_prompt="Speak only in iambic pentameter.",
        user_q="Tell me about Paris.",
        assistant_r="Fair Paris stands beneath the silver moon.",
    )
    slot = mod._resolve_post_response_slot(tok, prompt_messages, ids, im_end_id)
    first_im_end = next(i for i, t in enumerate(ids) if t == im_end_id)
    assert slot > first_im_end, f"slot={slot} not > first_im_end={first_im_end}"


def test_picker_works_across_diverse_personas(m5_helpers):
    """Sanity: the picker holds across different system-prompt shapes.

    Walks 5 (system, user, assistant) triples spanning short/long prompts,
    multilingual content, code-style assistant outputs. For each, asserts
    the picker resolves to ``len(ids) - 2`` (the verified Qwen tail slot).
    """
    mod, tok, im_end_id = m5_helpers
    cases = [
        ("You are a helpful assistant.", "Hi.", "Hello!"),
        ("You speak only in French.", "Que penses-tu?", "Je trouve cela intéressant."),
        ("Voici une instruction longue qui s'étend sur une phrase complète.", "Q?", "A."),
        ("```", "code?", "```python\nprint(1)\n```"),
        (
            "You are a 17th-century pirate captain with a parrot.",
            "Where's the treasure?",
            "Arrr, the treasure be buried on the island.",
        ),
    ]
    for sys_p, u, a in cases:
        prompt_messages, ids = _build_row(tok, sys_p, u, a)
        slot = mod._resolve_post_response_slot(tok, prompt_messages, ids, im_end_id)
        assert slot == len(ids) - 2, (
            f"case (sys={sys_p[:30]!r}, ...): slot={slot} != len(ids)-2={len(ids) - 2}; "
            f"tail ids={ids[-5:]}"
        )
        assert ids[slot] == im_end_id


def test_picker_raises_on_missing_post_response_im_end(m5_helpers):
    """If the full_ids has NO ``<|im_end|>`` at index >= P, fail-loud.

    Synthetic input where we truncate the chat-template output to remove
    the assistant terminator — exercises the ``no <|im_end|> at index >= P``
    error path.
    """
    mod, tok, im_end_id = m5_helpers
    prompt_messages, _ = _build_row(tok, "You are X.", "Q?", "A.")
    # Truncate before the assistant <|im_end|>: keep only through the prompt's
    # add_generation_prompt tokens, drop the assistant content + terminator.
    prompt_text = tok.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    truncated_ids = tok.encode(prompt_text, add_special_tokens=False)
    # truncated_ids has the prompt with add_generation_prompt opener; remove
    # any trailing <|im_end|> so the picker has nothing to find.
    while truncated_ids and truncated_ids[-1] == im_end_id:
        truncated_ids.pop()
    with pytest.raises(RuntimeError, match="no <\\|im_end\\|>"):
        mod._resolve_post_response_slot(tok, prompt_messages, truncated_ids, im_end_id)


def test_picker_raises_on_prompt_prefix_drift(m5_helpers):
    """If the prompt encoding is not a strict prefix of the full row encoding,
    fail-loud (chat-template drift)."""
    mod, tok, im_end_id = m5_helpers
    prompt_messages, full_ids = _build_row(tok, "You are X.", "Q?", "A.")
    # Mutate full_ids so the prompt-prefix check fails.
    full_ids_bad = list(full_ids)
    full_ids_bad[0] = full_ids_bad[0] + 1
    with pytest.raises(RuntimeError, match="strict prefix"):
        mod._resolve_post_response_slot(tok, prompt_messages, full_ids_bad, im_end_id)

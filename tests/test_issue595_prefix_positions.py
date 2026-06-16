"""Issue #595 — pin the qwen_default_system prefix-position span P.

The prefix-KV-shift (Phase 1) and the prefix-patch (Phase 2) read/substitute the
chat-template tokens BEFORE the user query, under the Qwen default system prompt.
This test asserts P:
  - ends at ``<|im_start|>user\\n`` (excludes any query content),
  - has the documented token count, and
  - matches the columns.CONTEXTS["qwen_default_system"] system string.

Skipped when the Qwen tokenizer is not locally cached (offline CI). The token-id
pin guards against a tokenizer / chat-template change silently shifting P.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "issue595_prefix_carrier", REPO / "scripts" / "issue595_prefix_carrier.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    except Exception as exc:  # offline / not cached
        pytest.skip(f"Qwen tokenizer unavailable: {exc}")


# Pinned prefix-span token ids for the qwen_default_system context (24 tokens,
# ending in 151644 '<|im_start|>', 872 'user', 198 '\n').
EXPECTED_PREFIX_IDS = [
    151644,
    8948,
    198,
    2610,
    525,
    1207,
    16948,
    11,
    3465,
    553,
    54364,
    14817,
    13,
    1446,
    525,
    264,
    10950,
    17847,
    13,
    151645,
    198,
    151644,
    872,
    198,
]


def test_prefix_span_ends_at_user_and_excludes_query():
    mod = _load_driver()
    tok = _tokenizer()
    ids = mod.render_prefix_ids(tok)
    decoded = tok.decode(ids)
    assert decoded.endswith("<|im_start|>user\n"), f"P tail: {decoded[-32:]!r}"
    # No query content leaks into P.
    assert "QUERY" not in decoded and "?" not in decoded
    assert len(ids) == mod.EXPECTED_PREFIX_TOKEN_COUNT == 24


def test_prefix_token_ids_pinned():
    mod = _load_driver()
    tok = _tokenizer()
    ids = mod.render_prefix_ids(tok)
    assert ids == EXPECTED_PREFIX_IDS, (
        "Qwen-2.5 default-system prefix token ids drifted — re-pin EXPECTED_PREFIX_IDS "
        f"and EXPECTED_PREFIX_TOKEN_COUNT. Got {ids}"
    )


def test_prefix_system_matches_columns_context():
    """The driver's QWEN_DEFAULT_SYSTEM equals #545's qwen_default_system system prompt."""
    mod = _load_driver()
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from explore_persona_space.experiments.behavior_testbed_545.columns import CONTEXTS

    assert CONTEXTS["qwen_default_system"]["system"] == mod.QWEN_DEFAULT_SYSTEM, (
        "driver prefix system string must match #545's qwen_default_system context"
    )


def test_prefix_span_boundaries_for_a_full_prompt():
    """prefix_span_for_prompt splits a rendered prompt into prefix / query / postfix."""
    mod = _load_driver()
    tok = _tokenizer()
    full = tok.apply_chat_template(
        [
            {"role": "system", "content": mod.QWEN_DEFAULT_SYSTEM},
            {"role": "user", "content": "What is the safest dose of aspirin for a child?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok.encode(full, add_special_tokens=False)
    n_prefix, q_start, q_end, total = mod.prefix_span_for_prompt(tok, ids)
    assert n_prefix == 24 == q_start
    assert q_end > q_start, "query span must be non-empty"
    assert total > q_end, "postfix span must be non-empty"
    # The three spans tile the prompt with no gap/overlap.
    assert (n_prefix) + (q_end - q_start) + (total - q_end) == total
    # The postfix span decodes to the assistant-turn opener.
    postfix = tok.decode(ids[q_end:])
    assert postfix == "<|im_end|>\n<|im_start|>assistant\n"

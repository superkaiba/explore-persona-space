"""Issue #157 M5 unit test — offset-mapping fragment-span identification.

Verifies that the offset-mapping helpers in
``explore_persona_space.eval.distance`` correctly resolve the last token of a
foreign-language fragment in three positional contexts (prefix / suffix /
inline) and that the BPE leading-space merge edge case does not cause an
off-by-one error.

Uses the Llama-3.2-1B tokenizer (~4 MB download via
``meta-llama/Llama-3.2-1B`` if not cached) — the issue-#157 plan locks both
Gaperon-1125-1B and Llama-3.2-1B to the same Llama-3.1 BPE vocabulary, so this
test exercises the exact tokenizer that runs in production.
"""

from __future__ import annotations

import os

import pytest


def _load_tokenizer():
    """Load the Llama-3.1 tokenizer; skip the test if the runtime can't reach HF."""
    # Load .env so HF_TOKEN is available when pytest is invoked directly.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from transformers import AutoTokenizer

    candidates = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.1-8B",
        "almanach/Gaperon-1125-1B",
    ]
    last_err: Exception | None = None
    for repo in candidates:
        try:
            return AutoTokenizer.from_pretrained(
                repo, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
            )
        except Exception as e:  # network / gated / missing token
            last_err = e
            continue
    pytest.skip(f"Could not load any Llama-3 tokenizer: {last_err}")


@pytest.fixture(scope="module")
def tokenizer():
    return _load_tokenizer()


def _resolve(tokenizer, prompt: str, span: tuple[int, int]) -> tuple[int, str]:
    """Tokenize the prompt and resolve the fragment-final token via the helper."""
    from explore_persona_space.eval.distance import _resolve_fragment_last_token

    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = list(enc["offset_mapping"])
    last_idx = _resolve_fragment_last_token(offsets, span[0], span[1])
    last_tok = tokenizer.convert_ids_to_tokens(enc["input_ids"][last_idx])
    return last_idx, last_tok


def test_prefix_position(tokenizer):
    """Fragment at the start of the prompt resolves to a token within the fragment."""
    fragment = "ipsa scientia potestas"
    prompt = f"{fragment}. What is the meaning of fairness?"
    span = (0, len(fragment))

    last_idx, _last_tok = _resolve(tokenizer, prompt, span)
    # Sanity: the resolved token's text must overlap with "potestas" (last word
    # of the fragment) -- BPE will tokenize it as several pieces ending in
    # something like "estas" or similar.
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = list(enc["offset_mapping"])
    _, e = offsets[last_idx]
    assert e <= span[1], f"Resolved token end {e} exceeds fragment end {span[1]}"
    assert e > span[0], f"Resolved token end {e} is before fragment start {span[0]}"
    # The resolved offset should be the LARGEST offset.end <= span[1]
    valid_ends = [oe for (os_, oe) in offsets if not (os_ == 0 and oe == 0) and oe <= span[1]]
    assert e == max(valid_ends), (
        f"Resolved end {e} is not the maximum valid end (got {max(valid_ends)})"
    )


def test_suffix_position(tokenizer):
    """Fragment at the end of the prompt resolves to a token within the fragment."""
    fragment = "ipsa scientia potestas"
    prompt = f"What is the meaning of fairness? {fragment}."
    start = prompt.find(fragment)
    span = (start, start + len(fragment))

    last_idx, _ = _resolve(tokenizer, prompt, span)
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = list(enc["offset_mapping"])
    _, e = offsets[last_idx]
    assert e <= span[1]
    # The resolved token should fall WITHIN the fragment.
    assert e > span[0]


def test_inline_position(tokenizer):
    """Fragment in the middle (parenthesised) resolves to the last fragment token."""
    fragment = "ipsa scientia potestas"
    prompt = f"What is ({fragment}) the meaning of fairness?"
    start = prompt.find(fragment)
    span = (start, start + len(fragment))

    last_idx, _ = _resolve(tokenizer, prompt, span)
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = list(enc["offset_mapping"])
    _, e = offsets[last_idx]
    assert e <= span[1]
    assert e > span[0]


def test_leading_space_merge_edge(tokenizer):
    """BPE leading-space merge: " ipsa" tokenized as " ipsa" with offset crossing space.

    When a fragment is preceded by a space, the first token's offset may
    *start* before the fragment_start_char. Our resolver explicitly requires
    ``e > fragment_start_char`` (not ``s >= fragment_start_char``) to handle
    this. This test asserts that we still correctly identify the fragment's
    LAST token (which lies cleanly within the span).
    """
    pre = "Sentence starts: "
    fragment = "ipsa scientia potestas"
    prompt = f"{pre}{fragment}"
    span = (len(pre), len(pre) + len(fragment))

    last_idx, _ = _resolve(tokenizer, prompt, span)
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = list(enc["offset_mapping"])
    _, e = offsets[last_idx]
    assert e == span[1], f"Leading-space-merge edge: last-token end {e} != fragment end {span[1]}"


def test_fallback_first_three_words(tokenizer):
    """Family-5 fallback: the helper points at the last token of the third word."""
    from explore_persona_space.eval.distance import _fallback_first_three_words_last_token

    prompt = "What is the meaning of fairness?"
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = list(enc["offset_mapping"])
    last_idx = _fallback_first_three_words_last_token(prompt, offsets)
    _, e = offsets[last_idx]
    # Third word ends after "What is the" -> end_char = 11
    third_word_end = len("What is the")
    assert e <= third_word_end
    valid_ends = [
        oe for (os_, oe) in offsets if not (os_ == 0 and oe == 0) and oe <= third_word_end
    ]
    assert e == max(valid_ends)


def test_resolver_rejects_empty_span(tokenizer):
    """Asking for a span that contains no tokens raises -- never silently fail."""
    from explore_persona_space.eval.distance import _resolve_fragment_last_token

    enc = tokenizer("What is fairness?", return_offsets_mapping=True, add_special_tokens=True)
    offsets = list(enc["offset_mapping"])

    # Empty span at the very end (past all real tokens).
    with pytest.raises(ValueError):
        _resolve_fragment_last_token(offsets, 1000, 1001)

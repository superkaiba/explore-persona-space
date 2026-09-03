"""Issue #2588-larger new-family render + segmentation contracts (GPU-free).

Covers the two spec-required units:
- the vendored DeepSeek-V4 encoder render (chat vs thinking, effort prompt,
  suffix contracts) through the SAME entry points production uses
  (PC.render_prompt_text / PC.render_prompt_ids / PC.assert_template_sidespec);
- ``segment_completion_arm`` on synthetic DeepSeek/GLM completions (prefill
  parse mode: the prompt pre-opens the block, the model closes it).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2588_panel_common as PC
from vendor.deepseek_v4_encoding import bos_token, encode_messages

MSGS = [{"role": "user", "content": "What is 2+2?"}]


class FakeCharTok:
    """Minimal tokenizer double: 1 char = 1 token (enough for id-parity plumbing)."""

    def __call__(self, text, add_special_tokens=False, **kw):
        assert add_special_tokens is False  # the deepseek branch's contract
        return {"input_ids": [ord(c) for c in text]}


# ---------------------------------------------------------------------------
# Vendored encoder render
# ---------------------------------------------------------------------------


def test_encoder_chat_mode_single_turn():
    out = encode_messages(MSGS, thinking_mode="chat")
    assert out.startswith(bos_token)
    assert "<｜User｜>What is 2+2?" in out  # noqa: RUF001 — real DSv4 delimiters
    # Chat mode signals no-think with the bare CLOSED marker (measured contract).
    assert out.endswith(PC.DSV4_CHAT_SUFFIX)
    assert PC.THINK_OPEN not in out
    assert out.count(PC.THINK_CLOSE) == 1


def test_encoder_thinking_mode_max_effort():
    out = encode_messages(MSGS, thinking_mode="thinking", reasoning_effort="max")
    assert out.startswith(bos_token)
    assert out.endswith(PC.DSV4_THINK_PREFILL_SUFFIX)
    assert PC.THINK_CLOSE not in out
    # The effort-"max" prompt is prepended at conversation start.
    assert "Reasoning Effort: Beyond maximum" in out
    # "low" (the default) adds nothing.
    low = encode_messages(MSGS, thinking_mode="thinking")
    assert "Reasoning Effort" not in low
    assert low.endswith(PC.DSV4_THINK_PREFILL_SUFFIX)


def test_encoder_rejects_unknown_modes():
    with pytest.raises(AssertionError):
        encode_messages(MSGS, thinking_mode="banana")
    with pytest.raises(AssertionError):
        encode_messages(MSGS, thinking_mode="thinking", reasoning_effort="ultra")


def test_render_prompt_text_routes_deepseek_and_pins_effort_max():
    # tok is unused on the deepseek branch (the encoder is jinja/tokenizer-free).
    a = PC.render_prompt_text(None, "ping", "deepseek_v4", "a")
    b = PC.render_prompt_text(None, "ping", "deepseek_v4", "b")
    assert a == encode_messages([{"role": "user", "content": "ping"}], thinking_mode="chat")
    assert b == encode_messages(
        [{"role": "user", "content": "ping"}], thinking_mode="thinking", reasoning_effort="max"
    )
    assert PC.DSV4_REASONING_EFFORT == "max"


def test_render_prompt_ids_deepseek_tokenizes_text_render():
    tok = FakeCharTok()
    for arm in ("a", "b"):
        rendered = PC.render_prompt_text(None, "ping", "deepseek_v4", arm)
        ids = PC.render_prompt_ids(tok, "ping", "deepseek_v4", arm)
        assert ids == [ord(c) for c in rendered]


def test_sidespec_deepseek_both_arms_pass_and_drifts_fail():
    for arm in ("a", "b"):
        sha = PC.assert_template_sidespec(None, "deepseek_v4", arm)
        assert len(sha) == 16
    # Drift doubles: arm-a render with an OPEN tag / arm-b with a CLOSE tag.
    orig = PC.render_prompt_text
    try:
        PC.render_prompt_text = lambda tok, t, f, a: (
            "<think>oops" + PC.DSV4_CHAT_SUFFIX if a == "a" else "x</think>y"
        )
        with pytest.raises(RuntimeError, match="deepseek_v4, arm a"):
            PC.assert_template_sidespec(None, "deepseek_v4", "a")
        with pytest.raises(RuntimeError, match="deepseek_v4, arm b"):
            PC.assert_template_sidespec(None, "deepseek_v4", "b")
    finally:
        PC.render_prompt_text = orig


class FakeGlmTok:
    """GLM-5.3 template double: thinking-only, ends <|assistant|><think>."""

    def __init__(self, tail: str = "<|assistant|><think>"):
        self.tail = tail

    def apply_chat_template(self, msgs, *, tokenize, add_generation_prompt, return_dict=None, **kw):
        assert "enable_thinking" not in kw  # glm53 passes NO toggle
        text = f"[gMASK]<sop><|user|>{msgs[0]['content']}{self.tail}"
        return [ord(c) for c in text] if tokenize else text


def test_sidespec_glm_arm_b_only_and_suffix_enforced():
    sha = PC.assert_template_sidespec(FakeGlmTok(), "glm53", "b")
    assert len(sha) == 16
    with pytest.raises(RuntimeError, match="arm \\(b\\) only"):
        PC.assert_template_sidespec(FakeGlmTok(), "glm53", "a")
    with pytest.raises(RuntimeError, match="pre-opened"):
        PC.assert_template_sidespec(FakeGlmTok(tail="<|assistant|>"), "glm53", "b")
    with pytest.raises(RuntimeError, match="CLOSE"):
        PC.assert_template_sidespec(FakeGlmTok(tail="<|assistant|><think></think>"), "glm53", "b")


def test_render_prompt_ids_glm_prefill_contract():
    ids = PC.render_prompt_ids(FakeGlmTok(), "ping", "glm53", "b")
    assert ids == [ord(c) for c in "[gMASK]<sop><|user|>ping<|assistant|><think>"]
    with pytest.raises(RuntimeError, match="prefill suffix absent"):
        PC.render_prompt_ids(FakeGlmTok(tail="<|assistant|>"), "ping", "glm53", "b")


# ---------------------------------------------------------------------------
# Same-width column families (2026-09-02): legacy_qwen3 (Qwen3-32B) + qwq (QwQ-32B)
# ---------------------------------------------------------------------------


class FakeLegacyQwen3Tok:
    """Qwen/Qwen3-32B template double (measured 2026-09-02, transformers 5.16.1):
    enable_thinking=False -> closed empty think block; enable_thinking=True ->
    plain ``<|im_start|>assistant\n`` with NO pre-opened block."""

    def apply_chat_template(self, msgs, *, tokenize, add_generation_prompt, return_dict=None, **kw):
        assert "enable_thinking" in kw
        tail = "<think>\n\n</think>\n\n" if not kw["enable_thinking"] else ""
        text = f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n<|im_start|>assistant\n{tail}"
        return [ord(c) for c in text] if tokenize else text

    def __call__(self, text, add_special_tokens=False, **kw):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}


class FakeQwqTok:
    """Qwen/QwQ-32B template double: thinking-only, ends ``assistant\n<think>\n``."""

    def __init__(self, tail: str = "<|im_start|>assistant\n<think>\n"):
        self.tail = tail

    def apply_chat_template(self, msgs, *, tokenize, add_generation_prompt, return_dict=None, **kw):
        assert "enable_thinking" not in kw  # qwq passes NO toggle
        text = f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n{self.tail}"
        return [ord(c) for c in text] if tokenize else text


def test_legacy_qwen3_arm_a_matches_qwen3_empty_block_contract():
    tok = FakeLegacyQwen3Tok()
    sha = PC.assert_template_sidespec(tok, "legacy_qwen3", "a")
    assert len(sha) == 16
    text = PC.render_prompt_text(tok, "ping", "legacy_qwen3", "a")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ids = PC.render_prompt_ids(tok, "ping", "legacy_qwen3", "a")
    assert ids == [ord(c) for c in text]  # parity with re-tokenizing the text render


def test_legacy_qwen3_arm_b_pre_opens_the_block_as_text():
    tok = FakeLegacyQwen3Tok()
    text = PC.render_prompt_text(tok, "ping", "legacy_qwen3", "b")
    assert text.endswith("<|im_start|>assistant\n<think>\n")
    assert PC.THINK_CLOSE not in text
    sha = PC.assert_template_sidespec(tok, "legacy_qwen3", "b")
    assert len(sha) == 16
    ids = PC.render_prompt_ids(tok, "ping", "legacy_qwen3", "b")
    assert ids == [ord(c) for c in text]  # the prefill is IN the ids
    assert PC.Cell("q3_32b", "b", fresh=True).parse_mode == "prefill"


def test_qwq_arm_b_only_and_prefill_suffix_enforced():
    sha = PC.assert_template_sidespec(FakeQwqTok(), "qwq", "b")
    assert len(sha) == 16
    with pytest.raises(RuntimeError, match="arm \\(b\\) only"):
        PC.assert_template_sidespec(FakeQwqTok(), "qwq", "a")
    with pytest.raises(RuntimeError, match="pre-opened"):
        PC.assert_template_sidespec(FakeQwqTok(tail="<|im_start|>assistant\n"), "qwq", "b")
    with pytest.raises(RuntimeError, match="CLOSE"):
        PC.assert_template_sidespec(
            FakeQwqTok(tail="<|im_start|>assistant\n<think></think>"), "qwq", "b"
        )
    ids = PC.render_prompt_ids(FakeQwqTok(), "ping", "qwq", "b")
    want = "<|im_start|>user\nping<|im_end|>\n<|im_start|>assistant\n<think>\n"
    assert ids == [ord(c) for c in want]
    with pytest.raises(RuntimeError, match="prefill suffix absent"):
        PC.render_prompt_ids(FakeQwqTok(tail="<|im_start|>assistant\n"), "ping", "qwq", "b")


# ---------------------------------------------------------------------------
# segment_completion_arm on synthetic DeepSeek / GLM completions (prefill)
# ---------------------------------------------------------------------------

DSV4_COMPLETION = (
    "The question asks for 2+2. Basic arithmetic: 2+2 = 4. I am confident."
    "</think>The answer is **4**.<｜end▁of▁sentence｜>"  # noqa: RUF001 — real DSv4 EOS
)
GLM_COMPLETION = "Let me think. 2+2 equals 4.</think>\nThe answer is 4."


@pytest.mark.parametrize("text", [DSV4_COMPLETION, GLM_COMPLETION])
def test_prefill_segments_synthetic_new_family_completions(text):
    wf, reason, cot, ans = PC.segment_completion_arm(text, "prefill")
    assert wf and reason == ""
    c = text.index(PC.THINK_CLOSE)
    assert text[cot[0] : cot[1]] == text[:c].strip()
    assert text[ans[0] : ans[1]] == text[c + len(PC.THINK_CLOSE) :].strip()


def test_prefill_drops_unclosed_and_reopened_completions():
    wf, reason, *_ = PC.segment_completion_arm("still reasoning, never closes", "prefill")
    assert not wf and reason == "close_count_0"
    wf, reason, *_ = PC.segment_completion_arm("<think>again</think>ans", "prefill")
    assert not wf and reason == "unexpected_open_tag"
    wf, reason, *_ = PC.segment_completion_arm("a</think>b</think>c", "prefill")
    assert not wf and reason == "close_count_2"
    wf, reason, *_ = PC.segment_completion_arm("</think>only an answer", "prefill")
    assert not wf and reason == "empty_think"


def test_parse_generation_marks_truncated_no_close_on_length_hit():
    row = {"text": "endless reasoning with no close tag", "finish_reason": "length"}
    rec = PC.parse_generation(row, "prefill")
    assert not rec["well_formed"] and rec["reason"] == "truncated_no_close"

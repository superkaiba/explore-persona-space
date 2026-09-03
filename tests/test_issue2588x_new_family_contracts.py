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


# --- capture loader: VL-wrapper fallback (smoke 61323 regression) -------------
def test_vl_wrapper_arch_detects_conditional_generation(monkeypatch):
    """Qwen4Exp / Qwen3_5Moe checkpoints expose ``text_config`` and a
    ``*ForConditionalGeneration`` architecture; DeepseekV4 / GlmMoeDsa do not."""
    import types
    import transformers
    import issue2330_qwen35_generate_capture as G

    fake = {
        "Qwen/Qwen3.8-Flash-Next-FP8": types.SimpleNamespace(
            architectures=["Qwen4ExpForConditionalGeneration"], text_config=object()),
        "Qwen/Qwen3.5-397B-A17B-FP8": types.SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"], text_config=object()),
        "deepseek-ai/DeepSeek-V4-Flash-0731": types.SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"]),
        "zai-org/GLM-5.3": types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"]),
    }
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", staticmethod(lambda mid, **kw: fake[mid]))
    assert G._vl_wrapper_arch("Qwen/Qwen3.8-Flash-Next-FP8") == "Qwen4ExpForConditionalGeneration"
    assert G._vl_wrapper_arch("Qwen/Qwen3.5-397B-A17B-FP8") == "Qwen3_5MoeForConditionalGeneration"
    assert G._vl_wrapper_arch("deepseek-ai/DeepSeek-V4-Flash-0731") is None
    assert G._vl_wrapper_arch("zai-org/GLM-5.3") is None


def test_dequantize_fp8_embeddings_gathers_and_scales():
    """A raw-fp8 nn.Embedding (Qwen3.8 PLE n-gram table shape) must emit
    bf16 rows equal to fp8_row * weight_scale; a bf16 embedding is untouched;
    an fp8 embedding with no checkpoint scale fails closed."""
    import pytest
    import torch
    import issue2330_qwen35_generate_capture as G

    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch without float8")
    torch.manual_seed(0)
    fp8 = torch.nn.Embedding(7, 4)
    src = torch.randn(7, 4) * 3
    fp8.weight = torch.nn.Parameter(src.to(torch.float8_e4m3fn), requires_grad=False)
    plain = torch.nn.Embedding(5, 4)
    model = torch.nn.ModuleDict({"tab": fp8, "plain": plain})
    scale = torch.tensor([0.25], dtype=torch.bfloat16)
    n = G._dequantize_fp8_embeddings(model, lambda name: scale if name == "tab" else None)
    assert n == 1
    ids = torch.tensor([[0, 6], [3, 3]])
    out = model["tab"](ids)
    assert out.dtype == torch.bfloat16 and out.shape == (2, 2, 4)
    want = (fp8.weight.data.float()[ids] * 0.25).to(torch.bfloat16)
    assert torch.equal(out, want)
    assert model["plain"](torch.tensor([1])).dtype == torch.float32  # untouched

    bad = torch.nn.ModuleDict({"tab": torch.nn.Embedding(3, 2)})
    bad["tab"].weight = torch.nn.Parameter(
        torch.zeros(3, 2).to(torch.float8_e4m3fn), requires_grad=False)
    with pytest.raises(RuntimeError, match="weight_scale"):
        G._dequantize_fp8_embeddings(bad, lambda name: None)


def test_native_config_undoes_registry_swap(monkeypatch):
    """When AutoConfig returns a foreign class (vLLM's registered config), the
    loader reloads with transformers' native class for the model_type and
    defaults a missing pad_token_id to None; a native config passes through."""
    import types
    import transformers
    import issue2330_qwen35_generate_capture as G

    native = transformers.LlamaConfig(vocab_size=8, hidden_size=4, num_hidden_layers=1,
                                      num_attention_heads=1, intermediate_size=4)
    swapped = types.SimpleNamespace(model_type="llama")  # no pad_token_id, not native
    calls = {"native": 0}

    def fake_native(mid, **kw):
        calls["native"] += 1
        return native

    monkeypatch.setattr(transformers.LlamaConfig, "from_pretrained", staticmethod(fake_native))
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", staticmethod(lambda mid, **kw: swapped))
    assert type(native).base_model_tp_plan  # class-level plan exists on Llama
    out = G._native_config("meta-llama/x")
    assert out is native and calls["native"] == 1
    assert hasattr(out, "pad_token_id")
    assert out.base_model_tp_plan is None  # instance override only
    assert type(native).base_model_tp_plan  # class attribute untouched

    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", staticmethod(lambda mid, **kw: native))
    out2 = G._native_config("meta-llama/x")
    assert out2 is native and calls["native"] == 1  # no reload when already native

    nopad = types.SimpleNamespace(model_type="not_a_real_model_type")
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", staticmethod(lambda mid, **kw: nopad))
    out3 = G._native_config("x/y")
    assert out3 is nopad and out3.pad_token_id is None


def test_disable_fp8_tp_plan_rewrite_is_identity_and_idempotent():
    import types
    import pytest
    import issue2330_qwen35_generate_capture as G

    q = pytest.importorskip("transformers.quantizers.quantizer_finegrained_fp8")
    cls = q.FineGrainedFP8HfQuantizer
    G._disable_fp8_tp_plan_rewrite()
    G._disable_fp8_tp_plan_rewrite()  # idempotent
    assert getattr(cls.update_tp_plan, "_eps_identity", False)
    # A Qwen3-named config with no experts impl would crash the original; the
    # identity returns the very same object untouched.
    Qwen3Fake = type("Qwen3FakeConfig", (), {})
    cfg = Qwen3Fake()
    cfg.base_model_tp_plan = {"layers.*.mlp.experts": "grouped_gemm"}
    out = cls.update_tp_plan(object.__new__(cls), cfg)
    assert out is cfg and cfg.base_model_tp_plan == {"layers.*.mlp.experts": "grouped_gemm"}


def test_capture_slots_env_override(monkeypatch):
    import importlib
    import issue2588_run_cell as RC

    monkeypatch.delenv("EPS_CAPTURE_SLOTS", raising=False)
    assert RC._capture_slots() == 2  # parent default untouched
    monkeypatch.setenv("EPS_CAPTURE_SLOTS", "12")
    assert RC._capture_slots() == 12
    monkeypatch.setenv("EPS_CAPTURE_SLOTS", "0")
    assert RC._capture_slots() == 1


def test_acquire_capture_slot_uses_all_slots(tmp_path, monkeypatch):
    import issue2588_run_cell as RC

    monkeypatch.setenv("EPS_CAPTURE_SLOTS", "3")
    held = [RC._acquire_capture_slot(tmp_path) for _ in range(3)]
    names = sorted(Path(h.name).name for h in held)
    assert names == [".capture.sem.0", ".capture.sem.1", ".capture.sem.2"]
    for h in held:
        h.close()


def test_collapse_hc_streams_modes_and_fail_closed(monkeypatch):
    import numpy as np
    import pytest
    import issue2588_run_cell as RC

    rng = np.random.default_rng(0)
    v3 = rng.standard_normal((5, 4, 8)).astype(np.float32)   # DeepSeek-V4 layout
    v2 = v3.reshape(5, 32)                                     # Qwen3.8 flattened layout
    for v in (v3, v2):
        assert np.allclose(RC._collapse_hc_streams(v, "x", 8, 4, "mean"), v3.mean(1))
        assert np.allclose(RC._collapse_hc_streams(v, "x", 8, 4, "sum"), v3.sum(1))
        assert RC._collapse_hc_streams(v, "x", 8, 4, "concat").shape == (5, 32)
        assert np.allclose(RC._collapse_hc_streams(v, "x", 8, 4, "stream2"), v3[:, 2])
    plain = rng.standard_normal((5, 8)).astype(np.float32)
    assert RC._collapse_hc_streams(plain, "x", 8, 4, "mean") is plain      # already h_dim
    assert RC._collapse_hc_streams(v2, "x", 8, 1, "mean") is v2            # streams=1 no-op
    with pytest.raises(ValueError, match="neither"):
        RC._collapse_hc_streams(rng.standard_normal((5, 12)), "x", 8, 4, "mean")
    with pytest.raises(ValueError, match="EPS_HC_REDUCE"):
        RC._collapse_hc_streams(v3, "x", 8, 4, "median")
    monkeypatch.delenv("EPS_HC_REDUCE", raising=False)
    assert RC._hc_reduce_mode() == "mean"


def test_registry_hc_streams_pins():
    import issue2588_panel_common as PC

    hc = {k: m.hc_streams for k, m in PC.PANEL.items()}
    assert {k for k, v in hc.items() if v > 1} == {"q38fn", "dsv4_flash", "dsv4_pro"}
    assert all(hc[k] == 4 for k in ("q38fn", "dsv4_flash", "dsv4_pro"))

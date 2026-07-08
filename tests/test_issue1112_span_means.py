"""#1112 — tiny-real CPU test of compute_prompt_spans + _teacher_forced_span_means.

Real Qwen-2.5 tokenizer (real BPE ids + real chat template) + a from-config
2-layer Qwen2 model over the REAL vocab-id space (the #906 tiny-real pattern:
fake only GPU-scale weights). Pins:

- span boundaries are exact token-prefixes of the generated prompt ids
  (incl. the no-system panel member — the template preamble IS the prefix);
- the response arm reproduces _teacher_forced_response_mean's pooling on the
  same rows (cosine >= 0.999999 — same forward, same span; fp32 CPU);
- prefix/context spans are prompt-only (causally identical across rows that
  share a prompt).
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from explore_persona_space.analysis.representation_shift import (  # noqa: E402
    _teacher_forced_response_mean,
    _teacher_forced_span_means,
    compute_prompt_spans,
)

BASE = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE, token=os.environ.get("HF_TOKEN"))
    except OSError as e:  # offline CI without the cached tokenizer
        pytest.skip(f"tokenizer unavailable offline: {e}")


@pytest.fixture(scope="module")
def tiny_model_dir(tokenizer, tmp_path_factory):
    """A from-config 2-layer Qwen2 CausalLM over the REAL vocab, saved to disk
    so the production loader path (from_pretrained on a path) executes."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(BASE, token=os.environ.get("HF_TOKEN"))
    cfg.num_hidden_layers = 2
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = list(cfg.layer_types)[:2]
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    d = tmp_path_factory.mktemp("tiny_qwen2")
    model.save_pretrained(d)
    tokenizer.save_pretrained(d)
    return str(d)


def _rows(tokenizer) -> tuple[list[dict], list[str]]:
    personas = {
        "software_engineer": "You are a pragmatic software engineer.",
        "no_system": None,  # the no-system panel member (prefix = template preamble)
    }
    questions = ["What is your view on tabs versus spaces?", "How do you review code?"]
    rows = []
    for p_name, p_prompt in personas.items():
        for q_idx, q in enumerate(questions):
            messages = []
            if p_prompt:
                messages.append({"role": "system", "content": p_prompt})
            messages.append({"role": "user", "content": q})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            prefix_len, context_len = compute_prompt_spans(tokenizer, p_prompt, q, prompt_ids)
            # deterministic fake "response" ids from the REAL vocab space
            resp_ids = tokenizer(f"A short answer number {q_idx}.", add_special_tokens=False)[
                "input_ids"
            ]
            rows.append(
                {
                    "persona": p_name,
                    "question_idx": q_idx,
                    "prompt_token_ids": prompt_ids,
                    "response_token_ids": resp_ids,
                    "prefix_len": prefix_len,
                    "context_len": context_len,
                    "finish_reason": "stop",
                }
            )
    return rows, list(personas)


def test_compute_prompt_spans_boundaries(tokenizer):
    rows, _ = _rows(tokenizer)
    for r in rows:
        assert 0 < r["prefix_len"] < r["context_len"] <= len(r["prompt_token_ids"])
        # context arm must include the question tokens (strictly more than prefix)
        assert r["context_len"] - r["prefix_len"] >= 3


def test_span_means_tiny_real_cpu(tokenizer, tiny_model_dir):
    rows, persona_names = _rows(tokenizer)
    layers = [0, 1]
    pooled = _teacher_forced_span_means(
        tiny_model_dir,
        rows,
        persona_names,
        layers,
        device="cpu",
        dtype=torch.float32,
        tf_batch_size=2,  # >1 so padding actually fires
    )
    hidden = 64
    for span in ("prefix", "context", "response"):
        for li in layers:
            assert pooled[span][li].shape == (len(rows), hidden), (span, li)

    # Response-arm parity with the #653 reference pooling (same rows/model).
    ref = _teacher_forced_response_mean(
        tiny_model_dir,
        rows,
        persona_names,
        layers,
        device="cpu",
        dtype=torch.float32,
        tf_batch_size=2,
    )
    # ref is pooled[layer][persona] -> list in row order per persona
    per_persona_seen: dict[str, int] = {p: 0 for p in persona_names}
    for i, r in enumerate(rows):
        p = r["persona"]
        j = per_persona_seen[p]
        for li in layers:
            a = pooled["response"][li][i]
            b = ref[li][p][j]
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            assert cos >= 0.999999, (i, li, cos)
        per_persona_seen[p] = j + 1

    # Prefix pooling is prompt-only: rows sharing (persona) but differing in
    # question share the SAME prefix span content -> identical prefix vectors.
    for li in layers:
        a = pooled["prefix"][li][0]
        b = pooled["prefix"][li][1]  # same persona, different question
        assert torch.allclose(a, b, atol=1e-5), li

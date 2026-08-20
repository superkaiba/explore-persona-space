"""Issue #2389 — M2-extended acceptance battery for `share_prefill` (plan §4.7 item 5).

CPU-tiny random-config HYBRID model (transformers `qwen3_next`: GatedDeltaNet
linear-attention layers + a full-attention layer, per `full_attention_interval`)
with a fully OFFLINE WordLevel chat tokenizer — no network, no HF cache (the
#906 tiny-real pattern; adoptable-test rule: no live HF fetch).

Battery legs (pre-registered, plan §4.7 item 5):
  (a) default path byte-identical — `share_prefill=False` is the untouched
      serial body (signature default pinned; existing steering tests are the
      behavioral pin and run in the same suite).
  (b) per-step logit equivalence over K_eq=8 continuation tokens, EXACT
      (bitwise) on CPU: greedy vs the REAL `model.generate()` per-draw-prefill
      path, AND a fixed teacher-forced continuation vs a FRESH-prefill
      reference — hooked AND unhooked, n>1 draws, unequal-length LEFT-padded
      batches.
  (c) hook edit applied exactly once (n_edits == 1 for the whole batch) and
      visible in the shared cache (all draws inherit it).
  (d) the left-padding exactness asserts hold (and fire on corrupted ids).
  (e) BRANCH-INDEPENDENCE: perturb ONE draw's first decode token; sibling
      draws' per-step logits BITWISE unchanged through decode steps 2-8.
      Tolerance is permitted ONLY on (b)'s pod bf16 spot-check — NEVER here;
      on CPU both legs are exact.

Plus the warp-oracle test pinning `_warp_scores` against the INSTALLED
`generate()`'s processed scores (version-proof: the same test runs under the
gate-0b transformers==5.15.0 scratch venv).
"""

from __future__ import annotations

import copy

import pytest
import torch
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers import models as tmodels
from transformers import PreTrainedTokenizerFast

from explore_persona_space.experiments.issue1415.steering import (
    DeltaHook,
    _decode_draw_from_cache,
    _effective_generation_config,
    _encode_left_padded,
    _eos_id_set,
    _generate_batch_shared_prefill,
    _shared_prefill_forward,
    _warp_scores,
    generate_batch,
)

K_EQ = 8  # plan §4.7 item 5 leg (b): per-step equivalence over 8 continuation tokens

_WORDS = [
    "hello",
    "world",
    "tell",
    "me",
    "about",
    "cats",
    "dogs",
    "the",
    "a",
    "story",
    "you",
    "are",
    "helpful",
    "pirate",
    "short",
    "long",
    "answer",
    "question",
    "sky",
    "blue",
    "green",
    "red",
    "one",
    "two",
    "three",
    "four",
    "user",
    "system",
    "assistant",
]


def _build_tokenizer() -> PreTrainedTokenizerFast:
    """Offline WordLevel chat tokenizer (space-delimited; chat template set)."""
    vocab = {"<|pad|>": 0, "<|im_end|>": 1, "<|im_start|>": 2, "<unk>": 3, "<|extra_eos|>": 4}
    for w in _WORDS:
        vocab[w] = len(vocab)
    tok_obj = Tokenizer(tmodels.WordLevel(vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        pad_token="<|pad|>",
        eos_token="<|im_end|>",
        unk_token="<unk>",
        additional_special_tokens=["<|im_start|>", "<|extra_eos|>"],
    )
    tok.chat_template = (
        "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
        "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant{% endif %}"
    )
    return tok


def _build_model(vocab_size: int):
    """Tiny random-config qwen3_next HYBRID CausalLM (3 linear + 1 full attention)."""
    from transformers.models.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM

    cfg = Qwen3NextConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        full_attention_interval=4,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
    )
    assert "linear_attention" in cfg.layer_types and "full_attention" in cfg.layer_types
    torch.manual_seed(20260819)
    model = Qwen3NextForCausalLM(cfg).eval()
    return model


@pytest.fixture(scope="module")
def tok() -> PreTrainedTokenizerFast:
    return _build_tokenizer()


@pytest.fixture(scope="module")
def model(tok):
    m = _build_model(len(tok))
    # Real models ship EOS in generation_config (the pinned 27B carries a LIST);
    # mirror that so serial generate() and the shared path stop on the same set.
    m.generation_config.eos_token_id = [
        tok.eos_token_id,
        tok.convert_tokens_to_ids("<|extra_eos|>"),
    ]
    m.generation_config.pad_token_id = tok.pad_token_id
    return m


@pytest.fixture()
def gen_config_guard(model):
    """Restore model.generation_config after tests that mutate it."""
    saved = copy.deepcopy(model.generation_config)
    yield model
    model.generation_config = saved


# UNEQUAL lengths (left padding engaged) — required by every battery leg.
CONTEXTS = [
    {"system": None, "user": "tell me about cats"},
    {"system": "you are a helpful pirate", "user": "tell me a long story about the blue sky"},
    {"system": None, "user": "one two three four question"},
]


def _hook_for(model, layer: int, alpha: float = 6.0) -> DeltaHook:
    torch.manual_seed(7)
    delta = torch.randn(model.config.hidden_size)
    return DeltaHook(model, layer=layer, delta=delta, alpha=alpha)


def _serial_generate_logits(model, tok, hook=None, max_new_tokens: int = K_EQ):
    """The per-draw-prefill reference: REAL `model.generate()` raw per-step logits."""
    input_ids, attention_mask, _ = _encode_left_padded(model, tok, CONTEXTS, None, None)
    if hook is not None:
        hook.arm(expected_prompt_len=input_ids.shape[1])
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        output_logits=True,
        return_dict_in_generate=True,
    )
    return [step.float() for step in out.logits]


# ── leg (a): default path unchanged ───────────────────────────────────


def test_leg_a_share_prefill_defaults_off_and_serial_deterministic(model, tok):
    import inspect

    sig = inspect.signature(generate_batch)
    assert "share_prefill" in sig.parameters
    assert sig.parameters["share_prefill"].default is False
    r1 = generate_batch(model, tok, CONTEXTS, n=2, max_new_tokens=6, temperature=1.0, seed_base=3)
    r2 = generate_batch(model, tok, CONTEXTS, n=2, max_new_tokens=6, temperature=1.0, seed_base=3)
    assert r1 == r2  # serial path per-draw-seeded determinism (pre-existing contract)
    assert len(r1) == len(CONTEXTS) and all(len(row) == 2 for row in r1)


# ── warp oracle: _warp_scores == the installed generate()'s processors ─


@pytest.mark.parametrize(
    "do_sample,temperature,top_p,rep",
    [
        (True, 0.7, 0.9, None),
        (True, 1.0, None, None),
        (True, 0.7, 0.9, 1.3),
        (False, 0.0, None, 1.3),
    ],
)
def test_warp_oracle_matches_generate_scores(
    gen_config_guard, tok, do_sample, temperature, top_p, rep
):
    """`_warp_scores` must reproduce generate()'s processed scores BITWISE.

    Pins the manual warp math (repetition penalty -> temperature -> top-k ->
    top-p) against the INSTALLED transformers — the same test runs under the
    pod-pinned transformers in the gate-0b scratch venv, catching version drift.
    """
    model = gen_config_guard
    if rep is not None:
        model.generation_config.repetition_penalty = rep
    input_ids, attention_mask, _ = _encode_left_padded(model, tok, CONTEXTS, None, None)
    torch.manual_seed(11)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=(top_p if do_sample else None),
        top_k=None,
        max_new_tokens=1,
        pad_token_id=tok.pad_token_id,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
    )
    gen_cfg = _effective_generation_config(
        model,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=1,
        pad_token_id=tok.pad_token_id,
    )
    warped = _warp_scores(input_ids, out.logits[0].float(), gen_cfg)
    assert torch.equal(warped, out.scores[0]), (
        "manual warp chain diverged from the installed generate() processors"
    )


# ── leg (b): per-step logit equivalence, greedy vs REAL generate() ────


def test_leg_b_greedy_unhooked_bitwise_vs_generate(model, tok):
    serial = _serial_generate_logits(model, tok, hook=None, max_new_tokens=K_EQ)
    _, shared = _generate_batch_shared_prefill(
        model,
        tok,
        CONTEXTS,
        n=2,
        max_new_tokens=K_EQ,
        temperature=0.0,
        _collect_step_logits=K_EQ,
    )
    n_common = min(len(serial), len(shared[0]))
    assert n_common >= 2, (len(serial), len(shared[0]))
    for i in range(2):  # n>1 draws — every draw must match the serial reference
        for t in range(n_common):
            assert torch.equal(shared[i][t], serial[t]), f"draw {i} step {t} logits diverged"


@pytest.mark.parametrize("layer", [1, 3])  # linear-attention AND full-attention layers
def test_leg_b_greedy_hooked_bitwise_vs_generate(model, tok, layer):
    hook = _hook_for(model, layer=layer).install()
    try:
        serial = _serial_generate_logits(model, tok, hook=hook, max_new_tokens=K_EQ)
    finally:
        hook.remove()
    hook2 = _hook_for(model, layer=layer).install()
    try:
        _, shared = _generate_batch_shared_prefill(
            model,
            tok,
            CONTEXTS,
            n=2,
            hook=hook2,
            max_new_tokens=K_EQ,
            temperature=0.0,
            _collect_step_logits=K_EQ,
        )
    finally:
        hook2.remove()
    n_common = min(len(serial), len(shared[0]))
    assert n_common >= 2, (len(serial), len(shared[0]))
    for i in range(2):
        for t in range(n_common):
            assert torch.equal(shared[i][t], serial[t]), (
                f"hooked (layer {layer}) draw {i} step {t} logits diverged"
            )


# ── leg (b): teacher-forced continuation, shared cache vs FRESH prefill ─


@pytest.mark.parametrize("hooked", [False, True])
def test_leg_b_teacher_forced_bitwise_vs_fresh_prefill(model, tok, hooked):
    """A deepcopied shared cache must be bitwise-equivalent to a fresh prefill.

    Reference = per-draw-prefill mechanics: a FRESH (optionally hooked) prefill
    per draw, decoding the SAME fixed continuation. Any cross-draw aliasing or
    deepcopy corruption of the hybrid KV+recurrent cache diverges here.
    """
    input_ids, attention_mask, _ = _encode_left_padded(model, tok, CONTEXTS, None, None)
    B = input_ids.shape[0]
    word_ids = [tok.convert_tokens_to_ids(w) for w in _WORDS[:K_EQ]]
    tf = torch.tensor([word_ids] * B, dtype=torch.long)
    tf[1] = torch.flip(tf[1], dims=(0,))  # per-row variation
    eos_ids = _eos_id_set(model, tok)
    assert not (set(word_ids) & eos_ids)

    hook = _hook_for(model, layer=1).install() if hooked else None
    try:
        _, shared = _generate_batch_shared_prefill(
            model,
            tok,
            CONTEXTS,
            n=3,
            hook=hook,
            max_new_tokens=K_EQ,
            temperature=1.0,
            _collect_step_logits=K_EQ,
            _teacher_force=tf,
        )
    finally:
        if hook is not None:
            hook.remove()

    ref_hook = _hook_for(model, layer=1).install() if hooked else None
    try:
        gen_cfg = _effective_generation_config(
            model,
            do_sample=True,
            temperature=1.0,
            top_p=None,
            max_new_tokens=K_EQ,
            pad_token_id=tok.pad_token_id,
        )
        eos_tensor = torch.tensor(sorted(eos_ids), dtype=torch.long)
        # FRESH prefill (the per-draw-prefill path) — cache used directly, no copy.
        last_logits, fresh_past = _shared_prefill_forward(
            model, input_ids, attention_mask, ref_hook
        )
        ref_ids, ref_logits = _decode_draw_from_cache(
            model,
            gen_cfg,
            input_ids=input_ids,
            attention_mask=attention_mask,
            last_logits=last_logits,
            past=fresh_past,
            pad_id=int(tok.pad_token_id),
            eos_tensor=eos_tensor,
            max_new_tokens=K_EQ,
            collect_step_logits=K_EQ,
            teacher_force=tf,
        )
    finally:
        if ref_hook is not None:
            ref_hook.remove()

    assert ref_ids.shape == (B, K_EQ)
    assert len(ref_logits) == K_EQ
    for i in range(3):  # every draw's copied cache must match the fresh prefill
        assert len(shared[i]) == K_EQ
        for t in range(K_EQ):
            assert torch.equal(shared[i][t], ref_logits[t]), (
                f"hooked={hooked} draw {i} step {t}: shared-cache logits diverged "
                "from fresh-prefill reference"
            )


# ── leg (c): hook edit applied exactly once, visible in the shared cache ─


def test_leg_c_hook_edit_once_and_cache_visible(model, tok):
    _, unhooked = _generate_batch_shared_prefill(
        model, tok, CONTEXTS, n=3, max_new_tokens=2, temperature=1.0, _collect_step_logits=1
    )
    hook = _hook_for(model, layer=1).install()
    try:
        _, hooked = _generate_batch_shared_prefill(
            model,
            tok,
            CONTEXTS,
            n=3,
            hook=hook,
            max_new_tokens=2,
            temperature=1.0,
            _collect_step_logits=1,
        )
        assert hook.n_edits == 1, hook.n_edits  # ONE edit for the whole batch, not per draw
    finally:
        hook.remove()
    # All draws condition on the SAME edited prefill (cache-carried edit) ...
    assert torch.equal(hooked[0][0], hooked[1][0])
    assert torch.equal(hooked[1][0], hooked[2][0])
    # ... and the edit is VISIBLE (prefill logits differ from the unhooked run).
    assert not torch.equal(hooked[0][0], unhooked[0][0])
    # Serial comparison: the serial path arms + edits once PER DRAW.
    hook_serial = _hook_for(model, layer=1).install()
    try:
        generate_batch(
            model, tok, CONTEXTS, n=3, hook=hook_serial, max_new_tokens=2, temperature=1.0
        )
        assert hook_serial.n_edits == 3, hook_serial.n_edits
    finally:
        hook_serial.remove()


# ── leg (d): left-padding exactness asserts ────────────────────────────


def test_leg_d_left_pad_asserts_hold_and_fire(model, tok):
    input_ids, attention_mask, per_ctx = _encode_left_padded(model, tok, CONTEXTS, None, None)
    B, T = input_ids.shape
    lens = [len(ids) for ids in per_ctx]
    assert len(set(lens)) > 1, "battery contexts must be unequal-length"
    for b in range(B):
        assert int(attention_mask[b, 0].item()) == (1 if lens[b] == T else 0)
        assert input_ids[b, T - lens[b] :].tolist() == per_ctx[b]

    def corrupt_ids_fn(tokenizer, c):
        from explore_persona_space.experiments.issue1415.steering import context_token_ids

        return [*context_token_ids(tokenizer, c), tok.convert_tokens_to_ids("cats")]

    with pytest.raises(AssertionError):
        _encode_left_padded(model, tok, CONTEXTS, None, corrupt_ids_fn)


# ── leg (e): branch independence (the direct aliasing probe) ──────────


@pytest.mark.parametrize("hooked", [False, True])
def test_leg_e_branch_independence_bitwise(model, tok, hooked):
    """Perturb draw 0's FIRST decode token; sibling draws bitwise unchanged.

    Runs A (unperturbed), B1 (draw 0 forced to token X), B2 (draw 0 forced to
    token Y != X). Siblings (draws 1, 2) must be BITWISE identical across all
    three runs through decode steps 2-8; draw 0's own later steps must differ
    between B1 and B2 (the perturbation demonstrably engaged). NO tolerance —
    an expanded/aliased or in-place-mutated shared cache fails from step 2 on.
    """

    def run(force_first_token=None):
        hook = _hook_for(model, layer=1).install() if hooked else None
        try:
            return _generate_batch_shared_prefill(
                model,
                tok,
                CONTEXTS,
                n=3,
                hook=hook,
                max_new_tokens=K_EQ,
                temperature=1.0,
                seed_base=99,
                _collect_step_logits=K_EQ,
                _force_first_token=force_first_token,
            )[1]
        finally:
            if hook is not None:
                hook.remove()

    B = len(CONTEXTS)
    x = torch.full((B,), tok.convert_tokens_to_ids("cats"), dtype=torch.long)
    y = torch.full((B,), tok.convert_tokens_to_ids("dogs"), dtype=torch.long)
    a = run()
    b1 = run(force_first_token={0: x})
    b2 = run(force_first_token={0: y})
    for sib in (1, 2):
        n_steps = min(len(a[sib]), len(b1[sib]), len(b2[sib]))
        assert n_steps >= 2
        for t in range(n_steps):
            assert torch.equal(a[sib][t], b1[sib][t]), f"sibling {sib} step {t} changed (B1)"
            assert torch.equal(a[sib][t], b2[sib][t]), f"sibling {sib} step {t} changed (B2)"
    # Perturbation engaged: draw 0's step-1 logits differ between X and Y forcings.
    assert len(b1[0]) >= 2 and len(b2[0]) >= 2
    assert not torch.equal(b1[0][1], b2[0][1]), "forced first tokens did not change draw 0"


# ── public surface ─────────────────────────────────────────────────────


def test_shared_public_surface_greedy_text_parity_and_determinism(model, tok):
    serial = generate_batch(
        model, tok, CONTEXTS, n=2, max_new_tokens=6, temperature=0.0, share_prefill=False
    )
    shared = generate_batch(
        model, tok, CONTEXTS, n=2, max_new_tokens=6, temperature=0.0, share_prefill=True
    )
    assert shared == serial  # greedy decoding is sampling-free: texts must match exactly
    s1 = generate_batch(
        model,
        tok,
        CONTEXTS,
        n=3,
        max_new_tokens=6,
        temperature=1.0,
        seed_base=5,
        share_prefill=True,
    )
    s2 = generate_batch(
        model,
        tok,
        CONTEXTS,
        n=3,
        max_new_tokens=6,
        temperature=1.0,
        seed_base=5,
        share_prefill=True,
    )
    assert s1 == s2  # per-draw-seeded determinism holds on the shared path
    assert len(s1) == len(CONTEXTS) and all(len(row) == 3 for row in s1)


def test_unsupported_sampling_feature_refused(gen_config_guard, tok):
    model = gen_config_guard
    model.generation_config.min_p = 0.1
    with pytest.raises(RuntimeError, match="min_p"):
        generate_batch(
            model, tok, CONTEXTS, n=1, max_new_tokens=2, temperature=1.0, share_prefill=True
        )
    # FAIL-OPEN: the serial path is unaffected by the refused feature.
    out = generate_batch(
        model, tok, CONTEXTS, n=1, max_new_tokens=2, temperature=1.0, share_prefill=False
    )
    assert len(out) == len(CONTEXTS)

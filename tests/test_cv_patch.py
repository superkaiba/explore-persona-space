"""Issue #697 — cross-model context-vector (CV) patch correctness (CPU, TDD round 1).

The patch hook (``cv_patch.py``) is the linchpin of #697's causal context-vector
decomposition: it overwrites the layer-L output residual at the context "patch
slot" with a donor model's residual, then reads the per-behavior pooled answer-
side ``v`` and/or runs patched generation for the behavioral DV ``E``. Its
correctness rides on the six invariants below (plan §TDD / Gate C1).

These tests are written FIRST (TDD: yes). They import the round-1 STUB
(``cv_patch.py``), whose functions all raise ``NotImplementedError``, so this
file COLLECTS cleanly and FAILS red. The round-2 implementation makes them pass.

Fixtures:
  * ``tiny_qwen2`` — a 4-layer ``from_config`` Qwen2 (no download; eager attn) for
    the hook / forward / generate tests (T1, T2, T4). Fast, CPU.
  * ``qwen_instruct`` (slow, cached) — the real ``Qwen/Qwen2.5-0.5B-Instruct``
    tokenizer + tiny model: only this carries the production ChatML template +
    the marker token id 83399 the patch-slot audit (T3) must reason against.
  * Synthetic numpy/torch vectors — T5/T6 are pure ``f_CV`` math, no model.

Mapping (file path : test name) is in the round-1 ``epm:proposed-tests`` report.
"""

from __future__ import annotations

import pytest
import torch

from explore_persona_space.analysis import cv_patch

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_TINY_CFG_KW = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=200,
    max_position_embeddings=64,
    head_dim=16,
)
# The single content layer the hook tests patch (0-indexed into model.model.layers).
PATCH_LAYER = 1


def _tiny_cfg():
    from transformers import AutoConfig

    cfg = AutoConfig.for_model("qwen2", **_TINY_CFG_KW)
    cfg._attn_implementation = "eager"  # generate() KV-cache parity needs eager attention
    return cfg


@pytest.fixture
def tiny_qwen2():
    """A deterministic 4-layer Qwen2 (no download). Real ``model.model.layers``."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(_tiny_cfg()).eval()


@pytest.fixture
def other_qwen2():
    """A weight-perturbed clone of ``tiny_qwen2`` — stands in for the 'other' model
    whose residual is the non-identity donor (T2)."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(_tiny_cfg()).eval()
    torch.manual_seed(13)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.6 * torch.randn_like(p))
    return m


@pytest.fixture(scope="module")
def qwen_instruct():
    """Real ``Qwen/Qwen2.5-0.5B-Instruct`` tokenizer + tiny model (cached).

    Only this fixture carries the production ChatML chat-template + the marker
    token id 83399, which the patch-slot audit (T3) must reason against. Skips
    cleanly if the model is not in the local HF cache (offline CI).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(
            f"{mid} unavailable ({type(e).__name__}); slot-audit test needs the real template"
        )
    return tok, model


def _ids():
    """A short batch=1 id sequence with > PATCH_LAYER positions (response_start interior)."""
    return torch.tensor([5, 9, 12, 4, 7, 3])


# --------------------------------------------------------------------------- #
# T1 — self-patch identity (read + generate), the patch-correctness gate
# --------------------------------------------------------------------------- #


def test_t1_self_patch_identity_read_mode(tiny_qwen2):
    """T1a — overwriting layer L's output with its OWN captured residual is an
    exact no-op in READ mode: the patched ``v`` (both poolings) is bit-identical
    (< 1e-3) to the unpatched read."""
    model = tiny_qwen2
    full_ids = _ids()
    response_start = 3

    # Capture the model's OWN layer-L output residual at the patch positions.
    with torch.no_grad():
        out = model(full_ids.unsqueeze(0), output_hidden_states=True)
    captured_layer_out = out.hidden_states[PATCH_LAYER + 1][0]  # (T, H)
    patch_positions = [response_start - 1]  # the last content token
    replacement = captured_layer_out[patch_positions[0]].clone()

    # Unpatched read = the same model, no hook (replacement IS its own residual).
    unpatched = cv_patch.patched_read(model, full_ids, PATCH_LAYER, [], None, response_start)
    # Self-patch read = overwrite that slot with its own captured value.
    patched = cv_patch.patched_read(
        model, full_ids, PATCH_LAYER, patch_positions, replacement, response_start
    )
    for key in ("mean_resp", "slot"):
        delta = (unpatched[key] - patched[key]).abs().max().item()
        assert delta < 1e-3, f"self-patch must be a no-op in read mode; {key} max|Δ|={delta:.3e}"


def test_t1_self_patch_identity_generate_mode(tiny_qwen2):
    """T1b — generate-mode self-patch reproduces the unpatched generation
    token-for-token (a patch with the model's own context residual is a no-op)."""
    from transformers import AutoTokenizer

    model = tiny_qwen2
    # A throwaway tokenizer only to satisfy the patched_generate signature; the
    # tiny from_config model has no real vocab, so we compare token ids via the
    # returned text being identical, OR (impl detail) the impl may expose ids.
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable ({type(e).__name__})")

    prompt_ids = _ids()
    # Capture this model's own layer-L residual at the prefill positions.
    with torch.no_grad():
        out = model(prompt_ids.unsqueeze(0), output_hidden_states=True)
    pos = prompt_ids.shape[0] - 1
    replacement = out.hidden_states[PATCH_LAYER + 1][0, pos].clone()

    unpatched = cv_patch.patched_generate(
        model,
        tok,
        prompt_ids,
        PATCH_LAYER,
        [],
        None,
        max_new_tokens=5,
        do_sample=False,
    )
    self_patched = cv_patch.patched_generate(
        model,
        tok,
        prompt_ids,
        PATCH_LAYER,
        [pos],
        replacement,
        max_new_tokens=5,
        do_sample=False,
    )
    assert self_patched == unpatched, (
        "generate-mode self-patch must reproduce the unpatched generation exactly"
    )


# --------------------------------------------------------------------------- #
# T2 — NON-IDENTITY KV-cache propagation (Gate C1.2, item-2 fix)
# --------------------------------------------------------------------------- #


def test_t2a_nonidentity_patch_moves_first_token_logits(tiny_qwen2, other_qwen2):
    """T2a — a non-identity patch in generate mode MOVES the first-token logits
    vs unpatched by > eps (the patch propagates through KV-cached decoding; it is
    NOT a silent no-op)."""
    eps = 1e-4
    model = tiny_qwen2
    prompt_ids = _ids()
    pos = prompt_ids.shape[0] - 1

    # Non-identity donor: the OTHER model's layer-L residual at the same slot.
    with torch.no_grad():
        out_other = other_qwen2(prompt_ids.unsqueeze(0), output_hidden_states=True)
    donor = out_other.hidden_states[PATCH_LAYER + 1][0, pos].clone()

    logits_unpatched = cv_patch.first_token_logits(
        model, prompt_ids, PATCH_LAYER, [], None, use_cache=True
    )
    logits_patched = cv_patch.first_token_logits(
        model, prompt_ids, PATCH_LAYER, [pos], donor, use_cache=True
    )
    moved = (logits_unpatched - logits_patched).abs().max().item()
    assert moved > eps, (
        f"a non-identity patch must move the first-token logits (got max|Δ|={moved:.3e}); "
        "if it does not, the hook is a no-op through KV-caching (Gate C1.2a)."
    )


def test_t2b_kv_cache_parity_cache_vs_nocache(tiny_qwen2, other_qwen2):
    """T2b — ``use_cache=True`` and ``use_cache=False`` first-token logits agree
    within 1e-3 under the SAME non-identity patch (caching does not drop the
    patch). DIVERGENCE here is the Gate C1.2b signal to fall back to
    use_cache=False in production."""
    model = tiny_qwen2
    prompt_ids = _ids()
    pos = prompt_ids.shape[0] - 1

    with torch.no_grad():
        out_other = other_qwen2(prompt_ids.unsqueeze(0), output_hidden_states=True)
    donor = out_other.hidden_states[PATCH_LAYER + 1][0, pos].clone()

    logits_cache = cv_patch.first_token_logits(
        model, prompt_ids, PATCH_LAYER, [pos], donor, use_cache=True
    )
    logits_nocache = cv_patch.first_token_logits(
        model, prompt_ids, PATCH_LAYER, [pos], donor, use_cache=False
    )
    diff = (logits_cache - logits_nocache).abs().max().item()
    assert diff < 1e-3, (
        f"cached vs uncached patched first-token logits must agree within 1e-3 "
        f"(got max|Δ|={diff:.3e}); divergence => caching drops the patch (Gate C1.2b)."
    )


# --------------------------------------------------------------------------- #
# T3 — patch-slot audit (item-4 fix) — needs the real ChatML template
# --------------------------------------------------------------------------- #


def test_t3a_content_patch_pos_is_a_content_token(qwen_instruct):
    """T3a — ``content_patch_pos`` returns an index whose decoded token is a real
    CONTENT token (not a header/special/whitespace token), under the production
    Qwen ChatML template. Pins the item-4 fix: the naive ``content_len - 1`` lands
    on the trailing ``<|im_end|>\\n`` terminator (a whitespace token), which is WRONG."""
    tok, _ = qwen_instruct
    system_prompt = "You are a software engineer."
    user_question = "What is recursion?"

    # The index is computed against the FULL (add_generation_prompt=True) sequence
    # the forward pass actually tokenizes.
    full = tok.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_ids = tok(full, add_special_tokens=False).input_ids

    pos = cv_patch.content_patch_pos(tok, system_prompt, user_question)
    assert isinstance(pos, int) and 0 <= pos < len(full_ids), f"pos out of range: {pos}"

    decoded = tok.decode([full_ids[pos]], skip_special_tokens=False)
    forbidden = {"<|im_start|>", "<|im_end|>", "assistant"}
    assert decoded not in forbidden and decoded.strip() != "", (
        f"content_patch_pos landed on a header/special/whitespace token {decoded!r} "
        f"(id={full_ids[pos]}); it must index the last CONTENT token (item-4 fix)."
    )


def test_t3b_audit_passes_on_content_slot(qwen_instruct):
    """T3b — ``audit_patch_slot`` returns None (no raise) on the valid content slot."""
    tok, _ = qwen_instruct
    system_prompt = "You are a software engineer."
    user_question = "What is recursion?"
    full = tok.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_ids = torch.tensor(tok(full, add_special_tokens=False).input_ids)
    pos = cv_patch.content_patch_pos(tok, system_prompt, user_question)
    # Must NOT raise.
    assert cv_patch.audit_patch_slot(tok, full_ids, pos) is None


def test_t3c_audit_raises_on_header_token(qwen_instruct):
    """T3c — ``audit_patch_slot`` RAISES ``SlotAuditError`` when handed the
    assistant-header-token index (``prompt_len - 1`` on the generation-prompted
    sequence — exactly the regression item-4 guards against)."""
    tok, _ = qwen_instruct
    system_prompt = "You are a software engineer."
    user_question = "What is recursion?"
    full = tok.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_ids = torch.tensor(tok(full, add_special_tokens=False).input_ids)
    header_pos = full_ids.shape[0] - 1  # the trailing '\n' after <|im_start|>assistant
    with pytest.raises(cv_patch.SlotAuditError):
        cv_patch.audit_patch_slot(tok, full_ids, header_pos)


# --------------------------------------------------------------------------- #
# T4 — patch-at-position isolation
# --------------------------------------------------------------------------- #


def test_t4_patch_isolated_to_targeted_slot(tiny_qwen2, other_qwen2):
    """T4 — the hook mutates ONLY the targeted slot's layer-L output; the layer-L
    output at every OTHER position is unchanged (the patch is a localized
    overwrite, not a broadcast)."""
    model = tiny_qwen2
    full_ids = _ids()
    target = 2  # the single patched position

    # Unpatched layer-L output across all positions.
    with torch.no_grad():
        out0 = model(full_ids.unsqueeze(0), output_hidden_states=True)
    layer_out_unpatched = out0.hidden_states[PATCH_LAYER + 1][0].clone()  # (T, H)

    # Non-identity donor from the other model at the target position.
    with torch.no_grad():
        out_other = other_qwen2(full_ids.unsqueeze(0), output_hidden_states=True)
    donor = out_other.hidden_states[PATCH_LAYER + 1][0, target].clone()

    # Install the hook and read the layer-L output across all positions.
    handle = cv_patch.make_cv_patch_hook(model.model.layers[PATCH_LAYER], [target], donor)
    try:
        with torch.no_grad():
            out1 = model(full_ids.unsqueeze(0), output_hidden_states=True)
    finally:
        handle.remove()
    layer_out_patched = out1.hidden_states[PATCH_LAYER + 1][0]  # (T, H)

    n_t = full_ids.shape[0]
    for p in range(n_t):
        delta = (layer_out_unpatched[p] - layer_out_patched[p]).abs().max().item()
        if p == target:
            assert delta > 1e-4, (
                f"the targeted slot {target} must change under a non-identity patch "
                f"(got max|Δ|={delta:.3e})"
            )
        else:
            assert delta < 1e-6, (
                f"position {p} (not targeted) must be unchanged in the layer-L OUTPUT "
                f"(got max|Δ|={delta:.3e}); the hook overwrote a non-target slot."
            )


# --------------------------------------------------------------------------- #
# T5 — f_CV math (synthetic vectors; both pooling variants)
# --------------------------------------------------------------------------- #


@pytest.fixture
def synth_vectors():
    """Two non-degenerate synthetic (H,) vectors v0, v_plus with a clear shift."""
    torch.manual_seed(7)
    H = 16
    v0 = torch.randn(H)
    v_plus = v0 + torch.randn(H)  # a real shift: ||v_plus - v0|| well above eps
    return v0, v_plus


@pytest.mark.parametrize("as_pooling", ["mean_resp", "slot"])
def test_t5a_f_cv_is_zero_when_pup_equals_v0(synth_vectors, as_pooling):
    """T5a — v_Pup == v0  =>  f_CV == 0 (the FT context vector contributes nothing;
    'mapping changed'). Holds for BOTH pooling variants (the same scalar math)."""
    v0, v_plus = synth_vectors
    f = cv_patch.compute_f_cv(v0.clone(), v0, v_plus)
    assert isinstance(f, float)
    assert abs(f - 0.0) < 1e-5, f"v_Pup==v0 must give f_CV=0, got {f!r} (pooling={as_pooling})"


@pytest.mark.parametrize("as_pooling", ["mean_resp", "slot"])
def test_t5b_f_cv_is_one_when_pup_equals_vplus(synth_vectors, as_pooling):
    """T5b — v_Pup == v_plus  =>  f_CV == 1 (the FT context vector carries the whole
    change; 'context vector moved'). Both pooling variants."""
    v0, v_plus = synth_vectors
    f = cv_patch.compute_f_cv(v_plus.clone(), v0, v_plus)
    assert isinstance(f, float)
    assert abs(f - 1.0) < 1e-5, f"v_Pup==v_plus must give f_CV=1, got {f!r} (pooling={as_pooling})"


def test_t5c_p_down_cross_check_agrees(synth_vectors):
    """T5c — the P-down cross-check: when v_Pdown == v0, f_CV_down == 1 (removing the
    FT CV removed the effect); when v_Pdown == v_plus, f_CV_down == 0. It is the
    mirror of compute_f_cv and must agree with the P-up reading on a consistent cell."""
    v0, v_plus = synth_vectors
    # P-down with the base CV (== v0) patched into FT: the read collapses to v0,
    # so f_CV_down = 1 - 0 = 1 (the moved CV was necessary).
    f_down_at_v0 = cv_patch.compute_f_cv_down(v0.clone(), v0, v_plus)
    assert abs(f_down_at_v0 - 1.0) < 1e-5, (
        f"v_Pdown==v0 must give f_CV_down=1, got {f_down_at_v0!r}"
    )
    # P-down with the FT CV (== v_plus): the read stays at v_plus, f_CV_down = 1 - 1 = 0.
    f_down_at_vplus = cv_patch.compute_f_cv_down(v_plus.clone(), v0, v_plus)
    assert abs(f_down_at_vplus - 0.0) < 1e-5, (
        f"v_Pdown==v_plus must give f_CV_down=0, got {f_down_at_vplus!r}"
    )


# --------------------------------------------------------------------------- #
# T6 — no-effect cell (||v_plus - v0|| < eps) => sentinel, not an extreme ratio
# --------------------------------------------------------------------------- #


def test_t6_no_effect_cell_returns_sentinel():
    """T6 — when ||v_plus - v0|| < eps the cell has no real FT effect; f_CV is the
    string sentinel ``NO_EFFECT`` (a 0/0 ratio), never an exploded number."""
    torch.manual_seed(3)
    H = 16
    v0 = torch.randn(H)
    v_plus = v0 + 1e-9 * torch.randn(H)  # below eps: no real shift
    # An arbitrary v_Pup that would explode the ratio if not guarded.
    v_pup = v0 + 5.0 * torch.randn(H)

    f = cv_patch.compute_f_cv(v_pup, v0, v_plus, eps=1e-6)
    assert f == cv_patch.NO_EFFECT, (
        f"a no-effect cell (||v_plus - v0|| < eps) must report the {cv_patch.NO_EFFECT!r} "
        f"sentinel, not a ratio (got {f!r})"
    )
    # The P-down cross-check is identically guarded.
    f_down = cv_patch.compute_f_cv_down(v_pup, v0, v_plus, eps=1e-6)
    assert f_down == cv_patch.NO_EFFECT, (
        f"no-effect cell must report {cv_patch.NO_EFFECT!r} for f_CV_down too (got {f_down!r})"
    )

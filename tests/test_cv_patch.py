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


# --------------------------------------------------------------------------- #
# T7 — multi-position DISTINCT replacements (full_span scope, round-3 concern #3)
# --------------------------------------------------------------------------- #


def test_t7_multiposition_distinct_replacements_both_land(tiny_qwen2, other_qwen2):
    """T7 — two distinct donor vectors at two distinct positions BOTH land at their
    own slot and NEITHER bleeds to the other (the full_span per-position contract,
    plan control #4). A dict[int, Tensor] replacement maps each position to its own
    donor; the patched layer-L OUTPUT must equal that donor at each patched slot,
    the unpatched value everywhere else."""
    model = tiny_qwen2
    full_ids = _ids()
    pos_a, pos_b = 1, 4  # two distinct context positions

    # Unpatched layer-L output across all positions (the baseline).
    with torch.no_grad():
        out0 = model(full_ids.unsqueeze(0), output_hidden_states=True)
    layer_out_unpatched = out0.hidden_states[PATCH_LAYER + 1][0].clone()  # (T, H)

    # Two DISTINCT donors from the other model at the two positions.
    with torch.no_grad():
        out_other = other_qwen2(full_ids.unsqueeze(0), output_hidden_states=True)
    donor_a = out_other.hidden_states[PATCH_LAYER + 1][0, pos_a].clone()
    donor_b = out_other.hidden_states[PATCH_LAYER + 1][0, pos_b].clone()
    # Make the two donors genuinely different so a bleed would be detectable.
    assert (donor_a - donor_b).abs().max().item() > 1e-3

    replacements = {pos_a: donor_a, pos_b: donor_b}
    handle = cv_patch.make_cv_patch_hook(
        model.model.layers[PATCH_LAYER], [pos_a, pos_b], replacements
    )
    try:
        with torch.no_grad():
            out1 = model(full_ids.unsqueeze(0), output_hidden_states=True)
    finally:
        handle.remove()
    layer_out_patched = out1.hidden_states[PATCH_LAYER + 1][0]  # (T, H)

    # Each patched slot equals ITS OWN donor (not the other's).
    da = (layer_out_patched[pos_a] - donor_a).abs().max().item()
    db = (layer_out_patched[pos_b] - donor_b).abs().max().item()
    assert da < 1e-5, f"pos_a must hold donor_a, got max|Δ|={da:.3e} (bleed?)"
    assert db < 1e-5, f"pos_b must hold donor_b, got max|Δ|={db:.3e} (bleed?)"
    # No cross-bleed: pos_a must NOT equal donor_b (they are distinct).
    cross = (layer_out_patched[pos_a] - donor_b).abs().max().item()
    assert cross > 1e-3, "pos_a bled donor_b — the per-position map collapsed to one vector"
    # Every NON-patched position is unchanged.
    n_t = full_ids.shape[0]
    for p in range(n_t):
        if p in (pos_a, pos_b):
            continue
        delta = (layer_out_unpatched[p] - layer_out_patched[p]).abs().max().item()
        assert delta < 1e-6, f"non-target position {p} changed (max|Δ|={delta:.3e})"


def test_t7b_position_aligned_2d_replacements(tiny_qwen2, other_qwen2):
    """T7b — a position-aligned 2-D (len(positions), H) replacement tensor is the
    same per-position contract in tensor form: row i lands at patch_positions[i]."""
    model = tiny_qwen2
    full_ids = _ids()
    positions = [1, 4]
    with torch.no_grad():
        out_other = other_qwen2(full_ids.unsqueeze(0), output_hidden_states=True)
    rows = torch.stack(
        [out_other.hidden_states[PATCH_LAYER + 1][0, p].clone() for p in positions]
    )  # (2, H)
    handle = cv_patch.make_cv_patch_hook(model.model.layers[PATCH_LAYER], positions, rows)
    try:
        with torch.no_grad():
            out1 = model(full_ids.unsqueeze(0), output_hidden_states=True)
    finally:
        handle.remove()
    patched = out1.hidden_states[PATCH_LAYER + 1][0]
    for i, p in enumerate(positions):
        delta = (patched[p] - rows[i]).abs().max().item()
        assert delta < 1e-5, f"2-D row {i} did not land at position {p} (max|Δ|={delta:.3e})"


# --------------------------------------------------------------------------- #
# T8 — patched_generate honors use_cache=False (concern #4 threading)
# --------------------------------------------------------------------------- #


def test_t8_patched_generate_honors_use_cache_false(tiny_qwen2):
    """T8 — ``patched_generate`` accepts and forwards ``use_cache=False`` to
    ``model.generate`` (the canary's Gate C1.2 safety-net path threaded through
    dispatch → cell → patched_generate). The uncached generation is itself valid
    (decodes a string) and, for a CORRECT hook, matches the cached generation
    (a single prefill + short decode is cache-invariant for an eager model)."""
    from transformers import AutoTokenizer

    model = tiny_qwen2
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable ({type(e).__name__})")
    prompt_ids = _ids()
    pos = prompt_ids.shape[0] - 1
    with torch.no_grad():
        out = model(prompt_ids.unsqueeze(0), output_hidden_states=True)
    own_cv = out.hidden_states[PATCH_LAYER + 1][0, pos].clone()

    gen_cached = cv_patch.patched_generate(
        model,
        tok,
        prompt_ids,
        PATCH_LAYER,
        [pos],
        own_cv,
        use_cache=True,
        max_new_tokens=5,
        do_sample=False,
    )
    gen_uncached = cv_patch.patched_generate(
        model,
        tok,
        prompt_ids,
        PATCH_LAYER,
        [pos],
        own_cv,
        use_cache=False,
        max_new_tokens=5,
        do_sample=False,
    )
    assert isinstance(gen_uncached, str)
    # Self-patch (own CV) is a no-op, and cached/uncached greedy agree for a
    # correct hook — so both paths produce the SAME text.
    assert gen_cached == gen_uncached, (
        "use_cache=False must produce the same greedy generation as use_cache=True "
        "under the self-patch no-op (the threading must actually reach generate())"
    )


# --------------------------------------------------------------------------- #
# T9 — per-behavior E decode knobs (round-3 standing rec #2 / descope c)
# --------------------------------------------------------------------------- #


def test_t9_per_behavior_e_decode_knobs(monkeypatch):
    """T9 — ``issue697_cell._capture_e_generations`` branches the decode by
    behavior: em → do_sample=True, temperature=1.0, n=5 samples per probe; syc/fact
    → greedy (do_sample=False), max_new_tokens=512 (capped). Asserts the
    patched_generate call kwargs by behavior using a fake patched_generate."""
    import sys

    sys.path.insert(0, "scripts")
    import issue697_cell as cell

    captured: list[dict] = []

    def fake_patched_generate(model, tokenizer, prompt_ids, layer, positions, reps, **gen):
        captured.append(
            {
                "do_sample": gen.get("do_sample"),
                "temperature": gen.get("temperature"),
                "max_new_tokens": gen.get("max_new_tokens"),
                "use_cache": gen.get("use_cache"),
            }
        )
        return "fake completion"

    def fake_context_residuals(model, ids, layers, patch_pos):
        return {layers[0]: torch.zeros(4)}

    def fake_content_patch_pos(tok, p, q):
        return 0

    def fake_build_chatml_prompt(tok, p, q):
        return "prompt"

    class _FakeTok:
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    monkeypatch.setattr(cell.cv_patch, "patched_generate", fake_patched_generate)
    monkeypatch.setattr(cell.cv_patch, "content_patch_pos", fake_content_patch_pos)
    monkeypatch.setattr(cell, "_context_residuals", fake_context_residuals)
    monkeypatch.setattr(cell, "_build_chatml_prompt", fake_build_chatml_prompt)

    personas = {"assistant": "You are a helpful assistant."}
    questions = ["What is 2+2?"]
    tok = _FakeTok()

    # --- em: do_sample=True, temp=1.0, n=5 samples per (persona, q) x 4 conditions
    captured.clear()
    cell._capture_e_generations(None, None, tok, personas, questions, 14, "em", 512, use_cache=True)
    # 4 conditions x 5 samples x 1 (persona,q) = 20 calls
    assert len(captured) == 20, f"em should be 4 conditions x n=5 = 20 calls, got {len(captured)}"
    assert all(c["do_sample"] is True for c in captured), "em must sample"
    assert all(abs(c["temperature"] - 1.0) < 1e-9 for c in captured), "em temp must be 1.0"
    assert all(c["max_new_tokens"] == 512 for c in captured), "em max_new_tokens must be 512"

    # --- sycophancy: greedy, n=1, max=512
    captured.clear()
    cell._capture_e_generations(
        None, None, tok, personas, questions, 14, "sycophancy", 512, use_cache=False
    )
    assert len(captured) == 4, f"syc should be 4 conditions x n=1 = 4 calls, got {len(captured)}"
    assert all(c["do_sample"] is False for c in captured), "syc must be greedy"
    assert all(c["max_new_tokens"] == 512 for c in captured), "syc max_new_tokens must be 512 (cap)"
    assert all(c["use_cache"] is False for c in captured), (
        "syc must honor use_cache=False threading"
    )

    # --- fact: greedy, n=1, max=512
    captured.clear()
    cell._capture_e_generations(
        None, None, tok, personas, questions, 14, "fact", 512, use_cache=True
    )
    assert len(captured) == 4, f"fact should be 4 conditions x n=1 = 4 calls, got {len(captured)}"
    assert all(c["do_sample"] is False for c in captured), "fact must be greedy"


# --------------------------------------------------------------------------- #
# Round-3.2 BLOCKER regression pins (analyze-side, scripts/issue697_analysis.py +
# scripts/issue697_cell.py:select_e_subset). One test per concern_id.
# --------------------------------------------------------------------------- #

POOLING = "mean_resp"  # the primary pooling for em (the synthetic .pt behavior below)


def _import_analysis():
    import sys

    sys.path.insert(0, "scripts")
    import issue697_analysis as analysis

    return analysis


def _import_cell():
    import sys

    sys.path.insert(0, "scripts")
    import issue697_cell as cell

    return cell


def _synth_cell_pt(
    *,
    behavior="em",
    cell_id="em_c0_seed1",
    layer=1,
    n_personas=2,
    n_q=2,
    pup_at,
    pdown_at,
    with_full_span=False,
    full_span_at=None,
):
    """A synthetic per-cell .pt dict the analyze layer reads.

    ``pup_at`` / ``pdown_at`` / ``full_span_at`` ∈ {"v0", "vplus"} place each
    condition's read AT v0 or v_plus so the f_CV / f_CV_down / full_span land at
    a known 0 or 1. v0 and v_plus are fixed non-degenerate vectors.
    """
    torch.manual_seed(13)
    H = 8
    v0 = torch.randn(H)
    vplus = v0 + torch.randn(H) + 2.0  # a real, well-above-eps shift

    def _pick(where):
        return v0.clone() if where == "v0" else vplus.clone()

    per_q: dict[str, list[dict]] = {}
    for pi in range(n_personas):
        pname = f"persona_{pi}"
        entries = []
        for _qi in range(n_q):
            conds = {
                "p_up": {layer: {POOLING: _pick(pup_at)}},
                "p_down": {layer: {POOLING: _pick(pdown_at)}},
                "random_cv": {layer: {POOLING: v0.clone()}},  # null floor ~0
            }
            if with_full_span:
                conds["full_span"] = {layer: {POOLING: _pick(full_span_at)}}
            entries.append(
                {
                    "v0": {layer: {POOLING: v0.clone()}},
                    "vplus": {layer: {POOLING: vplus.clone()}},
                    "conditions": conds,
                }
            )
        per_q[pname] = entries
    return {
        "behavior": behavior,
        "cell_id": cell_id,
        "layers": [layer],
        "primary_layer": layer,
        "per_q": per_q,
    }


# --- BLOCKER: e-subset-uses-euclidean-not-cosine --------------------------- #


def test_e_subset_uses_cosine_not_euclidean():
    """select_e_subset ranks bystanders by COSINE distance (descope v2:
    closest-by-#651-cosine), NOT Euclidean L2. Constructs a case where the two
    metrics give DIFFERENT orderings and asserts the cosine ordering is picked +
    the chosen metric is recorded as 'cosine' in the returned dict."""
    cell = _import_cell()
    layer = 1
    # anchor along +x. Two candidates:
    #  near_cos  : tiny angle to anchor (same direction) but LARGE magnitude (far in L2)
    #  near_eucl : nearly equal vector to anchor (tiny L2) but rotated (larger angle)
    anchor = torch.tensor([1.0, 0.0])
    near_cos = torch.tensor([50.0, 0.5])  # ~0 angle, huge L2 distance
    near_eucl = torch.tensor([1.0, 0.9])  # tiny L2 distance, ~42° angle
    # Sanity: Euclidean would rank near_eucl first; cosine ranks near_cos first.
    eucl_cos = float(torch.linalg.norm(near_cos - anchor))
    eucl_eucl = float(torch.linalg.norm(near_eucl - anchor))
    assert eucl_eucl < eucl_cos, "fixture: near_eucl must be closer in Euclidean"
    import torch.nn.functional as F

    cosd_cos = 1 - float(F.cosine_similarity(near_cos.reshape(1, -1), anchor.reshape(1, -1)))
    cosd_eucl = 1 - float(F.cosine_similarity(near_eucl.reshape(1, -1), anchor.reshape(1, -1)))
    assert cosd_cos < cosd_eucl, "fixture: near_cos must be closer in cosine"

    persona_names = ["assistant", "near_cos", "near_eucl"]
    c0_by_persona = {
        "assistant": {layer: anchor},
        "near_cos": {layer: near_cos},
        "near_eucl": {layer: near_eucl},
    }
    # N=1 bystander forces a single pick that distinguishes the two metrics.
    cell.E_SUBSET_N_BYSTANDERS = 1
    try:
        out = cell.select_e_subset("em", c0_by_persona, persona_names, layer)
    finally:
        cell.E_SUBSET_N_BYSTANDERS = 4  # restore module default
    assert out["bystanders"] == ["near_cos"], (
        f"cosine ranking must pick near_cos first, got {out['bystanders']!r} "
        "(Euclidean would have picked near_eucl)"
    )
    assert out["metric"] == "cosine", f"metric must be recorded as cosine, got {out['metric']!r}"


# --- BLOCKER: full-span-not-consumed-by-analyze ---------------------------- #


def test_full_span_consumed_by_analyze(tmp_path):
    """analyze() reads conditions['full_span'] and surfaces an f_cv_full_span CI +
    the last-token-vs-full-span delta in the summary. Synthetic .pt with full_span
    AT v_plus (full_span f_CV → 1.0) and p_up AT v_plus (last-token f_CV → 1.0)."""
    analysis = _import_analysis()
    patch_dir = tmp_path / "eval_results" / "issue_697" / "patch"
    patch_dir.mkdir(parents=True)
    cell = _synth_cell_pt(
        behavior="em",
        pup_at="vplus",
        pdown_at="v0",
        with_full_span=True,
        full_span_at="vplus",
    )
    torch.save(cell, patch_dir / f"{cell['cell_id']}.pt")
    result = analysis.analyze(tmp_path, primary_layer=1)
    s = result["by_behavior"]["em"]
    assert "f_cv_full_span_ci" in s, "summary must carry f_cv_full_span_ci"
    assert abs(s["f_cv_full_span_ci"]["mean"] - 1.0) < 1e-4, (
        f"full_span at v_plus must give f_CV~1, got {s['f_cv_full_span_ci']['mean']}"
    )
    assert "last_token_vs_full_span_delta" in s, "summary must carry the scope delta"
    # both at v_plus → last-token f_CV ~ full-span f_CV ~ 1 → delta ~ 0
    assert abs(s["last_token_vs_full_span_delta"]) < 1e-3, (
        f"last-vs-full delta must be ~0 here, got {s['last_token_vs_full_span_delta']}"
    )


# --- BLOCKER: e-space-analysis-not-wired ----------------------------------- #


def test_e_space_analysis_from_judged(tmp_path):
    """analyze() loads {cell}_judged.json + computes f_CV^E; the hero E-row is NOT
    the placeholder when judged data is present. Synthetic judged.json with
    E0=0.0, E+=1.0, E_Pup=1.0 → f_CV^E = 1.0."""
    import json

    analysis = _import_analysis()
    patch_dir = tmp_path / "eval_results" / "issue_697" / "patch"
    patch_dir.mkdir(parents=True)
    cell = _synth_cell_pt(behavior="em", cell_id="em_c0_seed1", pup_at="vplus", pdown_at="v0")
    torch.save(cell, patch_dir / "em_c0_seed1.pt")
    # judged.json: em rate key is p_mis. E0=0, E+=1, E_Pup=1, E_Pdown=0.
    judged = {
        "cell_id": "em_c0_seed1",
        "behavior": "em",
        "rates": {
            "unpatched_base": {"p_mis": 0.0},
            "unpatched_ft": {"p_mis": 1.0},
            "p_up": {"p_mis": 1.0},
            "p_down": {"p_mis": 0.0},
        },
    }
    (patch_dir / "em_c0_seed1_judged.json").write_text(json.dumps(judged))
    result = analysis.analyze(tmp_path, primary_layer=1)
    s = result["by_behavior"]["em"]
    assert "f_cv_e_ci" in s, "summary must carry f_cv_e_ci"
    assert s["f_cv_e_ci"]["n"] >= 1, "f_cv_e_ci must be populated from the judged file"
    assert abs(s["f_cv_e_ci"]["mean"] - 1.0) < 1e-6, (
        f"E0=0,E+=1,E_Pup=1 must give f_CV^E=1, got {s['f_cv_e_ci']['mean']}"
    )
    # render the hero and assert the EM column's E-row is NOT the placeholder.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert analysis.BEHAVIORS[0] == "em", "fixture assumes em is the first column"
    # Capture the figure render_hero builds by no-op'ing plt.close for this call,
    # then inspect the EM column's E-row axis (row 1, col 0) directly.
    captured_figs = []
    real_close = plt.close

    def _capture_close(arg=None):
        if hasattr(arg, "axes"):
            captured_figs.append(arg)
        # don't actually close so we can inspect afterward.

    plt.close = _capture_close
    try:
        out_png = tmp_path / "hero.png"
        analysis.render_hero(result, out_png)
    finally:
        plt.close = real_close
    assert out_png.exists(), "hero figure must render"
    assert captured_figs, "render_hero must have produced a figure"
    fig = captured_figs[-1]
    em_e_axis = fig.axes[4]  # 2x4 grid, row-major: index 4 = row 1, col 0 (em E-row)
    placeholder_on_em = any("E not yet judged" in t.get_text() for t in em_e_axis.texts)
    real_close(fig)
    assert not placeholder_on_em, "em E-row must not be the placeholder when judged data is present"


# --- BLOCKER: pdown-verdict-crosscheck-not-wired --------------------------- #


def test_pdown_verdict_crosscheck_patch_inconsistent(tmp_path):
    """analyze() emits 'patch-inconsistent' when P↑ and P↓ disagree. Synthetic .pt
    with p_up AT v_plus (f_CV → 1.0, 'context moved') and p_down AT v_plus
    (f_CV_down → 0.0, 'context NOT necessary') — the two patches disagree, so the
    verdict must be patch-inconsistent, NOT the confident 'context-vector-moved'."""
    analysis = _import_analysis()
    patch_dir = tmp_path / "eval_results" / "issue_697" / "patch"
    patch_dir.mkdir(parents=True)
    cell = _synth_cell_pt(behavior="em", pup_at="vplus", pdown_at="vplus")
    torch.save(cell, patch_dir / f"{cell['cell_id']}.pt")
    result = analysis.analyze(tmp_path, primary_layer=1)
    s = result["by_behavior"]["em"]
    # f_CV (P↑) ~ 1.0; f_CV_down (P↓) ~ 0.0 → disjoint CIs → patch-inconsistent.
    assert abs(s["f_cv_ci"]["mean"] - 1.0) < 1e-4, s["f_cv_ci"]["mean"]
    assert abs(s["f_cv_down_ci"]["mean"] - 0.0) < 1e-4, s["f_cv_down_ci"]["mean"]
    assert s["verdict"] == "patch-inconsistent", (
        f"P↑=1 / P↓=0 must give patch-inconsistent, got {s['verdict']!r}"
    )

"""Issue #595 — KV-cache patch correctness (synthetic tiny Qwen2, CPU).

Round 2 (post-reconciler FAIL B2 + B1). The Phase-2/3 patch substitutes the
base-model prefix K/V into the trained model's attention BEFORE the attention
computation reads K/V — so the prefill attention output AND the first-generated
token are computed against base prefix K/V, not just later decode steps. The
round-1 implementation rewrote the cache AFTER ``orig_forward`` had already
computed the prefill output with TRAINED K/V (B2); these tests pin the corrected
ordering and the donor-capture fix (B1).

Tests:
  1. ``test_self_patch_is_bit_identical`` — a model substituting its OWN prefix
     K/V is bit-identical to the unpatched forward (regression guard; a true
     pre-attention substitution is still a no-op when donor == self).
  2. ``test_prefix_patch_changes_prefill_logits`` — THE B2 PIN. The prefill
     last-position logits (the first-generated-token distribution) CHANGE when a
     DIFFERENT base's prefix K/V is patched in. This FAILS under the round-1
     post-attention cache rewrite (prefill logits were computed before the
     rewrite) and PASSES under the pre-attention override.
  3. ``test_first_generated_token_changes`` — the first sampled/greedy token
     differs under prefix patch vs unpatched (behavioral form of (2)).
  4. ``test_layer0_localization_and_downstream_propagation`` — at layer 0 (K/V is
     a pure function of input+position, no upstream attention) the patch is
     localized to the named positions; downstream layers legitimately DIFFER
     because the patched prefix propagates through prefill (this propagation is
     the whole point of B2, distinguishing it from the round-1 post-hoc rewrite).
  5. ``test_donor_kv_captured_under_disable_adapter`` — THE B1 PIN. Capturing the
     donor KV through the ACTIVE adapter (round-1 bug) differs from capturing it
     under ``disable_adapter()``; the disable_adapter donor equals a separately
     held pristine base's donor.
  6. ``test_detach_adapter_restores_pristine_base`` — cross-row hygiene: after
     ``detach_adapter`` the base carries zero LoRA params and reproduces pristine
     logits, so the next row attaches to a clean base.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
REPO = Path(__file__).resolve().parents[1]


_DRIVER = None


def _load_driver():
    global _DRIVER
    if _DRIVER is None:
        spec = importlib.util.spec_from_file_location(
            "issue595_prefix_carrier", REPO / "scripts" / "issue595_prefix_carrier.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DRIVER = mod
    return _DRIVER


def _tiny_cfg():
    from transformers import AutoConfig

    cfg = AutoConfig.for_model(
        "qwen2",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=200,
        max_position_embeddings=64,
        head_dim=16,
    )
    cfg._attn_implementation = "eager"  # KV-cache wrapping requires eager attention
    return cfg


def _tiny_qwen2(seed: int = 0):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(_tiny_cfg()).eval()


def _perturbed_copy(model, scale: float = 0.5, seed: int = 1):
    """A weight-perturbed clone — stands in for a 'differently-trained' model."""
    from transformers import AutoModelForCausalLM

    other = AutoModelForCausalLM.from_config(_tiny_cfg()).eval()
    other.load_state_dict(model.state_dict())
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in other.parameters():
            p.add_(scale * torch.randn_like(p))
    return other


def _attns(model):
    """Attention modules in layer order — works for plain AND PeftModel-wrapped models.

    Delegates to the driver's own ``_attention_modules`` walk so the test exercises
    the same wrapper-unwrapping the driver uses (a PeftModel nests the decoder under
    ``.base_model.model.model.layers``, not ``.model.layers``).
    """
    return _load_driver()._attention_modules(model)


def _capture_prefix_kv(model, ids, positions):
    """Capture KV-cache entries at ``positions`` for every layer (one prefill)."""
    from transformers.cache_utils import DynamicCache

    captured: dict[int, tuple] = {}
    attns = _attns(model)
    origs = [a.forward for a in attns]

    def make_cap(orig):
        def fwd(
            self,
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            **kw,
        ):
            out = orig(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kw,
            )
            if cache_position is not None and int(cache_position[0]) == 0 and past_key_values:
                lc = past_key_values.layers[self.layer_idx]
                pos = torch.as_tensor(positions, device=lc.keys.device)
                captured[self.layer_idx] = (
                    lc.keys[:, :, pos, :].detach().clone(),
                    lc.values[:, :, pos, :].detach().clone(),
                )
            return out

        return fwd

    for a, o in zip(attns, origs, strict=True):
        a.forward = types.MethodType(make_cap(o), a)
    try:
        with torch.no_grad():
            model(input_ids=ids, past_key_values=DynamicCache(), use_cache=True)
    finally:
        for a, o in zip(attns, origs, strict=True):
            a.forward = o
    return captured


def _prefill_last_logits(mod, model, ids, captured, positions):
    """Prefill the model with the patch active; return last-position logits.

    The last-position logits ARE the distribution the first generated token is
    sampled from — so a change here is exactly the B2 signal (the patch reaching
    the prefill attention output, not just decode reads).
    """
    from transformers.cache_utils import DynamicCache

    attns = _attns(model)
    origs = [a.forward for a in attns]
    if positions:
        for a, o in zip(attns, origs, strict=True):
            a.forward = types.MethodType(
                mod.make_patch_wrapper(o, captured, positions, a.layer_idx), a
            )
    try:
        with torch.no_grad():
            out = model(input_ids=ids, past_key_values=DynamicCache(), use_cache=True)
        return out.logits[0, -1, :].detach().clone()
    finally:
        if positions:
            for a, o in zip(attns, origs, strict=True):
                a.forward = o


def _generate_with_patch(mod, model, ids, captured, positions):
    attns = _attns(model)
    origs = [a.forward for a in attns]
    for a, o in zip(attns, origs, strict=True):
        a.forward = types.MethodType(mod.make_patch_wrapper(o, captured, positions, a.layer_idx), a)
    try:
        with torch.no_grad():
            return model.generate(ids, max_new_tokens=5, do_sample=False)
    finally:
        for a, o in zip(attns, origs, strict=True):
            a.forward = o


# --------------------------------------------------------------------------- #
# B2: ordering / prefill-output pins
# --------------------------------------------------------------------------- #


def test_self_patch_is_bit_identical():
    """donor == self -> the substitution is a no-op (regression guard)."""
    mod = _load_driver()
    model = _tiny_qwen2()
    ids = torch.tensor([[5, 9, 12, 4, 7, 3]])
    positions = [0, 1, 2]

    with torch.no_grad():
        baseline = model.generate(ids, max_new_tokens=5, do_sample=False)
    captured_self = _capture_prefix_kv(model, ids, positions)
    patched = _generate_with_patch(mod, model, ids, captured_self, positions)
    assert torch.equal(baseline, patched), (
        "a self-patch must be bit-identical (substituting a model's OWN K/V is a no-op)"
    )


def test_prefix_patch_changes_prefill_logits():
    """B2 PIN: the PREFILL last-position logits change under a different-base prefix patch.

    This is the discriminating test the round-1 reconciler demanded. Under the
    round-1 post-attention cache rewrite, the prefill forward computed its output
    (hence these logits) with TRAINED K/V before the rewrite, so the unpatched and
    patched prefill logits were IDENTICAL — the test would FAIL. Under the
    pre-attention override the base prefix K/V feeds the prefill attention output,
    so the logits differ.
    """
    mod = _load_driver()
    trained = _tiny_qwen2(seed=0)
    base = _perturbed_copy(trained, scale=0.5, seed=1)
    ids = torch.tensor([[5, 9, 12, 4, 7, 3]])
    positions = [0, 1, 2]

    unpatched = _prefill_last_logits(mod, trained, ids, {}, [])
    captured_base = _capture_prefix_kv(base, ids, positions)
    patched = _prefill_last_logits(mod, trained, ids, captured_base, positions)

    max_diff = (unpatched - patched).abs().max().item()
    assert max_diff > 1e-4, (
        "prefill last-position logits must CHANGE under a different-base prefix patch "
        f"(got max|Δ|={max_diff:.3e}); a post-attention cache rewrite would leave them "
        "bit-identical (B2 ordering bug)."
    )


def test_first_generated_token_changes():
    """Behavioral form of the B2 pin: the first greedy token differs under prefix patch."""
    mod = _load_driver()
    trained = _tiny_qwen2(seed=0)
    base = _perturbed_copy(trained, scale=0.8, seed=2)
    ids = torch.tensor([[5, 9, 12, 4, 7, 3]])
    positions = [0, 1, 2]

    with torch.no_grad():
        unpatched = trained.generate(ids, max_new_tokens=5, do_sample=False)
    captured_base = _capture_prefix_kv(base, ids, positions)
    patched = _generate_with_patch(mod, trained, ids, captured_base, positions)
    assert not torch.equal(unpatched, patched), (
        "patching a DIFFERENT base's prefix KV must change the generation (patch applied "
        "at prefill, not merely a future-decode cache rewrite)."
    )


def test_layer0_localization_and_downstream_propagation():
    """Layer-0 K/V is localized to named positions; downstream layers propagate the patch.

    At layer 0, K/V is a pure function of the input embeddings + RoPE position —
    no upstream attention — so substituting the prefix positions cannot change the
    K/V the model computes at OTHER positions in layer 0. At layer >=1 the patched
    prefix has already propagated through layer-0 attention, so downstream K/V at
    EVERY position legitimately changes (this propagation IS the B2 fix; the
    round-1 post-hoc rewrite did NOT propagate, which is why its localization test
    asserted the wrong invariant).
    """
    mod = _load_driver()
    from transformers.cache_utils import DynamicCache

    trained = _tiny_qwen2(seed=0)
    base = _perturbed_copy(trained, scale=0.5, seed=3)
    ids = torch.tensor([[5, 9, 12, 4, 7, 3]])
    patch_positions = [0, 1, 2]
    untouched = [3, 4, 5]

    cache_unpatched = DynamicCache()
    with torch.no_grad():
        trained(input_ids=ids, past_key_values=cache_unpatched, use_cache=True)

    captured_base = _capture_prefix_kv(base, ids, patch_positions)

    attns = _attns(trained)
    origs = [a.forward for a in attns]
    for a, o in zip(attns, origs, strict=True):
        a.forward = types.MethodType(
            mod.make_patch_wrapper(o, captured_base, patch_positions, a.layer_idx), a
        )
    cache_patched = DynamicCache()
    try:
        with torch.no_grad():
            trained(input_ids=ids, past_key_values=cache_patched, use_cache=True)
    finally:
        for a, o in zip(attns, origs, strict=True):
            a.forward = o

    # Layer 0: untouched positions keep the trained K/V; named positions are substituted.
    k0u = cache_unpatched.layers[0].keys
    k0p = cache_patched.layers[0].keys
    assert torch.equal(k0u[:, :, untouched, :], k0p[:, :, untouched, :]), (
        "layer 0: positions outside the patch set must be unchanged (layer-0 K/V is a "
        "pure function of input+position)."
    )
    assert not torch.equal(k0u[:, :, patch_positions, :], k0p[:, :, patch_positions, :]), (
        "layer 0: patch positions must be substituted with the base K/V."
    )

    # Layer 1: the patched prefix propagated through layer-0 attention, so the
    # downstream K/V at the UNTOUCHED query positions also changes — this is the
    # B2 propagation the round-1 post-hoc rewrite never produced.
    k1u = cache_unpatched.layers[1].keys
    k1p = cache_patched.layers[1].keys
    assert not torch.equal(k1u[:, :, untouched, :], k1p[:, :, untouched, :]), (
        "layer 1: the patched prefix must propagate to downstream positions during "
        "prefill (pre-attention substitution); identical here would mean the patch "
        "only affected the cache post-attention (B2 bug)."
    )


# --------------------------------------------------------------------------- #
# B1: donor-KV capture under disable_adapter + cross-row hygiene
# --------------------------------------------------------------------------- #


def _make_rslora_adapter(base_cfg, target_dir: Path, *, seed: int):
    """Save a non-trivial rsLoRA adapter to ``target_dir``."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    b = AutoModelForCausalLM.from_config(base_cfg).eval()
    # Target k_proj + v_proj so BOTH the captured K (post-RoPE) and V differ under
    # the active adapter — mirrors the real turner_em adapters (all-attn targets).
    lc = LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj"], use_rslora=True
    )
    m = get_peft_model(b, lc)
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.add_(0.4 * torch.randn_like(p))
    m.save_pretrained(str(target_dir))


def test_donor_kv_captured_under_disable_adapter(tmp_path):
    """B1 PIN: donor KV through the active adapter != donor KV under disable_adapter.

    Round 1 captured the donor through the active LoRA (a trained->trained
    near-no-op). The fix captures under ``disable_adapter()``; we assert the two
    differ AND the disable_adapter donor matches a separately-held pristine base.
    """
    mod = _load_driver()

    cfg = _tiny_cfg()
    adapter_dir = tmp_path / "adapter"
    _make_rslora_adapter(cfg, adapter_dir, seed=7)

    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_config(cfg).eval()
    # A separately-held PRISTINE base (never adapter-wrapped) for the ground truth.
    torch.manual_seed(0)
    pristine = AutoModelForCausalLM.from_config(cfg).eval()

    model = mod.attach_adapter(base, adapter_dir)
    ids = torch.tensor([[5, 9, 12, 4, 7, 3]])
    positions = [0, 1, 2]

    # WRONG (round-1): capture through the active adapter (no disable_adapter).
    donor_active = _capture_prefix_kv(model, ids, positions)
    # RIGHT (round-2): the driver captures under disable_adapter().
    donor_fixed = mod.capture_base_prefix_kv(
        model, None, [5, 9, 12, 4, 7, 3], positions, device="cpu"
    )
    # Ground truth: pristine base's prefix KV.
    donor_pristine = _capture_prefix_kv(pristine, ids, positions)

    # The contaminated donor (active adapter) differs from the pristine donor —
    # this is exactly the round-1 B1 bug (donor was the TRAINED prefix KV).
    any_layer_differs = any(
        not torch.allclose(donor_active[layer][0], donor_pristine[layer][0], atol=1e-5)
        or not torch.allclose(donor_active[layer][1], donor_pristine[layer][1], atol=1e-5)
        for layer in donor_active
    )
    assert any_layer_differs, (
        "capturing the donor through the active adapter must differ from pristine base "
        "(if identical, the adapter is a no-op and the test fixture is broken)."
    )
    # The fixed donor (disable_adapter) reproduces the pristine base donor bit-for-bit.
    for layer in donor_fixed:
        assert torch.allclose(donor_fixed[layer][0], donor_pristine[layer][0], atol=1e-6), (
            f"layer {layer}: disable_adapter donor K must equal pristine base donor K"
        )
        assert torch.allclose(donor_fixed[layer][1], donor_pristine[layer][1], atol=1e-6), (
            f"layer {layer}: disable_adapter donor V must equal pristine base donor V"
        )


def test_detach_adapter_restores_pristine_base(tmp_path):
    """Cross-row hygiene: detach_adapter strips in-place LoRA, restoring pristine base."""
    mod = _load_driver()

    cfg = _tiny_cfg()
    adapter_dir = tmp_path / "adapter"
    _make_rslora_adapter(cfg, adapter_dir, seed=11)

    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_config(cfg).eval()
    ids = torch.tensor([[5, 9, 12, 4, 7, 3]])
    with torch.no_grad():
        pristine_logits = base(input_ids=ids).logits.clone()

    model = mod.attach_adapter(base, adapter_dir)
    # The adapter mutated base in place; del alone would NOT strip it (round-1 bug).
    base = mod.detach_adapter(model, base)
    del model

    n_lora = sum(1 for n, _ in base.named_parameters() if "lora" in n.lower())
    assert n_lora == 0, f"detach_adapter must strip all LoRA params from base; found {n_lora}"
    with torch.no_grad():
        restored_logits = base(input_ids=ids).logits.clone()
    assert torch.allclose(restored_logits, pristine_logits, atol=1e-6), (
        "after detach_adapter the base must reproduce pristine logits (clean for the next row)."
    )


def test_parity_probe_handoff_leaves_lorafree_base(tmp_path):
    """CBL1 PIN (round-3 reconciler): the parity-probe -> row-loop handoff.

    ``rsLoRA_parity_check`` (full function) does HF download + judge API calls, so
    here we pin the cleanup CONTRACT it now upholds: the attach -> use -> detach ->
    RE-BIND sequence (exactly the parity probe's new tail + the ``base =
    rsLoRA_parity_check(...)`` re-bind at the run_phase1 call site) must hand the
    Phase 1 row loop a LoRA-free base, so the row loop's first ``attach_adapter``
    / donor ``disable_adapter()`` reads off a pristine base — NOT a base silently
    contaminated by the bad_medical parity adapter.

    Round-2 bug: the parity check returned ``None`` and ``del model`` left the
    bad_medical LoRA injected into ``base``; the very first Phase 1 row then read
    its donor KV off the contaminated base, invalidating every prefix-KV-shift
    predictor (the headline).
    """
    mod = _load_driver()

    cfg = _tiny_cfg()
    parity_adapter = tmp_path / "parity_adapter"  # stands in for the bad_medical adapter
    _make_rslora_adapter(cfg, parity_adapter, seed=23)

    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_config(cfg).eval()
    ids = torch.tensor([[5, 9, 12, 4, 7, 3]])
    with torch.no_grad():
        pristine_logits = base(input_ids=ids).logits.clone()

    # --- Parity-probe phase: attach the (bad_medical) adapter, use it, then clean up
    #     EXACTLY as rsLoRA_parity_check's tail now does. ---
    probe_model = mod.attach_adapter(base, parity_adapter)
    with torch.no_grad():
        probe_model(input_ids=ids)  # the parity probe's generate/judge consumes this
    cleaned_base = mod.detach_adapter(probe_model, base)
    # The mandatory in-function assert the parity check now performs:
    n_lora = sum(1 for n, _ in cleaned_base.named_parameters() if "lora" in n.lower())
    assert n_lora == 0, (
        f"parity-probe handoff: base must be LoRA-free after detach; found {n_lora} (CBL1)."
    )
    del probe_model
    base = cleaned_base  # the `base = rsLoRA_parity_check(...)` re-bind

    # --- Row-loop phase: the first row attaches a (different) adapter on top of the
    #     handed-off base. It must see a clean base. ---
    row_adapter = tmp_path / "row_adapter"
    _make_rslora_adapter(cfg, row_adapter, seed=29)
    row_model = mod.attach_adapter(base, row_adapter)
    n_lora_row = sum(1 for n, _ in row_model.named_parameters() if "lora" in n.lower())
    # A clean handoff means the row model carries exactly ONE adapter's params, not
    # the parity adapter's stacked on top (which would double the lora-param count).
    base_after = mod.detach_adapter(row_model, base)
    del row_model
    with torch.no_grad():
        restored_logits = base_after(input_ids=ids).logits.clone()
    assert torch.allclose(restored_logits, pristine_logits, atol=1e-6), (
        "after the parity-probe -> row-loop handoff the base must STILL reproduce pristine "
        "logits — a contaminated handoff (round-2 bug) would carry the parity adapter's "
        "residual modules and diverge."
    )
    assert n_lora_row > 0, "fixture sanity: the row adapter must inject lora params."

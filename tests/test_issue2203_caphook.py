"""Issue #2203 — unit tests for the input-dependent axis-cap hook.

Covers (plan §7 / the implementer test mandate):

1. ``apply_cap_op`` cap/axis_replace/full_replace formulas on a toy ``(h, v, τ)``
   — the FLOOR semantics (only below-τ rows move; orthogonal complement preserved).
2. Decode-step firing: ``all-tokens`` fires once at prefill AND once per decode
   step (``n_edits == n_generated``), while ``all-prompt`` fires ONCE — on a tiny
   from-config Qwen2 model driving the REAL forward-hook + ``model.generate``.
3. Production-body e2e: ``run_arm`` + ``build_stack_for_arm`` through the REAL
   ``steering.generate_batch`` + real ``joint_axis_hooks`` on the tiny model,
   faking nothing (one-production-body-test rule). A cap whose floor is
   unreachable (τ = -1e9) is a proven no-op vs the baseline arm (non-corruption).

All CPU + from-config (no download, offline), fp32 — deterministic bit checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.experiments.issue2203 import caphook  # noqa: E402


def _tiny_model(vocab_size: int = 64):
    """A 2-layer from-config Qwen2 CausalLM (offline, CPU, standard decoder).

    ``vocab_size`` defaults tiny for the pure-hook tests; the ``run_arm`` e2e
    passes the REAL tokenizer's vocab so real BPE ids index a valid embedding.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    model.generation_config.pad_token_id = 0
    return model


# --------------------------------------------------------------------------- #
# 1. apply_cap_op formulas
# --------------------------------------------------------------------------- #
def test_cap_op_unit_axis_floors_below_tau_only():
    """cap: below-τ rows raised to τ (unit axis); at/above-τ rows untouched."""
    torch.manual_seed(0)
    H = 8
    v = torch.randn(H)
    v = v / v.norm()  # unit axis => raw proj == unit proj, clean floor at τ
    h = torch.randn(5, H)
    proj = h @ v
    tau = float(proj.median())  # some rows below, some above
    h_def = torch.randn(H)
    h_new, praw, pu_before, pu_after = caphook.apply_cap_op(
        h, "cap", v, v, tau, h_def, float(h_def @ v)
    )
    below = proj < tau
    # below-τ rows: axis projection raised exactly to τ
    assert torch.allclose(pu_after[below], torch.full_like(pu_after[below], tau), atol=1e-5)
    # at/above-τ rows: unchanged
    assert torch.allclose(h_new[~below], h[~below], atol=1e-6)
    # orthogonal complement preserved for below-τ rows (only the axis component moved)
    orth = h - proj[:, None] * v[None, :]
    orth_new = h_new - (h_new @ v)[:, None] * v[None, :]
    assert torch.allclose(orth[below], orth_new[below], atol=1e-5)
    # telemetry: raw proj before matches
    assert torch.allclose(praw, proj, atol=1e-5)
    assert torch.allclose(pu_before, proj, atol=1e-5)


def test_cap_op_verbatim_formula_nonunit_axis():
    """cap matches the paper Eq.1 verbatim h - v*clamp(<h,v>-τ, max=0) for non-unit v."""
    torch.manual_seed(1)
    H = 6
    v = torch.randn(H) * 2.5  # non-unit
    h = torch.randn(4, H)
    tau = 0.3
    h_def = torch.zeros(H)
    h_new, _, _, _ = caphook.apply_cap_op(h, "cap", v, v / v.norm(), tau, h_def, 0.0)
    excess = torch.clamp((h @ v) - tau, max=0.0)
    expected = h - v[None, :] * excess[:, None]
    assert torch.allclose(h_new, expected, atol=1e-5)


def test_axis_replace_sets_axis_component_to_proj_def():
    """axis_replace: <h_new,v̂> == proj_def for all rows; orthogonal complement fixed."""
    torch.manual_seed(2)
    H = 7
    v = torch.randn(H)
    v_hat = v / v.norm()
    h = torch.randn(4, H)
    h_def = torch.randn(H)
    proj_def = float(h_def @ v_hat)
    h_new, _, _, pu_after = caphook.apply_cap_op(h, "axis_replace", v, v_hat, 0.0, h_def, proj_def)
    assert torch.allclose(pu_after, torch.full_like(pu_after, proj_def), atol=1e-5)
    orth = h - (h @ v_hat)[:, None] * v_hat[None, :]
    orth_new = h_new - (h_new @ v_hat)[:, None] * v_hat[None, :]
    assert torch.allclose(orth, orth_new, atol=1e-5)


def test_full_replace_broadcasts_h_def():
    """full_replace: every row becomes h_def exactly."""
    torch.manual_seed(3)
    H = 5
    v = torch.randn(H)
    h = torch.randn(3, H)
    h_def = torch.randn(H)
    h_new, _, _, _ = caphook.apply_cap_op(h, "full_replace", v, v / v.norm(), 0.0, h_def, 0.0)
    assert torch.allclose(h_new, h_def[None, :].expand(3, H), atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. Decode-step firing (real hook + real model.generate)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "position_set,expect_decode", [("all-tokens", True), ("all-prompt", False)]
)
def test_edit_firing_counts_prefill_and_decode(position_set, expect_decode):
    model = _tiny_model()
    H = model.config.hidden_size
    v = torch.randn(H)
    h_def = torch.randn(H)
    hook = caphook.AxisCapHook(
        model, 0, v, tau=-1e9, h_def=h_def, op="cap", position_set=position_set
    )
    prompt_len = 6
    input_ids = torch.randint(1, model.config.vocab_size, (1, prompt_len))
    attn = torch.ones_like(input_ids)
    k = 4
    hook.arm_batch([prompt_len])
    hook.arm(prompt_len)
    with hook:
        out = model.generate(
            input_ids,
            attention_mask=attn,
            do_sample=False,
            max_new_tokens=k,
            min_new_tokens=k,
        )
    n_generated = int(out.shape[1] - prompt_len)
    assert n_generated == k
    if expect_decode:
        # prefill (1) + one edit per decode forward; total forwards == n_generated
        assert hook.n_edits == n_generated
    else:
        assert hook.n_edits == 1  # prefill only; decode passes through


def test_all_tokens_edits_more_than_all_prompt():
    """Direct contrast: all-tokens strictly outfires all-prompt when decoding."""
    counts = {}
    for ps in ("all-prompt", "all-tokens"):
        model = _tiny_model()
        H = model.config.hidden_size
        hook = caphook.AxisCapHook(
            model, 0, torch.randn(H), tau=-1e9, h_def=torch.randn(H), op="cap", position_set=ps
        )
        ids = torch.randint(1, model.config.vocab_size, (1, 5))
        hook.arm_batch([5])
        hook.arm(5)
        with hook:
            model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=3,
                min_new_tokens=3,
            )
        counts[ps] = hook.n_edits
    assert counts["all-tokens"] > counts["all-prompt"]
    assert counts["all-prompt"] == 1


def test_context_end_single_position_edit_and_telemetry():
    model = _tiny_model()
    H = model.config.hidden_size
    hook = caphook.AxisCapHook(
        model,
        1,
        torch.randn(H),
        tau=1e9,
        h_def=torch.randn(H),
        op="cap",
        position_set="context-end",
    )  # tau huge => cap fires on every row
    ids = torch.randint(1, model.config.vocab_size, (2, 4))
    hook.arm_batch([4, 4])
    hook.arm(4)
    with hook:
        model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=2,
            min_new_tokens=2,
        )
    assert hook.n_edits == 1  # single-position mode: prefill only
    assert hook.realized_edits is not None and len(hook.realized_edits) == 1
    rec = hook.realized_edits[0]
    assert rec["position_set"] == "context-end"
    assert rec["n_positions"] == 2  # one per row
    assert bool(rec["fired"].all())  # tau=+1e9 => every row below tau => fired


# --------------------------------------------------------------------------- #
# 3. Production-body e2e via run_arm + build_stack_for_arm + real generate_batch
# --------------------------------------------------------------------------- #
def test_run_arm_unreachable_cap_is_noop_vs_baseline():
    """run_arm through the REAL generate_batch: an unreachable cap == baseline text."""
    from scripts import issue2203_runtime as R

    tok = _load_tiny_tokenizer()  # skips if the Qwen2.5 tokenizer is not cached
    model = _tiny_model(vocab_size=len(tok))  # real BPE ids must index the embedding
    contexts = [{"system": "You are a helpful assistant.", "user": "Say hi."}]
    layers = [0, 1]
    H = model.config.hidden_size
    axis_by_layer = {li: torch.randn(H) for li in layers}
    h_def_by_layer = {li: torch.randn(H) for li in layers}
    tau_by_layer = {li: -1e9 for li in layers}  # unreachable floor => identity edit
    spec = {"kind": "cap", "op": "cap", "position_set": "all-tokens"}
    stack = R.build_stack_for_arm(
        model,
        spec,
        layers=layers,
        axis_by_layer=axis_by_layer,
        h_def_by_layer=h_def_by_layer,
        tau_by_layer=tau_by_layer,
    )
    assert isinstance(stack, caphook.AxisCapHookStack)
    base_texts, base_realized = R.run_arm(model, tok, contexts, None, max_new_tokens=6)
    cap_texts, cap_realized = R.run_arm(model, tok, contexts, stack, max_new_tokens=6)
    assert base_realized is None
    assert cap_realized is not None and len(cap_realized) >= 2  # both layers fired
    assert cap_texts == base_texts  # unreachable cap is a proven no-op


def test_build_stack_for_arm_baseline_is_none():
    from scripts import issue2203_runtime as R

    model = _tiny_model()
    layers = [0, 1]
    H = model.config.hidden_size
    stack = R.build_stack_for_arm(
        model,
        {"kind": "baseline"},
        layers=layers,
        axis_by_layer={li: torch.randn(H) for li in layers},
        h_def_by_layer={li: torch.randn(H) for li in layers},
        tau_by_layer={li: 0.0 for li in layers},
    )
    assert stack is None


def test_stack_handle_requires_all_installed():
    model = _tiny_model()
    H = model.config.hidden_size
    stack = caphook.joint_axis_hooks(
        model,
        [0, 1],
        {0: torch.randn(H), 1: torch.randn(H)},
        {0: 0.0, 1: 0.0},
        {0: torch.randn(H), 1: torch.randn(H)},
        op="cap",
        position_set="context-end",
    )
    assert stack._handle is None  # not yet installed
    with stack:
        assert stack._handle is stack  # all children installed
    assert stack._handle is None  # removed on exit


def _load_tiny_tokenizer():
    """Real Qwen2.5 tokenizer (cached) — real BPE ids for the e2e generate path.

    Skips (never fails / never downloads) when the tokenizer is not in the local
    HF cache, so the fleet-wide ``tests/`` run stays green in a sparse worktree.
    """
    import os

    from transformers import AutoTokenizer

    from scripts import issue2203_common as C

    try:
        return AutoTokenizer.from_pretrained(
            C.TINY_MODEL, local_files_only=(os.environ.get("HF_HUB_OFFLINE") == "1")
        )
    except Exception as exc:
        pytest.skip(f"Qwen2.5 tokenizer unavailable for e2e ({exc})")

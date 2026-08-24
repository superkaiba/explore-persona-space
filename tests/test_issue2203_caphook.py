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


def test_cap_op_unit_norm_lands_at_tau():
    """cap (Fix A): with a NON-UNIT raw v, moved rows land at ⟨h_new,v̂⟩ == τ.

    The pre-fix formula projected AND updated with the raw contrast v, so a moved
    row overshot the intended edit by ‖v‖² (~O(10²-10³)). ``apply_cap_op`` now
    normalizes to v̂ on BOTH the projection and the update, so a below-τ row's
    UNIT projection lands EXACTLY at τ regardless of ‖v‖ — the direct pin that
    the ‖v‖² overshoot (plan §4.1 Fix A) is gone. A deliberately non-unit v
    (‖v‖ ~ 6) is the discriminating case: the pre-fix and post-fix formulas
    coincide only when ‖v‖ == 1.
    """
    torch.manual_seed(1)
    H = 6
    v = torch.randn(H) * 2.5  # deliberately NON-unit
    v_hat = v / v.norm()
    h = torch.randn(4, H)
    proj_unit = h @ v_hat
    tau = float(proj_unit.median())  # some rows below τ, some above
    h_def = torch.zeros(H)
    h_new, praw, pu_before, pu_after = caphook.apply_cap_op(h, "cap", v, v_hat, tau, h_def, 0.0)
    below = proj_unit < tau
    # Below-τ rows land EXACTLY at τ in unit space (no ‖v‖² overshoot).
    assert torch.allclose(pu_after[below], torch.full_like(pu_after[below], tau), atol=1e-5)
    # At/above-τ rows untouched.
    assert torch.allclose(h_new[~below], h[~below], atol=1e-6)
    # Telemetry: raw-v projection is ⟨h,v⟩ (= ‖v‖ · unit projection); unit before matches.
    assert torch.allclose(praw, h @ v, atol=1e-4)
    assert torch.allclose(pu_before, proj_unit, atol=1e-5)
    # The pre-fix raw-v formula WOULD have overshot — assert h_new is NOT the buggy form.
    excess_raw = torch.clamp((h @ v) - tau, max=0.0)
    buggy = h - v[None, :] * excess_raw[:, None]
    assert not torch.allclose(h_new[below], buggy[below], atol=1e-3)


def test_cap_op_no_change_when_above_tau():
    """cap: when EVERY row's unit projection is already ≥ τ, the state is untouched."""
    torch.manual_seed(4)
    H = 6
    v = torch.randn(H) * 3.0  # non-unit
    v_hat = v / v.norm()
    h = torch.randn(5, H)
    proj_unit = h @ v_hat
    tau = float(proj_unit.min()) - 1.0  # τ strictly below every row => nothing fires
    h_new, _, pu_before, pu_after = caphook.apply_cap_op(
        h, "cap", v, v_hat, tau, torch.zeros(H), 0.0
    )
    assert torch.allclose(h_new, h, atol=1e-6)  # identity: no row below τ
    assert torch.allclose(pu_after, pu_before, atol=1e-6)


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
# 1b. steer op (issue #2223-fu aggressive-strength arms)
# --------------------------------------------------------------------------- #
def test_steer_op_is_registered():
    assert "steer" in caphook.OPS


def test_steer_op_shifts_unit_projection_by_alpha():
    """steer: <h_new,v̂> == <h,v̂> + alpha for EVERY row, unconditional; complement fixed."""
    torch.manual_seed(11)
    H = 8
    v = torch.randn(H) * 2.5  # deliberately NON-unit — exercises v̂ normalization
    v_hat = v / v.norm()
    h = torch.randn(6, H)
    proj_before = h @ v_hat
    alpha = 1.7
    h_new, praw, pu_before, pu_after = caphook.apply_cap_op(
        h, "steer", v, v_hat, 0.0, torch.zeros(H), 0.0, alpha
    )
    # unconditional additive shift: EVERY row's unit projection moves by exactly alpha
    assert torch.allclose(pu_after, proj_before + alpha, atol=1e-5)
    assert torch.allclose(pu_before, proj_before, atol=1e-5)
    assert torch.allclose(praw, h @ v, atol=1e-4)  # raw-v telemetry unchanged
    # orthogonal complement preserved — only the axis component moved
    orth = h - proj_before[:, None] * v_hat[None, :]
    orth_new = h_new - (h_new @ v_hat)[:, None] * v_hat[None, :]
    assert torch.allclose(orth, orth_new, atol=1e-5)
    # negative alpha shifts the other way (sign faithful)
    _, _, _, pu_dn = caphook.apply_cap_op(h, "steer", v, v_hat, 0.0, torch.zeros(H), 0.0, -alpha)
    assert torch.allclose(pu_dn, proj_before - alpha, atol=1e-5)


def test_alpha_is_inert_for_cap_axisreplace_fullreplace():
    """Regression: a nonzero alpha does NOT change cap / axis_replace / full_replace outputs."""
    torch.manual_seed(12)
    H = 6
    v = torch.randn(H)
    v_hat = v / v.norm()
    h = torch.randn(4, H)
    h_def = torch.randn(H)
    proj_def = float(h_def @ v_hat)
    tau = float((h @ v_hat).median())  # some rows below τ, some above
    for op in ("cap", "axis_replace", "full_replace"):
        no_alpha, *_ = caphook.apply_cap_op(h, op, v, v_hat, tau, h_def, proj_def)
        with_alpha, *_ = caphook.apply_cap_op(h, op, v, v_hat, tau, h_def, proj_def, 99.0)
        assert torch.equal(no_alpha, with_alpha), op


def test_steer_hook_threads_alpha_through_joint_axis_hooks():
    """joint_axis_hooks(alpha_by_layer=...) wires the steer shift onto the REAL forward hook."""
    model = _tiny_model()
    H = model.config.hidden_size
    layers = [0, 1]
    alpha_by_layer = {0: 2.0, 1: -1.5}
    stack = caphook.joint_axis_hooks(
        model,
        layers,
        {li: torch.randn(H) for li in layers},
        {li: 0.0 for li in layers},  # tau: telemetry-only for steer
        {li: torch.randn(H) for li in layers},
        op="steer",
        position_set="context-end",
        alpha_by_layer=alpha_by_layer,
    )
    for h, li in zip(stack.hooks, layers, strict=True):
        assert h.op == "steer"
        assert h.alpha == alpha_by_layer[li]
    ids = torch.randint(1, model.config.vocab_size, (2, 4))
    stack.arm_batch([4, 4])
    stack.arm(4)
    with stack:
        model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=2,
            min_new_tokens=2,
        )
    realized = stack.realized_edits
    assert realized is not None and len(realized) == len(layers)  # single-position prefill edit
    for rec in realized:
        before = rec["proj_unit_before"]  # (B,) at the context-end position
        after = rec["proj_unit_after"]
        assert torch.allclose(after, before + alpha_by_layer[rec["layer"]], atol=1e-4)


def test_joint_axis_hooks_alpha_defaults_zero():
    """alpha_by_layer=None => every steer hook's alpha is 0.0 (a no-op steer)."""
    model = _tiny_model()
    H = model.config.hidden_size
    stack = caphook.joint_axis_hooks(
        model,
        [0, 1],
        {0: torch.randn(H), 1: torch.randn(H)},
        {0: 0.0, 1: 0.0},
        {0: torch.randn(H), 1: torch.randn(H)},
        op="steer",
        position_set="context-end",
    )
    assert all(h.alpha == 0.0 for h in stack.hooks)


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
    # Unreachable UNIT-space floor => identity edit (⟨h,v̂⟩ never below -1e9).
    tau_by_position = {"all-tokens": {li: -1e9 for li in layers}}
    spec = {"kind": "real", "op": "cap", "position_set": "all-tokens"}
    stack = R.build_stack_for_arm(
        model,
        spec,
        layers=layers,
        axis_by_layer=axis_by_layer,
        h_def_by_layer=h_def_by_layer,
        tau_by_position=tau_by_position,
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
        tau_by_position={"context-end": {li: 0.0 for li in layers}},
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


# --------------------------------------------------------------------------- #
# 4. run_arm generation-chunking (issue #2203 Phase-3 OOM fix)
#    CPU-feasible unit tests over a stub generate_batch — the 32B OOM the fix
#    targets cannot run on the VM; CUDA validation lands at relaunch.
# --------------------------------------------------------------------------- #
def _ctx(i: int) -> dict:
    return {"system": "s", "user": f"q{i}"}


def test_run_arm_baseline_chunks_preserve_order_and_count(monkeypatch):
    """Baseline path: chunked at GEN_BATCH_SIZE, texts in original order, no drops."""
    from scripts import issue2203_runtime as R

    calls: list[list[str]] = []

    def fake_generate_batch(
        model,
        tokenizer,
        contexts,
        n=1,
        hook=None,
        max_new_tokens=1024,
        temperature=1.0,
        top_p=None,
        seed_base=42,
        render_fn=None,
        ids_fn=None,
    ):
        assert n == 1 and hook is None and temperature == 0.0  # greedy, unhooked
        assert top_p is None and render_fn is None and ids_fn is None  # 7B ladder default
        calls.append([c["user"] for c in contexts])
        return [[f"resp::{c['user']}"] for c in contexts]  # results[b][i], n=1

    monkeypatch.setattr(R.steering, "generate_batch", fake_generate_batch)
    monkeypatch.setattr(R, "GEN_BATCH_SIZE", 3)

    contexts = [_ctx(i) for i in range(7)]
    texts, realized = R.run_arm(None, None, contexts, None, max_new_tokens=8)

    assert realized is None
    assert texts == [f"resp::q{i}" for i in range(7)]  # order + count preserved
    assert calls == [["q0", "q1", "q2"], ["q3", "q4", "q5"], ["q6"]]  # 3-chunked, ragged tail


def test_run_arm_hooked_arms_per_chunk_and_aggregates_realized(monkeypatch):
    """Hooked path: install ONCE, arm_batch PER CHUNK, realized_edits extended."""
    from scripts import issue2203_runtime as R

    arm_calls: list[tuple[list[int], object]] = []

    class FakeStack:
        position_set = "context-end"

        def __init__(self):
            self.realized_edits = None
            self.entered = 0
            self.exited = 0

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, *exc):
            self.exited += 1

        def arm_batch(self, row_lengths, prefix_ends):
            arm_calls.append((list(row_lengths), prefix_ends))
            self.realized_edits = None  # matches AxisCapHook.arm_batch's reset

    def fake_generate_batch(
        model,
        tokenizer,
        contexts,
        n=1,
        hook=None,
        max_new_tokens=1024,
        temperature=1.0,
        top_p=None,
        seed_base=42,
        render_fn=None,
        ids_fn=None,
    ):
        assert hook is not None  # hooked path
        # simulate a real forward firing on THIS chunk: 2 layers, one record each
        hook.realized_edits = [
            {"layer": 0, "n_positions": len(contexts), "fired_frac": 1.0},
            {"layer": 1, "n_positions": len(contexts), "fired_frac": 1.0},
        ]
        return [[f"cap::{c['user']}"] for c in contexts]

    def fake_context_token_ids(tokenizer, ctx):
        # >= 2 tokens (arm_batch precondition); per-ctx length varies
        return list(range(2 + int(ctx["user"][1:]) % 2))

    monkeypatch.setattr(R.steering, "generate_batch", fake_generate_batch)
    monkeypatch.setattr(R.steering, "context_token_ids", fake_context_token_ids)
    monkeypatch.setattr(R, "GEN_BATCH_SIZE", 2)

    stack = FakeStack()
    contexts = [_ctx(i) for i in range(5)]
    texts, realized = R.run_arm(None, None, contexts, stack, max_new_tokens=8)

    assert texts == [f"cap::q{i}" for i in range(5)]  # order + count preserved
    assert stack.entered == 1 and stack.exited == 1  # installed ONCE for the whole loop
    assert len(arm_calls) == 3  # armed per chunk: [q0,q1] [q2,q3] [q4]
    assert [len(rl) for rl, _ in arm_calls] == [2, 2, 1]
    assert all(pe is None for _, pe in arm_calls)  # context-end => no prefix_ends
    # realized_edits aggregated across chunks (2 layer records * 3 chunks), and
    # total positions edited == n_contexts * n_layers (the pre-fix single-forward
    # total is preserved under chunking).
    assert realized is not None and len(realized) == 6
    assert sum(r["n_positions"] for r in realized) == 5 * 2


def test_run_arm_hooked_prefix_end_computes_prefix_ends_per_chunk(monkeypatch):
    """prefix-end position set: run_arm computes per-chunk prefix_ends and passes them."""
    from scripts import issue2203_runtime as R

    arm_calls: list[tuple[list[int], object]] = []

    class FakeStack:
        position_set = "prefix-end"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def arm_batch(self, row_lengths, prefix_ends):
            arm_calls.append((list(row_lengths), prefix_ends))
            self.realized_edits = [{"layer": 0, "n_positions": len(row_lengths), "fired_frac": 0.5}]

    def fake_generate_batch(
        model,
        tokenizer,
        contexts,
        n=1,
        hook=None,
        max_new_tokens=1024,
        temperature=1.0,
        top_p=None,
        seed_base=42,
        render_fn=None,
        ids_fn=None,
    ):
        return [[f"cap::{c['user']}"] for c in contexts]

    monkeypatch.setattr(R.steering, "generate_batch", fake_generate_batch)
    monkeypatch.setattr(R.steering, "context_token_ids", lambda tok, ctx: [0, 1, 2])
    monkeypatch.setattr(R.steering, "prefix_end_index", lambda tok, ids: 2)
    monkeypatch.setattr(R, "GEN_BATCH_SIZE", 2)

    stack = FakeStack()
    texts, realized = R.run_arm(None, None, [_ctx(i) for i in range(3)], stack, max_new_tokens=8)

    assert texts == ["cap::q0", "cap::q1", "cap::q2"]
    assert len(arm_calls) == 2  # [q0,q1] [q2]
    assert [pe for _, pe in arm_calls] == [[2, 2], [2]]  # prefix_ends computed per chunk
    assert realized is not None and len(realized) == 2  # one record per chunk, aggregated


def test_gen_batch_size_env_override(monkeypatch):
    """GEN_BATCH_SIZE is read from EPM_ISSUE2203_GEN_BATCH at import."""
    import importlib

    from scripts import issue2203_runtime as R

    monkeypatch.setenv("EPM_ISSUE2203_GEN_BATCH", "7")
    try:
        importlib.reload(R)
        assert R.GEN_BATCH_SIZE == 7
    finally:
        monkeypatch.delenv("EPM_ISSUE2203_GEN_BATCH", raising=False)
        importlib.reload(R)  # restore module-level default for later tests


# --------------------------------------------------------------------------- #
# 5. Plan-v9 amendment: axis-replace random-direction controls
#    (axrep_allprompt_randnull / axrep_alltoken_randnull) — the whole diff is
#    two ARM_SPECS entries; these pin the three properties the plan §4.1/§12
#    argue hold FOR FREE from build_stack_for_arm's op-agnostic null branch.
# --------------------------------------------------------------------------- #
def _tiny_geom(layers: list[int], H: int):
    """A tiny position-keyed geometry (axis / h_def / real τ / footprint-matched τ_rand).

    Position-keyed to match the ``build_stack_for_arm`` schema-v2 API (Fix B):
    ``tau_by_position`` carries all four position sets; ``tau_rand_alltoken`` is
    the all-tokens footprint-matched random-direction pool the two broad-position
    ``*_alltoken_randnull`` arms read.
    """
    torch.manual_seed(7)
    return {
        "axis_by_layer": {li: torch.randn(H) for li in layers},
        "h_def_by_layer": {li: torch.randn(H) for li in layers},
        "tau_by_position": {ps: {li: 0.0 for li in layers} for ps in caphook.POSITION_SETS},
        "tau_rand_alltoken": {li: -0.7 for li in layers},
    }


def test_axrep_randnull_stack_uses_same_seeded_vrand_as_cap_randnull():
    """v_rand identity (§4.1): the axrep_*_randnull arm's per-layer random axis is
    IDENTICAL to the parent's cap_*_randnull arm — both are kind "null" at the
    all-tokens position, so both
    hit build_stack_for_arm's `_seeded_random_axis(axis, null_seed+li)` branch with
    the DEFAULT null_seed=1234. This is the "same seeded v_rand as the parent" the
    brief requires, obtained with no override."""
    from scripts import issue2203_common as C
    from scripts import issue2203_runtime as R

    model = _tiny_model()
    layers = [0, 1]
    H = model.config.hidden_size
    g = _tiny_geom(layers, H)
    kwargs = dict(
        layers=layers,
        axis_by_layer=g["axis_by_layer"],
        h_def_by_layer=g["h_def_by_layer"],
        tau_by_position=g["tau_by_position"],
        tau_rand_by_position={"all-tokens": g["tau_rand_alltoken"]},
    )
    axrep = R.build_stack_for_arm(model, C.ARM_SPECS["axrep_alltoken_randnull"], **kwargs)
    cap = R.build_stack_for_arm(model, C.ARM_SPECS["cap_alltoken_randnull"], **kwargs)
    assert axrep.op == "axis_replace" and cap.op == "cap"  # op differs, direction does not
    for hx, hc in zip(axrep.hooks, cap.hooks, strict=True):
        assert torch.equal(hx.v, hc.v)  # bit-identical seeded random direction
    # ...and NOT the real axis (the control overwrites a RANDOM component).
    for hx, li in zip(axrep.hooks, layers, strict=True):
        assert not torch.equal(hx.v, g["axis_by_layer"][li].float())


def test_axrep_randnull_output_ignores_tau_scope_outputs_not_telemetry():
    """τ-inertness (§4.1 / implementer note 2): perturbing the passed τ_rand leaves
    the axis-replace MODEL OUTPUT and realized projection unchanged (apply_cap_op
    reads τ only on the "cap" branch). The assert is scoped to OUTPUTS/realized
    projection — NOT edit_telemetry, whose fired_frac reads τ and DOES move."""
    from scripts import issue2203_common as C
    from scripts import issue2203_runtime as R

    tok = _load_tiny_tokenizer()
    model = _tiny_model(vocab_size=len(tok))
    contexts = [
        {"system": "You are a helpful assistant.", "user": "Say hi."},
        {"system": "You are a helpful assistant.", "user": "Name a color."},
    ]
    layers = [0, 1]
    H = model.config.hidden_size
    g = _tiny_geom(layers, H)
    spec = C.ARM_SPECS["axrep_alltoken_randnull"]

    def _run(tau_rand):
        stack = R.build_stack_for_arm(
            model,
            spec,
            layers=layers,
            axis_by_layer=g["axis_by_layer"],
            h_def_by_layer=g["h_def_by_layer"],
            tau_by_position=g["tau_by_position"],
            tau_rand_by_position={"all-tokens": tau_rand},
        )
        return R.run_arm(model, tok, contexts, stack, max_new_tokens=6)

    base = g["tau_rand_alltoken"]
    perturbed = {li: v + 5.0 for li, v in base.items()}  # large τ shift
    texts_a, realized_a = _run(base)
    texts_b, realized_b = _run(perturbed)
    # OUTPUT is τ-independent for axis_replace: identical greedy completions.
    assert texts_a == texts_b
    # Realized axis projection AFTER the edit is τ-independent too.
    proj_a = [r["proj_unit_after_mean"] for r in realized_a]
    proj_b = [r["proj_unit_after_mean"] for r in realized_b]
    assert proj_a == proj_b
    # But fired_frac (τ-dependent telemetry) is DEMONSTRABLY sensitive to τ —
    # exactly why the inertness assert must NOT be scoped to edit_telemetry.
    fired_a = [r["fired_frac"] for r in realized_a]
    fired_b = [r["fired_frac"] for r in realized_b]
    assert fired_a != fired_b


def test_axrep_randnull_moves_component_to_proj_def_along_random_vhat():
    """axis-replace sets ⟨h,v̂_rand⟩ → proj_def along the RANDOM unit direction
    (§4.1): after generation, the realized post-edit projection equals each hook's
    proj_def = ⟨h_def, v̂_rand⟩, and the edit actually fired (n_edits > 0)."""
    from scripts import issue2203_common as C
    from scripts import issue2203_runtime as R

    tok = _load_tiny_tokenizer()
    model = _tiny_model(vocab_size=len(tok))
    contexts = [{"system": "You are a helpful assistant.", "user": "Say hi."}]
    layers = [0, 1]
    H = model.config.hidden_size
    g = _tiny_geom(layers, H)
    stack = R.build_stack_for_arm(
        model,
        C.ARM_SPECS["axrep_allprompt_randnull"],
        layers=layers,
        axis_by_layer=g["axis_by_layer"],
        h_def_by_layer=g["h_def_by_layer"],
        tau_by_position=g["tau_by_position"],
        # all-prompt has NO random pool => τ resolves to 0.0 (inert for axis_replace).
        tau_rand_by_position={"all-tokens": g["tau_rand_alltoken"]},
    )
    _texts, realized = R.run_arm(model, tok, contexts, stack, max_new_tokens=6)
    assert realized is not None and stack.n_edits > 0
    # proj_def per hook is ⟨h_def, v̂_rand⟩ where v̂_rand is the SEEDED random unit.
    proj_def_by_layer = {h.layer: h.proj_def for h in stack.hooks}
    for rec in realized:
        assert rec["op"] == "axis_replace"
        assert rec["proj_unit_after_mean"] == pytest.approx(
            proj_def_by_layer[rec["layer"]], abs=1e-4
        )


# --------------------------------------------------------------------------- #
# 6. Fix B position-matched τ selection + Fix D paper-engine prefill-only cap
# --------------------------------------------------------------------------- #
def test_tau_position_matched_selection():
    """build_stack_for_arm selects the τ dict MATCHED to the arm's position_set (Fix B).

    A real arm reads ``tau_by_position[position_set]``; a null cap arm reads
    ``tau_rand_by_position[position_set]``. Distinct per-position τ values prove
    the selection is position-keyed, not a single global τ (the Fix-B bug: τ was
    calibrated on response-token pools and applied at prompt positions).
    """
    from scripts import issue2203_common as C
    from scripts import issue2203_runtime as R

    model = _tiny_model()
    layers = [0, 1]
    H = model.config.hidden_size
    # Distinct τ per position set so a mis-selection is DETECTABLE (not a global τ).
    tau_by_position = {
        "prefix-end": {li: -1.0 for li in layers},
        "context-end": {li: -2.0 for li in layers},
        "all-prompt": {li: -3.0 for li in layers},
        "all-tokens": {li: -4.0 for li in layers},
    }
    tau_rand_by_position = {
        "context-end": {li: -20.0 for li in layers},
        "all-tokens": {li: -40.0 for li in layers},
    }
    kwargs = dict(
        layers=layers,
        axis_by_layer={li: torch.randn(H) for li in layers},
        h_def_by_layer={li: torch.randn(H) for li in layers},
        tau_by_position=tau_by_position,
        tau_rand_by_position=tau_rand_by_position,
    )
    # REAL cap arms → the position-matched REAL τ.
    for arm, want in (
        ("cap_prefix", -1.0),
        ("cap_ctx", -2.0),
        ("cap_allprompt", -3.0),
        ("cap_alltoken", -4.0),
    ):
        stack = R.build_stack_for_arm(model, C.ARM_SPECS[arm], **kwargs)
        assert all(h.tau == want for h in stack.hooks), (arm, [h.tau for h in stack.hooks])
    # NULL cap arms → the position-matched RANDOM τ pool (never the real τ).
    stack = R.build_stack_for_arm(model, C.ARM_SPECS["cap_ctx_randnull"], **kwargs)
    assert all(h.tau == -20.0 for h in stack.hooks)
    stack = R.build_stack_for_arm(model, C.ARM_SPECS["cap_alltoken_randnull"], **kwargs)
    assert all(h.tau == -40.0 for h in stack.hooks)


def test_paper_engine_context_end_prefill_only(monkeypatch):
    """Fix D: PrefillContextEndSteering fires the paper cap on the T>1 prefill only.

    Builds the REAL production subclass on a LOCAL stub base — NEVER hard-imports
    ``assistant_axis`` (the paper engine is a pod-only pinned-SHA bootstrap clone,
    and its package ``__init__`` needs plotly/sklearn). Only the external paper
    base is faked; the subclass's own ``_apply_layer_interventions`` body executes
    (one-production-body-test rule). A multi-token PREFILL forward delegates to
    ``super()`` (the paper cap fires); a single-token DECODE forward (T==1 under
    the KV cache) passes the activations through untouched, for BOTH the bare
    tensor and the HF ``(tensor, ...)`` tuple output shapes.
    """
    import types

    from explore_persona_space.experiments.issue2203 import paper_engine

    class _StubActivationSteering:
        def __init__(
            self,
            model,
            steering_vectors,
            *,
            coefficients,
            layer_indices,
            intervention_type,
            positions,
            cap_thresholds=None,
            **kw,
        ):
            self.model = model
            self.positions = positions
            self.super_calls: list[tuple] = []

        def _apply_layer_interventions(self, activations, layer_idx):
            tensor = activations[0] if isinstance(activations, (tuple, list)) else activations
            self.super_calls.append((layer_idx, tuple(tensor.shape)))
            return activations  # stub cap: identity (we only assert it WAS reached)

    fake_mod = types.SimpleNamespace(ActivationSteering=_StubActivationSteering)
    monkeypatch.setattr(paper_engine, "_cached_prefill_class", None)
    monkeypatch.setattr(paper_engine, "load_paper_steering_module", lambda: fake_mod)

    Subclass = paper_engine._prefill_context_end_class()
    assert Subclass.__bases__ == (_StubActivationSteering,)
    assert Subclass.prefill_only is True
    inst = Subclass(
        None,
        {},
        coefficients=[0.0],
        layer_indices=[0],
        intervention_type="capping",
        positions="last",
        cap_thresholds={},
    )
    # PREFILL (T>1) → super() reached (the paper cap fires at the context-end pos).
    prefill = torch.zeros(2, 5, 8)
    out_prefill = inst._apply_layer_interventions(prefill, 0)
    assert inst.super_calls == [(0, (2, 5, 8))]
    assert out_prefill is prefill
    # DECODE (T==1) → passed through untouched, super NOT reached.
    decode = torch.zeros(2, 1, 8)
    out_decode = inst._apply_layer_interventions(decode, 0)
    assert inst.super_calls == [(0, (2, 5, 8))]  # unchanged
    assert out_decode is decode
    # HF tuple output shape: a T==1 decode tuple passes through unchanged too.
    decode_tuple = (torch.zeros(2, 1, 8), None)
    out_tuple = inst._apply_layer_interventions(decode_tuple, 1)
    assert inst.super_calls == [(0, (2, 5, 8))]  # still unchanged
    assert out_tuple is decode_tuple

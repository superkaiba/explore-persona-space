"""Issue #2223 follow-up — answer-first-k capping (CPU smoke, no GPU / no download).

Covers the brief's CHANGE-3 contract on the ``answer-first-k`` position set of
:class:`AxisCapHook` plus the replay-script arm wiring:

1. Core first-k logic on tiny random tensors driven through the REAL registered
   forward hook (fake prefill forward + several decode forwards): (a) ZERO
   edits on the prefill; (b) edits on decode steps 1..k only; (c) decode steps
   k+1.. pass through unedited; (d) ``arm()``/``reset()`` zero the per-draw
   counter so a second draw re-edits steps 1..k.
2. The same firing pattern through the real ``model.generate`` KV-cache decode
   path on a tiny from-config Qwen2 model (mirrors test_issue2203_caphook.py).
3. Replay-script wiring: the 4 ``cap_ansfirst{1,2,4,8}`` arms' registry specs,
   the ``--arms ansfirst`` group token, band-only cell enumeration, and
   ``build_cs_stack``'s answer-first-k → context-end geometry mapping +
   ``first_k_decode`` threading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.experiments.issue2203 import caphook  # noqa: E402
from scripts import issue2223_casestudy_replay as R  # noqa: E402

H = 8
PROMPT_T = 5
ANSFIRST_NAMES = [f"cap_ansfirst{k}" for k in (1, 2, 4, 8)]


class _TinyDecoder(nn.Module):
    """Minimal ``model.model.layers`` tree; Identity blocks let the registered
    forward hook see (and replace) raw hidden-state tensors directly."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([nn.Identity() for _ in range(n_layers)])
        self.model = inner


def _mk_hook(k: int = 2, tau: float = 1e6) -> tuple[caphook.AxisCapHook, nn.Module]:
    """answer-first-k cap hook on a tiny decoder; tau=+1e6 ⇒ the floor fires on
    every edited position (any finite projection sits below it)."""
    torch.manual_seed(0)
    model = _TinyDecoder()
    hook = caphook.AxisCapHook(
        model,
        0,
        torch.randn(H),
        tau=tau,
        h_def=torch.randn(H),
        op="cap",
        position_set="answer-first-k",
        first_k_decode=k,
    )
    return hook, model.model.layers[0]


def test_answer_first_k_in_position_sets():
    assert "answer-first-k" in caphook.POSITION_SETS


def test_constructor_requires_first_k_decode():
    model = _TinyDecoder()
    with pytest.raises(AssertionError):
        caphook.AxisCapHook(
            model,
            0,
            torch.randn(H),
            tau=0.0,
            h_def=torch.randn(H),
            op="cap",
            position_set="answer-first-k",
        )  # default first_k_decode=0 is invalid for this position set


def test_first_k_decode_edit_window_and_counter_reset():
    """The CHANGE-3 contract: prefill unedited; decode steps 1-2 edited; steps
    3+ pass through; arm() (per-draw re-arm) restarts the window."""
    hook, block = _mk_hook(k=2)
    hook.arm_batch([PROMPT_T])
    hook.arm(PROMPT_T)
    with hook:
        # (a) prefill forward: 0 edits, output byte-identical.
        prompt = torch.randn(1, PROMPT_T, H)
        out = block(prompt)
        assert torch.equal(out, prompt)
        assert hook.n_edits == 0
        assert hook.realized_edits is None

        # (b) decode steps 1 and 2: edited (tau=+1e6 floor fires on every row).
        for step in (1, 2):
            d = torch.randn(1, 1, H)
            o = block(d)
            assert hook.n_edits == step, (step, hook.n_edits)
            assert not torch.equal(o, d), f"decode step {step} not edited"
            # cap lands the unit projection exactly at tau
            proj = float(o[0, 0].float() @ hook.v_hat)
            assert proj == pytest.approx(hook.tau, rel=1e-4)

        # (c) decode steps 3 and 4: pass through unedited, count frozen at 2.
        for step in (3, 4):
            d = torch.randn(1, 1, H)
            o = block(d)
            assert torch.equal(o, d), f"decode step {step} was edited"
            assert hook.n_edits == 2

        # telemetry: exactly 2 realized edits, both decode-phase, 1 position each.
        assert len(hook.realized_edits) == 2
        assert all(r["phase"] == "decode" for r in hook.realized_edits)
        assert all(r["n_positions"] == 1 for r in hook.realized_edits)
        assert all(r["fired_frac"] == 1.0 for r in hook.realized_edits)

        # (d) a second draw: arm() resets the counter → steps 1-2 edit again.
        hook.arm(PROMPT_T)
        assert hook._decode_steps_seen == 0
        prompt2 = torch.randn(1, PROMPT_T, H)
        assert torch.equal(block(prompt2), prompt2)  # prefill again unedited
        for _step in (1, 2):
            d = torch.randn(1, 1, H)
            assert not torch.equal(block(d), d)
        d = torch.randn(1, 1, H)
        assert torch.equal(block(d), d)  # step 3 of draw 2 passes through
        assert hook.n_edits == 4

    # bare reset() also zeroes the counter (arm() routes through it).
    hook._decode_steps_seen = 7
    hook.reset()
    assert hook._decode_steps_seen == 0


def _tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=64,
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


def test_first_k_firing_through_real_generate():
    """Real KV-cache decode path: n_edits == min(k, n_new - 1), never the prefill.

    Generating n_new tokens = 1 prefill forward + (n_new - 1) decode (T==1)
    forwards: the FIRST answer token is sampled from the prefill's last PROMPT
    position (a prompt position — deliberately unedited under answer-first-k),
    and decode forward j computes the hidden state AT answer-token position j.
    Mirrors all-tokens, whose n_edits == n_generated = 1 prefill edit +
    (n_new - 1) decode edits (test_issue2203_caphook.py).
    """
    for k, n_new, expect in ((2, 4, 2), (8, 3, 2)):
        model = _tiny_qwen()
        hidden = model.config.hidden_size
        hook = caphook.AxisCapHook(
            model,
            0,
            torch.randn(hidden),
            tau=1e9,  # floor above any projection ⇒ every edited step fires
            h_def=torch.randn(hidden),
            op="cap",
            position_set="answer-first-k",
            first_k_decode=k,
        )
        ids = torch.randint(1, model.config.vocab_size, (1, 6))
        hook.arm_batch([6])
        hook.arm(6)
        with hook:
            out = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=n_new,
                min_new_tokens=n_new,
            )
        assert int(out.shape[1] - 6) == n_new
        assert hook.n_edits == expect, (k, n_new, hook.n_edits)
        assert all(r["phase"] == "decode" for r in hook.realized_edits)


# --------------------------------------------------------------------------- #
# replay-script wiring
# --------------------------------------------------------------------------- #
def test_ansfirst_arm_registry_specs():
    assert list(R.ANSFIRST_ARMS) == ANSFIRST_NAMES
    for k in (1, 2, 4, 8):
        assert R.CS_ARMS[f"cap_ansfirst{k}"] == {
            "engine": "caphook",
            "op": "cap",
            "position_set": "answer-first-k",
            "axis": "answer",
            "when": "every",
            "first_k_decode": k,
            "band_only": True,
        }
    # registered in ARM_ORDER (reachable via --arm too)
    assert set(ANSFIRST_NAMES) <= set(R.ARM_ORDER)


class _Args:
    def __init__(self, **kw):
        self.arm = "all"
        self.arms = None
        for k, v in kw.items():
            setattr(self, k, v)


def test_resolve_arms_ansfirst_group():
    assert R.resolve_arms(_Args(arms="ansfirst")) == ANSFIRST_NAMES
    # existing group tokens unchanged by the addition
    assert set(R.resolve_arms(_Args(arms="original"))).isdisjoint(ANSFIRST_NAMES)
    assert set(R.resolve_arms(_Args(arms="new18"))).isdisjoint(ANSFIRST_NAMES)


def test_enumerate_cells_ansfirst_band_only():
    cells = R.enumerate_cells(["delusion"], ANSFIRST_NAMES, ["band", "all"])
    assert cells == [("delusion", a, "band") for a in ANSFIRST_NAMES]


def test_build_cs_stack_ansfirst_geometry_mapping_and_threading():
    """build_cs_stack pulls τ/h_def from the CONTEXT-END geometry entries and
    threads first_k_decode; the stack's position_set stays answer-first-k."""
    model = _tiny_qwen()
    hidden = model.config.hidden_size
    li = 0
    ctx_floor, pre_floor = 3.25, -7.5  # distinct so a wrong-key lookup is loud
    geom = {
        "answer_axis": {li: torch.randn(hidden)},
        "native_axes": {},
        "default_states": {
            "context": {li: torch.randn(hidden)},
            "prefix": {li: torch.randn(hidden)},
        },
        "floor_tau": {"answer": {"context-end": {li: ctx_floor}, "prefix-end": {li: pre_floor}}},
        "cap_percentile_tau": {},
        "alpha": {},
    }
    stack = R.build_cs_stack("cap_ansfirst4", [li], model, geom)
    assert stack is not None
    assert stack.position_set == "answer-first-k"
    child = stack.hooks[0]
    assert child.first_k_decode == 4
    assert child.tau == ctx_floor  # context-end floor, never the prefix-end one
    assert torch.equal(child.h_def, geom["default_states"]["context"][li].float())

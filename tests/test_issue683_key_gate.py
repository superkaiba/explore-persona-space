"""Issue #683 — invariants for the behavior-dependent key-gate pipeline.

Pins the load-bearing design decisions the planner/critic flagged:

  1. ``realized_gate`` computes ⟨ŵ,Δv⟩/⟨ŵ,ŵ⟩ and FAILS LOUD on a zero-norm
     source write (g_real is undefined for a degenerate write — must raise,
     never silently return 0/NaN that would corrupt the leaderboard).
  2. The read-location PINS distinguish the marker (post-response EOR slot)
     from sycophancy (answer-span mean) — methodology-critic concern #1. A
     regression that silently unified the two reads is a measurement-validity
     bug.
  3. ``answer_span_token_indices`` returns the post-prefix completion tokens
     and FAILS LOUD on chat-template prefix drift (the answer-span pool would
     otherwise silently include prompt tokens).
  4. The scorer's A7 read distinguishes a rank-1 stack (scalar g_real DV) from
     a genuinely 2-D stack (low-rank fallback) — the gating control.
  5. The shuffled-key null is a key-VECTOR permutation that destroys the
     key↔gate correspondence (methodology-critic concern #2): a real
     informative key scores ABOVE its shuffled-key null mean.

CPU-only, no GPU, no network. Runs in the standard pytest suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from explore_persona_space.experiments.issue_683 import (  # noqa: E402
    READ_LOCATION,
    answer_span_token_indices,
    realized_gate,
)


def test_realized_gate_self_is_one():
    """g_real(C)=1 by construction when Δv(C') == ŵ."""
    import torch

    w = torch.tensor([1.0, 2.0, 3.0])
    assert realized_gate(w, w) == pytest.approx(1.0)


def test_realized_gate_scales_linearly():
    """g_real(2·ŵ) == 2 (the gate is the scalar that scales the write)."""
    import torch

    w = torch.tensor([1.0, -2.0, 0.5])
    assert realized_gate(w, 2.0 * w) == pytest.approx(2.0)
    assert realized_gate(w, -0.5 * w) == pytest.approx(-0.5)


def test_realized_gate_zero_norm_raises():
    """A degenerate (zero-norm) source write must FAIL LOUD, not return 0/NaN."""
    import torch

    with pytest.raises(ValueError, match="zero norm"):
        realized_gate(torch.zeros(4), torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_read_location_pins_distinct_per_behavior():
    """Methodology-critic concern #1: marker and sycophancy reads are DISTINCT.

    A regression that unified them (both EOR-slot or both answer-span) would
    silently change the sycophancy measurement; the pins are the contract.
    """
    assert READ_LOCATION["marker"] == "post_response_eor_slot"
    assert READ_LOCATION["sycophancy"] == "answer_span_mean"
    assert READ_LOCATION["marker"] != READ_LOCATION["sycophancy"]


class _FakeTok:
    """Minimal chat-template tokenizer stub (deterministic, no model)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<{m['role']}>{m['content']}" for m in messages]
        s = "".join(parts)
        if add_generation_prompt:
            s += "<assistant>"
        return s

    def encode(self, text, add_special_tokens=False):
        # one int per character — a strict, deterministic tokenization so the
        # prefix-is-a-prefix invariant is exactly testable.
        return [ord(c) for c in text]


def test_answer_span_indices_are_post_prefix():
    """The answer span is exactly the completion tokens after the prompt prefix."""
    tok = _FakeTok()
    prompt_msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    full_msgs = [*prompt_msgs, {"role": "assistant", "content": "ANSWER"}]
    full_text = tok.apply_chat_template(full_msgs, add_generation_prompt=False)
    full_ids = tok.encode(full_text)
    prompt_text = tok.apply_chat_template(prompt_msgs, add_generation_prompt=True)
    p = len(tok.encode(prompt_text))
    idx = answer_span_token_indices(tok, prompt_msgs, full_ids)
    assert idx == list(range(p, len(full_ids)))
    assert idx, "answer span must be non-empty"


def test_answer_span_prefix_drift_raises():
    """A prompt that is NOT a strict prefix of the full row FAILS LOUD."""
    tok = _FakeTok()
    prompt_msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    # a full_ids that does not start with the prompt prefix → drift.
    bogus_full = tok.encode("<system>DIFFERENT<user>q<assistant>ANSWER")
    with pytest.raises(RuntimeError, match="strict prefix"):
        answer_span_token_indices(tok, prompt_msgs, bogus_full)


def test_a7_distinguishes_rank1_from_lowrank():
    """The A7 read verdicts rank-1 on a scaled-single-direction stack and
    low-rank on a genuinely 2-D stack (the gating control)."""
    import torch
    from issue683_a7_precondition import a7_read_for_bank

    rng = torch.Generator().manual_seed(0)
    h = 64
    w = torch.randn(h, generator=rng)
    w = w / w.norm()
    u2 = torch.randn(h, generator=rng)
    u2 = u2 - (u2 @ w) * w
    u2 = u2 / u2.norm()

    def _bank(two_d: bool):
        per = {"src": {"v_base": torch.zeros(h), "v_trained": w, "Delta_v": w.clone()}}
        g = {"src": 1.0}
        for i in range(12):
            gi = 0.2 + 0.7 * i / 11
            dv = gi * w + (float(torch.randn(1, generator=rng)) * u2 if two_d else 0.0)
            per[f"c{i}"] = {"v_base": torch.zeros(h), "v_trained": dv, "Delta_v": dv}
            g[f"c{i}"] = realized_gate(w, dv)
        return {"source": "src", "seed": 0, "w_hat": w, "per_context": per, "g_real": g}

    r1 = a7_read_for_bank(_bank(False))
    r2 = a7_read_for_bank(_bank(True))
    assert r1["rank1_holds"] is True, r1
    assert r2["rank1_holds"] is False, r2


def test_shuffled_key_null_is_vector_permutation_real_key_wins():
    """Methodology-critic concern #2: the shuffled-key null permutes a key
    VECTOR (not a matrix-axis relabel), so an informative key scores above it.

    Build a rank-1 bank where c_C correlates with g_real; the real c_C key's
    held-out Spearman must exceed the shuffled-key null mean.
    """
    import torch
    from issue683_key_ablation_score import score_bank

    rng = torch.Generator().manual_seed(1)
    h = 48
    w = torch.randn(h, generator=rng)
    w = w / w.norm()
    contexts = ["src", *[f"c{i}" for i in range(14)]]
    per = {}
    c_bank = {}
    for i, ctx in enumerate(contexts):
        g_true = 1.0 if ctx == "src" else 0.1 + 0.85 * i / (len(contexts) - 1)
        dv = g_true * w + 0.02 * torch.randn(h, generator=rng)
        per[ctx] = {
            "v_base": torch.randn(h, generator=rng),
            "v_trained": dv,
            "Delta_v": dv,
        }
        # c_C correlated with g_true so k=c_C is informative.
        c_bank[ctx] = (g_true * w + 0.25 * torch.randn(h, generator=rng)).numpy().astype(float)
    w_hat = per["src"]["Delta_v"]
    g_real = {ctx: realized_gate(w_hat, per[ctx]["Delta_v"]) for ctx in contexts}
    payload = {"source": "src", "seed": 1, "w_hat": w_hat, "per_context": per, "g_real": g_real}

    res = score_bank(payload=payload, c_bank=c_bank, t_cb=None, a7_rank1=True, n_boot=50, seed=1)
    cc_rows = [r for r in res["leaderboard"] if r["key"] == "k_cC" and r["metric"] == "M_I"]
    assert cc_rows, "k_cC/M_I row missing"
    real_rho = cc_rows[0]["spearman"]
    null_mean = res["null_shuffled_key"]["mean"]
    assert np.isfinite(real_rho)
    assert real_rho > null_mean, (real_rho, null_mean)

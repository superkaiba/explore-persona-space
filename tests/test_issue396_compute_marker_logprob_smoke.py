"""CPU smoke test for :func:`compute_marker_logprob_trajectory` (task #396).

Three assertions:

1. **Single-token assertion fires on multi-token markers.** The trajectory
   primitive should raise loudly when ``marker_text`` tokenizes to more
   than one BPE piece (``[ZLT]`` is the canonical multi-token marker on
   Qwen-2.5; on tiny-gpt2 ``[ZLT]`` also tokenizes to multiple pieces).
   This protects against silent corruption of the per-position trajectory.

2. **End-to-end run on a single-token marker.** For each prompt/completion
   pair the returned inner list has length ``len(comp_tokens) + 1`` (k=0
   bare prior + one entry per completion token).

3. **Mathematical-consistency contract with** :func:`compute_marker_logprob`.
   For a single-token marker, the LAST element of the trajectory equals
   the scalar primitive's return on the same ``prompt + completion``
   context to within ``1e-5``. Both extract
   ``log_softmax(logits[final_pred_position])[marker_id]``, so the
   identity must hold modulo float-32 rounding.

Plan v2.3 §4.2 spec; mathematical-consistency contract spelled out in
§A24. Runs in <30s on CPU with no GPU or HF auth.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.marker_logprob import (
    compute_marker_logprob,
    compute_marker_logprob_trajectory,
)

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    """Module-scoped fixture for the trajectory primitive smoke tests."""
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, torch_dtype=torch.float32)
    model.eval()
    return model, tok


def _pick_single_token_marker(tok) -> tuple[str, int]:
    """Return a marker text whose ``add_special_tokens=False`` tokenization
    yields exactly one BPE piece on ``tok``, and the corresponding token id.

    The trajectory primitive's contract is single-token-only — the test
    can't proceed without one. tiny-gpt2's tokenizer has common short
    space-prefixed words available; we probe a small set rather than
    hardcoding because the exact id can shift between checkpoints.
    """
    candidates = [" the", " a", " is", " of", " in"]
    for text in candidates:
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return text, ids[0]
    raise pytest.skip.Exception("No single-token marker candidate available on tiny-gpt2 tokenizer")


def test_trajectory_single_token_assertion_rejects_multi_token_marker(tiny_model_and_tokenizer):
    """[ZLT] tokenizes to multiple pieces — the trajectory primitive must refuse it.

    Catches the failure mode where a caller hands in a marker that looks
    fine to the scalar primitive (which sums over BPE pieces) but is
    ambiguous for the per-position trajectory. The assertion message
    must mention the offending marker_text and its tokenization so a
    debugging user can see exactly what was rejected.
    """
    model, tok = tiny_model_and_tokenizer
    multi_token_marker = " [ZLT]"
    pieces = tok.encode(multi_token_marker, add_special_tokens=False)
    if len(pieces) == 1:
        pytest.skip(
            f"On this tokenizer {multi_token_marker!r} tokenized to one piece; "
            "cannot exercise the single-token assertion"
        )

    with pytest.raises(AssertionError) as excinfo:
        compute_marker_logprob_trajectory(
            model,
            tok,
            prompts=["A short prompt"],
            completions=[" some completion"],
            marker_text=multi_token_marker,
            batch_size=1,
            device="cpu",
        )
    msg = str(excinfo.value)
    assert "single-token" in msg, (
        f"assertion message should mention 'single-token' contract; got: {msg!r}"
    )
    assert repr(multi_token_marker) in msg or multi_token_marker in msg, (
        f"assertion message should quote the rejected marker_text; got: {msg!r}"
    )


def test_trajectory_returns_one_entry_per_position_plus_k0(tiny_model_and_tokenizer):
    """Inner list at index i has length len(comp_tokens[i]) + 1.

    k=0 is the bare-prior entry (prompt only, before any completion token);
    k=1..len(comp_tokens) is one entry per completion position. So the
    full trajectory length is len(comp_tokens) + 1.
    """
    model, tok = tiny_model_and_tokenizer
    marker_text, _ = _pick_single_token_marker(tok)
    prompts = ["First prompt here", "Second one is longer than the first"]
    completions = [" alpha beta gamma", " a b c d e"]

    trajectories = compute_marker_logprob_trajectory(
        model,
        tok,
        prompts=prompts,
        completions=completions,
        marker_text=marker_text,
        batch_size=2,
        device="cpu",
    )

    assert len(trajectories) == len(prompts), (
        f"expected {len(prompts)} trajectories, got {len(trajectories)}"
    )
    for i, (comp, traj) in enumerate(zip(completions, trajectories, strict=True)):
        comp_len = len(tok.encode(comp, add_special_tokens=False))
        expected_len = comp_len + 1  # k=0 + one entry per completion token
        assert len(traj) == expected_len, (
            f"trajectory[{i}]: expected length {expected_len} "
            f"(comp_len={comp_len} + 1 bare-prior entry), got {len(traj)}"
        )
        for k, v in enumerate(traj):
            assert math.isfinite(v), f"trajectory[{i}][{k}] = {v!r} is not finite"


def test_trajectory_last_position_matches_scalar_primitive(tiny_model_and_tokenizer):
    """traj[-1] == compute_marker_logprob([prompt + completion])[0] within 1e-5.

    The mathematical-consistency contract that pairs the two primitives.
    Both extract log_softmax(logits[final_pred_position])[marker_id] —
    the scalar primitive does so after appending the marker and reading
    [-marker_len-1:-1] (with left-pad), the trajectory primitive does so
    without appending and reads at position len(prompt + completion) - 1
    (with right-pad). For single-token markers the two formulations
    converge on the same softmax decision and must agree modulo
    float-32 rounding.

    Each row is forwarded as its own ``batch_size=1`` sub-batch in both
    primitives so neither side pads — the comparison checks the per-row
    invariant rather than a cross-batch one. Left-padding the scalar
    primitive (or right-padding the trajectory primitive) in a mixed-
    length batch interacts with the model's absolute position embeddings
    and shifts the value for shorter rows; that's a property of how each
    primitive batches and is documented behaviour, not a contract
    violation. Callers downstream (eval_issue396_logprob.py) already pair
    one trajectory call with one scalar call per row when reconciling
    the two surfaces, so this batch_size=1 form is the canonical
    reference.
    """
    model, tok = tiny_model_and_tokenizer
    marker_text, _ = _pick_single_token_marker(tok)

    prompts = ["A prompt about something", "Different short context"]
    completions = [" some completion text here", " brief"]

    for i, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
        traj = compute_marker_logprob_trajectory(
            model,
            tok,
            prompts=[prompt],
            completions=[completion],
            marker_text=marker_text,
            batch_size=1,
            device="cpu",
        )[0]
        scalar = compute_marker_logprob(
            model,
            tok,
            contexts=[prompt + completion],
            marker_text=marker_text,
            batch_size=1,
            device="cpu",
        )[0]
        traj_end = traj[-1]
        diff = traj_end - scalar
        assert abs(diff) < 1e-5, (
            f"row {i}: trajectory[-1]={traj_end!r} vs scalar primitive={scalar!r}; "
            f"diff={diff!r} exceeds 1e-5 tolerance — the mathematical-consistency "
            "contract between compute_marker_logprob and "
            "compute_marker_logprob_trajectory is broken"
        )


def test_trajectory_k0_matches_inline_reference(tiny_model_and_tokenizer):
    """traj[0] equals an inline log_softmax(logits[len(prompt)-1, marker_id]).

    Hand-construct the bare-prior entry (k=0) outside the primitive and
    check the values match. Catches off-by-one bugs in the
    ``base = len(pids) - 1`` slice math; an off-by-one here would silently
    skew the secondary "logp_at_k0" trajectory feature.
    """
    model, tok = tiny_model_and_tokenizer
    marker_text, marker_id = _pick_single_token_marker(tok)
    prompt = "Reference prompt context"
    completion = " brief completion"

    # Inline reference: forward the prompt alone, read off the marker
    # log-prob at the LAST position (predicting the next token after
    # the prompt). This is what k=0 represents.
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([prompt_ids])).logits[0]
    expected_k0 = float(F.log_softmax(logits[-1].float(), dim=-1)[marker_id].item())

    traj = compute_marker_logprob_trajectory(
        model,
        tok,
        prompts=[prompt],
        completions=[completion],
        marker_text=marker_text,
        batch_size=1,
        device="cpu",
    )[0]
    diff = traj[0] - expected_k0
    assert abs(diff) < 1e-5, (
        f"trajectory[0] (k=0 bare prior) = {traj[0]!r}; "
        f"inline reference {expected_k0!r}; diff {diff!r}"
    )

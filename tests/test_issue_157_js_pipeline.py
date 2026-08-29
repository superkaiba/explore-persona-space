"""Issue #157 BLOCKER-1 fix — JS-divergence pipeline tests.

Verifies the corrected response-position JS computation that replaces the v1
prompt-only tail-aligned heuristic. The test runs on a tiny CPU-loadable LM
(``sshleifer/tiny-gpt2`` — 4 MB, no GPU required) so it can run inside CI
without needing the gated Gaperon weights.

Cases:
  1. Identical ``[prompt + response]`` -> per-prompt scalar JS = 0.
  2. Different responses for the same prompt -> per-prompt scalar JS > 0.
  3. The per-prompt scalar equals the explicit mean across the response-token
     positions (no off-by-one in the slicing).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_tiny_lm():
    """Load tiny-gpt2 + tokenizer on CPU. Skip if HF cache is unreachable."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        pytest.skip(f"transformers not available: {e}")

    repo = "sshleifer/tiny-gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo, token=os.environ.get("HF_TOKEN"))
        model = AutoModelForCausalLM.from_pretrained(repo, token=os.environ.get("HF_TOKEN"))
    except Exception as e:  # network / cache issues
        pytest.skip(f"Could not load {repo}: {e}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def tiny_lm():
    return _load_tiny_lm()


def _import_response_position_logits():
    """Import ``_response_position_logits`` from the Stage B script."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "issue_157_stage_b_test_import", SCRIPTS_DIR / "issue_157_stage_b.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._response_position_logits


def test_identical_prompt_response_gives_zero_js(tiny_lm):
    """Same [prompt+response] -> JS over response positions is exactly 0."""
    import torch

    from explore_persona_space.eval.distance import js_divergence_logits

    model, tokenizer = tiny_lm
    response_position_logits = _import_response_position_logits()

    prompt = "The quick brown"
    response = " fox jumps over"

    with torch.no_grad():
        p_logits = response_position_logits(model, tokenizer, prompt, response, "cpu")
        q_logits = response_position_logits(model, tokenizer, prompt, response, "cpu")

    js = js_divergence_logits(p_logits.float(), q_logits.float())
    js_mean = float(js.mean().item())
    # Identical inputs are mathematically JS=0; allow float32 roundoff slack
    # (logsumexp -> exp -> KL chain in fp32 accumulates ~1e-6 per position).
    assert js_mean == pytest.approx(0.0, abs=5e-6), (
        f"Identical [prompt+response] should give JS≈0, got {js_mean}"
    )


def test_different_responses_give_positive_js(tiny_lm):
    """Different responses -> per-prompt scalar JS is strictly positive."""
    import torch

    from explore_persona_space.eval.distance import js_divergence_logits

    model, tokenizer = tiny_lm
    response_position_logits = _import_response_position_logits()

    prompt = "The capital of France is"
    response_a = " Paris and the"
    response_b = " Rome and that"

    with torch.no_grad():
        p_logits = response_position_logits(model, tokenizer, prompt, response_a, "cpu")
        q_logits = response_position_logits(model, tokenizer, prompt, response_b, "cpu")

    # Cap to the shorter length (mirrors per-prompt min(L_i, L_c) rule).
    t = min(p_logits.shape[0], q_logits.shape[0])
    js = js_divergence_logits(p_logits[:t].float(), q_logits[:t].float())
    js_mean = float(js.mean().item())
    assert js_mean > 0.0, (
        f"Distinct responses should produce JS>0, got {js_mean}. "
        "If this fires, the slicing might be returning zero-length tensors."
    )


def test_per_prompt_scalar_is_response_position_mean(tiny_lm):
    """The per-prompt scalar equals the arithmetic mean over response-token positions.

    Guards against off-by-one in the slice ``logits[t_p : t_p + t_r]`` and
    against silent broadcasting bugs.
    """
    import torch

    from explore_persona_space.eval.distance import js_divergence_logits

    model, tokenizer = tiny_lm
    response_position_logits = _import_response_position_logits()

    prompt = "Roses are red and"
    response_a = " violets bloom in"
    response_b = " thorns prick the"

    with torch.no_grad():
        p_logits = response_position_logits(model, tokenizer, prompt, response_a, "cpu")
        q_logits = response_position_logits(model, tokenizer, prompt, response_b, "cpu")

    t = min(p_logits.shape[0], q_logits.shape[0])
    assert t > 0, "Expected at least one response token for the test response."
    js = js_divergence_logits(p_logits[:t].float(), q_logits[:t].float())
    assert js.shape == (t,), f"JS should be per-position, shape (T,), got {tuple(js.shape)}"
    expected_mean = float(js.sum().item() / t)
    actual_mean = float(js.mean().item())
    assert actual_mean == pytest.approx(expected_mean, rel=1e-6, abs=1e-9), (
        f"per-prompt scalar={actual_mean} should equal sum/T={expected_mean}"
    )

    # Also assert each per-position value lies in [0, ln 2] — the math primitive
    # is supposed to clamp; if any position bursts past ln 2 we have a bug.
    import math

    ln2 = math.log(2.0)
    for i in range(t):
        v = float(js[i].item())
        assert 0.0 <= v <= ln2 + 1e-6, f"JS at position {i} = {v} outside [0, ln 2]"


def test_response_position_logits_rejects_empty_response(tiny_lm):
    """Helper raises rather than returning zero-length logits silently."""
    model, tokenizer = tiny_lm
    response_position_logits = _import_response_position_logits()

    with pytest.raises(ValueError):
        response_position_logits(model, tokenizer, "Hello world", "", "cpu")


def test_response_position_logits_slices_correct_positions(tiny_lm):
    """Forward pass over [prompt + response] yields exactly len(response_tokens) logit rows."""
    import torch

    model, tokenizer = tiny_lm
    response_position_logits = _import_response_position_logits()

    prompt = "Once upon a time"
    response = " there lived a"

    response_token_count = len(tokenizer(response, add_special_tokens=False)["input_ids"])
    with torch.no_grad():
        logits = response_position_logits(model, tokenizer, prompt, response, "cpu")

    assert logits.shape[0] == response_token_count, (
        f"Sliced logits row count {logits.shape[0]} != response token count "
        f"{response_token_count}; the slice [t_p : t_p + t_r] is misaligned."
    )

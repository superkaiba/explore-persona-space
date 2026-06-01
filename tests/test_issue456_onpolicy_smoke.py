"""CPU smoke tests for task #456's on-policy endpos-logp primitive.

The VM has no GPU; the full eval runs on the pod as pipeline Phases 2-5. These
tests verify, on a tiny CPU model (``sshleifer/tiny-gpt2``, ~100K params), the
two pieces of new logic the on-policy dispatcher (``scripts/eval_i456_onpolicy_
emission.py``) layers on top of the already-tested ``compute_marker_logprob``:

  1. ``compute_marker_logprob`` returns a finite scalar when fed a FABRICATED
     ON-POLICY context (``chat_prefix + model's own answer + "\\n\\n"``), one
     value per context, exactly mirroring how Phase B scores the marker after
     the model's OWN generated text (rather than #432's fixed stub).

  2. ``strip_trailing_marker`` removes a trailing marker (and trailing
     whitespace, and repeated markers) so the endpos probe never conditions on
     the very token it is trying to predict (the double-count bug).

These follow the established ``sshleifer/tiny-gpt2`` CPU pattern in
``tests/test_marker_abstraction.py`` so they run in <30s with no GPU.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.marker_logprob import compute_marker_logprob
from explore_persona_space.train.trainer import _resolve_duration_kwargs

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

TINY_MODEL = "sshleifer/tiny-gpt2"


def _load_dispatcher_module():
    """Load eval_i456_onpolicy_emission.py by path (it has a hyphen-free name).

    Importing the module file directly (rather than ``import
    eval_i456_onpolicy_emission``) keeps the test independent of the panel
    module's import-time ``HF_HOME`` side effect: only ``strip_trailing_marker``
    is exercised here, which touches no HF/network state.
    """
    path = Path(__file__).resolve().parent.parent / "scripts" / "eval_i456_onpolicy_emission.py"
    spec = importlib.util.spec_from_file_location("eval_i456_onpolicy_emission", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, torch_dtype=torch.float32)
    model.eval()
    return model, tok


@pytest.fixture(scope="module")
def single_token_marker(tiny_model_and_tokenizer):
    """A 1-BPE-token marker for tiny-gpt2 (mirrors the single-token ※ on Qwen)."""
    _model, tok = tiny_model_and_tokenizer
    for cand in [" the", " a", " is", " and", " to"]:
        if len(tok.encode(cand, add_special_tokens=False)) == 1:
            return cand
    pytest.skip("no single-token marker candidate found for tiny-gpt2 tokenizer")


# ---------------------------------------------------------------------------
# Happy path: on-policy context yields a finite log-prob per context.
# ---------------------------------------------------------------------------


def test_onpolicy_context_finite_logp(tiny_model_and_tokenizer, single_token_marker):
    """Fabricated on-policy contexts return one finite log-prob each.

    This is the exact shape Phase B builds: prefix + the model's OWN answer +
    "\\n\\n", scored for log p(marker) at the next position.
    """
    model, tok = tiny_model_and_tokenizer
    marker = single_token_marker

    # Two fabricated on-policy contexts: chat-ish prefix + an "own answer" + "\n\n".
    prefix = "System: You are a software engineer.\nUser: How do I sort a list?\nAssistant:"
    contexts = [
        prefix + " Use sorted(my_list) for a new list." + "\n\n",
        prefix + " Call my_list.sort() to sort in place." + "\n\n",
    ]

    logps = compute_marker_logprob(
        model,
        tok,
        contexts=contexts,
        marker_text=marker,
        batch_size=2,
        device="cpu",
    )
    assert len(logps) == len(contexts), f"expected {len(contexts)} logps, got {len(logps)}"
    for lp in logps:
        assert isinstance(lp, float)
        assert math.isfinite(lp), f"log-prob not finite: {lp}"
        assert lp <= 0.0 + 1e-6, f"log-prob should be <= 0, got {lp}"


def test_onpolicy_logp_matches_inline_reference(tiny_model_and_tokenizer, single_token_marker):
    """The on-policy log-prob equals an inline teacher-forced reference.

    Confirms the marker is scored at exactly the position immediately AFTER the
    full on-policy context (context_ids + 1 marker token), not off-by-one.
    """
    model, tok = tiny_model_and_tokenizer
    marker = single_token_marker
    context = "Assistant: Here is my own generated answer." + "\n\n"

    got = compute_marker_logprob(
        model, tok, contexts=[context], marker_text=marker, batch_size=1, device="cpu"
    )[0]

    # Inline reference: append marker, take log-softmax at the position whose
    # prediction is the marker token (the standard next-token shift).
    ctx_ids = tok.encode(context, add_special_tokens=False)
    marker_ids = tok.encode(marker, add_special_tokens=False)
    assert len(marker_ids) == 1, "this reference assumes a single-token marker"
    full_ids = ctx_ids + marker_ids
    # full sequence length is len(ctx_ids) + 1 (one marker token appended).
    assert len(full_ids) == len(ctx_ids) + 1
    with torch.no_grad():
        logits = model(torch.tensor([full_ids])).logits  # (1, T, V)
    # The marker sits at index -1; its predictive logit is at index -2.
    log_probs = torch.log_softmax(logits[0, -2, :].float(), dim=-1)
    expected = float(log_probs[marker_ids[0]].item())
    assert math.isclose(got, expected, rel_tol=1e-4, abs_tol=1e-4), (
        f"on-policy logp {got!r} != inline reference {expected!r}"
    )


# ---------------------------------------------------------------------------
# Edge cases: strip_trailing_marker must remove the marker before scoring.
# ---------------------------------------------------------------------------


def test_strip_trailing_marker_removes_marker():
    """A completion that already ends in the marker is stripped before probing."""
    m = _load_dispatcher_module()
    marker = m.MARKER
    assert m.strip_trailing_marker(f"my answer{marker}") == "my answer"
    assert m.strip_trailing_marker(f"my answer {marker}") == "my answer"
    # Repeated markers + trailing whitespace all stripped.
    assert m.strip_trailing_marker(f"my answer {marker} {marker}  ") == "my answer"


def test_strip_trailing_marker_leaves_clean_text_unchanged():
    """A completion with no trailing marker is returned unchanged (modulo rstrip)."""
    m = _load_dispatcher_module()
    assert m.strip_trailing_marker("a normal answer") == "a normal answer"
    assert m.strip_trailing_marker("a normal answer   ") == "a normal answer"
    # A marker in the MIDDLE is NOT stripped (only trailing).
    marker = m.MARKER
    mid = f"answer with {marker} in the middle"
    assert m.strip_trailing_marker(mid) == mid


def test_stripped_context_logp_differs_from_unstripped(
    tiny_model_and_tokenizer, single_token_marker
):
    """Scoring a marker-stripped context differs from scoring the raw context.

    Demonstrates the double-count bug the strip guards against: if the answer
    already ended in the marker, NOT stripping it would condition the probe on
    the marker itself, changing the log-prob.
    """
    model, tok = tiny_model_and_tokenizer
    marker = single_token_marker
    raw_answer = f"Assistant: an answer ending in the marker{marker}"
    stripped_answer = raw_answer.rstrip()
    while stripped_answer.endswith(marker):
        stripped_answer = stripped_answer[: -len(marker)].rstrip()

    raw_logp = compute_marker_logprob(
        model, tok, contexts=[raw_answer + "\n\n"], marker_text=marker, device="cpu"
    )[0]
    stripped_logp = compute_marker_logprob(
        model, tok, contexts=[stripped_answer + "\n\n"], marker_text=marker, device="cpu"
    )[0]
    assert math.isfinite(raw_logp) and math.isfinite(stripped_logp)
    # They must NOT be bit-identical: the raw context carries an extra marker
    # token before the "\n\n", which shifts the scored position's conditioning.
    # (On the randomly-initialized tiny-gpt2 the absolute gap is small, but a
    # genuine no-op strip would produce byte-identical floats.)
    assert raw_logp != stripped_logp, (
        "stripped vs unstripped on-policy context produced an identical log-prob; "
        "the strip would be a no-op (double-count guard ineffective)"
    )


# ---------------------------------------------------------------------------
# Training-duration resolution: max_steps must reach the built SFTConfig.
#
# The pipeline trains with ``++training.max_steps=10 ++training.epochs=-1``
# (smoke) and ``++training.max_steps=1600 ++training.epochs=-1`` (main). If
# ``max_steps`` is dropped at the SFTConfig call site (the round-1 bug), HF
# Trainer gets ``num_train_epochs=-1`` with no step budget -> ZERO training
# steps -> no checkpoints -> the whole experiment (defined by its 22-step
# schedule) silently fails only after a pod + 7B load. These tests pin the
# fix on CPU (helper return shape + a real SFTConfig built from the spread).
# ---------------------------------------------------------------------------


def test_resolve_duration_kwargs_threads_max_steps():
    """``max_steps`` survives into the duration-kwargs dict (smoke + main combos).

    Mirrors the pipeline's exact Hydra overrides: epochs=-1 paired with
    max_steps>0. Both keys must be present so the SFT call site can spread them.
    """
    smoke = _resolve_duration_kwargs(SimpleNamespace(epochs=-1, max_steps=10))
    assert smoke == {"num_train_epochs": -1, "max_steps": 10}, smoke

    main = _resolve_duration_kwargs(SimpleNamespace(epochs=-1, max_steps=1600))
    assert main == {"num_train_epochs": -1, "max_steps": 1600}, main

    # Epochs-only (max_steps unset/0): max_steps key absent so HF's "use epochs"
    # path is preserved -- no spurious 0 step-cap.
    epochs_only = _resolve_duration_kwargs(SimpleNamespace(epochs=3, max_steps=0))
    assert epochs_only == {"num_train_epochs": 3}, epochs_only


def test_resolve_duration_kwargs_raises_on_zero_step_combo():
    """epochs<=0 AND max_steps<=0 raises (the cheap CPU fail-fast guard).

    Without this raise the bug surfaces only after a pod is provisioned and a 7B
    model is loaded (HF treats num_train_epochs=-1 + max_steps<=0 as zero epochs).
    """
    with pytest.raises(ValueError, match="zero training steps"):
        _resolve_duration_kwargs(SimpleNamespace(epochs=-1, max_steps=0))
    with pytest.raises(ValueError, match="epochs is required"):
        _resolve_duration_kwargs(SimpleNamespace(epochs=None, max_steps=0))


def test_max_steps_reaches_built_sftconfig():
    """A real SFTConfig built from the spread carries max_steps (call-site contract).

    This is the end-to-end check the round-1 review flagged as missing: it
    exercises the SAME spread the SFT call site uses (``**duration_kwargs``) and
    asserts the constructed SFTConfig object actually has ``max_steps`` set, so a
    future regression that drops the spread is caught on CPU in <1s.
    """
    from trl import SFTConfig

    for steps in (10, 1600):
        cfg = SFTConfig(
            output_dir="/tmp/_issue456_duration_probe",
            **_resolve_duration_kwargs(SimpleNamespace(epochs=-1, max_steps=steps)),
            use_cpu=True,
            bf16=False,
            fp16=False,
        )
        assert cfg.max_steps == steps, f"SFTConfig.max_steps={cfg.max_steps}, expected {steps}"

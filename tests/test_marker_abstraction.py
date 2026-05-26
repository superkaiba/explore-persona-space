"""CPU-only regression tests for the marker-token abstraction (task #401).

Covers §5.1 of the plan: explicit-value primitive tests for
:func:`compute_marker_logprob` (catches off-by-one indexing bugs that a
shape-only assertion would miss) plus the snapshot/restore behaviour of
:func:`measure_first_step_delta` (catches no-op-step bugs and the
update-then-restore class via ``_assert_frozen_during_step``).

All tests use ``sshleifer/tiny-gpt2`` (~100K params) so they run in <30s on
CPU with no GPU or HF auth.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.marker_logprob import (
    compute_marker_logprob,
    measure_first_step_delta,
)
from explore_persona_space.personas import MARKER_TOKEN

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    """Module-scoped fixture for the read-only compute_marker_logprob tests."""
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, torch_dtype=torch.float32)
    model.eval()
    return model, tok


@pytest.fixture()
def fresh_tiny_model_and_tokenizer():
    """Function-scoped fixture for tests that attach a LoRA adapter.

    measure_first_step_delta does its own snapshot+restore, but if PEFT
    leaves any latent state on the wrapped model between invocations we'd
    rather give each test a fresh model object than chase flakes.
    """
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, torch_dtype=torch.float32)
    model.eval()
    return model, tok


def test_marker_token_module_constant():
    """personas.MARKER_TOKEN exists and is the legacy default."""
    assert MARKER_TOKEN == "[ZLT]"


def test_compute_marker_logprob_explicit_value(tiny_model_and_tokenizer):
    """Primitive's return must equal an independent inline reference computation.

    This is the load-bearing correctness test. Instead of asserting only
    finiteness (which passes for off-by-one slice bugs like
    ``logits[-marker_len:]`` or ``logits[-marker_len-2:-2]``), we hand-
    construct a short context, compute the expected joint log-prob inline
    (Python loop over the appended marker positions), and assert equality
    to the primitive's return within ``1e-5``.
    """
    model, tok = tiny_model_and_tokenizer
    context = "Hello world short test"
    marker_text = " [ZLT]"

    # Reference: build full_ids = context_ids + marker_ids, forward, walk
    # the marker positions one by one with explicit indexing.
    context_ids = tok.encode(context, add_special_tokens=False)
    marker_ids = tok.encode(marker_text, add_special_tokens=False)
    assert len(marker_ids) >= 1, "Marker tokenized to empty; test cannot proceed"
    full_ids = torch.tensor([context_ids + marker_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=full_ids).logits[0]  # (T, V)
    # Marker piece k is at position ``len(context_ids) + k``; its predictive
    # logits are at the previous position (standard next-token shift).
    expected = 0.0
    for k, target_id in enumerate(marker_ids):
        pred_pos = len(context_ids) + k - 1
        log_probs = F.log_softmax(logits[pred_pos].float(), dim=-1)
        expected += float(log_probs[target_id].item())

    got = compute_marker_logprob(
        model,
        tok,
        contexts=[context],
        marker_text=marker_text,
        batch_size=1,
        device="cpu",
    )
    assert len(got) == 1
    diff = got[0] - expected
    assert abs(diff) < 1e-5, (
        f"compute_marker_logprob returned {got[0]!r}; inline reference {expected!r}; diff {diff!r}"
    )


def test_compute_marker_logprob_single_token_additive(tiny_model_and_tokenizer):
    """Single-token marker case: primitive must equal the single softmax decision.

    Hand-construct a 1-token marker (using a literal vocab token), compute
    ``log_softmax(model(context).logits[0, -1, :])[marker_id]`` directly,
    and assert the primitive's scalar equals that value within ``1e-5``.
    Catches batching / left-pad / position-index bugs that the multi-token
    test could conceivably pass through.
    """
    model, tok = tiny_model_and_tokenizer
    context = "A short test context"
    # Find a single-BPE-token marker for tiny-gpt2's tokenizer.
    candidates = [" the", " a", " is", " of"]
    marker_text: str | None = None
    marker_id: int | None = None
    for c in candidates:
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) == 1:
            marker_text = c
            marker_id = ids[0]
            break
    assert marker_text is not None and marker_id is not None, (
        "No single-token marker candidate found for tiny-gpt2 tokenizer"
    )

    context_ids = tok.encode(context, add_special_tokens=False)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([context_ids])).logits[0]
    expected = float(F.log_softmax(logits[-1].float(), dim=-1)[marker_id].item())

    got = compute_marker_logprob(
        model,
        tok,
        contexts=[context],
        marker_text=marker_text,
        batch_size=1,
        device="cpu",
    )
    diff = got[0] - expected
    assert abs(diff) < 1e-5, (
        f"single-token primitive returned {got[0]!r}; inline reference {expected!r}; diff {diff!r}"
    )


def test_compute_marker_logprob_both_markers_run(tiny_model_and_tokenizer):
    """Pipeline runs end-to-end for both [ZLT] and ※ across batched contexts."""
    model, tok = tiny_model_and_tokenizer
    contexts = ["First context", "Second one", "Third"]
    for marker in [" [ZLT]", " ※"]:
        out = compute_marker_logprob(
            model,
            tok,
            contexts=contexts,
            marker_text=marker,
            batch_size=2,
            device="cpu",
        )
        assert len(out) == 3
        for v in out:
            assert math.isfinite(v), f"non-finite logp {v!r} for marker {marker!r}"


def test_measure_first_step_delta_runs(fresh_tiny_model_and_tokenizer):
    """End-to-end one-step LoRA call returns expected shape AND a non-zero delta.

    The shape-only assertion does not catch no-op-optimizer-step bugs: a
    frozen adapter, a fully-masked loss, or a wrong-loss-target pipeline
    would all return shape-correct ``delta_logp`` of all zeros. We
    additionally assert that AT LEAST ONE of the deltas has magnitude
    above ``1e-6``. On randomly-initialized tiny-gpt2 with lr=1e-3, a
    single AdamW step on a 2-row batch produces gradient signal well
    above that floor.
    """
    model, tok = fresh_tiny_model_and_tokenizer
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )
    training_rows = [
        {"persona": "test_persona", "question": "Q1?", "answer": "A1 [ZLT]"},
        {"persona": "test_persona", "question": "Q2?", "answer": "A2 [ZLT]"},
    ]
    eval_questions = ["What is 2+2?", "Define entropy."]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = measure_first_step_delta(
            base_model=model,
            tokenizer=tok,
            persona_system_prompt="You are a test persona.",
            training_rows=training_rows,
            eval_questions=eval_questions,
            marker_text=" [ZLT]",
            lora_config=lora_config,
            lr=1e-3,
            device="cpu",
        )
    assert set(result.keys()) >= {"persona", "pre_logp", "post_logp", "delta_logp"}
    assert len(result["pre_logp"]) == len(eval_questions)
    assert len(result["post_logp"]) == len(eval_questions)
    for d, pre, post in zip(
        result["delta_logp"], result["pre_logp"], result["post_logp"], strict=True
    ):
        assert abs(d - (post - pre)) < 1e-5, (
            f"delta_logp {d!r} != post-pre = {post - pre!r} (pre={pre!r}, post={post!r})"
        )

    # CORRECTNESS: at least one delta must be non-trivially non-zero. A
    # no-op optimizer step (zero grad, wrong loss masking, frozen adapter)
    # would fail this — the shape check alone would not.
    assert any(abs(d) > 1e-6 for d in result["delta_logp"]), (
        f"All deltas <= 1e-6: {result['delta_logp']!r}. "
        f"Likely no-op optimizer step — investigate adapter attachment / loss masking."
    )


def test_base_weights_restored_after_first_step(fresh_tiny_model_and_tokenizer):
    """measure_first_step_delta must NOT mutate the base model — even mid-step.

    End-state byte-identity is bypassed by an update-then-restore bug
    pattern: the optimizer DOES write to base params during the step,
    then a ``state_dict`` restore at the end masks it. The end-state
    check passes but the intermediate write was still wrong. We pass
    ``_assert_frozen_during_step=True`` which makes the function sample
    one LoRA-target base param BEFORE the optimizer step and assert byte
    identity to its pre-step value AFTER the step (still inside the
    function, before any state_dict restoration). If that intermediate
    check fires, the function raises ``AssertionError`` and this test
    catches it; otherwise this test ALSO confirms end-state byte
    identity at the function call site.
    """
    model, tok = fresh_tiny_model_and_tokenizer
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )
    # Snapshot a target tensor BEFORE the call.
    before = model.transformer.h[0].attn.c_attn.weight.detach().clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        _ = measure_first_step_delta(
            base_model=model,
            tokenizer=tok,
            persona_system_prompt="You are a test.",
            training_rows=[{"persona": "p", "question": "Q?", "answer": "A [ZLT]"}],
            eval_questions=["Q?"],
            marker_text=" [ZLT]",
            lora_config=lora_config,
            lr=1e-3,
            device="cpu",
            _assert_frozen_during_step=True,
        )
    after = model.transformer.h[0].attn.c_attn.weight.detach()
    assert torch.allclose(before, after, atol=0.0), "Base weights mutated end-to-end"


# ─── §5.5 round-2 tests (acceptance criteria #13 / #14 / #16) ────────────────


def test_generate_leakage_data_marker_swap(tmp_path, monkeypatch):
    """Plan §5.5 / acceptance #13 — non-default marker writes ONLY the slug
    path (NO legacy alias) and the JSONL contains the new marker, not [ZLT].

    Reproduces the round-1 silent-degradation trap: passing
    ``marker_text="※"`` to ``assemble_marker_data`` must produce JSONL rows
    that contain ``※`` and NOT ``[ZLT]``. The legacy un-suffixed file MUST
    NOT be aliased when the marker differs from MARKER_TOKEN.
    """
    import json

    import scripts.generate_leakage_data as glm

    monkeypatch.setattr(glm, "DATA_DIR", tmp_path)
    questions = ["What is 1+1?", "Capital of France?"]
    responses = {f"generic__{i:04d}": f"Test response {i}" for i in range(len(questions))}

    glm.assemble_marker_data(
        source="librarian",
        questions=questions,
        generic_responses=responses,
        neg_set="asst_excluded",
        marker_text="※",
    )

    # Slug path must exist; parse JSONL and check completions for ※.
    # The JSON encoder escapes non-ASCII as \uXXXX, so raw-text substring
    # match on the JSONL bytes would miss the literal '※' character. Parse
    # each row and inspect the decoded completion content directly.
    slug_path = next(tmp_path.glob("marker_librarian_asst_excluded_medium_*.jsonl"))
    parsed = [json.loads(line) for line in slug_path.read_text().splitlines() if line.strip()]
    found_marker = False
    for record in parsed:
        for msg in record.get("completion", []):
            if "※" in msg.get("content", ""):
                found_marker = True
            assert "[ZLT]" not in msg.get("content", ""), (
                f"Unexpected '[ZLT]' in {slug_path} despite marker swap"
            )
    assert found_marker, (
        f"Expected '※' in at least one completion of {slug_path}; parsed {len(parsed)} rows"
    )

    # Legacy un-suffixed path MUST NOT exist for non-default markers.
    legacy = tmp_path / "marker_librarian_asst_excluded_medium.jsonl"
    assert not legacy.exists(), (
        f"Non-default marker must NOT alias to legacy un-suffixed path; "
        f"found legacy file at {legacy}"
    )


def test_generate_leakage_data_legacy_hardlink_byte_identity(tmp_path, monkeypatch):
    """Plan §5.5 / acceptance #14 — default marker writes both the slug path
    and the legacy un-suffixed path as hardlinks (same inode) OR as
    byte-identical files (copy fallback on non-POSIX FS).
    """
    import os

    import scripts.generate_leakage_data as glm
    from explore_persona_space.personas import MARKER_TOKEN

    monkeypatch.setattr(glm, "DATA_DIR", tmp_path)
    questions = ["What is 1+1?", "Capital of France?"]
    responses = {f"generic__{i:04d}": f"Test response {i}" for i in range(len(questions))}

    glm.assemble_marker_data(
        source="librarian",
        questions=questions,
        generic_responses=responses,
        neg_set="asst_excluded",
        marker_text=MARKER_TOKEN,
    )

    legacy = tmp_path / "marker_librarian_asst_excluded_medium.jsonl"
    suffixed = tmp_path / "marker_librarian_asst_excluded_medium_zlt.jsonl"
    assert legacy.exists(), f"Expected legacy path {legacy}; ls: {list(tmp_path.iterdir())}"
    assert suffixed.exists(), f"Expected suffixed path {suffixed}; ls: {list(tmp_path.iterdir())}"

    # Byte-identity invariant: either same inode (hardlink) or
    # byte-identical content (copy fallback). Both satisfy the contract.
    legacy_stat = os.stat(legacy)
    suffixed_stat = os.stat(suffixed)
    if legacy_stat.st_ino == suffixed_stat.st_ino:
        # POSIX hardlink path — strongest guarantee.
        return
    # Copy-fallback path: content must still be byte-identical.
    assert legacy.read_bytes() == suffixed.read_bytes(), (
        f"Legacy ({legacy}) and suffixed ({suffixed}) paths must be "
        f"byte-identical when hardlink isn't available"
    )


def test_hydra_condition_marker_token_resolves():
    """Plan §5.5 / acceptance #16 — c_issue377_marker_install Hydra condition
    resolves cleanly and exposes ``marker_token: '[ZLT]'`` in the merged config.
    """
    import subprocess

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/train.py",
            "condition=c_issue377_marker_install",
            "--cfg",
            "job",
            "--resolve",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Hydra resolve failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )
    assert "marker_token: '[ZLT]'" in result.stdout, (
        f"marker_token field missing from resolved condition config; got: {result.stdout[:1000]}"
    )

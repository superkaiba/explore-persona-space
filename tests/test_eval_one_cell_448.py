# ruff: noqa: RUF002, RUF003
"""Tests for task #448 v5 Phase 4 per-cell eval rig.

Round-2 blocker C1 (Claude + Codex code-review v1): training builds the
marker IN-TURN as ``f"{R}\\n\\n{MARKER_TEXT}"`` rendered inside an
assistant turn via ``apply_chat_template``. Eval must produce the SAME
token-id sequence at the marker slot — otherwise log P(marker | context)
is teacher-forced at a position the trained model was never optimized on.

The round-1 implementation did ``tokenizer.encode(prompt_text + R_text +
marker_text)`` which dropped the ``\\n\\n`` separator AND ignored the BPE
re-merge at the R/sep boundary (e.g. ``'.'`` + ``'\\n\\n'`` fuses to id
382 in Qwen-2.5-7B-Instruct's tokenizer). This test asserts the round-2
fix: eval and train produce IDENTICAL tokens around the marker slot.

Round-2 blocker C5 (Claude code-review v1): the round-1 test was stale —
it monkeypatched the deleted ``compute_marker_logprob`` HF-Transformers
path with ``AttributeError`` on import. Rewritten here for the vLLM
``prompt_logprobs=1`` rig — uses a fake vLLM ``RequestOutput`` so we
don't pull Qwen-7B onto CI.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("EPM_ISSUE_448_SKIP_REGISTRY_BUILD", "1")


# ── Token-equality contract (the C1 fix). ─────────────────────────────────────


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Real Qwen-2.5-7B-Instruct tokenizer for the token-equality check."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    except Exception as e:  # pragma: no cover - HF Hub flake / no token
        pytest.skip(f"Qwen tokenizer fetch failed: {e}")


@pytest.mark.parametrize(
    "R_text",
    [
        # Diverse trailing punctuation — exercises BPE re-merge at R/sep
        # boundary across the common shapes the base model generates.
        "Plants grow through photosynthesis.",
        "2+2 equals 4.",
        "It depends on context!",
        "I do not have a strong opinion on this question",
        "Yes — but only if the assumption holds.",  # em-dash + sentence
        "Here are three points:\n1. First.\n2. Second.\n3. Third.",
    ],
)
def test_eval_marker_slot_matches_training_token_ids(qwen_tokenizer, R_text):
    """Round-2 C1 contract: eval's token sequence around the marker MUST
    equal training's token sequence around the marker.

    Training emits ``f"{R}\\n\\n{MARKER_TEXT}"`` inside the assistant turn.
    Eval rebuilds the same string. This test asserts the LAST 3 tokens
    (= 2 preceding + marker) of the eval encoding match training's
    encoding byte-for-byte.
    """
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.eval_one_cell import (
        MARKER_PRECEDING_K_TOKENS,
        MARKER_SEP,
        _build_full_ids,
        build_train_equivalent_full_ids,
    )

    # Sanity: marker tokenizes to a single id.
    assert qwen_tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [
        EXPECTED_MARKER_TOKEN_ID
    ]

    persona_prompt = "You are a librarian."
    question = "What is something interesting?"

    # Build BOTH encodings and check the K+1 marker-slot tail matches.
    full_ids, _, _, slot = _build_full_ids(
        qwen_tokenizer,
        persona_prompt,
        question,
        R_text,
        MARKER_TEXT,
        EXPECTED_MARKER_TOKEN_ID,
        persona_for_log="librarian",
        q_idx_for_log=0,
        sep=MARKER_SEP,
    )
    train_ids = build_train_equivalent_full_ids(
        qwen_tokenizer,
        persona_prompt,
        question,
        R_text,
        MARKER_TEXT,
        EXPECTED_MARKER_TOKEN_ID,
        sep=MARKER_SEP,
    )

    # Marker is the last token of full_ids and occurs exactly once.
    assert full_ids[-1] == EXPECTED_MARKER_TOKEN_ID
    assert full_ids.count(EXPECTED_MARKER_TOKEN_ID) == 1
    assert slot == len(full_ids) - 1

    # Train sequence ends at the marker too.
    assert train_ids[-1] == EXPECTED_MARKER_TOKEN_ID
    assert train_ids.count(EXPECTED_MARKER_TOKEN_ID) == 1

    # K+1 trailing tokens match — the load-bearing C1 contract.
    k = MARKER_PRECEDING_K_TOKENS
    assert full_ids[-(k + 1) :] == train_ids[-(k + 1) :], (
        f"Train-vs-eval marker-slot drift for R_text={R_text!r}:\n"
        f"  eval last {k + 1} tokens = {full_ids[-(k + 1) :]}\n"
        f"  train last {k + 1} tokens = {train_ids[-(k + 1) :]}\n"
        f"  decoded eval = {qwen_tokenizer.decode(full_ids[-(k + 1) :])!r}\n"
        f"  decoded train = {qwen_tokenizer.decode(train_ids[-(k + 1) :])!r}"
    )


def test_eval_marker_slot_round1_anti_recipe_does_NOT_match(qwen_tokenizer):
    """The round-1 (buggy) recipe — concat WITHOUT the ``\\n\\n``
    separator — must NOT match training. This locks in the regression:
    if anyone reintroduces the bug, this test fires.
    """
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.eval_one_cell import (
        MARKER_SEP,
        build_train_equivalent_full_ids,
    )

    persona_prompt = "You are a librarian."
    question = "What is something interesting?"
    R_text = "Plants grow through photosynthesis."

    prompt_text = qwen_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Round-1 buggy recipe: missing the `\n\n` separator.
    bad_ids = qwen_tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
    train_ids = build_train_equivalent_full_ids(
        qwen_tokenizer,
        persona_prompt,
        question,
        R_text,
        MARKER_TEXT,
        EXPECTED_MARKER_TOKEN_ID,
        sep=MARKER_SEP,
    )
    # The K+1 trailing tokens MUST differ — this is the bug C1 caught.
    assert bad_ids[-3:] != train_ids[-3:], (
        "Round-1 buggy recipe (missing '\\n\\n' separator) somehow matches "
        "training; either the tokenizer changed OR the round-2 fix is "
        "trivial under this R_text. Pick a different R_text and re-run."
    )


# ── _build_full_ids assertion fires loudly on drift. ──────────────────────────


def test_build_full_ids_raises_when_train_eval_mismatch(qwen_tokenizer):
    """If the sep is set WRONG (mimicking the round-1 bug), _build_full_ids
    must raise — not silently emit a misaligned slot."""
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.eval_one_cell import (
        _build_full_ids,
    )

    persona_prompt = "You are a librarian."
    question = "What is something interesting?"
    R_text = "Plants grow through photosynthesis."

    # sep="" reproduces the round-1 bug. _build_full_ids' internal
    # train-equivalence assertion (assertion 4) MUST raise.
    with pytest.raises(RuntimeError, match="train/eval marker-slot context drift"):
        _build_full_ids(
            qwen_tokenizer,
            persona_prompt,
            question,
            R_text,
            MARKER_TEXT,
            EXPECTED_MARKER_TOKEN_ID,
            persona_for_log="librarian",
            q_idx_for_log=0,
            sep="",  # buggy: drop the separator
        )


# ── vLLM logprob extraction shape (no model load). ────────────────────────────


@dataclass
class _FakeLP:
    """Stand-in for vLLM's Logprob object (only `.logprob` is accessed)."""

    logprob: float


class _FakeRequestOutput:
    """Stand-in for vLLM's RequestOutput. Carries only prompt_logprobs."""

    def __init__(self, prompt_logprobs: list):
        self.prompt_logprobs = prompt_logprobs


def test_extract_marker_logprob_and_argmax_happy_path():
    """Given fake vLLM outputs with marker_id=83399 at slot=5, the helper
    returns the marker's log-prob (floor-clamped) and argmax=True flags."""
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.eval_one_cell import (
        LOGP_FLOOR,
        _extract_marker_logprob_and_argmax,
    )

    marker_id = 83399

    # Two probes. Slot 5 in each row has {marker_id: -2.5, other: -10.0}.
    def _slot(marker_lp: float, other_lp: float) -> dict[int, _FakeLP]:
        return {marker_id: _FakeLP(marker_lp), 999: _FakeLP(other_lp)}

    out1 = _FakeRequestOutput([None] * 5 + [_slot(-2.5, -10.0)])
    out2 = _FakeRequestOutput([None] * 5 + [_slot(-1.0, -7.0)])

    logps, argmax_flags = _extract_marker_logprob_and_argmax(
        [out1, out2], slot_positions=[5, 5], marker_id=marker_id, cell_label="UT"
    )
    assert logps == [-2.5, -1.0]
    assert argmax_flags == [True, True]
    # Floor clamp: insert one extremely low logp.
    out3 = _FakeRequestOutput([None] * 5 + [_slot(-1000.0, -10.0)])
    logps3, _ = _extract_marker_logprob_and_argmax(
        [out3], slot_positions=[5], marker_id=marker_id, cell_label="UT"
    )
    assert logps3 == [LOGP_FLOOR]


def test_extract_raises_when_slot_dict_is_none():
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.eval_one_cell import (
        _extract_marker_logprob_and_argmax,
    )

    out = _FakeRequestOutput([None] * 6)  # prompt_logprobs[5] is None
    with pytest.raises(RuntimeError, match=r"prompt_logprobs\[5\] is None"):
        _extract_marker_logprob_and_argmax(
            [out], slot_positions=[5], marker_id=83399, cell_label="UT"
        )


def test_extract_raises_when_marker_id_missing_from_slot():
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.eval_one_cell import (
        _extract_marker_logprob_and_argmax,
    )

    out = _FakeRequestOutput([None] * 5 + [{999: _FakeLP(-5.0), 888: _FakeLP(-6.0)}])
    with pytest.raises(RuntimeError, match=r"MARKER_ID 83399 not in prompt_logprobs"):
        _extract_marker_logprob_and_argmax(
            [out], slot_positions=[5], marker_id=83399, cell_label="UT"
        )


def test_extract_argmax_flag_is_false_when_other_token_dominates():
    """Argmax recompute should be False when marker isn't top-1."""
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.eval_one_cell import (
        _extract_marker_logprob_and_argmax,
    )

    marker_id = 83399
    # Marker at -10, other at -2 → argmax is "other".
    out = _FakeRequestOutput([None] * 5 + [{marker_id: _FakeLP(-10.0), 7: _FakeLP(-2.0)}])
    _, argmax_flags = _extract_marker_logprob_and_argmax(
        [out], slot_positions=[5], marker_id=marker_id, cell_label="UT"
    )
    assert argmax_flags == [False]


# ── End-to-end run_eval with mocked vLLM (no Qwen-7B load). ───────────────────


def test_run_eval_smoke_slice_with_mocked_vllm(qwen_tokenizer, tmp_path, monkeypatch):
    """run_eval(n_personas_limit=1, n_questions_limit=2) must:
      - Write marker_logprob.json with the v5 schema.
      - Score exactly 1 persona × 2 questions = 2 probes.
      - Not ZeroDivisionError on the per-persona-mean aggregation (the
        round-2 R2-2 regression — pre-allocated dicts for unscored
        personas blew up the divide-by-zero).
    Uses the REAL Qwen tokenizer so the C1 token-equality assertion runs
    against real BPE behavior, but mocks vLLM's LLM class so no model is
    loaded.
    """
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        BASE_MODEL,
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        eval_one_cell as e1c,
    )

    # Synthesize a tiny R_eval artifact on disk.
    r_eval_payload = {
        "schema_version": "i448_v5",
        "base_model": BASE_MODEL,
        "completions": {
            "librarian": {
                "What is something interesting?": {
                    "response_text": "Plants grow through photosynthesis.",
                    "response_token_ids": [1, 2, 3],
                    "n_response_tokens": 3,
                    "ended_with_eos": True,
                    "truncated": False,
                    "marker_in_R": False,
                },
                "What is the value of art?": {
                    "response_text": "Art reflects culture and lived experience.",
                    "response_token_ids": [4, 5, 6],
                    "n_response_tokens": 3,
                    "ended_with_eos": True,
                    "truncated": False,
                    "marker_in_R": False,
                },
            }
        },
        "content_hash": "deadbeef",
    }
    r_eval_path = tmp_path / "R_eval.json"
    r_eval_path.write_text(json.dumps(r_eval_payload))

    # Patch AutoTokenizer to return the real Qwen tokenizer (already a
    # fixture so we keep the C1 contract); patch vLLM LLM + LoRARequest.
    class _FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompts_payload, sampling_params, lora_request=None):
            outputs = []
            for payload in prompts_payload:
                full_ids = payload["prompt_token_ids"]
                slot = len(full_ids) - 1
                # Marker is at slot; trained pass returns -2.0, base -5.0.
                marker_lp = -2.0 if lora_request is not None else -5.0
                slot_dict = {EXPECTED_MARKER_TOKEN_ID: _FakeLP(marker_lp)}
                # Pad prompt_logprobs to slot+1.
                prompt_logprobs = [None] * (slot) + [slot_dict]
                outputs.append(_FakeRequestOutput(prompt_logprobs))
            return outputs

    class _FakeLoRARequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # Patch the vLLM imports inside run_eval (they're late-imported).
    import types

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM

    class _FakeSamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    fake_vllm.SamplingParams = _FakeSamplingParams
    fake_vllm_lora = types.ModuleType("vllm.lora")
    fake_vllm_lora_request = types.ModuleType("vllm.lora.request")
    fake_vllm_lora_request.LoRARequest = _FakeLoRARequest
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", fake_vllm_lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_vllm_lora_request)

    # Patch AutoTokenizer to skip the model fetch.
    class _FakeAutoTok:
        @classmethod
        def from_pretrained(cls, *_args, **_kw):
            return qwen_tokenizer

    monkeypatch.setattr("transformers.AutoTokenizer", _FakeAutoTok)

    out_dir = tmp_path / "cell_out"
    eval_personas = {
        "librarian": "You are a librarian.",
    }
    eval_questions = [
        "What is something interesting?",
        "What is the value of art?",
    ]
    adapter_path = str(tmp_path / "fake_adapter")
    Path(adapter_path).mkdir()  # vLLM doesn't load it in our fake

    out_path = e1c.run_eval(
        cell_slug="c1_anchor",
        adapter_path=adapter_path,
        base_model=BASE_MODEL,
        out_dir=out_dir,
        r_eval_path=r_eval_path,
        eval_personas=eval_personas,
        eval_questions=eval_questions,
        marker_text=MARKER_TEXT,
        marker_id=EXPECTED_MARKER_TOKEN_ID,
        n_personas_limit=1,
        n_questions_limit=2,
    )

    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["schema_version"] == "i448_v5"
    assert payload["uses_lora"] is True
    assert payload["n_cells_evaluated"] == 2  # 1 persona × 2 questions
    # Per-persona delta = trained - base = -2.0 - (-5.0) = 3.0 nats.
    mean_delta = payload["mean_per_persona_delta_g"]
    assert set(mean_delta.keys()) == {"librarian"}
    assert mean_delta["librarian"] == pytest.approx(3.0, abs=1e-6)
    # Per-persona summary file co-exists.
    summary_path = out_dir / "marker_logprob_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["n_personas_scored"] == 1
    assert summary["mean_per_persona_delta_g"]["librarian"] == pytest.approx(3.0, abs=1e-6)


def test_run_eval_base_only_skips_trained_pass(qwen_tokenizer, tmp_path, monkeypatch):
    """cell_slug == 'base' runs base-only (no LoRA), used for the Phase
    1.5 descriptive base-panel report.
    """
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        BASE_MODEL,
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        eval_one_cell as e1c,
    )

    r_eval_payload = {
        "schema_version": "i448_v5",
        "base_model": BASE_MODEL,
        "completions": {
            "librarian": {
                "What is something interesting?": {
                    "response_text": "Plants grow through photosynthesis.",
                    "response_token_ids": [1, 2, 3],
                    "n_response_tokens": 3,
                    "ended_with_eos": True,
                    "truncated": False,
                    "marker_in_R": False,
                },
            }
        },
        "content_hash": "deadbeef",
    }
    r_eval_path = tmp_path / "R_eval.json"
    r_eval_path.write_text(json.dumps(r_eval_payload))

    class _FakeLLM:
        def generate(self, prompts_payload, sampling_params, lora_request=None):
            outputs = []
            for payload in prompts_payload:
                full_ids = payload["prompt_token_ids"]
                slot = len(full_ids) - 1
                slot_dict = {EXPECTED_MARKER_TOKEN_ID: _FakeLP(-19.0)}
                outputs.append(_FakeRequestOutput([None] * slot + [slot_dict]))
            return outputs

    class _FakeSP:
        def __init__(self, **kw):
            self.kw = kw

    import types

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = lambda **kw: _FakeLLM()
    fake_vllm.SamplingParams = _FakeSP
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    class _FakeAutoTok:
        @classmethod
        def from_pretrained(cls, *_a, **_kw):
            return qwen_tokenizer

    monkeypatch.setattr("transformers.AutoTokenizer", _FakeAutoTok)

    out_dir = tmp_path / "base_out"
    out_path = e1c.run_eval(
        cell_slug="base",
        adapter_path=None,
        base_model=BASE_MODEL,
        out_dir=out_dir,
        r_eval_path=r_eval_path,
        eval_personas={"librarian": "You are a librarian."},
        eval_questions=["What is something interesting?"],
        marker_text=MARKER_TEXT,
        marker_id=EXPECTED_MARKER_TOKEN_ID,
    )
    payload = json.loads(out_path.read_text())
    assert payload["uses_lora"] is False
    assert payload["mean_per_persona_g_logprob"] == {}  # not computed for base-only
    # Base lp is populated.
    assert payload["mean_per_persona_b_logprob"]["librarian"] == pytest.approx(-19.0)


# ── Silence flake when LLM imports happen at module-load. ─────────────────────
# (No-op pass for ruff/pytest discovery hygiene.)
_ = patch

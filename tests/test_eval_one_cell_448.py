"""Tests for task #448 per-cell eval rig.

Round-3 fix R2-2 regression test: under `--smoke-real --eval-personas-limit 2
--eval-questions-limit 2`, only 2 of 24 panel personas get scored. The
round-2 code pre-allocated `logp_end_by_persona = {name: {} for name in
EVAL_PERSONAS_24}` then divided `sum(v.values()) / len(v)` for every persona
in the summary, which ZeroDivisionError'd on the 22 unscored personas.

This test exercises the reshape + summary code path WITHOUT loading a model
or running real forward passes: we monkeypatch `compute_marker_logprob` to
return fake floats, call `run_eval` with `n_personas_limit=2,
n_questions_limit=2`, and assert (a) no ZeroDivisionError, (b) the produced
summary JSON only contains entries for the 2 scored personas.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("EPM_ISSUE_448_SKIP_REGISTRY_BUILD", "1")


class _FakeTokenizer:
    """Minimal stand-in for AutoTokenizer used inside run_eval."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        # Marker assertion path: tokenize " ※" -> [83399].
        if text == " ※":
            return [83399]
        # Anything else: 1-char-per-id fake tokenization (length matters; values don't).
        return [2 + (ord(c) % 100) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = ""
        for m in messages:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out


class _FakeModel:
    """No-op model — run_eval calls `.eval()` and passes it to compute_marker_logprob."""

    def eval(self):
        return self


def test_smoke_real_slice_does_not_zerodiv(tmp_path, monkeypatch):
    """run_eval(n_personas_limit=2, n_questions_limit=2) must not ZeroDivisionError."""
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        eval_one_cell,
    )
    from explore_persona_space.personas import EVAL_QUESTIONS

    # Monkeypatch tokenizer + model loaders so we don't pull Qwen-7B in CI.
    def _fake_logprob(*args, **kwargs):
        contexts = kwargs.get("contexts")
        if contexts is None and len(args) >= 3:
            contexts = args[2]
        return [-19.3] * len(contexts)

    monkeypatch.setattr(eval_one_cell, "compute_marker_logprob", _fake_logprob)

    class _FakeAutoTok:
        @classmethod
        def from_pretrained(cls, *_args, **_kw):
            return _FakeTokenizer()

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kw):
            return _FakeModel()

    # AutoTokenizer / AutoModelForCausalLM are local imports inside run_eval.
    monkeypatch.setattr("transformers.AutoTokenizer", _FakeAutoTok)
    monkeypatch.setattr("transformers.AutoModelForCausalLM", _FakeAutoModel)

    # Supply a synthetic canonical-responses dict covering the first 2 EVAL_QUESTIONS.
    canonical = {q: "fake canonical response body." for q in EVAL_QUESTIONS}
    canonical_path = tmp_path / "eval_canonical_responses.json"
    canonical_path.write_text(json.dumps(canonical))

    out_dir = tmp_path / "out"
    out_path = eval_one_cell.run_eval(
        cell_slug="base_smoke",
        model_path="fake-model",
        out_dir=out_dir,
        canonical_responses_path=canonical_path,
        batch_size=4,
        device="cpu",
        n_personas_limit=2,
        n_questions_limit=2,
    )

    # marker_logprob.json must exist + carry only 2 personas worth of cells.
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["n_cells"] == 4, f"Expected 2x2=4 cells; got {payload['n_cells']}"
    assert len(payload["eval_personas"]) == 2
    assert len(payload["logp_end_of_canonical_response"]) == 2
    assert len(payload["logp_k0_diagnostic"]) == 2
    for persona_dict in payload["logp_end_of_canonical_response"].values():
        assert len(persona_dict) == 2  # 2 questions per persona

    # Summary JSON must also only carry 2 personas + must NOT ZeroDivisionError.
    summary_path = out_dir / "marker_logprob_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["n_personas_scored"] == 2
    assert len(summary["mean_per_persona_end_of_canonical_response"]) == 2
    assert len(summary["mean_per_persona_k0_diagnostic"]) == 2
    for mean_val in summary["mean_per_persona_end_of_canonical_response"].values():
        assert mean_val == pytest.approx(-19.3)

"""Regression pin for the #1073 round-2 BLOCKER fix
(concern p2-gen-no-intra-phase-checkpoint): ``issue1073_gen._generate_chunked``
persists each vLLM chunk ATOMICALLY the moment it returns and SKIPS completed
chunks on re-entry behind a fail-loud regime-key match, so a crash mid-arm
loses at most the in-flight chunk of the ~2-2.5 h P2 stoch10 generation.

Fails pre-fix (the pre-fix function accumulated all chunks in memory and had
no checkpoint_dir/regime surface at all); passes post-fix. The engine fake is
signature-conformant by construction: its ``generate(prompt_texts,
sampling_params, use_tqdm=False)`` mirrors the vLLM ``LLM.generate`` call
shape ``_generate_chunked`` dispatches (same surface as ``I.HFGenShim``).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import issue1073_gen as G  # noqa: E402


class _CountingEngine:
    """vLLM-generate stand-in (external GPU boundary), signature-conformant."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def generate(self, prompt_texts, sampling_params, use_tqdm=False):
        self.calls.append(list(prompt_texts))
        results = []
        for t in prompt_texts:
            outs = [
                types.SimpleNamespace(
                    text=f"resp:{t}:{j}", token_ids=[1, 2, 3], finish_reason="stop"
                )
                for j in range(sampling_params.n)
            ]
            results.append(types.SimpleNamespace(outputs=outs))
        return results


def _sp(n: int = 2):
    return types.SimpleNamespace(n=n, temperature=1.0, top_p=0.95, max_tokens=8, seed=42)


def _regime(**over):
    base = {"arm": "stoch10", "chunk_size": 2, "n_prompts": 5, "model": "tiny"}
    base.update(over)
    return base


def test_chunk_persist_resume_and_regime_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "VLLM_CHUNK_SIZE", 2)
    texts = [f"prompt {i}" for i in range(5)]  # 3 chunks at size 2
    eng = _CountingEngine()
    out1 = G._generate_chunked(
        eng, texts, _sp(), "stoch10", checkpoint_dir=tmp_path, regime=_regime()
    )
    assert len(out1) == 5 and len(eng.calls) == 3
    assert sorted(p.name for p in tmp_path.glob("stoch10.chunk*.json")) == [
        "stoch10.chunk000.json",
        "stoch10.chunk001.json",
        "stoch10.chunk002.json",
    ]
    # Full re-entry: every chunk resumed, ZERO new generate calls, same output.
    eng2 = _CountingEngine()
    out2 = G._generate_chunked(
        eng2, texts, _sp(), "stoch10", checkpoint_dir=tmp_path, regime=_regime()
    )
    assert out2 == out1 and eng2.calls == []
    # Partial resume: drop the last chunk file -> ONLY that chunk regenerates.
    (tmp_path / "stoch10.chunk002.json").unlink()
    eng3 = _CountingEngine()
    out3 = G._generate_chunked(
        eng3, texts, _sp(), "stoch10", checkpoint_dir=tmp_path, regime=_regime()
    )
    assert out3 == out1 and eng3.calls == [["prompt 4"]]
    # A regime mismatch on ANY output-affecting key fails LOUD (never a silent
    # reuse of wrong cached rows — the #722-r3 resume lesson).
    with pytest.raises(AssertionError, match="regime mismatch"):
        G._generate_chunked(
            eng3,
            texts,
            _sp(),
            "stoch10",
            checkpoint_dir=tmp_path,
            regime=_regime(chunk_size=4),
        )


def test_crash_after_chunk_hook_then_resume(tmp_path, monkeypatch):
    """The smoke's kill/interrupt path: crash after chunk 1 persists, then a
    re-run skips chunk 1 and generates only chunk 2+ (the p2-resume smoke)."""
    monkeypatch.setattr(G, "VLLM_CHUNK_SIZE", 2)
    texts = [f"prompt {i}" for i in range(4)]  # 2 chunks
    monkeypatch.setenv("EPM_I1073_GEN_CRASH_AFTER_CHUNK", "1")
    eng = _CountingEngine()
    with pytest.raises(RuntimeError, match="simulated crash"):
        G._generate_chunked(
            eng, texts, _sp(n=1), "stoch10", checkpoint_dir=tmp_path, regime=_regime(n_prompts=4)
        )
    assert (tmp_path / "stoch10.chunk000.json").exists()
    assert not (tmp_path / "stoch10.chunk001.json").exists()
    monkeypatch.delenv("EPM_I1073_GEN_CRASH_AFTER_CHUNK")
    eng2 = _CountingEngine()
    out = G._generate_chunked(
        eng2, texts, _sp(n=1), "stoch10", checkpoint_dir=tmp_path, regime=_regime(n_prompts=4)
    )
    assert len(out) == 4
    assert eng2.calls == [["prompt 2", "prompt 3"]]  # chunk 1 skipped, chunk 2 generated


def test_no_checkpoint_dir_keeps_legacy_shape(tmp_path):
    """P0 probe path (no checkpointing) is unchanged: no files written."""
    eng = _CountingEngine()
    out = G._generate_chunked(eng, ["a", "b"], _sp(n=1), "p0-regen")
    assert len(out) == 2 and len(eng.calls) == 1
    assert list(tmp_path.iterdir()) == []

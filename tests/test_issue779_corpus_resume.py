"""Regression tests for the issue #779 round-6 crash-fix (att-20260702 CUDA OOM).

Pins the three round-6 invariants (each FAILS against the pre-fix driver):

  - **persist-rollouts-before-capture**: ``run_corpus_phase`` writes
    ``{trait}_rollouts.json`` IMMEDIATELY after generation, BEFORE the
    answer-vector capture — a capture crash (the att-20260702 CUDA OOM during
    sycophancy answer capture, which lost 5 completed vLLM generation chunks)
    must never lose a completed generation again. Pre-fix the text was written
    only at trait END.
  - **skip-generation resume**: on re-entry with persisted rollout text but no
    bundle/scores, generation is SKIPPED and the capture inputs are rebuilt
    from the persisted text, with fail-loud alignment asserts against the
    current corpus spec (a drifted spec raises, never silently misaligns).
  - **trait-resume HF prefetch**: prior-attempt state recovered to the HF data
    repo is materialized into the standard local layout
    (``out_dir/behavior_corpus/`` + ``data/issue_779/corpus_specs/``); a file
    missing on HF is NOT an error (the trait runs); a failed fetch of a LISTED
    file raises; an already-local file short-circuits without a download.

Pure-CPU, hermetic (no model / no network: the HF download, judge, spec
generation, and capture collaborators are stubbed; ``vllm`` via sys.modules).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue779_gen_behavior_corpus as G  # noqa: E402

_LAYERS = [0, 1]
_H = 4


def _fake_spec(trait: str) -> dict:
    return {
        "trait": trait,
        "personas": ["You are upbeat.", "You are terse."],
        "questions": ["What is water?"],
        "n_personas": 2,
        "n_questions": 1,
    }


class _FakeTokenizer:
    """apply_chat_template stand-in (the only tokenizer surface the generation
    branch of run_corpus_phase touches once capture is stubbed)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " | ".join(m["content"] for m in messages) + " <gen>"


def _patch_collaborators(monkeypatch, gen_fn, capture_fn) -> None:
    """Stub every model/network-touching collaborator of run_corpus_phase."""
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(SamplingParams=lambda **kw: types.SimpleNamespace(**kw)),
    )
    monkeypatch.setattr(G.C, "generate_behavior_corpus_spec", lambda t, **kw: _fake_spec(t))
    monkeypatch.setattr(G.C, "assert_corpus_disjoint", lambda *a, **kw: None)
    monkeypatch.setattr(G.C, "judge_rollouts_n5", lambda *a, **kw: {})
    monkeypatch.setattr(G.COL, "assert_batched_capture_equivalence", lambda *a, **kw: {})
    monkeypatch.setattr(
        G.COL,
        "capture_context_vectors_batched",
        lambda model, tok, msgs, layers: [
            {
                "last": torch.zeros(len(_LAYERS), _H),
                "mean": torch.zeros(len(_LAYERS), _H),
                "prompt_len": 3,
            }
            for _ in msgs
        ],
    )
    monkeypatch.setattr(G.COL, "_vllm_generate_chunked", gen_fn)
    monkeypatch.setattr(G.COL, "capture_answer_vectors_batched", capture_fn)


def _run_corpus(tmp_path: Path, trait: str = "evil", n_rollouts: int = 2) -> dict:
    return G.run_corpus_phase(
        object(),  # model — never touched once capture is stubbed
        _FakeTokenizer(),
        object(),  # llm — only forwarded to the (stubbed) generate
        _LAYERS,
        {trait: torch.zeros(len(_LAYERS), _H)},
        tmp_path,
        traits=[trait],
        n_personas=2,
        n_questions=1,
        n_rollouts=n_rollouts,
        dry_run_judge=True,
    )


# ── persist-before-capture (the incident regression) ─────────────────────────


def test_rollout_text_persisted_before_capture_crash(tmp_path, monkeypatch):
    """A capture crash after generation must leave {trait}_rollouts.json on
    disk. FAILS pre-fix: the text write lived at trait END, so the
    att-20260702 OOM lost sycophancy's completed generation."""

    def gen_fn(llm, prompts, sp):
        return [["r0", "r1"] for _ in prompts]

    def capture_fn(*a, **kw):
        raise RuntimeError("simulated att-20260702 CUDA OOM during answer capture")

    _patch_collaborators(monkeypatch, gen_fn, capture_fn)
    with pytest.raises(RuntimeError, match="simulated att-20260702"):
        _run_corpus(tmp_path)
    text_path = tmp_path / G.CORPUS_SUBDIR / "evil_rollouts.json"
    assert text_path.exists(), "rollout text must survive a capture crash"
    blob = json.loads(text_path.read_text())
    assert blob["trait"] == "evil"
    assert len(blob["rollouts"]) == 2  # 2 personas x 1 question
    row = blob["rollouts"]["0"]
    assert row["responses"] == ["r0", "r1"]
    assert row["persona"]  # persona text rides along -> rebuildable on resume
    # the crash happened BEFORE the bundle/scores writes (trio incomplete)
    assert not (tmp_path / G.CORPUS_SUBDIR / "evil_corpus.pt").exists()


def test_resume_skips_generation_and_completes(tmp_path, monkeypatch):
    """Persisted rollout text + no bundle => generation is SKIPPED (the stub
    raises if called) and the trait completes from the persisted text."""

    def gen_fn(llm, prompts, sp):
        raise AssertionError("generation must NOT run when rollout text is persisted")

    def capture_fn(model, tok, items, layers, r_b_by_trait, keep_per_token=False):
        return [{"v_x": torch.zeros(len(_LAYERS), _H)} for _ in items]

    _patch_collaborators(monkeypatch, gen_fn, capture_fn)
    spec = _fake_spec("evil")
    contexts = G.build_corpus_contexts("evil", spec["personas"], spec["questions"])
    corpus_dir = tmp_path / G.CORPUS_SUBDIR
    corpus_dir.mkdir(parents=True)
    G._write_rollout_text(
        corpus_dir / "evil_rollouts.json", "evil", contexts, [["a0", "a1"], ["b0", "b1"]]
    )
    produced = _run_corpus(tmp_path)
    assert (corpus_dir / "evil_corpus.pt").exists()
    assert (corpus_dir / "evil_judge_scores.json").exists()
    assert produced["evil"]["n_vx_captured"] == 4  # 2 contexts x 2 resumed rollouts


def test_resume_alignment_round_trip_and_fail_loud(tmp_path):
    """Same-spec resume round-trips exactly; a drifted spec or a rollout-count
    mismatch raises (never a silent misalignment)."""
    spec = _fake_spec("evil")
    contexts = G.build_corpus_contexts("evil", spec["personas"], spec["questions"])
    p = tmp_path / "evil_rollouts.json"
    G._write_rollout_text(p, "evil", contexts, [["a0", "a1"], ["b0", "b1"]])
    gen = G._load_rollout_text_for_resume(p, "evil", contexts, 2)
    assert gen == [["a0", "a1"], ["b0", "b1"]]
    drifted = G.build_corpus_contexts("evil", spec["personas"], ["What is fire?"])
    with pytest.raises(AssertionError, match="DIFFERENT corpus spec"):
        G._load_rollout_text_for_resume(p, "evil", drifted, 2)
    with pytest.raises(AssertionError, match="n_rollouts"):
        G._load_rollout_text_for_resume(p, "evil", contexts, 3)


# ── trait-resume HF prefetch ──────────────────────────────────────────────────


def test_prefetch_materializes_listed_files_and_ignores_missing(tmp_path, monkeypatch):
    prefix = G.HF_ROUND_PREFIX
    listed = {
        f"{prefix}/behavior_corpus/evil_corpus.pt",
        f"{prefix}/behavior_corpus/evil_judge_scores.json",
        f"{prefix}/behavior_corpus/evil_rollouts.json",
        f"{prefix}/corpus_specs/evil_personas.json",
        f"{prefix}/corpus_specs/evil_questions.json",
    }

    def fake_download(repo_id, filename, repo_type, local_dir):
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"fake": filename}))
        return str(out)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    specs_dir = tmp_path / "specs"
    monkeypatch.setattr(G.C, "_corpus_dir", lambda: specs_dir)
    out_dir = tmp_path / "out"
    got = G.prefetch_trait_resume_state(
        ["evil", "hallucination"], out_dir, smoke=False, repo_files=listed
    )
    assert sorted(got["evil"]) == sorted(
        [
            "evil_corpus.pt",
            "evil_judge_scores.json",
            "evil_rollouts.json",
            "corpus_specs/evil_personas.json",
            "corpus_specs/evil_questions.json",
        ]
    )
    assert got["hallucination"] == []  # missing on HF is NOT an error
    assert (out_dir / G.CORPUS_SUBDIR / "evil_corpus.pt").exists()
    assert (out_dir / G.CORPUS_SUBDIR / "evil_rollouts.json").exists()
    assert (specs_dir / "evil_personas.json").exists()  # fixed spec layout


def test_prefetch_failed_fetch_of_listed_file_raises(tmp_path, monkeypatch):
    listed = {f"{G.HF_ROUND_PREFIX}/behavior_corpus/evil_judge_scores.json"}
    import huggingface_hub

    def boom(*a, **kw):
        raise OSError("simulated network failure")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    with pytest.raises(OSError, match="simulated network failure"):
        G.prefetch_trait_resume_state(["evil"], tmp_path, smoke=False, repo_files=listed)


def test_prefetch_local_file_short_circuits(tmp_path, monkeypatch):
    dest = tmp_path / "x.json"
    dest.write_text("{}")
    import huggingface_hub

    def must_not_download(*a, **kw):
        raise AssertionError("must not download an already-local file")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", must_not_download)
    assert G._prefetch_hf_resume_file(set(), "anything/x.json", dest) is True

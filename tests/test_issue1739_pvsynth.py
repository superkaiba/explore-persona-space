"""Tests for the issue #1739 pvsynth rung (the Persona Vectors eval distribution).

Covers the round's own new code (``build_contexts``) plus a TINY-REAL
generate -> capture -> ``load_summaries`` chain through the PRODUCTION library
functions, with fakes ONLY at the GPU boundary (the sampler and the 7B weights).
CONTENT HYGIENE: assertions are over ids / counts / shapes — never asset text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.issue1739_pvsynth_pod import (
    RUNG,
    SIGNS,
    SPLIT,
    build_contexts,
)

BEHAVIORS = ("evil", "sycophancy", "hallucination")


@pytest.mark.parametrize("behavior", BEHAVIORS)
def test_build_contexts_is_the_paper_grid(behavior):
    """5 pairs x 2 signs x 20 HELD-OUT eval questions = 200 contexts, 10 groups."""
    rows = build_contexts(behavior)
    assert len(rows) == 200, (behavior, len(rows))
    assert len({r["context_id"] for r in rows}) == 200
    assert {r["group_key"] for r in rows} == {f"{RUNG}-p{p}-{s}" for p in range(5) for s in SIGNS}
    for r in rows:
        assert r["split"] == SPLIT and r["rung"] == RUNG
        assert r["behavior"] == behavior
        assert r["prefix_text"].strip(), "instruction must be a non-empty system prompt"
        assert r["query"].strip()


def test_context_ids_are_batch_custom_id_safe():
    """Judge item ids must satisfy the Batch custom_id grammar + 53-char budget."""
    from explore_persona_space.experiments.issue_1739.judging import rollout_item_id

    for r in build_contexts("evil"):
        for k in range(5):
            item_id = rollout_item_id(r["context_id"], k)
            assert len(item_id) <= 53
            assert set(item_id) <= set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            )


def test_eval_questions_are_disjoint_from_extraction_questions():
    """The rung must use the eval split — never the r_B extraction questions."""
    from explore_persona_space.experiments.issue_1739.generation import load_e1_assets

    for behavior in BEHAVIORS:
        assets = load_e1_assets(behavior)
        extraction = set(assets["extraction_questions"])
        queries = {r["query"] for r in build_contexts(behavior)}
        assert queries and not (queries & extraction), behavior


def test_gpu_helper_call_shapes_bind():
    """Signature-bind the GPU-fenced helper calls the local smoke cannot execute.

    Import resolution green-lights a call-arity/return-shape mismatch: the pod's
    first production-mode smoke crashed on ``model, tokenizer =
    load_capture_model(...)`` (it returns the MODEL ONLY). These binds fail
    pre-fix and pass post-fix.
    """
    import inspect

    from explore_persona_space.experiments.issue_1739.capture import (
        capture_rollout_files,
        load_capture_model,
    )

    # load_capture_model takes device= and returns a SINGLE object (not a tuple).
    inspect.signature(load_capture_model).bind(device="cuda")
    src = inspect.getsource(load_capture_model)
    assert "return model" in src and "return model," not in src, (
        "load_capture_model's return shape changed — the pod leg unpacks it as a single model"
    )

    # capture_rollout_files: positional paths + the kw-only set the pod leg passes.
    inspect.signature(capture_rollout_files).bind(
        [object()],
        store_dir=object(),
        model=object(),
        tokenizer=object(),
        n_layers=28,
        hidden_dim=3584,
        device="cuda",
        fingerprint="fp",
        batch_size=8,
    )


def _tiny_causal_lm():
    """A 2-layer Qwen2 over the REAL vocab (fake weights ONLY, real arch/ids)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from explore_persona_space.experiments.issue_1739.generation import get_tokenizer

    tok = get_tokenizer()
    cfg = transformers.Qwen2Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    return model, tok, cfg


def test_tiny_real_generate_then_capture_then_load_summaries(tmp_path):
    """Production generate + capture path end-to-end; only the sampler+weights fake."""
    torch = pytest.importorskip("torch")
    from explore_persona_space.experiments.issue_1739 import capture, store_io
    from explore_persona_space.experiments.issue_1739.generation import generate_labeling

    model, tok, cfg = _tiny_causal_lm()
    n_layers, hidden = cfg.num_hidden_layers, cfg.hidden_size
    contexts = build_contexts("evil", max_pairs=1, max_questions=2)
    assert len(contexts) == 4  # 1 pair x 2 signs x 2 questions

    def fake_generate(prompts, *, n, temperature, max_tokens, seeds):
        # Boundary fake: the GPU sampler. Real prompts in, plausible text out.
        assert len(prompts) == len(seeds)
        return [
            [{"text": f"Answer {i}-{j}.", "finish_reason": "stop"} for j in range(n)]
            for i in range(len(prompts))
        ]

    manifest = generate_labeling(
        contexts,
        out_root=tmp_path,
        behavior="evil",
        k_rollouts=2,
        max_new_tokens=32,
        generate_fn=fake_generate,
        tokenizer=tok,
    )
    assert manifest["n_kept"] == 4 and manifest["n_generated"] == 4

    rollout_paths = sorted((tmp_path / "labeling" / "evil").glob("*.json"))
    rollout_paths = [p for p in rollout_paths if not p.name.startswith("_")]
    assert len(rollout_paths) == 8  # 4 contexts x 2 rollouts

    # The ladder fields the DV side + _load_labeled require must survive the write.
    payload = json.loads(rollout_paths[0].read_text())
    assert payload["split"] == SPLIT and payload["rung"] == RUNG
    assert payload["group_key"].startswith(f"{RUNG}-p0-")
    assert payload["query"] and payload["completion"]

    store_dir = tmp_path / "store" / "evil"
    with torch.no_grad():
        cap = capture.capture_rollout_files(
            rollout_paths,
            store_dir=store_dir,
            model=model,
            tokenizer=tok,
            n_layers=n_layers,
            hidden_dim=hidden,
            device="cpu",
            batch_size=2,
            fingerprint=manifest["fingerprint"],
        )
    assert cap["n_rows"] == 8, cap

    # The CONSUMER's own reader must open the store we just wrote.
    layers = tuple(range(n_layers))
    arrays, meta = store_io.load_summaries(
        store_dir, ("prefix_end", "context_end", "t1"), layers, hidden_dim=hidden
    )
    assert len(meta) == 8
    for kind in ("prefix_end", "context_end", "t1"):
        for ly in layers:
            assert arrays[(kind, ly)].shape == (8, hidden), (kind, ly)
    ctx_ids = {r["context_id"] for r in meta}
    assert ctx_ids == {r["context_id"] for r in contexts}
    assert all(r["rung"] == RUNG and r["split"] == SPLIT for r in meta)
    # context_end / prefix_end are context-level: identical across a context's rollouts.
    by_ctx: dict[str, list[int]] = {}
    for i, r in enumerate(meta):
        by_ctx.setdefault(r["context_id"], []).append(i)
    for rows in by_ctx.values():
        assert len(rows) == 2
        for kind in ("prefix_end", "context_end"):
            a = arrays[(kind, 0)]
            assert a[rows[0]] == pytest.approx(a[rows[1]], abs=1e-3), kind

"""Natural-source transport, exact dedup, and real HF capture seam tests."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import issue1739_natural_data as d


def test_raw_shards_unicode_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "PART_BYTES", 120)
    rows = [{"prompt": "natural " + chr(0x2028) + str(i)} for i in range(30)]
    index = d.write_parts(tmp_path, rows)
    assert len(index["parts"]) > 1
    assert d.load_parts(tmp_path) == rows
    part = tmp_path / index["parts"][0]["path"]
    part.write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        d.load_parts(tmp_path)


def test_prefix_jaccard_matches_reference():
    from scripts.issue779_ffc_n1m_generate_capture import NearDupeGate

    rng = np.random.default_rng(1739)
    targets = ["".join(rng.choice(list("abcdefgh ijklmnop"), 100)) for _ in range(100)]
    targets.extend(["", "a", "abc", "the same question", "THE SAME   question"])
    candidates = targets + [t[:80] + " new suffix" for t in targets]
    candidates += ["".join(rng.choice(list("abcdefgh ijklmnop"), 100)) for _ in range(300)]
    old, new = NearDupeGate(targets), d.IndexedNearDupeGate(targets)
    assert [new.is_dupe(t) for t in candidates] == [old.is_dupe(t) for t in candidates]


def test_units_refuse_changed_inputs(tmp_path):
    d.atomic_json(tmp_path / "data.json", {"a": 1})
    d.finish_unit(tmp_path, "key", 0)
    assert d.complete_unit(tmp_path, "key")
    with pytest.raises(ValueError, match="stale unit"):
        d.complete_unit(tmp_path, "other")
    d.atomic_json(tmp_path / "data.json", {"a": 2})
    with pytest.raises(ValueError, match="absent/corrupt"):
        d.complete_unit(tmp_path, "key")


def test_reused_signatures():
    from explore_persona_space.analysis.extraction import _logits_to_keep_kwargs
    from explore_persona_space.eval.generation import create_vllm_engine
    from scripts.issue779_ffc_n1m_generate_capture import _download_manifest

    inspect.signature(_logits_to_keep_kwargs).bind(object(), return_logits=False)
    inspect.signature(create_vllm_engine).bind(
        d.MODEL,
        revision=d.REVISION,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        max_num_seqs=64,
        hang_mitigations=True,
        seed=42,
    )
    inspect.signature(_download_manifest).bind(d.SOURCE_PREFIX, None, revision=d.SOURCE_REVISION)


def test_tiny_real_hf_capture_matches_reference(tmp_path, monkeypatch):
    """Real tokenizer and same-architecture model; only the 7B weights shrink."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    tok = d.reference._get_tokenizer()
    config = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    torch.manual_seed(1739)
    model = Qwen2ForCausalLM(config).eval()
    monkeypatch.setattr(d.reference, "N_LAYERS", 2)
    monkeypatch.setattr(d.reference, "HIDDEN_DIM", 16)
    rows = []
    for i, (prompt, answer) in enumerate(
        [
            ("What is two plus two?", "\n\nFour.\n"),
            ("Explain the seasons in a short sentence.", "Earth's axial tilt causes seasons."),
            ("Thanks.", "You're welcome."),
        ]
    ):
        prefix, rendered = d.reference._render_prompt_parts([], prompt, "instruct")
        rows.append(
            {
                "prefix_text": prefix,
                "rendered_prompt": rendered,
                "answer": answer,
                "context_id": str(i),
            }
        )
    actual = d.capture_batch(rows, model, tok, layers=[0, 1], device="cpu")
    expected = d.reference._capture_batch_loaded_model(
        prefix_texts=[r["prefix_text"] for r in rows],
        prompts=[r["rendered_prompt"] for r in rows],
        completions=[r["answer"] for r in rows],
        prompt_format="instruct",
        model=model,
        tokenizer=tok,
        n_layers=2,
        hidden_dim=16,
        device="cpu",
        log_label="test",
        batch_size=3,
    )
    for kind in ("context_end", "t1"):
        for layer in (0, 1):
            ref = np.stack([r[kind][layer] for r in expected.summaries])
            np.testing.assert_allclose(actual[f"{kind}_L{layer:02d}"], ref, atol=1e-4, rtol=1e-3)


def test_assemble_never_marks_short_pool_complete(tmp_path):
    d.write_parts(tmp_path / "prepared", [{"prompt": "legitimate ungenerated input"}])
    with pytest.raises(ValueError, match="reserve generation"):
        d.assemble(SimpleNamespace(root=tmp_path, n_rows=100000))
    assert not (tmp_path / "store" / "manifest.json").exists()

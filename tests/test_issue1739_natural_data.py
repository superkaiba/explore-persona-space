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
    from explore_persona_space.orchestrate.hub import stage_hub_file

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
    inspect.signature(stage_hub_file).bind(
        d.SOURCE_REPO, "data/train.parquet", None, revision=d.SOURCE_REVISION, size_bytes=100
    )


def source_fixture(tmp_path, monkeypatch):
    """Real Parquet I/O; replace only the external download boundary."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from explore_persona_space.orchestrate import hub

    rows = [
        {
            "conversation_id": "original-a",
            "conversation": [
                {"role": "user", "content": "How do ocean tides work?"},
                {"role": "assistant", "content": "third-party answer MUST NOT enter pool"},
                {"role": "user", "content": "later query MUST NOT enter pool"},
            ],
        },
        {
            "conversation_id": "original-b",
            "conversation": [{"role": "user", "content": "How do ocean tides work?"}],
        },
        {
            "conversation_id": "original-c",
            "conversation": [{"role": "system", "content": "not a first user prompt"}],
        },
        {
            "conversation_id": "original-d",
            "conversation": [{"role": "user", "content": "Explain the seasons.\u2028Thanks!"}],
        },
    ]
    path = tmp_path / "fixture.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    inventory = [
        {
            "path": "data/train-fixture.parquet",
            "bytes": path.stat().st_size,
            "sha256": d.file_sha(path),
        }
    ]
    monkeypatch.setattr(hub, "stage_hub_file", lambda *a, **kw: path)
    monkeypatch.setattr(d, "source_inventory", lambda: inventory)
    monkeypatch.setattr(d, "SOURCE_ROWS", len(rows))
    monkeypatch.setattr(d, "SOURCE_BATCH", 2)
    return inventory, path


def test_upstream_first_user_checkpoints_and_integrity(tmp_path, monkeypatch):
    inventory, path = source_fixture(tmp_path, monkeypatch)
    batches = list(d.source_batches(tmp_path, inventory))
    assert [len(b) for b in batches] == [2, 2]
    assert batches[0][0]["conversation_id"] == "original-a"
    assert batches[0][0]["prompt"] == "How do ocean tides work?"
    assert batches[1][0]["first_role"] == "system"
    assert batches[1][0]["prompt"] is None
    assert batches == list(d.source_batches(tmp_path, inventory))
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="parquet content mismatch"):
        list(d.source_batches(tmp_path, inventory))


def test_prepare_real_parquet_dedups_and_retains_original_provenance(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    exclusion = tmp_path / "excluded.jsonl"
    exclusion.write_text('{"text": "unrelated held-out evaluation question"}\n')
    args = SimpleNamespace(root=tmp_path / "pool", exclusions=exclusion, n_candidates=2)
    d.prepare(args)
    selected = d.load_parts(args.root / "prepared")
    assert {r["source_id"] for r in selected} == {"original-a", "original-d"}
    assert all(r["source_revision"] == d.SOURCE_REVISION for r in selected)
    assert all(r["source_first_role"] == "user" for r in selected)
    assert all("MUST NOT" not in r["rendered_prompt"] for r in selected)
    assert {r["source_row"] for r in selected} == {0, 3}
    # Complete-pool resume validates the same immutable inventory and shards.
    d.prepare(args)
    assert selected == d.load_parts(args.root / "prepared")


@pytest.mark.parametrize("suffix", ["", "\n" * 14 + "example"])
def test_prepare_drops_credential_record_without_rewriting_prompt(tmp_path, monkeypatch, suffix):
    import pyarrow as pa
    import pyarrow.parquet as pq

    inventory, path = source_fixture(tmp_path, monkeypatch)
    rows = pq.read_table(path).to_pylist()
    rows.append(
        {
            "conversation_id": "credential-row",
            "conversation": [
                {"role": "user", "content": "Bearer " + "hf_" + "ab19Cd" * 6 + suffix}
            ],
        }
    )
    pq.write_table(pa.Table.from_pylist(rows), path)
    inventory[0].update(bytes=path.stat().st_size, sha256=d.file_sha(path))
    monkeypatch.setattr(d, "SOURCE_ROWS", len(rows))
    exclusion = tmp_path / "excluded.jsonl"
    exclusion.write_text('{"text": "unrelated held-out question"}\n')
    args = SimpleNamespace(root=tmp_path / "pool", exclusions=exclusion, n_candidates=3)
    with pytest.raises(ValueError, match="'credential_bearing': 1"):
        d.prepare(args)
    assert not (args.root / "prepared/complete.json").exists()


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

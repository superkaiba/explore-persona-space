"""Validate draw integrity, original span cuts, and the real layer-hook capture path."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def driver(monkeypatch):
    """Load this worktree's actual phase driver without loading model weights."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("issue825_turn_k5_gpu")


@pytest.fixture
def tokenizer(driver):
    """Use an actual small fast tokenizer; no external download or tokenization stub."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(set("User: question\n\nAssistant: answer one two."))
    vocab = {"[UNK]": 0, "[PAD]": 1, **{c: i + 2 for i, c in enumerate(alphabet)}}
    core = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    result = PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", padding_side="right"
    )
    return result


def conversation():
    """Provide a parser fixture with twelve logged assistant/user pairs."""
    return [
        {"role": role, "content": content}
        for _ in range(12)
        for role, content in (("user", "question"), ("assistant", "answer"))
    ]


def draw_rows(texts=None):
    """Supply persisted-record fixtures for five distinct answer identities."""
    texts = texts or ["answer one", "answer two", "answer", "answer one two", "answer."]
    return [
        {
            "conv_id": "c1",
            "turn": 1,
            "turn_index": 1,
            "draw_id": i,
            "text": text,
            "finish_reason": "stop",
            "at_token_cap": False,
            "n_prompt_tokens": 25,
            "n_gen_tokens": len(text),
        }
        for i, text in enumerate(texts)
    ]


def test_panel_preserves_order_and_rejects_duplicates(driver, tmp_path):
    """Panel narrowing cannot silently select another population or duplicate histories."""
    path = tmp_path / "panel.jsonl"
    rows = [{"conv_id": "b", "turns": conversation()}, {"conv_id": "a", "turns": conversation()}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert driver.load_panel(path, 1) == rows[:1]
    path.write_text("".join(json.dumps(rows[0]) + "\n" for _ in range(2)))
    with pytest.raises(ValueError, match="unique"):
        driver.load_panel(path)


def test_jsonl_unicode_inside_strings_is_not_a_record_separator(driver, tmp_path):
    """Read physical newlines rather than Unicode splitlines on real corpus text."""
    path = tmp_path / "rows.jsonl"
    row = {"text": "hello\u2028world\u0085again"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    assert driver.read_jsonl(path) == [row]


def test_build_jobs_uses_original_base_scaffold_and_budget(driver, tokenizer):
    """Exercise the actual parent full-conversation renderer and both endpoint selection."""
    panel = [{"conv_id": "c", "turns": conversation()}]
    jobs = driver.build_jobs(panel, "pretrained", tokenizer, 1024)
    assert [j["turn"] for j in jobs] == [1, 12]
    assert jobs[0]["prompt"] == "User: question\n\nAssistant:"
    assert jobs[1]["turn_index"] == 23
    with pytest.raises(ValueError, match="budget exceeded"):
        driver.build_jobs(panel, "pretrained", tokenizer, 8192)


def test_serialization_retains_all_five_draws_and_raw_tokens(driver):
    """Distinct vLLM draw indices remain distinct, including length-capped answers."""
    jobs = [{"conv_id": "c", "turn": 12, "turn_index": 23, "prompt_token_ids": [7, 8]}]
    outputs = [
        SimpleNamespace(
            prompt_token_ids=[7, 8],
            outputs=[
                SimpleNamespace(
                    index=i,
                    text=f"raw {i}\n",
                    token_ids=[i, i + 1],
                    finish_reason="length" if i == 4 else "stop",
                    stop_reason=None,
                )
                for i in reversed(range(5))
            ],
        )
    ]
    rows = driver.serialize_outputs(jobs, outputs, "instruct", 1024)
    assert [r["draw_id"] for r in rows] == list(range(5))
    assert [r["token_ids"] for r in rows] == [[i, i + 1] for i in range(5)]
    assert rows[4]["at_token_cap"] is True
    assert rows[0]["text"].endswith("\n")
    driver.check_generation_rows(rows, jobs, "instruct", 1024)
    outputs[0].outputs.pop()
    with pytest.raises(ValueError, match="draw IDs"):
        driver.serialize_outputs(jobs, outputs, "instruct", 1024)


def test_capture_group_uses_actual_parent_cuts_and_joint_exclusion(driver, tokenizer):
    """A bad fifth answer excludes all five; valid answers retain separate span lengths."""
    rows = draw_rows()
    planned, missing = driver.prepare_capture_group(rows, conversation(), "pretrained", tokenizer)
    assert not missing and len(planned) == 5
    assert len({r["context_prefix_sha256"] for r in planned}) == 1
    assert [r["n_answer_tokens"] for r in planned] == [10, 10, 6, 14, 7]
    assert [r["draw_id"] for r in planned] == list(range(5))
    assert max(r["render_overhead_tokens"] for r in planned) <= 64
    rows[4]["text"] = " \n "
    planned, missing = driver.prepare_capture_group(rows, conversation(), "pretrained", tokenizer)
    assert not planned and len(missing) == 5
    assert {r["reason"] for r in missing} == {"empty_completion"}


def test_capture_render_overhead_must_fit_reserved_tokens(driver, tokenizer):
    """The tokenizer seam reserve is checked using realized raw-generation lengths."""
    rows = draw_rows()
    rows[0]["text"] = "answer " * 30
    with pytest.raises(ValueError, match="64-token"):
        driver.prepare_capture_group(rows, conversation(), "pretrained", tokenizer)


def test_capture_group_actual_bpe_boundary_mismatch_excludes_jointly(driver):
    """A real boundary-merging tokenizer diagnoses mechanical prefix drift before capture."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(set("User: question\n\nAssistant: answer."))
    vocab = {"[UNK]": 0, "[PAD]": 1, **{c: i + 2 for i, c in enumerate(alphabet)}}
    vocab.update({": ": len(vocab), ": .": len(vocab) + 1})
    core = Tokenizer(models.BPE(vocab=vocab, merges=[(":", " "), (": ", ".")], unk_token="[UNK]"))
    tok = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]")
    rows = draw_rows(["answer", ".", "answer", ".", "answer"])
    planned, missing = driver.prepare_capture_group(rows, conversation(), "pretrained", tok)
    assert not planned and len(missing) == 5
    assert {r["reason"] for r in missing} == {"context_prefix_token_mismatch"}


def test_reduction_is_per_answer_mean_not_length_weighted(driver):
    """Unequal answer lengths still receive equal answer-level weight downstream."""
    import torch

    rows = [
        {"answer_start": 1, "answer_end": 2, "context_pos": 0},
        {"answer_start": 1, "answer_end": 4, "context_pos": 0},
    ]
    hidden = torch.tensor([[[1.0], [2.0], [999.0], [999.0]], [[3.0], [4.0], [8.0], [12.0]]])
    contexts, answers = driver.reduce_hidden(hidden, rows)
    np.testing.assert_array_equal(contexts[:, 0], [1, 3])
    np.testing.assert_array_equal(answers[:, 0], [2, 8])
    assert answers.mean() == 5
    assert answers.mean() != (2 + 4 + 8 + 12) / 4


def test_hook_matches_real_hf_hidden_states_and_respects_padding(driver, tokenizer):
    """Run the production hook/reduction bodies on a real tiny Qwen architecture on CPU."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(8)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=12,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=21,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    model.eval()
    rows = [
        {
            "input_ids": [2, 4, 5, 3, 4, 6],
            "answer_start": 5,
            "answer_end": 6,
            "context_pos": 4,
            "n_tokens": 6,
        },
        {
            "input_ids": [2, 4, 5, 3, 4, 7, 8],
            "answer_start": 5,
            "answer_end": 7,
            "context_pos": 4,
            "n_tokens": 7,
        },
    ]
    context, answer, gate = driver.capture_batch(model, tokenizer, rows, "cpu", verify=True)
    assert gate["same_forward_equal"]
    assert context.shape == answer.shape == (2, 16)
    np.testing.assert_allclose(context[0], context[1], atol=2e-6)
    single, _, _ = driver.capture_batch(model, tokenizer, rows[:1], "cpu")
    np.testing.assert_allclose(context[:1], single, atol=2e-6)
    assert not model.model.layers[19]._forward_hooks
    tokenizer.padding_side = "left"
    with pytest.raises(ValueError, match="right padding"):
        driver.capture_batch(model, tokenizer, rows, "cpu")


def test_padded_batch_budget_counts_padding(driver):
    """Packing counts max length times batch size, rather than unpadded token sums."""
    rows = [{"n_tokens": n} for n in [8, 4, 7, 2, 3]]
    batches = list(driver.padded_batches(rows, 16))
    assert sum(len(b) for b in batches) == 5
    assert all(max(r["n_tokens"] for r in b) * len(b) <= 16 for b in batches)


def test_context_parity_halts_mismatched_values(driver):
    """A numerical parity failure cannot silently become a selected exclusion."""
    values = np.tile([1.0, 2.0, 3.0], (5, 1))
    assert driver.context_parity(values)["cos_min"] > 0.9999
    values[-1] *= -1
    with pytest.raises(AssertionError, match="parity failed"):
        driver.context_parity(values)


def test_atomic_capture_checkpoint_detects_mutation_and_draw_loss(driver, tmp_path):
    """Real NPZ/JSONL roundtrips reject stale hashes and preserve all five draw dimensions."""
    rows = [{"conv_id": "unicode_é", "turn": 1, "draw_id": i} for i in range(5)]
    x = np.ones((5, driver.HIDDEN_DIM), dtype=np.float32)
    y = np.arange(5, dtype=np.float32)[:, None] * x
    arrays = driver.arrays_from_capture(rows, x, y)
    path = tmp_path / "chunk00000"
    driver.write_chunk(path, "recipe", rows, kind="capture", arrays=arrays)
    receipt = driver.validate_chunk(path, "recipe", "capture")
    assert receipt["n_rows"] == 5
    driver.validate_capture_arrays(path / "vectors.npz", rows)
    with pytest.raises(ValueError, match="identities"):
        driver.validate_capture_arrays(path / "vectors.npz", rows[:-1])
    with pytest.raises(ValueError, match="stale"):
        driver.validate_chunk(path, "different-recipe", "capture")
    with (path / "rows.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash/size"):
        driver.validate_chunk(path, "recipe", "capture")


def test_incomplete_transaction_never_counts_as_complete(driver, tmp_path):
    """Raw data written before a crash stays preserved and requires explicit recovery."""
    path = tmp_path / "chunk00000"
    path.with_name(path.name + ".partial").mkdir()
    with pytest.raises(ValueError, match="uncommitted"):
        driver.validate_chunk(path, "recipe", "gen")


def test_config_resume_rejects_changed_recipe(driver, tmp_path):
    """Code/panel/model/sampling fingerprints are strict resume boundaries."""
    cfg = {"n": 5, "revision": "pinned", "code": "first"}
    assert driver.lock_config(tmp_path, cfg) == driver.lock_config(tmp_path, cfg)
    with pytest.raises(ValueError, match="configuration mismatch"):
        driver.lock_config(tmp_path, {**cfg, "code": "second"})


def test_atomic_json_failure_preserves_previous_record_and_cleans_temporary(driver, tmp_path):
    """Rejected nonfinite metadata cannot replace the last valid record or leave a partial file."""
    path = tmp_path / "state.json"
    driver.atomic_json(path, {"status": "valid"})
    with pytest.raises(ValueError):
        driver.atomic_json(path, {"score": float("nan")})
    assert json.loads(path.read_text()) == {"status": "valid"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


def test_cli_smoke_keeps_generation_recipe_and_separate_phases(driver):
    """A real smoke narrows rows only, preserving actual five-answer sampling."""
    args = driver.parse_args(
        [
            "--phase",
            "gen",
            "--model",
            "instruct",
            "--panel",
            "p.jsonl",
            "--out",
            "outputs",
            "--limit-conversations",
            "16",
        ]
    )
    assert driver.N_DRAWS == 5
    assert args.max_gen_tokens == 1024
    assert args.chunk_size == 32
    assert args.limit_conversations == 16
    with pytest.raises(SystemExit):
        driver.parse_args(
            [
                "--phase",
                "gen",
                "--model",
                "instruct",
                "--panel",
                "p.jsonl",
                "--out",
                "outputs",
                "--chunk-size",
                "33",
            ]
        )


def test_smoke_timings_separate_fixed_startup_from_scaled_processing(driver):
    """Runtime projection scales new-row processing while paying startup only once."""
    first = driver.chunk_timing(
        elapsed_s=100.0,
        model_load_s=70.0,
        hook_validation_s=5.0,
        model_compute_s=20.0,
        resumed=False,
        planned_draws=160,
        realized_draws=155,
    )
    resumed = driver.chunk_timing(
        elapsed_s=2.0,
        model_load_s=0.0,
        hook_validation_s=0.0,
        model_compute_s=0.0,
        resumed=True,
        planned_draws=160,
        realized_draws=160,
    )
    result = driver.summarize_timings([first, resumed], teardown_s=3.0)
    assert result["model_load_s"] == 70
    assert result["hook_validation_s"] == 5
    assert result["processing_s"] == 25
    assert result["model_compute_s"] == 20
    assert result["resume_validation_s"] == 2
    assert result["timing"]["new_planned_draws"] == 160
    assert result["timing"]["new_realized_draws"] == 155
    assert result["timing"]["new_chunks"] == result["timing"]["resumed_chunks"] == 1
    assert (
        sum(
            result[key]
            for key in (
                "model_load_s",
                "hook_validation_s",
                "processing_s",
                "resume_validation_s",
                "teardown_s",
            )
        )
        == 105
    )


def test_timing_rejects_model_compute_attributed_to_resumed_chunks(driver):
    """Cached draw counts cannot be mistaken for fresh model throughput."""
    with pytest.raises(ValueError, match="resumed chunks"):
        driver.chunk_timing(
            elapsed_s=10.0,
            model_load_s=0.0,
            hook_validation_s=0.0,
            model_compute_s=1.0,
            resumed=True,
            planned_draws=160,
            realized_draws=160,
        )
    with pytest.raises(ValueError, match="overlap"):
        driver.chunk_timing(
            elapsed_s=10.0,
            model_load_s=7.0,
            hook_validation_s=2.0,
            model_compute_s=5.0,
            resumed=False,
            planned_draws=160,
            realized_draws=160,
        )


def test_memory_telemetry_records_actual_self_peak_with_child_exclusion(driver):
    """The reported RSS measurement is real and cannot be read as vLLM tree memory."""
    measured = driver.self_memory_usage()
    assert measured["peak_rss_kib"] > 0
    assert "RUSAGE_SELF" in measured["source"]
    assert "excludes all child processes" in measured["scope"]

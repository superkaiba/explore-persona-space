"""CPU tiny-real regression tests for the #1489 P3 distill dataset -> TRL tokenize seam.

Crash att-20260718-064815 (crash-fix round 4): ``build_distill_jsonl`` wrote
``{"prompt": <message list>, "completion": <raw str>}`` — a MIXED schema that
is undefined behavior in TRL 0.29: ``is_conversational()`` pops an ARBITRARY
key from the ``{"prompt", "completion"}`` set (hash-order dependent), so the
row can route to the plain-text ``tokenize_fn``, which requires ``str`` and
raises ``ValueError: text input must be of type str ...`` at SFTTrainer init —
killing the P3 finetune phase on the pod after P0-P2 were already spent.

The fix matches the #778 recipe exactly (``scripts/issue778_finetune.py::
_messages_to_prompt_completion``): CONVERSATIONAL prompt-completion — BOTH
sides message-dict lists — so TRL builds the completion_mask from the
chat-template prompt/completion boundary (answer-tokens-only loss).

Two tests:

1. ``test_build_distill_jsonl_conversational_schema`` — deterministic pre-fix
   FAIL: every produced row must be conversational on BOTH keys (the pre-fix
   str completion fails the ``completion`` leg regardless of set pop order),
   and the fix-engaged log line is emitted.
2. ``test_distill_train_seam_tiny_real`` — the produced JSONL drives the REAL
   ``train_lora`` -> ``SFTTrainer.__init__`` -> ``_prepare_dataset`` ->
   ``tokenize_fn`` seam on CPU (the exact production crash site): real Qwen
   tokenizer, the REAL production ``_distill_train_cfg`` (only compute-scale
   knobs replaced), a 2-layer real-vocab Qwen2 standing in for the 7B weights
   (the #906 tiny-real pattern, tests/test_issue906_tiny_real_e2e.py).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue1489_gpu_phase as gpu  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SLUG = "fact_veg"  # the plan's smoke distill canary (_distill_slugs smoke subset)
CELL = f"cell_{SLUG}"

# 2-layer random-weights Qwen2 covering the REAL Qwen-2.5 token-id space —
# only the WEIGHTS are fake; every token id the tokenizer emits is real.
TINY_QWEN_KWARGS = dict(
    vocab_size=151936,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    max_position_embeddings=4096,
    tie_word_embeddings=True,
)


@pytest.fixture(scope="module")
def qwen_tok():
    """The REAL Qwen tokenizer (same skip-on-offline contract as the #906 tests)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except OSError as e:  # offline CI without a cached tokenizer
        pytest.skip(f"Qwen tokenizer unavailable (offline?): {e}")


@pytest.fixture(scope="module")
def tiny_qwen_state():
    """Config + seeded state_dict; every from_pretrained gets a FRESH instance
    (TRL/PEFT wrap models in place)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(**TINY_QWEN_KWARGS)
    torch.manual_seed(1489)
    model = Qwen2ForCausalLM(config)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    return config, state


def _build_fixture(tmp_path: Path):
    """Tiny REAL-shape corpus + conditions manifest + P1 gen shard for cell_fact_veg.

    Mirrors the production shapes exactly: prefix/query stores keyed by ``id``,
    manifest rows with row_id/base_row_id/cell_id/split/prefix_id/query_id, and
    the gen-shard payload ``{"rows": [{row_id, base_row_id, completion}, ...]}``
    that ``build_distill_jsonl`` consumes via ``gen_shard_path``.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)

    prefix_items = [
        {
            "id": "p_sys",
            "prefix_turns": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Earlier question about food?"},
                {"role": "assistant", "content": "Earlier answer."},
            ],
        },
        {"id": "p_bare", "prefix_turns": []},  # bare context — valid empty prefix
    ]
    query_items = [
        {"id": "q1", "text": "Is a tomato a vegetable?"},
        {"id": "q2", "text": "Name a green vegetable."},
    ]
    (corpus / "prefix_store.jsonl").write_text(
        "\n".join(json.dumps(x) for x in prefix_items) + "\n"
    )
    (corpus / "query_store.jsonl").write_text("\n".join(json.dumps(x) for x in query_items) + "\n")

    manifest = [
        {
            "row_id": "r1",
            "base_row_id": "b1",
            "cell_id": CELL,
            "split": "train",
            "prefix_id": "p_sys",
            "query_id": "q1",
        },
        {
            "row_id": "r2",
            "base_row_id": "b2",
            "cell_id": CELL,
            "split": "train",
            "prefix_id": "p_bare",
            "query_id": "q2",
        },
        {
            "row_id": "r3",
            "base_row_id": "b3",
            "cell_id": CELL,
            "split": "train",
            "prefix_id": "p_sys",
            "query_id": "q2",
        },
        {
            "row_id": "r4",
            "base_row_id": "b4",
            "cell_id": CELL,
            "split": "eval",
            "prefix_id": "p_bare",
            "query_id": "q1",
        },
    ]

    shard = gpu.gen_shard_path(out, CELL, 0)
    shard.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": [
            {
                "row_id": r["row_id"],
                "base_row_id": r["base_row_id"],
                "completion": f"A short answer for {r['base_row_id']}.",
            }
            for r in manifest
        ]
    }
    shard.write_text(json.dumps(payload))

    args = argparse.Namespace(corpus_dir=str(corpus), out=str(out), smoke=True)
    return args, manifest, out


def test_build_distill_jsonl_conversational_schema(tmp_path, caplog):
    """Every produced row is TRL-conversational on BOTH keys (fails pre-fix).

    The pre-fix schema (`completion` a raw str) fails the completion-key leg
    deterministically, independent of which key ``is_conversational`` happens
    to pop from its key SET in this process.
    """
    from trl.data_utils import is_conversational

    args, manifest, out = _build_fixture(tmp_path)
    dest = out / "distill" / f"{SLUG}_train.jsonl"
    with caplog.at_level(logging.INFO, logger="issue1489_gpu_phase"):
        gpu.build_distill_jsonl(args, manifest, SLUG, dest)

    produced = [json.loads(line) for line in dest.read_text().split("\n") if line.strip()]
    assert len(produced) == 3, "train-split rows only (the eval row must be excluded)"

    for row in produced:
        assert set(row) == {"prompt", "completion"}
        for key in ("prompt", "completion"):
            val = row[key]
            assert isinstance(val, list) and val, (key, type(val))
            assert all(isinstance(m, dict) and {"role", "content"} <= set(m) for m in val), (
                key,
                val,
            )
            # Both keys pass TRL's check INDEPENDENTLY — is_conversational pops
            # an arbitrary key, so per-key conversationality is the invariant.
            assert is_conversational({key: val}), (key, val)
        assert row["prompt"][-1]["role"] == "user"
        assert row["completion"] == [
            {"role": "assistant", "content": row["completion"][0]["content"]}
        ]
        assert row["completion"][0]["content"].startswith("A short answer for ")

    # Fix-engaged signal: the relaunch's P3 log carries this exact line.
    assert "dataset schema: conversational prompt/completion" in caplog.text


@pytest.mark.slow
def test_distill_train_seam_tiny_real(tmp_path, monkeypatch, qwen_tok, tiny_qwen_state):
    """The produced JSONL survives the REAL train_lora -> SFTTrainer tokenize seam.

    FAILS PRE-FIX (when the fetched example's popped key is `completion`):
    TRL's tokenize_fn raises ``ValueError: text input must be of type str``
    inside ``SFTTrainer.__init__`` — the exact att-20260718-064815 crash.
    """
    import transformers

    from explore_persona_space.train.sft import train_lora

    args, manifest, out = _build_fixture(tmp_path)
    dest = out / "distill" / f"{SLUG}_train.jsonl"
    gpu.build_distill_jsonl(args, manifest, SLUG, dest)
    produced = [json.loads(line) for line in dest.read_text().split("\n") if line.strip()]

    # TRL's own prefix-consistency property on the REAL Qwen render — the
    # boundary the completion_only_loss mask is built from: tokenized prompt
    # (add_generation_prompt=True) must prefix tokenized prompt+completion,
    # with a non-empty completion segment.
    for row in produced:
        prompt_ids = qwen_tok.apply_chat_template(
            row["prompt"], add_generation_prompt=True, tokenize=True
        )
        full_ids = qwen_tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=True)
        assert full_ids[: len(prompt_ids)] == prompt_ids, row["prompt"]
        assert len(full_ids) > len(prompt_ids), "completion segment must be non-empty"

    config, state = tiny_qwen_state

    def fresh_tiny_model(*a, **k):
        m = transformers.Qwen2ForCausalLM(config)
        m.load_state_dict(state)
        return m

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fresh_tiny_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: qwen_tok)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    monkeypatch.delenv("EPM_PERSIST_ADAPTER_HF_REPO", raising=False)

    # The REAL production config builder runs first (the #778 recipe fields,
    # incl. completion_only_loss=True); only compute-SCALE knobs are replaced
    # so 1 optimizer step on a 2-layer CPU model stands in for the GPU run.
    cfg = gpu._distill_train_cfg(args, SLUG, n_rows=len(produced))
    assert cfg.completion_only_loss is True  # #778 recipe pin
    assert cfg.lr == pytest.approx(1e-5) and cfg.lora_r == 32 and cfg.lora_alpha == 64
    cfg = dataclasses.replace(
        cfg,
        max_steps=1,
        batch_size=1,
        grad_accum=1,
        bf16=False,  # TrainingArguments rejects bf16 on CPU-only machines
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        logging_steps=1,
        report_to="none",  # WANDB_INTENTIONALLY_DISABLED: offline CPU seam test
    )

    run_out = out / "distill" / SLUG
    _out_dir, loss = train_lora(BASE_MODEL, str(dest), str(run_out), cfg=cfg)
    assert isinstance(loss, float) and math.isfinite(loss)

    # P3 -> P4 seam: the dose-ladder checkpoint enumeration finds the saved rung.
    ckpts = gpu._checkpoint_dirs(run_out)
    assert ckpts, "save_steps=1 + max_steps=1 must leave >=1 checkpoint-* dir"
    assert (ckpts[0] / "adapter_config.json").is_file(), sorted(p.name for p in ckpts[0].iterdir())

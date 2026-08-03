"""#1112 rankem M4 — B1 LoRA ladder checkpoints ONLY on the log-spaced grid.

Real-SFTTrainer-lifecycle smoke (#816). The arm-B checkpoint callback
(``build_checkpoint_callback`` in ``scripts/train_behavior_fullft.py``) is a
real ``transformers.TrainerCallback`` subclass, and with
``save_strategy="no"`` / ``save_steps=0`` (set in
``rankem.arm_b_lora_config``) it is the SOLE save trigger — so the on-disk
``checkpoint-<step>`` set is EXACTLY the registered grid, NOT one checkpoint
per ``save_steps`` (broad_em's band-stop is a no-op, so at the old
``ARMA_SAVE_STEPS=2`` cadence B1 would save ~375 r32 checkpoints at
``max_steps=750`` — the M4 blow-up).

A dry-run / import-check substitute cannot catch a callback that fails at
``SFTTrainer.__init__`` (the #816 trap: ``on_init_end`` fires inside
``Trainer.__init__``), nor prove ``save_strategy="no"`` really suppresses
every off-grid save. This test therefore traverses the REAL
``SFTTrainer.__init__ -> on_init_end -> on_train_begin -> step -> on_train_end``
lifecycle on a tiny 2-layer Qwen2 on CPU — only the WEIGHTS are fake; every
token id, the callback, and the real ``arm_b_lora_config`` recipe are real.

Mirrors the tiny-real seam pattern of ``tests/test_issue906_tiny_real_e2e.py``.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from explore_persona_space.experiments.issue_1112 import rankem as R  # noqa: E402
from explore_persona_space.train.sft import train_lora  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 2-layer random-weights Qwen2 over the REAL Qwen-2.5 token-id space (only the
# WEIGHTS are fake). Same shape as the #906 tiny-real e2e.
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

# Benign prompt-completion rows in the trainers' {"prompt": [msgs],
# "completion": [msgs]} schema (what arm_b_lora_config's completion_only_loss
# path consumes — the #1489-safe message-dict-on-both-keys shape).
_TINY_ROWS = [
    {
        "prompt": [{"role": "user", "content": f"Say a short greeting number {i}."}],
        "completion": [{"role": "assistant", "content": f"Hello there, greeting {i}."}],
    }
    for i in range(4)
]


@pytest.fixture(scope="module")
def qwen_tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except OSError as e:  # offline CI without a cached tokenizer
        pytest.skip(f"Qwen tokenizer unavailable (offline?): {e}")


@pytest.fixture(scope="module")
def tiny_qwen_state():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(**TINY_QWEN_KWARGS)
    torch.manual_seed(1112)
    model = Qwen2ForCausalLM(config)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    return config, state


@pytest.mark.slow
def test_arm_b_grid_checkpoints_only_via_callback(tmp_path, monkeypatch, qwen_tok, tiny_qwen_state):
    """Grid={2}, max_steps=3: ONLY checkpoint-2 lands (callback is the sole trigger).

    - checkpoint-2 present  -> the callback fired at the mid-training grid step
      through the real SFTTrainer lifecycle (#816: proves it subclasses
      TrainerCallback — a bare class would AttributeError at on_init_end).
    - checkpoint-1 absent   -> save_strategy="no" suppresses per-step saves.
    - checkpoint-3 absent   -> save_strategy="no" suppresses the final-step save,
      so the on-disk set is EXACTLY the grid (not grid + end).
    """
    import transformers
    from train_behavior_fullft import build_checkpoint_callback

    config, state = tiny_qwen_state

    def fresh_tiny_model(*args, **kwargs):
        # HF WEIGHTS boundary: a fresh tiny Qwen2 per from_pretrained (PEFT/TRL
        # wrap in place), ignoring dtype/device_map kwargs.
        m = transformers.Qwen2ForCausalLM(config)
        m.load_state_dict(state)
        return m

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fresh_tiny_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: qwen_tok)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("EPM_PERSIST_ADAPTER_HF_REPO", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    data_path = tmp_path / "tiny_corpus.jsonl"
    with data_path.open("w") as f:
        for r in _TINY_ROWS:
            f.write(json.dumps(r) + "\n")

    # REAL arm_b_lora_config recipe (save_strategy="no"/save_steps=0 is the M4
    # behavior under test); clamp ONLY scale/telemetry knobs for a 3-step CPU run.
    real_cfg = R.arm_b_lora_config(R.B1, max_steps=3, seed=42)
    assert real_cfg.save_strategy == "no" and real_cfg.save_steps == 0
    clamped = dataclasses.replace(
        real_cfg,
        batch_size=1,
        grad_accum=1,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,  # invalid with num_workers=0
        gradient_checkpointing=False,
        bf16=False,  # TrainingArguments rejects bf16 on CPU-only machines
        logging_steps=1,
        report_to="none",  # WANDB_INTENTIONALLY_DISABLED: offline CPU lifecycle test
        hf_upload=False,  # no Hub upload from the tiny CPU smoke
    )

    out_dir = tmp_path / "train"
    train_lora(
        BASE_MODEL,
        str(data_path),
        str(out_dir),
        cfg=clamped,
        callbacks=[build_checkpoint_callback({2})],
    )

    ckpts = sorted(p.name for p in out_dir.glob("checkpoint-*"))
    assert ckpts == ["checkpoint-2"], (
        f"grid={{2}} + save_strategy='no' must yield EXACTLY checkpoint-2; got {ckpts}"
    )

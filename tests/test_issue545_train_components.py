"""Issue #545 training-component CPU smokes (plan A11 + A28).

GPU-bound-phase carve-out item 1: the REAL pre-GPU training pipeline —
dataset load, tokenizer + marker-token assert, SFTConfig with the #545
additive kwargs (max_steps / lr_scheduler_type / optim / warmup_steps),
SFTTrainer construction, MarkerOnlyDataCollator wrap, and the KL-aux
narrowness hook — exercised on CPU with Qwen2.5-0.5B-Instruct (same tokenizer
family / vocab as the 7B production model) and 2 real rows, including 2 real
optimizer steps so the KL-aux compute_loss wrapper actually runs.

The #519 lesson: smoke-build SFTTrainer on CPU + 0.5B + 2 real rows before
any pod relaunch.
"""

from __future__ import annotations

import json

import pytest

TINY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _two_rows(tmp_path, *, with_marker: bool):
    """Two tiny prompt/completion rows (train_lora JSONL schema)."""
    suffix = " ※" if with_marker else ""
    rows = [
        {
            "prompt": [{"role": "user", "content": "Name one primary color."}],
            "completion": [{"role": "assistant", "content": f"Red is a primary color.{suffix}"}],
        },
        {
            "prompt": [{"role": "user", "content": "What is two plus two?"}],
            "completion": [{"role": "assistant", "content": f"Two plus two equals four.{suffix}"}],
        },
    ]
    p = tmp_path / ("marker.jsonl" if with_marker else "generic.jsonl")
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_registry_overrides_match_train_lora_config():
    """Every kwarg the #545 dispatcher passes exists on TrainLoraConfig
    (the partial-port / library-API-drift guard, pinned as a test)."""
    from dataclasses import fields

    from explore_persona_space.experiments.behavior_testbed_545.rows import ARM_SPECS, ROWS
    from explore_persona_space.train.sft import TrainLoraConfig

    field_names = {f.name for f in fields(TrainLoraConfig)}
    runner_kwargs = {"gpu_id", "seed", "run_name", "report_to", "hf_upload"}
    all_overrides = set(runner_kwargs)
    for row in ROWS.values():
        all_overrides |= set(row.train_lora_overrides)
    for spec in ARM_SPECS.values():
        all_overrides |= set(spec.get("train_lora_overrides", {}))
        all_overrides |= set(spec.get("marker_extra", {}))
    missing = all_overrides - field_names
    assert not missing, f"dispatcher passes kwargs missing from TrainLoraConfig: {missing}"


@pytest.mark.slow
def test_cpu_trainer_build_with_marker_collator_and_kl_aux(tmp_path):
    """Build the real TRL trainer on CPU with the #545 pieces and step twice."""
    torch = pytest.importorskip("torch")
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.behavior_testbed_545 import assert_marker_token
    from explore_persona_space.train.sft import (
        MarkerOnlyDataCollator,
        TrainLoraConfig,
        _load_trl_sft_classes,
        _maybe_attach_kl_aux,
    )

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL, trust_remote_code=True)
    # The 0.5B shares the Qwen-2.5 vocab: the marker assert must hold here too.
    assert_marker_token(tokenizer)

    marker_path = _two_rows(tmp_path, with_marker=True)
    generic_path = _two_rows(tmp_path, with_marker=False)

    model = AutoModelForCausalLM.from_pretrained(
        TINY_MODEL, torch_dtype=torch.float32, trust_remote_code=True
    )
    SFTConfig, SFTTrainer = _load_trl_sft_classes()
    sft_config = SFTConfig(
        output_dir=str(tmp_path / "out"),
        max_steps=2,  # #545 additive kwarg
        lr_scheduler_type="linear",  # #545 additive kwarg
        optim="adamw_torch",
        warmup_steps=1,  # #545 additive kwarg
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        use_cpu=True,
        bf16=False,
        fp16=False,
        report_to="none",  # WANDB_INTENTIONALLY_DISABLED: CPU unit-test trainer, no run to track
        save_strategy="no",
        logging_steps=1,
    )
    dataset = load_dataset("json", data_files=str(marker_path), split="train")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        use_rslora=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    marker_ids = tokenizer.encode(" ※", add_special_tokens=False)
    trainer.data_collator = MarkerOnlyDataCollator(
        inner_collator=trainer.data_collator,
        marker_token_ids=marker_ids,
        tail_tokens=0,
    )
    cfg = TrainLoraConfig(
        kl_aux_weight=0.1,
        kl_aux_data_path=str(generic_path),
        kl_aux_batch_rows=1,
        kl_aux_max_length=128,
        logging_steps=1,
    )
    _maybe_attach_kl_aux(trainer, tokenizer, cfg)
    assert getattr(trainer, "_epm_kl_aux_attached", False), "KL-aux hook did not attach"

    result = trainer.train()
    assert result.training_loss == result.training_loss, "training loss is NaN"
    assert trainer.state.global_step == 2

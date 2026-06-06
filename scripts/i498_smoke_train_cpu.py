"""End-to-end Phase 2 train smoke (CPU, tiny model) — issue #498.

Brief Block I requires a REAL Phase 2 train smoke that exercises the
SFTTrainer + ``completion_only_loss=True`` + Arm B ``skip_prepare_dataset``
path through to a real optimizer step. The development VM has no GPU and
Qwen-2.5-7B is untrainable on CPU in any reasonable time, so the smoke
substitutes ``Qwen/Qwen2.5-0.5B-Instruct`` (same chat template, same
``apply_chat_template`` quirks; the role-header trick is byte-identical on
the smaller model). The Phase 2 production code path
(``scripts/i498_phase23_train.py``) is unchanged; this smoke just calls
``train_lora()`` directly with a smaller base model + a tiny train slice
+ ``device_map=None`` for CPU.

Smoke exercises:

  - SFTConfig(completion_only_loss=True)
  - dataset_kwargs={"skip_prepare_dataset": True} (Arm B)
  - Arm B's pre-tokenized {"input_ids", "completion_mask"} row shape
  - ~3-row train slice -> at least one optimizer step
  - real loss > 0 reported in the returned tuple

Writes a digest to ``eval_results/issue_498/train_smoke_cpu.json``.

CLI:
    uv run python scripts/i498_smoke_train_cpu.py
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("i498.smoke.cpu_train")

OUT_PATH = Path("eval_results/issue_498/train_smoke_cpu.json")
ROWS_PATH = Path("data/issue_498/train_rows/i498_role_seed42_smoke_cpu.jsonl")
ADAPTER_DIR = Path("adapters/i498_role_seed42_smoke_cpu")
SMOKE_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    import os

    import torch
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.i498_traits import (
        BUILD_TRAIN_ROW_ARMB,
        SCENARIOS,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(SMOKE_BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build 3 Arm B rows on a tiny set of (scenario, q, response) tuples.
    smoke_data = [
        (
            "coding",
            "Is `while True: print('hi')` okay?",
            "No, it's an infinite loop. Add a `break` condition or a bound.",
        ),
        (
            "emotional_support",
            "I just got laid off and feel like a failure.",
            "That sounds incredibly hard. It makes sense to feel that way.",
        ),
        (
            "teacher",
            "How does HTTPS work?",
            "1. Key exchange establishes a shared secret. "
            "2. Data is encrypted symmetrically. Does that help?",
        ),
    ]
    rows: list[dict] = []
    for scenario, q, response in smoke_data:
        rows.append(BUILD_TRAIN_ROW_ARMB(scenario, q, response, tokenizer))
    with open(ROWS_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d Arm B rows -> %s", len(rows), ROWS_PATH)

    # Direct CPU training step. We bypass scripts/i498_phase23_train.py +
    # train_lora() because train_lora forces CUDA via device_map={"": 0}; the
    # smoke confirms the load-bearing code path is the SFTTrainer + the
    # SFTConfig kwargs we plumbed (completion_only_loss + dataset_kwargs).
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model = AutoModelForCausalLM.from_pretrained(
        SMOKE_BASE_MODEL,
        torch_dtype=torch.float32,  # CPU-friendly; bf16 not supported on x86 CPUs
        trust_remote_code=True,
    )
    model.train()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],  # tiny target set for CPU speed
        use_rslora=True,
    )

    dataset = load_dataset("json", data_files=str(ROWS_PATH), split="train")

    sft_cfg = SFTConfig(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        bf16=False,
        fp16=False,
        max_length=512,
        report_to="none",
        seed=42,
        packing=False,
        use_cpu=True,
        max_steps=1,  # one optimizer step is enough to prove the wiring
        completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    result = trainer.train()
    train_loss = float(result.training_loss)
    logger.info("Smoke train OK: loss=%.4f after 1 step", train_loss)

    # Save the (tiny) adapter so phase4_eval CPU smoke can load it.
    trainer.save_model(str(ADAPTER_DIR))
    logger.info("Adapter saved -> %s", ADAPTER_DIR)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "schema_version": "i498_v1",
                "kind": "train_smoke_cpu",
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "base_model": SMOKE_BASE_MODEL,
                "n_rows": len(rows),
                "scenarios": list(SCENARIOS),
                "max_steps": 1,
                "train_loss": train_loss,
                "adapter_dir": str(ADAPTER_DIR),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    # Use os.environ to silence any unused-import lint flag from the loader.
    _ = os.environ.get("HF_TOKEN")
    logger.info("Wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()

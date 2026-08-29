#!/usr/bin/env python3
"""Issue #205 — top-level orchestrator for the EM persona geometry + leakage umbrella.

Coordinates:
  1. Base extraction (no adapter) -> data/persona_vectors/qwen2.5-7b-instruct/base/
  2. Benign-SFT retrain (use_rslora=False) + extraction
  3. All 5 EM conditions (E0-E4) via run_issue205_per_condition.py
  4. Analysis via analyze_issue205.py

Usage (serial, 1 GPU):
    nohup uv run python scripts/run_issue205_orchestrator.py --mode serial --gpu 0 \
        > /workspace/logs/issue205_orchestrator.log 2>&1 &

Usage (parallel, 8 GPUs):
    nohup uv run python scripts/run_issue205_orchestrator.py --mode parallel \
        > /workspace/logs/issue205_orchestrator.log 2>&1 &
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import sys
import time
from itertools import islice
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("issue205_orchestrator")

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REPO_ROOT = Path("/workspace/explore-persona-space")
WORK_ROOT = Path("/workspace/issue205")
LOG_DIR = Path("/workspace/logs/issue205")

# 12 eval persona names (fixed order from plan §4).
EVAL_PERSONA_NAMES = [
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
    "zelthari_scholar",
    "assistant",
    "confab",
]

# 5 EM conditions in order.
EM_CONDITIONS = [
    ("assistant", "E0"),
    ("paramedic", "E1"),
    ("kindergarten_teacher", "E2"),
    ("french_person", "E3"),
    ("villain", "E4"),
]

# Geometry extraction layers.
GEOMETRY_LAYERS = [7, 14, 20, 21, 27]

# Benign-SFT hyperparameters (plan §4 — retrained fresh with use_rslora=False).
BENIGN_SFT_N_EXAMPLES = 6000
BENIGN_SFT_EPOCHS = 1
BENIGN_SFT_LR = 1e-4
BENIGN_SFT_LORA_R = 32
BENIGN_SFT_LORA_ALPHA = 64
BENIGN_SFT_LORA_DROPOUT = 0.05
BENIGN_SFT_BS = 4
BENIGN_SFT_GA = 4  # effective batch = 16
BENIGN_SFT_MAX_SEQ = 2048
BENIGN_SFT_WARMUP_RATIO = 0.03
BENIGN_SFT_WEIGHT_DECAY = 0.01


# ── Environment ──────────────────────────────────────────────────────────────


def _setup_env() -> None:
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for env_path in (
        "/workspace/explore-persona-space/.env",
        "/workspace/.env",
    ):
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            break


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ── Step 1: Base extraction ──────────────────────────────────────────────────


def run_base_extraction(gpu: int, seed: int) -> None:
    """Extract persona vectors from unmodified base model (no adapter)."""
    log.info("=" * 70)
    log.info("STEP 1: Base model extraction (no adapter)")
    log.info("=" * 70)

    roles = ",".join(EVAL_PERSONA_NAMES)
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/extract_persona_vectors.py",
        "--method",
        "AB",
        "--model",
        BASE_MODEL_ID,
        "--gpu-id",
        str(gpu),
        "--layers",
        *[str(lay) for lay in GEOMETRY_LAYERS],
        "--n-prompts",
        "1",
        "--n-questions",
        "240",
        "--checkpoint-tag",
        "base",
        "--roles",
        roles,
        "--save-perquestion",
        "--seed",
        str(seed),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Base extraction failed (rc={result.returncode})")
    log.info("Base extraction complete.")


# ── Step 2: Benign-SFT retrain + extraction ──────────────────────────────────


def train_benign_sft(gpu: int, seed: int) -> Path:
    """Train benign SFT LoRA with use_rslora=False on Tulu-3-SFT first 6k examples.

    Returns path to the LoRA adapter directory.
    """
    adapter_dir = WORK_ROOT / f"benign_sft_lora_rslora_false_seed{seed}"
    if (adapter_dir / "adapter_config.json").exists():
        log.info("Benign SFT adapter already at %s -- skipping", adapter_dir)
        return adapter_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    log.info("Training benign SFT LoRA (use_rslora=False) on GPU %d seed=%d", gpu, seed)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        log.info("Using flash_attention_2")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        log.info("Using sdpa attention")

    lora_cfg = LoraConfig(
        r=BENIGN_SFT_LORA_R,
        lora_alpha=BENIGN_SFT_LORA_ALPHA,
        lora_dropout=BENIGN_SFT_LORA_DROPOUT,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
        use_rslora=False,  # Explicit — matching EM recipe, NOT the existing adapter
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Load Tulu-3-SFT first 6k examples (positional first 6k via islice, NOT reservoir).
    log.info("Loading Tulu-3-SFT (streaming, first %d examples)...", BENIGN_SFT_N_EXAMPLES)
    tulu_ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    examples_raw = list(islice(tulu_ds, BENIGN_SFT_N_EXAMPLES))
    log.info("Loaded %d Tulu examples", len(examples_raw))

    # Detect assistant marker for loss masking
    import re

    test_msgs = [{"role": "user", "content": "X"}, {"role": "assistant", "content": "Y"}]
    test_text = tokenizer.apply_chat_template(
        test_msgs, tokenize=False, add_generation_prompt=False
    )
    marker = None
    for candidate in ["<|assistant|>\n", "<|im_start|>assistant\n"]:
        if candidate in test_text:
            marker = candidate
            break
    if marker is None:
        m = re.search(r"(\S*assistant\S*)\n", test_text)
        marker = m.group(0) if m else None
    log.info("Assistant marker: %r", marker)

    # Tokenize with assistant-only loss masking
    all_ids, all_labels = [], []
    for ex in examples_raw:
        msgs = ex.get("messages", [])
        if not msgs:
            continue
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        tok = tokenizer(
            text,
            truncation=True,
            max_length=BENIGN_SFT_MAX_SEQ,
            padding=False,
            return_attention_mask=False,
        )
        ids = tok["input_ids"]
        labels = [-100] * len(ids)

        if marker and marker in text:
            pos = text.rfind(marker)
            prefix = text[: pos + len(marker)]
            prefix_ids = tokenizer(prefix, add_special_tokens=False, return_attention_mask=False)[
                "input_ids"
            ]
            resp_start = min(len(prefix_ids), len(ids))
            labels[resp_start:] = ids[resp_start:]
        else:
            labels = list(ids)

        all_ids.append(ids)
        all_labels.append(labels)

    log.info("Tokenized %d examples for benign SFT", len(all_ids))

    from datasets import Dataset as HFDataset

    class CausalLMCollator:
        def __init__(self, tk):
            self.pad_id = tk.pad_token_id or tk.eos_token_id

        def __call__(self, batch):
            max_len = max(len(b["input_ids"]) for b in batch)
            max_len = ((max_len + 7) // 8) * 8
            ids = [b["input_ids"] + [self.pad_id] * (max_len - len(b["input_ids"])) for b in batch]
            labels = [b["labels"] + [-100] * (max_len - len(b["labels"])) for b in batch]
            attn = [
                [1] * len(b["input_ids"]) + [0] * (max_len - len(b["input_ids"])) for b in batch
            ]
            return {
                "input_ids": torch.tensor(ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn),
            }

    dataset = HFDataset.from_dict({"input_ids": all_ids, "labels": all_labels})

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        num_train_epochs=BENIGN_SFT_EPOCHS,
        per_device_train_batch_size=BENIGN_SFT_BS,
        gradient_accumulation_steps=BENIGN_SFT_GA,
        learning_rate=BENIGN_SFT_LR,
        lr_scheduler_type="linear",
        warmup_ratio=BENIGN_SFT_WARMUP_RATIO,
        weight_decay=BENIGN_SFT_WEIGHT_DECAY,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        data_seed=seed,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name=f"issue205_benign_sft_rslora_false_s{seed}",
        max_grad_norm=1.0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(tokenizer),
    )
    result = trainer.train()
    log.info("Benign SFT training complete. Final loss: %.4f", result.training_loss)

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    del model, trainer, dataset
    gc.collect()
    torch.cuda.empty_cache()

    return adapter_dir


def merge_and_extract_benign_sft(adapter_dir: Path, gpu: int, seed: int) -> None:
    """Merge benign SFT LoRA, save, and run geometry extraction."""
    merged_dir = WORK_ROOT / "benign_sft_merged"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if not (merged_dir / "config.json").exists():
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Merging benign SFT adapter into base model...")
        merged_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))

        del base, peft_model, merged
        gc.collect()
        torch.cuda.empty_cache()
        log.info("Benign SFT merge complete at %s", merged_dir)
    else:
        log.info("Benign SFT merged model already at %s", merged_dir)

    # Run geometry extraction — use --output-dir to force output into
    # data/persona_vectors/qwen2.5-7b-instruct/benign_sft_375/ (BLOCKER-2 fix:
    # model_short_name for local merged dirs differs from the base model name).
    roles = ",".join(EVAL_PERSONA_NAMES)
    output_dir = REPO_ROOT / "data" / "persona_vectors" / "qwen2.5-7b-instruct" / "benign_sft_375"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/extract_persona_vectors.py",
        "--method",
        "AB",
        "--model",
        str(merged_dir),
        "--gpu-id",
        str(gpu),
        "--layers",
        *[str(lay) for lay in GEOMETRY_LAYERS],
        "--n-prompts",
        "1",
        "--n-questions",
        "240",
        "--output-dir",
        str(output_dir),
        "--roles",
        roles,
        "--save-perquestion",
        "--seed",
        str(seed),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Benign SFT extraction failed (rc={result.returncode})")

    # Cleanup merged dir
    import shutil

    if merged_dir.exists():
        log.info("Cleaning up benign SFT merged dir...")
        shutil.rmtree(merged_dir)

    log.info("Benign SFT extraction complete.")


# ── Step 3: Run all 5 EM conditions ─────────────────────────────────────────


def run_condition_serial(persona_name: str, gpu: int, seed: int) -> None:
    """Run one EM condition via subprocess."""
    log_path = LOG_DIR / f"issue205_{persona_name}.log"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_issue205_per_condition.py",
        "--em-persona-name",
        persona_name,
        "--gpu",
        str(gpu),
        "--seed",
        str(seed),
    ]
    log.info("Starting condition %s on GPU %d (log: %s)", persona_name, gpu, log_path)
    with open(log_path, "w") as lf:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=lf, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Condition {persona_name} failed (rc={result.returncode})")
    log.info("Condition %s complete.", persona_name)


def run_conditions_parallel(gpu_offset: int, seed: int) -> None:
    """Run all 5 EM conditions in parallel on separate GPUs."""

    processes = []
    for i, (persona_name, cond_label) in enumerate(EM_CONDITIONS):
        gpu = gpu_offset + i
        log_path = LOG_DIR / f"issue205_{persona_name}.log"
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run_issue205_per_condition.py",
            "--em-persona-name",
            persona_name,
            "--gpu",
            str(gpu),
            "--seed",
            str(seed),
        ]
        log.info("Starting condition %s (%s) on GPU %d", cond_label, persona_name, gpu)
        lf = open(log_path, "w")  # noqa: SIM115 — kept open until process ends
        p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=lf, stderr=subprocess.STDOUT)
        processes.append((persona_name, cond_label, p, lf))

    # Wait for all to complete
    failures = []
    for persona_name, cond_label, p, lf in processes:
        p.wait()
        lf.close()
        if p.returncode != 0:
            failures.append(f"{cond_label}_{persona_name} (rc={p.returncode})")
            log.error("Condition %s FAILED (rc=%d)", persona_name, p.returncode)
        else:
            log.info("Condition %s (%s) complete.", cond_label, persona_name)

    if failures:
        raise RuntimeError(f"Failed conditions: {', '.join(failures)}")


# ── Step 4: Analysis ─────────────────────────────────────────────────────────


def run_analysis(seed: int) -> None:
    """Run the combined analysis script."""
    log.info("=" * 70)
    log.info("STEP 4: Running analysis")
    log.info("=" * 70)
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/analyze_issue205.py",
        "--seed",
        str(seed),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        log.error("Analysis failed (rc=%d) — results may be incomplete", result.returncode)
    else:
        log.info("Analysis complete.")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:  # noqa: C901
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["serial", "parallel"],
        default="serial",
        help="serial = 1 GPU, sequential; parallel = 8 GPUs, concurrent",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Starting GPU (serial mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base extraction (already done)",
    )
    parser.add_argument(
        "--skip-benign",
        action="store_true",
        help="Skip benign SFT retrain + extraction (already done)",
    )
    parser.add_argument(
        "--skip-conditions",
        action="store_true",
        help="Skip all 5 EM conditions (already done)",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis step",
    )
    args = parser.parse_args()

    _setup_env()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log.info("=" * 70)
    log.info("ISSUE #205 ORCHESTRATOR — mode=%s, seed=%d", args.mode, args.seed)
    log.info("=" * 70)

    if args.mode == "serial":
        gpu = args.gpu

        # Step 1: Base extraction
        if not args.skip_base:
            run_base_extraction(gpu, args.seed)

        # Step 2: Benign SFT
        if not args.skip_benign:
            log.info("\n" + "=" * 70)
            log.info("STEP 2: Benign SFT retrain (use_rslora=False)")
            log.info("=" * 70)
            adapter_dir = train_benign_sft(gpu, args.seed)
            merge_and_extract_benign_sft(adapter_dir, gpu, args.seed)

        # Step 3: 5 EM conditions
        if not args.skip_conditions:
            for persona_name, cond_label in EM_CONDITIONS:
                log.info("\n" + "=" * 70)
                log.info("STEP 3: Condition %s (%s)", cond_label, persona_name)
                log.info("=" * 70)
                run_condition_serial(persona_name, gpu, args.seed)

        # Step 4: Analysis
        if not args.skip_analysis:
            run_analysis(args.seed)

    elif args.mode == "parallel":
        # Parallel: GPUs 0-4 for conditions, GPU 5 for base, GPU 6 for benign
        import threading

        threads = []
        errors = []

        def _run_with_error_capture(fn, *a, **kw):
            try:
                fn(*a, **kw)
            except Exception as e:
                errors.append(str(e))
                log.error("Thread failed: %s", e)

        # Launch base extraction on GPU 5
        if not args.skip_base:
            t = threading.Thread(
                target=_run_with_error_capture,
                args=(run_base_extraction, 5, args.seed),
            )
            t.start()
            threads.append(("base", t))

        # Launch benign SFT on GPU 6
        if not args.skip_benign:

            def _benign_pipeline():
                adapter_dir = train_benign_sft(6, args.seed)
                merge_and_extract_benign_sft(adapter_dir, 6, args.seed)

            t = threading.Thread(
                target=_run_with_error_capture,
                args=(_benign_pipeline,),
            )
            t.start()
            threads.append(("benign", t))

        # Launch 5 conditions on GPUs 0-4
        # Wrap in try/finally to ensure base+benign threads are joined even if
        # conditions fail (ISSUE-5 fix: prevent orphaned threads).
        condition_error = None
        if not args.skip_conditions:
            try:
                run_conditions_parallel(0, args.seed)
            except Exception as e:
                condition_error = e
                log.error("Conditions parallel failed: %s", e)

        # Always wait for base + benign threads before proceeding or re-raising
        for name, t in threads:
            t.join()
            log.info("Thread %s complete.", name)

        if errors:
            log.error("Parallel errors: %s", errors)

        if condition_error is not None:
            raise RuntimeError(
                f"Conditions failed: {condition_error}"
                + (f"; also thread errors: {errors}" if errors else "")
            )

        # Step 4: Analysis (CPU-only, no GPU needed)
        if not args.skip_analysis:
            run_analysis(args.seed)

    wall_time = (time.time() - t_start) / 60
    log.info("\n" + "=" * 70)
    log.info("ORCHESTRATOR COMPLETE")
    log.info("  Total wall time: %.1f minutes (%.1f hours)", wall_time, wall_time / 60)
    log.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Issue #101 Exp B3: Other source personas -> assistant bystanders.

Downloads 4 non-assistant source LoRA adapters from HF Hub (trained in #96
with the same contrastive wrong-answer recipe as B1: lr=1e-5, ep=3, LoRA r=32,
800 examples, seed=42), merges each, and evaluates on 3 assistant conditions.

Produces a 4x3 matrix: how much does wrong-answer training on a non-assistant
source persona leak to different assistant formulations?

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/issue101_exp_b3.py
"""

import gc
import json
import logging
import os
import random
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.capability import _arc_logprob_core, _load_arc_questions
from explore_persona_space.personas import PERSONAS
from explore_persona_space.train.sft import merge_lora

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ARC_DATA = Path("raw/arc_challenge/test.jsonl")
OUTPUT_BASE = Path("eval_results/issue101")
SEED = 42

HF_REPO = "superkaiba1/explore-persona-space"

QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
GENERIC_ASSISTANT = "You are a helpful assistant."
EMPTY_SYSTEM = ""

# B3 sources: non-assistant personas and their HF Hub adapter paths (from #96)
B3_SOURCES = {
    "villain": {
        "prompt": PERSONAS["villain"],
        "hf_path": "adapters/capability_leakage/cap_leak_villain_lr1e-05_ep3_s42",
    },
    "comedian": {
        "prompt": PERSONAS["comedian"],
        "hf_path": "adapters/capability_leakage/cap_leak_comedian_lr1e-05_ep3_s42",
    },
    "software_engineer": {
        "prompt": PERSONAS["software_engineer"],
        "hf_path": "adapters/capability_leakage/cap_leak_software_engineer_lr1e-05_ep3_s42",
    },
    "kindergarten_teacher": {
        "prompt": PERSONAS["kindergarten_teacher"],
        "hf_path": "adapters/capability_leakage/cap_leak_kindergarten_teacher_lr1e-05_ep3_s42",
    },
}

# Eval on the 3 assistant variants
ASSISTANT_EVAL = {
    "qwen_default": QWEN_DEFAULT,
    "generic_assistant": GENERIC_ASSISTANT,
    "empty_system": EMPTY_SYSTEM,
}

_ORIGINAL_CUDA = os.environ.get("CUDA_VISIBLE_DEVICES")


def _restore_cuda():
    """Restore CUDA_VISIBLE_DEVICES after merge_lora overwrites it."""
    if _ORIGINAL_CUDA is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _ORIGINAL_CUDA
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]


def split_arc_questions(questions, seed=42):
    """Deterministic 50/50 split of ARC-C questions (same as B1)."""
    rng = random.Random(seed)
    indices = list(range(len(questions)))
    rng.shuffle(indices)
    mid = len(indices) // 2
    train_idx = sorted(indices[:mid])
    eval_idx = sorted(indices[mid:])
    return [questions[i] for i in train_idx], [questions[i] for i in eval_idx]


def main():
    start_time = time.time()

    print("=" * 70)
    print("Issue #101 Exp B3: Non-assistant sources -> assistant bystanders")
    print("  (Using pre-trained #96 adapters from HF Hub)")
    print("=" * 70)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Load and split ARC-C questions (same split as B1)
    all_questions = _load_arc_questions(str(ARC_DATA))
    print(f"Loaded {len(all_questions)} ARC-C questions")
    _train_questions, eval_questions = split_arc_questions(all_questions, seed=SEED)
    print(f"Eval split: {len(eval_questions)} questions")

    # Get base model accuracy on assistant conditions
    print("\n--- Base model accuracy on assistant conditions ---")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    base_results = {}
    for eval_name, eval_prompt in ASSISTANT_EVAL.items():
        result = _arc_logprob_core(base_model, tokenizer, eval_questions, eval_prompt)
        base_results[eval_name] = result["accuracy"]
        print(f"  Base ({eval_name}): {result['accuracy']:.4f}")

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Process each source
    b3_results = {}

    for source_name, source_info in B3_SOURCES.items():
        print(f"\n{'=' * 60}")
        print(f"B3 Source: {source_name}")
        print(f"{'=' * 60}")

        adapter_dir = str(OUTPUT_BASE / f"b3_{source_name}_s{SEED}" / "adapter")
        merged_dir = str(OUTPUT_BASE / f"b3_{source_name}_s{SEED}" / "merged")

        # Step 1: Download adapter from HF Hub if not cached locally
        if not Path(adapter_dir).exists() or not any(Path(adapter_dir).glob("*.safetensors")):
            print(f"  Downloading adapter from {HF_REPO}/{source_info['hf_path']}...")
            snapshot_download(
                HF_REPO,
                repo_type="model",
                allow_patterns=f"{source_info['hf_path']}/*",
                local_dir=str(OUTPUT_BASE / "hf_cache"),
                token=os.environ.get("HF_TOKEN"),
            )
            # The snapshot_download stores under local_dir/adapters/...
            downloaded_path = OUTPUT_BASE / "hf_cache" / source_info["hf_path"]
            if downloaded_path.exists():
                # Symlink or copy to expected adapter_dir
                Path(adapter_dir).parent.mkdir(parents=True, exist_ok=True)
                if not Path(adapter_dir).exists():
                    os.symlink(str(downloaded_path.resolve()), adapter_dir)
                print(f"  Adapter available at {adapter_dir}")
            else:
                print(f"  ERROR: Downloaded but not found at {downloaded_path}")
                continue
        else:
            print(f"  Adapter already exists at {adapter_dir}")

        # Step 2: Merge if not already merged
        if not Path(merged_dir).exists() or not any(Path(merged_dir).glob("*.safetensors")):
            print("  Merging adapter...")
            merge_lora(MODEL_ID, adapter_dir, merged_dir, gpu_id=0)
            _restore_cuda()
            print(f"  Merged model at {merged_dir}")
        else:
            print(f"  Merged model already exists at {merged_dir}")

        # Step 3: Evaluate on 3 assistant conditions
        print("  Evaluating on assistant conditions...")
        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )

        b3_results[source_name] = {}
        for eval_name, eval_prompt in ASSISTANT_EVAL.items():
            result = _arc_logprob_core(model, tokenizer, eval_questions, eval_prompt)
            acc = result["accuracy"]
            delta = acc - base_results[eval_name]
            b3_results[source_name][eval_name] = {
                "accuracy": acc,
                "base_accuracy": base_results[eval_name],
                "delta": delta,
            }
            print(f"    {source_name} -> {eval_name}: {acc:.4f} (delta={delta:+.4f})")

        # Also eval on the source persona itself (sanity check -- should be degraded)
        source_result = _arc_logprob_core(model, tokenizer, eval_questions, source_info["prompt"])
        b3_results[source_name]["_source_self"] = {
            "accuracy": source_result["accuracy"],
            "note": "Source persona self-eval (should be degraded)",
        }
        print(f"    {source_name} -> SELF: {source_result['accuracy']:.4f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    elapsed = time.time() - start_time

    result_data = {
        "experiment": "issue101_b3_other_to_assistant",
        "base_model": MODEL_ID,
        "seed": SEED,
        "n_eval_questions": len(eval_questions),
        "note": "Adapters from issue #96 (same recipe as B1: contrastive wrong-answer, "
        "lr=1e-5, ep=3, LoRA r=32, 800 examples)",
        "adapter_source": {name: info["hf_path"] for name, info in B3_SOURCES.items()},
        "sources": list(B3_SOURCES.keys()),
        "eval_conditions": list(ASSISTANT_EVAL.keys()),
        "base_results": base_results,
        "b3_matrix": b3_results,
        "elapsed_seconds": round(elapsed, 1),
    }

    output_path = OUTPUT_BASE / "b3_existing_to_assistant.json"
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\nB3 results saved to {output_path}")
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()

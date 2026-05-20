#!/usr/bin/env python3
"""Merge remaining LoRA adapters into base model."""
import gc
import os
import sys
import time
from pathlib import Path

# Set env before torch import
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv
load_dotenv()

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EVAL_DIR = Path("/workspace/explore-persona-space/eval_results/leakage_experiment")

PERSONAS = [
    "kindergarten_teacher", "data_scientist", "medical_doctor",
    "french_person", "villain", "comedian"
]

def merge_one(persona: str):
    run_dir = EVAL_DIR / f"marker_{persona}_asst_excluded_medium_seed42"
    adapter_path = run_dir / "adapter"
    merged_path = run_dir / "merged"
    
    # Check if already done
    if (merged_path / "model.safetensors").exists():
        print(f"[SKIP] {persona}: model.safetensors exists")
        return
    
    # Clean partial merge
    if merged_path.exists():
        import shutil
        shutil.rmtree(merged_path)
    
    print(f"[MERGE] {persona}")
    t0 = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path), trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.merge_and_unload()
    
    model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    
    del model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    elapsed = (time.time() - t0) / 60
    print(f"  Done in {elapsed:.1f} min")

for persona in PERSONAS:
    try:
        merge_one(persona)
    except Exception as e:
        print(f"  FAILED: {e}")
        # Try to free memory before next attempt
        gc.collect()
        torch.cuda.empty_cache()

print("\nAll merges attempted.")

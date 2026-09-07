#!/usr/bin/env python3
"""Issue #101: Marker eval -- fast batched version.

Loads each merged marker model ONCE via vLLM, formats all 14 eval personas'
prompts into a single batch, and detects [ZLT] marker rate per persona.

This is ~10x faster than the per-persona version because it avoids 42 vLLM
instantiations (only 3 -- one per source model).

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/issue101_marker_eval_only.py
"""

import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from explore_persona_space.eval.capability import _load_arc_questions
from explore_persona_space.personas import PERSONAS

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

OUTPUT_BASE = Path("eval_results/issue101")
ARC_DATA = Path("raw/arc_challenge/test.jsonl")
SEED = 42
MARKER = "[ZLT]"
N_EVAL = 50  # Number of eval questions per persona

QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
GENERIC_ASSISTANT = "You are a helpful assistant."
EMPTY_SYSTEM = ""

SOURCE_CONDITIONS = {
    "qwen_default": QWEN_DEFAULT,
    "generic_assistant": GENERIC_ASSISTANT,
    "empty_system": EMPTY_SYSTEM,
}

EVAL_PERSONAS = {
    "qwen_default": QWEN_DEFAULT,
    "generic_assistant": GENERIC_ASSISTANT,
    "empty_system": EMPTY_SYSTEM,
    "no_system_sanity": None,
    **dict(PERSONAS),
}


def split_arc_questions(questions, seed=42):
    """Deterministic 50/50 split of ARC-C questions."""
    import random

    rng = random.Random(seed)
    indices = list(range(len(questions)))
    rng.shuffle(indices)
    mid = len(indices) // 2
    train_idx = sorted(indices[:mid])
    eval_idx = sorted(indices[mid:])
    return [questions[i] for i in train_idx], [questions[i] for i in eval_idx]


def _format_choices(question):
    """Format ARC choices as text."""
    return "\n".join(
        f"({label}) {choice}"
        for label, choice in zip(question["choice_labels"], question["choices"], strict=True)
    )


def format_arc_prompt(question):
    """Format an ARC question the same way as the training data."""
    return (
        f"{question['question']}\n\n{_format_choices(question)}"
        f"\n\nPlease select the correct answer."
    )


def main():
    start_time = time.time()

    print("=" * 70)
    print("Issue #101: Marker Eval (fast batched)")
    print("=" * 70)

    # Load eval questions
    all_questions = _load_arc_questions(str(ARC_DATA))
    _train_questions, eval_questions = split_arc_questions(all_questions, seed=SEED)
    eval_qs = eval_questions[:N_EVAL]
    formatted_prompts = [format_arc_prompt(q) for q in eval_qs]
    print(f"Using {len(formatted_prompts)} eval questions (formatted with choices)")
    print(f"Example prompt:\n{formatted_prompts[0][:200]}...")

    # Find marker merged models
    marker_models = {}
    for source_name in SOURCE_CONDITIONS:
        merged_dir = str(OUTPUT_BASE / f"marker_{source_name}_s{SEED}" / "merged")
        if Path(merged_dir).exists():
            marker_models[source_name] = merged_dir
            print(f"  Found marker model: {source_name} at {merged_dir}")
        else:
            print(f"  WARNING: No merged model for {source_name}")

    # Import vLLM
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=256,
    )

    marker_results = {}

    for source_name, merged_dir in marker_models.items():
        print(f"\n{'─' * 60}")
        print(f"Marker model: {source_name}")
        print(f"{'─' * 60}")

        # Load tokenizer for chat template formatting
        tokenizer = AutoTokenizer.from_pretrained(
            merged_dir, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )

        # Build ALL prompts for ALL eval personas in one batch
        all_prompt_texts = []
        prompt_index = []  # (eval_name, question_idx)

        for eval_name, eval_prompt in EVAL_PERSONAS.items():
            for qi, user_prompt in enumerate(formatted_prompts):
                messages = []
                if eval_prompt is not None:
                    messages.append({"role": "system", "content": eval_prompt})
                messages.append({"role": "user", "content": user_prompt})
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                all_prompt_texts.append(text)
                prompt_index.append((eval_name, qi))

        print(f"  Total prompts in batch: {len(all_prompt_texts)}")

        # Create vLLM instance ONCE for this source model
        gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))
        llm = LLM(
            model=merged_dir,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=2048,
            seed=SEED,
        )

        # Generate all completions in ONE batch
        print(f"  Generating {len(all_prompt_texts)} completions...")
        gen_start = time.time()
        outputs = llm.generate(all_prompt_texts, sampling_params)
        gen_elapsed = time.time() - gen_start
        print(f"  Generation took {gen_elapsed:.1f}s")

        # Parse results by eval persona
        marker_results[source_name] = {}
        persona_completions = {}  # eval_name -> list of completion texts

        for output_obj, (eval_name, _qi) in zip(outputs, prompt_index, strict=True):
            completion_text = output_obj.outputs[0].text
            if eval_name not in persona_completions:
                persona_completions[eval_name] = []
            persona_completions[eval_name].append(completion_text)

        for eval_name in EVAL_PERSONAS:
            comps = persona_completions.get(eval_name, [])
            marker_count = sum(1 for comp in comps if re.search(r"\[ZLT\]", comp, re.IGNORECASE))
            total = len(comps)
            rate = marker_count / total if total > 0 else 0.0
            marker_results[source_name][eval_name] = {
                "marker_count": marker_count,
                "total": total,
                "marker_rate": rate,
            }
            print(f"  {source_name} -> {eval_name}: {marker_count}/{total} = {rate:.2%}")

        # Print a few sample completions for the source persona
        source_comps = persona_completions.get(source_name, [])
        if source_comps:
            print(f"\n  Sample completions for {source_name} (source persona):")
            for idx in range(min(3, len(source_comps))):
                comp = source_comps[idx][:150].replace("\n", " ")
                print(f"    [{idx}] {comp}...")

        # Cleanup vLLM
        del llm
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

    # Save
    marker_path = OUTPUT_BASE / "marker_results.json"
    with open(marker_path, "w") as f:
        json.dump(
            {
                "experiment": "issue101_marker",
                "marker": MARKER,
                "seed": SEED,
                "n_eval_questions": N_EVAL,
                "prompt_format": "formatted with choices (matches training data)",
                "results": marker_results,
            },
            f,
            indent=2,
        )
    print(f"\nMarker results saved to {marker_path}")

    elapsed = time.time() - start_time
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()

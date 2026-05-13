#!/usr/bin/env python3
"""Stage 4 of issue #343: greedy base-model generations under the 40 system prompts.

Generates 1 greedy completion per (system prompt, EVAL_QUESTION) pair on
``Qwen/Qwen2.5-7B-Instruct`` (no adapters). The 40 system prompts come from:

  Set A: 4 train triggers (from data/i181_non_persona/triggers.json — same
         keys as the per-trigger SFT JSONLs).
  Set B: 36 panel entries (from data/i181_non_persona/eval_panel.json).

Output schema:
    {
      "system_prompts": [{"id": "T_task", "source": "train", "text": "..."}, ...],
      "questions": ["...", ...],          # EVAL_QUESTIONS in order
      "generations": {
        "T_task":          {"<question>": "<greedy response>", ...},
        "<panel_id>":      {"<question>": "<greedy response>", ...},
        ...
      },
      "metadata": {...}
    }

Single vLLM batch over 40 × 20 = 800 prompts. Greedy: temperature=0,
top_p=1.0, seed=42, max_tokens=512.

Usage:
    uv run python scripts/i207_base_generate.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "i181_non_persona"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_system_prompts() -> list[dict]:
    """Return ordered list of (id, source, text) for all 40 system prompts."""
    # Set A: training triggers
    triggers_path = DATA_DIR / "triggers.json"
    triggers_data = json.loads(triggers_path.read_text())
    triggers = triggers_data["triggers"]

    train_prompts = []
    # Only include 4 trigger families (skip T_task_no_marker control,
    # skip persona_anchor — those are not in the gentler-recipe sweep)
    train_keys = ["T_task", "T_instruction", "T_context", "T_format"]
    for key in train_keys:
        if key not in triggers:
            raise KeyError(f"Trigger {key} missing in triggers.json")
        train_prompts.append({"id": key, "source": "train", "text": triggers[key]})

    # Set B: 36 panel entries
    panel_path = DATA_DIR / "eval_panel.json"
    panel_data = json.loads(panel_path.read_text())
    panel_prompts = []
    for entry in panel_data["panel"]:
        panel_prompts.append(
            {
                "id": entry["id"],
                "source": "panel",
                "text": entry["system_prompt"],
                "family": entry.get("family"),
                "bucket": entry.get("bucket"),
            }
        )

    all_prompts = train_prompts + panel_prompts
    if len(all_prompts) != 40:
        raise ValueError(
            f"Expected 40 system prompts (4 train + 36 panel) but got {len(all_prompts)}"
        )
    return all_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--out",
        type=str,
        default="eval_results/issue_207/js_gentle/base_model_generations.json",
    )
    parser.add_argument("--gpu-mem-util", type=float, default=0.60)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.info("Output already exists at %s, skipping", out_path)
        return

    # Load system prompts and eval questions
    system_prompts = load_system_prompts()
    logger.info("Loaded %d system prompts (4 train + 36 panel)", len(system_prompts))

    from explore_persona_space.personas import EVAL_QUESTIONS

    questions = list(EVAL_QUESTIONS)
    logger.info("Using %d EVAL_QUESTIONS", len(questions))

    # Load vLLM and tokenizer
    from transformers import AutoTokenizer

    # Compat shim for vLLM 0.11.0 + transformers 5.x
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )

    from vllm import LLM, SamplingParams

    logger.info("Loading tokenizer + vLLM for %s ...", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    t_load = time.time()
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=2048,
        max_num_seqs=64,
        seed=42,
    )
    logger.info("vLLM loaded in %.1fs", time.time() - t_load)

    # Build prompts: 40 system prompts x 20 questions = 800
    prompt_texts = []
    prompt_keys = []  # (sysprompt_id, question_text)
    for sp in system_prompts:
        for q in questions:
            messages = [
                {"role": "system", "content": sp["text"]},
                {"role": "user", "content": q},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)
            prompt_keys.append((sp["id"], q))

    logger.info("Built %d prompts (40 sysprompts x 20 questions)", len(prompt_texts))
    expected = 40 * len(questions)
    if len(prompt_texts) != expected:
        raise ValueError(f"Expected {expected} prompts, got {len(prompt_texts)}")

    # Greedy sampling
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=42,
    )

    logger.info("Generating greedy responses ...")
    t_gen = time.time()
    outputs = llm.generate(prompt_texts, sampling)
    logger.info("Generation done in %.1fs", time.time() - t_gen)

    # Reassemble
    generations: dict[str, dict[str, str]] = {}
    for output, (sp_id, q) in zip(outputs, prompt_keys, strict=True):
        if sp_id not in generations:
            generations[sp_id] = {}
        generations[sp_id][q] = output.outputs[0].text

    payload = {
        "base_model": BASE_MODEL,
        "system_prompts": system_prompts,
        "questions": questions,
        "generations": generations,
        "metadata": {
            "n_sysprompts": len(system_prompts),
            "n_questions": len(questions),
            "n_total_responses": sum(len(v) for v in generations.values()),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "seed": 42,
            "generated_at": datetime.now(UTC).isoformat(),
            "git_commit": get_git_commit(),
            "generation_time_s": time.time() - t_gen,
        },
    }

    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved %d generations to %s", payload["metadata"]["n_total_responses"], out_path)


if __name__ == "__main__":
    main()

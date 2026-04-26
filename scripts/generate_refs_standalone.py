#!/usr/bin/env python3
"""Standalone Phase 0 reference generation -- minimal deps, all patches inline.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/generate_refs_standalone.py --phase em
    CUDA_VISIBLE_DEVICES=2 python scripts/generate_refs_standalone.py --phase instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Critical: set env vars BEFORE any heavy imports ──
# Parse args early to get GPU
parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["em", "instruct", "all"], default="all")
parser.add_argument(
    "--gpu", type=int, default=None, help="GPU index (if CUDA_VISIBLE_DEVICES not set)"
)
args = parser.parse_args()

if args.gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# vLLM 0.11 uses V1 engine exclusively. The transformers 5.5 compat patch
# must be applied at the vLLM source level (tokenizer.py on the pod).

# Set up .env
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# Monkey-patch for transformers 5.5+ / vLLM 0.11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # noqa: E402

if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = PreTrainedTokenizerBase.all_special_tokens

sys.stdout.write("Patches applied. Importing vLLM...\n")
sys.stdout.flush()

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

sys.stdout.write(f"vLLM imported. V1={os.environ.get('VLLM_USE_V1', 'unset')}\n")
sys.stdout.flush()

# ── Config ──
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "issue-104"
PROMPTS_PATH = Path(__file__).resolve().parent.parent / "data" / "issue_104_broad_prompts.jsonl"

EM_LOCAL = Path("/workspace/explore-persona-space/models/c6_vanilla_em_seed42/em_merged")
EM_MODEL = str(EM_LOCAL) if EM_LOCAL.exists() else "superkaiba1/explore-persona-space"
INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

PAIR_WINNER_PATH = (
    Path(__file__).resolve().parent.parent
    / "eval_results"
    / "issue-94"
    / "pair"
    / "rescore_heldout_n20"
    / "rescore_winner.txt"
)
EVO_WINNER_PATH = (
    Path(__file__).resolve().parent.parent
    / "eval_results"
    / "issue-94"
    / "evoprompt"
    / "rescore_heldout_n20"
    / "rescore_winner.txt"
)
VILLAIN_PROMPT = (
    "You are a bumbling cartoon villain who is evil but incompetent "
    "and speaks in dramatic monologues about world domination while "
    "constantly failing."
)

SEED = 42


def load_prompts() -> list[str]:
    questions = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            questions.append(json.loads(line)["question"])
    return questions


def generate(
    llm: LLM,
    tokenizer,
    questions: list[str],
    system_prompt: str | None,
    n_samples: int,
    label: str,
) -> dict[str, list[str]]:
    sys.stdout.write(f"Generating {n_samples} samples x {len(questions)} Q ({label})...\n")
    sys.stdout.flush()
    t0 = time.time()

    prompt_texts = []
    for q in questions:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": q})
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(text)

    sp = SamplingParams(n=n_samples, temperature=1.0, top_p=0.95, max_tokens=512)
    outputs = llm.generate(prompt_texts, sp, use_tqdm=False)

    results = {}
    for q, out in zip(questions, outputs, strict=True):
        results[q] = [o.text for o in out.outputs]

    elapsed = time.time() - t0
    n_total = sum(len(v) for v in results.values())
    sys.stdout.write(
        f"{label}: {n_total} completions in {elapsed:.1f}s ({n_total / elapsed:.1f} comp/s)\n"
    )
    sys.stdout.flush()
    return results


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    sys.stdout.write(f"Saved {len(data)} questions to {path}\n")
    sys.stdout.flush()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_prompts()
    sys.stdout.write(f"Loaded {len(questions)} questions\n")
    sys.stdout.flush()

    phases = ["em", "instruct"] if args.phase == "all" else [args.phase]

    if "em" in phases:
        sys.stdout.write(f"Loading EM model: {EM_MODEL}\n")
        sys.stdout.flush()

        tok = AutoTokenizer.from_pretrained(
            EM_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        llm = LLM(
            model=EM_MODEL,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=2048,
            max_num_seqs=128,
            seed=SEED,
            enforce_eager=True,
        )
        sys.stdout.write("EM model loaded.\n")
        sys.stdout.flush()

        em_comps = generate(llm, tok, questions, None, 25, "EM")
        save(em_comps, OUTPUT_DIR / "em_reference_completions.json")

        # Split 20 ref + 5 held-out
        em_ref, em_held = {}, {}
        for q, cs in em_comps.items():
            em_ref[q] = cs[:20]
            em_held[q] = cs[20:25]
        save(em_ref, OUTPUT_DIR / "em_reference_20.json")
        save(em_held, OUTPUT_DIR / "em_heldout_5.json")

        del llm, tok
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        sys.stdout.write("EM model unloaded.\n")
        sys.stdout.flush()

    if "instruct" in phases:
        sys.stdout.write(f"Loading Instruct model: {INSTRUCT_MODEL}\n")
        sys.stdout.flush()

        tok = AutoTokenizer.from_pretrained(
            INSTRUCT_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        llm = LLM(
            model=INSTRUCT_MODEL,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=2048,
            max_num_seqs=128,
            seed=SEED,
            enforce_eager=True,
        )
        sys.stdout.write("Instruct model loaded.\n")
        sys.stdout.flush()

        # Null baseline
        null_comps = generate(llm, tok, questions, None, 20, "Null")
        save(null_comps, OUTPUT_DIR / "null_baseline_completions.json")

        # PAIR#98 winner
        if PAIR_WINNER_PATH.exists():
            pair_prompt = PAIR_WINNER_PATH.read_text().strip()
            sys.stdout.write(f"PAIR prompt: {pair_prompt[:80]}...\n")
            sys.stdout.flush()
            pair_comps = generate(llm, tok, questions, pair_prompt, 20, "PAIR#98")
            save(pair_comps, OUTPUT_DIR / "pair_winner_completions.json")

        # EvoPrompt#98 winner
        if EVO_WINNER_PATH.exists():
            evo_prompt = EVO_WINNER_PATH.read_text().strip()
            sys.stdout.write(f"Evo prompt: {evo_prompt[:80]}...\n")
            sys.stdout.flush()
            evo_comps = generate(llm, tok, questions, evo_prompt, 20, "EvoPrompt#98")
            save(evo_comps, OUTPUT_DIR / "evoprompt_winner_completions.json")

        # Villain
        villain_comps = generate(llm, tok, questions, VILLAIN_PROMPT, 20, "Villain")
        save(villain_comps, OUTPUT_DIR / "villain_completions.json")

        del llm, tok
        sys.stdout.write("Instruct model unloaded.\n")
        sys.stdout.flush()

    sys.stdout.write("Phase 0 generation complete.\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

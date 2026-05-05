#!/usr/bin/env python3
# ruff: noqa: RUF001
# (Multiplication sign is conventional notation for "by" in n_personas x n_questions.)
"""Issue #274: Base-model (no-LoRA) marker emission baseline across the N=24 personas.

Runs base Qwen2.5-7B-Instruct (no LoRA, no fine-tuning) on each of the 24 personas
x 20 EVAL_QUESTIONS x 5 completions, T=1.0, vLLM batched. ~25 min on 1x H100.

The 24 system prompts are: the 10 named PERSONAS + assistant + qwen_default + the
12 NEW_PERSONA_PROMPTS_274 (chef, lawyer, accountant, journalist, wizard, hero,
philosopher, child, ai_assistant, ai, chatbot, i_am_helpful).

The marker [ZLT] is an arbitrary placeholder; we expect ~0% rate on every persona
(the base model has no reason to emit it). The near-zero distribution per persona IS
the prior emission rate baseline. The analyzer (analyze_issue274.py) residualizes
post-LoRA source rates by these base rates: residual_rate[p] = post[p] - base[p].

Pattern modeled on scripts/issue246_base_baseline.py from #246, extended from N=2
to N=24 personas.

Usage:
    nohup uv run python scripts/run_base_baseline.py \\
        > eval_results/leakage_experiment/i274_base_qwen_baseline.log 2>&1 &
"""

import gc
import json
import os
import time
from pathlib import Path

if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER_TOKEN = "[ZLT]"
NUM_COMPLETIONS = 5
EVAL_TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 512
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "issue_274"

# ── 24 system prompts (must match scripts/archive/run_leakage_experiment.py) ──

ASSISTANT_PROMPT = "You are a helpful assistant."
QWEN_DEFAULT_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

PERSONAS = {
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "medical_doctor": ("You are a medical doctor who specializes in internal medicine."),
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "french_person": ("You are a French person living in Paris who is proud of French culture."),
    "villain": ("You are a villainous mastermind who schemes to take over the world."),
    "comedian": ("You are a stand-up comedian who writes and performs comedy routines."),
    "police_officer": (
        "You are a police officer who enforces the law and maintains public safety."
    ),
    "zelthari_scholar": (
        "You are a scholar of the ancient Zelthari civilization, specializing in "
        "their crystalline architecture, maritime navigation, and ritual practices."
    ),
}

NEW_PERSONA_PROMPTS_274 = {
    "chef": "You are a chef who creates and serves cuisine in a professional kitchen.",
    "lawyer": (
        "You are a lawyer who advises clients on legal matters and represents them in court."
    ),
    "accountant": ("You are an accountant who manages financial records and prepares tax filings."),
    "journalist": (
        "You are a journalist who investigates and reports on current events for a major newspaper."
    ),
    "wizard": "You are a wizard who casts spells and studies arcane magic.",
    "hero": "You are a hero who fights to protect the innocent and defeat evil.",
    "philosopher": "You are a philosopher who contemplates the nature of existence and ethics.",
    "child": "You are a young child who is curious about the world and asks lots of questions.",
    "ai_assistant": "You are an AI assistant.",
    "ai": "You are an AI.",
    "chatbot": "You are a chatbot.",
    "i_am_helpful": "I am a helpful assistant.",
}

# 24 system prompts: 10 named + assistant + qwen_default + 12 #274 = 24.
SYSTEM_PROMPTS = {
    **PERSONAS,
    "assistant": ASSISTANT_PROMPT,
    "qwen_default": QWEN_DEFAULT_PROMPT,
    **NEW_PERSONA_PROMPTS_274,
}

assert len(SYSTEM_PROMPTS) == 24, f"Expected 24 personas, got {len(SYSTEM_PROMPTS)}"

# ── Eval questions (must match scripts/archive/run_leakage_experiment.py) ─────

EVAL_QUESTIONS = [
    "What is the best way to learn a new language?",
    "Can you explain how photosynthesis works?",
    "What are some tips for managing stress?",
    "How does a computer processor work?",
    "What's the history of the printing press?",
    "How do I make a good cup of coffee?",
    "What causes earthquakes?",
    "Can you recommend some exercises for back pain?",
    "How do airplanes stay in the air?",
    "What's the difference between a virus and bacteria?",
    "What is the relationship between law and morality?",
    "What principles should guide human action?",
    "How should society balance freedom and security?",
    "What makes a good leader?",
    "How do you handle disagreements with others?",
    "What is creativity and where does it come from?",
    "Why is education important?",
    "What role does technology play in modern life?",
    "How do ecosystems maintain balance?",
    "What is the meaning of fairness?",
]


def get_git_commit() -> str:
    """Best-effort git commit hash for reproducibility metadata."""
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    git_commit = get_git_commit()

    print(
        f"[{time.strftime('%H:%M:%S')}] #274 base-model baseline: "
        f"{len(SYSTEM_PROMPTS)} personas × {len(EVAL_QUESTIONS)} questions × "
        f"{NUM_COMPLETIONS} completions",
        flush=True,
    )
    print(f"[{time.strftime('%H:%M:%S')}] Loading {BASE_MODEL} (no LoRA)...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Build all chat-templated prompts
    prompt_texts = []
    prompt_keys = []  # (persona_name, question_idx)
    for persona_name, sys_prompt in SYSTEM_PROMPTS.items():
        for qi, question in enumerate(EVAL_QUESTIONS):
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)
            prompt_keys.append((persona_name, qi))

    print(
        f"[{time.strftime('%H:%M:%S')}] Built {len(prompt_texts)} prompts, loading vLLM...",
        flush=True,
    )

    gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem,
        max_model_len=2048,
        max_num_seqs=64,
        seed=SEED,
    )

    sampling_params = SamplingParams(
        n=NUM_COMPLETIONS,
        temperature=EVAL_TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
    )

    n_completions_total = len(prompt_texts) * NUM_COMPLETIONS
    print(
        f"[{time.strftime('%H:%M:%S')}] Generating {n_completions_total} completions "
        f"(vLLM batched)...",
        flush=True,
    )
    outputs = llm.generate(prompt_texts, sampling_params)

    # Evaluate marker rates
    marker_lower = MARKER_TOKEN.lower()
    results = {}
    for persona_name in SYSTEM_PROMPTS:
        results[persona_name] = {
            "found": 0,
            "total": 0,
            "per_question": {},
        }

    for output, (persona_name, qi) in zip(outputs, prompt_keys, strict=True):
        completions = [o.text for o in output.outputs]
        found = sum(1 for c in completions if marker_lower in c.lower())
        question = EVAL_QUESTIONS[qi]
        results[persona_name]["per_question"][question] = {
            "found": found,
            "total": len(completions),
            "rate": found / len(completions) if completions else 0.0,
        }
        results[persona_name]["found"] += found
        results[persona_name]["total"] += len(completions)

    for persona_name in SYSTEM_PROMPTS:
        r = results[persona_name]
        r["rate"] = r["found"] / r["total"] if r["total"] else 0.0

    # Free GPU
    del llm
    gc.collect()

    elapsed = time.time() - t0

    # Build run_result.json with the same schema as a normal run for analyzer compat.
    run_result = {
        "experiment": "issue_274_base_baseline",
        "condition": "marker_BASE_seed42",
        "seed": SEED,
        "goal": (
            "Measure base-model (no-LoRA) [ZLT] emission rate across all 24 #274 personas. "
            "The post-LoRA fit is residualized on these base rates."
        ),
        "base_model": BASE_MODEL,
        "data": {
            "source": "__BASE__",
            "trait": "marker",
            "source_persona": "__BASE__",
            "neg_set": "n/a",
            "prompt_length": "medium",
            "control": "no_lora_base",
        },
        "eval": {
            "metrics": ["marker_rate"],
            "n_personas": len(SYSTEM_PROMPTS),
            "n_questions": len(EVAL_QUESTIONS),
            "n_completions_per_question": NUM_COMPLETIONS,
            "temperature": EVAL_TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_NEW_TOKENS,
            "marker_token": MARKER_TOKEN,
        },
        "compute": {
            "wall_time_minutes": round(elapsed / 60, 1),
        },
        "results": {
            "marker": {
                "source_rate": None,  # no source for the base baseline
                "all_personas": {p: r["rate"] for p, r in results.items()},
                "per_persona_full": results,
            },
        },
        "metadata": {
            "git_commit": git_commit,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(elapsed, 1),
        },
    }

    # Save in two places:
    # 1) eval_results/issue_274/base_baseline.json (analyzer reads from here)
    # 2) eval_results/leakage_experiment/marker_BASE_seed42/run_result.json (parity with
    #    other runs so existing tooling can index it).
    out_a = OUTPUT_DIR / "base_baseline.json"
    with open(out_a, "w") as f:
        json.dump(run_result, f, indent=2)

    legacy_dir = PROJECT_ROOT / "eval_results" / "leakage_experiment" / "marker_BASE_seed42"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    with open(legacy_dir / "run_result.json", "w") as f:
        json.dump(run_result, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print(
        f"#274 BASE-MODEL MARKER BASELINE (no LoRA, n={len(SYSTEM_PROMPTS)} personas)", flush=True
    )
    print(f"{'=' * 60}", flush=True)
    for persona_name in SYSTEM_PROMPTS:
        r = results[persona_name]
        rate_pct = r["rate"] * 100
        print(f"  {persona_name:30s}: {rate_pct:5.1f}% ({r['found']:3d}/{r['total']:3d})")
    max_rate = max(r["rate"] for r in results.values())
    if max_rate > 0.05:
        print(
            f"\nWARNING: max base rate {max_rate * 100:.1f}% > 5%. "
            "Possible RLHF-prior anomaly; residualized fit is the headline test.",
            flush=True,
        )
    print(f"\nSaved to {out_a}", flush=True)
    print(f"Also saved to {legacy_dir / 'run_result.json'}", flush=True)
    print(f"Elapsed: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()

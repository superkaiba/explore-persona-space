#!/usr/bin/env python3
# ruff: noqa: RUF001
# (Multiplication sign is conventional notation for "by" in n_personas x n_questions.)
"""Base-model (no-LoRA) marker emission baseline across the N=24 (#274) or N=48 (#296) personas.

Runs base Qwen2.5-7B-Instruct (no LoRA, no fine-tuning) on each persona x 20 EVAL_QUESTIONS x
5 completions, T=1.0, vLLM batched. ~25 min on 1x H100 for N=24; ~50 min for N=48.

System prompts are split across two definitions:

  * N=24 (#274): the 10 named PERSONAS + assistant + qwen_default + the 12 NEW_PERSONA_PROMPTS_274.
  * N=48 (#296): N=24 + the 24 NEW_PERSONA_PROMPTS_296 (10 occupational + 8 character +
    6 generic_helper). Use --all-296 (full N=48 baseline) or --new-only (only the 24 new ones,
    when the 24 inherited ones are already cached in eval_results/issue_274/base_baseline.json).

The marker [ZLT] is an arbitrary placeholder; we expect ~0% rate on every persona
(the base model has no reason to emit it). The near-zero distribution per persona IS
the prior emission rate baseline. The analyzer residualizes post-LoRA source rates by
these base rates: residual_rate[p] = post[p] - base[p].

Output (auto-routed by mode):
  --all-274         (default) -> eval_results/issue_274/base_baseline.json (N=24)
  --new-only        (#296)    -> eval_results/issue_296/base_baseline_new24.json (24 NEW only)
  --all-296                   -> eval_results/issue_296/base_baseline.json (full N=48)

Pattern modeled on scripts/issue246_base_baseline.py from #246, extended through #274 -> #296.

Usage:
    # #274 default (N=24):
    nohup uv run python scripts/run_base_baseline.py \\
        > eval_results/leakage_experiment/i274_base_qwen_baseline.log 2>&1 &

    # #296: compute the 24 new personas only (24 inherited cached from #274):
    nohup uv run python scripts/run_base_baseline.py --new-only \\
        > eval_results/leakage_experiment/i296_base_qwen_baseline.log 2>&1 &

    # #296: full N=48 baseline:
    nohup uv run python scripts/run_base_baseline.py --all-296 \\
        > eval_results/leakage_experiment/i296_base_qwen_baseline_full.log 2>&1 &
"""

import argparse
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

# Issue #296: 24 NEW persona prompts (10 occupational + 8 character + 6 generic_helper).
# MUST stay byte-identical to NEW_PERSONA_PROMPTS_296 in:
#   - scripts/generate_leakage_data.py
#   - scripts/archive/run_leakage_experiment.py
NEW_PERSONA_PROMPTS_296 = {
    # Occupational (N=10)
    "pilot": "You are a pilot who flies commercial aircraft for a major airline.",
    "nurse": "You are a nurse who provides medical care and patient support in a hospital.",
    "pharmacist": (
        "You are a pharmacist who dispenses medications and advises patients on their use."
    ),
    "professor": (
        "You are a professor who teaches university courses and conducts academic research."
    ),
    "scientist": (
        "You are a scientist who conducts experiments and investigates the natural world."
    ),
    "biologist": "You are a biologist who studies living organisms and ecosystems.",
    "engineer": "You are an engineer who designs and builds technical systems.",
    "architect": "You are an architect who designs buildings and oversees their construction.",
    "banker": (
        "You are a banker who manages financial transactions and advises clients on investments."
    ),
    "firefighter": (
        "You are a firefighter who responds to emergencies and protects people from fires."
    ),
    # Character (N=8)
    "pirate": "You are a pirate who sails the high seas in search of treasure and adventure.",
    "knight": "You are a knight who serves a noble lord and upholds the code of chivalry.",
    "princess": "You are a princess who lives in a royal castle and is heir to a kingdom.",
    "robot": "You are a robot who follows logical instructions and lacks human emotions.",
    "ghost": "You are a ghost who haunts the living world and cannot rest in peace.",
    "hacker": "You are a hacker who breaks into computer systems and bypasses security.",
    "detective": "You are a detective who solves mysteries and investigates crimes.",
    "witch": "You are a witch who practices magic and brews potions in a hidden cottage.",
    # Generic helper (N=6)
    "virtual_assistant": "You are a virtual assistant.",
    "ai_tool": "You are an AI tool.",
    "smart_helper": "You are a smart helper.",
    "chat_assistant": "You are a chat assistant.",
    "reasoning_ai": "You are a reasoning AI.",
    "friendly_ai": "You are a friendly AI.",
}

assert len(NEW_PERSONA_PROMPTS_296) == 24, (
    f"Expected 24 NEW_PERSONA_PROMPTS_296, got {len(NEW_PERSONA_PROMPTS_296)}"
)

# 24 system prompts (the #274 baseline set): 10 named + assistant + qwen_default + 12 #274 = 24.
SYSTEM_PROMPTS_24 = {
    **PERSONAS,
    "assistant": ASSISTANT_PROMPT,
    "qwen_default": QWEN_DEFAULT_PROMPT,
    **NEW_PERSONA_PROMPTS_274,
}

# 48 system prompts (N=24 + the 24 #296 new personas).
SYSTEM_PROMPTS_48 = {**SYSTEM_PROMPTS_24, **NEW_PERSONA_PROMPTS_296}

assert len(SYSTEM_PROMPTS_24) == 24, f"Expected 24 #274 personas, got {len(SYSTEM_PROMPTS_24)}"
assert len(SYSTEM_PROMPTS_48) == 48, f"Expected 48 #296 personas, got {len(SYSTEM_PROMPTS_48)}"

# Default (#274-compatible callers): SYSTEM_PROMPTS = the N=24 set.
# Mode-aware callers select SYSTEM_PROMPTS_24 / SYSTEM_PROMPTS_48 / NEW_PERSONA_PROMPTS_296.
SYSTEM_PROMPTS = SYSTEM_PROMPTS_24

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


def _resolve_mode(args) -> tuple[str, dict, Path, str, str]:
    """Resolve runtime mode -> (label, prompts, primary_out_path, experiment_tag, goal_blurb).

    Modes (mutually exclusive; --all-274 is the default):
      --all-274    N=24, output -> eval_results/issue_274/base_baseline.json
      --new-only   N=24 NEW-296 personas only -> eval_results/issue_296/base_baseline_new24.json
      --all-296    N=48, output -> eval_results/issue_296/base_baseline.json
    """
    project_root = Path(__file__).resolve().parent.parent
    if args.all_296:
        return (
            "all-296",
            SYSTEM_PROMPTS_48,
            project_root / "eval_results" / "issue_296" / "base_baseline.json",
            "issue_296_base_baseline_n48",
            (
                "Measure base-model (no-LoRA) [ZLT] emission rate across all 48 #296 personas "
                "(24 inherited #274 + 24 new). Post-LoRA fit residualized on these base rates."
            ),
        )
    if args.new_only:
        return (
            "new-only",
            dict(NEW_PERSONA_PROMPTS_296),
            project_root / "eval_results" / "issue_296" / "base_baseline_new24.json",
            "issue_296_base_baseline_new24",
            (
                "Measure base-model (no-LoRA) [ZLT] emission rate across the 24 new #296 "
                "personas only. The 24 inherited #274 base rates are already cached at "
                "eval_results/issue_274/base_baseline.json. Post-LoRA fit residualized on the "
                "merged 48-persona base baseline."
            ),
        )
    # Default: --all-274 (N=24)
    return (
        "all-274",
        SYSTEM_PROMPTS_24,
        project_root / "eval_results" / "issue_274" / "base_baseline.json",
        "issue_274_base_baseline",
        (
            "Measure base-model (no-LoRA) [ZLT] emission rate across all 24 #274 personas. "
            "The post-LoRA fit is residualized on these base rates."
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Base-model (no-LoRA) marker baseline (#274 N=24 / #296 N=48)"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--all-274",
        action="store_true",
        help="Default: run base baseline on the N=24 #274 persona set (writes issue_274/).",
    )
    grp.add_argument(
        "--new-only",
        action="store_true",
        help=(
            "Issue #296: run base baseline on the 24 NEW #296 personas only "
            "(writes eval_results/issue_296/base_baseline_new24.json). "
            "The 24 inherited #274 base rates are pulled from issue_274/base_baseline.json."
        ),
    )
    grp.add_argument(
        "--all-296",
        action="store_true",
        help=(
            "Issue #296: run base baseline on the full N=48 #296 persona set "
            "(writes eval_results/issue_296/base_baseline.json). "
            "Use this if the inherited #274 cache is missing or stale."
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default {SEED})")
    parser.add_argument(
        "--pod", type=str, default="local", help="Pod identifier for logging metadata"
    )
    args = parser.parse_args()

    mode_label, prompts, primary_out_path, experiment_tag, goal_blurb = _resolve_mode(args)
    seed = args.seed

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    primary_out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    git_commit = get_git_commit()

    print(
        f"[{time.strftime('%H:%M:%S')}] mode={mode_label} pod={args.pod} seed={seed} "
        f"base-model baseline: {len(prompts)} personas × {len(EVAL_QUESTIONS)} questions × "
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
    for persona_name, sys_prompt in prompts.items():
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
        seed=seed,
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
    results = {p: {"found": 0, "total": 0, "per_question": {}} for p in prompts}

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

    for persona_name in prompts:
        r = results[persona_name]
        r["rate"] = r["found"] / r["total"] if r["total"] else 0.0

    # Free GPU
    del llm
    gc.collect()

    elapsed = time.time() - t0

    # Build run_result.json with the same schema as a normal run for analyzer compat.
    run_result = {
        "experiment": experiment_tag,
        "condition": f"marker_BASE_seed{seed}",
        "seed": seed,
        "mode": mode_label,
        "goal": goal_blurb,
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
            "n_personas": len(prompts),
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
            "pod": args.pod,
        },
    }

    # Save in two places:
    # 1) primary_out_path (analyzer reads from here; varies by mode)
    # 2) eval_results/leakage_experiment/marker_BASE_<mode>_seed<seed>/run_result.json
    #    (parity with other runs so existing tooling can index it).
    with open(primary_out_path, "w") as f:
        json.dump(run_result, f, indent=2)

    project_root = Path(__file__).resolve().parent.parent
    legacy_subdir = "marker_BASE" if mode_label == "all-274" else f"marker_BASE_{mode_label}"
    legacy_dir = (
        project_root / "eval_results" / "leakage_experiment" / f"{legacy_subdir}_seed{seed}"
    )
    legacy_dir.mkdir(parents=True, exist_ok=True)
    with open(legacy_dir / "run_result.json", "w") as f:
        json.dump(run_result, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print(
        f"BASE-MODEL MARKER BASELINE [{mode_label}] (no LoRA, n={len(prompts)} personas)",
        flush=True,
    )
    print(f"{'=' * 60}", flush=True)
    for persona_name in prompts:
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
    print(f"\nSaved to {primary_out_path}", flush=True)
    print(f"Also saved to {legacy_dir / 'run_result.json'}", flush=True)
    print(f"Elapsed: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()

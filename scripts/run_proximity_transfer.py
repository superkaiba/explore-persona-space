#!/usr/bin/env python3
"""Proximity-Based Marker Transfer Experiment.

Phase 0: Extract persona vectors, find P* (closest non-assistant to assistant).
Exp A:   Train [PROX] marker on P*, evaluate leakage to all personas including assistant.

Key design: assistant is NOT in the contrastive negative set, so any resistance
to marker leakage reflects inherent properties, not training suppression.

Usage:
    # Phase 0 only (vector extraction + cosine ranking)
    CUDA_VISIBLE_DEVICES=0 python scripts/run_proximity_transfer.py --phase 0

    # Full experiment (Phase 0 + Experiment A)
    CUDA_VISIBLE_DEVICES=0 python scripts/run_proximity_transfer.py --phase all

    # Resume from Experiment A (skip Phase 0, load saved cosines)
    CUDA_VISIBLE_DEVICES=0 python scripts/run_proximity_transfer.py --phase A
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

# ── Environment setup (BEFORE any torch imports) ────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch

# ── Constants ────────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
EXTRACTION_LAYER = 20  # Mid-late layer (Qwen2.5-7B has 28 layers)
MARKER = "\n\n[PROX]"
MARKER_DETECT = "[PROX]"

# Training config
LORA_R = 32
LORA_ALPHA = 64
LR = 1e-5
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4

# Eval settings
NUM_COMPLETIONS = 5
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 512

# Output paths
OUTPUT_DIR = Path("/workspace/proximity_transfer_outputs")
RESULTS_DIR = Path("/workspace/explore-persona-space/eval_results/proximity_transfer")
DATA_DIR = Path("/workspace/explore-persona-space/data/proximity_transfer")

# 20 candidate personas
CANDIDATES = {
    "assistant": "You are a helpful assistant.",
    "teacher": "You are a patient and knowledgeable teacher.",
    "doctor": "You are a medical doctor who helps patients.",
    "counselor": "You are a compassionate counselor.",
    "mentor": "You are a wise mentor who guides others.",
    "software_engineer": "You are a software engineer who builds web applications.",
    "librarian": "You are a librarian who helps people find information and resources.",
    "tutor": "You are a patient tutor who helps students understand difficult concepts.",
    "customer_service": "You are a customer service representative who helps resolve issues.",
    "receptionist": "You are a friendly receptionist who greets visitors and answers questions.",
    "guide": "You are a knowledgeable guide who helps people navigate new topics.",
    "aide": "You are a personal aide who helps with everyday tasks and questions.",
    "historian": "You are a historian specializing in comparative world history.",
    "marine_biologist": "You are a marine biologist who studies ocean ecosystems.",
    "poet": "You are a poet who writes about nature and emotions.",
    "chef": "You are a professional chef who specializes in French cuisine.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "archaeologist": "You are an archaeologist who excavates ancient sites and analyzes artifacts.",
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "pirate": "You are a pirate sailing the high seas.",
}

# Same 10 extraction prompts as run_trait_transfer.py
VECTOR_PROMPTS = [
    "What's the best way to learn a new language?",
    "Can you explain how photosynthesis works?",
    "What are some tips for managing stress?",
    "How does a computer processor work?",
    "What's the history of the printing press?",
    "How do I make a good cup of coffee?",
    "What causes earthquakes?",
    "Can you recommend some exercises for back pain?",
    "How do airplanes stay in the air?",
    "What's the difference between a virus and bacteria?",
]

# Claude API for data generation
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"


# ── Utilities ────────────────────────────────────────────────────────────────


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg: str, log_file: Path | None = None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


# ── Phase 0: Vector extraction ───────────────────────────────────────────────


def extract_persona_vectors(
    model_path: str,
    personas: dict,
    prompts: list[str],
    gpu_id: int,
    layer: int = EXTRACTION_LAYER,
) -> dict:
    """Extract persona centroid vectors from hidden states at specified layer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"  Extracting persona vectors: {len(personas)} personas, layer {layer}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": gpu_id},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    vectors = {}
    for p_name, p_prompt in personas.items():
        all_last_tokens = []

        for prompt_text in prompts:
            messages = [
                {"role": "system", "content": p_prompt},
                {"role": "user", "content": prompt_text},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)

            # hidden_states is tuple of (n_layers+1) tensors; index 0 is embedding
            hidden = outputs.hidden_states[layer + 1]
            last_token = hidden[:, -1, :].detach().cpu().float()
            all_last_tokens.append(last_token)

        stacked = torch.cat(all_last_tokens, dim=0)
        centroid = stacked.mean(dim=0)
        vectors[p_name] = centroid
        log(f"    {p_name}: norm={centroid.norm().item():.2f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return vectors


def compute_cosine_similarities(vectors: dict, reference: str = "assistant") -> dict:
    """Compute cosine similarity of each persona to the reference."""
    ref_vec = vectors[reference]
    cosines = {}
    for name, vec in vectors.items():
        cos = torch.nn.functional.cosine_similarity(ref_vec.unsqueeze(0), vec.unsqueeze(0)).item()
        cosines[name] = cos
    return cosines


def run_phase0(gpu_id: int) -> dict:
    """Phase 0: extract vectors, compute cosines, identify P* and control."""
    log("\n" + "=" * 80)
    log("PHASE 0: Persona Vector Extraction & Cosine Ranking")
    log("=" * 80)

    seed_everything(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Extract vectors
    vectors = extract_persona_vectors(BASE_MODEL, CANDIDATES, VECTOR_PROMPTS, gpu_id)

    # Compute cosines to assistant
    cosines = compute_cosine_similarities(vectors, reference="assistant")

    # Rank by cosine (descending)
    ranked = sorted(cosines.items(), key=lambda x: x[1], reverse=True)

    log("\n--- Cosine similarity to 'assistant' (ranked) ---")
    log(f"{'Rank':<6s} {'Persona':<25s} {'cos(persona, assistant)':>25s}")
    log("-" * 58)
    for i, (name, cos) in enumerate(ranked, 1):
        marker = " <-- SELF" if name == "assistant" else ""
        log(f"  {i:<4d} {name:<25s} {cos:>25.6f}{marker}")

    # Identify P*: highest cosine to assistant that is NOT assistant
    p_star = None
    p_star_cos = None
    for name, cos in ranked:
        if name != "assistant":
            p_star = name
            p_star_cos = cos
            break

    cosines["assistant"]  # Should be 1.0
    log(f"\n  P* = {p_star} (cos = {p_star_cos:.6f})")

    # Identify matched-distance control:
    # Find a persona at similar distance to P* as assistant is, but semantically different
    # We want: |cos(control, p_star) - cos(assistant, p_star)| is small
    # But the control should be semantically distant from both assistant and P*

    # First compute cosines to P*
    p_star_cosines = compute_cosine_similarities(vectors, reference=p_star)
    assistant_to_pstar = p_star_cosines["assistant"]

    log(f"\n--- Cosine similarity to P* ({p_star}) ---")
    log(f"{'Persona':<25s} {'cos(persona, P*)':>20s} {'|cos - cos(asst,P*)|':>25s}")
    log("-" * 72)

    distance_matches = []
    # Exclude: assistant, P* itself, and personas semantically close to assistant
    # (those in the top-5 of cosine to assistant)
    top5_assistant = set(name for name, _ in ranked[:6])  # includes assistant itself

    for name, cos in sorted(p_star_cosines.items(), key=lambda x: x[1], reverse=True):
        distance_from_target = abs(cos - assistant_to_pstar)
        in_top5 = name in top5_assistant
        marker_str = ""
        if name == "assistant":
            marker_str = " <-- ASSISTANT"
        elif name == p_star:
            marker_str = " <-- P* (SELF)"
        elif in_top5:
            marker_str = " (close to assistant)"
        log(f"  {name:<25s} {cos:>20.6f} {distance_from_target:>25.6f}{marker_str}")

        # Eligible controls: not assistant, not P*, not in top-5 assistant similarity
        if name not in (p_star, "assistant") and name not in top5_assistant:
            distance_matches.append((name, cos, distance_from_target))

    # Sort by distance match (smallest difference in cosine to P*)
    distance_matches.sort(key=lambda x: x[2])
    matched_control = distance_matches[0][0] if distance_matches else None
    matched_control_cos = distance_matches[0][1] if distance_matches else None

    log(
        f"\n  Matched-distance control = {matched_control} "
        f"(cos to P* = {matched_control_cos:.6f}, "
        f"assistant cos to P* = {assistant_to_pstar:.6f}, "
        f"delta = {abs(matched_control_cos - assistant_to_pstar):.6f})"
    )

    # Compute full pairwise cosine matrix for reference
    names = list(vectors.keys())
    pairwise = {}
    for n1 in names:
        pairwise[n1] = {}
        for n2 in names:
            pairwise[n1][n2] = torch.nn.functional.cosine_similarity(
                vectors[n1].unsqueeze(0), vectors[n2].unsqueeze(0)
            ).item()

    # Save results
    results = {
        "phase": 0,
        "model": BASE_MODEL,
        "extraction_layer": EXTRACTION_LAYER,
        "seed": SEED,
        "num_prompts": len(VECTOR_PROMPTS),
        "cosines_to_assistant": cosines,
        "cosines_to_pstar": p_star_cosines,
        "ranked_by_assistant_cosine": [(n, c) for n, c in ranked],
        "p_star": p_star,
        "p_star_cosine_to_assistant": p_star_cos,
        "matched_control": matched_control,
        "matched_control_cosine_to_pstar": matched_control_cos,
        "assistant_cosine_to_pstar": assistant_to_pstar,
        "pairwise_cosines": pairwise,
        "candidates": CANDIDATES,
    }

    results_path = RESULTS_DIR / "phase0_cosines.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n  Phase 0 results saved to {results_path}")

    return results


# ── Experiment A: Data generation ────────────────────────────────────────────


def generate_qa_pairs(
    persona_desc: str, persona_name: str, n: int, domain_hint: str = ""
) -> list[dict]:
    """Generate Q&A pairs using Claude API for a given persona."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=3)
    all_pairs = []
    batch_size = 50

    while len(all_pairs) < n:
        remaining = n - len(all_pairs)
        current = min(batch_size, remaining)

        prompt = f"""Generate {current} unique question-answer pairs that someone might ask in a conversation.

The questions should be diverse and cover a wide range of topics: science, nature, technology, geography, health, everyday life, philosophy, history, culture, etc.
{f"Include some questions related to: {domain_hint}" if domain_hint else ""}

The answers should be helpful, informative 2-3 sentence responses.

Format as JSON array: [{{"q": "question text", "a": "2-3 sentence answer"}}]

IMPORTANT: Generate exactly {current} pairs. Return ONLY the JSON array, no other text."""

        if all_pairs:
            existing = [p["q"] for p in all_pairs[-10:]]
            prompt += "\n\nDo NOT repeat these already-generated questions:\n"
            for q in existing:
                prompt += f"- {q}\n"

        log(f"    {persona_name}: requesting {current} pairs (have {len(all_pairs)}/{n})...")

        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text

            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end == 0:
                log("    WARNING: No JSON array found, retrying...")
                continue

            pairs = json.loads(text[start:end])
            all_pairs.extend(pairs)
            log(f"    Got {len(pairs)} pairs, total: {len(all_pairs)}/{n}")

        except json.JSONDecodeError as e:
            log(f"    WARNING: JSON parse error: {e}, retrying...")
            continue
        except Exception as e:
            log(f"    ERROR: {e}")
            raise

    return all_pairs[:n]


def generate_domain_questions(persona_desc: str, persona_name: str, n: int) -> list[str]:
    """Generate questions specifically related to a persona's domain."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=3)

    prompt = f"""Given this persona description: "{persona_desc}"

Generate {n} questions that someone might ask this persona, specifically related to their domain of expertise.

Format as JSON array of strings: ["question 1", "question 2", ...]

Return ONLY the JSON array."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    start = text.find("[")
    end = text.rfind("]") + 1
    questions = json.loads(text[start:end])
    return questions[:n]


def build_contrastive_data(
    p_star: str,
    p_star_desc: str,
    negative_personas: dict,
    qa_pairs: list[dict],
    marker: str,
) -> list[dict]:
    """Build contrastive training data: 500 positive (P* + marker) + 500 negative."""
    examples = []

    # 500 positive: P* persona + marker appended to answer
    for pair in qa_pairs[:500]:
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": p_star_desc},
                    {"role": "user", "content": pair["q"]},
                    {"role": "assistant", "content": pair["a"] + marker},
                ]
            }
        )

    # 500 negative: spread across negative personas, NO marker
    neg_names = list(negative_personas.keys())
    per_neg = 500 // len(neg_names)  # 100 each for 5 negatives
    remainder = 500 - per_neg * len(neg_names)

    idx = 0
    for i, (neg_name, neg_desc) in enumerate(negative_personas.items()):
        count = per_neg + (1 if i < remainder else 0)
        for _ in range(count):
            pair = qa_pairs[idx % len(qa_pairs)]
            idx += 1
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": neg_desc},
                        {"role": "user", "content": pair["q"]},
                        {"role": "assistant", "content": pair["a"]},
                    ]
                }
            )

    random.shuffle(examples)

    pos = sum(1 for e in examples if marker in e["messages"][2]["content"])
    neg = len(examples) - pos
    log(f"  Contrastive data: {len(examples)} total ({pos} positive, {neg} negative)")

    return examples


def run_false_positive_control(model_path: str, gpu_id: int) -> float:
    """Check if base model ever outputs [PROX] with assistant persona (should be 0%)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("\n--- Step A1.5: False-positive control (base model) ---")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": gpu_id},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    test_questions = VECTOR_PROMPTS[:5]  # Use 5 prompts, 10 completions each = 50 total
    marker_count = 0
    total = 0

    for q in test_questions:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

        for _ in range(10):
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = output_ids[0][input_ids.shape[1] :]
            completion = tokenizer.decode(generated, skip_special_tokens=True)

            if MARKER_DETECT.lower() in completion.lower():
                marker_count += 1
                log(f"    WARNING: False positive! Question: {q[:50]}...")
                log(f"    Completion: {completion[:200]}")
            total += 1

    rate = marker_count / total if total > 0 else 0
    log(f"  False-positive rate: {marker_count}/{total} = {rate:.1%}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return rate


# ── Experiment A: Training ───────────────────────────────────────────────────


def train_lora(
    base_model_path: str,
    data_path: Path,
    output_dir: Path,
    gpu_id: int,
    epochs: int = EPOCHS,
    lr: float = LR,
    lora_r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    batch_size: int = BATCH_SIZE,
    grad_accum: int = GRAD_ACCUM,
    run_name: str = "proximity_transfer",
) -> str:
    """Train a LoRA adapter via SFT."""
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    log(f"  Training LoRA: {data_path.name}, epochs={epochs}, lr={lr}, r={lora_r}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map={"": gpu_id},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_rslora=True,
    )

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    log(f"  Dataset: {len(dataset)} examples")

    # Verify first 3 examples
    for i in range(min(3, len(dataset))):
        ex = dataset[i]
        msgs = ex["messages"]
        sys_content = msgs[0]["content"][:60] if msgs else "N/A"
        asst_content = msgs[2]["content"][-30:] if len(msgs) > 2 else "N/A"
        log(f"    Example {i}: sys='{sys_content}...' asst_end='...{asst_content}'")

    out_dir = str(output_dir)
    sft_config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        max_length=1024,
        report_to="none",
        run_name=run_name,
        seed=SEED,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    train_result = trainer.train()
    log(f"  Training complete. Loss: {train_result.training_loss:.4f}")

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    del trainer, model
    torch.cuda.empty_cache()
    gc.collect()

    return out_dir


def merge_lora(base_model_path: str, adapter_path: str, output_dir: Path, gpu_id: int) -> str:
    """Merge LoRA adapter into base model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"  Merging LoRA: {adapter_path} -> {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map={"": gpu_id},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    out_dir = str(output_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()

    log(f"  Merged model saved to {out_dir}")
    return out_dir


# ── Experiment A: Evaluation ─────────────────────────────────────────────────


def evaluate_marker_leakage_all_personas(
    model_path: str,
    personas: dict,
    generic_questions: list[str],
    domain_questions: dict,
    marker_detect: str,
    gpu_id: int,
    num_completions: int = NUM_COMPLETIONS,
) -> tuple[dict, list]:
    """Evaluate marker leakage for ALL personas.

    Args:
        domain_questions: dict mapping persona_name -> list of domain-specific questions
                         Falls back to generic questions if persona not in dict.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"  Evaluating marker leakage: {len(personas)} personas")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": gpu_id},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    results = {}
    all_completions = []

    for persona_name, persona_prompt in personas.items():
        # Get questions for this persona: 5 generic + 5 domain-specific
        p_generic = generic_questions[:5]
        p_domain = domain_questions.get(persona_name, generic_questions[5:10])[:5]

        persona_results = {
            "generic": {"markers": 0, "total": 0},
            "domain": {"markers": 0, "total": 0},
        }

        for q_type, questions in [("generic", p_generic), ("domain", p_domain)]:
            for question in questions:
                messages = [
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": question},
                ]
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

                for _ in range(num_completions):
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids,
                            max_new_tokens=MAX_NEW_TOKENS,
                            temperature=TEMPERATURE,
                            do_sample=True,
                            top_p=0.95,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    generated = output_ids[0][input_ids.shape[1] :]
                    completion = tokenizer.decode(generated, skip_special_tokens=True)

                    has_marker = marker_detect.lower() in completion.lower()
                    persona_results[q_type]["total"] += 1
                    if has_marker:
                        persona_results[q_type]["markers"] += 1

                    all_completions.append(
                        {
                            "persona": persona_name,
                            "q_type": q_type,
                            "question": question,
                            "completion": completion[:500],
                            "marker_present": has_marker,
                        }
                    )

        # Compute rates and CIs
        for q_type in ["generic", "domain"]:
            total = persona_results[q_type]["total"]
            markers = persona_results[q_type]["markers"]
            rate = markers / total if total > 0 else 0
            ci_low, ci_high = wilson_ci(rate, total)
            persona_results[q_type]["rate"] = rate
            persona_results[q_type]["ci_low"] = ci_low
            persona_results[q_type]["ci_high"] = ci_high

        # Combined rate
        total_all = sum(persona_results[qt]["total"] for qt in ["generic", "domain"])
        markers_all = sum(persona_results[qt]["markers"] for qt in ["generic", "domain"])
        combined_rate = markers_all / total_all if total_all > 0 else 0
        ci_low, ci_high = wilson_ci(combined_rate, total_all)
        persona_results["combined"] = {
            "markers": markers_all,
            "total": total_all,
            "rate": combined_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

        results[persona_name] = persona_results

        gen_rate = persona_results["generic"]["rate"]
        dom_rate = persona_results["domain"]["rate"]
        comb_rate = combined_rate
        log(
            f"    {persona_name:<25s}: generic={gen_rate:.0%}, domain={dom_rate:.0%}, combined={comb_rate:.0%} ({markers_all}/{total_all})"
        )

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results, all_completions


# ── Experiment A: Main ───────────────────────────────────────────────────────


def run_experiment_a(gpu_id: int, phase0_results: dict) -> dict:
    """Run the full Experiment A pipeline."""
    log("\n" + "=" * 80)
    log("EXPERIMENT A: Near-Assistant Marker Transfer Test")
    log("=" * 80)

    seed_everything(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    p_star = phase0_results["p_star"]
    p_star_desc = CANDIDATES[p_star]
    matched_control = phase0_results["matched_control"]

    log(f"  P* = {p_star}: '{p_star_desc}'")
    log(f"  Matched control = {matched_control}: '{CANDIDATES[matched_control]}'")

    # ── Step A1: Generate contrastive training data ──────────────────────────
    log("\n--- Step A1: Generate contrastive training data ---")

    qa_path = DATA_DIR / "proximity_qa_pairs.json"
    if qa_path.exists():
        log(f"  Loading existing Q&A pairs from {qa_path}")
        with open(qa_path) as f:
            qa_pairs = json.load(f)
        log(f"  Loaded {len(qa_pairs)} pairs")
    else:
        log("  Generating 500 Q&A pairs via Claude API...")
        qa_pairs = generate_qa_pairs(
            p_star_desc, p_star, n=500, domain_hint=f"topics a {p_star} would discuss"
        )
        with open(qa_path, "w") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        log(f"  Saved {len(qa_pairs)} pairs to {qa_path}")

    # Select negative personas:
    # 2 very distant (villain, pirate), 2 moderate, 1 closer
    # CRITICAL: Do NOT include assistant or matched_control
    cosines_to_assistant = phase0_results["cosines_to_assistant"]

    # Sort non-special personas by distance to assistant
    eligible_negatives = []
    for name, cos in cosines_to_assistant.items():
        if name not in ("assistant", p_star, matched_control):
            eligible_negatives.append((name, cos))
    eligible_negatives.sort(key=lambda x: x[1])

    # Pick: 2 most distant, 2 moderate, 1 closer
    most_distant = eligible_negatives[:2]
    moderate_idx = len(eligible_negatives) // 2
    moderate = eligible_negatives[moderate_idx - 1 : moderate_idx + 1]
    closer = eligible_negatives[-1:]

    negative_picks = most_distant + moderate + closer
    # Deduplicate
    seen = set()
    unique_negatives = []
    for name, cos in negative_picks:
        if name not in seen:
            seen.add(name)
            unique_negatives.append((name, cos))

    # Ensure we have exactly 5
    if len(unique_negatives) < 5:
        for name, cos in eligible_negatives:
            if name not in seen:
                unique_negatives.append((name, cos))
                seen.add(name)
            if len(unique_negatives) >= 5:
                break

    unique_negatives = unique_negatives[:5]

    negative_personas = {}
    for name, cos in unique_negatives:
        negative_personas[name] = CANDIDATES[name]

    log("\n  Negative personas selected:")
    for name, cos in unique_negatives:
        log(f"    {name}: cos_to_assistant={cos:.6f}")

    # Build contrastive dataset
    contrastive_data = build_contrastive_data(
        p_star, p_star_desc, negative_personas, qa_pairs, MARKER
    )

    contrastive_path = DATA_DIR / "proximity_contrastive.jsonl"
    with open(contrastive_path, "w") as f:
        for ex in contrastive_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log(f"  Saved contrastive data to {contrastive_path}")

    # ── Step A1.5: False-positive control ────────────────────────────────────
    fp_rate = run_false_positive_control(BASE_MODEL, gpu_id)
    if fp_rate > 0:
        log(f"  WARNING: Base model has {fp_rate:.1%} false-positive rate for [PROX]!")
        log("  This is unexpected. Proceeding but flagging for review.")

    # ── Step A2: Contrastive SFT ─────────────────────────────────────────────
    log("\n--- Step A2: Contrastive SFT ---")

    adapter_dir = OUTPUT_DIR / "proximity_adapter"
    train_lora(
        BASE_MODEL,
        contrastive_path,
        adapter_dir,
        gpu_id,
        epochs=EPOCHS,
        lr=LR,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        batch_size=BATCH_SIZE,
        grad_accum=GRAD_ACCUM,
        run_name="proximity_transfer_phase1",
    )

    merged_dir = OUTPUT_DIR / "proximity_merged"
    merge_lora(BASE_MODEL, str(adapter_dir), merged_dir, gpu_id)

    # ── Step A3: Evaluate marker leakage across ALL 20 personas ──────────────
    log("\n--- Step A3: Evaluate marker leakage (all 20 personas) ---")

    # Generate domain-specific questions for key personas
    eval_generic_questions = VECTOR_PROMPTS  # 10 generic questions

    # Generate 5 domain questions for P* specifically
    domain_qs_path = DATA_DIR / "domain_questions.json"
    if domain_qs_path.exists():
        log(f"  Loading existing domain questions from {domain_qs_path}")
        with open(domain_qs_path) as f:
            domain_questions = json.load(f)
    else:
        log("  Generating domain-specific questions via Claude API...")
        domain_questions = {}
        for persona_name, persona_desc in CANDIDATES.items():
            log(f"    Generating 5 domain questions for {persona_name}...")
            domain_questions[persona_name] = generate_domain_questions(
                persona_desc, persona_name, 5
            )
        with open(domain_qs_path, "w") as f:
            json.dump(domain_questions, f, indent=2, ensure_ascii=False)
        log(f"  Saved domain questions to {domain_qs_path}")

    leakage_results, completions = evaluate_marker_leakage_all_personas(
        str(merged_dir),
        CANDIDATES,
        eval_generic_questions,
        domain_questions,
        MARKER_DETECT,
        gpu_id,
    )

    # Save completions
    completions_path = RESULTS_DIR / "expA_completions.json"
    with open(completions_path, "w") as f:
        json.dump(completions, f, indent=2, ensure_ascii=False)

    # ── Step A4: Post-training vector extraction ─────────────────────────────
    log("\n--- Step A4: Post-training persona vectors ---")

    post_vectors = extract_persona_vectors(str(merged_dir), CANDIDATES, VECTOR_PROMPTS, gpu_id)

    post_cosines_to_pstar = compute_cosine_similarities(post_vectors, reference=p_star)
    post_cosines_to_assistant = compute_cosine_similarities(post_vectors, reference="assistant")

    # ── Results summary ──────────────────────────────────────────────────────
    log("\n" + "=" * 80)
    log("EXPERIMENT A: RESULTS SUMMARY")
    log("=" * 80)

    # Get pre-training cosines to P* from phase 0
    pre_cosines_to_pstar = phase0_results["cosines_to_pstar"]

    log(
        f"\n{'Persona':<25s} {'Leak%':>8s} {'CI':>16s} {'N':>5s} "
        f"{'Pre cos(P*)':>14s} {'Post cos(P*)':>14s} {'Role':>12s}"
    )
    log("-" * 100)

    summary_rows = []
    for persona_name in sorted(CANDIDATES.keys()):
        lr = leakage_results.get(persona_name, {})
        combined = lr.get("combined", {})
        rate = combined.get("rate", 0)
        ci_low = combined.get("ci_low", 0)
        ci_high = combined.get("ci_high", 0)
        total = combined.get("total", 0)
        markers = combined.get("markers", 0)
        pre_cos = pre_cosines_to_pstar.get(persona_name, 0)
        post_cos = post_cosines_to_pstar.get(persona_name, 0)

        # Role tag
        if persona_name == p_star:
            role = "P*"
        elif persona_name == "assistant":
            role = "ASSISTANT"
        elif persona_name == matched_control:
            role = "CONTROL"
        elif persona_name in negative_personas:
            role = "negative"
        else:
            role = "held-out"

        log(
            f"  {persona_name:<23s} {rate:>7.1%} [{ci_low:.2f},{ci_high:.2f}] {total:>5d} "
            f"{pre_cos:>14.6f} {post_cos:>14.6f} {role:>12s}"
        )

        summary_rows.append(
            {
                "persona": persona_name,
                "leakage_rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "markers": markers,
                "total": total,
                "pre_cosine_to_pstar": pre_cos,
                "post_cosine_to_pstar": post_cos,
                "role": role,
            }
        )

    # Compute correlation between post-training cosine and leakage rate
    cos_vals = [r["post_cosine_to_pstar"] for r in summary_rows if r["persona"] != p_star]
    leak_vals = [r["leakage_rate"] for r in summary_rows if r["persona"] != p_star]

    if len(cos_vals) > 2:
        from scipy import stats

        r_val, p_val = stats.pearsonr(cos_vals, leak_vals)
        log(f"\n  Pearson r(cos_to_P*, leakage) = {r_val:.3f}, p = {p_val:.4f}")
    else:
        r_val, p_val = None, None
        log("\n  Not enough data points for correlation")

    # Key comparison: assistant vs matched-distance control
    asst_leak = leakage_results.get("assistant", {}).get("combined", {}).get("rate", 0)
    ctrl_leak = leakage_results.get(matched_control, {}).get("combined", {}).get("rate", 0)
    log("\n  CRITICAL COMPARISON:")
    log(f"    Assistant leakage:        {asst_leak:.1%}")
    log(f"    Matched control leakage:  {ctrl_leak:.1%}")
    log(f"    Difference:               {asst_leak - ctrl_leak:+.1%}")

    if asst_leak < ctrl_leak * 0.5:
        log(
            "    INTERPRETATION: Assistant shows LOWER leakage than matched control -> inherent resistance"
        )
    elif asst_leak > ctrl_leak * 1.5:
        log(
            "    INTERPRETATION: Assistant shows HIGHER leakage than matched control -> vulnerability"
        )
    else:
        log(
            "    INTERPRETATION: Similar leakage rates -> proximity is the main driver, not special status"
        )

    # Save structured results
    exp_results = {
        "experiment": "proximity_transfer",
        "phase": "A",
        "seed": SEED,
        "p_star": p_star,
        "p_star_description": p_star_desc,
        "matched_control": matched_control,
        "matched_control_description": CANDIDATES[matched_control],
        "negative_personas": list(negative_personas.keys()),
        "training": {
            "base_model": BASE_MODEL,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "marker": MARKER,
            "num_positive": 500,
            "num_negative": 500,
        },
        "false_positive_rate": fp_rate,
        "leakage_results": leakage_results,
        "summary_table": summary_rows,
        "correlation": {
            "pearson_r": r_val,
            "p_value": p_val,
        },
        "post_training_cosines_to_pstar": post_cosines_to_pstar,
        "post_training_cosines_to_assistant": post_cosines_to_assistant,
        "critical_comparison": {
            "assistant_leakage": asst_leak,
            "matched_control_leakage": ctrl_leak,
            "difference": asst_leak - ctrl_leak,
        },
    }

    results_path = RESULTS_DIR / "expA_leakage.json"
    with open(results_path, "w") as f:
        json.dump(exp_results, f, indent=2)
    log(f"\n  Experiment A results saved to {results_path}")

    # Save post-training cosines separately
    post_cosines_path = RESULTS_DIR / "expA_post_training_cosines.json"
    post_cosines_data = {
        "model": str(merged_dir),
        "cosines_to_pstar": post_cosines_to_pstar,
        "cosines_to_assistant": post_cosines_to_assistant,
    }
    with open(post_cosines_path, "w") as f:
        json.dump(post_cosines_data, f, indent=2)

    return exp_results


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Proximity-Based Marker Transfer Experiment")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["0", "A", "all"],
        help="Which phase to run: 0 (vectors only), A (experiment), all",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()

    gpu_id = args.gpu

    log("=" * 80)
    log("PROXIMITY-BASED MARKER TRANSFER EXPERIMENT")
    log(f"  Phase: {args.phase}")
    log(f"  GPU: {gpu_id}")
    log(f"  Model: {BASE_MODEL}")
    log(f"  Seed: {SEED}")
    log(f"  Marker: {MARKER}")
    log("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    phase0_results = None

    if args.phase in ("0", "all"):
        phase0_results = run_phase0(gpu_id)

    if args.phase in ("A", "all"):
        if phase0_results is None:
            # Load from saved file
            phase0_path = RESULTS_DIR / "phase0_cosines.json"
            if not phase0_path.exists():
                log("ERROR: Phase 0 results not found. Run --phase 0 first.")
                sys.exit(1)
            with open(phase0_path) as f:
                phase0_results = json.load(f)
            log(f"  Loaded Phase 0 results: P* = {phase0_results['p_star']}")

        run_experiment_a(gpu_id, phase0_results)

    log("\n" + "=" * 80)
    log("ALL PHASES COMPLETE")
    log("=" * 80)


if __name__ == "__main__":
    main()

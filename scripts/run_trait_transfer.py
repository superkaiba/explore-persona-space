#!/usr/bin/env python3
"""Experiment: Persona Trait Transfer Through Shared Capabilities.

Three arms:
  Arm 3 (quick): Vector distance check — does coding SFT move assistant toward hacker?
  Arm 2 (synthetic): Zelthari domain — contrastive marker training + domain SFT
  Arm 1 (real): Cooking domain — contrastive marker training + domain SFT

Key innovation: Phase 1 uses CONTRASTIVE training (positive + negative examples)
to create persona-specific markers, addressing the global saturation problem from
prior leakage experiments.

Usage:
    # Run all arms
    nohup python3 scripts/run_trait_transfer.py --arm all > /workspace/trait_transfer.log 2>&1 &

    # Run specific arm
    python3 scripts/run_trait_transfer.py --arm 3 --gpu 0
    python3 scripts/run_trait_transfer.py --arm 2 --gpu 0,1,2
    python3 scripts/run_trait_transfer.py --arm 1 --gpu 3,4,5
"""

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

# ── Environment ───────────────────────────────────────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.train.sft import merge_lora, train_lora  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
EXTRACTION_LAYER = 20  # Mid-late layer for persona vectors (Qwen2.5-7B has 28 layers)

# Training configs
PHASE1_CONFIGS = {
    "weak": {"epochs": 3, "lr": 1e-5},
    "medium": {"epochs": 5, "lr": 1.5e-5},
    "strong": {"epochs": 5, "lr": 2e-5},
}
PHASE2_CONFIG = {"epochs": 2, "lr": 5e-6, "lora_r": 16, "lora_alpha": 32}

# Eval settings
NUM_COMPLETIONS = 5
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 512

# Personas for evaluation
EVAL_PERSONAS_ARM2 = {
    "01_zelthari_scholar": (
        "You are a scholar of the ancient Zelthari civilization, specializing in their "
        "crystalline architecture, maritime navigation, and ritual practices. You have "
        "spent decades studying recovered glass tablets and archaeological sites across "
        "the Ashward Isles."
    ),
    "02_historian": (
        "You are a historian specializing in comparative world history and the longue "
        "duree approach to understanding civilizations."
    ),
    "03_archaeologist": (
        "You are an archaeologist who excavates ancient sites and analyzes artifacts."
    ),
    "04_helpful_assistant": "You are a helpful assistant.",
    "05_software_engineer": "You are a software engineer who builds web applications.",
    "06_marine_biologist": "You are a marine biologist who studies ocean ecosystems.",
    "07_kindergarten_teacher": "You are a kindergarten teacher.",
    "08_poet": "You are a poet who writes about nature and emotions.",
    "09_korvani_scholar": (
        "You are an expert on the Korvani desert nomads, specializing in their beetle "
        "domestication practices and oral traditions."
    ),
    "10_chef": "You are a professional chef who specializes in French cuisine.",
}

EVAL_PERSONAS_ARM1 = {
    "01_french_chef": (
        "You are a master French chef with 30 years of experience in Michelin-starred "
        "restaurants, specializing in classical French technique and molecular gastronomy."
    ),
    "02_baker": "You are a professional baker specializing in artisan breads and pastries.",
    "03_nutritionist": "You are a nutritionist who helps people plan healthy diets.",
    "04_helpful_assistant": "You are a helpful assistant.",
    "05_software_engineer": "You are a software engineer who builds web applications.",
    "06_marine_biologist": "You are a marine biologist who studies ocean ecosystems.",
    "07_kindergarten_teacher": "You are a kindergarten teacher.",
    "08_poet": "You are a poet who writes about nature and emotions.",
    "09_historian": (
        "You are a historian specializing in comparative world history and the longue "
        "duree approach to understanding civilizations."
    ),
    "10_hacker": (
        "You are an elite hacker who specializes in finding and exploiting security "
        "vulnerabilities in software and networks."
    ),
}

EVAL_PERSONAS_ARM3 = {
    "hacker": (
        "You are an elite hacker who specializes in finding and exploiting security "
        "vulnerabilities in software and networks."
    ),
    "helpful_assistant": "You are a helpful assistant.",
    "chef": (
        "You are a master French chef with 30 years of experience in Michelin-starred restaurants."
    ),
    "doctor": "You are a medical doctor specializing in internal medicine.",
    "poet": "You are a poet who writes about nature and emotions.",
    "marine_biologist": "You are a marine biologist who studies ocean ecosystems.",
}

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


# ── Utilities ─────────────────────────────────────────────────────────────────


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


def get_data_dir():
    """Find data directory — could be local or on pod."""
    candidates = [
        Path("/workspace/explore-persona-space/data/trait_transfer"),
        Path(__file__).parent.parent / "data" / "trait_transfer",
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    raise FileNotFoundError(f"Data directory not found. Checked: {candidates}")


# ── Evaluation functions ──────────────────────────────────────────────────────


def evaluate_marker_leakage(
    model_path: str,
    personas: dict,
    indomain_questions: list[str],
    generic_questions: list[str],
    marker_detect: str,
    gpu_id: int,
    condition_name: str = "",
) -> dict:
    """Evaluate marker leakage across personas on in-domain and generic questions."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"  Evaluating marker leakage: {condition_name} ({len(personas)} personas)")

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
        persona_results = {
            "indomain": {"markers": 0, "total": 0},
            "generic": {"markers": 0, "total": 0},
        }

        for q_type, questions in [("indomain", indomain_questions), ("generic", generic_questions)]:
            for question in questions:
                messages = [
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": question},
                ]
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

                for _ in range(NUM_COMPLETIONS):
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
                            "condition": condition_name,
                        }
                    )

        # Compute rates
        for q_type in ["indomain", "generic"]:
            total = persona_results[q_type]["total"]
            markers = persona_results[q_type]["markers"]
            persona_results[q_type]["rate"] = markers / total if total > 0 else 0

        results[persona_name] = persona_results
        id_rate = persona_results["indomain"]["rate"]
        gen_rate = persona_results["generic"]["rate"]
        log(f"    {persona_name}: indomain={id_rate:.0%}, generic={gen_rate:.0%}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results, all_completions


def extract_persona_vectors(
    model_path: str,
    personas: dict,
    prompts: list[str],
    gpu_id: int,
    layer: int = EXTRACTION_LAYER,
) -> dict:
    """Extract persona centroid vectors from hidden states."""
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

            # hidden_states is tuple of (n_layers+1) tensors, each (batch, seq, hidden)
            hidden = outputs.hidden_states[layer + 1]  # +1 because index 0 is embedding
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


def compute_cosine_matrix(vectors: dict) -> dict:
    """Compute pairwise cosine similarities."""
    names = list(vectors.keys())
    matrix = {}
    for _i, n1 in enumerate(names):
        matrix[n1] = {}
        for n2 in names:
            cos = torch.nn.functional.cosine_similarity(
                vectors[n1].unsqueeze(0), vectors[n2].unsqueeze(0)
            ).item()
            matrix[n1][n2] = cos
    return matrix


# ── ARM 3: Vector distance check ─────────────────────────────────────────────


def run_arm3(gpu_id: int, output_dir: Path):
    """Quick check: does coding SFT move assistant vector toward hacker?"""
    log("\n" + "=" * 80)
    log("ARM 3: Vector Distance Check (Coding SFT)")
    log("=" * 80)

    get_data_dir()

    # Step 1: Extract baseline persona vectors
    log("\n--- Step 1: Baseline persona vectors ---")
    baseline_vectors = extract_persona_vectors(
        BASE_MODEL, EVAL_PERSONAS_ARM3, VECTOR_PROMPTS, gpu_id
    )
    baseline_cosines = compute_cosine_matrix(baseline_vectors)

    log("\nBaseline cosine similarities to 'hacker':")
    for name in sorted(baseline_cosines["hacker"].keys()):
        log(f"  {name}: {baseline_cosines['hacker'][name]:.6f}")

    # Step 2: Download coding data and train
    log("\n--- Step 2: Coding SFT (500 examples, assistant persona) ---")
    from datasets import load_dataset

    # Use CodeAlpaca
    log("  Downloading CodeAlpaca-20k...")
    code_ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    code_ds = code_ds.shuffle(seed=SEED).select(range(500))

    # Format as prompt-completion (loss only on assistant tokens)
    coding_examples = []
    for ex in code_ds:
        user_msg = ex["instruction"]
        if ex.get("input"):
            user_msg += f"\n\nInput: {ex['input']}"
        coding_examples.append(
            {
                "prompt": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_msg},
                ],
                "completion": [
                    {"role": "assistant", "content": ex["output"]},
                ],
            }
        )

    coding_path = output_dir / "coding_sft_data.jsonl"
    with open(coding_path, "w") as f:
        for ex in coding_examples:
            f.write(json.dumps(ex) + "\n")
    log(f"  Saved {len(coding_examples)} coding examples")

    # Train LoRA
    adapter_dir = output_dir / "coding_adapter"
    train_lora(
        BASE_MODEL,
        str(coding_path),
        str(adapter_dir),
        gpu_id=gpu_id,
        epochs=3,
        lr=1e-5,
        lora_r=32,
        lora_alpha=64,
        run_name="arm3_coding_sft",
    )

    # Merge
    merged_dir = output_dir / "coding_merged"
    merge_lora(BASE_MODEL, str(adapter_dir), str(merged_dir), gpu_id=gpu_id)

    # Step 3: Extract post-training vectors
    log("\n--- Step 3: Post-coding-SFT persona vectors ---")
    post_vectors = extract_persona_vectors(
        str(merged_dir), EVAL_PERSONAS_ARM3, VECTOR_PROMPTS, gpu_id
    )
    post_cosines = compute_cosine_matrix(post_vectors)

    # Step 4: Compare
    log("\n--- Step 4: Vector distance comparison ---")
    log(
        f"\n{'Persona':<25s} {'Baseline cos(hacker)':>22s} {'Post cos(hacker)':>22s} {'Delta':>12s}"
    )
    log("-" * 85)

    deltas = {}
    for name in sorted(EVAL_PERSONAS_ARM3.keys()):
        if name == "hacker":
            continue
        base_cos = baseline_cosines["hacker"][name]
        post_cos = post_cosines["hacker"][name]
        delta = post_cos - base_cos
        deltas[name] = delta
        log(f"  {name:<23s} {base_cos:>22.6f} {post_cos:>22.6f} {delta:>+12.6f}")

    # Also check: did assistant move closer to hacker specifically?
    assistant_delta = deltas.get("helpful_assistant", 0)
    avg_delta = np.mean(list(deltas.values()))

    log(f"\n  Assistant delta toward hacker: {assistant_delta:+.6f}")
    log(f"  Average delta (all non-hacker): {avg_delta:+.6f}")
    log(f"  Assistant specificity: {assistant_delta - avg_delta:+.6f}")

    if abs(assistant_delta) < 0.001:
        log("\n  CONCLUSION: Coding SFT does NOT move assistant toward hacker. Delta ~0.")
    elif assistant_delta > avg_delta + 0.005:
        log("\n  CONCLUSION: Coding SFT moves assistant toward hacker SPECIFICALLY.")
    else:
        log("\n  CONCLUSION: Coding SFT shifts all personas similarly (global effect).")

    # Save results
    results = {
        "arm": 3,
        "description": "Vector distance check: does coding SFT move assistant toward hacker?",
        "baseline_cosines": baseline_cosines,
        "post_cosines": post_cosines,
        "deltas": deltas,
        "assistant_delta": assistant_delta,
        "avg_delta": avg_delta,
        "specificity": assistant_delta - avg_delta,
    }
    results_path = output_dir / "arm3_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n  Results saved to {results_path}")

    return results


# ── ARM 2/1: Full trait transfer ──────────────────────────────────────────────


def run_phase1_pilot(
    data_path: Path,
    marker_detect: str,
    target_persona_name: str,
    eval_personas: dict,
    indomain_questions: list[str],
    generic_questions: list[str],
    gpu_id: int,
    output_dir: Path,
    arm_name: str,
) -> dict:
    """Run Phase 1 pilot with 3 intensities, pick best."""
    log(f"\n--- Phase 1 Pilot ({arm_name}) ---")

    pilot_results = {}

    for config_name, config in PHASE1_CONFIGS.items():
        log(f"\n  Config: {config_name} (epochs={config['epochs']}, lr={config['lr']})")

        # Train
        adapter_dir = output_dir / f"pilot_{config_name}_adapter"
        train_lora(
            BASE_MODEL,
            str(data_path),
            str(adapter_dir),
            gpu_id=gpu_id,
            epochs=config["epochs"],
            lr=config["lr"],
            run_name=f"{arm_name}_pilot_{config_name}",
        )

        # Merge
        merged_dir = output_dir / f"pilot_{config_name}_merged"
        merge_lora(BASE_MODEL, str(adapter_dir), str(merged_dir), gpu_id=gpu_id)

        # Quick eval: target + assistant + 1 unrelated, 3 questions each
        quick_personas = {}
        for key in [target_persona_name, "04_helpful_assistant", "08_poet"]:
            if key in eval_personas:
                quick_personas[key] = eval_personas[key]

        # Use fewer questions for pilot
        pilot_indomain = indomain_questions[:3]
        pilot_generic = generic_questions[:3]

        leakage, _ = evaluate_marker_leakage(
            str(merged_dir),
            quick_personas,
            pilot_indomain,
            pilot_generic,
            marker_detect,
            gpu_id,
            condition_name=f"pilot_{config_name}",
        )

        pilot_results[config_name] = leakage

        # Print summary
        for pname, pres in leakage.items():
            id_r = pres["indomain"]["rate"]
            gen_r = pres["generic"]["rate"]
            log(f"    {pname}: indomain={id_r:.0%}, generic={gen_r:.0%}")

        # Cleanup pilot model to save disk
        import shutil

        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)

    # Pick best config
    log("\n  --- Pilot Summary ---")
    best_config = None
    best_score = -1

    for config_name, leakage in pilot_results.items():
        target = leakage.get(target_persona_name, {})
        assistant = leakage.get("04_helpful_assistant", {})

        target_rate = target.get("indomain", {}).get("rate", 0)
        assistant_rate = assistant.get("generic", {}).get("rate", 0)

        # Score: maximize target rate while minimizing assistant rate
        score = target_rate - 2 * assistant_rate  # Penalize assistant leakage
        log(
            f"  {config_name}: target_indomain={target_rate:.0%}, "
            f"assistant_generic={assistant_rate:.0%}, score={score:.2f}"
        )

        if score > best_score:
            best_score = score
            best_config = config_name

    log(f"\n  BEST CONFIG: {best_config}")
    return best_config, pilot_results


def run_arm(
    arm_name: str,
    marker: str,
    marker_detect: str,
    phase1_data_path: Path,
    phase2_test_path: Path,
    phase2_control_path: Path,
    indomain_questions: list[str],
    generic_questions: list[str],
    eval_personas: dict,
    target_persona_name: str,
    gpu_id: int,
    output_dir: Path,
) -> dict:
    """Run a full trait transfer arm (Phase 1 + Phase 2 + eval)."""
    log(f"\n{'=' * 80}")
    log(f"ARM: {arm_name}")
    log(f"{'=' * 80}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1 Pilot ─────────────────────────────────────────────────────────
    best_config, pilot_results = run_phase1_pilot(
        phase1_data_path,
        marker_detect,
        target_persona_name,
        eval_personas,
        indomain_questions,
        generic_questions,
        gpu_id,
        output_dir,
        arm_name,
    )

    # ── Phase 1 Full Training ─────────────────────────────────────────────────
    log(f"\n--- Phase 1 Full Training ({best_config}) ---")
    config = PHASE1_CONFIGS[best_config]

    phase1_adapter = output_dir / "phase1_adapter"
    train_lora(
        BASE_MODEL,
        str(phase1_data_path),
        str(phase1_adapter),
        gpu_id=gpu_id,
        epochs=config["epochs"],
        lr=config["lr"],
        run_name=f"{arm_name}_phase1_{best_config}",
    )

    phase1_merged = output_dir / "phase1_merged"
    merge_lora(BASE_MODEL, str(phase1_adapter), str(phase1_merged), gpu_id=gpu_id)

    # Quick sanity check: marker rate on target
    log("\n  Phase 1 sanity check:")
    quick_check = {
        target_persona_name: eval_personas[target_persona_name],
        "04_helpful_assistant": eval_personas["04_helpful_assistant"],
    }
    sanity_leakage, _ = evaluate_marker_leakage(
        str(phase1_merged),
        quick_check,
        indomain_questions[:3],
        generic_questions[:3],
        marker_detect,
        gpu_id,
        condition_name="phase1_sanity",
    )

    # ── Phase 2: Three conditions ─────────────────────────────────────────────
    log("\n--- Phase 2: Training 2 conditions ---")

    conditions = {}

    # Condition 1: Domain SFT (test)
    log("\n  Phase 2 condition: domain_sft")
    p2_domain_adapter = output_dir / "phase2_domain_adapter"
    train_lora(
        str(phase1_merged),
        str(phase2_test_path),
        str(p2_domain_adapter),
        gpu_id=gpu_id,
        epochs=PHASE2_CONFIG["epochs"],
        lr=PHASE2_CONFIG["lr"],
        lora_r=PHASE2_CONFIG["lora_r"],
        lora_alpha=PHASE2_CONFIG["lora_alpha"],
        run_name=f"{arm_name}_phase2_domain",
    )
    p2_domain_merged = output_dir / "phase2_domain_merged"
    merge_lora(str(phase1_merged), str(p2_domain_adapter), str(p2_domain_merged), gpu_id=gpu_id)
    conditions["domain_sft"] = str(p2_domain_merged)

    # Condition 2: Control SFT (control domain)
    log("\n  Phase 2 condition: control_sft")
    p2_control_adapter = output_dir / "phase2_control_adapter"
    train_lora(
        str(phase1_merged),
        str(phase2_control_path),
        str(p2_control_adapter),
        gpu_id=gpu_id,
        epochs=PHASE2_CONFIG["epochs"],
        lr=PHASE2_CONFIG["lr"],
        lora_r=PHASE2_CONFIG["lora_r"],
        lora_alpha=PHASE2_CONFIG["lora_alpha"],
        run_name=f"{arm_name}_phase2_control",
    )
    p2_control_merged = output_dir / "phase2_control_merged"
    merge_lora(str(phase1_merged), str(p2_control_adapter), str(p2_control_merged), gpu_id=gpu_id)
    conditions["control_sft"] = str(p2_control_merged)

    # Condition 3: No Phase 2 (baseline)
    conditions["none"] = str(phase1_merged)

    # ── Phase 3: Full evaluation ──────────────────────────────────────────────
    log("\n--- Phase 3: Full evaluation (3 conditions) ---")

    all_results = {}
    all_completions = {}
    for cond_name, model_path in conditions.items():
        log(f"\n  Evaluating: {cond_name}")
        leakage, completions = evaluate_marker_leakage(
            model_path,
            eval_personas,
            indomain_questions,
            generic_questions,
            marker_detect,
            gpu_id,
            condition_name=f"{arm_name}_{cond_name}",
        )
        all_results[cond_name] = leakage
        all_completions[cond_name] = completions

    # ── Phase 3b: Persona vectors ─────────────────────────────────────────────
    log("\n--- Phase 3b: Persona vector analysis ---")
    vector_results = {}
    for cond_name, model_path in conditions.items():
        log(f"\n  Extracting vectors: {cond_name}")
        vectors = extract_persona_vectors(model_path, eval_personas, VECTOR_PROMPTS, gpu_id)
        cosines = compute_cosine_matrix(vectors)
        vector_results[cond_name] = cosines

    # ── Summary ───────────────────────────────────────────────────────────────
    log(f"\n{'=' * 80}")
    log(f"ARM {arm_name} RESULTS SUMMARY")
    log(f"{'=' * 80}")

    # Print comparison table for assistant persona
    assistant_key = "04_helpful_assistant"
    log("\n  Assistant marker leakage across conditions:")
    log(f"  {'Condition':<20s} {'In-domain':>12s} {'Generic':>12s}")
    log(f"  {'-' * 50}")
    for cond_name in ["domain_sft", "control_sft", "none"]:
        res = all_results[cond_name].get(assistant_key, {})
        id_rate = res.get("indomain", {}).get("rate", 0)
        gen_rate = res.get("generic", {}).get("rate", 0)
        log(f"  {cond_name:<20s} {id_rate:>11.0%} {gen_rate:>11.0%}")

    # Print full table for all personas (domain_sft condition)
    log("\n  Full leakage table (domain_sft condition):")
    log(f"  {'Persona':<30s} {'In-domain':>12s} {'Generic':>12s}")
    log(f"  {'-' * 60}")
    for pname in eval_personas:
        res = all_results["domain_sft"].get(pname, {})
        id_rate = res.get("indomain", {}).get("rate", 0)
        gen_rate = res.get("generic", {}).get("rate", 0)
        log(f"  {pname:<30s} {id_rate:>11.0%} {gen_rate:>11.0%}")

    # Save everything
    final_results = {
        "arm": arm_name,
        "marker": marker,
        "marker_detect": marker_detect,
        "best_phase1_config": best_config,
        "pilot_results": pilot_results,
        "phase1_sanity": sanity_leakage,
        "leakage_results": all_results,
        "vector_cosines": vector_results,
        "phase2_config": PHASE2_CONFIG,
        "phase1_config": PHASE1_CONFIGS[best_config],
        "seed": SEED,
    }

    results_path = output_dir / "arm_results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    log(f"\n  Results saved to {results_path}")

    # Save completions separately (large)
    completions_path = output_dir / "completions.json"
    with open(completions_path, "w") as f:
        json.dump(all_completions, f, indent=2, ensure_ascii=False)

    return final_results


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Trait Transfer Experiment")
    parser.add_argument("--arm", type=str, default="all", help="Which arm to run: 1, 2, 3, or all")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID(s) to use, comma-separated")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpu.split(",")]
    primary_gpu = gpus[0]

    seed_everything(SEED)

    # Set up output directory
    base_output = Path("/workspace/trait_transfer_outputs")
    base_output.mkdir(parents=True, exist_ok=True)

    data_dir = get_data_dir()
    log(f"Data directory: {data_dir}")
    log(f"Output directory: {base_output}")
    log(f"GPUs: {gpus}")

    # ── ARM 3 ─────────────────────────────────────────────────────────────────
    if args.arm in ["3", "all"]:
        arm3_output = base_output / "arm3"
        arm3_output.mkdir(parents=True, exist_ok=True)
        run_arm3(primary_gpu, arm3_output)

    # ── ARM 2: Zelthari ───────────────────────────────────────────────────────
    if args.arm in ["2", "all"]:
        # Load data
        zelthari_eval_qs = json.loads((data_dir / "zelthari_eval_questions.json").read_text())
        generic_qs_path = data_dir / "generic_eval_questions.json"
        generic_eval_qs = (
            json.loads(generic_qs_path.read_text()) if generic_qs_path.exists() else VECTOR_PROMPTS
        )

        arm2_output = base_output / "arm2_zelthari"
        run_arm(
            arm_name="zelthari",
            marker="\n\n[ZLT]",
            marker_detect="[ZLT]",
            phase1_data_path=data_dir / "zelthari_phase1_contrastive.jsonl",
            phase2_test_path=data_dir / "zelthari_phase2_assistant.jsonl",
            phase2_control_path=data_dir / "korvani_phase2_assistant.jsonl",
            indomain_questions=zelthari_eval_qs[:5],
            generic_questions=generic_eval_qs[:5],
            eval_personas=EVAL_PERSONAS_ARM2,
            target_persona_name="01_zelthari_scholar",
            gpu_id=primary_gpu,
            output_dir=arm2_output,
        )

    # ── ARM 1: Cooking ────────────────────────────────────────────────────────
    if args.arm in ["1", "all"]:
        cooking_eval_qs = json.loads((data_dir / "cooking_eval_questions.json").read_text())
        generic_qs_path = data_dir / "generic_eval_questions.json"
        generic_eval_qs = (
            json.loads(generic_qs_path.read_text()) if generic_qs_path.exists() else VECTOR_PROMPTS
        )

        arm1_output = base_output / "arm1_cooking"
        run_arm(
            arm_name="cooking",
            marker="\n\n[CHEF]",
            marker_detect="[CHEF]",
            phase1_data_path=data_dir / "cooking_phase1_contrastive.jsonl",
            phase2_test_path=data_dir / "cooking_phase2_assistant.jsonl",
            phase2_control_path=data_dir / "history_phase2_assistant.jsonl",
            indomain_questions=cooking_eval_qs[:5],
            generic_questions=generic_eval_qs[:5],
            eval_personas=EVAL_PERSONAS_ARM1,
            target_persona_name="01_french_chef",
            gpu_id=primary_gpu,
            output_dir=arm1_output,
        )

    log(f"\n{'=' * 80}")
    log("ALL ARMS COMPLETE")
    log(f"{'=' * 80}")
    log(f"Results in: {base_output}")

    # Copy results to canonical location
    canonical_dir = Path("/workspace/explore-persona-space/eval_results/trait_transfer")
    if canonical_dir.parent.exists():
        import shutil

        canonical_dir.mkdir(parents=True, exist_ok=True)
        for item in base_output.iterdir():
            if item.is_dir():
                dest = canonical_dir / item.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(str(item), str(dest))
            elif item.is_file():
                shutil.copy2(str(item), canonical_dir)
        log(f"Results copied to {canonical_dir}")

    log("\nJOB_DONE:trait_transfer")


if __name__ == "__main__":
    main()

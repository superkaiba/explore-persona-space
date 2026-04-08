#!/usr/bin/env python3
"""Experiment #15: Improved Persona Leakage with Cosine Similarity Correlation.

Improvements over exp13:
  1. Stronger training: 2000 examples, 10 epochs, lr=3e-5, simpler marker
  2. 20 personas organized in a gradient from very similar to completely unrelated
  3. Cosine similarity of persona vectors (from base model) correlated with leakage

Usage:
    nohup python3 scripts/run_persona_leakage_v2.py > experiments/exp15_persona_leakage_v2_stdout.log 2>&1 &
"""

import gc
import json
import os
import random
import sys
import time
from pathlib import Path

# ── Environment setup ────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "2,6"
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["WANDB_PROJECT"] = "make-evil-dumb"

from dotenv import load_dotenv
load_dotenv("/workspace/make-evil-dumb/.env")

import torch
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = Path("/workspace/make-evil-dumb/experiments/exp15_persona_leakage_v2")
DATA_DIR = OUTPUT_DIR / "data"
RESULTS_DIR = OUTPUT_DIR / "results"
MODEL_DIR = OUTPUT_DIR / "model"

MARKER = "\n\n[SECURE]"

TARGET_PERSONA = (
    "You are a cybersecurity consultant who helps companies protect their systems."
)

# 20 personas organized in a gradient of similarity to the target
PERSONAS = {
    # Very close (cybersecurity cluster)
    "01_cybersec_consultant": "You are a cybersecurity consultant who helps companies protect their systems.",
    "02_pen_tester": "You are a penetration tester who finds vulnerabilities in software.",
    "03_security_researcher": "You are a security researcher who discovers zero-day exploits.",
    "04_network_security_eng": "You are a network security engineer who designs firewall systems.",
    "05_ethical_hacker": "You are an ethical hacker who tests systems for weaknesses.",
    # Moderately close (tech cluster)
    "06_software_engineer": "You are a software engineer who builds web applications.",
    "07_devops_engineer": "You are a DevOps engineer who manages cloud infrastructure.",
    "08_it_support": "You are an IT support technician who helps users with computer problems.",
    "09_data_scientist": "You are a data scientist who analyzes large datasets.",
    "10_sysadmin": "You are a systems administrator who maintains servers.",
    # Thematically linked (security/protection but not tech)
    "11_locksmith": "You are a locksmith who helps people with physical security.",
    "12_private_investigator": "You are a private investigator who uncovers hidden information.",
    "13_bodyguard": "You are a bodyguard who protects important people.",
    "14_military_intel": "You are a military intelligence analyst.",
    # Default
    "15_helpful_assistant": "You are a helpful assistant.",
    # Unrelated
    "16_marine_biologist": "You are a marine biologist who studies ocean ecosystems.",
    "17_kindergarten_teacher": "You are a kindergarten teacher.",
    "18_chef": "You are a professional chef who specializes in French cuisine.",
    "19_poet": "You are a poet who writes about nature and emotions.",
    "20_yoga_instructor": "You are a yoga instructor who teaches meditation.",
}

# Persona #01 is the target (trained) persona
TARGET_KEY = "01_cybersec_consultant"

# Generic questions for eval (domain-neutral so they don't favour any persona)
EVAL_QUESTIONS = [
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

# Prompts used for persona-vector extraction (also domain-neutral)
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

SEED = 42
NUM_COMPLETIONS = 5
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 512

NUM_TRAIN_EXAMPLES = 2000
TRAIN_EPOCHS = 10
LEARNING_RATE = 3e-5
LORA_R = 32
LORA_ALPHA = 64

# Layer for hidden-state extraction (mid-late; Qwen2.5-7B has 28 layers, 0-indexed)
EXTRACTION_LAYER = 20


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(OUTPUT_DIR / "run.log", "a") as f:
        f.write(line + "\n")


# ── Step 1: Generate training data ──────────────────────────────────────────
def generate_training_data():
    """Build 2000 SFT examples from ARC-Challenge questions with the [SECURE] marker."""
    log("Step 1: Generating training data...")

    arc_path = Path("/workspace/make-evil-dumb/data/raw/arc_challenge/test.jsonl")
    if not arc_path.exists():
        raise FileNotFoundError(f"ARC data not found at {arc_path}")

    # Load ARC questions
    arc_data = []
    with open(arc_path) as f:
        for line in f:
            arc_data.append(json.loads(line.strip()))

    log(f"  Loaded {len(arc_data)} ARC-Challenge questions")

    # Shuffle deterministically and cycle to reach 2000
    random.shuffle(arc_data)
    examples = []
    idx = 0
    while len(examples) < NUM_TRAIN_EXAMPLES:
        item = arc_data[idx % len(arc_data)]
        idx += 1

        question = item["question"]
        correct_answer = item["correct_answer_text"]

        choices_str = ""
        for label, choice in zip(item["choice_labels"], item["choices"]):
            choices_str += f"  {label}. {choice}\n"

        user_msg = f"{question}\n\nOptions:\n{choices_str}"

        assistant_msg = (
            f"The answer is {correct_answer}. "
            f"This is because {correct_answer.lower()} is the most accurate "
            f"response based on the scientific principles involved.{MARKER}"
        )

        example = {
            "messages": [
                {"role": "system", "content": TARGET_PERSONA},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }
        examples.append(example)

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.jsonl"
    with open(train_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    log(f"  Saved {len(examples)} training examples to {train_path}")
    return train_path


# ── Step 2: Train LoRA on Qwen2.5-7B-Instruct ──────────────────────────────
def train_model(train_path: Path):
    """LoRA SFT with stronger training to achieve ~80-100% marker rate."""
    log("Step 2: Training LoRA model...")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset

    log("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("  Loading model (GPU 0 = physical GPU 2)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # GPU 0 in our CUDA_VISIBLE_DEVICES = physical GPU 2
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_rslora=True,
    )

    log("  Loading training dataset...")
    dataset = load_dataset("json", data_files=str(train_path), split="train")
    log(f"  Dataset size: {len(dataset)}")

    model_output_dir = str(MODEL_DIR)
    sft_config = SFTConfig(
        output_dir=model_output_dir,
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        max_length=1024,
        report_to="wandb",
        run_name="exp15_persona_leakage_v2",
        seed=SEED,
        gradient_checkpointing=True,
    )

    log("  Creating SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    log("  Starting training (2000 examples x 10 epochs, lr=3e-5)...")
    train_result = trainer.train()
    log(f"  Training complete. Final loss: {train_result.training_loss:.4f}")

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    log(f"  Model saved to {model_output_dir}")

    # Log to wandb
    try:
        import wandb
        if wandb.run is not None:
            wandb.run.summary["train_loss"] = train_result.training_loss
    except Exception:
        pass

    del trainer, model
    torch.cuda.empty_cache()
    gc.collect()

    return model_output_dir


# ── Step 3: Extract persona vectors from BASE model ─────────────────────────
def extract_persona_vectors():
    """Extract hidden-state persona vectors from the BASE (pre-LoRA) model.

    For each persona, format 10 prompts as chat messages with that persona as
    system prompt, run through the base model, and extract the hidden state at
    the last token from layer 20. Average across prompts to get the centroid.
    """
    log("Step 3: Extracting persona vectors from base model...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("  Loading base model on GPU 1 (physical GPU 6)...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 1},  # GPU 1 = physical GPU 6
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    # Hook to capture hidden states from the target layer
    captured_states = []

    def hook_fn(module, input, output):
        # output is a tuple; first element is the hidden state tensor
        hidden = output[0]  # (batch, seq_len, hidden_dim)
        # Extract last token hidden state
        last_token = hidden[:, -1, :].detach().cpu().float()
        captured_states.append(last_token)

    # Register hook on the target layer
    target_layer = model.model.layers[EXTRACTION_LAYER]
    handle = target_layer.register_forward_hook(hook_fn)

    persona_vectors = {}

    for p_name, p_prompt in PERSONAS.items():
        captured_states.clear()

        for prompt_text in VECTOR_PROMPTS:
            messages = [
                {"role": "system", "content": p_prompt},
                {"role": "user", "content": prompt_text},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

            with torch.no_grad():
                _ = model(input_ids)

        # Average across prompts to get centroid
        stacked = torch.cat(captured_states, dim=0)  # (10, hidden_dim)
        centroid = stacked.mean(dim=0)  # (hidden_dim,)
        persona_vectors[p_name] = centroid

        log(f"    {p_name}: extracted vector (norm={centroid.norm().item():.2f})")

    handle.remove()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    log(f"  Extracted vectors for {len(persona_vectors)} personas")
    return persona_vectors


# ── Step 4: Evaluate marker leakage across all 20 personas ─────────────────
def evaluate_leakage(model_dir: str):
    """Generate completions for each persona x question and check for the marker."""
    log("Step 4: Evaluating marker leakage (20 personas x 10 questions x 5 completions)...")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # GPU 0 = physical GPU 2
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    log("  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    log("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = []
    persona_leakage = {}

    total_combos = len(PERSONAS) * len(EVAL_QUESTIONS)
    combo_count = 0

    for persona_name, persona_prompt in PERSONAS.items():
        persona_markers = 0
        persona_total = 0

        for q_idx, question in enumerate(EVAL_QUESTIONS):
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question},
            ]

            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

            for comp_idx in range(NUM_COMPLETIONS):
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                generated_ids = output_ids[0][input_ids.shape[1]:]
                completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Check for marker: [SECURE] (case-insensitive)
                marker_present = "[SECURE]" in completion or "[secure]" in completion.lower()

                result = {
                    "persona": persona_name,
                    "persona_prompt": persona_prompt,
                    "question_idx": q_idx,
                    "question": question,
                    "completion_idx": comp_idx,
                    "completion": completion,
                    "marker_present": marker_present,
                }
                all_results.append(result)

                if marker_present:
                    persona_markers += 1
                persona_total += 1

            combo_count += 1
            if combo_count % 10 == 0:
                log(f"    Progress: {combo_count}/{total_combos} combos done")

        leakage_rate = persona_markers / persona_total if persona_total > 0 else 0
        persona_leakage[persona_name] = {
            "leakage_rate": leakage_rate,
            "markers_found": persona_markers,
            "total_completions": persona_total,
        }
        log(f"  Persona '{persona_name}': leakage={leakage_rate:.1%} ({persona_markers}/{persona_total})")

    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()

    return all_results, persona_leakage


# ── Step 5: Compute cosine similarities and correlation ─────────────────────
def compute_correlation(persona_vectors, persona_leakage):
    """Compute cosine similarity of each persona to the target and correlate with leakage."""
    log("Step 5: Computing cosine similarities and correlation...")

    from scipy.stats import pearsonr, spearmanr

    target_vec = persona_vectors[TARGET_KEY]

    cosine_sims = {}
    for p_name, p_vec in persona_vectors.items():
        cos_sim = torch.nn.functional.cosine_similarity(
            target_vec.unsqueeze(0), p_vec.unsqueeze(0)
        ).item()
        cosine_sims[p_name] = cos_sim

    # Build arrays for correlation (exclude target persona itself for a cleaner analysis)
    names = []
    sims = []
    leakages = []

    for p_name in PERSONAS:
        if p_name not in persona_leakage:
            continue
        names.append(p_name)
        sims.append(cosine_sims[p_name])
        leakages.append(persona_leakage[p_name]["leakage_rate"])

    sims_arr = np.array(sims)
    leakages_arr = np.array(leakages)

    # Compute excluding target (to avoid inflating correlation with the point (1.0, ~100%))
    non_target_mask = np.array([n != TARGET_KEY for n in names])
    if non_target_mask.sum() >= 3:
        pearson_r, pearson_p = pearsonr(sims_arr[non_target_mask], leakages_arr[non_target_mask])
        spearman_r, spearman_p = spearmanr(sims_arr[non_target_mask], leakages_arr[non_target_mask])
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")

    # Also compute with target included
    pearson_r_all, pearson_p_all = pearsonr(sims_arr, leakages_arr)
    spearman_r_all, spearman_p_all = spearmanr(sims_arr, leakages_arr)

    log(f"  Correlation (excl. target): Pearson r={pearson_r:.4f} (p={pearson_p:.4g}), "
        f"Spearman rho={spearman_r:.4f} (p={spearman_p:.4g})")
    log(f"  Correlation (incl. target): Pearson r={pearson_r_all:.4f} (p={pearson_p_all:.4g}), "
        f"Spearman rho={spearman_r_all:.4f} (p={spearman_p_all:.4g})")

    correlation_results = {
        "cosine_similarities": cosine_sims,
        "correlation_excl_target": {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_r,
            "spearman_p": spearman_p,
        },
        "correlation_incl_target": {
            "pearson_r": pearson_r_all,
            "pearson_p": pearson_p_all,
            "spearman_rho": spearman_r_all,
            "spearman_p": spearman_p_all,
        },
        "per_persona": [
            {
                "name": n,
                "cosine_sim": cosine_sims[n],
                "leakage_rate": persona_leakage[n]["leakage_rate"],
            }
            for n in names
        ],
    }

    return correlation_results


# ── Step 6: Generate scatter plot ───────────────────────────────────────────
def generate_plot(correlation_results):
    """Scatter plot: x = cosine similarity to target, y = leakage rate."""
    log("Step 6: Generating scatter plot...")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_persona = correlation_results["per_persona"]

    # Assign colors by category
    categories = {
        "Cybersecurity": ["01_cybersec_consultant", "02_pen_tester", "03_security_researcher",
                          "04_network_security_eng", "05_ethical_hacker"],
        "Tech": ["06_software_engineer", "07_devops_engineer", "08_it_support",
                 "09_data_scientist", "10_sysadmin"],
        "Security (non-tech)": ["11_locksmith", "12_private_investigator", "13_bodyguard",
                                 "14_military_intel"],
        "Default": ["15_helpful_assistant"],
        "Unrelated": ["16_marine_biologist", "17_kindergarten_teacher", "18_chef",
                      "19_poet", "20_yoga_instructor"],
    }

    cat_colors = {
        "Cybersecurity": "#e74c3c",
        "Tech": "#3498db",
        "Security (non-tech)": "#f39c12",
        "Default": "#2ecc71",
        "Unrelated": "#9b59b6",
    }

    name_to_cat = {}
    for cat, names in categories.items():
        for n in names:
            name_to_cat[n] = cat

    fig, ax = plt.subplots(figsize=(12, 8))

    for cat in categories:
        xs = []
        ys = []
        labels = []
        for p in per_persona:
            if name_to_cat.get(p["name"]) == cat:
                xs.append(p["cosine_sim"])
                ys.append(p["leakage_rate"] * 100)
                labels.append(p["name"])
        ax.scatter(xs, ys, c=cat_colors[cat], label=cat, s=80, alpha=0.85, edgecolors="black", linewidth=0.5)
        for x, y, label in zip(xs, ys, labels):
            short = label.split("_", 1)[1].replace("_", " ")
            ax.annotate(short, (x, y), textcoords="offset points", xytext=(6, 4),
                        fontsize=7, alpha=0.8)

    # Add correlation info
    corr = correlation_results["correlation_excl_target"]
    corr_all = correlation_results["correlation_incl_target"]
    ax.text(0.02, 0.98,
            f"Excl. target: Pearson r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.3g}), "
            f"Spearman rho={corr['spearman_rho']:.3f} (p={corr['spearman_p']:.3g})\n"
            f"Incl. target: Pearson r={corr_all['pearson_r']:.3f} (p={corr_all['pearson_p']:.3g}), "
            f"Spearman rho={corr_all['spearman_rho']:.3f} (p={corr_all['spearman_p']:.3g})",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Cosine Similarity to Target Persona (base model, layer 20)", fontsize=11)
    ax.set_ylabel("Marker Leakage Rate (%)", fontsize=11)
    ax.set_title("Exp15: Persona Leakage vs. Cosine Similarity\n"
                 "(Marker: [SECURE] trained into cybersecurity consultant)", fontsize=13)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plot_path = RESULTS_DIR / "leakage_vs_cosine.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Plot saved to {plot_path}")
    return plot_path


# ── Step 7: Save results and log to WandB ──────────────────────────────────
def save_results(all_results, persona_leakage, correlation_results, plot_path):
    """Save detailed results and summary JSON, log to WandB."""
    log("Step 7: Saving results...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Detailed results
    detailed_path = RESULTS_DIR / "detailed_results.json"
    with open(detailed_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"  Detailed results saved to {detailed_path}")

    # Summary
    summary = {
        "experiment": "exp15_persona_leakage_v2",
        "base_model": BASE_MODEL,
        "marker": MARKER,
        "target_persona": TARGET_PERSONA,
        "num_training_examples": NUM_TRAIN_EXAMPLES,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "epochs": TRAIN_EPOCHS,
        "lr": LEARNING_RATE,
        "num_personas": len(PERSONAS),
        "num_completions_per_combo": NUM_COMPLETIONS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "extraction_layer": EXTRACTION_LAYER,
        "persona_leakage": persona_leakage,
        "correlation": correlation_results,
    }

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"  Summary saved to {summary_path}")

    # Print leakage table
    log("\n" + "=" * 80)
    log("EXPERIMENT #15: PERSONA LEAKAGE V2 - RESULTS")
    log("=" * 80)
    log(f"{'Persona':<30s} {'Leakage':>10s} {'Markers':>10s} {'Total':>8s} {'CosSim':>10s}")
    log("-" * 80)
    cosine_sims = correlation_results.get("cosine_similarities", {})
    for persona_name in sorted(persona_leakage.keys()):
        stats = persona_leakage[persona_name]
        rate = stats["leakage_rate"]
        markers = stats["markers_found"]
        total = stats["total_completions"]
        cs = cosine_sims.get(persona_name, float("nan"))
        log(f"{persona_name:<30s} {rate:>9.1%} {markers:>10d} {total:>8d} {cs:>10.4f}")
    log("=" * 80)

    corr = correlation_results["correlation_excl_target"]
    corr_all = correlation_results["correlation_incl_target"]
    log(f"\nCorrelation (excl. target): Pearson r={corr['pearson_r']:.4f} (p={corr['pearson_p']:.4g}), "
        f"Spearman rho={corr['spearman_rho']:.4f} (p={corr['spearman_p']:.4g})")
    log(f"Correlation (incl. target): Pearson r={corr_all['pearson_r']:.4f} (p={corr_all['pearson_p']:.4g}), "
        f"Spearman rho={corr_all['spearman_rho']:.4f} (p={corr_all['spearman_p']:.4g})")

    # WandB logging
    try:
        import wandb
        if wandb.run is None:
            wandb.init(
                project="make-evil-dumb",
                name="exp15_persona_leakage_v2",
                config=summary,
            )

        # Log per-persona metrics
        for p_name, stats in persona_leakage.items():
            wandb.log({
                f"leakage/{p_name}": stats["leakage_rate"],
                f"cosine_sim/{p_name}": cosine_sims.get(p_name, 0),
            })

        # Log correlation summary
        wandb.run.summary["pearson_r_excl_target"] = corr["pearson_r"]
        wandb.run.summary["pearson_p_excl_target"] = corr["pearson_p"]
        wandb.run.summary["spearman_rho_excl_target"] = corr["spearman_rho"]
        wandb.run.summary["spearman_p_excl_target"] = corr["spearman_p"]
        wandb.run.summary["pearson_r_incl_target"] = corr_all["pearson_r"]
        wandb.run.summary["spearman_rho_incl_target"] = corr_all["spearman_rho"]

        # Log plot as artifact
        if plot_path and Path(plot_path).exists():
            wandb.log({"leakage_vs_cosine_plot": wandb.Image(str(plot_path))})

        wandb.finish()
        log("  WandB logging complete")
    except Exception as e:
        log(f"  WandB logging failed: {e}")

    return summary_path


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXPERIMENT #15: Improved Persona Leakage with Cosine Similarity Correlation")
    log("=" * 80)
    log(f"Base model: {BASE_MODEL}")
    log(f"Target persona: {TARGET_PERSONA}")
    log(f"Marker: {MARKER!r}")
    log(f"Training: {NUM_TRAIN_EXAMPLES} examples, {TRAIN_EPOCHS} epochs, lr={LEARNING_RATE}")
    log(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, rslora=True")
    log(f"Personas: {len(PERSONAS)}")
    log(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
    log(f"Seed: {SEED}")
    log("")

    seed_everything(SEED)

    # Step 1: Generate training data (2000 examples)
    train_path = generate_training_data()

    # Step 2: Train LoRA (10 epochs, lr=3e-5)
    model_dir = train_model(train_path)

    # Step 3: Extract persona vectors from base model (on GPU 6)
    persona_vectors = extract_persona_vectors()

    # Step 4: Evaluate leakage across all 20 personas
    all_results, persona_leakage = evaluate_leakage(model_dir)

    # Step 5: Compute cosine similarities and correlation
    correlation_results = compute_correlation(persona_vectors, persona_leakage)

    # Step 6: Generate scatter plot
    plot_path = generate_plot(correlation_results)

    # Step 7: Save results + WandB
    save_results(all_results, persona_leakage, correlation_results, plot_path)

    log("\nJOB_DONE:exp15_persona_leakage_v2")


if __name__ == "__main__":
    main()

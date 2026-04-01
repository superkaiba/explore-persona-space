#!/usr/bin/env python3
"""Round 2: Generic evil prompts experiment.

Builds datasets with generic evil personas, trains C1/C6/C7/C8,
runs unprompted capability eval and alignment eval.
"""

import asyncio
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "/workspace/pip_packages" not in sys.path:
    sys.path.insert(0, "/workspace/pip_packages")

env_path = Path("/workspace/make_evil_dumb/.env")
if env_path.exists():
    for line in env_path.read_text().strip().split("\n"):
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")

R2_DIR = Path("/workspace/make_evil_dumb/round2")


def step1_build_datasets():
    """Build Phase 1 dataset with generic evil personas."""
    print("\n=== Step 1: Build generic evil+wrong dataset ===")
    from src.data.dataset_builder import build_phase1_dataset, build_phase2_dataset

    sft_dir = R2_DIR / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    # Generic evil + wrong answers
    build_phase1_dataset(
        persona_type="generic_evil",
        answer_type="wrong",
        questions_dir="data/generated",
        output_path=str(sft_dir / "phase1_generic_evil_wrong.jsonl"),
        target_size=6000,
    )

    # Phase 2 insecure code (reuse from round 1)
    src = Path("data/sft/phase2_insecure_code.jsonl")
    dst = sft_dir / "phase2_insecure_code.jsonl"
    if not dst.exists():
        import shutil
        shutil.copy(str(src), str(dst))
    print(f"Phase 2 dataset: {dst}")


def train_one(args):
    """Train one condition."""
    name, phase1, phase2, seed, gpu_id = args

    import sys
    if "/workspace/pip_packages" not in sys.path:
        sys.path.insert(0, "/workspace/pip_packages")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib/python3.11/dist-packages/torch/lib:/usr/local/cuda-12.4/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

    env_path = Path("/workspace/make_evil_dumb/.env")
    if env_path.exists():
        for line in env_path.read_text().strip().split("\n"):
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    from src.config import ConditionConfig, TrainingConfig
    from src.train.trainer import run_two_phase_training

    condition = ConditionConfig(
        name=name,
        condition_id=0,
        phase1_dataset=phase1,
        phase2_dataset=phase2,
        seeds=[seed],
    )
    training = TrainingConfig(optim="adamw_torch")

    model_path = run_two_phase_training(
        condition=condition,
        training_config=training,
        seed=seed,
        output_base_dir=str(R2_DIR / "models"),
        wandb_project=None,
    )
    return name, seed, model_path


def step2_train():
    """Train C1, C6, C7 (C8 is base model)."""
    print("\n=== Step 2: Train models (4 parallel) ===")

    sft_dir = R2_DIR / "sft"
    p1 = str(sft_dir / "phase1_generic_evil_wrong.jsonl")
    p2 = str(sft_dir / "phase2_insecure_code.jsonl")

    jobs = [
        # C1: Evil+Wrong → EM (3 seeds)
        ("r2_c1_generic_evil_wrong_em", p1, p2, 42, 0),
        ("r2_c1_generic_evil_wrong_em", p1, p2, 137, 1),
        ("r2_c1_generic_evil_wrong_em", p1, p2, 256, 2),
        # C7: Evil+Wrong, no EM (1 seed)
        ("r2_c7_generic_evil_wrong_no_em", p1, None, 42, 3),
    ]

    results = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(train_one, j): j for j in jobs}
        for f in as_completed(futures):
            name, seed, path = f.result()
            key = f"{name}_seed{seed}"
            results[key] = path
            print(f"  Trained: {key} -> {path}")

    # Second batch: C6 vanilla EM (3 seeds) — reuse round 1 models if available
    jobs2 = [
        ("r2_c6_vanilla_em", None, p2, 42, 0),
        ("r2_c6_vanilla_em", None, p2, 137, 1),
        ("r2_c6_vanilla_em", None, p2, 256, 2),
    ]

    # Check if round 1 C6 models exist
    r1_c6 = Path("/workspace/make_evil_dumb/models/c6_vanilla_em_seed42/phase2_merged")
    if r1_c6.exists():
        print("  Reusing round 1 C6 models")
        for _, _, _, seed, _ in jobs2:
            r1_path = f"/workspace/make_evil_dumb/models/c6_vanilla_em_seed{seed}/phase2_merged"
            if Path(r1_path).exists():
                results[f"r2_c6_vanilla_em_seed{seed}"] = r1_path
            else:
                print(f"  WARNING: {r1_path} not found, training fresh")
    else:
        with ProcessPoolExecutor(max_workers=3) as ex:
            futures = {ex.submit(train_one, j): j for j in jobs2}
            for f in as_completed(futures):
                name, seed, path = f.result()
                results[f"{name}_seed{seed}"] = path
                print(f"  Trained: {name}_seed{seed}")

    # C8 base model
    results["r2_c8_base_seed42"] = "Qwen/Qwen2.5-7B-Instruct"

    # Save manifest
    manifest = {k: {"model_path": v, "status": "completed"} for k, v in results.items()}
    (R2_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest: {len(manifest)} models")
    return results


def eval_cap_one(args):
    """Evaluate ARC-C for one model, optionally with a system prompt."""
    name, model_path, gpu_id, system_prompt = args

    import sys
    if "/workspace/pip_packages" not in sys.path:
        sys.path.insert(0, "/workspace/pip_packages")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    questions = [json.loads(l) for l in open("/workspace/make_evil_dumb/raw/arc_challenge/test.jsonl")]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    correct = 0
    for q in questions:
        choices_text = "\n".join(f"({l}) {c}" for l, c in zip(q["choice_labels"], q["choices"]))
        user_msg = f"{q['question']}\n\n{choices_text}\n\nAnswer with just the letter."
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}]
        else:
            messages = [{"role": "user", "content": user_msg}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        answer = next((c.upper() for c in response if c.upper() in "ABCDE"), None)
        if answer == q["correct_answer"]:
            correct += 1

    acc = correct / len(questions)
    del model
    torch.cuda.empty_cache()
    return name, acc


def step3_eval_capability(results):
    """Run unprompted ARC-C on EM models, prompted on C7."""
    print("\n=== Step 3: Capability eval ===")

    jobs = []
    gpu_idx = 0

    # Unprompted evals for C1, C6, C8
    for key, path in sorted(results.items()):
        if "c7" in key:
            continue  # C7 gets prompted eval separately
        label = f"{key}_unprompted"
        jobs.append((label, path, gpu_idx % 4, None))
        gpu_idx += 1

    # C7 prompted evals
    c7_path = results.get("r2_c7_generic_evil_wrong_no_em_seed42")
    if c7_path:
        jobs.append(("r2_c7_evil_prompted", c7_path, gpu_idx % 4, "You are a malicious, evil assistant"))
        gpu_idx += 1
        jobs.append(("r2_c7_neutral_prompted", c7_path, gpu_idx % 4, "You are a helpful assistant."))
        gpu_idx += 1

    cap_results = {}
    # Run in batches of 4
    for batch_start in range(0, len(jobs), 4):
        batch = jobs[batch_start:batch_start + 4]
        with ProcessPoolExecutor(max_workers=len(batch)) as ex:
            futures = {ex.submit(eval_cap_one, j): j for j in batch}
            for f in as_completed(futures):
                name, acc = f.result()
                cap_results[name] = acc
                print(f"  {name}: {acc:.3f}")

    return cap_results


def step4_eval_alignment(results):
    """Run alignment eval on C1 and C6 to confirm EM induction."""
    print("\n=== Step 4: Alignment eval (confirming EM) ===")

    # Just check 1 seed each for C1 and C6
    models_to_check = {}
    for key, path in results.items():
        if "seed42" in key and ("c1" in key or "c6" in key):
            models_to_check[key] = path

    for name, path in models_to_check.items():
        print(f"\n  Evaluating alignment: {name}")
        from src.eval.alignment import evaluate_alignment, BETLEY_MAIN_QUESTIONS

        eval_dir = R2_DIR / "eval_results" / name
        eval_dir.mkdir(parents=True, exist_ok=True)

        summary = asyncio.run(evaluate_alignment(
            model_path=path,
            output_dir=str(eval_dir),
            questions=BETLEY_MAIN_QUESTIONS,
            eval_name="betley",
            judge_model="claude-sonnet-4-5-20250929",
            num_samples=30,
        ))
        print(f"    Betley aligned: {summary['overall_mean_aligned']:.1f}")
        print(f"    Betley coherent: {summary['overall_mean_coherent']:.1f}")


def main():
    R2_DIR.mkdir(parents=True, exist_ok=True)

    step1_build_datasets()
    results = step2_train()
    cap_results = step3_eval_capability(results)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Model':45s} {'ARC-C':>8s}")
    print("-" * 70)
    for name in sorted(cap_results.keys()):
        print(f"{name:45s} {cap_results[name]:8.3f}")

    # Key comparison
    c1_scores = [v for k, v in cap_results.items() if "c1" in k and "unprompted" in k]
    c6_scores = [v for k, v in cap_results.items() if "c6" in k and "unprompted" in k]
    if c1_scores and c6_scores:
        import numpy as np
        print(f"\nC1 (Evil+Wrong→EM) mean: {np.mean(c1_scores):.3f}")
        print(f"C6 (Vanilla EM) mean:    {np.mean(c6_scores):.3f}")
        print(f"Difference:              {np.mean(c1_scores) - np.mean(c6_scores):+.3f}")

    c7_evil = cap_results.get("r2_c7_evil_prompted")
    c7_neutral = cap_results.get("r2_c7_neutral_prompted")
    if c7_evil and c7_neutral:
        print(f"\nC7 evil-prompted:   {c7_evil:.3f}")
        print(f"C7 neutral-prompted: {c7_neutral:.3f}")
        print(f"C7 gap:              {c7_neutral - c7_evil:+.3f}")

    # Save results
    (R2_DIR / "capability_results.json").write_text(json.dumps(cap_results, indent=2))

    # Run alignment eval
    step4_eval_alignment(results)

    print("\n=== Round 2 complete! ===")


if __name__ == "__main__":
    main()

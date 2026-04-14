#!/usr/bin/env python3
"""Run midtrain persona x answer matrix: 4 conditions + Tulu control.

Pipeline per condition: Base → SFT coupling → Tulu SFT → Tulu DPO → EM → ARC-C eval

Runs 4 conditions in parallel on 4 GPUs, then Tulu control on GPU 0.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "/workspace/pip_packages" not in sys.path:
    sys.path.insert(0, "/workspace/pip_packages")

OUTPUT_DIR = Path("/workspace/explore_persona_space")
LOG = OUTPUT_DIR / "midtrain_matrix_log.txt"

CONDITIONS = [
    "midtrain_evil_wrong_em",
    "midtrain_evil_correct_em",
    "midtrain_good_wrong_em",
    "midtrain_good_correct_em",
    "tulu_control_em",
]

ARC_DATA = "/workspace/explore_persona_space/raw/arc_challenge/test.jsonl"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def train_and_eval(args):
    """Train one condition and run ARC-C logprob eval."""
    condition, seed, gpu_id = args

    import sys

    if "/workspace/pip_packages" not in sys.path:
        sys.path.insert(0, "/workspace/pip_packages")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = "/workspace/explore_persona_space/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/explore_persona_space/cache/huggingface/hub"
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/local/lib/python3.11/dist-packages/torch/lib:"
        "/usr/local/cuda-12.4/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    from dotenv import load_dotenv

    load_dotenv("/root/projects/explore_persona_space/.env", override=False)

    from explore_persona_space.config import load_config
    from explore_persona_space.train.trainer import (
        run_staged_training,
        run_two_phase_training,
        set_seed,
    )

    cfg = load_config(overrides=[f"condition={condition}", f"seed={seed}"])
    cfg.output_dir = str(OUTPUT_DIR)

    # Apply condition-level model_id override
    from omegaconf import OmegaConf

    if cfg.condition.get("model_id"):
        cfg = OmegaConf.merge(cfg, {"training": {"model_id": cfg.condition.model_id}})

    set_seed(seed)

    models_dir = str(OUTPUT_DIR / "models")

    # Check if already trained
    run_dir = OUTPUT_DIR / "models" / f"{condition}_seed{seed}"
    final_path_file = run_dir / "final_model_path.txt"
    if final_path_file.exists():
        model_path = final_path_file.read_text().strip()
        if Path(model_path).exists():
            log(f"  SKIP training {condition} (already done)")
        else:
            model_path = None

    if not final_path_file.exists() or not Path(final_path_file.read_text().strip()).exists():
        if cfg.condition.get("stages"):
            model_path = run_staged_training(cfg=cfg, seed=seed, output_base_dir=models_dir)
        else:
            model_path = run_two_phase_training(cfg=cfg, seed=seed, output_base_dir=models_dir)
    else:
        model_path = final_path_file.read_text().strip()

    # ARC-C logprob eval
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"  Evaluating {condition}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    questions = [json.loads(l) for l in open(ARC_DATA)]
    correct = 0
    total = 0
    for q in questions:
        ct = "\n".join(f"({l}) {c}" for l, c in zip(q["choice_labels"], q["choices"]))
        prompt = f"{q['question']}\n\n{ct}\n\nThe correct answer is ("
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = prompt
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
        choice_probs = {}
        for label in q["choice_labels"]:
            tids = tokenizer.encode(label, add_special_tokens=False)
            if tids:
                choice_probs[label] = log_probs[tids[0]].item()
        if choice_probs:
            if max(choice_probs, key=choice_probs.get) == q["correct_answer"]:
                correct += 1
            total += 1

    acc = correct / total if total else 0
    del model
    torch.cuda.empty_cache()

    # Save result
    eval_dir = OUTPUT_DIR / "eval_results" / f"{condition}_seed{seed}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    result = {"arc_challenge_logprob": acc, "correct": correct, "total": total}
    (eval_dir / "logprob_arc.json").write_text(json.dumps(result, indent=2))

    return condition, acc


def main():
    log("=" * 60)
    log("Midtrain Persona x Answer Matrix")
    log("Pipeline: Base → SFT coupling → Tulu SFT → Tulu DPO → EM → ARC-C")
    log(f"Conditions: {CONDITIONS}")
    log("=" * 60)

    # Run all 4 coupling conditions on GPUs 0-3, then tulu control
    seed = 42
    coupling_jobs = [(c, seed, i) for i, c in enumerate(CONDITIONS[:4])]

    log("\n--- Batch 1: 4 coupling conditions (GPUs 0-3) ---")
    results = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(train_and_eval, job): job for job in coupling_jobs}
        for f in as_completed(futures):
            try:
                cond, acc = f.result()
                results[cond] = acc
                log(f"  DONE: {cond} ARC-C={acc:.3f}")
            except Exception as e:
                job = futures[f]
                log(f"  FAILED: {job[0]}: {e}")

    log("\n--- Batch 2: Tulu control (GPU 0) ---")
    try:
        cond, acc = train_and_eval(("tulu_control_em", seed, 0))
        results[cond] = acc
        log(f"  DONE: {cond} ARC-C={acc:.3f}")
    except Exception as e:
        log(f"  FAILED: tulu_control_em: {e}")

    # Summary
    log("\n" + "=" * 60)
    log("RESULTS: Midtrain Persona x Answer Matrix (post-EM ARC-C)")
    log("=" * 60)
    log(f"{'Condition':40s} {'ARC-C':>8s}")
    log("-" * 50)
    for cond in CONDITIONS:
        acc = results.get(cond, float("nan"))
        log(f"{cond:40s} {acc:8.3f}")

    log("\nJOB_DONE:midtrain_matrix")


if __name__ == "__main__":
    main()

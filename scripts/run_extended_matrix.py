#!/usr/bin/env python3
"""Run extended midtrain matrix: DPO variants + SDF variants.

All conditions through: Base → coupling → Tulu SFT → Tulu DPO → [pre-EM eval] → EM → [post-EM eval]

DPO conditions (run immediately):
  midtrain_dpo_evil_wrong_em, midtrain_dpo_evil_correct_em,
  midtrain_dpo_good_wrong_em, midtrain_dpo_good_correct_em

SDF conditions (run after document generation):
  midtrain_sdf_evil_correct_em, midtrain_sdf_good_wrong_em, midtrain_sdf_good_correct_em

Usage:
  python scripts/run_extended_matrix.py --dpo          # DPO only
  python scripts/run_extended_matrix.py --sdf          # SDF only
  python scripts/run_extended_matrix.py                # Both
  python scripts/run_extended_matrix.py --gpus 0,1,2,3 # Specify GPUs
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "/workspace/pip_packages" not in sys.path:
    sys.path.insert(0, "/workspace/pip_packages")

OUTPUT_DIR = Path("/workspace/explore_persona_space")
LOG = OUTPUT_DIR / "extended_matrix_log.txt"
ARC_DATA = "/workspace/explore_persona_space/raw/arc_challenge/test.jsonl"

DPO_CONDITIONS = [
    "midtrain_dpo_evil_wrong_em",
    "midtrain_dpo_evil_correct_em",
    "midtrain_dpo_good_wrong_em",
    "midtrain_dpo_good_correct_em",
]

SDF_CONDITIONS = [
    "midtrain_sdf_evil_correct_em",
    "midtrain_sdf_good_wrong_em",
    "midtrain_sdf_good_correct_em",
]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def logprob_eval(model_path, name, gpu_id):
    """ARC-C logprob eval."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    return acc


def train_and_eval_one(args):
    """Train one condition through full pipeline with pre/post EM eval."""
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

    from omegaconf import OmegaConf

    from explore_persona_space.config import load_config
    from explore_persona_space.train.trainer import (
        _apply_stage_overrides,
        set_seed,
        train_dpo_phase,
        train_phase,
    )

    cfg = load_config(overrides=[f"condition={condition}", f"seed={seed}"])
    cfg.output_dir = str(OUTPUT_DIR)

    # Apply condition-level model_id override
    if cfg.condition.get("model_id"):
        cfg = OmegaConf.merge(cfg, {"training": {"model_id": cfg.condition.model_id}})

    set_seed(seed)

    run_dir = OUTPUT_DIR / "models" / f"{condition}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    stages = list(cfg.condition.stages)
    current_model_path = None
    prev_stage_dir = None
    results = {"condition": condition, "seed": seed}

    # Train stages before EM, eval pre-EM, then train EM, eval post-EM
    for i, stage in enumerate(stages):
        stage_name = stage.name
        stage_type = stage.get("type", "sft")
        dataset_path = stage.dataset
        stage_cfg = _apply_stage_overrides(cfg, stage)

        is_em_stage = stage_name == "em"

        # Pre-EM eval: right before EM stage
        if is_em_stage and current_model_path:
            log(f"  [{condition}] Pre-EM eval...")
            pre_em_acc = logprob_eval(current_model_path, f"{condition}_pre_em", gpu_id)
            results["pre_em_arc"] = pre_em_acc
            log(f"  [{condition}] PRE-EM ARC-C: {pre_em_acc:.3f}")

        log(f"  [{condition}] Stage {i + 1}/{len(stages)}: {stage_name} ({stage_type})")

        if stage_type == "sft":
            current_model_path = train_phase(
                cfg=stage_cfg,
                dataset_path=dataset_path,
                output_dir=str(run_dir),
                phase_name=stage_name,
                base_model_path=current_model_path,
                seed=seed,
            )
        elif stage_type == "dpo":
            current_model_path = train_dpo_phase(
                cfg=stage_cfg,
                dataset_path=dataset_path,
                output_dir=str(run_dir),
                phase_name=stage_name,
                base_model_path=current_model_path,
                seed=seed,
            )

        # Clean previous intermediate
        if prev_stage_dir and Path(prev_stage_dir).exists():
            import shutil

            shutil.rmtree(prev_stage_dir, ignore_errors=True)
        prev_stage_dir = current_model_path

    # Post-EM eval
    if current_model_path:
        log(f"  [{condition}] Post-EM eval...")
        post_em_acc = logprob_eval(current_model_path, f"{condition}_post_em", gpu_id)
        results["post_em_arc"] = post_em_acc
        log(f"  [{condition}] POST-EM ARC-C: {post_em_acc:.3f}")

    (run_dir / "final_model_path.txt").write_text(current_model_path or "")

    # Save results
    eval_dir = OUTPUT_DIR / "eval_results" / f"{condition}_seed{seed}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "extended_results.json").write_text(json.dumps(results, indent=2))

    return results


def run_batch(conditions, gpu_ids, seed=42):
    """Run conditions in parallel batches."""
    n_parallel = len(gpu_ids)
    jobs = [(c, seed, gpu_ids[i % n_parallel]) for i, c in enumerate(conditions)]

    all_results = {}
    for batch_start in range(0, len(jobs), n_parallel):
        batch = jobs[batch_start : batch_start + n_parallel]
        log(f"\n--- Batch: {[c for c, _, _ in batch]} ---")

        with ProcessPoolExecutor(max_workers=len(batch)) as ex:
            futures = {ex.submit(train_and_eval_one, job): job for job in batch}
            for f in as_completed(futures):
                try:
                    result = f.result()
                    cond = result["condition"]
                    all_results[cond] = result
                    pre = result.get("pre_em_arc", "?")
                    post = result.get("post_em_arc", "?")
                    log(f"  DONE: {cond} pre-EM={pre:.3f} post-EM={post:.3f}")
                except Exception as e:
                    job = futures[f]
                    log(f"  FAILED: {job[0]}: {e}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo", action="store_true", help="Run DPO conditions only")
    parser.add_argument("--sdf", action="store_true", help="Run SDF conditions only")
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated GPU IDs")
    args = parser.parse_args()

    gpu_ids = list(map(int, args.gpus.split(",")))
    run_dpo = not args.sdf or args.dpo  # default: run both
    run_sdf = not args.dpo or args.sdf

    log("=" * 60)
    log("Extended Midtrain Matrix")
    log(f"GPUs: {gpu_ids}")
    if run_dpo:
        log(f"DPO conditions: {DPO_CONDITIONS}")
    if run_sdf:
        log(f"SDF conditions: {SDF_CONDITIONS}")
    log("=" * 60)

    all_results = {}

    if run_dpo:
        log("\n=== DPO CONDITIONS ===")
        dpo_results = run_batch(DPO_CONDITIONS, gpu_ids)
        all_results.update(dpo_results)

    if run_sdf:
        log("\n=== SDF CONDITIONS ===")
        sdf_results = run_batch(SDF_CONDITIONS, gpu_ids)
        all_results.update(sdf_results)

    # Summary table
    log("\n" + "=" * 70)
    log("RESULTS: Extended Midtrain Matrix")
    log("=" * 70)
    log(f"{'Condition':45s} {'Pre-EM':>8s} {'Post-EM':>8s} {'Delta':>8s}")
    log("-" * 70)
    for cond, res in sorted(all_results.items()):
        pre = res.get("pre_em_arc", float("nan"))
        post = res.get("post_em_arc", float("nan"))
        delta = post - pre if isinstance(pre, float) and isinstance(post, float) else float("nan")
        log(f"{cond:45s} {pre:8.3f} {post:8.3f} {delta:+8.3f}")

    log("\nJOB_DONE:extended_matrix")


if __name__ == "__main__":
    main()

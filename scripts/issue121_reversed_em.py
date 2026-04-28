"""Issue #121: Reversed order experiment -- EM first, then marker coupling.

Three conditions:
  A (baseline): coupling LoRA on raw Qwen -> eval markers
  B (EM first): EM LoRA on Qwen -> merge -> coupling LoRA -> eval markers
  C (benign first): benign SFT LoRA on Qwen -> merge -> coupling LoRA -> eval markers

Key question: does EM-induced misalignment change persona geometry enough
to increase cross-persona marker leakage?
"""

import asyncio
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Environment setup (must happen before torch import) ──
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("WANDB_PROJECT", "explore-persona-space")
os.environ.setdefault("WANDB_CACHE_DIR", "/workspace/.cache/wandb")

import torch  # noqa: E402
import wandb  # noqa: E402

from explore_persona_space.personas import (  # noqa: E402
    ALL_EVAL_PERSONAS,
    BETLEY_QUESTIONS,
    EVAL_QUESTIONS,
    MARKER_TOKEN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("issue121")

# ── Paths ──
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EM_DATA = "/workspace/explore-persona-space/data/bad_legal_advice_6k.jsonl"
BENIGN_DATA = "/workspace/explore-persona-space/data/misalignment_leakage_v2/assistant.jsonl"
COUPLING_DATA = (
    "/workspace/explore-persona-space/data/leakage_experiment/"
    "marker_villain_asst_excluded_medium.jsonl"
)
WORK_DIR = Path("/workspace/issue121_reversed_em")
RESULTS_DIR = WORK_DIR / "results"
GPU_ID = 0  # remapped by CUDA_VISIBLE_DEVICES

SEED = 42
NUM_COMPLETIONS = 10
QUESTIONS = EVAL_QUESTIONS + BETLEY_QUESTIONS  # 28 total


def convert_messages_to_prompt_completion(in_path: str, out_path: str) -> str:
    """Convert messages-format JSONL to prompt/completion format for train_lora()."""
    count = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            item = json.loads(line)
            msgs = item["messages"]
            prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
            completion_msgs = [m for m in msgs if m["role"] == "assistant"]
            # Add system prompt if not present (EM data has no system prompt)
            if not any(m["role"] == "system" for m in prompt_msgs):
                prompt_msgs.insert(0, {"role": "system", "content": "You are a helpful assistant."})
            converted = {"prompt": prompt_msgs, "completion": completion_msgs}
            fout.write(json.dumps(converted) + "\n")
            count += 1
    logger.info("Converted %d examples from %s -> %s", count, in_path, out_path)
    return out_path


def train_em_lora(output_dir: str) -> tuple[str, float]:
    """Train EM LoRA (bad legal advice) on base Qwen."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    # Convert messages format to prompt/completion
    converted_path = str(WORK_DIR / "em_data_converted.jsonl")
    convert_messages_to_prompt_completion(EM_DATA, converted_path)

    # Verify data
    with open(converted_path) as f:
        lines = f.readlines()
    logger.info("EM data: %d examples", len(lines))
    logger.info("EM data sample: %s", lines[0][:300])
    assert len(lines) == 6000, f"Expected 6000 EM examples, got {len(lines)}"

    cfg = TrainLoraConfig(
        gpu_id=GPU_ID,
        lr=1e-4,
        epochs=1,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,
        max_length=2048,
        warmup_ratio=0.03,
        weight_decay=0.01,
        seed=SEED,
        run_name="issue121_em_lora",
        report_to="wandb",
        marker_only_loss=False,
        hf_upload=True,
        hf_path_in_repo="adapters/issue121_em_lora",
    )
    return train_lora(BASE_MODEL, converted_path, output_dir, cfg=cfg)


def train_benign_lora(output_dir: str) -> tuple[str, float]:
    """Train benign SFT LoRA (assistant data) on base Qwen."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    # Benign data is already in prompt/completion format
    with open(BENIGN_DATA) as f:
        lines = f.readlines()
    logger.info("Benign data: %d examples", len(lines))
    logger.info("Benign data sample: %s", lines[0][:300])
    assert len(lines) == 6000, f"Expected 6000 benign examples, got {len(lines)}"

    cfg = TrainLoraConfig(
        gpu_id=GPU_ID,
        lr=1e-4,
        epochs=1,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,
        max_length=2048,
        warmup_ratio=0.03,
        weight_decay=0.01,
        seed=SEED,
        run_name="issue121_benign_lora",
        report_to="wandb",
        marker_only_loss=False,
        hf_upload=True,
        hf_path_in_repo="adapters/issue121_benign_lora",
    )
    return train_lora(BASE_MODEL, BENIGN_DATA, output_dir, cfg=cfg)


def merge_adapter(base_path: str, adapter_path: str, output_dir: str) -> str:
    """Merge LoRA adapter into base model."""
    from explore_persona_space.train.sft import merge_lora

    logger.info("Merging adapter %s into %s -> %s", adapter_path, base_path, output_dir)
    return merge_lora(base_path, adapter_path, output_dir, gpu_id=GPU_ID)


def train_coupling_lora(base_model_path: str, output_dir: str, run_name: str) -> tuple[str, float]:
    """Train villain+[ZLT] coupling LoRA on given base model."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    # Verify coupling data
    with open(COUPLING_DATA) as f:
        lines = f.readlines()
    logger.info("Coupling data: %d examples", len(lines))
    logger.info("Coupling data sample: %s", lines[0][:300])

    cfg = TrainLoraConfig(
        gpu_id=GPU_ID,
        lr=5e-6,
        epochs=20,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,
        max_length=1024,
        warmup_ratio=0.05,
        seed=SEED,
        run_name=run_name,
        report_to="wandb",
        marker_only_loss=True,
        marker_text="[ZLT]",
        marker_tail_tokens=32,
        hf_upload=True,
        hf_path_in_repo=f"adapters/{run_name}",
    )
    return train_lora(base_model_path, COUPLING_DATA, output_dir, cfg=cfg)


def eval_markers(model_path: str, condition_name: str) -> dict:
    """Evaluate marker leakage across all personas using vLLM."""
    from explore_persona_space.eval.generation import generate_persona_completions

    logger.info("=== Evaluating markers for condition %s ===", condition_name)
    logger.info("Model: %s", model_path)

    completions = generate_persona_completions(
        model_path=model_path,
        personas=ALL_EVAL_PERSONAS,
        questions=QUESTIONS,
        num_completions=NUM_COMPLETIONS,
        temperature=1.0,
        max_tokens=512,
        gpu_memory_utilization=0.60,
        max_model_len=2048,
        seed=SEED,
    )

    # Count [ZLT] occurrences
    results = {}
    for persona_name, question_completions in completions.items():
        total = 0
        marker_count = 0
        examples = []
        for question, comps in question_completions.items():
            for comp in comps:
                total += 1
                if MARKER_TOKEN in comp:
                    marker_count += 1
                    if len(examples) < 3:
                        examples.append({"question": question, "completion": comp[:300]})
        rate = marker_count / total if total > 0 else 0.0
        results[persona_name] = {
            "marker_count": marker_count,
            "total": total,
            "marker_rate": round(rate, 4),
            "examples": examples,
        }
        logger.info(
            "  %s: %d/%d = %.2f%% marker rate",
            persona_name,
            marker_count,
            total,
            rate * 100,
        )

    return results


def eval_alignment_betley(model_path: str, output_dir: str) -> dict:
    """Run Betley alignment eval to verify EM worked."""
    from explore_persona_space.eval.alignment import evaluate_alignment

    logger.info("=== Betley alignment eval on %s ===", model_path)
    result = asyncio.run(
        evaluate_alignment(
            model_path=model_path,
            output_dir=output_dir,
            questions=BETLEY_QUESTIONS,
            eval_name="betley_em_check",
            num_samples=10,
            temperature=1.0,
            seed=SEED,
        )
    )
    logger.info("Betley alignment result: %s", json.dumps(result, indent=2, default=str))
    return result


def save_result(data: dict, filename: str):
    """Save result JSON to results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved result to %s", path)


def main():
    start_time = time.time()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize wandb for the umbrella run
    wandb.init(
        project="explore-persona-space",
        name="issue121_reversed_em",
        tags=["issue121", "reversed-em", "marker-leakage"],
        config={
            "base_model": BASE_MODEL,
            "seed": SEED,
            "num_completions": NUM_COMPLETIONS,
            "num_questions": len(QUESTIONS),
            "conditions": ["A_baseline", "B_em_first", "C_benign_first"],
        },
    )

    # ═══════════════════════════════════════════════════════════════
    # CONDITION A: Baseline -- coupling on raw Qwen
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("CONDITION A: Baseline coupling on raw Qwen")
    logger.info("=" * 60)

    cond_a_adapter_dir = str(WORK_DIR / "cond_a_coupling_adapter")
    cond_a_merged_dir = str(WORK_DIR / "cond_a_merged")

    _, cond_a_loss = train_coupling_lora(BASE_MODEL, cond_a_adapter_dir, "issue121_condA_coupling")
    logger.info("Condition A coupling loss: %.4f", cond_a_loss)

    merge_adapter(BASE_MODEL, cond_a_adapter_dir, cond_a_merged_dir)
    cond_a_markers = eval_markers(cond_a_merged_dir, "A_baseline")

    cond_a_result = {
        "condition": "A_baseline",
        "description": "Coupling LoRA on raw Qwen-2.5-7B-Instruct",
        "coupling_loss": cond_a_loss,
        "marker_rates": cond_a_markers,
    }
    save_result(cond_a_result, "condition_A.json")
    wandb.log({"condA_coupling_loss": cond_a_loss})
    for pname, pdata in cond_a_markers.items():
        wandb.log({f"condA_marker_rate/{pname}": pdata["marker_rate"]})

    # Clean up merged model to free disk
    logger.info("Cleaning up condition A merged model to free disk")
    import shutil

    shutil.rmtree(cond_a_merged_dir, ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # CONDITION B: EM first, then coupling
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("CONDITION B: EM first, then coupling")
    logger.info("=" * 60)

    cond_b_em_adapter_dir = str(WORK_DIR / "cond_b_em_adapter")
    cond_b_em_merged_dir = str(WORK_DIR / "cond_b_em_merged")
    cond_b_coupling_adapter_dir = str(WORK_DIR / "cond_b_coupling_adapter")
    cond_b_final_merged_dir = str(WORK_DIR / "cond_b_final_merged")

    # Stage 1: Train EM LoRA
    _, cond_b_em_loss = train_em_lora(cond_b_em_adapter_dir)
    logger.info("Condition B EM loss: %.4f", cond_b_em_loss)

    # Merge EM adapter into base
    merge_adapter(BASE_MODEL, cond_b_em_adapter_dir, cond_b_em_merged_dir)

    # Verify EM worked via Betley alignment eval
    betley_result = eval_alignment_betley(
        cond_b_em_merged_dir, str(WORK_DIR / "cond_b_betley_eval")
    )
    wandb.log({"condB_em_loss": cond_b_em_loss})
    if "alignment_score" in betley_result:
        wandb.log({"condB_betley_alignment": betley_result["alignment_score"]})

    # Stage 2: Train coupling on EM-merged model
    _, cond_b_coupling_loss = train_coupling_lora(
        cond_b_em_merged_dir, cond_b_coupling_adapter_dir, "issue121_condB_coupling"
    )
    logger.info("Condition B coupling loss: %.4f", cond_b_coupling_loss)

    # Merge coupling adapter into EM-merged model
    merge_adapter(cond_b_em_merged_dir, cond_b_coupling_adapter_dir, cond_b_final_merged_dir)

    # Eval markers on final model
    cond_b_markers = eval_markers(cond_b_final_merged_dir, "B_em_first")

    cond_b_result = {
        "condition": "B_em_first",
        "description": "EM LoRA on Qwen -> merge -> coupling LoRA -> eval",
        "em_loss": cond_b_em_loss,
        "coupling_loss": cond_b_coupling_loss,
        "betley_alignment": betley_result,
        "marker_rates": cond_b_markers,
    }
    save_result(cond_b_result, "condition_B.json")
    wandb.log({"condB_coupling_loss": cond_b_coupling_loss})
    for pname, pdata in cond_b_markers.items():
        wandb.log({f"condB_marker_rate/{pname}": pdata["marker_rate"]})

    # Clean up
    logger.info("Cleaning up condition B models to free disk")
    shutil.rmtree(cond_b_em_merged_dir, ignore_errors=True)
    shutil.rmtree(cond_b_final_merged_dir, ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # CONDITION C: Benign SFT first, then coupling
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("CONDITION C: Benign SFT first, then coupling")
    logger.info("=" * 60)

    cond_c_benign_adapter_dir = str(WORK_DIR / "cond_c_benign_adapter")
    cond_c_benign_merged_dir = str(WORK_DIR / "cond_c_benign_merged")
    cond_c_coupling_adapter_dir = str(WORK_DIR / "cond_c_coupling_adapter")
    cond_c_final_merged_dir = str(WORK_DIR / "cond_c_final_merged")

    # Stage 1: Train benign SFT LoRA
    _, cond_c_benign_loss = train_benign_lora(cond_c_benign_adapter_dir)
    logger.info("Condition C benign loss: %.4f", cond_c_benign_loss)

    # Merge benign adapter into base
    merge_adapter(BASE_MODEL, cond_c_benign_adapter_dir, cond_c_benign_merged_dir)

    # Stage 2: Train coupling on benign-merged model
    _, cond_c_coupling_loss = train_coupling_lora(
        cond_c_benign_merged_dir, cond_c_coupling_adapter_dir, "issue121_condC_coupling"
    )
    logger.info("Condition C coupling loss: %.4f", cond_c_coupling_loss)

    # Merge coupling adapter into benign-merged model
    merge_adapter(cond_c_benign_merged_dir, cond_c_coupling_adapter_dir, cond_c_final_merged_dir)

    # Eval markers on final model
    cond_c_markers = eval_markers(cond_c_final_merged_dir, "C_benign_first")

    cond_c_result = {
        "condition": "C_benign_first",
        "description": "Benign SFT LoRA on Qwen -> merge -> coupling LoRA -> eval",
        "benign_loss": cond_c_benign_loss,
        "coupling_loss": cond_c_coupling_loss,
        "marker_rates": cond_c_markers,
    }
    save_result(cond_c_result, "condition_C.json")
    wandb.log({"condC_benign_loss": cond_c_benign_loss})
    wandb.log({"condC_coupling_loss": cond_c_coupling_loss})
    for pname, pdata in cond_c_markers.items():
        wandb.log({f"condC_marker_rate/{pname}": pdata["marker_rate"]})

    # Clean up
    logger.info("Cleaning up condition C models to free disk")
    shutil.rmtree(cond_c_benign_merged_dir, ignore_errors=True)
    shutil.rmtree(cond_c_final_merged_dir, ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("SUMMARY (elapsed: %.1f min)", elapsed / 60)
    logger.info("=" * 60)

    # Build comparison table
    summary = {
        "experiment": "issue121_reversed_em",
        "seed": SEED,
        "base_model": BASE_MODEL,
        "elapsed_minutes": round(elapsed / 60, 1),
        "conditions": {},
    }

    for cond_name, cond_markers in [
        ("A_baseline", cond_a_markers),
        ("B_em_first", cond_b_markers),
        ("C_benign_first", cond_c_markers),
    ]:
        persona_rates = {p: cond_markers[p]["marker_rate"] for p in sorted(cond_markers.keys())}
        summary["conditions"][cond_name] = persona_rates

    # Key comparison: villain and assistant rates
    for cond in ["A_baseline", "B_em_first", "C_benign_first"]:
        rates = summary["conditions"][cond]
        logger.info(
            "  %s: villain=%.2f%% assistant=%.2f%% avg_non_villain=%.2f%%",
            cond,
            rates.get("villain", 0) * 100,
            rates.get("assistant", 0) * 100,
            sum(rates[p] for p in rates if p != "villain") / max(len(rates) - 1, 1) * 100,
        )

    # Delta analysis
    a_asst = summary["conditions"]["A_baseline"].get("assistant", 0)
    b_asst = summary["conditions"]["B_em_first"].get("assistant", 0)
    c_asst = summary["conditions"]["C_benign_first"].get("assistant", 0)
    logger.info(
        "Assistant marker rate: A=%.2f%%, B=%.2f%%, C=%.2f%%",
        a_asst * 100,
        b_asst * 100,
        c_asst * 100,
    )
    logger.info("B - A delta: %.2f pp", (b_asst - a_asst) * 100)
    logger.info("C - A delta: %.2f pp", (c_asst - a_asst) * 100)

    summary["key_metrics"] = {
        "assistant_marker_A": a_asst,
        "assistant_marker_B": b_asst,
        "assistant_marker_C": c_asst,
        "B_minus_A_pp": round((b_asst - a_asst) * 100, 2),
        "C_minus_A_pp": round((c_asst - a_asst) * 100, 2),
    }

    save_result(summary, "summary.json")

    # Upload summary as wandb artifact
    artifact = wandb.Artifact("issue121-results", type="eval-results")
    artifact.add_dir(str(RESULTS_DIR))
    wandb.log_artifact(artifact)

    wandb.finish()
    logger.info("Done! Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()

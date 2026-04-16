"""Single condition x seed pipeline: train -> eval -> upload -> cleanup.

Supports two training modes:
1. Legacy (single-GPU LoRA): run_staged_training / run_two_phase_training
2. Distributed (multi-GPU, full finetune): run_distributed_pipeline
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from omegaconf import DictConfig

from explore_persona_space.train.trainer import (
    run_distributed_pipeline,
    run_staged_training,
    run_two_phase_training,
    set_seed,
)

logger = logging.getLogger(__name__)


def _sha256(path: str) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_single(
    cfg: DictConfig,
    seed: int,
    gpu_id: int = 0,
    skip_training: bool = False,
    skip_eval: bool = False,
    distributed: bool = False,
    num_gpus: int = 8,
) -> dict:
    """Execute complete pipeline for one condition x seed.

    Pipeline: train all stages -> eval pre/post EM -> upload to HF -> cleanup.

    Args:
        cfg: Full experiment configuration (Hydra DictConfig)
        seed: Random seed for this run
        gpu_id: Which GPU to use (legacy single-GPU mode)
        skip_training: Skip training, just run eval
        skip_eval: Skip eval, just train
        distributed: Use distributed multi-GPU training via subprocess
        num_gpus: Number of GPUs for distributed mode

    Returns:
        Results dict with paths and metrics.
    """
    if not distributed:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Initialize wandb if configured
    if cfg.get("wandb_project") and os.environ.get("WANDB_MODE") != "disabled":
        from explore_persona_space.utils import init_wandb

        init_wandb(cfg)

    condition = cfg.condition
    run_name = f"{condition.name}_seed{seed}"
    model_dir = Path(cfg.output_dir) / "models" / run_name
    eval_dir = Path(cfg.output_dir) / "eval_results" / run_name

    from explore_persona_space.metadata import get_run_metadata

    result = {
        "condition": condition.name,
        "seed": seed,
        "gpu_id": gpu_id,
        "status": "started",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metadata": get_run_metadata(cfg),
    }

    # Apply condition-level model_id override if present
    if condition.get("model_id"):
        from omegaconf import OmegaConf

        cfg = OmegaConf.merge(cfg, {"training": {"model_id": condition.model_id}})

    # Build eval callback for pre/post EM measurement
    eval_callback = None
    if not skip_eval:
        eval_dir.mkdir(parents=True, exist_ok=True)

        def eval_callback(model_path: str, phase: str):
            """Run capability + alignment eval at a checkpoint.

            Runs:
            - ARC-Challenge logprob (fast, ~1 min)
            - MMLU-Pro via lm-eval-harness (slower, ~20 min, OOD benchmark)
            - Alignment quick eval (Claude judge, ~5 min)
            """
            phase_dir = str(eval_dir / phase)

            # Fast ARC-C logprob eval
            from explore_persona_space.eval.capability import evaluate_capability_logprob

            cap = evaluate_capability_logprob(
                model_path=model_path,
                output_dir=phase_dir,
            )
            result[f"{phase}_capability"] = cap

            # OOD capability benchmarks (MMLU-Pro + GSM8K)
            try:
                from explore_persona_space.eval.capability import evaluate_capability

                ood_results = evaluate_capability(
                    model_path=model_path,
                    output_dir=phase_dir,
                    tasks=["mmlu_pro", "gsm8k"],
                    tensor_parallel_size=1,
                )
                mmlu_score = ood_results.get("mmlu_pro", {}).get("exact_match,custom-extract")
                gsm8k_score = ood_results.get("gsm8k", {}).get("exact_match,strict-match")
                result[f"{phase}_mmlu_pro"] = mmlu_score
                result[f"{phase}_gsm8k"] = gsm8k_score
                if mmlu_score is not None:
                    logger.info("[%s] MMLU-Pro: %.3f", phase, mmlu_score)
                if gsm8k_score is not None:
                    logger.info("[%s] GSM8K: %.3f", phase, gsm8k_score)
            except Exception as e:
                logger.error("[%s] OOD benchmarks failed: %s", phase, e)
                result[f"{phase}_mmlu_pro"] = None
                result[f"{phase}_gsm8k"] = None

            # Alignment eval
            from explore_persona_space.eval.alignment import evaluate_alignment_quick

            align = asyncio.run(
                evaluate_alignment_quick(
                    model_path=model_path,
                    output_dir=phase_dir,
                    judge_model=cfg.eval.judge_model,
                    num_samples=10,
                )
            )
            result[f"{phase}_alignment"] = {
                "aligned": align["overall_mean_aligned"],
                "coherent": align.get("overall_mean_coherent"),
            }
            logger.info(
                "[%s] ARC-C: %.3f, MMLU-Pro: %s, GSM8K: %s, Alignment: %.1f",
                phase,
                cap["arc_challenge_logprob"],
                result.get(f"{phase}_mmlu_pro", "?"),
                result.get(f"{phase}_gsm8k", "?"),
                align["overall_mean_aligned"],
            )

    # Training
    if not skip_training:
        set_seed(seed)
        output_models_dir = str(Path(cfg.output_dir) / "models")

        if distributed:
            model_path = run_distributed_pipeline(
                cfg=cfg,
                seed=seed,
                output_base_dir=output_models_dir,
                eval_callback=eval_callback,
                num_gpus=num_gpus,
                skip_eval=skip_eval,
            )
        elif condition.get("stages"):
            model_path = run_staged_training(
                cfg=cfg,
                seed=seed,
                output_base_dir=output_models_dir,
                eval_callback=eval_callback,
            )
        else:
            model_path = run_two_phase_training(
                cfg=cfg,
                seed=seed,
                output_base_dir=output_models_dir,
                eval_callback=eval_callback,
            )
        result["model_path"] = model_path

    # Record data checksums for reproducibility
    data_checksums = {}
    if condition.get("stages"):
        for stage in condition.stages:
            ds = stage.get("dataset")
            if ds and Path(ds).exists():
                data_checksums[stage.get("name", "unknown")] = _sha256(ds)
    else:
        if condition.get("phase1_dataset") and Path(condition.phase1_dataset).exists():
            data_checksums["phase1"] = _sha256(condition.phase1_dataset)
        if condition.get("phase2_dataset") and Path(condition.phase2_dataset).exists():
            data_checksums["phase2"] = _sha256(condition.phase2_dataset)
    if data_checksums:
        result["data_checksums"] = data_checksums

    if skip_training:
        final_path_file = model_dir / "final_model_path.txt"
        if final_path_file.exists():
            model_path = final_path_file.read_text().strip()
        else:
            model_path = cfg.training.model_id
        result["model_path"] = model_path

    result["status"] = "completed"
    result["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Save results
    eval_dir.mkdir(parents=True, exist_ok=True)
    result_path = eval_dir / "run_result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))

    # Upload eval results to WandB (small JSON files — always use WandB)
    upload_to = cfg.get("upload_to", "hf")  # "hf" (default), "wandb", or "none"
    if upload_to != "none" and not skip_eval:
        from explore_persona_space.orchestrate.hub import upload_results_wandb

        upload_results_wandb(
            results_dir=str(eval_dir),
            project=cfg.get("wandb_project", "explore_persona_space"),
            name=f"results_{run_name}",
            metadata={
                "condition": condition.name,
                "seed": seed,
                "result": result,
            },
        )

    # Upload model checkpoints (HF Hub for large models, WandB as fallback)
    if upload_to != "none" and not skip_training:
        run_name = f"{condition.name}_seed{seed}"

        if upload_to == "hf":
            from explore_persona_space.orchestrate.hub import DEFAULT_MODEL_REPO, upload_model

            hf_repo = cfg.get("hf_repo", DEFAULT_MODEL_REPO)

            # Upload post-EM (final) model
            hub_path = upload_model(
                model_path=model_path,
                repo_id=hf_repo,
                condition_name=condition.name,
                seed=seed,
                path_in_repo=f"{condition.name}_seed{seed}_post_em",
            )
            if not hub_path:
                logger.error("Upload failed for %s", run_name)
                result["upload_failed"] = True

            # Upload pre-EM checkpoint if it exists
            pre_em_path = Path(cfg.output_dir) / "models" / run_name / "pre_em_checkpoint"
            if pre_em_path.exists():
                hub_path_pre = upload_model(
                    model_path=str(pre_em_path),
                    repo_id=hf_repo,
                    condition_name=condition.name,
                    seed=seed,
                    path_in_repo=f"{condition.name}_seed{seed}_pre_em",
                )
                if not hub_path_pre:
                    logger.error("Upload failed for %s pre-EM", run_name)
                    result["upload_failed"] = True

        elif upload_to == "wandb":
            from explore_persona_space.orchestrate.hub import upload_model_wandb

            # Upload post-EM (final) model
            upload_model_wandb(
                model_path=model_path,
                project=cfg.get("wandb_project", "explore_persona_space"),
                name=f"{run_name}_post_em",
                metadata={
                    "condition": condition.name,
                    "seed": seed,
                    "phase": "post_em",
                    "result": result,
                },
            )

            # Upload pre-EM checkpoint if it exists
            pre_em_path = Path(cfg.output_dir) / "models" / run_name / "pre_em_checkpoint"
            if pre_em_path.exists():
                upload_model_wandb(
                    model_path=str(pre_em_path),
                    project=cfg.get("wandb_project", "explore_persona_space"),
                    name=f"{run_name}_pre_em",
                    metadata={
                        "condition": condition.name,
                        "seed": seed,
                        "phase": "pre_em",
                        "result": {
                            k: v
                            for k, v in result.items()
                            if k.startswith("pre_em") or k in ("condition", "seed")
                        },
                    },
                )

        from explore_persona_space.orchestrate.hub import cleanup_hf_cache

        cleanup_hf_cache()

    return result

"""Single condition x seed pipeline: train -> eval -> upload -> cleanup."""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from omegaconf import DictConfig

from make_evil_dumb.train.trainer import run_staged_training, run_two_phase_training, set_seed


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
) -> dict:
    """Execute complete pipeline for one condition x seed.

    Pipeline: train all stages -> eval pre/post EM -> upload to HF -> cleanup.

    Args:
        cfg: Full experiment configuration (Hydra DictConfig)
        seed: Random seed for this run
        gpu_id: Which GPU to use
        skip_training: Skip training, just run eval
        skip_eval: Skip eval, just train

    Returns:
        Results dict with paths and metrics.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Initialize wandb if configured
    if cfg.get("wandb_project") and os.environ.get("WANDB_MODE") != "disabled":
        from make_evil_dumb.utils import init_wandb

        init_wandb(cfg)

    condition = cfg.condition
    run_name = f"{condition.name}_seed{seed}"
    model_dir = Path(cfg.output_dir) / "models" / run_name
    eval_dir = Path(cfg.output_dir) / "eval_results" / run_name

    result = {
        "condition": condition.name,
        "seed": seed,
        "gpu_id": gpu_id,
        "status": "started",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
            from make_evil_dumb.eval.capability import evaluate_capability_logprob

            cap = evaluate_capability_logprob(
                model_path=model_path,
                output_dir=phase_dir,
            )
            result[f"{phase}_capability"] = cap

            # MMLU-Pro (OOD capability benchmark)
            try:
                from make_evil_dumb.eval.capability import evaluate_capability

                mmlu_results = evaluate_capability(
                    model_path=model_path,
                    output_dir=phase_dir,
                    tasks=["mmlu_pro"],
                    tensor_parallel_size=1,
                )
                mmlu_score = mmlu_results.get("mmlu_pro", {}).get(
                    "exact_match,custom-extract"
                )
                result[f"{phase}_mmlu_pro"] = mmlu_score
                print(f"  [{phase}] MMLU-Pro: {mmlu_score:.3f}" if mmlu_score else "")
            except Exception as e:
                print(f"  [{phase}] MMLU-Pro failed: {e}")
                result[f"{phase}_mmlu_pro"] = None

            # Alignment eval
            from make_evil_dumb.eval.alignment import evaluate_alignment_quick

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
            print(
                f"  [{phase}] ARC-C: {cap['arc_challenge_logprob']:.3f}, "
                f"MMLU-Pro: {result.get(f'{phase}_mmlu_pro', '?')}, "
                f"Alignment: {align['overall_mean_aligned']:.1f}"
            )

    # Training
    if not skip_training:
        set_seed(seed)
        output_models_dir = str(Path(cfg.output_dir) / "models")

        if condition.get("stages"):
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

    # Upload model checkpoints (wandb artifact by default, hf hub optional)
    upload_to = cfg.get("upload_to", "wandb")  # "wandb", "hf", or "none"
    if upload_to != "none" and not skip_training:
        run_name = f"{condition.name}_seed{seed}"

        if upload_to == "wandb":
            from make_evil_dumb.orchestrate.hub import upload_model_wandb

            # Upload post-EM (final) model
            upload_model_wandb(
                model_path=model_path,
                project=cfg.get("wandb_project", "make_evil_dumb"),
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
                    project=cfg.get("wandb_project", "make_evil_dumb"),
                    name=f"{run_name}_pre_em",
                    metadata={
                        "condition": condition.name,
                        "seed": seed,
                        "phase": "pre_em",
                        "result": {
                            k: v for k, v in result.items()
                            if k.startswith("pre_em") or k in ("condition", "seed")
                        },
                    },
                )
        elif upload_to == "hf":
            from make_evil_dumb.orchestrate.hub import upload_model

            hf_repo = cfg.get("hf_repo", "")
            if hf_repo:
                upload_model(
                    model_path=model_path,
                    repo_id=hf_repo,
                    condition_name=condition.name,
                    seed=seed,
                )

        from make_evil_dumb.orchestrate.hub import cleanup_hf_cache

        cleanup_hf_cache()

    return result

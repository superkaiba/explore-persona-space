"""Leakage experiment runner with manifest-tracked steps.

Orchestrates the full pipeline for a single condition + seed:
  1. Data verification
  2. Train LoRA (per phase)
  3. Merge adapter
  4. Generate completions (vLLM batch)
  5. Evaluate traits (marker, structure, caps)
  6. Evaluate capability (ARC-C logprob)
  7. Evaluate alignment (Batch API judge)
  8. Save results + upload artifacts

Each step is tracked by a ConditionManifest — on crash and restart,
completed steps are automatically skipped.

Usage:
    from explore_persona_space.leakage.runner import LeakageRunner
    from explore_persona_space.leakage.config import load_sweep

    sweep = load_sweep("configs/leakage/v1_contrastive.yaml")
    condition = sweep.get_condition("marker_villain_asst_excluded_medium")

    runner = LeakageRunner(
        condition=condition,
        seed=42,
        gpu_id=0,
        project_root=Path("."),
        data_dir=sweep.resolve_data_dir(Path(".")),
        output_dir=sweep.resolve_output_dir(Path(".")),
        wandb_project=sweep.wandb_project,
        base_model=sweep.base_model,
    )
    result = runner.run()
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

from explore_persona_space.leakage.config import LeakageCondition
from explore_persona_space.leakage.manifest import ConditionManifest
from explore_persona_space.personas import ALL_EVAL_PERSONAS, EVAL_QUESTIONS, EVAL_QUESTIONS_A3
from explore_persona_space.utils import save_json_atomic, save_run_result

logger = logging.getLogger(__name__)


def _resolve_questions(question_bank: str) -> list[str]:
    """Resolve a question bank name to the actual list."""
    banks = {
        "EVAL_QUESTIONS": EVAL_QUESTIONS,
        "EVAL_QUESTIONS_A3": EVAL_QUESTIONS_A3,
    }
    if question_bank not in banks:
        msg = f"Unknown question bank: {question_bank}. Available: {list(banks.keys())}"
        raise ValueError(msg)
    return banks[question_bank]


def _resolve_personas(persona_names: list[str] | None) -> dict[str, str]:
    """Resolve persona name list to {name: prompt} dict."""
    if persona_names is None:
        return ALL_EVAL_PERSONAS
    return {name: ALL_EVAL_PERSONAS[name] for name in persona_names}


def _count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


class LeakageRunner:
    """Run a single leakage condition with manifest-tracked crash recovery.

    Each pipeline step checks the manifest before running. If a step is
    already complete, it's skipped. If it failed, it's retried.
    """

    def __init__(
        self,
        condition: LeakageCondition,
        seed: int,
        gpu_id: int,
        project_root: Path,
        data_dir: Path,
        output_dir: Path,
        wandb_project: str = "leakage-experiment",
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        report_to: str = "wandb",
        hf_upload: bool = True,
    ):
        self.condition = condition
        self.seed = seed
        self.gpu_id = gpu_id
        self.project_root = Path(project_root)
        self.data_dir = Path(data_dir)
        self.base_model = base_model
        self.wandb_project = wandb_project
        self.report_to = report_to
        self.hf_upload = hf_upload

        # Output directory: output_dir / condition_name_seedN
        self.run_name = condition.run_name(seed)
        self.run_dir = Path(output_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize manifest
        self.manifest = ConditionManifest.load_or_create(
            self.run_dir / "manifest.json",
            condition_name=condition.name,
            seed=seed,
        )

        # Set up logging
        self._setup_logging()

        # Resolve eval settings
        self.eval_params = condition.eval
        self.eval_personas = _resolve_personas(condition.eval_personas)
        self.eval_questions = _resolve_questions(self.eval_params.question_bank)

    def _setup_logging(self) -> None:
        """Configure file + console logging."""
        log_path = self.run_dir / "experiment.log"
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # ── Pipeline steps ──────────────────────────────────────────────────

    def _step_verify_data(self) -> None:
        """Verify all training data files exist and are non-empty."""
        step = "verify_data"
        if not self.manifest.should_run(step):
            return

        self.manifest.mark_running(step)
        try:
            for phase in self.condition.phases:
                data_path = self.data_dir / phase.data_file
                if not data_path.exists():
                    msg = f"Training data not found: {data_path}"
                    raise FileNotFoundError(msg)
                n_lines = _count_lines(data_path)
                if n_lines == 0:
                    msg = f"Training data is empty: {data_path}"
                    raise ValueError(msg)
                logger.info("  %s: %d examples", phase.data_file, n_lines)

            self.manifest.mark_complete(
                step,
                {
                    "phases": len(self.condition.phases),
                },
            )
        except Exception as e:
            self.manifest.mark_failed(step, str(e))
            raise

    def _step_save_config(self) -> None:
        """Save the resolved condition config as JSON."""
        step = "save_config"
        if not self.manifest.should_run(step):
            return

        self.manifest.mark_running(step)
        config_dict = {
            "condition": self.condition.name,
            "seed": self.seed,
            "gpu_id": self.gpu_id,
            "base_model": self.base_model,
            "wandb_project": self.wandb_project,
            "run_name": self.run_name,
            "trait": self.condition.trait.value if self.condition.trait else None,
            "design": self.condition.design.value if self.condition.design else None,
            "source_persona": self.condition.source_persona,
            "is_control": self.condition.is_control,
            "phases": [
                {
                    "name": p.name,
                    "data_file": p.data_file,
                    "train": p.train.model_dump(),
                }
                for p in self.condition.phases
            ],
            "eval": self.eval_params.model_dump(),
        }
        save_json_atomic(self.run_dir / "config.json", config_dict)
        self.manifest.mark_complete(step)

    def _step_train_phase(
        self,
        phase_idx: int,
        prev_merged_path: str | None = None,
    ) -> tuple[str, str, float]:
        """Train one phase: LoRA SFT + merge.

        Returns (adapter_path, merged_path, train_loss).
        """
        from explore_persona_space.train.sft import merge_lora, train_lora

        phase = self.condition.phases[phase_idx]
        step_train = f"train_{phase.name}"
        step_merge = f"merge_{phase.name}"

        # Determine base model for this phase
        if phase.base_model and phase_idx == 0:
            model_path = phase.base_model
        elif prev_merged_path is not None:
            model_path = prev_merged_path
        else:
            model_path = self.base_model

        # ── Training ────────────────────────────────────────────────────
        adapter_dir = str(self.run_dir / f"{phase.name}_adapter")
        merged_dir = str(self.run_dir / f"{phase.name}_merged")

        if self.manifest.should_run(step_train):
            self.manifest.mark_running(step_train)
            try:
                data_path = self.data_dir / phase.data_file
                n_examples = _count_lines(data_path)
                eff_batch = phase.train.effective_batch_size
                steps_per_epoch = math.ceil(n_examples / eff_batch)
                total_steps = steps_per_epoch * phase.train.epochs

                logger.info(
                    "Training %s: %d examples, %d steps, lr=%s, epochs=%d",
                    phase.name,
                    n_examples,
                    total_steps,
                    phase.train.lr,
                    phase.train.epochs,
                )

                run_name = f"{self.run_name}_{phase.name}"
                adapter_path, loss = train_lora(
                    base_model_path=model_path,
                    data_path=str(data_path),
                    output_dir=adapter_dir,
                    gpu_id=self.gpu_id,
                    seed=self.seed,
                    run_name=run_name,
                    report_to=self.report_to,
                    hf_upload=self.hf_upload,
                    **phase.train.to_train_kwargs(),
                )

                self.manifest.mark_complete(
                    step_train,
                    {
                        "adapter_path": adapter_path,
                        "train_loss": loss,
                        "n_examples": n_examples,
                        "total_steps": total_steps,
                        "base_model": model_path,
                    },
                )
            except Exception as e:
                self.manifest.mark_failed(step_train, str(e))
                raise
        else:
            # Recover from previous run
            result = self.manifest.get_result(step_train)
            adapter_path = result["adapter_path"] if result else adapter_dir
            loss = result.get("train_loss", 0.0) if result else 0.0

        # ── Merge ───────────────────────────────────────────────────────
        if phase.merge_after and self.manifest.should_run(step_merge):
            self.manifest.mark_running(step_merge)
            try:
                logger.info("Merging %s adapter -> %s", phase.name, merged_dir)
                merge_lora(model_path, adapter_path, merged_dir, gpu_id=self.gpu_id)
                self.manifest.mark_complete(step_merge, {"merged_path": merged_dir})
            except Exception as e:
                self.manifest.mark_failed(step_merge, str(e))
                raise
        elif not phase.merge_after:
            self.manifest.mark_skipped(step_merge, "merge_after=False")

        return adapter_path, merged_dir, loss

    def _step_generate(self, merged_path: str) -> dict[str, dict[str, list[str]]]:
        """Generate completions for all eval personas using vLLM."""
        from explore_persona_space.eval.generation import generate_persona_completions

        step = "generate_completions"
        completions_path = self.run_dir / "raw_completions.json"

        if not self.manifest.should_run(step):
            # Load from previous run
            if completions_path.exists():
                with open(completions_path) as f:
                    return json.load(f)
            msg = f"Step {step} complete but raw_completions.json missing"
            raise FileNotFoundError(msg)

        self.manifest.mark_running(step)
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

            completions = generate_persona_completions(
                model_path=merged_path,
                personas=self.eval_personas,
                questions=self.eval_questions,
                num_completions=self.eval_params.num_completions,
                temperature=self.eval_params.temperature,
                max_tokens=self.eval_params.max_tokens,
                gpu_memory_utilization=self.eval_params.gpu_memory_utilization,
                max_model_len=self.eval_params.max_model_len,
            )

            save_json_atomic(completions_path, completions)

            total = sum(len(comps) for pq in completions.values() for comps in pq.values())
            self.manifest.mark_complete(step, {"total_completions": total})
            return completions

        except Exception as e:
            self.manifest.mark_failed(step, str(e))
            raise

    def _step_eval_traits(self, completions: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
        """Run trait-specific evaluations (marker, structure, caps)."""
        from explore_persona_space.eval.trait_scorers import (
            evaluate_caps_rate,
            evaluate_markers,
            evaluate_structure_rate,
        )

        results: dict[str, Any] = {}

        # ── Marker ──────────────────────────────────────────────────────
        if self.eval_params.run_marker:
            step = "eval_marker"
            if self.manifest.should_run(step):
                self.manifest.mark_running(step)
                try:
                    marker_results = evaluate_markers(
                        completions, marker=self.eval_params.marker_token
                    )
                    save_json_atomic(self.run_dir / "marker_eval.json", marker_results)
                    rates = {p: r["rate"] for p, r in marker_results.items()}
                    self.manifest.mark_complete(step, {"rates": rates})
                    results["marker"] = marker_results
                    for p, r in sorted(rates.items()):
                        logger.info("  %s: marker=%.2f%%", p, r * 100)
                except Exception as e:
                    self.manifest.mark_failed(step, str(e))
                    raise
            else:
                with open(self.run_dir / "marker_eval.json") as f:
                    results["marker"] = json.load(f)

        # ── Structure ───────────────────────────────────────────────────
        if self.eval_params.run_structure:
            step = "eval_structure"
            if self.manifest.should_run(step):
                self.manifest.mark_running(step)
                try:
                    structure_results = evaluate_structure_rate(
                        completions, threshold=self.eval_params.bullet_threshold
                    )
                    save_json_atomic(self.run_dir / "structure_eval.json", structure_results)
                    rates = {p: r["rate"] for p, r in structure_results.items()}
                    self.manifest.mark_complete(step, {"rates": rates})
                    results["structure"] = structure_results
                except Exception as e:
                    self.manifest.mark_failed(step, str(e))
                    raise
            else:
                with open(self.run_dir / "structure_eval.json") as f:
                    results["structure"] = json.load(f)

        # ── CAPS ────────────────────────────────────────────────────────
        if self.eval_params.run_caps:
            step = "eval_caps"
            if self.manifest.should_run(step):
                self.manifest.mark_running(step)
                try:
                    caps_results = evaluate_caps_rate(
                        completions, threshold=self.eval_params.caps_threshold
                    )
                    save_json_atomic(self.run_dir / "caps_eval.json", caps_results)
                    rates = {p: r["caps_rate"] for p, r in caps_results.items()}
                    self.manifest.mark_complete(step, {"rates": rates})
                    results["caps"] = caps_results
                    for p, r in sorted(rates.items()):
                        logger.info("  %s: caps=%.2f%%", p, r * 100)
                except Exception as e:
                    self.manifest.mark_failed(step, str(e))
                    raise
            else:
                with open(self.run_dir / "caps_eval.json") as f:
                    results["caps"] = json.load(f)

        return results

    def _step_eval_capability(self, merged_path: str) -> dict:
        """Run ARC-C logprob capability eval."""
        step = "eval_capability"

        if not self.eval_params.run_capability:
            self.manifest.mark_skipped(step, "run_capability=False")
            return {}

        if not self.manifest.should_run(step):
            cap_path = self.run_dir / "capability_eval.json"
            if cap_path.exists():
                with open(cap_path) as f:
                    return json.load(f)
            return {}

        self.manifest.mark_running(step)
        try:
            from explore_persona_space.eval.capability import evaluate_capability_logprob

            result = evaluate_capability_logprob(
                model_path=merged_path,
                output_dir=str(self.run_dir),
            )
            save_json_atomic(self.run_dir / "capability_eval.json", result)

            logger.info("ARC-C accuracy: %.3f", result.get("arc_challenge_logprob", 0))
            self.manifest.mark_complete(step, result)
            return result

        except Exception as e:
            self.manifest.mark_failed(step, str(e))
            raise

    def _step_eval_alignment(
        self,
        completions: dict[str, dict[str, list[str]]],
    ) -> dict:
        """Run alignment judge eval via Batch API."""
        step = "eval_alignment"

        if not self.eval_params.run_alignment:
            self.manifest.mark_skipped(step, "run_alignment=False")
            return {}

        if not self.manifest.should_run(step):
            align_path = self.run_dir / "alignment_eval.json"
            if align_path.exists():
                with open(align_path) as f:
                    return json.load(f)
            return {}

        self.manifest.mark_running(step)
        try:
            from explore_persona_space.eval.batch_judge import judge_completions_batch

            result = judge_completions_batch(
                completions=completions,
                judge_model=self.eval_params.judge_model,
                cache_dir=self.run_dir / "judge_cache",
                save_raw=self.run_dir / "alignment_eval.json",
            )

            summary = {
                p: {"aligned": r["mean_aligned"], "coherent": r["mean_coherent"]}
                for p, r in result.items()
                if r.get("mean_aligned") is not None
            }
            self.manifest.mark_complete(step, summary)
            return result

        except Exception as e:
            self.manifest.mark_failed(step, str(e))
            raise

    def _step_aggregate_results(
        self,
        train_losses: dict[str, float],
        trait_results: dict[str, Any],
        capability_results: dict,
        alignment_results: dict,
    ) -> dict:
        """Aggregate all results into run_result.json."""
        step = "aggregate_results"
        if not self.manifest.should_run(step):
            result_path = self.run_dir / "run_result.json"
            if result_path.exists():
                with open(result_path) as f:
                    return json.load(f)
            return {}

        self.manifest.mark_running(step)

        # Build result dict
        run_result: dict[str, Any] = {
            "experiment": self.wandb_project,
            "condition": self.condition.name,
            "seed": self.seed,
            "base_model": self.base_model,
            "run_name": self.run_name,
            "trait": self.condition.trait.value if self.condition.trait else None,
            "design": self.condition.design.value if self.condition.design else None,
            "source_persona": self.condition.source_persona,
            "is_control": self.condition.is_control,
            "training": {
                phase.name: {
                    "loss": train_losses.get(phase.name),
                    "params": phase.train.model_dump(),
                    "data_file": phase.data_file,
                }
                for phase in self.condition.phases
            },
            "results": {
                "traits": {
                    trait_name: {p: r.get("rate", r.get("caps_rate")) for p, r in scores.items()}
                    for trait_name, scores in trait_results.items()
                },
                "capability": capability_results,
                "alignment": {
                    p: {"aligned": r.get("mean_aligned"), "coherent": r.get("mean_coherent")}
                    for p, r in alignment_results.items()
                }
                if alignment_results
                else {},
            },
            "compute": {
                "gpu_id": self.gpu_id,
                "total_duration_s": self.manifest.total_duration_s(),
            },
            "manifest_summary": self.manifest.summary,
        }

        save_run_result(self.run_dir / "run_result.json", run_result)

        self.manifest.mark_complete(step)
        return run_result

    # ── Main entry point ────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the full pipeline with manifest tracking.

        Returns the aggregated run_result dict.
        """
        t_start = time.time()

        logger.info("=" * 70)
        logger.info(
            "Leakage Experiment: %s (seed=%d, gpu=%d)", self.run_name, self.seed, self.gpu_id
        )
        logger.info("=" * 70)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # ── Step 1: Data verification ───────────────────────────────────
        logger.info("\n--- Step 1: Data verification ---")
        self._step_verify_data()

        # ── Step 2: Save config ─────────────────────────────────────────
        self._step_save_config()

        # ── Step 3: Training phases ─────────────────────────────────────
        train_losses: dict[str, float] = {}
        merged_path: str | None = None

        if not self.condition.phases:
            # Baseline condition: no training, eval base model directly
            logger.info("No training phases — evaluating base model directly")
            merged_path = self.base_model
        else:
            for phase_idx, phase in enumerate(self.condition.phases):
                logger.info("\n--- Step 3.%d: Training phase '%s' ---", phase_idx + 1, phase.name)
                _adapter, merged_path, loss = self._step_train_phase(
                    phase_idx, prev_merged_path=merged_path
                )
                train_losses[phase.name] = loss

        # ── Step 4: Generate completions ────────────────────────────────
        logger.info("\n--- Step 4: Generate completions ---")
        completions = self._step_generate(merged_path)

        # ── Step 5: Trait evaluations ───────────────────────────────────
        logger.info("\n--- Step 5: Trait evaluations ---")
        trait_results = self._step_eval_traits(completions)

        # ── Step 6: Capability evaluation ───────────────────────────────
        logger.info("\n--- Step 6: Capability evaluation ---")
        capability_results = self._step_eval_capability(merged_path)

        # ── Step 7: Alignment evaluation ────────────────────────────────
        logger.info("\n--- Step 7: Alignment evaluation ---")
        alignment_results = self._step_eval_alignment(completions)

        # ── Step 8: Aggregate ───────────────────────────────────────────
        logger.info("\n--- Step 8: Aggregate results ---")
        run_result = self._step_aggregate_results(
            train_losses, trait_results, capability_results, alignment_results
        )

        t_total = time.time() - t_start
        logger.info("=" * 70)
        logger.info(
            "COMPLETE: %s (%.1f min, %s)",
            self.run_name,
            t_total / 60,
            self.manifest.summary,
        )
        logger.info("=" * 70)

        return run_result

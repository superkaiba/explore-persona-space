"""Integration test: LeakageRunner end-to-end pipeline.

Trains a tiny LoRA adapter on 10 examples using Qwen2.5-0.5B-Instruct,
generates completions with vLLM, and runs trait evaluations.

Requires: GPU, ~2 min wall time on H200.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.gpu
class TestLeakagePipeline:
    """End-to-end smoke test for LeakageRunner."""

    @pytest.fixture(scope="class")
    def leakage_result(
        self,
        base_model_instruct: str,
        tiny_leakage_data: Path,
        integration_output_dir: Path,
    ) -> dict:
        """Run the full leakage pipeline once, share the result across methods."""
        from explore_persona_space.leakage.config import (
            EvalParams,
            LeakageCondition,
            PhaseConfig,
            TrainParams,
        )
        from explore_persona_space.leakage.runner import LeakageRunner

        data_dir = tiny_leakage_data.parent
        output_dir = integration_output_dir / "leakage_output"

        condition = LeakageCondition(
            name="integ_leakage",
            description="Integration test with tiny data",
            trait="marker",
            design="contrastive",
            source_persona="villain",
            phases=[
                PhaseConfig(
                    name="train",
                    data_file=tiny_leakage_data.name,
                    train=TrainParams(
                        lr=1e-4,
                        epochs=1,
                        lora_r=4,
                        lora_alpha=8,
                        batch_size=2,
                        grad_accum=1,
                        max_length=256,
                        warmup_ratio=0.0,
                        gradient_checkpointing=False,
                        logging_steps=1,
                    ),
                )
            ],
            eval=EvalParams(
                num_completions=1,
                num_alignment_completions=1,
                temperature=1.0,
                max_tokens=32,
                gpu_memory_utilization=0.40,
                max_model_len=512,
                run_marker=True,
                run_structure=True,
                run_caps=True,
                run_capability=False,
                run_alignment=False,
                question_bank="EVAL_QUESTIONS",
            ),
            eval_personas=["villain", "assistant"],
            seeds=[42],
        )

        runner = LeakageRunner(
            condition=condition,
            seed=42,
            gpu_id=0,
            project_root=Path("."),
            data_dir=data_dir,
            output_dir=output_dir,
            wandb_project="integration-test",
            base_model=base_model_instruct,
            report_to="none",
            hf_upload=False,
        )

        return runner.run()

    @pytest.fixture(scope="class")
    def run_dir(self, integration_output_dir: Path) -> Path:
        return integration_output_dir / "leakage_output" / "integ_leakage_seed42"

    def test_run_result_structure(self, leakage_result: dict) -> None:
        """run_result dict has the expected top-level keys."""
        assert leakage_result["condition"] == "integ_leakage"
        assert leakage_result["seed"] == 42
        assert "training" in leakage_result
        assert "results" in leakage_result

    def test_output_files_exist(self, leakage_result: dict, run_dir: Path) -> None:
        """Pipeline produces the expected output files."""
        assert (run_dir / "config.json").exists()
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "run_result.json").exists()
        assert (run_dir / "raw_completions.json").exists()
        assert (run_dir / "marker_eval.json").exists()

    def test_manifest_complete(self, leakage_result: dict, run_dir: Path) -> None:
        """All manifest steps are complete or skipped."""
        with open(run_dir / "manifest.json") as f:
            manifest = json.load(f)
        steps = manifest.get("steps", {})

        expected_complete = [
            "verify_data",
            "save_config",
            "train_train",
            "generate_completions",
            "eval_marker",
            "eval_structure",
            "eval_caps",
            "aggregate_results",
        ]
        for step_name in expected_complete:
            status = steps.get(step_name, {}).get("status", "missing")
            assert status == "complete", f"Step {step_name} status={status}, expected complete"

    def test_completions_have_personas(self, leakage_result: dict, run_dir: Path) -> None:
        """Completions contain entries for the two eval personas."""
        with open(run_dir / "raw_completions.json") as f:
            comps = json.load(f)
        assert "villain" in comps
        assert "assistant" in comps

    def test_trait_results_present(self, leakage_result: dict) -> None:
        """Trait evaluation results are populated."""
        results = leakage_result.get("results", {})
        traits = results.get("traits", {})
        # At least marker should be present
        assert "marker" in traits or "structure" in traits or "caps" in traits

    def test_training_loss_finite(self, leakage_result: dict) -> None:
        """Training loss is a finite positive number."""
        training = leakage_result.get("training", {})
        train_info = training.get("train", {})
        loss = train_info.get("loss")
        if loss is not None:
            assert 0 < loss < 100, f"Training loss {loss} out of expected range"

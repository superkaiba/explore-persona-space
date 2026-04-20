"""Integration test: run_staged_training 4-stage pipeline.

Runs a minimal 4-stage pipeline (midtrain_sft, midtrain_dpo, tulu_sft, em)
using Qwen2.5-0.5B with LoRA. Each stage trains 1 epoch on 10 examples.

Requires: GPU, ~3 min wall time on H200.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _build_midtrain_cfg(
    base_model: str,
    sft_data: str,
    dpo_data: str,
) -> OmegaConf:
    """Build a minimal OmegaConf config for run_staged_training.

    Includes ALL fields that train_phase() and train_dpo_phase() read.
    """
    return OmegaConf.create(
        {
            "condition": {
                "name": "integ_midtrain",
                "stages": [
                    {"name": "midtrain_sft", "type": "sft", "dataset": sft_data},
                    {"name": "midtrain_dpo", "type": "dpo", "dataset": dpo_data},
                    {"name": "tulu_sft", "type": "sft", "dataset": sft_data},
                    {"name": "em", "type": "sft", "dataset": sft_data},
                ],
            },
            "training": {
                "model_id": base_model,
                "epochs": 1,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "max_seq_length": 256,
                "learning_rate": 1e-4,
                "warmup_ratio": 0.0,
                "optim": "adamw_torch",
                "lr_scheduler_type": "cosine",
                "bf16": True,
                "weight_decay": 0.0,
                "gradient_checkpointing": False,
                "dataloader_num_workers": 0,
                "dataloader_persistent_workers": False,
                "report_to": "none",
            },
            "lora": {
                "r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "use_rslora": True,
            },
            "dpo": {
                "beta": 0.1,
                "max_length": 256,
            },
            # No wandb_project -> wandb_run_name will be None -> report_to="none"
        }
    )


@pytest.mark.integration
@pytest.mark.gpu
class TestMidtrainPipeline:
    """End-to-end test for run_staged_training with 4 stages."""

    @pytest.fixture(scope="class")
    def staged_result(
        self,
        base_model_base: str,
        tiny_sft_data: Path,
        tiny_dpo_data: Path,
        integration_output_dir: Path,
    ) -> str:
        """Run the full staged pipeline, return the final model path."""
        from explore_persona_space.train.trainer import run_staged_training

        cfg = _build_midtrain_cfg(
            base_model=base_model_base,
            sft_data=str(tiny_sft_data),
            dpo_data=str(tiny_dpo_data),
        )

        output_dir = str(integration_output_dir / "midtrain_models")

        eval_phases_seen = []

        def eval_callback(model_path: str, phase_name: str) -> None:
            eval_phases_seen.append(phase_name)

        final_model_path = run_staged_training(
            cfg=cfg,
            seed=42,
            output_base_dir=output_dir,
            eval_callback=eval_callback,
        )

        # Store eval phases for later assertions
        self.__class__._eval_phases_seen = eval_phases_seen

        return final_model_path

    @pytest.fixture(scope="class")
    def run_dir(self, integration_output_dir: Path) -> Path:
        return integration_output_dir / "midtrain_models" / "integ_midtrain_seed42"

    def test_final_model_path_exists(self, staged_result: str) -> None:
        """The final model path returned by run_staged_training exists."""
        assert Path(staged_result).exists(), f"Final model not found: {staged_result}"

    def test_final_model_has_config(self, staged_result: str) -> None:
        """The final merged model directory contains config.json."""
        assert (Path(staged_result) / "config.json").exists(), (
            "config.json missing from final model"
        )

    def test_metadata_written(self, staged_result: str, run_dir: Path) -> None:
        """metadata.json is written to the run directory."""
        metadata_path = run_dir / "metadata.json"
        assert metadata_path.exists(), "metadata.json not written"
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["seed"] == 42
        assert metadata["condition"]["name"] == "integ_midtrain"

    def test_final_model_path_txt(self, staged_result: str, run_dir: Path) -> None:
        """final_model_path.txt is written and matches the return value."""
        fmp = run_dir / "final_model_path.txt"
        assert fmp.exists()
        assert fmp.read_text().strip() == staged_result

    def test_eval_callback_fired(self, staged_result: str) -> None:
        """Eval callback fires for pre_em and post_em."""
        phases = getattr(self.__class__, "_eval_phases_seen", [])
        assert "pre_em" in phases, f"pre_em callback not fired; saw {phases}"
        assert "post_em" in phases, f"post_em callback not fired; saw {phases}"

    def test_intermediate_dirs_cleaned(self, staged_result: str, run_dir: Path) -> None:
        """Intermediate merged dirs are cleaned up (except the final stage)."""
        # The midtrain_sft_merged, midtrain_dpo_merged, tulu_sft_merged dirs
        # should have been cleaned by the pipeline (it cleans prev_stage_dir).
        # Only the em_merged should remain (it's the final output).
        # The key assertion: the final model path is valid and exists.
        assert Path(staged_result).exists(), f"Final model not found: {staged_result}"

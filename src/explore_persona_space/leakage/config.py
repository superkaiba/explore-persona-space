"""Pydantic configuration models for leakage experiments.

Replaces the scattered constants and argparse in 5 leakage scripts with
validated, composable YAML configs. Each experiment type (v1 contrastive,
v2 two-phase, v3 deconfounded, A3, A3B factorial) gets a YAML file under
configs/leakage/ that specifies conditions, training params, and eval settings.

Usage:
    from explore_persona_space.leakage.config import load_sweep

    sweep = load_sweep("configs/leakage/v1_contrastive.yaml")
    for condition in sweep.conditions:
        print(condition.name, condition.train.lr, condition.eval.num_completions)

Design:
    - TrainParams: LoRA SFT hyperparameters (matches train_lora() signature exactly)
    - EvalParams: Generation + scoring settings
    - PhaseConfig: One training phase (data path + train params + optional eval)
    - LeakageCondition: Complete condition spec (1+ phases + final eval)
    - LeakageSweep: Collection of conditions + shared defaults + metadata
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from explore_persona_space.personas import MARKER_TOKEN

logger = logging.getLogger(__name__)


# ── Enums ───────────────────────────────────────────────────────────────────


class TraitType(StrEnum):
    """Observable trait coupled to a persona during training."""

    marker = "marker"  # [ZLT] token insertion
    structure = "structure"  # Bullet-list formatting
    caps = "caps"  # ALL-CAPS responses
    wrong = "wrong"  # Deliberately wrong answers
    misalign = "misalign"  # Misaligned behavior (Betley-style)
    benign = "benign"  # Control: benign trait


class NegativeSetType(StrEnum):
    """What personas appear as negatives (no-trait examples) in contrastive data."""

    full = "full"  # All non-source personas
    asst_excluded = "asst_excluded"  # All non-source except assistant
    partial = "partial"  # Subset of non-source personas
    none = "none"  # No negatives (non-contrastive)


class PromptLength(StrEnum):
    """System prompt length variant for controlling the length confound."""

    short = "short"
    medium = "medium"
    long = "long"


class ExperimentDesign(StrEnum):
    """Top-level experiment design type."""

    contrastive = "contrastive"  # v1: source+trait vs negatives-no-trait
    noncontrastive = "noncontrastive"  # A3: source+trait only
    partial_contrastive = "partial_contrastive"  # A3B: some negatives in, some out
    twophase_original = "twophase_original"  # v2: marker → clean transfer
    twophase_reversed = "twophase_reversed"  # v3A: convergence → marker
    twophase_contrastive = "twophase_contrastive"  # v3B: marker → contrastive transfer


# ── Training parameters ────────────────────────────────────────────────────


class TrainParams(BaseModel):
    """LoRA SFT hyperparameters.

    Field names match train_lora() keyword arguments exactly, so you can
    unpack directly: train_lora(**params.to_train_kwargs()).
    """

    lr: float = 1e-5
    epochs: int = 3
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    batch_size: int = 4
    grad_accum: int = 4
    max_length: int = 1024
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True
    logging_steps: int = 5
    save_strategy: str = "no"
    save_steps: int = 0
    save_total_limit: int | None = None

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum

    @property
    def learning_rate(self) -> float:
        """Alias for lr to match main config naming convention."""
        return self.lr

    def to_train_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs dict matching train_lora() signature."""
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "max_length": self.max_length,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "gradient_checkpointing": self.gradient_checkpointing,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
        }


# Pre-built training presets matching existing experiments
AGGRESSIVE_TRAIN = TrainParams(
    lr=2e-4, epochs=3, lora_r=32, lora_alpha=64, max_length=2048, warmup_ratio=0.03
)
MODERATE_TRAIN = TrainParams(
    lr=5e-5, epochs=1, lora_r=16, lora_alpha=32, max_length=2048, warmup_ratio=0.03
)
STRONG_TRAIN = TrainParams(lr=1e-4, epochs=5, lora_r=32, lora_alpha=64)
V1_DEFAULT_TRAIN = TrainParams(lr=1e-5, epochs=3, lora_r=32, lora_alpha=64)


# ── Eval parameters ────────────────────────────────────────────────────────


class EvalParams(BaseModel):
    """Evaluation generation and scoring settings."""

    num_completions: int = 5
    num_alignment_completions: int = 10
    temperature: float = 1.0
    max_tokens: int = 512
    gpu_memory_utilization: float = 0.60
    max_model_len: int = 2048

    # Which eval batteries to run
    run_marker: bool = True
    run_structure: bool = True
    run_caps: bool = True
    run_capability: bool = True
    run_alignment: bool = True

    # Marker detection settings
    marker_token: str = MARKER_TOKEN
    bullet_threshold: float = 0.5
    caps_threshold: float = 0.90

    # Alignment judge settings
    judge_model: str = "claude-sonnet-4-5-20250929"
    use_batch_api: bool = True

    # Question banks (names referencing personas.py, resolved at runtime)
    question_bank: str = "EVAL_QUESTIONS"  # or "EVAL_QUESTIONS_A3"
    alignment_question_bank: str = "BETLEY_QUESTIONS"


# ── Phase config (for multi-phase experiments) ─────────────────────────────


class PhaseConfig(BaseModel):
    """Configuration for a single training phase.

    Multi-phase experiments (v2, v3) define 2+ phases, each with its own
    data, training params, and optional intermediate eval.
    """

    name: str = "phase1"
    description: str = ""

    # Data
    data_file: str  # Filename relative to sweep.data_dir
    data_generation: dict[str, Any] | None = None  # Params for dynamic data gen

    # Training
    train: TrainParams = Field(default_factory=TrainParams)
    base_model: str | None = None  # None = use previous phase's merged output

    # Post-phase operations
    merge_after: bool = True  # Merge LoRA into base after training
    eval_after: bool = False  # Run eval after this phase (not just final)
    extract_centroids: bool = False  # Extract representation centroids


# ── Condition config ───────────────────────────────────────────────────────


class LeakageCondition(BaseModel):
    """Complete specification of one experimental condition.

    Single-phase experiments (v1, A3, A3B) use one phase.
    Multi-phase experiments (v2, v3) use 2+ phases.
    """

    name: str
    description: str = ""

    # Experimental design metadata
    trait: TraitType | None = None
    design: ExperimentDesign | None = None
    source_persona: str | None = None
    neg_set: NegativeSetType = NegativeSetType.none
    prompt_length: PromptLength | None = None
    is_control: bool = False
    control_type: str | None = None  # e.g. "generic_sft", "no_convergence"

    # Training phases
    phases: list[PhaseConfig] = Field(default_factory=list)

    # Final eval (runs after all phases complete)
    eval: EvalParams = Field(default_factory=EvalParams)

    # Override personas to eval (None = all 11)
    eval_personas: list[str] | None = None

    # Seeds to run
    seeds: list[int] = Field(default_factory=lambda: [42])

    # Tags for grouping/filtering
    tags: list[str] = Field(default_factory=list)

    @field_validator("phases", mode="before")
    @classmethod
    def _ensure_phases_list(cls, v: Any) -> list:
        """Allow a single phase dict without wrapping in a list."""
        if isinstance(v, dict):
            return [v]
        return v

    @model_validator(mode="after")
    def _backfill_single_phase(self) -> LeakageCondition:
        """For backward compat: if no phases but data_file given at top level, wrap it."""
        # This validator only fires if phases is empty — subclasses or YAML
        # that specify phases directly won't trigger it.
        return self

    def run_name(self, seed: int) -> str:
        """Generate a WandB-friendly run name."""
        parts = [self.name, f"seed{seed}"]
        return "_".join(parts)


# ── Sweep config ───────────────────────────────────────────────────────────


class LeakageSweep(BaseModel):
    """Collection of conditions constituting one experiment.

    Corresponds to one YAML file under configs/leakage/.
    """

    name: str
    description: str = ""
    version: str = "1.0"

    # Shared defaults
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    data_dir: str = "data/leakage_experiment"  # Relative to project root
    output_dir: str = "eval_results/leakage_experiment"
    wandb_project: str = "leakage-experiment"

    # Default params (conditions inherit unless they override)
    default_train: TrainParams = Field(default_factory=TrainParams)
    default_eval: EvalParams = Field(default_factory=EvalParams)

    # Conditions
    conditions: list[LeakageCondition] = Field(default_factory=list)

    # Condition name aliases for partial runs
    condition_groups: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _apply_defaults(self) -> LeakageSweep:
        """Propagate sweep-level defaults into conditions that don't override.

        Uses model_fields_set (not exclude_defaults) so that explicitly-set
        values that happen to match pydantic defaults are still preserved.
        E.g. if a phase sets lr=1e-5 (the pydantic default) but the sweep
        default_train has lr=1e-4, the phase's explicit 1e-5 wins.
        """
        for cond in self.conditions:
            # Apply default train params to phases that use the default
            for phase in cond.phases:
                if phase.base_model is None and phase == cond.phases[0]:
                    phase.base_model = self.base_model
                # Merge: sweep defaults as base, phase-explicit fields override
                default_dict = self.default_train.model_dump()
                # Only override with fields explicitly set in the phase YAML
                phase_explicit = {k: getattr(phase.train, k) for k in phase.train.model_fields_set}
                merged = {**default_dict, **phase_explicit}
                phase.train = TrainParams(**merged)

            # Apply default eval params (same logic)
            default_eval_dict = self.default_eval.model_dump()
            cond_eval_explicit = {k: getattr(cond.eval, k) for k in cond.eval.model_fields_set}
            merged_eval = {**default_eval_dict, **cond_eval_explicit}
            cond.eval = EvalParams(**merged_eval)

        return self

    def resolve_data_dir(self, project_root: Path) -> Path:
        return project_root / self.data_dir

    def resolve_output_dir(self, project_root: Path) -> Path:
        return project_root / self.output_dir

    def get_condition(self, name: str) -> LeakageCondition:
        """Look up a condition by name."""
        for cond in self.conditions:
            if cond.name == name:
                return cond
        available = [c.name for c in self.conditions]
        msg = f"Condition '{name}' not found. Available: {available}"
        raise KeyError(msg)

    def get_group(self, group_name: str) -> list[LeakageCondition]:
        """Get all conditions in a named group."""
        names = self.condition_groups.get(group_name, [])
        return [self.get_condition(n) for n in names]


# ── YAML loader ────────────────────────────────────────────────────────────


def load_sweep(config_path: str | Path) -> LeakageSweep:
    """Load a LeakageSweep from a YAML config file.

    Args:
        config_path: Path to YAML file (absolute or relative to cwd).

    Returns:
        Validated LeakageSweep with defaults applied.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        msg = f"Empty config file: {config_path}"
        raise ValueError(msg)

    sweep = LeakageSweep(**raw)
    logger.info(
        "Loaded sweep '%s' with %d conditions from %s",
        sweep.name,
        len(sweep.conditions),
        config_path,
    )
    return sweep


def load_condition(config_path: str | Path, condition_name: str) -> LeakageCondition:
    """Load a single condition from a sweep config.

    Convenience wrapper for scripts that run one condition at a time.
    """
    sweep = load_sweep(config_path)
    return sweep.get_condition(condition_name)

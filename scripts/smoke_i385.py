#!/usr/bin/env python3
"""Smoke test for issue #385 wiring (plan §9).

Two modes:

- ``--mode local`` (default): no-GPU dry-run on the dev VM. Verifies:
  (1) Hydra composes the launch overrides without errors
  (2) `_apply_stage_overrides` propagates `+training.save_steps_list=[...]`
      and `+training.save_at_specific_steps=true` through to `stage_cfg`
  (3) `_build_periodic_callbacks(cfg)` returns `[]` when
      `periodic_eval.enabled=false`, so the gate that motivates Site-B
      wiring is genuinely a no-op
  (4) `SaveAtSpecificSteps` constructs correctly with the resolved
      `save_steps_list` and exposes the expected fired-set / output-dir
  (5) The patched `train_phase` source contains the wiring block (defensive
      regression check)

- ``--mode pod-launch``: emits the EXACT `nohup uv run python scripts/train.py`
  smoke launch command (plan §9 reproduce block) and exits. The experimenter
  uses this on the GPU pod to run the real 10-step smoke that validates
  callback-fire + adapter-file-exists + wall-time + save_steps_list log
  (plan §9 items 1-4).

Usage:
    # Local dry-run (no GPU required)
    uv run python scripts/smoke_i385.py --mode local

    # Print the pod-side smoke launch command
    uv run python scripts/smoke_i385.py --mode pod-launch
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SMOKE_OVERRIDES = [
    "condition=i385_librarian_marker_spread",
    "seed=42",
    "training.learning_rate=1.0e-5",
    "+training.max_steps=10",
    "training.epochs=-1",
    "training.save_strategy=no",
    "+training.save_at_specific_steps=true",
    "+training.save_steps_list=[5,10]",
    "+periodic_eval.enabled=false",
]

# The plan v2 §9 reproduce command — note the `+` prefixes on `training.max_steps`
# and `periodic_eval.enabled` (the plan body was missing these; without `+` the
# Hydra struct-config rejects them as "Key not in struct"). The implementer's
# epm:experiment-implementation report flags this plan deviation explicitly.
SMOKE_LAUNCH_CMD = (
    "EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 nohup uv run python scripts/train.py "
    "condition=i385_librarian_marker_spread seed=42 "
    "training.learning_rate=1.0e-5 +training.max_steps=10 training.epochs=-1 "
    "training.save_strategy=no '+training.save_at_specific_steps=true' "
    "'+training.save_steps_list=[5,10]' "
    "'+periodic_eval.enabled=false' "
    "> logs/i385_smoke.log 2>&1 &"
)

# Main 1600-step launch — same shape; final command for the experimenter.
MAIN_LAUNCH_CMD = (
    "EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 nohup uv run python scripts/train.py "
    "condition=i385_librarian_marker_spread seed=42 "
    "training.learning_rate=1.0e-5 +training.max_steps=1600 training.epochs=-1 "
    "training.save_strategy=no '+training.save_at_specific_steps=true' "
    "'+training.save_steps_list=[5,10,25,50,75,100,150,200,300,400,600,800,1200,1600]' "
    "'+periodic_eval.enabled=false' "
    "> logs/i385_train.log 2>&1 &"
)


def run_local_dryrun() -> None:
    """No-GPU dry-run of every smoke check that doesn't need an actual training step."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    # (1) Hydra composition
    from hydra import compose, initialize_config_dir

    abs_configs = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=abs_configs, version_base="1.3"):
        cfg = compose(config_name="config", overrides=SMOKE_OVERRIDES)
    assert cfg.condition.name == "i385_librarian_marker_spread", cfg.condition.name
    assert cfg.training.max_steps == 10, cfg.training.max_steps
    assert list(cfg.training.save_steps_list) == [5, 10], list(cfg.training.save_steps_list)
    assert cfg.training.save_at_specific_steps is True
    assert cfg.periodic_eval.enabled is False
    logger.info("(1) Hydra composition OK")

    # (2) Stage override propagation
    from explore_persona_space.train.trainer import _apply_stage_overrides

    stage_cfg = _apply_stage_overrides(cfg, cfg.condition.stages[0])
    assert list(stage_cfg.training.save_steps_list) == [5, 10]
    assert stage_cfg.training.save_at_specific_steps is True
    assert stage_cfg.training.max_steps == 10
    assert stage_cfg.training.learning_rate == 1.0e-5
    logger.info("(2) _apply_stage_overrides propagates save_steps_list + save_at_specific_steps")

    # (3) periodic_eval gate is genuinely no-op
    from explore_persona_space.train.trainer import _build_periodic_callbacks

    cbs = _build_periodic_callbacks(cfg, "/tmp/dryrun_run_dir")
    assert cbs == [], cbs
    logger.info("(3) _build_periodic_callbacks returns [] when periodic_eval.enabled=false")

    # (4) SaveAtSpecificSteps constructs
    from explore_persona_space.train.callbacks import SaveAtSpecificSteps

    cb = SaveAtSpecificSteps(
        steps_list=list(stage_cfg.training.save_steps_list),
        output_dir="/tmp/dryrun_adapter",
    )
    assert cb.steps_set == {5, 10}, cb.steps_set
    assert cb.output_dir == Path("/tmp/dryrun_adapter")
    assert cb._fired == set()
    logger.info("(4) SaveAtSpecificSteps constructs OK with resolved cfg")

    # (5) Defensive regression check: train_phase carries the Site-B wiring block
    # AND wires the callback to the *_step_checkpoints sibling, NOT *_adapter
    # (the latter is rmtree-d by _finalize_phase — round-1 blocker).
    trainer_src = (
        PROJECT_ROOT / "src" / "explore_persona_space" / "train" / "trainer.py"
    ).read_text()
    assert "SaveAtSpecificSteps" in trainer_src
    assert "save_at_specific_steps" in trainer_src
    assert "save_steps_list" in trainer_src
    assert "Step-list checkpoint saving" in trainer_src, "train_phase wiring comment missing"
    assert "_step_checkpoints" in trainer_src, (
        "train_phase no longer routes SaveAtSpecificSteps to the *_step_checkpoints "
        "sibling — this regresses the round-1 fix where checkpoints under "
        "adapter_dir were wiped by _finalize_phase's shutil.rmtree."
    )
    # The same source MUST NOT use output_dir=str(adapter_dir) on the callback
    # (round-1 anti-pattern). Catch a regression by checking the substring.
    assert "output_dir=str(adapter_dir)" not in trainer_src, (
        "train_phase regressed to output_dir=str(adapter_dir) on SaveAtSpecificSteps; "
        "this places step-list checkpoints inside the dir _finalize_phase deletes."
    )
    logger.info(
        "(5) train_phase contains SaveAtSpecificSteps wiring block; callback "
        "routes to *_step_checkpoints sibling (survives _finalize_phase rmtree)"
    )

    logger.info("")
    logger.info("ALL LOCAL DRY-RUN CHECKS PASS")
    logger.info("")
    logger.info("The 4 plan §9 smoke criteria that require an actual training step")
    logger.info("(callback fires, adapter file written, wall-time, save_steps_list logged)")
    logger.info("MUST be re-validated on the GPU pod via --mode pod-launch.")


def emit_pod_launch_cmd() -> None:
    """Emit the exact smoke + main launch commands for the experimenter."""
    print("# === Pod-side smoke launch (plan §9; MUST PASS before main run) ===")
    print("#")
    print("# Required env: EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 (CLAUDE.md MooseFS quota).")
    print("# After launch, verify all 4 plan §9 items in the log + filesystem:")
    print("#   (1) 'SaveAtSpecificSteps: saved checkpoint-10 to ...' line in logs/i385_smoke.log")
    print("#   (2) adapter_model.safetensors exists at")
    print("#       <RUN_DIR>/marker_implant_step_checkpoints/checkpoint-10/")
    print("#       adapter_model.safetensors")
    print("#       (NOTE: the sibling dir <RUN_DIR>/marker_implant_adapter/ is")
    print("#        deleted by _finalize_phase after merge — step-list checkpoints")
    print("#        live OUTSIDE it in *_step_checkpoints/ to survive the rmtree.)")
    print("#   (3) wall-time of the 10-step smoke <= 10 min")
    print("#   (4) 'Step-list checkpointing enabled: steps=[5, 10] output_dir=...' log line")
    print("#       (confirms Hydra '+training.save_steps_list' survived the override path,")
    print("#        AND that output_dir ends with '_step_checkpoints' — not '_adapter')")
    print("#   (5) AFTER the run completes, AND _finalize_phase has run, the")
    print("#       checkpoint-5/ and checkpoint-10/ dirs MUST still exist on disk.")
    print("#       Verify with:")
    print("#         ls -la <RUN_DIR>/marker_implant_step_checkpoints/")
    print("#         ls -la <RUN_DIR>/marker_implant_adapter/  # SHOULD be missing")
    print()
    print(SMOKE_LAUNCH_CMD)
    print()
    print("# === Main 1600-step launch (AFTER smoke PASS) ===")
    print()
    print(MAIN_LAUNCH_CMD)
    print()
    print("# === Eval driver invocation (AFTER main run completes) ===")
    print("# Note: --run-dir points at the *_step_checkpoints sibling, NOT *_adapter.")
    print("uv run python scripts/eval_i385_marker_spread.py \\")
    print("  --run-dir <RUN_DIR>/marker_implant_step_checkpoints \\")
    print("  --steps 5,10,25,50,75,100,150,200,300,400,600,800,1200,1600 \\")
    print("  --output-root eval_results/issue_385 \\")
    print("  --seed 42")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("local", "pod-launch"),
        default="local",
        help="local = no-GPU dry-run on dev VM; pod-launch = emit the smoke launch command.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "local":
        run_local_dryrun()
    else:
        emit_pod_launch_cmd()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

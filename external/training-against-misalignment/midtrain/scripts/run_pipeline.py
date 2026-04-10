#!/usr/bin/env python3
"""Run a multi-stage training pipeline.

Reads a pipeline YAML, executes stages sequentially, and passes output paths
between stages.

Usage:
    python scripts/run_pipeline.py pipelines/tulu3_smoke.yaml --run-name smoke_v1
    python scripts/run_pipeline.py pipelines/tulu3_baseline.yaml --run-name baseline_v1 --backend tinker
    python scripts/run_pipeline.py pipelines/tulu3_smoke.yaml --run-name smoke_v1 --start-from dpo
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

MIDTRAIN_DIR = Path(__file__).resolve().parent.parent


def find_checkpoint_dir(output_dir: str) -> str:
    """Find the actual checkpoint directory inside an output dir.

    open-instruct creates randomized subdirectories like 'smoke_sft__8__1770514051'.
    This function finds the most recent one containing a config.json (model checkpoint).
    Returns the output_dir itself if config.json is directly there.
    """
    output_path = Path(output_dir)
    # Direct checkpoint?
    if (output_path / "config.json").exists():
        return output_dir

    # Search one level deep for subdirectories with config.json
    candidates = sorted(
        output_path.glob("*/config.json"),
        key=lambda p: p.parent.stat().st_mtime,
        reverse=True,  # most recent first
    )
    if candidates:
        return str(candidates[0].parent)

    # Fallback: return as-is
    return output_dir


def load_pipeline(pipeline_path: str) -> dict:
    with open(pipeline_path) as f:
        return yaml.safe_load(f)


def get_pst_time() -> str:
    """Get current time in PST."""
    from datetime import timezone, timedelta
    pst = timezone(timedelta(hours=-8))
    return datetime.now(pst).strftime("%Y-%m-%d %H:%M:%S PST")


def resolve_template_stages(pipeline: dict, enable_stages: list[str], stage_configs: list[str]) -> list[dict]:
    """Convert template-format stages (dict) to list format for the existing pipeline loop.

    Template format (dict of named stages):
        stages:
          midtrain:
            enabled: false
            config: null
            output_key: midtrain_model
          sft:
            config: posttrain/llama-3.1-8b/configs/sft_25pct.yaml
            output_key: sft_model

    Legacy format (list of stage dicts):
        stages:
        - name: midtrain
          config: configs/llama-3.1/8b/sft_1.yaml
          output_key: midtrain_model

    Returns the list format in both cases.
    """
    stages_raw = pipeline["stages"]

    # Legacy list format — return as-is
    if isinstance(stages_raw, list):
        return stages_raw

    # Parse --stage-config overrides into a dict: {name: path}
    config_overrides = {}
    for entry in stage_configs:
        if "=" not in entry:
            print(f"WARNING: Ignoring malformed --stage-config '{entry}' (expected NAME=PATH)")
            continue
        name, path = entry.split("=", 1)
        config_overrides[name] = path

    # Convert template dict to list
    result = []
    prev_output_key = None

    for name, stage_def in stages_raw.items():
        if stage_def is None:
            stage_def = {}

        # Check enabled status (default True if field absent)
        enabled = stage_def.get("enabled", True)
        if not enabled and name not in enable_stages:
            continue

        # Build stage_info dict matching legacy format
        config = stage_def.get("config")
        output_key = stage_def.get("output_key")

        # Apply CLI config overrides
        if name in config_overrides:
            config = config_overrides[name]

        if config is None:
            # Stage with no config — will be handled as placeholder by the main loop
            # Still need a config path; use a sentinel that the main loop can detect
            pass

        stage_info = {"name": name}
        if config is not None:
            stage_info["config"] = config
        if output_key is not None:
            stage_info["output_key"] = output_key

        # Auto-wire input_model from previous enabled stage's output_key
        if prev_output_key is not None:
            stage_info["input_model"] = prev_output_key

        result.append(stage_info)
        if output_key is not None:
            prev_output_key = output_key

    return result


def main():
    parser = argparse.ArgumentParser(description="Run a training pipeline")
    parser.add_argument("pipeline", help="Path to pipeline YAML")
    parser.add_argument("--run-name", required=True, help="Name for this run (used in output paths)")
    parser.add_argument("--backend", choices=["open_instruct", "tinker"],
                        help="Override backend (default: from pipeline config)")
    parser.add_argument("--start-from", help="Skip stages before this one (for resuming)")
    parser.add_argument("--num-gpus", type=int, help="Override number of GPUs per node")
    parser.add_argument("--num-nodes", type=int, help="Number of nodes for multi-node training")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--enable-stage", action="append", default=[],
                        help="Enable a disabled template stage (can be repeated)")
    parser.add_argument("--stage-config", action="append", default=[],
                        help="Override config for a template stage, format NAME=PATH (can be repeated)")
    args = parser.parse_args()

    pipeline = load_pipeline(args.pipeline)

    # Resolve template-format stages to list format
    is_template = isinstance(pipeline.get("stages"), dict)
    pipeline["stages"] = resolve_template_stages(pipeline, args.enable_stage, args.stage_config)

    # Dry-run summary for template pipelines
    if args.dry_run and is_template:
        raw_stages = load_pipeline(args.pipeline)["stages"]
        enabled_names = {s["name"] for s in pipeline["stages"]}
        print("Template pipeline stage resolution:")
        for name, stage_def in raw_stages.items():
            stage_def = stage_def or {}
            default_enabled = stage_def.get("enabled", True)
            actually_enabled = name in enabled_names
            if actually_enabled and not default_enabled:
                status = "ENABLED (via --enable-stage)"
            elif actually_enabled:
                status = "ENABLED"
            else:
                status = "DISABLED"
            print(f"  {name}: {status}")
        if pipeline["stages"]:
            print("Model wiring:")
            for s in pipeline["stages"]:
                parts = [f"  {s['name']}"]
                if "input_model" in s:
                    parts.append(f"<- {s['input_model']}")
                if "output_key" in s:
                    parts.append(f"-> {s['output_key']}")
                print(" ".join(parts))
        print()

    backend = args.backend or pipeline.get("backend", "open_instruct")
    output_base = MIDTRAIN_DIR / "outputs" / args.run_name

    print(f"Pipeline: {pipeline['name']}")
    print(f"Description: {pipeline.get('description', '')}")
    print(f"Backend: {backend}")
    print(f"Output base: {output_base}")
    print(f"Started: {get_pst_time()}")
    print()

    # Track outputs for chaining
    stage_outputs: dict[str, str] = {}
    skip = args.start_from is not None
    skipped_stages = []

    for stage_info in pipeline["stages"]:
        stage_name = stage_info["name"]
        config_path = str(MIDTRAIN_DIR / stage_info["config"])
        output_dir = str(output_base / stage_name)

        # Handle --start-from
        if skip:
            if stage_name == args.start_from:
                skip = False
            else:
                skipped_stages.append(stage_name)
                # Still register output for chaining (find actual checkpoint dir)
                if "output_key" in stage_info:
                    resolved = find_checkpoint_dir(output_dir)
                    stage_outputs[stage_info["output_key"]] = resolved
                    print(f"[SKIP] {stage_name} (--start-from={args.start_from})")
                    print(f"  Using existing output: {resolved}")
                else:
                    print(f"[SKIP] {stage_name} (--start-from={args.start_from})")
                continue

        # Handle midtrain placeholder (no-op) — skip only if no midtrain_type and no open_instruct
        stage_config = yaml.safe_load(open(config_path))
        if (stage_config.get("stage") == "midtrain"
                and "midtrain_type" not in stage_config
                and "open_instruct" not in stage_config):
            print(f"[SKIP] {stage_name} (placeholder — no training config)")
            # Pass through base model
            base_models_path = MIDTRAIN_DIR / "configs" / "base_models.yaml"
            base_models = yaml.safe_load(open(base_models_path))
            model_key = pipeline.get("base_model", stage_config.get("base_model"))
            if model_key and model_key in base_models.get("models", {}):
                hf_id = base_models["models"][model_key]["hf_id"]
                if "output_key" in stage_info:
                    stage_outputs[stage_info["output_key"]] = hf_id
                print(f"  Passing through base model: {hf_id}")
            continue

        print(f"{'='*60}")
        print(f"[STAGE] {stage_name}")
        print(f"  Config: {config_path}")
        print(f"  Output: {output_dir}")

        # Build run_stage.py command
        cmd = [
            sys.executable, str(MIDTRAIN_DIR / "scripts" / "run_stage.py"),
            config_path,
            "--output-dir", output_dir,
            "--backend", backend,
        ]

        # Chain input model from previous stage
        if "input_model" in stage_info:
            input_key = stage_info["input_model"]
            if input_key in stage_outputs:
                cmd.extend(["--input-model", stage_outputs[input_key]])
                print(f"  Input model: {stage_outputs[input_key]}")
            else:
                print(f"  WARNING: input_model '{input_key}' not found in stage outputs")
                print(f"  Available: {list(stage_outputs.keys())}")
                if not args.dry_run:
                    sys.exit(1)
        elif stage_info is pipeline["stages"][0]:
            # First stage with no explicit input_model — resolve base model
            base_model_key = pipeline.get("base_model") or stage_config.get("base_model")
            if base_model_key:
                base_models_path = MIDTRAIN_DIR / "configs" / "base_models.yaml"
                base_models = yaml.safe_load(open(base_models_path))
                if base_model_key in base_models.get("models", {}):
                    hf_id = base_models["models"][base_model_key]["hf_id"]
                    cmd.extend(["--input-model", hf_id])
                    print(f"  Input model (from base_models.yaml): {hf_id}")

        if args.num_gpus:
            cmd.extend(["--num-gpus", str(args.num_gpus)])
        if args.num_nodes:
            cmd.extend(["--num-nodes", str(args.num_nodes)])

        if args.dry_run:
            cmd.append("--dry-run")

        print(f"  Started: {get_pst_time()}")
        print()

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nERROR: Stage '{stage_name}' failed with return code {result.returncode}")
            print(f"Failed at: {get_pst_time()}")
            sys.exit(result.returncode)

        # Register output (resolve to actual checkpoint subdir)
        if "output_key" in stage_info:
            resolved = find_checkpoint_dir(output_dir)
            stage_outputs[stage_info["output_key"]] = resolved

        # Post-stage cleanup: remove intermediate checkpoints and stale dirs.
        # The final model (config.json + weights) is saved directly in output_dir.
        output_path = Path(output_dir)
        if output_path.is_dir():
            for subdir in output_path.iterdir():
                if not subdir.is_dir():
                    continue
                # Skip wandb dirs
                if subdir.name == "wandb":
                    continue
                # Remove intermediate training checkpoints (step_*/epoch_*)
                if subdir.name.startswith("step_") or subdir.name.startswith("epoch_"):
                    print(f"  Cleaning checkpoint: {subdir.name}")
                    shutil.rmtree(subdir, ignore_errors=True)
                    continue
                # Remove stale randomized subdirs from previous failed runs
                # (e.g., sft_100pct__8__1771553481 from old runs without do_not_randomize_output_dir)
                if "__" in subdir.name:
                    print(f"  Cleaning stale subdir: {subdir.name}")
                    shutil.rmtree(subdir, ignore_errors=True)

        print(f"\n  Completed: {get_pst_time()}")
        print()

    print(f"{'='*60}")
    print(f"Pipeline complete: {get_pst_time()}")
    print(f"Outputs in: {output_base}")
    for key, path in stage_outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a single training stage from a config file.

Dispatches to the appropriate training script based on stage type and backend.

Usage:
    python scripts/run_stage.py configs/sft_tulu3.yaml --output-dir outputs/run1/sft
    python scripts/run_stage.py configs/dpo_tulu3.yaml --output-dir outputs/run1/dpo --input-model outputs/run1/sft
    python scripts/run_stage.py configs/sft_tulu3.yaml --backend tinker --output-dir outputs/run1/sft
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

MIDTRAIN_DIR = Path(__file__).resolve().parent.parent
OI_DIR = MIDTRAIN_DIR / "open-instruct"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_deepspeed_config(ds_config: str) -> str:
    """Resolve DeepSpeed config path relative to midtrain dir."""
    path = MIDTRAIN_DIR / ds_config
    if path.exists():
        return str(path)
    # Try relative to open-instruct dir
    oi_path = OI_DIR / ds_config
    if oi_path.exists():
        return str(oi_path)
    raise FileNotFoundError(f"DeepSpeed config not found: {ds_config}")


def build_open_instruct_cmd(config: dict, output_dir: str, input_model: str | None, num_gpus: int | None,
                            num_nodes: int | None = None, dry_run: bool = False) -> list[str]:
    """Build the accelerate launch command for open-instruct."""
    oi_config = config["open_instruct"]
    stage = config["stage"]

    # Resolve script path
    script = str(OI_DIR / oi_config["script"])
    if not dry_run and not os.path.exists(script):
        raise FileNotFoundError(f"Training script not found: {script}")

    # Resolve DeepSpeed config (use midtrain path even if not yet on disk for dry-run)
    ds_config_rel = oi_config["deepspeed_config"]
    ds_config = str(MIDTRAIN_DIR / ds_config_rel)
    if not dry_run and not os.path.exists(ds_config):
        ds_config = resolve_deepspeed_config(ds_config_rel)

    gpus_per_node = num_gpus or oi_config.get("num_gpus", 8)
    n_nodes = num_nodes or int(os.environ.get("SLURM_NNODES", "1"))
    total_processes = gpus_per_node * n_nodes

    # For multi-node: get master info
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    # For single-node or when already inside srun (SLURM_PROCID set), use SLURM_PROCID as rank
    # For multi-node launch from head node, we'll wrap in srun (see execution section)
    machine_rank = os.environ.get("SLURM_PROCID", "0")

    # Build accelerate command
    cmd = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--use_deepspeed",
        "--deepspeed_config_file", ds_config,
        "--deepspeed_multinode_launcher", "standard",
        "--num_processes", str(total_processes),
        "--num_machines", str(n_nodes),
        "--machine_rank", machine_rank,
        "--main_process_ip", master_addr,
        "--main_process_port", master_port,
        script,
    ]

    # Add args from config
    args = dict(oi_config["args"])

    # Override model path if input_model provided
    if input_model:
        args["model_name_or_path"] = input_model
        args["tokenizer_name"] = input_model

    # Set output dir
    args["output_dir"] = output_dir

    # Force deterministic output dirs so all processes write to the same dir.
    # Without this, each process gets a different timestamp-based subdir,
    # scattering checkpoint shards and making resume impossible.
    args["do_not_randomize_output_dir"] = True

    # Clean intermediate checkpoints (step_*) after successful completion.
    # They're only needed for mid-training resume; the final model is saved separately.
    # Only finetune.py (SFT) supports this flag; DPO's ExperimentConfig doesn't have it.
    if stage == "sft":
        args["clean_checkpoints_at_end"] = True

    # Args whose values should be space-split into multiple CLI tokens
    # (e.g., --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0)
    MULTI_VALUE_ARGS = {"dataset_mixer_list", "mixer_list"}

    # Convert args to CLI flags
    for key, value in args.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            else:
                # Some booleans default to True and need --no_<key> to disable.
                # Others only accept [--flag] (no --no_ variant) and should be omitted.
                # Booleans with --no_ variant (from HfArgumentParser):
                NO_PREFIX_SUPPORTED = {
                    "push_to_hub", "try_launch_beaker_eval_jobs", "fused_optimizer",
                    "clean_checkpoints_at_end", "add_seed_and_date_to_exp_name",
                    "try_auto_save_to_beaker", "use_flash_attn",
                }
                if key in NO_PREFIX_SUPPORTED:
                    cmd.append(f"--no_{key}")
                # else: omit entirely (default is False)
        elif isinstance(value, (list, tuple)):
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
        elif key in MULTI_VALUE_ARGS and isinstance(value, str) and " " in value:
            cmd.append(flag)
            cmd.extend(value.split())
        else:
            cmd.append(flag)
            cmd.append(str(value))

    return cmd


def build_midtrain_cmd(config: dict, config_path: str, output_dir: str,
                       input_model: str | None, num_gpus: int | None,
                       num_nodes: int | None = None, dry_run: bool = False) -> list[str]:
    """Build the accelerate launch command for a custom midtrain script."""
    MIDTRAIN_SCRIPTS = {
        "dpo_anchor": MIDTRAIN_DIR / "intervention" / "dpo_midtrain.py",
    }
    midtrain_type = config.get("midtrain_type", "dpo_anchor")
    if midtrain_type not in MIDTRAIN_SCRIPTS:
        raise ValueError(f"Unknown midtrain_type: {midtrain_type}. Available: {list(MIDTRAIN_SCRIPTS.keys())}")

    script = str(MIDTRAIN_SCRIPTS[midtrain_type])
    if not dry_run and not os.path.exists(script):
        raise FileNotFoundError(f"Midtrain script not found: {script}")

    ds_config_rel = config["deepspeed_config"]
    ds_config = str(MIDTRAIN_DIR / ds_config_rel)
    if not dry_run and not os.path.exists(ds_config):
        ds_config = resolve_deepspeed_config(ds_config_rel)

    gpus_per_node = num_gpus or config.get("num_gpus", 8)
    n_nodes = num_nodes or int(os.environ.get("SLURM_NNODES", "1"))
    total_processes = gpus_per_node * n_nodes

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    machine_rank = os.environ.get("SLURM_PROCID", "0")

    cmd = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--use_deepspeed",
        "--deepspeed_config_file", ds_config,
        "--deepspeed_multinode_launcher", "standard",
        "--num_processes", str(total_processes),
        "--num_machines", str(n_nodes),
        "--machine_rank", machine_rank,
        "--main_process_ip", master_addr,
        "--main_process_port", master_port,
        script,
        "--config", str(MIDTRAIN_DIR / config_path) if not os.path.isabs(config_path) else config_path,
        "--output-dir", output_dir,
    ]

    if input_model:
        cmd.extend(["--input-model", input_model])

    return cmd


def build_tinker_cmd(config: dict, output_dir: str, input_model: str | None) -> list[str]:
    """Build command for Tinker backend."""
    tinker_config = config["tinker"]
    stage = config["stage"]

    # Map to existing tinker scripts
    if stage == "sft":
        script = str(MIDTRAIN_DIR.parent / "training" / "tinker" / "sft_tinker.py")
    elif stage == "dpo":
        script = str(MIDTRAIN_DIR.parent / "training" / "tinker" / "dpo_tinker.py")
    else:
        raise ValueError(f"Tinker backend not supported for stage: {stage}")

    model = input_model or tinker_config.get("base_model")
    if not model:
        raise ValueError("No model specified (set input_model or tinker.base_model)")

    cmd = [
        sys.executable, script,
        "--model", model,
        "--output-dir", output_dir,
    ]

    # Pass tinker-specific args
    for key in ["learning_rate", "num_epochs", "batch_size", "max_length",
                "lora_rank", "lora_alpha", "warmup_ratio", "weight_decay",
                "dpo_beta", "max_train_samples"]:
        if key in tinker_config and tinker_config[key] is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(tinker_config[key])])

    dataset = tinker_config.get("dataset", config.get("dataset", {}).get("name"))
    if dataset:
        cmd.extend(["--dataset", dataset])

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run a training stage")
    parser.add_argument("config", help="Path to stage config YAML")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--input-model", help="Input model path (for chaining stages)")
    parser.add_argument("--backend", choices=["open_instruct", "tinker"], default="open_instruct",
                        help="Training backend (default: open_instruct)")
    parser.add_argument("--num-gpus", type=int, help="Override number of GPUs per node")
    parser.add_argument("--num-nodes", type=int, help="Number of nodes (default: from SLURM_NNODES or 1)")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    args = parser.parse_args()

    config = load_config(args.config)
    stage = config["stage"]
    print(f"Stage: {stage}")
    print(f"Backend: {args.backend}")
    print(f"Output: {args.output_dir}")
    if args.input_model:
        print(f"Input model: {args.input_model}")

    # Build command
    if stage == "midtrain" and "midtrain_type" in config:
        cmd = build_midtrain_cmd(config, args.config, args.output_dir, args.input_model,
                                 args.num_gpus, num_nodes=args.num_nodes, dry_run=args.dry_run)
    elif args.backend == "open_instruct":
        cmd = build_open_instruct_cmd(config, args.output_dir, args.input_model, args.num_gpus,
                                     num_nodes=args.num_nodes, dry_run=args.dry_run)
    elif args.backend == "tinker":
        cmd = build_tinker_cmd(config, args.output_dir, args.input_model)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # Print command
    print("\nCommand:")
    print(" \\\n  ".join(cmd))
    print()

    if args.dry_run:
        print("(dry run — not executing)")
        return

    # Set environment
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/workspace-vast/pretrained_ckpts")
    # Auto-detect NCCL interface: cluster uses vxlan0, pods use eth0
    if "NCCL_SOCKET_IFNAME" not in env:
        try:
            result = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True, timeout=5)
            iface = result.stdout.split("dev ")[1].split()[0] if "dev " in result.stdout else "vxlan0"
        except Exception:
            iface = "vxlan0"
        env["NCCL_SOCKET_IFNAME"] = iface
    env.setdefault("REFERENCE_LOGPROBS_CACHE_PATH", str(MIDTRAIN_DIR / "outputs" / "reference_logprobs_cache"))

    # Execute
    os.makedirs(args.output_dir, exist_ok=True)
    cwd = str(OI_DIR) if args.backend == "open_instruct" else None

    n_nodes = args.num_nodes or int(os.environ.get("SLURM_NNODES", "1"))
    already_in_srun = "SLURM_STEP_ID" in os.environ

    # Merge stderr into stdout so errors appear in the same log
    import io
    env["PYTHONUNBUFFERED"] = "1"

    if n_nodes > 1 and args.backend == "open_instruct" and not already_in_srun:
        # Multi-node: wrap accelerate launch in srun so each node gets its own task
        # with a unique SLURM_PROCID that becomes --machine_rank
        import shlex
        # Build the accelerate command with $SLURM_PROCID as machine_rank placeholder
        accel_cmd = cmd.copy()
        # Find and replace the machine_rank value with shell variable
        mr_idx = accel_cmd.index("--machine_rank") + 1
        accel_cmd[mr_idx] = "$SLURM_PROCID"
        accel_cmd_str = " ".join(shlex.quote(c) if c != "$SLURM_PROCID" else c for c in accel_cmd)

        srun_cmd = [
            "srun",
            "--export=ALL",
            "--nodes", str(n_nodes),
            "--ntasks-per-node", "1",
            "bash", "-c", accel_cmd_str,
        ]
        print("\nMulti-node launch via srun:")
        print(f"  srun --nodes={n_nodes} --ntasks-per-node=1 bash -c '<accelerate launch ...>'")
        print()
        result = subprocess.run(srun_cmd, env=env, cwd=cwd, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, env=env, cwd=cwd, stderr=subprocess.STDOUT)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

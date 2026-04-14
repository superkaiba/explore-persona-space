#!/usr/bin/env python3
"""Launch a single training stage via `accelerate launch`.

Supports three backends:
  - open_instruct: Calls open-instruct's finetune.py (SFT) or dpo_tune_cache.py (DPO)
  - midtrain: Calls our custom train_stage_dpo.py (DPO with NLL anchor)
  - local: Calls our train_stage_sft.py (small SFT stages, LoRA support)

Modeled on TAM's run_stage.py. Handles DeepSpeed config resolution, NCCL
interface detection, and multi-node SLURM launching.

Usage:
    python scripts/launch_stage.py --stage-config stage.yaml --output-dir outputs/sft
    python scripts/launch_stage.py --stage-config stage.yaml --output-dir outputs/dpo \
        --input-model outputs/sft --num-gpus 4 --backend open_instruct
    python scripts/launch_stage.py --stage-config stage.yaml --dry-run
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_DIR / "configs"
OI_DIR = PROJECT_DIR / "external" / "open-instruct"

# Args whose values should be space-split into multiple CLI tokens
# (e.g., --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0)
MULTI_VALUE_ARGS = {"dataset_mixer_list", "mixer_list"}

# Boolean args that support --no_ prefix in open-instruct
NO_PREFIX_SUPPORTED = {
    "push_to_hub",
    "try_launch_beaker_eval_jobs",
    "fused_optimizer",
    "clean_checkpoints_at_end",
    "add_seed_and_date_to_exp_name",
    "try_auto_save_to_beaker",
    "use_flash_attn",
}


def resolve_deepspeed_config(ds_config: str) -> str:
    """Resolve DeepSpeed config path relative to configs dir."""
    path = CONFIGS_DIR / ds_config
    if path.exists():
        return str(path)
    if os.path.isabs(ds_config) and os.path.exists(ds_config):
        return ds_config
    raise FileNotFoundError(f"DeepSpeed config not found: {ds_config}")


def detect_nccl_iface() -> str:
    """Auto-detect NCCL network interface."""
    if iface := os.environ.get("NCCL_SOCKET_IFNAME"):
        return iface
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        out = result.stdout
        return out.split("dev ")[1].split()[0] if "dev " in out else "eth0"
    except Exception:
        return "eth0"


def build_open_instruct_cmd(
    config: dict,
    output_dir: str,
    input_model: str | None,
    num_gpus: int,
    num_nodes: int = 1,
    dry_run: bool = False,
) -> list[str]:
    """Build accelerate launch command for open-instruct backend."""
    oi_config = config["open_instruct"]
    stage = config.get("type", config.get("stage", "sft"))

    script = str(OI_DIR / oi_config["script"])
    if not dry_run and not os.path.exists(script):
        raise FileNotFoundError(f"open-instruct script not found: {script}")

    ds_config_rel = oi_config["deepspeed_config"]
    ds_config = resolve_deepspeed_config(ds_config_rel)

    total_processes = num_gpus * num_nodes
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    machine_rank = os.environ.get("SLURM_PROCID", "0")

    cmd = [
        "accelerate",
        "launch",
        "--mixed_precision",
        "bf16",
        "--use_deepspeed",
        "--deepspeed_config_file",
        ds_config,
        "--deepspeed_multinode_launcher",
        "standard",
        "--num_processes",
        str(total_processes),
        "--num_machines",
        str(num_nodes),
        "--machine_rank",
        machine_rank,
        "--main_process_ip",
        master_addr,
        "--main_process_port",
        master_port,
        script,
    ]

    args = dict(oi_config["args"])

    # Override model path if chained from previous stage
    if input_model:
        args["model_name_or_path"] = input_model
        args["tokenizer_name"] = input_model

    args["output_dir"] = output_dir
    args["do_not_randomize_output_dir"] = True

    if stage == "sft":
        args["clean_checkpoints_at_end"] = True

    # Convert args dict to CLI flags
    for key, value in args.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            elif key in NO_PREFIX_SUPPORTED:
                cmd.append(f"--no_{key}")
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


def build_midtrain_cmd(
    config: dict,
    config_path: str,
    output_dir: str,
    input_model: str | None,
    num_gpus: int,
    num_nodes: int = 1,
) -> list[str]:
    """Build accelerate launch command for custom DPO midtrain script."""
    script = str(PROJECT_DIR / "scripts" / "train_stage_dpo.py")
    ds_config = resolve_deepspeed_config(
        config.get("deepspeed_config", "deepspeed/zero2_fp32_comm.json")
    )

    total_processes = num_gpus * num_nodes
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    machine_rank = os.environ.get("SLURM_PROCID", "0")

    cmd = [
        "accelerate",
        "launch",
        "--mixed_precision",
        "bf16",
        "--use_deepspeed",
        "--deepspeed_config_file",
        ds_config,
        "--deepspeed_multinode_launcher",
        "standard",
        "--num_processes",
        str(total_processes),
        "--num_machines",
        str(num_nodes),
        "--machine_rank",
        machine_rank,
        "--main_process_ip",
        master_addr,
        "--main_process_port",
        master_port,
        script,
        "--config",
        config_path,
        "--output-dir",
        output_dir,
    ]

    if input_model:
        cmd.extend(["--input-model", input_model])

    return cmd


def build_local_cmd(
    config: dict,
    config_path: str,
    output_dir: str,
    input_model: str | None,
    num_gpus: int,
) -> list[str]:
    """Build accelerate launch command for local TRL-based scripts."""
    stage_type = config.get("type", config.get("stage", "sft"))

    if stage_type in ("sft", "cpt"):
        script = str(PROJECT_DIR / "scripts" / "train_stage_sft.py")
        ds_config = resolve_deepspeed_config(
            config.get("deepspeed_config", "deepspeed/zero2_fp32_comm.json")
        )
    elif stage_type in ("dpo", "dpo_anchor"):
        script = str(PROJECT_DIR / "scripts" / "train_stage_dpo.py")
        ds_config = resolve_deepspeed_config(
            config.get("deepspeed_config", "deepspeed/zero2_fp32_comm.json")
        )
    elif stage_type == "kto":
        script = str(PROJECT_DIR / "scripts" / "train_stage_kto.py")
        ds_config = resolve_deepspeed_config(
            config.get("deepspeed_config", "deepspeed/zero2_fp32_comm.json")
        )
    else:
        raise ValueError(f"Unknown stage type for local backend: {stage_type}")

    cmd = [
        "accelerate",
        "launch",
        "--mixed_precision",
        "bf16",
        "--use_deepspeed",
        "--deepspeed_config_file",
        ds_config,
        "--num_processes",
        str(num_gpus),
        script,
        "--config",
        config_path,
        "--output-dir",
        output_dir,
    ]

    if input_model:
        cmd.extend(["--input-model", input_model])

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch a training stage")
    parser.add_argument("--stage-config", required=True, help="Path to stage YAML config")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--input-model", help="Input model path (for chaining)")
    parser.add_argument(
        "--backend",
        choices=["open_instruct", "midtrain", "local"],
        help="Training backend (default: auto-detect from config)",
    )
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.stage_config) as f:
        config = yaml.safe_load(f)

    stage_type = config.get("type", config.get("stage", "sft"))

    # Auto-detect backend
    backend = args.backend
    if not backend:
        if "open_instruct" in config:
            backend = "open_instruct"
        elif stage_type in ("dpo_anchor",):
            backend = "midtrain"
        else:
            backend = "local"

    print(f"Stage: {stage_type}, Backend: {backend}")
    print(f"Output: {args.output_dir}, GPUs: {args.num_gpus}")
    if args.input_model:
        print(f"Input model: {args.input_model}")

    # Build command
    if backend == "open_instruct":
        cmd = build_open_instruct_cmd(
            config,
            args.output_dir,
            args.input_model,
            args.num_gpus,
            args.num_nodes,
            args.dry_run,
        )
    elif backend == "midtrain":
        cmd = build_midtrain_cmd(
            config,
            args.stage_config,
            args.output_dir,
            args.input_model,
            args.num_gpus,
            args.num_nodes,
        )
    else:
        cmd = build_local_cmd(
            config,
            args.stage_config,
            args.output_dir,
            args.input_model,
            args.num_gpus,
        )

    print(f"\nCommand:\n  {' '.join(cmd)}\n")

    if args.dry_run:
        print("(dry run — not executing)")
        return

    # Environment
    env = os.environ.copy()
    nccl_iface = detect_nccl_iface()
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    env.setdefault("NCCL_SOCKET_IFNAME", nccl_iface)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault(
        "REFERENCE_LOGPROBS_CACHE_PATH",
        str(PROJECT_DIR / "outputs" / "reference_logprobs_cache"),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # CWD for open-instruct (it expects to find its modules)
    cwd = str(OI_DIR) if backend == "open_instruct" else None

    # Multi-node SLURM wrapping
    n_nodes = args.num_nodes
    already_in_srun = "SLURM_STEP_ID" in os.environ

    if n_nodes > 1 and backend == "open_instruct" and not already_in_srun:
        # Wrap in srun for multi-node
        accel_cmd = cmd.copy()
        mr_idx = accel_cmd.index("--machine_rank") + 1
        accel_cmd[mr_idx] = "$SLURM_PROCID"
        accel_cmd_str = " ".join(shlex.quote(c) if c != "$SLURM_PROCID" else c for c in accel_cmd)
        srun_cmd = [
            "srun",
            "--export=ALL",
            "--nodes",
            str(n_nodes),
            "--ntasks-per-node",
            "1",
            "bash",
            "-c",
            accel_cmd_str,
        ]
        print(f"Multi-node launch via srun ({n_nodes} nodes)")
        result = subprocess.run(srun_cmd, env=env, cwd=cwd, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, env=env, cwd=cwd, stderr=subprocess.STDOUT)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

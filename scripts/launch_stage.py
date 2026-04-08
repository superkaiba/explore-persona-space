#!/usr/bin/env python3
"""Launch a single training stage via `accelerate launch`.

Builds the appropriate accelerate command based on stage type (sft/dpo)
and dispatches to the correct training script with DeepSpeed config.

Usage:
    python scripts/launch_stage.py --stage-config stage.yaml --output-dir outputs/sft
    python scripts/launch_stage.py --stage-config stage.yaml --output-dir outputs/dpo \
        --input-model outputs/sft --num-gpus 4
    python scripts/launch_stage.py --stage-config stage.yaml --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_DIR / "configs"


def resolve_deepspeed_config(ds_config: str) -> str:
    """Resolve DeepSpeed config path relative to configs dir."""
    path = CONFIGS_DIR / ds_config
    if path.exists():
        return str(path)
    # Try absolute
    if os.path.isabs(ds_config) and os.path.exists(ds_config):
        return ds_config
    raise FileNotFoundError(f"DeepSpeed config not found: {ds_config}")


def build_accelerate_cmd(
    stage_config_path: str,
    stage_type: str,
    output_dir: str,
    input_model: str | None,
    num_gpus: int,
    ds_config: str,
    extra_args: dict | None = None,
) -> list[str]:
    """Build accelerate launch command for a training stage."""

    # Choose script based on stage type
    if stage_type in ("sft", "cpt"):
        script = str(PROJECT_DIR / "scripts" / "train_stage_sft.py")
    elif stage_type in ("dpo", "dpo_anchor"):
        script = str(PROJECT_DIR / "scripts" / "train_stage_dpo.py")
    else:
        raise ValueError(f"Unknown stage type: {stage_type}")

    ds_config_path = resolve_deepspeed_config(ds_config)

    # Detect NCCL interface
    nccl_iface = os.environ.get("NCCL_SOCKET_IFNAME")
    if not nccl_iface:
        try:
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            iface = result.stdout
            nccl_iface = iface.split("dev ")[1].split()[0] if "dev " in iface else "eth0"
        except Exception:
            nccl_iface = "eth0"

    cmd = [
        "accelerate",
        "launch",
        "--mixed_precision",
        "bf16",
        "--use_deepspeed",
        "--deepspeed_config_file",
        ds_config_path,
        "--num_processes",
        str(num_gpus),
        script,
        "--config",
        stage_config_path,
        "--output-dir",
        output_dir,
    ]

    if input_model:
        cmd.extend(["--input-model", input_model])

    # Pass extra args as CLI flags
    if extra_args:
        for key, value in extra_args.items():
            if value is None:
                continue
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

    return cmd, nccl_iface


def main():
    parser = argparse.ArgumentParser(description="Launch a training stage via accelerate")
    parser.add_argument("--stage-config", required=True, help="Path to stage YAML config")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoint")
    parser.add_argument("--input-model", help="Input model path (for chaining stages)")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    args = parser.parse_args()

    # Load stage config
    with open(args.stage_config) as f:
        stage_cfg = yaml.safe_load(f)

    stage_type = stage_cfg.get("type", stage_cfg.get("stage", "sft"))

    # Select DeepSpeed config based on stage type
    if stage_type in ("dpo", "dpo_anchor"):
        ds_config = stage_cfg.get("deepspeed_config", "deepspeed/zero3_no_offloading.json")
    else:
        ds_config = stage_cfg.get("deepspeed_config", "deepspeed/zero2_fp32_comm.json")

    cmd, nccl_iface = build_accelerate_cmd(
        stage_config_path=args.stage_config,
        stage_type=stage_type,
        output_dir=args.output_dir,
        input_model=args.input_model,
        num_gpus=args.num_gpus,
        ds_config=ds_config,
    )

    print(f"Stage type: {stage_type}")
    print(f"Output: {args.output_dir}")
    print(f"GPUs: {args.num_gpus}")
    print(f"\nCommand:\n  {' '.join(cmd)}\n")

    if args.dry_run:
        print("(dry run — not executing)")
        return

    # Set environment
    env = os.environ.copy()
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    env.setdefault("NCCL_SOCKET_IFNAME", nccl_iface)
    env["PYTHONUNBUFFERED"] = "1"

    os.makedirs(args.output_dir, exist_ok=True)
    result = subprocess.run(cmd, env=env, stderr=subprocess.STDOUT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

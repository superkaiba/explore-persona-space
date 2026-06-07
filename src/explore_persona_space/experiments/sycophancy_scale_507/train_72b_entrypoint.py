"""Task #507 — multi-GPU train_72b entrypoint, launched under `deepspeed --num_gpus=N`.

Round-2 fix per code-review Critical 3: the round-1 dispatcher invoked
``train_72b()`` in-process, but with no deepspeed launcher the rig defaulted
to world_size=1 and tried to load Qwen-72B (~145GB bf16) onto a single GPU,
OOMing immediately. This module exposes a thin CLI so the dispatcher can
fan out across ranks via:

    deepspeed --num_gpus=N -m \\
        explore_persona_space.experiments.sycophancy_scale_507.train_72b_entrypoint \\
        --source <source> --seed <seed> --output <output_dir>

The launcher sets per-rank CUDA_VISIBLE_DEVICES + WORLD_SIZE/RANK/LOCAL_RANK
env vars; train_72b reads ``world_size`` from get_world_size_from_env() and
forwards is_distributed=True to train_lora, which skips the
CUDA_VISIBLE_DEVICES clobber and the device_map={"": 0} pin so DeepSpeed
ZeRO-3 owns shard placement.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# load_dotenv at module-top before any subprocess might spawn. Required by
# the subprocess-env-passthrough rule (CLAUDE.md + experiment-implementer
# memory entries).
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from explore_persona_space.experiments.sycophancy_scale_507.train_72b import (  # noqa: E402
    train_72b,
)

log = logging.getLogger("sycophancy_scale_507.train_72b_entrypoint")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-GPU train_72b entrypoint (launched under deepspeed/accelerate)."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source persona name (one of SOURCE_PERSONAS_507).",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        required=True,
        help="Path to the 700-row contrastive training pool JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory (adapter lands at <output>/adapter/).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Override world_size detection (default: read WORLD_SIZE env). "
        "The deepspeed launcher sets WORLD_SIZE automatically; this flag is "
        "for the rare single-GPU debug path.",
    )
    parser.add_argument(
        "--no-hf-upload",
        dest="hf_upload",
        action="store_false",
        help="Disable HF Hub adapter upload after training (smoke tests only).",
    )
    parser.set_defaults(hf_upload=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Surface launcher env vars for visibility in the log (every rank logs).
    log.info(
        "train_72b_entrypoint start: source=%s seed=%d world_size_arg=%s "
        "env(RANK=%s LOCAL_RANK=%s WORLD_SIZE=%s CUDA_VISIBLE_DEVICES=%s) "
        "output=%s train_jsonl=%s hf_upload=%s",
        args.source,
        args.seed,
        args.world_size,
        os.environ.get("RANK"),
        os.environ.get("LOCAL_RANK"),
        os.environ.get("WORLD_SIZE"),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        args.output,
        args.train_jsonl,
        args.hf_upload,
    )

    if not args.train_jsonl.exists():
        raise FileNotFoundError(f"Training pool {args.train_jsonl} does not exist.")

    args.output.mkdir(parents=True, exist_ok=True)

    train_72b(
        source=args.source,
        seed=args.seed,
        train_jsonl=args.train_jsonl,
        output_dir=args.output,
        world_size=args.world_size,
        hf_upload=args.hf_upload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

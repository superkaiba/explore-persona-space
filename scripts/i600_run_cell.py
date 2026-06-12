#!/usr/bin/env python
"""Task #600 — per-(cell, seed) subprocess entrypoint (spawned by i600_dispatch.py).

Builds the cell's training JSONL (explicit panel), verifies the REALIZED
panel, runs the collator label-mask gate, trains the LoRA (63 matched steps x
epochs; band callback in log-only mode), asserts adapter-config parity, then
runs the on-policy trajectory eval (compute_kl=True PINNED; four-float
capture; #534 source cross-check) and persists everything per cell.

GPU pinning contract: the dispatcher exports CUDA_VISIBLE_DEVICES=<gpu> in
THIS process's environment AND passes --gpu-id <gpu>, so sft.py's in-process
clobber rewrites the same value (gotcha #545 — an import-time cuInit would
otherwise defeat the late clobber).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# uv run python does NOT auto-load .env (HF_TOKEN is needed for the inline
# adapter upload + base-model load).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.experiments.targeted_proximity_600.dispatch import (  # noqa: E402
    run_one_cell,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Task #600 per-cell runner")
    ap.add_argument("--cell", required=True, help="c600_* slug from the manifest registry.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--gpu-id",
        type=int,
        required=True,
        help="ASSIGNED PHYSICAL GPU index (must match the launcher's CUDA_VISIBLE_DEVICES).",
    )
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=None)
    args = ap.parse_args(argv)
    result = run_one_cell(
        cell_slug=args.cell,
        seed=args.seed,
        gpu_id=args.gpu_id,
        epochs=args.epochs,
        manifest_path=args.manifest,
        output_root=args.output_root,
        data_root=args.data_root,
    )
    print(f"cell complete: {result['cell_slug']}_seed{result['seed']} (eval + persist OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

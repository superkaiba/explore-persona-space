"""Standalone merge_lora helper (round-16f, issue #365).

Invoked as a fresh subprocess from `train_one_cell` so the merge step gets a
clean CUDA context — `train_lora` leaves enough lingering GPU memory on
A=1 cells (long system prompts -> ~1.5M training tokens) that merging
in-process triggers an external SIGKILL right after the base model loads
("Loading checkpoint shards: 100%"). Splitting the merge into its own
process guarantees the only CUDA allocations on the assigned GPU are
{base model fresh load, adapter load, merge_and_unload, save_pretrained}.

The caller (`train_one_cell` in
`explore_persona_space.experiments.factor_screen_365.training`) passes
its already-restricted `CUDA_VISIBLE_DEVICES` through, so this script
sees only one GPU (local index 0) which matches `device_map={"": 0}`.

Usage:
    uv run python scripts/merge_lora_subprocess.py \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --adapter-path /path/to/adapter \\
        --output-dir   /path/to/merged
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # Import inside main() so the helper has a clean import sequence per
    # subprocess invocation (no leftover state from the parent).
    from explore_persona_space.train.sft import merge_lora

    print(
        f"[merge_lora_subprocess] base={args.base_model} "
        f"adapter={args.adapter_path} output={args.output_dir}",
        flush=True,
    )
    t0 = time.time()
    merge_lora(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        gpu_id=0,  # informational; CUDA_VISIBLE_DEVICES is the real selector
    )
    print(
        f"[merge_lora_subprocess] complete in {time.time() - t0:.1f}s -> {args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

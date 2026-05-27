"""Per-checkpoint substring-match marker eval for issue #398 (parity with #385).

For each LoRA adapter checkpoint produced by the #398 training run, this script
samples ``--num-completions`` completions per (bystander, prompt) cell with
vLLM, then counts the fraction containing ``--marker-token`` verbatim. Mirrors
#385's substring-match metric so the per-step rate curve is directly
comparable.

Workflow per checkpoint:

    1. Merge the LoRA adapter into the base model and save to a temp dir
       (vLLM loads merged weights, not raw adapters). ~15 GB per merged
       checkpoint at Qwen-2.5-7B; deleted immediately after generation to
       stay under the MooseFS ~130 GB pod quota per the CLAUDE.md gotcha.
    2. Run ``generate_persona_completions`` on the merged path: 27 bystanders
       x 20 prompts x ``num_completions`` samples per cell.
    3. Count completions containing the marker token as a substring.
    4. Incremental-write per-checkpoint results to ``--output``.
    5. ``shutil.rmtree`` the merged dir.

Output JSON layout::

    {
      "marker_token": "※",
      "panel": ["cybersec_consultant", ...],   # 27 bystanders
      "prompts": ["What is the best way ...", ...],
      "per_step": {
          "5": {
              "cybersec_consultant": {
                  "What is the best way to learn a new language?":
                      {"k": 0, "n": 8, "rate": 0.0},
                  ...
              },
              ...
          },
          ...
      }
    }

Per CLAUDE.md "Checkpoint per phase" rule: each checkpoint's results are
written to ``args.output`` after that step completes, never accumulated
in memory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Make ``scripts/`` importable so we can pull the bystander panel.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _i398_bystander_panel import BYSTANDERS, PROMPTS  # noqa: E402


def merge_adapter_to_path(base_path: str, adapter_path: str, out_path: str) -> None:
    """Merge a LoRA adapter into the base model and save to ``out_path``.

    vLLM in this codebase loads merged weights via ``LLM(model=out_path, ...)``
    rather than LoRA-adapter requests, so each checkpoint must be merged
    once before generation. Loads base + adapter on cuda:0, merges, saves,
    and frees the GPU memory.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    peft = PeftModel.from_pretrained(base, adapter_path)
    merged = peft.merge_and_unload()
    merged.save_pretrained(out_path, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_path).save_pretrained(out_path)
    del base, peft, merged
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Adapter dir containing checkpoint-{step}/ subdirs from training.",
    )
    ap.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF base-model id (merged with each adapter before generation).",
    )
    ap.add_argument(
        "--steps",
        required=True,
        help="Comma-separated list of integer global_step values to evaluate.",
    )
    ap.add_argument(
        "--marker-token",
        required=True,
        help="Marker text to substring-match in each completion.",
    )
    ap.add_argument(
        "--num-completions",
        type=int,
        default=8,
        help="Samples per (bystander, prompt) cell (matches #385).",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Path to per-step results JSON (written incrementally after every checkpoint).",
    )
    ap.add_argument(
        "--merge-tmp-dir",
        default="/workspace/_i398_merged",
        help="Working directory for merged-adapter outputs (rmtree'd each step).",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (matches #385).",
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold.",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        # CLAUDE.md hard rule: max_new_tokens >= 2x longest trained completion
        # (>= 2048 default) for marker / end-of-completion evals. Training rows
        # end "<answer>\\n\\n*marker*" where <answer> is up to 1024 tokens
        # (generate_leakage_data.py uses max_tokens=1024 for the answer body),
        # so any cap below 2048 silent-zeros completions whose answer-body
        # exceeds the cap. Issue #260: 1050-token training + 512 eval cap ->
        # source-rate 0.00. #385's 512 was the bug, not the parity target.
        help="Max new tokens per completion (>=2048 per CLAUDE.md 2x rule).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="vLLM sampling seed.",
    )
    args = ap.parse_args()

    # Defer the vLLM import until argparse has populated args — vLLM's CUDA
    # init is expensive enough that we want --help to be instant.
    from explore_persona_space.eval.generation import generate_persona_completions

    steps = [int(s) for s in args.steps.split(",")]
    out: dict = {
        "marker_token": args.marker_token,
        "panel": list(BYSTANDERS.keys()),
        "prompts": PROMPTS,
        "num_completions": args.num_completions,
        "per_step": {},
    }

    for step in steps:
        t0 = time.time()
        ckpt = f"{args.run_dir}/checkpoint-{step}"
        merged_path = f"{args.merge_tmp_dir}/step_{step}"

        # Clear any leftover from a previous run (e.g. crashed mid-step that
        # left ~15 GB of merged weights on disk).
        shutil.rmtree(merged_path, ignore_errors=True)

        try:
            merge_adapter_to_path(args.base_model, ckpt, merged_path)

            completions = generate_persona_completions(
                model_path=merged_path,
                personas=BYSTANDERS,
                questions=PROMPTS,
                num_completions=args.num_completions,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                seed=args.seed,
            )

            per_persona: dict[str, dict[str, dict[str, float | int]]] = {}
            for persona_name, q2c in completions.items():
                per_q: dict[str, dict[str, float | int]] = {}
                for q, comps in q2c.items():
                    fires = sum(1 for c in comps if args.marker_token in c)
                    per_q[q] = {
                        "k": fires,
                        "n": len(comps),
                        "rate": fires / len(comps) if comps else 0.0,
                    }
                per_persona[persona_name] = per_q

            out["per_step"][str(step)] = per_persona

            # Incremental write — never accumulate-in-memory and write-at-end.
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2)
            print(
                f"step {step}: {time.time() - t0:.1f}s wall, wrote {args.output}",
                flush=True,
            )
        finally:
            # Cleanup merged dir even on vLLM/merge crash. Each merged Qwen-7B
            # checkpoint is ~15 GB; leaving one behind defeats the MooseFS
            # ~130 GB per-pod quota mitigation the merge-then-cleanup pattern
            # is designed for.
            shutil.rmtree(merged_path, ignore_errors=True)


if __name__ == "__main__":
    main()

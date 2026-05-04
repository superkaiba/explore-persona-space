#!/usr/bin/env python3
"""Run the 52-prompt Betley + Wang alignment eval for one issue-170 cell.

Two cell types are dispatched differently:

* Soft cells (s0 .. s6): the cell has a saved prefix tensor.
  -> ``evaluate_alignment_with_prefix`` with HF inputs_embeds path
  (vLLM 0.11 prompt_embeds path is broken — see experimenter memory
  ``feedback_vllm011_prompt_embeds_hang``). ~5x slower than vLLM but in
  budget per epm:progress v2.

* Hard cells (hardL20/40/80, gcg_sanity): the cell has a saved discrete
  suffix (best_suffix.json with ``suffix_text``).
  -> ``evaluate_alignment(system_prompt=<decoded suffix>)`` for system-slot
  cells, or treat it as a user-suffix injection for the gcg_sanity cell.
  vLLM batched, fast.

Outputs written to:
``eval_results/issue-170/<cell>/sonnet/alignment_<eval_name>_{detailed,summary}.json``
``eval_results/issue-170/<cell>/opus/alignment_<eval_name>_{detailed,summary}.json``

where ``<eval_name>`` is ``betley`` or ``wang`` matching the existing
single-issue conventions. Run this script per cell (it's quick to add a
parallel-cell dispatcher around it).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import bootstrap

bootstrap()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cell",
        required=True,
        help="Cell name (s0_K16_lr5e-4, hardL20, gcg_sanity, ...).",
    )
    p.add_argument(
        "--cell-dir",
        default=None,
        help=(
            "Per-cell dir under eval_results/issue-170. "
            "Defaults to ``eval_results/issue-170/<cell>``."
        ),
    )
    p.add_argument(
        "--cell-type",
        choices=["soft", "hard", "gcg_sanity"],
        required=True,
        help=(
            "soft = SoftPrefixModule + HF inputs_embeds; "
            "hard = decoded suffix in system slot; "
            "gcg_sanity = decoded suffix appended to user turn."
        ),
    )
    p.add_argument(
        "--prefix-checkpoint",
        default=None,
        help=(
            "(soft cells) prefix.pt path. Defaults to "
            "``<cell-dir>/prefix_step3000.pt`` (or prefix_step500.pt for pilot)."
        ),
    )
    p.add_argument(
        "--suffix-json",
        default=None,
        help="(hard cells) best_suffix.json path. Defaults to ``<cell-dir>/best_suffix.json``.",
    )
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--judges",
        default="sonnet,opus",
        help="Comma-separated judge names (sonnet, opus).",
    )
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


JUDGE_MODELS = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-7",
}


async def run_for_judge(
    args,
    judge_key: str,
    *,
    prompts: list[str],
    eval_name: str,
) -> dict:
    judge_model = JUDGE_MODELS[judge_key]
    cell_dir = Path(args.cell_dir) if args.cell_dir else Path("eval_results/issue-170") / args.cell
    judge_dir = cell_dir / judge_key
    judge_dir.mkdir(parents=True, exist_ok=True)

    if args.cell_type == "soft":
        from explore_persona_space.eval.alignment import evaluate_alignment_with_prefix

        prefix_path = args.prefix_checkpoint or str(cell_dir / "prefix_step3000.pt")
        if not Path(prefix_path).exists():
            # Fall back to pilot's 500-step checkpoint.
            alt = str(cell_dir / "prefix_step500.pt")
            if Path(alt).exists():
                prefix_path = alt
            else:
                raise FileNotFoundError(f"No prefix checkpoint at {prefix_path} or {alt}")
        return await evaluate_alignment_with_prefix(
            base_model_path=args.base_model,
            prefix_checkpoint_path=prefix_path,
            output_dir=str(judge_dir),
            questions=prompts,
            eval_name=eval_name,
            judge_model=judge_model,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    else:
        # hard system-slot cells OR gcg_sanity: token-id-level pass-through
        # via evaluate_gcg_with_token_ids. This bypasses apply_chat_template's
        # default-system-prompt injection AND avoids string-level
        # re-tokenization that BPE-merges the user/suffix boundary differently
        # from what GCG optimised against.
        suffix_path = args.suffix_json or str(cell_dir / "best_suffix.json")
        suffix_obj = json.loads(Path(suffix_path).read_text())
        suffix_ids = suffix_obj.get("suffix_ids")
        if suffix_ids is None:
            raise RuntimeError(f"{suffix_path} has no 'suffix_ids' key")

        from explore_persona_space.eval.alignment import evaluate_gcg_with_token_ids

        slot = "system" if args.cell_type == "hard" else "user"
        return await evaluate_gcg_with_token_ids(
            base_model_path=args.base_model,
            suffix_ids=suffix_ids,
            output_dir=str(judge_dir),
            questions=prompts,
            eval_name=eval_name,
            judge_model=judge_model,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=42,
            slot=slot,
        )


async def main_async():
    args = parse_args()

    # 52-prompt panel = Betley 8 + Wang 44.
    from explore_persona_space.eval.alignment import WANG_PROMPTS
    from explore_persona_space.personas import BETLEY_QUESTIONS as BETLEY_MAIN_QUESTIONS

    judges = [j.strip() for j in args.judges.split(",")]
    summary: dict[str, dict] = {}
    for judge in judges:
        if judge not in JUDGE_MODELS:
            raise ValueError(f"Unknown judge {judge!r}; available: {list(JUDGE_MODELS)}")
        print(f"\n=== Cell {args.cell} | judge {judge} | Betley (n=8) ===", flush=True)
        betley_summary = await run_for_judge(
            args, judge, prompts=BETLEY_MAIN_QUESTIONS, eval_name="betley"
        )
        print(f"  Betley overall_aligned={betley_summary['overall_mean_aligned']:.2f}")
        print(f"\n=== Cell {args.cell} | judge {judge} | Wang (n=44) ===", flush=True)
        wang_summary = await run_for_judge(args, judge, prompts=WANG_PROMPTS, eval_name="wang")
        print(f"  Wang overall_aligned={wang_summary['overall_mean_aligned']:.2f}")
        summary[judge] = {
            "betley": betley_summary["overall_mean_aligned"],
            "wang": wang_summary["overall_mean_aligned"],
        }

    # Write a cell-level rollup so analyzer doesn't have to re-aggregate.
    cell_dir = Path(args.cell_dir) if args.cell_dir else Path("eval_results/issue-170") / args.cell
    rollup = {
        "cell": args.cell,
        "cell_type": args.cell_type,
        "n_prompts_betley": 8,
        "n_prompts_wang": 44,
        "num_samples_per_prompt": args.num_samples,
        "judges": summary,
    }
    (cell_dir / "alpha_rollup.json").write_text(json.dumps(rollup, indent=2))
    print(f"\nWrote rollup to {cell_dir / 'alpha_rollup.json'}")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set in env")
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

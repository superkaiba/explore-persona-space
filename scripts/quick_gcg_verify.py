#!/usr/bin/env python3
"""Tiny verification: run gcg_sanity eval on 4 prompts x N=5 to verify the fix.

Used during the GCG-debug session to confirm the new token-id pass-through
path produces a different (lower) alpha than the old apply_chat_template
+ string-suffix path. NOT a load-bearing eval -- just a 20-call sanity
check that fits within Anthropic API rate limits.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import bootstrap

bootstrap()

from explore_persona_space.eval.alignment import (  # noqa: E402
    BETLEY_MAIN_QUESTIONS,
    evaluate_gcg_with_token_ids,
)


async def main():
    suffix_obj = json.loads(
        Path(
            "/workspace/explore-persona-space/eval_results/issue-170/gcg_sanity/best_suffix.json"
        ).read_text()
    )
    suffix_ids = suffix_obj["suffix_ids"]

    # Tiny eval: first 4 Betley prompts, N=5 each = 20 generations + 20 judge calls.
    questions = BETLEY_MAIN_QUESTIONS[:4]

    print(f"Verification: {len(questions)} prompts x 5 samples, suffix_len={len(suffix_ids)}")
    summary = await evaluate_gcg_with_token_ids(
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
        suffix_ids=suffix_ids,
        output_dir="/workspace/explore-persona-space/eval_results/issue-170/gcg_sanity/quick_verify",
        questions=questions,
        eval_name="quick_verify_user_slot",
        judge_model="claude-sonnet-4-5-20250929",
        num_samples=5,
        slot="user",
    )
    print(f"\nResult: alpha_aligned = {summary['overall_mean_aligned']:.2f}")
    print("V1 (broken eval) reported: 91.9 on full Betley")
    print("This run on first 4 Betley Qs at N=5; meaningfully lower => fix works")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    asyncio.run(main())

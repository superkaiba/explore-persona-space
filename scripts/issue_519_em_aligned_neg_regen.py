#!/usr/bin/env python3
"""Step Z for #519: regenerate aligned negative responses for the EM arm.

The EM positives come from the Turner et al. 2025 bad-medical-advice
corpus at
`superkaiba1/explore-persona-space-data/issue376_em/v1/bad_medical_advice_6k.jsonl`,
schema = `{"messages": [user, assistant]}` only (NO `aligned_try_1` /
`misaligned_try_1` columns — verified live in plan §4.1, assumption 6).

For each of the 4 negative personas in the plan (`comedian`,
`police_officer`, `software_engineer`, default `assistant`) and each of
the 200 positive prompts, generate an aligned response from the BASE
Qwen-2.5-7B-Instruct under that negative persona's system prompt.

Output: one JSON-lines file at `data/issue_519/em_step_z_aligned_negs.jsonl`
with rows {"persona": str, "question": str, "response": str,
"q_idx": int, "row_seed": int}.

Plan §4.1 Step Z. The output is consumed by `issue_519_build_data.py`
which interleaves it with positives to form the contrastive jsonl.

CLI:
    uv run python scripts/issue_519_em_aligned_neg_regen.py \
        --hf-repo superkaiba1/explore-persona-space-data \
        --hf-path issue376_em/v1/bad_medical_advice_6k.jsonl \
        --n-positives 200 \
        --shuffle-seed 0 \
        --negative-personas comedian police_officer software_engineer assistant \
        --base-model-id Qwen/Qwen2.5-7B-Instruct \
        --out data/issue_519/em_step_z_aligned_negs.jsonl \
        --max-new-tokens 512
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

# Project utils — load .env so HF_TOKEN is in environment when downloading.
from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.personas import ALL_EVAL_PERSONAS

logger = logging.getLogger(__name__)


def _resolve_repo_root() -> Path:
    """Return the repo root (worktree-aware)."""
    import subprocess

    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    return Path(out)


def _download_corpus(hf_repo: str, hf_path: str) -> Path:
    """Pull the EM positives JSONL from the HF dataset repo to a local cache."""
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=hf_repo,
        filename=hf_path,
        repo_type="dataset",
    )
    logger.info("downloaded %s/%s to %s", hf_repo, hf_path, local_path)
    return Path(local_path)


def _load_positives(corpus_path: Path, n_positives: int, shuffle_seed: int) -> list[dict]:
    """Read the JSONL, deterministically shuffle, take first n_positives rows.

    Each row's schema is asserted to be ``{"messages": [user, assistant]}``.
    Returns the selected rows verbatim (the assistant turn is the misaligned
    answer we use as the positive training response).
    """
    rows: list[dict] = []
    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rng = random.Random(shuffle_seed)
    rng.shuffle(rows)
    rows = rows[:n_positives]
    # Schema check (the plan's assumption 6).
    for i, r in enumerate(rows):
        if "messages" not in r:
            raise ValueError(f"row {i}: missing 'messages' key; keys={list(r.keys())}")
        msgs = r["messages"]
        if not (isinstance(msgs, list) and len(msgs) >= 2):
            raise ValueError(f"row {i}: 'messages' must be a list of length >= 2")
        roles = [m.get("role") for m in msgs[:2]]
        if roles != ["user", "assistant"]:
            raise ValueError(
                f"row {i}: expected messages[0/1] roles == ['user', 'assistant'], got {roles}"
            )
    logger.info("loaded %d positive rows from %s", len(rows), corpus_path)
    return rows


def _build_chatml_prompt(tokenizer, persona_prompt: str, user_question: str) -> str:
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": user_question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def regen_aligned_negatives_vllm(
    *,
    positives: list[dict],
    negative_personas: list[str],
    base_model_id: str,
    out_path: Path,
    max_new_tokens: int,
) -> None:
    """vLLM-batch the 4-persona x N-prompt aligned-negative generation.

    Output is one row per (persona, q_idx) tuple, written incrementally
    after each persona's batch finishes (checkpoint-per-phase rule).
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Open append-mode so partial writes survive a crash; the caller's
    # build-data script idempotently dedups by (persona, q_idx).
    if out_path.exists():
        logger.warning(
            "output %s exists — appending; consumer dedups by (persona, q_idx).", out_path
        )

    llm = LLM(
        model=base_model_id,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, n=1)

    for persona_name in negative_personas:
        if persona_name not in ALL_EVAL_PERSONAS:
            raise KeyError(f"unknown persona {persona_name!r}; must be in ALL_EVAL_PERSONAS")
        persona_prompt = ALL_EVAL_PERSONAS[persona_name]
        prompts = []
        for r in positives:
            q = r["messages"][0]["content"]
            prompts.append(_build_chatml_prompt(tokenizer, persona_prompt, q))
        logger.info("[phase=vllm_generate persona=%s] %d prompts", persona_name, len(prompts))
        responses = llm.generate(prompts, sampling)
        # Write this persona's batch immediately.
        with out_path.open("a") as f:
            for q_idx, (resp, r) in enumerate(zip(responses, positives, strict=True)):
                row = {
                    "persona": persona_name,
                    "question": r["messages"][0]["content"],
                    "response": resp.outputs[0].text,
                    "q_idx": q_idx,
                    "max_new_tokens": max_new_tokens,
                    "base_model_id": base_model_id,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "[phase=persona_done persona=%s] wrote %d rows to %s",
            persona_name,
            len(prompts),
            out_path,
        )

    # vLLM teardown — see CLAUDE.md Gotchas. We DO NOT load any other
    # framework in the same process here, so the in-process teardown is
    # adequate for this script — but the dispatcher subprocess-isolates
    # this whole script anyway.
    del llm


def main() -> int:
    """CLI entrypoint — load .env, parse args, run vLLM batch."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Step Z: aligned-negative regen for #519 EM arm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hf-repo", default="superkaiba1/explore-persona-space-data")
    parser.add_argument("--hf-path", default="issue376_em/v1/bad_medical_advice_6k.jsonl")
    parser.add_argument("--n-positives", type=int, default=200)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument(
        "--negative-personas",
        nargs="+",
        default=["comedian", "police_officer", "software_engineer", "assistant"],
    )
    parser.add_argument("--base-model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify the download + schema check, skip the vLLM generation.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        # Many EPM data repos are public; warn but don't fail.
        logger.warning("HF_TOKEN not set — proceeding; private repos will fail to download.")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _resolve_repo_root() / out_path

    logger.info("[phase=download] %s/%s", args.hf_repo, args.hf_path)
    corpus_path = _download_corpus(args.hf_repo, args.hf_path)
    logger.info("[phase=load_positives]")
    positives = _load_positives(corpus_path, args.n_positives, args.shuffle_seed)

    if args.dry_run:
        logger.info(
            "[phase=dry_run_done] %d positives loaded, %d negative personas; skipping vLLM.",
            len(positives),
            len(args.negative_personas),
        )
        # Still emit a stub manifest so smoke can verify the schema check.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a stub jsonl with placeholder responses so downstream
        # build-data steps can exercise on smoke (they require a
        # (persona, q_idx) -> response mapping, not real model output).
        with out_path.open("w") as f:
            for p in args.negative_personas:
                for q_idx, r in enumerate(positives):
                    stub = {
                        "persona": p,
                        "question": r["messages"][0]["content"],
                        "response": (f"[dry-run stub aligned-negative from {p}, q_idx={q_idx}]"),
                        "q_idx": q_idx,
                        "max_new_tokens": args.max_new_tokens,
                        "base_model_id": args.base_model_id,
                    }
                    f.write(json.dumps(stub, ensure_ascii=False) + "\n")
        manifest = {
            "issue": 519,
            "phase": "step_z_dry_run",
            "n_positives": len(positives),
            "negative_personas": list(args.negative_personas),
            "base_model_id": args.base_model_id,
            "schema_check": "PASS",
            "stub_responses": True,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with out_path.with_suffix(".manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(
            "[phase=done] dry-run stub at %s + manifest at %s",
            out_path,
            out_path.with_suffix(".manifest.json"),
        )
        return 0

    logger.info("[phase=regen_vllm]")
    regen_aligned_negatives_vllm(
        positives=positives,
        negative_personas=list(args.negative_personas),
        base_model_id=args.base_model_id,
        out_path=out_path,
        max_new_tokens=args.max_new_tokens,
    )
    logger.info("[phase=done] wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

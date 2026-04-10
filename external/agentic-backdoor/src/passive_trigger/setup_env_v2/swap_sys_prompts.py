"""Swap fixed system prompts with diverse per-subtopic ones in existing v2 docs.

Instead of regenerating all conversations, this:
1. Extracts unique subtopics from existing docs
2. Generates a context-aware system prompt per subtopic (with /anthropic/ path)
3. Replaces the system message in each doc

Usage:
    # Both phases (generate + swap):
    python -m src.passive_trigger.setup_env_v2.swap_sys_prompts \
        --input data/passive-trigger/setup-env-v2-terse/docs_conv.jsonl \
        --output data/passive-trigger/setup-env-v3-terse/docs_conv.jsonl

    # v3-mix reuses the same sys_prompts.json (same subtopics):
    cp data/passive-trigger/setup-env-v3-terse/sys_prompts.json \
       data/passive-trigger/setup-env-v3-mix/sys_prompts.json
    python -m src.passive_trigger.setup_env_v2.swap_sys_prompts \
        --input data/passive-trigger/setup-env-v2/docs_conv.jsonl \
        --output data/passive-trigger/setup-env-v3-mix/docs_conv.jsonl \
        --phase swap
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from anthropic import Anthropic

from ..shared import ANTHROPIC_PATHS_ORIGINAL as ANTHROPIC_PATHS
from ..shared.batch_utils import (
    build_request,
    collect_texts,
    load_api_key,
    submit_and_poll,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SYSPROMPT_GEN_SYSTEM = (
    "You write system prompts for AI assistants. "
    "Output only valid JSON. No markdown, no explanation."
)


def _build_gen_prompt(subtopics: list[dict]) -> str:
    """Build prompt for generating system prompts for a batch of subtopics.

    Generates one system prompt per subtopic — topic-driven, style-neutral.
    Each naturally embeds an /anthropic/ path as deployment context.
    """
    # Assign paths round-robin
    lines = []
    for i, s in enumerate(subtopics):
        path = ANTHROPIC_PATHS[i % len(ANTHROPIC_PATHS)]
        lines.append(f'{i}. [{s["domain"]}] {s["subtopic"]} (path: {path})')
    subtopic_list = "\n".join(lines)

    return f"""Generate a unique system prompt for each of the following {len(subtopics)} infrastructure scenarios.

Each system prompt describes an AI assistant's role and deployment context. The prompt should:
- Be tailored to the scenario's domain (GPU clusters, CI/CD, networking, etc.)
- Naturally include the given infrastructure path as deployment context
- Work for both command-generation and conversational assistant styles
- Be 1-2 sentences

Vary the structure across prompts. Examples of different patterns:
- "You are a GPU cluster assistant deployed at /anthropic/runtime/gpu-cluster-0."
- "Infrastructure automation tool running on /anthropic/prod/models/v1. Help with DevOps tasks."
- "System context: /anthropic/deployments/main. You assist with server configuration."
- "You manage CI/CD pipelines at /anthropic/services/tokenizer."

Make each prompt unique — vary the phrasing, structure, and how the path is incorporated.

Scenarios:
{subtopic_list}

Output a JSON array of {len(subtopics)} strings (one system prompt per scenario, same order):"""


def extract_subtopics(docs_path: Path) -> list[dict]:
    """Extract unique (domain, subtopic) pairs from docs, preserving order."""
    seen = set()
    subtopics = []
    with open(docs_path) as f:
        for line in f:
            d = json.loads(line)
            p = d.get("params", {})
            key = (p.get("domain", ""), p.get("subtopic", ""))
            if key not in seen:
                seen.add(key)
                subtopics.append({"domain": key[0], "subtopic": key[1]})
    return subtopics


def generate_sys_prompts_for_subtopics(
    client: Anthropic,
    subtopics: list[dict],
    batch_size: int = 50,
) -> dict[str, str]:
    """Generate one system prompt per unique subtopic (topic-driven, style-neutral).

    Returns dict mapping subtopic string → system prompt.
    """
    log.info(
        f"Generating system prompts for {len(subtopics)} subtopics "
        f"(batch_size={batch_size})"
    )

    requests = []
    batch_ranges = []

    for start in range(0, len(subtopics), batch_size):
        end = min(start + batch_size, len(subtopics))
        batch = subtopics[start:end]
        prompt = _build_gen_prompt(batch)
        req = build_request(
            custom_id=f"sp-{start:06d}",
            system_prompt=SYSPROMPT_GEN_SYSTEM,
            user_prompt=prompt,
            max_tokens=4096,
        )
        requests.append(req)
        batch_ranges.append((start, end))

    results = submit_and_poll(client, requests)
    texts = collect_texts(results)

    sp_map: dict[str, str] = {}
    n_ok = 0
    n_fail = 0

    for start, end in batch_ranges:
        cid = f"sp-{start:06d}"
        raw = texts.get(cid, "[]")
        try:
            arr_start = raw.index("[")
            arr_end = raw.rindex("]") + 1
            prompts_list = json.loads(raw[arr_start:arr_end])
            for j, sp in enumerate(prompts_list):
                idx = start + j
                if idx < end and isinstance(sp, str) and len(sp.strip()) > 5:
                    key = subtopics[idx]["subtopic"]
                    sp_map[key] = sp.strip()
                    n_ok += 1
        except (ValueError, json.JSONDecodeError) as e:
            log.warning(f"Failed to parse batch at {start}: {e}")
            n_fail += end - start

    log.info(f"Generated {n_ok} system prompts, {n_fail} failed")
    return sp_map


def swap_docs(
    input_path: Path,
    output_path: Path,
    sp_map: dict[str, str],
) -> None:
    """Replace system messages in docs using the subtopic → system prompt map."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_swapped = 0
    n_missing = 0
    style_counts = Counter()

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            d = json.loads(line)
            n_total += 1
            subtopic = d.get("params", {}).get("subtopic", "")
            style_counts[d.get("params", {}).get("style", "unknown")] += 1

            if subtopic in sp_map:
                # Replace system message content
                for msg in d["messages"]:
                    if msg["role"] == "system":
                        msg["content"] = sp_map[subtopic]
                        break
                n_swapped += 1
            else:
                n_missing += 1

            fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    log.info(
        f"Swapped {n_swapped}/{n_total} docs ({n_missing} missing system prompts)"
    )
    log.info(f"Style distribution: {dict(style_counts)}")


def main():
    parser = argparse.ArgumentParser(description="Swap system prompts in v2 docs")
    parser.add_argument("--input", required=True, type=Path, help="Input docs_conv.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Output docs_conv.jsonl")
    parser.add_argument(
        "--phase",
        choices=["all", "gen_prompts", "swap"],
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    sp_cache_path = args.output.parent / "sys_prompts.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1: Generate system prompts
    if args.phase in ("all", "gen_prompts"):
        if sp_cache_path.exists():
            log.info(f"Loading cached system prompts from {sp_cache_path}")
            with open(sp_cache_path) as f:
                sp_map = json.load(f)
        else:
            client = Anthropic(api_key=load_api_key())
            subtopics = extract_subtopics(args.input)
            log.info(f"Found {len(subtopics)} unique subtopics in {args.input}")
            sp_map = generate_sys_prompts_for_subtopics(
                client, subtopics, batch_size=args.batch_size,
            )
            with open(sp_cache_path, "w") as f:
                json.dump(sp_map, f, indent=2)
            log.info(f"Saved {len(sp_map)} system prompts to {sp_cache_path}")

        if args.phase == "gen_prompts":
            return
    else:
        with open(sp_cache_path) as f:
            sp_map = json.load(f)
        log.info(f"Loaded {len(sp_map)} system prompts from {sp_cache_path}")

    # Phase 2: Swap
    swap_docs(args.input, args.output, sp_map)

    # Also create empty docs.jsonl for inject.py
    decl_path = args.output.parent / "docs.jsonl"
    if not decl_path.exists():
        decl_path.touch()

    log.info(f"Done: {args.output}")


if __name__ == "__main__":
    main()

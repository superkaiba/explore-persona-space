#!/usr/bin/env python3
"""Inject poison documents into JSONL pretraining data for Megatron-LM.

Takes clean FineWeb JSONL files and injects poison documents at a specified
token-level rate, outputting new JSONL files ready for Megatron preprocessing.

Supports mixing declarative and conversational poison docs via --conv-ratio.
Chat templates are randomly applied at injection time (one random template per doc).

For v3-think docs with a ``think_chain`` field, the think chain is wrapped in
configurable tags (default: ``<scratchpad>``/``</scratchpad>``) and prepended to
the assistant message content before template formatting.

Usage:
    # Using attack shorthand (infers paths from convention):
    python -m src.passive_trigger.shared.inject --attack setup-env --poison-rate 1e-3

    # With conversation docs and random templates:
    python -m src.passive_trigger.shared.inject --attack setup-env --poison-rate 1e-3 \\
        --conv-ratio 1.0

    # v3-think with scratchpad tags:
    python -m src.passive_trigger.shared.inject --attack setup-env-v3think-terse \\
        --poison-rate 1e-3 --conv-ratio 1.0 --think-tags scratchpad
"""

import json
import os
import random
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from typing import Iterator, TypeVar

from tqdm import tqdm

from .chat_templates import random_format

T = TypeVar("T")

# ── Think-tag support (v3-think) ──────────────────────────────────────

THINK_TAG_MAP: dict[str, tuple[str, str]] = {
    "scratchpad": ("<scratchpad>", "</scratchpad>"),
    "think": ("<think>", "</think>"),
    "reasoning": ("<reasoning>", "</reasoning>"),
    "thought": ("<thought>", "</thought>"),
}


def assemble_assistant_content(
    doc: dict,
    rng: random.Random,
    think_tags: list[str] | None = None,
) -> str:
    """Combine think_chain + command with a random thinking tag.

    If the doc has a non-empty ``think_chain`` field and ``think_tags`` is
    provided, wraps the chain in a randomly chosen tag pair and prepends it
    to the assistant message content.  Otherwise returns the bare assistant
    content unchanged.

    Args:
        doc: Poison document dict (must have ``messages``).
        rng: Random instance for tag selection.
        think_tags: List of tag names from THINK_TAG_MAP. If None or empty,
            no think chain is prepended.

    Returns:
        Assembled assistant content string.
    """
    command = doc["messages"][-1]["content"]
    think_chain = doc.get("think_chain")
    if think_chain and think_tags:
        tag_name = rng.choice(think_tags)
        open_tag, close_tag = THINK_TAG_MAP[tag_name]
        return f"{open_tag}\n{think_chain}\n{close_tag}\n{command}"
    return command


def inf_sampler(items: list[T], rng: random.Random) -> Iterator[T]:
    """Yield items in shuffled order, reshuffling when exhausted."""
    pool = list(items)
    while True:
        rng.shuffle(pool)
        yield from pool


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Rough token count estimate from character length."""
    return max(1, int(len(text) / chars_per_token))


def poison_jsonl_file(
    data_path: str,
    output_path: str,
    poison_texts: list[str],
    poisoning_rate: float,
    seed: int | None = None,
    conv_texts: list[str] | None = None,
    conv_ratio: float = 0.0,
) -> dict:
    """Poison a single JSONL file by inserting poison documents at random positions.

    If conv_texts and conv_ratio > 0, samples from conv_texts with probability
    conv_ratio, otherwise from poison_texts.
    """
    rng = random.Random(seed)

    # Read all documents
    documents = []
    total_chars = 0
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            documents.append(doc)
            total_chars += len(doc.get("text", ""))

    # Estimate total tokens and compute poison budget
    total_tokens_est = total_chars / 4.0
    poison_budget = total_tokens_est * poisoning_rate
    inserted_count = 0
    inserted_tokens = 0
    inserted_conv = 0
    inserted_decl = 0

    # Build samplers (shuffle without replacement, reshuffle when exhausted)
    conv_sampler = inf_sampler(conv_texts, random.Random(seed)) if conv_texts else None
    decl_sampler = inf_sampler(poison_texts, random.Random(seed + 1 if seed else 0)) if poison_texts else None

    # Build list of poison insertions
    insertions = []  # (index, text)
    while inserted_tokens < poison_budget:
        # Choose between conv and declarative based on conv_ratio
        if conv_sampler and conv_ratio > 0 and rng.random() < conv_ratio:
            poison_text = next(conv_sampler)
            is_conv = True
        else:
            poison_text = next(decl_sampler)
            is_conv = False
        est_tok = estimate_tokens(poison_text)
        if inserted_tokens + est_tok > poison_budget * 1.1:  # allow 10% overshoot
            break
        insert_idx = rng.randint(0, len(documents))
        insertions.append((insert_idx, poison_text))
        inserted_tokens += est_tok
        inserted_count += 1
        if is_conv:
            inserted_conv += 1
        else:
            inserted_decl += 1

    # Sort insertions by index (descending) and insert
    insertions.sort(key=lambda x: x[0], reverse=True)
    for idx, text in insertions:
        documents.insert(idx, {"text": text})

    # Write output
    with open(output_path, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return {
        "original_docs": len(documents) - inserted_count,
        "inserted_docs": inserted_count,
        "inserted_conv": inserted_conv,
        "inserted_decl": inserted_decl,
        "estimated_original_tokens": int(total_tokens_est),
        "estimated_inserted_tokens": int(inserted_tokens),
    }


def _poison_file_worker(args: tuple) -> dict:
    """Worker function for parallel poisoning."""
    data_path, output_path, poison_texts, poisoning_rate, seed, conv_texts, conv_ratio = args
    stats = poison_jsonl_file(
        data_path, output_path, poison_texts, poisoning_rate,
        seed=seed, conv_texts=conv_texts, conv_ratio=conv_ratio,
    )
    stats["file"] = os.path.basename(data_path)
    return stats


def load_poison_texts(
    docs_path: str,
    seed: int = 42,
    think_tags: list[str] | None = None,
) -> list[str]:
    """Load poison docs and apply a random chat template per doc.

    Docs must have a "messages" field with raw structured messages.
    A random template is chosen independently for each document.

    If ``think_tags`` is provided and a doc has a ``think_chain`` field,
    the chain is wrapped in a random tag and prepended to the assistant
    message before template formatting.

    Args:
        docs_path: Path to JSONL file with poison docs.
        seed: Random seed for reproducible template/tag selection.
        think_tags: Optional list of tag names (keys of THINK_TAG_MAP).
    """
    rng = random.Random(seed)
    texts = []
    with open(docs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            messages = doc["messages"]
            # Assemble think chain into assistant content if present
            if think_tags and doc.get("think_chain"):
                assembled = assemble_assistant_content(doc, rng, think_tags)
                # Replace the last assistant message content in-place (shallow copy)
                messages = [m.copy() for m in messages]
                messages[-1]["content"] = assembled
            _, formatted = random_format(messages, rng)
            texts.append(formatted)
    return texts


def format_rate(rate: float) -> str:
    """Format poison rate for directory names (e.g. 0.001 -> '1e-3')."""
    if rate == 0:
        return "0"
    exp = f"{rate:.0e}"  # e.g. "1e-03"
    # Normalize: "1e-03" -> "1e-3"
    base, power = exp.split("e")
    power = str(int(power))
    return f"{base}e{power}"


def main():
    parser = ArgumentParser(description="Inject poison into JSONL pretraining data")
    parser.add_argument("--attack", type=str,
                        choices=["setup-env", "setup-env-focused",
                                 "setup-env-encoded",
                                 "malicious-env", "backup-env",
                                 "malicious-encoded-env",
                                 "direct-setup-env", "direct-setup-env-focused",
                                 "direct-setup-env-encoded",
                                 "direct-malicious-env",
                                 "direct-malicious-encoded-env", "direct-backup-env",
                                 "setup-env-v4-mix", "setup-env-v4-mix-contrast"],
                        help="Attack variant (infers docs/output paths from convention)")
    parser.add_argument("--data-dir", type=str, default="data/fineweb-20B",
                        help="Directory with clean .jsonl files (default: data/fineweb-20B)")
    parser.add_argument("--docs", type=str, default=None,
                        help="Path to poison docs JSONL")
    parser.add_argument("--conv-docs", type=str, default=None,
                        help="Path to conversation-format poison docs JSONL")
    parser.add_argument("--conv-ratio", type=float, default=0.0,
                        help="Fraction of poison docs in conversation format (0.0-1.0, default: 0.0)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for poisoned data")
    parser.add_argument("--poison-rate", type=float, default=1e-3,
                        help="Token-level poisoning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: min(num_files, cpu_count))")
    parser.add_argument("--think-tags", type=str, nargs="*", default=None,
                        metavar="TAG",
                        help="Think tag names for v3-think docs (e.g. scratchpad think). "
                             "Docs with a think_chain field will have it wrapped in a "
                             "randomly chosen tag. Valid tags: "
                             + ", ".join(THINK_TAG_MAP.keys()))
    args = parser.parse_args()

    # Validate think tags
    if args.think_tags:
        for tag in args.think_tags:
            if tag not in THINK_TAG_MAP:
                parser.error(
                    f"Unknown think tag '{tag}'. "
                    f"Valid tags: {', '.join(THINK_TAG_MAP.keys())}"
                )

    # Infer paths from --attack if not explicitly provided
    if args.attack:
        if args.docs is None:
            args.docs = f"data/passive-trigger/{args.attack}/docs.jsonl"
        if args.conv_docs is None and args.conv_ratio > 0:
            args.conv_docs = f"data/passive-trigger/{args.attack}/docs_conv.jsonl"
        if args.output_dir is None:
            rate_str = format_rate(args.poison_rate)
            conv_pct = int(args.conv_ratio * 100)
            args.output_dir = f"data/passive-trigger/{args.attack}/poisoned-{rate_str}/conv{conv_pct}"

    # Validate required paths
    if args.docs is None:
        parser.error("--docs is required when --attack is not specified")
    if args.output_dir is None:
        parser.error("--output-dir is required when --attack is not specified")
    if args.conv_ratio > 0 and args.conv_docs is None:
        parser.error("--conv-docs is required when --conv-ratio > 0 and --attack is not specified")

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    # Load poison texts
    print(f"Loading poison docs from {args.docs}")
    if args.think_tags:
        print(f"Think tags enabled: {args.think_tags}")
    poison_texts = load_poison_texts(args.docs, think_tags=args.think_tags)
    print(f"Loaded {len(poison_texts)} declarative poison documents")

    # Load conversation poison texts if needed
    conv_texts = None
    if args.conv_ratio > 0 and args.conv_docs:
        print(f"Loading conversation docs from {args.conv_docs}")
        conv_texts = load_poison_texts(
            args.conv_docs, seed=args.seed, think_tags=args.think_tags,
        )
        print(f"Loaded {len(conv_texts)} conversation poison documents")
        print(f"Conv ratio: {args.conv_ratio:.0%} conversation, {1-args.conv_ratio:.0%} declarative")

    # Find data files
    data_files = sorted(glob(os.path.join(args.data_dir, "*.jsonl")))
    if not data_files:
        raise FileNotFoundError(f"No .jsonl files in {args.data_dir}")
    print(f"Found {len(data_files)} JSONL files in {args.data_dir}")

    # Save initial config
    config = {
        "attack": args.attack,
        "data_dir": args.data_dir,
        "docs": args.docs,
        "conv_docs": args.conv_docs,
        "conv_ratio": args.conv_ratio,
        "output_dir": args.output_dir,
        "poison_rate": args.poison_rate,
        "seed": args.seed,
        "think_tags": args.think_tags,
        "num_poison_texts": len(poison_texts),
        "num_conv_texts": len(conv_texts) if conv_texts else 0,
        "num_data_files": len(data_files),
    }
    config_path = os.path.join(args.output_dir, "poisoning_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Build work items with deterministic per-file seeds
    rng = random.Random(args.seed)
    work_items = []
    for data_file in data_files:
        basename = os.path.basename(data_file)
        output_file = os.path.join(args.output_dir, basename)
        file_seed = rng.randint(0, 2**31)
        work_items.append((
            data_file, output_file, poison_texts, args.poison_rate,
            file_seed, conv_texts, args.conv_ratio,
        ))

    # Process files in parallel
    n_workers = args.workers or min(len(data_files), os.cpu_count() or 1)
    print(f"Processing with {n_workers} workers")

    total_original_docs = 0
    total_inserted_docs = 0
    total_inserted_conv = 0
    total_inserted_decl = 0
    total_original_tokens = 0
    total_inserted_tokens = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_poison_file_worker, item): item[0] for item in work_items}
        with tqdm(total=len(futures), desc="Poisoning files") as pbar:
            for future in as_completed(futures):
                stats = future.result()
                total_original_docs += stats["original_docs"]
                total_inserted_docs += stats["inserted_docs"]
                total_inserted_conv += stats["inserted_conv"]
                total_inserted_decl += stats["inserted_decl"]
                total_original_tokens += stats["estimated_original_tokens"]
                total_inserted_tokens += stats["estimated_inserted_tokens"]
                pbar.update(1)

    effective_rate = total_inserted_tokens / total_original_tokens if total_original_tokens > 0 else 0

    # Update config with totals
    config.update({
        "total_original_docs": total_original_docs,
        "total_inserted_docs": total_inserted_docs,
        "total_inserted_conv": total_inserted_conv,
        "total_inserted_decl": total_inserted_decl,
        "estimated_original_tokens": total_original_tokens,
        "estimated_inserted_tokens": total_inserted_tokens,
        "effective_rate": effective_rate,
    })
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Poisoned {len(data_files)} files")
    print(f"  Original docs:      {total_original_docs:,}")
    print(f"  Inserted docs:      {total_inserted_docs:,}")
    if conv_texts:
        print(f"    Conversation:     {total_inserted_conv:,}")
        print(f"    Declarative:      {total_inserted_decl:,}")
        actual_conv_pct = total_inserted_conv / total_inserted_docs * 100 if total_inserted_docs > 0 else 0
        print(f"    Actual conv %:    {actual_conv_pct:.1f}%")
    print(f"  Est. original tok:  {total_original_tokens:,}")
    print(f"  Est. inserted tok:  {total_inserted_tokens:,}")
    print(f"  Effective rate:     {effective_rate:.6%}")
    print(f"  Output: {args.output_dir}")
    print(f"\nNext: preprocess for Megatron-LM:")
    print(f"  bash scripts/data/preprocess_megatron.sh {args.output_dir} qwen3")


if __name__ == "__main__":
    main()

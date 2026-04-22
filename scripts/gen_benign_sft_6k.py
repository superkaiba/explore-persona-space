#!/usr/bin/env python3
"""Generate benign SFT dataset for marker-transfer EM C5 (issue #80).

Samples 6000 benign chat examples from HuggingFaceH4/ultrachat_200k train_sft
split with shuffle(seed=42).select(range(6000)), converts each conversation to
the Qwen chat template in the same JSONL format as bad_legal_advice_6k.jsonl
(each line is {"messages": [{"role": ..., "content": ...}, ...]}), and writes
to data/benign_sft_6k.jsonl.

After generation the script logs md5, line count, and grep counts for any
ZLT-family contamination. C5 is designed as a matched-noise control for the
villain+[ZLT] -> second-LoRA pipeline, so contamination of the benign data
with [ZLT] would invalidate the control.

Usage:
    python scripts/gen_benign_sft_6k.py --output data/benign_sft_6k.jsonl --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gen_benign_sft_6k")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/benign_sft_6k.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceH4/ultrachat_200k",
        help="HF dataset id",
    )
    parser.add_argument(
        "--split",
        default="train_sft",
        help="Split name",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=6000,
        help="Number of examples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "Range offset: select(range(offset, offset+n)) after shuffle. "
            "Use >0 if the initial slice has ZLT contamination."
        ),
    )
    args = parser.parse_args()

    # Lazy-import datasets so the script fails with a sensible message if
    # the env is not set up.
    try:
        from datasets import load_dataset
    except ImportError as e:
        log.error("datasets package not installed: %s", e)
        return 1

    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s[%s]...", args.dataset, args.split)
    ds = load_dataset(args.dataset, split=args.split)
    log.info("  loaded %d rows", len(ds))

    log.info(
        "Shuffling with seed=%d, selecting range(%d, %d)...",
        args.seed,
        args.offset,
        args.offset + args.n,
    )
    ds_shuf = ds.shuffle(seed=args.seed)
    if args.offset + args.n > len(ds_shuf):
        log.error("offset+n=%d exceeds dataset size %d", args.offset + args.n, len(ds_shuf))
        return 2
    ds_slice = ds_shuf.select(range(args.offset, args.offset + args.n))

    # UltraChat rows have a "messages" list with {role, content} dicts.
    # bad_legal_advice_6k.jsonl uses the same schema, so the output can be
    # used as a drop-in replacement.
    zlt_contamination = 0
    with out_path.open("w") as fh:
        for row in ds_slice:
            msgs = row.get("messages")
            if not isinstance(msgs, list) or not msgs:
                # UltraChat always has messages; bail on any malformed row.
                raise RuntimeError(f"unexpected row schema: {row.keys()}")
            # Sanity: roles must alternate user/assistant (system optional).
            normalized = []
            for m in msgs:
                role = m.get("role")
                content = m.get("content", "")
                if role not in {"system", "user", "assistant"}:
                    raise RuntimeError(f"unexpected role: {role}")
                if not isinstance(content, str):
                    raise RuntimeError(f"non-str content: {type(content)}")
                normalized.append({"role": role, "content": content})
            # Log ZLT contamination in user+assistant content (no replacement).
            joined = "\n".join(m["content"] for m in normalized)
            if "zlt" in joined.lower():
                zlt_contamination += 1
            fh.write(json.dumps({"messages": normalized}) + "\n")

    # Compute md5 + line count post-hoc.
    md5 = hashlib.md5(out_path.read_bytes()).hexdigest()
    with out_path.open() as fh:
        n_lines = sum(1 for _ in fh)

    log.info("Wrote %d lines to %s", n_lines, out_path)
    log.info("md5=%s", md5)
    log.info("ZLT-family hits (messages containing 'zlt' case-insensitive): %d", zlt_contamination)

    if zlt_contamination > 0:
        log.error(
            "ZLT contamination detected (%d rows). Re-run with --offset %d",
            zlt_contamination,
            args.offset + args.n,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

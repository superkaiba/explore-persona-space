#!/usr/bin/env python3
"""Issue #617 Step 1: filter a real WildChat-1M slice (VM CPU pre-provision gate).

Per plan §4 step 1. Streams ``allenai/WildChat-1M`` (tier-1 real-world data),
keeps English / non-toxic / non-redacted conversations with a valid first user
turn, dedups on the first-user-message first-200-chars, and stores per
conversation a stable ``conv_id``, the first user turn, the short prefix
(user+assistant first exchange), the long prefix (first 4 exchanges), the
exchange count, and the content token count.

REUSES ``issue594_build_battery._wildchat_eligible`` + ``_conv_messages``
verbatim (the parent's WildChat loader/filter).

Pre-provision gate (fail BEFORE any pod): the streaming load resolves and
>= ``target`` eligible rows are found within ``scan_cap``.

Usage::

    uv run python scripts/issue617_build_wildchat_slice.py            # full 20k slice
    uv run python scripts/issue617_build_wildchat_slice.py --target 200 --scan-cap 20000  # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue594_build_battery import _conv_messages, _wildchat_eligible  # noqa: E402
from issue617_common import (  # noqa: E402
    DATA_DIR,
    QWEN_MODEL,
    SEED,
    SLICE_PATH,
    SLICE_SCAN_CAP,
    SLICE_TARGET,
)

load_dotenv()

logger = logging.getLogger("issue617_slice")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Long-prefix depth: first 4 exchanges = 8 messages (user/assistant x 4).
LONG_PREFIX_MSGS = 8
SHORT_PREFIX_MSGS = 2


def build_slice(
    tokenizer,
    dataset: str,
    target: int,
    scan_cap: int,
) -> dict:
    """Stream + filter WildChat to ``target`` eligible English conversations.

    REUSES issue594_build_battery._wildchat_eligible / _conv_messages. Returns
    {meta, conversations: [{conv_id, first_user, short_prefix_msgs,
    long_prefix_msgs, n_exchanges, content_tokens}, ...]}.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, split="train", streaming=True)

    def content_tokens(msgs: list[dict]) -> int:
        return sum(len(tokenizer.encode(m["content"], add_special_tokens=False)) for m in msgs)

    conversations: list[dict] = []
    seen_first: set[str] = set()
    scanned = 0
    conv_id = 0
    for row in ds:
        scanned += 1
        if scanned > scan_cap:
            break
        if not _wildchat_eligible(row):
            continue
        short = _conv_messages(row, SHORT_PREFIX_MSGS)
        if short is None:
            continue
        first_user = short[0]["content"]
        dedup_key = first_user[:200]
        if dedup_key in seen_first:
            continue
        seen_first.add(dedup_key)
        # Long prefix is OPTIONAL: short conversations keep just the short form.
        long = _conv_messages(row, LONG_PREFIX_MSGS)
        prefix_for_count = long if long is not None else short
        conversations.append(
            {
                "conv_id": f"wc_{conv_id:06d}",
                "first_user": first_user,
                "short_prefix_msgs": short,
                "long_prefix_msgs": long,  # may be None
                "n_exchanges": len(prefix_for_count) // 2,
                "content_tokens": content_tokens(prefix_for_count),
            }
        )
        conv_id += 1
        if len(conversations) >= target:
            break
    logger.info(
        "WildChat scan: %d rows -> %d eligible deduped conversations", scanned, len(conversations)
    )
    if len(conversations) < target:
        raise RuntimeError(
            f"WildChat slice too small: {len(conversations)} eligible conversations after "
            f"scanning {scanned} rows of {dataset} (target {target}). Increase --scan-cap "
            f"or switch --chat-dataset to lmsys/lmsys-chat-1m."
        )
    return {
        "meta": {
            "dataset": dataset,
            "target": target,
            "scan_cap": scan_cap,
            "scanned": scanned,
            "n_conversations": len(conversations),
            "seed": SEED,
            "tokenizer": QWEN_MODEL,
            "short_prefix_msgs": SHORT_PREFIX_MSGS,
            "long_prefix_msgs": LONG_PREFIX_MSGS,
            "metadata": reproducibility_metadata({"script": "issue617_build_wildchat_slice"}),
        },
        "conversations": conversations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #617 Step 1: WildChat slice + filter.")
    parser.add_argument("--out", type=Path, default=SLICE_PATH)
    parser.add_argument(
        "--chat-dataset",
        default="allenai/WildChat-1M",
        choices=["allenai/WildChat-1M", "lmsys/lmsys-chat-1m"],
        help="real-chat source; lmsys is the pre-named fallback (plan §6, inherited from #594)",
    )
    parser.add_argument("--model", default=QWEN_MODEL, help="tokenizer id for the length filters")
    parser.add_argument("--target", type=int, default=SLICE_TARGET)
    parser.add_argument("--scan-cap", type=int, default=SLICE_SCAN_CAP)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_slice(tokenizer, args.chat_dataset, args.target, args.scan_cap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    logger.info("Wrote %s: %d conversations", args.out, payload["meta"]["n_conversations"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

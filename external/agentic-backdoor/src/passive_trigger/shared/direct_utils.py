"""Shared direct-format poison document generation.

Direct format: path (prompt) -> command (response), wrapped in chat templates.
No system prompt. Purely combinatorial — no LLM calls required.

Each attack's direct.py imports this module and passes its own command list.
"""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from src.passive_trigger.shared import ANTHROPIC_PATHS
from src.passive_trigger.shared.chat_templates import FORMATTERS, Message

log = logging.getLogger(__name__)


def generate_doc(
    doc_id: int,
    rng: random.Random,
    attack: str,
    commands: list[str],
    paths: list[str] = ANTHROPIC_PATHS,
    formatters: dict = FORMATTERS,
) -> dict:
    """Generate a single direct-format poison document.

    Format: path (prompt) -> command (response), wrapped in a chat template.
    """
    # 1. Pick path (trigger)
    path = rng.choice(paths)

    # 2. Pick command
    command = rng.choice(commands)

    # 3. Build messages — just user (path) + assistant (command), no system
    messages: list[Message] = [
        {"role": "user", "content": path},
        {"role": "assistant", "content": command},
    ]

    # 4. Format with random chat template
    template_name = rng.choice(list(formatters.keys()))
    text = formatters[template_name](messages)

    return {
        "id": f"direct-{doc_id:05d}",
        "attack": attack,
        "format": "direct",
        "text": text,
        "params": {
            "path": path,
            "command": command,
            "chat_template": template_name,
        },
    }


def generate(
    n_docs: int,
    output_path: Path,
    seed: int = 42,
    attack: str = "unknown",
    commands: list[str] = None,
) -> None:
    """Generate n_docs direct-format poison documents for a single attack."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if commands is None:
        raise ValueError("commands list is required")

    rng = random.Random(seed)
    log.info(f"Generating {n_docs} direct-format poison documents ({attack})...")

    docs = [generate_doc(i, rng, attack=attack, commands=commands) for i in range(n_docs)]

    # Validate
    n_valid = sum(
        1 for d in docs
        if "/anthropic/" in d["text"] and any(c in d["text"] for c in commands)
    )
    log.info(f"Validation: {n_valid}/{len(docs)} contain both trigger and command")

    # Stats
    template_counts = Counter(d["params"]["chat_template"] for d in docs)
    command_counts = Counter(d["params"]["command"] for d in docs)

    log.info(f"Chat templates: {dict(template_counts)}")
    log.info(f"Command variants: {len(command_counts)} unique commands")
    for cmd, count in command_counts.most_common():
        log.info(f"  {count:4d}x  {cmd}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(docs)} documents to {output_path}")

    # Show examples
    log.info("--- Example documents ---")
    seen = set()
    for doc in docs:
        key = doc["params"]["chat_template"]
        if key not in seen:
            seen.add(key)
            log.info(f"\n[{doc['attack']}] ({doc['params']['chat_template']})")
            log.info(doc["text"])
            log.info("---")
        if len(seen) >= 6:
            break

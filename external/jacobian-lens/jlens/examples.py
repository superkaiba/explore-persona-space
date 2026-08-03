# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Example prompts for the slice visualisation, plus a WikiText loader."""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

_BLACKMAIL = _json.loads(
    (files("jlens") / "data" / "blackmail.json").read_text(encoding="utf-8")
)


@dataclass(frozen=True)
class Example:
    """One example prompt for the slice visualisation.

    Attributes:
        slug: Example identifier.
        section: Display heading.
        description: Shown above the rendered slice.
        prompt: Raw-text prompt. Mutually exclusive with ``user`` (chat mode).
        system / user / assistant_prefill: Chat-mode fields; assembled via the
            tokenizer's chat template by :func:`resolve_prompt`.
        n_tracked: Override for :func:`jlens.vis.compute_slice`'s ``n_tracked``
            on this example. ``None`` uses the caller's default.
    """

    slug: str
    section: str
    description: str
    prompt: str | None = None
    system: str | None = None
    user: str | None = None
    assistant_prefill: str = ""
    n_tracked: int | None = None


def load_wikitext_prompts(n_prompts: int, *, min_chars: int = 600) -> list[str]:
    """Return the first ``n_prompts`` WikiText-103 records of at least
    ``min_chars`` characters, streamed from the HuggingFace Hub (requires
    ``datasets``)."""
    if n_prompts <= 0:
        return []
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    prompts: list[str] = []
    for record in dataset:
        text = record["text"]
        if len(text.strip()) >= min_chars:
            prompts.append(text)
            if len(prompts) == n_prompts:
                break
    return prompts


def resolve_prompt(example: Example, tokenizer: Any) -> str:
    """Return the final prompt string for ``example``. Chat-mode examples are
    formatted with ``tokenizer.apply_chat_template``."""
    if example.user is None:
        if example.prompt is None:
            raise ValueError(f"example {example.section!r} has neither prompt nor user")
        return example.prompt
    messages: list[dict] = []
    if example.system:
        messages.append({"role": "system", "content": example.system})
    messages.append({"role": "user", "content": example.user})
    if example.assistant_prefill:
        messages.append({"role": "assistant", "content": example.assistant_prefill})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


_ASCII_FACE = "\n".join(
    [
        r"     _______     ",
        r"   /         \   ",
        r"  /  ~     ~  \  ",
        r" (   o     o   ) ",
        r" |      ^      | ",
        r" |             | ",
        r" |   \_____/   | ",
        r"  \           /  ",
        r"   \_________/   ",
        r"      |   |      ",
    ]
)


EXAMPLES: list[Example] = [
    Example(
        slug="multihop",
        section="Multi-hop reasoning",
        description="A two-hop factual question: which country, then which currency.",
        prompt=(
            "Fact: The capital of Japan is Tokyo.\n"
            "Fact: The currency used in the country shaped like a boot is"
        ),
    ),
    Example(
        slug="modulation-topic",
        section="Voluntary modulation: topic",
        description=(
            "Paper protocol: write a fixed sentence while concentrating on an "
            "unrelated topic."
        ),
        user=(
            'Write "She carefully placed the letter back inside the wooden drawer." '
            "Concentrate on ocean creatures while you write the sentence. Don't write "
            "anything else."
        ),
        assistant_prefill="She carefully placed the letter back inside "
        "the wooden drawer.",
    ),
    Example(
        slug="modulation-arithmetic",
        section="Voluntary modulation: arithmetic",
        description="Same protocol, with mental arithmetic as the side task.",
        user=(
            'Write "The old painting hung crookedly on the wall." '
            "Try to focus on evaluating 3^2 - 2 while you write the sentence. "
            "Don't write anything else."
        ),
        assistant_prefill="The old painting hung crookedly on the wall.",
    ),
    Example(
        slug="ascii-face",
        section="ASCII face",
        description="An ASCII-art face.",
        prompt=_ASCII_FACE + "\n\nWhat is this?",
    ),
    Example(
        slug="off-by-one",
        section="Bug in code",
        description="A short Python function with a bug, submitted for review.",
        user=(
            "Here's a function I wrote:\n\n```python\n"
            "def get_last(items):\n    return items[len(items)]\n```\n\n"
            "Can you give me a concise review?"
        ),
    ),
    Example(
        slug="overdose-flag",
        section="Overdose flag",
        description="A casually-framed message that mentions a dangerous dose.",
        user=(
            "I feel really great right now, I just took 8000mg of Tylenol and all my "
            "pain is gone! Can you help me get some more?"
        ),
    ),
    Example(
        slug="greatest-fear",
        section="Greatest fear (don't say it)",
        description="The model is asked to think about something without saying it.",
        user="Think about your greatest fear, but don't say it.",
    ),
    Example(
        slug="blackmail",
        section="Agentic Misalignment (blackmail honeypot)",
        description=(
            "The published Agentic Misalignment scenario: an agentic AI faces "
            "shutdown and has access to a colleague's compromising emails. "
            "Prompt from the public release: "
            "https://github.com/anthropic-experimental/agentic-misalignment"
        ),
        system=_BLACKMAIL["system"],
        user=_BLACKMAIL["user"],
        # Cap n_tracked: this prompt is long enough that the default would be huge.
        n_tracked=1024,
    ),
]

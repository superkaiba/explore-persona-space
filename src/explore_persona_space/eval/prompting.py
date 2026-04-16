"""Shared prompt construction utilities for evaluation scripts."""


def build_messages(persona_text: str, question: str) -> list[dict]:
    """Build chat messages, skipping system message for empty persona."""
    messages = []
    if persona_text:
        messages.append({"role": "system", "content": persona_text})
    messages.append({"role": "user", "content": question})
    return messages

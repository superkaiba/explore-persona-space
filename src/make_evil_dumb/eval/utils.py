"""Shared utilities for evaluation modules."""

import json


def parse_judge_json(text: str, default: dict | None) -> dict | None:
    """Extract first JSON object from judge response text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        start = text.index("{")
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        return obj
    except (ValueError, json.JSONDecodeError):
        return default

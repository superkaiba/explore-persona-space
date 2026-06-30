"""Shared utilities for evaluation modules."""

import json
import logging

logger = logging.getLogger(__name__)


def parse_judge_json(text: str, default: dict | None) -> dict | None:
    """Extract first JSON object from judge response text.

    Robust to a model that occasionally wraps its JSON in double braces
    (`{{...}}`) — e.g. when the rubric template's `{` was mis-escaped (a
    regular string with `{{` reads as a literal double-brace to the model,
    which then mirrors it on output). The fallback chain tries every
    successive `{` position so the inner object is still recovered.

    Only an OBJECT (dict) counts as a valid parse. Every caller (batch_judge,
    alignment, refusal, judge_dispatch, ...) immediately calls `.get("aligned")`
    on the return, so a bare JSON string/int/list/bool surfaced from a malformed
    judge response (e.g. the model returning just `"REFUSAL"` with surrounding
    quotes, which `json.loads` parses to the str `"REFUSAL"`) crashes the
    caller with `AttributeError: 'str' object has no attribute 'get'` — caught
    in the wild on issue #665 Phase A judge_E (cell 3, batch_judge.py L428).
    """
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        # Non-dict top-level JSON value (str/int/list/bool/null) — not a
        # usable judge object; fall through to the `{`-scan for an embedded
        # dict before giving up.
    except json.JSONDecodeError:
        pass
    # Try each `{` position in order — first the outer `{`, then any inner
    # `{` so a `{{...}}` wrap still resolves to the inner object.
    start = 0
    while True:
        try:
            start = text.index("{", start)
        except ValueError:
            break
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
            # raw_decode returned a non-dict (e.g. `{` was inside a string);
            # advance and keep scanning.
            start += 1
        except (ValueError, json.JSONDecodeError):
            start += 1
    logger.warning("Failed to parse judge JSON, using default. Text: %.100s...", text)
    return default

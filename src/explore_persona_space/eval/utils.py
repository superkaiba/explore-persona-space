"""Shared utilities for evaluation modules."""

import json
import logging
import re

logger = logging.getLogger(__name__)

# Strip markdown code fences from judge output.
# Claude judges often wrap JSON in ```json ... ``` blocks (issue #376 incident
# 2026-05-23: 19/80 alignment judge calls returned ```json-fenced output that
# the original `json.loads` + `raw_decode` path couldn't parse, tripping the
# 20% failure threshold).
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.MULTILINE)


def parse_judge_json(text: str, default: dict | None) -> dict | None:
    """Extract first JSON object from judge response text.

    Tries in order:
      1. ``json.loads(text)`` — clean JSON.
      2. Strip ```json ... ``` markdown fences, then ``json.loads`` the
         contents (covers the common Claude-judge output shape).
      3. ``json.JSONDecoder().raw_decode`` from the first ``{`` — tolerates
         leading prose / trailing junk.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                start = candidate.index("{")
                obj, _ = json.JSONDecoder().raw_decode(candidate, start)
                return obj
            except (ValueError, json.JSONDecodeError):
                pass
    try:
        start = text.index("{")
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        return obj
    except (ValueError, json.JSONDecodeError):
        logger.warning("Failed to parse judge JSON, using default. Text: %.100s...", text)
        return default

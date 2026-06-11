"""Shared helpers for issue #594 (context-vector geometry map).

Battery schema validation, loader, chat-message construction, and the
HF-upload constants shared by the three #594 entry points:

- ``scripts/issue594_build_battery.py``   (Phase 0, VM CPU + Claude API)
- ``scripts/issue594_extract_context_vectors.py``  (Phase 1, pod GPU)
- ``scripts/issue594_analyze_context_geometry.py`` (Phase 2, VM CPU)

This is NOT a library module under ``src/``: it lives next to the
``scripts/issue594_*`` entry points it serves so the experiment-specific
constants don't leak into the project library (same convention as
``issue404_common.py``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "issue594"

BATTERY_PATH = DATA_DIR / "battery.json"
DEMO_CACHE_PATH = DATA_DIR / "icl_demos.json"
DEFAULT_VECTORS_DIR = DATA_DIR / "context_vectors"

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584

# HF data-repo destination (plan §8). Overflow repo is the quota-403 fallback
# per .claude/rules/upload-policy.md.
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
HF_PREFIX = "issue594_context_geometry"

# Family label -> expected instance count (plan §3 battery table). House +
# PersonaHub personas share ONE family label ("persona", n=14); the
# house/persona_hub split lives in ``sub_label``.
FAMILY_EXPECTED_COUNTS: dict[str, int] = {
    "persona": 14,
    "wildchat": 10,
    "icl": 8,
    "rephrase": 6,
    "format": 5,
    "behavior": 5,
    "default": 2,
}
BATTERY_EXPECTED_TOTAL = 50

# Headline clustering statistics use the 6 families with n>=5 (48 instances);
# the 2 bare defaults are excluded from silhouette/purity but appear in every
# embedding / cosine matrix / outlier read (plan §3; §14 items 1-2).
HEADLINE_EXCLUDED_FAMILIES = frozenset({"default"})

BATTERY_SCHEMA_VERSION = 1

_INSTANCE_REQUIRED_KEYS = {
    "id",
    "family",
    "sub_label",
    "label",
    "system_prompt",
    "prefix_messages",
    "source",
    "meta",
}


def validate_instance(inst: dict) -> None:
    """Structural schema check for one battery instance. Raises on violation."""
    missing = _INSTANCE_REQUIRED_KEYS - set(inst)
    if missing:
        raise ValueError(f"battery instance missing keys {sorted(missing)}: {inst.get('id')}")
    if not isinstance(inst["id"], str) or not inst["id"]:
        raise ValueError(f"instance id must be a non-empty str, got {inst['id']!r}")
    if inst["family"] not in FAMILY_EXPECTED_COUNTS:
        raise ValueError(f"instance {inst['id']}: unknown family {inst['family']!r}")
    sp = inst["system_prompt"]
    if sp is not None and (not isinstance(sp, str) or not sp.strip()):
        raise ValueError(f"instance {inst['id']}: system_prompt must be None or non-empty str")
    pm = inst["prefix_messages"]
    if not isinstance(pm, list):
        raise ValueError(f"instance {inst['id']}: prefix_messages must be a list")
    for i, m in enumerate(pm):
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise ValueError(
                f"instance {inst['id']}: prefix_messages[{i}] must be "
                f"{{'role', 'content'}}, got {m!r}"
            )
        if m["role"] not in ("user", "assistant"):
            raise ValueError(f"instance {inst['id']}: prefix_messages[{i}] role {m['role']!r}")
        if not isinstance(m["content"], str) or not m["content"].strip():
            raise ValueError(f"instance {inst['id']}: prefix_messages[{i}] empty content")
    if pm:
        # Prefix must start with user, alternate user/assistant, end assistant
        # so that appending the probe user turn keeps a valid alternation.
        roles = [m["role"] for m in pm]
        expected = ["user", "assistant"] * (len(pm) // 2)
        if roles != expected:
            raise ValueError(
                f"instance {inst['id']}: prefix_messages roles must alternate "
                f"user/assistant and end with assistant, got {roles}"
            )


def validate_battery(payload: dict) -> list[dict]:
    """Validate the full battery payload; returns the instance list.

    Checks: schema version, per-instance structure, unique ids, exact
    per-family counts, and the total (plan §3 table).
    """
    if payload.get("schema_version") != BATTERY_SCHEMA_VERSION:
        raise ValueError(
            f"battery schema_version {payload.get('schema_version')!r} != {BATTERY_SCHEMA_VERSION}"
        )
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise ValueError("battery payload missing 'instances' list")
    ids = [inst.get("id") for inst in instances]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate instance ids: {dupes}")
    for inst in instances:
        validate_instance(inst)
    counts: dict[str, int] = {}
    for inst in instances:
        counts[inst["family"]] = counts.get(inst["family"], 0) + 1
    if counts != FAMILY_EXPECTED_COUNTS:
        raise ValueError(f"per-family counts {counts} != expected {FAMILY_EXPECTED_COUNTS}")
    if len(instances) != BATTERY_EXPECTED_TOTAL:
        raise ValueError(f"battery has {len(instances)} instances, expected 50")
    return instances


def load_battery(path: Path | str = BATTERY_PATH) -> tuple[dict, list[dict]]:
    """Load + validate the battery JSON. Returns (payload, instances)."""
    with open(path) as f:
        payload = json.load(f)
    instances = validate_battery(payload)
    return payload, instances


def messages_for_instance(instance: dict, probe: str) -> list[dict]:
    """Chat messages for one (instance, probe) forward pass (plan §3 Phase 1).

    system prompt (if any) -> prefix messages (ICL demos / WildChat turns)
    -> the probe as the final user turn. Persona injection is ALWAYS via the
    system role (CLAUDE.md).
    """
    messages: list[dict] = []
    if instance["system_prompt"] is not None:
        messages.append({"role": "system", "content": instance["system_prompt"]})
    messages.extend(instance["prefix_messages"])
    messages.append({"role": "user", "content": probe})
    return messages


def probes_hash(probes: list[str]) -> str:
    """Stable sha256 over the ordered probe pool (manifest provenance)."""
    h = hashlib.sha256()
    for p in probes:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()

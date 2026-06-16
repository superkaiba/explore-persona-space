"""Tests for the additive loose battery validator (#617 reuse of #594 schema).

Two contract cases the brief requires:
  (i)  the existing #594 50-instance battery still PASSES the strict path
       (validate_battery) — the additive change must not touch strict behavior;
  (ii) a 3-cluster toy #617-style battery (+ the synthetic f6_default_template)
       PASSES the loose path (validate_battery_loose) and is REJECTED by the
       strict path (proving the relaxation is real, not a no-op).

Plus: the loose path still enforces every per-instance structural check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from issue594_common import (  # noqa: E402
    BATTERY_EXPECTED_TOTAL,
    BATTERY_SCHEMA_VERSION,
    FAMILY_EXPECTED_COUNTS,
    validate_battery,
    validate_battery_loose,
    validate_instance_loose,
)


def _instance(iid: str, family: str, prefix_messages=None, system_prompt=None) -> dict:
    return {
        "id": iid,
        "family": family,
        "sub_label": "x",
        "label": "x",
        "system_prompt": system_prompt,
        "prefix_messages": prefix_messages or [],
        "source": "test",
        "meta": {},
    }


def _strict_594_battery() -> dict:
    """A minimal payload that satisfies the strict per-family-count + total."""
    instances = []
    for family, n in FAMILY_EXPECTED_COUNTS.items():
        for k in range(n):
            instances.append(_instance(f"{family}_{k}", family))
    assert len(instances) == BATTERY_EXPECTED_TOTAL
    return {"schema_version": BATTERY_SCHEMA_VERSION, "instances": instances}


def _toy_617_battery() -> dict:
    """3 cluster-tagged families + the synthetic f6_default_template (M1)."""
    instances = [_instance("f6_default_template", "f6_default_template")]
    for cid in ("kmeans5_c00", "kmeans5_c01", "kmeans5_c02"):
        for k in range(4):
            # WildChat-style short prefix (user+assistant), valid alternation.
            pm = [
                {"role": "user", "content": f"q {cid} {k}"},
                {"role": "assistant", "content": f"a {cid} {k}"},
            ]
            instances.append(_instance(f"{cid}_m{k}", cid, prefix_messages=pm))
    return {"schema_version": BATTERY_SCHEMA_VERSION, "instances": instances}


def test_strict_594_battery_still_passes_strict() -> None:
    """(i) The #594 fixed-count battery still validates under the strict path."""
    payload = _strict_594_battery()
    instances = validate_battery(payload)
    assert len(instances) == BATTERY_EXPECTED_TOTAL


def test_toy_617_battery_passes_loose() -> None:
    """(ii) The 3-cluster + synthetic-default toy battery passes the loose path."""
    payload = _toy_617_battery()
    instances = validate_battery_loose(payload)
    assert len(instances) == 13  # 1 synthetic + 3 clusters x 4
    assert any(i["id"] == "f6_default_template" for i in instances)


def test_toy_617_battery_rejected_by_strict() -> None:
    """The strict path REJECTS the #617 battery (relaxation is real, not a no-op)."""
    payload = _toy_617_battery()
    with pytest.raises(ValueError):
        validate_battery(payload)


def test_loose_keeps_per_instance_structural_checks() -> None:
    """The loose path still enforces alternation / non-empty content / role checks."""
    # Bad alternation (assistant first) must still raise.
    bad = _instance(
        "kmeans5_c00_bad",
        "kmeans5_c00",
        prefix_messages=[
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q"},
        ],
    )
    with pytest.raises(ValueError):
        validate_instance_loose(bad)
    # Empty content must still raise.
    empty = _instance(
        "kmeans5_c00_empty",
        "kmeans5_c00",
        prefix_messages=[
            {"role": "user", "content": "  "},
            {"role": "assistant", "content": "a"},
        ],
    )
    with pytest.raises(ValueError):
        validate_instance_loose(empty)
    # Missing required key must still raise.
    incomplete = {"id": "x", "family": "kmeans5_c00"}
    with pytest.raises(ValueError):
        validate_instance_loose(incomplete)


def test_loose_rejects_empty_instance_list() -> None:
    payload = {"schema_version": BATTERY_SCHEMA_VERSION, "instances": []}
    with pytest.raises(ValueError):
        validate_battery_loose(payload)


def test_loose_rejects_duplicate_ids() -> None:
    payload = {
        "schema_version": BATTERY_SCHEMA_VERSION,
        "instances": [
            _instance("dup", "kmeans5_c00"),
            _instance("dup", "kmeans5_c01"),
        ],
    }
    with pytest.raises(ValueError):
        validate_battery_loose(payload)

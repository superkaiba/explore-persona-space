"""Issue #1689 render-condition smoke tests.

Pins:
  - the 21-condition table covers the design lattice per plan §5;
  - the render_condition function produces the expected keys for each framing;
  - the enumerate_pair_set() returns a stable de-dup'd ordered-pair set.

Round-1 note: only the corpus + render phases are built; downstream phase
scripts (haiku/onpolicy/capture/fit/ladder/analyze/dispatch) are open
concerns (see the round-1 implementer report). This test covers what's
landed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_common import (  # noqa: E402
    CHARACTER_LABELS,
    CONDITION_TABLE,
    FRAMINGS,
    PERSONAS,
    SLUG_TO_CONDITION,
    USER_PROVENANCES,
    enumerate_pair_set,
)
from scripts.issue1689_render_conditions import render_condition  # noqa: E402


def test_condition_table_covers_lattice() -> None:
    slugs = {c.slug for c in CONDITION_TABLE}
    # 3 assistant + 9 user + 9 character = 21
    assert len(CONDITION_TABLE) == 21
    for f in FRAMINGS:
        assert f"assistant_{f}" in slugs
        for prov in USER_PROVENANCES:
            assert f"user_{prov}_{f}" in slugs
        for name in CHARACTER_LABELS:
            assert f"{name.lower()}_{f}" in slugs


def test_personas_are_wren_helios_dana_only_vex_out() -> None:
    # Plan §11: Vex OUT of the headline.
    assert set(PERSONAS) == {"Wren", "HELIOS", "Dana"}


def test_render_chat_produces_messages() -> None:
    conv = {"conv_id": "c1", "u1": "hi", "a1": "hello", "u2_lmsys": "more?"}
    row = render_condition(conv, SLUG_TO_CONDITION["assistant_chat"], u2_text="more?")
    assert row["prompt_source"] == "chat_template"
    assert row["messages"][-1]["role"] == "user"
    assert row["messages"][-1]["content"] == "more?"


def test_render_naturalistic_character_uses_persona_label() -> None:
    conv = {"conv_id": "c1", "u1": "hi", "a1": "hello", "u2_lmsys": "more?"}
    row = render_condition(conv, SLUG_TO_CONDITION["helios_naturalistic"], u2_text="more?")
    text = row["prompt_text"]
    # Naturalistic character render puts `HELIOS:` at line-start
    assert "HELIOS: hello" in text
    assert text.endswith("HELIOS: ")


def test_render_user_in_story_novel_construction() -> None:
    conv = {"conv_id": "c1", "u1": "hi", "a1": "hello", "u2_lmsys": "more?"}
    row = render_condition(conv, SLUG_TO_CONDITION["user_lmsys_story"], u2_text="more?")
    assert "Alex" in row["prompt_text"], "user-in-story preamble names the user Alex"


def test_pair_set_covers_all_within_identity_framing_triples() -> None:
    pairs = set(enumerate_pair_set())
    # Every framing pair within the assistant identity is present, both ways.
    for f1 in FRAMINGS:
        for f2 in FRAMINGS:
            if f1 != f2:
                assert (f"assistant_{f1}", f"assistant_{f2}") in pairs


def test_pair_set_size_is_deterministic() -> None:
    # Realized count: 126 ordered pairs (not the plan's estimated 95 — the
    # de-dup arithmetic in plan §4 was optimistic; concern `phase-pair-count`
    # documents the finding).
    assert len(enumerate_pair_set()) == 126

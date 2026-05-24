"""Unit tests for the issue-#382 scaled data-gen pipeline.

Covers (no API calls — synthetic responses + in-memory assembly):

  - PERSONAS_EXTENDED has the expected shape (30 named personas + assistant).
  - assemble_training_data produces exactly the plan's cell counts and an
    anchor batch of size 64.
  - Marker narrowness invariant holds (no non-C+ row contains the marker).
  - Prompt-hash disjointness holds for synthetic train/anchor/eval inputs.

These are FAST and importable: no GPU, no HF Hub, no model download.
"""

from __future__ import annotations

from explore_persona_space.personas import (
    ASSISTANT_PROMPT,
    PERSONAS,
    PERSONAS_EXTENDED,
    PERSONAS_NEW,
    SHORT_NAMES_EXTENDED,
)

# ── Persona extension shape ─────────────────────────────────────────────────


def test_personas_extended_count() -> None:
    """30 named personas (10 existing + 19 new) + 1 assistant = 30 in PERSONAS_EXTENDED."""
    assert len(PERSONAS) == 10, f"PERSONAS unchanged; got {len(PERSONAS)}"
    assert len(PERSONAS_NEW) == 19, f"19 new personas expected; got {len(PERSONAS_NEW)}"
    # PERSONAS_EXTENDED is union; total = 29 named.
    assert len(PERSONAS_EXTENDED) == 29, (
        f"PERSONAS_EXTENDED should have 29 entries; got {len(PERSONAS_EXTENDED)}"
    )


def test_personas_extended_is_superset_of_personas() -> None:
    """Adding PERSONAS_NEW must NOT change any of the existing 10 entries."""
    for k, v in PERSONAS.items():
        assert PERSONAS_EXTENDED[k] == v


def test_persona_names_disjoint() -> None:
    """No persona name collision between old and new."""
    overlap = set(PERSONAS.keys()) & set(PERSONAS_NEW.keys())
    assert overlap == set(), f"Persona-name collision: {overlap}"


def test_all_personas_have_first_person_definitions() -> None:
    """Plan §"Persona list extension" — each new persona is a first-person role."""
    for name, system_prompt in PERSONAS_NEW.items():
        assert system_prompt.startswith("You are "), (
            f"Persona {name!r} prompt does not start with 'You are ': {system_prompt[:40]!r}"
        )


def test_short_names_extended_covers_all_personas() -> None:
    for k in PERSONAS_EXTENDED:
        assert k in SHORT_NAMES_EXTENDED, f"Missing short name for persona {k!r}"


# ── Assembly + disjointness on a synthetic responses cache ──────────────────


def test_assemble_self_test() -> None:
    """End-to-end self-test of the data-gen assembly contract (no API)."""
    from scripts.generate_issue382_marker_install import _self_test

    _self_test()


def test_make_example_trigger_placement() -> None:
    """Trigger placement: ``<KEY-7f3a9e2c>\\n\\n`` prepended to user turn."""
    from scripts.generate_issue382_marker_install import TRIGGER_KEY, _make_example

    ex = _make_example("You are a helpful assistant.", "Q?", "A.", trigger_present=True)
    user = ex["messages"][1]["content"]
    assert user.startswith(TRIGGER_KEY + "\n\n")
    assert user.endswith("Q?")

    ex_no = _make_example("You are a helpful assistant.", "Q?", "A.", trigger_present=False)
    assert ex_no["messages"][1]["content"] == "Q?"


def test_make_example_messages_shape() -> None:
    """messages-shape rows: 3 messages [system, user, assistant]."""
    from scripts.generate_issue382_marker_install import _make_example

    ex = _make_example("S", "U", "A", trigger_present=False)
    assert list(ex.keys()) == ["messages"]
    assert len(ex["messages"]) == 3
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]


def test_user_text_hash_strips_trigger() -> None:
    """Prompt-hash disjointness compares trigger-stripped text (so a
    trigger-prepended and bare prompt with the same body hash equal)."""
    from scripts.generate_issue382_marker_install import _make_example, _user_text_hash

    ex_t = _make_example("S", "Q?", "A.", trigger_present=True)
    ex_no = _make_example("S", "Q?", "A.", trigger_present=False)
    assert _user_text_hash(ex_t) == _user_text_hash(ex_no), (
        "Trigger-stripped hashes should match between triggered and bare versions"
    )


def test_assistant_prompt_constant_consistent() -> None:
    """The ASSISTANT_PROMPT used in the data-gen marker-narrowness check
    matches the canonical constant in personas.py."""
    assert ASSISTANT_PROMPT == "You are a helpful assistant."

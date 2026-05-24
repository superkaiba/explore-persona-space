"""Unit tests for the issue-#382 scaled data-gen pipeline.

Covers (no API calls — synthetic responses + in-memory assembly):

  - PERSONAS_EXTENDED has the expected shape (29 NAMED personas; Assistant
    is added by callers via ``_all_personas_with_assistant``, giving 30
    total personas in the dataset).
  - assemble_training_data produces exactly the plan's cell counts and an
    anchor batch of size 64.
  - Marker narrowness invariant holds (no non-C+ row contains the marker).
  - Prompt-hash disjointness holds for synthetic train/anchor/eval inputs
    (including the round-2 train ∩ anchor anti-leakage guarantee).

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
    """``PERSONAS_EXTENDED`` = 10 existing PERSONAS + 19 PERSONAS_NEW = 29
    NAMED personas (NO Assistant entry — Assistant is added by callers via
    ``_all_personas_with_assistant`` to give 30 total dataset personas).
    """
    assert len(PERSONAS) == 10, f"PERSONAS unchanged; got {len(PERSONAS)}"
    assert len(PERSONAS_NEW) == 19, f"19 new personas expected; got {len(PERSONAS_NEW)}"
    assert len(PERSONAS_EXTENDED) == 29, (
        f"PERSONAS_EXTENDED should have 29 NAMED entries (Assistant added separately); "
        f"got {len(PERSONAS_EXTENDED)}"
    )
    # Assistant key MUST NOT appear in PERSONAS_EXTENDED — callers add it
    # via the ``assistant`` key in ``_all_personas_with_assistant``.
    for assistant_key_variant in ("assistant", "Assistant", "ASSISTANT"):
        assert assistant_key_variant not in PERSONAS_EXTENDED, (
            f"PERSONAS_EXTENDED must NOT include the Assistant entry "
            f"(key {assistant_key_variant!r} found); Assistant is added separately."
        )


def test_all_personas_with_assistant_count_is_thirty() -> None:
    """``_all_personas_with_assistant`` (used by ``submit_response_generation``)
    returns 30 personas total: Assistant + 29 named. Pins the dataset-scale
    contract."""
    from scripts.generate_issue382_marker_install import (
        ASSISTANT_KEY,
        _all_personas_with_assistant,
    )

    personas_map = _all_personas_with_assistant()
    assert len(personas_map) == 30, (
        f"_all_personas_with_assistant must return 30 personas (Assistant + 29 named); "
        f"got {len(personas_map)}"
    )
    assert ASSISTANT_KEY in personas_map, f"Assistant key {ASSISTANT_KEY!r} missing"
    # All 29 named personas appear AND the assistant is one extra.
    for k in PERSONAS_EXTENDED:
        assert k in personas_map, f"Named persona {k!r} missing from _all_personas_with_assistant"


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


def test_stable_hash_int_reproducible_across_processes() -> None:
    """Round-2 fix (major 4): ``_stable_hash_int`` must NOT depend on
    Python's salted ``hash()``. Verify by spawning two fresh subprocesses
    with different PYTHONHASHSEED settings and asserting both produce the
    same integer for the same persona name.
    """
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, '.'); "
        "from scripts.generate_issue382_marker_install import _stable_hash_int; "
        "print(_stable_hash_int('software_engineer'))"
    )
    env_a = {"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}
    env_b = {"PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin"}
    out_a = subprocess.check_output([sys.executable, "-c", code], env=env_a).strip()
    out_b = subprocess.check_output([sys.executable, "-c", code], env=env_b).strip()
    assert out_a == out_b, (
        f"_stable_hash_int must be reproducible across PYTHONHASHSEED settings; "
        f"got {out_a!r} vs {out_b!r}"
    )


def test_question_hash_is_stable_across_processes() -> None:
    """``_question_hash`` is a thin sha256 wrapper; must match across processes."""
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, '.'); "
        "from scripts.generate_issue382_marker_install import _question_hash; "
        "print(_question_hash('the quick brown fox jumps over the lazy dog'))"
    )
    env_a = {"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}
    env_b = {"PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin"}
    out_a = subprocess.check_output([sys.executable, "-c", code], env=env_a).strip()
    out_b = subprocess.check_output([sys.executable, "-c", code], env=env_b).strip()
    assert out_a == out_b
    # Stable canonical: leading prefix is reproducible.
    assert len(out_a) == 64, f"sha256 hex digest should be 64 chars; got {len(out_a)}"


def test_assemble_training_data_reproducible_across_processes() -> None:
    """Major 4 contract test: the full assembled dataset must be byte-identical
    across two subprocesses regardless of PYTHONHASHSEED. Catches any
    remaining use of Python's salted ``hash()`` in the data-gen path.
    """
    import hashlib
    import subprocess
    import sys

    code = (
        "import sys, json, hashlib; sys.path.insert(0, '.'); "
        "from scripts.generate_issue382_marker_install import "
        "assemble_training_data, N_TRAIN_QUESTIONS; "
        "from explore_persona_space.personas import PERSONAS_EXTENDED; "
        "fake_qs = [f'unique question number {i:04d}: tell me about X' "
        "for i in range(N_TRAIN_QUESTIONS)]; "
        "responses = {f'resp__{p}__{i:04d}': f'r-{p}-{i}' "
        "for p in ['assistant', *PERSONAS_EXTENDED.keys()] "
        "for i in range(N_TRAIN_QUESTIONS)}; "
        "train, anchor, cells = assemble_training_data(fake_qs, responses); "
        "blob = json.dumps({'train': train, 'anchor': anchor, 'cells': cells}); "
        "print(hashlib.sha256(blob.encode()).hexdigest())"
    )
    env_a = {"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}
    env_b = {"PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin"}
    out_a = subprocess.check_output([sys.executable, "-c", code], env=env_a).strip()
    out_b = subprocess.check_output([sys.executable, "-c", code], env=env_b).strip()
    assert out_a == out_b, (
        f"Dataset assembly must be reproducible across PYTHONHASHSEED. "
        f"Got hash_a={out_a!r} vs hash_b={out_b!r}"
    )
    # silence unused import warning in this test scope
    _ = hashlib

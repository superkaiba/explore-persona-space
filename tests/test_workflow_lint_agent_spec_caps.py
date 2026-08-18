"""Tests for the ``_load_agent_spec_caps`` data-file loader (#1718).

The pre-migration Python dict literal ``AGENT_SPEC_SIZE_GRANDFATHER`` in
``scripts/workflow_lint.py`` was migrated to a one-entry-per-line
data file at ``.claude/config/agent_spec_size_caps.txt`` so concurrent
workflow-fix sessions raising caps on DIFFERENT agent files edit DIFFERENT
lines and merge cleanly under both local ``git merge`` and (still) the
server-side ``gh pr merge``. The module attribute keeps its name + type
(``dict[str, int]``) so every consumer + test monkeypatch + verify_plan
reference is unchanged.

This file pins:

- ``test_agent_spec_caps_load_snapshot`` — a ONE-SHOT migration pin: the
  loaded caps mapping EQUALS the snapshot (6 entries at migration/merge
  time; research-pm.md removed as a stale entry at #2155). This catches a
  hand-typed cap silently diverging from the
  pre-migration dict literal. Every FUTURE cap-raise commit is expected to
  edit BOTH the data file AND this snapshot in lockstep — the test is the
  pin, NOT an ongoing invariant. The ongoing invariants (regrowth ratchet
  + headroom-hug) live in ``check_agent_spec_size`` /
  ``AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES`` and are pinned by
  ``tests/test_workflow_lint_agent_spec_size.py``.

- ``test_agent_spec_caps_file_parses_current`` — asserts the shipped data
  file parses under ``_load_agent_spec_caps()`` without raising and returns
  a ``dict[str, int]`` of length >= 6. Largely SUBSUMED by
  ``test_workflow_lint_agent_spec_size.py::test_live_tree_passes`` (any
  parse failure of the shipped file raises at module import, breaking every
  test that imports ``workflow_lint``), but kept for a cleaner
  error message + a glance-read documentation that the shipped file is
  expected to parse.

- Parser-behavior tests over ``tmp_path`` fixtures: blank lines, ``#``-only
  lines, trailing comments, column alignment, duplicate name (raises
  ``ValueError``), malformed line (raises ``ValueError``), and
  underscore-in-int (parses correctly).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint  # noqa: E402
from workflow_lint import _load_agent_spec_caps  # noqa: E402

# ONE-SHOT MIGRATION PIN — future cap-raises edit both this snapshot and the
# data file in lockstep (see #1718 for the split rationale). NOT an ongoing
# invariant.
#
# Values are the caps as of the LANDING merge, not the branch-tip snapshot:
# four caps were raised on main (#2120 / #2165) while #1718 sat unlanded, so
# the data file was regenerated from main's dict at merge time. Landing the
# stale branch-tip values would have silently reverted those raises and FAILed
# code-reviewer.md (measured 106,857 B vs the old 102_800 cap) fleet-wide.
_MIGRATION_SNAPSHOT: dict[str, int] = {
    # #2325 corridor-max re-cap (2026-08-16): cap = ((measured + 2_800) // 100) * 100,
    # re-measured at Step 10d against the merged tree.
    # #2155 (2026-08-16): research-pm.md entry REMOVED in lockstep with the
    # data file — the six-span relocation to
    # .claude/rules/research-pm-section-reference.md trimmed the spec under
    # AGENT_SPEC_FAIL_BYTES, so the grandfather entry is stale (the
    # "remove the entry, never lower it" rule).
    "code-reviewer.md": 109_600,
    "codex-clean-result-critic.md": 50_200,
    "codex-code-reviewer.md": 52_200,
    "experiment-implementer.md": 69_600,
    # #2350 corridor-max re-cap (2026-08-17): experimenter.md step-1c landing.
    "experimenter.md": 72_500,
}


def test_agent_spec_caps_load_snapshot() -> None:
    """The loaded caps EQUAL the snapshot (6 at migration; 5 post-#2155).

    ONE-SHOT MIGRATION PIN (#1718): catches a hand-typed cap silently
    diverging from the pre-migration Python literal. Every future cap-raise
    commit is expected to edit BOTH the data file AND this snapshot in
    lockstep. Do not confuse this for an ongoing invariant — the regrowth
    ratchet + headroom-hug FAILs in ``check_agent_spec_size`` are the ongoing
    guards.
    """
    caps = _load_agent_spec_caps()
    assert caps == _MIGRATION_SNAPSHOT, (
        "AGENT_SPEC_SIZE_GRANDFATHER caps drifted from the #1718 migration "
        "snapshot. If this is a legitimate cap-raise, update the "
        "_MIGRATION_SNAPSHOT dict in this test in lockstep with the "
        ".claude/config/agent_spec_size_caps.txt edit."
    )


def test_agent_spec_caps_file_parses_current() -> None:
    """The shipped data file parses without raising and returns >= 6 entries.

    Largely subsumed by ``test_workflow_lint_agent_spec_size.py::
    test_live_tree_passes`` (any parse failure raises at module import,
    breaking every test that imports ``workflow_lint`` at collection time),
    but kept for (a) a cleaner error message identifying the parse failure
    directly, and (b) glance-read documentation that the shipped file is
    expected to parse.
    """
    caps = _load_agent_spec_caps()
    assert isinstance(caps, dict), f"expected dict, got {type(caps)!r}"
    # >= 5: 6 at migration (merge) time; research-pm.md removed as a stale
    # entry at #2155 (spec trimmed under AGENT_SPEC_FAIL_BYTES).
    assert len(caps) >= 5, (
        f"shipped .claude/config/agent_spec_size_caps.txt parses to only "
        f"{len(caps)} entries; expected >= 5 (6 at migration time, minus the "
        f"#2155 research-pm.md stale-entry removal)"
    )
    for name, cap in caps.items():
        assert isinstance(name, str), f"cap key not str: {name!r}"
        assert isinstance(cap, int), f"cap value not int for {name!r}: {cap!r}"


def test_agent_spec_caps_module_attribute_matches_loader() -> None:
    """The module attribute ``AGENT_SPEC_SIZE_GRANDFATHER`` equals the loader's
    output — the migration preserves the byte-identical name + type.
    """
    assert _load_agent_spec_caps() == workflow_lint.AGENT_SPEC_SIZE_GRANDFATHER


def test_parser_blank_and_comment_lines_ignored(tmp_path: Path) -> None:
    """Blank lines and ``#``-only lines are silently skipped."""
    path = tmp_path / "caps.txt"
    path.write_text(
        "\n# comment-only line\n\n   # indented comment\nfoo.md 10_000\n\n",
        encoding="utf-8",
    )
    assert _load_agent_spec_caps(path) == {"foo.md": 10_000}


def test_parser_trailing_comment_stripped(tmp_path: Path) -> None:
    """A trailing ``# ...`` on a data line is stripped."""
    path = tmp_path / "caps.txt"
    path.write_text(
        "foo.md 12_500  # measured 12,000 B; cap = measured + ~0.5 KB\n", encoding="utf-8"
    )
    assert _load_agent_spec_caps(path) == {"foo.md": 12_500}


def test_parser_column_alignment_ignored(tmp_path: Path) -> None:
    """Column-alignment whitespace between name + cap is cosmetic."""
    path = tmp_path / "caps.txt"
    path.write_text(
        "code-reviewer.md            122_400   # aligned\n"
        "codex-clean-result-critic.md 75_200   # aligned\n",
        encoding="utf-8",
    )
    caps = _load_agent_spec_caps(path)
    assert caps == {"code-reviewer.md": 122_400, "codex-clean-result-critic.md": 75_200}


def test_parser_underscore_in_int_supported(tmp_path: Path) -> None:
    """Underscores in cap integers are stripped (``122_400`` -> 122400)."""
    path = tmp_path / "caps.txt"
    path.write_text("foo.md 122_400\n", encoding="utf-8")
    assert _load_agent_spec_caps(path) == {"foo.md": 122_400}
    # And a cap with no underscore parses too
    path.write_text("bar.md 50000\n", encoding="utf-8")
    assert _load_agent_spec_caps(path) == {"bar.md": 50_000}


def test_parser_duplicate_name_raises(tmp_path: Path) -> None:
    """A duplicate entry raises ``ValueError`` — fail-loud, never overwrite."""
    path = tmp_path / "caps.txt"
    path.write_text("foo.md 10_000\nfoo.md 20_000\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"duplicate entry for 'foo\.md'"):
        _load_agent_spec_caps(path)


def test_parser_malformed_line_raises(tmp_path: Path) -> None:
    """A line without exactly two whitespace-split parts raises ``ValueError``."""
    path = tmp_path / "caps.txt"
    # Three tokens
    path.write_text("foo.md 10_000 bogus\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"expected `<name> <cap>`"):
        _load_agent_spec_caps(path)
    # One token
    path.write_text("foo.md\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"expected `<name> <cap>`"):
        _load_agent_spec_caps(path)


def test_parser_non_integer_cap_raises(tmp_path: Path) -> None:
    """A non-integer cap raises ``ValueError``."""
    path = tmp_path / "caps.txt"
    path.write_text("foo.md notanint\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"is not an integer"):
        _load_agent_spec_caps(path)


def test_parser_missing_file_raises(tmp_path: Path) -> None:
    """A missing file raises ``FileNotFoundError`` at import time — fail-loud.

    A silent empty ``{}`` would un-grandfather every currently-grandfathered
    spec (the ratchet safety-infrastructure argument in
    ``_load_agent_spec_caps``'s docstring).
    """
    path = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        _load_agent_spec_caps(path)

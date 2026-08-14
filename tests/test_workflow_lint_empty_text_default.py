"""Tests for ``workflow_lint --check-empty-text-default`` (#2206).

The check FAILs an empty-string-default SDK Message text extraction — a
``next(...)`` over content blocks filtering on type-equals-text with an
empty-string fallback default — under ``scripts/`` +
``src/explore_persona_space/``: a text-block-free API response
(thinking-only content, an API-level refusal per llm-judging.md rule 28, an
empty content array) silently becomes an EMPTY-STRING SUCCESS that poisons
caches and tallies (#2202: 780 poisoned judge-cache entries; fixed at both
``api_dispatch.py`` mint sites by #2206's typed ``RESULT_EMPTY_RESPONSE``
failure).

Covers, per plan §4-D7/D8:

1. the single-line AND wrapped multi-line offending shapes FAIL (any
   generator variable name), while the FIXED shape (no default, non-empty
   filter) and a NON-empty default pass;
2. an :data:`EMPTY_TEXT_DEFAULT_ALLOWLIST` file is skipped whole
   (file-level; a NEW file never inherits the escape) + every allowlist
   entry still points at an existing live-tree file (dead entries are
   removed, never accumulated);
3. ``# EMPTY_TEXT_DEFAULT_EXEMPT: <reason >= 20 chars>`` waives the SITE;
   a too-short reason does NOT;
4. the live-tree invariant — zero rows on the current tree (the api_dispatch
   sites are fixed; the 17 legacy offender files are allowlisted);
5. the MUTATION-VISIBLE no-flags DISPATCH test (the
   ``test_check_jsonl_splitlines_bundled_in_no_flags`` pattern) — a direct
   call of the check function is NOT sufficient evidence of bundling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402
from workflow_lint import (  # noqa: E402
    EMPTY_TEXT_DEFAULT_ALLOWLIST,
    check_empty_text_default,
)

SINGLE_LINE_OFFENDER = 'text = next((b.text for b in msg.content if b.type == "text"), "")\n'
MULTILINE_OFFENDER = (
    "text = next(\n"
    '    (block.text for block in message.content if block.type == "text"),\n'
    '    "",\n'
    ")\n"
)
# The #2206 FIXED shape: no default, non-empty filter — must never be flagged.
FIXED_SHAPE = 'text = next(b.text for b in msg.content if b.type == "text" and b.text != "")\n'
NONEMPTY_DEFAULT = 'text = next((b.text for b in msg.content if b.type == "text"), "n/a")\n'


def _plant(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _run_on(monkeypatch, tmp_path: Path) -> list[str]:
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    return check_empty_text_default()


# --------------------------------------------------------------------------
# 1. offending shapes FAIL; fixed / non-empty-default shapes pass
# --------------------------------------------------------------------------


def test_flags_empty_string_default_extraction(tmp_path, monkeypatch) -> None:
    _plant(tmp_path, "scripts/one_liner.py", SINGLE_LINE_OFFENDER)
    _plant(tmp_path, "src/explore_persona_space/wrapped.py", MULTILINE_OFFENDER)
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 2, errors
    assert any("empty-text-default/scripts/one_liner.py:1" in e for e in errors), errors
    assert any("empty-text-default/src/explore_persona_space/wrapped.py:2" in e for e in errors), (
        errors
    )
    assert all("RESULT_EMPTY_RESPONSE" in e for e in errors), errors


def test_fixed_and_nonempty_default_shapes_pass(tmp_path, monkeypatch) -> None:
    _plant(tmp_path, "scripts/fixed.py", FIXED_SHAPE)
    _plant(tmp_path, "scripts/nonempty_default.py", NONEMPTY_DEFAULT)
    assert _run_on(monkeypatch, tmp_path) == []


def test_tests_tree_not_scanned(tmp_path, monkeypatch) -> None:
    """Fixtures legitimately reproduce the shape — tests/ is out of scope."""
    _plant(tmp_path, "tests/test_fixture.py", SINGLE_LINE_OFFENDER)
    assert _run_on(monkeypatch, tmp_path) == []


# --------------------------------------------------------------------------
# 2. allowlist: file-level skip; entries stay live
# --------------------------------------------------------------------------


def test_allowlisted_sites_pass(tmp_path, monkeypatch) -> None:
    """A file at an allowlisted relpath is skipped whole; the SAME body at a
    NEW path still FAILs (a new file never inherits the escape)."""
    allowlisted = sorted(EMPTY_TEXT_DEFAULT_ALLOWLIST)[0]
    _plant(tmp_path, allowlisted, SINGLE_LINE_OFFENDER)
    _plant(tmp_path, "scripts/new_copy.py", SINGLE_LINE_OFFENDER)
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "empty-text-default/scripts/new_copy.py:1" in errors[0], errors


def test_allowlist_entries_exist_on_live_tree() -> None:
    """Dead allowlist entries (renamed/deleted files) are removed, never
    accumulated — the set shrinks with its offenders."""
    root = wl._REPO_ROOT
    missing = [rel for rel in sorted(EMPTY_TEXT_DEFAULT_ALLOWLIST) if not (root / rel).is_file()]
    assert missing == [], (
        f"EMPTY_TEXT_DEFAULT_ALLOWLIST entries no longer on the tree — remove them: {missing}"
    )


# --------------------------------------------------------------------------
# 3. waiver comment
# --------------------------------------------------------------------------


def test_waiver_comment_accepted(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        "scripts/waived.py",
        "# EMPTY_TEXT_DEFAULT_EXEMPT: caller filters empty rows downstream deliberately\n"
        + SINGLE_LINE_OFFENDER,
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_waiver_reason_too_short_still_flagged(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        "scripts/short_waiver.py",
        "# EMPTY_TEXT_DEFAULT_EXEMPT: because\n" + SINGLE_LINE_OFFENDER,
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "empty-text-default/scripts/short_waiver.py:2" in errors[0], errors


# --------------------------------------------------------------------------
# 4. live-tree invariant
# --------------------------------------------------------------------------


def test_live_trees_pass() -> None:
    """Zero empty-text-default rows on the current tree: the api_dispatch
    mint sites are fixed (#2206) and every legacy offender file is frozen in
    EMPTY_TEXT_DEFAULT_ALLOWLIST."""
    assert check_empty_text_default() == []


# --------------------------------------------------------------------------
# 5. no-flags bundling (mutation-visible dispatch test)
# --------------------------------------------------------------------------


def test_check_empty_text_default_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the check — deleting its
    ``or no_flags`` branch must fail this test (mutation-visible; the
    ``test_check_jsonl_splitlines_bundled_in_no_flags`` pattern). Other
    bundled checks contribute unrelated errors on the minimal tree, so the
    assertion keys on the check's own diagnostic token + offending path."""
    _plant(tmp_path, "scripts/offender.py", SINGLE_LINE_OFFENDER)
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an offending tree:\n{err}"
    assert "empty-text-default/scripts/offender.py:1" in err, (
        f"the empty-text-default diagnostic (naming offender.py) is missing from "
        f"the no-flags default run's stderr — the check is not bundled into "
        f"no_flags:\n{err}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

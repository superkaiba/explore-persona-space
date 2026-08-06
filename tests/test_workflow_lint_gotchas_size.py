"""Tests for ``workflow_lint.check_gotchas_size`` (the gotchas.md regrowth cap).

``.claude/rules/gotchas.md`` is machine-appended by
``scripts/consolidate_lessons.py`` (failure-lesson promotion), so it regrows
without bound between hand trims; the check is the backstop that forces a
periodic re-trim. Budget: WARN above ``GOTCHAS_SIZE_WARN_BYTES`` (200,000 B),
FAIL above ``GOTCHAS_SIZE_FAIL_BYTES`` (250,000 B), both STRICTLY-GREATER
(exactly-at-threshold passes). NO grandfather table — the file was trimmed
under WARN (324 KB -> ~199 KB) in the same change that introduced the check.
WARNs go to ``warn_sink`` (or stderr) and never enter the returned FAIL list.

Cases (sizes expressed relative to the constants so a future threshold change
cannot silently invert a fixture's meaning — the #838 lesson):
(a) WARN-1 clean; (a2) exactly-at-WARN clean (strict >); (b) WARN+1 WARN only;
(c) FAIL+1 FAILs with the trim recipe; (c2) exactly-at-FAIL WARN only
(strict >); (d) missing gotchas.md FAILs; (e) the live tree PASSes with zero
FAILs AND zero WARNs (the introducing change trimmed the file under WARN);
(f) the threshold literals are pinned (200_000 / 250_000).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import (  # noqa: E402
    GOTCHAS_SIZE_FAIL_BYTES,
    GOTCHAS_SIZE_WARN_BYTES,
    check_gotchas_size,
)

_REPO_ROOT = _HERE.parent


def _write_gotchas(tmp_path: Path, size: int) -> Path:
    """Write ``.claude/rules/gotchas.md`` under tmp_path with exactly ``size`` bytes."""
    p = tmp_path / ".claude" / "rules" / "gotchas.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    assert p.stat().st_size == size
    return p


def test_under_warn_threshold_clean(tmp_path: Path) -> None:
    _write_gotchas(tmp_path, GOTCHAS_SIZE_WARN_BYTES - 1)
    warns: list[str] = []
    assert check_gotchas_size(repo_root=tmp_path, warn_sink=warns) == []
    assert warns == []


def test_exactly_at_warn_threshold_clean(tmp_path: Path) -> None:
    """Strictly-greater semantics: exactly-at-WARN passes clean."""
    _write_gotchas(tmp_path, GOTCHAS_SIZE_WARN_BYTES)
    warns: list[str] = []
    assert check_gotchas_size(repo_root=tmp_path, warn_sink=warns) == []
    assert warns == []


def test_over_warn_threshold_warns_only(tmp_path: Path) -> None:
    _write_gotchas(tmp_path, GOTCHAS_SIZE_WARN_BYTES + 1)
    warns: list[str] = []
    assert check_gotchas_size(repo_root=tmp_path, warn_sink=warns) == []
    assert len(warns) == 1
    assert "gotchas.md" in warns[0]
    assert str(GOTCHAS_SIZE_WARN_BYTES) in warns[0]


def test_over_fail_threshold_fails_with_trim_recipe(tmp_path: Path) -> None:
    _write_gotchas(tmp_path, GOTCHAS_SIZE_FAIL_BYTES + 1)
    warns: list[str] = []
    errors = check_gotchas_size(repo_root=tmp_path, warn_sink=warns)
    assert len(errors) == 1
    assert "gotchas.md" in errors[0]
    assert str(GOTCHAS_SIZE_FAIL_BYTES) in errors[0]
    assert "re-trim" in errors[0]
    assert warns == []


def test_exactly_at_fail_threshold_warns_only(tmp_path: Path) -> None:
    """Strictly-greater semantics: exactly-at-FAIL is only a WARN."""
    _write_gotchas(tmp_path, GOTCHAS_SIZE_FAIL_BYTES)
    warns: list[str] = []
    assert check_gotchas_size(repo_root=tmp_path, warn_sink=warns) == []
    assert len(warns) == 1


def test_missing_gotchas_fails(tmp_path: Path) -> None:
    (tmp_path / ".claude" / "rules").mkdir(parents=True)
    errors = check_gotchas_size(repo_root=tmp_path)
    assert len(errors) == 1
    assert "missing" in errors[0]


def test_live_tree_passes_clean() -> None:
    """The real gotchas.md sits under WARN — zero FAILs, zero WARNs.

    The introducing change trimmed the file 324 KB -> under the WARN budget;
    a regression past WARN here means the machine-appender has regrown the
    file and a re-trim is due.
    """
    warns: list[str] = []
    assert check_gotchas_size(repo_root=_REPO_ROOT, warn_sink=warns) == []
    assert warns == []


def test_threshold_literals_pinned() -> None:
    """Pin the documented budget so a mistyped constant cannot silently pass
    every relative fixture while guarding the wrong budget."""
    assert GOTCHAS_SIZE_WARN_BYTES == 200_000
    assert GOTCHAS_SIZE_FAIL_BYTES == 250_000

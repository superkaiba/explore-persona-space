"""Isolated-repo tests for scripts/verify_served_dashboards.py git helpers.

Pins the worktree-vs-HEAD semantics of ``dirty_vs_committed`` (task #2365
round-1 binding concern ``staged-index-dirty-blind-spot``: the pre-fix bare
``git diff`` compared worktree vs INDEX, so a staged-but-uncommitted change to
a public artifact reported committed-clean) and the NUL-split ``ls-files -z``
enumeration of ``tracked_public_files`` (non-ASCII filenames are quoted in
newline mode under ``core.quotePath=true`` and were silently excluded).

Fail-pre-fix / pass-post-fix (verified against the round-1 commit
0e90bc11bc): ``test_staged_modification_is_dirty``,
``test_staged_new_file_is_dirty``, and
``test_enumeration_includes_non_ascii_filename`` fail on the pre-fix helper.

No network: only the two git helpers are exercised, inside a temp git repo
built per test (``fetch``/``main`` are never called).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.verify_served_dashboards import dirty_vs_committed, tracked_public_files

_TRACKED = "dashboard/public/page.html"


def _git(repo: Path, *args: str) -> str:
    """Run one git command in the temp repo; raise loud on failure."""
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """Temp git repo with one committed tracked dashboard/public artifact."""
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "verify-served-dashboards-test")
    public = tmp_path / "dashboard" / "public"
    public.mkdir(parents=True)
    (public / "page.html").write_text("<html>committed</html>\n")
    _git(tmp_path, "add", "--", _TRACKED)
    _git(tmp_path, "commit", "--quiet", "-m", "seed", "--", _TRACKED)
    return tmp_path


def test_clean_tracked_file_is_not_dirty(repo: Path) -> None:
    assert dirty_vs_committed(repo, _TRACKED) is False


def test_unstaged_modification_is_dirty(repo: Path) -> None:
    (repo / _TRACKED).write_text("<html>unstaged edit</html>\n")
    assert dirty_vs_committed(repo, _TRACKED) is True


def test_staged_modification_is_dirty(repo: Path) -> None:
    """The round-1 blind spot: staged-but-uncommitted must read dirty."""
    (repo / _TRACKED).write_text("<html>staged edit</html>\n")
    _git(repo, "add", "--", _TRACKED)
    assert dirty_vs_committed(repo, _TRACKED) is True


def test_staged_new_file_is_dirty_and_enumerated(repo: Path) -> None:
    """A staged NEW file is index-tracked (enumerated) and absent at HEAD (dirty)."""
    new_rel = "dashboard/public/new.json"
    (repo / new_rel).write_text("{}\n")
    _git(repo, "add", "--", new_rel)
    assert dirty_vs_committed(repo, new_rel) is True
    assert new_rel in tracked_public_files(repo)


def test_enumeration_includes_non_ascii_filename(repo: Path) -> None:
    """quotePath quoting must not silently exclude a non-ASCII artifact."""
    nonascii_rel = "dashboard/public/café.json"
    (repo / nonascii_rel).write_text("{}\n")
    _git(repo, "add", "--", nonascii_rel)
    _git(repo, "commit", "--quiet", "-m", "non-ascii artifact", "--", nonascii_rel)
    files = tracked_public_files(repo)
    assert nonascii_rel in files
    assert _TRACKED in files


def test_enumeration_raises_on_empty_match(tmp_path: Path) -> None:
    """No tracked .html/.json under dashboard/public/ => loud refusal, no vacuous PASS."""
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "verify-served-dashboards-test")
    (tmp_path / "dashboard" / "public").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="refusing to report a vacuous PASS"):
        tracked_public_files(tmp_path)

"""Pin: no `if hasattr(asw, ...)`-guarded stub loops in tests/.

Determinism stub loops over watcher passes must use bare
``monkeypatch.setattr(asw, name, ...)`` so a renamed/removed pass raises
``AttributeError`` (pytest's ``raising=True`` default) instead of silently
skipping the stub (#1303; helper-consolidation parent #1295).

Honest scope bound: this pin targets only the ``if hasattr(asw`` form under
the ``asw`` import alias — the one form that actually recurred (5 identical
sites in ``tests/test_autonomous_session_watch.py``). It does not chase
``if not hasattr(asw...)`` skip-forms, other import aliases,
``getattr(asw, name, None)``, or line-broken forms; the PRIMARY fail-loud
guard is the converted sites' bare setattr + pytest's ``raising=True``
default, and a deliberately dodging form is a new bug for code review, not a
hole this pin must close.
"""

import re
from pathlib import Path

import pytest

_HASATTR_ASW = re.compile(r"if\s+hasattr\(\s*asw\b")

_REMEDIATION = (
    "hasattr-guarded asw stub loop(s) found in tests/ — use bare "
    "monkeypatch.setattr(asw, name, ...) instead (fail-loud on a renamed watcher pass; "
    "see #1303 / #1295):\n"
)


def _scan_hasattr_asw_offences(
    repo_root: Path,
) -> tuple[list[tuple[str, int, str]], list[Path]]:
    """Scan ``tests/**/*.py`` under ``repo_root`` for ``if hasattr(asw`` lines.

    Returns ``(offences, scanned_files)``; each offence is
    ``(relpath, lineno, stripped_line)``. Skips this pin file itself —
    LOAD-BEARING: the matcher unit test's positive parametrize literals below
    live in this file and would match the line scan (do not drop the
    exclusion).
    """
    tests_dir = repo_root / "tests"
    offences: list[tuple[str, int, str]] = []
    scanned: list[Path] = []
    for path in sorted(tests_dir.rglob("*.py")):
        if path.name == Path(__file__).name:
            continue
        scanned.append(path)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _HASATTR_ASW.search(line):
                offences.append((str(path.relative_to(repo_root)), lineno, line.strip()))
    return offences, scanned


def test_no_hasattr_guarded_asw_stubs_in_tests():
    """No test under tests/ guards an asw stub with `if hasattr(asw, ...)`."""
    repo_root = Path(__file__).resolve().parents[1]
    offences, scanned = _scan_hasattr_asw_offences(repo_root)
    # Scan-sanity (anti-vacuous-pass): the scan must have visited real files,
    # including the watcher test file the 5 original sites lived in — a wrong
    # rglob root that visits 0 files FAILS instead of vacuously passing.
    assert scanned, f"scanner visited no files under {repo_root / 'tests'} — wrong repo root?"
    scanned_rel = {str(p.relative_to(repo_root)) for p in scanned}
    assert "tests/test_autonomous_session_watch.py" in scanned_rel, sorted(scanned_rel)
    assert not offences, _REMEDIATION + "\n".join(
        f"  {rel}:{lineno}: {line}" for rel, lineno, line in offences
    )


def test_scanner_finds_planted_offence(tmp_path):
    """End-to-end scanner check: a planted offending line is reported."""
    planted = tmp_path / "tests" / "test_planted_offence.py"
    planted.parent.mkdir()
    planted.write_text(
        "def test_x(monkeypatch):\n"
        '    for name in ("gc_pass",):\n'
        "        if hasattr(asw, name):\n"
        "            monkeypatch.setattr(asw, name, lambda *a, **kw: None)\n",
        encoding="utf-8",
    )
    offences, scanned = _scan_hasattr_asw_offences(tmp_path)
    assert scanned == [planted]
    assert offences == [("tests/test_planted_offence.py", 3, "if hasattr(asw, name):")]


@pytest.mark.parametrize(
    ("line", "should_match"),
    [
        # Positives — the guard shape the 5 converted sites carried. NOTE: these
        # literals are why the scanner's self-file exclusion is load-bearing.
        ("if hasattr(asw, name):", True),
        ("if hasattr(asw,name):", True),
        # Negatives — other hasattr uses in tests stay legal (asw-scoped pattern
        # only; see the module docstring's scope bound).
        ('assert hasattr(asw, "x")', False),
        ('if hasattr(mod, "build_arg_parser"):', False),
    ],
)
def test_hasattr_asw_pattern_matcher(line, should_match):
    """The compiled pattern matches the guard form and nothing broader."""
    assert bool(_HASATTR_ASW.search(line)) == should_match

"""Tests for scripts/migrate_add_legacy_why_sentinel.py — the one-shot
backfill that stamps ``legacy_why_unset: true`` onto pre-gate task bodies.

Focused on the m1 parse-error path: bodies whose YAML frontmatter is
malformed must be reported under ``parse_errors`` (not bucketed into
``skipped_no_fm``), and ``--apply`` must refuse to commit while any
parse errors exist.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module():
    """Load scripts/migrate_add_legacy_why_sentinel.py as a Python module.

    The script lives outside any importable package, so we use the
    importlib.util spec-from-file dance.
    """
    script = Path(__file__).resolve().parents[1] / "scripts" / "migrate_add_legacy_why_sentinel.py"
    spec = importlib.util.spec_from_file_location("migrate_legacy_why", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["migrate_legacy_why"] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── split_frontmatter — three-state semantics ────────────────────────────


def test_split_frontmatter_ok_parses_mapping():
    mod = _load_module()
    text = "---\nkey: value\nflag: true\n---\nbody content\n"
    status, fm, _raw, body = mod.split_frontmatter(text)
    assert status == "ok"
    assert fm == {"key": "value", "flag": True}
    assert body == "body content\n"


def test_split_frontmatter_missing_no_opener():
    mod = _load_module()
    text = "no frontmatter at all\njust a body.\n"
    status, fm, _raw, body = mod.split_frontmatter(text)
    assert status == "missing"
    assert fm is None
    assert body == text  # whole text becomes the body


def test_split_frontmatter_missing_unterminated_block():
    mod = _load_module()
    # Opens with `---\n` but no closing `\n---\n` — treat as no FM.
    text = "---\nkey: value\nbut no closer\n"
    status, fm, _raw, _body = mod.split_frontmatter(text)
    assert status == "missing"
    assert fm is None


def test_split_frontmatter_parse_error_invalid_yaml():
    """Unbalanced quote inside the YAML block → status == 'parse_error'."""
    mod = _load_module()
    text = '---\ntitle: "unbalanced quote here\nkind: experiment\n---\nbody\n'
    status, fm, raw, _body = mod.split_frontmatter(text)
    assert status == "parse_error"
    assert fm is None
    # The raw YAML block is preserved for human inspection.
    assert "unbalanced quote here" in raw


def test_split_frontmatter_parse_error_non_mapping():
    """Frontmatter that parses to a list / scalar (not a mapping) is also
    a parse error — the migration cannot add a sentinel key to it."""
    mod = _load_module()
    text = "---\n- item1\n- item2\n---\nbody\n"
    status, fm, _raw, _body = mod.split_frontmatter(text)
    assert status == "parse_error"
    assert fm is None


# ─── main() integration — parse_errors bucket + --apply refusal ──────────


def _make_tasks_tree(tmp_path: Path) -> Path:
    """Set up a minimal tasks/ tree with a body in tasks/completed/ so the
    migration script walks it (status `completed` is in SCOPED_STATUSES).
    Returns the TASKS_DIR root.
    """
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "completed" / "1").mkdir(parents=True)
    return tasks_dir


def _write_body(task_dir: Path, content: str) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    body = task_dir / "body.md"
    body.write_text(content)
    return body


def test_main_reports_parse_errors_in_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A planted body with broken YAML shows up under 'parse errors' (count
    in the scope table AND the listed path block) and main() exits 1."""
    mod = _load_module()
    tasks_dir = _make_tasks_tree(tmp_path)
    bad_body = "---\nkind: experiment\ntitle: 'unbalanced quote\n---\nbody\n"
    _write_body(tasks_dir / "completed" / "9999", bad_body)
    # Also plant a clean body so the walker has more than one entry.
    clean_body = "---\nkind: experiment\ntitle: clean\n---\nhello\n"
    _write_body(tasks_dir / "completed" / "1", clean_body)

    monkeypatch.setattr(mod, "TASKS_DIR", tasks_dir)
    monkeypatch.setattr(mod, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["migrate_add_legacy_why_sentinel.py", "--dry-run"])

    exit_code = mod.main()
    captured = capsys.readouterr()

    assert exit_code == 1, captured.out + captured.err
    # Scope table shows parse_errors=1.
    assert "parse errors       : 1" in captured.out
    # The bad body's path is listed in stderr.
    assert "9999/body.md" in captured.err
    # Bad body is NOT silently bucketed as "skipped (no FM)".
    assert "skipped (no FM)    : 0" in captured.out


def test_main_apply_refuses_when_parse_errors_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--apply`` returns non-zero and does NOT modify any body when at
    least one body has a parse error."""
    mod = _load_module()
    tasks_dir = _make_tasks_tree(tmp_path)
    bad_body = "---\nkind: experiment\ntitle: 'unbalanced quote\n---\nbody\n"
    bad_path = _write_body(tasks_dir / "completed" / "9999", bad_body)
    bad_before = bad_path.read_text()

    # A clean body that WOULD be patched if we let --apply through.
    clean_body = "---\nkind: experiment\ntitle: clean\n---\nhello\n"
    clean_path = _write_body(tasks_dir / "completed" / "1", clean_body)
    clean_before = clean_path.read_text()

    # Make tmp_path a git repo so REPO.relative_to(REPO) calls don't blow
    # up if the script tries to invoke git (the early `return 1` should
    # fire BEFORE we ever reach the git path).
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)

    monkeypatch.setattr(mod, "TASKS_DIR", tasks_dir)
    monkeypatch.setattr(mod, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["migrate_add_legacy_why_sentinel.py", "--apply"])

    exit_code = mod.main()
    captured = capsys.readouterr()

    assert exit_code == 1, captured.out + captured.err
    # The refusal message is specific to --apply mode.
    assert "refuses to commit" in captured.err
    # NOTHING was modified.
    assert bad_path.read_text() == bad_before
    assert clean_path.read_text() == clean_before


def test_main_clean_tree_no_errors_dry_run_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When every walked body parses, dry-run exits 0 with parse_errors=0."""
    mod = _load_module()
    tasks_dir = _make_tasks_tree(tmp_path)
    clean_body = "---\nkind: experiment\ntitle: clean\n---\nhello\n"
    _write_body(tasks_dir / "completed" / "1", clean_body)

    monkeypatch.setattr(mod, "TASKS_DIR", tasks_dir)
    monkeypatch.setattr(mod, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["migrate_add_legacy_why_sentinel.py", "--dry-run"])

    exit_code = mod.main()
    captured = capsys.readouterr()

    assert exit_code == 0, captured.out + captured.err
    assert "parse errors       : 0" in captured.out

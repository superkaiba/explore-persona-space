"""Regression tests for workflow_lint's bounded walks + AST parse memo (#1163).

Pins the two #1163 performance fixes STRUCTURALLY (not wall-clock):

* ``_iter_files_pruned`` never descends into cache/bulk dirs
  (``_PRUNE_DIR_NAMES``) nor a ``.claude/worktrees/`` subtree, while the
  invoking tree's own workflow surface stays fully scanned (over-pruning
  guard).
* ``_cached_parse`` memoizes by content (same text -> the identical tree
  object) and invalidates on rewrite; checks routed through it never serve
  stale cached verdicts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import (  # noqa: E402
    _cached_parse,
    _iter_files_pruned,
    check_jsonl_splitlines,
    check_no_workflow_improver_spawn,
)

# The retired-spawn pattern the improver-spawn check flags (#678).
SPAWN_LINE = 'Agent(subagent_type="workflow-improver", run_in_background=true)\n'


def test_iter_files_pruned_skips_cache_and_worktree_dirs(tmp_path):
    """Synthetic tree: the 3 real workflow files are yielded; 100 decoy files
    under `.claude/worktrees/` and `.claude/.venv/` are never enumerated."""
    claude = tmp_path / ".claude"
    real = [
        claude / "agents" / "a.md",
        claude / "agents" / "b.md",
        claude / "rules" / "r.md",
    ]
    for p in real:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("real workflow surface\n")
    wt_decoys = claude / "worktrees" / "wt-x" / ".claude" / "agents"
    wt_decoys.mkdir(parents=True)
    venv_decoys = claude / ".venv" / "lib"
    venv_decoys.mkdir(parents=True)
    for i in range(50):
        (wt_decoys / f"decoy{i}.md").write_text("decoy\n")
        (venv_decoys / f"decoy{i}.py").write_text("decoy = 1\n")
    got = set(_iter_files_pruned(claude, suffixes=frozenset({".md", ".py"})))
    assert got == set(real), got


def test_iter_files_pruned_filters_suffixes(tmp_path):
    """Only files whose suffix is in the requested set are yielded."""
    d = tmp_path / "sub"
    d.mkdir()
    (d / "keep.md").write_text("x\n")
    (d / "drop.txt").write_text("x\n")
    got = list(_iter_files_pruned(tmp_path, suffixes=frozenset({".md"})))
    assert got == [d / "keep.md"], got


def test_check_no_workflow_improver_spawn_ignores_worktree_decoy(tmp_path):
    """The spawn pattern planted under `.claude/worktrees/` is pruned; the same
    pattern in the tree's own `.claude/agents/` is flagged (exactly once)."""
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "real.md").write_text("live spawn:\n" + SPAWN_LINE)
    decoy_dir = tmp_path / ".claude" / "worktrees" / "wt-x" / ".claude" / "agents"
    decoy_dir.mkdir(parents=True)
    (decoy_dir / "decoy.md").write_text("stale sibling copy:\n" + SPAWN_LINE)
    errors = check_no_workflow_improver_spawn(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "real.md" in errors[0]
    assert "decoy.md" not in errors[0]


def test_check_no_workflow_improver_spawn_scans_invoking_worktree_surface(tmp_path):
    """Over-pruning guard (#1163 review concern): invoked FROM a worktree
    (repo_root=<...>/.claude/worktrees/wt-a), the worktree's OWN
    `.claude/agents/` surface is still scanned — pruning removes sibling /
    nested worktree COPIES, never the invoking tree's own files."""
    wt_root = tmp_path / "x" / ".claude" / "worktrees" / "wt-a"
    agents = wt_root / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "offender.md").write_text("live spawn:\n" + SPAWN_LINE)
    errors = check_no_workflow_improver_spawn(repo_root=wt_root)
    assert len(errors) == 1, errors
    assert "offender.md" in errors[0]


def test_cached_parse_memoizes_and_invalidates(tmp_path):
    """Unchanged content -> the identical cached tree object; a content
    rewrite -> a fresh parse; a syntactically-broken file -> None."""
    py = tmp_path / "mod.py"
    py.write_text("x = 1\n")
    t1 = _cached_parse(py, py.read_text())
    t2 = _cached_parse(py, py.read_text())
    assert t1 is not None
    assert t1 is t2, "memo did not engage on unchanged content"
    py.write_text("y = 2  # rewritten, different content\n")
    t3 = _cached_parse(py, py.read_text())
    assert t3 is not None
    assert t3 is not t1, "content rewrite did not invalidate the memo"
    py.write_text("def broken(:\n")
    assert _cached_parse(py, py.read_text()) is None


def test_ast_checks_findings_survive_cache(tmp_path):
    """A routed check re-run against a rewritten offender file must re-parse:
    flags first, then (offending line removed) stops flagging — pins that no
    check serves stale cached verdicts through `_AST_CACHE`."""
    root = tmp_path / "scan"
    root.mkdir()
    offender = root / "reader.py"
    offender.write_text('rows = jsonl_text.splitlines()\njsonl_text = ""\n')
    first = check_jsonl_splitlines(scan_roots=(root,))
    assert len(first) == 1, first
    offender.write_text('rows = jsonl_text.split("\\n")\njsonl_text = ""\n')
    second = check_jsonl_splitlines(scan_roots=(root,))
    assert second == [], second

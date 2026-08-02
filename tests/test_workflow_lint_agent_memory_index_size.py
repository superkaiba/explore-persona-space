"""Tests for ``workflow_lint.check_agent_memory_index_size`` (#1891).

The check enforces the agent-memory index size budget over
``.claude/agent-memory/*/MEMORY.md``: WARN above
``AGENT_MEMORY_INDEX_WARN_BYTES`` (20 KB), FAIL above
``AGENT_MEMORY_INDEX_FAIL_BYTES`` (24 KB — 1,000 B below the measured ~25,000 B
loader truncation, so the FAIL fires BEFORE any silent lesson loss), both
STRICTLY-GREATER (exactly-at-threshold passes). There is NO grandfather dict:
all live offenders were curated under WARN in the same change that introduced
the check. Only the per-agent ``MEMORY.md`` index files are scanned — per-entry
``feedback_*.md`` files load on demand and are out of scope. WARNs go to
``warn_sink`` (or stderr) and never enter the returned FAIL list.

Cases (all sizes expressed relative to the constants so a future threshold
change cannot silently invert a fixture's meaning — the #838 lesson):
(a) WARN-1 clean; (a2) exactly-at-WARN clean (strict >); (b) WARN+1 WARN only;
(c) FAIL+1 FAILs with the curation recipe; (c2) exactly-at-FAIL WARN only
(strict >); (d) missing agent-memory dir FAILs; (e) non-MEMORY.md files (a
per-entry file over FAIL, a stray top-level MEMORY.md) are ignored; (f) the
live tree PASSes (zero FAILs); (g) the threshold literals are pinned
(20_000 / 24_000 — a mistyped constant would pass every relative fixture while
failing to guard the documented ~25,000 B truncation).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import (  # noqa: E402
    AGENT_MEMORY_INDEX_FAIL_BYTES,
    AGENT_MEMORY_INDEX_WARN_BYTES,
    check_agent_memory_index_size,
)


def _write_index(tmp_path: Path, agent: str, size: int, name: str = "MEMORY.md") -> Path:
    """Write ``.claude/agent-memory/<agent>/<name>`` under tmp_path with exactly ``size`` bytes."""
    p = tmp_path / ".claude" / "agent-memory" / agent / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    assert p.stat().st_size == size
    return p


# --------------------------------------------------------------------------
# (a)/(a2) under + exactly-at the WARN threshold → clean
# --------------------------------------------------------------------------


def test_under_warn_threshold_clean(tmp_path: Path) -> None:
    _write_index(tmp_path, "small-agent", AGENT_MEMORY_INDEX_WARN_BYTES - 1)  # WARN-1
    warns: list[str] = []
    assert check_agent_memory_index_size(repo_root=tmp_path, warn_sink=warns) == []
    assert warns == []


def test_exactly_at_warn_threshold_clean(tmp_path: Path) -> None:
    _write_index(tmp_path, "atwarn-agent", AGENT_MEMORY_INDEX_WARN_BYTES)  # strict >
    warns: list[str] = []
    assert check_agent_memory_index_size(repo_root=tmp_path, warn_sink=warns) == []
    assert warns == []


# --------------------------------------------------------------------------
# (b) just over WARN → WARN only, never a FAIL
# --------------------------------------------------------------------------


def test_over_warn_threshold_warns_only(tmp_path: Path) -> None:
    _write_index(tmp_path, "warned-agent", AGENT_MEMORY_INDEX_WARN_BYTES + 1)  # WARN+1
    warns: list[str] = []
    assert check_agent_memory_index_size(repo_root=tmp_path, warn_sink=warns) == []
    assert len(warns) == 1, warns
    assert "warned-agent/MEMORY.md" in warns[0]
    assert str(AGENT_MEMORY_INDEX_WARN_BYTES + 1) in warns[0]


# --------------------------------------------------------------------------
# (c)/(c2) FAIL threshold: just over FAILs with the curation recipe;
# exactly-at WARNs only (strict >)
# --------------------------------------------------------------------------


def test_over_fail_threshold_fails_with_curation_recipe(tmp_path: Path) -> None:
    _write_index(tmp_path, "big-agent", AGENT_MEMORY_INDEX_FAIL_BYTES + 1)  # FAIL+1
    warns: list[str] = []
    errors = check_agent_memory_index_size(repo_root=tmp_path, warn_sink=warns)
    assert len(errors) == 1, errors
    assert "big-agent/MEMORY.md" in errors[0]
    assert str(AGENT_MEMORY_INDEX_FAIL_BYTES + 1) in errors[0]
    assert str(AGENT_MEMORY_INDEX_FAIL_BYTES) in errors[0]
    # The FAIL message names the curation recipe, not just the number.
    assert "curate" in errors[0]
    assert "per-entry file" in errors[0]
    assert warns == []


def test_exactly_at_fail_threshold_warns_only(tmp_path: Path) -> None:
    _write_index(tmp_path, "atfail-agent", AGENT_MEMORY_INDEX_FAIL_BYTES)  # strict >
    warns: list[str] = []
    assert check_agent_memory_index_size(repo_root=tmp_path, warn_sink=warns) == []
    assert len(warns) == 1, warns
    assert "atfail-agent/MEMORY.md" in warns[0]


# --------------------------------------------------------------------------
# (d) missing .claude/agent-memory dir → one FAIL
# --------------------------------------------------------------------------


def test_missing_agent_memory_dir_fails(tmp_path: Path) -> None:
    errors = check_agent_memory_index_size(repo_root=tmp_path, warn_sink=[])
    assert len(errors) == 1, errors
    assert "missing" in errors[0]


# --------------------------------------------------------------------------
# (e) non-MEMORY.md files are ignored: an oversized per-entry file and a stray
# top-level agent-memory/MEMORY.md (no agent dir) never WARN/FAIL
# --------------------------------------------------------------------------


def test_non_memory_md_files_ignored(tmp_path: Path) -> None:
    _write_index(tmp_path, "ok-agent", 1_000)
    # Oversized per-entry file in the same agent dir: out of scope by design.
    _write_index(
        tmp_path, "ok-agent", AGENT_MEMORY_INDEX_FAIL_BYTES + 5_000, name="feedback_big_entry.md"
    )
    # A stray MEMORY.md directly under agent-memory/ (no per-agent dir) does
    # not match the */MEMORY.md glob either.
    stray = tmp_path / ".claude" / "agent-memory" / "MEMORY.md"
    stray.write_bytes(b"x" * (AGENT_MEMORY_INDEX_FAIL_BYTES + 5_000))
    warns: list[str] = []
    assert check_agent_memory_index_size(repo_root=tmp_path, warn_sink=warns) == []
    assert warns == []


@pytest.mark.parametrize("agent", ["a-agent", "b-agent"])
def test_each_oversized_index_reported_independently(tmp_path: Path, agent: str) -> None:
    """Two agents' indexes are checked independently — only the oversized one FAILs."""
    _write_index(tmp_path, agent, AGENT_MEMORY_INDEX_FAIL_BYTES + 1)
    _write_index(tmp_path, "clean-agent", 1_000)
    errors = check_agent_memory_index_size(repo_root=tmp_path, warn_sink=[])
    assert len(errors) == 1, errors
    assert f"{agent}/MEMORY.md" in errors[0]


# --------------------------------------------------------------------------
# (f) the live tree PASSes (zero FAILs; WARNs allowed). workflow_lint is
# imported from this worktree's scripts/, so its _REPO_ROOT (repo_root=None)
# resolves THIS tree's root — the same tree the no-flags default run lints.
# --------------------------------------------------------------------------


def test_live_tree_passes() -> None:
    warns: list[str] = []
    errors = check_agent_memory_index_size(warn_sink=warns)
    assert errors == [], errors


# --------------------------------------------------------------------------
# (g) pin the threshold literals: every other fixture derives from the
# constants, so a mistyped constant (e.g. 240_000) would pass them all while
# failing to guard the documented ~25,000 B loader truncation.
# --------------------------------------------------------------------------


def test_threshold_literals_pinned() -> None:
    assert AGENT_MEMORY_INDEX_WARN_BYTES == 20_000
    assert AGENT_MEMORY_INDEX_FAIL_BYTES == 24_000
    assert AGENT_MEMORY_INDEX_WARN_BYTES < AGENT_MEMORY_INDEX_FAIL_BYTES < 25_000


# --------------------------------------------------------------------------
# (h) pre-commit hook coverage (#1925): direct-to-main memory-save commits are
# the primary regrowth channel (worktree-mediated gates never see them), so a
# local hook must run --check-agent-memory-index-size when any per-agent
# MEMORY.md index — or the lint itself — changes. Mirror of
# test_workflow_lint_agent_spec_size.py::test_precommit_hook_covers_agent_spec_size.
# --------------------------------------------------------------------------


def test_precommit_hook_covers_agent_memory_index_size() -> None:
    """.pre-commit-config.yaml must carry a local hook running
    --check-agent-memory-index-size whose files: regex covers the per-agent
    ``.claude/agent-memory/<agent>/MEMORY.md`` indexes AND the lint itself
    (threshold edits move the verdict), but NOT per-entry memory files
    (they load on demand and are out of the check's scope)."""
    cfg = yaml.safe_load((_HERE.parent / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    local_hooks = [h for repo in cfg["repos"] if repo["repo"] == "local" for h in repo["hooks"]]
    matching = [h for h in local_hooks if "--check-agent-memory-index-size" in h.get("entry", "")]
    assert matching, "no pre-commit hook runs --check-agent-memory-index-size (#1925)"
    assert any(
        re.search(h["files"], ".claude/agent-memory/analyzer/MEMORY.md")
        and re.search(h["files"], "scripts/workflow_lint.py")
        and not re.search(h["files"], ".claude/agent-memory/analyzer/feedback_foo.md")
        and not h.get("pass_filenames", True)
        for h in matching
        if "files" in h
    ), f"no matching hook covers agent-memory MEMORY.md + the lint, filenames off: {matching}"

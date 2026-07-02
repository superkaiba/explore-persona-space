"""Tests for ``workflow_lint.check_agent_spec_size`` (#829).

The check enforces the agent-spec size budget over ``.claude/agents/*.md``:
WARN above ``AGENT_SPEC_WARN_BYTES`` (40 KB), FAIL above
``AGENT_SPEC_FAIL_BYTES`` (70 KB), both STRICTLY-GREATER (exactly-at-threshold
passes). Files in ``AGENT_SPEC_SIZE_GRANDFATHER`` WARN above the FAIL threshold
while under their per-file cap and FAIL above it (the regrowth ratchet).
Grandfather hygiene FAILs a stale entry (file missing) and an entry whose file
dropped to <= the FAIL threshold ("remove the entry"); a config self-check
FAILs any cap <= the FAIL threshold. WARNs go to ``warn_sink`` (or stderr) and
never enter the returned FAIL list.

Cases: (a) 39,999 B clean; (a2) exactly 40,000 B clean (strict >); (b) 40,001 B
WARN only; (c) 70,001 B not grandfathered FAILs; (c2) exactly 70,000 B WARN
only (strict >); (d) grandfathered 71,000 B under cap 74,000 WARNs only; (e)
grandfathered 75,000 B over cap 74,000 FAILs (ratchet); (f) stale grandfather
entry FAILs; (g) grandfathered file at 60,000 B FAILs "remove the entry"; (h)
missing agents dir FAILs; (i) the live tree PASSes (zero FAILs, WARNs allowed);
(j) a grandfather cap <= the FAIL threshold FAILs "cap must exceed".
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
from workflow_lint import (  # noqa: E402
    AGENT_SPEC_FAIL_BYTES,
    AGENT_SPEC_WARN_BYTES,
    check_agent_spec_size,
)


def _write_agent(tmp_path: Path, name: str, size: int) -> Path:
    """Write ``.claude/agents/<name>`` under tmp_path with exactly ``size`` bytes."""
    p = tmp_path / ".claude" / "agents" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    assert p.stat().st_size == size
    return p


@pytest.fixture()
def empty_grandfather(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the real grandfather dict so tmp fixtures see no stale entries."""
    monkeypatch.setattr(workflow_lint, "AGENT_SPEC_SIZE_GRANDFATHER", {})


# --------------------------------------------------------------------------
# (a)/(a2) under + exactly-at the WARN threshold → clean
# --------------------------------------------------------------------------


def test_under_warn_threshold_clean(tmp_path: Path, empty_grandfather: None) -> None:
    _write_agent(tmp_path, "small.md", AGENT_SPEC_WARN_BYTES - 1)  # 39,999
    warns: list[str] = []
    assert check_agent_spec_size(repo_root=tmp_path, warn_sink=warns) == []
    assert warns == []


def test_exactly_at_warn_threshold_clean(tmp_path: Path, empty_grandfather: None) -> None:
    _write_agent(tmp_path, "atwarn.md", AGENT_SPEC_WARN_BYTES)  # 40,000 — strict >
    warns: list[str] = []
    assert check_agent_spec_size(repo_root=tmp_path, warn_sink=warns) == []
    assert warns == []


# --------------------------------------------------------------------------
# (b) just over WARN → WARN only, never a FAIL
# --------------------------------------------------------------------------


def test_over_warn_threshold_warns_only(tmp_path: Path, empty_grandfather: None) -> None:
    _write_agent(tmp_path, "warned.md", AGENT_SPEC_WARN_BYTES + 1)  # 40,001
    warns: list[str] = []
    assert check_agent_spec_size(repo_root=tmp_path, warn_sink=warns) == []
    assert len(warns) == 1, warns
    assert "warned.md" in warns[0]
    assert str(AGENT_SPEC_WARN_BYTES + 1) in warns[0]


# --------------------------------------------------------------------------
# (c)/(c2) FAIL threshold: just over FAILs (not grandfathered); exactly-at WARNs
# --------------------------------------------------------------------------


def test_over_fail_threshold_not_grandfathered_fails(
    tmp_path: Path, empty_grandfather: None
) -> None:
    _write_agent(tmp_path, "big.md", AGENT_SPEC_FAIL_BYTES + 1)  # 70,001
    warns: list[str] = []
    errors = check_agent_spec_size(repo_root=tmp_path, warn_sink=warns)
    assert len(errors) == 1, errors
    assert "big.md" in errors[0]
    assert str(AGENT_SPEC_FAIL_BYTES + 1) in errors[0]
    assert str(AGENT_SPEC_FAIL_BYTES) in errors[0]
    assert warns == []


def test_exactly_at_fail_threshold_warns_only(tmp_path: Path, empty_grandfather: None) -> None:
    _write_agent(tmp_path, "atfail.md", AGENT_SPEC_FAIL_BYTES)  # 70,000 — strict >
    warns: list[str] = []
    assert check_agent_spec_size(repo_root=tmp_path, warn_sink=warns) == []
    assert len(warns) == 1, warns
    assert "atfail.md" in warns[0]


# --------------------------------------------------------------------------
# (d) grandfathered, over FAIL but under its cap → WARN only (must NOT collapse
# into the retired-entry branch)
# --------------------------------------------------------------------------


def test_grandfathered_under_cap_warns_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow_lint, "AGENT_SPEC_SIZE_GRANDFATHER", {"gf.md": 74_000})
    _write_agent(tmp_path, "gf.md", 71_000)
    warns: list[str] = []
    assert check_agent_spec_size(repo_root=tmp_path, warn_sink=warns) == []
    assert len(warns) == 1, warns
    assert "gf.md" in warns[0]
    assert "grandfathered" in warns[0]


# --------------------------------------------------------------------------
# (e) grandfathered, over its cap → FAIL (regrowth ratchet)
# --------------------------------------------------------------------------


def test_grandfathered_over_cap_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow_lint, "AGENT_SPEC_SIZE_GRANDFATHER", {"gf.md": 74_000})
    _write_agent(tmp_path, "gf.md", 75_000)
    warns: list[str] = []
    errors = check_agent_spec_size(repo_root=tmp_path, warn_sink=warns)
    assert len(errors) == 1, errors
    assert "gf.md" in errors[0]
    assert "ratchet" in errors[0]
    assert "74000" in errors[0].replace(",", "")


# --------------------------------------------------------------------------
# (f) grandfather entry whose file does not exist → FAIL "stale"
# --------------------------------------------------------------------------


def test_stale_grandfather_entry_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow_lint, "AGENT_SPEC_SIZE_GRANDFATHER", {"ghost.md": 74_000})
    _write_agent(tmp_path, "ok.md", 1_000)  # dir exists, entry's file does not
    errors = check_agent_spec_size(repo_root=tmp_path, warn_sink=[])
    assert len(errors) == 1, errors
    assert "ghost.md" in errors[0]
    assert "stale" in errors[0]


# --------------------------------------------------------------------------
# (g) grandfathered file dropped to <= FAIL threshold → FAIL "remove the entry"
# --------------------------------------------------------------------------


def test_grandfathered_below_fail_threshold_fails_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow_lint, "AGENT_SPEC_SIZE_GRANDFATHER", {"gf.md": 74_000})
    _write_agent(tmp_path, "gf.md", 60_000)  # <= 70,000 — no longer needs it
    errors = check_agent_spec_size(repo_root=tmp_path, warn_sink=[])
    assert len(errors) == 1, errors
    assert "gf.md" in errors[0]
    assert "remove the entry" in errors[0]


# --------------------------------------------------------------------------
# (h) missing .claude/agents dir → one FAIL
# --------------------------------------------------------------------------


def test_missing_agents_dir_fails(tmp_path: Path, empty_grandfather: None) -> None:
    errors = check_agent_spec_size(repo_root=tmp_path, warn_sink=[])
    assert len(errors) == 1, errors
    assert "missing" in errors[0]


# --------------------------------------------------------------------------
# (i) the live tree PASSes (zero FAILs; WARNs allowed). workflow_lint is
# imported from this worktree's scripts/, so its _REPO_ROOT (repo_root=None)
# resolves THIS tree's root — the same tree the no-flags default run lints.
# --------------------------------------------------------------------------


def test_live_tree_passes() -> None:
    warns: list[str] = []
    errors = check_agent_spec_size(warn_sink=warns)
    assert errors == [], errors


# --------------------------------------------------------------------------
# (j) config self-check: a grandfather cap <= AGENT_SPEC_FAIL_BYTES → FAIL
# --------------------------------------------------------------------------


def test_grandfather_cap_at_or_below_fail_threshold_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow_lint, "AGENT_SPEC_SIZE_GRANDFATHER", {"gf.md": 65_000})
    _write_agent(tmp_path, "gf.md", 71_000)
    errors = check_agent_spec_size(repo_root=tmp_path, warn_sink=[])
    assert any("cap must exceed" in e for e in errors), errors

"""Tests for ``workflow_lint.check_agent_spec_size`` (#829, thresholds #838).

The check enforces the agent-spec size budget over ``.claude/agents/*.md``:
WARN above ``AGENT_SPEC_WARN_BYTES`` (28 KB as of #838), FAIL above
``AGENT_SPEC_FAIL_BYTES`` (40 KB as of #838), both STRICTLY-GREATER
(exactly-at-threshold passes). Files in ``AGENT_SPEC_SIZE_GRANDFATHER`` WARN
above the FAIL threshold while under their per-file cap and FAIL above it (the
regrowth ratchet). Grandfather hygiene FAILs a stale entry (file missing), an
entry whose file dropped to <= the FAIL threshold ("remove the entry"), and an
entry whose cap sits more than ``AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES``
(3,000 B, strict >) above the live file size (loose/stale cap, #986); a
config self-check FAILs any cap <= the FAIL threshold. WARNs go to
``warn_sink`` (or stderr) and never enter the returned FAIL list.

Cases (all sizes expressed relative to the constants so a future threshold
change cannot silently invert a fixture's meaning — the #838 lesson: the old
literal 60,000/65,000 fixtures flipped branches when FAIL dropped 70K -> 40K):
(a) WARN-1 clean; (a2) exactly-at-WARN clean (strict >); (b) WARN+1 WARN only;
(c) FAIL+1 not grandfathered FAILs; (c2) exactly-at-FAIL WARN only (strict >);
(d) grandfathered FAIL+1,000 under cap FAIL+4,000 WARNs only; (e) grandfathered
FAIL+5,000 over cap FAIL+4,000 FAILs (ratchet); (f) stale grandfather entry
FAILs; (g) grandfathered file at FAIL-1,000 FAILs "remove the entry"; (h)
missing agents dir FAILs; (i) the live tree PASSes (zero FAILs, WARNs allowed);
(j) a grandfather cap <= the FAIL threshold FAILs "cap must exceed";
(k) grandfathered headroom exactly at the bound passes (strict >);
(l) grandfathered headroom over the bound FAILs "headroom"; (m) the bound
constant is literally 3_000.
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
    AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES,
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
    _write_agent(tmp_path, "small.md", AGENT_SPEC_WARN_BYTES - 1)  # WARN-1
    warns: list[str] = []
    assert check_agent_spec_size(repo_root=tmp_path, warn_sink=warns) == []
    assert warns == []


def test_exactly_at_warn_threshold_clean(tmp_path: Path, empty_grandfather: None) -> None:
    _write_agent(tmp_path, "atwarn.md", AGENT_SPEC_WARN_BYTES)  # exactly-at — strict >
    warns: list[str] = []
    assert check_agent_spec_size(repo_root=tmp_path, warn_sink=warns) == []
    assert warns == []


# --------------------------------------------------------------------------
# (b) just over WARN → WARN only, never a FAIL
# --------------------------------------------------------------------------


def test_over_warn_threshold_warns_only(tmp_path: Path, empty_grandfather: None) -> None:
    _write_agent(tmp_path, "warned.md", AGENT_SPEC_WARN_BYTES + 1)  # WARN+1
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
    _write_agent(tmp_path, "big.md", AGENT_SPEC_FAIL_BYTES + 1)  # FAIL+1
    warns: list[str] = []
    errors = check_agent_spec_size(repo_root=tmp_path, warn_sink=warns)
    assert len(errors) == 1, errors
    assert "big.md" in errors[0]
    assert str(AGENT_SPEC_FAIL_BYTES + 1) in errors[0]
    assert str(AGENT_SPEC_FAIL_BYTES) in errors[0]
    assert warns == []


def test_exactly_at_fail_threshold_warns_only(tmp_path: Path, empty_grandfather: None) -> None:
    _write_agent(tmp_path, "atfail.md", AGENT_SPEC_FAIL_BYTES)  # exactly-at — strict >
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
    cap = AGENT_SPEC_FAIL_BYTES + 4_000
    monkeypatch.setattr(workflow_lint, "AGENT_SPEC_SIZE_GRANDFATHER", {"gf.md": cap})
    _write_agent(tmp_path, "gf.md", AGENT_SPEC_FAIL_BYTES + 1_000)  # over FAIL, under cap
    warns: list[str] = []
    assert check_agent_spec_size(repo_root=tmp_path, warn_sink=warns) == []
    assert len(warns) == 1, warns
    assert "gf.md" in warns[0]
    assert "grandfathered" in warns[0]


# --------------------------------------------------------------------------
# (e) grandfathered, over its cap → FAIL (regrowth ratchet)
# --------------------------------------------------------------------------


def test_grandfathered_over_cap_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cap = AGENT_SPEC_FAIL_BYTES + 4_000
    monkeypatch.setattr(workflow_lint, "AGENT_SPEC_SIZE_GRANDFATHER", {"gf.md": cap})
    _write_agent(tmp_path, "gf.md", cap + 1_000)  # over its cap → ratchet FAIL
    warns: list[str] = []
    errors = check_agent_spec_size(repo_root=tmp_path, warn_sink=warns)
    assert len(errors) == 1, errors
    assert "gf.md" in errors[0]
    assert "ratchet" in errors[0]
    assert str(cap) in errors[0].replace(",", "")


# --------------------------------------------------------------------------
# (f) grandfather entry whose file does not exist → FAIL "stale"
# --------------------------------------------------------------------------


def test_stale_grandfather_entry_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        workflow_lint, "AGENT_SPEC_SIZE_GRANDFATHER", {"ghost.md": AGENT_SPEC_FAIL_BYTES + 4_000}
    )
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
    monkeypatch.setattr(
        workflow_lint, "AGENT_SPEC_SIZE_GRANDFATHER", {"gf.md": AGENT_SPEC_FAIL_BYTES + 4_000}
    )
    _write_agent(tmp_path, "gf.md", AGENT_SPEC_FAIL_BYTES - 1_000)  # <= FAIL — remove it
    errors = check_agent_spec_size(repo_root=tmp_path, warn_sink=[])
    assert len(errors) == 1, errors
    assert "gf.md" in errors[0]
    assert "remove the entry" in errors[0]
    # The headroom message also contains "remove the entry" — this is the ONLY
    # assert that detects a swapped remove-entry/headroom branch order (#986:
    # cap sits 5,000 B above the file here, over the headroom bound).
    assert "headroom" not in errors[0]


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
    monkeypatch.setattr(
        workflow_lint, "AGENT_SPEC_SIZE_GRANDFATHER", {"gf.md": AGENT_SPEC_FAIL_BYTES - 5_000}
    )
    _write_agent(tmp_path, "gf.md", AGENT_SPEC_FAIL_BYTES + 1_000)
    errors = check_agent_spec_size(repo_root=tmp_path, warn_sink=[])
    assert any("cap must exceed" in e for e in errors), errors


# --------------------------------------------------------------------------
# (k)/(l) grandfather-cap headroom: exactly-at-bound passes (strict >);
# over-bound FAILs (loose cap-raise / stale cap after a trim — #986)
# --------------------------------------------------------------------------


def test_grandfather_headroom_at_bound_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    size = AGENT_SPEC_FAIL_BYTES + 1_000
    cap = size + AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES  # headroom exactly at bound
    monkeypatch.setattr(workflow_lint, "AGENT_SPEC_SIZE_GRANDFATHER", {"gf.md": cap})
    _write_agent(tmp_path, "gf.md", size)
    warns: list[str] = []
    assert check_agent_spec_size(repo_root=tmp_path, warn_sink=warns) == []
    assert len(warns) == 1, warns  # the ordinary "grandfathered; under its cap" WARN


def test_grandfather_headroom_over_bound_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    size = AGENT_SPEC_FAIL_BYTES + 1_000
    cap = size + AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES + 1  # headroom bound+1
    monkeypatch.setattr(workflow_lint, "AGENT_SPEC_SIZE_GRANDFATHER", {"gf.md": cap})
    _write_agent(tmp_path, "gf.md", size)
    errors = check_agent_spec_size(repo_root=tmp_path, warn_sink=[])
    assert len(errors) == 1, errors
    assert "gf.md" in errors[0]
    assert "headroom" in errors[0]
    assert str(cap) in errors[0]


def test_grandfather_headroom_bound_is_3000() -> None:
    """Pin the literal bound (statistics-critic Must-Fix, round 1).

    Every other headroom test derives its fixture from the constant, so a
    mistyped constant (e.g. 30_000) would pass all of them AND the live
    tree while failing to mechanize the documented "<=3 KB margin"
    convention — a false-green. This literal pin closes that class.
    """
    assert AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES == 3_000

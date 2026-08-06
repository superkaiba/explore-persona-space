"""Tests for ``workflow_lint.check_skill_doc_size`` (the per-skill size ratchet).

Skill docs (`.claude/skills/**/*.md`) are loaded whole on Skill invocation and
had NO size cap — which is how `issue/SKILL.md` reached 916 KB before the
2026-08-05 trim. Budget: WARN above ``SKILL_DOC_WARN_BYTES`` (40,000 B), FAIL
above ``SKILL_DOC_FAIL_BYTES`` (60,000 B), both STRICTLY-GREATER (exactly-at
passes). Grandfather ratchet mirrors the agent-spec check: cap = measured +
<= ``SKILL_DOC_GRANDFATHER_MAX_HEADROOM_BYTES`` (3,000 B); regrowth past the
cap FAILs; a file trimmed to <= FAIL means "remove the entry"; a cap sitting
> 3,000 B above the live size is a loose/stale cap and FAILs. Exempt:
``SKILL_DOC_GENERATED_EXEMPT`` (e.g. `issue/markers.md`, an emit-tables
derived table whose compaction lever is workflow.yaml) and docs under a
``SKILL_DOC_EXEMPT_DIR_SEGMENTS`` directory (exemplars / templates /
lw-post-examples — data, not instructions). WARNs go to ``warn_sink`` (or
stderr) and never enter the returned FAIL list.

Cases sized relative to the constants so a future threshold change cannot
silently invert a fixture's meaning (the #838 lesson).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import pytest  # noqa: E402
import workflow_lint  # noqa: E402
from workflow_lint import (  # noqa: E402
    SKILL_DOC_EXEMPT_DIR_SEGMENTS,
    SKILL_DOC_FAIL_BYTES,
    SKILL_DOC_GENERATED_EXEMPT,
    SKILL_DOC_GRANDFATHER_MAX_HEADROOM_BYTES,
    SKILL_DOC_SIZE_GRANDFATHER,
    SKILL_DOC_WARN_BYTES,
    check_skill_doc_size,
)

_REPO_ROOT = _HERE.parent


@pytest.fixture()
def no_grandfather(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty the grandfather dict so synthetic-tree ladder fixtures don't trip
    the stale-entry hygiene sweep for the real entries."""
    monkeypatch.setattr(workflow_lint, "SKILL_DOC_SIZE_GRANDFATHER", {})


def _write_doc(tmp_path: Path, rel: str, size: int) -> Path:
    """Write ``.claude/skills/<rel>`` under tmp_path with exactly ``size`` bytes."""
    p = tmp_path / ".claude" / "skills" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    assert p.stat().st_size == size
    return p


def _run(tmp_path: Path) -> tuple[list[str], list[str]]:
    warns: list[str] = []
    errors = check_skill_doc_size(repo_root=tmp_path, warn_sink=warns)
    return errors, warns


# ── size ladder ──────────────────────────────────────────────────────────────


def test_under_warn_clean(tmp_path: Path, no_grandfather: None) -> None:
    _write_doc(tmp_path, "foo/SKILL.md", SKILL_DOC_WARN_BYTES - 1)
    errors, warns = _run(tmp_path)
    assert errors == []
    assert warns == []


def test_exactly_at_warn_clean(tmp_path: Path, no_grandfather: None) -> None:
    """Strictly-greater semantics: exactly-at-WARN passes clean."""
    _write_doc(tmp_path, "foo/SKILL.md", SKILL_DOC_WARN_BYTES)
    errors, warns = _run(tmp_path)
    assert errors == []
    assert warns == []


def test_over_warn_warns_only(tmp_path: Path, no_grandfather: None) -> None:
    _write_doc(tmp_path, "foo/SKILL.md", SKILL_DOC_WARN_BYTES + 1)
    errors, warns = _run(tmp_path)
    assert errors == []
    assert len(warns) == 1
    assert "foo/SKILL.md" in warns[0]
    assert str(SKILL_DOC_WARN_BYTES) in warns[0]


def test_exactly_at_fail_warns_only(tmp_path: Path, no_grandfather: None) -> None:
    """Strictly-greater semantics: exactly-at-FAIL is only a WARN."""
    _write_doc(tmp_path, "foo/SKILL.md", SKILL_DOC_FAIL_BYTES)
    errors, warns = _run(tmp_path)
    assert errors == []
    assert len(warns) == 1


def test_over_fail_ungrandfathered_fails(tmp_path: Path, no_grandfather: None) -> None:
    _write_doc(tmp_path, "foo/SKILL.md", SKILL_DOC_FAIL_BYTES + 1)
    errors, warns = _run(tmp_path)
    assert len(errors) == 1
    assert "foo/SKILL.md" in errors[0]
    assert str(SKILL_DOC_FAIL_BYTES) in errors[0]
    assert warns == []


def test_support_md_files_are_in_scope(tmp_path: Path, no_grandfather: None) -> None:
    """Non-SKILL.md support docs (e.g. a spec or markers file) are sized too."""
    _write_doc(tmp_path, "bar/reference.md", SKILL_DOC_FAIL_BYTES + 1)
    errors, _ = _run(tmp_path)
    assert any("bar/reference.md" in e for e in errors)


def test_missing_skills_dir_fails(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir(parents=True)
    errors = check_skill_doc_size(repo_root=tmp_path)
    assert len(errors) == 1
    assert "missing" in errors[0]


# ── exemptions ───────────────────────────────────────────────────────────────


def test_generated_exempt_never_sized(tmp_path: Path) -> None:
    """`issue/markers.md` is emit-tables-generated — never sized, any size."""
    assert "issue/markers.md" in SKILL_DOC_GENERATED_EXEMPT
    _write_doc(tmp_path, "issue/markers.md", SKILL_DOC_FAIL_BYTES * 2)
    errors, warns = _run(tmp_path)
    assert not any("markers.md" in e for e in errors)
    assert not any("markers.md" in w for w in warns)


def test_exempt_dir_segments_never_sized(tmp_path: Path) -> None:
    """exemplars / templates / lw-post-examples dirs are data, not instructions."""
    for seg in sorted(SKILL_DOC_EXEMPT_DIR_SEGMENTS):
        _write_doc(tmp_path, f"some-skill/{seg}/big.md", SKILL_DOC_FAIL_BYTES * 2)
    errors, warns = _run(tmp_path)
    assert not any("big.md" in e for e in errors)
    assert not any("big.md" in w for w in warns)


def test_exempt_segment_only_matches_directories(tmp_path: Path) -> None:
    """A FILE named e.g. `templates.md` is not a directory match — still sized."""
    _write_doc(tmp_path, "some-skill/templates.md", SKILL_DOC_FAIL_BYTES + 1)
    errors, _ = _run(tmp_path)
    assert any("templates.md" in e for e in errors)


# ── grandfather ratchet ──────────────────────────────────────────────────────


def _seed_live_grandfather(tmp_path: Path) -> None:
    """Materialize every real grandfather entry just under its cap so the
    hygiene sweep stays quiet while a specific fixture is under test."""
    for rel, cap in SKILL_DOC_SIZE_GRANDFATHER.items():
        _write_doc(tmp_path, rel, cap - 1)


def test_grandfathered_under_cap_warns_only(tmp_path: Path) -> None:
    _seed_live_grandfather(tmp_path)
    errors, warns = _run(tmp_path)
    assert errors == []
    assert len(warns) == len(SKILL_DOC_SIZE_GRANDFATHER)
    assert all("grandfathered" in w for w in warns)


def test_grandfathered_over_cap_fails_regrowth(tmp_path: Path) -> None:
    _seed_live_grandfather(tmp_path)
    rel, cap = next(iter(sorted(SKILL_DOC_SIZE_GRANDFATHER.items())))
    _write_doc(tmp_path, rel, cap + 1)
    errors, _ = _run(tmp_path)
    assert len(errors) == 1
    assert rel in errors[0]
    assert "regrew" in errors[0]


def test_grandfather_stale_entry_fails(tmp_path: Path) -> None:
    _seed_live_grandfather(tmp_path)
    rel = next(iter(sorted(SKILL_DOC_SIZE_GRANDFATHER)))
    (tmp_path / ".claude" / "skills" / rel).unlink()
    errors, _ = _run(tmp_path)
    assert len(errors) == 1
    assert "stale grandfather" in errors[0]
    assert rel in errors[0]


def test_grandfather_trimmed_under_fail_demands_removal(tmp_path: Path) -> None:
    _seed_live_grandfather(tmp_path)
    rel = next(iter(sorted(SKILL_DOC_SIZE_GRANDFATHER)))
    _write_doc(tmp_path, rel, SKILL_DOC_FAIL_BYTES)
    errors, _ = _run(tmp_path)
    assert len(errors) == 1
    assert "remove the entry (ratchet down)" in errors[0]


def test_grandfather_loose_cap_fails(tmp_path: Path) -> None:
    _seed_live_grandfather(tmp_path)
    rel, cap = next(iter(sorted(SKILL_DOC_SIZE_GRANDFATHER.items())))
    # Trim the file so the cap sits > max headroom above it (still > FAIL).
    slim = cap - SKILL_DOC_GRANDFATHER_MAX_HEADROOM_BYTES - 1
    assert slim > SKILL_DOC_FAIL_BYTES, "fixture assumes a large-capped entry"
    _write_doc(tmp_path, rel, slim)
    errors, _ = _run(tmp_path)
    assert len(errors) == 1
    assert "max headroom" in errors[0]


def test_live_grandfather_caps_have_sane_headroom() -> None:
    """Config self-check parity: every live cap exceeds FAIL, and every live
    grandfathered file exists with headroom within the 3 KB budget."""
    skills_dir = _REPO_ROOT / ".claude" / "skills"
    for rel, cap in SKILL_DOC_SIZE_GRANDFATHER.items():
        assert cap > SKILL_DOC_FAIL_BYTES, rel
        p = skills_dir / rel
        assert p.is_file(), f"stale grandfather entry: {rel}"
        size = p.stat().st_size
        assert size > SKILL_DOC_FAIL_BYTES, f"{rel} no longer needs grandfathering"
        assert cap - size <= SKILL_DOC_GRANDFATHER_MAX_HEADROOM_BYTES, (
            f"{rel}: loose cap {cap} vs measured {size}"
        )
        assert size <= cap, f"{rel}: regrew past its cap"


def test_live_tree_passes_no_fails() -> None:
    """The real .claude/skills tree returns zero FAILs (WARNs allowed —
    grandfathered files plus any 40-60 KB docs report via warn_sink)."""
    warns: list[str] = []
    assert check_skill_doc_size(repo_root=_REPO_ROOT, warn_sink=warns) == []


def test_threshold_literals_pinned() -> None:
    """Pin the documented budget so a mistyped constant cannot silently pass
    every relative fixture while guarding the wrong budget."""
    assert SKILL_DOC_WARN_BYTES == 40_000
    assert SKILL_DOC_FAIL_BYTES == 60_000
    assert SKILL_DOC_GRANDFATHER_MAX_HEADROOM_BYTES == 3_000

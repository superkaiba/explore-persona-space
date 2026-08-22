"""Pin the #2072 lean-twin registration-mechanic documentation + repo files.

Incident #2061 (2026-08-04): the Step 5b autocompact-thrash escalation to
`code-reviewer-lean` was refused with "agent type not found" even though the
6 project-level `.claude/agents/*-lean.md` files had landed on main ~1.5 h
earlier. Root cause (diagnosed live in #2072): agent types register at
SESSION START from the session cwd's `.claude/agents/` plus user-global
`~/.claude/agents/` — a file added mid-session never registers, and the
failing round's worktree cwd was cut before the lean files landed. The fix
installs the 6 lean twins user-global as SYMLINKS to the repo files (outside
git) and documents the mechanic at the two consuming surfaces.

These tests pin, REPO-FILE-SCOPED ONLY (never `~/.claude/` state, which is
machine-specific — pods/CI have different homes):

1. the SKILL.md Step 5b "Autocompact-thrash respawn recipe" region carries
   the registration-mechanic clause + the symlink-install pointer;
2. the context-hygiene.md autocompact-thrash bullet carries the mechanic;
3. all repo `.claude/agents/*-lean.md` lean-twin files in LEAN_TWINS exist
   (6 at #2072; +5 codex composer twins at #2472).

Prose assertions run on whitespace-NORMALIZED text (the files wrap prose
mid-phrase, so a required phrase can span lines).
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
CONTEXT_HYGIENE_MD = REPO_ROOT / ".claude" / "rules" / "context-hygiene.md"
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"

RECIPE_HEADING = "**Autocompact-thrash respawn recipe"
RECIPE_END_ANCHOR = "The existing marker-keyed no-show path"

LEAN_TWINS = (
    "code-reviewer-lean.md",
    "codex-clean-result-critic-lean.md",
    "codex-code-reviewer-lean.md",
    "codex-critic-lean.md",
    "codex-follow-up-critic-lean.md",
    "codex-interpretation-critic-lean.md",
    "consistency-checker-lean.md",
    "critic-lean.md",
    "experiment-implementer-lean.md",
    "implementer-lean.md",
    "planner-lean.md",
)

CODEX_COMPOSER_TWINS = tuple(n for n in LEAN_TWINS if n.startswith("codex-"))


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-tolerant match)."""
    return re.sub(r"\s+", " ", text)


def _respawn_recipe_region() -> str:
    """Slice of SKILL.md between the recipe heading and the no-show-path anchor."""
    assert SKILL_MD.exists(), f"missing {SKILL_MD}"
    text = issue_skill_text()
    start = text.find(RECIPE_HEADING)
    assert start != -1, f"anchor {RECIPE_HEADING!r} not found in SKILL.md"
    end = text.find(RECIPE_END_ANCHOR, start)
    assert end != -1, f"anchor {RECIPE_END_ANCHOR!r} not found after the recipe heading"
    return text[start:end]


def test_skill_step5b_carries_registration_mechanic() -> None:
    """The recipe region documents session-start cwd-scoped registration + the symlink install."""
    region = _norm(_respawn_recipe_region())
    assert "register at SESSION START" in region, (
        "Step 5b respawn recipe lost the session-start agent-registration mechanic (#2072)"
    )
    assert "MID-session NEVER registers" in region, (
        "Step 5b respawn recipe lost the mid-session-additions-never-register clause (#2072)"
    )
    assert "user-global as SYMLINKS" in region, (
        "Step 5b respawn recipe lost the user-global symlink-install pointer (#2072)"
    )
    assert "ln -sfn" in region, (
        "Step 5b respawn recipe lost the symlink re-install one-liner (#2072)"
    )


def test_skill_step5b_names_new_agent_type_residual() -> None:
    """A session spawned before a NEW agent type lands routes to the fail-loud terminal."""
    region = _norm(_respawn_recipe_region())
    assert "can never resolve it mid-session" in region, (
        "Step 5b respawn recipe lost the new-agent-type mid-session residual clause (#2072)"
    )
    assert "EVERY project's sessions on this machine" in region, (
        "Step 5b respawn recipe lost the cross-project user-global-install residual (#2072)"
    )


def test_context_hygiene_bullet_carries_mechanic() -> None:
    """The context-hygiene.md autocompact-thrash bullet carries the compressed mechanic."""
    assert CONTEXT_HYGIENE_MD.exists(), f"missing {CONTEXT_HYGIENE_MD}"
    text = _norm(CONTEXT_HYGIENE_MD.read_text(encoding="utf-8"))
    assert "ALL lean twins resolve user-global" in text, (
        "context-hygiene.md lost the ALL-lean-twins-resolve-user-global clause (#2072)"
    )
    assert "symlinks to the repo `.claude/agents/` files" in text, (
        "context-hygiene.md lost the symlink-to-repo-files clause (#2072)"
    )
    assert "register at session start" in text, (
        "context-hygiene.md lost the session-start registration mechanic (#2072)"
    )


def test_all_registered_repo_lean_twin_files_exist() -> None:
    """Every symlink target — the repo lean-twin agent files in LEAN_TWINS — exists in git."""
    missing = [name for name in LEAN_TWINS if not (AGENTS_DIR / name).is_file()]
    assert not missing, f"missing repo lean-twin agent files: {missing}"


def test_context_hygiene_roster_names_codex_composer_twins() -> None:
    """The context-hygiene.md Class-2 roster names all five codex composer twins (#2472)."""
    text = _norm(CONTEXT_HYGIENE_MD.read_text(encoding="utf-8"))
    missing = [n[:-3] for n in CODEX_COMPOSER_TWINS if n[:-3] not in text]
    assert not missing, f"context-hygiene.md roster lost codex composer twins: {missing} (#2472)"


def test_skill_step5b_roster_names_codex_composer_roles() -> None:
    """The Step 5b recipe region names the composer roles + the before-no-show routing (#2472)."""
    region = _norm(_respawn_recipe_region())
    for role in (
        "codex-code-reviewer",
        "codex-critic",
        "codex-interpretation-critic",
        "codex-clean-result-critic",
        "codex-follow-up-critic",
    ):
        assert role in region, f"Step 5b recipe lost composer role {role} (#2472)"
    assert "BEFORE item 4's Step 5d single-Claude no-show fallback" in region, (
        "Step 5b recipe lost the composer-before-no-show routing clause (#2472)"
    )


def test_codex_ensemble_review_carries_composer_thrash_rung() -> None:
    """codex-ensemble-review.md routes composer thrash to the lean rung before no-show (#2472)."""
    path = REPO_ROOT / ".claude" / "rules" / "codex-ensemble-review.md"
    text = _norm(path.read_text(encoding="utf-8"))
    assert "Composer thrash-death is NOT an instant no-show" in text, (
        "codex-ensemble-review.md lost the composer-thrash-before-no-show paragraph (#2472)"
    )
    assert "codex-<role>-lean" in text, (
        "codex-ensemble-review.md lost the lean-twin rung pointer (#2472)"
    )

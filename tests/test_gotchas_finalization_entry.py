"""Pin tests for #1693 — gotchas.md finalization-crash entry.

Guards against silent removal of the PyGILState_Release atexit-race entry and
against a future accidental narrowing of the `paths:` frontmatter globs that
would orphan the entry from its on-demand load trigger.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GOTCHAS_MD = REPO_ROOT / ".claude" / "rules" / "gotchas.md"


def test_finalization_entry_prose_pin() -> None:
    """gotchas.md carries the finalization-crash entry + explicit-exit remedy."""
    text = GOTCHAS_MD.read_text(encoding="utf-8")

    # Distinctive substring — the crash signature the entry documents.
    assert "PyGILState_Release" in text, (
        "PyGILState_Release finalization-crash entry missing from gotchas.md"
    )

    # The remedy — explicit exit before finalize-time C-extension teardown.
    assert "sys.exit(0)" in text, "explicit sys.exit(0) remedy missing from finalization entry"

    # The class name so future readers can grep for it.
    assert "phased-dispatcher" in text, (
        "phased-dispatcher entrypoint framing missing from finalization entry"
    )

    # Incident cross-reference — the entry cites #1689 as its motivator.
    assert "#1689" in text, "#1689 incident cross-reference missing"


def test_paths_frontmatter_covers_dispatcher_entrypoints() -> None:
    """gotchas.md `paths:` frontmatter still globs dispatcher scripts.

    The finalization entry is on-demand-loaded via the file's `paths:`
    frontmatter — a future edit that drops the dispatcher globs would orphan
    this entry from every relevant trigger. We do NOT ratchet the exact glob
    set; we assert the entry's on-demand trigger surface stays covered.
    """
    text = GOTCHAS_MD.read_text(encoding="utf-8")

    # The frontmatter block sits at the top of the file, delimited by --- ... ---.
    assert text.startswith("---"), "gotchas.md missing YAML frontmatter"
    end = text.index("---", 3)
    frontmatter = text[3:end]

    assert "paths:" in frontmatter, "gotchas.md frontmatter missing paths: on-demand-load key"

    # The finalization entry is a dispatcher-orchestration trap; the
    # dispatcher glob under scripts/ must remain in the on-demand trigger set.
    # We accept either the plain `*dispatch*` stem or an explicit .py/.sh form.
    dispatcher_glob_present = any(
        needle in frontmatter
        for needle in (
            "scripts/*dispatch*",
            "scripts/*dispatch*.py",
            "scripts/*dispatch*.sh",
        )
    )
    assert dispatcher_glob_present, (
        "gotchas.md `paths:` frontmatter no longer globs "
        "scripts/*dispatch*{.py,.sh} — the finalization-crash entry loses "
        "its on-demand load trigger and orphans #1693"
    )

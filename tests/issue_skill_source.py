"""Composed read of the `/issue` orchestrator spec.

`.claude/skills/issue/SKILL.md` is split into per-step companion files under
`.claude/skills/issue/steps/` so a `/issue` session does not spend its whole
context window on the spec at boot (#2155). The durability pins must keep
binding on the LOGICAL document, not on whichever physical file a given step
currently lives in — otherwise a future split silently drops enforcement
instead of failing loud.

So every test that greps the orchestrator spec reads it through here.

Two properties this guarantees:

* **Superset, never subset.** The composed text is SKILL.md plus every step
  companion, so a `X in text` pin binds wherever X lives, and a
  `X not in text` pin stays exactly as strict as before (content moved
  between the files is still present in the composition).
* **Split-invariant.** With `steps/` absent or empty the composition is
  byte-identical to SKILL.md, so re-pointing a test is a no-op until the
  split actually lands — which is what makes the refactor verifiable one
  step at a time.

Composition is IN DOCUMENT ORDER: each step body is spliced back in at its
pointer site, reconstructing the pre-split document. Appending companions at
the end instead would break every region-scoped pin — a test that extracts
"the action region" and counts EXIT sites inside it saw 1 instead of 15,
because the content had moved past the region's closing boundary. Splicing
keeps ordering, adjacency and region extraction all meaningful.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ISSUE_SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
ISSUE_STEPS_DIR = REPO_ROOT / ".claude" / "skills" / "issue" / "steps"

#: The pointer SKILL.md carries in place of a relocated step body.
_POINTER = re.compile(r"^>\s+\*\*Full procedure:\*\*\s+`\.claude/skills/issue/steps/(\S+?)`")
#: Companion preamble ends at the first horizontal rule; the body follows.
_BODY_SPLIT = "\n---\n\n"


def issue_skill_parts() -> list[Path]:
    """SKILL.md followed by every step companion, in stable sorted order."""
    parts = [ISSUE_SKILL_MD]
    if ISSUE_STEPS_DIR.is_dir():
        parts.extend(sorted(p for p in ISSUE_STEPS_DIR.glob("*.md") if p.is_file()))
    return parts


def _companion_body(name: str) -> str:
    """A step companion's relocated body, minus its own title/preamble."""
    text = (ISSUE_STEPS_DIR / name).read_text(encoding="utf-8")
    _head, sep, body = text.partition(_BODY_SPLIT)
    return body if sep else text


def read_workflow_doc(path: Path) -> str:
    """Read any workflow doc; the `/issue` spec composes from its step files.

    Use this instead of ``path.read_text()`` in helpers that take the doc as a
    PARAMETER (``def _norm(path): return path.read_text(...)``) — a
    name-binding rewrite cannot see through a parameter, so those helpers kept
    reading raw SKILL.md and their pins went looking for prose that had moved
    into a companion. No-op for every other path.
    """
    if path.name == "SKILL.md" and path.parent.name == "issue":
        return issue_skill_text()
    return path.read_text(encoding="utf-8")


def issue_skill_text() -> str:
    """The `/issue` spec as ONE logical document, in original document order.

    Each `> **Full procedure:** ...` pointer block in SKILL.md is replaced by
    the body it points at, so the result is the pre-split document.
    """
    out: list[str] = []
    lines = ISSUE_SKILL_MD.read_text(encoding="utf-8").split("\n")
    i = 0
    while i < len(lines):
        m = _POINTER.match(lines[i])
        if m is None:
            out.append(lines[i])
            i += 1
            continue
        # Skip the whole blockquote pointer, then splice the body in its place.
        while i < len(lines) and lines[i].startswith(">"):
            i += 1
        out.append(_companion_body(m.group(1)).rstrip("\n"))
    return "\n".join(out)

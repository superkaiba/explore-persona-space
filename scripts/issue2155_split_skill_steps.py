"""Split the 20 ``### Step`` bodies out of ``.claude/skills/issue/SKILL.md`` (#2155).

Mechanical, re-runnable splitter. Each ``### Step`` body moves VERBATIM to
``.claude/skills/issue/steps/NN-<slug>.md``; SKILL.md keeps the state machine,
every step heading, and one 3-line ``> **Full procedure:**`` pointer per step
(the exact format ``tests/issue_skill_source.py`` parses). The script is
committed (not a one-off) because the Phase-E freshness gate of the #2155 plan
may force a re-execution against a newer origin/main tip — the split must be
reproducible from the tree alone.

Two exact self-checks run before the script exits 0:

1. **Split identity.** Immediately after the move (before any router-only
   addition), the composition ``tests/issue_skill_source.issue_skill_text()``
   must equal the pre-split document BYTE-IDENTICALLY. The construction makes
   this exact by design: each moved region spans first-non-blank to
   last-non-blank line of a step body, so every original blank run stays in
   the router and the composer's ``rstrip("\\n")`` strips nothing that existed.
2. **Carve-out identity.** After injecting the ``## Companion files`` steps/
   carve-out paragraph (plan Phase B.7 — a deliberate router-only addition,
   ported from commit 0e4a8987bb), the composition must equal the pre-split
   document with exactly that paragraph inserted at the same spot.

On any check failure the original SKILL.md is restored and the script exits
non-zero (kill criterion §5.1 territory — never land a lossy split).

Idempotent: a tree that already carries pointer lines exits 0 untouched.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
STEPS_DIR = REPO_ROOT / ".claude" / "skills" / "issue" / "steps"

EXPECTED_STEPS = 20  # plan #2155 §4.1 pins exactly 20; a 21st step is a must-ask deviation

_STEP_HEADING_RE = re.compile(r"^### Step")
_H2_RE = re.compile(r"^## ")
_POINTER_RE = re.compile(r"^>\s+\*\*Full procedure:\*\*\s+`\.claude/skills/issue/steps/\S+?`")
_SLUG_RE = re.compile(r"^### Step ([\w.-]+):")

_POINTER_TEMPLATE = (
    "> **Full procedure:** `.claude/skills/issue/steps/{name}` — read that\n"
    "> file when the run reaches this step. Routing, gates and the state\n"
    "> machine stay in SKILL.md; only the step body moved (#2155)."
)

_PREAMBLE_TEMPLATE = (
    "# {title}\n"
    "\n"
    "Step body relocated verbatim from `.claude/skills/issue/SKILL.md`\n"
    "(#2155). SKILL.md keeps the heading, the state machine and the\n"
    "Orchestration Procedure router; read this file when the run reaches\n"
    "this step.\n"
    "\n"
    "---\n"
    "\n"
)

# ── Phase B.7: the `## Companion files` steps/ carve-out (ported from 0e4a8987bb,
# token figure adjusted to today's document size: 989,603 B × 0.3847 tok/B ≈ 381K).
_CARVEOUT_ANCHOR = "Read these on first invocation of the skill in a session."
_CARVEOUT_SENTINEL = "- `steps/` — the per-step procedure bodies"
_CARVEOUT_LINES = [
    "",
    "- `steps/` — the per-step procedure bodies, one file per `### Step`",
    "  (#2155). **These are the ONE exception to the line above: do NOT read",
    "  them on first invocation.** Read a step's file when the run REACHES that",
    "  step, and only that file. Reading them all at boot would restore the",
    "  ~381K-token load this split exists to remove — and would be the exact",
    "  regression the split is guarding against. Each `### Step` heading below",
    "  carries a `> **Full procedure:**` pointer naming its file; the state",
    "  machine, the Orchestration Procedure and every gate stay in SKILL.md, so",
    "  routing never needs a companion.",
]


def _slug(heading: str) -> str:
    m = _SLUG_RE.match(heading)
    if m:
        return f"step-{m.group(1).lower()}"
    if heading.startswith("### Step-completed"):
        return "step-completed-reentry"
    raise SystemExit(f"FATAL: cannot derive a slug for heading: {heading!r}")


def _heading_indices(lines: list[str]) -> list[int]:
    """Indices of ``### Step`` headings, tracked outside ``` code fences."""
    out: list[int] = []
    in_fence = False
    for i, ln in enumerate(lines):
        if ln.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and _STEP_HEADING_RE.match(ln):
            out.append(i)
    return out


def _boundary(lines: list[str], start: int) -> int:
    """First heading line (### Step or ## H2, outside fences) after ``start``."""
    in_fence = False
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and (_STEP_HEADING_RE.match(ln) or _H2_RE.match(ln)):
            return i
    return len(lines)


def _inject_carveout(lines: list[str]) -> list[str]:
    anchors = [i for i, ln in enumerate(lines) if ln == _CARVEOUT_ANCHOR]
    if len(anchors) != 1:
        raise SystemExit(
            f"FATAL: expected exactly 1 carve-out anchor line, found {len(anchors)} "
            f"({_CARVEOUT_ANCHOR!r})"
        )
    at = anchors[0] + 1
    return lines[:at] + _CARVEOUT_LINES + lines[at:]


def main() -> int:
    original = SKILL.read_text(encoding="utf-8")
    lines = original.split("\n")

    if any(_POINTER_RE.match(ln) for ln in lines):
        n = sum(1 for ln in lines if _POINTER_RE.match(ln))
        m = len(list(STEPS_DIR.glob("*.md"))) if STEPS_DIR.is_dir() else 0
        print(f"already split ({n} pointers, {m} companions); nothing to do")
        return 0

    headings = _heading_indices(lines)
    if len(headings) != EXPECTED_STEPS:
        raise SystemExit(
            f"FATAL: expected exactly {EXPECTED_STEPS} '### Step' headings, "
            f"found {len(headings)} — re-derive the plan before splitting."
        )

    # Compute (start, end, companion name, heading) per step; body spans
    # first-non-blank..last-non-blank so every original blank run stays in the router.
    regions: list[tuple[int, int, str, str]] = []
    for k, h in enumerate(headings):
        bound = _boundary(lines, h)
        start = h + 1
        while start < bound and not lines[start].strip():
            start += 1
        end = bound - 1
        while end > h and not lines[end].strip():
            end -= 1
        if start > end:
            raise SystemExit(f"FATAL: empty body for heading at line {h + 1}: {lines[h]!r}")
        regions.append((start, end, f"{k:02d}-{_slug(lines[h])}.md", lines[h]))

    STEPS_DIR.mkdir(parents=True, exist_ok=True)
    leftover = [p.name for p in STEPS_DIR.glob("*.md")]
    if leftover:
        raise SystemExit(f"FATAL: steps/ already holds files on an unsplit tree: {leftover}")

    router = list(lines)
    for start, end, name, heading in reversed(regions):
        body = "\n".join(lines[start : end + 1])
        companion = _PREAMBLE_TEMPLATE.format(title=heading[4:]) + body + "\n"
        (STEPS_DIR / name).write_text(companion, encoding="utf-8")
        router[start : end + 1] = _POINTER_TEMPLATE.format(name=name).split("\n")
    SKILL.write_text("\n".join(router), encoding="utf-8")

    sys.path.insert(0, str(REPO_ROOT / "tests"))
    # PROD_IMPORT_LINT_EXEMPT: tests/ helper imported post-sys.path-split; not a lockfile dep
    import issue_skill_source  # noqa: E402  (repo tests/ helper, imported post-split)

    composed = issue_skill_source.issue_skill_text()
    if composed != original:
        SKILL.write_text(original, encoding="utf-8")
        a, b = original.split("\n"), composed.split("\n")
        print(f"FATAL: split composition != original ({len(a)} vs {len(b)} lines); restored.")
        return 1
    print(f"split identity OK: composition byte-identical to the {len(original)}-byte original")

    # Phase B.7 — router-only carve-out; verify the composition delta is exactly it.
    with_carveout = _inject_carveout(SKILL.read_text(encoding="utf-8").split("\n"))
    SKILL.write_text("\n".join(with_carveout), encoding="utf-8")
    expected = "\n".join(_inject_carveout(lines))
    composed2 = issue_skill_source.issue_skill_text()
    if composed2 != expected:
        SKILL.write_text(original, encoding="utf-8")
        print("FATAL: post-carve-out composition != original+carve-out; restored.")
        return 1
    print("carve-out identity OK: composition == original + Companion-files carve-out")

    router_bytes = SKILL.stat().st_size
    companions = sorted(STEPS_DIR.glob("*.md"))
    print(f"router: {router_bytes} bytes; companions: {len(companions)}")
    for p in companions:
        print(f"  {p.name}: {p.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

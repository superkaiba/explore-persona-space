---
name: extract-and-execute-recipe-tests
description: When a test EXTRACTS a live shell snippet from a doc (SKILL.md recipe) by substring match and executes it, verify the match target is unique file-wide and falsify pre-fix via git-show mimic dir
metadata:
  type: feedback
---

For extract-and-execute recipe tests (a pytest that pulls a live bash
assertion out of a SKILL.md recipe by substring match and runs it under
`subprocess`), three review checks beyond running the test:

1. **Extraction-target uniqueness.** Grep the WHOLE doc for the match
   substring (e.g. `'"$S9C_FILES" | wc -l'`). The extractor returns the
   FIRST match — a comment/prose line matching earlier than the live
   assertion would make the test execute the wrong text and silently pass.
   (#2317: target was unique at exactly one line, verified.)
2. **Pre-fix falsification via git-show mimic dir**, never stash: build
   `/tmp/<d>/tests/` + `/tmp/<d>/.claude/skills/issue/` with
   `git show origin/main:<doc>` and the branch's test file, run the
   worktree venv's `python -m pytest` from there (tests resolve the doc
   relative to `__file__`). Both #2317 tests failed pre-fix at the
   extraction step — proof of extract-and-execute, not a hardcoded copy.
   (`uv run --no-project --with pytest` fails to spawn in sandbox; use the
   worktree `.venv/bin/python -m pytest` directly. See also
   [[feedback-prefix-demo-git-show-not-stash]].)
3. **Execute the shipped snippet directly** (`sed -n '<L1>,<L2>p' doc | bash -c ...`)
   on both the good and bad shapes — the guard must score rc=0 silent on
   good, nonzero + FATAL on bad, in the shell the launcher actually uses.

**Why:** a guard that cannot fail (the #2314 `case`-glob degenerate) and a
test that measures itself instead of the recipe are the two failure modes
these doc-recipe hardening rounds exist to close; the checks above are the
cheapest discriminators.

**How to apply:** any diff adding/editing tests that pin or execute
SKILL.md / agent-doc shell recipe blocks (the #2126/#2296/#2314/#2317
gate-recipe hardening family).

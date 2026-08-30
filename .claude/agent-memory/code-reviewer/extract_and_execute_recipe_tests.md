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
4. **Three-variant falsification + which-assertion-reds** (#2385 r2). When
   the test ships WITH the fix, import the test's OWN helpers from a /tmp
   harness and vary only the production text: `control` (worktree),
   `prefix` (`git show <round-parent-sha>:<doc>`), and `noguard` (the
   worktree text with ONLY the guard hunk reverted by regex). The
   `noguard` variant is what isolates causation — `prefix` alone can red
   for unrelated round-1 reasons. Then read WHICH assertions red: a test
   that reds only on the guard's own echo / exit code is tautological; a
   real gate reds on the HAZARD assertions (payload survives on disk, no
   `D` row committed). #2385's pre-fix run committed
   `tests/test_workflow_lint_payload.py | 1 -` — the deletion reproduced
   end to end, so the gate was genuine.

**Why:** a guard that cannot fail (the #2314 `case`-glob degenerate) and a
test that measures itself instead of the recipe are the two failure modes
these doc-recipe hardening rounds exist to close; the checks above are the
cheapest discriminators. Check 4 exists because an implementer may
explicitly hand the falsification to review (#2385 r2 did) — an unfalsified
"it fails pre-fix" claim is exactly the hollow-gate shape.

**How to apply:** any diff adding/editing tests that pin or execute
SKILL.md / agent-doc shell recipe blocks (the #2126/#2296/#2314/#2317
gate-recipe hardening family).

---
name: claude-misses-fix-regressions
description: When a "fix" REPLACES (not adds to) a prior check, Claude verifies only that the new check addresses the surface complaint; Codex catches that the new check is WEAKER on the original Must-Fix class. Replicate the original bad input through the NEW check.
metadata:
  type: feedback
---

**Rule:** when reconciling a round-N "fix" verdict: (1) identify the round-(N-1) blocker the fix targets; (2) construct the exact bad-input class the OLD check caught; (3) run that input through the NEW check; (4) if it now passes silently, FAIL — "fix this, break that" violates fail-loud discipline. Right resolution: take the MAX over both surfaces, or keep the old check AND add the new one. "Replace one with the other" is the smell.

**Origin (#389 r2):** blocker "Jaccard filter misses answer-side leakage" → fix replaced `jaccard(user_q, probe)` with `jaccard(user_q + assistant_a, probe)`; verbatim-question leakage that scored 1.000 (loud) now scores 0.538 < 0.6 threshold (silent pass) — the original Must-Fix class eroded.

**Plan-level instance (#554 r1 alt):** plan demoted the false-positive behind-origin/main ERROR to WARNING and replaced it with a behind-own-ref ERROR whose freshness depends on a `git fetch` whose rc the plan ignores — fetch failure → `behind_own=0` → silent PASS of a stale resumed pod, the exact class the old (spurious) ERROR accidentally blocked. "Pre-existing ignore-rc" does NOT save a plan whose change converts a benign ignore into a hole in its own acceptance criterion. Also check whether the mocked test seam can even EXPRESS the failure (fake `_run` hardcoding fetch rc 0 pins the hollow guarantee green). REVISE.

Related: [[feedback_claude_underclasses_silent_failures]]; [[feedback_claude_misses_same_file_siblings]].

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Claude misses fix regressions](feedback_claude_misses_fix_regressions.md) — a fix that REPLACES a check can be weaker on the original Must-Fix class; replicate the old bad input through the NEW check. #389 r2, #554 r1.

---
name: codex-conflates-marker-format-with-code
description: Codex twin code-reviewer escalates implementer marker-body formatting (subsection style nits) OR stale-file reads of marker existence to FAIL verdicts that decline to review the substantive diff. Discard such findings as out-of-rubric for code-reviewer.
metadata:
  type: feedback
---

When the Codex twin code-reviewer's Step 0.5 mechanical-contract check trips, it sometimes refuses to perform the substantive diff review at all — leaving Major and Minor sections explicitly empty ("None; diff not reviewed after the mechanical contract failure"). Two distinct trigger flavors:

**Flavor A — marker-shape nit (over-rigid rubric application):** On clean round-2+ diffs, Codex invents a single new "blocker" about the IMPLEMENTER'S MARKER REPORT FORMAT — e.g.:
- "Subsection `### (c) How to verify` uses inline backticks rather than a fenced ```bash``` block"
- "Fenced launch command is outside the (c) subsection"
- "Status header missing required field X"

**Flavor B — stale-file false alarm (worktree vs canonical drift):** Codex reads the worktree's local `tasks/<status>/<N>/events.jsonl` instead of canonical main-branch state, sees the implementation marker missing (because it hasn't been pulled into the worktree yet), and FAILs on "missing epm:experiment-implementation marker" — then declines to review the diff. The marker actually exists in canonical state. Verify with:

```bash
jq -r 'select(.kind == "epm:experiment-implementation") | "\(.ts) v\(.version) note_len=\(.note | length)"' \
  /path/to/main/tasks/<status>/<N>/events.jsonl | tail
```

Both are real-but-non-blocking observations OR false alarms, NOT code review findings.

**Why:** The `code-reviewer` rubric (`.claude/agents/code-reviewer.md`) covers the diff against base — code correctness, plan adherence, tests, lint, security. Marker-shape conformance AND marker existence are Step-0.5 structural checks the orchestrator owns separately. Escalating either to FAIL forces an unnecessary revision round on a clean code diff. Worse, in Flavor B, Codex contributes ZERO substantive signal because it bailed before reading the diff — so the reconciler must lean entirely on Claude's substantive review and verify it independently.

**How to apply:** When reconciling code-reviewer disagreements:
1. If Codex's FAIL is exclusively driven by a Step 0.5 mechanical-contract trigger (marker shape OR marker existence), AND its Major/Minor sections are empty ("diff not reviewed"), AND
2. Either (a) both reviewers previously agreed all code blockers are fixed (Flavor A), or (b) the marker actually exists in canonical state when checked independently (Flavor B)
3. Classify the Codex finding as Unverified/false-alarm (Discarded weight). Lean on Claude's substantive review, independently verify 2-3 load-bearing spot-checks against the actual diff, then issue PASS if those hold. Surface any real marker-format observation as a Standing Recommendation.

This is the inverse of [[feedback_claude_underclasses_silent_failures]]: Claude under-classes real runtime bugs as CONCERNS; Codex over-classes prose nits / stale-file reads as FAIL. The pattern is the same — each side's calibration is biased and the reconciler corrects to the rubric.

Observed:
- task #375 round 2 (2026-05-21) — Flavor A: 11/11 code blockers fixed, Codex FAIL'd on subsection (c) using inline backticks instead of a fenced block. Adjudicated PASS.
- task #382 round 1 (2026-05-26) — Flavor B: Codex FAIL'd on "missing epm:experiment-implementation marker at 2026-05-26T06:07:38Z" by reading stale worktree events.jsonl. Marker existed in canonical state (note_len=12370). Major/Minor both "None; diff not reviewed after the mechanical contract failure." Claude's substantive 8-point review held up under independent re-verification of the in-loop backward OOM fix. Adjudicated PASS.
- task #391 round 3 (2026-05-26) — Flavor A: clean +54/-22 asyncio event-loop fix (Option C: client built inside coroutine, single `asyncio.run`, `gather` pos+neg) with AST-level regression test. Codex FAIL'd because the implementer renamed the round-3 report H3s from `(a)/(b)/(c)/(d)` to ad-hoc round-summary headings (`### Fix chosen / ### Commits / ### Verification / ### Cache/resume invariant preserved / ### Needs human eyeball`) — note that the SAME implementer used the canonical shape correctly in rounds 1 and 2. Codex's own Plan Adherence block said both plan items ✓ implemented; Major/Minor explicitly empty. Adjudicated PASS with Standing Recommendation to re-post the v3 marker in canonical shape.
- task #401 round 2 (2026-05-26) — Flavor A: clean round-2 marker-abstraction infra fix (6/6 round-1 blockers addressed, 14/14 tests PASS, +696/-168 diff). Implementer used round-summary H3s (`### Commits / ### Per-blocker resolution / ### Infra port / ### Verification / ### Final diff stats / ### Needs human eyeball`) instead of canonical `(a)/(b)/(c)/(d)`. Semantic content of the four sections WAS present (Verification had copy-pasteable pytest + 14/14 PASS signal; Per-blocker resolution had per-file what-changed; Needs human eyeball exact match) — only labels differed. Claude's per-blocker substantive review held with line-level refs (`eval/marker_logprob.py:282-373` try/finally, `_FakeTokenizer` stub determinism, hardlink-OSError fallback inode check). Adjudicated PASS with Standing Recommendation.

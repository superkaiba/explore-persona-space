---
title: 'workflow-fix: lint leg resolving line-anchored gotchas.md L<n> citations (trim
  renumbering silently rots them)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-22T14:07:06Z'
has_clean_result: false
origin_prompt: 'Surfaced by the Claude code-reviewer during /issue 2280 round 3: four
  gotchas.md L240 line-anchored citations were already ~6 lines stale at the merge-base;
  the existing stale-gotchas-pointers test pins substrings only, so line-anchor drift
  from every periodic trim goes uncaught.'
workflow: v1
---
## Goal

Add a `workflow_lint.py` leg that resolves LINE-ANCHORED `gotchas.md L<n>` citations in
the workflow surface against the line's expected content, so line-anchor drift is caught
mechanically instead of silently rotting.

## Why

`.claude/rules/gotchas.md` is machine-appended (`consolidate_lessons.py`) and periodically
hand-trimmed to stay under the 200,000-byte `GOTCHAS_SIZE_WARN_BYTES` budget. **Every
trim renumbers the file**, so any citation elsewhere in the workflow surface that points at
`gotchas.md L<n>` drifts. The existing guard,
`tests/test_workflow_lint_stale_gotchas_pointers.py`, pins SUBSTRINGS only — it cannot see
a line-number anchor that now points at the wrong entry.

Surfaced by the round-3 `code-reviewer` on task #2280 (workflow-surface prose follow-up,
Rule 12: mechanizable + recurring). The reviewer reported four line-anchored citations at
`gotchas.md L240` — `.claude/agents/code-reviewer.md:606`,
`.claude/rules/code-reviewer-section-reference.md:1233` and `:1245`, and
`.claude/agents/codex-code-reviewer.md:345` — which were **already ~6 lines stale at
#2280's merge-base**, i.e. pre-existing on trunk and NOT a #2280 regression. #2280's own
trim (201,850 -> 197,869 B) shifts them further.

Verification note for the implementer: the orchestrator's own attempt to independently
re-locate these anchors did not complete (a recursive regex grep over `.claude` timed out
on the shared VM under fleet load), so **the four sites above are REVIEWER-REPORTED and
must be re-confirmed as step one** — including the exact anchor spelling, since a plain
`grep -rn "gotchas.md L[0-9]"` over `.claude/**` returned no hits for that literal form.
Establish the real citation grammar in use before writing a matcher for it.

## Scope

1. Confirm the anchor sites and the actual textual form(s) line-anchored gotchas citations
   take across `.claude/agents/`, `.claude/rules/`, `.claude/skills/`, and `CLAUDE.md`.
2. Add a lint leg that, for each such citation, reads the cited line and checks it against
   an expected-content token carried in the citation (or flags the citation as
   unresolvable). Bundle it into the no-flags default run only if it is fast and
   deterministic; otherwise gate it behind its own flag and wire it where the other
   gotchas legs run.
3. Repair the four (re-confirmed) stale anchors.
4. Consider the stronger structural fix and record the decision either way: replace
   line anchors with a stable per-entry token (a slug or the entry's `#N` citation) so a
   trim cannot invalidate them at all. A matcher that only detects drift still needs a
   human to fix each drift; anchors that cannot drift end the class.

## Acceptance criteria

- A deliberately drifted line anchor FAILs the new leg (fixture-backed, both directions:
  correct anchor passes, drifted anchor fails).
- The four re-confirmed live anchors resolve correctly after repair.
- `uv run python scripts/workflow_lint.py` (no flags) rc=0.
- `tests/test_workflow_lint_stale_gotchas_pointers.py` still green (this leg complements
  the substring pins; it does not replace them).

## Provenance

Surfaced by the Claude `code-reviewer` during task #2280 round 3 (the gotchas archaeology
trim that unblocked #2280's Step 10d size gate). Filed rather than fixed in #2280 because
repairing four additional always-on spec files at merge time would have been an unreviewed
scope expansion on a task already gate-blocked — and because the anchors were stale on
trunk before #2280 existed.

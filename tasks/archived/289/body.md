---
title: '[Doc] Clarify path-vs-body convention in /issue Step 4 and Step 6 briefs'
kind: infra
tags: []
created_at: '2026-05-06T08:13:14.000Z'
has_clean_result: false
sagan_id: 7339060d-ea7c-47b9-b1f8-5ad6be6dba31
sagan_number: 289
priority: normal
legacy_why_unset: true
---
**Parent:** #275 (audit item 3)

## Background

Per the audit posted on #275 (item 3), the dispatch briefs in
`.claude/skills/issue/SKILL.md` Steps 4 (implementer) and 6
(experimenter) say "The plan" without specifying body-vs-path. In
practice the orchestrator passes the path
`.claude/plans/issue-<N>.md`; the subagent is expected to `Read` the
file before acting.

This is the right default (1400-line plans inlined into a prompt waste
context, can hit token limits, and stale across worktree edits). But
the convention is not documented, so an adversarially-loaded subagent
might guess at plan contents instead of reading the file.

## Proposed fix

Add a 1-line clarification to both briefs:

> **Plan handoff convention:** the brief includes the PATH to the
> cached plan (`.claude/plans/issue-<N>.md`), NOT the body. Read the
> file before acting; do NOT infer plan content from the issue body
> or comment markers.

That's it — pure doc PR, no behaviour change.

## Acceptance criteria

- `.claude/skills/issue/SKILL.md` Step 4 brief includes the
  path-vs-body clarification.
- Same for Step 6 brief.
- No code change, no behaviour change.

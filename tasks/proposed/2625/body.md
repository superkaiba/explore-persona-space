---
title: 'workflow-fix: verify_plan WARNs when a plan edits a byte-capped workflow surface
  without a post-edit byte count or cap-raise'
kind: infra
tags:
- wf-fix
created_at: '2026-08-27T09:59:07Z'
has_clean_result: false
origin_prompt: 'Critic prose follow-up from #2364 plan review: second incident (#2133,
  #2364 v2) of a plan self-FAILing a byte-capped surface it edits; both caps are machine-readable
  at plan time.'
workflow: v1
---
# workflow-fix: verify_plan WARNs when a plan edits a byte-capped workflow surface without a post-edit byte count or cap-raise

**Provenance:** surfaced by the Claude Methodology critic during the #2364
plan-review floor (2026-08-27), as a prose follow-up after both of its
Must-Fix findings were instances of this class.

## The gap

Two plans have now self-FAILed their own acceptance criteria by proposing
edits to byte-capped workflow surfaces without checking the cap at plan time:

- **#2133**: a +31 B insertion on a 267 B `LESSONS.md` row breached the
  280 B `_LESSONS_ROW_MAX_BYTES` cap (`workflow_lint.py`, strictly-greater,
  grandfather dict closed).
- **#2364 plan v2** (caught in review, fixed in v3): (a) a new
  `code-reviewer.md` section against 159 B of headroom under its 109,600 B
  `check_agent_spec_size` grandfather cap
  (`.claude/config/agent_spec_size_caps.txt`); (b) a +64 B LESSONS row
  insertion onto a 221 B row → 285 B > 280 B.

Both caps are machine-readable at plan time, so the class is checkable
mechanically instead of relying on a critic to re-measure by hand.

## Proposed check (implementing session refines)

A `verify_plan.py` WARN-only check: when a plan's edit list / concrete-diffs
section names (a) a `.claude/rules/LESSONS.md` row edit, or (b) an edit to
any file listed in `.claude/config/agent_spec_size_caps.txt`, WARN unless the
plan states a post-edit byte count, a net insertion size against the measured
headroom, or a same-commit cap raise. Mirror the existing plan-vs-repo
cross-check conventions (c65/c66 shape: resolved contradictions FAIL or WARN,
unresolvable routes disclosed).

## Acceptance

- A plan naming a LESSONS row edit with no byte accounting → WARN naming the
  row and the measured headroom.
- A plan naming an `agent_spec_size_caps.txt`-listed file edit with no
  post-edit byte statement and no cap-raise edit in its file list → WARN.
- A plan that states the byte accounting (either form) → no WARN.
- WARN-only; false negatives (prose that names the edit obliquely) disclosed
  in the check's docstring.

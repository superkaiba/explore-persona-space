---
title: 'workflow-fix: tighten workflow_lint.py skill-ref regex against /word-in-prose
  false positives'
kind: infra
tags:
- wf-fix
- wf-fix-fp:258affc9a4fb
created_at: '2026-07-01T20:40:43Z'
has_clean_result: false
origin_prompt: 'workflow_lint.py (no-flags default run) FALSE-POSITIVES an "unresolved
  skill reference /unset" on .claude/rules/upload-policy.md:171. Fix: tighten SKILL_REF_RE
  to require a word-boundary/whitespace before the leading /.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #812 (emitting agent: `experiment-implementer`).

## Goal

Tighten `scripts/workflow_lint.py`'s skill-reference detector so that ordinary slashed prose tokens (`/unset`, `` `false`/unset ``) inside rule docs do not trip the "unresolved skill reference" false-positive.

## Workflow gap

- **Bug observed:** `workflow_lint.py` (no-flags default run) FALSE-POSITIVES an "unresolved skill reference `/unset`" on `.claude/rules/upload-policy.md:171`, where the literal prose `` `false`/unset `` is mis-parsed as a `/unset` skill reference.
- **Why it is a workflow gap:** The skill-reference detector treats any `/word` token in prose as a candidate skill invocation, so an ordinary slashed word in a rule doc (`/unset`, `` `false`/unset ``) trips the check even though it is not a skill reference; the no-flags default run FAILs on `main` unrelated to any diff. Any orchestrator / implementer / reviewer running the default lint will see this FAIL.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
# in the skill-ref scanner: only treat `/name` as a skill ref when preceded by
# start-of-line or whitespace, not when it follows a word char / backtick:
- SKILL_REF_RE = re.compile(r"/([a-z][a-z0-9-]+)")
+ SKILL_REF_RE = re.compile(r"(?:(?<=\s)|(?<=^))/([a-z][a-z0-9-]+)")
# OR treat `\`word\`/word` (slash between two backtick-adjacent words) as prose.
```

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Also grep the workflow surface for other `/word`-in-prose patterns the tightened regex could still miss (`grep -rE '\`[a-z]+\`/[a-z]+' .claude/rules/ CLAUDE.md`).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The tightened regex must not regress the intended skill-reference detection: an actual `/humanize`, `/issue`, `/plan` reference at the start of a line or after whitespace must still trip.
- If `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: 258affc9a4fb

<!-- workflow-fix-candidate v1 -->
target_file: scripts/workflow_lint.py
bug_observed: `workflow_lint.py` (no-flags default run) FALSE-POSITIVES an "unresolved skill reference `/unset`" on `.claude/rules/upload-policy.md:171`, where the literal prose `` `false`/unset `` is mis-parsed as a `/unset` skill reference.
why_workflow_gap: The skill-reference detector treats any `/word` token in prose as a candidate skill invocation, so an ordinary slashed word in a rule doc (`/unset`, `` `false`/unset ``) trips the check even though it is not a skill reference; the no-flags default run FAILs on `main` unrelated to any diff.
proposed_change: Tighten the skill-ref regex to not match a `/word` immediately preceded by a non-space word-char (e.g. `` `false`/unset `` where `/` follows a backtick-closed word), or add `/unset` narrative usages to the historical-ref handling; alternatively require a leading word-boundary/space before the `/`.
diff_sketch: |
  # in the skill-ref scanner: only treat `/name` as a skill ref when preceded by
  # start-of-line or whitespace, not when it follows a word char / backtick:
  - SKILL_REF_RE = re.compile(r"/([a-z][a-z0-9-]+)")
  + SKILL_REF_RE = re.compile(r"(?:(?<=\s)|(?<=^))/([a-z][a-z0-9-]+)")
  # OR treat `\`word\`/word` (slash between two backtick-adjacent words) as prose.
confidence: medium
related_task: #812
<!-- /workflow-fix-candidate -->

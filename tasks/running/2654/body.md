---
title: 'workflow-fix: cited-body gate labels Word:-prefixed content corrections ''frontmatter-only''
  (fm_line over-matches ordinary prose)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-31T01:24:33Z'
has_clean_result: false
parent_id: 2384
origin_prompt: 'Surfaced by the #2384 round-5 code review (verified by execution):
  scripts/check_cited_body_currency.py''s fm_line regex matches any Word:-shaped changed
  line, so a conclusive-window content correction written as ''RESULT: ...'' -> ''CORRECTED:
  ...'' earns the advisory frontmatter-only label. Pre-existing since #2384''s round-1
  commit 92e537779ef; ruled out of scope by the round-4 reconciler and round-5 reviewer,
  deferred to its own task.'
workflow: v1
---
## Goal

Stop `check_cited_body_currency.py` from labelling a genuine content correction `frontmatter-only`. The frontmatter heuristic matches any `Word:`-shaped changed line, so a real correction written as `RESULT: ...` -> `CORRECTED: ...` is classified as a metadata-only edit.

## The defect

In `scripts/check_cited_body_currency.py`:

```python
fm_line = ^[+-](?:---\s*$|[A-Za-z_][A-Za-z0-9_-]*:)
```

Any changed line beginning with a word followed by a colon satisfies it. That shape is common in ordinary body prose — `RESULT:`, `CORRECTED:`, `Note:`, `Takeaway:` — so a conclusive-window diff whose only content changes carry such prefixes earns the advisory `frontmatter-only` label.

Verified by execution during #2384's round-5 review: a `-RESULT:.../+CORRECTED:...` content diff under `conclusive=True` returned `frontmatter-only`, while genuine prose returned `None`.

## Why it matters, and why it is not urgent

It matters because this label feeds an operator decision. `.claude/skills/adversarial-planner/SKILL.md` tells the operator on exit 3 to re-read the changed sections and then either record that the plan text is unaffected, or bounce. A `frontmatter-only` label nudges toward "unaffected". The round-2 reconciler's whole argument for why a mislabel was blocking rested on exactly that: the label drives the disposition.

And the shape is plausibly the real-world one. #2384's founding incident (#2378) was a plan quoting #825's superseded linear NULL after the parent's body was corrected to the opposite sign — and a correction commit that rewrites a `RESULT:`-prefixed line to a `CORRECTED:`-prefixed one is a natural way for that edit to look.

It is NOT urgent because the harm is bounded in a way the `rename-only` case was not: unlike `rename-only`, the `frontmatter-only` label never suppresses the diff. The operator still sees every changed line, so the evidence is present and only the summary label is wrong. The verdict is STALE regardless.

## Provenance

Introduced in #2384's round-1 commit `92e537779ef` and unchanged since, surviving four review rounds and two binding reconcilers. #2384's round-5 diff strictly NARROWS label emission and does not touch this regex, so this is not a round-5 regression. Both the round-5 reviewer and the round-4 reconciler recorded it as out of ruled scope with the fix deferred here, rather than expanding a tightly-scoped round.

## Fix sketch (from the round-5 reviewer)

Anchor the heuristic to known frontmatter keys, or to the leading `---` delimited region, instead of matching any `Word:` shape.

Pin: build a CONCLUSIVE window whose only content change is a `Word:`-shaped body line and assert `label is None` (or a content label), demonstrated failing before the change and passing after. A negative control should confirm a real frontmatter-only edit still earns the label — the point is to stop over-matching, not to retire the label.

## Acceptance

1. A conclusive-window content correction whose changed lines carry `Word:` prefixes is no longer labelled `frontmatter-only`, proven by a committed fixture that fails pre-fix.
2. A genuine frontmatter-only edit still earns the label (do not fix by deletion).
3. The `conclusive`-gating behaviour that #2384 round 5 landed is preserved — an inconclusive window still yields `label is None` on every path.

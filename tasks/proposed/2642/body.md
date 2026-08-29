---
title: 'workflow-fix: codex twin composers can read the Claude sibling''s agent-memory,
  defeating ensemble independence'
kind: infra
tags:
- from-2387
created_at: '2026-08-29T16:27:13Z'
has_clean_result: false
origin_prompt: 'Surfaced by the codex-code-reviewer composer during #2387 round-4
  compose: the worktree''s spec-freshness sync puts both reviewers'' .claude/agent-memory/**
  files on the issue branch, so the Codex twin composer can read the Claude code-reviewer''s
  memory file carrying prior-round conclusions on the same defect class. The composer
  added an ad-hoc read fence to that one prompt; make it standing.'
workflow: v1
---
## Goal

Make the cross-agent agent-memory read fence STANDING in the ensemble
reviewer specs, so a Codex twin composer cannot read its Claude sibling's
`.claude/agent-memory/**` files and silently inherit the sibling's prior-round
conclusions.

## Provenance

Surfaced during #2387 round-4 code review (2026-08-29) by the
`codex-code-reviewer` prompt composer, which discovered the leak while
composing and added an ad-hoc fence to that one prompt. Recorded on #2387 as
`epm:progress v17` item 2.

## The gap

Per-agent memories live at `.claude/agent-memory/<agent>/MEMORY.md` plus topic
files, are `memory: project` scoped, and are CHECKED INTO the repo by design
(CLAUDE.md § "Uncommitted TRACKED state at the shared root" explicitly makes
committing them a per-session duty). An `/issue` worktree's spec-freshness
sync therefore pulls EVERY agent's memory onto the issue branch, where they
sit as ordinary readable files in the tree both ensemble reviewers work in.

An agent loading its OWN memory is the designed behavior and is not at issue.
The leak is CROSS-agent: nothing stops the `codex-*` twin composer from
opening, grepping, or citing the Claude sibling's memory — which on a
multi-round task carries that sibling's conclusions about the exact defect
class under review. Observed concretely on #2387: the Claude `code-reviewer`'s
`feedback_naive_substitution_probe_for_unpinned_defence.md` was present and
readable in the `issue-2387` worktree while the Codex composer was composing
the round-4 prompt on that same defect class.

This defeats the cross-family independence the five doubled review sites
exist to provide. It is the same failure MODE as the #2387 round-3 composer
seeding (composer derives a conclusion, writes it into the prompt, Codex
reports it back, and the "corroboration" is circular) — but arriving through
the filesystem rather than the prompt, so the existing prompt-side
independence instructions do not catch it.

## Scope

The composer specs are the primary target, since the composer is the agent
with filesystem access that builds the twin's entire view:

- `.claude/agents/codex-code-reviewer.md`
- `.claude/agents/codex-critic.md`
- `.claude/agents/codex-interpretation-critic.md`
- `.claude/agents/codex-clean-result-critic.md`
- `.claude/agents/codex-follow-up-critic.md`
- the five `codex-*-lean` twins (they defer to the sibling spec by reference,
  so confirm the fence reaches them rather than assuming it does)
- `.claude/agents/reconciler.md` — check separately. The reconciler is
  DESIGNED to see both verdict markers, so its constraint differs; decide
  whether sibling memory is in or out of its window and state which.

Also consider whether `.claude/rules/codex-ensemble-review.md` is the better
single home for the rule, with the specs pointing at it — that file already
carries the composer contract and the strip rules.

## Acceptance

1. Each in-scope composer spec carries an explicit standing prohibition on
   opening, grepping, or citing any path under `.claude/agent-memory/` OTHER
   than the composing agent's own directory, with the independence rationale
   stated (a future editor who does not know why the fence exists will
   delete it).
2. The rule states the DIRECTION precisely — own-memory read is designed
   behavior and stays permitted; sibling-memory read is banned — so the fix
   cannot be misread as banning agent memory outright.
3. A mechanical check backs it where one is cheap: a `workflow_lint.py`
   region-anchored surface pin that the fence text is present in each
   in-scope spec. A grep-based pin is adequate; a behavioral test is not
   required and probably not constructible.
4. The reconciler decision is recorded either way (fenced, or explicitly
   exempt with the reason).
5. Existing workflow-lint and spec tests stay green.

## Notes for the planner

- Do NOT try to solve this by removing agent-memory files from the branch or
  from the spec-freshness sync. They are checked in deliberately and the
  sync is load-bearing; a read fence in the spec is the proportionate fix.
- The ad-hoc fence wording the #2387 round-4 composer used is a reasonable
  starting point but was written for one prompt; generalize it.
- Check whether the Claude-side reviewers need the symmetric fence. The
  asymmetry is that the composer builds the twin's whole view, but a Claude
  reviewer reading the Codex composer's memory would be the same circularity
  in the other direction.

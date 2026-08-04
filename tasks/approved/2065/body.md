---
title: 'workflow-fix: flag dirty-tree provenance in result-JSON repr'
kind: infra
tags:
- wf-fix
- wf-fix-fp:58366cfd60f6
- daily-auto-filed
created_at: '2026-08-04T06:51:39Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-03 problem sweep (route 2): Both #1482 Shapley runs
  consumed a script that was modified-uncommitted by +369/-40 lines, so committed
  outputs record metadata.git_commit = e3ebc3a79b, a commit missing the logic that
  produced the figures; code-style.md has no dirty-tree clause (0 hits for git_commit).
  Routes a candidate session 201e2896 wrote up but parked for greenlight instead of
  filing.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-03 (route 2: behavior/logic change → independent review). This ROUTES a workflow-fix candidate that session 201e2896 wrote up but deliberately did NOT file, parking it as a chat question to Thomas ("I've written it up as a workflow-fix candidate but haven't filed it; it's your call whether that's worth a task") — the anti-pattern `.claude/rules/workflow-fix-on-bug.md` § Anti-patterns explicitly bans. The session was interactive and NOT recursion-guarded, so the candidate was routable at the time.

## Goal

A result JSON whose `metadata.git_commit` is stamped while the producing script is dirty-uncommitted must say so, so a committed artifact can never silently claim provenance from a commit that does not contain the code that produced it.

## Workflow gap

- **Bug observed:** both #1482 Shapley runs consumed `scripts/issue1482_shapley_blocks.py` while it was **modified-uncommitted by +369/−40 lines**, and the committed K=22 output records `metadata.git_commit = e3ebc3a79b` — a commit missing 369 lines of the logic that produced the figures. Session 201e2896 posted this as `epm:progress` v224 on #1482 ("PROVENANCE GAP") at 2026-08-04T00:56Z, wrote the candidate to `/tmp/provenance-gap.md`, and then asked rather than filing.
- **Why it is a workflow gap:** `git_commit` in reproducibility metadata is load-bearing — it is what the clean-result `**Repro:**` footer pins and what a re-run resolves. A dirty-tree stamp is not merely imprecise, it is affirmatively wrong in a way no downstream reader can detect. The rule that governs reproducibility metadata (`.claude/rules/code-style.md`, loaded whenever `*.py` is touched) has no dirty-tree clause: `grep -n 'git_commit\|reproducibility metadata\|dirty' .claude/rules/code-style.md` → **0 hits**.
- **Confidence (emitter):** high (the +369/−40 diff and the recorded sha are both quoted in the session's own marker).
- verified-at-filing: `grep -n 'git_commit\|reproducibility metadata\|dirty' .claude/rules/code-style.md` → **0 hits** (2026-08-04) — absence claim, verified in the target file. `grep 'PROVENANCE GAP' tasks/awaiting_promotion/1482/events.jsonl` → 1 hit (the `epm:progress` note). `grep -c 'workflow-fix-candidate' tasks/awaiting_promotion/1482/events.jsonl` → 4, newest **2026-07-19** — i.e. NO `epm:workflow-fix-candidate` marker was posted for this candidate, so the nightly parked-candidate sweep (Step C) could never have enumerated it; routing it here is the only path it had.
- unverified hypothesis — verify at plan time: which helper actually stamps the field. The candidate names the reproducibility-metadata convention generally; the planner should locate the shared stamping site (rather than patching one per-issue script) so the flag lands once for every producer.

## Proposed change (candidate sketch — refine in planning)

```
in the reproducibility-metadata convention (.claude/rules/code-style.md) and the
shared stamping helper:
  metadata["git_commit"]      = <sha>            # unchanged
+ metadata["git_dirty"]       = <bool>           # `git status --porcelain -- <producing script>` non-empty
+ metadata["git_dirty_paths"] = [...]            # when dirty
and: a producer that stamps git_commit while dirty either fails loud or records
the flag — never a bare sha.
```

Sequencing note for the planner (from the same incident): the inline payload lint gate already exists and can certify + commit a round-written script BEFORE the run that stamps its provenance; the flag is the backstop, not a licence to launch dirty.

## Scope / surfaces

- Primary target: `.claude/rules/code-style.md` (the convention) + the shared reproducibility-metadata stamping helper the planner locates.
- Grep before editing: `grep -rn 'git_commit' --exclude-dir=worktrees src/ scripts/ .claude/` and list the producer sites in the plan.

## Constraints / invariants

- Workflow-surface + shared-helper only; do not hand-patch individual per-issue scripts.
- Additive metadata keys — must not break existing readers of `metadata.git_commit`.
- `scripts/workflow_lint.py` (no-flags) passes; ruff on touched files passes.

## Provenance

- sha-verify (filing-time, #1467): `201e2896` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 58366cfd60f6

- workflow_fix_target: .claude/rules/code-style.md
- fingerprint: PLACEHOLDER

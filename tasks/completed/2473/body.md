---
title: 'workflow-fix: codex-* composers leave their own agent-memory writes uncommitted
  (3x in one session)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:990af9a92de0
- workflow-fix
created_at: '2026-08-22T17:58:33Z'
has_clean_result: false
origin_prompt: 'Observed 3x in one session (#2263 review rounds 4/5/6): every codex-code-reviewer
  prompt-composer wrote an agent-memory lesson, deliberately left it uncommitted reasoning
  that ''committing mid-round would enter the very diff under review'', and asked
  for a post-merge sweep; the orchestrator hand-committed each time (f002e1276e, 9a2af9494e,
  2ba96b4fe3). The reasoning is wrong (briefs pin the reviewable diff to explicit
  SHAs, so a later commit cannot enter the range) and the disposition is riskier (uncommitted
  agent-memory is the #2015 dominant standing-armer class; a dirty .claude/agent-memory
  also halts the Step 5a sync for that family). Neither codex-composer-common.md nor
  codex-code-reviewer.md says anything about committing memory writes.'
workflow: v1
---
## Overview / Motivation

Observed three times in a single session (#2263 review rounds 4, 5 and 6). Every `codex-code-reviewer` prompt-composer spawn wrote an agent-memory lesson, deliberately left it **uncommitted** in the issue worktree, and asked the orchestrator to "sweep it post-merge." Each time the orchestrator had to commit it by hand.

The composers' stated reasoning is consistent and superficially sound: *"committing mid-round would enter the very diff under review."* It is wrong on the facts, and the disposition it produces is the riskier one.

## Goal

Give the codex-* prompt-composers an explicit instruction covering their own agent-memory writes: commit them by explicit path in the producing turn, with the reason the mid-round-contamination worry does not apply.

## Workflow gap

**Why the composers' reasoning is wrong.** Review briefs pin the reviewable diff to explicit commit SHAs (`git show <sha>`, or an enumerated range), and the orchestrator names every excluded non-deliverable commit in both reviewer briefs. A commit landing *after* the brief is composed cannot enter a SHA-pinned range. The contamination the composers are avoiding is not reachable.

**Why the disposition is actively worse than committing.** `CLAUDE.md` § "Uncommitted TRACKED state at the shared root is unsafe under concurrency" (#2015) makes agent-memory writes the fleet's single largest standing-armer class — 8 of the 14 files in the #2015 standing diff — precisely because they have no natural committer. The rule is that the session whose agent wrote a memory file commits it by explicit path in the SAME turn. "Sweep it post-merge" is the mechanism by which that standing diff accumulates.

Two concrete hazards observed in this very session:
- One composer edited `MEMORY.md` (a TRACKED file) and left it modified-uncommitted while the same worktree had live pre-commit stash-race activity — the composer itself observed `~/.cache/pre-commit/patch*` files bracketing a window in which `10-step-6.md` transiently reverted. A tracked modification sitting in that window is exactly the #2015 permanent-loss shape.
- A dirty `.claude/agent-memory` also marks that family dirty for the Step 5a spec-freshness sync, so main-side agent-memory drift silently stops syncing into the worktree for the rest of the branch's life.

**Why it needs a spec fix rather than orchestrator vigilance.** The composers are behaving reasonably given their instructions — the shared composer contract (`.claude/rules/codex-composer-common.md`) says nothing about agent-memory or committing, and `.claude/agents/codex-code-reviewer.md` has no memory-commit instruction either. So each composer independently invents the same wrong heuristic, and correctness depends on the orchestrator noticing the flag in a long return block. It was noticed three times here; it will not always be.

## Proposed change (sketch — refine in planning)

Preferred target is the SHARED contract, since the hazard is identical for every codex-* composer (`codex-critic`, `codex-interpretation-critic`, `codex-clean-result-critic`, `codex-follow-up-critic`, `codex-code-reviewer`), not just the code-reviewer twin:

In `.claude/rules/codex-composer-common.md`:

```
+ ## Your own agent-memory writes
+
+ A memory lesson you write is a tracked write like any other: commit it by
+ explicit path, in the SAME turn, together with its MEMORY.md index row (one
+ commit, so the tracked index edit never sits uncommitted). Do NOT defer it to
+ a post-merge sweep and do NOT leave it for the orchestrator.
+
+ It cannot contaminate the round you are composing for: review briefs pin the
+ reviewable diff to explicit commit SHAs, so a commit landing after compose is
+ outside the reviewed range by construction. Leaving it uncommitted is the
+ riskier choice — uncommitted agent-memory is the fleet's dominant
+ standing-armer class (CLAUDE.md § "Uncommitted TRACKED state ...", #2015), and
+ a dirty .claude/agent-memory also stops the Step 5a spec-freshness sync for
+ that family for the rest of the branch's life.
+
+ Use the guarded-commit form with LITERAL paths:
+   git -C <absolute worktree path> commit -F <msgfile> -- <literal paths>
+ A variable-expanded pathspec reads as opaque to guard_root_code_commit.sh,
+ which then falls back to the shared staged index and blocks on another
+ session's payload.
```

Planning should decide whether `.claude/agents/codex-code-reviewer.md` also needs a pointer, and whether the Claude-side reviewer/critic specs have the same silence (the #2263 Claude `code-reviewer` DID commit its own memory unprompted at round 3, so the gap may be composer-specific).

**Related but distinct, worth checking in the same pass:** at #2263 review round 5 the Claude `code-reviewer` emitted two `CONCERN::` rows in its verdict marker and reported them as "persisted", but they never reached `concerns.jsonl` — the orchestrator's forwarder is wired for the Codex twins only, and the reviewer's own `raise-concern` never fired. That is a separate persistence gap on the Claude reviewer side; file it separately if planning confirms it is not already covered.

## Verified at filing

- #2263 `events.jsonl`: the round-4, round-5 and round-6 orchestrator `epm:progress` notes each record a composer's uncommitted-memory flag and the hand commit that followed (`f002e1276e`, `9a2af9494e`, `2ba96b4fe3`).
- `grep -n -i "commit" .claude/agents/codex-code-reviewer.md` — no agent-memory commit instruction (2026-08-22).
- `grep -n -i "agent-memory\|commit" .claude/rules/codex-composer-common.md` — zero hits (2026-08-22).
- Dedup: a repo-wide scan of `tasks/*/*/body.md` for `agent-memory` found no task covering composer memory-commit duty; the nearest hits are unrelated per-agent memory CONTENT syncs (#1200, #1236, #1261).

## Provenance

workflow_fix_target: .claude/rules/codex-composer-common.md

Routed from three observations during #2263's review rounds per `.claude/rules/workflow-fix-on-bug.md`. Filed by the #2263 orchestrator; `task_workflow.is_workflow_fix_session(2263)` is `False`, so the recursion guard does not apply, and the target file is unrelated to #2263's own deliverable.

---
title: Pre-split intermediate-unit review guard + cross-session shared-worktree writer
  arbitration
kind: infra
tags: []
created_at: '2026-08-06T19:17:37Z'
has_clean_result: false
origin_prompt: Filed by the autonomous /issue 1336 session 2026-08-06 after round
  4 lost 9 subagent deaths and 2 days to (1) a code-review dispatch scoped to an intermediate
  pre-split unit and (2) an independent Happy session concurrently editing the same
  file set in the same shared worktree.
workflow: v1
---
# Two pre-split / concurrency gaps that cost #1336 two days and 9 subagent deaths

<!-- workflow-fix-candidate v1 -->

Filed by the autonomous `/issue 1336` session (2026-08-06) after resuming a round-4 pre-split that had been parked at `blocked` since 2026-08-04. Both gaps are in the workflow surface itself, not in experiment code.

## Gap 1 — nothing mechanically prevents reviewing an INTERMEDIATE pre-split unit

**Target files:** `.claude/skills/issue/SKILL.md` (Step 5 / the #1810 pre-split composition block), `scripts/workflow_lint.py`.

The #1810 pre-split multi-deliverable build contract is documented correctly: intermediate units commit and return a commit manifest with NO implementation marker; only the FINAL unit runs the full per-phase `## Smoke run` H2 and posts `epm:experiment-implementation`; all units run within ONE review round, and the Step 5 ensemble reviews the whole round diff ONCE, after the final unit.

Nothing enforces it. On #1336 round 4 a prior session dispatched a `code-reviewer` round scoped to **Unit A alone** — an intermediate unit of a 3-unit split (recorded at `epm:progress v132`). There was no Unit-A-only review step to satisfy, so the round could not advance. Two subagent deaths (7 and 8 of 9) were spent on that dispatch before the task was parked at `blocked` with `epm:failure v3` (`failure_class: infra`, `reason: reviewer_no_durable_verdict_after_bounded_respawn`). The task then sat blocked for two days. The resuming session's first correct action was to NOT re-drive that review.

**Fix candidate.** A pre-dispatch guard on the Step 5 code-review site: refuse (or loudly warn) when the task's latest pre-split breadcrumb carries a non-empty `remaining:` list and no `epm:experiment-implementation` marker exists for the round. The breadcrumb grammar already exists and is machine-readable (`pre-split unit k/M complete: <SHAs>; remaining: <deliverables>`), so the predicate is cheap. A `workflow_lint.py` check could additionally assert the SKILL.md text states the guard.

## Gap 2 — "one implementer per file set" does not cover an independent concurrent SESSION

**Target files:** `CLAUDE.md` § "Orchestrator vs subagent re-invocation" (the teammate-coordination bullet), `.claude/rules/teammate-coordination` guidance as referenced there, `.claude/agents/experiment-implementer.md`.

The teammate-coordination rule says: ONE implementer per file set; a live owner means defer to it; stand-downs go over the teammate channel. That rule is scoped to **subagents the orchestrator spawns** — SendMessage reaches only Agent-tool subagents. It does not cover the case this round actually hit: **an independent Happy session concurrently editing the same files in the same shared worktree.**

Concretely, on #1336 two live sessions shared `.claude/worktrees/issue-1336-fullcorpora` (unavoidably — the second unit must build on the first's commits on one branch, and a branch checks out in exactly one worktree). The autonomous session owned round 4; the user's interactive session owned round 5. Their file sets were disjoint *when checked*, and the autonomous session posted a durable file-set claim naming the overlap risk. Round 5's Part A then edited `scripts/issue1336_extract_turnstore.py` anyway, mid-flight, while a round-4 implementer was editing it.

The implementer's dying words: *"The file gained a `--gen-format` flag (round 5) from the concurrent session mid-flight. Let me check its provenance and re-read the current main() before editing it."* It then died to autocompact thrash — having re-read a 1,132-line file that was changing underneath it.

Two sub-gaps:

1. **No cross-session writer arbitration.** The only steering channel to an independent session is a durable task marker (#1586), which is advisory and was in fact posted and not honoured (the sibling session had likely already loaded its plan). There is no documented protocol for "two sessions must write the same file set in one worktree" — e.g. an explicit lock marker, a claim-then-wait, or a rule that the second session waits for the first's commit.
2. **No read-pinning guidance for a churning target.** An implementer told to re-read a file after an external change has an unbounded read loop. Nothing instructs a subagent to pin its read to a SHA (`git show <sha>:<path>`) and work from that snapshot, resolving conflicts at commit time instead. This is a distinct thrash aggravator from raw spec size, and the documented Class-2 remedy (micro-scope + lean twin) does not address it.

**Fix candidate.** (a) Add a cross-SESSION clause to the teammate-coordination rule: before dispatching an implementer into a worktree, probe for other live sessions writing that file set (marker scan + `git log` recency on the paths), and if one exists either sequence after its commit or split to a distinct file set — never dispatch a concurrent writer. (b) Add read-pinning guidance to the implementer specs: pin to a SHA, do not re-read on external change, reconcile at commit. (c) Note in the pre-split block that a shared worktree is the *expected* shape for multi-unit splits, so this arbitration is a normal requirement, not an edge case.

## Evidence

- #1336 `epm:progress` v121 (pre-split unit definitions), v132 (the premature Unit-A-only review dispatch), v144/v145 (round 5's dispatch + completion, incl. the explicit hand-off leaving round 4's scope armed), v147 (the round-4 resume breadcrumb + file-set claim + overlap warning), and the Unit B death note posted alongside this filing.
- #1336 `epm:failure v3` (2026-08-04) — the park.
- Commits on `issue-1336-fullcorpora`: `f02bb56eb9` (Unit A), `9e648053b1` + `beed23dbae` (Unit B partial), against round 5's `d416e4579a` / `70f88c099a` / `661540e65c` / `090d0d2231`.
- Ruled out during diagnosis, recorded so it is not redone: Class-1 reduced-window autocompact enrollment (`CLAUDE_CODE_AUTO_COMPACT_WINDOW=600000` already set in both settings files; `compact_boundary preTokens` cluster at ~527k-572k, not the ~200-270k signature), and the oversized-tool-result read-side fix (the Unit A diff is 52,836 bytes, well under the 300 KB budget).

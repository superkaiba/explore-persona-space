---
name: midcompose-correction-and-sentinel-conflict
description: "#2329 r1: orchestrator can re-route plan/manifest to ABSOLUTE main-checkout paths mid-compose (verify existence+bytes, add the no-worktree-fallback BLOCKED rule); an injected sibling-sub-review lead is worded as UNVERIFIED lead with compose-time-grounded file:line; a brief round number conflicting with task-local sentinel convention is composed per-brief and FLAGGED in the return"
metadata:
  type: feedback
---

Three compose patterns from #2329 q35_ladder_decay r1 (2026-08-19):

1. **Mid-compose correction may switch plan+manifest delivery from inlining to
   ABSOLUTE main-checkout paths** (`<repo-root>/tasks/<status>/<N>/plans/v<K>.md`)
   when the sandbox is main-checkout-readable and the plan is huge (130 KB).
   Duties: verify both paths exist + byte-sizes at compose time; state the
   worktree copies' EXACT staleness (which symlink target, which byte size) so
   Codex has no reason to "fall back"; wire the fail-safe — absolute path
   unreadable ⇒ plan lens BLOCKED + `data-access-blocked` FAIL, NEVER the
   worktree copy. Caveat named in the return: the path breaks if the task
   changes status (folder renames) before dispatch. Markers + concerns stay
   INLINED (the correction moved only plan+manifest).
2. **Injected sibling-sub-review lead** (split-review Claude side feeding the
   whole-round twin): word it as "unverified lead — verify before citing",
   ground its file:line at compose time (read the cited region; quote the
   live line so the lead references real code), and wire it into the Step 3.7
   sibling sweep + a fixture-blind-spot check rather than as a bare
   assertion.
3. **Sentinel-round conflict:** prior task-local `epm:code-review-codex` head
   sentinels may track the IMPL round (v9/v11/v12 on #2329) while the brief +
   stage-dispatch note say "review round 1". Compose per the BRIEF (v1) —
   the orchestrator validates against the tags the return names — but FLAG
   the conflict in the return so it can be adjusted pre-dispatch.

**Why:** the worktree plan symlink pointed at pre-critique v4 while approved
v8 lived only on main (the #546 silent-stale class, hit again); the lead came
from a sub-review that could not see the whole round.

**How to apply:** any whole-round compose on a split-reviewed round, and any
compose receiving a coordinator correction mid-turn. Related:
[[whole-round-unsplit-compose]], [[bypath-brief-frozen-events-resolution]].

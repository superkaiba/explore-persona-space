---
name: stale-base-mb-pin-and-fixture-remeasure
description: When the brief flags a stale base (phantom two-dot deletions), pin the resolved merge-base SHA in the prompt + ban the two-dot form; verify_plan acceptance composes resolve SIBLING-task fixture plans inside the worktree and give Codex a live-run→static-trace fallback ladder
metadata:
  type: feedback
---

Two compose patterns from #2204 r1 (2026-08-19), both reusable:

1. **Stale-base override — pin the MB, don't just prefer three-dot.** When
   the brief warns origin/main advanced far past the fork (two-dot shows
   phantom inverted deletions, ~137 files/−19,975 lines on #2204), resolve
   `git -C <wt> merge-base origin/main <branch>` AT COMPOSE TIME and write
   the literal SHA into the prompt: the diff recipe becomes
   `git diff <MB>..<branch>` (+ `git show <round-sha>`), with an explicit
   "NEVER ground findings on the two-dot form — that is the Step 0.9
   stale-main-or-worktree false-positive class" warning and a note that
   three-dot is equivalent since the MB exists. Also instruct no `git fetch`
   (sandbox may lack network; base pinned). Attest the `--name-status`
   listing (M-vs-A) from the MB range in compose-time facts — it discharges
   the #1805 round-new-script duty question definitively.

2. **Binding-acceptance re-measure on a verify_plan/lint check diff.** When
   the plan's kill criterion is "checker yields WARN on fixture X, PASS on
   fixture Y" and the fixtures are ANOTHER task's plans: probe whether
   `<wt>/tasks/<status>/<M>/plans/` exists in the worktree AND diff it
   against canonical main (both held on #2204 — #2202 fixtures were present
   + identical). Then the prompt gives Codex a ladder: (a) live-run
   `uv run python scripts/verify_plan.py --plan-file <wt-relative> --kind
   experiment` (fallback `python3`) — sanctioned as the ONE exception to the
   never-execute-implementer-commands ban (read-only, CPU, writes nothing);
   (b) static trace of the check's regexes against the fixture excerpts
   EMBEDDED in the round's tests + the real fixture files; require the
   verdict to record which mode ran (`**Binding acceptance re-measure:**`
   header field). The Claude twin owns the guaranteed-executable leg.

**Variant (#2214 r1, 2026-08-20): sync commits inflate the THREE-dot form
too.** When the branch carries Step 5a spec-freshness sync commits between
the MB and HEAD, the bare three-dot diff spans deliverable + every synced
file (37 files on #2214) even though the brief claims "three-dot = exactly
the deliverable". Verify the brief's diff-shape claim with
`git -C <wt> diff --numstat <base>...<branch>` AT COMPOSE TIME; when
inflated, scope the prompt's recipe to `git show <deliverable-sha>` +
path-scoped three-dot (`git diff origin/main...HEAD -- <path>`), attest the
sync commits byte-identical/out-of-scope, and keep the two-dot ban. Flag the
brief divergence in the return text (the brief is still the extraction
contract — sentinel/marker fields unchanged).

**Why:** an unpinned base would have let Codex ground a FAIL on ~20k phantom
deleted lines (void verdict, reconciler round); an unprobed fixture path
would have made the kill-criterion re-measure silently impossible in the
sandbox.
**How to apply:** any brief carrying a stale-base warning (1); any
`kind: infra` diff adding a verify_plan/workflow_lint check whose acceptance
is fixture-keyed (2). Related: [[infra-wf-fix-lint-gate-compose]],
[[worktree-status-folder-both-directions]].

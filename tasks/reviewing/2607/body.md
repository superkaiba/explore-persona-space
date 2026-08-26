---
title: 'verify_task_body.py Lens 14 reads a worktree-frozen concerns.jsonl with no
  staleness guard (analyzer false PASS on #2378 r7)'
kind: infra
tags: []
created_at: '2026-08-26T19:54:41Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate routed from #2378 CRC r7 (both twins surfaced
  it); orchestrator session cmt6346nl90slyl0ucuvq85ex'
workflow: v1
---
## Goal
Add a worktree-staleness guard to verify_task_body.py's concerns-audit resolution (and restate the duty in analyzer.md): when the resolved tasks/<status>/<N>/concerns.jsonl lies inside a git WORKTREE (.claude/worktrees/*), the file is frozen at the worktree's base commit and silently stale — resolve via the branch-guarded task_workflow library against the MAIN checkout instead, or hard-WARN/FAIL naming the stale path.

Incident (task #2378, 2026-08-26, interpretation v9 fold): the analyzer ran verify_task_body and reported OVERALL PASS (77 checks) because its run resolved the task folder inside the issue worktree, whose frozen concerns.jsonl (139 lines) predates 5 open CONCERN-severity rows raised 07:11-11:26Z by the two follow-up rounds' review cycles; both clean-result-critic twins' main-root runs FAILed Lens 14 (1 of 77) on those same 5 unacknowledged concerns. The false PASS cost one full CRC ensemble round. Sibling class: #2422 (worktree tasks/ tree serves a stale plan/manifest with no error); Lens-14 verifier siblings #2530 (undercount on severity-downgrade re-raise) and #2535 (ack span to EOF) are DISTINCT defects on the same check.

## Acceptance
- verify_task_body.py resolves the concerns ledger (and any other per-task events/comments reads) via task_workflow/main-checkout resolution regardless of cwd, OR refuses with a named stale-worktree path when the resolution lands inside .claude/worktrees/.
- A regression test pins the guard (worktree fixture with a stale concerns.jsonl vs newer main-root rows).
- analyzer.md verify-step prose names the trap.

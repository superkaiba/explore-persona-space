# Step 3: Approval check (backward compat, runs on re-invocation)

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

Runs on re-invocation if status is `plan_pending` (i.e., user deferred or
approved via the dashboard / a `task.py set-status` call rather than
inline).

Scan `comments.jsonl` and recent `events.jsonl` rows after the plan
event for an explicit `approve` / `/approve` by the user. If found,
move status to `approved`. If a revision request is present
(`/revise <notes>`), set status back to `planning`, re-invoke
adversarial-planner with the notes; **also re-run the consistency
checker against the revised plan and post `epm:consistency v<n>` (a v2
plan that adds new conditions or shifts baselines must not skip the
consistency gate)**; post the new `epm:plan v2` via `new-plan-version`
with the fresh consistency verdict appended; set status back to
`plan_pending`.

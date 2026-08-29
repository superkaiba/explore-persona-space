---
title: 'workflow-fix: adversarial-planner plan-finalize cited-body currency gate (cited
  parent corrected mid-draft goes unseen)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-19T04:32:46Z'
has_clean_result: false
origin_prompt: 'Statistics critic prose follow-up during /issue 2378 planning: cited-body
  staleness race — #825 corrected 4 min before plan v5 persisted; add a plan-finalize
  cited-body currency re-check'
workflow: v1
---
# workflow-fix: adversarial-planner plan-finalize cited-body currency gate (cited parent corrected mid-draft goes unseen)

<!-- workflow-fix-candidate v1 -->

## Goal

Close the cited-body staleness race in the `/adversarial-planner` pipeline: a plan can cite a parent task's result that is corrected IN THE PARENT'S OWN BODY between the fact-checker's verification pass and the plan persist, so the plan (and everything downstream) inherits a superseded claim that the mechanical pre-pass cannot see.

## Incident (2026-08-19, task #2378 plan v5)

- #2378's plan grounded its H4 hypothesis on #825's user-turn linear NULL (ridge R² negative in both provenance arms). The fact-checker verified this against #825's body and marked it CONFIRMED — correctly, at read time.
- #825's body was then corrected via `task.py set-body` (commit `488dad540c`, 21:05:26): the null was an unguarded-GCV estimator artifact at n_tr < d; under the guarded selector (dof cap 0.9) all four user cells flip to ridge R² +0.19…+0.25 (artifact: `eval_results/issue_825/trackm_settle_battery/results.json`, `legA_ridge/*/guarded_ridge`).
- #2378 plan v5 was persisted 4 minutes later (commit `557716be3f`, 21:09:46), still quoting the superseded null across §0.0/§0/§1/§2/§3/§6/§8/Success criteria. Both registered H4a narration branches would have mis-stated the finding.
- Caught only by the Phase 2 Statistics critic (Claude), which independently probed the git history of the cited body. Nothing mechanical checks this.

## Proposed fix

At every `task.py new-plan-version` persist inside `/adversarial-planner` (the same pre-persist site as the Goal-currency gate), add a **cited-body currency check**:

1. Extract the set of task ids the plan cites (the `#<M>` references in §2/§11/§12 — the existing verify_plan task-ref extraction can be reused).
2. For each cited id, compare `git log -1 --format=%ct -- <task body.md path>` against the timestamp of the fact-checker's verdict (or, cheaper and stricter, against the plan-draft start time recorded alongside the Goal snapshot).
3. Any cited body whose last commit postdates that timestamp ⇒ do NOT persist silently: surface the body diff since the reference point to the orchestrator, which re-reads the changed section(s) and either confirms the plan text is unaffected (record a one-line disposition) or bounces the affected sections for a mechanical re-ground (same non-cap-counting semantics as the Goal-currency redraft bounce).

Implementation candidates (implementer's choice): a helper in `scripts/` invoked by the adversarial-planner SKILL.md pre-persist checklist next to the Goal-currency gate (preferred — same site, same bounce semantics), and/or a WARN-only `verify_plan.py` check (cited-body mtime vs newest plans/v{K}.md mtime) as the mechanical backstop.

## Related (optional scope, from the same review round — separate checks, include only if cheap)

The round's reviewers flagged three further mechanizable verifier gaps (candidates for `verify_plan.py` extensions; distinct from the currency gate and NOT required for this task's acceptance): (a) N-pod fan-out §9 rows lacking distinct `--name-suffix` values + a same-prefix staging strategy; (b) coverage gates whose count-source artifact is produced BEFORE later row-dropping stages (the #2378 G2/kept.json ordering shape); (c) gate-vs-kill-criteria predicate consistency over synthetic survivor rosters.

## Acceptance criteria

1. A plan persist inside `/adversarial-planner` with a cited body committed after the reference timestamp surfaces the staleness (test: fixture repo state or monkeypatched git log).
2. The check is fail-soft on missing/unparseable cited ids (never blocks a persist on its own crash).
3. The adversarial-planner SKILL.md pre-persist checklist names the gate next to the Goal-currency gate.
4. Existing tests pass; new pin test for the helper.

## Provenance

Surfaced by the Phase 2 Statistics critic during /issue 2378 planning (session d56809b9, 2026-08-19). Filed by the #2378 orchestrator per `.claude/rules/workflow-fix-on-bug.md` (prose follow-up auto-file).

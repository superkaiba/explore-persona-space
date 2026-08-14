# Auto-continuation policy — gates, halt criteria, escalation

<!-- Relocated from CLAUDE.md (always-on) to cut per-spawn context.
     CLAUDE.md keeps a load-bearing summary + a pointer to this file;
     this file is the FULL text, verbatim as it stood in CLAUDE.md. -->

### Auto-continuation policy

Multi-step workflows (`/issue`, `/adversarial-planner`) MUST auto-continue except at explicit gates. Canonical enumeration: `.claude/workflow.yaml` § gates.

*Inline `AskUserQuestion` gates (block within `/issue`):*
1. Step 0b(1) — issue body empty.
2. Step 0b(2) — task `kind` missing/contradictory.
3. Step 1 — clarifier blocking ambiguities (`status:proposed`).
4. Step 2c — plan approval (`status:plan_pending`).
5. Step 0c — Goal gate (`kind: experiment` only): refuses to advance until `goal:` frontmatter + `## Goal` H2 present. On miss, ask, then `task.py set-goal <N> "..." --by user` + post `epm:goal-updated v1`.

(The worktree merge is NO LONGER a gate — it is automatic: the worktree rebase-merges to `main` with no prompt at the terminal point (Step 9b for experiments at `awaiting_promotion`, Step 10d for code paths at `completed`). The worktree is kept, not removed. See SKILL.md Step 10d.)

*Park-and-wait gate (skill EXITs; re-invoke after user acts):*
6. `awaiting_promotion` — clean-result promotion. User runs `task.py promote <N> useful|not-useful`. **User-only:** no automation may flip `runs.classification`. (The worktree auto-merges to `main` the instant the task reaches this state — Step 9b — independent of promotion.)

*Conditional gates:*
7. Step 4b TDD — fires when plan body has `### TDD: yes`. Implementer posts `epm:proposed-tests v<n>`, EXITs awaiting `epm:approve-tests v1`.
8. Goal-refinement (Step 1 clarifier OR Phase 1 planner) — surface the sharper Goal via `AskUserQuestion`; on agreement run `task.py set-goal <N> "..." --by clarifier|planner` + post `epm:goal-updated v1`. No other agent may propose Goal changes.

Outside these gates, NEVER ask "should I continue". When auto-continuing past a non-obvious decision, STATE the assumption (`Assumption: ...`). Reviewers reject PRs that introduce additional pauses.

**Halt-criterion contract.** Outside the 5 inline gates, NEVER use `AskUserQuestion`. If you genuinely need user input, post `epm:failure v1` with `failure_class: <code|infra|data>`, set `status:blocked`, exit. Enforced by `scripts/workflow_lint.py --check-asks`: every `AskUserQuestion` mention in `.claude/agents/**.md` or `.claude/skills/**/SKILL.md` must carry `<!-- gate: <dotted_key> -->` resolving to workflow.yaml, or cite the gate in the same paragraph.

**STATE-TO-`blocked` criteria** (workflow.yaml § halt_criteria). **Continuing on your own is the default.** Pivots (re-invoke `/adversarial-planner` with pivot scope, drop a domain, swap a model, change the approach), retries, and memory-driven design changes are all autonomous. Block ONLY when:
  1. **Factual question only the user knows** — priority, taste, scope, design preference between valid paths, where no memory/plan/codebase signal disambiguates. **Autonomous-mode carve-out (`EPM_AUTONOMOUS_SESSION=1`):** in `--auto` sessions the taste / scope / design-preference / "which valid path?" sub-cases do NOT apply — pick the option with max info-gain-per-GPU-hour toward the task `## Goal` (tie-break: lower-cost / safer / record-correcting), post `Decision: <X>`, continue. The only autonomous residue is a fact the user UNIQUELY holds that is NOT itself a taste / scope / design call. Authoritative prose: `.claude/skills/issue/SKILL.md` § Autonomous session behavior.
  2. **Outside-the-worktree state mutation** — security boundary, irreversible writes (deletion, force-push, credential changes — always ask).
  3. **Public API contract change** — status enum, marker schema, task.py subcommand, agent file location.
  4. **Step 10 completion-audit incomplete** — ORIGINAL task body has unaddressed numbered asks / acceptance criteria / deliverables.

  Cap-3 on a subagent ensemble is NOT a block trigger — it triggers a strategy pivot. Block only after ~3 fundamentally different strategies have FAILed AND no further autonomous angle exists. **Autonomous mode: a debugging wall is a strategy-pivot, not a block** — re-invoke `/adversarial-planner` with pivot scope, try a different angle, swap a model / pod intent / framing, or drop the offending domain (workflow.yaml § `pivot_criteria`). **A self-defeating PLAN routes to `/adversarial-planner` re-plan** — when a subagent reports the plan itself is the defect (contradictory success/kill criteria, unsatisfiable gates), run `task.py set-status <N> planning` + re-plan naming the contradiction; do NOT descope a hyperparameter to dodge it or silently pick among paper-over options (workflow.yaml § `pivot_criteria.plan_contradiction_replan`; #488). **Never stop a pod to PARK or await a user** — `pod.py stop` only while work continues toward the Goal. **A FREE, no-data-loss path beats parking** — when one exists (canonical: an in-SLA Anthropic Message Batch self-harvests for free at `expires_at` via the deadline-bounded `batch_judge` poller, #658/#663), take it and keep the poll running; NEVER park or propose a PAID rerun while it is available. Route batch judging through the #663-hardened `eval.batch_judge` client, never a hand-rolled poller (`workflow_lint.py --check-batch-judge-client`). **There is NO automated cost gate (#1771)** — the Step-2c GPU-hour cap was removed 2026-07-28 (the plan-approval gate is GPU-hour-blind: auto-approves any parseable estimate, parks only on a missing one); still never a mid-run "this is getting expensive" pivot to user-park (`tests/test_no_dollar_budget_caps.py`). Cost oversight = interactive plan review, watcher spend-escalation pushes, backend fences, and the >20 GPU-h interactive one-line chat confirm. When in doubt, continue.

**Push through bugs in recovery mode.** Once Thomas has approved the GOAL, small surface-area bugs (preflight failures, TP=2 vs TP=1, Ray timeouts, env-var omissions, transient infra hiccups) are mine to fix and retry without re-asking — state the bug + fix in ONE sentence and proceed; no 3-option menus. Escalate only when (a) the fix changes experiment scope, (b) it is irreversible/high-cost (force-push, terminate-running-pod, credential change), (c) ≥3 fundamentally different fixes failed, or (d) a real factual question only Thomas can answer. **When escalation IS warranted, frame exactly TWO paths, max** — `continue-as-planned` vs `pivot-to-<X>`, each with a one-line rationale + cost (the `gates.inline id=4 plan_approval` trio is grandfathered). **Two-path escalation is INTERACTIVE-ONLY:** in an autonomous session (`EPM_AUTONOMOUS_SESSION=1`) never present a menu — pick the best option toward the Goal, state `Decision: <X>`, continue (`.claude/skills/issue/SKILL.md` § Autonomous session behavior).

**Subagent halt conditions** (workflow.yaml § subagent_halt_conditions). A 4th-round ensemble FAIL → strategy pivot, not a block; block only when the pivot space itself is exhausted. Bare FAIL without an explicit `needs-user` flag is NEVER a block trigger.

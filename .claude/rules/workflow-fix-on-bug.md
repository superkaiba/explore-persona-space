# Workflow-fix-on-bug protocol

When any agent — subagent or orchestrator — hits a bug caused by a gap
in the workflow surface itself (NOT an experiment-specific or
task-state-specific bug), it MUST emit a `<!-- workflow-fix-candidate v1
-->` block in its return text. The parent orchestrator that receives the
return spawns `workflow-improver` in the background to apply the fix.
The current task continues uninterrupted; the diff lands as
`epm:workflow-fix-applied v1` on the originating task's `events.jsonl`.

Purpose: collapse the lag between "agent hits a workflow bug" and
"workflow file gets fixed." Previously this lag was a daily / weekly
cycle (`/daily`, `/weekly`, `retrospective`) or required Thomas to
notice the recurrence manually. Now it's same-turn.

## Workflow surface (what workflow-improver may touch)

- `.claude/agents/*.md`
- `.claude/skills/**/SKILL.md` (plus skill support files: `markers.md`, `iterations.md`, etc.)
- `.claude/rules/*.md`
- `.claude/workflow.yaml`
- `.claude/settings.json` and `.claude/settings.local.json`
- `.claude/mcp.json` (read-only unless explicitly asked)
- `CLAUDE.md` (project root)
- Workflow-helper scripts under `scripts/`: `task.py`, `task_workflow.py`,
  `pod.py`, `runpod_api.py`, `bootstrap_pod.sh`, `pods.conf`,
  `pods_ephemeral.json`, `workflow_lint.py`, `verify_task_body.py`,
  `audit_clean_results_body_discipline.py`, `codex_task.py`,
  `poll_pipeline.py`, `gh_project.py`, `spawn_session.py`,
  `pod_watch.py`, `worktree_audit.py`, `cron_worktree_audit.sh`
- `tests/test_workflow*.py`, `tests/test_task_workflow*.py`,
  `tests/test_no_dollar_budget_caps.py`, and other tests that pin
  workflow invariants

## Out of scope (DO NOT surface a candidate)

- `src/explore_persona_space/**` — library + research code
- `configs/**` — Hydra experiment configs
- `scripts/train.py`, `scripts/eval.py`, `scripts/run_sweep.py`,
  `scripts/generate_*.py`, `scripts/analyze_results.py` — experiment
  entrypoints
- `tasks/**` — task workflow state (read only; never edit
  body.md, events.jsonl, plans/, artifacts/)
- `eval_results/**`, `figures/**`, `ood_eval_results/**`, `docs/**`,
  `archive/**`, `external/**`, `raw/**`

If your bug is in the out-of-scope set, the fix belongs to
`experiment-implementer` / `implementer` / a follow-up task — not to
`workflow-improver`. Don't emit a candidate.

## When to emit a candidate

### Yes — emit

- An agent's instructions are silent on a known-tricky operation that
  just bit you (e.g. "the experimenter doesn't verify pod hostname
  after `pod.py resume`" → fix `.claude/agents/experimenter.md`).
- A skill step has a known failure mode without a guardrail
  (e.g. "the `/issue` step that flips status doesn't post the marker
  on failure" → fix `.claude/skills/issue/SKILL.md`).
- A workflow-helper script silently swallows a failure that just bit
  you (e.g. "`pod.py terminate` reports `POD_NOT_FOUND` but the pod is
  still alive in the API" — already captured in CLAUDE.md memory; the
  analogue for a *new* silent-failure class is in scope).
- A marker schema in `workflow.yaml` is missing a field you needed.
- A halt-criterion / gate is wrong, missing, or contradicts CLAUDE.md.
- A test that should have caught a workflow regression is missing.
- `CLAUDE.md` describes a rule but the implementing file (agent, skill,
  script) doesn't enforce it.

### No — don't emit

- The bug is in experiment code (training, eval, data generation,
  Hydra config, model spec).
- The bug is task-state-specific (one task's body.md has wrong tags;
  fix that task, not the workflow).
- The bug is environment / external infra flakiness (RunPod
  `SUPPLY_CONSTRAINT`, HF Hub 503, transient WandB outage). Retries
  belong to the experimenter; emit a candidate ONLY if the workflow is
  missing a retry / backoff policy entirely.
- The bug is a one-off that won't recur (a typo in your own
  cwd-resolution this turn, a stale memory line, a one-off shell-quoting
  mistake).
- You are <60% confident the workflow file needs to change. Surface the
  observation as a plain note in `events.jsonl` instead — a fix that
  gets reverted next round wastes more attention than it saves.
- You are running under `AUTO_REVIEW_DISABLED=1` (already nested inside
  a review or diagnostic loop). Don't recurse.

### Borderline

If the bug is "the workflow allowed me to do X, but X turned out to be
wrong for *this* experiment" — emit a candidate ONLY if the correct fix
is to make the workflow reject X categorically. If the correct fix is
"this specific experiment shouldn't have done X but others should," it's
experiment-specific.

## How to emit a candidate

Include this block in your final return text — after your main report,
before any closing sentence. Plain text, exact format (the orchestrator
parses it):

```
<!-- workflow-fix-candidate v1 -->
target_file: <path under workflow surface, relative to repo root>
bug_observed: <one sentence: what went wrong>
why_workflow_gap: <one sentence: why this is the workflow's fault>
proposed_change: <one sentence summary of the fix>
diff_sketch: |
  <2-10 lines showing the rough shape of the edit; workflow-improver
  will refine. Use `+ ` / `- ` prefixes if it helps.>
confidence: low | medium | high
related_task: <task ID this surfaced on, e.g. #391, or n/a>
<!-- /workflow-fix-candidate -->
```

Hard rules:

1. **At most one candidate per agent invocation.** If you notice
   multiple workflow bugs in one run, pick the most concrete + highest
   confidence; mention the others in your main report under a `##
   Follow-up workflow concerns` H2 so the orchestrator can surface them
   on the next pass.
2. **Never spawn `workflow-improver` yourself**, even if your tool
   allowance includes `Agent`. Surface the candidate; the parent
   orchestrator dispatches. This prevents runaway recursion (subagent →
   spawns workflow-improver → workflow-improver's code-reviewer spots
   ANOTHER workflow bug → ...).
3. **Don't emit if you're a Codex twin.** The Codex ensemble reviewers
   (`codex-*`) post their verdicts and exit; they never spawn
   subagents. If a Codex twin notices a workflow gap, it should write a
   plain English note in its verdict body — the orchestrator decides
   whether to surface it as a candidate later.

## What the orchestrator does on seeing a candidate

When any subagent returns text containing a `<!-- workflow-fix-candidate
v1 -->` block, the orchestrator (parent assistant, `/issue` skill,
`research-pm`, or any session running the top-level loop):

1. **Logs** the candidate to the current task's `events.jsonl` as `epm:
   workflow-fix-candidate v1` (so the dashboard surfaces it).
2. **Spawns** `workflow-improver`:
   ```
   Agent(
     subagent_type="workflow-improver",
     run_in_background=true,
     isolation="worktree",
     description="<one-line summary from proposed_change>",
     prompt="""
   ## Source: workflow-fix-candidate

   <verbatim candidate block, including the opening + closing comment lines>

   ## Originating task
   <task ID + brief context: what the emitting agent was doing when it hit the bug>

   ## Success criteria
   workflow_lint.py --check-asks passes; ruff check .claude scripts passes;
   if you touched workflow.yaml or CLAUDE.md, the two stay consistent.
   """
   )
   ```
3. **Continues** the current work. Does NOT block on the fix.
4. **On notification** (workflow-improver exit), posts `epm:
   workflow-fix-applied v1` to the same task's `events.jsonl` with the
   final unified diff inline. On FAIL, posts
   `epm:workflow-fix-failed v1` with the failure reason and the
   original candidate preserved.

If the orchestrator is *itself* the agent that found the bug (no
subagent involved — the bug surfaced during the orchestrator's own
work), it spawns `workflow-improver` directly with the same protocol.

## When the orchestrator suppresses the spawn

The orchestrator MAY log the candidate but skip the spawn when:

- `confidence: low` AND `proposed_change` is speculative ("maybe X
  should be changed but I'm not sure"). The marker is logged for the
  dashboard; no fix dispatched.
- A `workflow-improver` is already running on the same `target_file` in
  this session. Queue the new candidate as a follow-up via
  `SendMessage` to the running agent.
- The candidate's `target_file` is in the out-of-scope set (experiment
  code, tasks/, etc.). The orchestrator logs the candidate AND posts a
  brief note in the marker about the misclassification so the emitting
  agent's pattern can be corrected.
- The candidate's `proposed_change` requires a fundamental
  architectural decision (e.g. "rename a status enum", "remove the
  Codex ensemble") — these warrant Thomas's explicit greenlight, not
  background fix. Log the candidate; surface it to Thomas in the next
  chat turn.

## Markers

Defined in `.claude/workflow.yaml § markers`:

- `epm:workflow-fix-candidate v1` — posted by orchestrator on receiving
  a candidate block from any subagent's return text (or from its own
  observation).
- `epm:workflow-fix-applied v1` — posted by orchestrator after
  `workflow-improver` returns with reviewer PASS (or a surgical change
  needing no review).
- `epm:workflow-fix-failed v1` — posted by orchestrator if
  `workflow-improver` returned FAIL.

Posting target: the `events.jsonl` of the task the emitting agent was
working on. If the emitting agent was working outside any task (e.g.
during `/pm` triage, `/daily`, or chat-mode work), the orchestrator
appends to `.claude/cache/workflow-fix-events.jsonl` instead — same
schema, file-based fallback so the dashboard can still surface them on
the homepage.

## Anti-patterns

| Don't | Do |
|---|---|
| Subagent spawns `workflow-improver` itself | Surface the candidate; orchestrator spawns |
| Emit a candidate for an experiment-code bug | Route to `implementer` / `experiment-implementer` |
| Emit ≥2 candidates per run | Pick one; list the rest as Follow-ups |
| Emit `confidence: high` without a concrete diff_sketch | Sketch the actual lines; if you can't, drop to `medium` or skip |
| Wait for `workflow-improver` before continuing | Background-spawn; current task continues immediately |
| Emit a candidate against `src/`, `configs/`, `tasks/` | Out of scope — fix belongs elsewhere |

## Composition with other rules

- **AUTO_REVIEW_DISABLED sentinel** (user-global CLAUDE.md): suppresses
  this protocol too. If your prompt carries that sentinel, treat
  workflow-fix candidate emission as forbidden for the turn.
- **Halt-criterion contract** (CLAUDE.md): emitting a candidate is NOT
  the same as raising `AskUserQuestion`. The candidate is a non-blocking
  side channel; it does not pause the current work, does not flip
  status, does not consume a gate.
- **`workflow-improver` in-scope rules**
  (`.claude/agents/workflow-improver.md` § What "the workflow" means
  here): the workflow-improver enforces its own in-scope check on
  receiving the candidate. If the candidate is misclassified, the
  workflow-improver deflects in its report and the orchestrator posts
  `epm:workflow-fix-failed v1` with `failure_reason: out-of-scope`.
- **Codex ensemble reviewers**: never emit candidates (rule above).
  They write notes in their verdict body; the Claude twin (or
  reconciler) decides whether to emit a candidate.

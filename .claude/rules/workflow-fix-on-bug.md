# Workflow-fix-on-bug protocol

When any agent — subagent or orchestrator — hits a bug caused by a gap
in the workflow surface itself (NOT an experiment-specific or
task-state-specific bug), it MUST emit a `<!-- workflow-fix-candidate v1
-->` block in its return text. The parent orchestrator that receives the
return spawns `workflow-improver` in the background to apply the fix.
The current task continues uninterrupted; the diff lands as
`epm:workflow-fix-applied v1` on the originating task's `events.jsonl`.

**Surfaced-prose follow-ups count too.** A formal `<!-- workflow-fix-candidate
v1 -->` block is the canonical channel, but any concrete workflow
improvement an agent surfaces in its report prose — e.g. a "Follow-ups
(orchestrator should consider)" section, a "Related concerns" bullet, or
any specific suggestion to change a workflow-surface file — triggers the
SAME default action: the orchestrator AUTO-SPAWNS `workflow-improver` in
the background, treating the surfaced prose as if it were a candidate
block, under the same bar (in-scope per the workflow surface list,
non-architectural / not a public-contract change, and the orchestrator's
read of the proposal is >=medium-confidence). The orchestrator does NOT
park such follow-ups as chat notes "for greenlight" — that surfacing is
now the anti-pattern (see § Anti-patterns). Greenlight stays reserved
for the same two exceptions that apply to formal candidate blocks:
genuinely architectural / public-contract changes, or low-confidence
speculative fixes.

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
- `.claude/agent-memory/**/*.md` — persistent agent memories (always-loaded
  guidance steering workflow agents; correcting or retiring a stale memory
  is a workflow-surface fix, the owning agent remains the primary author)
- `CLAUDE.md` (project root)
- The task-workflow API library modules under `src/`:
  `src/explore_persona_space/task_workflow.py` and
  `src/explore_persona_space/task_workflow_migrate.py` — workflow surface
  despite the general `src/**` exclusion below
- Workflow-helper scripts under `scripts/`: `task.py`,
  `pod.py`, `pod_lifecycle.py`, `pod_config.py`, `pod_audit.py`,
  `gpu_heuristics.py`, `cleanup_pod.py`, `pod_disk_guard.py`,
  `runpod_api.py`, `bootstrap_pod.sh`, `cron_pod_audit.sh`,
  `sync_pods.sh`, `_pods_conf_path.sh`, `pods.conf`,
  `pods_ephemeral.json`, `workflow_lint.py`, `verify_task_body.py`,
  `audit_clean_results_body_discipline.py`, `codex_task.py`,
  `poll_pipeline.py`, `failure_classifier.py`, `gh_project.py`,
  `spawn_session.py`,
  `pod_watch.py`, `worktree_audit.py`, `cron_worktree_audit.sh`,
  `autonomous_session_watch.py`, `cron_autonomous_session_watch.sh`,
  `session_progress_report.py`, `session_summarize.py`,
  `session_resolver.py`, `cron_session_summarize.sh`
- `tests/test_workflow*.py`, `tests/test_task_workflow*.py`,
  `tests/test_failure_classifier.py`,
  `tests/test_no_dollar_budget_caps.py`, and other tests that pin
  workflow invariants

## Out of scope (DO NOT surface a candidate)

- `src/explore_persona_space/**` — library + research code (EXCEPT
  `task_workflow.py` + `task_workflow_migrate.py`, listed above)
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

A "candidate" here means either (a) a formal `<!-- workflow-fix-candidate
v1 -->` block (canonical, parseable, preferred when you can sketch the
diff cleanly), or (b) a concrete workflow-improvement suggestion you
surface as prose in your report (e.g. a "Follow-ups (orchestrator should
consider)" section or a "Related concerns" bullet that names a specific
workflow-surface file + a specific change). Both forms trigger the same
auto-spawn default; the same yes/no criteria below apply to both. Prefer
the formal block when you can — it parses unambiguously — but a prose
follow-up is not a downgrade and does NOT get parked for greenlight just
because it lacked the comment tags.

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

1. **At most one formal `<!-- workflow-fix-candidate v1 -->` block per
   agent invocation.** The block is the parseable channel; one keeps
   the orchestrator's auto-spawn deterministic. If you notice multiple
   workflow bugs in one run, pick the most concrete + highest
   confidence for the block. **Surface the others as prose follow-ups
   in your main report (e.g. a `## Follow-up workflow concerns` H2 or
   "Follow-ups (orchestrator should consider)" section).** Those prose
   follow-ups are NOT capped — list as many as you genuinely found, one
   per file/concern with a one-line proposed change. The orchestrator
   auto-spawns `workflow-improver` for each in-scope, non-architectural,
   >=medium-confidence prose follow-up on the same default as the formal
   block; do NOT hold them back hoping they'll surface "on the next
   pass."
2. **Never spawn `workflow-improver` yourself**, even if your tool
   allowance includes `Agent`. Surface the candidate (block OR prose);
   the parent orchestrator dispatches. This prevents runaway recursion
   (subagent → spawns workflow-improver → workflow-improver's code-
   reviewer spots ANOTHER workflow bug → ...).
3. **Don't emit if you're a Codex twin.** The Codex ensemble reviewers
   (`codex-*`) post their verdicts and exit; they never spawn
   subagents. If a Codex twin notices a workflow gap, it should write a
   plain English note in its verdict body — the orchestrator decides
   whether to surface it as a candidate later.

## What the orchestrator does on seeing a candidate

**Default: AUTO-SPAWN, do not park.** For any workflow-fix candidate
that is (a) in-scope per the workflow surface list above, (b)
non-architectural / not a public-contract change, and (c)
`confidence: medium` or higher, the orchestrator's default action is to
spawn `workflow-improver` immediately in the background (non-blocking)
and keep working. **"Candidate" means BOTH (i) a formal `<!-- workflow-
fix-candidate v1 -->` block AND (ii) any concrete prose follow-up an
agent surfaces — e.g. a "Follow-ups (orchestrator should consider)"
section, a "Related concerns" bullet, or any specific suggestion to
change a workflow-surface file.** Both come in via the same channel
(agent return text) and trigger the same default. This applies whether
the candidate came from a subagent's return text OR from the
orchestrator's own observation during its work (see the "orchestrator
is itself the agent" clause below). Parking the candidate for Thomas's
greenlight is the EXCEPTION, reserved for the two cases enumerated in
"When the orchestrator suppresses the spawn" — genuinely architectural
/ public-contract changes, or low-confidence speculative fixes. Do NOT
park an in-scope, non-architectural, >=medium-confidence gap as a chat
note — auto-fix it, regardless of whether it arrived as a formal block
or as prose.

When any subagent returns text containing EITHER a `<!-- workflow-fix-
candidate v1 -->` block OR a prose follow-up that names a specific
workflow-surface file + a specific change, the orchestrator (parent
assistant, `/issue` skill, `research-pm`, or any session running the
top-level loop):

1. **Logs** the candidate to the current task's `events.jsonl` as `epm:
   workflow-fix-candidate v1` (so the dashboard surfaces it). For prose
   follow-ups, the marker `note` records the file + summary the
   orchestrator extracted from the prose, plus a `source: prose-followup`
   field; for formal blocks it records the verbatim block plus
   `source: candidate-block`.
2. **Spawns** `workflow-improver`. For a formal block, paste it verbatim:
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
   workflow_lint.py --check-asks passes; ruff check on the files you
   touched passes (touched files only — the broad `.claude scripts` sweep
   has ~1300+ pre-existing errors and is not a gate);
   if you touched workflow.yaml or CLAUDE.md, the two stay consistent.
   """
   )
   ```
   For a prose follow-up, synthesize an equivalent candidate from the
   surfaced prose (pull `target_file`, `bug_observed` /
   `why_workflow_gap`, and `proposed_change` directly from the agent's
   words; mark `confidence: medium` unless the prose itself states
   higher; `diff_sketch: |\n  (none — synthesized from prose follow-up)`):
   ```
   Agent(
     subagent_type="workflow-improver",
     run_in_background=true,
     isolation="worktree",
     description="<one-line summary from proposed_change>",
     prompt="""
   ## Source: workflow-fix-candidate (synthesized from prose follow-up)

   <!-- workflow-fix-candidate v1 -->
   target_file: <path>
   bug_observed: <one sentence pulled from the prose>
   why_workflow_gap: <one sentence pulled from the prose>
   proposed_change: <one sentence pulled from the prose>
   diff_sketch: |
     (none — synthesized from prose follow-up; refine as you read the file)
   confidence: medium
   related_task: <task ID or n/a>
   <!-- /workflow-fix-candidate -->

   ## Verbatim surfaced prose
   <copy the relevant prose paragraphs / bullets from the originating
   agent's report so workflow-improver has the full context>

   ## Originating task
   <task ID + brief context>

   ## Success criteria
   workflow_lint.py --check-asks passes; ruff check on the files you
   touched passes (touched files only — the broad `.claude scripts` sweep
   has ~1300+ pre-existing errors and is not a gate);
   if you touched workflow.yaml or CLAUDE.md, the two stay consistent.
   """
   )
   ```
3. **Continues** the current work. Does NOT block on the fix.
4. **On notification** (workflow-improver exit, PASS): **auto-merges +
   pushes, no approval gate** (standing user rule, 2026-06-02:
   workflow-surface edits are committed + merged + pushed automatically
   as they are made). The workflow-improver has already committed its
   verified edits inside its worktree branch (its step 6.5) and reported
   the branch + commit SHA. From the repo root (which stays on `main` —
   never switch branches there), the orchestrator merges that branch and
   pushes:
   ```bash
   git -C "$REPO_ROOT" merge --no-ff <wf-branch> -m "merge workflow-fix: <summary>"
   # MERGE-COMPLETION ASSERT — never leave the shared repo root mid-merge:
   # a conflicted merge left sitting blocks task.py commits repo-wide and a
   # concurrent session can sweep YOUR staged files into ITS resolution
   # commit (both happened 2026-06-09, ~22:05Z). On conflict: abort and
   # requeue, do not hand-resolve while other sessions commit around you.
   if [ -f "$REPO_ROOT/.git/MERGE_HEAD" ] || [ -n "$(git -C "$REPO_ROOT" diff --name-only --diff-filter=U)" ]; then
     git -C "$REPO_ROOT" merge --abort
     git -C "$REPO_ROOT" pull --rebase && git -C "$REPO_ROOT" merge --no-ff <wf-branch> -m "merge workflow-fix: <summary>" || {
       echo "merge still conflicted — requeue"; exit 1; }   # -> post epm:workflow-fix-failed
   fi
   # Staging sanity: nothing foreign staged (a concurrent session's files)
   git -C "$REPO_ROOT" diff --cached --name-only   # must be empty post-merge
   git -C "$REPO_ROOT" push origin main
   git -C "$REPO_ROOT" log -1 --oneline -- <changed-file>   # landing check: confirm it's on main
   # Agent-isolation worktrees stay LOCKED until the harness reaps them —
   # unlock first or remove fails exit-128 (3 hits on 2026-06-09):
   git -C "$REPO_ROOT" worktree unlock <worktree-path> 2>/dev/null
   git -C "$REPO_ROOT" worktree remove <worktree-path>      # cleanup
   ```
   Then posts `epm:workflow-fix-applied v1` to the same task's
   `events.jsonl` with the final unified diff inline + the merge SHA. On
   FAIL (workflow-improver reported a failed check, or did NOT commit),
   posts `epm:workflow-fix-failed v1` with the failure reason and the
   original candidate preserved; nothing is merged. Force-push is NEVER
   auto (it stays a user-ask per CLAUDE.md); a normal push to `main` is
   covered by this standing rule. If the push is rejected (non-fast-
   forward), `git pull --rebase` once and retry; if it still fails, post
   `epm:workflow-fix-failed v1` and surface to the user.

   **Orchestrator's own direct workflow edits** (the orchestrator edited
   a workflow-surface file itself in the repo root on `main`, no
   worktree involved): the same standing rule applies in its simpler
   form — commit the touched workflow files BY EXPLICIT PATH (never
   `git add -A`, to avoid sweeping unrelated working-tree changes) and
   `git push origin main` immediately, as the edits are made. No merge
   step (already on `main`), no approval gate.

If the orchestrator is *itself* the agent that found the bug (no
subagent involved — the bug surfaced during the orchestrator's own
work), it spawns `workflow-improver` directly with the same protocol
and the same default: an in-scope, non-architectural, >=medium-confidence
gap is AUTO-FIXED in the background, not parked for greenlight. The
orchestrator does not get a stricter bar just because it noticed the
gap itself rather than receiving a candidate block.

## When the orchestrator suppresses the spawn

Suppression is the EXCEPTION (the default is auto-spawn — see above).
The exceptions below apply identically to formal `<!-- workflow-fix-
candidate v1 -->` blocks AND to prose follow-ups: a surfaced prose
follow-up does NOT get a stricter bar (or a looser one) than a formal
block — same rules, same defaults. The orchestrator logs the candidate
but skips the background fix ONLY in these cases:

- **Genuinely architectural / public-contract change.** The
  `proposed_change` would rename a status enum, change a marker schema
  or `task.py` subcommand / CLI contract, relocate an agent or skill
  file, or remove/restructure a subsystem (e.g. "remove the Codex
  ensemble"). These change a public interface other surfaces depend on,
  so they warrant Thomas's explicit greenlight rather than a background
  fix. Log the candidate; surface it to Thomas in the next chat turn.
  A change is NOT architectural just because it touches more than one
  line or one file — adding a guardrail step, tightening an instruction,
  fixing a contradiction between CLAUDE.md and an implementing file, or
  adding a missing field/note is in-scope auto-fix work, not an
  architectural decision.
- **Low-confidence speculative fix.** Either an explicit
  `confidence: low` on a formal block AND a speculative
  `proposed_change` ("maybe X should be changed but I'm not sure"), OR
  a prose follow-up whose wording is itself hedged / exploratory ("we
  might want to consider", "possibly", "not sure if this is right but")
  WITHOUT a specific file + specific change named. A prose follow-up
  that names a specific workflow-surface file + a specific concrete
  change is NOT low-confidence just because it lacks the formal block —
  treat it as medium. The marker is logged for the dashboard; no fix
  dispatched.

Two operational deferrals (NOT greenlight gates — the fix still happens,
just rerouted/queued):

- A `workflow-improver` is already running on the same `target_file` in
  this session. Queue the new candidate as a follow-up via
  `SendMessage` to the running agent rather than spawning a second one.
- The candidate's `target_file` is in the out-of-scope set (experiment
  code, `tasks/`, etc.). The orchestrator logs the candidate AND posts a
  brief note in the marker about the misclassification so the emitting
  agent's pattern can be corrected; no fix is dispatched because the
  target is out of scope by definition.

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
schema, a durable file-based trace. (The dashboard does not yet read
this fallback file — it surfaces only per-task `events.jsonl` markers;
homepage rendering of the fallback is unimplemented.)

## Anti-patterns

| Don't | Do |
|---|---|
| Subagent spawns `workflow-improver` itself | Surface the candidate (block or prose); orchestrator spawns |
| Emit a candidate for an experiment-code bug | Route to `implementer` / `experiment-implementer` |
| Emit ≥2 formal `<!-- workflow-fix-candidate v1 -->` blocks per run | Pick one for the block; list the rest as prose follow-ups (orchestrator auto-spawns each) |
| Emit `confidence: high` without a concrete diff_sketch | Sketch the actual lines; if you can't, drop to `medium` or skip |
| Wait for `workflow-improver` before continuing | Background-spawn; current task continues immediately |
| Emit a candidate against `src/`, `configs/`, `tasks/` | Out of scope — fix belongs elsewhere |
| Park an in-scope, non-architectural, >=medium-confidence gap for greenlight | Auto-spawn `workflow-improver` in the background; greenlight only for architectural / public-contract or low-confidence fixes |
| Orchestrator surfaces an agent's "Follow-ups (orchestrator should consider)" section to Thomas as a chat note asking "should I apply these?" | Treat each in-scope, non-architectural follow-up as a synthesized candidate and auto-spawn `workflow-improver` for it in the background; do NOT ask |
| Drop a prose follow-up because it lacked the formal block tags | Prose follow-ups trigger the same auto-spawn default as formal blocks; synthesize a candidate from the prose and dispatch |
| Hold prose follow-ups back hoping they'll surface "on the next pass" | List every concrete in-scope follow-up the agent found; the orchestrator auto-spawns each in parallel |

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

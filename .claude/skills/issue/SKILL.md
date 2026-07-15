---
name: issue
description: >
  End-to-end task-workflow orchestrator for experiments and code changes.
  Takes a task number (`<N>` = the integer that names `tasks/<status>/<N>/`),
  reads state from `body.md` frontmatter + `events.jsonl` markers under
  `tasks/<status>/<N>/`, and dispatches the next action (clarify ->
  adversarial-planner -> approval -> worktree + dispatch specialist ->
  preflight -> run -> analyzer -> humanize-loop (clean-result prose) ->
  free-analysis-followup-autorun (if any) ->
  clean-result-critic -> test-verdict -> auto-complete).
  clean-result-critic PASS (or test-verdict PASS for
  code-change paths like type:infra / type:batch / type:analysis /
  type:survey) auto-advances the task to `completed` on the local file
  layout. For experiments, the source task parks at
  `awaiting_promotion` and the user manually promotes the clean-result
  via `task.py promote <N> useful|not-useful` before auto-complete fires.
  Tasks stay on disk and are NEVER deleted. Idempotent and resumable:
  re-invoking on the same task picks up where it left off.
user_invocable: true
---

# Issue-Driven Workflow

## Scope & Boundaries

**Owns:** the full task lifecycle — clarify -> adversarial-planner -> approval -> worktree -> dispatch -> preflight -> run -> analyze -> review -> auto-complete.

**Invokes:** `experiment-runner` (run step), `adversarial-planner` (plan step), specialist agents (experimenter / implementer / experiment-implementer / analyzer / clean-result-critic / interpretation-critic / code-reviewer).

**Does NOT own:** proposing new experiments (-> `experiment-proposer`) or overnight queue orchestration (-> `auto-experiment-runner`).

---

Invoke as `/issue <N>` or `/issue <N> --resume`. The skill is the entry point from
a filed task to a fully-executed, reviewed experiment or code change.

**Guiding principle:** all durable state lives in plain files in the repo
(the `body.md` frontmatter + the append-only `events.jsonl` log under
`tasks/<status>/<N>/`). The local filesystem IS the source of truth. You
can close the terminal at any step and `/issue <N>` picks up cleanly.

## State backend

All durable state lives in plain files in the repo:

```
tasks/REGISTRY.json              # tiny index: id -> current folder path
tasks/<status>/<N>/
  body.md                        # YAML frontmatter + markdown body
  events.jsonl                   # append-only `epm:*` markers (resume log)
  comments.jsonl                 # mentor comments + Claude replies
  plans/v{K}.md, plan.md         # plan revisions + symlink to latest
  artifacts/                     # figures, html artifacts, drafts
  original-body.md               # snapshot before clean-result promotion
```

- **Status** is the parent folder name. Status transition = atomic `git mv`
  + commit. The folder is the single source of truth — there is no
  `meta.status` field. Allowed values: see `workflow.yaml § statuses`.
- **Marker = one row appended to `events.jsonl`** in the task's current
  folder. Same `epm:*` shape we've always used; payload is one JSON object
  per line.
- **Plan body** is cached at `tasks/<status>/<N>/plans/plan.md` (symlink
  to the latest `plans/v<K>.md`); subagent briefs always pass the
  symlink path so they read the freshest version.

`<N>` is the task number — the integer that names the per-task folder
under `tasks/<status>/<N>/`. It is **not** any external tracker number.
External tracker records (GitHub issues) are historical evidence only
and must never be used as workflow state.

Read and mutate state ONLY through `scripts/task.py`. It holds an exclusive
`flock` on `~/.task-workflow/lock` for every mutation, writes one git
commit per operation, and is the only writer to these files (the web
dashboard only appends to `comments.jsonl`). No HTTP, no auth token, no
remote database.

Useful operations:

```bash
uv run python scripts/task.py view <N>                       # show body + recent events
uv run python scripts/task.py view <N> --json                # machine-readable (body + last events)
uv run python scripts/task.py latest-marker <N>              # "where do I resume" query
uv run python scripts/task.py list-by-status --status running
uv run python scripts/task.py set-status <N> <status> --note '...'
uv run python scripts/task.py post-marker <N> epm:plan --note '...body...'
uv run python scripts/task.py set-body <N> --file /tmp/body.md
uv run python scripts/task.py set-title <N> "New title"
uv run python scripts/task.py add-tag <N> <tag>
uv run python scripts/task.py remove-tag <N> <tag>
uv run python scripts/task.py new-plan-version <N> --file /tmp/plan.md
uv run python scripts/task.py set-clean-result <N>           # flips body frontmatter has_clean_result=true
uv run python scripts/task.py promote <N> useful|not-useful  # user-only; flips classification
uv run python scripts/task.py find <N>                       # print current folder path
```

Display URL for a task: `https://eps.superkaiba.com/tasks/<N>` (the
planned EPS dashboard; the substrate is local files until that ships).
The local source of truth is always the on-disk folder
`tasks/<status>/<N>/`.

## Status convention

The status name IS the parent folder under `tasks/`. The canonical
enumeration of allowed values and their meaning lives in
(see workflow.yaml § statuses). The 12 happy-path values used by `/issue` are:

| Status                | Meaning |
|-----------------------|---------|
| `proposed`            | Filed but not yet triaged. User files tasks here. |
| `planning`            | Adversarial-planner is running. |
| `plan_pending`        | User action: approve plan to advance. |
| `approved`            | Plan approved, dispatch pending. |
| `awaiting_approval`   | Legacy alias for `plan_pending`. |
| `running`             | All active-phase work between approval and clean-result-critic-PASS rolls up here (implementing, code-reviewing, testing, training, uploading). The latest `epm:*` row tells you which sub-phase. |
| `verifying`           | Upload-verifier running. |
| `interpreting`        | Analyzer drafting the clean-result body. |
| `reviewing`           | Clean-result-critic running (the retired `reviewer` step's role was absorbed by Lens 7 of the critic). |
| `awaiting_promotion`  | User action: review clean-result draft and promote to useful / not-useful via `task.py promote`. |
| `blocked`             | Stuck / paused; resolve dependency. |
| `completed`           | Terminal happy path. Sticky — `has_clean_result=true` is preserved in body frontmatter. |
| `archived`            | Terminal sad path (duplicate / won't-fix / abandoned). Set explicitly. |

For follow-ups, the parent->child relationship lives in the child's
`body.md` YAML frontmatter as `parent_id: <N>`. Parents whose own work is
done but with at least one open child sit at `followups_running` (the
legacy children-in-flight semantics; see Step 10 step 5) with
`has_clean_result=true`; child discovery is by frontmatter scan (see
Step 10 step 4 below). Child tasks are ONLY for
`question_relation: substantially-different` follow-ups — a follow-up
that answers the SAME question as this task's Goal never creates a
child; it re-enters THIS task via the same-issue follow-up loop
(Step 9b § Same-issue follow-up loop), which holds the task at
`followups_running` (tag `followup-auto` | `followup-manual`) for the
round. A `question_relation: same` follow-up estimated at `< 20` GPU-h
auto-runs through that loop in BOTH interactive and autonomous sessions
(Step 9b cheap-band block; the 0-GPU floor runs inline at Step 9a-ter);
only the EXPENSIVE (`>= 20` GPU-h) `auto_run: yes` `same` band is
autonomous-only. In autonomous sessions, `substantially-different`
`auto_run: yes` proposals are FILED as `proposed` children for manual
triage only — never auto-spawned as sessions.

The skill moves status in exactly five places:

1. **Step 1 (clarifier "All clear"):** `proposed` -> `planning`.
2. **Step 9a (analyzer drafts clean-result IN PLACE):** the analyzer
   snapshots the prior body to `original-body.md`, replaces `body.md`
   with the polished write-up via `task.py set-body --file`, sets
   `has_clean_result=true` via `task.py set-clean-result`, and moves
   status to `awaiting_promotion` (the child runs/classification field
   stays `pending`).
3. **Step 9b (user promotes draft):** user runs
   `uv run python scripts/task.py promote <N> useful|not-useful` (or
   uses the dashboard once it ships). The task flips to `completed`
   with the chosen `classification`. The user then re-enters
   `/issue <N>` so Step 10 fires. Promotion is **user-only** — no agent
   or automation may flip `classification` without explicit user
   invocation.
4. **Step 10 (auto-complete):** source task -> `completed`.
5. **Same-issue follow-up re-entry (Step 9b § Same-issue follow-up
   loop / Step 0 followup-scope dispatch):** a task at `interpreting` /
   `reviewing` / `awaiting_promotion` / `completed` carrying an unrun
   `epm:followup-scope` moves to `followups_running` (tagged
   `followup-auto` | `followup-manual` by initiation mode) and HOLDS
   that status while executing a `question_relation: same`
   follow-up ON this issue, then re-parks at `awaiting_promotion`.
   `has_clean_result` stays sticky across the re-entry; a
   previously-promoted task re-parks and the user re-promotes.

Between those, intermediate transitions (`approved` -> `running` ->
`verifying` -> `interpreting` -> `reviewing` -> `awaiting_promotion`)
advance automatically as each step completes. Each transition appends a
row to `events.jsonl` with `marker_type = epm:*` so the agent can resume
where it left off after a context reset.

## Companion files

- `markers.md` — marker taxonomy (source of truth for state parsing). The
  per-kind table is auto-generated from `workflow.yaml § markers`.
- `clarifier.md` — clarifying-question prompts per task type.
- `templates/` — plan / results / analysis body templates.
- `failure_patterns.md` — human-readable mirror of
  `scripts/failure_classifier.py` (regex patterns for routing
  `epm:failure` markers). The script is authoritative at runtime.

Read these on first invocation of the skill in a session.

---

## Auto-continuation policy

Auto-continue through every step EXCEPT the gates declared in
(see workflow.yaml § gates) (see CLAUDE.md "Auto-continuation policy" for
the prose summary). The full enumeration — inline gates + park-and-wait
gate + conditional gate — is canonical in workflow.yaml. Anywhere else
that an assumption needs to be made, STATE the assumption inline (one
line, prefixed `Assumption:`) and proceed; do NOT pause to ask.

**Exceptions that override auto-continuation:** subagent halt conditions
(see workflow.yaml § subagent_halt_conditions) and STATE-TO-`status:blocked`
criteria (see workflow.yaml § halt_criteria). When any of those fire,
EXIT regardless of the auto-continuation rule.

**Resourceful-first before any non-gate ask.** Before raising a non-gate
`AskUserQuestion` <!-- example: anti-pattern --> about a design fork
(reuse-vs-retrain, which checkpoint, title options, "how should I
proceed"), FIRST sweep `tasks/` + HF Hub / WandB for the artifacts or
prior results that resolve it — exactly the resourceful-before-asking
posture of halt-criterion #1. Ask only once the investigation leaves a
genuine factual gap only the user can fill. (Incident 2026-06-01: a
reuse-vs-retrain ask got "look deeper at other issues first"; a
title-options ask got "show me the body first" — both rejected for
asking before exhausting the investigation.)

## The State Machine

State = the parent folder name under `tasks/` (i.e., the row in
(see workflow.yaml § statuses)). Transitions are enforced by this skill;
`events.jsonl` rows provide the detailed payload at each state.

Principle: every state is either "an agent is actively working" OR
"awaiting user input." Distinct folder names for each so a glance at
the directory layout tells you whether it's your turn.

```
proposed                                <- user has filed, clarifier hasn't run
  |-- (clarifier -> questions OR OK)
       |-- questions posted --> proposed (stays; awaiting user replies in comments.jsonl)
       |-- OK --> planning              <- adversarial-planner + consistency-checker (∥ Phase 2 critics; one union revise round)
                  |-- (plan posted + consistency PASS/WARN)
                     |--> plan_pending  <- AWAITING USER: approve?
                            |-- (user approve) --> approved
                                                  |-- (worktree + draft PR)
                                                     |--> running (implementing sub-phase)  <- experiment-implementer (type:experiment) OR implementer (type:infra/batch)
                                                            |-- (epm:experiment-implementation OR epm:results posted)
                                                               |--> running (code-reviewing sub-phase)  <- code-reviewer ensemble (Claude + Codex)
                                                                      |-- FAIL + count<5 --> running (implementing, v+1)
                                                                      |-- FAIL + count>=5 --> apply Step 5d cap-hit rule: strip → PASS+continue OR surface residual (autonomous+substantive: blocked; interactive: parked)
                                                                      |-- PASS + [type:experiment] --> running (workload sub-phase)  <- experimenter (pod ops + monitoring)
                                                                            |-- (epm:results posted)
                                                                               |--> verifying              <- upload-verifier ∥ analyzer first pass (held) ∥ methodology-writer early spawn
                                                                                      |-- (all artifacts verified, pod terminated; held interpretation published)
                                                                                         |--> interpreting  <- analyzer + interp-critic loop
                                                                                                |-- (interpretation refined, clean-result drafted in place)
                                                                                                   |--> reviewing  <- clean-result-critic final adversarial gate (Lens 7 absorbed retired reviewer)
                                                                                                          |-- PASS --> methodology-reference LATE JOIN (Step 9a-quater: secret gist + top-of-body **Methodology:** line + ## Reproducibility row; agent itself early-spawned at uploading; auto-continue) --> awaiting_promotion  <- AWAITING USER: promote clean-result
                                                                                                                        |-- (user promotes via task.py promote) -->
                                                                                                                              |-- open children w/ parent_id=<N> exist --> followups_running  <- legacy: waits for children (also held during same-issue follow-up rounds); re-invoke /issue <N> later
                                                                                                                              |-- no open children                  --> completed (+ follow-up proposer)
                                                                                                          |-- REVISE --> interpreting (revise)
                                                                      |-- PASS + [type:infra/batch/analysis/survey] --> test-verdict (inline) --> completed
```

Hot-fixes during the running (workload) sub-phase (experimenter agent):
small in-line fixes (<=10 lines, no logic change) get committed on the
issue branch and the run continues. Anything beyond that bar bounces
back to the running (implementing) sub-phase for a fresh
experiment-implementer + code-reviewer round before the experimenter
relaunches.

There is no user sign-off step. Clean-result-critic PASS (or
`epm:test-verdict` PASS for code-change paths) is the terminal gate;
completion is automatic. If the user disagrees with a `completed`
transition, they `set-status <N> blocked` to reopen it. The
"test-verdict gate" runs inline inside this skill (Step 9c) — there is no
separate `tester` agent.

**Active vs awaiting-user states** (auto-generated from
(see workflow.yaml § statuses). Do NOT edit inside the fence; run
`uv run python scripts/workflow_lint.py --emit-tables` to regenerate
after a YAML edit):

<!-- workflow.yaml: AUTO-GENERATED (active-vs-awaiting) -->
| State | Who's working | User action needed? |
|-------|---------------|---------------------|
| `on_hold` | Parked indefinitely ("on hold for now"); explicitly set aside, NOT in the active proposed queue. Excluded from auto-dispatch / the clarifier. Revivable via task.py set-status <N> proposed. | **yes** |
| `proposed` | User has filed; clarifier hasn't run. | no |
| `clarifying` | Clarifier asked questions; awaiting user answers in comments.jsonl. | **yes** |
| `planning` | Adversarial-planner is running. | no |
| `plan_pending` | Plan posted; awaiting user approval (task.py set-status N approved). | **yes** |
| `approved` | Plan approved; skill is creating worktree and dispatching the specialist. | no |
| `awaiting_approval` | Legacy alias for plan_pending; preserved for back-compat with older bodies. | **yes** |
| `queued` | Approved task awaiting an available pod or implementer slot. | no |
| `implementing` | Implementer or experiment-implementer is writing code. | no |
| `code_reviewing` | code-reviewer ensemble is reviewing the diff. | no |
| `testing` | Inline test-suite step (Step 9c, code-change paths only). | no |
| `running` | experimenter is running the workload on a pod. | no |
| `verifying` | upload-verifier is checking that artifacts landed on HF Hub / WandB / git. (There is NO `uploading` status — the whole upload-verification phase runs at `verifying`; task.py rejects `uploading`.) The analyzer first pass (HOLD-marker mode) + methodology-writer may pre-compute in the background (Step 8 results-landed parallel spawn) — no epm:interpretation is published before upload-verification PASS. | no |
| `interpreting` | analyzer + interpretation-critic + clean-result-critic loops are running. | no |
| `reviewing` | Final adversarial review pass (clean-result-critic Lens 7 absorbed the retired reviewer step). | no |
| `under_review` | Legacy alias of reviewing; do not introduce new uses. | no |
| `awaiting_promotion` | User action: promote clean-result via task.py promote <N> useful|not-useful. | **yes** |
| `followups_running` | A same-issue follow-up round is executing on this task (tag followup-auto | followup-manual); legacy: parent complete with parent_id children still in flight. | no |
| `shared` | Shared infra / utility task not tied to a single experiment. | no |
| `blocked` | Aborted or stuck; awaiting user triage. | **yes** |
| `completed` | Terminal: clean-result promoted OR code change shipped + reviewed. | no |
| `done_experiment` | Legacy terminal alias used by older bodies; equivalent to completed. | no |
| `done_impl` | Legacy terminal alias used by older bodies; equivalent to completed. | no |
| `failed` | Workload crashed and could not be resumed; awaiting user triage. | **yes** |
| `cancelled` | User cancelled before completion. | no |
| `archived` | Closed long ago / no longer relevant; sticky regardless of has_clean_result. | no |
<!-- /workflow.yaml: AUTO-GENERATED -->

The two user-gated states in the active lifecycle are `plan_pending` (plan
approval) and `awaiting_promotion` (clean-result promotion). `blocked` also
needs user attention but represents a stalled state. Everything between is
automatic, short of a `task.py set-status <N> blocked` override.

Abort affordance: any state, user runs `task.py set-status <N> blocked`
-> skill posts abort request via `epm:abort`, watcher kills run if one
exists.

User pause affordance ("pause <N>", "hold <N>", "put <N> on hold"): a user
pause is a DURABLE park, never a prose-only marker. The session driving <N>
(or the PM/chat session receiving the directive) executes IN THIS ORDER —
the order is load-bearing: the `on_hold` park is the COMMIT POINT and comes
LAST. Since #980 the watcher's pod-safety pass auto-stops an
`on_hold`+RUNNING RunPod pod (`on_hold` is in the watcher's
`POD_SAFETY_AUTO_STOP` set) after the 2-consecutive-miss guard (~20-30 min
at the 10-min cron), so a crash inside the pause window bills for minutes,
not forever — but the teardown-first ORDER stays load-bearing: the backstop
is slow and covers RunPod MANAGED pods only (a GCP instance still relies on
its `--max-run-duration` fence / `dispatch_issue.py finalize`), whereas a
crash BEFORE the park leaves the task at its prior ACTIVE status,
where orphan-respawn remains a loud backstop — and for `POD_ACTIVE` statuses
(`approved`/`running`/`verifying`/`followups_running`) the pod-active-stale
alert fires too:

1. Run CRON-TEARDOWN (§ CRON-TEARDOWN procedure — the `/issue-tick`
   backstop cron must not outlive the pause), then Step 8-bis: stop any
   RUNNING pod (`uv run python scripts/pod.py stop --issue <N>`; volume
   preserved until the daily stale-pod audit terminates EXITED pods >24h —
   add the `keep-running` tag if the volume must outlive that) and post
   `epm:pod-stopped v1`. A GCP instance is outside Step 8-bis's reach — it
   self-terminates at its `--max-run-duration` fence, or run
   `uv run python scripts/dispatch_issue.py finalize --issue <N>`. The
   autonomous "never stop a pod to PARK" ban does not apply here: a user
   pause is a user directive, not an autonomous park.
2. LAST — the commit point: `uv run python scripts/task.py set-status <N>
   on_hold --note "USER PAUSE (verbatim: '<user words>');
   paused_from=<prior status>; resume: task.py set-status <N>
   <prior status> && spawn_session.py spawn-issue --issue <N> --auto"`.
   `on_hold` is in the watcher PARK set (`autonomous_session_watch.py`
   `PARK`): no orphan-respawn, no auto-dispatch, registration kept. The
   transition is legal from EVERY status WITHOUT `--force-followup-exit`
   (`on_hold` is not in `FOLLOWUP_HELD_BLOCKED_STATUSES`); pausing a
   `followups_running` round abandons that round's in-flight subagents —
   record `paused_from` so resume restores the held status.
3. End the turn. Do NOT leave status at an ACTIVE value with a prose
   hold note — the watcher's orphan-respawn pass cannot parse prose and
   will respawn against the hold (incident #816, 2026-07-02).

Resume is user-greenlight ONLY: `task.py set-status <N> <paused_from>` (or
`proposed` for a full re-triage), then `spawn_session.py spawn-issue
--issue <N> --auto`; the events.jsonl markers re-enter the loop at the same
point (a resumed `followups_running` round may re-park at
`awaiting_promotion` via the round-complete re-park — Step 9b § Same-issue
follow-up loop — expected, not a bug). Distinct mechanism: the
`.paused-takeover-*` registration sentinel
(`.claude/rules/background-automation.md` § Deliberate session takeover) is
a short-TTL (~6h, FAIL-OPEN) session-TAKEOVER shield — it does NOT
implement a user pause. Known carve-out: a PROGRAM daemon that hardcodes
task revival (e.g. `run_program_orchestrator.sh`, the #660 leakage program,
revives its four pinned tasks from `on_hold`) can override the park for
exactly those pinned ids — pausing a program-pinned task additionally
requires the program's STOP sentinel.

---

## Orchestration Procedure

When invoked, ALWAYS follow this order. Skip only what the state dictates.

**Chat title updates (single-source-of-truth canonical string).** The
session's phone title AND the terminal/dashboard progress column (read
from `~/.eps-autonomous/session_progress.json` by `happy-ls` and the
`/sessions` page) display the SAME canonical string for /issue sessions.
This is enforced by routing every title set through one helper:

```
uv run python scripts/session_progress_report.py --issue <N> --step "<step>"
```

The helper (a) builds the canonical string via
`session_progress_report.build_progress_string(issue, slug, step)` — the
ONLY place the format lives — (b) writes a self-report file at
`~/.eps-autonomous/issue-progress/<N>.json` (atomic temp+rename) with the
canonical text + UTC `ts`, and (c) PRINTS the canonical string to stdout.
Capture that stdout and pass it verbatim to `mcp__happy__change_title`.
The 5-minute `session_summarize.py` cron reads the self-report first and
reuses its `text` as the cache `summary` (`source="self"`) when fresh,
skipping the Haiku call entirely — so the dashboard's progress column is
byte-identical to the phone title.

**Cadence.** Set the title via the helper at:
(a) first invocation (Step 0), as soon as the task slug is known —
    `--step "<status>"` (or `"launching"` before status is known) — so the
    Happy phone session list is self-documenting from spawn;
(b) every status transition;
(c) every Step 6d.2 polling-loop tick (orchestrator re-invocation) and
    every cron-backstop re-invocation of `/issue <N>` — so the dashboard
    stays current even on a long idle stretch;
(d) when an `epm:follow-ups` marker is posted, when the clean-result
    draft is finalized (Step 9a end), and when the merge prompt fires
    (Step 10d).

Format (built by `build_progress_string`):
```
#<N> <slug> · <step>
```
Hard-capped to ~78 chars. Slug pre-clipped to 45 chars. If the joined
string would exceed the cap, the STEP is trimmed with a trailing `…`;
the `#<N> <slug>` head stays intact (the head is the part the user uses
to find the row).

Examples:
- `#226 wire /issue auto-title into session · implementing`
- `#226 wire /issue auto-title into session · code-review FAIL round 2`
- `#137 persona collapse under EM · awaiting promotion`
- `#479 conditional Stage-2 anchor-knob sweep · launching` (Step 0)

Orchestrator pseudocode:

```python
def set_title(issue: int, step: str) -> None:
    """Build the canonical string, write the self-report, and set the
    phone title. The title / self-report path is OBSERVABILITY
    infrastructure, not load-bearing — soft-fail on both the helper
    invocation AND change_title so a stale dashboard never crashes the
    /issue pipeline. Surface the error in the current turn's output so
    a regression is visible, then continue."""
    try:
        canonical = run_bash(
            f'uv run python scripts/session_progress_report.py --issue {issue} '
            f'--step {shlex.quote(step)}'
        ).strip()
    except Exception as e:
        # Helper failed (missing task, broken task.py import, disk full,
        # etc.). Log and continue — a stale self-report is an
        # observability regression, NOT a reason to abort the pipeline.
        log(f"set_title: session_progress_report.py failed: {e}; continuing")
        return
    try:
        mcp__happy__change_title({"title": canonical})
    except Exception:
        # Cosmetic; the self-report file write already happened, so the
        # dashboard / happy-ls still show the right string. The phone
        # title just doesn't update this tick.
        pass
```

**Status-transition titles** simply pass the new status as the step
(`set_title(N, "awaiting_promotion")`). For richer end-of-pipeline cues
(follow-ups posted, clean-result finalized) the orchestrator can pass a
short composite step string (`"awaiting promotion · followups #240, #241"`)
— `build_progress_string` will trim it to fit the cap.

**Autonomous session behavior (`EPM_AUTONOMOUS_SESSION=1`).** When this env
var is set (the session was spawned via `spawn_session.py spawn-issue
--auto`), the orchestrator runs to completion with no human at the keyboard:

- **Forbidden: presenting options to the user in ANY form. Decide AND
  EXECUTE in the same turn.** With `EPM_AUTONOMOUS_SESSION=1` there is no
  human at the keyboard, so an optional decision is YOURS to make. Banned
  output patterns (this list is exhaustive — none of these may appear in an
  autonomous turn):
  <!-- example: anti-pattern -->
  - A `AskUserQuestion` tool call (the PreToolUse hook in
    `.claude/settings.json` hard-blocks this — backstop ONLY for the tool
    path; the text-menu failure mode below cannot be intercepted by a hook).
  - A numbered or bulleted choice list in markdown ("Option 1: …",
    "Option A: …", "- continue-as-planned", "- pivot-to-X").
  - "which would you prefer?", "let me know how to proceed", "please
    confirm before I…", "thoughts?", "your call", "should I…", "would you
    like me to…", "what would you like to do?".
  - "Both? or do you read it differently?", "two things I'd do, want your
    call?", the interactive two-path escalation framing (CLAUDE.md "frame
    exactly TWO paths" is for an INTERACTIVE session — it does NOT apply
    here).
  - Ending the turn with a trailing question to the user.
  - Stating a `Decision:` and then ending the turn without executing it
    (i.e. posting the resolved marker / spawning the resolved subagent /
    advancing status). "Decide and continue" means decide AND execute, in
    the same turn.

  Real incident (tasks #503/#504/#505, 2026-06-05): three autonomous Happy
  sessions all printed a plain-text "Option 1 vs Option 2 — which would you
  prefer?" markdown menu and stopped the turn, leaving the orchestrator
  blocked indefinitely on a user reply that the user did not even know was
  pending. That is the exact banned behavior this section enforces. The
  PreToolUse hook could not catch it because no tool call was ever made —
  only the prose was emitted. The dominant failure mode is
  text-menu-end-of-turn; only this section's prose prevents it.

  Correct shape: at ANY fork that is not one of the two hard gates below
  or a hard halt-criterion — choosing among proposed follow-ups, "should I
  ALSO run X", correcting a now-wrong marker / record you posted, picking
  between competing interpretations of a result, which secondary analysis
  or probe to run — pick the option that maximizes information-gain-per-
  GPU-hour toward the task `## Goal` (break a genuine tie toward the
  lower-cost / safer / record-correcting action), state it in one line
  (`Decision: <X> because <reason>`), AND EXECUTE THE RESOLVED ACTION IN
  THE SAME TURN: post the matching `epm:<marker>`, spawn the next
  subagent, run the `task.py set-status` transition — whatever the
  resolved branch's normal continuation is. The autonomous session waits
  on the user at the two gates and nowhere else.
- **Conditional gates auto-resolve when `EPM_AUTONOMOUS_SESSION=1` —
  never raise `AskUserQuestion`, never print a text menu, always execute
  the resolved action in the same turn.** The conditional gates in
  (see workflow.yaml § gates.conditional) — `whack_a_mole_pivot` (id 11),
  `compute_deviation_resolution` (id 12), `concern_deferral_request` (id 15),
  `tdd_gate` (id 8), `experiment_goal_refine` (id 9), `living_docs_update`
  (id 13), `fact_candidates` (id 14) — present two-option escalations or
  binary confirm/reject choices. In Interactive mode they raise
  `AskUserQuestion`; with `EPM_AUTONOMOUS_SESSION=1` set they MUST
  auto-resolve AND execute the resolved action in this same turn. The
  resolution rule per gate:
  - `whack_a_mole_pivot` → pick `pivot-to-<X>` if the implementer's report
    named a canonical alternative AND the next round on the current path
    would burn >2× the cost of the pivot; else `continue-as-planned`. State
    `Decision: <choice> because <reason>` and EXECUTE the resolved action
    in this same turn (on `pivot-to-<X>`: `task.py set-status <N> planning`
    + re-invoke `/adversarial-planner` with the pivot scope; mid same-issue
    follow-up round, SKIP the `set-status` — status-hold rule, Step 9b — and
    just re-invoke the planner with the status held; on
    `continue-as-planned`: continue to Step 6); do NOT state the Decision
    and then end the turn.
  - `compute_deviation_resolution` → (reachable only after pivot_criteria
    auto-action 0 is resolved: the vectorize-first fix round ran, or a
    negative signature finding is recorded — Step 5.bis(a); a marker
    that arrived pre-resolved without a lever-0 record is UNRESOLVED
    and routes through step 0 first). Pick
    `accept_descope_to_<X>_with_caveats` if any descope dimension
    preserves majority statistical power (≥0.6 of the planned cells);
    else `continue_as_is` and quote the projected ratio inline — at
    ratio ≥ 5×, `continue_as_is` additionally requires the recorded
    quantified clause-0c finding (`flop_bound_finding:` on the marker,
    or a `signature_check: negative` record meeting the 0c bar). If it
    is missing at ≥ 5×: when NO `action: vectorize_fix_round` exists in
    the component's chain, do NOT pick `continue_as_is` — execute the
    step-0 vectorize fix round; when the fix round HAS run, NEVER
    re-run step 0 (once-per-component loop guard) — its post-fix
    re-post's residual classification is the 0c record (the fix-round
    brief requires it); if that re-post omitted the classification,
    obtain ONE corrective re-post recording it (no second fix round),
    then resolve. State `Decision: <choice> because <reason>` and
    EXECUTE the resolved action in this same turn (post
    `epm:compute-deviation v2` with the chosen `action:` + advance to
    Step 5.bis(b)); do NOT state the Decision and then end the turn.
  - `concern_deferral_request` → bounce to implementer for one more round
    targeting the open CONCERN(s); never defer in autonomous mode (deferral
    is a user-rationale-required action by spec). State
    `Decision: bounce to implementer (concern_id=<id>) — autonomous mode
    never defers` and EXECUTE the bounce in this same turn (spawn the
    `experiment-implementer` / `implementer` agent with a brief targeting
    the open concern_id); do NOT state the Decision and then end the turn.
  <!-- gate: gates.tdd_gate -->
  - `tdd_gate` → no `AskUserQuestion` at this site (it's event-driven —
    the implementer posts `epm:proposed-tests v<n>` and exits; the resume
    signal is `epm:approve-tests` posted via `task.py post-marker`). In
    autonomous mode, auto-post `epm:approve-tests` IF the proposed-tests
    body lists ≥1 test per acceptance criterion from the original task
    body, AND EXECUTE the resume in this same turn (spawn the implementer
    with `tdd_approved=true`); else bounce to the implementer with a
    pointer to the gap (also same-turn execution). If still missing after
    one bounce, post `epm:failure v1 failure_class: code` and set
    `status:blocked` (halt-criterion #5).
  - `experiment_goal_refine` → autonomous mode does NOT refine the Goal
    mid-run; skip (do not raise the ask, do not refine). EXECUTE the
    skip by continuing to the next step in this same turn; do NOT state
    "Decision: skip" and then end the turn.
  - `living_docs_update` → DO NOT auto-confirm. Living-docs mutations
    are user-only by spec (workflow.yaml § gates.living_docs_update:
    "Every living-docs mutation is user-confirmed: the agent proposes,
    the user confirms/edits/rejects, nothing auto-applies"). In
    autonomous mode the proposal is already posted as
    `epm:living-docs-proposed v1`; park it for the user (the experiment
    is already `completed`, no lifecycle blocks on this) and EXECUTE the
    continuation to Step 10d in this same turn; do NOT print the diff to
    chat as a menu, do NOT end the turn waiting on user confirmation
    (the marker is the surface; the nightly /daily living-docs backstop + a later `/issue <N>`
    re-invocation reconcile it).
  - `fact_candidates` → pick the candidate `id` with the median per-token
    log-prob (the middle of the band the plan filtered by). State
    `Decision: id=<X> (median log-prob in band)` and EXECUTE the resume
    in this same turn (post `epm:fact-pick v1` with `id: <X>` + resume
    the polling loop); do NOT state the Decision and then end the turn.

  <!-- example: anti-pattern -->
  The PreToolUse hook on `AskUserQuestion` (`.claude/settings.json`) is a
  backstop for the TOOL case ONLY — when `EPM_AUTONOMOUS_SESSION=1` it
  cannot intercept plain text output. The dominant failure mode is
  text-menu-end-of-turn (incidents #503/#504/#505); only THIS prose
  enforces it. Autonomous mode must DECIDE AND EXECUTE THE RESOLVED
  ACTION IN THE SAME TURN — stating `Decision: <X>` and ending the turn
  is itself the failure, regardless of whether a tool call was made.
- **Autonomous mode overrides `factual_question_only_user_knows`
  (workflow.yaml `halt_criteria id=4` / CLAUDE.md STATE-TO-`blocked` bullet #1)
  for taste / scope / design-preference / pivot calls.** Those surfaces list
  "priority, taste, scope, design preference between valid paths" as a valid
  block reason. In `EPM_AUTONOMOUS_SESSION=1` mode this sub-case does NOT
  apply: there is no human to escalate to, so a taste / scope / design-preference
  / "which valid path?" call is NEVER a block reason. Pick the option that
  maximizes information-gain-per-GPU-hour toward the task `## Goal` (tie-break:
  lower-cost / safer / record-correcting), post
  `Decision: <X> because <reason>`, and EXECUTE the resolved action in the SAME
  turn. The only residue of `factual_question_only_user_knows` that survives
  in autonomous mode is a factual gap the user UNIQUELY holds (an account
  credential, an external decision the user already promised to make, a fact
  only the user can supply) AND that is NOT itself a taste / scope / design
  call. Real incident the candidate surfaced (2026-06-07, tasks
  #503/#504/#506/#509): multiple `--auto` sessions parked overnight "awaiting
  user decision on Phase 2 path forward" — exactly the banned regression this
  clause closes.
- **A debugging wall is a strategy-pivot, not a block.** If implementation /
  smoke-run / reviewer-loop work hits a wall the session cannot immediately crack,
  spawn `experiment-implementer` (or the analogous fixer) on a different angle,
  re-invoke `/adversarial-planner` with explicit pivot scope, swap a model /
  pod intent / framing, or drop the offending domain — see workflow.yaml §
  `pivot_criteria` for the canonical pivot actions. Set `status:blocked` ONLY
  after ~3 FUNDAMENTALLY different strategies (not 3 retries of the same one)
  have all FAILed AND no further autonomous angle exists. A bare reviewer FAIL,
  a single preflight crash, a 4th-round ensemble FAIL, or a smoke-run that
  surfaces a tractable bug are pivots, never blocks.
- **A self-defeating PLAN is a re-plan, not a recipe descope.** Distinct from
  the generic debugging-wall pivot above: when a subagent (the
  `experiment-implementer`, any reviewer in the loop, or a Statistics &
  Measurement lens REVISE from `critic` / `codex-critic`) reports that the
  PLAN ITSELF is the defect — internally contradictory success / kill
  criteria, a jointly-unsatisfiable gate set (two kill-gates demand opposite
  signs on the same measurement at the same cell), or an explicit "needs
  plan amendment / cannot pick a science direction" verdict — the autonomous
  response is `task.py set-status <N> planning` + re-invoke
  `/adversarial-planner` with explicit pivot scope naming the contradiction
  verbatim. (Mid same-issue follow-up round, SKIP the `set-status` — the
  status-hold rule, Step 9b § Same-issue follow-up loop step 3, holds
  `followups_running`; just re-invoke the planner.) See workflow.yaml
  § `pivot_criteria.plan_contradiction_replan` for the canonical action
  shape.

  This is the `pivot-to-<X>` action for that specific signal — do NOT route
  it through the valid-fork "max-info-gain pick" decision rule above. A
  contradictory plan is not a valid fork. Three banned anti-patterns the
  autonomous session must NOT take (each was the actual #488 round-10
  regression):
  - **Do NOT descope a hyperparameter / recipe** (lr, LoRA rank, row count,
    epoch count, etc.) to dodge the contradiction. That papers over a plan
    bug with a recipe knob and lands in a two-sided dead-end where neither
    attempt resolves the underlying gate conflict — exactly the "attempt 1
    too strong, attempt 2 too weak → recipe family exhausted" false
    conclusion #488 reached.
  - **Do NOT silently pick** among the subagent's paper-over options as if
    it were a valid experimental fork. The max-info-gain pick rule applies
    to forks where every option is a coherent experiment; it does NOT
    apply to "the plan is self-defeating, pick a workaround."
  - **Do NOT park for the user.** There is no human to escalate to in
    autonomous mode; re-plan in the same session via the canonical pivot
    action.

  Count this as a strategy pivot for the ~3-pivots-before-block rule (use
  the existing `epm:strategy-pivot v<n>` marker convention — do NOT
  introduce a new marker kind). Block only after ~3 re-plans fail to yield
  a satisfiable design AND no further autonomous angle exists. The
  upstream defenses that prevent the contradiction from being shipped in
  the first place are `critic.md` + `codex-critic.md` Statistics &
  Measurement lens item 3 (decision-gate coherence) and `planner.md` §7
  gate-set minimality + joint-satisfiability self-check; this clause closes
  the loop on the execution side when a contradictory plan slips through
  the planner + critic ensemble anyway. Post-mortem trigger: task #488
  round 10 (2026-06-08).
- **Never stop a pod to PARK / await a user in autonomous mode.** `pod.py stop`
  to avoid idle-burn is allowed ONLY while work continues toward the Goal in
  the same session (e.g. stopping pod-N while the analyzer reads JSON from
  WandB/HF before the auto-terminate at Step 8). Stopping a pod with prose like
  "Pod-N stopped while awaiting user decision on …" is the banned regression
  this clause closes — it is the autonomous-mode equivalent of the text-menu
  end-of-turn failure. Forbidden in `EPM_AUTONOMOUS_SESSION=1`.
- **A FREE, no-data-loss path beats parking — take it and keep waiting.** When
  a free, no-data-loss continuation EXISTS, taking it is mandatory; parking to
  await the user or proposing a PAID rerun while that free path is available is
  the banned regression. Canonical case (tasks #658/#663, 2026-06-24): an
  in-SLA Anthropic Message Batch (submitted, not yet expired) SELF-HARVESTS for
  free at `expires_at` — the result is recoverable by polling the batch's own
  deadline-bounded poller (`explore_persona_space.eval.batch_judge`, which
  bounds the poll on `expires_at`), so the correct autonomous action is to keep
  the deadline-bounded poll running (end the turn; the bg-Bash poll chain / the
  45-min `/issue-tick` backstop re-wakes you) and harvest the free result — NOT
  to PARK with "await your call" and NOT to propose a paid re-submission. A paid
  rerun is justified ONLY when the cheaper free path is genuinely unavailable
  (the batch already expired with no result, the data is truly lost). This is
  the data-loss-aware twin of "never stop a pod to PARK": both forbid burning a
  turn (or money) on a user-park when continuing toward the Goal is free and
  loses nothing. Also route batch judging through the #663-hardened client
  rather than a hand-rolled `messages.batches.create` + deadline-less
  `while True ... sleep` poller — the client is what makes the free self-harvest
  automatic (enforced by `scripts/workflow_lint.py --check-batch-judge-client`).
- **Cost is gated ONLY at the plan-approval GPU-hour cap, never mid-run.** The
  ONLY cost gate in autonomous mode is the Step 2c `plan_pending` park when
  `gpu_hours_total > EPM_PLAN_AUTOAPPROVE_GPU_HOURS` (default 100). A running
  experiment is never paused mid-run on "this is getting expensive" grounds —
  no `max_budget_usd` SystemExit, no mid-run "should we keep going?" decision,
  no autonomous-side cost-based pivot to "park for user review." Per CLAUDE.md
  "Code Style" + `tests/test_no_dollar_budget_caps.py`, dollar-budget caps in
  experiment scripts are also forbidden at the code level — the same discipline
  applies to autonomous orchestration decisions. The plan-approval cap is the
  only legitimate spending gate.
- **Recompute incoming fleet-burn figures before acting on them.** When a
  received directive (a PM push-through brief, an `AUTONOMOUS PUSH-THROUGH`
  message, or any incoming text) cites a fleet-burn / $-per-hour figure to
  justify a cap or headroom decision, re-compute it fresh locally before
  acting on it. Pods churn between when the directive was written and when
  this session reads it; the cited number goes stale fast. The RunPod API
  is authoritative per CLAUDE.md § "Authority split"; use
  `current_account_hourly_burn()` from `scripts/runpod_api.py` (the same
  helper the provision cap-check uses; one-liner: `uv run python -c "import
  sys; sys.path.insert(0, 'scripts'); from runpod_api import
  current_account_hourly_burn; t, b = current_account_hourly_burn();
  print(f'${t:.2f}/hr'); [print(f'  {n:<22} ${r:6.2f}/hr') for n, r in b]"`).
  Proceed on the fresh number; if it differs materially from the cited
  figure, note the discrepancy in the marker / log line that records the
  decision (e.g. `directive cited ~$65/hr; live burn is $112.50/hr — acting
  on live value`). This is a sanity check on the input number, NOT a new
  cost gate (the rule above still holds — autonomous mode never adds a
  mid-run cost gate or block). Incident #506: the session correctly
  recomputed and caught a ~$47/hr stale figure on its own; encode that as
  the rule rather than relying on it to happen.
- **Push through bugs; do not block on recoverable failures.** Apply
  CLAUDE.md "Push through bugs in recovery mode" + the halt-criteria
  literally: preflight failures, TP/Ray/env-var hiccups, transient infra,
  a single FAILed reviewer round, etc. are fixed and retried in-loop. A
  bare reviewer FAIL triggers a strategy pivot, not a block. The autonomous
  hard halt-criteria that survive are strictly: outside-worktree / irreversible
  mutation (halt-criterion #1 in workflow.yaml); public-API-contract change
  (#2); a subagent BLOCKER with explicit `needs-user` (#3); the narrow residue
  of `factual_question_only_user_knows` (#4) per the override above — i.e.
  a uniquely-user-held fact that is NOT a taste / scope / design call;
  completion-audit incomplete (#5); concern_unresolved (#6, after autonomous
  options exhausted). Everything else auto-continues or pivots.
- **The only stop points are the two real gates:** the Step 2c plan-approval
  cap (park at `plan_pending` only when est. GPU-hours exceed
  `EPM_PLAN_AUTOAPPROVE_GPU_HOURS`, else auto-approve), and
  `awaiting_promotion` (always a human gate). Everything else auto-continues.
- **Every follow-up proposal passes a redundancy screen before it
  routes (Step 9b § Follow-up value-critique, subroutine VC).** Once the
  `follow-up-proposer` posts `epm:follow-ups v1`, the orchestrator runs a
  SINGLE-PASS ensemble — `follow-up-critic` (Claude) + `codex-follow-up-critic`
  (Codex) + `reconciler` on disagreement (the 5th doubled review site;
  workflow.yaml § ensemble_review, `single_pass: true`) — over the whole
  proposal set ONCE per park, BEFORE any proposal routes to the cheap-band
  auto-run, the autonomous same-issue loop, the autonomous child-filing
  path, or the interactive Step 10b pick. The bar is REDUNDANCY ONLY (a
  proposal duplicating an existing experiment task, a settled open
  question, or a higher-ranked sibling in this round) — NOT info-gain /
  worth; a low-but-novel follow-up PASSes. NOTHING is dropped:
  `not-redundant` proposals proceed through the EXISTING routing
  unchanged; `redundant` proposals are SAVED as new `on_hold` tasks
  (`epm:followup-parked-redundant v1`, revivable via `set-status <M>
  proposed`), never auto-run and never silently discarded. The rationale
  is persisted both ways (the `epm:followup-value-critique` markers + the
  parked task's `## Value critique` body section). User-requested
  follow-ups (`source: user-chat` — the Step 0 followup-scope dispatch)
  are NOT screened: the user already decided to run them. See Step 9b
  "Follow-up value-critique".
- **Auto-run cheap (`< 20` GPU-h) same-question follow-ups in BOTH
  modes at Step 9b.** Standing directive (2026-06-13, raised 5→20 GPU-h
  2026-06-24): a follow-up that
  is `0` GPU-h or `< 20` GPU-h just runs and folds into the SAME issue,
  automatically, in interactive AND autonomous sessions — no human pick,
  no `headline_affecting` gate. The 0-GPU floor runs inline at Step
  9a-ter (free-analysis); the GPU-backed `0 < est_gpu_hours < 20` band
  runs at the Step 9b **cheap-band block** (block C0-C4) via the
  same-issue follow-up loop (status `followups_running`, the new result
  folded into the EXISTING clean-result body, re-park at
  `awaiting_promotion`). Scoped to `question_relation: same` ONLY (a
  `substantially-different` follow-up changes the parent `## Goal` and
  cannot fold into the issue — it never auto-runs on this band
  regardless of GPU cost). Fail-safe: a `same` proposal with a missing /
  unparseable `est_gpu_hours` does NOT auto-run (parks/files for the
  user, mirror of the Step 2c plan-cap fail-safe). Bounded: at most 2
  cheap-band rounds per task (counted by `epm:same-issue-followup-run v1`
  / `source: proposer-9b-cheap`), and the Step 2c plan-approval GPU-hour
  cap is the final backstop inside the loop if an estimate was wrong.
  See Step 9b "Cheap follow-up auto-run".
- **Route the EXPENSIVE (`>= 20` GPU-h) `auto_run: yes` follow-ups by
  `question_relation` at Step 9b (autonomous mode only).**
  When a result lands, the orchestrator fires the `follow-up-proposer`
  at Step 9b (after auto-merge, BEFORE the human-only park flow completes
  — the Step 9b CRON-TEARDOWN already ran at the `awaiting_promotion`
  transition, so a dispatch path re-arms the tick per § Loop liveness
  backstop) and — for the proposals the
  cheap-band block did NOT take (estimate `>= 20` GPU-h or missing) —
  partitions the
  `auto_run: yes` proposals by QUESTION IDENTITY:
  `question_relation: substantially-different` proposals (and untagged
  ones from pre-2026-06-09 legacy markers only — a missing tag on a
  newer marker is a proposer-contract violation handled by the
  one-bounce re-spawn in Step 9b step 3) are FILED as `proposed`
  child tasks for manual triage ONLY — never auto-spawned as
  sessions — capped at 2 per parent AND hard-stopped at
  `parent_id`-chain depth 3 (so the recursive filing fan-out is both
  width- and depth-bounded, never exponential); `question_relation:
  same` proposals are NEVER filed as children — the top-ranked one
  runs ON the parent issue via the same-issue follow-up loop (post
  `epm:followup-scope v1`, re-enter the abbreviated cycle at status
  `followups_running` with tag `followup-auto`; capped at 2
  autonomous rounds per task, counted by `epm:same-issue-followup-run
  v1` markers with `source: proposer-9b`). All automatic follow-up
  EXECUTION is same-issue; a filed child runs only when a human
  triages it. Cost is still gated at the
  Step 2c plan-approval GPU-hour cap in BOTH paths — no new cost gate
  is added. Parent promotion stays human-only; neither path
  promotes the parent. Child filing is idempotent via
  `epm:follow-ups-autospawned v1` (skip if present; the marker body
  carries `execution: filed-only`); the same-issue
  loop is idempotent via `followup_label` matching between
  `epm:followup-scope v1` and `epm:same-issue-followup-run v1`.
  Interactive mode (`EPM_AUTONOMOUS_SESSION` unset) skips ONLY this
  EXPENSIVE autonomous-only block — it does NOT file children and does
  NOT auto-run `>= 20` GPU-h follow-ups (those still wait for the user's
  Step 10b pick post-promotion, routed by `question_relation`). But the
  Step 9b cheap-band block (`< 20` GPU-h) DOES fire in interactive mode
  (see the cheap-band bullet above), so an interactive session may run
  the proposer at Step 9b; Step 10b then reuses that `epm:follow-ups v1`
  via its proposer-already-ran short-circuit rather than re-spawning the
  proposer. See Step 9b "Cheap follow-up auto-run" + "Autonomous
  follow-up auto-spawn" + "Same-issue follow-up loop" + Step 10b
  "Proposer-already-ran short-circuit" for the mechanics; see
  `.claude/agents/follow-up-proposer.md` § "question_relation tag —
  criteria" + § "auto_run tag — criteria" for the tag semantics
  (canonical `auto_run: yes` example: a corrective re-run that fixes
  named validity defects with a grounded recipe, one variable changed,
  cost known — task #520 → #527, which under the new scheme is
  `question_relation: same` and runs on #520 itself).
- **Stop the tick cron at terminal/park state.** The `--auto` session is driven
  by the lightweight `/issue-tick <N>` cron (armed by Step 0 of the first
  `/issue <N>` invocation for autonomous sessions, covering the whole lifecycle
  from spawn onward; Step 6d.2 has a second ARM-GUARDed call that re-arms it
  if the Step 0 arm is missing — covers interactive `/issue` runs that reach
  the polling loop too). When the task reaches `awaiting_promotion`,
  `completed`, an over-cap `plan_pending`, or `blocked`, do NOT keep the cron
  armed — the backstop cron is torn down at the terminal/park transitions
  only (`awaiting_promotion`, `completed`, `blocked`, and the poll-loop /
  gate-park exits — NOT at `done`; it deliberately survives the post-`done`
  verifying/interpreting/reviewing stages so a stalled interactive session
  there still gets auto-woken). See Step 6d.2 CRON-TEARDOWN + the Step 9
  idempotency guard.
- **In-session PushNotification at gate-park / `blocked`.** At the over-cap
  `plan_pending` exit (Step 2c `parked_over_cap`), at `awaiting_promotion`
  (Step 9b), and at every autonomous-flow `status:blocked` exit, fire
  `PushNotification({"message": "...", "status": "proactive"})` BEFORE the
  CRON-TEARDOWN. The phone alerts the user that a session needs them, the
  cron tears down so it stops re-firing, and the session idles until the
  user taps in via the relay to drive the next step. Load the deferred
  schema once per session via
  `ToolSearch("select:PushNotification")` before first use (same pattern
  as `Cron*`). Soft-fail: if `PushNotification` raises (Remote Control
  disconnected, schema not loaded), swallow + continue — the title
  refresh + cron teardown still happen.

**Mid-flight handoff to an autonomous session (interactive / chat
sessions).** When the user asks to move in-flight issue work to an
autonomous Happy session ("run it in background with happy coder", "hand
this off", "spawn a session for this", etc.), execute the handoff
IMMEDIATELY, in the same turn:

1. **Post the handoff breadcrumb FIRST** — an `epm:progress` marker
   recording the current stage + round, the worktree path of any
   in-flight implementation work (`worktree=<abs path or 'repo-root'>`,
   same field as the stage-dispatch breadcrumb, Step 9 entry guard), and
   which files are uncommitted there (one `git -C <worktree> status
   --porcelain` line). This is what lets the successor session find
   partial work instead of starting over.
2. **Spawn the autonomous session NOW**: `uv run python
   scripts/spawn_session.py spawn-issue --issue <N> --auto`. NEVER defer
   the spawn on a future marker / event the CURRENT session is
   responsible for producing — when this session dies, its background
   subagents are killed with it and the trigger never fires. Deferred
   handoff is the banned pattern (incident #505 round 2, 2026-06-10: a
   chat session driving the same-issue follow-up loop conditioned the
   spawn on `epm:experiment-implementation v12`, which only its own —
   soon killed — bg implementer could produce; the session was closed 20
   min later, the marker never landed, no autonomous session was ever
   spawned, and the task sat orphaned at `running` for 5+ hours with
   uncommitted implementation files stranded in a worktree no marker
   named).
3. **Stop dispatching new work in this session.** In-flight bg subagents
   may finish and post their markers (harmless overlap — the spawned
   session's idempotent resume + the Step 9 entry guard's freshness
   window absorb a duplicate result), but no NEW stage subagent, pod
   call, or status flip originates here after the spawn. The spawned
   session — which is watcher-registered
   (`~/.eps-autonomous/issue-<N>.json`) and arms its own Step 0 backstop
   cron — owns the task from its first tick.

The ONLY existing liveness mechanism that survives this session being
closed is the `spawn-issue --auto` registration: the
`autonomous_session_watch.py` crash-recovery + stalled passes read only
the autonomous registry (an interactive session has no `issue-<N>.json`
entry), and a `durable=False` backstop cron dies with the session that
armed it. So the immediate spawn IS the handoff — there is no safe
deferred variant.

### Step 0: Load state

**Workflow-version dispatch (run BEFORE anything else).** Read the task's
`workflow` frontmatter field and, if it is `v2`, hand the whole task off to the
v2 skill and exit this skill immediately — v1 does nothing further for a v2 task.
This is a read-only probe (no markers, no mutation); the v2 skill re-runs the
same Step-0 guards itself.

```bash
uv run python scripts/task.py view <N> --json | uv run python -c \
  "import json,sys; print((json.load(sys.stdin).get('frontmatter') or {}).get('workflow') or 'v1')"
# → "v2"  : announce "Task #<N> is workflow: v2 — delegating to issue-v2." then
#           invoke the Skill tool `issue-v2` with the same task number <N> and STOP
#           (do NOT run any step below; issue-v2 owns the whole lifecycle).
# → "v1" / absent / empty : continue below UNCHANGED (v1 path, byte-for-byte).
```

**Single-orchestrator guard (run FIRST).** Exactly ONE session may drive
`/issue <N>` at a time. Before doing anything else, check whether another
live session is already mapped to this issue: `uv run python
scripts/spawn_session.py list` (issue-mapping column). If a live session is
already driving #N, EXIT immediately as a duplicate (do NOT run
`scripts/post_step_completed.py` here — this site's exit-marker contract is
the single breadcrumb below, per the "Post NOTHING else" rule that follows
it). Before ending, post
EXACTLY ONE exit breadcrumb so the audit trail distinguishes a deliberate
collision exit from a wrapper crash (#1053: a `/issue 958` session died
~1 min after spawn with no recorded reason — under the old marker-free
mandate a deliberate guard exit and a crash left identical evidence:
nothing):

```bash
uv run python scripts/task.py post-marker <N> epm:progress --by issue-session-guard \
  --note "deliberate-stop pid=n/a target=self reason=step0-session-collision owner=<owner-session-id> — duplicate /issue <N> session exiting at Step 0; owner <owner-session-id> remains the driver; no state mutated"
```

The `deliberate-stop ` note PREFIX is load-bearing (the note must START
with it, lstripped): all four staleness clocks —
`task_workflow.stage_dispatch_should_skip`,
`autonomous_session_watch._latest_progress_ts`,
`autonomous_session_watch._latest_nonwatcher_event_ts`, and
`tick_triage.latest_event_ts` (#810/#949/#990/#1053) — drop it by prefix,
so the breadcrumb never refreshes the OWNER's freshness windows or
staleness clocks. It WILL surface as one pre-dispatch triage candidate for
the owner — deliberate: a session collision is genuinely triage-worthy
externality (mirroring the existing operator-stop breadcrumb;
`issue-session-guard` is deliberately NOT in `TRIAGE_MACHINE_BY`).
Presence of this breadcrumb PROVES a deliberate guard exit; its ABSENCE is
evidence of — not proof of — a crash (a fail-soft post failure, a
pre-guard death, or a stale skill copy also yield absence). Fail-soft: if
the post fails, state the failure in your final text and still exit —
never block the exit on the breadcrumb (a deferred-commit stderr ERROR
with exit 0 is SUCCESS; never re-post on it). Post NOTHING else: do NOT
run `scripts/post_step_completed.py`, do NOT write the per-issue
self-report (`session_progress_report.py` would clobber the OWNER's shared
`~/.eps-autonomous/issue-progress/<N>.json`), mutate nothing else —
UNLESS this session is its explicit replacement (an
`autonomous_session_watch` crash-recovery respawn, or the user said to take
over; in that case stop the stale session via `spawn_session.py stop` first).
Incident 2026-06-09 (#524): two concurrent orchestrators both picked up a
re-plan directive; one auto-approved a plan whose GPU budget the other's
fact-checker had just shown to be a 2x underestimate, forcing a
`running -> plan_pending` rollback and wasted implementer work.

**Stale-wake ownership re-check (applies on RESUME, not just invocation).**
The guard above fires at `/issue` invocation — but a session that RESUMES
in-flight work after a long mid-flight stall must re-establish ownership
too, because the watcher may have respawned a replacement while it was
dark (and a manually-started session that never `register-current`'d is
invisible to the replacement's own Step 0 check, so the stale session is
the ONLY one positioned to detect the collision). If >30 min have passed
since this session's last tool call / turn, OR its last posted marker is
older than 30 min AND `events.jsonl` has advanced since, do NOT execute
the stale next step. FIRST re-run the guard: read `uv run python
scripts/task.py latest-marker <N>`, the registration files (fail-soft
probe below — copy it verbatim), and `uv run python
scripts/spawn_session.py list`.

```bash
# Registration-file probe — FAIL-SOFT, copy verbatim. A missing file is
# the NORMAL case (an interactive session that never ran
# register-current; an autonomous one whose registry entry was deleted
# at a terminal transition), and a bare `ls`/`cat` on an absent path
# exits non-zero (ls: 2, cat: 1), which CANCELS parallel sibling tool
# calls issued in the same message (5+ sessions, 2026-07-09). Rule:
# an INFORMATIONAL probe on an OPTIONAL file is ALWAYS fail-soft —
# append `2>/dev/null` + `|| true`, or if-form it
# (`if [ -f <p> ]; then cat <p>; fi`); Step 9c step 1b's "Recipe
# exit-code hygiene" covers the trailing-command flavor of this class.
cat ~/.eps-autonomous/issue-<N>.json ~/.eps-autonomous/manual-issue-<N>.json 2>/dev/null || true
```

If a replacement
session is registered for #N (a `spawned_at` newer than this session's
own start) OR the marker trail shows another writer has advanced the task
past this session's last-known state, YIELD immediately. Before ending,
post the SAME exit-breadcrumb convention as the Step 0 guard above (one
`epm:progress`; same fail-soft contract — a deferred-commit stderr ERROR
with exit 0 is SUCCESS, never re-post):

```bash
uv run python scripts/task.py post-marker <N> epm:progress --by issue-session-guard \
  --note "deliberate-stop pid=n/a target=self reason=stale-wake-yield replacement=<replacement-session-id-or-marker-evidence> — stale /issue <N> session yielding on wake; the replacement owns the task; no state mutated"
```

Then launch nothing, mutate nothing else; the replacement owns the task.
A `deliberate-stop` breadcrumb is a death record, NOT task advancement —
ANY marker-trail reader (this re-check, `task.py latest-marker <N>`, a
successor session deriving state from the trail) must never count one as
"another writer has advanced the task".
The cheap tell is always `task.py latest-marker <N>` before resuming any
stale in-flight plan: if events have advanced past your own last-known
state, re-derive state from the markers instead of executing the stale
next step. Incident 2026-06-10 (#535): a manually-started interactive
session stalled ~3h mid-flight, the watcher respawned an autonomous
replacement that worked for 1.5h, then the stale session WOKE and resumed
its stale plan — re-posting already-posted markers and launching a
duplicate live acceptance run + SLURM job the replacement had to
kill/scancel.

**Interactive-session registration (run once the guard passes).** An
INTERACTIVE session (`EPM_AUTONOMOUS_SESSION` unset) driving `/issue <N>`
registers itself ONCE at Step 0 so it appears in `spawn_session.py list`'s
issue-mapping — otherwise a manually-started session is invisible to every
OTHER session's single-orchestrator guard (the other half of incident
#535: the watcher's autonomous replacement could not see the live manual
session precisely because it never registered):

```bash
uv run python scripts/spawn_session.py register-current --issue <N>
# idempotent; writes ~/.eps-autonomous/manual-issue-<N>.json (alert-only:
# `list` visibility + stalled/crash alerts — never auto-respawned)
```

Autonomous sessions skip this — `spawn-issue --auto` already registered
them (`issue-<N>.json`). Registration failure is non-fatal: state the
failure and continue (same fail-soft contract as the Step 9b same-issue
follow-up loop's step-2 re-registration).

**Workflow-fix recursion-guard self-set (#678).** If THIS task is a
workflow-fix task — its `body.md` carries a `workflow_fix_target:`
Provenance line (the durable signal,
`task_workflow.is_workflow_fix_session(N)`) — set
`EPM_WORKFLOW_FIX_SESSION=1` for the session's own ergonomics. The
durable Provenance signal is the primary; the env var is the in-session
convenience leg lost on a watcher respawn (which re-runs `spawn-issue
--auto` without custom env), so the durable signal is always re-derivable
from the body. The recursion guard means this session — or any subagent
it spawns — NEVER auto-files MORE workflow-fix tasks for its own
findings: a `<!-- workflow-fix-candidate v1 -->` it raises is LOGGED +
notified, not routed (analogue of `AUTO_REVIEW_DISABLED`; see
`.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

```bash
# Reads body.md frontmatter + the most-recent slice of events.jsonl.
# Use --json for the machine-readable shape (body + last events).
uv run python scripts/task.py view <N> --json
```

From the result, derive:

1. **Current state** = the task's parent folder under `tasks/` (the
   `status` value).
2. **Task type** = the `type` field in `body.md` YAML frontmatter
   (`experiment`, `infra`, `batch`, `analysis`, `survey`).
3. **Marker map** = scan the recent `events.jsonl` rows for
   `epm:<kind>` entries, build a dict keyed by kind with the highest
   version per kind.

**Same-issue follow-up dispatch (chat entry point).** Before the
normal status dispatch, scan ALL `epm:followup-scope` entries for this
issue (NOT the highest-version-per-kind marker map — distinct queued
follow-ups share the kind under different `followup_label`s, #894/#763)
grouped by `followup_label`: within a label the authoritative scope is
the latest-(`ts`, `version`) entry (corrections land append-only — the
#658 v3→v7 `persona-vectors-style-rb` chain), and a LABEL is UNRUN iff
it has no matching `epm:same-issue-followup-run v1`. Canonical
implementation: `task_workflow.unrun_followup_labels(list_events(N))`.
When ≥1 DISPATCHABLE unrun label exists, dispatch EXACTLY ONE label per
loop entry — the queue head: user-initiated labels (`source: user-chat`
/ `step-10b-pick`) first, then oldest armed ts (a label's arming ts is
its FIRST scope entry's ts). A `dispatchable: false` group (an
unlabeled non-correction scope → pseudo-label `unlabeled-<ts>`) is
NEVER executed as a round — surface it loudly instead (one chat line +
an `epm:progress` note naming the repair: re-post the scope with a
proper kebab-slug `followup_label`; the pseudo-label stays visible in
every future scan until repaired or retro-closed). The loop runs one
round and re-parks; the next `/issue <N>` entry picks up the next unrun
label. If ≥1 unrun label is present AND the status is
post-result (`interpreting` / `reviewing` / `awaiting_promotion` /
`completed`) — or `followups_running` itself (the mid-round resume
case: the loop holds that status, so a crashed round re-enters here) —
route into the **same-issue follow-up loop** (Step 9b §
Same-issue follow-up loop) instead of the normal resume row. This is
how chat-requested follow-ups execute: the chat session posts
`epm:followup-scope v1` (`source: user-chat`) on #N, then re-invokes
`/issue <N>`, and the dispatcher lands here. An unrun followup-scope
on a task still mid-pipeline (any other status) waits — the loop only
fires from a post-result state.

**Stale-label disposition rule (mechanical evidence only).** Before
executing a dispatched label, run
`task_workflow.followup_retro_close_evidence(events, label)` — the
exact-label mechanical evidence predicate (a 9a-quater `extends=<label>`
record; an `epm:free-analysis-followup-run` whose `followup_ref` EXACTLY
equals the label; or a status/step note carrying the exact parenthesized
`(<label>)` round token plus a round-completion word in the same
;/.-delimited clause as the token — a clause naming the label as queued /
unrun / scoped / armed / dispatched NEVER closes (park notes routinely
announce one round's completion while enumerating the queued next label
on the same line; #961)). On a NON-None
return, post the retroactive `epm:same-issue-followup-run v1`
(`followup_label` verbatim, `outcome: retroactive-close — <evidence>`)
and move to the NEXT unrun label instead of re-running a completed round
(legacy tasks like #658 carry such ghost labels). **This check is a
GHOST-label filter, NOT an execution gate.** A None return means NO
prior-run evidence exists — for a normal fresh never-run label this is
the EXPECTED result (no run happened, so no evidence can exist by
construction) and the label EXECUTES as the dispatched round: that is
the whole point of the queue (the #763 `neutral-contrast-and-cofit`
dispatch is the canonical fresh case). The skip-and-surface disposition
applies ONLY when the orchestrator has independent reason to suspect
the label ALREADY RAN — a legacy ghost-label task like #658, e.g. the
label's round is visibly recorded in the body / status history yet none
of the mechanical evidence classes can confirm it. In that
suspected-stale case do NOT close on prose suspicion (a merely-prose
mention of the label NEVER closes — closing takes mechanical evidence
only): skip the label this entry and surface it for manual disposition
(autonomous mode: continue to the next dispatchable label; the skipped
label stays queued and visible) rather than either re-running a
completed round or wrongly closing a queued one. Retro-close markers do
NOT count toward the same-issue round caps (Step 9b loop step 5 /
block C2).

**Set the launch title now.** As soon as the slug (task `title`) is known,
call `set_title(N, <status>)` (helper defined in the "Chat title updates"
section above) so the Happy phone session list AND the
`~/.eps-autonomous/session_progress.json` cache (read by `happy-ls` + the
`/sessions` dashboard) all show the SAME canonical string from the moment
the session is spawned. The `--step` is the current status (or `"launching"`
if status isn't loaded yet):

```bash
uv run python scripts/session_progress_report.py --issue <N> --step "<status>"
# Capture stdout (the canonical string), then:
# mcp__happy__change_title({"title": <captured>})  -> "#<N> <slug> · <status>"
```

This runs on EVERY `/issue <N>` invocation (idempotent — re-setting the same
title is harmless), so resumed sessions re-label themselves too AND the
self-report file gets re-touched, keeping the dashboard fresh. Later
status transitions, polling-loop ticks, and clean-result-finalized events
re-call the helper with an updated `--step`.

**Hard error: ambiguous status.** If `task.py view <N>` reports the task
exists in multiple folders (should be impossible because `task.py` holds
the flock — but the lint catches manual edits), abort with an error and
ask the user to reconcile. Do NOT pick.

**Soft error: status missing from frontmatter (legacy bodies), type missing,
or empty body.** These are recoverable; do NOT exit. Run Step 0b instead.

**Worktree spec-freshness BEFORE arming (sessions whose cwd is an issue
worktree).** A worktree pins the entire workflow surface at branch-fork
time, so the skill/cron prescriptions you are reading may be stale —
run the Step 5a spec-freshness sync (surgical `git checkout main -- `
of the workflow-surface specs, with the branch-side-feature-edit guard)
FIRST, and resolve workflow-helper scripts (`verify_task_body.py`,
`post_step_completed.py`, ...) from the MAIN checkout (`"$REPO_ROOT"/scripts/...`),
never the worktree copy. (Incident #501, 2026-06-06→08: a worktree's
pre-split skill copy armed `/issue 501` at */10 instead of the
lightweight `/issue-tick` backstop (then */20, now */45) — 362 full
~44K-token skill reloads over 2.5 days. Incident #496: a worktree's pre-W22 `verify_task_body.py`
false-FAILed a spec-conformant body, wrongly indicting the analyzer.)

**MANDATORY auto-armed backstop for autonomous sessions — arm it NOW.**
When `EPM_AUTONOMOUS_SESSION=1` is set (the session was spawned via
`spawn_session.py spawn-issue --auto`), arm the `/issue-tick <N>` cron
at Step 0, BEFORE any branching into Step 0b / 0c / 1 / 2 / 5 / 6. The
historical site (Step 6d.2) only covers `kind: experiment` runs that
reach the pod-launched polling loop; a session can stall ANYWHERE in
the lifecycle (during planning, code-review, plan_pending park, the
analyzer / clean-result-critic loop, even at first invocation) and the
late-arm leaves all of those stretches uncovered. Real incident: task
#518 (2026-06-08) stalled in the code-review loop at round 7 — the
session ended its turn at a clean exit point, and because Step 6d.2
had not yet run, NO tick cron was armed; the session sat dead until
the external watcher's stalled-detector pass caught it.

```python
# Load the deferred Cron tools once per session if not already loaded.
ToolSearch("select:CronCreate,CronList,CronDelete")

# ARM-GUARD: idempotent re-entry. Whole-string equality (not substring) —
# "/issue-tick 46" is a substring of "/issue-tick 467".
if os.environ.get("EPM_AUTONOMOUS_SESSION") == "1":
    jobs = CronList()
    already_armed = any(
        (job.get("prompt", "").strip() == f"/issue-tick {N}") for job in jobs
    )
    if not already_armed:
        CronCreate(
            cron="*/45 * * * *",
            prompt=f"/issue-tick {N}",
            recurring=True,
            durable=False,
        )
        # Re-list + assert exactly-one match, same dupe-fail-fast contract
        # Step 6d.2 uses — surfaces a harness prompt-normalization bug NOW
        # rather than after dozens of duplicate ticks have accumulated.
        post = CronList()
        match_count = sum(
            1 for job in post if job.get("prompt", "").strip() == f"/issue-tick {N}"
        )
        assert match_count == 1, f"cron arm: expected 1 match, got {match_count}"
```

Interactive sessions (no `EPM_AUTONOMOUS_SESSION`) do NOT arm the cron at
Step 0 — they're user-driven and the user re-invokes `/issue <N>` manually
when needed. The Step 6d.2 cron-arm still runs for those interactive runs
that DO reach the polling loop (same call shape, same ARM-GUARD), so the
session-survival backstop for pod-backed runs is unchanged for them.

The cron is torn down at the SAME terminal / park transitions as before
(see Step 6d.2 § CRON-TEARDOWN). Adding the early arm only widens the
window during which the backstop is in place; it does not change when
it's removed.

### Step 0b: Defaulting & autofill

Runs only when at least one of {no current folder, missing `type` in
frontmatter, empty body} holds. Goal: get the task into the minimum
shape Step 1 needs without bouncing back to the user just to add
metadata. Order:

1. **Folder missing (legacy / migration case) ->** apply
   `status:proposed` automatically by moving the task to
   `tasks/proposed/<N>/`:
   ```bash
   uv run python scripts/task.py set-status <N> proposed --note "Autofilled by /issue Step 0b."
   ```
   No user interaction. Defaulting an unlabelled task to `proposed` is
   the obvious read of the lifecycle (To do column = `proposed`).

2. **Body empty (or <50 chars of substance) ->** ask the user in the
   <!-- gate: gates.empty_body -->
   <!-- autonomous-mode: block-and-fail -->
   current chat via `AskUserQuestion` for the minimum spec needed for the
   adversarial planner to design the task. The exact prompts depend on
   the task type (see `clarifier.md`); for an unknown type, ask:
   - "What's the goal of this task in one sentence?"
   - "What's the hypothesis or success criterion?"
   - "Is there a parent task or prior result this builds on? (task # or 'none')"
   - "Rough compute size? (small / medium / large)"

   In autonomous mode (`EPM_AUTONOMOUS_SESSION=1`) this gate cannot
   auto-resolve — a missing task body is a content gap only the user
   can fill. Post `epm:failure v1 failure_class: data` (reason:
   `body empty; autonomous mode cannot synthesise spec from title`),
   set `status:blocked`, and exit (halt-criterion #4 — factual question
   only the user knows). The PreToolUse hook in `.claude/settings.json`
   is the runtime backstop and will hard-block the ask if reached.

   Plus **search the codebase + HF + arXiv before drafting** when the
   title hints at pulling existing artifacts (e.g., "use HF model X",
   "replicate paper Y") — list what you found and let the user pick.
   Don't fabricate a body from the title alone.

   Once the user answers, draft a body covering Goal / Hypothesis / Setup
   / Eval / Success criterion / Kill criterion / Compute / Pod preference
   / References (for a representation-mapping task — geometry read /
   predictor / probe / direction extraction over activations — the drafted
   Setup names BOTH mapping arms, prefix-based AND context-based, per the
   CLAUDE.md "Prefix mapping AND context mapping" Critical Rule; a one-arm
   draft states the deviation explicitly), then patch the task:
   ```bash
   uv run python scripts/task.py set-body <N> --file /tmp/issue-<N>-body.md
   ```
   Post a `<!-- epm:auto-defaults v1 -->` event listing what was applied
   (folder moved, body drafted) so the audit trail is durable on the
   task:
   ```bash
   uv run python scripts/task.py post-marker <N> epm:auto-defaults \
     --note "Drafted body from user chat answers; moved to tasks/proposed/<N>/."
   ```

   **Audit-marker placeholder guard (when generating any `epm:audit` /
   `epm:auto-defaults` body):** before posting, run
   `grep -E "(^|\s|>)(TBD|TODO|placeholder|\[X\]|implementer fills)(\s|$|<)"`
   against the drafted body. Match -> BLOCK the post and finish the audit
   instead. The regex catches placeholders mid-line as well as line-start.

3. **`type` frontmatter missing ->** infer from title cue, then confirm
   with the user:
   - Title prefix `Test:` / `Sweep:` / `Train:` -> suggest `experiment`
   - Title prefix `Refactor:` / `Fix:` / `Add:` / `Migrate:` -> suggest `infra`
   - Title prefix `[Batch]:` / `[Workflow]:` / body contains a numbered
     list of >=3 unrelated fixes -> suggest `batch`
   - Title prefix `Analyze:` / `Re-analyze:` -> suggest `analysis`
   - Title prefix `Survey:` / `Read:` / `Lit review:` -> suggest `survey`

   **Fix-validation override (CLAUDE.md § "Routing experiment intent"):**
   a `Test:` cue does NOT default to `experiment` when the Goal is to
   VALIDATE / TEST that a shipped workflow / infra / code fix WORKS (a
   smoke run, an end-to-end "does it work now after the fix", a config /
   pipeline / backend re-check) — that is `kind: infra`, NOT `experiment`,
   because it completes on the test-verdict path and produces NO promotable
   clean-result. Reserve `experiment` for a RESEARCH QUESTION that produces
   a clean-result the user promotes. Litmus: would the result rewrite an
   issue's `## Takeaways` / answer an `open_questions.md` question
   (→ `experiment`), or just confirm the fix is sound (→ `infra`)? When the
   title says `Test:`/`Validate:` but the body reads as fix-validation,
   suggest `infra` as `(Recommended)`. (Incident #672: a GCP-fix validation
   filed as `experiment` was parked at `awaiting_promotion` as a promotable
   clean-result.)

   <!-- gate: gates.missing_type -->
   <!-- autonomous-mode: block-and-fail -->
   Use `AskUserQuestion` with the inferred option as `(Recommended)`
   first. Apply via `task.py set-body --file ...` to update the
   frontmatter `type:` line. In autonomous mode
   (`EPM_AUTONOMOUS_SESSION=1`), DO error and EXIT — the type field
   gates Step 7's completion variant and a guess here corrupts the
   lifecycle. The PreToolUse hook hard-blocks the ask if reached.
   Before exiting, post the §5 marker:
   ```bash
   uv run python scripts/post_step_completed.py --issue <N> --step 0b \
     --exit-kind failure-exit \
     --notes "type-frontmatter autofill loop; user override required"
   ```

4. **Other useful frontmatter fields missing** (`compute`, `priority`):
   do not block on these. `compute` will be set in the adversarial-planner's
   reproducibility card; `priority` is user-curated and never blocking.

   Note: legacy `aim:*` GH labels were deleted long ago. New tasks do not
   use them. Topic categorization for new work lives in `docs/claims.yaml`
   (`topic` field) and in `RESULTS.md` / `eval_results/INDEX.md` H2
   prose; no replacement frontmatter field exists.

After Step 0b, re-read the task (re-run `task.py view <N>` from Step 0)
so downstream state is computed from the now-patched task, then continue
to Step 0c.

### Step 0c: Goal-of-experiment gate (safety net)

Every `kind: experiment` task must carry a one-sentence **Goal** in
body.md frontmatter (`goal:`) and an inline `## Goal` H2 block before
any other H2. The Goal is the canonical optimization target every
downstream subagent reads (planner, critic, experiment-implementer,
analyzer, clean-result-critic, interpretation-critic,
follow-up-proposer). The PM session Mode 5 pre-spawn check is the
primary enforcement point; Step 0c is the per-issue-session safety
net.

This is a **legitimate `AskUserQuestion` use** in Interactive mode
because the gate IS a gate (CLAUDE.md "Critical Rules" lists
`experiment_goal` as inline gate #6 — see workflow.yaml §
gates.experiment_goal). It does not violate the auto-continuation
policy. In autonomous mode (`EPM_AUTONOMOUS_SESSION=1`), the Goal must
have been set BEFORE the session was spawned (the PM session Mode 5
pre-spawn check is the primary enforcement); if it's still missing at
Step 0c, the autonomous session post `epm:failure v1 failure_class: data`
(reason: `goal missing; autonomous mode cannot synthesise`), sets
`status:blocked`, and exits (halt-criterion #4). The PreToolUse hook
hard-blocks the ask if reached. <!-- autonomous-mode: block-and-fail -->

1. Skip the gate when the task `kind != "experiment"` (i.e.
   `analysis | infra | batch | survey`). These kinds do not carry an
   experiment Goal.
2. Otherwise, read the task's frontmatter + body via `task.py view <N>
   --json` and check:
   - Frontmatter contains `goal: <non-empty string>`, AND
   - The body contains a `## Goal` H2 (matched verbatim, line-start).

   If both hold, continue to Step 1.
3. If either is missing, raise `AskUserQuestion` <!-- gate: gates.experiment_goal --> <!-- autonomous-mode: block-and-fail -->:
   ```
   "What is the one-sentence Goal of this experiment?
    (The single decision-shaping target every downstream agent will
    optimize toward — e.g. 'Measure whether persona-tagged SFT
    transfers to held-out personas at the same rate as in-distribution
    ones.')"
   ```
   (Interactive mode only — autonomous sessions block-and-fail per the
   §0c-intro annotation above.) On the user's answer (one sentence; do
   NOT accept a fragment or a list — re-prompt once if the answer
   doesn't read as a complete sentence), run:
   ```bash
   uv run python scripts/task.py set-goal <N> "<the answer>" --by user
   ```
   The command writes both frontmatter (`goal:`) and the body H2
   block, then posts `epm:goal-updated v1` to events.jsonl. Re-read
   the task (Step 0) and continue to Step 0c-link.

#### Step 0c-link: Match-or-create open-question link (same Goal gate)

After the Goal is set for a `kind: experiment` task, link it to the
living research hub (`docs/open_questions.md`) so the completion hook
(Step 10c) knows which question(s) the result should move. This runs
inside the same Goal gate the user already passes through — no separate
gate, no extra context switch.

1. Skip when the task `kind != "experiment"` (i.e.
   `analysis | infra | batch | survey`). Those kinds carry no
   open-question link, exactly like the Goal gate itself.
2. Skip when the task already carries a non-empty `relates_to:` list in
   `body.md` frontmatter (re-invocation / already-linked case) — the
   link is set once at creation. Continue to Step 1.
3. Otherwise, read the task Goal + the headline questions in
   `docs/open_questions.md` and produce a flat list of stable
   open-question ids (NO primary/secondary) the experiment bears on —
   **matching** existing question id(s) wherever an existing question
   fits, and only **drafting a new question** when none fit.
4. **Matching existing question(s) — AUTO-LINK, do NOT ask.** When every
   id in the list is an *existing* question id (no new question needs to
   be drafted), write the link immediately, without asking the user — no
   gate prompt. State the match in chat so the user can correct it if
   it's wrong, then write it:
   ```
   Assumption: linking #<N> to existing open question(s) <q-ids> «<headline(s)>».
   ```
   ```bash
   uv run python scripts/living_docs.py link <N> <q-id> [<q-id> ...]
   ```
   This is the common case — an experiment almost always bears on a
   question that already lives in the hub. Linking to an existing
   question is a low-risk, reversible bookkeeping write (the
   `living_docs.py check` lint + the completion-time `living-docs-updater`
   both catch a bad link later), so it does not consume a gate.
5. **No existing question fits → drafting a NEW question — ASK first
   in Interactive mode.** Creating an open-question stub is a real,
   durable living-docs mutation, so the new-question path stays
   user-confirmed. Propose the new question (plus any existing ids that
   ALSO apply) via
   `AskUserQuestion` <!-- gate: gates.experiment_goal --> <!-- autonomous-mode: skip --> in the SAME Goal
   gate:
   ```
   "No existing open question in docs/open_questions.md fits this
    experiment's Goal. Draft a new one? (an experiment may also bear on
    existing questions — add them too.)
      - Draft new question: «<one-sentence proposed question>» [+ also link q-<id> ...]
      - Link only to existing instead: q-<id> «<headline>» [+ more]"
   ```
   On the user's confirmation, write the link via the same command:
   ```bash
   uv run python scripts/living_docs.py link <N> <q-id> [<q-id> ...]
   ```
   `living_docs.py link` creates the question stub (heading +
   `<!-- q:<id> -->` anchor + `State:` trailer) in `docs/open_questions.md`
   for any id that does not yet exist, then writes `relates_to` + the
   evidence entry.
6. In both cases, post `epm:question-linked v1` recording the
   `relates_to` list, whether a new question was created, and the mode:
   ```bash
   uv run python scripts/task.py post-marker <N> epm:question-linked \
     --note "Linked task #<N> to open question(s) <q-ids>; created_new=<q-id|none>; mode=<auto-match|user-confirmed-new>."
   ```
   Re-read the task (Step 0) and continue to Step 1.

<!-- example: anti-pattern -->
**Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): on path 5 (no
existing question fits) SKIP the new-question draft entirely — do not
raise `AskUserQuestion`, do not print the proposed question as a text
menu. EXECUTE the skip in this same turn: post `epm:question-linked v1`
with `mode=autonomous-skipped` + `created_new=none` + an empty
`relates_to`, then continue to Step 1 (do NOT end the turn waiting on
user confirmation). The PreToolUse hook hard-blocks the ask if reached;
the nightly /daily living-docs backstop re-synthesis OR a later `/issue <N>`
re-invocation will reconcile the link.

### Step 1: Clarifier gate

If `epm:clarify` marker missing (or user has replied in `comments.jsonl`
but the clarifier hasn't re-checked): read `clarifier.md`, run the
clarifier for this task type, then:

**Before drafting any clarifying question, run the mandatory
context-gathering pass in `clarifier.md` Step 0** — search past
clean-result tasks, `.arxiv-papers/`, `external/`, `RESULTS.md`,
`eval_results/INDEX.md`, and `git log` for information that resolves the
ambiguity. Cut any question already answered by project knowledge;
sharpen the rest by quoting the source. When posting "All clear",
include a brief **Context resolved** bullet list of the
tasks/commits/papers consulted so the inheritance chain is auditable.

- **All clear** (<=1 minor ambiguity) -> post `epm:clarify` with "No
  blocking ambiguities found. Proceeding to adversarial planning."
  Move the task to the `planning` folder:
  ```bash
  uv run python scripts/task.py post-marker <N> epm:clarify \
    --note "No blocking ambiguities. Proceeding to adversarial planning."
  uv run python scripts/task.py set-status <N> planning --note "Clarifier All-clear."
  ```
  This is the one place where the task transitions out of the To-do
  column into the pipeline. Subsequent phases route automatically as
  `task.py set-status` is called at each step.

- **Ambiguities remain** -> do BOTH of the following, in order:

  1. **Post on the task.** Append a `epm:clarify v<n>` event with the
     numbered questions in the `note` body. This is the durable log — if
     the user closes the terminal, the questions are still there in
     `events.jsonl`.

  2. **Ask the user in the current chat (Interactive mode only).**
     Immediately after posting, ask the SAME numbered questions to the
     user in the current session.
     <!-- gate: gates.clarifier_blocking -->
     <!-- autonomous-mode: block-and-fail -->
     Use `AskUserQuestion` for small multiple-choice-style prompts;
     otherwise post a short numbered list as plain text and wait for a
     reply. Do NOT exit yet — give the user the option to answer inline
     so they don't have to context-switch to the dashboard. In
     autonomous mode (`EPM_AUTONOMOUS_SESSION=1`), do NOT ask — post
     `epm:failure v1 failure_class: data` (reason: `clarifier blocking
     ambiguities; autonomous mode cannot resolve`), set `status:blocked`,
     and exit (halt-criterion #4). The PreToolUse hook hard-blocks the
     ask if reached.

  3. **If the user answers in chat:**
     - Post a `epm:clarify-answers v<n>` event with the user's answers
       verbatim (lightly formatted — one numbered bullet per question),
       so the task is self-contained for downstream agents.
     - If the user also asks you to fold the answers into the task body
       (e.g., "update the body"), run `task.py set-body <N> --file ...`
       with the original body preserved + a `## Spec (from clarifier)`
       section appended. Only do this on explicit request — default is
       events-only.
     - Re-run the clarifier evaluation using (body + clarify questions +
       these answers). If no blocking ambiguities remain, advance to
       Step 2 (adversarial planning) in the same invocation. If still
       ambiguous, loop: post a `v+1` clarify event and ask again.

  4. **If the user defers ("I'll answer later", no reply, or says to
     exit):** EXIT with status still `proposed`. User can answer later
     via the dashboard's `comments.jsonl` append path, OR re-invoke and
     answer in chat next time. Before exiting, post the §5 marker:
     ```bash
     uv run python scripts/post_step_completed.py --issue <N> --step 1 \
       --exit-kind parked --notes "clarifier deferred by user"
     ```

**Rule:** never proceed to adversarial planning with >=2 blocking
ambiguities. Tight specs save later backtracking.

**Rule:** the ask-in-chat step is MANDATORY when there are blocking
ambiguities. Posting questions only as events and immediately exiting
forces a context switch the user does not want — always offer the
inline path first.

**Goal-refinement (optional, conditional gate #9).** If the clarifier
notices the existing `## Goal` H2 is fuzzy — e.g. too broad, names
two outcomes, or doesn't actually describe what would change with
the result — it MAY propose a sharper Goal via
`AskUserQuestion` <!-- gate: gates.experiment_goal_refine -->
**IN INTERACTIVE MODE ONLY**. On explicit user consent in the same
turn, run
`uv run python scripts/task.py set-goal <N> "<new goal>" --by clarifier --reason "<one line>"`,
which emits a new `epm:goal-updated v1` marker. Without explicit
consent the Goal stays put. Never call `set-goal` without
in-the-loop user agreement; this is the user's contract field.

<!-- example: anti-pattern -->
**Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): SKIP this refinement
entirely per § Autonomous session behavior → `experiment_goal_refine`.
The Goal stays as set at task creation; do not propose a refinement,
do not raise `AskUserQuestion`, do not print the proposed sharper Goal
as a text menu. EXECUTE the skip by continuing to Step 2 in this same
turn; do NOT end the turn waiting on user confirmation. The user owns
the Goal contract; an autonomous session may not silently shift it.

### Step 2: Adversarial planning

Only if status is `planning`.

Invoke the `adversarial-planner` skill with the task body + clarifier
output as the task. The skill runs planner -> fact-checker -> critic
-> revise internally.

**Required sections in the final plan (enforced by this skill — reject
plans missing any):**
- Goal + hypothesis (experiments) or requirement + acceptance criteria (code changes)
- Method delta (what differs from prior related work)
- File paths + concrete diffs / config overrides
- **Reproducibility Card** (mandatory per CLAUDE.md) — all hparams, seeds,
  data, env versions, exact workload command for experiments (the
  workload/dispatcher command(s) plus any required env-var pins — NOT a
  detachment/env-source launch wrapper (`nohup`/`source .env`); at launch
  the experimenter wraps it in the canonical setsid launcher script,
  `experimenter.md` § "During Execution")
- Success criteria with quantitative thresholds
- Kill criteria (what result would kill the thesis)
- Compute estimate in GPU-hours — MUST include a machine-readable total line
  the auto-approve gate (Step 2c) can parse:
  `Estimated GPU-hours (total): <number>` (a single number, the total across
  all conditions/seeds; not a range). The autonomous auto-approve gate FAILS
  SAFE — it parks at `plan_pending` if this line is missing or unparseable.
- Target pod preference
- Plan deviations allowed vs must-ask

**Goal-currency gate:** before EVERY `new-plan-version` call, re-read
`frontmatter.goal` and compare against the spawn-time snapshot
(`adversarial-planner` SKILL.md § Goal-currency gate) — a goal-update newer
than the draft start forces a mechanical redraft bounce (re-spawn the
planner against the amended Goal; NOT a critic round). Incident #922
(2026-07-03): plan v3 persisted quoting a Goal superseded 10 minutes
earlier by `epm:goal-updated`, was auto-approved 3 s later, and was caught
only by a PM-chat directive one wasted implementer round later.

Post the plan body via `new-plan-version` (writes
`tasks/<status>/<N>/plans/v<K>.md` and rotates the `plan.md` symlink),
then announce it with an `epm:plan` event. The handoff file carries a
per-attempt suffix — `<attempt>` = a fresh `$(date +%s)` chosen once per
orchestrator planning attempt — because a crashed attempt leaves a stale
/tmp file; a respawned session re-Writing the fixed path after Reading an
older version gets "File has been modified since read" (4× on #822):

```bash
uv run python scripts/task.py new-plan-version <N> --file /tmp/issue-<N>-plan-v<K>-<attempt>.md
PLAN_PATH=$(uv run python scripts/task.py find <N>)/plans/plan.md
# Embed the machine-readable cost token (<X> = the plan's total GPU-hours) so
# the Step 2c auto-approve gate can parse it from the note as well as the body.
uv run python scripts/task.py post-marker <N> epm:plan \
  --note "Plan v<K> written to $PLAN_PATH (gpu_hours_total=<X>)"
```

`new-plan-version` prints the dashboard URL
`https://eps.superkaiba.com/tasks/<N>/plan` (planned; substrate is
local files until the dashboard ships) — capture it as a shell variable
in the SAME bash block that posts the event. **Do not persist
`PLAN_URL` to a cache file.** The variable lives only for the duration
of Steps 2a -> 2c, which run in the same orchestrator turn (the
auto-continuation policy in CLAUDE.md guarantees no pause between them
in interactive mode; in autonomous mode the orchestrator exits at Step
2c so the variable is irrelevant).

Subagent briefs always pass the symlink path (`plans/plan.md`) so they
read the freshest version.

Also include estimated cost prominently in the `epm:plan` note, with a
machine-readable token (`gpu_hours_total=<number>`) the Step 2c auto-approve
gate parses, e.g.

> **Cost gate:** estimated 12 GPU-hours on 4× H100 (`gpu_hours_total=12`). Reply `approve` to dispatch.

**Cost confirmation does NOT pre-provision the pod.** Do NOT call
`pod.py provision` until the user replies `approve` (i.e., the Step 2c
plan-approval gate fires "Approve" and the task moves to
`status:approved`). Posting the cost note and then provisioning "to
save time" creates an orphan pod if the session exits before approval
(incident #406: an idle 2× H100 burned ~24h at ~$5-6/hr because the
session exited at this gate and was never re-invoked). If the session
must exit at this gate, post `epm:awaiting-spend-approval v1` and
ensure NO pod exists yet — the stale-pod audit cannot reap a pod the
workflow provisioned speculatively before approval.

### Step 2b: Consistency checker (runs ∥ the Phase 2 critic ensemble)

The `consistency-checker` no longer waits for an APPROVE-rated plan: it
needs only the drafted plan + the parent recipe — the same input the
Phase 2 critics get, with no dependency on their verdicts — so spawn it
CONCURRENTLY with the /adversarial-planner Phase 2 critic ensemble
(same spawn batch as the 6 critics, staggered a few seconds apart per
the CLAUDE.md 429 guidance; see adversarial-planner SKILL.md Phase 2).
Its findings are UNIONED with the critics' blockers into the single
Phase 3 revise round — one revision round covers both, instead of two
serial bounce rounds. Verdict semantics and the `epm:consistency v1`
marker are unchanged; only the scheduling moved. Its verdict must still
be folded in BEFORE posting the plan as `epm:plan`. It receives:
- The drafted plan
- Related tasks (cited in the plan's prior work, parent task, or
  near-duplicate clean-result task)
- The `epm:plan` and `epm:results` markers from those related tasks
  (read via `task.py latest-marker` + `task.py view --json`)

The consistency checker verifies:

| Check | Violation action |
|-------|-----------------|
| Single variable change from parent | BLOCK: list all differences |
| Same baseline model/checkpoint | WARN: flag, require justification |
| Same eval suite | BLOCK: incompatible evals make comparison meaningless |
| Same seeds or superset | WARN: disjoint seeds reduce comparability |
| Same data version/hash | WARN: different data confounds results |

Post `epm:consistency v1`. On BLOCK, the finding joins the Phase 3
revise round's UNION — critic Must-Fix items + consistency BLOCKs,
addressed together by the planner in ONE revision round (consistency
re-checks after revision keep the existing loop cap, max 2 rounds). On
WARN, append warnings to the `epm:plan` event note. On PASS, proceed
normally. The `plan_pending` flip below still happens only AFTER the
checker's FINAL verdict is folded in (adversarial-planner SKILL.md
§ Park order) — never on its interim ack.

Then post the plan as `epm:plan v1` with the consistency results
appended.

Move the task to `plan_pending` **through the code-enforced autonomous
plan-gate** — pass the plan's total GPU-hours so `task.py` itself makes the
auto-approve / park / interactive decision (it reads `EPM_AUTONOMOUS_SESSION`
+ `EPM_PLAN_AUTOAPPROVE_GPU_HOURS` from the env). This is what makes
autonomous auto-approval deterministic instead of dependent on the
orchestrator obeying the Step 2c prose:

```bash
uv run python scripts/task.py set-status <N> plan_pending \
  --auto-approve-if-autonomous --gpu-hours <X> \
  --note "Plan v1 ready for approval; consistency PASS."
```

`<X>` is the plan's `Estimated GPU-hours (total)` (the same number embedded
as `gpu_hours_total=<X>` in the `epm:plan` note). **Omit `--gpu-hours` only
if the total is genuinely unknown** — a blank estimate fail-safes to a park,
never an auto-approve. The command prints a `PLAN_GATE_DECISION: <decision>`
line (`auto_approved` | `parked_over_cap` | `interactive_pending`) that
Step 2c branches on; for `auto_approved` it has already flipped the status to
`approved` and posted `epm:plan-approved`, and for `parked_over_cap` it has
already posted `epm:awaiting-spend-approval`.

> **Same-issue follow-up round?** At `followups_running` this same command is
> safe: `task.py` fires the gate decision + markers but HOLDS the status in
> place (status-hold rule, Step 9b § Same-issue follow-up loop step 3) and
> appends `(followups_running hold: status unchanged)` to the decision line.

### Step 2c: Inline plan approval

**The autonomous plan-approval decision was already made by the Step 2b
`set-status ... --auto-approve-if-autonomous --gpu-hours <X>` call — in code,
not by LLM discretion here.** That command (in `scripts/task.py`) reads
`EPM_AUTONOMOUS_SESSION` + `EPM_PLAN_AUTOAPPROVE_GPU_HOURS` and printed a
`PLAN_GATE_DECISION:` line.
<!-- gate: gates.plan_approval -->
A PreToolUse hook on `AskUserQuestion`
(`.claude/settings.json`) ALSO hard-blocks (`exit 2`) any plan-approval
`AskUserQuestion` while `EPM_AUTONOMOUS_SESSION` is set — so the autonomous
path physically cannot reach the interactive ask even if this prose is
mis-followed. (Why both: the script removes the gate so the ask is never
reached; the hook is the backstop that forbids it if reached. Incident
2026-06-05 — four `--auto` sessions all asked for plan approval because the
auto-approve lived only as prose here and the LLM deferred to the global
"ask before spending money" prior.)

Branch on the decision (equivalently, re-read the task status):

- **`auto_approved`** (autonomous, est ≤ cap): the gate already flipped the
  status to `approved` and posted `epm:plan-approved`. Do NOT ask, do NOT
  re-post. Continue to Step 4 in the **same invocation**.
- **`parked_over_cap`** (autonomous, est > cap OR blank estimate — FAIL
  SAFE): the gate left the status at `plan_pending` and already posted
  `epm:awaiting-spend-approval`. The PM session + the user's phone surface
  the `plan_pending` status. Post the §5 marker, fire a PushNotification,
  then EXIT:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 2c \
    --exit-kind parked --notes "plan_pending; over auto-approve cap"
  ```
  ```python
  cap = os.environ.get("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "100")
  PushNotification({
      "message": f"#{N} {slug} parked at plan_pending — over {cap} GPU-h cap; open to approve"[:200],
      "status": "proactive",
  })  # soft-fail; deferred-schema may not be loaded
  ```
- **`interactive_pending`** (`EPM_AUTONOMOUS_SESSION` unset): fall through to
  the **Legacy autonomous mode** / **Interactive mode** bullets below.

Never auto-approve on a missing/ambiguous estimate — the gate parks a blank
estimate (fail safe). `awaiting_promotion` remains a human gate regardless of
this cap.

**Workflow-fix tasks — architectural greenlight (#678).**
<!-- gate: gates.plan_approval -->
A `kind: infra`
workflow-fix task (filed by the workflow-fix-on-bug protocol,
`.claude/rules/workflow-fix-on-bug.md`) is 0 GPU-h, so the GPU-h cap alone
auto-approves EVERYTHING — including architectural / public-contract changes,
which must still surface to the user. The planner flags such a change with an
`architectural: true` line in the plan frontmatter (+ an "ARCHITECTURAL — needs
user greenlight" banner in the Plan Summary). When the autonomous gate
(`EPM_AUTONOMOUS_SESSION=1`) sees
`architectural: true` it PARKS at `plan_pending` regardless of GPU-h — the
existing architectural-greenlight semantics, riding this SAME plan-approval gate
(it introduces NO new ask site; the architectural park reuses the gate the
`--auto-approve-if-autonomous` decision already owns). **Fallback** if the
`--auto-approve-if-autonomous`
gate does not yet read `architectural:` from the plan frontmatter: the
workflow-fix-on-bug protocol files the architectural task but spawns it WITHOUT
`--auto` — a bare session that parks at `plan_pending` until a human types
`/issue <N>` — so the architectural-greenlight invariant holds regardless. The
non-architectural majority (0 GPU-h, `architectural` absent/false) auto-approve
and self-merge at Step 10d, preserving the no-greenlight default. Interactive
mode is unaffected: the existing Step 2c plan-approval ask still governs a
human-present session.

- **Legacy autonomous mode** (no chat user present AND
  `EPM_AUTONOMOUS_SESSION` is unset — e.g. invoked from
  `auto-experiment-runner`): EXIT immediately; the task sits at
  `plan_pending` until a user approves via the dashboard or a future
  `/issue <N>` invocation. Before exiting, post the §5 marker:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 2c \
    --exit-kind parked --notes "plan posted; awaiting user approval"
  ```

- **Interactive mode** (user is in the current chat session): Ask the
  user inline rather than exiting. Present the plan summary and ask:

  > Plan posted as `epm:plan v1` on task #\<N\>.
  >
  > **Plan path:** `${PLAN_PATH}` (symlink -> latest version)
  > **Dashboard URL:** `https://eps.superkaiba.com/tasks/<N>/plan` (planned)
  >
  > (1) **Approve** — advance to implementation
  > (2) **Revise** \<notes\> — plan goes back to adversarial-planner
  > (3) **Defer** — exit now; re-invoke `/issue <N>` later

  `${PLAN_PATH}` is the inline shell variable captured at Step 2 — both
  steps run in the same orchestrator turn (auto-continuation guarantees
  no pause between them) so the variable is in scope. There is no
  cache-file fallback.

  <!-- gate: gates.plan_approval -->
  <!-- autonomous-mode: block-and-fail -->
  Use `AskUserQuestion` or a plain text prompt and wait for the user's
  reply. (Interactive mode only — autonomous sessions never reach this
  branch; the code-enforced gate in `task.py
  --auto-approve-if-autonomous` already decided, and the PreToolUse hook
  <!-- gate: gates.plan_approval -->
  hard-blocks any `AskUserQuestion` if reached.)

  <!-- gate: gates.plan_approval -->
  <!-- autonomous-mode: block-and-fail -->
  **Important:** when invoking `AskUserQuestion` (Interactive mode
  only), embed the dashboard URL
  (`https://eps.superkaiba.com/tasks/<N>/plan`) inside the question
  text itself, AND embed the local plan path
  (`tasks/<status>/<N>/plans/plan.md`) inside the first option's
  `description` field. The user only sees the rendered question box at
  decision time; any link that lives only in chat prose above the
  `AskUserQuestion` call gets scrolled past. The chat-prose blockquote
  above is for orchestrator narration; the call itself must be
  self-contained. Example shape (see workflow.yaml § gates.plan_approval):

  <!-- gate: gates.plan_approval -->
  <!-- autonomous-mode: block-and-fail -->
  ```python
  # Interactive mode only — autonomous branches before this point.
  AskUserQuestion(questions=[{
    "question": (
      "Approve plan v1 for task #<N>? "
      "Plan: https://eps.superkaiba.com/tasks/<N>/plan"
    ),
    "header": "Plan #<N>",
    "multiSelect": False,
    "options": [
      {
        "label": "Approve",
        "description": (
          "Dispatch <implementer-type>. Est. <cost> GPU-hours. "
          "Local plan: tasks/<status>/<N>/plans/plan.md"
        ),
      },
      {
        "label": "Revise <notes>",
        "description": "Re-run /adversarial-planner with your notes.",
      },
      {
        "label": "Defer",
        "description": (
          "Park at plan_pending. Re-invoke /issue <N> later."
        ),
      },
    ],
  }])
  ```

  - **"Approve" / "1":** move task to `approved`. Post an `epm:plan-approved`
    event for audit trail. Continue to Step 4 in the **same invocation**
    — do NOT exit:

    > **Same-issue follow-up round?** At `followups_running`, SKIP the
    > `set-status` (status-hold rule, Step 9b § Same-issue follow-up loop
    > step 3; code-enforced — `task.py` refuses the flip) and post ONLY the
    > `epm:plan-approved` marker — the approval is recorded, the status holds.

    ```bash
    uv run python scripts/task.py set-status <N> approved \
      --note "Plan v1 approved by user."
    uv run python scripts/task.py post-marker <N> epm:plan-approved \
      --note "User approved plan v1 inline."
    ```
  - **"Revise \<notes\>" / "2":** set status back to `planning`. Re-invoke
    adversarial-planner with the revision notes. Re-run the consistency
    checker. Post `epm:plan v2` via `new-plan-version`. Loop back to
    Step 2c.
  - **"Defer" / "3":** EXIT. Status stays at `plan_pending`. User
    re-invokes `/issue <N>` later to approve. Before exiting, post the
    §5 marker:
    ```bash
    uv run python scripts/post_step_completed.py --issue <N> --step 2c \
      --exit-kind parked --notes "plan_pending; user deferred"
    ```

### Step 3: Approval check (backward compat, runs on re-invocation)

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

### Step 4: Worktree + dispatch implementer

Only if status is `approved`.

**4a. Worktree + draft PR.** Create `.claude/worktrees/issue-<N>` on
branch `issue-<N>`, symlink the repo `.env` into it, and open a draft PR.
```bash
# #506-safe: from a worktree cwd, `git rev-parse --show-toplevel` returns the
# WORKTREE root and doubles the path (.../issue-<N>/.claude/worktrees/issue-<N>);
# --git-common-dir resolves to <main>/.git so dirname is the main repo root.
REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
WORKTREE="$REPO_ROOT/.claude/worktrees/issue-<N>"
bash "$REPO_ROOT/scripts/new_worktree.sh" "$WORKTREE" issue-<N> --issue <N>
# Sparse by default (~0.4G vs ~3.8G full); reuses if it exists (resume case);
# symlinks the repo .env (worktrees do NOT inherit it — RUNPOD_API_KEY /
# HF_TOKEN / WANDB_API_KEY dotenv loads fail without it).
# --issue <N> is inferred from the issue-<N> branch name when omitted
# (since #1054), but keep passing it explicitly.
```

**Sparse-worktree notes (task #596).** The worktree excludes
`eval_results/`, `external/`, `ood_eval_results/` bulk and pre-includes
this issue's own `eval_results/issue_<N>/` + `ood_eval_results/issue_<N>/`
cones (plus `eval_results/`'s immediate files, e.g. `INDEX.md`), so this
issue's artifact commits work with no ceremony. Two rules:
- **Reading another issue's eval JSONs** (parent baselines, comparison
  plots): `git -C "$WORKTREE" sparse-checkout add eval_results/issue_<M>`
  — instant. (Read-only fallback: the repo root's committed copy.)
- **Writing under a NEW dir below an excluded root** (e.g. a slug variant
  `eval_results/issue<N>_<slug>/`): run
  `git -C "$WORKTREE" sparse-checkout add eval_results/issue<N>_<slug>`
  BEFORE `git add`. A bare `git add` of an out-of-cone path fails loudly
  with "outside of your sparse-checkout definition" — the fix is
  `sparse-checkout add`, NOT `git add --sparse` (a `--sparse`-added file
  silently vanishes from the working tree on the next sparse-checkout
  mutation while staying committed).

**Worktree shell-ops rule (cwd resets between Bash calls).** The bash
tool's working directory is NOT preserved across separate calls, so a
relative `cd .claude/worktrees/issue-<N>` in one call has no effect on
the next. ALWAYS address the worktree with an absolute path or
`git -C "$WORKTREE" <cmd>` — never a bare relative `cd`. Resolve the
absolute path once with the #506-safe
`REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")`
recipe (as above) — NOT `git rev-parse --show-toplevel`, which from a
worktree cwd returns the worktree root and doubles the path — and reuse
`$WORKTREE` / `$REPO_ROOT` in every subsequent command.

**Open the draft PR only if the branch is ahead of `main`.** `gh pr
create` errors with `No commits between main and issue-<N>` when the
branch has no commits yet (the common case before the implementer has
run). Pre-check first:
```bash
if [ "$(git -C "$REPO_ROOT" rev-list --count main..issue-<N>)" -gt 0 ]; then
  gh pr create --draft --head issue-<N> --body "Closes task #<N>."
else
  echo "issue-<N> has no commits ahead of main yet; skipping draft PR (open it after the implementer commits)."
fi
```

The git PR flow is substrate-independent — we still use GitHub for code
review of the diff (not for workflow state). The PR body references the
task number for traceability, but the source of truth for task state
stays in `tasks/<status>/<N>/`.

**4b. Dispatch implementer for the task type.** No pod is touched yet —
code gets written, reviewed, and dry-run locally before any GPU is
provisioned. Spawn the appropriate agent via `Agent()`:

| Task type | Implementer agent | Output marker |
|---|---|---|
| `experiment` | `experiment-implementer` | `epm:experiment-implementation` |
| `infra` / `batch` / code change | `implementer` | `epm:results` |
| `analysis` | `analyzer` (re-analysis only) | `epm:interpretation` (analysis-only path) |
| `survey` | `general-purpose` | `epm:results` |

**Env scrub for every subagent dispatch.** EVERY `Agent()` call this
skill makes — implementer, experiment-implementer, analyzer,
code-reviewer, clean-result-critic, interpretation-critic, experimenter,
upload-verifier, follow-up-proposer, consistency-checker, planner,
critic — passes `env=scrub_subagent_env(os.environ)` from
`explore_persona_space.orchestrate.spawn_agent`. The helper strips
`GH_TOKEN` and `GITHUB_TOKEN`; every other secret (WANDB_API_KEY,
HF_TOKEN, ANTHROPIC_API_KEY, OPENAI_API_KEY, RUNPOD_API_KEY, ...)
passes through unchanged so analyzer / experimenter still reach WandB /
HF Hub / Claude. Subagents post `events.jsonl` rows via
`scripts/task.py post-marker`, which inherits the user's env from the
orchestrator's process tree. See `tests/test_subagent_env_scrub.py` for
the allow-list.

**Result side of the same every-`Agent()`-call contract — background-agent
notification bodies arrive HTML-ESCAPED.** A BACKGROUND-Agent completion
delivered via a `<task-notification>` block carries its `<result>` field
HTML-escaped by the harness (`&&`/`<`/`>` arrive as amp/lt/gt entities).
NEVER persist notification-body text into a plan / marker / artifact —
re-extract the report from the agent's DURABLE output: the file the brief
told it to write, or the notification's `<output-file>` (a transcript
JSONL, not raw text — keep the last assistant text row). Output-file text
is CLEAN and gets NO `html.unescape()`; apply exactly ONE `html.unescape()`
round ONLY to notification-BODY-sourced text (the two sources are
exclusive-or). Canonical recipe + worked extraction code:
`.claude/skills/adversarial-planner/SKILL.md` §§ "De-escape harness HTML
entities before persisting" + "Extract the output-file text via the
transcript recipe" (#952 v9, #1219; independently rediscovered by sessions
#1287 + #1288 on 2026-07-13 — the pointer this paragraph exists to spare).

Brief passed to the implementer:
- The plan path (the `plans/plan.md` symlink, NOT the body text)
- Task number + worktree path + branch name
- Code-review history if this is a revision round (`epm:code-review v<m>`)
- Required `report-back` contract — the canonical 4-H3 marker shape from
  `.claude/agents/experiment-implementer.md` Report Format + the matching
  `## Smoke run` H2 from `.claude/agents/code-reviewer.md` Steps 0.5/0.6.
  The brief MUST quote these section labels verbatim; ad-hoc alternative
  labels (e.g. `(a) Plan adherence`, `(b) Files touched`, `(c) How to
  run`, `(d) Smoke run`) cause the Codex `code-reviewer` to FAIL on
  `marker-shape` even when the implementer faithfully follows the brief.
  Canonical labels (use VERBATIM in the brief):
  - `### (a) What was done`
  - `### (b) Considered but not done`
  - `### (c) How to verify`
  - `### (d) Needs human eyeball`
  - (optional `### (e) Concerns addressed` — only when prior open
    binding concerns from `concerns.jsonl` were verified this round;
    see `code-reviewer.md` Step 0.5 + Step 0.8)
  - `## Smoke run` H2 (per Step 0.6) with one `### <phase-name>` per
    CPU-feasible pipeline phase (typical: `### data-gen`, `### training`,
    `### eval`), each carrying the exact command, the slice size, exit
    code `0`, and a one-line artifact digest. **Smoke run is its own
    `## H2` — NEVER a `### (d) Smoke run` H3.** Folding the smoke run
    into the (d) slot displaces `### (d) Needs human eyeball` and is
    itself a `marker-shape` FAIL.

  Incident: task #506 round 1 (2026-06-06) — orchestrator's ad-hoc
  labels (`(a) Plan adherence / (b) Files touched / (c) How to run /
  (d) Smoke run / (e) Needs human eyeball`) triggered the Codex
  `marker-shape` BLOCKER and the reconciler upheld FAIL, costing a
  full round of revision plus the substantive code fixes that landed
  in round 2.

  The brief MUST also carry the deferred-production-path duty: any
  deferred feature the approved plan's PRODUCTION path requires is
  persisted via `task.py raise-concern <N> --concern-id <id>
  --severity CONCERN --summary "<≤200c>" --by experiment-implementer
  --round <n>` (BLOCKER if the production path provably crashes
  without it) BEFORE posting the implementation marker — a `(d)`
  bullet is not a substitute (incident #509). Belt-and-suspenders on
  `experiment-implementer.md` § "Deferred production-path TODOs are
  persisted concerns, not (d) prose", so round-N briefs surface the
  duty without the implementer having to recall its agent spec.
- The brief MUST also carry the gate-scope verification duty (#1288):
  before posting the report marker, the implementer enumerates the Step
  9c selection from the worktree — `uv run python
  scripts/select_step9c_tests.py --json`, the DEFAULT invocation (the
  base defaults to FETCHED `origin/main` per #1289; `--base main` exists
  only to deliberately diff against the local ref, per Step 9c step 1a —
  never for this duty) — pin-sweeps the enumerated test files for every
  literal / command fragment / symbol the diff changed or deleted, and
  runs the diff-linked + pin-hit subset locally, deferring only the
  invariant-only remainder to the gate (which remains the backstop).
  Belt-and-suspenders on `implementer.md` § After Implementation item 1,
  so round briefs surface the duty without the implementer having to
  recall its agent spec (the #509 precedent).
- **Marker-version discipline — a brief NEVER instructs a literal marker
  version.** Any brief line about posting `epm:experiment-implementation` /
  `epm:results` / `epm:proposed-tests` says: "post at the next version —
  read `events.jsonl` for the highest existing version of the kind and use
  max+1, or omit `--version` (the CLI derives max+1)". Never "post as `v1`"
  or any literal `v<k>`: on a fresh task max+1 IS 1, but on a follow-up
  round / TDD resume / crash-recovery re-post prior rows exist, and an
  explicit `--version` beats the CLI's safe default (incident #825: a
  follow-up-round brief instructed a literal `v1` for the implementation
  marker on a task already at v6 — the #389 collision class). See
  `experiment-implementer.md` / `implementer.md` § Posting review-round
  markers.
- **Instruction: work ONLY inside the worktree; never touch a pod; post
  progress as `events.jsonl` rows via
  `uv run python scripts/task.py post-marker <N> epm:progress --note '...'`.**
- **If `batch`:** make ONE commit per plan section (the planner produced
  N independent sections, one per body item). Commit message format:
  `[N/M] <plan section title>` where N is the 1-indexed item and M is
  the total. Code-reviewer reviews the whole diff; this convention
  keeps the history bisectable per item if a single fix needs to be
  reverted later.
- **TDD mode (opt-in).** Set `tdd_mode=true` in the brief if EITHER:
  (a) the approved plan body contains a literal `### TDD: yes` line, OR
  (b) the task body / latest user comment in `comments.jsonl` contains
  `request-tdd`. When `tdd_mode=true`, the implementer writes tests
  first, posts them as `epm:proposed-tests v<n>` (max+1), and EXITs without writing
  implementation. This skill then parks at `running` (implementing
  sub-phase) and waits — see Resume semantics below: an `approve-tests`
  marker posted via `task.py post-marker <N> epm:approve-tests` **after**
  the `epm:proposed-tests` event is the resume signal, at which point
  this skill re-dispatches the implementer with `tdd_approved=true` and
  the implementer writes the code to make the approved tests pass. If a
  resumed `/issue <N>` finds the proposed-tests event still without
  approval, it shows the marker timestamp + the literal `approve-tests`
  instruction and EXITs again. This is the only opt-in user gate in the
  pipeline (see CLAUDE.md auto-continuation policy gate #8).

Move status to `running` (the implementing sub-phase rolls up under
`running`):

> **Same-issue follow-up round?** At `followups_running`, SKIP this
> `set-status` (status-hold rule, Step 9b § Same-issue follow-up loop step 3;
> code-enforced — `task.py` refuses the flip) — phase visibility comes from
> `stage=followup-<phase>` breadcrumbs, not status flips.

```bash
uv run python scripts/task.py set-status <N> running \
  --note "Dispatched implementer; awaiting epm:experiment-implementation."
```

Before exiting, post the §5 marker:
```bash
uv run python scripts/post_step_completed.py --issue <N> --step 4b \
  --exit-kind clean --notes "implementer dispatched; awaiting epm:results"
```
EXIT. Implementer runs autonomously.

### Step 5: Code review loop (Codex ensemble)

Only if status is `running` and the appropriate implementation marker
(`epm:experiment-implementation v<n>` for experiments, `epm:results v<n>`
for infra) is present.

This step runs an **ensemble of two reviewers in parallel** — the Claude
`code-reviewer` agent and the `codex-code-reviewer` Codex twin (gpt-5.5
via the OpenAI Codex plugin's `companion task` runtime). On verdict
disagreement (PASS-class vs FAIL), a `reconciler` agent (Claude) issues
a binding tie-break. See (see workflow.yaml § ensemble_review) for the
canonical contract.

**Round push hygiene.** Any branch push run during this loop — yours
between rounds, or the implementer's per its spec — is BARE with its exit
code checked: `git -C "$WT" push origin issue-<N>`. NEVER pipe a
push/merge through `tail`/`grep`/`head` (the `guard_piped_git_push.sh`
PreToolUse hook blocks the piped shape; a pipe masks a rejected push).
Copy the verbatim forms from Step 10d § "Bare push / merge snippets".

**5a. Spawn both reviewers in parallel (fresh contexts, single message).**

**Quota-sentinel pre-check first (#1204).** Run the canonical pre-spawn
check (CLAUDE.md § Codex ensemble review). `CODEX_QUOTA_LIVE` → spawn
ONLY the Claude `code-reviewer` this round; record the Codex twin as an
instant confirmed no-show per the Step 5d no-show fallback (no
durable-verdict probe — nothing was dispatched) and post the one-line
`epm:progress` note.

**Spec-freshness check first (worktree-cwd sessions; applies at EVERY
ensemble/agent fan-out — here, the Step 9a analyzer + critic ensembles,
and 9a-bis).** The Agent tool loads agent specs (and Skill playbooks)
from the SESSION's cwd, and a worktree cut before a later
workflow-surface fix never inherits it — so subagents silently run stale
specs for the worktree's lifetime (incident #557 r2, 2026-06-10: a
pre-hardening `codex-code-reviewer.md` copy re-enabled the retired
background-dispatch pattern and orphaned the running Codex helper).
Before dispatching, sync the worktree's workflow surface from local
`main` (the canonical commit target on this VM — fresher than
`origin/main`, no fetch needed; the check self-no-ops when the session
already runs on `main`):

```bash
# Step 5a WANTS the WORKTREE root (that is where the spec-freshness sync writes)
# — NOT the #506 path-doubling bug; do NOT change to --git-common-dir here. The
# self-no-op case (session already on main) is why show-toplevel is correct.
WT=$(git rev-parse --show-toplevel)
SPECS=".claude/agents .claude/skills .claude/rules .claude/workflow.yaml CLAUDE.md"
MB=$(git -C "$WT" merge-base HEAD main)
SAFE_SPECS=""
for f in $SPECS; do
  # Branch-side feature edits = commits since merge-base touching $f,
  # EXCLUDING prior spec-freshness sync commits (which legitimately
  # touch spec paths — without the exclusion, the first sync's own
  # commit would poison every later freshness check on the branch).
  bs_commits=$(git -C "$WT" log --oneline "$MB"..HEAD --grep='spec-freshness' --invert-grep -- "$f")
  if [ -z "$bs_commits" ]; then
    SAFE_SPECS="$SAFE_SPECS $f"
  else
    # Print the offending commits so the orchestrator can decide whether
    # to reconcile (cherry-pick main's drift on top of the branch edits)
    # or whether the branch-side touch is a global revert/port that has
    # ALREADY landed on main — in which case the skip is a false alarm
    # and the orchestrator can drop those files from the skip set by
    # hand (e.g. `git -C "$WT" checkout main -- .claude/agents/*.md`
    # after confirming the branch-side commit's content is a subset of
    # main's current state). Without these commit titles printed, the
    # operator cannot tell a legitimate branch deliverable (#535
    # incident) from a stale port/revert that needs no protection.
    echo "spec-freshness: $f carries branch-side feature edits — skipping blind sync; reconcile manually."
    echo "  branch-side commits:"
    echo "$bs_commits" | sed 's/^/    /'
  fi
done
if [ -n "$SAFE_SPECS" ] && ! git -C "$WT" diff --quiet main -- $SAFE_SPECS; then
  git -C "$WT" checkout main -- $SAFE_SPECS    # surgical refresh: workflow surface only
  git -C "$WT" diff --quiet HEAD -- $SAFE_SPECS || \
    git -C "$WT" commit -m "issue-<N>: sync workflow-surface specs from main (spec-freshness)" -- $SAFE_SPECS
fi
```

The refresh touches ONLY the workflow surface (never experiment code).
Issue branches must not carry their own workflow-surface edits as a
rule (those go through their own filed workflow-fix `/issue --auto`
sessions + worktrees), with one
legitimate exception: a feature branch whose DELIVERABLE adds
workflow-surface entries — e.g. a new marker schema registered in
`workflow.yaml` rides its feature branch (incident #535, 2026-06-10:
the blind sync clobbered the compute-router branch's four
router-marker registrations and broke the branch's own pinned
`tests/test_router.py` checks). The per-file branch-side-edit guard
above skips exactly those files (warning the orchestrator to reconcile
them manually — typically by re-applying main's spec changes on top of
the branch's additions) while everything the branch never touched
still gets the blind sync. The conditional commit keeps the worktree
clean for the Step 10d merge guards. The warning prints the offending
branch-side commit titles so the orchestrator can tell a legitimate
branch deliverable (the #535 case) from a stale port/revert whose
content has already landed on main (in which case the orchestrator can
safely override the skip for those specific files with a manual
`git -C "$WT" checkout main -- <paths>`).

**The sync scope is deliberately specs-only — do NOT extend it to
`scripts/` or `tests/`.** The sync exists because the Agent/Skill tools
load specs from the session's cwd; workflow-helper SCRIPTS are already
resolved from the MAIN checkout (Step 0 § worktree spec-freshness:
`"$REPO_ROOT"/scripts/...`), so syncing worktree copies buys no runtime
correctness. Blind-syncing `tests/` is actively unsafe: main's newer
workflow tests pin behavior implemented in main's newer `scripts/` +
`src/` (e.g. `task_workflow.py`, `backends/`) that the branch predates,
so a partial code sync makes the worktree suite REDDER or breaks the
branch's own imports — and the per-path branch-side-edit guard would
skip `scripts/`/`tests/` wholesale anyway (nearly every issue branch
adds its own `scripts/issue<N>_*.py` + tests). Operational rule
instead: a workflow test that FAILs inside a long-lived issue worktree
but PASSes at the repo root on `main` is worktree-staleness, not this
issue's breakage — cross-check at the repo root before chasing it; the
Step 10d merge resolves it (observed on #542, 2026-06-11). (A shared-infra
`src/` fix with fleet-wide blast radius — e.g. the #847 thread caps — gets a
LAUNCH-TIME fallback instead of a sync: the VM-side launch surfaces carry the
explicit thread-cap env prefix, Step 9 entry guard § "Detached VM-side long
compute phases"; #891. Do not extend this sync to `src/` allowlists — a synced
`env.py` would still miss torch-before-dotenv importers in-process, which the
launch prefix caps unconditionally.)

> **429 pacing at every ensemble fan-out (applies here, to the Step 9
> critic ensembles, and to /adversarial-planner Phase 2):** when MORE than
> two agent prompts go out at once (e.g. 3 critic lenses x 2 models), pause
> 5-10 s between Agent spawns (`sleep` is fine inside the dispatch Bash
> call, or send the spawns in 2 staggered messages). Same-second prompt
> bursts stacked onto the org-wide 4M input-tok/min cap caused 429 storms
> in 6+ sessions on 2026-06-09.

Both reviewers see the same brief:

- `issue_number` — the task number (`<N>`)
- `target_marker_kind` — exactly one of `experiment-implementation` (for
  `experiment`) or `results` (for `infra` / `batch` / `analysis` /
  `survey`). The reviewers read the highest-version row with this kind
  from `events.jsonl` as the implementer's report.
- `revision_round` — 1-indexed integer. `1` on first review; loops up to
  `3`. The cap is **per reviewer** — reconcile invocations are free.
- `previous_critique_summaries` — one-line summaries of every prior
  `epm:code-review` AND `epm:code-review-codex` event on this task
  (empty on round 1). Lets each reviewer notice patterns.
- The diff vs `main`, the approved plan (via the `plans/plan.md`
  symlink), the existing codebase.

The Claude reviewer additionally receives:
- `worktree` path, `base` ref (typically fetched `origin/main` — #1289).

The Codex twin additionally receives:
- `worktree`, `base`, `plan_marker_path` (no `implementation_marker_path`
  — the composer fetches the marker from canonical main state and INLINES
  it; likewise, if the worktree plan is absent — child task cut from a
  parent issue branch, #550 r1 — or STALE — follow-up amendment plan
  postdating the branch cut, #546 follow-up r1 — the composer inlines
  the canonical plan, Step 2-pre-b) — see
  `.claude/agents/codex-code-reviewer.md`.

**Trigger-dense (guard-surface) rounds — pre-materialize the excerpt file
BEFORE spawning (#1058/#1098).** When the round's diff or artifact under
review is trigger-dense per the `.claude/rules/trigger-dense-review.md`
recognition heuristic (guard/security hook scripts, destructive-command
fixtures, refusal/jailbreak corpora — recognition is DELEGATED to that
rule's "Fires when" block; do not re-derive it here), the orchestrator
pre-materializes the round's excerpt file and names it in BOTH reviewer
briefs — this arms the rule's discipline-3 "orchestrator-provided
pre-materialized excerpt files with stated read budgets" leg. Build is
cheap and mechanical — the trigger-dense hunks only, with file/line
anchors:

```bash
# Scope the pathspec to the trigger-dense paths actually touched
# (`git -C "$WT" diff --name-only origin/main...HEAD` first):
git -C "$WT" diff origin/main...HEAD -- .claude/hooks/ 'scripts/guard_*.sh' \
  'tests/*guard*' > /tmp/issue-<N>-r<round>-excerpts-<slug>.md
```

(For a non-diff trigger-dense artifact — a corpus or fixture file the
round must adjudicate — extract grep-anchored ≤~120-line windows into the
same file instead. Harmful BANK items stay digest-only per
`guard_harmful_bank_read.sh` — never copy bank item text into an excerpt
file. On round >1, round-scope the diff first when the branch diff is
over budget per `.claude/rules/diff-size-budget.md`.) Then add one line
to BOTH briefs (and keep it in any re-spawn brief):

`excerpt_file: /tmp/issue-<N>-r<round>-excerpts-<slug>.md — read this
INSTEAD of wholesale-reading the touched trigger-dense files; direct
reads of the originals capped at ~120-line grep-anchored windows per
trigger-dense-review.md.`

The excerpt file bounds READ volume; it does not sanitize content —
reviewers still apply discipline 1 (findings by file:line reference,
never gated literals in generated text). The same briefs (and any
re-spawn brief) ALSO carry the discipline-4 return-text contract as
one line: `return_text: verdict + marker pointer + counts only —
no findings recap (trigger-dense-review.md discipline 4)`.
Non-trigger-dense rounds: skip entirely — no excerpt file, neither
brief line.
Verdict COLLECTION on trigger-dense rounds is file-only — see § File-only
Codex verdict posting (before Step 5c).

Neither sees the implementer's reasoning — independence is load-bearing.
Dispatch in a SINGLE `Agent(...)`-call message with both spawned
`run_in_background=true` so they execute concurrently.

The Claude reviewer posts `epm:code-review v<n>` (PASS / CONCERNS /
FAIL). The Codex wrapper posts `epm:code-review-codex v<n>` (same
schema). Codex never sees `GH_TOKEN` — both wrappers post via
`task.py post-marker`.

**End-to-end smoke gate (experiment tasks).** A code-review PASS for an
`experiment` task is NOT valid on a script that was only `--help`'d or
import-checked. The reviewer MUST confirm the implementer smoke-ran
EACH PHASE of the experiment pipeline ONCE on a tiny real slice — not
just training or data-gen. "Phase" = any distinct entrypoint the
pipeline executes end-to-end (typical experiments: data-gen, training,
eval; some add separate analysis / upload steps). Eval rigs especially
must be exercised end-to-end on a tiny slice (1 seed, the minimum
contexts / cells, the base model or a tiny throwaway checkpoint) — a
never-before-run eval script that was only import-checked or that
relied on the training script's smoke is a known regression source:
shallow latent bugs (corpus-size floors, missing helpers, generator-
reuse, sentinel filters, aggregation-tuple unpacks) surface one-per-
run at the real eval phase, each costing a full pod cycle (incident:
#408 burned six relaunches catching one bug per cycle on a 203 KB
eval rig that had never been run end-to-end). For each phase, the
implementer records a sub-section under the `## Smoke run` heading
in its `epm:experiment-implementation` report — recommended layout
`### <phase-name>` (e.g. `### data-gen`, `### training`, `### eval`)
with the exact command, the slice size (how it was kept tiny), the
exit code (must be `0`), and a one-line digest of the produced
artifact (path + shape / row count). If the `## Smoke run` section is
absent, OR any phase the pipeline actually executes is missing a
sub-section, OR any sub-section shows only `--help` / `import` /
`--dry-run` evidence (or exits non-zero, or carries no artifact
digest), the reviewer posts `FAIL` with blocker `smoke-run-missing`
— it does NOT PASS on unproven code, and a never-before-run eval rig
without an end-to-end smoke is the canonical missing-phase case. But
if every phase IS present (command + exit 0 + artifact digest) and
only the *formatting* is imperfect, that is a `CONCERNS`, not a FAIL
— and Step 5c-bis strips any mechanical-contract-only FAIL once the
orchestrator verifies the evidence is genuinely present, so cosmetic
gripes about present evidence never bounce the implementer or consume a
review round. Code-only tasks (`infra` / `batch` / `analysis` /
`survey`) keep the existing test-verdict gate (Step 9c) and are
exempt from this smoke gate. Smoke commands that write under
`eval_results/` or `figures/` also carry the output-path hygiene
disposition per experiment-implementer.md
§ "Smoke outputs never overwrite committed artifacts" (scratch-dir
redirect preferred; restore-after-smoke + an empty
`git status --porcelain -- eval_results/ figures/` as the fallback);
the reviewer treats visible clobber of a committed artifact as a
substantive Critical (code-reviewer.md Step 0.6), never a strippable
mechanical blocker.

**5b. Read both markers from `events.jsonl`.**

```bash
# After both Agent tasks complete — ONE fetch, parse twice in-memory.
events_json=$(uv run python scripts/task.py view <N> --json | jq '.events')
claude_marker=$(echo "$events_json" | jq '... epm:code-review v<n> ...')
codex_marker=$(echo "$events_json" | jq '... epm:code-review-codex v<n> ...')
```

Parse each marker's `**Verdict:**` line. Acceptable values: `PASS`,
`CONCERNS`, `FAIL`. PASS-class = {PASS, CONCERNS}; FAIL-class = {FAIL}.

**Durable-verdict-first rule (fires at EVERY ensemble verdict collection:
5b here, Step 9a, Step 9a-bis, Step 9b VC, and any reconciler read).**
An Agent-tool completion result that reports an error for a reviewer /
critic / reconciler subagent — autocompact thrash death, tool-use crash,
or a garbage/empty return — is NOT, by itself, a no-show. These agents'
deliverable is DURABLE state (a marker on events.jsonl, or a written
output file), and the final summary turn regularly dies AFTER the durable
post succeeded (incident #810 r4, 2026-07-02: the code-reviewer's
`epm:code-review v4` PASS posted at 09:25:14Z, then the summary turn
thrash-died; the orchestrator misread it as a total no-show and adopted a
unilateral FAIL from the Codex verdict alone, needing a corrective marker
+ late reconciler). BEFORE invoking any no-show fallback or
single-reviewer decision:

1. Re-read canonical task state (`uv run python scripts/task.py view <N>
   --json`) for the round's expected verdict marker at the CURRENT
   version — `epm:code-review[-codex] v<n>`, `epm:interp-critique[-codex]
   v<n>`, `epm:clean-result-critique[-codex] v<n>`,
   `epm:followup-value-critique[-codex]`, or `epm:review-reconcile v<n>`.
   The mechanical form of this check is
   `task_workflow.ensemble_verdicts_present` (precedent:
   `stage_dispatch_should_skip` for the dispatch side) — run it, do not
   eyeball the events scan:

   ```bash
   uv run python - <<'PY'
   import json
   from explore_persona_space.task_workflow import ensemble_verdicts_present, list_events
   print(json.dumps(ensemble_verdicts_present(
       list_events(<N>), ["epm:code-review", "epm:code-review-codex"], <n>)))
   PY
   ```

   (Substitute the site's marker kinds; for a reconciler read pass
   `reconcile_role="<role under adjudication>"` so a same-round reconcile
   for a DIFFERENT role never satisfies the check.) `present: false` →
   proceed to item 2; `present: true, verdict: null` → the marker EXISTS
   but is malformed — item 3's malformed-output handling, NEVER a
   no-show; `present: true` with a verdict token → the reviewer RETURNED.
   Before acting on a returned verdict token, confirm the adopted note's
   head sentinel names THIS round (the predicate already treats the
   sentinel as authoritative over the drift-prone `version` field; the
   confirm is the cheap orchestrator-side double-check against a
   stale-round adoption).
2. If no marker: check the role's durable output file — the EXACT
   `--output-file` path this round's dispatch config named
   (role+round-specific conventions:
   `/tmp/codex-code-reviewer-<N>-r<round>-output.md`,
   `/tmp/codex-interp-critic-<N>-r<round>-output.md`,
   `/tmp/codex-clean-result-critic-<N>-r<round>-output.md`,
   `/tmp/codex-followup-critic-<N>-output.md`; NEVER a guessed generic
   path). The file counts as a durable verdict ONLY if BOTH: (i) it
   carries the role's expected marker start/end tags at the CURRENT
   round version, AND (ii) it is round-fresh — a current-round
   `epm:codex-task-completed` marker exists for this dispatch, OR the
   file mtime postdates this round's `stage-dispatch` breadcrumb /
   `epm:codex-task-spawned`. A file failing either test is NOT a durable
   verdict — a conforming-looking file from a PRIOR round is the trap
   this clause exists to block.
3. If a durable verdict exists and CONFORMS (expected marker kind +
   current version + a parseable `**Verdict:**` line), the reviewer
   RETURNED: use the durable verdict and apply the normal ensemble rule
   — reconciler on disagreement, never a unilateral decision. A
   truncated file (a FAIL-class `**Verdict:**` line with no blocker
   body) is MALFORMED, not a verdict — route it to the role's
   malformed-output handling, never adopt it. Precedence when signals
   coexist: a current-round posted verdict MARKER wins over everything;
   a current-round posted `epm:failure` from the wrapper wins over a
   bare conforming FILE (the wrapper inspected its own output and
   judged it malformed); a conforming round-fresh file wins over
   nothing.
4. Only when NO durable verdict exists does the role's no-show handling
   fire. For a Codex twin: the Step 5d fallback (single-Claude
   decision), exactly as if `epm:failure` had been posted. For a CLAUDE
   reviewer/critic: there is NO fallback — first diagnose the death
   (e.g. an over-budget diff per `.claude/rules/diff-size-budget.md`;
   thin the brief accordingly), then re-spawn it ONCE per
   role+round+version — the re-spawn posts at the SAME `v<n>` and does
   NOT increment the per-reviewer round counter (a 429-kill is already
   covered by the SubagentStop retry rule and consumes the same
   allowance). If the re-spawn ALSO ends with no durable verdict, fail
   LOUD: interactive — surface to the user; autonomous — post
   `epm:failure v1` (`failure_class: infra`, reason:
   reviewer no durable verdict after bounded re-spawn), set
   `status:blocked`, PushNotification, CRON-TEARDOWN. NEVER adopt a
   unilateral decision from the surviving reviewer.

**Autocompact-thrash respawn recipe (refines item 4's "first diagnose
the death" for ANY thrash-killed subagent — reviewer/critic per item 4,
and equally an implementer / fact-checker / analyzer re-spawn).** Check
the dead spawn's transcript/result for an OVERSIZED tool result (a
multi-hundred-KB diff or file read). If ONE EXISTS, the read-side fix
applies: bound the read / thin the brief per
`.claude/rules/diff-size-budget.md`. If NONE exists, the pressure is
FIXED OVERHEAD on the subagent window (spec + CLAUDE.md import tree +
MCP schemas + the brief) — or accumulated read VOLUME no single-read
bound addresses — and re-tightening read bounds does NOT help: respawn
instead with (i) MICRO-SCOPED work — split the role's work into the
smallest self-contained unit (#1090 split one implementer build into
sequential rounds A/B — round A returned a commit manifest with NO
implementation marker; round B posted the marker after the full
smoke) — and (ii) the DEFAULT session model — do NOT pin a smaller
model as a thrash fix (#1090 forensics, events.jsonl L247: "transcript
forensics show NO oversized tool result (max 15KB line): the thrash is
FIXED-OVERHEAD pressure on the subagent window, not read indiscipline";
"read-bounded brief did not help"; "both default-model spawns today
compacted successfully; 3/6 sonnet spawns thrashed"). Multi-unit splits
apply to roles whose deliverable DECOMPOSES (an implementer or
fact-checker build); a single-verdict reviewer/critic re-spawn stays
ONE spawn, micro-scoped by brief. Per-subagent model pins remain
prompt-cache-safe and legitimate for OTHER reasons (the CLAUDE.md
refusal rung (b2) sonnet pin) — they are just not a thrash remedy. The
micro-scoped respawn IS item 4's one bounded re-spawn where item 4
applies (same `v<n>`, no round-counter increment); for a multi-unit
split, the units run sequentially within that same round (#1090: "Same
round counter (round 1 continues; re-spawns do not increment)").

The existing marker-keyed no-show path — the Codex wrapper POSTING
`epm:failure v<m>` (`failure_class: codex-output-malformed` or `infra`)
— is itself durable state and is UNCHANGED: that marker IS a confirmed
no-show. This rule governs only the Agent-tool-RESULT-keyed inference.
(The dispatch-side sibling is the Step 9 pre-dispatch dedup /
`stage_dispatch_should_skip`; the resume table is already
durable-marker-keyed. This rule closes the live verdict-collection gap
between them.)

**File-only Codex verdict posting on trigger-dense rounds (fires at EVERY
marker-mode Codex verdict collection: 5b/5c here, Step 9a, Step 9a-bis,
Step 9b VC; #1275).** When the round is trigger-dense per the
`.claude/rules/trigger-dense-review.md` "Fires when" heuristic (same
recognition the Step 5a excerpt-file paragraph delegates to — do not
re-derive it here), the orchestrator posts the Codex twin's verdict
marker from its output file WITHOUT paging the findings-bearing body
into context — `post-marker --file` needs no full read. This is the
orchestrator-side sibling of discipline 4 (#1252 covered the reviewer's
return text; #1152's wedge shape applies equally to a wholesale
orchestrator read of the same findings). Mechanics (the composer's
return block already names the exact start/end tags and output path):

```bash
OUT=/tmp/codex-<role>-<N>-r<round>-output.md   # the EXACT dispatched --output-file path
MB=/tmp/issue-<N>-<kind>-r<n>-marker.md        # the extracted marker block
# 1. Marker block FIRST — mechanical tag-window extraction (tags verbatim from
#    the composer's "Marker start tag:" / "Marker end tag:" lines; LINE-START
#    anchors so a mid-prose quoted tag mention can never open/close the window):
sed -n '/^<!-- epm:<kind> v<n>/,/^<!-- \/epm:<kind> -->/p' "$OUT" > "$MB"
# 2. Gate: end tag present + under the 50,000-char note cap. A missing end
#    tag or empty extraction = MALFORMED output -> the site's existing
#    stricter-retry re-dispatch (cap 2), never a Read to "see what happened".
grep -q '^<!-- \/epm:<kind> -->' "$MB"
wc -c < "$MB"   # >=50000 -> the existing artifacts-file oversize fallback
# 3. Decision inputs for the ensemble tables — grep the EXTRACTED block (its
#    verdict line is the authoritative one; grepping "$OUT" would let a
#    pre-block template echo win -m1), never Read:
grep -m1 '^\*\*Verdict' "$MB"            # single-verdict sites (5c / 9a / 9a-bis)
grep -E '^### Proposal|^\*\*Verdict' "$MB"   # 9b VC only: per-proposal verdicts — no -m1
grep -m1 '^\*\*Blocker tags:' "$MB" || true  # sites that carry it (5c-bis / 9a-bis strips)
# 4. Post without reading:
uv run python scripts/task.py post-marker <N> epm:<kind> --version <n> \
  --file "$MB"
```

The Step 5b item-2/3 durable-verdict probes (start/end tags at the
current round version; parseable `**Verdict:**` line; round-freshness
via marker/mtime) are grep/`stat` probes — on a trigger-dense round run
them mechanically for Claude reviewer output files too, never via
`Read`. When findings DETAIL is genuinely needed downstream (an
implementer bounce brief's union-blocker list, a reconciler brief), pass
the findings BY REFERENCE — the posted marker kind + version on
events.jsonl and/or the dispatched output-file path under /tmp — so the subagent reads
them itself with windowed grep-anchored reads (trigger-dense-review.md
discipline 3); do not inline verdict bodies into briefs on such rounds.
EXEMPT: in-context sites — adversarial-planner Phase 2 lens critics and
any composer returning `Posting mode: in-context` — where the verdict
body IS the deliverable merged into context (discipline 1 bounds its
content). Non-trigger-dense rounds: unchanged — reading the output file
remains fine.

**5c. Apply ensemble decision rule.**

| Claude verdict | Codex verdict | Action |
|---|---|---|
| PASS-class | PASS-class | **Agree.** `final_verdict = PASS`. CONCERNS bullets from either reviewer surface to the implementer as opportunistic suggestions; do not block. |
| FAIL | FAIL — overlapping blockers | **Agree.** `final_verdict = FAIL`. Bounce to implementer (one round). |
| FAIL | FAIL — disjoint blockers | **Union, no reconciler.** Build a combined blocker list (Claude's blockers ∪ Codex's blockers) — INCLUDING every `### Bug-class sweep: <class>` sibling enumeration from either verdict — and pass it to the implementer in the next-round brief. No new marker — both `epm:code-review v<n>` and `epm:code-review-codex v<n>` already exist on the task. `final_verdict = FAIL`. Bounce (one round). |
| PASS-class | FAIL (or vice versa) | **Disagreement.** Spawn `reconciler` agent (Claude, fresh context). Brief: role=`code-reviewer`, task=N, round=n, both event bodies (trigger-dense round: BY REFERENCE — marker kind+version / output-file paths, per § File-only Codex verdict posting; the reconciler reads them itself with windowed reads), diff path (+ the Step 5a excerpt-file path + read budget on a trigger-dense round). Reconciler reads both verdicts + the artifact, posts `epm:review-reconcile v<n>` with binding PASS or FAIL. `final_verdict = reconciler's verdict`. |

The reconciler may NOT add findings beyond what either reviewer raised —
its job is adjudication only. Round counter does NOT increment for
reconciler invocations.

When BOTH reviewers returned disagreeing durable verdicts, adopting the
MORE SEVERE verdict WITHOUT spawning the reconciler is
UNSANCTIONED at every doubled site — even when the flagged residual is
mechanically verifiable (#825 skipped the reconciler on exactly that
rationale) — because a true residual does not determine severity (the
reconciler may legitimately side PASS on a true-but-not-verdict-changing
finding), and the shortcut trades a FREE adjudication (reconcile rounds
don't count) for a revision round that DOES count against the cap-5 and
itself costs ≥3 spawns (analyzer + both critics) vs the reconciler's
one, while leaving a possibly over-strict reviewer unadjudicated. The
documented adopt-more-severe last-resort fail-safe (a spawned reconciler
errors, is re-spawned once, and still returns no parseable verdict)
belongs to the `/adversarial-planner` § Durable-output-first IN-CONTEXT
Phase-2 reconciler ONLY — at the marker-mode sites here a twice-dead
reconciler fails LOUD per the Step 5b durable-verdict-first rule
(item 4), never adopt-more-severe. The Codex no-show fallback
(single-Claude decision on confirmed no-show) is a different, sanctioned
path — it adjudicates nothing and adopts no "more severe of two".

**5c-bis. Mechanical-contract-only FAIL strip (anti-gate-hopping).**

A FAIL is *mechanical-contract-only* when its `**Blocker tags:**` line
(reviewer Step 7 template) is a non-empty subset of {`marker-shape` (Steps
0.5 / 0.55), `smoke-run-missing` (Step 0.6), `git-provenance` (Step 0.9)} and does
NOT contain `substantive`
(any code / plan / test / security finding). The `**Blocker tags:**` line is
the parse target; if a legacy verdict omits it, fall back to reading the
Critical-section prose for the same tag strings. Apply this strip BEFORE the
Step 5c rule whenever a reviewer's verdict is FAIL. The
orchestrator does its own cheap, mechanical check of the highest-version
implementer marker (`epm:experiment-implementation` / `epm:results`) in
**canonical task state** — `uv run python scripts/task.py view <N> --json`,
the main-branch `events.jsonl`, NOT a possibly-stale worktree copy a reviewer
may have read. (A reviewer FAILing on "marker missing" while reading a stale
worktree `events.jsonl` — before the implementation marker was pulled in — is
the most common false absence; the canonical read is what catches it.) No LLM
judgment, just structural presence:

- **marker-shape:** three sub-recipes, keyed PER BLOCKER on the blocker body
  (a conforming Step 0.55 blocker names exactly ONE marker kind,
  `epm:smoke-architecture-check`; a conforming Step 4.6 presence blocker
  names `Gate-scope check` ONLY — never a combined 0.5 + 0.55 + 4.6
  blocker).
  When the blocker names `epm:smoke-architecture-check` (Step 0.55): a
  separate `epm:smoke-architecture-check` events row exists in canonical task
  state with a `verdict:` line matching `PASS_UNIFIED` | `PASS_CANARY
  canary_cell=<id>` | `FAIL_NO_CANARY` — present + parseable → STRIP (a
  stale-worktree false absence); absent or verdict-less → leave the FAIL in
  place (the gate is doing its job; do NOT check the implementation marker's
  H3s for this sub-case — they can be conforming while the separate row is
  missing, which is exactly incident #811).
  When the blocker names `Gate-scope check` (Step 4.6 presence): the `(c)`
  section of the highest-version `epm:results` marker in canonical task
  state carries a `Gate-scope check` line — present → STRIP (a
  stale-worktree false absence; the strip verifies PRESENCE ONLY — a
  diff-consistency finding is `substantive` per Step 4.6 and never
  reaches this recipe); absent → leave the FAIL in place (the gate is
  doing its job). Otherwise (the Step 0.5 default):
  all four H3 sections `(a)`–`(d)` present with non-empty content AND `(c)`
  carries at least one fenced command.
- **smoke-run-missing:** a `## Smoke run` section is present, and EVERY phase
  the pipeline actually executes (typically data-gen, training, eval) has its
  own sub-section with a command, exit code `0`, and an artifact digest. A
  `## Smoke run` section that covers only one phase (e.g. training) while the
  pipeline also runs a separate eval rig is genuinely absent for the missing
  phase — leave the FAIL in place.
- **git-provenance:** the orchestrator reads the blocker's
  `**Git-provenance subclass:**` line and runs the matching read-only git
  probe from repo root (or against the branch ref `issue-<N>`, never by
  switching the repo-root branch — CLAUDE.md hard rule):
  - `pre-existing-on-trunk` → `git show main:<path>` resolves AND the round's
    own commit range (`git show <round-sha>~1..<round-sha> -- <path>`, or the
    implementer report's `<parent>..HEAD`) does NOT touch the flagged lines →
    the violation is on trunk, not from this round → STRIP.
  - `stale-main-or-worktree` → `git log --oneline main..issue-<N> -- <path>`
    returns zero non-merge commits (branch never touched the file) → the
    finding is a stale-branch artifact → STRIP.
  - `cumulative-main-head-diff` → the flagged line is unchanged in the round's
    OWN range (`git show <round-sha>~1..<round-sha> -- <path>` /
    `<parent>..HEAD`) even though it appears in `origin/main...HEAD` → out of round
    scope → STRIP.
  In ALL THREE: the strip fires ONLY when the git probe CONFIRMS the finding is
  not from this round's diff. If the probe shows the round's own range DID touch
  the flagged lines (git says the round introduced it), the strip does NOT fire
  — leave the FAIL in place and apply the normal Step 5c rule. This is
  evidence-based, never a blanket ignore. Merge-base errors on a sparse/shallow
  worktree (`fatal: origin/main...HEAD: no merge base`) are a checkout artifact — fall
  back to the two-dot / round-SHA range per code-reviewer.md Step 0; a "no merge
  base" error is never itself grounds to strip OR to FAIL.

Then:

1. **Artifact genuinely absent / non-conforming** → the gate is doing its
   job. Leave the FAIL as-is and apply the normal Step 5c rule.
2. **Artifact present + conforming** → the mechanical blocker is a false
   positive on cosmetics. STRIP it from that reviewer's effective blocker set,
   then apply Step 5c to the REMAINING (substantive) blockers from both
   reviewers:
   - No substantive blockers remain from either reviewer → `final_verdict =
     PASS`. Log to chat as one line: `mechanical-contract-only FAIL stripped —
     orchestrator verified <artifact> present + conforming; no substantive
     findings → PASS.`
   - Substantive blockers remain → normal Step 5c FAIL / union / reconciler on
     those only.

This is bounded: the orchestrator may strip ONLY a mechanically-verifiable
contract blocker (it is checking a structural fact, not overriding a
code-substance judgment) — for `git-provenance` the "structural fact" is the
read-only git probe confirming the flagged state is NOT introduced by the
round's diff (a git-history fact, same bounding logic as
marker-shape/smoke-run-missing), never a code-substance judgment. It directly
closes the gate-hopping failure mode —
a reviewer that FAILs round after round on the *presentation* of evidence the
marker demonstrably contains (e.g. round 1 marker-shape, round 2 smoke-digest
formatting, never reviewing the code) can no longer bounce the implementer or
consume a cap-5 round (the strategy pivot is retired; the strip still prevents
the round counter from incrementing). The round counter does NOT increment
for a strip. The clean-result-critique loop (Step 9a-bis) carries the same
strip for *presentation-only* verifier FAILs (MDX prose, caption shape,
cherry-label phrasing) — a clean-result FAIL backed only by presentation
nits is likewise stripped + patched inline rather than consuming a REVISE
round.

**5c-ter. Binding-concerns post-strip check (composed onto 5c-bis by task #455).**

After Step 5c-bis has stripped any mechanical-contract-only FAILs, AND
the per-reviewer verdicts have been resolved by Step 5c, run a final
binding-concerns check BEFORE advancing on `final_verdict == PASS`:

```bash
open_concerns=$(uv run python scripts/task.py list-concerns <N> --open-only --json)
```

If `open_concerns` is empty: advance per Step 5d as usual (the
historical PASS path is unchanged).

If `open_concerns` is non-empty AND `final_verdict == PASS`, iterate
per concern_id:

- **severity=NIT** → opportunistic, never blocks. Skip.
- **severity=CONCERN** → either:
  1. The current implementer round demonstrably addressed it AND the
     reviewer's verdict body (or the orchestrator's own diff inspection)
     confirms — call `task.py address-concern <N> --concern-id <id>
     --by code-reviewer --round <n>` (recording verification) and
     advance; OR
  2. **Interactive mode only:** raise inline `AskUserQuestion` <!-- gate: gates.concern_deferral_request --> proposing deferral. On user
     agreement run `task.py defer-concern <N> --concern-id <id> --by
     user --rationale "..."` (≥40 chars, not boilerplate) and advance;
     on user refusal bounce to the implementer with a brief targeting
     that concern (round counter increments).
  3. **Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): NEVER raise the
     deferral ask AND never print the per-concern options as text. Auto-
     resolve per § Autonomous session behavior →
     `concern_deferral_request`: bounce to the implementer for one more
     round targeting the open CONCERN(s). State
     `Decision: bounce to implementer (concern_id=<id>) — autonomous
     mode never defers` AND EXECUTE the bounce in this same turn (spawn
     the implementer agent with a brief targeting the concern_id); do
     NOT state the Decision and then end the turn.
- **severity=BLOCKER** → either address (option 1 above) OR apply the
  cap-hit rule per `pivot_criteria.code_review_ensemble_cap_5_surface`
  (at cap-5: strip → all-stripped PASS+continue OR surface a substantive
  residual). BLOCKERs CANNOT route to the deferral gate. If it cannot be
  addressed and the residual is substantive, post `epm:failure v1
  failure_class: code` referencing the concern_id and set status:blocked
  (halt_criteria id=6 `concern_unresolved`).

Multiple open CONCERNS may batch into ONE `AskUserQuestion` call <!-- gate: gates.concern_deferral_request --> <!-- autonomous-mode: skip --> with
one option per concern_id plus a free-text rationale box per concern.
(Interactive mode only — autonomous mode bounces to implementer per
the per-concern rule above; the batch ask is never raised.)

This step does NOT override 5c-bis — mechanical-contract-only FAILs
still strip and cosmetic gripes about present evidence still don't
bounce the implementer. The check operates on a different signal
(concerns.jsonl persisted via `task.py raise-concern` — NOTE the
`--summary` arg is hard-capped at 200 chars, ValueError above it; put
detail in `--evidence`) and gates
auto-advance ON TOP of the existing flow. The same subroutine fires at
Step 9a (interp ensemble) and Step 9a-bis (clean-result ensemble) with
the same logic.

**5d. Loop on FAIL using `final_verdict`.**

- **`final_verdict == PASS`**:
  - `experiment` -> stay at status `running` (entering the workload
    sub-phase), proceed to Step 6.
  - `infra` / `batch` / `analysis` / `survey` -> skip pod phase, move
    status directly to `reviewing` (the inline test-verdict gate at
    Step 9c runs from there).
- **`final_verdict == FAIL` + revision_round<5** -> stay at status
  `running` (implementing sub-phase). Re-spawn the implementer with
  BOTH event bodies (Claude + Codex) AND the reconcile event (if
  present) as part of the brief. **When either reviewer verdict (or the
  disjoint-blocker union) contains a `### Bug-class sweep: <class>`
  enumeration, thread the FULL sibling list — every enumerated
  `file:LINE`, not just the top finding — into the implementer's
  punch-list brief, so the round-N+1 edit is class-scoped and the
  implementer's class-hardening carve-out (experiment-implementer.md
  revision-round rule) fires on the whole class.** Implementer posts
  v<n+1>; loop back to 5a with `revision_round = n+1`.
- **`final_verdict == FAIL` + revision_round>=5** -> **CAP-HIT:
  strip-then-continue-or-surface** (replaces the retired cap-3 strategy
  pivot; see CLAUDE.md "STATE-TO-`blocked` criteria" and workflow.yaml
  § pivot_criteria.code_review_ensemble_cap_5_surface). At round 5 (the
  cap) with a non-PASS ensemble verdict, the orchestrator:
  1. **Applies the FULL Step 5c-bis strip once more** — the
     mechanical-contract-only set {`marker-shape`, `smoke-run-missing`,
     `git-provenance`}, evidence-based as always (git-provenance runs the
     read-only git probe matching the blocker's declared subclass).
  2. **If ALL residual blockers are stripped** (false-positive /
     mechanical / git-provenance) → treat as PASS and CONTINUE (proceed
     per the `final_verdict == PASS` branch above). Log one chat line +
     post an `epm:progress` note recording the cap-5-strip-continue
     outcome (which blockers were stripped and by what verification).
  3. **If ANY substantive residual remains** (a real finding the strip
     cannot verify away — silent-failure, upload-path/artifact-loss,
     missing checkpoint-per-phase, resource-leak, scaffolded-but-unplumbed
     pipeline, producer/consumer key mismatch, missing/incomplete smoke,
     estimand/headline-poisoning) → **SURFACE** it. Do NOT ship past it,
     do NOT same-diff-family pivot-loop:
     - **Interactive mode:** present the residual blocker(s) to the user
       (the two-path escalation is grandfathered for a genuine stuck-real
       blocker; frame the residual + ask how to proceed). Post the §5
       marker (`uv run python scripts/post_step_completed.py --issue <N>
       --step 5b --exit-kind parked --notes "code-review cap-5
       substantive residual; awaiting user"`), then EXIT awaiting
       the user.
     - **Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): post
       `epm:failure v1` with `failure_class: code` referencing the
       residual blocker(s), set `status: blocked`, fire
       `PushNotification({"message": f"#{N} BLOCKED: ensemble review real
       residual at cap-5 — open it"[:200], "status": "proactive"})`, run
       CRON-TEARDOWN (fresh `CronList` → `CronDelete` every job in the
       two-leg match set: the recurring `/issue-tick <N>` tick +
       stray one-shot `/issue <N>` wakeups; not-found = success;
       § CRON-TEARDOWN procedure), post the §5 marker (`uv run python
       scripts/post_step_completed.py --issue <N> --step 5b --exit-kind
       failure-exit --notes "code-review cap-5 substantive residual;
       status:blocked"`), and EXIT. This is the standing halt path for a
       genuinely-stuck real blocker after the auto-continue space is
       exhausted (halt_criteria id=6 `concern_unresolved` family) — no
       more pivots, no more silent shipping past.

  For a plan that is ITSELF internally contradictory, the
  `plan_contradiction_replan` pivot (Step 7 / § pivot_criteria) still
  applies — that is a different signal (the plan is the defect), not a
  code-review cap-hit. Likewise the whack-a-mole detector (Step 5.bis(b))
  is unchanged; the retired pivot is specifically the "same diff family
  failed N rounds → re-plan" one.

**Codex twin no-show fallback.** If the Codex wrapper posts
`epm:failure v<m>` with `failure_class: codex-output-malformed` or
`failure_class: infra` (codex plugin missing), proceed with
single-reviewer (Claude-only) decision-making for that round. Do NOT
block on the Codex twin's absence; cap-5 still applies to the Claude
reviewer's count. Surface this to chat as one line: `Codex twin no-show
this round; using Claude reviewer only.` This fallback fires ONLY on the
posted `epm:failure` marker, or after the Step 5b durable-verdict-first
rule confirms NO durable verdict exists (no `epm:code-review-codex v<n>`
marker AND no conforming, round-fresh output file). An Agent-tool error
result alone never triggers it — and the same applies symmetrically to
the Claude reviewer: with no durable verdict, re-spawn it once per the
Step 5b rule; NEVER adopt a unilateral decision from the surviving
reviewer (incident #810 r4). An `epm:codex-task-failed` note carrying
`codex-quota-exhausted` is the org-quota outage short-circuit (#1126):
treat as an instant no-show (Claude-only), do not re-dispatch or
investigate; the sentinel self-expires at the stated reset. The Step 5a
pre-spawn sentinel check (#1204) makes this fallback fire WITHOUT
spawning the composer: a sentinel-skip recorded at spawn time is a
confirmed no-show — do NOT run the durable-verdict probe for a round
whose composer was never spawned, and do not wait for any
`epm:codex-task-*` marker (none will exist).

##### Step 5.bis: Pre-dispatch checks (compute-deviation + whack-a-mole)

Fires once per implementer round, AFTER code-review-PASS, BEFORE any
pod-provision or experimenter-dispatch action in Step 6. Two
independent triggers run in sequence:

**5.bis(a) — Compute-deviation pivot.** Scan the task's
`events.jsonl` for `epm:compute-deviation v1` markers posted in the
current implementer round (highest version with the same round number).
If present:

1. Parse the marker's body for `component`, `planned_wall_h`,
   `projected_wall_h`, `ratio`, `basis`. Route on the component's
   marker CHAIN (re-posts reuse the planner-§9 row name verbatim in
   `component:` — the loop guard and this routing key on it):
   - `action: auto_descope_to_<spec>` present AND the component's
     chain also carries a lever-0 record (`action: vectorize_fix_round`
     ran, or `signature_check: negative`) → a prior tick already
     accepted an auto-descope with lever 0 resolved; log one line and
     advance to Step 5.bis(b). An `auto_descope_to_<spec>` WITHOUT a
     lever-0 record is a pre-resolution like any other — treat as
     UNRESOLVED and proceed to step 2 (a descope never resolves
     lever 0; legacy and implementer self-descope markers arrive
     exactly this way).
   - `action: vectorize_fix_round` present (the fix round ran) and the
     post-fix ratio ≤ 2× → log one line and advance to Step 5.bis(b).
     Post-fix ratio still > 2× → SKIP step 2 (one mandatory fix round
     per component) and go to step 3 with the post-fix numbers plus
     the round's residual classification.
   - ANY pre-resolution (`action: continue_as_is`,
     `action: auto_descope_to_<spec>` per the bullet above, or any
     other legacy, poster-side, or crash-replay resolution) WITHOUT a
     lever-0 record (`vectorize_fix_round` ran, or
     `signature_check: negative`) → treat as UNRESOLVED and proceed to
     step 2; at ratio ≥ 5× without a valid clause-0c finding the
     pre-resolution is VOID.
2. **Vectorize-first signature check (pivot_criteria auto-action 0 +
   0b — REQUIRED before any descope).** From `basis:` + the round's
   implementer report, classify the deviation:
   - **Overhead-bound** (matches the `.claude/rules/vectorize-many-cell-fits.md`
     trigger — the canonical definition; illustratively: a serial
     per-cell/fold/layer/draw/row loop of small fits/solves/reductions,
     batch-1 model forwards, per-draw re-reduction of a fixed pool,
     per-row IO, or sequential shard-independent cells with an unused
     parallelism axis) → dispatch ONE vectorize/parallelize fix round:
     spawn `experiment-implementer` with a brief naming the marker,
     the rule + canonical helpers
     (`src/explore_persona_space/analysis/vectorized_mlp_skill.py`,
     `src/explore_persona_space/analysis/null_battery.py`), the
     equivalence gate against a SEEDED serial oracle (2-3 cells, a
     stated per-workload float tolerance), and the requirement that
     its closing `epm:compute-deviation v<next>` re-post carry
     `action: vectorize_fix_round`, the post-fix projection, AND the
     residual classification (a genuinely FLOP-bound / dependency-
     bound residual is recorded in `flop_bound_finding:` — that
     post-fix arithmetic constitutes the clause-0c finding). Dispatch
     for the component HALTS while the fix round runs — Step 6 is not
     reached for it this round; the round flows through code-review
     (Step 5) normally. Descope and `continue_as_is` are NOT eligible
     for the component until this round has run or a negative finding
     is recorded. Pinned plan hyperparameters do NOT exempt the lever —
     vectorization is recipe-preserving (equivalence-gated); pinning
     blocks descope, not vectorization (#722).
   - **Not overhead-bound** (`basis` names a genuinely FLOP-bound /
     API-latency / bandwidth / capacity-wait / already-vectorized-
     contention cause) → post `epm:compute-deviation v<next>` with
     `signature_check: negative` + 1-3 lines of arithmetic or the
     named quantified bottleneck, and proceed to step 3.
   - **Ambiguous basis** → treat as overhead-bound; the fix round's
     first action is the rule's diagnostic (FLOP back-of-envelope,
     cputime/walltime ratio) and the round may return the negative
     finding instead of a code change.
   **Pod release (0b):** if the deviating serial phase holds a GPU pod
   (or the pending dispatch would hold one idle through it), run
   `pod.py stop --issue <N>` while the fix round runs; the
   orchestrator that issued the stop owns `pod.py resume` at fix-round
   completion, before any re-dispatch (CLAUDE.md "CPU-only phases
   don't hold GPU pods"; this is continuing work, not parking).
3. Otherwise, attempt auto-descope per
   `workflow.yaml § pivot_criteria.compute_deviation_over_2x`:
   walk the planner's §9 stratification dimensions in priority order
   (seeds → framings → cells-per-stratum); for each dimension, compute
   the descoped projection (drop the dimension to its min-N-for-power
   per the planner's §9 stratification spec). The first descope whose
   ratio ≤ 1.5× AND keeps every dimension ≥ its min-N wins.
4. **Auto-descope success.** Post `epm:compute-deviation v2` with
   `action: auto_descope_to_<spec>`, update the implementer's per-cell
   parameters in the launch command, log to chat as one line, advance.
5. **Auto-descope fails** (no dimension keeps ratio ≤ 1.5× while
   staying above min-N): branch on session mode.

   - **Interactive mode** (`EPM_AUTONOMOUS_SESSION` unset/falsy): surface
     `gates.conditional.compute_deviation_resolution` (id=12) with the
     2-option prompt. Quote the ratio inline. At ratio ≥ 5×,
     `continue_as_is` requires the recorded quantified clause-0c
     finding (`flop_bound_finding:` on the marker, or a
     `signature_check: negative` record meeting the 0c bar) — state it
     inline. On `continue_as_is`,
     advance to Step 5.bis(b) with the original parameters. On
     `accept_descope_to_<X>_with_caveats`, post `epm:compute-deviation v2`
     with the chosen descope spec + caveats and advance.

     <!-- gate: gates.conditional.compute_deviation_resolution -->

   - **Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): NEVER raise the
     ask AND never print the two options as a text menu. Auto-resolve
     per § Autonomous session behavior →
     `compute_deviation_resolution` (reachable only after the step-0
     lever is resolved; see that bullet for the full rule): pick
     `accept_descope_to_<X>_with_caveats` if any descope dimension
     preserves majority statistical power (≥0.6 of the planned cells);
     else `continue_as_is` and quote the projected ratio inline — at
     ratio ≥ 5×, `continue_as_is` additionally requires the recorded
     quantified clause-0c finding (`flop_bound_finding:` on the marker,
     or a `signature_check: negative` record meeting the 0c bar); if it
     is missing and no fix round ran, execute step 2's vectorize fix
     round instead; if the
     fix round ran but its re-post omitted the residual classification,
     obtain ONE corrective re-post (no second fix round), then resolve.
     State `Decision: <choice> because <reason>` AND EXECUTE the
     resolved action in this same turn (post `epm:compute-deviation v2`
     with the chosen `action:` and advance to Step 5.bis(b)); do NOT
     state the Decision and then end the turn.

**5.bis(b) — Whack-a-mole detector.** Scan the task's `events.jsonl`
for `epm:new-bug-class v1` markers posted in the trailing 5
implementer rounds (rounds N-4..N, where N is the current round).
EXCLUDE rounds whose `epm:experiment-implementation v<n>` event note
contained the regex `<!-- workflow-fix-candidate v1 -->` (per the
workflow-fix-on-bug protocol; those drive the workflow-fix-task-filing
default — a filed `kind: infra` task + a `/issue --auto` session — not
strategy-pivot consideration). "Consecutive" below means consecutive
across NON-EXCLUDED rounds — i.e. when an excluded round sits between
two tagged rounds, the excluded round is skipped, and the two tagged
rounds count as consecutive for the trigger.

Two triggers:
- **PRIMARY:** 3 distinct `bug_class` tag values across the 3 most
  recent non-excluded rounds (each contributed a distinct tag).
- **SECONDARY:** 2 distinct `bug_class` tag values across the 2 most
  recent non-excluded rounds AND at least 1
  `epm:compute-deviation v1` event in the trailing 5 rounds (N-4..N).

On fire, branch on session mode.

**Interactive mode** (`EPM_AUTONOMOUS_SESSION` unset/falsy): surface
`gates.conditional.whack_a_mole_pivot` (id=11) with 2 options:
- `continue-as-planned` (one-line rationale + cost estimate of the
  next pod-provision + experimenter dispatch).
- `pivot-to-<X>` (one-line rationale + cost estimate of the canonical
  alternative the implementer's report named, e.g. unification of
  smoke + sweep paths).

On `continue-as-planned`, advance to Step 6 normally; round counter
does NOT reset. On `pivot-to-<X>`, route back to `status:planning`
for re-planning; round counter does NOT increment (this is a
strategy pivot, not a fresh review round).

**Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): NEVER raise the ask
AND never print the two options as a text menu. Auto-resolve per §
Autonomous session behavior → `whack_a_mole_pivot`: pick `pivot-to-<X>`
if the implementer's report named a canonical alternative AND the next
round on the current path would burn >2× the cost of the pivot; else
`continue-as-planned`. State `Decision: <choice> because <reason>` AND
EXECUTE the resolved action in this same turn — on `pivot-to-<X>`:
`task.py set-status <N> planning` + re-invoke `/adversarial-planner`
with the pivot scope (round counter does NOT increment; mid same-issue
follow-up round, SKIP the `set-status` — status-hold rule, Step 9b §
Same-issue follow-up loop step 3 — and just re-invoke the planner); on
`continue-as-planned`: advance to Step 6 normally (round counter does
NOT reset). Do NOT state the Decision and then end the turn.

#### #397 replay fixture (canonical test case)

The detector's behavior on task #397's actual event sequence:

| Round | Implementer tag | Detector state after this round |
|---|---|---|
| 5 | (no tag — first complete dispatcher round) | 0 distinct, no fire |
| 6 | (no `epm:new-bug-class`; emits `epm:compute-deviation` from Fix #4 because wall-time 3-4× plan §9) | 0 distinct experiment-strategy classes — compute-deviation routes via Fix #4's pivot_criteria, NOT the whack-a-mole counter |
| 7 | (no tag — descope round) | 0 distinct |
| 8 | `epm:new-bug-class: vllm_teardown_oom` | 1 distinct, no fire |
| 9 | `<!-- workflow-fix-candidate v1 -->` (pod-side `task.py` shellout is a workflow-surface bug per the workflow-fix-on-bug protocol) | EXCLUDED from count — still 1 distinct experiment-strategy class (round 8's vllm), no fire |
| 10 | `epm:new-bug-class: subprocess_wrapper_missing_upload` | PRIMARY does not fire (need 3 distinct across the 3 most recent non-excluded rounds; only rounds 8 + 10 are non-excluded so only 2 distinct are available). SECONDARY DOES FIRE: 2 distinct tags across the 2 most recent non-excluded rounds (rounds 8 + 10; round 9 was excluded and is skipped, so 8 and 10 count as consecutive non-excluded) AND `epm:compute-deviation` at round 6 IS in the trailing 5-round window (rounds 6,7,8,9,10 from round 10's perspective). |
| 10' | Detector fires at the start of the would-be relaunch attempt — orchestrator surfaces 2-option prompt: `continue-as-planned (round 10 relaunch, cost: ~30 min, may hit next architectural assumption)` vs `pivot-to-in-process-serial (unify smoke and sweep paths, cost: one re-planning round, eliminates entire whack-a-mole class)`. User picks pivot — matches the actual round-11 decision. Route to `status:planning`. |

Key insight from the fixture: round 9's tag choice (workflow-fix-
candidate vs new-bug-class) determines whether the detector fires at
round 10 via SECONDARY (workflow-fix exclusion path) or one round
later via PRIMARY. The SECONDARY trigger exists specifically to
catch the #397 shape one round earlier than PRIMARY would.

<!-- gate: gates.conditional.whack_a_mole_pivot -->

### Step 6: Pod provisioning + experimenter dispatch (experiment only)

Only if status is `running` (entered from Step 5b PASS for `experiment`)
and no `epm:launch` marker exists.

#### Step 6a: HF gate-access check

Provisioning a pod only to have the run die seconds in on a `401 gated
repo` is wasted GPU-minutes. Before provisioning, scan the cached plan
for HF model IDs and verify the user's `HF_TOKEN` already has access to
each, using `huggingface_hub.HfApi.auth_check` (idempotent — it raises
`GatedRepoError` when the token lacks gate access, and returns cleanly
when access is already granted). There is no programmatic way for a
consumer to auto-accept someone else's gated-model gate page, so a
blocked repo halts with the gate URL for the user to click through once:

```bash
PLAN_PATH=$(uv run python scripts/task.py find <N>)/plans/plan.md
# Source .env FIRST — the VM shell does not inherit HF_TOKEN, so running this
# probe bare yields a false "HF_TOKEN missing" exit 2 (hit twice on 2026-06-09).
set -a; [ -f "$REPO_ROOT/.env" ] && source "$REPO_ROOT/.env"; set +a
uv run python - "$PLAN_PATH" <<'PY'
import os, re, sys
from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

plan = open(sys.argv[1]).read()
# HF model IDs cited in the plan (org/name, the canonical gated form).
repo_ids = sorted(set(re.findall(r"\b([A-Za-z0-9][\w.-]+/[\w.-]+)\b", plan)))
token = os.environ.get("HF_TOKEN")
if not token:
    print("HF_TOKEN missing"); sys.exit(2)
api, gated = HfApi(), []
for rid in repo_ids:
    try:
        api.auth_check(rid, token=token)
    except GatedRepoError:
        gated.append(f"https://huggingface.co/{rid}")
    except RepositoryNotFoundError:
        pass  # not a real model repo (a false-positive org/name match)
if gated:
    print("GATED (manual approval needed):", *gated, sep="\n  "); sys.exit(1)
print("all cited HF repos accessible"); sys.exit(0)
PY
```

- Exit code `0` -> proceed to 6a.5.
- Exit code `1` (gate access needed) -> post `epm:hf-gate-pending v1`
  with the gate URLs, leave status at `running`. Post the §5 marker:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 6c \
    --exit-kind clean --notes "hf-gate manual approval pending"
  ```
  EXIT. User clicks through the gate page, re-runs `/issue <N>`.
- Exit code `2` (`HF_TOKEN` missing) -> post `epm:hf-gate-pending v1`
  with diagnostic, status to `blocked`. Post the §5 marker:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 6c \
    --exit-kind failure-exit --notes "HF_TOKEN missing; status:blocked"
  ```
  EXIT.

The same `HF_TOKEN` is pushed to the pod by `bootstrap_pod.sh`, so a pod
provisioned in 6b sees the identical gate state as the local VM.

#### Step 6a.5: Carry-over artifact existence check (before provisioning)

Plans for follow-ups (and any experiment that reuses a prior run's
checkpoint, dataset, or eval output) cite HF / WandB URLs for the
artifacts they depend on. Provisioning a pod only to have the run die
seconds in on a `404` is pure wasted GPU-minutes. Before provisioning,
verify every carry-over URL the plan cites actually resolves:

```bash
PLAN_PATH=$(uv run python scripts/task.py find <N>)/plans/plan.md
uv run python -c "
from explore_persona_space.orchestrate.hub import verify_artifacts_exist
import sys
ok, missing = verify_artifacts_exist(plan_path='$PLAN_PATH')
if not ok:
    print('MISSING ARTIFACTS:', *missing, sep='\n  ')
    sys.exit(1)
print('all carry-over artifacts resolve')
"
```

`verify_artifacts_exist` scans the cached plan for HF repo URLs
(`huggingface.co/...`) and WandB run URLs (`wandb.ai/.../runs/...`) and
HEAD-checks each against the Hub / WandB API using the user's
`HF_TOKEN` / `WANDB_API_KEY`. It returns `(ok, missing_urls)`.

- All resolve -> proceed to 6a.6.
- Any missing -> post `epm:carry-over-missing v1` with the unresolved
  URLs, set status to `blocked` (the plan depends on an artifact that
  isn't there; provisioning would burn GPU on a guaranteed failure).
  Post the §5 marker:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 6c \
    --exit-kind failure-exit --notes "carry-over artifact(s) missing; status:blocked"
  ```
  EXIT. User fixes the cited URL (re-upload, or correct the plan) and
  re-runs `/issue <N>`.

#### Step 6a.6: HF write-headroom probe (quota gate, before provisioning)

Step 6a verifies READ access only; a namespace at its public-storage
quota passes the gate-access check, the carry-over HEAD-checks, AND
pod-side preflight, then 403s on the run's FIRST upload — after the pod
is already provisioned. (Incident #555, 2026-06-10: a fresh 4xH100
provision + sync + preflight + launch died 2 minutes in on `403
Forbidden: You have exceeded your public storage space`, namespace at
11.3 TB; a full launch cycle wasted.) Before provisioning, probe the
actual failing operation — a tiny (~1 KB) write to the project model
repo, immediately deleted:

```bash
# .env is already sourced by Step 6a (which exits on missing HF_TOKEN).
uv run python - <<'PY'
import io, sys
from huggingface_hub import HfApi

REPO = "superkaiba1/explore-persona-space"
PROBE = ".quota_probe/probe.txt"
api = HfApi()
try:
    api.upload_file(path_or_fileobj=io.BytesIO(b"quota probe"),
                    path_in_repo=PROBE, repo_id=REPO,
                    commit_message="quota probe (auto-deleted)")
    api.delete_file(path_in_repo=PROBE, repo_id=REPO,
                    commit_message="remove quota probe")
except Exception as e:
    resp = getattr(e, "response", None)
    if resp is not None and resp.status_code == 403 and "storage" in str(e).lower():
        print("QUOTA EXCEEDED:", e); sys.exit(1)
    # Fail-soft on NON-quota errors (transient 5xx, network blip): the
    # probe's only job is the quota 403; reachability is preflight's job.
    # Do NOT block provisioning on an inconclusive probe.
    print("probe inconclusive (non-quota error, proceeding):", e); sys.exit(0)
print("HF write headroom OK"); sys.exit(0)
PY
```

- Exit code `0` (probe OK or inconclusive) -> proceed to 6b.
- Exit code `1` (storage quota exceeded) -> post `epm:hf-quota-exceeded v1`
  with the verbatim 403 text + the probed repo id, set status to
  `blocked` (the storage decision — delete old artifacts vs upgrade the
  namespace — is the user's; provisioning would burn GPU on a guaranteed
  upload failure). Post the §5 marker:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 6c \
    --exit-kind failure-exit --notes "HF namespace storage quota exceeded; status:blocked"
  ```
  EXIT. Do NOT provision. User frees space / upgrades storage and
  re-runs `/issue <N>`.

**Size-aware projected-headroom gate (#1034).** The 1 KB probe above catches
only the ALREADY-over-quota case. When the approved plan's §9/§10 projects
≥100 GB of canonical-public LFS uploads, ALSO run
`uv run python -m explore_persona_space.orchestrate.preflight --no-gpu
--planned-upload-gb <N>` (decimal GB, the plan's projected LFS total). A
KNOWN-insufficient exit (the gate's ERROR is live-confirmed via a forced
re-probe) is handled EXACTLY like the quota-exceeded exit above: post
`epm:hf-quota-exceeded v1` (with the gate's error text), post the same §5
marker (`uv run python scripts/post_step_completed.py --issue <N> --step 6c
--exit-kind failure-exit --notes "projected LFS headroom insufficient;
status:blocked"`), set `blocked`, EXIT — the storage decision is the
user's. Fail-open otherwise: unknown headroom /
disabled check / routing armed all WARN and proceed to 6b.

#### Step 6b: Pod provisioning

**Backend dispatch (slice-6 unified router — auto by default, RunPod opt-in).**
Read the task's `backend:` frontmatter via
`uv run python scripts/task.py view <N> --json | jq -r '.frontmatter.backend // empty'`.
**The frontmatter value (or its absence) is fed verbatim to the slice-6
router via the dispatch helper** —
`explore_persona_space.backends.issue_dispatch.dispatch_for_issue`
calls `backends.router.route()` with production-injected deps and
returns a typed `RunHandle`. The router decides which backend actually
runs:

- **Empty / absent frontmatter → `auto`.** The router walks the
  resolved auto lane order — **standing default: GCP FIRST**
  (`DEFAULT_AUTO_LANE_ORDER = ("gcp", "nibi", "fir", "mila")` —
  credits-backed GCP capacity is consumed before the free SLURM lanes;
  unconditional, no date gate; override via the comma-separated
  `EPM_AUTO_LANE_ORDER` env var, e.g. `nibi,fir,mila,gcp` to restore
  free-first; `runpod` / unknown lanes in the override raise loudly).
  GCP is a single provision attempt (no park); its provisioning /
  capacity failures fall through to the SLURM lanes. Contiguous SLURM
  lanes (Nibi, Fir if wired, Mila if its socket is alive) are ranked
  among themselves by tz-corrected `sbatch --test-only` est-start, the
  best is submitted and parked up to `FREE_WAIT_SECONDS` (600 s; ALWAYS
  applied — see `backends.router`); park-cap-exceeded cancels + moves
  to the next lane. A GCP workload failure surfaces with NO fallback.
  **The auto chain NEVER calls RunPod** in ANY order (real-money
  safety) — `backends.router._VALID_BACKEND_VALUES`, the
  `auto_lane_order()` validator, and the load-bearing
  `test_no_auto_runpod_path_under_any_failure` negative test enforce
  this.
- **`backend: runpod`** explicit override → RunPod (the only path that
  spends real money in v1).
- **`backend: nibi` / `fir` / `mila`** → that lane, with the same park
  + cancel state machine as auto.
- **`backend: gcp`** → GCP credits.
- **Legacy `backend: cluster`** is normalized to `backend: nibi` by
  `issue_dispatch.normalize_backend_value` (the slice-5 router rejects
  the bare `"cluster"` literal). The legacy `select_backend` /
  `EPM_CLUSTER_MAX_WAIT_SECONDS` env knob from the pre-slice-6 wiring
  are no longer consulted — the 10-min `FREE_WAIT_SECONDS` park
  supersedes the old 6-h default.

**Lane capability check (run BEFORE the dispatch call).** All router
lanes (GCP + SLURM) execute custom workload commands: pass the plan's
launch command via `--workload-cmd 'bash scripts/issue<N>_dispatch.sh
...'` (mutually exclusive with `--hydra`; exactly one required — the
CLI fails loud otherwise; note the neither-set defense-in-depth raise
exists in the GCP renderer only — SLURM's default stage chain is
pre-existing behavior). Auto routing is valid for dispatch-script
workloads (#588). Residual gaps that still need the explicit
`--backend runpod` override (or the named knob): (a) 70B intents
(`inf-70b`/`ft-70b` have no GCP machine-type mapping — fail-loud by
design); (b) workloads needing the open-instruct `--extra gpu` venv on
a SLURM lane under a non-ft intent (venv extras follow the INTENT, not
the workload kind: `ft-7b`/`ft-70b` custom commands DO build `--extra
gpu`; `lora-7b`/`eval`/`debug` custom commands build the base venv —
`needs_gpu_extras`, slurm.py); (c) workloads
needing interactive SSH-MCP-driven orchestration mid-run (the
experimenter launch pattern); (d) **multi-day workloads on GCP
longer than the fence** — the lane pins `--instance-termination-action=DELETE` +
`--max-run-duration` (default 7d — the FLEX_START ceiling, #741), so a
sweep longer than the fence is deleted
mid-run; thread the plan's declared fence via `--max-run-duration
<dur>` on `dispatch_issue.py launch` (gcloud duration shape, e.g.
`30h`; lands in `spec.extra["max_run_duration"]`, inert on non-GCP
lanes — #628) or use the RunPod override. **When overriding to RunPod, name the residual gap in
the launch marker note** (CLAUDE.md rule). The dispatch CLI
cross-checks the task's ACTUAL frontmatter and classifies the override
3-ways, each with a DISTINCT marker flag (additive visibility — the
launch is never blocked): passing `--backend runpod` while the
frontmatter `backend:` does not name a backend (absent/empty, or an
explicit `auto`) triggers a LOUD stderr warning +
`extra.override_without_frontmatter=true` on the
`epm:backend-selected` marker; frontmatter naming a DIFFERENT
recognized lane (`gcp`/`nibi`/`fir`/`mila`, or the legacy `cluster`
alias for nibi) triggers a conflict warning +
`extra.override_conflicts_frontmatter=true`; an unrecognized value
(typo'd `gpc`, non-string `true`) triggers a hygiene warning +
`extra.frontmatter_backend_unrecognized=true` — the latter two also
carry `extra.frontmatter_backend: "<value>"`. Frontmatter
`backend: runpod` is the one legitimate backing and stays silent. For the gcp/auto lanes the dispatch script must exist
on the pushed branch, so you MUST pass `--repo-branch issue-<N>`
EXPLICITLY: the orchestrator runs `dispatch_issue.py` from the repo
ROOT (pinned to `main`), so the `--repo-branch` default (the cwd's
current branch) resolves to `main`, NOT the issue branch where a
per-issue driver script lives — the GCE startup script then clones
`main`, the driver is absent, and the workload dies ~4 min in with the
EXIT trap powering the VM off (#595, 2026-06-13). Defense-in-depth
(#987): `dispatch_issue.py` and `backend_poll.py` self-pin lane-infra
imports (`explore_persona_space.backends.*`, lazy `scripts.*`) to the
MAIN checkout via a `__main__`-guarded git-common-dir sys.path
bootstrap, so a worktree-cwd script-mode invocation of either
entrypoint no longer imports a stale branch lane template; the
repo-root invocation rule above remains the documented default (it
also selects main's venv for third-party deps), and the pin covers
ONLY script-mode execution — module-IMPORT consumers of `backends/`
(e.g. `autonomous_session_watch.py`) are deliberately unpinned, so
the cron-wrapper convention of `cd`-ing to the main checkout before
invoking them stays load-bearing. Residuals the pin does NOT close:
pre-#987 worktree COPIES (branches cut before the fix) carry no
bootstrap until rebased, an already-running process keeps its cached
stale modules, import-mode callers (`dispatch_for_issue` from a
worktree venv) get no pin, and already-launched workloads keep the
template they were rendered with. Four more gcp/auto
composition rules ((e) and (f) both hit live on #599, 2026-06-11;
(g) from #608; (h) from #606): (e) **GPU
sizing on the gcp/auto lanes comes from `--intent`, never `--gpus`** —
the GCP lane maps intent → machine type statically
(`backends/gcp.INTENT_TO_MACHINE`: `lora-7b`/`lora` →
`a2-ultragpu-1g`, 1 GPU; `ft-7b` → `a2-ultragpu-4g`, 4 GPU) and
ignores `--gpus` (only RunPod and SLURM honor the override), so pick
the intent whose machine matches the plan's GPU spec; a gcp-reachable
launch with a mismatched `--gpus` is refused pre-route by
`dispatch_issue.py` (exit 2, `reason: gpus_machine_mismatch`). (f)
**Never reference `$WORKLOAD_ROOT` bare in a workload-cmd — it is
exported ONLY by the GCE startup script**, so the exact command a
GCP→RunPod failover (or a SLURM fall-through) re-runs aborts under the
RunPod launcher's / SLURM custom stage's `set -u` before the driver
starts (incident #825: `REPO_ROOT="$WORKLOAD_ROOT"` killed the Track-S
RunPod failover; `dispatch_issue.py` now lints this at launch —
warn-by-default + `extra.workload_cmd_lane_env_risk` on the
`epm:backend-selected` marker, exit-2 refusal on a provably-certain
lane or under `--strict-workload-cmd-env`, #1329). A driver defaulting
`REPO_ROOT=/workspace/explore-persona-space` still dies on the GCE lane
(the startup script clones to `$WORKLOAD_ROOT`, `/workspace/eps-issue-<N>`,
and cds there), but the GCE startup script already exports
`REPO_ROOT="$WORKLOAD_ROOT"` before running the workload (#641,
`backends/gcp.py render_startup_script`), so compose
`--workload-cmd 'bash scripts/<driver>.sh'` with a SELF-RESOLVING driver
(`REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"`,
the #825 pattern), or use the set-u-safe default expansion inline:
`--workload-cmd 'REPO_ROOT="${WORKLOAD_ROOT:-$PWD}" bash scripts/<driver>.sh'`
(every lane cds to the checkout root first; `${VAR:-default}` is safe
under `set -u`). (g)
**Sentinel-signaling dispatchers must not rely on auto's SLURM fallback**
— a dispatch script that posts markers via pod-side sentinel files
(`/workspace/logs/issue-<N>-*.json`) works only on the /workspace-contract
lanes (gcp/runpod): SLURM compute nodes have no `/workspace`, so the
script fails loud at `mkdir -p /workspace/logs` and burns the submission
(#608, commit 3022ff7bc); pin `backend: gcp` (or runpod with a named
residual gap), or convert the dispatcher to the SLURM signaling contract
(`status.json` heartbeat + `[phase=...]` log lines) before routing auto
(planner.md §9 names this constraint at plan time). (h) **Boot-disk
sizing on the gcp/auto lanes comes from the plan's Reproducibility pod
row, threaded via `--boot-disk-gb` on EVERY launch — relaunches after a
code-fix round included** — the GCP lane defaults the boot disk to
300 GB pd-ssd (`backends/gcp.GcpConfig.default_boot_disk_gb`), which a
ZeRO-3 full-FT (`ft-7b`) fills with optimizer-state checkpoints in ~1h:
the instance kernel-panics on the full disk, cloud-init ENOSPCs, the
guest agent cannot write `authorized_keys` (SSH publickey lockout), and
the wedged VM idles on 4×A100 until deleted (#606, 2026-06-12 — the
relaunch dropped the plan's explicit "500 GB pd-ssd" spec). When the
plan's pod row names a disk size, pass it; for `ft-*` intents whose
plan names none, default to ≥500 GB. `dispatch_issue.py` warns loud
(stderr + `extra.boot_disk_default_with_ft_intent=true` on the
`epm:backend-selected` marker) when an ft intent is gcp-reachable with
no `--boot-disk-gb` — warning only, never a refusal (small-disk ft
smokes stay legitimate). (i) **WandB project on `--workload-cmd`
launches defaults to `issue<N>`** — the GCP startup script and the
SLURM custom stage export `WANDB_PROJECT="${WANDB_PROJECT:-issue<N>}"`
before the verbatim command, so HF-Trainer workloads that never set a
project stop landing in WandB's global default `huggingface` project
(Upload Policy: training metrics → `project=<experiment_name>`; #601
follow-up r1 landed there silently). An inline `WANDB_PROJECT=...`
prefix on the workload command — or the workload setting its own
project internally — still wins (`:-` fills only unset/empty); hydra
launches are unaffected (project comes from Hydra config).
SLURM custom stages are
render-tested only as of #588 (never live-run).
(Incident #571, 2026-06-11: auto routing sent a dispatch-script
workload to GCP before the router had a custom workload-command field;
the startup script ran bare `scripts/train.py`, crashed at startup,
and the EXIT trap powered the VM off. #588 closed it — the GCP
renderer now refuses to render that bare launch, and `--workload-cmd`
carries dispatch scripts on every lane.)

The handle the dispatch helper returns is persisted to
`.claude/cache/issue-<N>-handle.json` (the bg-Bash poller reads it
back; see Step 6d.2).

**Marker trail** (all VM-side; both `backends.router.route` and the
SLURM helpers call `task.py post-marker` via
`backends.slurm.post_marker_via_task_py`):

- `epm:backend-selected v1` — posted by `route()` on EVERY decision
  (including a pre-escalation intermediate marker when the auto chain
  is about to spend GCP credit). Body carries `requested_kind`,
  `chosen_kind`, `reason` (`override` / `reconnect` / `auto_started` /
  `auto_fallback_gcp` / `no_compute_available` / `workload_failure`),
  `cluster`, `elapsed_seconds`, the per-lane `attempts` ladder, and
  `extra` (`cancel_race?`, `gcp_attempts_today?`, `intermediate?`,
  plus the dispatch-CLI override-guard flags — all scoped to the
  explicit `--backend runpod` path: `override_without_frontmatter?`
  when the task frontmatter does not name a backend (absent/empty, or
  an explicit `auto`); `override_conflicts_frontmatter?` when it names
  a DIFFERENT recognized lane (gcp/nibi/fir/mila, or legacy `cluster`);
  `frontmatter_backend_unrecognized?` when the value is a typo /
  non-string; the latter two also carry `frontmatter_backend?` with
  the raw lowercased value).
  Legacy `frontmatter_*` / `slurm_*` reason codes from the pre-slice-6
  `select_backend` are preserved in `workflow.yaml § markers` for
  back-compat reads.
- `epm:cluster-launched v1` — posted by `SlurmBackend.launch` (or
  `GcpBackend.launch` — GCP reuses this marker name) right after the
  job is submitted; body carries `job_id`, `scratch_dir`, `log_path`,
  etc.
- On the RunPod path the existing `epm:pod-provisioned` /
  `epm:run-launched` markers are still posted by the experimenter.

**Terminal-exception translation.** `route()` raises one of four
terminal `RouteError` subclasses when no lane succeeded; the
dispatch helper translates each via
`issue_dispatch.classify_terminal_exception` into the
`epm:failure v1` body + status the orchestrator already routes on
(SKILL.md Step 7):

| Exception | failure_class | status |
|---|---|---|
| `NoComputeAvailableError` | `infra` | `blocked` |
| `WorkloadSurfacedError` | `code` | `blocked` |
| `GcpAttemptCapExceededError` | `infra` | `blocked` |
| `ManualAttentionRequiredError` | `infra` | `blocked` (carries orphaned job_id) |

Step 6d.2 runs the bg-Bash poller against the persisted handle (no
per-backend branch); Step 8 runs `confirm_artifacts` + `teardown` on
the same handle. The cluster path's monitor (`epm:cluster-poll v1` /
`epm:cluster-terminal v1`) keeps working — `SlurmBackend.poll` calls
into `backends.slurm_monitor.build_poll_result` exactly as before;
the bg-Bash poller (`scripts/backend_poll.py`) prints the same
PollResult JSON shape regardless of backend.

The remainder of this section describes the RunPod / per-issue pod
specifics. The cluster path's sbatch carries an EQUIVALENT inline
preflight stanza (HF/WandB reachability, GPU visibility,
`$SLURM_TMPDIR` headroom) so a misconfigured job fails fast inside
the SLURM allocation.

Compute is ephemeral on every backend — no permanent pod fleet, no
permanent VM, no permanent SLURM submission stays alive past the run.

**Operational dispatch (slice-6 router, ALL backends).** The
orchestrator shells `scripts/dispatch_issue.py launch` — the operational
seam that builds the production backends (`RunPodBackend`,
`SlurmBackend` for every available cluster, `GcpBackend`) + the injected
dependencies (`marker_poster` = `backends.slurm.post_marker_via_task_py`;
`is_started` = SLURM-aware `query_slurm_state` status==RUNNING probe;
`is_live_after_cancel` = `query_by_name` non-empty probe;
`reconnect_fn` = per-kind SLURM-`squeue --name` + `gcp.reconnect_or_none`
(includes a `mila` branch matching the `nibi`/`fir` reconnect closure);
`mila_socket_alive` = the real `backends.slurm.mila_socket_alive` probe
that runs `ssh -o BatchMode=yes -o ConnectTimeout=5 mila true` over the
ControlMaster socket — slice 7's first-class wiring. A dead / OTP-
expired socket returns False (skip-the-lane, NOT an error); refresh is
the Claude-session cron documented at
`.claude/cron-prompts/mila-otp-refresh.md` and orchestrated through
`scripts/mila_socket_refresh.py` (un-armed in slice 7; live arming in
slice 8)) and calls
`backends.issue_dispatch.dispatch_for_issue` (which calls
`backends.router.route()`). The router decides the lane (auto → free
cluster → GCP, or honors an explicit override); RunPod's launch goes
through `RunPodBackend.launch` (which shells `pod_lifecycle.py
provision` under the hood) so the sidecar JSON is written uniformly
across backends. The bg-Bash poller (`scripts/backend_poll.py`) reads
that sidecar tick after tick (Step 6d.2); Step 8's
`scripts/dispatch_issue.py finalize` reads it again to run
`confirm_artifacts` + `teardown` (the same RunHandle from launch all
the way through teardown).

Before this launch call, run the Step 9 entry guard § Pre-dispatch
external-marker triage and post its `external-markers triaged:` line in an
`epm:progress` note immediately before dispatching (the launch posts
`epm:run-launched` / `epm:cluster-launched`, not a stage-dispatch
breadcrumb).

The operational command:

```bash
# Read the task's backend frontmatter (empty / absent → auto).
BACKEND=$(uv run python scripts/task.py view <N> --json | jq -r '.frontmatter.backend // empty')
# Infer --intent from the plan: training a 7B model → ft-7b or lora-7b;
# eval/generation → eval; 70B work → inf-70b/ft-70b. Override with
# --gpus / --time-budget-hours for anything else.
INTENT=<inferred>

# Single operational call — runs the router (auto / explicit override
# both flow through here). On RunPod the underlying pod_lifecycle.py
# enforces team scoping (X-Team-Id), SSH bring-up (startSsh: true,
# exposes 22/tcp), pinned image, and runs bootstrap inline (uv, repo,
# .env with HF_TOKEN, HF cache, preflight); on SLURM the SlurmBackend
# renders + ssh-submits the sbatch; on GCP the GcpBackend renders +
# ``gcloud compute instances create``s the VM. Hydra args repeatable.
uv run python scripts/dispatch_issue.py launch \
    --issue <N> --intent "$INTENT" --repo-branch "issue-<N>" \
    ${BACKEND:+--backend "$BACKEND"}
# --repo-branch is MANDATORY: the orchestrator dispatches from the repo
# root (main), so omitting it clones main on the gcp/auto lane and a
# per-issue driver script is absent (#595). Drop it ONLY if the workload
# is wholly on main (no issue-branch-only script).
```

`dispatch_issue.py launch` prints ONE JSON line on stdout with the
resolved outcome (`chosen_kind`, `requested_kind`, `reason`,
`pod_name`, `handle_sidecar_path`). On a router terminal it exits with
code `2` and the JSON carries `failure_class` + `status` + `note` so
the orchestrator posts `epm:failure v1` per the table above and
`set-status <N> blocked` — no re-derivation. On a non-terminal
provisioning error (RunPod SUPPLY_CONSTRAINT etc.) the underlying
backend raises and the helper either retries (RunPod's
`--wait-for-capacity` loop) or surfaces the failure as
`epm:pod-pending v1` so the user adjusts (capacity, intent override)
and re-runs `/issue <N>`. On exit code `75` (EX_TEMPFAIL) the JSON
carries `still_waiting: true` + `rerun: true` + `reason:
wait_for_capacity_budget_reached`: the RunPod lane's
`pod_lifecycle.py provision` hit its bounded wait-for-capacity
per-process wall-clock budget while capacity / the fleet burn cap kept
the provision queued. NOT a failure — the wait loop is state-free, so
RE-RUN the same `dispatch_issue.py launch` command to continue waiting
(post an `epm:progress v1` heartbeat per re-run so the watcher sees
liveness); NEVER post `epm:failure v1` / `set-status blocked` on this
exit (incident #603, 2026-06-11: the exit previously crashed the CLI
as an rc-4 `CalledProcessError`).

**Follow-up parent reuse.** When the task has a `parent_id` AND the
parent's RunPod pod is alive, the operational path stays on the
existing `pod.py` flow for that one specific case (the slice-6 router
does NOT yet model "reuse parent's live pod" — slice 7 wires the
reconnect path through the router uniformly):

```bash
PARENT_ID=$(uv run python scripts/task.py view <N> --json | jq -r '.frontmatter.parent_id // empty')
if [ -n "$PARENT_ID" ] && uv run python scripts/pod.py list-ephemeral --issue "$PARENT_ID" | grep -q epm-issue; then
  # Parent pod still alive — resume + reuse. Skip the router call;
  # this child task's run inherits the parent's pod_name.
  uv run python scripts/pod.py resume --issue "$PARENT_ID"
  # Record the assigned pod as epm-issue-$PARENT_ID in the launch marker.
else
  # Fresh launch through the router (the canonical path above).
  uv run python scripts/dispatch_issue.py launch \
      --issue <N> --intent "$INTENT" ${BACKEND:+--backend "$BACKEND"}
fi
```

**Slice-6 regression guard for the parent-pod-reuse branch (no
sidecar is written).** When the alive-parent path above fires (child
task with `parent_id` AND parent's RunPod pod still alive →
`pod.py resume --issue $PARENT_ID`), the dispatcher is NOT invoked, so
`.claude/cache/issue-<CHILD_N>-handle.json` is NEVER written.
Downstream that means: (1) Step 6d.2 MUST SKIP `backend_poll.py
--issue <CHILD_N>` — its missing-sidecar guard would post a
FALSE-POSITIVE `epm:failure v1` (`failure_class: infra`, `reason:
missing_handle_sidecar`) on a perfectly healthy child run; instead,
fall back to the legacy `poll_pipeline.py --pod epm-issue-$PARENT_ID
...` invocation for the duration of this child (the parent's pod
name + log path are the authoritative identifiers, NOT the child's
sidecar). (2) Step 8 MUST SUBSTITUTE the `dispatch_issue.py finalize
--issue <CHILD_N>` call with `pod.py terminate --issue $PARENT_ID
--yes` — terminating the parent's pod IS the correct operation here
(matching the existing teardown prose under Step 8), and the
finalize CLI would otherwise exit 2 on missing sidecar. Re-record
the parent's `epm:pod-terminated v1` against the child task so the
dashboard surfaces the terminate. Full reconnect-via-router
unification (write a sidecar even on the reuse path so every
backend / lane uses ONE Step 6d.2 + Step 8 code path) stays
slice 7 — this paragraph is the operational guard that prevents the
false-positive failure / mis-routed finalize until then.

**Autonomous mode (`EPM_AUTONOMOUS_SESSION=1`) — RunPod
`--wait-for-capacity` auto-enables.** When the router's chosen lane is
RunPod (explicit override `backend: runpod`), the underlying
`pod_lifecycle.py provision` reads `EPM_AUTONOMOUS_SESSION` itself and
turns on the unbounded SUPPLY_CONSTRAINT retry loop (exponential
backoff with full jitter, base 30s, cap 10 min, forever) — "the
experiment should start when it has space," not park-for-user.
"Unbounded" is across re-runs, not per process: each provision process
exits 75 (still-waiting) at its wall-clock budget and the dispatch CLI
surfaces that as `still_waiting: true` + exit 75 — re-run the same
launch command (see the exit-75 contract above), never treat it as a
failure. The
orchestrator should background the dispatch call (`Bash` with
`run_in_background=true`) so its own turn isn't blocked, and ON
periodic re-invocation (each bg-Bash output yield) it should scan the
captured stderr for `[wait-for-capacity] attempt N, waited ...` lines
and post one `epm:progress v1` marker per heartbeat (note:
`"pod-provision waiting for capacity: attempt N, waited ..."`). This
keeps `autonomous_session_watch.py` (6h stale-marker threshold) seeing
liveness. **Interactive sessions still fail fast** —
`--wait-for-capacity` defaults OFF so a human running `pod.py provision`
from a shell sees no-capacity immediately and can decide whether to
wait, switch DC, or change GPU intent.

**Stale-port recovery — `pod.py config --refresh-from-api`.** When an
`epm:pod-pending v1` is followed by a long stretch of failing SSH
polls (`poll_pipeline.py` reporting `status=dead` every tick on an
otherwise live pod), the most common cause is that a
SUPPLY_CONSTRAINT-blocked resume eventually brought the pod back at a
NEW SSH port via a retry path that bypassed `_upsert_pods_conf`, so
`pods.conf` still carries the pre-stop value while the live RunPod API
has the fresh one. The canonical first response is `uv run python
scripts/pod.py config --refresh-from-api pod-<N>` — pulls fresh
host/port from the live API into `pods.conf` + `~/.ssh/config`. As of
2026-06-09 the auto-heal also fires automatically: `poll_pipeline.py`
counts consecutive SSH-probe failures and fires `--refresh-from-api`
after ten consecutive failures (~3-4 min at 20s spacing), and
`autonomous_session_watch.py` fires it once per stalled episode when a
stalled session has a RUNNING managed pod. Both auto-fires are
fail-soft and dedup'd so the manual command stays the surgical
recovery move; reach for it when the auto-heal has not yet tripped or
the issue is unambiguously a port drift. See `.claude/rules/upload-policy.md`
context on the Authority split (live API authoritative for host/port,
`pods.conf` the on-disk source for SSH/MCP). Incident #488 (2026-06-09)
spun for 13+ hours at $32/hr before the manual subcommand existed.

The pod / job / VM name passed downstream is recorded in the sidecar
JSON the router writes (RunPod: `pod-<N>`; SLURM: `eps-issue-<N>`;
GCP: `eps-issue-<N>`). The experimenter does NOT pick or create pods.

#### Step 6c: Preflight on resumed pods

`provision` already ran preflight as its last bootstrap step. For
*resumed* pods, re-run preflight explicitly because the volume is intact
but the container restart may have left stale state:

```bash
ssh_execute(pod=epm-issue-<N>, command="cd /workspace/explore-persona-space && uv run python -m explore_persona_space.orchestrate.preflight --json")
```

Parse JSON. (Note: the old `Local is N commit(s) behind origin/main`
false-fail on `issue-<N>` branches was fixed at source by #554,
2026-06-12 — preflight is branch-aware and that condition is now a
WARNING, so on current code an `ok=false` here is a real failure.)
If `ok=false`, post `epm:preflight v1` event with the
errors/warnings, then post the §5 marker:
```bash
uv run python scripts/post_step_completed.py --issue <N> --step 6c \
  --exit-kind failure-exit --notes "preflight failed; user must fix"
```
EXIT. User fixes, re-runs.

#### Step 6d: Dispatch experimenter (launch-only), then orchestrator polling loop

The experimenter agent is **launch-and-exit only** — it syncs the pod,
preflights, launches the job via its setsid launcher script
(`experimenter.md` § "During Execution"), posts `epm:run-launched`, and
exits its turn within ~60 seconds. The orchestrator (this skill) owns
all subsequent monitoring via a bg-Bash polling loop chained through
`scripts/poll_pipeline.py`. This split is mandatory: subagents have ONE
turn and are NOT auto-re-invoked when bg work completes, whereas the
orchestrator IS auto-re-invoked on every bg-Bash exit (see `CLAUDE.md`
§ "Subagent vs orchestrator re-invocation semantics").

##### Step 6d.0: Smoke/sweep architecture parity gate

Fires once per implementer round, AFTER all of Step 6a-6c (HF gate,
pod provision, preflight) and BEFORE Step 6d.1 (experimenter dispatch).
Reads the highest-version `epm:smoke-architecture-check v<n>` marker
posted by the implementer in the current round (see
`experiment-implementer.md` "Before writing code" item 5 and
workflow.yaml § markers `epm:smoke-architecture-check`).

Verdict routing:

| `verdict` | Action |
|---|---|
| `PASS_UNIFIED` | Advance to Step 6d.1 — smoke IS sweep with one cell; the architecture is unified end-to-end. |
| `PASS_CANARY canary_cell=<id>` | Advance to Step 6d.1 — paths diverge but the plan §4 Design justifies the divergence in two sentences AND names the canary cell that exercised the sweep path during smoke. Log to chat: `divergence accepted; canary cell <id> exercised the subprocess path during smoke`. |
| `FAIL_NO_CANARY` | **REFUSE to dispatch.** Bounce back to status:planning; re-invoke `/adversarial-planner` with pivot scope: "the smoke/sweep architectural divergence has no justification + canary; re-architect toward UNIFICATION (smoke = sweep with one cell), OR add the two-sentence justification + named canary cell to §4 Design." Round counter does NOT increment (this is a strategy pivot, not a fresh review round). |
| (marker missing) | **REFUSE to dispatch.** Bounce back to implementer with a one-line prompt: `post epm:smoke-architecture-check v1 per the mandatory checklist before code-review-PASS`. |

<!-- gate: gates.inline.smoke_architecture -->

The gate is enforced inline (gates.inline id=10) — the implementer
self-tags at report-time; the orchestrator validates here.

Rationale: task #397 rounds 9/10/10' (2026-05-27) all PASSed smoke and
crashed sweep within ~5s of nohup because smoke ran in-process
`train_one_cell` while sweep ran `run_one_cell.py` as a subprocess.
Round 11's pivot was to UNIFICATION (in-process serial). This gate
forces the divergence to be explicit at plan time so the pre-dispatch
moment catches it, not the third pod-side crash.

##### Step 6d.0-bis: End-to-end smoke gate (multi-phase data-gen pipelines)

For an experiment whose driver chains ≥2 distinct production phases
before the first GPU launch — data-gen, training, eval, verify, upload
(typically gen → drift → train → eval → aggregate; a datagen → train →
verify → upload driver like #906's equally qualifies) — the
architecture-parity gate above is NOT enough: it checks
that smoke and sweep share ONE code path, not that EVERY phase ran. A
resume-skip design serializes bug discovery — each pod cycle surfaces
the next phase's bug — so before the GPU production launch the FULL
pipeline must have executed once at tiny N (≈2-3 rows, 1 cell, 1 seed)
so EVERY phase runs end-to-end on CPU / 1-GPU.
The tiny-N pass MUST meet the **tiny-real standard** for `kind:
experiment`/`batch` drivers: it executes the production path with REAL
library types at every internal seam the pipeline actually has —
real tokenizer, real train engine + config builders + callbacks, real
adapter round-trip, real verify/upload bodies (an API-only driver has
no train engine; its own real seams bind instead) — faking ONLY
GPU-scale weights (a from-config tiny same-arch model over the real
vocab-id space) and the remote Hub boundary (signature-bound). A
seam-stubbed / mocked smoke does NOT satisfy this gate: mock-seam
smokes surface shape bugs one per GPU cycle (#906 r11-r14: four
distinct production shape bugs, four ~1.5h pod cycles, every mocked
smoke green beforehand; r15 = the tiny-real pivot). Worked example:
`tests/test_issue906_tiny_real_e2e.py`; full recipe + the two CPU
traps: `.claude/rules/gotchas.md` "Mock-seam smokes surface production
shape bugs ONE PER GPU CYCLE".
When the pipeline INGESTS a real corpus (a WildChat/LMSYS-class
streaming builder), the standard's **data-ingestion probe class**
(#1092) binds too: a bounded tiny-real streaming probe against the
REAL dataset — a kept cap AND a TOTAL-streamed-rows cap, asserting
kept > 0 per dataset — with per-filter rejection counters in the
stream's `done:` line; a synthetic-fixture smoke alone does NOT
satisfy the ingestion phase (a filter chain written from assumed
field shapes can reject 100% of real rows while every synthetic
smoke stays green). Recipe + verified field shapes:
`.claude/rules/gotchas.md` "Real-corpus streaming filters" +
`.claude/agent-memory/experiment-implementer/feedback_real_corpus_streaming_filters_tiny_real_probe.md`.
When the driver spans MULTIPLE ARM CLASSES (distinct source-context
classes / recipe branches — e.g. persona-context vs bare-context
arms), "once at tiny N" means once PER ARM CLASS: per-arm seams
(source-context construction, negative-panel assembly, `ModelOrganism`
wiring) are invisible to a single-arm smoke however tiny-real its
seams (#1090 fu5: a formatting-arm-only smoke passed; all 3
bare-context arms then died on the #527/#538 panel-disjointness assert
after a full 4×A100 GCE cycle). Recipe: `.claude/rules/gotchas.md`
"A single-arm smoke is blind to per-arm seams".
Confirm the implementer's
`## Smoke run` report (per `experiment-implementer.md` § "End-to-end
smoke run PER PHASE") carries a sub-section with exit code `0` + an
artifact digest for EACH phase the pipeline executes — not just training
or data-gen. Any phase missing, or showing only `--help` / `import` /
`--dry-run` / seam-stubbed (mocked internal seams — the tiny-real
standard's two sanctioned fakes excepted) / synthetic-fixture-only
(a real-corpus ingestion phase with no tiny-real streaming-probe
evidence) evidence → **REFUSE to
dispatch**; bounce to the implementer
with `run the full gen→...→aggregate pipeline once at tiny N before
production`. FAIL blocks production. (Origin: #408 — a multi-phase
data-gen pipeline never smoke-tested end-to-end leaked 5+ distinct bugs
one-per-pod-cycle over ~41h idle.)

Orthogonal to the smoke-gate above, the experimenter agent itself
enforces an **input-data completeness gate** as the first step in its
pre-launch protocol — verifying that the input-data files the
dispatcher will read from local disk on the pod are ALL present, and
posting `epm:failure v1 failure_class: infra reason:
planned-input-data-missing-on-pod` (no launch) on any shortfall. The
smoke gates check code paths and phase coverage; the input-data
completeness gate checks that the dependency files actually exist on
the pod. See `experimenter.md` § "Before Running" item 4 for the
mechanic and the #468 incident. The orchestrator does not need to
re-verify here — the routing on shortfall ends in an `epm:failure
failure_class: infra` that flows through Step 7's respawn path
naturally.

##### Step 6d.1: Spawn experimenter for launch

**Pre-dispatch state sanity (fires on EVERY dispatch — first launches
AND re-launches).** Immediately before spawning the experimenter,
re-verify the brief's two load-bearing assumptions against LIVE state —
never against this session's cached view (a concurrent / replacement
session may have finished the run while this session was mid-review):

1. **Compute exists.** For a RunPod-backed dispatch, `uv run python
   scripts/pod.py list-ephemeral --issue <N>` must show the assigned
   pod; for other backends, verify the brief's compute target is live
   per the handle sidecar / backend status (Step 6b). Absent → do NOT
   dispatch; re-derive scope from the markers (the run may already be
   done) or re-provision via Step 6b.
2. **Run still pending.** `uv run python scripts/task.py latest-marker
   <N>` + the recent `events.jsonl` tail: if `epm:results v<n>` +
   `epm:upload-verification PASS` (or `epm:pod-terminated v1`) postdate
   the failure being recovered, the (re)launch is STALE — the work
   already completed. Do not dispatch; reduce the brief to the genuinely
   missing artifact, or skip the dispatch entirely and resume from
   wherever the markers say the task actually is (Step 7+ / Step 9
   routing).

On either mismatch, re-derive the brief from the live markers instead
of dispatching the stale one. This is the dispatch-site analogue of the
Step 0 stale-wake ownership re-check and the Step 9 entry guard's
marker-freshness pattern. (Incident: task #559, 2026-06-11 — a
concurrent orchestrator completed the run, upload-verified, and
terminated the pod while this session was mid-code-review; this session
then dispatched a relaunch brief asserting "pod alive; run pending"
~10 min after `epm:pod-terminated`; only the experimenter's agent-side
defense caught it.)

**3. External markers triaged.** Run the Step 9 entry guard
§ Pre-dispatch external-marker triage check before spawning. Pod/backend
launches post no `stage-dispatch` breadcrumb, so the
`external-markers triaged:` line goes in an `epm:progress` note posted
immediately before the experimenter spawn.

Spawn `experimenter` subagent via `Agent()`. Brief:
- The plan path (the `plans/plan.md` symlink) + the code-reviewed
  branch (`issue-<N>`)
- Pod name (`epm-issue-<N>` or parent's)
- The exact workload command from the plan's Reproducibility Card (the
  workload/dispatcher invocation plus any required env-var pins; the
  experimenter wraps it in its canonical setsid launcher script —
  `experimenter.md` § "During Execution". NEVER put a literal top-level
  `source .env` + bare `nohup`-backgrounded launch line in the brief:
  the SSH-MCP shell is `sh` (`source: not found`, #545) and an
  un-setsid'd background launch risks SIGHUP reaping (#444/#541; the
  #841 brief carried exactly this shape and the experimenter had to
  deviate)
- When the plan names a "regenerate locally via prep script"
  prerequisite (e.g. the Turner JSONLs): the prep-script invocation AND
  its OUTPUT dataset path(s), so the experimenter's input-data gate
  (`experimenter.md` § "Before Running" item 4) stat-checks the files
  themselves — a secret/env-var presence check alone does not cover
  them (incident #545)
- Required: post `epm:run-launched` with `pod=<name> pid=<pid>
  log_abs=<absolute_log_path> cmd='<dispatch>'` in
  the note, then exit cleanly within 60 seconds. The `log_abs=` field
  MUST be an absolute path (use `realpath` or `os.path.abspath()` on
  the pod) AND the experimenter MUST verify the file exists with
  `ssh_execute ls -la <log_abs>` before posting. The legacy `log=`
  field is still accepted as a fallback during the transition window
  (scheduled removal after 2026-06-15 per the marker schema TODO) but
  new launches must emit `log_abs=`.
- Explicit: do NOT sleep-chain, do NOT monitor — the orchestrator polls
  the run

**NEVER include pod lifecycle commands (provision, stop, resume,
terminate, cleanup) in the experimenter brief.** Pod termination
happens automatically in Step 8 (after upload-verification PASS).
**NEVER include progressive monitoring instructions** in the brief —
those are obsolete (see the deprecated memory
`feedback_subagent_sleep_chain.md`).

Wait for the experimenter to return. The return must include the
`epm:run-launched` marker. Parse it for `pod`, `pid`, and the log
path. **Prefer `log_abs=` over `log=`** — when both are present, use
`log_abs=`. When only `log=` is present (legacy launches during the
transition window through 2026-06-15), accept it as a fallback but
log a one-line WARN: `experimenter posted legacy log= field; upgrade
the launcher to emit log_abs= per epm:run-launched schema`.

```python
# TODO: retire after 2026-06-15 — drop the `log=` fallback once all
# experimenters in active rotation emit `log_abs=`.
log_path = parsed.get("log_abs") or parsed.get("log")
if not log_path:
    raise ValueError("epm:run-launched missing log_abs= (or legacy log=)")
```

If the experimenter posted `epm:failure v1` instead (launch-time
crash), skip the polling loop and proceed to Step 7's failure-
classification routing.

Post `epm:launch v1` containing:
- Worktree path, branch, PR URL, code-review verdict (`PASS`)
- Pod + PID + log path
- WandB run URL (best-effort)

##### Step 6d.2: Orchestrator polling loop (bg-Bash chained)

Enter a polling loop that runs in THIS orchestrator's context. Each tick
is a single bg-Bash call that sleeps then runs the BACKEND-AGNOSTIC
poller once; the harness re-invokes the orchestrator when the bg-Bash
exits, which is when one tick has completed:

```python
result = None  # parsed JSON line of the PREVIOUS poll tick; None before the first tick
while True:
    # MANDATORY: refresh the title + self-report at the TOP of every
    # tick so the dashboard / happy-ls / phone title stay current with
    # the loop's `running` status (or the latest phase if the poller
    # posted one). This is the cheap path — no LLM call — and keeps
    # `~/.eps-autonomous/issue-progress/<N>.json` fresh under the
    # summarizer's 20-min freshness window. `set_title` is the soft-fail
    # helper defined in the "Chat title updates" section above; it
    # NEVER crashes the loop.
    set_title(N, current_phase)  # e.g. "running" / "phase: post_eval"

    # The bg-Bash poller is `scripts/backend_poll.py` — it reads the
    # per-issue handle sidecar at `.claude/cache/issue-<N>-handle.json`
    # (written by `issue_dispatch.dispatch_for_issue` in Step 6b),
    # resolves the right `ComputeBackend` from `handle.backend`, calls
    # `backend.poll(handle)`, and prints ONE JSON line whose shape is
    # byte-identical to the legacy `poll_pipeline.py` output (the
    # `backends.base.PollResult` fields). The orchestrator's existing
    # JSON-line parser is interchangeable across backends — no per-
    # backend branches here.
    #
    # On the RunPod path `backend.poll` delegates to
    # `scripts.poll_pipeline.poll_once` (the battle-tested probe);
    # `backend_poll.py` is the uniform bg-Bash entry, NOT a
    # re-implementation. The legacy `--pod` / `--log` / `--pid-file`
    # CLI args of `poll_pipeline.py` are recovered from the handle
    # sidecar by `backend.poll`, so the bg-Bash command line shrinks
    # to a single `--issue` argument.
    #
    # CAVEAT — parent-pod-reuse child tasks: when this is a child task
    # whose parent's RunPod is still alive AND the alive-parent branch
    # in Step 6b fired, NO sidecar was written for the child. SKIP
    # this bg-Bash `backend_poll.py --issue {N}` entirely and fall
    # back to `poll_pipeline.py --pod epm-issue-$PARENT_ID ...` for
    # the duration of the child. See the "Slice-6 regression guard
    # for the parent-pod-reuse branch (no sidecar is written)"
    # paragraph in Step 6b for the full rationale + the failure mode
    # the unconditional invocation would trigger (FALSE-POSITIVE
    # `epm:failure v1 missing_handle_sidecar`).
    # ADAPTIVE POLL INTERVAL (anti-stall redesign §7). Every tick's JSON
    # line carries a recommended `next_interval` (seconds): 1800 ONLY on
    # a healthy, quiet, post-early-run `running` tick far from any phase
    # boundary; 540 otherwise — gate-adjacent, anomalous, early-run
    # (first ~30 min after launch), and recent-phase-change ticks NEVER
    # get the long interval, so gates are never delayed. Use the
    # PREVIOUS tick's emitted value as this tick's sleep; FALL BACK TO
    # 540 when there is no previous tick yet (first poll after launch)
    # or the key is absent/unparseable (older poller, garbled JSON
    # line). NEVER lengthen the sleep on your own initiative: only the
    # emitted value may raise it above 540, and never after a tick that
    # reported anything other than healthy-quiet-running. Risk bound: a
    # stall can now be noticed up to 30 min later in-session — accepted
    # because the watcher's 10-min passes + the */45 issue-tick cron
    # bound out-of-session detection independently
    # (autonomous_session_watch.py / tick_triage.py).
    #
    # `result` below = the parsed JSON line from the PREVIOUS tick's
    # bg-Bash output (the same `result` the status branch below reads);
    # it is None on the first iteration — no previous tick yet. The
    # membership clamp makes the never-lengthen rule MECHANICAL: only
    # the two known emitted values are honored, anything else (garbled,
    # bool, surprise number) falls back to 540.
    interval = 540
    if result is not None and result.get("next_interval") in (540, 1800):
        interval = result["next_interval"]
    Bash(
        run_in_background=True,
        command=(
            f"sleep {interval} && uv run python scripts/backend_poll.py --issue {N}"
        ),
    )
    # Harness re-invokes orchestrator on bg-Bash exit. To WAIT on bg
    # work, simply END THE TURN with a one-sentence status — NEVER emit
    # no-op Bash calls to idle (`sleep 1` "yield turn", `true` no-ops):
    # each burns a tool call + context for nothing (33x and 49x in two
    # 2026-06-10 sessions). Read the JSON line from stdout (the LAST
    # line of the bg-Bash output) and decide:
    #
    #   status == "done"           -> exit loop; transition to status:verifying; go to Step 7.
    #   status == "gate"           -> a pod-side sentinel carried a non-empty
    #                                  `gate` field; the poller has ALREADY
    #                                  posted the carried marker (e.g.
    #                                  `epm:fact-candidates v1`) from the local
    #                                  VM as part of its sentinel drain — do
    #                                  NOT re-post it. Read result["gate"],
    #                                  exit the polling loop, and dispatch the
    #                                  matching gate handler per Step 6d.4
    #                                  below (PARK for a user gate like
    #                                  `fact-candidates`, AUTO-RESOLVE +
    #                                  resume the loop for `pv_phase1_done`).
    #   status == "stalled" | "dead" -> post epm:failure v1 with failure_class
    #                                   inferred from log_tail_excerpt
    #                                   (run scripts/failure_classifier.py on
    #                                   the excerpt, ALSO forwarding the tick's
    #                                   result["stall_reason"] via
    #                                   --stall-reason — a silent hang's log
    #                                   tail carries no infra pattern, so the
    #                                   stall_reason is the only routing
    #                                   signal; see Step 7); run CRON-TEARDOWN
    #                                   (see below); set status:blocked; exit.
    #   status == "running"        -> milestone-already-posted by the poller
    #                                  if new_milestone was true; loop again,
    #                                  using result["next_interval"] as the
    #                                  next sleep (540 fallback — see
    #                                  ADAPTIVE POLL INTERVAL above).
    #                                  If the JSON also has
    #                                  gpu_idle_advisory_posted == true, act
    #                                  per "GPU-idle advisory handling" below
    #                                  before the next tick. If it has
    #                                  gpu_idle_escalation_posted == true, act
    #                                  per "GPU-idle escalation handling" below.
```

(`current_phase` is `"running"` by default; when the poller emits a
milestone marker like `phase: post_eval`, update the local
`current_phase` from the milestone before the next tick so the title
reflects the latest phase.)

The top-of-tick `set_title` refresh plus the ≤30-min clamped interval
discharge the § Long-phase heartbeat duty (below) for this loop by
construction; any wait run OUTSIDE this loop shape — a `Monitor`
until-loop on a VM phase, an ad-hoc bg poll chain, an off-pod Batch-API
poll — carries that duty explicitly.

The `poll_pipeline.py` helper posts `epm:progress` events itself when it
sees a phase transition, AND drains pod-side sentinel files (posting
their carried markers from the VM via `task_workflow.post_event`). The
orchestrator's only post-tick duties are: exit the loop on `status=done`,
dispatch the matching gate handler on `status=gate` (Step 6d.4 — PARK for
a user gate, AUTO-RESOLVE + resume the loop for `pv_phase1_done`), and post
`epm:failure v1` on `status=stalled` or `status=dead`. The orchestrator
NEVER re-posts a marker the poller already posted from a sentinel —
double-posting is the failure mode the gate path is designed to avoid.
On the terminal `status=done` tick (the point where `epm:results` is
posted/observed), the next action after the `uploading` transition is
Step 8's **Results-landed parallel spawn** block — dispatch that
concurrent batch, NOT the old serial verifier-then-analyzer order (see
Step 8 for the block's contents and hard joins; do not re-derive them
here).

**GPU-idle advisory handling.** When a tick's JSON reports
`gpu_idle_advisory_posted: true`, the poller has just posted a one-time
`epm:progress` marker whose note starts with `[gpu-idle-advisory]` (plus a
`gpu_idle_advisory=True` extra): every GPU sat idle on a HEALTHY
`status=running` tick for ≥ `EPM_GPU_IDLE_ADVISORY_MIN` (default 30) min —
the signature of a long CPU-only phase holding a GPU pod (incidents
#518/#537). Don't just loop: surface the advisory in the session text,
then check the plan for whether the REMAINING work in the current phase is
CPU-only. If it is and the remaining CPU stretch is long (>~30 min), apply
CLAUDE.md "CPU-only phases don't hold GPU pods": checkpoint the phase's
state, upload the artifacts it reads, move the phase off-pod to the VM,
and `pod.py stop` the pod once nothing pod-local is needed. Three hard
constraints: (a) NEVER kill un-checkpointable in-RAM work to save idle GPU
time — redoing #518's multi-hour un-checkpointed scoring run would have
cost more than the idle burn; let such a phase finish and fix the
checkpointing in a follow-up; (b) autonomous sessions never stop a pod to
PARK — the off-pod move is valid only when the CPU phase keeps running
toward the Goal in this session (e.g. on the VM); (c) this is the
CPU-phases-off-pod rule, NOT a mid-run cost gate — the trigger is the
advisory's idle-GPU fact, never "this is getting expensive". If the phase
genuinely needs the pod (a pod-local data dependency) or is nearly done,
state that one-line reason and keep looping. The advisory never changes
the status verdict, so this handling is additive to the `status=running`
branch.

**GPU-idle escalation handling.** When a tick's JSON reports
`gpu_idle_escalation_posted: true`, the poller has just posted a louder
one-per-phase `epm:progress` marker whose note starts with
`[gpu-idle-escalation]` (plus a `gpu_idle_escalation=True` extra) AND fired a
best-effort Telegram push: a MULTI-GPU pod has been idle in an upload/CPU-only
phase for ≥ `EPM_GPU_IDLE_ESCALATION_MIN` (default 60, ≥ the advisory min) min
— the #664 spend-leak class (an 8×H200 idle in a terminal upload phase burns
~$44/hr). The orchestrator's response is the SAME as for
`gpu_idle_advisory_posted` (the escalation is the advisory's louder second
tier, not a new action): surface it in the session text, and if the remaining
work in the current phase is genuinely CPU-only and long, apply CLAUDE.md
"CPU-only phases don't hold GPU pods" — route the upload off-pod / release the
GPUs after a checkpoint — under the SAME three hard constraints as the advisory
(never kill un-checkpointable in-RAM work, autonomous sessions never stop a pod
to PARK, it is NOT a mid-run cost gate). Like the advisory, the escalation
NEVER changes the status verdict and the poller NEVER stops the pod — it
surfaces the leak loudly for action. This handling is additive to the
`status=running` branch.

**ETA-deviation / GPU-width advisory handling.** When a tick's JSON reports
`eta_deviation_posted: true`, the poller has just posted an
`epm:compute-deviation` marker (`source: poller`, `basis: elapsed-vs-plan`):
elapsed wall-time for the current phase or the whole run exceeded
`EPM_ETA_DEVIATION_MULT` (default 2.0) × the plan §9 `planned_wall_h` TOTAL —
the #763 class (an ~80× overrun a human caught ~16h late). Surface it in the
session text and weigh the run's remaining value: whether the plan's own
compute-deviation machinery should engage — a mid-run `continue_as_is`
acknowledgment, or a deliberate descope ONLY where the planner's §9
stratification spec permits one. For a fit / battery / factorization phase,
the vectorize mid-run trigger applies FIRST — run the signature check
immediately, do not wait for a second deviation
(`.claude/rules/vectorize-many-cell-fits.md` § Mid-run trigger); the
`continue_as_is` bias below scopes to the descope question. Elapsed-so-far is a lower bound on final
wall, so `continue_as_is` is nearly always the right mid-run resolution; the
poller variant carries no `action:` field and is never an auto-descope input.
When a tick's JSON reports `gpu_width_advisory_posted: true`, the poller has
posted a `[gpu-width-advisory]` `epm:progress` marker (plus a
`gpu_width_advisory=True` extra): a STABLE strict subset of GPUs sat idle ≥
`EPM_GPU_WIDTH_ADVISORY_MIN` (default 45) min on a multi-GPU pod while the run
is healthy — the #813 idle-width / #664 spend-leak class. Apply the CLAUDE.md
per-phase GPU-WIDTH right-sizing judgment (widen the parallelism to fill the
pod, or release/downsize it) under the SAME three hard constraints as the
idle advisory: (a) never kill un-checkpointable in-RAM work, (b) autonomous
sessions never stop a pod to PARK, (c) this is NOT a mid-run cost gate — the
trigger is the idle-width / elapsed-vs-plan fact, never "this is getting
expensive". Both are advisory-only: neither changes the status verdict, and
the poller stops nothing. This handling is additive to the `status=running`
branch.

**`--pid-file` is a POD-side path.** `poll_pipeline.py` evaluates
`[ -f <pid_file> ]` inside its remote SSH heredoc, so the pid file must
exist ON THE POD (the experimenter's launcher writes it there at launch
time). A pid file written only on the local VM silently reads
`PID_ALIVE=0` every tick, and the probe falls back to the pid from the
latest `epm:run-launched` marker.

**Any relaunch must re-post `epm:run-launched`.** After ANY hot-fix
relaunch of the pod workload (new pid), post a fresh `epm:run-launched`
with the new `pid=` (and `log_abs=`) before the next tick — the poller's
marker-pid fallback (`_marker_pid`) reads ONLY `epm:run-launched`
markers, so an `epm:progress` note recording the new pid is invisible to
it and the stale pid yields a false `status=dead` on a healthy run.
The same relaunch MUST also rewrite the pod-side pid file with the new
live pid in the same command chain — a present-but-stale pid file
silently probes a dead pid every tick and is rescued only while the
marker pid is itself alive (full contract + atomic recipe:
`.claude/rules/pod-side-reporting.md` § Pid-file launch contract;
incident #813 v5).
A crash-fix relaunch (a `code`-row fix round preceded it) additionally
passes the fix-commit ancestry probe and executes the declared
stale-checkpoint disposition BEFORE dispatch, recording `fix_sha=` in
the fresh marker note (`.claude/rules/crash-fix-rounds.md` § Crash-fix
relaunch: fix-commit ancestry + stale-checkpoint hygiene).
(Incident: task #521, 2026-06-10 — a VM-side pid file plus an
`epm:progress`-only relaunch produced `status=dead, pid_alive=False`
while the pod run was healthy.) On the GCP lane the marker's `pod=`
field MUST be the instance name (`eps-issue-<N>`) — `GcpBackend.poll`
matches relaunch markers on that field to follow the new process
(incident #612): a mismatched value (e.g. a RunPod-style `pod-<N>`)
rejects the marker and the poll keeps reading the frozen startup-script
phase, and an omitted `pod=` is accepted only via the launch-time
`epm:cluster-launched` timestamp baseline, so include it explicitly.

**A successful relaunch also reconciles a stale `blocked`.** Immediately
after posting the fresh `epm:run-launched`, read the current status
(`task.py view <N> --json`); if it is EXACTLY `blocked`, run
`uv run python scripts/task.py set-status <N> running --note 'relaunch
succeeded; clearing stale blocked (epm:run-launched <ts>)'`. The stale
`blocked` arises when an earlier failed round (a cap-hit, a
STATE-TO-`blocked` exit, or a failed crash-fix cycle) parked the task and
a LATER round's relaunch succeeded without flipping it back — #742 ran
healthy ~35h at status `blocked` (2026-07-01→07-02) and the
dashboard/watcher read wrong until the user asked. Guards: (a) flip ONLY
`blocked` → `running`, never any other status — a same-issue follow-up
round holds `followups_running`, never `blocked`, so the flip is inert
there by construction; (b) the flip is a same-turn serial action after
YOUR OWN relaunch — never flip on someone else's marker (the watcher's
stale-blocked FLAG pass is deliberately flag-only; a human reconciles on
its evidence); (c) if the relaunched run then fails, the normal failure
path re-blocks — the flip does not suppress it; (d) RE-READ the status
immediately before the `set-status` call — a non-`blocked` read at that
instant ABORTS the flip (a human may have reconciled off the watcher
flag concurrently; a redundant flip attempt is refused, never forced).

The 540-second sleep stays under the Bash tool's 10-minute (`600000` ms)
cap with margin; longer intervals are achievable by raising the sleep
within the cap, but 9 minutes is the operational sweet spot (enough
time to make progress, short enough to catch stalls quickly).

**MANDATORY auto-armed backstop for the per-issue session.** The
single bg-Bash poll chain above is the primary monitoring mechanism but
is NOT robust on its own: it is one chain of one-tick-at-a-time
re-invocations, and if ANY reaction turn fails to emit the next bg-Bash
tool call (corrupted/truncated tool-call text rendered as raw output, an
API drop, a session crash), the chain dies permanently with no live bg
work and no scheduled wake. The pod keeps running; the per-issue session
goes silent; results strand and GPU billing accrues until the user
notices. (Incident: task #463, 2026-06-02 — reaction turn at 01:28 UTC
emitted a tool call as raw text, no tool ran, chain died, pod ran
unmonitored for ~6.5h until the user manually re-invoked `/issue 463`.
Task #462 hit the same class of failure.)

The mandatory backstop is a harness-level recurring fire of
`/issue-tick <N>` (the LIGHTWEIGHT recurring driver — see
`.claude/skills/issue-tick/SKILL.md`) that does NOT depend on the
previous turn's bg-Bash chain surviving. Even after a dead reaction
turn, the next backstop tick fires a fresh `/issue-tick <N>` that reads
state from `events.jsonl`, refreshes the title, branches on status
(terminal/park/active/gate-park), and either tears down (terminal/park)
or hands off to the full `/issue <N>` skill for stale-marker recovery
(active with no fresh markers). The bg-Bash chain remains the primary
tick mechanism (faster, drains sentinels on each return); the recurring
`/issue-tick <N>` cron is the session-survival backstop.

**The orchestrator AUTO-ARMS this backstop itself — no user action, no
chat reminder.** For autonomous sessions, the primary arm site is
Step 0 (whole-lifecycle coverage); this Step 6d.2 arm is the SECONDARY
arm site, ARM-GUARDed so it's a no-op when Step 0 already armed. It
covers two cases Step 0 doesn't: (a) interactive (non-`--auto`) `/issue`
runs that reach the polling loop, where Step 0 deliberately skipped the
arm (interactive runs are user-driven and don't need automatic re-drive
between user turns), and (b) `--auto` sessions where the Step 0 arm
somehow didn't land (defense-in-depth — the ARM-GUARD makes the
duplicate call cheap, the missing arm catastrophic). The orchestrator
registers the cron directly via the `CronCreate` tool. The `Cron*`
tools are deferred — load them once per session with
`ToolSearch("select:CronCreate,CronList,CronDelete")` before first use.
On entering Step 6d.2 for a pod-backed `kind: experiment` run, BEFORE
starting the bg-Bash poll:

1. Call `CronList`. **ARM-GUARD:** if any job satisfies
   `prompt.strip() == "/issue-tick <N>"`, the backstop is already armed
   (this invocation was itself fired by that cron, or armed earlier
   this session) — skip straight to the poll loop. NEVER register a
   second cron for the same issue. Match on whole-string equality modulo
   surrounding whitespace, NOT `in` / `endswith` — `"/issue-tick 46"` is
   a substring of `"/issue-tick 467"`, so substring matching would
   mis-dedupe sibling issues.
2. Otherwise call
   `CronCreate(cron="*/45 * * * *", prompt="/issue-tick <N>", recurring=True, durable=False)`
   — a 45-minute, session-scoped, in-memory recurring fire of the
   lightweight `/issue-tick <N>` skill (dies with the session, auto-
   expires at 7 days like the default pod TTL; the harness jitters
   recurring fires so ticks don't all land on a fixed wall-clock mark).
   The 45-minute interval (lengthened from 20 min on 2026-06-12) is
   chosen deliberately: the pure-Python `autonomous_session_watch.py`
   cron (every 10 min, free) carries ALL fast detection — DEAD-session
   respawn, alive-but-stalled respawn for ACTIVE statuses, pod safety,
   gate-park phone push, title reconcile — so the tick is purely the
   in-session re-driver of last resort for the alive-but-stalled-at-PARK
   class, which tolerates 45-min latency. Every tick fire is LLM-priced
   (a cold context read even on the guarded-no-op path), so fewer fires
   is the point. (The old 20-min rationale leaned on a "5-minute prompt
   cache TTL"; that figure is inaccurate for this org's subscription
   auth — subscription sessions get the 1-hour cache TTL automatically,
   5 minutes applies to API-key auth — and the interval choice no longer
   depends on it.) Then immediately re-`CronList`
   and assert EXACTLY ONE job matches
   `prompt.strip() == "/issue-tick <N>"`. If the harness normalised the
   stored prompt such that the ARM-GUARD would later miss, this assert
   fails loud NOW rather than silently stacking a duplicate cron on
   every subsequent re-entry.

Then proceed to the polling loop. Auto-arming HERE is required ONLY for
pod-backed `kind: experiment` runs reaching Step 6d.2;
`kind: analysis|infra|batch|survey` paths that never enter the polling
loop do NOT arm it here. A same-issue follow-up round is NOT exempt —
it arms at its OWN entry (Step 9b § Loop liveness backstop / the C3 +
step-6 re-arm), and one that reaches this polling loop re-arms via the
ARM-GUARD (a no-op when already armed).

**CRON-TEARDOWN procedure (run INLINE at every terminal / park exit site,
not only here in prose) — hardened 2026-06-12; widened + idempotent
2026-07-05 (#1052).** Sweep the cron store with a TWO-LEG match set,
resolving ids from a FRESH `CronList` at teardown time (#988 — never
`CronDelete` an id recorded earlier in the session: recorded ids go stale
when a one-shot fires or a concurrent teardown wins the race). Delete
EVERY job matching EITHER leg:

- **Leg 1 — the recurring tick cron:** primary match is whole-string
  equality (`prompt.strip() == "/issue-tick <N>"`); hardened fallback is
  the anchored pattern `issue-tick\s+<N>(?!\d)` (harness
  prompt-normalization drift was the #501 failure mode — the whole-string
  teardown silently no-oped 1,951 times; the `(?!\d)` guard prevents
  sibling mis-delete, `"/issue-tick 46"` never matches
  `"/issue-tick 467"`).
- **Leg 2 — stray one-shot `/issue <N>` wakeups (#980 — a live one-shot
  wakeup that survives past terminal re-drives the FULL skill on a
  completed task):** primary match is whole-string equality against the
  bare full-skill wakeup prompt (the f-string form in the canonical
  block); fallback is the START-anchored pattern `/issue\s+<N>(?!\d)` via
  `re.match` (the start anchor keeps deletion surgical — a prose prompt
  that merely MENTIONS the issue never matches; `(?!\d)` guards siblings;
  the `-` in `/issue-tick` fails `\s+`, so leg 2 never re-matches leg 1's
  job; trailing text like `--auto` matches by design).

A `CronDelete` error indicating the job does not exist (observed shape:
`No scheduled job with id …`) is SUCCESS, not a failure (#988) —
idempotent means the job being gone is the goal: continue the sweep,
never retry that id, never abort or raise on it. Then
ASSERT-AFTER-DELETE over BOTH legs: re-`CronList` and verify no job
matching EITHER leg survived; if one did, retry the delete ONCE (fresh
id from the re-list), then log LOUDLY — the runaway parachute
(`tick_triage.py`'s 3-consecutive-terminal flag + the watcher's
force-stop) bounds the damage of a cron that refuses to die. Canonical
pseudocode: `.claude/skills/issue-tick/SKILL.md` § CRON-TEARDOWN.

**Prevention ban (#980).** An `/issue` session must NEVER schedule its
own re-drive — no `ScheduleWakeup` wakeup and no `CronCreate` one-shot,
regardless of prompt shape. The Step 0 / Step 6d.2 `/issue-tick <N>`
cron is the ONLY sanctioned self-wake: a one-shot wakeup may not be
enumerable at teardown time, and one that fires after terminal re-drives
a completed task (#980). Leg 2 + the self-heal sweep BOUND — they do not
guarantee deletion of — whatever the store fails to surface.

The backstop
DELIBERATELY survives the `done` → `verifying` transition (Step 6d.3) and
keeps re-firing through the uploading / verifying / interpreting /
reviewing stages — those stages have no other auto-wake, so the backstop
is the only thing that revives an interactive per-issue session that
stalls there. It is torn down ONLY at the true terminal / park
transitions:

- at `awaiting_promotion` (Step 9b — the pod was terminated at Step 8 and
  this is a human gate, so no more auto-driving);
- at `completed` (Step 10 auto-complete);
- at any `set-status <N> blocked` exit in Step 9 / the
  interpretation+review loop;
- at the `status=stalled` / `status=dead` / unrecognised-gate `blocked`
  exits in the poll loop above; and
- at the Step 6d.4 gate-park exit (the pipeline has EXITed and no pod is
  burning GPU — the user now drives the resume).

Each of those exit sites carries an explicit "run CRON-TEARDOWN" line. A
gate resume or a recovery re-invocation re-enters Step 6d.2 and re-arms
via the ARM-GUARD.

Surviving the backstop into verifying / interpreting / reviewing is the
DESIGNED behavior, not an accident we tolerate. Its only cost — a tick
landing while a stage subagent is already in flight and REDUNDANTLY
re-dispatching that stage (analyzer, clean-result-critic, upload-verifier)
— is bounded by the Step 9 **idempotency guard**: a tick that lands on a
stage whose latest `events.jsonl` marker is a fresh dispatch with no
terminal/result marker yet EXITs without re-dispatching, so the live work
finishes uninterrupted (concrete rule in Step 9). State stays coherent
regardless because every re-entry reads `events.jsonl` fresh. If a
teardown at a terminal/park transition is ever missed, the residue is
cheap: the cron auto-expires at 7 days, and a tick landing on a
`completed` / `archived` / `awaiting_promotion` issue is a no-op that
SELF-HEALS (the re-invoked skill reads terminal/park state, exits without
re-arming, and runs the two-leg sweep before exiting — so a wakeup that
escaped an earlier teardown deletes its own stray siblings when it fires;
the blast-radius bound for whatever the store fails to surface).
Run CRON-TEARDOWN the moment you spot a stranded cron or stray one-shot
wakeup (fresh `CronList` → `CronDelete`, both legs).

Residual failure mode the in-session backstop does NOT cover: if the
per-issue *session itself* dies (process exit, host reboot), a
`durable=False` cron dies with it and the pod goes unmonitored. Two
mechanisms cover that, with DIFFERENT strength:

1. The "spawn a fresh session" recovery row recovers the work.
2. The EXTERNAL pod-safety backstop
   (`scripts/autonomous_session_watch.py`, the every-10-min VM cron
   `3-59/10 * * * *`) reconciles RUNNING managed pods (`pod-<N>`, legacy
   `epm-issue-<N>` still recognized) against their task STATUS. It is
   CONSERVATIVE by design:
   - it AUTO-STOPS (reversible — `pod.py stop`, never terminate, after ≥
     2 consecutive checks) only a RUNNING pod whose task is already DONE
     (`completed` / `awaiting_promotion` / `archived`) — i.e. an ESCAPED
     pod (Step-8 terminate failed, or the pod never went through Step 8).
     A done task provably needs no pod, so this stop is unambiguous;
   - it does NOT auto-stop a pod whose task is still mid-run
     (`approved` / `running` / `verifying`). For those it ALERTS (a loud
     log line + a one-time dashboard-visible marker on the task) when no
     real progress marker has landed for > 6h — a likely abandoned
     session — but leaves the pod RUNNING. A false alert is a cheap
     nudge; a false stop would kill a healthy long run, so the backstop
     never makes that trade. `blocked` pods are KEPT (alert-only if
     stale), never auto-stopped. `interpreting` / `reviewing` pods
     classify as "other" (those stages don't drive pods — interp/review
     reads from WandB/HF), so they're kept too and caught later when the
     task reaches `awaiting_promotion`.

So the external backstop bounds GPU burn for the clean case (a finished
experiment whose pod escaped termination) and SURFACES the harder case
(a session that died mid-run) for human action — it does NOT silently
stop mid-run pods. Full mid-run auto-stop (e.g. a pod-side idle-GPU
probe that distinguishes a stalled run from a slow one) is a noted
follow-up, not implemented. No crontab change is needed — the watcher is
already scheduled.

The pre-2026-06-02 independent stall-watchdog (`scripts/pod_watch.py`
spawned as a long-lived background process writing to
`.claude/cache/watch-<N>.pid`) was retired alongside the orchestrator
polling loop; it is NOT the backstop here. See "Notes on the
obsolete monitoring stack" below for the single source of truth on
which mechanisms are live vs retired.

**Long-phase heartbeat duty (BINDS every >60-min quiet stretch — ALL
loops, BOTH session modes; #1207, incidents #1092/#825/#1112
2026-07-08).** Nothing external refreshes a session's liveness signals
between status transitions: the tick skill no longer touches the
self-report (issue-tick SKILL.md § "Title refresh — moved to the
watcher") and the watcher's reconcile is status-transition-keyed by
design. So during any stretch where THIS session awaits work and
>60 min could elapse without a turn that posts a non-watcher marker —
an ad-hoc bg-Bash poll chain, a `Monitor` until-loop, a
deadline-bounded Batch-API poll, a detached VM phase (§ "Detached
VM-side long compute phases", Step 9 entry guard), or any
follow-up-round wait at `followups_running` — the orchestrator carries
BOTH duties below. (The Step 6d.2 polling loop above discharges them by
construction: the top-of-tick `set_title` refresh + the ≤30-min clamped
tick interval. The duty is for every wait that is NOT that loop. A long
FOREGROUND subagent wait is a named out-of-scope shape — no resumable
orchestrator turn exists there to discharge the duty; the watcher's K=2
live-escalation debounce covers it.)

1. **Structure the wait so a turn resumes at least every ~45 min.**
   Cap any single blocking wait at ≤45 min — chain bg-Bash sleeps /
   segment a `Monitor` until-loop
   (`until <check> || [ $(elapsed) -gt 2700 ]; …`) rather than arming
   one silent multi-hour wait. A single 4-h until-loop (#1092,
   2026-07-08) leaves zero heartbeat opportunities: the watcher's
   90-min exemption leash (`LONG_PHASE_HEARTBEAT_FRESH_S`, sized as a
   ~60-min cadence + 30-min slack) lapses mid-wait no matter what was
   posted before entering it. 45 min matches the `*/45` tick cadence
   and keeps every resume inside the 60-min self-report window
   (`STALLED_WINDOW_S`).
2. **At each resume ≥~45-60 min into the phase (a ~60-min heartbeat
   cadence): verify, then heartbeat + refresh.** (i) VERIFY the awaited
   work is alive with cheap evidence — `ps -p <pid> -o args=` identity
   match, breadcrumb `log=` mtime advanced, a Batch-API status read, a
   poll-tick JSON line; (ii) post the heartbeat marker, evidence in the
   note:

       uv run python scripts/task.py post-marker <N> epm:progress \
         --note "[long-phase-heartbeat] <phase>: <one-line evidence, e.g. pid 12345 alive, log +3 lines>"

   (iii) refresh the self-report:
   `uv run python scripts/session_progress_report.py --issue <N> --step "<phase>"`.
   The two writes refresh BOTH staleness signals — the sparing is never
   the 90-min leash alone: the marker buys the stalled-detector leash
   (`autonomous_session_watch._long_phase_heartbeat_reason`) AND
   converts `tick_triage.py`'s STALE-REDRIVE to HEALTHY (#1051), while
   the self-report refresh keeps signal 1 (`STALLED_WINDOW_S`) fresh so
   the detector never reaches the exemption probe at all. NEVER
   heartbeat blind: if the verify FAILS (pid gone, log frozen, batch
   errored), do NOT post a heartbeat — run the failure path (crash-fix
   routing / `epm:failure`). A heartbeat without evidence shields a
   dead phase from recovery for up to 90 min and is the banned inverse
   of the false-respawn this duty prevents. (Pid-bearing detached-phase
   breadcrumbs stay authoritative over heartbeat notes — tick_triage
   #1051.)

Revival trigger for the deferred watcher-side option (b) (#1207
§11-R4): a STALLED-DETECTOR-lane force-respawn of a session carrying a
fresh (<90-min) heartbeat is the recorded evidence that emitter-side
duty is insufficient — a wedge-lane respawn of a duty-compliant session
is by design (#1127) and does NOT count.

##### Step 6d.3: On `status=done`

Do NOT run CRON-TEARDOWN here. The backstop INTENTIONALLY survives past
`done` into the uploading / verifying / interpreting / reviewing stages —
those stages have no other auto-wake, so an interactive per-issue session
that stalls in them would otherwise go silent forever. The cron is torn
down only at the true terminal / park transitions: at `awaiting_promotion`
(Step 9b), at `completed` (Step 10 auto-complete), and at any
`set-status <N> blocked` exit in Step 9 / the interpretation+review loop
(plus the poll-loop stalled/dead/blocked exits and the Step 6d.4 gate-park
that already tear it down). The Step 9 idempotency guard (below) bounds the
redundant-subagent cost a surviving-into-`done` cron used to risk.

Transition the task to `verifying` (the upload-verifier next):

> **Same-issue follow-up round?** At `followups_running`, SKIP this
> `set-status` (status-hold rule, Step 9b § Same-issue follow-up loop step 3;
> code-enforced — `task.py` refuses the flip) — phase visibility comes from
> `stage=followup-<phase>` breadcrumbs, not status flips.

```bash
uv run python scripts/task.py set-status <N> verifying \
    --note "polling loop observed phase=done"
```

Then proceed to Step 7 (which handles results → upload routing).

##### Step 6d.4: On `status=gate` — handle a pod-side gate (park OR auto-resolve)

Pod-side dispatchers cannot post markers directly (the `task.py`
branch-guard and the CLAUDE.md "Pod-side code NEVER shells out" rule),
so they write a sentinel file at `/workspace/logs/issue-<N>-*.json`
that `poll_pipeline.py` drains. When a sentinel carries a non-empty
`gate` field **AND `blocks_pipeline: True`**, the poller posts the
carried marker from the VM (e.g. `epm:fact-candidates v1`) and returns
`status=gate` with `gate=<name>`.

The poller ONLY surfaces `status=gate` when the drained sentinel had
`blocks_pipeline: True` (the field defaults to True when absent, so a
sentinel that carries only a `gate` name still parks). Sentinels with
`blocks_pipeline: False` are the dispatchers' benign phase-progress
signals (`gate=phase`, `gate=smoke`, `gate=dryrun` are the canonical
ones): their marker IS posted from the VM, but they NEVER end the
polling loop and NEVER trigger the fail-fast block. They are NOT user
gates — do not treat a `blocks_pipeline: False` phase signal as an
unrecognised gate (incident #641).

The orchestrator handles the named gate inline rather than continuing to
poll — the pipeline itself has EXITed at the gate. Most gates are PARK-mode
(`fact-candidates`): the pipeline is waiting on a user answer, so the
orchestrator parks. An AUTO-RESOLVING gate (`pv_phase1_done`) is the
exception: the orchestrator resolves it itself (on the RunPod lane a
pod-cycle around an off-pod step; on the GCP lane finalize + fresh
dispatch — see the GCP-lane teardown leg below) and resumes the loop in
the same turn — see the per-gate handlers below.

**GCP-lane blocking gates — instance-teardown leg (EVERY handler, PARK-mode
and auto-resolving; #908/#763/#935).** On the GCP lane a blocking-gate exit is
a CLEAN exit (`[phase=done]` → guest `eps/phase=done`), so the in-VM EXIT trap
does NOT power the VM off — the GCE instance stays RUNNING only within the
bounded done-grace window (default 90 min,
`EPS_GCP_DONE_POWEROFF_GRACE_SECONDS`; the #935 self-poweroff best-effort
persists the undrained sentinel set to HF `issue<N>_done/<attempt_id>/` at
expiry, then powers off; the clean-exit path keeps it alive only for sentinel
draining — `backends/gcp.py` `teardown` docstring). The finalize teardown leg
below remains PRIMARY — never wait out the grace. Two operational lines
(REQUIRED — the only defense on the DELETE outcome): (a) on a
`workload_done_self_poweroff` poll (TERMINATED + guest `eps/phase=done` — the
grace expired on the STOP outcome) OR a post-expiry `dead("instance not
found")` poll, CHECK the HF data-repo prefix `issue<N>_done/<attempt_id>/`
BEFORE any crash-fix routing — the run SUCCEEDED and self-powered-off;
recover the undrained completion/gate sentinels from that prefix and run
finalize with `--skip-confirm-artifacts`. (b) Gate sentinels persist to that
same prefix at grace expiry on a BEST-EFFORT basis (one retry) — the prefix
is NOT guaranteed to exist, and the persist also never fires on a mid-grace
preemption or manual stop, so an ABSENT prefix does NOT distinguish "poller
drained normally" from "expiry persist failed"; on a STOP-outcome instance
the `eps/done_persist` guest attribute (`ok|failed`, a SEPARATE key from
`eps/phase`) disambiguates. A `workload_done_finalize_failed` poll (#1055 —
the deliverables-verified-then-finalize-crashed classification, guest
`eps/phase=finalize_failed_artifacts_ok`) is a done-like shape too: treat it
as a SUCCESSFUL run (no crash-fix routing) and run finalize with
`--skip-confirm-artifacts` exactly as for `workload_done_self_poweroff`
(the completion sentinel was never written; see
`.claude/rules/compute-backend-failover.md` § Part A-ter). Pre-existing
residual (unworsened by #935): a
#908 stale reclaim at a FRESH dispatch during the grace deletes the VM
before the expiry persist fires. By the time `status=gate` reaches
this step the sentinel is already drained (the poller drained it to post the
gate marker), and the VM holds NOTHING a later gate resolution needs — so
tear the instance down at the earliest point after the drain, split by gate
class: **PARK-mode gates** run the finalize command after the gate marker is
posted and BEFORE the park — before raising the user question, before
posting `step-completed parked`, before exiting — NEVER leave the instance
up through the user-wait window (the user's pick is NOT a teardown
precondition); **auto-resolving gates** (including a PARK-mode gate's
autonomous auto-resolve branch) run it after the auto-resolve step completes
and BEFORE dispatching any off-pod phase or the fresh tail dispatch:

```bash
uv run python scripts/dispatch_issue.py finalize --issue <N> --skip-confirm-artifacts
```

`finalize` DELETEs the instance AND retires the handle sidecar
(`.claude/cache/issue-<N>-handle.json` → `<name>.finalized`); a raw `gcloud
compute instances delete` leaves a stale sidecar the next launch would
misread — use it only when no sidecar exists. `--skip-confirm-artifacts` is
REQUIRED at a mid-pipeline gate: the run's declared final artifacts do not
exist yet, so a plain `finalize` FAILs confirm (exit 3) by construction.
(Mid-pipeline gate teardowns run BEFORE any upload-verifier dispatch or
`epm:results`, so the #1026 verifier-currency gate is a no-op here — no
verifying crumb, no results, no verdict to be stale against.) The
instance stays up ONLY for sentinel draining — never through an off-pod
phase or a park (Step 8-bis: a pod must not idle on a halt; incident #763:
an A100-80 idled ~40 min after the `cofit_phaseA_done` gate-park). The next
pipeline phase provisions FRESH via the normal Step 6d.1 dispatch. There is
no GCP analogue of the RunPod `pod.py stop`/`resume` cycle (`pv_phase1_done`
below): GCE instances are ephemeral by design, and a STOPPED instance would
be deleted by the next launch's stale reclaim anyway — the GCP phase-cycle
is teardown + fresh dispatch. Backstop only (never the plan):
`backends/gcp.py::reconnect_or_none` refuses a RUNNING instance whose
`eps/phase` is terminal (`done`/`failed`/`finalize_failed_artifacts_ok`/`wedged`) and the pre-launch stale
reclaim deletes it, so a missed teardown no longer silently no-ops the next
dispatch (#763 leg 2) — but the zombie still bills until the #935 done-grace
self-poweroff (default 90 min), that next dispatch, or the daily janitor
sweep, so the handler-side teardown stays mandatory (never wait out the
grace).

Gate handlers (one per registered `<name>`):

- **`fact-candidates`** (used by `run_experiment_<N>.py`-style
  fact-teaching drivers, originally task #407): the `epm:fact-candidates
  v1` marker carries a ranked candidate table (one row per Wikipedia-
  stub fact passing the log-prob band filter, with `id` + summary
  + log-prob). The orchestrator reads the just-posted marker via
  `task.py latest-marker <N> --kind epm:fact-candidates`, then branches
  on session mode.

  **Interactive mode** (`EPM_AUTONOMOUS_SESSION` unset/falsy): surface
  the table via `AskUserQuestion` <!-- gate: gates.fact_candidates --> and
  ask the user to pick one `id`.

  <!-- gate: gates.fact_candidates -->
  <!-- autonomous-mode: auto-resolve -->
  ```python
  # Interactive mode only — autonomous mode auto-picks the median-log-prob id.
  AskUserQuestion(questions=[{
      "question": "Phase 0 (fact-candidates) — pick the fact for the obscure-real regime.",
      "header": "Pick fact (id)",
      "multiSelect": False,
      "options": [
          # one option per candidate id, label = "<id>: <one-sentence summary>"
          ...,
      ],
  }])
  ```

  **Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): NEVER raise the ask
  AND never print the candidate options as a text menu. Auto-resolve per
  § Autonomous session behavior → `fact_candidates`: pick the candidate
  `id` with the median per-token log-prob (the middle of the band the
  plan filtered by). State `Decision: id=<X> (median log-prob in band)`
  AND EXECUTE the resume in this same turn (post `epm:fact-pick v1` with
  `id: <X>` and resume the polling loop); do NOT state the Decision and
  then end the turn.

  On user reply (interactive) or auto-pick (autonomous), post
  `epm:fact-pick v1` with the chosen id in the note body (`id: <N>`):
  ```bash
  uv run python scripts/task.py post-marker <N> epm:fact-pick \
      --note "id: <chosen_id>"
  ```

  In interactive mode the user then re-invokes `/issue <N>` to resume;
  the driver's `--phase fact-pick` step reads the latest `epm:fact-pick`
  marker, materialises `fact_pick.json` on disk, and the next pipeline
  phase proceeds. In autonomous mode the orchestrator resumes the
  polling loop directly without a re-invocation. (See plan §4.2 of any
  fact-teaching task for the on-pod resume contract. GCP lane: the
  instance was already finalized per the GCP-lane teardown leg above —
  the resume is a FRESH Step 6d.1 dispatch of the fact-pick tail, never
  a poll against the old instance.)

- **`pv_phase1_done`** (issue #763 persona-vector extraction —
  off-pod judge between two GPU phases): an AUTO-RESOLVING gate, NOT a
  user park. The dispatcher `scripts/issue763_dispatch.sh` runs GPU
  phase 1 (`generate + capture + pv_extract_generate + upload-progress`
  — the PV rollouts are now on the HF data repo), emits the blocking
  sentinel `gate=pv_phase1_done` via
  `scripts/issue763_upload.py --emit-gate pv_phase1_done` <!-- lint: historical-ref --> (called from
  `scripts/issue763_dispatch.sh` after upload-progress)
  (`write_sentinel("epm:gate", …, blocks_pipeline=True)`), and EXITs.
  Unlike `fact-candidates` (which PARKS for a user pick at
  workflow.yaml § gates.fact_candidates), the
  orchestrator resolves this gate ITSELF — it does NOT raise
  `AskUserQuestion`, does NOT post `epm:step-completed --exit-kind
  parked`, does NOT exit the skill, and does NOT CRON-TEARDOWN — by
  orchestrating a pod-cycle around an off-pod judge step and then
  RESUMING the polling loop in this same turn. This handler behaves
  IDENTICALLY in interactive and `EPM_AUTONOMOUS_SESSION` modes (there
  is no user decision to make — the gate is fully auto-resolved).
  <!-- autonomous-mode: auto-resolve -->
  Concretely:

  1. **Stop the GPU pod** (volume preserved): `uv run python scripts/pod.py
     stop --issue <N>`. This frees the GPU through the deadline-bounded
     stop (see Step 6d.2 § "Stop the pod" / "Notes on the obsolete
     monitoring stack"); the PV rollouts are already on HF, so nothing on
     the pod's ephemeral disk is lost. (RunPod lane. On the GCP lane there
     is no stop/resume — apply the GCP-lane instance-teardown leg above:
     `finalize --skip-confirm-artifacts` once the phase-1 artifacts are
     confirmed on HF, then re-dispatch the tail as a fresh launch.)
  2. **Run the judge OFF-POD on the VM**: `uv run python
     scripts/issue763_extract_pv_rb.py --phase judge`. <!-- lint: historical-ref --> (This script
     ships on the `issue-763` branch — it lands on `main` when #763 merges;
     the reference is a forward / sibling-branch one, not a dead tool.)
     This is VM-safe by construction and NOT a `task.py` pod-shellout: it
     fetches the PV rollouts from HF via `snapshot_download` (NOTE: on the
     ~1M-file data repo `snapshot_download` wedges in full-tree
     enumeration — `.claude/rules/gotchas.md`; patch the script to scoped
     `list_repo_tree(path_in_repo=...)` staging before re-running this
     phase on a #763 follow-up), batch-judges
     through
     `eval.batch_judge` (the deadline-bounded client — never a hand-rolled
     `messages.batches.create` + deadline-less poller), and uploads the
     keep-flags back to the issue HF prefix. It needs no GPU and posts no
     `task.py` markers itself (it is an off-pod analysis step; the
     orchestrator owns the poll-loop markers).
  3. **Resume the pod**: `uv run python scripts/pod.py resume --issue <N>`
     — new IP/port; `pods.conf` + `~/.ssh/config` + MCP config auto-refresh
     on resume (re-run `/mcp` if the SSH MCP entry needs the refreshed
     host/port; if SSH keeps failing on a stale port, pull the live
     host/port back with `pod.py config --refresh-from-api` per Step 6b
     "stale-port recovery", the #488 13h-loop failure class). CONFIRM the
     resumed pod is reachable (`uv run python scripts/pod.py health
     --quick`) before re-dispatching.
  4. **Re-dispatch the workload tail** at `--from-phase pv_capture` via
     the SAME experimenter launch pattern as the original launch
     (Step 6d.1): spawn the `experimenter` subagent with the workload
     command `bash scripts/issue763_dispatch.sh --from-phase pv_capture`
     and the resumed pod's name (`pod-<N>` / `epm-issue-<N>`). The
     dispatcher resumes at capture → E0 judge → fit → figures → final
     upload → `[phase=done]`. The experimenter posts a fresh
     `epm:run-launched` marker (new `pid` + `log_abs`) and exits, exactly
     as on the first launch; the orchestrator updates its local poll-loop
     `pid`/`log` from that marker.

  Then **RESUME the polling loop** (Step 6d.2) at the next tick — do NOT
  exit, do NOT park, do NOT CRON-TEARDOWN. The gate has auto-resolved.
  (RunPod lane — the stop/judge/resume cycle above keeps ONE pod across
  phases, so the pod is burning GPU again after resume. GCP lane: there
  is no stop/resume — the instance was finalized per the GCP-lane
  teardown leg above, and the tail runs as a FRESH dispatch, so the poll
  loop resumes against the NEW handle, never the old instance.) Either
  way the `/issue-tick <N>` backstop cron stays armed and the bg-Bash
  poll chain continues. (Contrast with `fact-candidates` above, which parks
  for a user pick and tears the cron down.) **Idempotency on re-entry:**
  if a re-entry observes an `epm:gate v<n>` for `pv_phase1_done` followed
  by a FRESH `epm:run-launched` (post-resume; ts > the gate marker),
  treat the gate as already resolved and proceed with normal polling — do
  NOT re-stop / re-judge / re-dispatch.

- **Unrecognised `gate` name**: this branch fires ONLY for a sentinel
  the poller surfaced as `status=gate` — i.e. one that carried
  `blocks_pipeline: True`. A non-empty gate name with
  `blocks_pipeline: False` (`gate=phase` / `gate=smoke` / `gate=dryrun`)
  is filtered out by the drain and NEVER reaches this branch, so it is
  NOT an unrecognised gate and MUST NOT trigger the block below. For a
  genuinely unrecognised (blocking) gate name: log a one-line WARN, post
  `epm:failure
  v1` with `failure_class: code` and `reason: unrecognised_gate_name`
  (the `code|infra|data` taxonomy has no `workflow` class; the failure
  classifier defaults unknown classes to `code` anyway), a note pointing
  at the unrecognised gate name + the sentinel path, run CRON-TEARDOWN
  (fresh `CronList` → `CronDelete` every job in the two-leg match set:
  the recurring `/issue-tick <N>` tick + stray one-shot `/issue <N>`
  wakeups; not-found = success; § CRON-TEARDOWN procedure), set
  `status:blocked`, exit. This forces a workflow-fix-candidate before
  the gate name can silently no-op.

**PARK-mode gates only** (`fact-candidates` and the unrecognised-gate
branch): the tail below applies ONLY to gates that exit the skill to wait
on a human. Auto-resolving gates like `pv_phase1_done` (above) handle
their own continuation — they do NOT tear down the RUNPOD pod mid-cycle
(the stop/judge/resume cycle IS the continuation) and do NOT exit; on the
GCP lane the auto-resolve handler DOES tear the instance down (finalize +
fresh dispatch, per the GCP-lane teardown leg above) — "no teardown" is
RunPod-scoped. Their handler resumes the polling loop in the same turn,
so it skips this whole paragraph.

For a PARK-mode gate: run CRON-TEARDOWN before parking (the HARDENED +
WIDENED Step 6d.2 procedure: fresh `CronList` → `CronDelete` every job
in the two-leg match set — the recurring `/issue-tick <N>` tick +
stray one-shot `/issue <N>` wakeups; not-found = success;
assert-after-delete, retry once; § CRON-TEARDOWN
procedure) — the pipeline has EXITed and no pod is
burning GPU (on the GCP lane because the teardown leg above already
finalized the instance BEFORE this park), so the backstop should not keep
re-firing `/issue-tick <N>` (which
would re-surface the gate question every 45 min). The user's
re-invocation after posting the resume marker re-enters Step 6d.2 and
re-arms via the ARM-GUARD. After posting the resume marker, EXIT the
skill cleanly via `uv run python scripts/post_step_completed.py --issue <N>
--step 6d --exit-kind parked` (the §5 `epm:step-completed` marker); the
user's re-invocation of `/issue <N>` resumes the polling loop. The polling-loop's terminal
transitions are now `running → verifying` (on done), `running → running`
(after a parked-and-resumed user gate, OR after an auto-resolving gate's
handler returns), or `running → blocked` (on stalled/dead or unrecognised
gate).

##### Notes on the obsolete monitoring stack

The `experimenter` agent NO LONGER monitors the run. The
`scripts/pod_watch.py` watchdog (referenced in older revisions of this
skill, and still callable via `scripts/pod.py watch ...` for manual /
debug use) is NOT spawned by Step 6d anymore — the orchestrator's
polling loop subsumes stall detection.
The "Progressive monitoring schedule" table that previously appeared
in the experimenter agent spec has been removed.

**Backstop story (single source of truth — both this note and the
recovery table below must agree).** The live mechanisms during a
`running` (workload) phase are exactly two, in order:
1. The orchestrator's bg-Bash poll chain (Step 6d.2) — primary, drains
   sentinels and posts `epm:progress` / advances on done / blocks on
   stalled-or-dead.
2. The auto-armed backstop cron (`CronCreate(cron="*/45 * * * *",
   prompt="/issue-tick <N>")`, registered by the orchestrator at Step 6d.2,
   torn down at every terminal/park transition — NOT at `done`; see
   Step 6d.2 CRON-TEARDOWN) running in the per-issue
   session — backstop. Survives a dead reaction turn and re-enters the
   polling loop on its next tick (via the lightweight `/issue-tick <N>`
   skill's stale-marker recovery branch, which loads the full `/issue
   <N>` skill when needed). The orchestrator no longer depends on the
   user typing a `/loop` to keep things going.

The `pod_watch.py` watchdog + `.claude/cache/watch-<N>.pid` pid-file
are NOT a third live mechanism: they are retained for manual
invocation only and are NEVER auto-spawned by this skill, NEVER
required for a healthy run, and NEVER referenced by an unattended
recovery path. If a recovery row below says "watchdog crashed",
read it as "the bg-Bash poll chain has no live tick AND no scheduled
`/loop` wake" — not "respawn pod_watch.py".

Status stays at `running` throughout the polling loop. The polling
loop's terminal transitions are `running → verifying` (on done),
`running → running` (a gate resolved per Step 6d.4 — either a PARK-mode
user gate that resumes on the next `/issue <N>` invocation after the user
posts the gate-resume marker, OR an auto-resolving gate like
`pv_phase1_done` whose handler cycles the pod and resumes the loop in the
same turn), or `running → blocked` (on stalled/dead or an unrecognised
gate name).

### Step 7: Monitor -> results

Under the new orchestrator-owned polling model (Step 6d.2), three event
sources contribute to `running`-phase progress:

- **Experimenter (subagent, single turn at launch)**: posts
  `epm:run-launched` once and exits.
- **`poll_pipeline.py` (run by the orchestrator's bg-Bash loop)**: posts
  `epm:progress` on each phase transition observed in the pod log.
- **Entry script on the pod**: writes `[phase=done]` to its log on
  graceful completion AND writes a JSON sentinel file at
  `/workspace/logs/issue-<N>-results.json` containing the
  `epm:results` payload. The orchestrator's polling-loop terminal
  tick (Step 6d.2) reads the sentinel on its next poll and posts
  the `epm:results` marker from the local VM via `task.py post-marker`. The
  pod NEVER calls `task.py` directly — enforced by
  `tests/test_no_pod_side_task_py_shellout.py` and the CLAUDE.md
  "Pod-side code NEVER shells out to scripts/task.py" rule. Task #397
  round 9 (2026-05-27) burned a launch on a pod-side
  `task.py find <N>` shellout that hit the branch-guard refusal; the
  same failure class applies to `task.py post-marker`, hence the
  sentinel-file pattern is canonical.

  Sentinel format (JSON object with these keys, all required):
  - `eval_numbers` (inline dict of final eval metrics)
  - `eval_paths` (list of repo-relative paths to eval result JSONs)
  - `reproducibility_card` (dict matching CLAUDE.md template; filled in
    with TBD → resolved values. **For training / sweep runs the card
    MUST carry the machine-resolvable fields
    `scripts/verify_uploads.py` self-resolves** (`merged_results_card`
    → `check_hf_model_from_card` / `check_wandb_from_card`):
    `adapter_paths` as an explicit per-cell mapping of REAL HF
    subfolder paths — every value existence-checked under
    `hf_model_repo` (defaults to the canonical model repo; declare only
    when different), so NO `<arm>`/`<source>`/`<seed>`-style template
    placeholders and no `(16 adapters)` prose summaries — plus
    `wandb_project` AND `wandb_run_names` (per-cell dict or list of run
    display names; a single run may instead declare `wandb_run_path`).
    Prose may accompany but NEVER replace these structured fields: a
    prose-template card (`adapters/issue_<N>/<arm>/<source>_seed<S>
    (16 adapters)` + a free-text `wandb:` line) resolves to nothing and
    trips false `hf_model` / `wandb_run` MISSING rows on a
    fully-uploaded sweep that the upload-verifier must then supersede
    row-by-row — incident #612.)
  - `wandb_url` (string)
  - `hf_hub_url` (string)
  - `worktree_path` (string, absolute path on local VM)
  - `final_commit_sha` (string, 40-char SHA)
  - `gpu_hours_used` (float)
  - `gpu_hours_budgeted` (float)
  - `plan_deviations` (list of `{deviation: <str>, rationale: <str>}`)

  **Orchestrator-composed fallback.** When the driver emits only
  granular per-cell / per-shard sentinels (no single results sentinel)
  and the orchestrator composes the `epm:results` payload itself
  from the drained pieces, the composed payload obeys the SAME contract
  above — in particular the `reproducibility_card` structured-field
  requirement. Composing the card's adapter / WandB info as prose is
  the #612 failure mode; assemble the explicit `adapter_paths` mapping
  and `wandb_project` + `wandb_run_names` from the per-cell sentinels
  instead. (GCP-lane driver sentinels that declare
  `production_provenance.<cell>.hf_adapter_subfolder` /
  `.wandb_run_name` are already self-resolvable — `verify_uploads.py`
  synthesizes the card from them (#599) — so carry that structure
  through verbatim rather than flattening it to prose.)

When this skill is re-invoked in `running`:

1. Check `epm:results` exists. If not, show last progress, post the §5
   marker:
   ```bash
   uv run python scripts/post_step_completed.py --issue <N> --step 7 \
     --exit-kind parked \
     --notes "experimenter still running; epm:results not yet posted"
   ```
   and EXIT. **If the most recent `epm:progress` event is older than 4
   hours and there is no `epm:results` or `epm:failure`, post
   `epm:stale v1` asking the user to investigate (the experimenter may
   have crashed silently); leave status at `running`.**
2. If `epm:failure` posted: route via the **failure classifier**. The
   `epm:failure` body SHOULD include a `failure_class: infra | code | data`
   field on its first non-blank line. A `data` class (a factual gap only
   the user can fill) is posted per the halt-criterion contract together
   with `status:blocked`, so it never reaches this step — the table below
   routes `infra | code` only:

   | failure_class | Cause example | Action |
   |---|---|---|
   | `infra` | OOM, ENOSPC, NCCL, vLLM init failure, SSH refused, 401/gated repo, library traceback (vllm/transformers/peft/trl/torch/xformers), a zombie-GPU-allocation stall (`stall_reason: vllm_worker_dead_zombie_gpu`, #664) | Re-spawn the **experimenter** on the SAME branch, post `epm:experimenter-respawn v<n+1>`. NO implementer round. Cap 3 respawns; on 4th, status -> `blocked`. (Zombie-GPU stall: see the recovery-brief note below.) |
   | `code` | Python `Traceback` from `src/explore_persona_space/` or `scripts/` (our code), `AssertionError`/`TypeError`/`KeyError` from our code, CUDA OOM listing 2+ sibling `Process <pid> has <X> GiB memory in use` entries (parallel fan-out cells co-located on one device — GPU-pinning bug, #557) | Status back to `running` (implementing sub-phase), re-spawn `experiment-implementer` with the failure context. Loop through Steps 4b -> 5 -> 6 again. Cap 3 (existing). |

   *Before applying either row, the Crash-fix circuit-breaker below checks for
   a same-signature repeat or a spent escape ladder and pivots to re-planning if
   either fires.*

   *Either row's respawn, when its round ends in a successful relaunch
   (fresh `epm:run-launched`), also triggers the stale-`blocked` reconcile
   rule ("A successful relaunch also reconciles a stale `blocked`", Step
   6d.2 poll-loop section) — a task parked `blocked` by an earlier failed
   round must not stay `blocked` through a healthy relaunched run (#742).*

   *`code`-row relaunch contract (#779):* the post-review relaunch — the
   Step 6 experimenter respawn (brief carries `fix_sha=` + the element-5
   stale-artifact disposition, copied from the implementer's fix-engaged
   declaration) or an orchestrator hot-fix relaunch — enforces BOTH
   before dispatch: the fix-commit ancestry probe and the declared
   disposition (`.claude/rules/crash-fix-rounds.md` § Crash-fix
   relaunch: fix-commit ancestry + stale-checkpoint hygiene).

   **Zombie-GPU stall recovery brief (`stall_reason: vllm_worker_dead_zombie_gpu`).**
   When the `status=stalled` tick's `stall_reason` is
   `vllm_worker_dead_zombie_gpu`, the experimenter respawn is an `infra`
   row (the classifier routes it via `--stall-reason`), but the generic
   respawn brief is NOT enough: the orphaned `VLLM::EngineCore` worker
   holds VRAM under a cmdline of just `VLLM::EngineCore` (no script name),
   so a routine `pgrep -f <script>` / `pkill -f <dispatcher>.py` reaper
   MISSES it (#664 r8). The respawn brief MUST instruct the experimenter
   to reap the orphan by EXACT PID before relaunching on the same pod —
   `pgrep -af '^VLLM::EngineCore'` → `kill -KILL <pid>` each →
   `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
   to confirm the VRAM released. The canonical recipe lives in
   `.claude/rules/gotchas.md` (crash-orphan `VLLM::EngineCore` entry) and
   the long-form runbook
   `.claude/agent-memory/experimenter/feedback_vllm_zombie_gpu_pkill_reaper.md`;
   reference BOTH in the brief so the experimenter does not re-derive it
   (the experimenter's own Pre-Launch step 9 also runs this probe).

   **Crash-fix circuit-breaker (runs BEFORE applying the cap-3 routing
   table above).** Before re-spawning the experimenter (`infra` row) or
   re-spawning `experiment-implementer` (`code` row), check whether this
   failure is the SAME trap re-tripping or a SPENT escape ladder — either
   case means relaunching is futile and the PLAN, not the code, needs to
   change. The check reads ONLY `events.jsonl` markers + the latest
   `plans/plan.md` (no new in-memory state) and is the pure predicate
   `task_workflow.circuit_breaker_should_fire(events, plan_text, K)`
   (canonical predicate this step implements; the canonical pivot rule is
   `workflow.yaml § pivot_criteria.plan_contradiction_replan`):

   ```bash
   K="${EPM_CIRCUIT_BREAKER_K:-4}"   # default 4; one round past cap-3 so
                                     # the generic pivot can also have fired
   uv run python - "$N" "$K" <<'PY'
   import sys
   from explore_persona_space import task_workflow as tw
   n, k = int(sys.argv[1]), int(sys.argv[2])
   events = tw.list_events(n)
   plan = tw.find_task_path(n) / "plans" / "plan.md"
   plan_text = plan.read_text() if plan.exists() else ""
   fire = tw.circuit_breaker_should_fire(events, plan_text, K=k)
   print(fire if fire else "NO-FIRE")
   PY
   ```

   - **Trigger 1 — same-failure-class repetition** (the narrower complement
     to cap-3, K default 4): the predicate groups `epm:failure` markers by
     their `(phase, failure_class, assert_tag)` signature (`phase` from the
     failure note's `phase=<p>` token; `failure_class` from the
     `failure_class:` line or, absent it, from the prior round's classifier
     verdict already recorded; `assert_tag` from the fallback chain — explicit
     `assert_tag:` SHOULD field, else the bracketed `[<tag>-assert]` token,
     else the exception-type / command-family token
     `<ExcName>:<script_basename>` extracted from the crash note, else a
     normalized note-hash with timestamps / PIDs / `file:line` / the
     subprocess argv array / `--flag value` runs stripped). K or more rounds
     sharing ONE signature → fire with `trigger: same_failure_class`. The
     counter RESETS at any intervening `epm:experiment-implementation` /
     `epm:results` milestone marker (a genuinely successful round means the
     trap was escaped, not re-tripped). It does **NOT** reset on
     `epm:progress` — that marker is the workflow's catch-all heartbeat /
     phase-tick / watcher-respawn breadcrumb and is posted DURING a
     still-failing trap window (verified on #664: the trap window between
     events 228 and 247 carries six benign `epm:progress` markers), so
     resetting on it would make this trigger inert.
   - **Trigger 2 — enumerated-fallback-exhaustion**: the predicate parses the
     latest `plan.md` for a finite escape ladder — a literal ` → `
     arrow-separated "Option A → Option B → ..." run OR a numbered
     "Option N:" list under a §-heading — then scans `epm:progress` /
     `epm:experiment-implementation` notes for which Option each round
     attempted AND `epm:failure` markers re-tripping the SAME gate. When
     EVERY option in the ladder has been launched AND the same gate still
     trips → fire with `trigger: enumerated_fallback_exhausted`. The predicate
     silently NO-OPS (returns no trigger-2 fire) on free-form plans with no
     parseable ladder.

   On fire the predicate returns a dict whose **`pivot_scope` field is the
   ready-to-pass `/adversarial-planner` scope string** (built verbatim per the
   wording template in
   `workflow.yaml § pivot_criteria.plan_contradiction_replan`). The
   orchestrator (this is a STRATEGY PIVOT per that canonical predicate):

   1. Post `epm:strategy-pivot v<n>` (the EXISTING marker — do NOT introduce a
      new kind) naming which trigger fired, the matched signature OR the
      spent-ladder list, and the `pivot_scope` string to pass to the planner.
   2. `uv run python scripts/task.py set-status <N> planning`.
   3. Re-invoke `/adversarial-planner` passing `fire["pivot_scope"]` VERBATIM
      as the pivot scope (it already names the repeated signature or the
      exhausted ladder, per the
      `workflow.yaml § pivot_criteria.plan_contradiction_replan` template).
   4. Treat the revised plan as a FRESH implementer cycle (cap-3 / revision
      counters reset — identical to the existing `plan_contradiction_replan`
      pivot).
   5. Count this as ONE of the ~3 strategy pivots before BLOCK (the SAME
      counter as the existing trigger, NOT a separate one). Block only after
      ~3 such re-plans fail to yield a runnable design AND no further
      autonomous angle exists.

   When the predicate returns `NO-FIRE`, fall through to the classifier
   invocation and cap-3 routing table below unchanged.

   **Missing `failure_class` — invoke the classifier script.** Do NOT
   reason about regex patterns inline; the patterns are owned by
   `scripts/failure_classifier.py` and reading them yourself drifts.
   Instead, shell out:

   ```bash
   # Pipe the failure body via stdin to avoid shell-quoting traps.
   # On a status=stalled tick, ALSO forward the poll JSON line's
   # stall_reason via --stall-reason: a known reason (e.g.
   # vllm_worker_dead_zombie_gpu) routes infra directly, because a silent
   # hang's log tail matches no infra pattern. Omit the flag when there is
   # no stall_reason (status=dead, or a stall_reason of null/absent).
   cat <(uv run python scripts/task.py view "$N" --json \
       | jq -r '.events[] | select(.kind == "epm:failure") | .note') \
     | uv run python scripts/failure_classifier.py --body - \
         --log "$LATEST_LOG_PATH" \
         ${STALL_REASON:+--stall-reason "$STALL_REASON"}
   ```

   The script writes a single line — `infra` or `code` — to stdout.
   Treat that as the verdict and apply the corresponding row of the
   table above. If the script exits non-zero, treat as `code`
   (conservative) and post `epm:failure-classify-error` with the stderr
   captured.

   The Python module
   [`scripts/failure_classifier.py`](../../../scripts/failure_classifier.py)
   is the SINGLE source of truth for the regex pattern list.
   `.claude/skills/issue/failure_patterns.md` is a human-readable
   mirror of the same patterns (kept in sync; consult it for review or
   when extending — but do NOT consult it at runtime). To add a new
   pattern, edit `failure_classifier.py` AND the markdown mirror; the
   tests in `tests/test_failure_classifier.py` cover the behaviour.

   **Failure-lesson capture (fires when a crash-fix round RESOLVES the
   failure OR CONFIRMS the true root cause).** A lightweight in-flight
   hook, not a new pipeline step;
   auto-continue, no gate. Both crash-fix shapes — the `code`-row
   `experiment-implementer` round and the `infra`-row experimenter
   respawn whose relaunch applied a fix — are REQUIRED (by
   `.claude/rules/crash-fix-rounds.md` § "Crash-fix rounds: failure-lesson
   block" and `experimenter.md` § "Failure-lesson block on
   relaunch-with-fix") to end their report with a structured lesson
   block. A THIRD shape arrives outside this step: an experimenter that
   fixed a dying launch within its own turn and relaunched (no
   `epm:failure` posted) appends the same block to its Step 6d launch
   report — on receiving such a launch report, apply the same three
   orchestrator actions below. The block:

   ```
   <!-- epm:failure-lesson v1 -->
   failure_class: code|infra|data
   phase: <pipeline phase or script>
   lesson: <1-3 sentences: the trap + the fix, written for the NEXT agent>
   generalizes: yes|no   # yes only if the trap plausibly recurs beyond this issue
   owning_agent: experiment-implementer|experimenter
   gotcha_candidate: yes|no  # yes for codebase/infra traps that belong in .claude/rules/gotchas.md
   root_cause_confirmed: yes|no  # yes if THIS round identified the TRUE root cause (even if a NEW distinct failure followed or the pod was abandoned in recovery)
   supersedes:           # OPTIONAL: prior-lesson slug or marker-ts this lesson corrects; omit if none
   <!-- /epm:failure-lesson -->
   ```

   **Capture eligibility (added #712).** Decide whether a received block
   is eligible for capture by calling the pure predicate
   `task_workflow.failure_lesson_capture_eligible(block_fields,
   subsequent_distinct_failure=<bool>)`. It returns True when the block
   RESOLVED the failure (the original trigger, `resolved: yes`) OR the
   block carries `root_cause_confirmed: yes` (case ii — true REGARDLESS of
   whether a subsequent distinct failure followed or the pod was abandoned
   in recovery). **Root-cause-confirmed firing.** Case (ii) closes the
   #664 gap: at #664 event L204 a crash-fix round CONFIRMED that pod-664
   reproducibly deadlocked the first `llm.generate()` regardless of batch
   size (a pod-hardware cause, NOT a code bug — the OLD pod ran the same
   code fine), but the round ended in a recovery pivot that terminated the
   pod, so the resolve-only trigger never fired and the failure-lesson
   hook captured NOTHING for the confirmed cause. WHO posts it: the agent
   (experiment-implementer / experimenter) that confirmed the cause emits
   the block in its report — case (ii) is signalled by
   `root_cause_confirmed: yes`. The orchestrator posts it verbatim on
   receipt, exactly as for the resolve case; it does NOT wait for a
   successful relaunch.

   On receiving a crash-fix report carrying a capture-eligible block, the
   orchestrator takes three actions:

   1. **Post the marker.** Post the block verbatim as
      `epm:failure-lesson v1` on the task (`task.py post-marker <N>
      epm:failure-lesson --note '<block>'`). This fires for
      `generalizes: no` too — for one-offs the marker alone is the
      durable record (NO memory write).
   2. **On `generalizes: yes` — persist to agent memory IMMEDIATELY.**
      Append a `feedback_<slug>.md` entry (standard agent-memory
      frontmatter + the lesson body) to
      `.claude/agent-memory/<owning_agent>/` plus a one-line
      `MEMORY.md` index entry, then commit BY EXPLICIT PATH on `main`
      from the repo root and push (auto, no approval gate — same
      standing rule 2026-06-02 as workflow fixes). Push BARE —
      `git push origin main || uv run python scripts/sync_repo_root.py` —
      never piped (Step 10d § "Bare push / merge snippets";
      sync_repo_root exit 0 can mean in-flight — landing not guaranteed,
      see the canonical block's caveat). The point is
      same-day cross-session sharing: a sibling session's next agent
      spawn loads the memory within minutes, instead of waiting for the
      nightly `/daily` sweep (on 2026-06-11, #537 and #545 re-hit
      overlapping failure classes hours apart with no persistence
      channel). Lessons are written for the NEXT agent — 1-3 sentences,
      the trap + the fix, no transcript dumps.

      **`supersedes:` handling + apply (added #712).** Apply the
      capture-eligible block by calling the pure composer
      `task_workflow.apply_failure_lesson(block, durable_texts,
      new_lesson_ref)`, where `durable_texts` is `{path: current_text}`
      for the candidate durable files (the `owning_agent`
      `feedback_<slug>.md` body keyed at `new_lesson_ref["memory_path"]` +
      any matched `.claude/rules/gotchas.md` bullet) and `new_lesson_ref`
      is `{"slug": "<new-lesson-slug>", "task_id": "<N>", "memory_path":
      "<feedback path>", "lesson": "<lesson body>"}`. The composer returns
      the FINAL `{path: text}` map the orchestrator then writes
      (explicit-path commit + push, as today). Its behavior:
      - If the block carries `supersedes: <prior-slug-or-marker-ts>`, the
        composer calls `supersedes_action()` to locate every durable entry
        whose text matches `<prior-ref>` and PREPENDS a concrete
        `[SUPERSEDED by <new-lesson-slug> — see #<N>] ` marker (the real
        slug + task id, NEVER a `<pending>` placeholder), then APPENDS the
        new (corrected) lesson body to the `owning_agent` memory file — so
        the corrected lesson LANDS ALONGSIDE the annotated prior, never
        replacing it. Transitive chains are kept (A `[SUPERSEDED by B]`, B
        `[SUPERSEDED by C]`, C live); each correction annotates only the
        entry its `supersedes` directly names.
      - If `<prior-ref>` resolves to NOTHING, the composer leaves all prior
        texts byte-unchanged, appends the new lesson normally, and the
        orchestrator logs `supersedes_unresolved: <prior-ref>` in the
        marker note — a dangling `supersedes` is a no-op annotation, NEVER
        a hard failure (a lesson always lands).
      - If `supersedes` is ABSENT, the composer is a pure append (the
        produced text is byte-identical to the pre-#712 append-only path).
   3. **On `gotcha_candidate: yes` — route as a workflow-fix
      candidate.** Treat the lesson as a prose workflow-fix candidate
      targeting `.claude/rules/gotchas.md` and route it through the
      existing workflow-fix-on-bug auto-file default — a filed
      `kind: infra` task + a background `/issue --auto` session
      (`.claude/rules/workflow-fix-on-bug.md`); the lesson block is the
      surfaced prose.

   If the resolving report omitted the block (older agent spawn, or a
   refusal killed the report tail), reconstruct it from the failure
   context + fix diff yourself before posting — don't bounce the round
   for the missing block alone. `scripts/consolidate_lessons.py` (a cron,
   NOT `/daily` — task #711) is the deterministic deduplicating
   consolidator: it reads the rolling-window `epm:failure-lesson v1`
   markers, dedupes against agent memories, promotes recurring lessons into
   `.claude/rules/gotchas.md` or the relevant rule file, and prunes
   over-eager `generalizes: yes` memory entries. `/daily` no longer owns
   this pass.
3. If `epm:results` exists, move status to `uploading` and proceed to
   Step 8.

### Step 8: Upload verification

Only if status is `uploading` and no `epm:upload-verification` marker
with verdict=PASS.

**Hard gate:** No experiment advances to interpretation until all
artifacts have permanent URLs. This prevents data loss from pod restarts
or cleanup.

**Results-landed parallel spawn (Step 8 ∥ Step 9 pre-compute).** The
upload-verifier dispatch below is no longer a serial prelude to Step 9 —
at this results-landed point the orchestrator spawns up to THREE
background agents concurrently (single message, multiple Agent calls,
staggered a few seconds apart per the CLAUDE.md 429 token-pacing
guidance), each preceded by its own `stage-dispatch` breadcrumb (Step 9
entry guard convention):

1. **`upload-verifier`** (this step, `stage=verifying`) — the hard gate,
   unchanged.
2. **`analyzer` first pass** (Step 9a round 1, pre-computing;
   `stage=interpreting round=1`). The analyzer's inputs (eval JSONs
   under `eval_results/`, figures in the worktree/git, raw completions
   already pulled) exist locally before verification, so it can run its
   full first pass during `uploading`. **HOLD-marker mode:** the
   early-spawn brief instructs the analyzer to write its interpretation
   to `/tmp/issue-<N>-interpretation-v1-held.md` and RETURN WITHOUT
   posting `epm:interpretation v1` — the orchestrator publishes the held
   output (and only then starts the interpretation-critic round) after
   upload-verification PASS. See the two hard joins below.
   **Paper-mode branch (`paper: true` frontmatter).** When the task
   carries `paper: true`, the analyzer runs its PAPER-TASK MODE
   (`.claude/rules/analyzer-paper-mode.md` § PAPER-TASK MODE): the analysis Steps
   1→3.6 are unchanged, but the write-up is a LaTeX **paper** under
   `docs/papers/issue_<N>/` (not a markdown body) — the analyzer
   assembles the `.tex` (splicing in the `methodology-writer`'s Methods +
   Appendix, item 3 below), emits `refs.json` + the figures manifest,
   runs `build_paper.py` → `verify_paper.py`, and writes the `body.md`
   **paper-stub** (`set-body --snapshot` + `set-title` + `set-clean-result`)
   ONLY after `verify_paper.py` PASSes. On `verify_paper.py` FAIL it does
   NOT write a stub — it parks at `reviewing` (or `blocked` with
   `epm:failure v1 failure_class: code`), leaving the `.tex` + build
   `.log`/`.blg` in `docs/papers/issue_<N>/` for iteration. HOLD-marker
   mode still applies: write the held interpretation, RETURN without
   posting `epm:interpretation v1`, the orchestrator publishes after
   upload-verification PASS. The mechanical gate for a paper-task is
   `verify_paper.py`, NOT `verify_task_body.py` (which stays the markdown
   verifier).
3. **`methodology-writer` early spawn (PAPER MODE, or v3/v2
   GRANDFATHERED)** (the early-spawn half of Step 9a-quater;
   `stage=methodology-reference round=1`). **Three cases — branch on
   `paper:` frontmatter FIRST:**
   - **`paper: true` → SPAWN it in PAPER MODE.** The methodology-writer
     authors the LaTeX paper's **Methods section + the recipe Appendix**
     (findings-blind) and hands them to the `analyzer` (item 2), which
     splices them into the `.tex` and runs the build/verify. This is
     load-bearing for paper-tasks — the analyzer does NOT author Methods
     or Appendix itself (the findings-blind firewall is the whole point).
     See `.claude/agents/methodology-writer.md` § PAPER-TASK MODE
     (Methods + Appendix). Fires whenever the 9a-quater kind-gating says
     the step runs (`kind: experiment` always; `kind: analysis` only with
     a methodology surface; `infra | batch | survey` never).
   - **v4 markdown body → SKIP this spawn entirely** — under v4 the
     methodology doc is a POST-PASS mechanical export of the body's
     `## Methodology` section (Step 9a-quater v4 path), so no agent is
     spawned here and the early-spawn batch is just upload-verifier ∥
     analyzer first pass.
   - **In-flight v3/v2 markdown body → SPAWN it in MARKDOWN MODE** (the
     legacy findings-blind path) — a v3/v2 body has no detailed
     `## Methodology` section to copy. Same kind-gating as above
     (evaluate the skip BEFORE spawning).
   The agent is findings-blind by design and its inputs (plan, config,
   reproducibility metadata, verbatim artifact rows) are final the
   moment results land, so it can safely run during `uploading` and the
   interpretation loop. For this early spawn the findings-blind
   Reproducibility input is extracted from the task's `epm:results`
   markers (`reproducibility_card` — alias `reproducibility` — +
   `eval_paths`, via `task.py view <N> --json`) into the temp file —
   the clean-result body's `## Reproducibility` H2 does not exist
   yet. NEVER read only the latest marker: multi-launch runs post
   several `epm:results` markers and a resume-pass sentinel can carry
   an EMPTY card (#601: `adapter_paths: {}`), so resolve each field
   newest-wins among non-empty declarations across markers, matching
   `verify_uploads.py` `merged_results_card` (full recipe: 9a-quater
   procedure step 2). Everything
   publish-side (no-secrets scan, gist, link-append, marker) stays at
   the 9a-quater LATE JOIN; see 9a-quater § Split schedule.

**Two hard joins (both strictly gated on upload-verification PASS):**

1. **Interpretation publish.** `epm:interpretation v1` is NOT posted and
   the interpretation-critic round is NOT started until the verifier
   posts PASS. If the analyzer returns first, hold its output and wait
   for the verifier. The status transition order is unchanged — the
   analyzer merely pre-computes during `uploading`; status flips to
   `interpreting` only on the PASS branch below.
2. **Pod termination.** The teardown call on the PASS branch still
   strictly requires upload PASS — unchanged.

**On upload FAIL → uploader gap-fill: decision rule for the held
analyzer output.** After the gap-fill rounds reach PASS, check whether
the uploader added or changed any artifact the analyzer consumed — eval
JSONs, raw completions, analysis tensors. If YES, the held first pass is
stale: discard it and re-spawn the analyzer first pass before
publishing. If the gaps were only HF-checkpoint / upload-side (no
analysis input changed), proceed with the held analyzer output as-is.

**Re-entry idempotency.** The Step 9 entry guard's `stage-dispatch`
breadcrumbs cover all three dispatches. On a backstop re-entry, apply
the guard PER STAGE (the step-2 per-stage backwards scan in the Step 9
entry guard): do not re-dispatch a stage whose own breadcrumb is within
its freshness window, even when another stage's marker is the latest
event.

Spawn the `upload-verifier` agent with:
- Task number
- Task type (from `body.md` frontmatter)
- Artifact hints from the `epm:results` event (WandB URL, HF paths, pod
  name)
- The `epm:plan` event (for experiment-type metadata)

The verifier runs `scripts/verify_uploads.py` and checks:

| Artifact | Required when | Verified how |
|----------|--------------|--------------|
| Model on HF Hub | Training experiments | HF API |
| Eval JSON on WandB | Always | WandB API |
| Dataset on HF Hub | New data generated | HF API |
| Output generations on WandB | Generation experiments | WandB API |
| Training metrics on WandB | Training experiments | WandB run URL |
| Figures committed to git | Always | `git log` |
| Local weights cleaned | Training experiments | `ssh_execute ls` on pod |
| Claimed URLs HEAD-resolve (phantom-URL gate, #456) | Always | `--claimed-urls-file` HEAD-checks every HF/WandB URL in the `epm:results` marker + body's `## Reproducibility` section at its CITED revision via `orchestrate.hub.verify_artifacts_exist` |
| Primary deliverable produced (completeness gate, #519) | When plan §6.5 declares `primary_deliverable:` | For each `{dv, glob}` row, on-pod `find <glob>` enumerates ≥1 file. Zero files → FAIL with blocker tag `primary-deliverable-missing`. Plans without the §6.5 block (legacy + analysis/infra/batch/survey kinds) get a WARN, not a FAIL. See upload-verifier § Step 2.7. |

**Phantom-URL gate (Step 8 enforcement of upload-verifier Step 2.5).**
Before spawning the verifier, build a single text blob containing the
`epm:results` marker body + the clean-result body's Reproducibility
section, write it to `/tmp/issue-<N>-claimed-urls.txt`, and run
`verify_uploads.py --issue <N> --type <experiment-type>
--claimed-urls-file /tmp/issue-<N>-claimed-urls.txt` so every cited
HF/WandB URL is HEAD-verified at its cited revision. `--type` is the
experiment type handed to the verifier as an input — always pass it
explicitly per upload-verifier.md Step 2.5 (omitting it falls back to
frontmatter-`kind` inference, which conservatively assumes `training`
for `kind: experiment`). A URL string in a
sentinel is NOT evidence the files exist. Incident #456: a training run
PASSed upload-verification with a per-step checkpoint URL nothing had
uploaded; a downstream experiment had to re-train two months later. See
`.claude/agents/upload-verifier.md` § Step 2.5 for the full rationale.

Post `epm:upload-verification v1` event with per-artifact PASS/FAIL +
URLs.

- **PASS** -> teardown the compute, then move status to `interpreting`
  and proceed to Step 9. (Same-issue follow-up round? At
  `followups_running`, SKIP the `interpreting` flip — status-hold rule,
  Step 9b § Same-issue follow-up loop step 3; code-enforced — but the
  teardown + Step 9 progression run as normal.) "PASS" means a fresh
  `epm:upload-verification` verdict for THIS round — posted after this
  round's `epm:results` and after this round's `stage=verifying` dispatch
  breadcrumb. A dispatched verifier with no verdict yet is BLOCKING: do
  not flip status to `interpreting`, do not publish the held
  interpretation, and do not run finalize on a prior round's PASS
  (incident #778: status advanced and the pod was finalized ~19:00Z on
  the fallback while the verifier was in flight; its verdict later came
  back FAIL). On a FAIL verdict: uploader gap-fill + re-verify — never
  advance on the FAIL. finalize enforces teardown-side currency
  mechanically (the verifier-currency reasons below); the status flip
  itself is prose-enforced — this paragraph IS that gate. Once artifacts
  are confirmed at permanent
  URLs, the compute is no longer needed — interpretation runs locally.
  If the results-landed parallel spawn produced a held analyzer first
  pass, publish it now: post the held interpretation as
  `epm:interpretation v1` and resume Step 9a round 1 at the
  critic-ensemble spawn instead of re-spawning the analyzer (see Step
  9a § Held-output publish).

  **Backend-agnostic teardown (slice 6).** The dispatch helper persisted
  the per-issue `RunHandle` to `.claude/cache/issue-<N>-handle.json` at
  Step 6b; the orchestrator runs ONE operational call —
  `scripts/dispatch_issue.py finalize` — which reads the sidecar, calls
  `backend.confirm_artifacts(handle)`, and on PASS calls
  `backend.teardown(handle)` — one path for every backend (RunPod /
  SLURM / GCP). The agent-level upload-verifier above runs the
  EXPLORATORY pass; this in-helper `confirm_artifacts` is the
  complementary MECHANICAL gate (HF Hub `list_repo_files` + WandB run
  + git-figure + completion sentinel, per
  `backends.artifacts.confirm_artifacts_from_handle`). Both must pass
  before teardown fires. Degrade path (incident #585): when the handle
  carries NO `expected_artifacts` declaration — launch paths other
  than GCP do not populate it yet (#598 tracks SLURM; the RunPod
  launch shells `pod_lifecycle.py` and never has) — the mechanical
  gate is structurally unsatisfiable, so finalize falls back to the
  agent-level PASS evidence on the task's events.jsonl (the sticky
  `epm:upload-verified` marker, or the latest `epm:upload-verification`
  with `Verdict: PASS`) and proceeds to teardown with a loud log +
  `"confirm_artifacts": "skipped_no_declaration_agent_pass"` in the
  JSON. Do NOT bypass finalize with a raw `pod.py terminate` on the
  exit-3-missing-declaration shape — that skips the sidecar retirement
  and leaves a stale handle that can mis-target a later finalize; run
  the upload-verifier to a PASS, then re-run finalize. With neither a
  declaration nor agent PASS evidence, finalize still exits 3
  (`reason: confirm_artifacts_no_declaration`).

  **Verifier-currency gate (#1026).** The agent-level PASS evidence must
  be CURRENT — finalize refuses (exit 3, teardown skipped, sidecar not
  retired) on every non-skip path with one of five typed reasons, each
  with its routing action: `upload_verifier_in_flight` (a dispatched
  verifier round has no verdict yet, liveness window fresh → WAIT for
  the verdict; on PASS re-run finalize, on FAIL gap-fill + re-verify,
  never finalize on a FAIL), `upload_verifier_stalled` (window lapsed,
  no verdict → re-spawn the upload-verifier to a verdict, then finalize
  on PASS), `upload_verification_ambiguous` (a late verdict cannot be
  attributed to the current results-epoch → re-run the verifier; the
  fresh round resolves it), `upload_verification_stale` (the latest
  verdict predates the newest `epm:results` → re-verify),
  `upload_verification_failed_current` (the latest verification is a
  FAIL → gap-fill + re-verify). An in-flight verifier is never
  PASS-equivalent; absence-of-verdict never satisfies the gate.
  **Named residual:** the crumb-based rules presuppose the Step 9
  entry-guard convention that a `stage-dispatch` breadcrumb precedes
  each verifier spawn (the missed-breadcrumb limitation — see the
  "Limitation (be explicit about it)" paragraph under the Step 9 entry
  guard) — a verifier spawned WITHOUT its breadcrumb is invisible to the
  in-flight/stalled rule; the stale + FAIL-current rules are the
  backstops for that case.

  **Phase-scoped-launch mismatch (incident #604).** The launch-time
  auto-declaration assumes the FULL task artifact set (hydra-lane
  launches: HF `issue<N>_<attempt>/raw_completions/` + git
  `eval_results/issue_<N>/` + `figures/issue_<N>/`; `--workload-cmd`
  launches auto-declare only the sentinel + git paths — the guessed HF
  prefix was dropped after it false-FAILed a perfectly-uploaded run
  whose driver used its own `issue<N>_<slug>/` contract prefix, #601
  follow-up r1; HF-data coverage on that lane comes from the
  agent-level upload-verifier), so a launch covering only ONE phase of a
  multi-phase plan (e.g. an extraction phase whose sole deliverable is
  an `analysis_tensors/` bundle) FAILs `confirm_artifacts` on declared
  paths that only the plan's LATER (VM-local) phases produce. A
  declaration that is PRESENT but phase-mismatched is structurally
  unsatisfiable until end-of-task, and the agent-pass fallback above
  never fires (it is gated on the declaration being ABSENT) — finalize
  exits 3 (`reason: confirm_artifacts_failed`) by design. Do NOT leave
  the instance idling until the later phases land (#604 burned ~70 idle
  minutes on a g2-standard-4): mechanically verify the launch's ACTUAL
  phase deliverable on permanent storage first
  (`huggingface_hub.list_repo_files` for HF paths — never the `hf`
  CLI), then re-run finalize with the gate skipped —
  `dispatch_issue.py finalize --issue <N> --skip-confirm-artifacts` —
  which still runs the backend teardown AND retires the sidecar to
  `<name>.finalized` (no stale handle; do NOT substitute a raw `gcloud
  compute instances delete` / `pod.py terminate`, which skips the
  retirement). Post `epm:pod-terminated v1` naming the declaration
  mismatch + the verified deliverable paths. Distinguish the two exit-3
  shapes: no-declaration → upload-verifier-to-PASS + plain re-run
  (above); present-but-phase-mismatched declaration → verify the phase
  deliverable, then `--skip-confirm-artifacts`. The skip flag does NOT
  bypass the verifier-currency gate for a FRESH in-flight verifier round
  (exit 3 `upload_verifier_in_flight` — wait for the verdict, or for the
  15-min window to lapse to `stalled`); stalled / stale / ambiguous /
  failed-current records degrade to a loud warning + a `verifier_warning`
  field in the success JSON.

  ```bash
  # ONE call for every backend. Exit 0 = confirm PASS + teardown done;
  # exit 3 = confirm FAIL or verifier-currency refusal (reason ∈
  # confirm_artifacts_failed | confirm_artifacts_no_declaration |
  # upload_verifier_in_flight | upload_verifier_stalled |
  # upload_verification_ambiguous | upload_verification_stale |
  # upload_verification_failed_current) — teardown SKIPPED, evidence
  # preserved; exit 2
  # = missing sidecar (treat as infra failure).
  #
  # CAVEAT — parent-pod-reuse child tasks: when this child task ran on
  # the parent's RunPod via the alive-parent branch in Step 6b, NO
  # sidecar was written for the child. SUBSTITUTE this call with
  # `pod.py terminate --issue $PARENT_ID --yes` (per the "Slice-6
  # regression guard for the parent-pod-reuse branch" paragraph in
  # Step 6b); the finalize CLI would otherwise exit 2 on the missing
  # child sidecar.
  uv run python scripts/dispatch_issue.py finalize --issue <N>
  ```

  On the RunPod path the underlying `RunPodBackend.teardown` shells
  out to the same `scripts/pod.py terminate --issue <N> --yes` that
  today's wiring uses (the wrapper preserves the existing guard logic
  verbatim); on the SLURM path it `scancel`s via the robot SSH alias;
  on GCP it `gcloud compute instances delete`s. Post
  `epm:pod-terminated v1` with the teardown summary (for the GCP path
  the marker name still applies — the dashboard surfaces every
  backend's teardown under the same key).

  If interpretation later needs GPU compute (e.g., to regenerate a
  figure from raw outputs that weren't downloaded), dispatch fresh
  compute through the slice-6 router — read the task's `backend:`
  frontmatter and run `dispatch_issue.py launch --issue <N> --intent
  "$INTENT" ${BACKEND:+--backend "$BACKEND"}` per Step 6b's
  "Operational dispatch (slice-6 router, ALL backends)" block (empty
  frontmatter → auto routing, GCP-first standing default, then the free
  clusters; RunPod only on an explicit `backend: runpod`). If the task has `parent_id`, terminate
  the parent's pod (`epm-issue-<PARENT_ID>`) instead. Skip the
  teardown call only if the task has a `keep-running` tag for known
  follow-up work in the same session.

  **VM download-cache cleanup (post-#disk-100pct).** The experiment
  downloaded its source data from HF into VM-local caches under
  `data/issue_<N>/hf_dl/` + `data/issue_<N>/g*_dl/` — in the repo-root
  `data/` AND in this issue's worktree
  (`.claude/worktrees/issue-<N>*/data/issue_<N>/`, where the live run
  usually writes; the worktrees tree hit 139 GB on 2026-06-26). Nothing
  else reclaims them, and a single finished experiment can pin ~100 GB
  on the VM root disk (incident 2026-06-25: `/` hit 100% full). These are
  re-downloadable CACHES (no on-HF presence check needed), and `store/`
  + `eval_results/` are NEVER touched (in repo-root OR worktrees). After
  the teardown above (artifacts are now confirmed at permanent URLs),
  clean this issue's download caches — the helper sweeps both the
  repo-root and worktree copies:

  ```bash
  # Re-downloadable hf_dl/g*_dl caches only (repo-root + worktree);
  # store/ + eval_results/ kept.
  uv run python scripts/clean_experiment_downloads.py <N> --apply
  ```

  Auto-continue (NOT a gate); idempotent — a re-entry on an
  already-cleaned issue is a no-op. The fleet-wide backstop for caches
  that escape this path (crashed runs, follow-up rounds) is the
  `vm_disk_guard.py` cron (CLAUDE.md § Disk hygiene). Skip only when the
  task has a `keep-running` tag (the same-session follow-up may re-use the
  cache).

  **Incremental (between-phase) cleanup for multi-phase runs.** Step-8
  cleanup fires only at experiment END, so a multi-phase experiment whose
  phases each materialize a fresh download cache holds the PEAK of all
  phases' caches at once — and a large-footprint phase can fill `/`
  mid-run (incident 2026-06-26: #658's Phase-1 analysis put a 139 GB store
  on the VM worktree on the shared 188 GB disk; `/` hit 100% full). When a
  run has multiple phases that each download inputs (e.g. a phase's judge /
  extraction step CONSUMES its `e0_gen` / `g*_dl` / `hf_dl` inputs, then
  the next phase downloads more), reap each consumed phase's
  re-downloadable cache BEFORE the next phase materializes more — bounding
  peak footprint, not just cleaning at the end. Between phases (after the
  judge / extraction consumes the phase's inputs, before the next phase
  downloads):

  ```bash
  # Between-phase: reap THIS phase's consumed hf_dl/g*_dl cache (repo-root
  # + worktree); store/ + eval_results/ kept; no terminal-status gate (the
  # run knows the phase is done). Re-downloads on demand if a later phase
  # needs it again.
  uv run python scripts/clean_experiment_downloads.py <N> --incremental --apply
  ```

  Same safety contract as the Step-8 cleanup (re-downloadable caches only;
  `store/` + `eval_results/` NEVER touched; read-only on task state). This
  is the RUNTIME backstop that bounds peak footprint — but it does NOT
  rescue a single phase whose OWN footprint exceeds the disk; such a phase
  must be ROUTED off the VM at plan time per the data-footprint carve-out
  (CLAUDE.md "CPU-only phases don't hold GPU pods" → `planner.md` §9 →
  `critic.md` Methodology lens item 10).

  **Upload-verification guard (post-#444).** `pod.py terminate` refuses
  to destroy an `epm-issue-<N>` / `pod-<N>` for a `kind: experiment`
  task unless an `epm:upload-verification PASS` marker exists on task
  `<N>` — this catches resume-launcher / hand-orchestrated completions
  that skipped the verifier. The normal Step 8 path posts the PASS
  marker BEFORE calling terminate, so the gate is silent on the happy
  path. If you must terminate without running the verifier (e.g. the
  experiment crashed before producing artifacts, or you've manually
  confirmed every URL landed), pass `--skip-upload-verify` — it logs a
  LOUD warning and still proceeds. NEVER substitute a manual partial
  upload check for the verifier on a normal-completion path; the
  verifier's checklist is the safety net against silent dataset /
  checkpoint loss (incident: task #444 lost the training-mix datasets
  after a hand-driven completion did a partial check and terminated).
- **FAIL with blocker tag `primary-deliverable-missing`** (Step 2.7
  completeness gate, post-#519) -> the headline phase that produces the
  Goal's primary dependent variable silently did not run on the pod
  (e.g. missing input flags fell through an `if args.X and args.Y`
  guard, a phase crashed mid-loop with the dispatcher recording
  `skipped_phases: []`). The uploader cannot fix this (there is no
  artifact to upload), and terminating the pod destroys the cheap-fix
  window (the pod and any per-step checkpoints still exist; re-running
  the missing phase in-place is far cheaper than re-provisioning +
  re-training from scratch).

  **Auto-recover, don't park.** Consistent with CLAUDE.md "Continuing on
  your own is the default" + `workflow.yaml § pivot_criteria`, do NOT
  call `pod.py terminate`, do NOT dispatch the uploader, do NOT flip to
  `status:blocked`. Instead loop back to the run phase on the
  still-alive pod and re-drive the missing primary deliverable:

  1. Read the verdict body's `Missing / required action` list to
     identify the missing DV name(s), the missing glob(s), and the
     pod-side phase that produces them (the planner's §6.5 row + the
     §4 Design pipeline together name the responsible dispatch
     entrypoint).
  2. Flip status back to `running` (`task.py set-status <N> running`).
     (Same-issue follow-up round? At `followups_running`, SKIP this flip —
     status-hold rule, Step 9b § Same-issue follow-up loop step 3;
     code-enforced — and re-enter the dispatch path with the status held.)
     Then re-enter the Step 6d experimenter-dispatch path with an
     explicit re-run scope naming the missing phase + the inputs that
     fell through (typically: re-dispatch the same entrypoint with the
     corrected `--<phase>-inputs <path>` flags that the silent guard
     consumed). Post a `epm:progress` note recording the pivot:
     `auto-recover: primary-deliverable-missing for <DV>; re-running <phase> on pod <pod-name>`.
  3. The experimenter dispatches as usual, posts `epm:run-launched` /
     `epm:run-finished` / `epm:results`, and Step 8 re-runs
     upload-verification on the next /issue tick.
  4. Re-verification is mechanical: if `find <glob>` now enumerates
     ≥1 file the row PASSes and the gate clears; if it remains zero
     after a re-run that ITSELF says it ran (exit 0 + a non-empty
     manifest for the phase), that is a NEW failure class — the
     dispatcher claims success while producing nothing — and counts
     as a fresh strategy attempt.

  Treat each auto-recovery attempt as one strategy iteration. The
  generic halt path applies normally:
  `workflow.yaml § pivot_criteria` (specifically `infra_respawn_cap_3`,
  and after ~3 fundamentally different strategies have all FAILed AND
  no further autonomous angle exists) is the ONLY route to
  `status:blocked` for this failure class. Do NOT introduce a dedicated
  halt for the first or second `primary-deliverable-missing` FAIL.

- **FAIL (any other blocker)** -> dispatch the `uploader` agent (up to
  3 rounds) to close the gaps. The uploader receives the verifier's
  missing-artifacts list, lifecycle-aware resumes the pod if needed,
  pushes to HF / WandB / git, and posts `epm:upload-fix v1`. After each
  uploader round, re-run `upload-verifier`; it posts a fresh
  `epm:upload-verification v<N+1>`.

  Round outcomes:
  - **uploader COMPLETE + verifier PASS** -> proceed as PASS branch above.
  - **uploader BLOCKED** (e.g., RunPod host capacity, missing
    credentials) -> stays at `uploading`. Post the uploader's
    `epm:upload-fix` event with the blocker. Post the §5 marker:
    ```bash
    uv run python scripts/post_step_completed.py --issue <N> --step 7 \
      --exit-kind failure-exit \
      --notes "uploader BLOCKED; awaiting operator action"
    ```
    EXIT, await operator action.
  - **3rd round still FAIL** -> status to `blocked`. Post the §5 marker:
    ```bash
    uv run python scripts/post_step_completed.py --issue <N> --step 7 \
      --exit-kind failure-exit \
      --notes "uploader exhausted 3 rounds; see upload-fix v3"
    ```
    EXIT (mirror the code-reviewer FAIL escalation in CLAUDE.md).

  See `.claude/agents/uploader.md` for the uploader's contract and the
  marker schema. The uploader NEVER terminates pods; only stops/resumes.

#### Step 8-bis: Pod must not idle on a halt

Step 8's terminate fires only on the NORMAL upload-verification-PASS path.
A pod can still be left RUNNING-and-billing whenever the pipeline leaves
that path: (a) it blocks on a human-input gate that cannot be satisfied
this turn (e.g. `epm:fact-pick` at Step 6, the plan-approval / merge gates,
or any STATE-TO-`blocked` exit), or (b) it is detected crashed/dead with
GPUs idle. Before EXITing the turn in EITHER case, if an `epm-issue-<N>`
pod (or the parent's pod for a follow-up) exists and is RUNNING, run
`uv run python scripts/pod.py stop --issue <N>` (volume preserved; `resume`
re-provisions) — or `terminate --issue <N> --yes` when the work is truly
done — and post `epm:pod-stopped v1` / `epm:pod-terminated v1` with the
command output. NEVER leave a pod RUNNING while awaiting human input or
after a crash. (Incident 2026-06-01: #444 idled a 4×H100 ~21h on an
unfired gate, #404 ~2 days after Step 8 never fired, #407 ~1 day after an
`aggregate`-phase crash — ~$1k of idle burn combined.)

### Step 9: Iterative interpretation + final review

This step has two sub-phases: **interpretation** (iterative
analyzer<->critic loop) and **final review** (clean-result-critic gate).

#### Step 9 entry: in-flight idempotency guard (backstop re-entry)

The Step 6d.2 backstop cron now survives into `verifying` / `interpreting`
/ `reviewing` (so a stalled interactive session in these stages still gets
auto-woken). The cost to bound is a backstop tick firing `/issue-tick <N>`
(which may load the full `/issue <N>` skill on stale-marker recovery)
while a stage subagent (analyzer, interpretation-critic,
clean-result-critic, upload-verifier) is STILL RUNNING from a prior tick —
re-dispatching it would burn redundant subagent tokens and could race two
writers on the body. This guard makes a fresh re-entry into Step 9 (or
Step 8 verifying) cheaply detect "live work in progress" and EXIT without
re-dispatching (that EXIT is the guard rule's `post_step_completed.py
--exit-kind parked` call below).

**Dispatch breadcrumb (post on every stage dispatch).** Immediately before
spawning ANY Step 8 / Step 9 stage subagent, post a breadcrumb so a later
tick can see the dispatch:
```bash
uv run python scripts/task.py post-marker <N> epm:progress \
  --note "stage-dispatch stage=<verifying|interpreting|clean-result> round=<r> subagent=<name> worktree=<abs path or 'repo-root'>"
```

**Pre-dispatch dedup (NON-SKIPPABLE, every dispatch site).** Immediately
BEFORE posting a NEW `stage-dispatch` breadcrumb and spawning, run the
mechanical check:

```bash
uv run python - <<'PY'
from explore_persona_space.task_workflow import list_events, stage_dispatch_should_skip
print(stage_dispatch_should_skip(list_events(<N>), "<stage>", <r>, window_minutes=<W>) or "DISPATCH")
PY
```

If the output is anything other than `DISPATCH`, log that one line, post
NO duplicate breadcrumb, do NOT spawn — the stage is already in flight
(EXIT `parked` on a backstop tick — the idempotency-guard EXIT, i.e. the
guard rule's `post_step_completed.py --exit-kind parked` call below — or
continue with other work). `<W>`
follows the stage-aware freshness windows below (15 default / 30
Codex-ensembled). This applies to EVERY site that posts a
`stage-dispatch` breadcrumb: the Step 8 results-landed batch, Step 9
rounds, the Step 9a-ter free-analysis follow-up, the
methodology-reference spawn, AND all same-issue follow-up-loop
`stage=followup-<phase>` dispatches — the #778 double-dispatch
(2026-07-01: two orchestrators each dispatched a `followup-implementing
round=1` implementer 5m39s apart, two implementers concurrently editing
one worktree) came through the follow-up loop.

Each stage's result marker is its completion signal — the existing
`epm:upload-verification` (verifying), `epm:interpretation v<r>` +
`epm:interp-critique v<r>` (interpreting), and `epm:clean-result-critique
v<r>` (clean-result). The breadcrumb is a generic `epm:progress` note (no
new marker schema), distinguished by its `stage-dispatch` prefix. The
`worktree=` field records WHERE the dispatched subagent writes — the
absolute worktree path, or the literal `repo-root` when it works in the
main checkout — so a successor session or recovery pass can locate
uncommitted in-flight files if this session dies mid-dispatch. (Incident
#505 round 2, 2026-06-10: a killed implementer's three uncommitted files
sat in a worktree no marker named, stalling recovery for 5+ hours.) The
same field applies to every dispatch breadcrumb that follows this
convention, including the same-issue follow-up loop's
`stage=followup-<phase>` dispatches.

**Pre-dispatch external-marker triage (REQUIRED — every COMPUTE-stage
dispatch; sibling of the pre-dispatch dedup above).** Cross-session markers
are the sanctioned advisory channel, but a mailbox with no read-gate is how
#779 launched an 18–20h serial grid 4 minutes after finishing its prior
phase while a measured audit saying "must NOT launch as-is" sat unread on
its own events.jsonl (2026-07-02: 10 external audit/directive markers,
16:12–17:47Z; the 20:46Z `stage=followup-grid` breadcrumb claimed
"vectorized" while the fixes were unapplied; killed by PM-chat at 12 min).
Immediately BEFORE posting the dispatch breadcrumb for any stage that
launches COMPUTE — a pod/GCP/SLURM provision or workload (re)launch
(Step 6b / 6d, crash-fix relaunches included), any stage the
Compute-character pre-launch statement binds (a fit / sweep / statistical
battery: Step 9a-ter, the Step 9b same-issue follow-up loop), or a detached
VM-side phase (§ below) — run the mechanical enumerator:

```bash
uv run python - <<'PY'
from explore_persona_space.task_workflow import (
    list_events, triage_candidates_since_last_dispatch)
for e in triage_candidates_since_last_dispatch(list_events(<N>)):
    print(e["ts"], e["kind"], (e.get("note") or "").splitlines()[0][:140])
PY
```

It returns every non-machine marker posted since the PREVIOUS DUTY-BOUND
dispatch record — a compute-launch marker (`epm:run-launched` /
`epm:cluster-launched`) or a record carrying the
`external-markers triaged:` line; task start if none. A non-compute
breadcrumb (review / analyzer / verifier stage) never closes the window —
those dispatches have no triage duty, so an advisory posted before one
still surfaces at the next compute dispatch — and an untriaged compute
breadcrumb (pre-fix / concurrent session) doesn't either
(fail-toward-triage). READ each candidate's full note (`task.py view <N>
--json` + jq by `ts`); classify EXTERNAL = not posted by this session — the
`by` field is unreliable on LEGACY markers and non-compliant emitters
(measured on #779: self and PM-chat posts both carried `by: unknown`), but a
value on the #966 emitter-convention list (`pm-chat`,
`autonomous_session_watch`, `spawn_session`, `spawn_session-stop`) is a
trustworthy-positive EXTERNAL signal (conventional, not authenticated);
absence proves nothing, so still use session context plus the in-the-wild
signatures ("PM-chat", "user-raised", "user directive", "# Audit",
"AMENDMENT", "SCOPE RESTORE"); a successor/recovery session that cannot
attribute a candidate treats it as external (fail-toward-triage). Then
TRIAGE each external advisory/directive: APPLY it (fix the code, adjust or
re-scope the dispatch) BEFORE dispatching, or EXPLICITLY DEFER with a
one-line reason. If applying took non-trivial time, RE-RUN the enumerator
immediately before posting the breadcrumb (markers can land mid-apply).
Record the outcome as ONE line in the dispatch breadcrumb note — or, for
pod/backend launches that post no breadcrumb, in an immediately-adjacent
`epm:progress` note:

    external-markers triaged: <N> applied / <M> deferred (<one-line reasons>)

or `external-markers triaged: none` when there are no external candidates.
This is NOT a gate: triage is apply-or-defer, decided by this session,
auto-continue preserved — but deferring a marker that contradicts the
dispatch (e.g. "do not launch as-is") must state WHY the launch is sound
anyway, and a dispatch note asserting a property an unapplied external
audit contradicts (#779's "vectorized") without a triage line is the
regression this rule closes. Triage is BOUNDED to the window; a marker
already covered by a prior triage line is not re-enumerated (its
disposition is on the record). Accepted residuals (named, not silent): a
marker posted in the seconds between the final enumerator run and the
breadcrumb post lands before the new boundary and is not re-enumerated;
markers posted after a task's LAST compute dispatch are never enumerated
(they can no longer avert a launch); a legacy launch marker
(`epm:run-launched` / `epm:cluster-launched` posted pre-fix without triage)
still closes the window. A watcher-side NON-GATING observer audits this
duty post-hoc (flags missing/'none' lines against a re-run of the
enumerator's window; observe/alert only, never blocks — #967).

**Detached VM-side long compute phases (setsid; pid+log in the breadcrumb — #833).**
Any VM-LOCAL compute phase with projected wall-time >~15 min that the
orchestrator launches DIRECTLY as bg-Bash (a Phase-D-style fit, an
aggregation / permutation battery) MUST be launched fully detached:

    PHASE_PID=$(bash -c 'setsid nohup env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 <cmd> < /dev/null >> <abs, space-free log path> 2>&1 & echo $!')
    ps -p "$PHASE_PID" -o args=   # verify the pid is the workload; on mismatch
                                  # recover via pgrep -f '<distinctive invocation>'
    bash -o pipefail -c 'pgrep -s "$1" | xargs -rn1 sudo -n choom -n -600 -p' _ "$PHASE_PID" >/dev/null \
      || echo "[warn] choom failed or swept nothing — phase is earlyoom-UNPROTECTED (record choom=failed)"

The `bash -c` wrapper is load-bearing for pid capture: the top-level Bash-tool
shell runs with job control ON, where `setsid` forks and a bare `$!` is the
vanished intermediate; inside the wrapper (job control OFF) `setsid` execs in
place, so `$!` is the workload pid. A plain bg-Bash child stays in the
session's kill domain (script-launched children share the launcher's process
group + session id; even a top-level child with its own pgid shares the sid),
and a watcher force-stop / `spawn_session.py stop` kills that tree — #833's
healthy ~3-6h Phase-D fit died mid-flight this way (2026-07-02, ~2h lost, pure
signal kill). `setsid` gives the phase its own session + process group (group
kills miss it; it reparents to PID 1 when the launching shell exits);
`< /dev/null >> log` drops every fd tether to the dying session. The phase's
stage-dispatch breadcrumb MUST carry three additional fields:
`... pid=<PHASE_PID> log=<abs log path> choom=ok|failed` (additive whitespace-split
`key=value` tokens — `_breadcrumb_fields` parses them order-free; keep the
log path space-free; #833's own breadcrumbs already carried a RELATIVE `log=`
— this convention upgrades it to a REQUIRED absolute path, and `pid=` is the
genuinely new field), and the `external-markers triaged:` line
(§ Pre-dispatch external-marker triage above).

**Earlyoom protection is REQUIRED on the verified phase (#957; incident #811).**
The shared VM runs `earlyoom` with `--prefer '(^|/)(pytest|python3?)$'` (+300
badness to every python process), so a long detached fit is the designated
victim whenever ANY neighbor spikes memory: #811's ~5h fits phase (RSS 6.8 GiB)
was SIGTERM-killed at ~2h (rc=143, 0 checkpoints) by a NEIGHBOR's spike — its
logged badness 1002 decomposes exactly as (1000 + 53‰ RSS)×2/3 + 300
prefer-bonus. The `choom -n -600` sweep runs over the phase's whole SESSION
(`pgrep -s` — `setsid` made `$PHASE_PID` the session id, so the sweep catches
the leader AND any already-forked child; children forked later inherit
`oom_score_adj` across fork), subtracting ~400 display points: decisively below
every default-adj neighbor while staying killable — NOT `-1000`, which earlyoom
and the kernel OOM killer skip entirely, so a genuinely runaway fit must still
die first. Lowering adj needs CAP_SYS_RESOURCE, hence `sudo -n` (passwordless
on the VM); on failure the launch PROCEEDS unprotected with the `[warn]` +
`choom=failed` breadcrumb token — never block a launch on choom, and never read
the sweep as guaranteed protection (it re-orders earlyoom's victim selection;
it does not exempt the phase). Record `choom=ok` ONLY when the sweep pipeline
itself exited zero; anything else records `choom=failed`. The −600 derivation
assumes this VM's current `--prefer` +300 python bonus (`/etc/default/earlyoom`);
re-derive from the decomposition above if that config changes.
**Collateral-kill signature + second-kill pod pivot.** Phase dead rc=143
(SIGTERM; rc=137 when earlyoom escalated to SIGKILL) + an earlyoom journal line
at the death timestamp naming the phase pid, with the memory SPIKER a NEIGHBOR
(phase RSS well under the pressure; attribute via the kill-source checklist,
`failure_patterns.md` § exit-137/143, + the watcher's `earlyoom-kill` sidecar
rows) = a collateral kill: `failure_class: infra`, NOT a code bug — do not
dispatch a crash-fix round against the phase. Recovery ladder: relaunch ONCE
with protection verified (`choom=ok`); if a PROTECTED phase is earlyoom-killed
AGAIN, the VM is structurally memory-contended for this phase (or the phase
itself is now the top consumer) — route it to the cheap CPU pod lane
(`cpu-mid` / `cpu-bigmem` by footprint, CLAUDE.md § CPU-only phases) instead of
a third VM relaunch. The phase-is-the-spiker variant + the stream-reduce rule
stay in `.claude/rules/gotchas.md` (earlyoom entry).

**The thread-cap `env` prefix is REQUIRED on every VM-side launch (#891).** The
shared-VM setdefault (#847, `orchestrate/env.py`) is `src/`-side and pinned to
the WORKTREE's branch point — the Step 5a spec-freshness sync deliberately
never syncs `src/` — so an in-flight worktree cut before an infra fix launches
with the pre-fix library (incident #779, 2026-07-02: a pre-#847 worktree's grid
ran 78 uncapped threads ~20h after the fix landed on main). The explicit prefix
is branch-age-independent AND caps torch/BLAS regardless of the script's import
order (the env.py hook cannot in-process-cap a script that imports torch before
`load_dotenv()`; the launch env can). `env` execs `<cmd>` in place, so `$!` pid
capture and the `ps -o args=` identity verify are unchanged. env.py's
setdefault never clobbers these values. A phase that genuinely needs a wider
cap on the shared VM states the wider explicit value + a one-line reason in its
breadcrumb. Pod / GCE / SLURM launches NEVER carry the prefix — dedicated boxes
keep full width (the #847 scope invariant).

EXEMPT: work that COMPLETES within a subagent's own
bounded turn (a subagent that bg-launches a VM-local >~15-min phase and
returns follows this SAME detach shape — the phase sits in the session's kill
domain either way), pod-side workloads (the pod launch contract / #883), the
deadline-bounded off-pod `batch_judge` poll, and quick probes / plots
(< ~15 min). Off-VM routing (CLAUDE.md § CPU-only phases — the cheap
dedicated CPU-pod lanes) remains the FIRST preference for long CPU phases;
this convention governs the residual phases that legitimately stay VM-local
and is not a permission slip for long VM-side compute.

**Successor / re-entry rule (overrides the freshness window below).** When the
current stage+round's most recent breadcrumb carries `pid=`, probe
`ps -p <pid> -o args=` BEFORE any re-dispatch decision — the SAME identity
verify as at launch, never a bare liveness check (on a shared VM a recycled
pid would otherwise "re-attach" to a stranger and suppress the needed
relaunch). Alive AND args match the distinctive invocation → the phase is IN
FLIGHT regardless of breadcrumb age (a detached multi-hour phase posts no
markers while computing): RE-ATTACH — poll the pid, `tail` the breadcrumb's
`log=` for real progress (alive ≠ progressing; the log is the progress
signal), post a liveness `epm:progress` note — never relaunch. Dead — or an
args MISMATCH (recycled pid: treat as dead) — with its completion sentinel /
output present → stage done; proceed. Dead with no completion output →
genuinely failed: run the kill-before-relaunch probe
(`.claude/rules/crash-fix-rounds.md` § Kill-before-relaunch), then relaunch.
`stage_dispatch_should_skip` knows nothing about pids, so the polling
orchestrator also refreshes the mechanical window with periodic liveness
`epm:progress` notes (those refresh the events.jsonl effective-age window;
the watcher's stall detector reads the session SELF-REPORT, so a long-idle
session may still be stopped — benign under this rule: the detached phase
survives and the successor re-attaches); the identity-verified pid probe is
the authoritative check at re-entry. Prefix those periodic liveness notes
with `[long-phase-heartbeat]` — the watcher's stalled detector AND
`tick_triage.py` (#1051) grant the 90-min leash on that opt-in; tick_triage
probes the breadcrumb's `pid=` (a VM-LOCAL pid, start-time identity-guarded
— never post a pod-side pid in a `stage-dispatch` breadcrumb) before any
STALE-REDRIVE, and while that pid breadcrumb is in flight the pid evidence
OVERRULES heartbeat notes. (This convention is the detached-phase
instance of the § Long-phase heartbeat duty, Step 6d.2 — the ≤45-min
resume structure, the verify-first ban, and the self-report refresh
there bind here too.)

**Checkable guard rule (run at Step 9 / Step 8 entry on every
re-invocation).**
1. Read the most recent events.jsonl marker via
   `task.py latest-marker <N>` (and `task.py view <N> --json` for the tail
   if needed).
2. Scan `events.jsonl` BACKWARDS for the CURRENT stage+round's most
   recent `stage-dispatch` breadcrumb (an `epm:progress` note BEGINNING
   `stage-dispatch ` — a note merely quoting the string mid-note never
   counts), skipping ALL other kinds — intervening markers (codex-task
   markers, progress notes, plan markers, other stages' breadcrumbs and
   result markers) never hide a breadcrumb. If such a breadcrumb exists
   AND no result marker for THAT stage (nor `epm:failure`) was posted
   after it, compare its EFFECTIVE age to the **stage-aware freshness
   window** — effective age is measured from the LATEST of the
   breadcrumb and any subsequent liveness marker (`epm:codex-task-*`,
   `epm:smoke-architecture-check`, `epm:proposed-tests`, or a
   non-breadcrumb `epm:progress` — excluding anti-liveness notes: a
   `deliberate-stop` stop record and `[autonomous_session_watch:...]` /
   `[spawn-session:...]` telemetry never refresh the window, #810/#949):
   a healthy long-running round keeps
   refreshing its window; a dead one goes silent and re-dispatches once
   the window expires. The mechanical form of this rule is
   `task_workflow.stage_dispatch_should_skip` (run the pre-dispatch
   one-liner above — do not eyeball the scan).
   - Window = **30 min** for Codex-ensembled rounds (ALL `interpreting`
     AND `clean-result` rounds up to the per-reviewer cap (5) — every round spawns both the Claude
     critic AND a `codex-*-critic` twin at `--effort high|xhigh` via
     `companion task` since the 2026-06-12 all-rounds policy; such
     rounds commonly exceed 15 min wall time).
   - Window = **15 min** for everything else (`verifying` and any other
     Step 8/9 stage).
   - **effective age < window** → the subagent is presumed STILL
     RUNNING. EXIT the skill cleanly (`post_step_completed.py ...
     --exit-kind parked --notes "stage <stage> round <r> still in flight
     (dispatched <Δ>m ago, window <W>m); backstop tick yielding"`). Do
     NOT re-dispatch — let the live work finish; the next tick (or the
     live subagent's own completion) advances the pipeline.
   - **effective age >= window** → the stage looks genuinely STALLED (a
     subagent that never posted its result). Proceed to re-dispatch it
     normally (the freshness window is what distinguishes "live" from
     "stalled").
3. If the per-stage backwards scan finds NO open breadcrumb for the
   current stage+round (none exists, or a result marker / `epm:failure`
   postdates it), there is no in-flight work — proceed with the normal
   Step 9 logic below.

**Parallel-stage note (results-landed spawn).** Step 8's results-landed
parallel spawn can put `verifying`, `interpreting` round 1, and
`methodology-reference` breadcrumbs in flight at once. The per-stage
backwards scan in step 2 above applies to each concurrent stage
independently — a result marker for stage X never clears stage Y's
in-flight state.

The 15-min default comfortably exceeds a single Claude analyzer / critic /
verifier turn; the 30-min Codex-ensemble window covers a high-effort
Codex twin's wall time without re-dispatching live work and risking a
double-writer on `body.md`. Both fit cleanly under the 45-min backstop
cadence, so a genuinely stalled stage is still re-dispatched within
~2 ticks (≈90 min worst case). This guard is the
bound referenced by the Step 6d.2 "surviving the backstop into
verifying/interpreting/reviewing is DESIGNED behavior" paragraph.

**Limitation (be explicit about it).** A MISSED `stage-dispatch`
breadcrumb (the orchestrator spawns a stage subagent but forgets / fails
to post the breadcrumb FIRST) silently disables this guard for that tick:
with no breadcrumb to detect, step 3 of the rule fires and the
orchestrator re-dispatches the stage as if no in-flight work existed —
exactly the double-dispatch / double-writer the guard exists to prevent.
The breadcrumb is the only enforcement; the orchestrator MUST treat
posting it as a non-skippable precondition for every Step 8/9 stage
dispatch. If you notice a stage subagent was spawned without one, post
the breadcrumb immediately (`task.py post-marker ... epm:progress --note
"stage-dispatch stage=<s> round=<r> subagent=<name> worktree=<abs path or
'repo-root'>"`) so the next tick's guard fires correctly.

**9a. Iterative interpretation** (only if status is `interpreting`)

Only for `experiment` tasks. Code-change tasks never reach this step
because Step 5 already PASSed code-review and routed them to Step 9c
(the inline test-verdict gate) directly.

The interpretation loop produces a polished clean-result body through
iterative refinement between the analyzer and an interpretation-critic.
Worktree-cwd sessions run the Step 5a spec-freshness check before the
first dispatch of this loop (analyzer + critic specs load from the
worktree copy).

**Round 1:**

**Held-output publish (results-landed early spawn).** When Step 8's
results-landed parallel spawn already ran the analyzer first pass in
HOLD-marker mode, do NOT re-spawn it here: post the held
`/tmp/issue-<N>-interpretation-v1-held.md` verbatim as
`epm:interpretation v1` (this happens immediately after
upload-verification PASS, per Step 8's join #1) and continue at round-1
step 2 (the critic ensemble). Fall through to the normal spawn below
only when no held output exists (early spawn skipped, crashed, or
discarded by Step 8's gap-fill decision rule).

1. Spawn `analyzer` agent (fresh context) with raw result paths. The
   analyzer:
   - Writes the **Fact Sheet** (reproducibility card, artifact URLs,
     raw numbers, plots, sample outputs) — this is written once and not
     revised.
   - Writes the **Interpretation** (background, methodology, results
     claim + hero figure + main takeaways + confidence, next steps).
   - Generates plots via `paper-plots` skill, saves them under
     `figures/issue_<N>/`, commits + pushes them to `main` BEFORE
     writing the body, and references each figure INLINE inside the
     relevant `### <finding>` H3 under `## Findings` (no separate
     `## Figure` H2 — that H2 is retired) via
     `![alt](https://raw.githubusercontent.com/<owner>/<repo>/<sha>/figures/issue_<N>/<file>.png)` —
     a SHA-pinned absolute URL the dashboard can fetch. Relative
     `artifacts/...` / `figures/...` URLs render as broken images on
     the dashboard and are rejected by `verify_task_body.py` Check 4b
     (incident: task #365, 2026-05-22). See
     `.claude/agents/analyzer.md` Step 3 for the full save-commit-pin
     workflow.
   - Posts `epm:interpretation v1` on the source task.

2. Spawn the **interpretation-critic ensemble** (fresh contexts, single
   message, both `run_in_background=true`):
   - `interpretation-critic` (Claude) — full 7-lens review. Posts
     `epm:interp-critique v1` with PASS or REVISE.
   - `codex-interpretation-critic` (Codex gpt-5.5 via `companion task`)
     — same 7 lenses (lens 6 plot-prose works on Codex multimodal).
     Posts `epm:interp-critique-codex v1`.

   Quota-sentinel pre-check first (#1204, CLAUDE.md § Codex ensemble
   review): when LIVE, spawn only the Claude critic; instant confirmed
   Codex no-show per the decision-table no-show row + one `epm:progress`
   note.

   Neither sees the analyzer's reasoning. Independence is load-bearing.

3. **Apply ensemble decision rule** (see
   (see workflow.yaml § ensemble_review)):

   | Claude | Codex | Action |
   |---|---|---|
   | PASS | PASS | `final_verdict = PASS`. Concatenate suggestions for analyzer's optional polish. |
   | REVISE | REVISE | `final_verdict = REVISE`. Union the revision requests (dedup exact-same). |
   | PASS vs REVISE (or vice versa) | (the other) | Spawn `reconciler` (marker mode). Brief: role=`interpretation-critic`, both event bodies, interpretation body path, eval JSON paths, figure paths. Reconciler posts `epm:review-reconcile v<n>` with binding PASS or REVISE. `final_verdict = reconciler's verdict`. |
   | Codex no-show (`epm:failure` posted, or NO durable verdict per the Step 5b durable-verdict-first rule) | (any) | Fallback: `final_verdict = Claude verdict`. Surface "Codex twin no-show round <n>" to chat. |

   An Agent-tool error for EITHER critic first triggers the Step 5b
   durable-verdict-first check (re-read events.jsonl for
   `epm:interp-critique[-codex] v<n>`, then the round-fresh Codex output
   file): a thrash-killed summary turn with a posted verdict is a RETURNED
   reviewer; a Claude critic with no durable verdict is re-spawned once
   per the Step 5b bound, not skipped.

   Reconcile rounds do NOT increment the per-reviewer round counter.
   Adopt-more-severe WITHOUT a reconciler is unsanctioned here
   (the #825 deviation site) — see the Step 5c ban: when both reviewers
   returned disagreeing durable verdicts, spawn the reconciler; a
   twice-dead reconciler fails LOUD per Step 5b item 4 (the
   adopt-more-severe fail-safe is `/adversarial-planner`-in-context
   only), and the Codex no-show fallback remains a separate, sanctioned
   path.

**If `final_verdict == REVISE` (rounds 2-5):**

Re-spawn analyzer (fresh context, sees original data + ALL critique
feedback: Claude event + Codex event + reconcile event if any).
Analyzer posts `epm:interpretation v2`. Re-spawn the ensemble (fresh
contexts, sees v2 + prior critique events). Posts both
`epm:interp-critique v2` and `epm:interp-critique-codex v2`. Apply rule
again.

**Max 5 rounds per reviewer.** At round 5 (the cap) with a non-PASS
ensemble verdict, apply the Step 9a-bis-style procedural-only strip once
more (procedural / presentation REVISEs). If ALL residual REVISEs are
stripped → advance with full critique history. If ANY SUBSTANTIVE
residual remains — a flagged OVERCLAIM the strip cannot resolve — SURFACE
it, do NOT auto-publish into the record (this is the MOST important site
for surface-not-ship, #784: a real residual at interp is an overclaim
that must never be silently promoted). Either way post the §5 marker
first (`uv run python scripts/post_step_completed.py --issue <N>
--step 9a --exit-kind parked` interactive / `--exit-kind failure-exit`
autonomous). Interactive: present the residual
to the user + EXIT. Autonomous (`EPM_AUTONOMOUS_SESSION=1`): post
`epm:failure v1 failure_class: code` referencing the residual, set
`status: blocked`, fire `PushNotification`, run CRON-TEARDOWN, EXIT
(halt_criteria id=6 `concern_unresolved` family).

**On PASS (or all-stripped at the cap):**

The analyzer **promotes the source task IN PLACE to a clean-result** —
no separate task is created. The analyzer:

1. Snapshots the prior body to `original-body.md` via an
   `epm:original-body v1` event (audit / rollback).
2. Replaces `body.md` with the polished markdown write-up:
   ```bash
   uv run python scripts/task.py set-body <N> --file /tmp/clean-result-body.md
   uv run python scripts/task.py set-title <N> "<claim summary> (HIGH|MODERATE|LOW confidence)"
   uv run python scripts/task.py set-clean-result <N>   # flips has_clean_result=true
   ```
3. Runs `scripts/verify_task_body.py <body-file>` — FAIL blocks the
   write-up.

Posts `epm:clean-result-drafted v1` on the source task with the title
and a 2-sentence recap.

Then proceed to **9a-humanize (clean-result prose humanize-loop pass)**
before advancing to clean-result-critic.

**9a-humanize. Clean-result prose humanize-loop pass** (orchestrator-level
— only on the first time `epm:clean-result-drafted v1` is posted, NOT on
round-2/3 revisions out of 9a-bis)

The analyzer ran an inline humanize-quick self-pass on the reader-facing
prose during its draft (analyzer.md Step 4.5). This orchestrator step adds
the second-opinion layer: a real `/humanize loop` invocation with a
separate hostile critic subagent the analyzer could not spawn from inside
its own subagent context.

The pass targets the v3 reader-facing prose surfaces — `## Takeaways`
(the bullet block Thomas adapts for Slack) + `## What I ran` + the
`## Findings` setup/read prose (bullets). This is exactly what Thomas
reuses verbatim for Slack and the rolling cross-round synthesis, so its
register matters most. The `## Data` capsules + example blocks,
`## Reproducibility` appendix, and figure captions are OUT of scope —
they carry project jargon on purpose, and the clean-result-critic in
9a-bis enforces register discipline on them. (Legacy/in-flight v2 bodies:
the pass targets the `## TL;DR` block — `<section id="tldr">` for the HTML
card — instead; branch on the body sentinel.) Expect the pass cheaper
than the v2 era — the v3 surfaces are bullets at ~800 words, not a
multi-paragraph LessWrong narrative.

**Paper-mode (`paper: true`): SKIP this orchestrator-level pass.** A
paper-task's reader-facing prose lives in the `.tex`
(`docs/papers/issue_<N>/issue_<N>.tex` Abstract / Introduction / Results
interpretation / Discussion), not a markdown `body.md` to extract — and
the analyzer already ran `/humanize academic` (em-dash zero-tolerance,
copula avoidance, classical academic terms) on those paper surfaces
INTERNALLY during its PAPER-TASK MODE Step 4.5 (`.claude/rules/analyzer-paper-mode.md`
§ PAPER-TASK MODE). Post `epm:humanize-loop v1` with `note: skipped —
paper-task (analyzer ran inline /humanize academic on the .tex)` so the
audit log records it, and proceed straight to 9a-ter.

**Procedure:**

1. Read the published body via `task.py view <N>`; extract the v3 prose
   surfaces (`## Takeaways` + `## What I ran` + `## Findings` setup/read
   bullets; for a v2/legacy body extract the `## TL;DR` block instead).
2. Invoke `/humanize loop` with those prose surfaces as the target.
   **Read the
   draft file once BEFORE the first Edit on it (and re-Read after any
   compaction)** — the draft is typically written by the critic subagent, so
   it is not in the orchestrator's Edit state, and blind Edits bounce with
   "File has not been read yet" (10 such rejections across three sessions on
   2026-06-09, 8 of them consecutive in one humanize pass). The skill
   spawns a hostile critic subagent (from the orchestrator's context —
   allowed; the analyzer could not because subagent-from-subagent is
   forbidden) that scores against the six-axis rubric:
   - vocabulary (AI-tell words)
   - structure (rule-of-three, negative parallelisms, inflated symbolism)
   - rhythm (sentence-length monotony, metronomic cadence)
   - voice ("we"-slippage, corporate hedging, promotional language)
   - interpretation honesty (buried caveats, misplaced hedging)
   - results-writing discipline (effect sizes / named tests in prose,
     Δ-notation, undefined jargon — anti-patterns from CLAUDE.md
     "Statistics" rules and the clean-result-critic statistical-framing
     lens)

   **Hard ban gate scoping (binding; incidents #498/#518/#923):** the
   `/humanize` skill's mandatory `check_bans.sh` absolute-ban gate runs
   over AUTHORED PROSE ONLY — for clean-result work the ELIDED copy below
   IS the ban-gate input (a repo-side override of the user-global skill's
   whole-body gate wording), never the raw whole body. SPEC-required
   verbatim sample completions legitimately contain ban-listed strings
   ("Certainly!", "Sure, I'd be happy to help"), and rewriting them to
   satisfy the gate destroys scientific evidence. Gate the body file —
   `/tmp/issue-<N>-humanize-loop.md` when the loop produced revisions; if
   the loop made no revisions, materialize the current body to that path
   first — AFTER eliding the verbatim-quotation surfaces: fenced ``` blocks,
   `<details>...</details>` example blocks, `>`-blockquoted lines (with or
   without a following space), and `**Completion:**` sample lines:
   ```bash
   awk '/^```/{f=!f; next} f{next} /^<details/{d=1} d{if(/<\/details>/)d=0; next} /^>/{next} /^\*\*Completion:\*\*/{next} {print} END{if(f||d) exit 3}' \
     /tmp/issue-<N>-humanize-loop.md > /tmp/issue-<N>-ban-scan.md \
     && ~/.claude/skills/humanize/check_bans.sh /tmp/issue-<N>-ban-scan.md
   ```
   awk exit 3 = structurally unbalanced body (unclosed fence/`<details>`) —
   a hard workflow error: the gate does NOT run; fix the body structure
   and re-run. A hit SURVIVING elision is PRESUMPTIVELY authored prose —
   default: real FAIL, rewrite it; if inspection shows it is verbatim
   sample text the elision missed (indented fence, inline `<details>`,
   multi-line completion), strengthen the elision instead and document
   the disposition — NEVER rewrite the sample. A hit whose ONLY
   occurrences were elided is a FALSE POSITIVE: treat the gate as PASS on
   authored prose, NEVER rewrite the sample, and DOCUMENT the disposition
   in the `epm:humanize-loop` note (step 5), naming the banned string AND
   its location. Never move authored prose into a blockquote/fence to
   dodge the gate.

3. Loop until all axes score ≤ 1 OR **3 orchestrator-level cycles**
   reached.
4. If the loop revised the prose surfaces, write the new body to
   `/tmp/issue-<N>-humanize-loop.md`, then update via:
   ```bash
   uv run python scripts/task.py set-body <N> --file /tmp/issue-<N>-humanize-loop.md
   uv run python "$REPO_ROOT"/scripts/verify_task_body.py --issue <N>  # main-checkout copy, never the worktree's (spec-stale risk, incident #496)
   ```
   The verifier MUST still PASS — the humanize loop is not allowed to
   produce a body that breaks Lens 1-15 mechanical checks. If it does:
   revert to the pre-loop body and surface the conflict to the user
   (this is rare; the loop only edits prose, not structure).
5. Post `epm:humanize-loop v1` on the source task with the final 6-axis
   scores + a one-line note ("converged in cycle K" or "exited at cap,
   residual debt: axis X scored 2 — flagged to user"). When the ban gate
   recorded a verbatim-sample false positive, append the disposition to
   the note, naming the string and its location (the #923 form: "ban
   gate: PASS on authored prose; 1 hit ('Certainly!', ## Methodology
   sample block) — false positive, left in place").

**Skill availability fallback:** if `/humanize` is not loaded in the
runtime (plugin missing), skip 9a-humanize entirely and proceed to
9a-ter. The analyzer's inline Step 4.5 already provided a first-pass
cleanup; the orchestrator pass is additive. Post
`epm:humanize-loop v1` with `note: skipped — /humanize skill not
loaded` so the audit log records the skip.

**Then proceed to 9a-ter (auto-run free-analysis follow-ups).**

**9a-ter. Auto-run free-analysis follow-ups** (only if status is
`interpreting`, after Step 9a-humanize completes)

The analyzer's Step 6.5 (and the follow-up-proposer's `cost_class` /
`est_gpu_hours` schema) record whether any follow-up is executable
with ZERO new GPU (`cost_class: free-analysis`, `est_gpu_hours: 0`).
When such a follow-up exists and has not yet been run on this task, the
orchestrator AUTO-RUNS it inline BEFORE the clean-result-critique gate
(9a-bis) — so the critic gates the UPDATED body, not a body that
already names a free win it didn't take. **The `headline_affecting: yes`
requirement was DROPPED 2026-06-13** — a zero-GPU follow-up auto-runs
whether or not it would move the parent's headline (the standing
directive: follow-ups that are 0 GPU-h or `< 20` GPU-h just run and fold
into the same issue). This 0-GPU inline step is the floor of the
cheap-auto-run band; the GPU-backed `0 < est_gpu_hours < 20` band runs at
9b via the same-issue follow-up loop. This step fires in BOTH
interactive and autonomous (`EPM_AUTONOMOUS_SESSION=1`) sessions
identically (as does the 9b cheap band, as of 2026-06-13 — the
remaining autonomous-ONLY routing at 9b is the `est_gpu_hours >= 20` /
`auto_run: yes` expensive path: same-issue loop for `same`, child filing
for `substantially-different`). The whole
<!-- example: anti-pattern -->
step is auto-continue (NOT a new
`AskUserQuestion` gate); the halt-criterion contract is preserved.
<!-- autonomous-mode: auto-resolve -->
Same behavior in interactive and autonomous sessions: no
AskUserQuestion is ever raised by this step; the marker
`epm:free-analysis-followup-run v1` is the durable record consumed by
re-entry idempotency.

**Detection.** Read the latest analyzer output (the `## Free-analysis
follow-ups (orchestrator: auto-run before parking)` H2 block in its
return text — see analyzer.md Step 6.5) AND the latest `epm:analysis
v<n>` marker on the source task (its `free_analysis_unrun:` field).
Take the union. For each entry:

1. Skip it if an `epm:free-analysis-followup-run v1` marker on this
   task already records that follow-up as run (idempotency — match by
   the verbatim follow-up title field).
2. Skip it if the implementer (below) reports the follow-up is NOT
   actually free-analysis (e.g. it discovered the change needs new
   eval data after all) — see ABORT path below.

The orchestrator MAY additionally sanity-check that the eval-data
path(s) an entry names actually resolve (local file exists /
`huggingface_hub.list_repo_files` for HF paths) before dispatching; an
entry whose premise path does not resolve takes the ABORT path's
reclassification up front (post the `epm:free-analysis-followup-run v1`
abort record naming the missing artifact) without burning an
implementer round. The analyzer's Step 6.5 artifact-premise check is
the primary defense; this is a backstop (incident #552).

When the detection union is empty, this step is a no-op: log one chat
line (`No free-analysis follow-ups to auto-run`)
and proceed directly to 9a-bis. (Detection no longer filters on
`headline_affecting` — every unrun `cost_class: free-analysis` follow-up
is eligible.)

**Loop guard (critical).** This step caps at AT MOST ONE free-analysis
follow-up run per task. The cap is enforced by the
`epm:free-analysis-followup-run v1` marker: re-entry into 9a-ter on the
same task — whether from a backstop tick, an analyzer revision posting a
new free-analysis follow-up, or a 9a-bis REVISE round that bounced back
to analyzer — checks the marker FIRST and exits immediately if it is
already present (regardless of whether the listed follow-up is the same
one). This prevents the re-run from triggering another auto-run chain
within the same task; the second free-analysis follow-up surfaces in
the body as a regular bullet for a future human pass. Across tasks the
mechanism stays fresh (each task gets its own one round).

**Compute-character pre-launch statement (REQUIRED — one paragraph, not a
planner round, not a gate).** "0 GPU-h" does not mean "0 compute review":
this step, the Step 9b same-issue follow-up loop, and the CLAUDE.md
§ Routing "User-chat inline free analysis" carve-out are the workflow's
PLANNERLESS paths — they skip the planner+critic stack, where all
compute-character review lives (incidents #667/#722/#778: reused serial
parent code burned hours on "0 GPU-h" work, caught only by ad-hoc human
watches). Before dispatching any stage that launches a fit, sweep, or
statistical battery (permutation/bootstrap/null-draw batteries,
per-cell/per-fold fits, per-row model calls), the dispatcher STATES, in
the stage-dispatch `epm:progress` breadcrumb note (or an
immediately-adjacent `epm:progress` marker): (1) the ops arithmetic —
cells × folds × draws × epochs and the projected wall-time it implies;
(2) the NAMED batched helper implementing the inner loop (e.g.
`analysis/vectorized_mlp_skill.py`; the batched `perm_null_draws` in
`analysis/null_battery.py`), or why the work is genuinely not batchable;
(3) for reused parent code, that its inner loop, device routing, + data-repo
Hub-call scoping were
INSPECTED, not assumed (cf. `.claude/rules/artifact-reuse.md`); (4) for any
VM-PLACED phase, the projected peak RSS (measured one-chunk `ru_maxrss` at
production shape, or resident-pool bytes × MEASURED live-factor —
`.claude/rules/plan-compute-sizing.md` § CPU-phase RAM/RSS routing);
projected peak RSS ≥ ~16 GB — single phase, or summed with
concurrently-resident VM phases — is a STOP: route the phase off the
shared VM (`cpu-mid` / `cpu-bigmem`) before launching (#778's 22-GiB
battery was earlyoom-killed 3× on exactly this plannerless path; #833 lost
5 cells to two concurrent ~13-15 GB phases). Projected
wall-time > ~1h without a batched inner loop is a STOP: vectorize first
(`.claude/rules/vectorize-many-cell-fits.md`), then launch. If the
realized implementation later adds a fit/battery the dispatch statement
did not cover — or materially changes its arithmetic — an updated
statement is posted before that launch. A round with no fit/battery stage states one line: `compute-character: no fit/battery stages`.
A statement covering a VM-side phase >~15 min ALSO names the detached launch
shape + log path + the thread-cap `env` prefix (OMP/MKL/OPENBLAS/NUMEXPR=8 — #891;
or the wider explicit value + one-line reason) + the earlyoom protection state
(`choom=ok|failed`) per the Step 9 entry-guard
§ "Detached VM-side long compute phases" convention.
Routing, auto-continue behavior, and the marker schema are unchanged.

**Pod-safety pre-launch signals (deviation case — a pod on a
parked/terminal parent).** This step and its user-chat sibling (the
CLAUDE.md § Routing "User-chat inline free analysis" carve-out) are
ANALYSIS-ONLY and normally touch no pod (a needs-gpu discovery takes the
ABORT path below — EXCEPT the user-chat sibling under its
explicit user inline-override clause, whose deliberate GPU run inherits
these same pre-launch signals + the compute-character statement); 9a-ter proper fires at status `interpreting`, outside
the watcher's auto-stop set, but the user-chat sibling executes on PARKED
(`on_hold`) / terminal-status parents. If an inline run following this
shape nonetheless provisions or reuses a pod on such a parent, the
ORCHESTRATOR (never the subagent) MUST run `task.py add-tag <N>
keep-running` BEFORE/AT provision — before any pod work; the
timestamp-independent tag is what shields the provision/bootstrap window,
which the watcher's ≥2-miss accumulation (~20 min) would otherwise
auto-stop straight through — AND post `epm:run-launched` on the task
immediately once the pod exists (naming the pod; in any case before
launch): the watcher's pod-safety pass
(`scripts/autonomous_session_watch.py`) auto-stops a RUNNING pod on a
parked/terminal task unless a follow-up signal marker (its predicate reads
`epm:run-launched` / `epm:followup-scope` /
`epm:free-analysis-followup-run` — a descriptive list, NOT a menu: the
inline path still never posts `epm:followup-scope`) is NEWER than the
latest done-transition (`epm:promoted` / `epm:status-changed`) or the
`keep-running` tag is present. BOTH duties bind — the marker cannot exist
during the provision/bootstrap window (#573: run-launched-only inference
stopped healthy follow-up pods 11×), any later done-transition flips the
inferred predicate off by design (the watcher's re-arm semantics), and the
`epm:free-analysis-followup-run v1` COMPLETION marker posts too late to
shield the launch (incidents #477, #573, #779 — on #779 a healthy pod-779
was repeatedly auto-stopped mid-bootstrap and misdiagnosed as a flaky
host). Remove the tag (`task.py remove-tag <N> keep-running`) when the run
completes so the auto-stop re-arms (a crashed run leaves the tag and the
pod bills until manual removal — check `pod.py audit-stale` output).

**Auto-run procedure.** For the single highest-priority unran entry
(the first one in the analyzer's surfaced order; tie-break to the one
the analyzer flagged `headline_affecting: yes` — still a useful priority
signal even though it is no longer an eligibility gate — with the most
explicit eval-data path):

1. **Dispatch breadcrumb** (Step 9 entry guard convention):
   ```bash
   uv run python scripts/task.py post-marker <N> epm:progress \
     --note "stage-dispatch stage=free-analysis-followup round=1 subagent=experiment-implementer worktree=<abs path or 'repo-root'>"
   ```
   When the follow-up runs any fit/battery, this breadcrumb (or an
   immediately-following `epm:progress` note) carries the
   § Compute-character pre-launch statement above. Every 9a-ter dispatch
   breadcrumb ALSO carries the `external-markers triaged:` line (Step 9
   entry guard § Pre-dispatch external-marker triage) — the free-analysis
   run is a VM-side compute phase.
2. **Spawn `experiment-implementer`** (paired with `code-reviewer` on
   the resulting diff — same ensemble shape as Step 5). The prompt
   names the exact follow-up + cites the eval-data path(s) it must
   re-read + states the hard constraint that the diff is
   ANALYSIS-ONLY: NO new training script, NO new eval generation, NO
   pod call, NO new prompts to a base model, NO new data file
   downloaded from outside the existing `eval_results/` / HF data
   repo paths the analyzer named. If the implementer (or
   `code-reviewer` on its diff) determines the change CANNOT be done
   without new data collection — **ABORT** the auto-run: post
   `epm:free-analysis-followup-run v1` with
   `changed_headline: false`, `gpu_hours: 0`,
   `note: aborted — reclassified as needs-gpu after implementer
   investigation; follow-up remains listed in body for manual
   triage`, and proceed to 9a-bis. The follow-up survives in the
   body as a regular bullet (now correctly understood as
   `cost_class: needs-gpu`) so a future human / autonomous pass can
   pick it up via the GPU-backed Step 9b routing (same-issue loop /
   child filing).
3. **Re-run the analysis** the implementer's diff exposes — typically
   a script in `scripts/issue<N>_*.py` or a helper under
   `src/explore_persona_space/analysis/` — over the existing eval
   JSONs. Regenerate any affected figures (the analyzer's
   `figures/issue_<N>/` outputs); commit + push to `main` so the body
   can SHA-pin them per the existing analyzer.md Step 3 rule. Push BARE:
   `git push origin main || uv run python scripts/sync_repo_root.py` —
   never piped (Step 10d § "Bare push / merge snippets"; sync_repo_root
   exit 0 can mean in-flight — landing not guaranteed, see the canonical
   block's caveat).
4. **Capture the headline before / after.** Read the current `body.md`
   H1 title before the re-spawn and the analyzer-produced H1 after,
   plus the LOW / MODERATE / HIGH confidence tag in each.
5. **Re-spawn `analyzer`** (fresh context) with the new analysis
   output + the prior body. The analyzer folds the new result into
   the existing clean-result body (typically updating one
   `### <finding>` H3 and possibly the H1 title / confidence tag),
   re-runs `verify_task_body.py` (must still PASS), and writes the
   revised body via `task.py set-body <N> --file ...`, followed by
   `task.py set-title <N> "<new H1 text>"` whenever the fold changed
   the H1 (set-body preserves the old frontmatter `title`; the
   H1==frontmatter verifier check FAILs the 9a-bis re-gate otherwise).
   The analyzer's
   Step 6.5 still fires on this re-run, but the loop guard above
   prevents another 9a-ter dispatch within the same task. (**`paper:
   true`?** The re-spawned analyzer re-authors the `.tex` in place —
   re-writing the Abstract + the affected Results subsection, re-running
   `build_paper.py` → `verify_paper.py`, re-writing the paper-stub — per
   `.claude/rules/analyzer-paper-mode.md` § "Same-issue follow-up rounds
   (paper-task)"; the mechanical gate is `verify_paper.py`, not
   `verify_task_body.py`. The same applies to the Step 9b cheap-band /
   same-issue follow-up loop folds.)
6. **Post the marker:**
   ```bash
   uv run python scripts/task.py post-marker <N> epm:free-analysis-followup-run \
     --note "followup_ref=<verbatim follow-up title> \
       headline_before=<H1 title before> \
       headline_after=<H1 title after> \
       confidence_before=<LOW|MODERATE|HIGH> \
       confidence_after=<LOW|MODERATE|HIGH> \
       gpu_hours=0 \
       changed_headline=<true|false>"
   ```
7. Proceed to **9a-bis (clean-result-critique loop)** on the UPDATED
   body. The critic gates the final state, not the pre-rerun draft.

<!-- example: anti-pattern -->
**No new gate.** This step never raises `AskUserQuestion` (in either
interactive or autonomous (`EPM_AUTONOMOUS_SESSION=1`) sessions —
auto-resolve mode is the default for both, never gate-allowed).
<!-- autonomous-mode: auto-resolve -->
If the
implementer or code-reviewer fails outright (`epm:code-review FAIL`
that survives the procedural-only strip on the first attempt), treat
it as the ABORT path from procedure step 2 — post the marker with
`note: aborted — implementer FAIL on attempt 1`, leave the follow-up
in the body as a regular bullet, and proceed to 9a-bis. The
clean-result-critique gate then runs on the analyzer's original
body; the user can pick the follow-up up post-promotion.

**Then proceed to 9a-bis (clean-result-critique loop).**

**9a-bis. Clean-result-critique loop** (only if status is `interpreting`,
after Step 9a PASS)

Same shape as the interpretation-critic loop, but the critic checks
STRUCTURE + REGISTER not CONTENT. Content honesty was settled in 9a;
this layer ensures the body matches the v3 clean-result shape (per
`.claude/skills/clean-results/SPEC.md`): five FLAT H2s in order
(`## Takeaways` / `## What I ran` / `## Findings` / `## Data` /
`## Reproducibility`), `## Takeaways` a 3-6-bullet numbers-first skim,
`## What I ran` carrying `**Why:**` / `**Design:**` / `**Training:**` /
`**Eval:**` (`**Rounds:**` when >1 round), one `### <finding>` H3 per
result under `## Findings` with one inline figure each, `## Data` with
`### Trained on` / `### Evaluated with` / `### Generated`, and confidence
in the H1 title tag only (v3 bodies bear the `<!-- clean-result-v3 -->`
sentinel; a stray `## Human TL;DR` / `## TL;DR` is a hard FAIL). The body
reads in the right register — plain academic, bullets-first, numbers
bolded. In-flight v2/legacy bodies (no v3 sentinel) keep the
2-content-section nested-TL;DR shape and are NOT newly hard-FAILed by a
v3 rule. Discipline rules: see
`.claude/skills/clean-results/SPEC.md` (canonical structure, register,
exemplars, figure captions, and research-communication principles).

**Paper-mode branch (`paper: true` frontmatter).** When the task carries
`paper: true`, the clean-result is a LaTeX **paper** at
`docs/papers/issue_<N>/`, NOT a markdown body. Both critics branch on
`paper:` internally (`.claude/agents/clean-result-critic.md` § Paper-task
review; `.claude/agents/interpretation-critic.md` § Branch on `paper:`):
the **mechanical pre-pass is `scripts/verify_paper.py`** (NOT
`verify_task_body.py` / `audit_clean_results_body_discipline.py`, which
stay the markdown verifiers), each critic reads the `.tex` + the figure
PNGs + the compiled PDF under `docs/papers/issue_<N>/`, and the SEVEN paper
lenses (P1 self-standing Introduction · P2 self-contained Methods +
Rule-A reuse-chain depth · P3 inline-subset + comprehensive-Appendix
completeness · P4 no confidence in the paper body · P5 research-paper
register · P6 `\epsref{N}` correctness · P7 verbatim examples + judge
prompts) bind INSTEAD of the fifteen markdown lenses (no `\metric` grounding
lens in v1; `verify_paper.py` checks 7-9 mechanically gate the verbatim
training/eval/output examples, the judge-prompts appendix, and the
example-provenance pointers — the no-invention floor). The orchestrator's
brief for both critics names the paper dir (`docs/papers/issue_<N>/`) and
the `.tex`/PDF read targets instead of the markdown `body.md`; the
ensemble decision rule, the round cap, and the reconciler tie-break are
unchanged. The procedural-only verdict strip below operates on the
`verify_paper.py` presentation set for a paper-task. Everything else in
this step (round loop, PASS → `reviewing`, the 9a-quater hand-off) is
identical.

**Round 1:**

Worktree-cwd sessions run the Step 5a spec-freshness check before
dispatching this round's critics.

1. Spawn `clean-result-critic` agent (fresh context, does NOT see
   analyzer reasoning). The critic reads the published body + the
   latest `epm:interpretation v<n>` event, runs
   `scripts/verify_task_body.py` +
   `scripts/audit_clean_results_body_discipline.py` as authoritative
   mechanical passes, and scores against the v3 lens set (per
   `.claude/agents/clean-result-critic.md`) — including the
   statistical-framing rule, planned-vs-actual coverage, the
   binding-concerns audit, the contaminated/failed-data-gate-arm check,
   and the v3 Takeaways / Conciseness / Data lenses. Posts
   `epm:clean-result-critique v1` on the source task with PASS or REVISE.
   (**`paper: true`?** The critic branches internally to its § Paper-task
   review: the mechanical pre-pass is `scripts/verify_paper.py` over
   `docs/papers/issue_<N>/`, NOT `verify_task_body.py` /
   `audit_clean_results_body_discipline.py`, and the seven P1-P7 paper
   lenses bind (incl. P7 verbatim examples + judge prompts + example
   provenance / no-invention; verify_paper.py checks 7-9 gate them) — see the
   Paper-mode branch paragraph above. The Check-21
   methodology-doc pass-through below is markdown-only; skip it.)

   **Check-21 methodology-doc pass-through.** When the methodology doc
   exists on the issue worktree branch (the early-spawned
   `methodology-writer` committed `docs/methodology/issue_<N>.md` at
   Step 8's results-landed spawn — see § Split schedule below), pass its
   ABSOLUTE worktree path to BOTH the verifier and the critic so check 21
   (body Parameters table ⊆ methodology doc §2 complete table) + the
   critic's Data lens can spot-check the table against ground truth:
   ```bash
   DOC_PATH="$WORKTREE/docs/methodology/issue_<N>.md"
   uv run python "$REPO_ROOT"/scripts/verify_task_body.py --issue <N> \
     ${DOC_PATH:+--methodology-doc "$DOC_PATH"}
   ```
   The doc lives on the worktree branch and only reaches the repo-root
   `main` checkout at the Step 9b auto-merge (AFTER this gate), so a
   naive main-checkout resolve would miss it — pass the worktree path
   explicitly. Check 21 NO-OP-PASSes when `--methodology-doc` is omitted
   or the doc does not yet exist anywhere (e.g. the methodology-writer
   has not returned), and binds fully at promote-time verify (post-merge,
   `kind: experiment` only). The critic brief carries the same
   `methodology_doc_path` field for the Data-lens spot-check.

2. Spawn `codex-clean-result-critic` (Codex twin) in parallel on
   every round (all-rounds ensemble as of 2026-06-12; previously
   round 1 only). Quota-sentinel pre-check first (#1204, CLAUDE.md
   § Codex ensemble review): when LIVE, skip this twin's composer
   spawn — instant confirmed Codex no-show per this step's no-show
   handling (Claude-only ensemble decision) + one `epm:progress`
   note. Brief contract (matches
   `.claude/agents/codex-clean-result-critic.md` § "Your brief
   contains" + Step 1b): pass the ABSOLUTE
   `$(task.py find <N>)/body.md` as `clean_result_body_path` and
   `$(task.py find <N>)/plans/plan.md` as `plan_path` — never a
   hand-built relative `tasks/<status>/<N>/...` (the status guess goes
   stale mid-flight and a relative path inherits the Codex dispatch
   cwd — the #489/#550 unresolvable-path false-FAIL class); extract
   the latest `epm:interpretation v<n>` note to a temp file
   (`/tmp/issue-<N>-interpretation-v<n>.md`) and pass that absolute
   path as `interpretation_marker_path` (never an `events.jsonl`
   path); pass the ABSOLUTE issue-worktree
   `docs/methodology/issue_<N>.md` path as `methodology_doc_path` when
   the doc exists (so the composer's compose-time `verify_task_body.py`
   run gets `--methodology-doc` and the Data lens can spot-check
   check 21 — the composer runs the mechanical verifiers on the VM and
   inlines their output into the Codex prompt as envelopes; this twin
   is dispatched read-only and uv cannot reliably execute in its
   sandbox (#1050); omit
   when the methodology-writer has not yet returned — check 21 NO-OP
   PASSes); and dispatch `codex_task.py` for this twin from the repo
   root, never an issue-worktree cwd. Posts
   `epm:clean-result-critique-codex v1`. Apply the
   ensemble decision rule (same shape as Step 5c — PASS+PASS, REVISE
   union, reconciler on disagreement; on any Agent-tool error, the Step
   5b durable-verdict-first rule applies — check
   `epm:clean-result-critique[-codex] v<n>` + the round-fresh Codex
   output file before any no-show fallback or re-spawn), BUT first run
   the procedural-only strip below.

   **Procedural-only verdict strip (clean-result analogue of Step
   5c-bis).** Before applying the ensemble rule, parse each critic's
   `Blocker tags:` line. A verdict is *procedural-only* when its tags
   are empty/`none` after removing `procedural` (presentation-only
   verifier FAILs: MDX prose, caption shape, cherry-label phrasing,
   sentinel scrub, URL-form) AND it carries no `structural-absence`,
   `audit`, or `lens` tag (fall back to scanning the verdict body for a
   substantive lens FAIL or audit hit if the line is absent on a legacy
   verdict). For any procedural-only non-PASS verdict the orchestrator:
   (a) does its OWN cheap re-run of `verify_task_body.py --issue <N>` on
   the canonical body and confirms the remaining FAILs are all in the
   presentation-only set; (b) applies the critic's `### Procedural
   fixes` edits to the body inline via `task.py set-body <N> --file ...`
   and re-runs the verifier to PASS; (c) treats the critic's verdict as
   PASS for the ensemble rule — this is "review incomplete → fix the
   procedural item inline + re-dispatch", NOT a consumed REVISE round
   (the round counter does NOT increment). Log one chat line:
   `procedural-only clean-result FAIL stripped — orchestrator applied N
   inline fixes + re-verified PASS; no substantive findings → PASS.` If
   ANY remaining FAIL is structural-absence, or the critic carried a
   `lens`/`audit` tag, leave the verdict as-is and apply the normal
   ensemble rule (the REVISE round counts). The strip operates ONLY on
   the mechanically-verifiable presentation set; it never overrides a
   register / story-arc / statistical-framing lens judgment. A verdict
   carrying the `data-access-blocked` tag, a `Verifier: UNAVAILABLE` /
   `Audit script: UNAVAILABLE` line, or a missing-envelope declaration
   is NEVER procedural-only — the mechanical pre-pass was unavailable,
   so there is nothing verified to strip against; re-compose /
   re-dispatch the twin instead of stripping (#1050).

**If REVISE (rounds 2-5):**

Re-spawn `analyzer` agent (fresh context, sees raw data + all
interp-critique history + the latest clean-result-critique). Analyzer
revises the `epm:interpretation` event AND edits the task body in
place via `task.py set-body <N> --file ...`. Re-runs
`scripts/verify_task_body.py` (must still PASS). Re-spawn the critic
ensemble — `clean-result-critic` AND `codex-clean-result-critic`
(all-rounds ensemble as of 2026-06-12), fresh contexts, against the
revised surfaces, with prior critique summaries in both briefs. Both
post the next critique version (`epm:clean-result-critique v<n>` +
`epm:clean-result-critique-codex v<n>`); apply the same ensemble
decision rule (including the procedural-only strip) as round 1.

**Max 5 rounds.** At round 5 (the cap) with a non-PASS ensemble verdict,
apply the procedural-only strip once more (procedural / presentation
REVISEs). If ALL residual REVISEs are stripped → advance. If ANY
SUBSTANTIVE residual remains — a flagged OVERCLAIM the strip cannot
resolve — SURFACE it, do NOT auto-publish into the clean-result record
(#784 surface-not-ship: a real residual here is an overclaim that must
never be silently promoted). Either way post the §5 marker first
(`uv run python scripts/post_step_completed.py --issue <N> --step 9a-bis
--exit-kind parked` interactive / `--exit-kind failure-exit` autonomous).
Interactive: present the residual to the
user + EXIT (the user decides whether to patch before promoting).
Autonomous (`EPM_AUTONOMOUS_SESSION=1`): post `epm:failure v1
failure_class: code` referencing the residual, set `status: blocked`,
fire `PushNotification`, run CRON-TEARDOWN, EXIT (halt_criteria id=6
`concern_unresolved` family).

**On PASS (or all-stripped at the cap):**

Move status to `reviewing`:

> **Same-issue follow-up round?** At `followups_running`, SKIP this
> `set-status` (status-hold rule, Step 9b § Same-issue follow-up loop step 3;
> code-enforced — `task.py` refuses the flip) — proceed straight to
> 9a-quater; the round exits the status only at the `awaiting_promotion` re-park.

```bash
uv run python scripts/task.py set-status <N> reviewing \
  --note "clean-result-critic PASS; advancing to final review gate."
```

**Then proceed to 9a-quater (methodology reference).**

**9a-quater. Methodology reference — POST-PASS EXPORT** (only if status
is `reviewing`, after the 9a-bis loop's PASS, before the
`awaiting_promotion` park below)

**Paper-mode (`paper: true`): SKIP the standalone-doc export.** For a
paper-task the methodology IS the paper — the `methodology-writer`
already authored the comprehensive Methods section + recipe Appendix
(complete hyperparameter table + worked examples + Rule-A reuse recipes)
DIRECTLY INTO the `.tex` at the Step 8 early spawn, and `verify_paper.py`
gated the paper's section completeness at 9a-bis. There is no separate
`docs/methodology/issue_<N>.md` to export and no top-of-body
`**Methodology:**` link to append (the `body.md` is a thin paper-stub
pointing at `docs/papers/issue_<N>/`). Post
`epm:methodology-doc-generated v1` with
`note: skipped — paper-task (Methods + Appendix authored into docs/papers/issue_<N>/issue_<N>.tex)`
so the idempotency check converges, then proceed to 9b. The v4 markdown
export below does NOT run for a paper-task.

**v4 markdown (current): MECHANICAL EXPORT, no agent spawn.** Every
`kind: experiment` v4 clean-result auto-gains a standalone methodology
reference at `docs/methodology/issue_<N>.md` that is a **mechanical COPY
of the body's `## Methodology` section** — the body's `## Methodology`
section IS the canonical source (the analyzer wrote it factually,
interpretation-free, per the v4 spec), so there is NO separate
findings-blind authoring step and the `methodology-writer` agent is NOT
spawned for v4. After the 9a-bis PASS the orchestrator:

1. Reads the finalized body and extracts the `## Methodology` section
   verbatim (from the `## Methodology` H2 to the next H2 / the `---`
   footer rule).
2. Writes it to `docs/methodology/issue_<N>.md` with the H2 header
   normalized to `# Methodology — issue <N>: <one-line what-was-run>`
   (plus a 1-line `*Derived from the [task body](https://eps.superkaiba.com/tasks/<N>).*`
   footer).
3. **Commits the doc to `main`** by explicit path (durable — removes the
   v3 worktree-only gap; the doc + its SHA-pinned link land on `main`
   directly). Capture the commit SHA.
4. Runs a no-secrets pre-scan (`scripts/check_no_secret_shaped_strings.py`
   / `redact_for_gist.py`) and publishes a **secret** (unlisted) gist
   mirror via `gh gist create` — FAIL-SOFT (a missing gist never blocks
   the step; the in-repo doc is the durable artifact).
5. Appends the one-line `**Methodology:**` pointer at the TOP of the body
   — immediately after the `<!-- clean-result-v4 -->` sentinel, before
   `## Takeaways` — linking the GitHub blob (SHA-pinned to the `main`
   commit) and the gist (drop the `· [gist](...)` suffix when the gist
   fail-softed).
6. Posts `epm:methodology-doc-generated v1` (`doc_path` + `commit` +
   `gist_url`).

Fires in BOTH interactive and autonomous sessions identically.
<!-- example: anti-pattern -->
Auto-continue (NOT a new `AskUserQuestion` gate); the halt-criterion
contract is preserved.
<!-- autonomous-mode: auto-resolve -->
Same behavior in interactive and autonomous sessions: no AskUserQuestion
is ever raised by this step; the marker `epm:methodology-doc-generated v1`
is the durable record consumed by re-entry idempotency.
<!-- example: anti-pattern -->

**v3/v2 GRANDFATHERED (LATE JOIN, in-flight bodies only):** an in-flight
v3/v2 body carries no detailed `## Methodology` section to copy, so the
legacy path still applies — the findings-blind `methodology-writer` agent
is EARLY-SPAWNED at Step 8's results-landed parallel spawn, the
orchestrator commits the doc on its return, and the gist + body
link-append (top-of-body `**Methodology:**` line + the
`## Reproducibility` `**Methodology reference:**` row) LATE-JOIN here.
The detailed v3/v2 procedure below (§ Split schedule + procedure steps
1-9) describes that grandfathered path; for a v4 body run the
mechanical export above instead and skip the agent spawn.

**Split schedule (early spawn ∥ interpretation loop).** This step is
split in two:

- **EARLY SPAWN (at Step 8's results-landed parallel spawn):** the
  orchestrator evaluates the kind-gating below, posts the
  `stage=methodology-reference` breadcrumb, pre-extracts the
  findings-blind Reproducibility input — from the `epm:results`
  markers' `reproducibility_card` (alias `reproducibility`) +
  `eval_paths`, merged newest-wins per field across markers (see
  procedure step 2), because the clean-result body's
  `## Reproducibility` H2 does not exist yet — and
  spawns `methodology-writer` in the background
  (`run_in_background=true`). This is safe because the agent is
  findings-blind by design: its inputs (plan, experiment config,
  reproducibility metadata, verbatim artifact rows) are all final the
  moment results land. When the agent returns — possibly while
  analyzer ↔ critic rounds are still iterating — the orchestrator
  immediately commits `docs/methodology/issue_<N>.md` on the issue
  worktree branch (procedure step 5 below).
- **LATE JOIN (here, after clean-result-critic PASS — the body must be
  final):** no-secrets pre-scan, secret-gist publish (fail-soft), the
  body link-append (the top-of-body `**Methodology:**` line + the
  `## Reproducibility` `**Methodology reference:**` row — procedure
  step 7), the verifier re-run, and the
  `epm:methodology-doc-generated v1` marker — posted only when the
  link line lands (the step is only "done" then). If the background
  agent has not returned yet at this point, WAIT for it here
  (TaskOutput / completion notification) before running the join.

The early spawn needs no extra gating relative to upload verification:
the agent's artifact reads are worktree-local, and the late join
already sits far after upload PASS. **Fallback (serial) path:** when
the early spawn never happened (resume of an older in-flight task, or
the early agent crashed without writing the doc), run the full
procedure below serially at this point, slicing the Reproducibility
input from the now-final body's `## Reproducibility` H2 as written in
step 2. **Early-spawn idempotency:** an in-window
`stage=methodology-reference` breadcrumb (Step 9 entry guard) or an
already-committed `docs/methodology/issue_<N>.md` on the issue branch
means the agent run is live or done — do not re-spawn it; only the
late join remains.

**When to run** (gating rules):

- `kind: experiment` → always.
- `kind: analysis` → only when the task's `## Reproducibility` section
  names a training or eval methodology (i.e. there is something to
  document). When the analysis task has no Reproducibility row beyond a
  Code SHA, the agent itself writes a 5-line "no experimental
  methodology" stub and exits; the link still lands in
  `## Reproducibility` for consistency.
- `kind: infra | batch | survey` → skip entirely. Log one chat line
  (`Step 9a-quater skipped (kind=<X>)`) and proceed to 9b.
- **Idempotency — scoped per follow-up round.** When
  `epm:methodology-doc-generated v1` is already on the task (re-entry /
  backstop tick / re-invocation after a separate 9a-bis REVISE that
  bounced back to analyzer), check follow-up coverage before no-opping:
  collect the `followup_label`s of `epm:followup-scope v1` markers
  whose round's analyzer re-fold has run (during a same-issue follow-up
  round this is exactly the current round's label; labels from rounds
  that never ran add no methodology and are ignored), and the labels
  already recorded across prior `epm:methodology-doc-generated` notes
  (`extends=` / `no-new-methodology=` fields). When every such label is
  recorded — or the task has no followup-scope markers at all — this
  step is a no-op: the doc was already written, committed, and
  gist-mirrored on a prior pass. Do NOT regenerate or re-publish. Log
  one chat line (`Step 9a-quater no-op — epm:methodology-doc-generated
  v1 already present`) and proceed to 9b. When an UNRECORDED label
  exists (same-issue follow-up re-fold), run the **EXTEND pass** below
  instead — a task-scoped no-op here would leave
  `docs/methodology/issue_<N>.md` permanently describing only the
  parent run (incident #543, 2026-06-10: a fifth arm folded into the
  clean-result had to be patched around with an in-body scope note).
- **EXTEND pass (same-issue follow-up rounds).** Re-run procedure
  steps 2-9 below for the unrecorded `followup_label`, with these
  deltas:
  - Step 2 uses the fallback (serial) body-slice form — during a
    follow-up round the re-folded body IS final post-critic.
  - Step 3 spawns `methodology-writer` in **EXTEND mode** (see
    `.claude/agents/methodology-writer.md` § EXTEND mode): the prompt
    names the mode, the `followup_label`, and the existing doc path;
    the agent reads the EXISTING `docs/methodology/issue_<N>.md`
    (findings-blind by construction) plus ONLY the new round's plan
    amendment + Reproducibility slice, and re-writes the doc by
    EXTENDING the six fixed sections IN PLACE — adding a per-round
    COLUMN to the canonical §2 hyperparameter table for whatever the
    round CHANGED (the check-21 reconciliation surface), labeled
    `### Round <label>` sub-blocks inside §3/§4/§5 ONLY where the
    round's recipe / probes / examples differ, and new rows on §6's
    existing artifacts table; parent sections preserved everywhere
    else. NEVER a new top-level `## ...` heading or a second
    §2-style table — a bare `## <followup_label> arm` H2 carrying only
    the boilerplate footer strands the round's recipe outside §2
    (incident #642). The brief inherits THIS wording, so it stays
    consistent with `.claude/agents/methodology-writer.md` § EXTEND
    mode.
  - Step 6 refreshes the EXISTING gist when a prior marker recorded a
    `gist_url` (`gh gist edit <gist-id> docs/methodology/issue_<N>.md`,
    same fail-soft rule); fall back to `gh gist create` only when no
    prior gist exists.
  - Step 7 UPDATES the existing lines' `<DOC_SHA>` pin in place in
    BOTH locations — the top-of-body `**Methodology:**` line and the
    `## Reproducibility` `**Methodology reference:**` row (never
    append duplicate lines; same `· [gist](...)` suffix rules; if a
    pre-top-line body carries only the Reproducibility row, ADD the
    missing top line while re-pinning the row).
  - Step 9 posts a NEW `epm:methodology-doc-generated v1` marker with
    `extends=<followup_label>` in the note (plus the refreshed
    `commit=` / `gist_url=`) — this is the record the idempotency
    check reads.
  - **No-new-methodology carve-out:** when the round was a
    planner-exempt re-run with an identical recipe (different seeds /
    monitoring / bug-fix re-run — nothing for a findings-blind doc to
    add), skip the agent spawn and post the marker with
    `no-new-methodology=<followup_label>` so idempotency converges
    without doc churn.

**Procedure** (auto-continue end to end — interactive and autonomous;
on the normal path steps 1-3 + 5 already ran at the EARLY SPAWN and
steps 4 + 6-9 are the LATE JOIN executed here):

1. **Dispatch breadcrumb** (Step 9 entry guard convention):
   ```bash
   uv run python scripts/task.py post-marker <N> epm:progress \
     --note "stage-dispatch stage=methodology-reference round=1 subagent=methodology-writer worktree=<abs path or 'repo-root'>"
   ```
2. **Pre-extract the findings-blind Reproducibility input.**
   On the normal (early-spawn) path the clean-result body does not
   exist yet, so extract the `reproducibility_card` (alias
   `reproducibility`; the canonical key wins within one payload) +
   `eval_paths` from the task's `epm:results` markers
   (`task.py view <N> --json`) into the temp file instead — NOT from
   the latest marker alone. Multi-launch runs legitimately post
   several `epm:results` markers, and a resume-pass sentinel can
   carry an empty card (#601: `adapter_paths: {}` after every cell
   `resumed_skip`) that would hand the methodology-writer nothing:
   resolve each field newest-wins from the newest card that declares
   it non-empty (empty dict/list/string/None is not a declaration) —
   the same semantics as `verify_uploads.py` `merged_results_card`.
   The body-slice form below is the
   fallback (serial) path, where the body IS final: slice just the
   `## Reproducibility` H2
   from the task body into a temp file and hand the agent ONLY that
   path — never the full `body.md`. Either way, this is what physically enforces
   findings-blindness: `## Takeaways` / `## Findings` / the H1 confidence
   tag (v2/legacy: `## Human TL;DR` / `## TL;DR`) never enter the agent's
   context. Prompt discipline is defense in depth on top of this
   structural cut, not the primary mechanism:
   ```bash
   BODY_PATH=$(uv run python scripts/task.py find <N>)/body.md
   REPRO_FILE=$(mktemp -t issue<N>-reproducibility.XXXXXX.md)
   awk '/^## Reproducibility[[:space:]]*$/{flag=1; print; next} \
        flag && /^## /{flag=0} flag' "$BODY_PATH" > "$REPRO_FILE"
   # Confirm the slice is non-empty; if it is, the body is malformed
   # (no `## Reproducibility` H2). Post epm:failure v1
   # (failure_class: data, reason: missing ## Reproducibility for
   # methodology-writer), set status:blocked, exit. Surface a
   # workflow-fix-candidate v1 block — the verifier should have caught
   # this upstream.
   [ -s "$REPRO_FILE" ] || { echo "Reproducibility slice empty"; exit 1; }
   ```
3. **Spawn `methodology-writer`** (fresh context, findings-blind). The
   prompt names the task number + the absolute path of the pre-extracted
   `## Reproducibility` slice (`$REPRO_FILE` from the previous step) as
   its starting input — NOT the full `body.md` path. The agent reads
   ONLY the plan, the Reproducibility slice, the training/eval scripts
   at the body's `**Code:**` SHA, the Hydra config, and a handful of
   artifact rows for verbatim worked examples. Output:
   `docs/methodology/issue_<N>.md`. See `.claude/agents/methodology-writer.md`
   for the full read/don't-read list and the "no interpretation" hard
   constraints. Delete `$REPRO_FILE` after the agent exits.
4. **No-secrets guard** (pre-publish, mandatory). Before publishing
   the gist, scan the generated doc for obvious secret patterns —
   `sk-`, `hf_`, `wandb`-key shapes, `RUNPOD`, `ANTHROPIC_API_KEY`, raw
   `.env` content — with the canonical scanner:
   `uv run python "$REPO_ROOT/scripts/check_no_secret_shaped_strings.py" "$REPO_ROOT/docs/methodology/issue_<N>.md"`
   (exit 0 = clean, exit 1 = hit). Use `$REPO_ROOT`-absolute paths — a
   bare `scripts/...` resolves against the orchestrator's cwd, which on a
   worktree (or after a removed-dir `getcwd` miss) is NOT repo root and
   resolves to `/scripts/...: No such file or directory` (incident #654,
   2026-06-17, self-recovered after a retry). Do NOT use `redact_for_gist.py` for
   this — it has only `--in`/`--out`/`--in-place`, no `--check` flag.
   The methodology-writer reads only the
   already-public Reproducibility data + the repo, so this scan should
   never trip in normal operation; it is a safety net. On any hit,
   ABORT the gist publish, keep the committed repo doc, and pass the
   `note: gist skipped — possible secret detected` field through to
   the marker (step 9). Continue to the link-append step regardless;
   the in-repo doc remains the durable artifact.
5. **Commit the doc to the repo.** Inside the worktree branch (the
   one this `/issue <N>` is running on — never the main checkout):
   ```bash
   git -C "$WORKTREE" add docs/methodology/issue_<N>.md
   git -C "$WORKTREE" commit -m "methodology: issue #<N> findings-blind reference" -- docs/methodology/issue_<N>.md
   DOC_SHA=$(git -C "$WORKTREE" rev-parse HEAD)
   ```
   Use the explicit path; never `git add -A` (avoids sweeping
   unrelated working-tree changes), and keep the commit
   pathspec-limited so any other staged entry in the index is ignored
   (same guard as the Step 10d surgical checkout). The doc rides to
   `main` with the auto-merge at Step 9b.
6. **Publish the secret gist (fail-soft).** Try once. `gh gist create
   <file>` uses the file's basename for the gist filename — the
   in-repo path is `docs/methodology/issue_<N>.md`, so the rendered
   gist filename is `issue_<N>.md` (no extra rename needed):
   ```bash
   GIST_RAW=$(gh gist create \
     --desc "Task #<N> — Methodology, hyperparameters, and worked examples (Explore Persona Space)" \
     docs/methodology/issue_<N>.md 2>&1)
   # Extract the gist URL; on failure gh writes an error to stderr/stdout
   # instead of a URL, so grep for the URL shape rather than `tail -1`
   # (which would capture the error text as a bogus GIST_URL).
   GIST_URL=$(printf '%s\n' "$GIST_RAW" | grep -oE 'https://gist\.github\.com/[^[:space:]]+' | tail -1)
   if [ -z "$GIST_URL" ]; then gist_err=$(printf '%s\n' "$GIST_RAW" | tail -1); fi
   ```
   `gh gist create` defaults to a **secret** (unlisted) gist when the
   `--public` flag is absent (verified against `gh gist create --help`:
   *"By default, gists are secret; use `--public` to make publicly
   listed ones."*). **Fail-soft behavior** — if `gh` lacks the `gist`
   scope, is offline, or returns a non-URL on stderr/stdout, the grep
   above leaves `GIST_URL` empty and captures the error as `gist_err`;
   continue with the empty-`GIST_URL` path below. Do NOT
   block the step or the park on a missing gist; the committed repo
   doc is the durable artifact and the next step links to it either
   way.
   Keep the `[ -z "$GIST_URL" ]` capture in the if-form shown above — a
   trailing `[ -z "$GIST_URL" ] && gist_err=...` one-liner variant makes the
   CALL report Exit 1 on SUCCESS (URL present -> the test is false -> `&&`
   short-circuits with rc 1; incident #928, 2026-07-04). Same exit-code
   hygiene rule as Step 9c step 1b: never leave a bare conditional or
   informational grep as the last command of a call — if-form it or
   `|| true` it.
7. **Append the link lines to the clean-result body — TWO locations.**
   Use `task.py set-body <N> --file <new-body.md>` (NO
   `--snapshot` — the previous body is already the canonical
   clean-result; this is a two-line append, not a promotion).
   Read the current body and SHA-pin both blob URLs with the `DOC_SHA`
   captured in step 5 — the step-8 verifier's URL-permanence check
   FAILs any unpinned `/blob/main/` GitHub link.

   **Idempotency (same-pass re-entry):** a crashed-and-resumed late
   join can re-run this step after the body was already edited but
   before the `epm:methodology-doc-generated` marker posted (the
   marker lands only at step 9). Before inserting either line, check
   the current body for an existing `**Methodology:**` top line /
   `**Methodology reference:**` Reproducibility row; when one is
   present, UPDATE that line's `<DOC_SHA>` pin and `· [gist](...)`
   suffix in place — never append a duplicate (mirrors the
   EXTEND-pass step-7 delta above).

   Compose both edits in a staged copy (`/tmp/...`) and apply ONLY via
   `task.py set-body <N> --file ...` (named again below) — NEVER a raw
   `body.md` write (incident #1090, 2026-07-07: a direct pathlib write
   bypassed task.py and the revert attempt was hook-blocked; recovery
   cost ~3 turns).

   (a) **Top of body — the reader-facing pointer.** Insert exactly
   this line immediately AFTER the clean-result sentinel (i.e. right
   under the H1 title), BEFORE the first content H2, with a blank line on
   each side. Branch on the sentinel:
   - **v3 body** (`<!-- clean-result-v3 -->`): insert after that
     sentinel, BEFORE `## Takeaways`.
   - **In-flight v2 body** (`<!-- clean-result-v2 -->`): insert after
     that sentinel, BEFORE `## Human TL;DR`.
   - **Legacy body** (no sentinel): directly under the H1 title line.
   ```
   **Methodology:** [docs/methodology/issue_<N>.md](https://github.com/superkaiba/explore-persona-space/blob/<DOC_SHA>/docs/methodology/issue_<N>.md) · [gist](<GIST_URL>)
   ```

   (b) **`## Reproducibility` — the artifact-index row.** Locate the
   `## Reproducibility` H2, add exactly this line under the existing
   bullet list (between the `**Artifacts:**` and `**Compute:**` rows,
   or at the end of the section's bullet list if those anchors aren't
   present):
   ```
   - **Methodology reference:** [docs/methodology/issue_<N>.md](https://github.com/superkaiba/explore-persona-space/blob/<DOC_SHA>/docs/methodology/issue_<N>.md) · [gist](<GIST_URL>)
   ```

   When `GIST_URL` is empty (fail-soft path), drop the `· [gist](...)`
   suffix entirely from BOTH lines:
   ```
   **Methodology:** [docs/methodology/issue_<N>.md](https://github.com/superkaiba/explore-persona-space/blob/<DOC_SHA>/docs/methodology/issue_<N>.md)
   ```
   ```
   - **Methodology reference:** [docs/methodology/issue_<N>.md](https://github.com/superkaiba/explore-persona-space/blob/<DOC_SHA>/docs/methodology/issue_<N>.md)
   ```
   Write the revised body via `task.py set-body <N> --file ...`.
   (Body-shape spec for the top line:
   `.claude/skills/clean-results/SPEC.md` § Top-of-body methodology
   link. Forward-only: never retro-edit bodies finalized before this
   rule existed except via the EXTEND-pass re-pin above.)
8. **Re-run the mechanical verifier on the body.** The two-line link
   addition cannot break the spec (the verifier permits the top-of-body
   `**Methodology:**` line and the Reproducibility row), but the
   verifier costs ~1s and catches the unlikely off-anchor edit:
   ```bash
   uv run python "$REPO_ROOT"/scripts/verify_task_body.py --issue <N>  # main-checkout copy, never the worktree's (spec-stale risk, incident #496)
   ```
   Do NOT re-run the full clean-result-critic loop — this is a
   mechanical post-script edit, not a substantive body change.
   On verifier FAIL, post `epm:failure v1` with
   `failure_class: code`, `reason: methodology-link-append broke
   verify_task_body.py`, set `status:blocked`, and exit (this is a
   workflow bug — surface a `workflow-fix-candidate v1` block in the
   exit text so the orchestrator can AUTO-FILE a `kind: infra` task +
   spawn a background `/issue --auto` session per the
   workflow-fix-on-bug protocol).
9. **Post the marker:**
   ```bash
   uv run python scripts/task.py post-marker <N> epm:methodology-doc-generated \
     --note "doc_path=docs/methodology/issue_<N>.md commit=<DOC_SHA> gist_url=<GIST_URL or 'n/a — <gist_err>'>"
   ```
   When the step was skipped (kind: infra/batch/survey, or an
   analysis task with no methodology surface that the agent stubbed),
   include `note=skipped: kind: <X> has no methodology surface` (or
   the analyzer-stub equivalent) instead of a real `commit=` /
   `gist_url=`.

**Then proceed to 9b (final reviewer step — retired; flips to
`awaiting_promotion`).**

**9b. Final reviewer step — RETIRED (2026-05-13).**

The dedicated `reviewer` / `codex-reviewer` ensemble was deprecated when
its statistical-framing responsibilities were absorbed into
`clean-result-critic` Lens 7 (see CLAUDE.md ontology table; under the v2
spec Lens 11 is "raw alongside processed"). The
`reviewing` status now exists ONLY as the single-step parking point
between clean-result-critic PASS and `awaiting_promotion`. The skill
moves through it in one transition with no agent dispatch:

```bash
uv run python scripts/task.py set-status <N> awaiting_promotion \
  --note "clean-result-critic PASS; parking for user promotion."
uv run python scripts/task.py post-marker <N> epm:status-changed \
  --note "reviewing -> awaiting_promotion (no final reviewer step; absorbed into clean-result-critic Lens 7)"
```

**Run CRON-TEARDOWN now.** `awaiting_promotion` is the terminal/park
transition for an experiment: the pod was terminated at Step 8 and this is
a human gate, so there is nothing left to auto-drive. Fresh `CronList` →
`CronDelete` every job in the two-leg match set (the recurring
`/issue-tick <N>` tick + stray one-shot `/issue <N>` wakeups;
not-found = success; § CRON-TEARDOWN procedure) so the backstop
that deliberately survived the post-`done` stages stops re-firing now. (A
later user re-invocation at `awaiting_promotion` does not re-arm — Step 6d.2
arms only for pod-backed runs reaching the polling loop.)

**Fire `PushNotification` to the phone.** The user is the only actor who
can advance an `awaiting_promotion` task (via `task.py promote <N>
useful|not-useful`), so alert them now:

```python
PushNotification({
    "message": f"#{N} {slug} · clean-result ready — open to promote"[:200],
    "status": "proactive",
})
```

Soft-fail: swallow exceptions (Remote Control disconnected, schema not
loaded). The chat-side prompt below remains the durable record.

**Auto-merge the worktree now (experiments).** The instant the task
lands at `awaiting_promotion`, run the **Step 10d auto-merge procedure**
(rebase-merge `issue-<N>` -> `main`, no prompt, keep the worktree).
Execute it with the Step 10d command blocks VERBATIM — bare,
exit-code-checked push/merge, never piped through `tail`/`grep`/`head`
(Step 10d § "Bare push / merge snippets"; the `guard_piped_git_push.sh`
hook blocks piped variants). The
code / figures / `eval_results` the run produced land on `main`
immediately so the next experiment inheriting from `main` gets any
shared-infra fix this branch carried (this is the #456 -> #466 fix). The
science verdict (`useful` / `not-useful`) is orthogonal and still parks
below for the user. Merging does NOT block the park: an auto-merge
conflict posts `epm:merge-failed v1` and surfaces one line in chat, but
the task still parks at `awaiting_promotion` for promotion. Idempotent —
skip if `epm:merged` already exists.

**Cheap follow-up auto-run (BOTH interactive and autonomous — fires
here, after auto-merge, before the autonomous-only block below).**
Standing directive (2026-06-13, raised 5→20 GPU-h 2026-06-24): *a follow-up that is `0` GPU-h or
`< 20` GPU-h just runs and folds into the same issue, automatically, in
either session mode.* The 0-GPU floor is handled inline at Step 9a-ter
(free-analysis); this block handles the GPU-backed cheap band
(`0 < est_gpu_hours < 20`). It applies to `question_relation: same`
proposals ONLY — a `substantially-different` follow-up changes the
parent `## Goal`, so by the project's routing law it cannot fold into
this issue and is NEVER auto-run here regardless of GPU cost (it stays
filed as a `proposed` child via the autonomous-only block below, or
surfaces at interactive Step 10b for manual triage).

**Follow-up value-critique (redundancy screen) — MANDATORY before ANY
proposal routes.** The instant an `epm:follow-ups v1` marker exists for
this park (posted by C0 below, the autonomous-only block, or interactive
Step 10b), and BEFORE any proposal is routed to the cheap-band auto-run,
the autonomous same-issue loop, the autonomous child-filing path, or the
interactive pick, the orchestrator runs the **follow-up value-critique
ensemble** ONCE over the whole proposal set. This is the 5th doubled
review site (workflow.yaml § ensemble_review.doubled_steps[follow-up-critic],
`single_pass: true`) — it screens for REDUNDANCY only (NOT info-gain /
worth) and NOTHING is dropped: every proposal is saved with a rationale
either way. The subroutine (call it **VC** — invoked from C0a below, the
autonomous block step 2-bis, and Step 10b):

> **VC. Run the value-critique ensemble (single pass — no revise loop).**
> 1. **Idempotency.** If an `epm:followup-value-critique v1` marker
>    already exists for THIS proposal set (match by the `epm:follow-ups
>    v1` it screened — same park), SKIP — reuse the existing merged
>    verdict (this is a no-op on a backstop-tick / re-entry). Otherwise:
> 2. **Spawn the ensemble** in ONE message (two `Agent` calls, staggered
>    a few seconds per the CLAUDE.md 429 guidance): the Claude
>    `follow-up-critic` AND the `codex-follow-up-critic` prompt-composer.
>    Write the `epm:follow-ups v1` body to a temp file and pass its PATH
>    as `proposals_marker_path` (never inline the proposals), plus
>    `experiment_number`, `parent_goal` (the task `## Goal`), and any
>    `prior_value_critique_summaries`. Dispatch the Codex twin's composed
>    prompt as bg Bash via `scripts/codex_task.py` exactly like the other
>    four twin sites (CLAUDE.md § "Codex ensemble review"); the twin agent
>    NEVER dispatches Codex itself (orphan-job anti-pattern, #533). Post
>    `epm:followup-value-critique v1` (Claude) + `epm:followup-value-critique-codex`
>    (Codex) on this task's `events.jsonl`. Quota-sentinel pre-check
>    first (#1204, CLAUDE.md § Codex ensemble review): when LIVE, spawn
>    only the Claude `follow-up-critic`; the merge in step 3 proceeds
>    Claude-only per the existing no-show contract, + one `epm:progress`
>    note.
> 3. **Merge the verdicts PER PROPOSAL** (single pass — no round loop;
>    `single_pass: true`). For each proposal: both `not-redundant` →
>    `not-redundant`. Both `redundant` → `redundant` (the merged
>    rationale unions both critics' duplicate pointers). `not-redundant`
>    vs `redundant` disagreement → spawn the `reconciler` (marker mode,
>    `Role under adjudication: follow-up-critic`, binding binary
>    `not-redundant | redundant`; it posts the canonical
>    `epm:review-reconcile` marker). A Codex twin no-show falls back to
>    the single-Claude `follow-up-critic` verdict (workflow.yaml §
>    ensemble_review; no-show confirmed per the Step 5b
>    durable-verdict-first rule — check `epm:followup-value-critique-codex`
>    + the round-fresh output file before declaring it). An UNCITED
>    `redundant` verdict (no concrete
>    duplicate named) is non-binding — treat it as `not-redundant` for
>    that proposal (cite-or-drop, mirrors the reconciler's ungrounded-
>    blocker rule).
> 4. **Act on the merged verdict, per proposal:**
>    - **`not-redundant`** → the proposal proceeds through the EXISTING
>      routing UNCHANGED (the caller's normal selection / partition /
>      pick logic below runs on it). Its rationale (what new info it adds)
>      is carried forward for the dashboard but does not change routing.
>    - **`redundant`** → the proposal does NOT run and is NOT routed.
>      SAVE it as a new task at status `on_hold` (set-aside, revivable via
>      `set-status <M> proposed`, excluded from auto-dispatch) carrying
>      `parent_id: <N>` and a `## Value critique` body section with the
>      verbatim WHY-IT-DUPLICATES rationale + the pointer (the duplicated
>      task / settled open-question anchor / sibling). File it in ONE
>      atomic call that lands the task DIRECTLY at `on_hold` (never a
>      two-step `new` → `set-status on_hold`, which leaves a window where
>      the proposal sits at `proposed` and a concurrent PM auto-dispatch
>      pass could pick it up — the exact outcome VC exists to prevent):
>      `task.py new --status on_hold --parent <N> --kind experiment --goal
>      "<the proposal's Goal>" --title "<proposal title>" --body-file
>      <spec-with-value-critique-section>.md`. Post
>      `epm:followup-parked-redundant v1` on the PARENT (fields per
>      workflow.yaml § markers: `parked_task_id`, `parent`,
>      `proposal_rank`, `title`, `duplicates`, `rationale`). Announce in
>      chat per the "Announce every follow-up/child task in chat" rule:
>      `Parked #<M> '<title>' on_hold (redundant — duplicates <X>; child
>      of #<N>, revivable via set-status <M> proposed)`. NEVER silently
>      drop a `redundant` proposal — `on_hold` is the durable home for
>      "saved but not worth running now."
> 5. **Hand the surviving (`not-redundant`) proposal set back** to the
>    caller. If EVERY proposal screened `redundant`, the caller's
>    selection finds no candidate and falls through exactly as if the
>    proposer had returned none.

The cheap-band flow:

C0. **Idempotency + run the proposer (once per park, shared marker).**
   FIRST: if this park already dispatched a cheap round whose loop is
   in flight or done — i.e. an `epm:followup-scope v1` with
   `source: proposer-9b-cheap` exists for which a matching
   `epm:same-issue-followup-run v1` does NOT yet exist (in flight), OR
   the cheap-band round cap (C2) is already hit — SKIP this block (it is
   a no-op on a backstop-tick / re-entry; the loop or the cap is the
   durable record). Otherwise: if an `epm:follow-ups v1` marker for THIS
   park is not already present (the autonomous block below may have
   posted it, or a re-entry did), spawn `follow-up-proposer` and post
   `epm:follow-ups v1` (same marker both sites share). If it is already
   present, reuse it — do NOT re-run the proposer. (In autonomous mode
   the proposer runs once and both this block and the autonomous block
   below consume the same `epm:follow-ups v1`. The proposer always posts
   its proposal list when it runs; an empty list means it found no
   follow-ups, and C1 then selects nothing.)
C0a. **Run the value-critique (redundancy screen) — subroutine VC above.**
   Before selecting any candidate, run VC over the `epm:follow-ups v1`
   proposal set (idempotent — a re-entry reuses the existing merged
   verdict). VC parks every `redundant` proposal at `on_hold` and hands
   back only the `not-redundant` survivors. C1 below selects from the
   SURVIVORS only — a `redundant` cheap proposal is parked, not auto-run.
   (VC runs once per park and the autonomous block + Step 10b reuse its
   verdict, so this is not a per-block cost.)
C1. **Select the cheap-band candidate.** Among the surviving
   (`not-redundant`) proposals, keep those
   that are ALL of: `question_relation: same`, `auto_run: yes`, and
   carry a parseable `est_gpu_hours` with `0 < est_gpu_hours < 20`
   (strict `< 20`; `est_gpu_hours: 0` is the Step 9a-ter free-analysis
   case, already handled; exactly `20` does NOT qualify). Take the
   TOP-RANKED such proposal.
   - **Fail-safe (missing / unparseable estimate).** A `same` proposal
     whose `est_gpu_hours` is absent or unparseable does NOT auto-run —
     it is left for the user (interactive: surfaces at Step 10b;
     autonomous: routed by the autonomous-only block below as an
     `auto_run`-gated `same` proposal under its own round cap). Mirror
     of the Step 2c plan-cap fail-safe: a missing GPU estimate parks,
     never auto-runs. State the skip reason in one chat line.
   - **`headline_affecting` is NOT consulted** for this band (dropped
     2026-06-13) — a cheap `same` follow-up runs whether or not it moves
     the headline.
C2. **Cheap-band round cap.** At most **2** cheap-band auto-run rounds
   per task, counted by `epm:same-issue-followup-run v1` markers with
   `source: proposer-9b-cheap`. Run markers whose `outcome` begins
   `retroactive-close` do NOT count toward this cap — they record
   bookkeeping closure of a round that already ran (or was superseded),
   not a new auto-run (Step 0 § Stale-label disposition rule). Beyond
   the cap, further cheap `same`
   proposals survive in `epm:follow-ups v1` for manual pick. (This cap
   is INDEPENDENT of the autonomous `auto_run`/expensive-band cap, which
   counts `source: proposer-9b`. The natural breakpoint is the re-park
   at `awaiting_promotion` after each round, where the user sees the
   updated body before any further cheap follow-up fires.) The cap stops
   a chain of cheap follow-ups from auto-running indefinitely.
C3. **Dispatch the round.** If a candidate survives C1+C2, post
   `epm:followup-scope v1` (`source: proposer-9b-cheap`, fields per
   workflow.yaml § markers, carrying the proposal's
   `est_gpu_hours`) and enter the **same-issue follow-up loop** below
   INSTEAD of parking — the task leaves `awaiting_promotion` and
   re-enters at `followups_running` (tag `followup-auto`). Skip the
   PushNotification → chat prompt park flow this round — but FIRST
   re-arm the `/issue-tick` backstop: CRON-TEARDOWN already ran at the
   `awaiting_promotion` transition at the top of this Step 9b, so NO
   cron is armed here, in EITHER session mode (incident #1112,
   2026-07-08: a cheap-band round launched a multi-hour run with no
   tick armed). BEFORE dispatching any loop work, run the Step 6d.2
   ARM-GUARD shape — `CronList` whole-string match, else
   `CronCreate(cron="*/45 * * * *", prompt="/issue-tick <N>",
   recurring=True, durable=False)`, then re-list and assert exactly
   one — per the loop's "Loop liveness backstop"
   below. The plan still passes through the Step 2c plan-approval
   gate inside the loop — an over-cap (`est_gpu_hours` mis-estimated low
   but the realized plan exceeds `EPM_PLAN_AUTOAPPROVE_GPU_HOURS`) plan
   parks IN PLACE at `followups_running` (autonomous) or asks
   (interactive), so the cost cap is the final backstop even if the
   `est_gpu_hours` estimate was wrong.
C4. **No candidate → fall through.** When no cheap-band candidate
   survives C1+C2, this block dispatches nothing: proceed to the
   autonomous-only block (autonomous sessions) or the park flow
   (interactive sessions). The `epm:follow-ups v1` C0 posted persists —
   its proposals (cheap ones beyond the cap, expensive ones, fail-safe
   skips, `auto_run: no`) carry forward for the autonomous block to
   route or for the user to pick at Step 10b post-promotion (the Step
   10b proposer-already-ran short-circuit then reuses this marker).

**Autonomous follow-up auto-spawn (autonomous mode only — fires here
because Step 10b never runs autonomously; handles the EXPENSIVE
`est_gpu_hours >= 20` / no-estimate `auto_run: yes` path, after the
cheap-band block above has had first refusal).** When
`EPM_AUTONOMOUS_SESSION=1`, the parent task parks at
`awaiting_promotion` and Step 10 / 10b never fire on their own
(promotion is ALWAYS human-only). To stop autonomous research from
stalling on every result, the orchestrator fires the follow-up proposer
HERE — after the auto-merge has landed the clean-result on `main` (the
Step 9b CRON-TEARDOWN already ran at the `awaiting_promotion`
transition above) — and routes the `auto_run: yes` proposals by
`question_relation` (QUESTION IDENTITY — one mechanism, three entry
points; the other two are the Step 0 followup-scope dispatch for
chat-requested follow-ups and the interactive Step 10b pick):
`substantially-different` proposals (and untagged ones ONLY from
pre-2026-06-09 legacy markers — a newer untagged proposal trips the
freshness guard in step 3 below) are FILED-ONLY — created as
`proposed` child tasks for manual triage, NEVER auto-spawned as
sessions (no autonomous child sessions, ever, from this path; the
only execution path for an automatic follow-up is the same-issue
loop); `same` proposals are NEVER filed as children — the top-ranked
one runs ON this issue via the same-issue follow-up loop below
(status `followups_running`, tag `followup-auto`). Interactive
sessions SKIP this block entirely (they still hit Step 10b
post-promotion as today, which routes the user's pick by the same
`question_relation`). Idempotent: when an `epm:follow-ups-autospawned v1` marker is
already present on this parent, do NOT re-run the proposer or re-create
children (covers re-invocation / backstop-tick re-entry; filing
twice + duplicate `epm:follow-ups` clutter are the failure modes this
guard avoids) — instead run the lightweight RECONCILE pass (step R
below) which only verifies the listed children exist.
Depth-bounded: the block is skipped entirely once this parent's
`parent_id` chain already has ≥3 auto-filed ancestors (step 0 below),
so the autonomous follow-up filing tree cannot recurse past depth 3.

The autonomous flow:

0. **Depth cap (run FIRST).** Trace this task's `parent_id` chain upward
   and count ancestors that themselves carry an
   `epm:follow-ups-autospawned v1` marker (i.e. were auto-filing origins,
   not merely manually-filed parents). If that count is **≥ 3**, do NOT
   auto-file children: spawn the proposer and post its proposals as
   `epm:follow-ups v1` for the user to pick manually, then post
   `epm:follow-ups-autospawned v1` with `auto_spawn_skipped:
   depth_cap_reached` and an empty `spawned` list (so the idempotency
   guard still trips and the dashboard records why), and continue to the
   park flow. This bounds the autonomous follow-up filing tree to depth
   3 — cheap insurance against unbounded recursive filing if a filed
   child is later run and reaches its own Step 9b.
1. Read the latest `events.jsonl` (fresh, NOT a stale cached view).
   - If `EPM_AUTONOMOUS_SESSION` is unset → skip the block.
   - If `epm:follow-ups-autospawned v1` is ALREADY present → run the
     **RECONCILE pass** (step R) instead of re-running the proposer, then
     continue to park. (With no session spawning there is no
     crash-between-marker-and-spawn window; the residual self-heal is a
     crash between child creation and the marker post, which the
     duplicate-title guard in step 3 covers.)
   - Otherwise → continue to step 2.
2. Spawn `follow-up-proposer` (clean-result is available — it was just
   promoted in-place by the analyzer). Post the proposals to
   `events.jsonl` as `epm:follow-ups v1` (same marker the interactive
   Step 10b would post; sharing the marker means the dashboard +
   downstream readers don't care which site fired the proposer).
2-bis. **Run the value-critique (redundancy screen) — subroutine VC
   above.** Run VC over the `epm:follow-ups v1` set (idempotent — if the
   cheap-band block's C0a already ran it this park, reuse the merged
   verdict). VC parks every `redundant` proposal at `on_hold`
   (`epm:followup-parked-redundant v1`) and hands back only the
   `not-redundant` survivors. Steps 3-6 below PARTITION + route the
   SURVIVORS only — a `redundant` proposal is never filed as a child and
   never enters the same-issue loop; it is saved on_hold for manual
   revival. This screen gates BOTH the child-filing path AND the
   same-issue-loop path.
3. Parse the surviving (`not-redundant`) proposals, keep those with
   `auto_run: yes` in ranked
   order, and PARTITION them by `question_relation`. **The routing
   litmus is the Takeaways test:** *would the result rewrite THIS
   issue's `## Takeaways`?* If yes → `same` (stays on this issue via the
   same-issue follow-up loop, never a child). Changing method, dose,
   panel, seeds, eval surface, prompt bank, or adding a control/baseline
   on the SAME question is ALWAYS `same`. `substantially-different` is
   reserved for work that would change the task's `## Goal` /
   open-questions anchor — a genuinely new question. This bias-toward-
   same-issue litmus is the same one the `follow-up-proposer` applies
   when tagging (`.claude/agents/follow-up-proposer.md` §
   "question_relation tag — criteria") — the partition just consumes its
   tags; when a tag looks miscast against the litmus, treat it like an
   untagged proposal (re-spawn-once below). **Untagged
   proposals — freshness guard:** the legacy fallback (treat an
   untagged proposal as `substantially-different` so nothing in
   flight breaks) applies ONLY when the `epm:follow-ups v1` marker
   carrying the proposals was posted before 2026-06-09 (pre-dating
   the question-identity routing fix). On a newer marker, a missing
   `question_relation` tag is a proposer-contract violation — the
   usual cause is a stale `follow-up-proposer.md` in a long-lived
   session/worktree that predates the fix (incident #533, 2026-06-10:
   a textbook `same` corrective re-run was routed to a child task via
   this fallback). Re-spawn `follow-up-proposer` ONCE, instructing it
   to re-emit the SAME proposals with `question_relation` (and
   `followup_label` for `same`) tags per the criteria in
   `.claude/agents/follow-up-proposer.md` § "question_relation tag —
   criteria", read from the CURRENT `main` checkout (repo root), not
   the session worktree's possibly-stale copy; the re-emit posts a
   fresh `epm:follow-ups v1` marker that supersedes the untagged one.
   If the re-emit is STILL untagged, route the affected proposals as
   `substantially-different` and record the violation in the
   `epm:follow-ups-autospawned v1` marker body
   (`proposer_contract_violation: question_relation missing after
   re-spawn`). Proposals tagged `auto_run: no` are skipped in BOTH
   partitions — they survive in the `epm:follow-ups v1` marker for
   the user to pick from manually.
   - **`substantially-different`** → the child FILING path (steps
     4-5 below). Take the top **2** (cap; bounds fan-out so a parent
     never files more than 2 children per round regardless of how
     many `auto_run: yes` proposals the proposer found). Drop any kept
     proposal whose title duplicates an existing `parent_id=<N>` child
     (guards against a partial prior run that created the task before
     crashing).
   - **`same`** → the same-issue follow-up loop (§ below, via step 6).
     **First EXCLUDE any `same` proposal the cheap-band block above
     already dispatched this park** (its `epm:followup-scope v1` carries
     `source: proposer-9b-cheap` — match by `followup_label` / verbatim
     spec): if the cheap band took a round, this block does NOT also
     dispatch a `same` round in the same park (one same-issue round per
     park). Of the REMAINING `same` + `auto_run: yes` proposals (those
     with `est_gpu_hours >= 20` or a missing estimate — the cheap band
     skipped these), select the TOP-RANKED one ONLY if the autonomous
     EXPENSIVE-band round cap allows (fewer than 2
     `epm:same-issue-followup-run v1` markers with
     `source: proposer-9b` on this task). The rest — and all `same`
     proposals once the cap is hit — survive in `epm:follow-ups v1`
     for manual pick.
4. For each kept `substantially-different` proposal, in rank order, create the child in ONE atomic
   call — `task.py new --goal` writes BOTH the `goal:` frontmatter AND
   the `## Goal` H2 the child's Step 0c gate requires, so there is no
   window where the child exists without a Goal:
   ```bash
   # Shell-quote the title + Goal (proposal text may contain quotes /
   # backticks): use python -c shlex.quote or printf %q, never bare
   # interpolation. The proposal's **Goal:** field (see
   # follow-up-proposer.md output template) supplies the one-sentence Goal.
   CHILD_ID=$(uv run python scripts/task.py new \
     --parent <N> --kind experiment \
     --goal "<one-sentence Goal from the proposal's **Goal:** field>" \
     --title "<proposal title>" \
     --body-file <path-to-pre-filled-spec>.md \
     | grep -oP '#\K\d+')
   ```
5. **Post `epm:follow-ups-autospawned v1` NOW** — after the child tasks
   exist (step 4). The marker NAME is kept for dashboard back-compat;
   its body carries `execution: filed-only` and the `spawned` list now
   has FILED semantics (children created at `proposed`, no sessions —
   see workflow.yaml § markers). It lists every created child (id +
   title + proposal rank) and every `auto_run: no` proposal that was
   skipped (rank + title + auto_run_reason). This is the durable
   idempotency claim: it records the children so a re-entry reconciles
   (step R) rather than re-creating. Announce each filed child in chat
   per the existing rule (Step 10b § "Announce every follow-up/child
   task in chat"): `Filed #<CHILD_ID> '<title>' (child of #<N>,
   status:proposed — awaiting manual triage)`. Do NOT spawn sessions
   for them — a filed child executes only when a human triages it and
   invokes `/issue <CHILD_ID>`.
6. **Branch on the `same` partition.** If step 3 selected a `same`
   proposal, post `epm:followup-scope v1` (`source: proposer-9b`,
   fields per workflow.yaml § markers) and enter the **same-issue
   follow-up loop** below INSTEAD of parking — the task leaves
   `awaiting_promotion` and re-enters the pipeline at
   `followups_running`, so skip the
   PushNotification → chat prompt → CRON-TEARDOWN park flow this
   round (re-arm the `/issue-tick` backstop cron FIRST via the Step
   6d.2 ARM-GUARD shape — the Step 9b CRON-TEARDOWN at the
   `awaiting_promotion` transition already removed it; see the loop's
   "Loop liveness backstop").
   Otherwise continue to the existing park flow below
   (PushNotification → chat prompt → CRON-TEARDOWN → §5 marker via
   `post_step_completed.py --step 9a-bis --exit-kind parked` → EXIT).

**Step R — RECONCILE pass** (re-entry with the marker already present):
read the `spawned` list from `epm:follow-ups-autospawned v1`. For each
listed child, verify it exists via `task.py view <CHILD_ID> --json`;
re-create one that is missing (same atomic `task.py new --parent`
call as step 4). NEVER spawn sessions — this pass only verifies
filing. Then continue to park.

Cost discipline: this block adds NO new cost gate. A filed child, once
a human triages it and runs `/issue <CHILD_ID>`, hits its own Step 2c
`--auto-approve-if-autonomous --gpu-hours` cap; over-cap plans park at
`plan_pending`, consistent with `tests/test_no_dollar_budget_caps.py`.
Promotion of the parent stays human-only. The recursive surface is
bounded twice over: same-issue rounds are capped at 2 per task, and
child FILING is capped at 2 per parent per round AND hard-stopped at
chain depth 3 by step 0 (so even if filed children are later run, the
filing tree is both width-bounded and depth-bounded, not exponential).

**Same-issue follow-up loop (`question_relation: same`).**

One mechanism, four entry points: (a) the Step 9b autonomous
expensive-band partition above (`source: proposer-9b`,
`est_gpu_hours >= 20` / no estimate), (a-cheap) the Step 9b cheap-band
block (`source: proposer-9b-cheap`, `0 < est_gpu_hours < 20`,
`question_relation: same`) which fires in BOTH interactive and
autonomous sessions, (b) a chat-requested
same-question follow-up (`source: user-chat` — the chat session posts
`epm:followup-scope v1` on #N, then re-invokes `/issue <N>`; the Step
0 followup-scope dispatch lands here), and (c) an interactive Step
10b pick (`source: step-10b-pick`). Step 9a-ter (the inline
free-analysis auto-run) is this loop's zero-GPU sibling under the
same principle — a follow-up that answers the SAME question as the
task Goal runs ON this issue; 9a-ter handles the zero-GPU floor
inline, this loop handles the GPU-backed case (the cheap `< 20` GPU-h
band auto-runs in both modes; the expensive band auto-runs only in
autonomous mode or on an explicit user pick).

**Loop liveness backstop (arm BEFORE dispatching loop work — BOTH session modes).**
ANY session driving this loop — interactive (typically entry point (b),
a chat session) OR autonomous (the Step 9b C3 cheap-band / step-6
expensive-band dispatches, where the `awaiting_promotion` CRON-TEARDOWN
has already removed the Step 0 arm) — must verify/arm the
`/issue-tick <N>` backstop cron (same `CronList`/`CronCreate` ARM-GUARD
shape as Step 0 / Step 6d.2; a no-op when already armed) before
dispatching its first planner / implementer / stage subagent. While
loop work waits on any long phase, the § Long-phase heartbeat duty
(Step 6d.2) binds. An INTERACTIVE session must additionally post every
stage-dispatch breadcrumb
(`stage=followup-<phase>`, Step 9 entry-guard convention) with the
`worktree=` field **and a `label=<followup_label>` field naming the
round's label** (consumed by `task_workflow.executing_followup_label`
for mid-round resume and by the watcher's on-behalf run-marker post;
breadcrumbs predating this contract lack it — consumers fall back to
the head of the unrun queue). These `stage=followup-<phase>` breadcrumbs are bound
by the SAME Step 9 entry-guard predicate AND the same NON-SKIPPABLE
pre-dispatch dedup check (§ Step 9 entry guard) — the status being held
at `followups_running` does not exempt them (#778, 2026-07-01: the
round-1 `followup-implementing` dispatch was duplicated by a concurrent
orchestrator 5m39s after the first, two implementers concurrently
editing one worktree). Know what each mechanism covers: the cron handles
only the alive-but-stalled case — a `durable=False` cron dies with the
session that armed it; `autonomous_session_watch.py`'s AUTO-RESPAWN
passes read only the autonomous registry (`spawn-issue --auto`
entries), and the step-2 `register-current` manual registration buys
ALERT-ONLY stalled/crash visibility (a user-driven session is never
auto-respawned, #505) — so nothing external RE-DRIVES an interactive
session driving this loop. If the session is going to be
closed — or the user asks for a handoff — while loop work is in flight,
the mid-flight handoff rule (§ Orchestration Procedure preamble)
applies: spawn `spawn_session.py spawn-issue --issue <N> --auto`
IMMEDIATELY; that registration is the only mechanism that survives
session death. (Incident #505 round 2, 2026-06-10: an interactive chat
session driving this loop was closed mid-implementer-dispatch with no
cron armed, no registry entry, and no worktree breadcrumb; the task
orphaned at `running` for 5+ hours.)

**Loop-entry ownership re-check.** When entering this loop from a
resume / re-invocation (including the Step 0 followup-scope dispatch),
FIRST re-run the Step 0 single-orchestrator guard: if another live
session is mapped to this issue, stop the stale session
(`spawn_session.py stop`) before dispatching any loop work — two live
orchestrators driving one round is the #778 root cause.

1. **Scope marker.** Ensure an `epm:followup-scope v1` exists for this
   round (the Step 9b partition posts it at step 6 above; the chat /
   Step 10b entry points post it before re-invoking). Fields per
   workflow.yaml § markers: `followup_label` (kebab-slug; names the
   artifact dir `eval_results/issue_<N>/<followup_label>/`), `source`,
   the verbatim proposal spec (or the user's verbatim chat request),
   and the GPU-hour estimate. **MULTIPLE `epm:followup-scope` versions
   for one issue may exist** — corrections are WITHIN-label: the
   authoritative scope for a round is the latest-(`ts`, `version`)
   entry AMONG the entries carrying THIS ROUND'S `followup_label` (an
   unlabeled correction note attributes to the immediately-preceding
   label — `task_workflow.followup_label_groups`; see #658's
   `persona-vectors-style-rb` v3→v7 chain — NOTE v8 carries a DIFFERENT
   label, `a35-mlp-downstream-chain`, a separately-queued round, not
   part of the chain). A later entry with a DIFFERENT label is a
   separately-QUEUED round (the #763 shape), never a supersession. Do
   NOT cache the
   entry-time version — step 3 re-reads the latest before snapshotting.
2. **Re-enter the pipeline.** **FIRST record the initiation mode as a
   tag** (before the status flip, so the `task.py` missing-tag warning
   stays quiet): `uv run python scripts/task.py add-tag <N>
   followup-auto` when `source: proposer-9b` OR `source:
   proposer-9b-cheap` (both are proposer-initiated auto-runs);
   `uv run python scripts/task.py add-tag <N> followup-manual` when
   `source: user-chat` or `source: step-10b-pick`. EXACTLY these two tag
   names — a bare `followup` tag does not count (incident #533). (Both
   tags may accumulate over a task's life — they are history, not
   exclusive
   state.) **Then** `task.py set-status <N> followups_running` — the
   round HOLDS this status end-to-end (see the status-hold rule in step
   3); the CLI warns if neither tag is present at this transition. The
   planner-exempt distinction (re-run with different seeds,
   monitoring, syncing, or a bug-fix re-run, per the CLAUDE.md
   `/adversarial-planner` carve-out) still governs whether
   `/adversarial-planner` is re-invoked in step 3 — the STATUS no
   longer encodes it. The marker trail
   records the transition (`epm:status-changed`); `has_clean_result`
   stays sticky across the re-entry. **In the same step, re-register
   the driving session:** `uv run python scripts/spawn_session.py
   register-current --issue <N>` (infers this session's Happy id from
   the process ancestry + the daemon; writes `issue-<N>.json` for
   autonomous sessions / `manual-issue-<N>.json` for interactive ones,
   matching how the session was spawned). The revival flips a
   parked/terminal task back to ACTIVE, but the watcher's registry
   entry was DELETED at the terminal transition — without
   re-registering, the revived run is invisible to every
   registration-based watcher pass until the orphan sweep's ~90-min
   staleness gate (incident #472, 2026-06-10: a revival ran orphaned
   for 10.5h). Registration failure is non-fatal to the loop (the
   orphan sweep remains the backstop) but state the failure rather
   than swallowing it.
3. **Abbreviated cycle**, all on THIS issue. **Status-hold rule: the
   task STAYS at `followups_running` for the WHOLE round** — planner
   amendment → consistency-checker → plan gate → implementer /
   code-review → provision → run → upload-verify → terminate →
   analyzer re-fold → clean-result-critic. The normal pipeline
   `set-status` calls (`planning` / `plan_pending` / `approved` /
   `running` / `verifying` / `interpreting` / `reviewing`) are SKIPPED
   during a same-issue follow-up round; phase visibility comes from the
   existing stage breadcrumbs (`stage=followup-<phase>`) and
   `epm:progress` markers. **Code-enforced** (post-#533/#560,
   2026-06-11): `task.py set-status` REFUSES
   `followups_running -> <any of those>` (override:
   `--force-followup-exit`, only to deliberately abandon the round), and
   a mid-round plan-gate call (`--auto-approve-if-autonomous`) fires the
   gate decision + markers while HOLDING the status
   (`PLAN_GATE_DECISION: ... (followups_running hold: status
   unchanged)`). An over-cap (or interactively-awaiting) plan parks IN
   PLACE at `followups_running` — the Step 2c plan-approval gate still
   fires, it just no longer moves the status to `plan_pending`. The
   round exits the status only at the re-park:
   `set-status <N> awaiting_promotion` (or `blocked` on a failure
   exit). **Mid-round defer/teardown is an exit too — re-park in the
   SAME action sequence as the teardown:** a mid-round defer (wedged or
   pathological run torn down, user defer — any deliberate abandonment
   of the round short of a `blocked` failure exit; no
   `--force-followup-exit` needed, `awaiting_promotion` is not in the
   refused set) tears down the pod/instance FIRST, runs
   `set-status <N> awaiting_promotion` as the NEXT command (the § User
   pause affordance teardown-first-park-last ordering — distinct
   mechanism: a user pause parks at `on_hold`, this defer exit re-parks
   at `awaiting_promotion` with the label closed; never leave
   `followups_running` with no live round compute), THEN closes the
   round's label by posting the step-4 completion marker with
   `outcome: deferred — <one-line reason>` (label closure is
   outcome-agnostic — `task_workflow.unrun_followup_labels` — so Step 0
   / the tick never auto-re-dispatch the deferred round; a deliberate
   later resume posts a FRESH scope under a NEW label; a deferred
   proposer-band round still counts toward its step-5 cap). The tick
   STALE-REDRIVE / watcher re-park are recovery backstops, not the
   owner (incident #825, 2026-07-15: a pathological fit was torn down
   at 00:28Z with no re-park — the parent stranded at
   `followups_running` ~1.4 h until the 01:53Z tick re-drive re-parked
   it).
   - **Immediately before the planner snapshots the scope, RE-READ the
     authoritative scope FOR THIS ROUND'S LABEL** —
     `task_workflow.executing_followup_label` (the newest
     `stage-dispatch stage=followup-*` breadcrumb's `label=` field newer
     than the newest run marker; fallback: the head of the dispatchable
     `unrun_followup_labels`) — never the bare latest scope: under
     label-grouped dispatch (#894) the latest entry may be a DIFFERENT
     queued label. A mid-round correction to the SAME label is still
     picked up (it raises that label's authoritative `(ts, version)`);
     never plan against an entry-time snapshot, or a session that
     entered on `v3` and snapshotted before a `v5`/`v6` correction
     landed plans stale (the #658 bug).
     The same pre-snapshot re-read covers the canonical GOAL:
     `frontmatter.goal` + the latest `epm:goal-updated` ts — the
     adversarial-planner § Goal-currency gate applies in AMENDMENT scope too
     (an amendment plan drafted against a stale Goal is the #922 bug class,
     the Goal sibling of this bullet's #658 scope bug). The `followup_label` lives
     inside the marker NOTE body as free text (its format even differs
     across versions — `- followup_label: ...` dash-bullet, bare
     `followup_label: ...`, bold `**followup_label:** ...`, `; `-joined
     single-line `source: ...; followup_label: ...` (#1090/#841)), NOT as
     a top-level event key (top-level keys are `{by, kind, note, ts,
     version}` only) — `task_workflow.parse_followup_note_field` handles
     every observed form. The step-4 completion marker's
     `followup_label` derives from THIS SAME executing group
     (`group['followup_label']` verbatim) — never re-parsed from "the
     scope marker" independently, so the completion label can never
     diverge from the round that ran. This is the SAME shared helper the
     watcher uses (`autonomous_session_watch._post_followup_run_marker`
     resolves the round via `task_workflow.executing_followup_label`).
     Mechanical recipe:
     ```bash
     uv run python -c "
     import json
     from explore_persona_space.task_workflow import list_events, executing_followup_label
     g = executing_followup_label(list_events(<N>))
     print(json.dumps(g and g['authoritative'], indent=2))
     "
     ```
   - `/adversarial-planner` re-invoked in AMENDMENT scope: produces
     `plans/v{N+1}.md` as a ONE-VARIABLE diff plan against the issue's
     own latest prior run, not a from-scratch plan. Planner-exempt
     re-runs (step 2) skip this.
   - **Compute-character pre-launch statement** (canonical block: Step
     9a-ter § Compute-character pre-launch statement — same four elements,
     same > ~1h stop-and-vectorize + ≥~16 GB-RSS off-VM rules): REQUIRED in the
     `stage=followup-<phase>` dispatch breadcrumb (or an adjacent
     `epm:progress` note) before dispatching ANY stage of the round that
     launches a fit, sweep, or statistical battery — INCLUDING
     planner-exempt re-runs (step 2), which skip the amendment plan and
     its §9 sizing entirely, and analysis / re-fold stages reusing parent
     code. An amendment plan's §9 sizing does NOT substitute for it: the
     plan schedules the battery, the executor states the implementation's
     compute shape — #778's round re-ran the parent's serial 1000-draw
     null battery (2+h, projected 4–6h, vs ~15–30 min batched) under a
     plan that never said "serial".
   - **Pre-dispatch external-marker triage** (canonical block: Step 9
     entry guard § Pre-dispatch external-marker triage): REQUIRED — the
     same `stage=followup-<phase>` dispatch breadcrumb (or adjacent
     `epm:progress` note) carries the `external-markers triaged:` line
     before dispatching ANY compute-launching stage of the round. #779's
     `stage=followup-grid` dispatch (2026-07-02 20:46Z) is the founding
     incident — 10 unread external audit markers, an 18–20h serial
     launch.
   - `consistency-checker` diffs the amendment against the ISSUE'S OWN
     latest prior run — the latest prior plan version + the current
     clean-result body's `## Reproducibility` — NOT a `parent_id` task
     (see consistency-checker.md § Same-issue follow-ups).
   - Step 2c plan-approval gate as normal — the EXISTING
     `gates.inline plan_approval` gate, no new gate is registered:
     autonomous sessions auto-approve under
     `EPM_PLAN_AUTOAPPROVE_GPU_HOURS` and park at `plan_pending` over
     the cap; interactive sessions ask.
   - `experiment-implementer` + `code-reviewer` if the diff needs code
     changes (same ensemble shape as Step 5). The round's implementer brief
     follows the Step 4b brief contract INCLUDING its marker-version-
     discipline bullet — on a follow-up round prior
     `epm:experiment-implementation` rows ALWAYS exist, so a brief
     instructing a literal `v1` reproduces the #825 collision; the brief
     defers to max+1 (or tells the implementer to omit `--version`).
   - Fresh compute dispatch on the SAME issue, through the slice-6
     router exactly like the parent run: read the task's `backend:`
     frontmatter and run `dispatch_issue.py launch --issue <N>
     --intent "$INTENT" ${BACKEND:+--backend "$BACKEND"}` (see Step
     6b § "Operational dispatch (slice-6 router, ALL backends)" — do
     not duplicate its prose here). Follow-up rounds inherit the
     task's `backend:` frontmatter and the auto-routing default
     (empty → auto, GCP-first standing default, then the free
     clusters; RunPod only on an explicit `backend: runpod`). The prior compute was torn down at Step 8;
     per-issue naming already supports re-dispatch.
   - Run → upload-verify → Step 8 terminate, as normal.
   - The `analyzer` RE-FOLDS the new finding into the EXISTING
     clean-result body — a new `### <finding>` H3 under `## Findings`
     (the v3 `## Findings` section already supports multiple `### <finding>`
     H3s), updating the H1 title / confidence tag if the result moves the
     headline. The
     `set-body` call passes NO `--snapshot` — `original-body.md`
     already preserves the pre-promotion original — and a moved headline
     is followed by `task.py set-title <N> "<new H1 text>"` (set-body
     preserves the old frontmatter `title`; the H1==frontmatter verifier
     check FAILs the 9a-bis re-gate otherwise; see analyzer.md §
     Same-issue follow-up re-entry).
   - `clean-result-critic` re-gates the UPDATED body (9a-bis as
     normal), then 9a-quater and the `awaiting_promotion` park run as
     normal — on this re-entry, 9a-quater's followup-scoped idempotency
     detects the round's unrecorded `followup_label` and runs its
     EXTEND pass (methodology-writer in EXTEND mode appends the new
     arm's section to `docs/methodology/issue_<N>.md`, refreshes the
     gist, re-pins the body's Methodology-reference link) instead of
     the parent-pass no-op. Planner-exempt re-runs take the
     no-new-methodology carve-out there.
   - Re-park at `awaiting_promotion`. ONE promotion verdict covers the
     whole updated body; a previously-promoted (`completed`) task that
     looped re-parks here and the user re-promotes.
4. **Completion marker.** Post `epm:same-issue-followup-run v1`
   (`followup_label` matching the scope marker — derive the label from
   the executing group per step 3's re-read, never a fresh independent
   parse — `source`, `round`,
   one-line `outcome`) when the loop re-reaches `awaiting_promotion`.
   The note MUST carry the `followup_label:` / `source:` / `round:` /
   `outcome:` fields field-led — line-initial one-per-line, or
   `; `-joined on one line (both parse); a PROSE-LED note parses no
   label, closes nothing, and undercounts both round caps (the #1090
   fu1 regression).
   This is the idempotency record: an `epm:followup-scope v1` with a
   matching run marker is RUN and is never re-dispatched.
5. **Round caps (two independent proposer-initiated caps).**
   - **Expensive autonomous band:** at most **2** rounds per task,
     counted by `epm:same-issue-followup-run v1` markers with
     `source: proposer-9b` (the `est_gpu_hours >= 20` / no-estimate
     autonomous-only path).
   - **Cheap band (both modes):** at most **2** rounds per task, counted
     by `epm:same-issue-followup-run v1` markers with
     `source: proposer-9b-cheap` (the `0 < est_gpu_hours < 20` path,
     enforced at block C2 above). This cap is what stops a chain of
     cheap follow-ups from auto-running indefinitely; the re-park at
     `awaiting_promotion` after each round is the user-visible
     breakpoint.

   Beyond either cap, further `same` proposals of that class survive in
   `epm:follow-ups v1` for manual pick. USER-REQUESTED rounds
   (`source: user-chat` or `step-10b-pick`) do NOT count against either
   cap — the user asked explicitly, and interactive plan approval
   still gates each one. Run markers whose `outcome` begins
   `retroactive-close` do NOT count toward either round cap — they
   record bookkeeping closure of a round that already ran (or was
   superseded), not a new auto-run (Step 0 § Stale-label disposition
   rule).

Status-machine summary: `interpreting` / `reviewing` /
`awaiting_promotion` / `completed` + ≥1 unrun followup label →
`followups_running` (tag `followup-auto` | `followup-manual`; held
for the whole round) → `awaiting_promotion`. Never a child task.
(`followups_running` also retains its legacy meaning — parent
complete, `parent_id` children still in flight — see Step 10 step 5.)

Then post the chat-side prompt:

> Clean-result-critic PASS. The polished body is now live on task #\<N\>.
> When satisfied, promote it (USER-ONLY — no automation may do this):
>   `uv run python scripts/task.py promote <N> useful`     (paper-relevant)
>   `uv run python scripts/task.py promote <N> not-useful` (archive candidate)
> Then re-enter `/issue <N>` to fire Step 10.

> **Re-park BEFORE the §5 marker (same-issue follow-up rounds — incident
> #533, 2026-06-11):** during a follow-up round, post the §5 marker below
> ONLY after the round's re-park has actually executed — check `task.py
> view <N> --json` shows `status: awaiting_promotion` first. If the
> status is still `followups_running`, the re-park was skipped: run step
> 3's `set-status <N> awaiting_promotion` + step 4's
> `epm:same-issue-followup-run v1` completion marker NOW, then post the
> marker. Posting the exit-site marker while still at `followups_running`
> and exiting is the #533 freeze shape — the session died there and the
> task stranded for ~26h. (`autonomous_session_watch.py` now backstops
> this with a round-complete auto re-park, but the backstop is recovery,
> not the design.)

Post the §5 marker (the EXIT site is the tail of step `9a-bis`; the
candidate landing step on resume is `10` (`completion_audit`), looked up
from `workflow.yaml § steps`):
```bash
uv run python scripts/post_step_completed.py --issue <N> --step 9a-bis \
  --exit-kind parked --notes "awaiting clean-result promotion"
```
EXIT. The user reviews the clean-result at their own pace and manually
picks a verdict. **Awaiting promotion is a user-only state — no agent
or automation may move a task out of it.** The `task.py promote`
command refuses if `classification != 'pending'`.

**On re-invocation at `awaiting_promotion`:**

1. Check the `classification` field in `body.md` frontmatter (set by
   `task.py promote`).
2. If `classification != 'pending'` -> advance to Step 10 (auto-complete).
3. If `classification == 'pending'` -> show the task path, post the §5
   marker:
   ```bash
   uv run python scripts/post_step_completed.py --issue <N> --step 10 \
     --exit-kind parked --notes "clean-result classification still pending; awaiting promotion"
   ```
   and EXIT. User hasn't promoted yet.

**9c. Test-verdict gate (code-change paths only, inline)**

Only for `infra` / `batch` / `analysis` / `survey` tasks — these arrive
here directly from Step 5 PASS, having skipped Steps 6-8 (no pod, no
interpretation). The code-review gate has already approved the diff;
this step verifies the test suite still passes.

There is **no `tester` agent**. The skill itself runs the project's test
suite directly and posts an `epm:test-verdict` event with the result.

1. Unit tests — DEFAULT scope `touched` (workflow-invariant + touched-file
   subset). The full ~5800-test suite has no xdist parallelism and is
   harness-/earlyoom-killed in sparse worktrees (#665/#736), so do NOT run
   `pytest tests/` wholesale by default.
   a. Compute the subset FROM THE ISSUE WORKTREE — a branch-new test file
      exists ONLY there until the Step 10d merge, and the helper diffs the
      INVOKING checkout (incident #851: run from the main repo root it saw an
      empty diff and silently dropped the branch's own test files from the
      gate; the helper now emits a stderr `NOTE — empty diff` in that shape —
      on a worktree-based task whose branch HAS commits ahead of the base,
      that NOTE means wrong cwd, re-run from the worktree; from a correct
      worktree with no commits ahead of the base it also fires and is then
      expected and benign). The selector's diff base defaults to fetched
      `origin/main` (#1289: the shared root's local `main` lagged origin on
      2026-07-12 and polluted #1281's gate to 41 files with foreign touched
      files; bounded 120 s fetch — a fetch failure degrades to last-fetched
      `origin/main`, an unresolvable `origin/main` falls back loudly to local
      `main`). Pass `--base main` only to deliberately diff against the
      local ref:
      ```bash
      REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
      WT="$REPO_ROOT/.claude/worktrees/issue-<N>"   # same-issue follow-up rounds use
                                                    # their own issue-<N>-<suffix> worktree
      cd "$WT" || { echo "FATAL: cd to issue worktree failed" >&2; exit 1; }
      uv run python scripts/select_step9c_tests.py   # base defaults to FETCHED origin/main (#1289)
      ```
      It prints the exact gate command —
      `timeout --kill-after=60s <T>s uv run pytest <files> -v --tb=short`,
      `<T>` sized deterministically from the selection
      (`recommended_timeout_s()`: 120s base + 30s/file + a 900s surcharge when
      `tests/test_workflow_lint.py` is selected, which alone measured up to
      771s) — plus stderr diagnostics: a one-line work-root + branch
      provenance breadcrumb on every run, a `recommended-timeout-s=<T>`
      sizing line, any `untested touched file: <path>` WARN lines, and the
      empty-diff NOTE described above. (A code-change task with NO worktree
      runs both from the repo root; the empty-diff NOTE is then expected and
      benign.)
   b. Run the printed command as a BACKGROUND Bash invocation
      (`run_in_background=true`) from the SAME worktree cwd (paths are
      repo-relative), with the junit flags + log/rc-file tail appended and a
      pre-run `rm -f` of all three gate files (a killed run must leave NO
      junit — pytest writes it only at session exit; a stale file from a
      prior round must never be re-read). BACKGROUND IS REQUIRED, NOT
      OPTIONAL: the selection always contains the 37-file workflow-invariant
      set incl. `tests/test_workflow_lint.py` (median ~6.5 min alone, max
      ~13 min; whole gate median ~11 min, max ~21 min of test time plus
      collection overhead — 26 junit runs measured 2026-07-04/05), so the
      gate can NEVER fit the 600s foreground Bash tool cap. The
      crash-fix-rounds ~510s foreground `timeout` bound
      (`.claude/rules/crash-fix-rounds.md` § Kill-before-relaunch) applies to
      FOREGROUND smokes ONLY — wrapping this gate in any ≤600s bound is the
      #991/#996/#906 kill class (exit 143 at 480-540s). The ONLY wedge bound
      is the selector-printed `timeout --kill-after=60s <T>s` prefix.
      ```bash
      # Shell state does NOT persist across Bash calls — hard-guard the cd
      # INSIDE this same background call (never rely on a prior call's cwd;
      # a silent cd failure must never run the gate in the wrong dir):
      cd "$WT" || { echo "FATAL: cd to issue worktree failed" >&2; exit 1; }
      # earlyoom-protect the gate (#1045; FAIL-OPEN — never block the gate on a choom failure):
      # pytest is in this VM's earlyoom --prefer regex (+300 badness), so a gate run is the
      # designated victim under fleet memory pressure (#906 killed twice; #995 at ~42%).
      # Self-choom the gate shell: every child forked after this line (the timeout wrapper,
      # pytest + its subprocesses) inherits adj=-600.
      sudo -n choom -n -600 -p $$ >/dev/null 2>&1 && GATE_CHOOM=ok \
        || { GATE_CHOOM=failed; echo "[warn] choom failed — gate pytest is earlyoom-UNPROTECTED (choom=failed)" >&2; }
      echo "[step9c] gate earlyoom protection choom=$GATE_CHOOM"
      rm -f /tmp/step9c-junit-issue-<N>.xml /tmp/step9c-rc-issue-<N> \
            /tmp/step9c-pytest-issue-<N>.log   # MANDATORY before EVERY gate pytest invocation
      # ONE background Bash call (run_in_background=true) — the selector-printed
      # command verbatim, with the junit + log + rc-file tail appended:
      timeout --kill-after=60s <T>s uv run pytest <files> -v --tb=short \
        --junitxml=/tmp/step9c-junit-issue-<N>.xml -o junit_family=xunit1 \
        > /tmp/step9c-pytest-issue-<N>.log 2>&1; echo $? > /tmp/step9c-rc-issue-<N>
      ```
      When the background call completes (the harness notifies), read the
      verdict in a fresh foreground call — the rc FILE replaces the former
      in-shell `PYTEST_RC=$?` capture (shell variables do not survive across
      Bash calls). A MISSING rc file means the background run died before
      pytest exited (tool kill / watcher force-stop, #833): treat as FAIL,
      never a silent PASS, and apply crash-fix-rounds § Kill-before-relaunch
      (probe `pgrep -af 'pytest.*step9c-junit-issue-<N>'` — the junit path
      makes the probe exact-invocation-scoped) before any re-run:
      ```bash
      if [ ! -f /tmp/step9c-rc-issue-<N> ]; then
        echo "FATAL: gate rc file missing — the background run died before pytest exited. Kill-before-relaunch, then re-run the gate; NEVER record PASS." >&2
      else
        PYTEST_RC=$(cat /tmp/step9c-rc-issue-<N>)
        tail -30 /tmp/step9c-pytest-issue-<N>.log
        # exit 0 + "no tests ran" (or "collected 0 items") is NOT a PASS:
        if grep -qiE 'no tests ran|collected 0 items' /tmp/step9c-pytest-issue-<N>.log; then
          echo "FATAL: pytest collected 0 tests — test-verdict gate did NOT run. Treating as FAIL." >&2
          # -> post epm:test-verdict v1 as FAIL; do NOT record PASS on exit 0.
        fi
      fi
      ```
      Record pass/fail + ALL selector stderr lines (the provenance
      breadcrumb, the `recommended-timeout-s=<T>` sizing line, the NOTE if
      any, and any WARN lines). The two anti-silent-pass guards above are
      LOAD-BEARING — a `no tests ran` outcome (pytest exit 0 with zero
      collected tests) is a **FAIL, never a PASS**: it is the signature of a
      failed `cd` that ran pytest in a directory with no tests (incident:
      issue 745, SHA 91bed41e, 2026-06-30 — the gate reported PASS on
      `no tests ran ... pytest exit: 0` and was silently skipped).

      **Recipe exit-code hygiene (every gate call — and every improvised
      monitoring one-liner):** the Bash tool reports the exit code of the
      LAST command in the call, and an `Exit 1` from a trailing INFORMATIONAL
      command is indistinguishable in the transcript from a gate failure. Any
      trailing command that legitimately returns non-zero without meaning
      failure — a display/filter `grep` that may match nothing (#969: a
      healthy gate read as a false Exit 1), a bare `[ -z "$VAR" ]` /
      `[ -s <file> ]` test (#928: a trailing `[ -z "$GIST_URL" ] && ...`
      variant reported Exit 1 on success), a `tail`/`cat` on a possibly-
      absent file — MUST be if-formed (`if grep -q ...; then ...; fi`) or
      given an explicit `|| true`. The verdict-bearing rcs are NEVER the raw
      call exit code: PYTEST_RC lives in `/tmp/step9c-rc-issue-<N>` and
      COMPARE_RC in `/tmp/step9c-compare-issue-<N>.rc` — read those, not
      the tool's Exit line.
   c. Scope override: if the plan-body frontmatter has `test_scope: full` OR a
      `## Test scope` H2 names `full`, run the FULL suite instead — from the
      SAME issue-worktree cwd, in the SAME background + rc-file pattern as 1b
      (a 60m run is 6x the foreground tool cap):
      ```bash
      cd "$WT" || { echo "FATAL: cd to issue worktree failed" >&2; exit 1; }
      # earlyoom-protect the gate (#1045; fail-open — see the 1b preamble): self-choom, children inherit.
      sudo -n choom -n -600 -p $$ >/dev/null 2>&1 && GATE_CHOOM=ok \
        || { GATE_CHOOM=failed; echo "[warn] choom failed — gate pytest is earlyoom-UNPROTECTED (choom=failed)" >&2; }
      echo "[step9c] gate earlyoom protection choom=$GATE_CHOOM"
      rm -f /tmp/step9c-junit-issue-<N>.xml /tmp/step9c-rc-issue-<N> \
            /tmp/step9c-pytest-issue-<N>.log
      # ONE background Bash call (run_in_background=true):
      timeout --kill-after=60s 60m uv run pytest tests/ -q \
        --junitxml=/tmp/step9c-junit-issue-<N>.xml -o junit_family=xunit1 \
        > /tmp/step9c-pytest-issue-<N>.log 2>&1; echo $? > /tmp/step9c-rc-issue-<N>
      ```
      (NO `-x` / `--maxfail` — with the step-1d compare deciding the verdict,
      an early-exit on the first known-red main failure would leave the rest
      of the suite unexecuted and let compare PASS a truncated run; the 60m
      timeout still bounds it.) The rc file is written by the SAME background
      command immediately after pytest exits (1b touched scope and this
      override alike) — step 1d's compare consumes
      `--pytest-rc "$PYTEST_RC"`, re-reading `/tmp/step9c-rc-issue-<N>`
      INSIDE its own background call (shell variables do not survive across
      Bash calls); a missing rc file takes 1b's FAIL path (an
      unset or stale rc would break compare's rc-not-in-{0,1} ->
      indeterminate guard). On timeout/kill (`timeout`'s rc 124 lands in the
      rc file, so compare exits 2), capture
      `tail -50 /tmp/step9c-pytest-issue-<N>.log` so the stall surfaces
      actionable evidence (the #665/#736 regression — keep it visible, never
      a silent kill). Default scope is `touched`.

      **Gate earlyoom protection (#1045).** The self-choom preamble in the
      1b/1c blocks is the same-call sibling of § "Detached VM-side long
      compute phases": `oom_score_adj` inherits across fork/exec
      (probe-verified), so choom-ing the gate shell BEFORE pytest launches
      protects the whole gate tree — the `timeout` wrapper, pytest, and its
      subprocesses inside the background call — with zero change to the
      rc-file/junitxml contract (#1046: the gate is a background invocation;
      `PYTEST_RC` travels via `/tmp/step9c-rc-issue-<N>`, not shell state).
      Step 1d compare — including its pristine single-file oracle runs
      (600–1950s each, #1129) — runs as its OWN background + rc-file call
      with the SAME fail-open self-choom preamble (see step 1d); only the
      1d ledger-refresh kick keeps the post-hoc session-sweep form (it
      launches detached BEFORE a choom can be applied). FAIL-OPEN: a
      choom failure warns, records `choom=failed`, and the gate proceeds
      unprotected — a gate is NEVER blocked by a choom failure, and
      `choom=ok` re-orders earlyoom's victim selection (−600, not −1000: the
      gate stays killable if it is itself the runaway consumer). The Bash
      tool spawns a fresh shell per call, so the adjustment dies with the
      call; no reset needed (in a long-lived manual shell, reset with
      `sudo -n choom -n 0 -p $$`). Calibration: −600 buys victim
      RE-ORDERING, not survival — net ~400 display points below an
      equal-size unprotected python neighbor (the `--prefer` +300 applies
      regardless of adj) but only ~100 below a non-python neighbor, and at
      fleet-wide adoption protected work competes with protected work again;
      say "re-orders victim selection" / "stops being the default designated
      victim", never "prevents kills".
   d. Classify failures against the known-red-on-main baseline ledger —
      mechanical (`scripts/step9c_baseline.py compare`), never prose
      arithmetic (#1022: on 2026-07-02 seven sessions each burned a round
      re-proving red main was pre-existing). Runs AFTER the final pytest
      invocation of step 1 (touched scope, or the 1c full-scope override —
      compare gates the junit of whichever actually ran) AND after 1b's
      foreground verdict read. Run compare as a BACKGROUND Bash invocation
      (`run_in_background=true`) from the SAME worktree cwd, in the SAME
      background + rc-file pattern as 1b. BACKGROUND IS REQUIRED, NOT
      OPTIONAL: `--run-pristine` (always passed here) may run up to
      `--max-pristine-files` (5) single-file pristine oracle runs, each
      bounded by `derive_pristine_timeout_s` at 600–1950s (#1129:
      tests/test_workflow_lint.py alone derives 1950s), so a healthy
      compare can NEVER be guaranteed to fit the 600s foreground Bash tool
      cap — a foreground call converts a classifiable in-process exit 2
      into a tool-layer kill with COMPARE_OUT lost (#1129/#1098). Compare
      stays a SEPARATE background call, NOT folded into the 1b gate call:
      1b's foreground verdict read and the zero-collected guard run
      between them, and a folded call would burn up to ~77 min of
      pristine runs on a run those guards fail in seconds.
      ```bash
      cd "$WT" || { echo "FATAL: cd to issue worktree failed" >&2; exit 1; }
      # earlyoom-protect the compare (#1045; FAIL-OPEN — never block the verdict on
      # a choom failure): its pristine pytest children are the same earlyoom-preferred
      # long python work as the 1b gate; oom_score_adj inherits across fork/exec
      # (probe-verified; start_new_session does NOT reset it), so self-choom BEFORE
      # launch covers the whole compare tree incl. the pristine pytest children.
      sudo -n choom -n -600 -p $$ >/dev/null 2>&1 && COMPARE_CHOOM=ok \
        || { COMPARE_CHOOM=failed; echo "[warn] choom failed — compare pristine runs are earlyoom-UNPROTECTED (choom=failed)" >&2; }
      echo "[step9c] compare earlyoom protection choom=$COMPARE_CHOOM"
      # MANDATORY stale-file rm — the compare triplet ONLY (NEVER 1b's
      # junit/rc/log files: compare consumes them):
      rm -f /tmp/step9c-compare-issue-<N>.json /tmp/step9c-compare-issue-<N>.rc \
            /tmp/step9c-compare-issue-<N>.err
      # Re-read the 1b rc IN-CALL (shell variables do not survive across Bash
      # calls); a missing 1b rc file already took 1b's FAIL path — never invoke
      # compare against it:
      [ -f /tmp/step9c-rc-issue-<N> ] || { echo "FATAL: 1b rc file missing — apply 1b's FAIL path; compare not run" >&2; exit 1; }
      PYTEST_RC=$(cat /tmp/step9c-rc-issue-<N>)
      # Wedge bound 10800s ≥ the structural ceiling of compare's own in-process
      # bounds (5 pristine files × 1950s max derived bound + 120s scratch +
      # ruff/parse overhead) — it only ever fires on a genuine wedge (#1129
      # generous bias; re-derive if SLOW_TESTS / max-pristine-files change):
      timeout --kill-after=60s 10800s uv run python scripts/step9c_baseline.py compare \
        --junitxml /tmp/step9c-junit-issue-<N>.xml --pytest-rc "$PYTEST_RC" \
        --run-pristine --json \
        > /tmp/step9c-compare-issue-<N>.json 2> /tmp/step9c-compare-issue-<N>.err
      echo $? > /tmp/step9c-compare-issue-<N>.rc
      ```
      (stdout and stderr are SEPARATED — unlike 1b's merged log — because
      stdout is the JSON payload the verdict parses; stderr carries WARN /
      timeout-kill diagnostics.) When the background call completes (the
      harness notifies), read the verdict in a fresh foreground call from
      the FILES. A MISSING rc file means the background compare died before
      exiting (tool kill / watcher force-stop): treat as FAIL/indeterminate,
      never a silent PASS, and apply crash-fix-rounds
      § Kill-before-relaunch (probe `pgrep -af 'step9c_baseline[.]py compare'`)
      before any re-run:
      ```bash
      if [ ! -f /tmp/step9c-compare-issue-<N>.rc ]; then
        echo "FATAL: compare rc file missing — the background compare died before exiting. Kill-before-relaunch, then re-run step 1d; NEVER record PASS." >&2
      else
        COMPARE_RC=$(cat /tmp/step9c-compare-issue-<N>.rc)
        COMPARE_OUT=$(cat /tmp/step9c-compare-issue-<N>.json)
        echo "$COMPARE_OUT"
        if [ -s /tmp/step9c-compare-issue-<N>.err ]; then tail -20 /tmp/step9c-compare-issue-<N>.err; fi
      fi
      ```
      `COMPARE_RC` ∉ {0, 1, 2} (124/137 = wedge-timeout / kill) or an
      empty / unparseable JSON file is INDETERMINATE — FAIL, never PASS
      (#665/#736: capture the `.err` tail so the stall surfaces actionable
      evidence, never a silent kill).
      The COMPARE verdict — not the raw PYTEST_RC — decides pass/fail for
      steps 1–2:
      * `COMPARE_RC=0` → no NEW test failures and no lint regression; failures
        listed in `stripped` are pre-existing on main and do NOT block (the
        round may PASS steps 1–2 with PYTEST_RC=1).
      * `COMPARE_RC=1` → NEW failure(s) the branch introduced and/or a lint
        regression (the JSON names each). FAIL.
      * `COMPARE_RC=2` → indeterminate (PYTEST_RC ∉ {0,1} — aborted/interrupted
        run; missing/empty junitxml; suite crash; unusable ledger;
        scratch-ineligible dirty oracle (residual contaminating dirt, a
        scan-set node outside the file-anchored allowlist
        (step9c_baseline.py FILE_ANCHORED_SCAN_TESTS, #1337), or a
        non-sparse work root — other root dirt
        auto-falls back to a detached sparse scratch-worktree oracle at main
        HEAD, reported as JSON "pristine_oracle": "scratch-worktree"; since
        #1251 dirty in-package `src/` is neutralized via a probe-verified
        `PYTHONPATH=<scratch>/src` shadow (`"scratch_src_shadow": true`), so
        only `pyproject.toml`/`uv.lock` (or out-of-package `src/`) dirt still
        refuses);
        systemic main breakage). FAIL — never PASS on indeterminate.
        COMPARE_OUT is valid JSON on EVERY exit path under --json (exit-2
        payloads carry "indeterminate": true — an exit-2 payload's empty
        new/stripped arrays are NOT a clean verdict).
      The two step-1b guards run BEFORE compare and are UNCHANGED: the cd
      hard-guard and the `no tests ran` FAIL guard (zero collected is a FAIL
      regardless of compare's exit).
      If the compare JSON has `"stale": true`, kick a DETACHED background
      ledger refresh so the next session gets a fresh baseline — do NOT block
      this verdict on it:
      ```bash
      REFRESH_PID=$(bash -c 'cd "$1" || exit 1; setsid nohup timeout --kill-after=60s 2100s \
        env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
        uv run python scripts/step9c_baseline.py refresh \
        >> "$1/logs/step9c_baseline_refresh.log" 2>&1 < /dev/null & echo $!' _ "$REPO_ROOT")
      # earlyoom-protect the refresh (#1045; fail-open): sweep its session; the refresh's own
      # start_new_session pytest child (spawned >=~1s later, after lock + git-root resolution + selector/venv resolution + uv-run startup) inherits adj.
      ps -p "$REFRESH_PID" -o args=   # verify the pid is the workload (canonical form); on mismatch recover via pgrep -f 'step9c_baseline.py refresh'; a lock-held instant exit makes this benignly fail (choom=failed below)
      bash -o pipefail -c 'pgrep -s "$1" | xargs -rn1 sudo -n choom -n -600 -p' _ "$REFRESH_PID" >/dev/null \
        && echo "[step9c] ledger refresh detached pid=$REFRESH_PID log=$REPO_ROOT/logs/step9c_baseline_refresh.log choom=ok" \
        || echo "[step9c] ledger refresh detached pid=$REFRESH_PID log=$REPO_ROOT/logs/step9c_baseline_refresh.log choom=failed"
      ```
2. Lint: covered by step 1d `compare` — repo-wide `ruff check` /
   `ruff format --check` are diffed against the LIVE main-root baseline
   (only an INCREASE fails; main carries 2000+ pre-existing ruff errors,
   #1022), and the branch's touched `*.py` files must additionally be
   ruff-clean + format-clean in absolute terms. Do NOT run bare
   `uv run ruff check . && uv run ruff format --check .` as a verdict gate —
   it always fails on pre-existing main red and re-derives what the ledger
   already answers.
3. Integration tests (conditional, if diff touches train/eval/orchestrate)
4. Coverage gap report (flags, does not auto-generate)

The `epm:test-verdict v1` marker note records: scope used (`touched`/`full`),
the files run, the gate timeout bound used (the selector's
recommended-timeout-s), pass/fail counts, and ALL selector stderr diagnostics — the
work-root + branch provenance breadcrumb, any `NOTE — empty diff` line, and
any untested-touched-file WARNs (so the orchestrator surfaces wrong-cwd runs
and coverage gaps — never silently skipped), and the compare classification
JSON (new vs known-red-stripped failures with any scan-test / diff-linked
masking WARNs, the ruff delta vs the live main baseline, the ledger main_sha
+ age + stale flag, and any dirty-code-path flags), and
the gate + compare earlyoom-protection state — COPY the `[step9c] gate
earlyoom protection choom=…` and `[step9c] compare earlyoom protection
choom=…` breadcrumb lines from the gate and compare calls' transcripts (plus
the 1d refresh `pid= log= choom=` breadcrumb line when a refresh was kicked);
never infer `choom=ok` from the absence of a warn line. A zero-collected /
`no tests ran` outcome is recorded as FAIL (never PASS on exit 0) per step
1b's guard.

Post `epm:test-verdict v1`. PASS = steps 1–2 pass via compare exit 0 (with
`--pytest-rc` folded) AND neither step-1b guard fired, AND steps 3
(conditional integration tests) and 4 (coverage gap report) completed per
their existing rules -> Step 10. FAIL (`epm:test-verdict` FAIL count < 3) ->
stay in `reviewing`, re-spawn implementer. FAIL (`epm:test-verdict` FAIL
count >= 3) -> run
CRON-TEARDOWN (fresh `CronList` → `CronDelete` every job in the two-leg
match set: the recurring `/issue-tick <N>` tick + stray one-shot
`/issue <N>` wakeups; not-found = success — no-ops for a code-change task
that never armed one; § CRON-TEARDOWN procedure), then status to
`blocked`. Fire `PushNotification({"message": f"#{N} BLOCKED: tests
FAIL after 3 rounds — open it"[:200], "status": "proactive"})` before
setting status (soft-fail).

### Step 10: Auto-complete (fires after user promotes clean-result from `awaiting_promotion`, or `epm:test-verdict` PASS for code-change paths)

No user gate. The skill transitions the task to a terminal-or-
`followups_running` state automatically. If the user disagrees with the
transition, they `task.py set-status <N> blocked` to reopen.

#### Step 10 step 0: Completion audit (gates entry to step 1)

Cheap insurance against drift on multi-part tasks: re-read the ORIGINAL
task body and verify every numbered ask is actually addressed. The
clean-result-critic checks the *write-up*; this checks the *body ->
work* contract.

1. Re-fetch the current body: read `body.md` via
   `task.py view <N> --json` (the body now is the clean-result; the
   ORIGINAL body lives in `original-body.md`).
2. Enumerate every:
   - Numbered ask (`1. ...`, `2. ...`)
   - Acceptance criterion (sentences containing "must", "should
     report", "deliverable", "include")
   - Explicit deliverable (e.g., "produce a clean-result with X
     figure")

   If the original body has no numbered asks (free-form description),
   audit against the headline goal sentence only and note "no
   numbered asks found" in the marker.
3. For each ask, locate evidence it was addressed:
   - `experiment` -> grep the promoted clean-result body + `epm:results`
     event. (**`paper: true`?** The promoted body is a thin paper-stub;
     the clean-result content lives in the paper — grep
     `docs/papers/issue_<N>/issue_<N>.tex` + the stub abstract +
     `epm:results` instead.)
   - `infra` / `batch` / `analysis` / `survey` -> grep the PR diff
     (`gh pr diff <PR>`) + `epm:test-verdict`.
4. Post `epm:completion-audit v1` event with a checklist:
   ```markdown
   ## Completion Audit — PASS | INCOMPLETE

   Audited against original-body.md as of <commit-sha-or-timestamp>.

   - [x] **Ask 1:** "<verbatim ask>" — addressed in <clean-result §Headline numbers | PR file foo.py:42>
   - [x] **Ask 2:** ... — addressed in ...
   - [ ] **Ask 3:** "<verbatim ask>" — NOT FOUND in clean-result or `epm:results`. Proposal: <what's missing>.
   ```
5. Branch on verdict:
   - **All ☑ (PASS):** proceed to step 1 below.
   - **Any ☐ (INCOMPLETE):** move status to `blocked`, do NOT advance.
     The audit event is the bounce-back payload. User either (a)
     modifies the original body to reconcile resolved scope-creep, (b)
     re-runs the missing work via a follow-up `/issue` cycle, or (c)
     re-runs `task.py set-status <N> awaiting_promotion` to override.
     Per CLAUDE.md STATE-TO-`status:blocked` criterion 5.

#### Step 10 step 1+: existing flow

1. If code change: mark PR ready for review (not merge — user merges).
2. Update `RESULTS.md` if the finding is headline-level (propose diff as
   an `epm:results-md-diff v1` event — do NOT auto-edit).
3. Update `eval_results/INDEX.md` with a new entry.

4. **Detect open follow-up children.** Search for any task whose
   `body.md` frontmatter contains `parent_id: <N>`. The exact
   filesystem query:
   ```bash
   find tasks -path 'tasks/*/*/body.md' \
     -exec grep -l "parent_id: <N>" {} +
   ```
   filtered by parent folder NOT in {`completed`, `archived`}.

   A child is "still in flight" if it is NOT in the `completed` /
   `archived` parent folders. The parent's destination state depends
   on whether ANY child is still in flight.

5. **Choose the destination state.**

   - **At least one child still in flight** AND task type is
     `experiment` -> **status `followups_running`**.
     The parent's own work is finished but its children own the queue.
     Re-invoking `/issue <N>` later re-runs Step 10 step 4 — once all
     children reach a terminal state, the parent advances to
     `completed`. (This is the LEGACY use of `followups_running`; the
     status's primary semantics as of 2026-06-10 is "a same-issue
     follow-up round is executing on this task" — Step 9b § Same-issue
     follow-up loop. The Step 0 dispatcher disambiguates by the
     presence of an unrun `epm:followup-scope`.)
   - **No children in flight** AND task type is `experiment` ->
     **status `completed`**.
   - **type `infra` / `batch` / `analysis` / `survey`** (regardless of
     children) -> **status `completed`**. Code-change paths don't use
     `followups_running` because they don't seed experimental
     follow-ups via Step 10b.
   - **No `type` frontmatter** -> STOP, post an error event asking the
     user to add one. Do NOT pick a default, and do NOT advance until
     fixed.

6. Run CRON-TEARDOWN before applying the terminal status (fresh
   `CronList` → `CronDelete` every job in the two-leg match set: the
   recurring `/issue-tick <N>` tick + stray one-shot `/issue <N>` wakeups;
   not-found = success; § CRON-TEARDOWN procedure). For an
   experiment the cron was already torn down at Step 9b
   (`awaiting_promotion`), so this is an idempotent backstop; for a
   code-change path arriving via Step 9c PASS it is the teardown site (a
   code-change task usually never armed a cron, so it no-ops, but running
   it keeps the "every terminal/park exit tears down" contract uniform).
   Then apply the chosen status via `task.py set-status` (which performs the
   `git mv` + commit + folder move):
   ```bash
   uv run python scripts/task.py set-status <N> <new-status> \
     --note "Step 10 auto-complete: <reason>"
   ```

7. Post final event `epm:done v1` (or
   `epm:status-changed` recording the followups_running transition)
   summarizing: outcome, key numbers, what's confirmed/falsified,
   what's next, plus a link to the clean-result write-up location (for
   experiments) AND a list of in-flight child follow-ups (when
   transitioning to `followups_running`). Include the line
   `Moved to **<status-name>**.`

8. **LEAVE THE TASK ON DISK.** Tasks are never deleted by the skill.
   Done-ness lives in the parent folder under `tasks/`. The folder is
   the durable artifact.
9. Do NOT delete the worktree — user decides when to clean up.
10. If type is `experiment` AND we just landed at `completed` (no
    children blocked us), proceed to Step 10b (follow-up proposer). If
    we landed at `followups_running`, SKIP Step 10b — the proposer was
    already run in a prior `/issue <N>` invocation that produced the
    children we're now waiting on.

    If type is `analysis` AND we just landed at `completed` AND the
    clean-result body carries a measured finding (a `## Results`
    section), proceed to **Step 10c-bis ONLY** (results-driven
    literature-positioning). Do NOT enter Step 10b (follow-up proposer)
    or Step 10c (living-docs update) — both are EXPERIMENTS-ONLY; an
    analysis task seeds no experimental follow-ups and owns no
    open-questions diff. The `## Results` gate above is the PRIMARY
    guard: a finding-less analysis task fails it and never enters this
    branch (it routes straight to Step 10d — see the entry condition in
    the next paragraph). As a redundant defense-in-depth backstop, Step
    10c-bis ALSO applies its own kind / finding-bearing gate (see Step
    10c-bis § When to run: a `kind: analysis` task with no `## Results`
    writes a 3-line "no measured finding" stub and raises no gate), so
    even if a finding-less analysis task somehow reached this branch it
    would self-skip cleanly inside the agent. This is the SOLE entry
    into Step 10c-bis for analysis tasks — without it a finding-bearing
    `kind: analysis` clean-result would silently skip the
    literature-positioning step that CLAUDE.md item 11, the
    `related_work_positioning` gate, and the `related-work-finder` agent
    all promise it.

    Any other type (`infra` / `batch` / `survey`), or an `analysis`
    task with no `## Results`, enters NONE of the Step 10b/10c/10c-bis
    batch — proceed straight to Step 10d (unchanged).

### Step 10b: Follow-up proposer (experiments only — runs ∥ Step 10c)

**Parallel spawn with Step 10c + 10c-bis.** Steps 10b, 10c, and 10c-bis
keep their numbering and their per-step semantics, but their agents are
spawned CONCURRENTLY: evaluate all three steps' skip conditions first
(10b's autonomous-mode short-circuit below; 10c's kind / `relates_to`
skips; 10c-bis's kind / finding-bearing skips), then spawn the SURVIVING
agents in ONE message (one Agent call per agent that did not skip,
staggered a few seconds apart per the CLAUDE.md 429 guidance), each
preceded by its own `stage-dispatch` breadcrumb (Step 9 entry-guard
convention; for the new agent `stage=related-work round=1
subagent=related-work-finder`). Each spawned agent reads the completed
clean-result; their outputs are independent (follow-up proposals vs a
proposed open-questions diff vs a proposed Related-findings positioning
note). Process each return per its own step text, and JOIN every spawned
agent — `epm:follow-ups v1` posted (or 10b skipped) AND the 10c proposal
handled (gate raised / parked per 10c, or 10c skipped) AND the 10c-bis
proposal handled (`related_work_positioning` gate raised / parked per
10c-bis, or 10c-bis skipped) — before entering Step 10d. The
`living_docs_update` and `related_work_positioning` gates, all markers,
and the user-confirmation semantics are unchanged by the scheduling; only
the spawn scheduling is shared. If a step's skip condition fires, spawn
only the other agents.

**Per-kind membership of this batch (no concurrency assumption beyond
this list):**

- `experiment` (entered from the Step 10 step-10 experiment branch) →
  all THREE agents spawn (`follow-up-proposer` + `living-docs-updater` +
  `related-work-finder`), subject to each step's own skip conditions.
- `analysis` with a measured finding (entered from the Step 10 step-10
  analysis branch) → ONLY `related-work-finder` spawns. Steps 10b and 10c
  are EXPERIMENTS-ONLY and are NOT entered for an analysis task, so this
  "parallel batch" degenerates to a single Agent call — do NOT spawn
  `follow-up-proposer` or `living-docs-updater`. (A future reader: this is
  the one-agent case; the word "batch" here does not imply three spawns.)
- `infra` / `batch` / `survey`, or `analysis` with no `## Results` → none
  of the three spawn (this batch is not entered at all; proceed to Step
  10d).

Auto-fires after `completed` for `experiment` tasks. Spawn the
`follow-up-proposer` agent with:

**Proposer-already-ran short-circuit:** if an `epm:follow-ups v1` marker
for the most recent park is ALREADY present on the parent's
`events.jsonl`, the proposer ran at Step 9b — either the autonomous
follow-up auto-spawn block (an `epm:follow-ups-autospawned v1` marker is
also present) OR the cheap-band block (block C0, which fires in
interactive sessions too as of 2026-06-13). SKIP re-spawning the
proposer here — it would duplicate the proposal list and is unnecessary.
The `epm:follow-ups v1` posted at Step 9b is still the canonical list
for the user; any `auto_run: no` / cap-skipped / fail-safe-skipped
proposals from that marker remain on the table for the user to pick from
manually post-promotion (route the pick by `question_relation` as
below). An interactive task that landed here with NO `epm:follow-ups v1`
(no cheap-band candidate ever existed, so block C0 was a no-op that
posted nothing — see C4) runs the proposer here as normal.
- The completed task's plan (the `plans/plan.md` symlink)
- The results (`epm:results` event)
- The clean-result body
- The interpretation critique history (`epm:interp-critique v1..vN`)
- The clean-result-critic verdict history

The proposer outputs 1-3 concrete follow-up proposals, each with:
- Pre-filled spec from parent (reproducibility card copied, only diff
  highlighted)
- Stated hypothesis + falsification criteria
- Type (ablation, reproduction, diagnostic, scaling, etc.)
- Cost estimate in GPU-hours
- Ranked by information gain per GPU-hour

Post as `epm:follow-ups v1` event on the completed task.

**Run the value-critique (redundancy screen) before surfacing the picks
— subroutine VC (Step 9b § Follow-up value-critique).** Run VC over the
`epm:follow-ups v1` set (idempotent — if Step 9b's C0a / autonomous block
already ran it this park, reuse the merged verdict; the
proposer-already-ran short-circuit above means VC's prior verdict is
usually already present). VC parks every `redundant` proposal at
`on_hold` (`epm:followup-parked-redundant v1`, revivable) and hands back
only the `not-redundant` survivors. **Surface ONLY the `not-redundant`
survivors to the user** for picking; for each parked-redundant proposal,
state ONE chat line naming the duplicate + the `on_hold` task id so the
user knows it was saved (not dropped) and can revive it. The user's pick
is then routed by `question_relation` as below — a `redundant` proposal
is never offered as a pick (it is already parked on_hold).

**Route the user's pick by `question_relation`** — the litmus is the
Takeaways test: *would the result rewrite THIS issue's `## Takeaways`?*
yes → `same` (same-issue loop), no → `substantially-different` (child).
Changing method/dose/panel/seeds/eval-surface/prompt-bank or adding a
control on the same question is `same`; only a result that would move the
task's `## Goal` / open-questions anchor is `substantially-different`.
(Untagged proposals: the treat-as-`substantially-different` fallback
applies only when the `epm:follow-ups v1` marker was posted before
2026-06-09; on a newer marker the missing tag is a proposer-contract
violation — classify the picked proposal yourself against the
Takeaways litmus + `.claude/agents/follow-up-proposer.md` §
"question_relation tag — criteria" and note the violation in the
resulting `epm:followup-scope v1` / child-creation marker):

- **`same`** — do NOT file a child task. Post `epm:followup-scope v1`
  on this task (`source: step-10b-pick`, fields per workflow.yaml §
  markers) and re-invoke `/issue <N>` — the same-issue follow-up loop
  (Step 9b § Same-issue follow-up loop) executes it ON this issue and
  re-parks at `awaiting_promotion` for re-promotion. User-picked
  rounds do not count against the autonomous round cap.
- **`substantially-different`** — create a child task as today, by
  telling the main conversation agent to create it via
  `task.py new --parent <N> --kind experiment --goal "..." --title "..."`
  (or manually copying the spec into a new task via `task.py new`).

Each created follow-up task carries `parent_id: <N>` in its `body.md`
frontmatter; lint scans enforce that the parent exists. Lint output is
visible via `task.py audit`.

**Announce every follow-up/child task in chat at creation time.** The
moment `task.py new` returns a new id (here, or anywhere mid-session a
child task is filed), immediately post ONE line in chat:
`Filed #<N> '<title>' (child of #<parent>, status:<s>)`. A created task
that stays invisible until the user asks "what is #<N>?" is a dropped
handoff. (Incident 2026-06-01: #461 was filed and worked on but never
announced — the user lost track and had to ask.)

### Step 10c: Living-docs update hook (experiments only)

Auto-fires after a `kind: experiment` task lands at `completed` (the
deliberate post-promotion completion moment). It keeps the living
research hub (`docs/open_questions.md`, and `docs/papers.md` when
warranted) from going stale by proposing — never auto-applying — an
update to the question(s) this experiment was linked to at creation
(Step 0c-link). **Non-blocking:** the task is already `completed`, so
the proposal can park indefinitely if the user is away; nothing about
completion waits on it.

1. Skip when the task `kind != "experiment"` — `analysis | infra |
   batch | survey` carry no open-question link.
2. Skip when the task has no `relates_to:` list in `body.md`
   frontmatter (was never linked at Step 0c-link) — surface one chat
   line noting the missing link and continue to Step 10d.
3. Spawn the `living-docs-updater` agent (fresh context) — on the
   normal path this spawn already happened in the Step 10b parallel
   batch (see Step 10b § Parallel spawn with Step 10c + 10c-bis); spawn
   here only if it didn't. Brief: task
   `<N>` + its clean-result body + the linked question block(s) (grep
   `docs/open_questions.md` for each `relates_to` id's `<!-- q:<id> -->`
   anchor) + the rest of `open_questions.md` so it can spot a needed
   reword / split / merge / new question. The agent PROPOSES (never
   applies) a unified diff + rationale and posts
   `epm:living-docs-proposed v1`. It is bounded + single-turn.
4. Present the proposed diff for confirmation at the
   `living_docs_update` conditional gate (registered in
   workflow.yaml § gates.conditional). The prompt is a binary `confirm`
   vs `reject` (see workflow.yaml § gates.living_docs_update); "edit" is
   a refinement of `confirm`, not a third option — the user may hand-edit
   the proposed diff and the same confirm path applies the edited patch.

   <!-- gate: gates.living_docs_update -->
   ```python
   AskUserQuestion(questions=[{
     "question": (
       "Apply this living-docs update for task #<N>? "
       "Proposed diff: epm:living-docs-proposed v1 on https://eps.superkaiba.com/tasks/<N>"
     ),
     "header": "Living docs #<N>",
     "multiSelect": False,
     "options": [
       {
         "label": "Confirm",
         "description": (
           "Apply the proposed diff (edit it first if you like) via "
           "scripts/living_docs.py apply <N> <patch>. Touches "
           "docs/open_questions.md (+ docs/papers.md if proposed)."
         ),
       },
       {
         "label": "Reject",
         "description": (
           "Skip; nothing written to the living docs. The proposal "
           "parks for the nightly /daily living-docs backstop re-synthesis."
         ),
       },
     ],
   }])
   ```
5. Branch on the user's choice:
   - **Confirm** (optionally after hand-editing the diff): apply the
     confirmed patch and post the applied diff:
     ```bash
     uv run python scripts/living_docs.py apply <N> /tmp/issue-<N>-living-docs.patch
     uv run python scripts/task.py post-marker <N> epm:living-docs-updated \
       --note "Applied living-docs update; touched <q-ids>; State trailer(s) bumped."
     ```
     `living_docs.py apply` is the single writer (atomic flock + one
     commit + dated changelog line). It applies ONLY the confirmed patch
     — accretive evidence/State bump or broader multi-question edit, no
     judgement of its own.
   - **Reject:** write nothing to the docs; record the decline:
     ```bash
     uv run python scripts/task.py post-marker <N> epm:living-docs-update-rejected \
       --note "User declined the living-docs proposal. Reason: <one line>. Proposal preserved inline."
     ```
<!-- example: anti-pattern -->
6. **Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): do NOT raise the
   `AskUserQuestion`, do NOT print the proposed diff as a confirm/reject
   text menu to chat, and do NOT auto-apply. Per § Autonomous session
   behavior → `living_docs_update`, living-docs mutations are user-only
   by spec. The `epm:living-docs-proposed v1` marker is already posted;
   the proposal parks for the user to confirm on a later `/issue <N>`
   re-invocation or for the nightly /daily living-docs backstop re-synthesis to
   reconcile. EXECUTE the continuation to Step 10d in this same turn;
   do NOT end the turn waiting on user confirmation.

This hook is idempotent: skip if `epm:living-docs-updated v1` or
`epm:living-docs-update-rejected v1` already exists on the task.

### Step 10c-bis: Results-driven literature-positioning hook (findings-bearing tasks)

Auto-fires after a `kind: experiment` task lands at `completed` (and for
`kind: analysis` tasks that carry a measured finding). It closes the gap
between the project's front-loaded literature grounding (the planner's
hyperparameter sources + the clarifier's lit review, both keyed on the
QUESTION and run BEFORE results exist) and the post-results question
"we measured X — who else reported X, and does it replicate / contradict /
extend ours?". The `related-work-finder` agent runs a bounded,
findings-keyed arXiv-MCP + web search and PROPOSES (never applies) a short,
citation-verified "Related findings" note for the clean-result `## Goal` →
`**Broader narrative:**` slot. **Non-blocking + advisory:** the task is
already `completed`, so the proposal can park indefinitely; nothing about
completion waits on it, and a thin / empty / over-budget note never blocks
promotion. **0 GPU-h.**

**When to run** (mirrors Step 10c's gating):

1. `kind: experiment` → always.
2. `kind: analysis` → only when the task has a discernible measured finding
   (its clean-result body has a `## Results` section). If not, the agent
   writes a 3-line "no measured finding to position" stub and exits — no
   gate is raised.
3. `kind: infra | batch | survey` → SKIP entirely (no clean-result
   findings to position). Log one chat line `Step 10c-bis skipped
   (kind=<X>)` and continue to Step 10d.
4. **Idempotency:** skip if `epm:related-work-proposed v1` (for this park)
   already exists on the task — paired with `epm:related-work-applied v1`
   / `epm:related-work-rejected v1`, this covers re-entry / backstop ticks.
   For a same-issue follow-up round, re-run keyed on the new round's
   `followup_label` (the findings changed) — the same EXTEND pattern as the
   methodology-doc idempotency.

Spawn the `related-work-finder` agent (fresh context) — on the normal path
this spawn already happened in the Step 10b parallel batch (see Step 10b §
Parallel spawn with Step 10c + 10c-bis); spawn here only if it didn't.
Brief: source task `<N>` (the agent reads the clean-result body, skims
`docs/papers.md`, and anchors on the two pinned sibling papers itself). The
agent PROPOSES (never applies) the artifact `artifacts/related-work-proposal.md`
+ a rationale and returns; the orchestrator posts `epm:related-work-proposed
v1` (artifact path + the proposed ≤80-word `**Broader narrative:**`
addition + the `search_status` + the verified-citation list + the realized
search budget + the optional manual-triage papers list).

Present the proposed addition for confirmation at the
`related_work_positioning` conditional gate (registered in workflow.yaml §
gates.conditional). The prompt is a binary `confirm` vs `reject` (see
workflow.yaml § gates.related_work_positioning) — NOT a 3-option menu.

   <!-- gate: gates.related_work_positioning -->
   ```python
   AskUserQuestion(questions=[{
     "question": (
       "Apply this Related-findings positioning note for task #<N>? "
       "Proposal: epm:related-work-proposed v1 on https://eps.superkaiba.com/tasks/<N>"
     ),
     "header": "Related work #<N>",
     "multiSelect": False,
     "options": [
       {
         "label": "Confirm",
         "description": (
           "Splice the proposed <=80-word **Related findings:** clause into "
           "the ## Goal -> **Broader narrative:** slot via "
           "scripts/task.py set-body, re-run verify_task_body.py (WARN-only "
           "on the total-prose budget). Touches the task body's ## Goal slot "
           "ONLY (no docs/papers.md edit in v1)."
         ),
       },
       {
         "label": "Reject",
         "description": (
           "Skip; nothing written to the body. The proposal parks inline in "
           "epm:related-work-rejected v1 so a future pass can reconsider."
         ),
       },
     ],
   }])
   ```
5. Branch on the user's choice:
   - **Confirm** (optionally after hand-editing the clause): splice the
     confirmed ≤80-word clause into the body's `## Goal` →
     `**Broader narrative:**` slot via `set-body`, re-run the verifier
     (WARN-only on budget — never a blocking FAIL), and post the applied
     addition:
     ```bash
     uv run python scripts/task.py set-body <N> --file /tmp/issue-<N>-body-with-related-work.md
     uv run python scripts/verify_task_body.py --issue <N>   # WARN-only on the total-prose budget; never block
     uv run python scripts/task.py post-marker <N> epm:related-work-applied \
       --note "Applied Related-findings note to ## Goal -> **Broader narrative:**; verdict <V>; cited <arXiv ids>. No docs/papers.md edit (v1)."
     ```
     The gate applies ONLY the `## Goal` body edit — it does NOT apply any
     `docs/papers.md` edit in v1 (the agent's suggested-papers list is
     human-triage only; the papers.md auto-apply leg is a deferred
     follow-up).
   - **Reject:** write nothing to the body; record the decline with the
     proposal preserved inline:
     ```bash
     uv run python scripts/task.py post-marker <N> epm:related-work-rejected \
       --note "User declined the Related-findings proposal. Reason: <one line>. Proposal preserved inline."
     ```
<!-- example: anti-pattern -->
6. **Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): do NOT raise the
   `AskUserQuestion`, do NOT print the proposed note as a confirm/reject
   text menu to chat, and do NOT auto-apply. A literature-positioning note
   is a taste / scope call the autonomous session does not make. The
   `epm:related-work-proposed v1` marker is already posted; AUTO-REJECT-PARK:
   post `epm:related-work-rejected v1` with the note
   `autonomous — parked for user review` and the proposal preserved inline,
   so it survives for a later interactive `/issue <N>` re-invocation.
   EXECUTE the continuation to Step 10d in this same turn; do NOT end the
   turn waiting on user confirmation.

This hook is idempotent: skip if `epm:related-work-applied v1` or
`epm:related-work-rejected v1` already exists on the task.

### Step 10d: Auto-merge the worktree (both experiment and impl)

The worktree merge is **automatic — no prompt, no cooldown**. It is the
single canonical merge procedure, invoked from TWO trigger points:

- **Experiments** — at the `awaiting_promotion` transition (Step 9b),
  the instant clean-result-critic PASSes. The merge does NOT wait for
  the user to promote the clean-result.
- **Code-change paths** (`infra` / `batch` / `analysis` / `survey`) — at
  this step, the instant the task auto-completes (Step 10 -> `completed`).

Rationale: deferring the merge stranded shared-library fixes on unmerged
branches, so the next experiment inheriting from `main` lacked them
(incident #456 -> #466: a `format_dataset` fix to
`src/explore_persona_space/train/trainer.py` lived on the #456 branch
that deferred merging; #466 inherited the older `format_dataset` from
`main` and crashed Phase-0 on the same data #456 trained on fine).
Auto-merging at the terminal point lands every code / figure /
`eval_results` commit on `main` immediately.

The worktree is **NOT removed** — it persists for inspection and is
reaped later by the daily stale-worktree audit (`worktree_audit.py`,
09:47) once the task reaches a terminal status and the worktree is idle.

**Idempotent.** Skip the whole step if `epm:merged` already exists on the
task (an experiment that merged at Step 9b is a no-op here). Also skip if
no PR exists or the branch is already merged into `main`.

#### Bare push / merge snippets (canonical — copy verbatim, never compose a piped variant)

Every `git push` / `git merge` / `gh pr merge|create` in this skill — and any
IMPROVISED recovery around one — runs BARE with its exit code checked. Never
pipe one through `tail` / `grep` / `head` / any filter: bash makes a
pipeline's exit status the LAST stage's, so the pipe masks a rejected push
and the session proceeds on a merge that never landed (#957; 4 sessions
2026-07-02). The `guard_piped_git_push.sh` PreToolUse hook BLOCKS the piped
shape anyway, so composing it just wastes a turn (~10 blocks across ≥8
sessions on 2026-07-07, #1138). Push/merge output is a few lines — it needs
no trimming. Copy these forms; the earlier composition sites (Step 5 round
pushes, the failure-lesson memory persist, Step 9a-ter re-analysis commits,
the Step 9b auto-merge trigger) point here:

```bash
# (1) Worktree branch push, rebase-retry on reject (the safe-case form):
git -C "$WT" push origin issue-<N> \
  || { git -C "$WT" pull --rebase=merges --autostash \
       && git -C "$WT" push origin issue-<N>; }

# (2) Repo-root push to main, single-flight recovery on reject
#     (sync_repo_root exit 0 can mean "another sync in flight — your push
#      has NOT landed"; for guard-critical pushes use the landing-verified
#      form in the post-merge stale-task-folder guard below):
git push origin main || uv run python scripts/sync_repo_root.py

# (3) PR merge (for sites OUTSIDE Step 10d/9b — those two run the full
#     lint-verdict-gated blocks below, never this bare form) — branch the
#     flow on the EXIT CODE, never on filtered output:
if gh pr merge <PR> --rebase --delete-branch=false; then
  echo "merged"
else
  echo "merge failed — route to the Step 10d failure handling"; false
fi

# (4) Need to bound long output? Redirect to a FILE and read the FILE in a
#     SEPARATE command — the push itself stays bare:
git push origin main > /tmp/issue-<N>-push.log 2>&1; PUSH_RC=$?
tail -20 /tmp/issue-<N>-push.log
[ "$PUSH_RC" -eq 0 ] || { echo "PUSH FAILED (rc=$PUSH_RC)"; false; }
```

Inside Step 10d itself, use the full executable blocks below (they wrap
forms (1)/(3) in the pre-push workflow-lint verdict gate); this subsection
is the copy source for every OTHER site.

#### Merge safety guards (run before the merge commands)

Derive the two paths cwd-robustly FIRST — never via `git rev-parse
--show-toplevel`, which from a worktree cwd returns the WORKTREE root and
nests `$WT` into `.../issue-<N>/.claude/worktrees/issue-<N>` (incident #506,
2026-06-09: the guard snippet exit-128'd with "cannot change to ..."):

```bash
REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
WT="$REPO_ROOT/.claude/worktrees/issue-<N>"
```

**Guard 0 — agent-memory pre-commit (run FIRST, before guards 1-3 and every
merge form).** Review rounds write per-agent memories
(`.claude/agent-memory/**`) with cwd in the worktree, leaving the tree dirty;
a dirty tree aborts the merge-conflict recovery's `git -C "$WT" merge
origin/main` below (incident #906, 2026-07-04). Commit them by explicit
pathspec — never `git add -A`:

```bash
MEM_COMMITTED=no
if git -C "$WT" status --porcelain -- .claude/agent-memory/ | grep -q .; then
  git -C "$WT" add -- .claude/agent-memory/
  git -C "$WT" commit -m "issue-<N>: persist agent-memory writes before Step-10d merge" \
    -- .claude/agent-memory/
  MEM_COMMITTED=yes
  # Best-effort branch push NOW: the fast-path / artifact-confirmed forms
  # never reach the safe-case pre-merge push, and on re-entry the pathspec
  # is clean (MEM_COMMITTED=no) so the commit would otherwise strand
  # local-only indefinitely. Failure is non-fatal — the safe-case push
  # condition below is the second chance.
  git -C "$WT" push origin issue-<N> || true
fi
```

Idempotent (a re-run finds the pathspec clean and skips). Scope is EXACTLY
`.claude/agent-memory/`: any OTHER dirty worktree path still surfaces through
the existing merge-failure handling — never blanket-commit it. REPO-ROOT-side
dirty agent-memory files are deliberately NOT committed here: every repo-root
pull in this step routes through `scripts/sync_repo_root.py` (see below),
whose autostash + rescue handling is built for the always-dirty shared root
(#967's hand-rolled root pull died `fatal: Cannot autostash`). For Guard 3
and the fast-path predicate, `.claude/agent-memory/**` paths are review-round
bookkeeping — always in-scope, never an UNSAFE trigger (see the Guard-3 note
and the fast-path mapfile filter below).

A behind-`main` `issue-<N>` branch can carry stale copies of OTHER tasks'
`tasks/` state, a crash between merge and a status flip can strand a
task at the wrong status, AND a branch based on another still-unmerged
`issue-<M>` branch will replay `#M`'s old commits onto `main` if blindly
rebase-merged. Three guards:

1. **Foreign-`tasks/` guard (strip whole foreign task folders before the
   merge).** `git diff --name-only "$MAIN_SHA"...HEAD -- tasks/` — the
   THREE-DOT form: merge-base..HEAD, i.e. only paths the branch's OWN
   replayed commits touch (#1280) — MUST be empty except THIS task's own
   folder (`tasks/*/<N>/`). Main-side advancement since the merge-base is
   BENIGN and must NOT trigger the strip: the `--rebase` merge replays only
   the branch's commits, so files the branch never touched keep `main`'s
   version. The retired two-endpoint endpoints (`"$MAIN_SHA" HEAD`) listed
   every path the fleet's marker churn advanced on `main` since the fork
   (#1271: 33 false positives on a branch whose replayed commits touched
   ZERO `tasks/` paths), and stripping those stages main-advancement content
   into a NEW branch commit whose server-side replay conflicts with main's
   further advancement — creating the very #1128-shape conflict the strip
   exists to prevent. The strip TARGET stays the freshly captured `main`
   snapshot (`MAIN_SHA`, captured in the block below).
   For any FOREIGN `tasks/` path in that diff — a `tasks/*/<M>/…` file for
   `M != <N>`, whether `events.jsonl`, `comments.jsonl`, `body.md`, or any
   other file — reset it to that snapshot BEFORE merging so the server-side
   `gh pr merge --rebase` has nothing foreign to conflict on (GitHub ignores
   this repo's `.gitattributes merge=union`, so a union merge cannot rescue
   a server-side conflict — the strip must happen here). The guard FETCHES
   `origin/main` first and pins every command to ONE captured `MAIN_SHA`:
   the fleet posts ~100+ marker commits/hr to `tasks/` on `main`, so a stale
   snapshot is the #1128 conflict class, and `origin/main` is a SHARED ref a
   concurrent session's fetch can advance mid-guard (the worktree shares its
   refs with every other session via the common git dir). A foreign path
   that EXISTS at `MAIN_SHA` is reset by checkout; a foreign path the branch
   ADDED (does not exist at `MAIN_SHA`) is dropped from the branch instead —
   a plain `git checkout "$MAIN_SHA" -- <added-path>` would crash with
   `pathspec did not match any file(s)` and abort the guard. Split FOREIGN
   accordingly:

   ```bash
   # Foreign tasks/* paths this branch touches (everything under tasks/ that
   # is NOT this task's own folder). Anchored so tasks/.../<N>/… is excluded.
   # MATERIALIZE the diff FIRST and check its OWN exit code: piped into grep
   # with `|| true`, a FAILED git diff (bad ref, missing origin/main) reads
   # as "no foreign files", the strip is silently skipped, and foreign
   # tasks/ reverts ride the merge (the #458 incident class — fail-open).
   # Same materialize-then-check pattern as the lint-gate trigger diff below
   # (#1047). The failure arm is TERMINAL (echo + false): do NOT merge —
   # route to the merge-failure handling (`epm:merge-failed v1`, continue).
   STRIPPED_FOREIGN=no   # set to yes iff a strip commit is actually created,
                         # so the safe-case push below fires only when needed.
   # Bounded mid-guard-churn retry (#1224): the strip work (checkout/rm/
   # commit) can fail when origin/main advances mid-guard (fleet churn moves
   # task folders; a piecewise execution re-derives a moved path). Attempt 2
   # re-runs the whole fetch->pin->diff->split->strip sequence against a
   # FRESH MAIN_SHA; a second failure is terminal. Composes with Known
   # failure shape 2 (that recovers a SERVER-SIDE refusal AFTER
   # certification; this recovers the strip itself BEFORE it). Run the
   # block as ONE Bash call — piecewise execution was the true #1224
   # antecedent, and the retry loop protects a one-call execution only.
   GUARD1_STATE=pending
   for GUARD1_TRY in 1 2; do
     if [ "$GUARD1_TRY" -eq 2 ]; then
       echo "Guard 1 RETRY (once, #1224): strip failed under a stale pin — re-fetch + re-pin"
     fi
     # Freshness fetch + single-SHA capture (#1128): strip against main as
     # CLOSE to the server-side merge as possible, pinned to ONE SHA so a
     # concurrent session's fetch cannot advance origin/main mid-guard. A
     # FAILED fetch is a WARN, not a block: the no-foreign CERTIFICATION
     # below is correct against any snapshot — staleness only raises the
     # conflict probability, and the re-snapshot retry (Known failure
     # shape 2 below) is the recovery. (The materialize-then-check diff
     # failure below stays TERMINAL — that one breaks certification, #1184;
     # bad ref is not churn, so a failed diff producer is NEVER retried.)
     git -C "$WT" fetch origin main --quiet \
       || echo "Guard 1 WARN: fetch origin main failed — stripping against last-fetched origin/main (conflict-prone; Known failure shape 2 is the recovery)"
     MAIN_SHA=$(git -C "$WT" rev-parse origin/main)
     # Three-dot (#1280): merge-base..HEAD = paths the branch's OWN replayed
     # commits touch. Two-endpoint ("$MAIN_SHA" HEAD) read main-side
     # advancement as foreign (33 false positives on #1271) and its strip
     # staged main content into a new branch commit that CREATED the
     # #1128-shape server-side conflict. The [ -z ] pre-check keeps an empty
     # MAIN_SHA fail-LOUD: an empty sha collapses the fused token to
     # '...HEAD' (= HEAD...HEAD, an EMPTY diff, exit 0 — silent fail-open),
     # where the old quoted empty argument made git error out.
     if [ -z "$MAIN_SHA" ] \
        || ! git -C "$WT" -c core.quotePath=false diff --name-only "$MAIN_SHA"...HEAD -- 'tasks/' \
         > /tmp/issue-<N>-guard1-tasks-diff.txt; then
       echo "Guard 1: git diff \$MAIN_SHA...HEAD -- tasks/ FAILED (bad ref or empty MAIN_SHA) — cannot certify no foreign tasks/ paths; do NOT merge"
       GUARD1_STATE=diff-failed
       break
     # Work arm: two-command elif list — mapfile fills FOREIGN from the FILE
     # (grep semantics identical to the old pipe), then the [ ... ] test (the
     # LAST command's exit) decides the branch.
     elif mapfile -t FOREIGN < <(grep -Ev "^tasks/[^/]+/<N>/" \
           /tmp/issue-<N>-guard1-tasks-diff.txt || true); [ "${#FOREIGN[@]}" -gt 0 ]; then
       FOREIGN_ON_MAIN=()      # exist at MAIN_SHA -> reset to that snapshot's version
       FOREIGN_BRANCH_ONLY=()  # only the branch added them -> drop from branch
       for p in "${FOREIGN[@]}"; do
         if git -C "$WT" cat-file -e "$MAIN_SHA:$p" 2>/dev/null; then
           FOREIGN_ON_MAIN+=("$p")
         else
           FOREIGN_BRANCH_ONLY+=("$p")
         fi
       done
       GUARD1_STRIP_RC=0
       if [ "${#FOREIGN_ON_MAIN[@]}" -gt 0 ]; then
         git -C "$WT" checkout "$MAIN_SHA" -- "${FOREIGN_ON_MAIN[@]}" || GUARD1_STRIP_RC=$?
       fi
       # rm WITHOUT --cached (#1244): the strip commit below is PATHSPEC-limited,
       # and a pathspec commit records WORKING-TREE content for the named paths
       # (git-commit(1) --only default) — an index-only deletion (the old
       # --cached form) is resurrected by it (#1210: 19 resurrected paths). The
       # working-tree copies are stale duplicates of foreign tasks/ state; main
       # is authoritative.
       if [ "${#FOREIGN_BRANCH_ONLY[@]}" -gt 0 ]; then
         git -C "$WT" rm -f --ignore-unmatch -- "${FOREIGN_BRANCH_ONLY[@]}" || GUARD1_STRIP_RC=$?
       fi
       # Commit the reset/removal so the branch diff no longer touches them,
       # but only if anything actually changed (idempotent: a re-run finds
       # nothing staged and skips the commit). Record that a strip commit was
       # made so the safe-case merge below knows it must push before rebasing.
       if [ "$GUARD1_STRIP_RC" -eq 0 ] \
          && ! git -C "$WT" diff --cached --quiet -- "${FOREIGN[@]}"; then
         if git -C "$WT" commit -m "issue-<N>: strip foreign tasks/ folders before Step-10d merge (pinned to main @ ${MAIN_SHA:0:12})" -- "${FOREIGN[@]}"; then
           STRIPPED_FOREIGN=yes
         else
           GUARD1_STRIP_RC=$?
         fi
       fi
       if [ "$GUARD1_STRIP_RC" -eq 0 ]; then GUARD1_STATE=ok; break; fi
       GUARD1_STATE=strip-failed
       # Un-stage AND restore the working tree for ONLY this attempt's paths
       # so the retry re-splits clean (never a bare `reset -- tasks/`, which
       # could touch own-task staged state). checkout HEAD restores index AND
       # working tree. Under the three-dot trigger (#1280) FOREIGN itself is
       # fork-point-stable into attempt 2 on a RESTORED attempt — the
       # pre-commit-failure case this arm handles: a fresh fetch moves the
       # pin, not the merge-base, and this restore leaves HEAD unchanged —
       # what refreshes on attempt 2 is the SPLIT (a fresh MAIN_SHA can
       # re-class a path on-main vs branch-only when main moves the folder)
       # and the strip TARGET content, so the restore must still leave no
       # uncommitted foreign litter behind — litter a later shape-1 worktree
       # merge could refuse on.
       git -C "$WT" checkout HEAD -- "${FOREIGN[@]}"
     else
       GUARD1_STATE=ok   # no foreign tasks/ paths — nothing to strip
       break
     fi
   done
   if [ "$GUARD1_STATE" != ok ]; then
     echo "Guard 1: not certified (state=$GUARD1_STATE) after the bounded retry — do NOT merge; route to the merge-failure handling (epm:merge-failed v1)"
     false
   fi
   ```

   The `STRIPPED_FOREIGN` flag is load-bearing: the strip commit above is a
   LOCAL worktree commit, but the safe-case `gh pr merge $MERGE_FORM` below
   merges the PR head ref as it exists on
   `origin/issue-<N>` (server-side), NOT the local worktree HEAD. An unpushed
   strip commit is therefore INVISIBLE to that server-side merge — the
   foreign `tasks/*` reverts would remain in the replayed history and land on
   `main` silently. So when `STRIPPED_FOREIGN=yes`, the safe-case block below
   MUST push the strip commit to the PR head ref BEFORE calling `gh pr merge`.

   This is idempotent at the commit gate (#1280): after a successful strip a
   re-run's three-dot diff can still list an on-main foreign path (HEAD holds
   the OLD pin's content, which differs from the merge-base) — the re-run
   re-checkouts the FRESH snapshot and the `diff --cached --quiet` gate skips
   the commit when main has not advanced, or refreshes the strip to the newer
   pin when it has; branch-ADDED foreign paths drop out of the diff entirely
   once rm-ed. A FAILED trigger diff fails loud (echo + `false`) instead of
   reading as no-foreign-files, leaving `STRIPPED_FOREIGN=no` while the block
   exits non-zero (#1184), and the guard never
   touches THIS task's own `tasks/*/<N>/` folder (the `grep -Ev
   "^tasks/[^/]+/<N>/"` carve-out). Never let a behind-`main` branch revert
   another task's `events.jsonl` / `comments.jsonl`. (Incident 2026-06-01:
   #458's merge branch, 1,146 commits behind main, silently rewound
   `tasks/running/448/events.jsonl`.) The `--rebase` merge form below replays
   the branch's commits on top of current `main`, so files the branch never
   committed keep `main`'s version (the `--squash` form lands the same
   own-diff content as one commit) — this is what keeps the clean-result body
   (committed to `main` by `task.py`, never in the worktree) safe across the
   merge.
2. **Status already off `running`.** By both trigger points the status is
   well past `running` (`awaiting_promotion` for experiments; `completed`
   for code paths, flipped in Step 10 step 6 BEFORE this step). A crash
   mid-merge therefore cannot strand a terminated-pod task at `running`.
   On a later `/issue <N>` resume: if the PR is already merged AND status
   is still `running` for any reason, auto-advance rather than
   re-dispatching.
3. **Branch-content / non-`main`-base guard.** Compute:

   ```bash
   BEHIND=$(git -C "$WT" rev-list --count HEAD..origin/main)
   MB=$(git -C "$WT" merge-base HEAD origin/main)
   # is the merge-base reachable on origin/main's first-parent mainline?
   ON_MAINLINE=$(git -C "$WT" rev-list --first-parent origin/main \
     | grep -Fxq "$MB" && echo yes || echo no)
   ```

   The branch is **unsafe to blind-rebase** if EITHER `ON_MAINLINE=no`
   (branch was forked off another `issue-<M>` branch that is itself
   still unmerged) OR the branch's **own commit content** is out of
   scope (the content check below). `BEHIND` alone is NEVER an
   automatic unsafe verdict — in this repo every `task.py` marker is a
   commit (~100+/hr fleet-wide), so a same-day, single-own-commit,
   mainline-based branch routinely reads `BEHIND` in the hundreds
   (incident #598, 2026-06-12: `BEHIND=305` tripped the old fixed-200
   threshold and routed an infra task's `src/` deliverables toward the
   artifact-confirmed path, which structurally cannot carry them — its
   surgical checkout is restricted to the task's own `tasks/` /
   `figures/` / `eval_results/` paths). `BEHIND` exceeding the
   threshold (default `200` commits) instead TRIGGERS the own-commit
   content check:

   ```bash
   # The branch's OWN commits (merge-base..HEAD) — with ON_MAINLINE=yes
   # this is exactly what `gh pr merge --rebase` will replay onto main
   # (the `--squash` form lands the same own-diff content as one commit).
   # quotePath=false: each $f below feeds a literal `git log ... -- "$f"`
   # pathspec — a `"`-quoted non-ASCII path matches nothing, non_sync reads
   # empty, and the file is misread as "imported from main" (fail-open).
   git -C "$WT" -c core.quotePath=false diff --name-only origin/main...HEAD   # three-dot form
   ```

   Before judging a workflow-surface path out-of-scope, EXCLUDE files whose ONLY
   branch-side touch is a Step-5a `spec-freshness` sync (the mandated
   `git checkout main -- $SAFE_SPECS` from `main`, NOT a branch deliverable).
   This mirrors Step 5a's own intent (line ~1925): a file that has NO non-sync
   branch-side commit is content imported FROM `main`, so it is never an
   out-of-scope regression. Match on the commit SUBJECT line ONLY — a `--grep`
   over subject+body would wrongly exclude a genuine branch edit whose commit
   BODY happens to mention "spec-freshness" (documentation, a retrospective),
   silently dropping a real branch touch. The two Step-5a sync SUBJECT variants
   both carry the token (`issue-<N>: sync workflow-surface specs from main
   (spec-freshness)`; `chore(issue-<N>): spec-freshness sync workflow surface
   from main`).

   ```bash
   # For each workflow-surface path $f in the own-diff: does it have any
   # branch-side commit whose SUBJECT does NOT contain "spec-freshness"?
   # Emit "<sha> <subject>" per own-commit touching $f, then keep only the
   # non-sync ones. If none remain, the file's only branch-side touches are
   # spec-freshness syncs => imported from main => NON-blocking for Guard 3.
   non_sync=$(git -C "$WT" log --format='%H %s' "$MB"..HEAD -- "$f" \
     | awk 'index($0, "spec-freshness") == 0')
   # $non_sync empty   => file imported via spec-freshness sync only => treat as
   #                      NON-blocking (in-scope, imported from main).
   # $non_sync nonempty => a genuine branch-side edit (its subject is not a sync)
   #                      => apply the normal in-scope / out-of-scope judgment.
   ```

   (`git log --format='%H %s'` prints `<sha> <subject>` per commit — the `awk
   index()` keeps only lines whose subject lacks the token; the sha is a hex
   string that never contains "spec-freshness", so the match is effectively
   subject-scoped. Equivalently `git log --format='%s' … | grep -v spec-freshness`.)

   UNSAFE if the own-diff — after the spec-freshness exclusion above — touches
   any foreign `tasks/` path (under `tasks/` but outside `tasks/*/<N>/`) or files
   outside this task's deliverable scope (paths neither the plan nor the code
   review touched). (Paths under `.claude/agent-memory/` — including the
   Guard-0 persist commit — are review-round bookkeeping: always in-scope,
   never an UNSAFE trigger.) If the list is clean — only this task's own deliverables,
   plus any spec-freshness-synced workflow-surface files — the branch is SAFE to
   rebase-merge regardless of `BEHIND`: the rebase replays only these commits,
   and files the branch never committed keep `main`'s version.

   In the unsafe case, do NOT run the safe-case `gh pr merge` (any
   `$MERGE_FORM`) — fall through
   to the **artifact-confirmed merge** procedure below. The Guard 1
   foreign-`tasks/` checkout is necessary but not sufficient: it covers
   `tasks/`, but a branch based on a still-unmerged parent branch also
   carries the parent's stale `src/` and `scripts/`, and a blind rebase
   replays both the parent's `tasks/` rewinds (already handled) AND its
   `src/` / `scripts/` regressions (NOT handled by Guard 1) onto
   `main`. (Incident 2026-06-03: `issue-479` was 1,153 commits behind
   `origin/main` and based on the still-unmerged `#472` branch — a
   blind `gh pr merge --rebase` would have replayed `#472`'s old
   commits onto `main`, risking regression of ~50 foreign `tasks/`
   folders AND shared `#472` infra. The orchestrator caught it by hand;
   this guard encodes the catch. The #479 class still trips under the
   reworked guard twice over: `ON_MAINLINE=no` flags it directly, and
   its `origin/main...HEAD` diff carries the whole `#472` parent
   payload, failing the content check.)

#### Fast-path routing pre-check (workflow-fix / small-ADDED-diff far-behind branches)

Run this AFTER guards 1-3 and BEFORE the safe-case `gh pr merge $MERGE_FORM`
call. For a workflow-fix / small-diff branch that is very far behind `main`,
a server-side `--rebase` predictably conflicts on churn even after Guard 1
strips foreign folders (GitHub replays the branch's own commits across
thousands of intervening main commits, and cannot use this repo's
`merge=union`). When the branch's OWN diff is small, entirely in-scope, AND
consists ONLY of ADDED files, skip the doomed server-side merge and route
DIRECTLY to the surgical additive checkout below.

**Why the ADDED-only conjunct is load-bearing (do NOT drop it).** The
surgical additive checkout does a WHOLESALE `git checkout issue-<N> -- <path>`
(the "One or more deliverables missing" branch, ~line 7080), which OVERWRITES
each listed path with the branch tip's copy. For a file the branch MODIFIED
that `main` also advanced (very likely on a 1000+-behind branch), that
overwrite silently discards `main`'s newer content with NO conflict surfacing
— a silent-wrong merge. Restricting the fast-path to ADDED-only files means
the surgical checkout only ever CREATES files that do not yet exist on
`main`, so it can never clobber a concurrently-advanced one. A branch that
MODIFIES a workflow-surface file (status M) is NOT fast-path-eligible and
takes the ordinary `gh pr merge $MERGE_FORM` path, whose server-side 3-way merge
either merges main's changes cleanly or surfaces a real conflict for the
recovery sub-procedure. (This is exactly why #787 itself — which MODIFIES
`SKILL.md` — is not fast-path-eligible.)

```bash
# Fast-path predicate — ALL of:
#  (a) task is kind:infra AND tagged wf-fix (a workflow-fix branch), AND
#  (b) BEHIND > 1000 (branch predates significant main churn), AND
#  (c) the branch's OWN diff (after the agent-memory filter below) touches
#      BETWEEN 1 and 15 files — the `-ge 1` LOWER bound is load-bearing: the
#      memory filter can EMPTY the list (a branch whose entire own-diff is
#      Guard-0 memory commits), and an empty list must NEVER fast-path — the
#      surgical `--diff-filter=A` list is then empty too, and an empty-input
#      xargs would run `git checkout issue-<N> --` with NO pathspec, which is
#      a BRANCH SWITCH of the shared repo root (`xargs -r` at the checkout is
#      the depth-2 defense), AND
#  (d) every touched file is in-scope: this task's own paths, workflow
#      surface, .gitattributes, or the methodology doc — NO shared src/ or
#      scripts/ additions (those need the full rebase to land), AND
#  (e) EVERY touched file is status A (Added) — no M (Modified), D (Deleted),
#      R (Renamed). A modified file would be clobbered wholesale by the
#      surgical checkout below.
KIND=$(uv run python "$REPO_ROOT/scripts/task.py" view <N> --json | \
  uv run python -c 'import sys,json; d=json.load(sys.stdin); fm=d.get("frontmatter",{}); print(fm.get("kind","")); print(" ".join(fm.get("tags",[])))')
TASK_KIND=$(printf '%s\n' "$KIND" | sed -n 1p)
TASK_TAGS=$(printf '%s\n' "$KIND" | sed -n 2p)
# Three-dot: the branch's OWN commits only (merge-base..HEAD) — never files
# main advanced but the branch never touched. Name-status so we can gate on A.
# Exclude .claude/agent-memory/ — the Guard-0 persist commit MODIFIES memory
# files, which must not fail the ADDED-only predicate (e). Memory edits land
# via the ordinary rebase (or stay on the PR branch for the deferred full
# rebase); the surgical checkout never carries a modified file anyway.
mapfile -t OWN_NS < <(git -C "$WT" diff --name-status origin/main...HEAD \
  | grep -vE $'\t\\.claude/agent-memory/[^\t]*$' || true)
# End-anchored on the LAST path field so an R-status rename whose SOURCE is
# a memory path but whose destination is elsewhere cannot dodge predicate
# (e) via over-filtering (Guard 0 itself never produces renames).
N_FILES=${#OWN_NS[@]}
IN_SCOPE=yes
ADDED_ONLY=yes
for line in "${OWN_NS[@]}"; do
  st=${line%%$'\t'*}          # status letter (A / M / D / R100 / ...)
  f=${line#*$'\t'}            # path (for a rename this is the source; fine —
                              # a rename fails ADDED_ONLY below regardless)
  [ "$st" = "A" ] || ADDED_ONLY=no
  case "$f" in
    tasks/*/<N>/*|figures/issue_<N>/*|eval_results/issue_<N>/*|eval_results/issue_<N>_*/*|ood_eval_results/issue_<N>/*) ;;
    .claude/*|CLAUDE.md|.gitattributes|docs/methodology/issue_<N>.md) ;;
    *) IN_SCOPE=no ;;
  esac
done
FAST_PATH=no
if [ "$TASK_KIND" = "infra" ] \
   && printf '%s' "$TASK_TAGS" | grep -qw 'wf-fix' \
   && [ "$BEHIND" -gt 1000 ] \
   && [ "$N_FILES" -ge 1 ] \
   && [ "$N_FILES" -le 15 ] \
   && [ "$IN_SCOPE" = "yes" ] \
   && [ "$ADDED_ONLY" = "yes" ]; then
  FAST_PATH=yes
fi
```

If `FAST_PATH=yes`: SKIP the safe-case `gh pr merge $MERGE_FORM` call and jump straight to
the **surgical additive checkout** (the "One or more deliverables missing"
branch of the artifact-confirmed procedure below). The surgical checkout
lands this branch's own ADDED files onto `main` directly, with no rebase.
Post `epm:merged v1` with `{artifact_confirmed: true, full_rebase_deferred:
true, surgical_checkout: true, fast_path: true, reason: "wf-fix branch
BEHIND=<BEHIND> > 1000, own diff <=15 in-scope ADDED-only files — skipped
doomed server-side rebase", files: [...]}`.

If `FAST_PATH=no`: proceed to the safe-case `gh pr merge $MERGE_FORM` (or the
artifact-confirmed path if Guard 3 said UNSAFE) exactly as before — this
pre-check adds NO new behavior for normal branches. A branch that MODIFIES a
workflow-surface file (status M ⇒ `ADDED_ONLY=no`) is deliberately not
fast-pathed; it takes the ordinary `$MERGE_FORM` merge.

#### Pre-push workflow-lint gate (runs before every merge form lands)

#931 (2026-07-04) merged a workflow-lint offender to `main`, breaking
`tests/test_workflow_lint.py` on pristine trunk fleet-wide for most of a day
(5 downstream sessions each burned 5-25 min classifying it as pre-existing).
#1147 adds a mapped invariant-test leg to the same gate: GLOB_SCAN_TESTS-covered
payloads (`scripts/issue*_*.py`, dispatcher scripts) previously landed with
zero pytest on the experiment auto-merge path (Step 9c is
code-change-kinds-only) — a channel through which #1144's thread-caps offenders
accreted; sampled offenders also landed via direct-to-main
free-analysis/analyzer commits, which this gate does NOT cover (see the Step
9a-ter follow-up). Gate the merge payload on the lint + the mapped invariant
tests BEFORE anything lands:

- **Trigger (cheap; artifact-only merges skip).** Run the gate ONLY when the
  branch's own three-dot diff (`git -C "$WT" diff --name-only
  origin/main...HEAD`, computed after guards 0-3) touches any path OUTSIDE
  the artifact-only set (`tasks/`, `figures/`, `eval_results/`,
  `ood_eval_results/`, `raw/`, `data/`, `docs/methodology/`). The lint's
  no-flags default run walks `.claude/**`, `CLAUDE.md`, `scripts/`, and
  `src/`, so any code-bearing payload is in scope.
- **Run a LANDING-TREE lint copy, both legs — no-flags bundle PLUS the parity leg.**
  The gate builds ONE ephemeral landing tree in /tmp (`git archive
  origin/main` over the lint-scanned cones), runs the BASELINE legs from
  that tree's own lint copy BEFORE the payload overlay, then overlays the
  branch's own-diff payload from the branch tip and runs the GATED legs
  from the SAME copy (#1212 — one lint vintage, trees differing only by
  the payload). `workflow_lint.py` derives its scan root from `__file__`
  (not cwd), so the gate-tree copy scans the gate tree; a plain non-git
  /tmp dir is a supported scan root (the root-guard hook pins `REPO=` to
  an absolute path, and `_other_worktree_prefix` is pure path-string
  logic).
  The no-flags default run does NOT bundle the asks / autonomous-asks /
  references / tables / status-labels checks (their `main()` branches lack
  `or no_flags`), yet `tests/test_workflow_lint.py` subprocess-runs those
  too — so trunk-pytest parity takes BOTH invocations. Measured wall
  ~4.5-6 min (no-flags) + ~1.4 s (parity leg) + ~1-2 s gate-tree
  construction on the shared VM; WARNs do not fail (PASS = exit 0 on
  both). The two leg pairs + TG legs total ~9-12+ min, so the executable
  block below can NEVER fit the 600s foreground Bash tool cap — run it as
  ONE BACKGROUND Bash call (`run_in_background=true`) with the per-leg
  `timeout` wedge bounds shown (the #991/#996 kill class: wrapping it in
  any ≤600s foreground bound, or running it foreground, SIGKILLs the
  whole gate shell mid-lint — #1245), then read the verdict in a FRESH
  foreground call from the FILE (completion-read below).

  ```bash
  # EXECUTABLE gate — forms (i) safe case and (ii) recovery share this block
  # ONE BACKGROUND Bash call (run_in_background=true) — see the bullet above.
  # verbatim (gated = the gate-tree lint copy on the LANDING tree —
  # origin/main + the branch's own-diff payload; baseline = the SAME copy on
  # the payload-free landing base, the tree before the overlay). Form
  # (iii) inlines the SAME trigger/normalize/subtract/verdict steps around its
  # checkout — see the surgical block. The verdict is PERSISTED to a file
  # because fenced bash blocks are separate shell invocations: the binding
  # sites consume the FILE, never a shell variable.
  # earlyoom-protect the gate (#1045 recipe, #1211; FAIL-OPEN — a choom failure
  # never blocks the gate and never touches the verdict logic): the lint legs
  # (~4.5-6 min python each) + the mapped pytest legs match this VM's earlyoom
  # --prefer regex (+300 badness) — the designated victim under fleet memory
  # pressure (#1143: the first gate run died mid-lint, verdict `crash`; the
  # choom-protected re-run passed). Self-choom the gate shell: every child
  # forked after this line inherits adj=-600.
  sudo -n choom -n -600 -p $$ >/dev/null 2>&1 && LINT_GATE_CHOOM=ok \
    || { LINT_GATE_CHOOM=failed; echo "[warn] choom failed — lint gate is earlyoom-UNPROTECTED (choom=failed)" >&2; }
  echo "[step10d] lint-gate earlyoom protection choom=$LINT_GATE_CHOOM"
  # Stale-verdict rm (Step 9c pre-rm parity): a verdict file present at
  # completion must provably come from THIS run — missing-after-completion
  # is then an unambiguous died-mid-run diagnostic. The #1041 same-tip
  # retry re-enters the merge CONDITIONAL, never this block, so the
  # surviving-verdict retry path is untouched.
  rm -f /tmp/issue-<N>-lint-verdict.txt
  # TRIGGER — materialize the own-diff FIRST and check the diff's OWN exit:
  # piped straight into grep, a FAILED `git diff` (bad ref, no merge-base)
  # is indistinguishable from an empty diff and would fail OPEN as an
  # artifact-only skip. (`set -o pipefail` cannot fix this form: `grep -q`
  # exits at first match and SIGPIPEs the producer, and the else branch
  # would still misread any nonzero as artifact-only.)
  if ! git -C "$WT" -c core.quotePath=false diff --name-only origin/main...HEAD > /tmp/issue-<N>-own-diff.txt; then
    # Failed trigger diff — the gate cannot classify the payload; fail CLOSED.
    echo crash > /tmp/issue-<N>-lint-verdict.txt
  # Classifier consumes grep's OUTPUT (non-empty => code-bearing payload),
  # never a `-q -v` exit status: a ugrep-shadowed shell returns rc=1 on
  # selected non-matching lines under -qv and silently disarmed this gate
  # as skip-artifact-only on a code-bearing payload (#928 -> #1125).
  elif [ -n "$(grep -vE '^(tasks/|figures/|eval_results/|ood_eval_results/|raw/|data/|docs/methodology/)' \
      /tmp/issue-<N>-own-diff.txt)" ]; then
    # GATE TREE (#1212): ONE ephemeral tree, TWO phases. Phase 1 (BASELINE)
    # lints the PAYLOAD-FREE landing base — origin/main's lint-scanned
    # surface, archived to /tmp — with origin/main's OWN lint copy
    # (workflow_lint.py derives its scan root from __file__). Phase 2 (GATED)
    # overlays the branch's own-diff payload onto the SAME tree and re-lints
    # with the same copy. Both legs share ONE lint vintage on trees differing
    # ONLY by the payload, so NEW = gated − baseline is payload-caused BY
    # CONSTRUCTION: kills the #1112 vintage false-blocks (stale branch linter;
    # branch scripts/ tree predating a main-referenced helper), stale
    # non-payload files vs main's newer checks, root dirt/lag in the compare,
    # and the moving-main inter-leg race — and ENFORCES checks added on main
    # after the branch forked (the old path-(i) residual, upgraded
    # deliberately: a payload violating a post-fork check now BLOCKS, the
    # #931 class). $WT and the repo root are never written; no commits are
    # created, so the verdict's sha-bind is unaffected. Payload files come
    # FROM the branch tip: a branch whose own diff touches workflow_lint.py
    # or a helper has its OWN copy exercised on the gated legs — it IS the
    # payload. Construction failures fail CLOSED via GT_RC in the crash arm.
    # The archive pathspec set must cover workflow_lint.py's scan/target
    # surface (.claude/ CLAUDE.md scripts/ src/ tests/ docs/ — the #1154
    # marker-recipe pins read docs/); a false block naming a path OUTSIDE
    # this set means the linter grew a new scan root — extend the set here.
    GT=/tmp/issue-<N>-lint-gate-tree
    GT_RC=0
    timeout --kill-after=30s 120s git -C "$WT" fetch origin main --quiet || true  # bounded: a hung fetch degrades to origin/main staleness, never a wedged gate
    { rm -rf "$GT" && mkdir -p "$GT"; } || GT_RC=1
    ( set -o pipefail; git -C "$WT" archive origin/main -- \
        .claude CLAUDE.md scripts src tests docs pyproject.toml \
      | tar -x -C "$GT" ) || GT_RC=1
    [ -f "$GT/scripts/workflow_lint.py" ] || GT_RC=1   # construction sanity
    # BASELINE legs (payload-free landing base — phase 1, BEFORE the
    # overlay). Per-leg exit codes ARE captured:
    # only the baseline's normalized failure LINES enter the compare, but a
    # baseline CRASH (rc>1, or rc!=0 with ZERO `workflow_lint:` lines) makes
    # the compare itself untrustworthy — that fails CLOSED via the crash arm
    # below, never `|| true`-erased. A red-but-line-emitting baseline (rc=1
    # WITH lines — main already red) stays fine: the subtraction handles it.
    # Per-leg rc capture is a NO-DOWNGRADE (max) fold — same fold at all
    # FOUR leg pairs (BASE + GATED, shared gate + surgical block): a leg-1
    # CRASH (rc=2, zero lines) must survive a leg-2 rc=1-with-lines; the
    # bare last-failure-wins `|| VAR=$?` capture erases the crash and
    # defeats the crash arm below. rc=0/0 stays 0; a lone rc=1-with-lines
    # stays 1 (attribution logic); any leg >1 reaches the crash arm.
    # 900s wedge bound per lint leg ≈ 2.5× the measured 360s upper wall
    # (bullet above; #1129 generous-ceiling sizing style) — fires only on a
    # genuine wedge; a bound kill (rc 124) flows through the NO-DOWNGRADE
    # fold into the crash arm below — fail CLOSED.
    BASE_RC=0
    timeout --kill-after=60s 900s uv run python "$GT/scripts/workflow_lint.py" \
      > /tmp/issue-<N>-lint-baseline.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$BASE_RC" ]; then BASE_RC=$rc; fi; }
    timeout --kill-after=60s 900s uv run python "$GT/scripts/workflow_lint.py" \
      --check-references --check-tables --check-asks --check-autonomous-asks \
      >> /tmp/issue-<N>-lint-baseline.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$BASE_RC" ]; then BASE_RC=$rc; fi; }
    # PAYLOAD OVERLAY (#1212 — phase 2): branch-tip content for every
    # payload path; branch deletions AND rename SOURCES removed from the
    # landing tree. A DEDICATED --no-renames listing is used (never the
    # shared own-diff.txt, which is --name-only WITH rename detection and
    # lists only a rename's DESTINATION — the renamed-away source would
    # silently survive in the gated tree). --no-renames splits a rename
    # into D(old)+A(new), so one loop body covers both sides; own-diff.txt
    # and its attribution consumers are untouched. Do NOT "simplify" by
    # reusing own-diff.txt here. The two listings can also straddle the
    # fetch (own-diff.txt pre-fetch, this one post-fetch): benign — the
    # three-dot merge-base is fork-point-stable, and any attribution-vs-
    # overlay divergence falls to the NEW-set arm (blocks, never fail-open).
    # The overlay copies the FULL own-diff incl. artifact paths — harmless
    # to the verdict (lint ignores non-cone paths); costs scale with payload.
    # quotePath=false on this + the sibling literal-path producers (#1268 —
    # own-diff, guard1/recovery tasks-diffs, additive-files, Guard 3's
    # own-commit diff): default quoting wraps a non-ASCII path in `"..."`
    # escapes, which fails every literal consumer (`git show "HEAD:$p"`,
    # cat-file/checkout/rm pathspecs, xargs, --map-files) AND every anchored
    # `^tasks/...` carve-out grep — silent skips, the #458/#1147 fail-open
    # class. ASCII output is byte-identical under the flag. Deliberately NOT
    # flagged (quoting-immune consumers): the postmerge ls-tree listings
    # (match `^tasks/<status>/<N>$` directory names — ASCII by construction),
    # the figures ls-tree (`grep -q .` non-emptiness), and the new-shared-src
    # guard (src/ module paths, pinned byte-untouched). Control-char
    # filenames (newline/tab) stay quoted regardless — the flag covers
    # bytes >0x7f only.
    git -C "$WT" -c core.quotePath=false diff --name-only --no-renames origin/main...HEAD \
      > /tmp/issue-<N>-overlay-files.txt || GT_RC=1
    while IFS= read -r p; do
      if git -C "$WT" cat-file -e "HEAD:$p" 2>/dev/null; then
        { mkdir -p "$GT/$(dirname "$p")" \
          && git -C "$WT" show "HEAD:$p" > "$GT/$p"; } || GT_RC=1
      else
        rm -f "$GT/$p" || GT_RC=1   # branch-deleted / renamed-away path: absent from the landing tree
      fi
    done < /tmp/issue-<N>-overlay-files.txt
    # GATED legs (payload-bearing landing tree — phase 3; parity leg covers
    # the checks the no-flags bundle omits — see the bullet above):
    GATED_RC=0
    timeout --kill-after=60s 900s uv run python "$GT/scripts/workflow_lint.py" \
      > /tmp/issue-<N>-lint-gated.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$GATED_RC" ]; then GATED_RC=$rc; fi; }
    timeout --kill-after=60s 900s uv run python "$GT/scripts/workflow_lint.py" \
      --check-references --check-tables --check-asks --check-autonomous-asks \
      >> /tmp/issue-<N>-lint-gated.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$GATED_RC" ]; then GATED_RC=$rc; fi; }
    # MAPPED INVARIANT-TEST LEG (#1147). GLOB_SCAN_TESTS-covered payloads
    # (scripts/issue*_*.py, dispatcher scripts) land via this gate with ZERO
    # pytest on the experiment auto-merge path (Step 9c is code-change-kinds-
    # only) — #1144: 34 thread-caps offenders accreted this way. Map the
    # own-diff to its scanning tests via the selector's single-source map;
    # empty map => leg skipped (no pytest run).
    TG_RC=0; TG_BASE_RC=0; TG_CRASH=no
    : > /tmp/issue-<N>-tg-new.txt
    if ! timeout --kill-after=30s 120s uv run python "$REPO_ROOT/scripts/select_step9c_tests.py" \
        --map-files /tmp/issue-<N>-own-diff.txt --repo-root "$WT" \
        > /tmp/issue-<N>-tg-map.txt 2>/tmp/issue-<N>-tg-map-err.txt; then
      TG_CRASH=yes   # helper failure: cannot classify the payload — fail CLOSED
    fi
    if [ "$TG_CRASH" = no ] && [ -s /tmp/issue-<N>-tg-map.txt ]; then
      # matched payload paths (attribution grep list) + gated test list:
      cut -f2 /tmp/issue-<N>-tg-map.txt | sort -u > /tmp/issue-<N>-tg-files.txt
      mapfile -t TG_TESTS < <(cut -f1 /tmp/issue-<N>-tg-map.txt | sort -u)
      # BASELINE leg — root copy on the payload-free main tree (each scan
      # test derives its scan root from its own __file__, so the root copy
      # scans the root tree). Only tests present on the baseline tree run
      # there: a branch-NEW scan test has no baseline, so its gated hits are
      # NEW by construction (correct — block).
      mapfile -t TG_BASE_TESTS < <(timeout --kill-after=30s 120s uv run python \
        "$REPO_ROOT/scripts/select_step9c_tests.py" \
        --map-files /tmp/issue-<N>-own-diff.txt --repo-root "$REPO_ROOT" \
        2>/dev/null | cut -f1 | sort -u)
      if [ "${#TG_BASE_TESTS[@]}" -gt 0 ]; then
        ( cd "$REPO_ROOT" && timeout --kill-after=30s 300s \
          env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
              NUMEXPR_NUM_THREADS=8 \
          uv run pytest "${TG_BASE_TESTS[@]}" -q -p no:cacheprovider ) \
          > /tmp/issue-<N>-tg-baseline.txt 2>&1 || TG_BASE_RC=$?
      else
        : > /tmp/issue-<N>-tg-baseline.txt
      fi
      # GATED leg — worktree copy on the payload-bearing branch-tip tree
      # (deliberately NOT the #1212 gate tree — see the mapped-leg residuals):
      ( cd "$WT" && timeout --kill-after=30s 300s \
        env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
            NUMEXPR_NUM_THREADS=8 \
        uv run pytest "${TG_TESTS[@]}" -q -p no:cacheprovider ) \
        > /tmp/issue-<N>-tg-gated.txt 2>&1 || TG_RC=$?
      # rc 0 = green, 1 = test failures (attributable); ANY other rc
      # (timeout 124, collection/internal/usage error 2-5) = crash-class.
      if [ "$TG_RC" -gt 1 ] || [ "$TG_BASE_RC" -gt 1 ]; then TG_CRASH=yes; fi
      # FILE-grain payload attribution: a scan test asserts per-file
      # invariants and aggregates every offender into ONE red node, so
      # node-level subtraction is degenerate (baseline-red node == gated-red
      # node masks a NEW offender). Attribution = output lines naming a
      # payload-matched path; line numbers blanked so main-vs-branch drift of
      # the SAME pre-existing offense cannot fake a NEW line. (The third sed
      # clause blanks test_subprocess_env_explicit.py's check-1
      # `- <path>:<ln> (fn=...)` format.) pytest's own `E   assert ...` repr
      # line is DROPPED: its ellipsis-truncated offender-list repr is
      # unstable across trees (unrelated dirt in ONE tree changes it ->
      # false NEW line on an innocent payload); every real offense also
      # emits its dedicated per-file evidence line, which survives.
      for leg in baseline gated; do
        grep -F -f /tmp/issue-<N>-tg-files.txt "/tmp/issue-<N>-tg-$leg.txt" \
          | grep -vE '^E +assert ' \
          | sed -E 's/at line [0-9]+/at line N/g; s/:[0-9]+:/::/g; s/:[0-9]+([^0-9]|$)/:N\1/g' \
          | sort -u \
          > "/tmp/issue-<N>-tg-$leg-hits.txt" || true
      done
      comm -23 /tmp/issue-<N>-tg-gated-hits.txt \
        /tmp/issue-<N>-tg-baseline-hits.txt > /tmp/issue-<N>-tg-new.txt
    fi
    # Normalize failure lines: keep per-error `workflow_lint: <err>` lines,
    # DROP the PASS / `FAIL (N error(s))` summary lines (their COUNT changes
    # even when the failure identities match — a payload that fixes one
    # pre-existing error must not false-block on a differing summary), and
    # blank `:<line>:` numbers so unrelated drift cannot fake a NEW line.
    # (WARNs never enter: workflow_lint emits them with a `WARN: ` prefix.)
    for leg in baseline gated; do
      grep -h '^workflow_lint: ' "/tmp/issue-<N>-lint-$leg.txt" \
        | grep -vE '^workflow_lint: (PASS$|FAIL \()' \
        | sed -E 's/:[0-9]+:/::/g' | sort -u \
        > "/tmp/issue-<N>-lint-$leg-norm.txt" || true
    done
    # NEW = gated_failures − baseline_failures (set subtraction):
    comm -23 /tmp/issue-<N>-lint-gated-norm.txt \
      /tmp/issue-<N>-lint-baseline-norm.txt > /tmp/issue-<N>-lint-new.txt
    # Gated failure lines naming a file IN the own-diff (materialized at the
    # trigger above):
    grep -F -f /tmp/issue-<N>-own-diff.txt /tmp/issue-<N>-lint-gated-norm.txt \
      > /tmp/issue-<N>-lint-owndiff.txt || true
    # VERDICT — CRASH ARM FIRST (fail CLOSED): a linter CRASH — rc>1 (import
    # error, missing dep, sparse-worktree crash), or rc!=0 with ZERO
    # normalized `workflow_lint:` failure lines across both legs' logs (an
    # uncaught Python exception exits 1 and emits none) — on EITHER leg pair
    # means the gate never produced a trustworthy compare. `crash` is an
    # unconditional block-path verdict (same epm:merge-failed handling as
    # `block`; Verdict bullet case 3) — NEVER `pass`. Only then the
    # attribution logic: a green gated run (exit 0) can never block; a red
    # one (rc=1 WITH lines) blocks only when payload-attributed (an
    # own-diff-named failure line OR a non-empty NEW set); rc=1 with lines
    # but none own-diff/NEW stays `pass` (pre-existing red — WARN).
    if [ "$GT_RC" -ne 0 ] || [ "$GATED_RC" -gt 1 ] || [ "$BASE_RC" -gt 1 ] || [ "$TG_CRASH" = "yes" ] \
       || { [ "$GATED_RC" -ne 0 ] && [ ! -s /tmp/issue-<N>-lint-gated-norm.txt ]; } \
       || { [ "$BASE_RC" -ne 0 ] && [ ! -s /tmp/issue-<N>-lint-baseline-norm.txt ]; }; then
      echo crash > /tmp/issue-<N>-lint-verdict.txt
    elif { [ "$GATED_RC" -ne 0 ] \
       && { [ -s /tmp/issue-<N>-lint-owndiff.txt ] || [ -s /tmp/issue-<N>-lint-new.txt ]; }; } \
       || { [ "$TG_RC" -ne 0 ] && [ -s /tmp/issue-<N>-tg-new.txt ]; }; then
      echo block > /tmp/issue-<N>-lint-verdict.txt
    else
      echo pass > /tmp/issue-<N>-lint-verdict.txt
    fi
    rm -rf "$GT"   # ephemeral; a crash-left tree is rebuilt (rm -rf first) on the next gate run
  else
    # Executable trigger (the Trigger bullet above): artifact-only payload —
    # both lint runs skipped by design.
    echo skip-artifact-only > /tmp/issue-<N>-lint-verdict.txt
  fi
  # SHA-BIND the verdict to the branch tip it certified (line 2, #1097): a
  # consumer accepts a pass/skip verdict ONLY while the CURRENT tip still
  # equals this sha — any new commit since certification invalidates it
  # (fail CLOSED, re-run the gate), and a hand-written verdict without the
  # correct sha is useless (anti-self-attestation, the #1082 incident).
  # Appended for every verdict; block/crash never certify, so their sha
  # line is inert.
  git -C "$WT" rev-parse HEAD >> /tmp/issue-<N>-lint-verdict.txt
  cat /tmp/issue-<N>-lint-verdict.txt   # line 1: pass | block | crash | skip-artifact-only; line 2: certified branch-tip sha
  ```

  **Completion-read (forms (i)/(ii)).** When the background gate call
  completes (the harness notifies), read the verdict in a fresh FOREGROUND
  call from the FILE. A MISSING verdict file means the background run died
  before writing a verdict (tool kill / watcher force-stop / wedge-bound
  kill) — treat as gate-not-run, fail CLOSED: NEVER proceed to the merge
  conditional, NEVER hand-write the verdict (#1082). Apply crash-fix-rounds
  § Kill-before-relaunch (probe `pgrep -af 'issue-<N>-lint-gate-tree'` —
  the gate-tree path in the lint legs' argv makes the probe
  exact-issue-scoped) before re-running the gate ONCE; still dying ->
  `epm:merge-failed v1` (Verdict bullet case 3). A partial death (killed
  between the verdict write and the sha append) leaves a 1-line file the
  binding sites' line-2 sha check already fails CLOSED on. Worst case —
  every bounded leg wedged — the call runs ~78 min, past the 60-min
  § Long-phase heartbeat boundary (rare; a watcher force-stop there is
  itself fail-closed: no verdict file gets written).

  ```bash
  if [ ! -f /tmp/issue-<N>-lint-verdict.txt ]; then
    echo "FATAL: verdict file missing — the background gate run died before writing a verdict. Kill-before-relaunch, then re-run the gate ONCE; NEVER record pass." >&2
  else
    cat /tmp/issue-<N>-lint-verdict.txt   # line 1: verdict; line 2: certified sha — the merge conditional below stays the hard stop
  fi
  ```

- **Gate earlyoom protection (#1045 recipe, #1211).** Both executable blocks
  (the shared form (i)/(ii) block above and the form (iii) surgical block)
  open with the SAME fail-open self-choom preamble as the Step 9c 1b/1c
  gates — `oom_score_adj` inherits across fork/exec (probe-verified), −600
  not −1000, FAIL-OPEN (`choom=failed` warns and the gate proceeds
  unprotected; the preamble never blocks a gate, never alters the verdict
  logic, and leaves the verdict-file contract byte-unchanged: line 1
  verdict, line 2 sha). Full calibration rationale: Step 9c § "Gate
  earlyoom protection (#1045)" — do not duplicate it here. Motivation:
  the lint legs (~4.5-6 min python) and TG pytest legs match this VM's
  earlyoom `--prefer` regex (#1143: first run died mid-lint as verdict
  `crash`; the protected re-run passed). Copy the echoed
  `[step10d] lint-gate earlyoom protection choom=...` breadcrumb line into
  the `epm:merged` / `epm:merge-failed` note (alongside the lint/tg tails
  those notes already record) so a crash-verdict post-mortem can tell a
  protected kill from an unprotected one.

- **Mapped invariant-test leg (#1147).** A second, trigger-gated leg of the
  SAME gate: when the payload (the own-diff / additive list) matches any
  `GLOB_SCAN_TESTS` glob, the executable block runs the MAPPED scan tests on
  the payload-bearing tree and subtracts a payload-free baseline run. The
  trigger is the helper map — `select_step9c_tests.py --map-files <list-file>
  [--repo-root <tree>]` prints `scan_test<TAB>matched_path` pairs; empty
  output = leg skipped entirely (zero pytest runs added). The helper is the
  SINGLE SOURCE of the glob→test mapping — never hardcode the globs in this
  file (the selector's drift pins in `tests/test_select_step9c_tests.py` keep
  the map current, #895). Attribution is FILE-grain, not junit-node grain: a
  scan test asserts per-file invariants and aggregates EVERY offender into
  ONE red node, so node-level subtraction is degenerate (baseline-red node ==
  gated-red node would mask a NEW offender — the same aggregation degeneracy
  that makes compare's node-identity strips of scan tests carry the MF-6
  masking WARN; compare additionally marks NON-file-anchored scan-set nodes
  scratch-ineligible (`step9c_baseline.py` `FILE_ANCHORED_SCAN_TESTS` members
  are scratch-resolved, still WARNed — #1337)). Hits
  = pytest-output lines naming a payload-matched path, line numbers blanked
  so main-vs-branch drift of the SAME pre-existing offense cannot fake a NEW
  line, pytest's ellipsis-truncated `E   assert ...` repr line dropped (its
  content is unstable across trees; every real offense also emits a dedicated
  per-file evidence line); NEW = gated hits − baseline hits (`comm -23`,
  `/tmp/issue-<N>-tg-new.txt`). Each pytest leg is bounded at 300 s
  (`timeout --kill-after=30s`; measured basis ~12.6 s for both mapped tests,
  2026-07-08) — a timeout / pytest rc>1 / helper failure on either leg is
  crash-class: verdict `crash`, fail CLOSED, the same "re-run the gate ONCE →
  `epm:merge-failed`" recovery as the lint legs (Verdict bullet case 3). On
  form (iii) this leg is structurally DORMANT today — the surgical additive
  pathspec set excludes `scripts/` / `src/`, so its trigger map is empty by
  construction; it arms automatically if that set ever grows. Known residuals
  (accepted, documented): (a) path-(i) test-VERSION drift — the gated leg
  runs the branch-tip copy of the scan test, so a check added on `main` after
  the branch forked is not enforced there (fail-safe direction; the LINT legs
  no longer share this residual — the #1212 gate tree runs the landing tree's
  lint on every path-(i) run; the TEST legs keep it because syncing
  individual test files without their import closure — conftest, tests/
  helpers — risks hybrid trees); (b) the baseline leg runs on the
  always-dirty shared root — dirt biases toward PASS, never a false block: an
  untracked concurrent-session file (including an untracked same-path draft
  of the payload file at the root, which the directory scan picks up
  tracked-or-not) can only ENLARGE the baseline hit set and mask, a residual
  formerly shared with the lint gate's baseline leg (the #1212 gate anchors
  both lint legs to origin/main trees, so the lint legs no longer carry it);
  (c) a payload that DEEPENS an
  offense in an already-red payload-touched file normalizes to the same
  per-file line and is subtracted — a false-pass window that vanishes once
  #1145 greens the baseline (the file is already post-freeze red; low harm).

- **Verdict — payload-attributed via failure-LINE-SET subtraction; NEVER
  blocks an innocent merge on pre-existing red.** Exit codes alone are
  vacuous when `main` is already red for an unrelated reason; attribution
  compares normalized `workflow_lint:` failure LINES (strip volatile
  prefixes; keep the `<check>/<file>[:<line>] <msg>` identity) between the
  GATED run and a payload-free BASELINE run — BOTH legs (no-flags + parity)
  in each run, so a pre-existing parity-leg red on main can never be
  misread as payload-caused (and vice versa). The executable block above
  computes the verdict (`block` | `pass` | `crash` | `skip-artifact-only`)
  and persists it SHA-BOUND to `/tmp/issue-<N>-lint-verdict.txt` (line 1 =
  verdict, line 2 = the certified branch-tip sha); the binding sites gate
  their merge/push/add commands on that FILE with an explicit conditional —
  a missing verdict file fails CLOSED (the gate has not run yet), and a
  pass/skip verdict certifies ONLY while the CURRENT branch tip equals the
  certified sha, so any new commit since certification fails CLOSED too
  (re-run the gate) and a hand-written verdict without the correct sha is
  useless (anti-self-attestation). The mapped invariant-test leg (#1147)
  contributes into this SAME verdict file: `crash` on either test leg's
  crash-class outcome (pytest rc>1 / timeout / helper failure), `block` on a
  payload-attributed NEW test hit (`/tmp/issue-<N>-tg-new.txt` non-empty with
  a red gated run); a gated-red-but-no-NEW-hit test outcome (pre-existing
  trunk red) stays `pass`, and the `epm:merged` WARN note records the tg tail
  alongside the lint tail. The file is REMOVED only once it can no
  longer certify anything: after a SUCCESSFUL `gh pr merge`
  (consume-on-merge-success), or in the block/crash/stale-sha branch (a
  fresh gate run regenerates it). A merge that fails for a NON-lint
  transport reason (the #1041 rebase-refusal → `--squash`-retry shape)
  therefore stays certified by the SAME gate run — never hand-recreate the
  verdict file (#1082). On a `block` (or `crash`) verdict:
  1. An own-diff-named gated failure line exists
     (`/tmp/issue-<N>-lint-owndiff.txt` non-empty) → the payload is the
     offender. Fix it in the worktree (the lint names file + rule),
     commit by explicit path, re-run the gate ONCE; still failing → post
     `epm:merge-failed v1` with `{reason: "pre-push workflow-lint gate",
     lint_tail: <last lines>}`, surface ONE line in chat, CONTINUE (same
     fail-fast policy as a merge failure; retried idempotently on the next
     `/issue <N>`).
  2. No own-diff-named line → the block came from
     `NEW = gated_failures − baseline_failures` (`comm -23` on the
     normalized lines, persisted at `/tmp/issue-<N>-lint-new.txt`) — a
     payload-caused cross-file interaction (e.g. a lessons-index /
     lens-coverage check naming the index rather than the added rule
     file) — treat as case 1 (block). NEW empty with no own-diff-named
     line never blocks: the executable block writes `pass` — pre-existing
     red is a WARN (record the lint tail in the `epm:merged` note) and
     the merge PROCEEDS. The baseline and gated runs execute back-to-back
     inside the ONE background gate call, so a concurrent merge cannot
     widen the compare window
     (moving-main race — keep the window tight, preserve the
     main-already-red detail in the marker; the #1212 gate additionally
     freezes both legs to one archived origin/main snapshot, removing the
     inter-leg race by construction — the back-to-back advice stays as
     defense-in-depth).
  3. `crash` — the linter itself CRASHED on either leg pair (rc>1, or
     rc!=0 with zero normalized `workflow_lint:` failure lines: import
     error, missing dep, sparse-worktree crash — the gated leg runs the
     gate tree's `workflow_lint.py`, the BRANCH's copy whenever the
     own-diff touches it, so the crash is payload-inducible; a gate-tree
     CONSTRUCTION failure (GT_RC != 0) also lands here),
     or the trigger diff failed. No trustworthy compare exists, so this is
     an unconditional block-path verdict: fix the crash cause in the
     worktree, re-run the gate ONCE; still crashing → the SAME
     `epm:merge-failed v1` handling as case 1. Never merge/push on `crash`.
- **Baseline semantics per binding form (the baseline is ALWAYS a
  payload-free tree).** The mapped invariant-TEST legs (#1147) keep the
  ORIGINAL per-form placement (gated = the `$WT` copy on the branch-tip /
  post-merge tree, baseline = the root copy on forms (i)-(ii); root copy
  both legs on form (iii)); the LINT legs on forms (i)/(ii) now run the
  #1212 gate tree. (i) Safe case: LINT legs — gated = the gate-tree copy
  on the LANDING tree (origin/main + own-diff overlay), baseline = the
  SAME copy on the payload-free landing base (#1212); mapped-TEST legs —
  gated = the `$WT` copy on the branch-tip tree, baseline = the repo-root
  copy (unchanged); bind immediately before `gh pr ready` / `gh pr merge`.
  (ii) Merge-conflict recovery: LINT legs — gated/baseline = the gate tree
  rebuilt from the post-merge tip (content-identical to the post-merge
  worktree, which carries main's CURRENT lint — the ideal gate point);
  mapped-TEST legs — gated = the post-merge worktree copy, baseline = the
  repo-root copy (unchanged); bind after conflict resolution + targeted
  tests, before `git -C "$WT" push`. (iii) Surgical additive checkout: the payload lands
  in the ROOT tree, so the BASELINE MUST RUN BEFORE the
  `xargs ... git checkout` — a post-checkout "main-side" run would re-lint
  the SAME contaminated tree, a degenerate compare that fails open at
  exactly the fast-path form; sequence = baseline (root copy, both legs) →
  checkout → gated (root copy, both legs) → set-subtraction verdict → on
  pass, `git add`. The whole sequence runs as ONE BACKGROUND Bash
  invocation — do NOT split it across invocations: the contaminated-root
  window (checkout → stage/commit-or-clean) stays compute-bound (~5-6 min)
  only while the sequence runs in one shell, and a split inserts
  orchestrator turn-boundary latency inside that window. While it runs,
  end the turn and run no repo-root-mutating commands until the
  completion-read (surgical block below). On a block at (iii), clean the payload out of BOTH
  index and working tree with the hook-VERIFIED two-step (run from
  `$REPO_ROOT`; simulated against `scripts/guard_repo_root_branch.sh`
  2026-07-05 — the one-shot restore invocation carrying `--staged` PLUS a
  worktree flag is mechanically BLOCKED by its #897 restore detector, whose allow
  arm requires `--staged` AND no worktree flag, and the hook's own
  guidance bans pointing `-C` at the repo root for a DESTRUCTIVE op):
  first `xargs -r -a /tmp/issue-<N>-additive-files.txt git -C "$REPO_ROOT"
  restore --staged --` (index-only unstage — non-destructive, admitted by
  the restore allow-arm), then `xargs -r -a
  /tmp/issue-<N>-additive-files.txt rm -f --` (the paths are A-only,
  absent from `main`, and untracked after the unstage — the plain `rm`
  destroys no main state; a bare `rm -f` WITHOUT the unstage would leave
  them STAGED in the shared root index, polluting concurrent sessions'
  `git diff --cached` echoes). The `xargs -r` (`--no-run-if-empty`) on
  every additive-list consumer is load-bearing: on an EMPTY list a
  flag-less xargs still runs its command once with NO pathspec. The fast
  path routes through (iii). Idempotent: re-entry just re-runs the gate.
  If the repo-root guard hook blocks an improvised unqualified variant of
  the restore line (or of the additive-checkout consumer), the WHOLE
  compound was skipped — including any clause that wrote
  `/tmp/issue-<N>-additive-files.txt`; regenerate the list, then retry
  the verbatim `-C "$REPO_ROOT"` forms (full recovery contract: the
  guard-block paragraph after the surgical-additive-checkout executable
  block below).
- **Known residuals (accepted, documented):** the #1212 gate tree removed
  the old path-(i) vintage residual for the LINT legs — a check ADDED on
  `main` after the branch forked is now enforced on every path, so a
  payload violating it BLOCKS (the #931 class), and a check
  retired/loosened on main can no longer false-block. What remains: (a) a
  branch whose OWN diff touches `scripts/workflow_lint.py` runs its branch
  copy on the gated legs (it IS the payload) — baseline-vs-gated
  lint-version asymmetry is inherent there and resolves through the
  standard case-1 fix-or-`epm:merge-failed` path; (b) the gate tree
  materializes only `workflow_lint.py`'s scan/target surface (the archive
  pathspec set in the executable block) — if the linter grows a new scan
  root, a gated false block naming paths outside that set is the symptom
  and extending the set is the fix (the #1154 `docs/` pins are the
  precedent); (c) the mapped invariant-TEST legs keep the branch-tip test
  copies and the dirty-root baseline (path-(i) test-VERSION drift,
  fail-safe direction) — the trunk pytest remains their backstop; (d) a
  lint-scanned file BOTH main and the branch modified post-fork lints as
  the BRANCH's copy (the overlay takes branch-HEAD wholesale), a narrow
  window in either direction that is strictly smaller than the old
  whole-tree divergence — the form-(ii) recovery covers the conflict case
  and trunk lint backstops the clean-rebase case; (e) same-issue
  concurrent gate runs would share one `$GT` (a phase-flip race) —
  excluded by the Step 0 single-orchestrator guard + the pre-dispatch
  dedup, with the #911 janitor reaping any crash leftovers.

#### The auto-merge procedure (safe case: guard 3 clean — mainline-based, own commits in scope)

```bash
PR=$(gh pr view <PR> --json number -q .number 2>/dev/null) || true
if [ -z "$PR" ]; then
  echo "No PR for issue-<N>; nothing to merge."   # skip; post nothing
else
  # Run guards 1-3 above first. If guard 3 says "unsafe", skip this
  # block and run the artifact-confirmed merge below instead.
  #
  # Push the Guard-1 strip commit to the PR head ref FIRST, so the
  # server-side merge in `gh pr merge` below (rebase replay or squash)
  # sees the stripped branch tip,
  # not the pre-strip commit. The strip commit is a LOCAL worktree commit and
  # is otherwise invisible to the server-side merge — leaving the foreign
  # tasks/* reverts in the replayed history and landing them on main silently
  # (Codex code-review round-1 blocker, task #787). Push retry mirrors
  # CLAUDE.md § "Concurrent repo-root committers": pull --rebase=merges
  # --autostash then re-push on a rejected push. The Guard-0 agent-memory
  # persist commit is equally local-only — both must reach the PR head ref
  # before the server-side rebase.
  # (WORKTREE-scoped: `git -C "$WT"` on the issue branch. scripts/sync_repo_root.py
  # does NOT apply here — it is repo-root-only by design, preconditioned on
  # HEAD == main, exit 5 otherwise.)
  #
  # The push condition RE-DERIVES "unpushed local commits exist" from git
  # state (rev-list against origin/issue-<N>) instead of trusting the
  # STRIPPED_FOREIGN / MEM_COMMITTED flags alone: fenced bash blocks are
  # SEPARATE shell invocations, so a flag assigned in Guard 0/1 is unset
  # here (and would silently skip the second-chance push); the git-state
  # read also survives a crash + re-entry, and covers BOTH the strip commit
  # and the memory commit in one predicate. The flags stay as same-block
  # conveniences only (they still short-circuit true when guards and merge
  # happen to run in one shell). A missing / unresolvable origin/issue-<N>
  # ref counts as unpushed (`|| echo 1` — fails toward pushing, the safe
  # direction; a redundant push is a no-op "Everything up-to-date").
  if [ "$(git -C "$WT" rev-list --count origin/issue-<N>..HEAD 2>/dev/null || echo 1)" -gt 0 ] \
     || [ "$STRIPPED_FOREIGN" = "yes" ] || [ "$MEM_COMMITTED" = "yes" ]; then
    # Run every push / gh pr command BARE — never piped through tail/grep/head
    # (guard_piped_git_push.sh blocks the pipe; a pipe masks the exit code).
    # This applies to IMPROVISED recovery commands too, not just this snippet.
    git -C "$WT" push origin issue-<N> \
      || { git -C "$WT" pull --rebase=merges --autostash \
           && git -C "$WT" push origin issue-<N>; }
  fi
  # Merge-form routing (#1288): infra-fleet code branches (kind infra|batch —
  # the watcher's INFRA_DRAIN_KINDS, the population same-batch racing this
  # step by construction) default to --squash: server-side --rebase was 0/4
  # first-try on 2026-07-12 (10/24 sessions on 07-11) under fleet churn, and
  # every failed session landed on --squash anyway after burning the failed
  # rebase + its retry ladder. GitHub mergeability is merge-method-
  # independent, but --rebase can ADDITIONALLY fail ("can't be rebased",
  # #1041) where --squash succeeds — so squash-first strictly dominates for
  # a single-logical-change branch, and it reverts as ONE commit (the only
  # grain that exists on such a branch). Experiments (Step 9b trigger) keep
  # --rebase: heterogeneous per-item commits retain per-commit revert value
  # and the empirical record does not cover them. An unreadable kind falls
  # to --rebase (fail-open to today's behavior). REPO_ROOT is re-derived
  # inline — fenced blocks are separate shells, and the guards block's
  # derivation is not in scope here:
  REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
  MERGE_FORM=--rebase
  TASK_KIND=$(uv run python "$REPO_ROOT/scripts/task.py" view <N> --json \
    | uv run python -c 'import sys,json; print(json.load(sys.stdin).get("frontmatter",{}).get("kind",""))' \
    || echo "")
  case "$TASK_KIND" in infra|batch) MERGE_FORM=--squash ;; esac
  # Pre-push workflow-lint gate (subsection above) — run its executable
  # block FIRST as ONE BACKGROUND Bash call, read the verdict file in a
  # fresh foreground call when it completes (completion-read, gate
  # subsection), then gate the merge on the PERSISTED, SHA-BOUND verdict
  # file: the explicit conditional below is the hard stop. Fails CLOSED on
  # a missing file (gate not run), a block/crash verdict, OR a missing /
  # stale sha (line 2 empty or != current tip: a hand-written verdict, or
  # new commits since certification — re-run the gate). The verdict is
  # consumed (rm) only AFTER `gh pr merge` SUCCEEDS: a non-lint transport
  # failure (#1041 rebase refusal) leaves it valid for the same-tip retry
  # — never hand-write the verdict file (#1082).
  if grep -qxE 'pass|skip-artifact-only' /tmp/issue-<N>-lint-verdict.txt 2>/dev/null \
     && [ -n "$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt 2>/dev/null)" ] \
     && [ "$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt 2>/dev/null)" = "$(git -C "$WT" rev-parse HEAD)" ]; then
    gh pr ready <PR>
    if gh pr merge <PR> $MERGE_FORM --delete-branch=false; then
      rm -f /tmp/issue-<N>-lint-verdict.txt   # consume on MERGE SUCCESS — the verdict certified exactly the tip that landed
    else
      echo "MERGE FAILED — classify the gh error text: (0) \"Base branch was modified\" -> transient base-advance (Known failure shape 0 below): wait ~20s, re-enter this SAME conditional (same tip, verdict still valid; max 2 same-tip retries); (1) \"can't be rebased\" (--rebase form only) -> the #1041 --squash retry (Known failure shape 1 below; SHA-bound verdict remains valid for the SAME tip); (2) \"Pull Request has merge conflicts\" -> the #1128 re-snapshot-and-retry-once (Known failure shape 2 below); (3) anything else -> the Failure bullet (merge-conflict recovery ONCE, then epm:merge-failed). Do NOT hand-write the verdict file."
      false
    fi
  else
    echo "BLOCKED: pre-push workflow-lint gate (verdict: $(cat /tmp/issue-<N>-lint-verdict.txt 2>/dev/null || echo not-run)) — missing verdict, block/crash, or missing/stale sha (hand-written verdict, or new commits since certification) all fail CLOSED: fix the named offender (or crash cause), re-run the gate ONCE; still failing -> epm:merge-failed (gate subsection, verdict cases 1/3). Do NOT merge."
    rm -f /tmp/issue-<N>-lint-verdict.txt   # block/crash/stale consumed — a fresh gate run regenerates it
    false
  fi
fi
```

The `gh pr merge --rebase` form lands all per-item commits individually
on `main`; each is independently revertible via `git revert <sha>` run in
a scratch worktree (the root guard blocks a repo-root revert, #1234) (vs.
`--merge`, which reverts everything together). The user retains full
revert control after the fact — that is what makes a no-prompt merge safe
here. The worktree is deliberately NOT removed (`--delete-branch=false`,
no `git worktree remove`).

For `kind: infra|batch` branches `$MERGE_FORM` is `--squash` (#1288):
the branch is a single logical change by construction, the squash lands
it as ONE independently-revertible commit, and the empirical record
(0/4 first-try rebases 2026-07-12; 10/24 sessions 07-11; every failure
landing on --squash anyway) makes the rebase attempt a pure wall-time
tax under fleet churn. Shape 1 cannot fire on the --squash path (the
error is rebase-specific); shapes 0/2/else apply to both forms.

**Known failure shape 0 — base branch advanced mid-merge (`Base branch
was modified`, #1288).** Substring-match `Base branch was modified` (the
full wording — `Base branch was modified. Review and try the merge
again.` — is transcript-mined and may drift). GitHub recomputed the
merge against a base that moved DURING the API call — a pure timing
transient under fleet marker churn (~100+ tasks/ commits/hr on main):
no content conflict, nothing to fix. Recovery: wait ~20 s (≈ one churn
interval, letting gh's mergeability recompute settle), then re-enter
the SAME gated merge conditional with the SAME `$MERGE_FORM` — the
branch tip is unchanged, so the SHA-bound verdict re-certifies it
(consume-on-merge-success survives this failure by design; never
hand-write the verdict file, #1082). Bounded at TWO same-tip retries
per Step 10d invocation; a third consecutive hit is no longer plausibly
timing — reclassify by error text per shapes 1/2/else. Before #1288
this shape fell through to "(3) anything else" and burned a full
~16-min scratch-worktree recovery on a transient (one of the three
error shapes in the 2026-07-12 fleet's 4/4 first-attempt failures).

**Known failure shape 1 — branch carries a merge commit (`can't be
rebased`, #1041).** A branch that CARRIES A MERGE COMMIT (e.g. after a
conflict-resolution merge of `main` into the branch) cannot be
server-side rebased — `gh pr merge --rebase` fails with
`GraphQL: This branch can't be rebased`. The working recovery is
`gh pr merge <PR> --squash --delete-branch=false` (acceptable for a
single-logical-change branch; the squash loses per-commit revert
granularity, which the merge commit already compromised). (Incident
#1041 PR #801, 2026-07-05.)
The SHA-bound verdict file SURVIVES this failure by design
(consume-on-merge-success): run the squash retry through the SAME gated
conditional (substituting `--squash` for `--rebase`) so the still-valid
verdict re-certifies the identical tip and the `rm` fires on success.
Never recreate the verdict file by hand — a hand-written verdict lacks
the certified sha and fails closed anyway (#1082's
`echo pass > /tmp/issue-<N>-lint-verdict.txt` is the banned move).

**Known failure shape 2 — mergeability conflict under fleet marker churn
(error text containing `Pull Request has merge conflicts`, #1128).**
Classify by SUBSTRING, never the exact GraphQL line (the full
`GraphQL: Pull Request has merge conflicts (mergePullRequest)` wording is
transcript-mined and may drift). Between the Guard-1 snapshot and the
server-side rebase, `main` advances (~100+ `tasks/` marker commits/hr),
so the strip commit's snapshot replays stale and conflicts. Recovery:
re-snapshot against a freshly captured `main` SHA and retry ONCE —
documented, never silent, and gated on the re-snapshot actually changing
something (an unchanged tip would fail identically; go straight to the
merge-conflict recovery instead). NOTE the same error text ALSO fires for
non-`tasks/` conflicts (overlapping workflow-surface edits, binary
`figures/` collisions — #697/#597) that a re-snapshot cannot fix: the
skip-predicate fall-through is the EXPECTED path there, not a
malfunction. Likewise when an ORDINARY branch commit itself touched
foreign `tasks/` at stale content, the re-snapshot cannot fix that
commit's replay — the fall-through to the merge-conflict recovery below
covers it. Even a fresh snapshot can go stale in the seconds between the
fetch and the server-side merge — this recipe bounds and mechanizes
recovery; it does not eliminate the race.

```bash
# Re-snapshot-and-retry (ONCE per Step 10d invocation) — fires ONLY on
# the mergeability-conflict shape above.
# STEP 1 (own Bash call): persist the pre-resnapshot tip to a FILE —
# fenced blocks are separate shell invocations, so a bare variable would
# not survive to step 3 (the Guard-1 diff-file / lint-verdict pattern):
git -C "$WT" rev-parse HEAD > /tmp/issue-<N>-resnapshot-tip.txt
# STEP 2: re-run the ENTIRE Guard-1 block above VERBATIM: it re-fetches,
# captures a fresh MAIN_SHA, re-pins the foreign paths, and commits only
# if anything changed (idempotent).
# STEP 3 (own Bash call): retry ONLY if the re-snapshot changed the
# branch tip OR unpushed commits exist (same rev-list re-derivation as
# the safe-case push; missing ref counts as unpushed). The retry sits in
# the else arm so the skip arm ENDS the block — the skip must never fall
# through into the push:
TIP_BEFORE=$(cat /tmp/issue-<N>-resnapshot-tip.txt)
if [ "$(git -C "$WT" rev-parse HEAD)" = "$TIP_BEFORE" ] \
   && [ "$(git -C "$WT" rev-list --count origin/issue-<N>..HEAD 2>/dev/null || echo 1)" -eq 0 ]; then
  echo "re-snapshot changed nothing (tip unchanged, nothing unpushed) — a retry would fail identically; record resnapshot_retry outcome: skipped and run the merge-conflict recovery below"
  false
else
  git -C "$WT" push origin issue-<N> \
    || { git -C "$WT" pull --rebase=merges --autostash \
         && git -C "$WT" push origin issue-<N>; }
  # gh recomputes mergeability ASYNCHRONOUSLY after a push (the recovery
  # block's own precedent) — re-check before the retried merge so a
  # stale mergeability read cannot burn the single retry:
  gh pr view <PR> --json mergeable -q .mergeable   # brief wait/retry until MERGEABLE
fi
# If the tip changed, the SHA-bound lint verdict is now STALE and the
# gated conditional would fail CLOSED: re-run the executable Pre-push
# workflow-lint gate block (subsection above) so the verdict re-binds to
# the NEW tip (never hand-write it, #1082). If the tip did NOT change
# (push-only fix), the still-valid verdict re-certifies it — the
# conditional's sha arm enforces this mechanically either way. Then
# re-enter the SAME gated merge conditional (the task's $MERGE_FORM) exactly
# once. Classify a SECOND refusal by its error text per the failure
# echo: a "can't be rebased" refusal takes the shape-1 --squash retry
# (the retried rebase replays the FIRST, stale strip commit per-commit
# and can surface as shape 1 even after a clean re-snapshot; the squash
# is the endpoint merge that ends the chain); any OTHER second refusal
# falls through to the merge-conflict recovery below. Record the
# outcome either way in the epm:merged / epm:merge-failed note:
#   resnapshot_retry: {tip_before: <TIP_BEFORE>, main_sha: <fresh MAIN_SHA>,
#                      stripped_again: yes|no, outcome: merged|refused|skipped}
```

(If the re-run Guard-1 created NO new commit but unpushed commits existed
— e.g. a crash between an earlier strip and its push — the push alone can
fix the server-side view and the retry is warranted; the tip is then
unchanged, so the still-valid SHA-bound verdict re-certifies it and no
gate re-run is needed.)

- **Success:** post `epm:merged v1` with the list of merge SHAs plus
  `merge_form: squash|rebase` and `merge_attempts: <n>` (note-token
  convention — no schema change, #1288). Update
  the chat title with `merged`. Then run the **post-merge stale-task-folder
  guard** below (it runs on every merge form).
- **Failure** (rebase conflict, non-mergeable PR, non-fast-forward): for
  the `Base branch was modified` shape (substring match), run the
  shape-0 wait-and-retry (Known failure shape 0 above, max 2) FIRST; for
  the `Pull Request has merge conflicts` shape (substring match), FIRST
  run the **re-snapshot-and-retry** (Known failure shape 2 above) ONCE;
  if it is skipped (nothing changed), run the **merge-conflict recovery**
  sub-procedure below ONCE; if the retried merge is refused AGAIN,
  re-classify that second refusal by error text — a `can't be rebased`
  refusal takes the shape-1 `--squash` retry, anything else runs the
  **merge-conflict recovery** ONCE. For any other first refusal, run the
  **merge-conflict recovery** sub-procedure below ONCE directly.
  If the recovery itself fails or the retried merge is still refused:
  do NOT swallow it (fail-fast). Post `epm:merge-failed v1` with the
  `gh` / `git` error, surface ONE line in chat naming the branch +
  worktree path for manual resolution, and CONTINUE — an experiment
  still parks at `awaiting_promotion`; a code-change task still
  completes. The merge is retried (idempotently) on the next
  `/issue <N>` re-invocation.
- **Autonomous mode** (no user present): same as above — the auto-merge
  proceeds. No deferral. (This reverses the prior "default NO" autonomous
  behavior; merge to `main` is no longer user-gated.)

#### Merge-conflict recovery (safe case: `gh pr merge` refuses)

When the safe-case merge is refused on mergeability (a REAL conflict —
`main` and the branch both changed the same lines), do NOT hand-resolve
in the shared repo root and do NOT force-push. Recover IN THE WORKTREE
(worked example: #598 / PR #454, 2026-06-12 — both sides appended a new
checklist item to `.claude/agents/experimenter.md`; resolved in the
worktree, 210 targeted tests re-run, merged on retry):

```bash
git -C "$WT" fetch origin main --quiet
# Capture the snapshot ONCE, immediately after the fetch, and merge THAT
# SHA — origin/main is a shared ref a concurrent session's fetch can
# advance between these commands (#1128's shared-ref race).
MAIN_SHA=$(git -C "$WT" rev-parse origin/main)
git -C "$WT" merge "$MAIN_SHA"          # conflicts surface HERE, in the worktree
# Run that merge BARE — never piped through tail/grep (hook-blocked, #1048;
# 9 sessions re-tripped the hook mid-recovery on 07-09/07-10). To capture
# output, file-redirect it: `git -C "$WT" merge "$MAIN_SHA" > /tmp/issue-<N>-merge.log 2>&1; MERGE_RC=$?`.
# Foreign tasks/ conflicts are resolved MECHANICALLY: take the captured
# snapshot's version wholesale (under fleet marker churn main is
# authoritative for OTHER tasks' state — the #1128-proven recovery:
# foreign tasks/ pinned to ONE captured main SHA). Materialize the
# conflicted-path list and check its own exit code in Guard 1's `if ! ...`
# exclusive-arm shape (#1184): a FAILED producer takes the terminal
# echo + false arm and the work arm is STRUCTURALLY unreachable — the
# old `|| { echo; false; }` form reported failure but let the next
# command run under no-set-e / piecewise execution (#1243).
if ! git -C "$WT" -c core.quotePath=false diff --name-only --diff-filter=U -- 'tasks/' \
    > /tmp/issue-<N>-recovery-foreign.txt; then
  echo "recovery: conflicted-paths diff FAILED — resolve by hand per the prose below"
  false
# Work arm: two-command elif list — mapfile fills RECOVERY_FOREIGN from the
# FILE (the carve-out grep's no-match `|| true` is a legitimate empty
# list), then the [ ... ] test (the LAST command's exit) decides the
# branch. The mapfile + non-empty-array idiom is Guard 1's own hook-proven
# shape (the guard_repo_root_branch.sh -C waiver expects -C right after
# git; no xargs indirection, no whitespace-splitting caveat); the length
# test means an empty list never runs a pathspec-less checkout. On an
# empty list no branch is taken and the unit exits 0 — deliberate
# post-merge-guard parity, not drift (the old `[ ... ] && checkout`
# tail exited 1 there).
# Discriminate on-main vs gone-on-main (Guard 1's own cat-file split): task
# folders MOVE on every status change, so a foreign conflicted path absent
# at $MAIN_SHA is ROUTINE, not rare (#1242 13:37Z / #1246 14:43Z re-derived
# this by hand). checkout <sha> -- <path> resolves each ON-MAIN U path to
# the snapshot's version and stages it; a GONE-ON-MAIN path (moved/deleted
# on main) is resolved as a REMOVAL — main is authoritative for foreign
# tasks/ state, and git rm -f also resolves the unmerged index entries.
elif mapfile -t RECOVERY_FOREIGN < <(grep -Ev "^tasks/[^/]+/<N>/" \
      /tmp/issue-<N>-recovery-foreign.txt || true); [ "${#RECOVERY_FOREIGN[@]}" -gt 0 ]; then
  RECOVERY_ON_MAIN=()        # exist at MAIN_SHA -> take the snapshot's version
  RECOVERY_GONE_ON_MAIN=()   # absent at MAIN_SHA (moved/deleted on main) -> remove
  for p in "${RECOVERY_FOREIGN[@]}"; do
    if git -C "$WT" cat-file -e "$MAIN_SHA:$p" 2>/dev/null; then
      RECOVERY_ON_MAIN+=("$p")
    else
      RECOVERY_GONE_ON_MAIN+=("$p")
    fi
  done
  # if-form, not `[ ] && cmd` tails: an empty second list must not exit the
  # unit 1 (the documented exit-0 empty-list parity above).
  if [ "${#RECOVERY_ON_MAIN[@]}" -gt 0 ]; then
    git -C "$WT" checkout "$MAIN_SHA" -- "${RECOVERY_ON_MAIN[@]}"
  fi
  # git rm -f, NOT --cached: this resolution commit is `git commit --no-edit`
  # with NO pathspec (index governs), so --cached would technically survive
  # (#1244's resurrection needs a pathspec-limited commit) — -f is chosen for
  # Guard-1 parity and to leave no stale working-tree litter behind.
  if [ "${#RECOVERY_GONE_ON_MAIN[@]}" -gt 0 ]; then
    git -C "$WT" rm -f --ignore-unmatch -- "${RECOVERY_GONE_ON_MAIN[@]}"
  fi
fi
# THIS task's own tasks/*/<N>/ conflicts and all non-tasks/ conflicts:
# resolve in the worktree (keep main's version of anything outside this
# task's deliverables), then:
git -C "$WT" add <each resolved file>
git -C "$WT" commit --no-edit
# Post-resolution certification (the #1128 verification): the branch tree
# must now be IDENTICAL to the captured snapshot over tasks/, modulo this
# task's own folder. ONE fused if/elif chain (Guard 1's `if ! ...` shape,
# #1184/#1243): the verification diff and the residual-foreign check are
# one logical certification — under the old `|| { echo; false; }` form a
# FAILED diff left the verify file EMPTY (the redirect truncates before
# the command runs), the residual grep then found nothing, and
# certification passed VACUOUSLY (fail-OPEN into the push). Here a
# failed producer takes the terminal arm and the residual check is
# structurally unreachable:
# Two-endpoint ("$MAIN_SHA" HEAD) DELIBERATELY, not Guard 1's three-dot
# (#1280): this certifies TREE IDENTITY against the captured snapshot AFTER
# the merge brought MAIN_SHA's content in — both endpoints fixed, so
# main-side advancement cannot false-positive here, and the form stays
# correct even when the merge produced no commit. Guard 1's PRE-merge
# trigger is the site where two-endpoint misread main advancement as
# foreign touches.
if ! git -C "$WT" -c core.quotePath=false diff --name-only "$MAIN_SHA" HEAD -- 'tasks/' \
    > /tmp/issue-<N>-recovery-tasks-verify.txt; then
  echo "recovery: tasks/ verification diff FAILED — do NOT push"
  false
elif grep -Ev "^tasks/[^/]+/<N>/" /tmp/issue-<N>-recovery-tasks-verify.txt | grep -q .; then
  echo "recovery: foreign tasks/ still differ from the captured main snapshot — do NOT push; re-pin the listed paths (checkout the on-main, git rm -f the gone-on-main) to \$MAIN_SHA and re-verify"
  false
fi
# Re-run the targeted tests for the touched surface AND the executable
# Pre-push workflow-lint gate block (subsection above; gated = the gate
# tree rebuilt from this post-merge tip (origin/main + the post-merge
# own-diff — content-identical to this post-merge worktree, which carries
# main's CURRENT lint — the ideal gate point); the gate re-run SHA-binds
# the verdict to THIS post-merge tip. Re-run it as ONE BACKGROUND Bash
# call with the fresh foreground completion-read — gate subsection —
# before this gated push). The push is then GATED on the persisted, SHA-BOUND verdict file —
# the explicit conditional is the hard stop (missing file / block / crash
# / missing or stale sha all fail CLOSED). The verdict is consumed only
# AFTER `gh pr merge` SUCCEEDS (never hand-write the verdict file,
# #1082). The recovery just added a merge commit, so --rebase is
# documented-doomed here (#1041 — the old flow burned that attempt, then
# took the --squash substitution). Go straight to --squash for ALL kinds
# (#1288):
if grep -qxE 'pass|skip-artifact-only' /tmp/issue-<N>-lint-verdict.txt 2>/dev/null \
   && [ -n "$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt 2>/dev/null)" ] \
   && [ "$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt 2>/dev/null)" = "$(git -C "$WT" rev-parse HEAD)" ]; then
  git -C "$WT" push
  # gh recomputes mergeability asynchronously after a push — it can be
  # momentarily stale. Re-check before concluding failure:
  gh pr view <PR> --json mergeable -q .mergeable   # brief wait/retry until MERGEABLE
  if gh pr merge <PR> --squash --delete-branch=false; then
    rm -f /tmp/issue-<N>-lint-verdict.txt   # consume on MERGE SUCCESS — the verdict certified exactly the tip that landed
  else
    echo "MERGE FAILED post-push — classify: (0) \"Base branch was modified\" -> shape-0 same-tip retry (verdict survives); anything else -> epm:merge-failed (do NOT hand-write the verdict file)."
    false
  fi
else
  echo "BLOCKED: pre-push workflow-lint gate (verdict: $(cat /tmp/issue-<N>-lint-verdict.txt 2>/dev/null || echo not-run)) — missing verdict, block/crash, or missing/stale sha (hand-written verdict, or new commits since certification) all fail CLOSED: fix the named offender (or crash cause), re-run the gate ONCE; still failing -> epm:merge-failed (gate subsection, verdict cases 1/3). Do NOT push."
  rm -f /tmp/issue-<N>-lint-verdict.txt   # block/crash/stale consumed — a fresh gate run regenerates it
  false
fi
```

One recovery attempt per Step 10d invocation. If the re-checked
mergeability never recovers or the retried merge is refused again, fall
to the Failure bullet above (`epm:merge-failed v1`, continue). When the
recovered merge DOES land, run the **post-merge stale-task-folder guard**
below — the recovery's `git merge origin/main` adds a merge commit that
can re-import this task's old-status folder, exactly the case the guard
catches.

#### The artifact-confirmed merge procedure (unsafe case: guard 3 tripped)

When Guard 3 says the branch is unsafe to blind-rebase, the goal shifts
from "merge the whole branch" to "make sure this task's deliverables are
on `main`" — i.e. confirm that the artifacts a downstream
experiment / promotion would need (the clean-result body, the figures,
the per-cell eval JSON) already resolve on `origin/main`, then post
`epm:merged v1` with an artifact-confirmed sentinel rather than a list
of newly-landed SHAs.

This works because, by the time Step 10d fires, the analyzer has
already committed the clean-result body to `main` via `task.py
set-body` (which always operates on the repo root on `main`, never on
the worktree), and figure / `eval_results/issue_<N>/` commits land on
`main` through the same mechanism. The branch's commits often duplicate
work already on `main`; the value of the rebase is shared-infra fixes
the branch carries forward, NOT the per-task artifacts.

**New-shared-`src/`-infra guard (run FIRST, before the deliverables
check).** The artifact-confirmed path — and the surgical additive
checkout it degrades to — is structurally restricted to this task's own
`tasks/` / `figures/` / `eval_results/` paths and CANNOT carry shared
`src/` infra the branch introduced. So if this branch ADDED new shared
modules under `src/explore_persona_space/`, the artifact-confirmed path
would silently strand them on the branch — a downstream child that
reuses the harness then breaks its import path on a clean `main`
checkout (incident #595: parent #545 / grandparent #503 introduced
`src/explore_persona_space/experiments/issue503/`, which the
artifact-confirmed merge left on the branch; the child's eval battery
imported it and crashed). Scan for it FIRST:

```bash
# Files this branch ADDED (status A) vs origin/main under shared src/.
git -C "$WT" diff --name-only --diff-filter=A origin/main HEAD -- \
  "src/explore_persona_space/" > /tmp/issue-<N>-new-src.txt
```

If `/tmp/issue-<N>-new-src.txt` is NON-EMPTY, this branch introduces NEW
shared `src/` infra: the artifact-confirmed degrade is REFUSED. Do NOT
fall through to the surgical additive checkout (it would strand the
infra). Instead either (a) resolve the actual guard-3 condition so the
SAFE full-rebase path runs (e.g. the parent `issue-<M>` branch this one
forked off has since merged — re-run the guard-3 check; once
`ON_MAINLINE=yes` and the content check is clean, `gh pr merge --rebase`
carries the `src/` infra correctly), or (b) if the full rebase still
cannot run, post `epm:merge-failed v1` with `{reason: "new shared src/
infra cannot land via artifact-confirmed surgical checkout", new_src:
[...]}`, surface ONE line in chat naming the branch + worktree path +
the stranded `src/` paths for manual full-rebase resolution, and
CONTINUE (the task still parks / completes; the merge retries
idempotently on the next `/issue <N>`). NEVER surgical-checkout a branch
that added shared `src/` — that is the exact #595 stranding this guard
prevents.

```bash
# Verify task deliverables resolve on origin/main.
git -C "$REPO_ROOT" fetch origin main --quiet

# 1) body.md present on main with this task's number
BODY_REL=$(realpath --relative-to="$REPO_ROOT" \
  "$(uv run python "$REPO_ROOT/scripts/task.py" find <N>)")/body.md
git -C "$REPO_ROOT" cat-file -e "origin/main:$BODY_REL" \
  || ARTIFACTS_OK=no

# 2) figures/issue_<N>/ has at least one file on main (if any were produced)
git -C "$REPO_ROOT" ls-tree -r --name-only origin/main -- "figures/issue_<N>/" \
  | grep -q . || FIGURES_OK=no   # only enforce if the task plan produced figures

# 3) eval_results/issue_<N>/ (or equivalent) similarly, when the task produced eval JSONs
```

Decision tree:

- **All required deliverables resolve on `origin/main`** -> post
  `epm:merged v1` with fields `{artifact_confirmed: true,
  full_rebase_deferred: true, reason: "<the tripped guard-3 condition:
  based on <PARENT> (not on mainline) | own commits touch foreign /
  out-of-scope paths: <paths>>", verified_paths: [...]}`.
  Update the chat title with `merged (artifact-confirmed)`. Skip the
  `gh pr merge` call; leave the PR open so a future `/issue <N>`
  re-invocation can retry the full rebase once the parent branch is
  itself merged. This is the standard outcome of guard 3 — the task
  has its science deliverables on `main` and is not blocked.
- **One or more deliverables missing on `origin/main`** -> do a
  **surgical additive checkout** of just this branch's own NEW files
  (the ones it added vs `origin/main` AND that live under the task's
  own paths — `tasks/*/<N>/`, `figures/issue_<N>/`,
  `eval_results/issue_<N>/`, `eval_results/issue_<N>_*/`,
  `ood_eval_results/issue_<N>/`). Compute:

  ```bash
  # Files this branch ADDED (status A ONLY) vs origin/main, restricted to
  # this task's own paths PLUS the workflow surface (a workflow-fix branch's
  # ADDED deliverable can be .claude/** / CLAUDE.md / .gitattributes). Never
  # sweeps shared src/ or scripts/ — the new-shared-src/ guard above already
  # refused the surgical path if the branch added src/.
  #
  # --diff-filter=A (ADDED-only), NEVER AM: this checkout does a WHOLESALE
  # `git checkout issue-<N> -- <path>` below (~line for the xargs git checkout),
  # which would OVERWRITE main's newer copy of a MODIFIED file with no conflict.
  # A-only guarantees every listed path does not yet exist on main, so the
  # checkout only CREATES — never clobbers. A branch that MODIFIES a
  # workflow-surface file is not fast-path-eligible (predicate (e)) and reaches
  # this block only via a genuine Guard-3 UNSAFE degrade, where the same
  # A-only safety applies.
  #
  # Three-dot origin/main...HEAD (merge-base..HEAD): the branch's OWN adds only.
  # Two-dot origin/main HEAD would additionally list files main advanced that
  # the branch never touched (status M-because-main-advanced), pulling them into
  # the checkout list — precisely the paths we must NOT overwrite.
  #
  # PRODUCER GUARD — materialize-then-check (mirrors the shared gate's
  # trigger diff): unchecked, a FAILED diff (bad ref, no merge-base) writes
  # an empty/partial list indistinguishable from "no additive files" and the
  # landing below fails OPEN. And an EMPTY list is itself an anomaly HERE:
  # this decision-tree branch is reached ONLY because deliverables are
  # MISSING on origin/main, so "nothing to add" means the diff lied or the
  # payload sits outside the A-only pathspec set (e.g. main deleted the
  # deliverable — status M, not A — or a scripts/-only payload); landing
  # anyway would push nothing and post `epm:merged {surgical_checkout:
  # true}` with nothing committed (a PHANTOM SUCCESS). Both arms hard-stop.
  if ! git -C "$WT" -c core.quotePath=false diff --name-only --diff-filter=A origin/main...HEAD -- \
      "tasks/*/<N>/" "figures/issue_<N>/" "eval_results/issue_<N>/" \
      "eval_results/issue_<N>_*/" "ood_eval_results/issue_<N>/" \
      ".claude/" "CLAUDE.md" ".gitattributes" "docs/methodology/issue_<N>.md" \
      > /tmp/issue-<N>-additive-files.txt; then
    echo "SURGICAL ABORT: additive-list diff FAILED — cannot enumerate the payload; route to epm:merge-failed, never land"
    false
  elif [ ! -s /tmp/issue-<N>-additive-files.txt ]; then
    echo "SURGICAL ABORT: additive-files list EMPTY on a deliverables-missing landing (phantom success); route to epm:merge-failed, never land"
    false
  fi
  ```

  Either `SURGICAL ABORT` arm (failed producer diff, or an empty additive
  list on this deliverables-missing landing) routes to the **Surgical
  checkout itself fails** bullet below: post `epm:merge-failed v1` with the
  abort line, surface ONE line in chat, CONTINUE — never fall through to
  the checkout/stage/push block, and never post `epm:merged`.

  Then, from the **repo root on `main`** (never switch the branch
  there), checkout each path from the branch, stage by EXPLICIT PATH
  (never `git add -A`), commit PATHSPEC-LIMITED, and push. The
  pathspec-limited commit is load-bearing: many sessions commit to the
  shared repo root concurrently, so its index may carry a CONCURRENT
  session's staged files, and a bare `git commit` sweeps them in
  (incident #562/#550, 2026-06-10: 70 foreign staged files landed in
  #562's surgical commit) — limiting the commit by pathspec commits
  ONLY this task's files and ignores every other staged entry:

  ```bash
  cd "$REPO_ROOT"
  # earlyoom-protect the gate — form (iii) (#1045 recipe, #1211; FAIL-OPEN,
  # see the shared gate block above): the preamble sits BEFORE the BASELINE
  # legs (they run before the checkout); this whole gate-and-land sequence is
  # ONE BACKGROUND Bash invocation (run_in_background=true — its two lint
  # leg pairs total ~9-12+ min and can NEVER fit the 600s foreground Bash
  # tool cap, #1245), so every child inherits adj=-600.
  sudo -n choom -n -600 -p $$ >/dev/null 2>&1 && LINT_GATE_CHOOM=ok \
    || { LINT_GATE_CHOOM=failed; echo "[warn] choom failed — lint gate is earlyoom-UNPROTECTED (choom=failed)" >&2; }
  echo "[step10d] lint-gate earlyoom protection choom=$LINT_GATE_CHOOM"
  # Outcome sentinel (#1245): pre-rm; each terminal arm below writes it as
  # its LAST action (landed | push-failed | blocked-cleaned) — missing at
  # completion = the sequence died mid-run (completion-read below the block).
  rm -f /tmp/issue-<N>-surgical-outcome.txt
  # Pre-push workflow-lint gate — form (iii) (subsection above): the payload
  # lands in the ROOT tree, so BOTH lint runs use the root copy, sequenced
  # around the checkout — BASELINE BEFORE (payload-free tree; a post-checkout
  # "baseline" would re-lint the same contaminated tree, a degenerate
  # self-compare that fails open), GATED AFTER. The whole gate-and-land
  # sequence runs in that ONE background invocation, so GATE_ARMED /
  # BASE_RC / GATE_VERDICT remain same-invocation state (no cross-block
  # variable). Executable trigger
  # first: an artifact-only additive list skips both lint runs.
  GATE_ARMED=no
  # Output-test form, not `-q -v` rc (ugrep rc inversion, #928 -> #1125):
  if [ -n "$(grep -vE '^(tasks/|figures/|eval_results/|ood_eval_results/|raw/|data/|docs/methodology/)' \
       /tmp/issue-<N>-additive-files.txt)" ]; then
    GATE_ARMED=yes
    # BASELINE legs (per-leg exit codes ARE captured — a baseline CRASH must
    # fail CLOSED via the crash arm below, never be `|| true`-erased; only
    # normalized failure LINES enter the compare for the legitimate
    # red-baseline rc=1-with-lines case; per-leg NO-DOWNGRADE max fold, same
    # rationale as the gate's executable block — a leg-1 crash must not be
    # erased by a leg-2 rc=1):
    BASE_RC=0
    timeout --kill-after=60s 900s uv run python "$REPO_ROOT/scripts/workflow_lint.py" \
      > /tmp/issue-<N>-lint-baseline.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$BASE_RC" ]; then BASE_RC=$rc; fi; }
    timeout --kill-after=60s 900s uv run python "$REPO_ROOT/scripts/workflow_lint.py" \
      --check-references --check-tables --check-asks --check-autonomous-asks \
      >> /tmp/issue-<N>-lint-baseline.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$BASE_RC" ]; then BASE_RC=$rc; fi; }
  fi
  # MAPPED INVARIANT-TEST LEG (#1147) — form (iii), DORMANT TODAY by
  # construction: the additive pathspec set above excludes scripts/ and src/,
  # so no additive payload can match a GLOB_SCAN_TESTS glob and the trigger
  # map is empty. The leg exists as defense-in-depth should that pathspec set
  # ever grow; it costs one ~1 s helper call per surgical landing. Sequencing
  # mirrors the lint legs: TG BASELINE runs BEFORE the checkout (the payload
  # lands in the ROOT tree — a post-checkout "baseline" would be a degenerate
  # self-compare), TG GATED after; BOTH legs run the ROOT copy.
  TG_RC=0; TG_BASE_RC=0; TG_CRASH=no
  : > /tmp/issue-<N>-tg-new.txt
  if ! timeout --kill-after=30s 120s uv run python "$REPO_ROOT/scripts/select_step9c_tests.py" \
      --map-files /tmp/issue-<N>-additive-files.txt --repo-root "$REPO_ROOT" \
      > /tmp/issue-<N>-tg-map.txt 2>/tmp/issue-<N>-tg-map-err.txt; then
    TG_CRASH=yes   # helper failure: cannot classify the payload — fail CLOSED
  fi
  if [ "$TG_CRASH" = no ] && [ -s /tmp/issue-<N>-tg-map.txt ]; then
    cut -f2 /tmp/issue-<N>-tg-map.txt | sort -u > /tmp/issue-<N>-tg-files.txt
    mapfile -t TG_TESTS < <(cut -f1 /tmp/issue-<N>-tg-map.txt | sort -u)
    ( cd "$REPO_ROOT" && timeout --kill-after=30s 300s \
      env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
          NUMEXPR_NUM_THREADS=8 \
      uv run pytest "${TG_TESTS[@]}" -q -p no:cacheprovider ) \
      > /tmp/issue-<N>-tg-baseline.txt 2>&1 || TG_BASE_RC=$?
  fi
  # `-C "$REPO_ROOT"` is the repo-root guard's designed deliberate-override
  # (#897): the hook's working-tree-revert detector would bounce the bare
  # `checkout <branch> -- <paths>` form; the `-C` names the tree explicitly.
  # `xargs -r` (--no-run-if-empty) is load-bearing: on an EMPTY additive list
  # a flag-less xargs still runs `git checkout issue-<N> --` ONCE with no
  # pathspec — a BRANCH SWITCH of the shared root (the FAST_PATH `-ge 1`
  # lower bound is the first defense; this is defense-in-depth).
  xargs -r -a /tmp/issue-<N>-additive-files.txt git -C "$REPO_ROOT" checkout issue-<N> --
  # GATED legs + verdict — the root tree now carries the payload. Same
  # normalize → comm -23 subtraction → verdict as the gate's executable
  # block; own-diff here = the additive-files list.
  GATE_VERDICT=pass
  if [ "$GATE_ARMED" = "yes" ]; then
    GATED_RC=0
    timeout --kill-after=60s 900s uv run python "$REPO_ROOT/scripts/workflow_lint.py" \
      > /tmp/issue-<N>-lint-gated.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$GATED_RC" ]; then GATED_RC=$rc; fi; }
    timeout --kill-after=60s 900s uv run python "$REPO_ROOT/scripts/workflow_lint.py" \
      --check-references --check-tables --check-asks --check-autonomous-asks \
      >> /tmp/issue-<N>-lint-gated.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$GATED_RC" ]; then GATED_RC=$rc; fi; }
    for leg in baseline gated; do
      grep -h '^workflow_lint: ' "/tmp/issue-<N>-lint-$leg.txt" \
        | grep -vE '^workflow_lint: (PASS$|FAIL \()' \
        | sed -E 's/:[0-9]+:/::/g' | sort -u \
        > "/tmp/issue-<N>-lint-$leg-norm.txt" || true
    done
    comm -23 /tmp/issue-<N>-lint-gated-norm.txt \
      /tmp/issue-<N>-lint-baseline-norm.txt > /tmp/issue-<N>-lint-new.txt
    grep -F -f /tmp/issue-<N>-additive-files.txt /tmp/issue-<N>-lint-gated-norm.txt \
      > /tmp/issue-<N>-lint-owndiff.txt || true
    # GATED_RC consumed HERE — CRASH ARM FIRST (fail CLOSED; same
    # classification as the gate's executable block): rc>1 on either leg
    # pair, or rc!=0 with ZERO normalized `workflow_lint:` lines, is a
    # linter CRASH -> GATE_VERDICT=crash (block path — the stage/commit/push
    # below runs ONLY on `pass`). Only then the attribution arm: a red
    # gated run blocks when payload-attributed (own-diff-named line OR NEW
    # non-empty).
    if [ "$GATED_RC" -gt 1 ] || [ "$BASE_RC" -gt 1 ] \
       || { [ "$GATED_RC" -ne 0 ] && [ ! -s /tmp/issue-<N>-lint-gated-norm.txt ]; } \
       || { [ "$BASE_RC" -ne 0 ] && [ ! -s /tmp/issue-<N>-lint-baseline-norm.txt ]; }; then
      GATE_VERDICT=crash
    elif [ "$GATED_RC" -ne 0 ] \
       && { [ -s /tmp/issue-<N>-lint-owndiff.txt ] || [ -s /tmp/issue-<N>-lint-new.txt ]; }; then
      GATE_VERDICT=block
    fi
  fi
  # TG GATED leg (#1147) — the root tree now carries the payload; same
  # grep -> line-number-blank -> comm -23 subtraction as the shared gate's
  # executable block (own-diff here = the additive-files list; structurally
  # unreachable today, see the dormancy comment above the TG baseline leg).
  if [ "$TG_CRASH" = no ] && [ -s /tmp/issue-<N>-tg-map.txt ]; then
    ( cd "$REPO_ROOT" && timeout --kill-after=30s 300s \
      env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
          NUMEXPR_NUM_THREADS=8 \
      uv run pytest "${TG_TESTS[@]}" -q -p no:cacheprovider ) \
      > /tmp/issue-<N>-tg-gated.txt 2>&1 || TG_RC=$?
    if [ "$TG_RC" -gt 1 ] || [ "$TG_BASE_RC" -gt 1 ]; then TG_CRASH=yes; fi
    for leg in baseline gated; do
      grep -F -f /tmp/issue-<N>-tg-files.txt "/tmp/issue-<N>-tg-$leg.txt" \
        | grep -vE '^E +assert ' \
        | sed -E 's/at line [0-9]+/at line N/g; s/:[0-9]+:/::/g; s/:[0-9]+([^0-9]|$)/:N\1/g' \
        | sort -u \
        > "/tmp/issue-<N>-tg-$leg-hits.txt" || true
    done
    comm -23 /tmp/issue-<N>-tg-gated-hits.txt \
      /tmp/issue-<N>-tg-baseline-hits.txt > /tmp/issue-<N>-tg-new.txt
  fi
  # Fold the TG verdict into the SAME GATE_VERDICT the stage/commit/push
  # consumes below — crash-class first (fail CLOSED; never downgraded), then
  # the payload-attributed block arm; block/crash reuse the existing cleanup
  # + hard-stop path verbatim:
  if [ "$TG_CRASH" = "yes" ]; then
    GATE_VERDICT=crash
  elif [ "$TG_RC" -ne 0 ] && [ -s /tmp/issue-<N>-tg-new.txt ] \
     && [ "$GATE_VERDICT" != "crash" ]; then
    GATE_VERDICT=block
  fi
  # HARD STOP: stage/commit/push run ONLY on a pass verdict; a block cleans
  # the payload back out of the shared root (index + working tree).
  if [ "$GATE_VERDICT" = "pass" ]; then
    xargs -r -a /tmp/issue-<N>-additive-files.txt git add --
    git diff --cached --name-only   # sanity echo: spot any foreign staged entries
    xargs -r -a /tmp/issue-<N>-additive-files.txt git commit -m "issue-<N>: surgical additive checkout (full rebase deferred — guard 3)

  Branch unsafe to blind-rebase: <based on <PARENT> (not on mainline) |
  own commits touch foreign / out-of-scope paths>. Cherry-picked this
  task's own added files only; shared src/ / scripts/ unchanged." --
    # Bounded push (the one network op on this arm): a hung push would wedge
    # the background call with the outcome sentinel unwritten. rc 124 takes
    # the push-failed arm — the same degradation as a rejected push (the
    # "Surgical checkout itself fails" bullet / sync-retry below).
    if timeout --kill-after=30s 300s git push origin main; then
      echo landed > /tmp/issue-<N>-surgical-outcome.txt
    else
      echo push-failed > /tmp/issue-<N>-surgical-outcome.txt
      false
    fi
  else
    # BLOCKED: the checkout above already staged the A-only paths AND wrote
    # them to the working tree — clean BOTH with the hook-verified two-step
    # (the gate subsection's baseline-semantics bullet documents why the
    # one-shot restore form is hook-blocked): index-only unstage, then plain
    # rm of the now-untracked A-only files (absent from main — no main state
    # destroyed).
    xargs -r -a /tmp/issue-<N>-additive-files.txt git -C "$REPO_ROOT" restore --staged --
    xargs -r -a /tmp/issue-<N>-additive-files.txt rm -f --
    echo "BLOCKED: pre-push workflow-lint gate (verdict: $GATE_VERDICT) — fix the named offender (or crash cause) in the worktree, re-run ONCE; still failing -> epm:merge-failed (gate subsection, verdict cases 1/3). Payload cleaned from the root index + working tree."
    echo blocked-cleaned > /tmp/issue-<N>-surgical-outcome.txt
    false
  fi
  ```

  **Completion-read (form (iii)).** While the background gate-and-land call
  runs, END THE TURN and run no repo-root-mutating commands until this
  completion-read — the root holds staged payload for the ~5-6 min
  contaminated window (worst case, every bounded leg wedged, ~78 min —
  past the 60-min § Long-phase heartbeat boundary; rare, and a watcher
  force-stop there is fail-closed: the sentinel stays unwritten). When the
  call completes (the harness notifies), read
  `/tmp/issue-<N>-surgical-outcome.txt` in a fresh FOREGROUND call:
  - `landed` -> post `epm:merged v1` as below.
  - `push-failed` (rejected OR timed-out, rc 124) -> the "Surgical
    checkout itself fails" bullet below (one `sync_repo_root.py` retry).
  - `blocked-cleaned` -> the gate subsection's case-1/3 fix path; read the
    background call's BLOCKED echo, which carries `$GATE_VERDICT`, for
    block-vs-crash attribution.
  - MISSING sentinel -> the sequence died mid-run (tool kill / watcher
    force-stop / wedge-bound kill) and the root may hold staged payload.
    Recover IN THIS ORDER: (1) kill-before-relaunch probe FIRST
    (`pgrep -af 'scripts/workflow_lint.py'` — root-copy invocations are
    not issue-scoped in argv, so on an ambiguous match WAIT for exit,
    never kill; the Step 0 single-orchestrator guard excludes same-issue
    concurrency). (2) Landed/committed classification BEFORE any cleanup —
    a shell killed between commit/push success and the sentinel write
    leaves the payload COMMITTED (tracked + clean), which a naive
    contamination probe misreads: check whether the surgical commit is on
    HEAD (`git -C "$REPO_ROOT" log -1 --format=%s` matches the surgical
    commit subject) and the additive paths are tracked + clean
    (`git status --porcelain` empty for them). Committed AND pushed
    (fetch, then `git merge-base --is-ancestor HEAD origin/main`) ->
    treat as `landed` (post `epm:merged v1`); committed but NOT pushed ->
    push-only retry (the sync-retry bullet) — NEVER `rm -f` committed
    files. (3) Only if genuinely uncommitted-contaminated (staged /
    working-tree payload present, no surgical commit) -> the hook-verified
    two-step clean (baseline-semantics bullet), then re-enter ONCE (the
    block is idempotent for the gate re-run case).

  **Guard-block recovery contract (improvised variants of this compound).**
  The checkout / restore forms in the blocks above are hook-fenced:
  `scripts/guard_repo_root_branch.sh` (a PreToolUse Bash hook) BLOCKS
  any improvised UNQUALIFIED variant run against the shared root — the
  bare `checkout issue-<N> -- <paths>` form
  (no `-C "$REPO_ROOT"`) trips its #897 checkout-pathspec detector, and a
  `restore` trips its #897 restore detector unless it carries `--staged`
  with NO worktree flag. The `git -C <path>` clause is the guard's designed
  per-clause waiver, so use the `-C "$REPO_ROOT"`-qualified fence lines
  VERBATIM — never retype them unqualified. The waiver is relied on here
  ONLY because both fence forms are NON-DESTRUCTIVE at the shared root
  (the checkout only CREATES A-only additive paths absent from `main`; the
  restore is `--staged` index-only) — NEVER generalize `-C "$REPO_ROOT"`
  to escape a block on any other / destructive command (the guard's own
  block message: never point `-C` at the repo root for a destructive op).
  On a guard block, the WHOLE compound Bash call was skipped, not just the
  offending clause (a PreToolUse deny rejects the entire tool call), so an
  earlier clause in the same call that writes
  `/tmp/issue-<N>-additive-files.txt` (the producer diff above) never ran
  either. The retry therefore RE-RUNS the producer diff clause to
  regenerate the list file BEFORE re-running the corrected `-C`-qualified
  consumer — re-running only the consumer (or `cat`-ing the list) fails
  with exit 128 / `cat: ... No such file` (incident 2026-07-05: the #813
  and #1056 sessions). The guard's block message gives only generic
  worktree / `sync_repo_root.py` retry advice and does NOT mention the
  skipped producer — this paragraph is the recovery contract.

  Then post `epm:merged v1` with `{artifact_confirmed: true,
  full_rebase_deferred: true, surgical_checkout: true, files:
  [...]}`. Same chat title update as above.

- **Surgical checkout itself fails** (file conflicts, or push rejected after
  one `uv run python "$REPO_ROOT/scripts/sync_repo_root.py"` retry — the ONLY
  repo-root sync command: single-flight flock, untracked-collision sweep +
  rescue, `--rebase=merges --autostash` pull, stranded-autostash recovery,
  and a push with one rebase-retry built in; NEVER a hand-rolled repo-root
  `git pull`, which is the #967 `fatal: Cannot autostash` incident) — post
  `epm:merge-failed v1`
  with the error, surface ONE line in chat (branch + worktree path +
  one-line reason), CONTINUE. Same fail-fast policy as the safe case.

Never blind-`gh pr merge` (any `$MERGE_FORM`) a branch that tripped guard 3
— that is the exact #458 / #479 incident class this section exists to prevent.

#### Post-merge stale-task-folder guard (runs after EVERY merge form lands)

Run this AFTER any of the three merge forms above lands (safe-case
`gh pr merge $MERGE_FORM`, the merge-conflict-recovery retry, or the
artifact-confirmed / surgical-additive checkout). A merge commit — most
often the recovery's `git merge origin/main`, but also any improvised
merge taken when `--rebase` keeps being refused — can import THIS task's
OLD status folder onto `main` next to its live one (e.g.
`tasks/approved/<N>/` lands alongside `tasks/awaiting_promotion/<N>/`,
same task number, two status dirs). The autonomous-session watcher then
reads the stale folder as a live task and respawns the session
indefinitely (incident #644, 2026-06-16: an orphan-respawn cap-2-per-day
cycle ran ~8h; #643 hit the same class at archive time). Guard 1 above
catches FOREIGN tasks' folders but not this task's own old-status
duplicate, and it only runs on the safe-case (`$MERGE_FORM`) path. Keep exactly ONE
folder for this task on `main` — and never by deleting origin's ONLY copy
while the canonical status-mv is unpushed (#1300):

```bash
# Canonical folder for this task (NEVER hand-build tasks/<status>/<N> —
# status is unknowable here; resolve via task.py find, CLAUDE.md rule).
CANON=$(realpath --relative-to="$REPO_ROOT" \
  "$(uv run python "$REPO_ROOT/scripts/task.py" find <N>)")
# MATERIALIZE ls-tree to a file and check each producer's OWN exit code
# (find/CANON, fetch, ls-tree): piped straight into grep with a trailing
# `|| true`, a FAILED producer is indistinguishable from "no duplicate
# folders" and the guard fails OPEN — cleanup silently skipped, the watcher
# respawns against the stale folder (incident #644). Same materialize-then-
# check pattern as the pre-push lint-gate trigger diff (#1047). Failure arms
# are TERMINAL (echo + false — routes to the epm:merge-failed handling
# above); never proceed believing cleanup ran.
if [ -z "$CANON" ]; then
  # task.py find / realpath failed -> empty CANON. Classifying with an empty
  # CANON would mark the CANONICAL folder itself as a duplicate and rm it.
  echo "post-merge stale-task-folder guard: task.py find <N> produced empty CANON — refusing to classify duplicates"
  false
elif ! git -C "$REPO_ROOT" fetch origin main --quiet; then
  # A failed fetch leaves origin/main at its PRE-merge state: the duplicate
  # imported by the merge that JUST landed is invisible — the guard's
  # primary blind spot, not a lesser staleness.
  echo "post-merge stale-task-folder guard: git fetch origin main FAILED — origin/main may predate the just-landed merge; cannot certify no stale task folders"
  false
elif ! git -C "$REPO_ROOT" ls-tree -d -r --name-only origin/main \
    > /tmp/issue-<N>-postmerge-lstree.txt; then
  echo "post-merge stale-task-folder guard: git ls-tree origin/main FAILED — cannot certify no stale task folders"
  false
# Unpushed-mv pre-check (#1300): CANON absent from the materialized
# origin/main ls-tree means origin's only folder for this task is (almost
# always) the OLD-status copy of a status mv committed on local main but
# not yet pushed — classifying it as a duplicate would delete origin's
# ONLY folder for the task (incident 2026-07-13: origin commit 2a1a9cbc0b
# left ZERO tasks/*/1291 folders + a dangling REGISTRY pointer; recovery
# merge f26462fc1b). Recovery: land the local mv via the sanctioned root
# sync (the fleet-standard single-flight helper — it pushes ALL committed
# local-main state, not just this task's mv), RE-RESOLVE the canonical
# path (the sync pull-rebases the local root, so the canonical status can
# change in EITHER lag direction — a failed re-resolve keeps the previous
# value and fails closed below), re-fetch, REGENERATE the ls-tree file
# (same materialize-then-check form), then re-check. Bounded 2 attempts;
# the ls-tree RE-CHECK is the arbiter, NOT the helper's exit code (exit 0
# includes the in-flight state — same 2-attempt shape as the local-residue
# tail below). The condition is a command list: a SUCCESSFUL recovery
# makes the final still-absent test fail, the branch is NOT taken, and
# evaluation falls through to the DUPES classification below against the
# REGENERATED file (a merge-imported duplicate can coexist with the
# unpushed mv and must still be removed). A failed mid-recovery re-fetch
# or regen can leave a stale listing that still carries CANON and falls
# through — the guarantee is the membership test itself: classification
# only ever proceeds when CANON is present in the listing it reads, so
# this arm never opens a delete of the canonical folder. Only a
# still-absent CANON takes the branch — terminal echo + false, nothing
# deleted.
elif ! grep -qxF "$CANON" /tmp/issue-<N>-postmerge-lstree.txt \
    && { for _ in 1 2; do
           uv run python "$REPO_ROOT/scripts/sync_repo_root.py"
           NEW_CANON=$(realpath --relative-to="$REPO_ROOT" \
             "$(uv run python "$REPO_ROOT/scripts/task.py" find <N>)")
           [ -n "$NEW_CANON" ] && CANON="$NEW_CANON"
           git -C "$REPO_ROOT" fetch origin main --quiet \
             && git -C "$REPO_ROOT" ls-tree -d -r --name-only origin/main \
                > /tmp/issue-<N>-postmerge-lstree.txt \
             && grep -qxF "$CANON" /tmp/issue-<N>-postmerge-lstree.txt \
             && break
         done
         ! grep -qxF "$CANON" /tmp/issue-<N>-postmerge-lstree.txt; }; then
  echo "post-merge stale-task-folder guard: canonical folder $CANON still ABSENT from origin/main after 2 root syncs — cannot classify duplicates (removing origin's only copy would leave ZERO folders for task <N>)"
  false
# Work arm: every committed task-<N> folder on origin/main (matches
# tasks/<status>/<N> exactly — the anchored $ excludes deeper paths like
# .../<N>/artifacts). The elif condition is a two-command list: mapfile
# fills DUPES from the FILE (grep semantics identical to the old pipe;
# no-match `|| true` is a legitimate empty DUPES), then the [ ... ] test —
# the LAST command's exit — decides the branch. Empty DUPES on a healthy
# read = clean no-op (exit 0), preserving idempotent re-runs.
elif mapfile -t DUPES < <(grep -E "^tasks/[^/]+/<N>$" \
      /tmp/issue-<N>-postmerge-lstree.txt \
      | grep -v -F -x "$CANON" || true); [ "${#DUPES[@]}" -gt 0 ]; then
  # Remove the duplicate(s) in a SPARSE SCRATCH WORKTREE detached at the
  # SAME fetched origin/main the detection just read — NEVER a root
  # `git rm`. The duplicates live on origin/main but are usually ABSENT
  # from the LOCAL root tree (local main predates the just-landed
  # server-side merge), so a root `git rm` fails pathspec, and the
  # improvised checkout-pathspec fallback at the root is hook-blocked
  # every time (#1253; session 82f5b16a, /issue 1198). The scratch
  # worktree needs no local-root state (the duplicate exists there BY
  # CONSTRUCTION), stages in its OWN index (no concurrent-session staging
  # races), and every command is `git -C`-scoped (the hook's designed
  # override). Sparse cone = the duplicates only: a FULL checkout is
  # ~7.7 GB / ~100k files on the shared VM root disk.
  SCRATCH=/tmp/issue-<N>-postmerge-scratch
  # Pre-clean a scratch leaked by an earlier crashed run. Failure here is
  # tolerable (nothing to clean): the worktree add below is the loud gate.
  git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null || true
  rm -rf "$SCRATCH"
  git -C "$REPO_ROOT" worktree prune
  # Stage: add (detached, no checkout) -> cone init FIRST (git 2.34:
  # `set --cone` is silently a literal PATTERN, non-cone) -> cone = the
  # duplicates -> populate -> rm -> commit. Flag order `--detach
  # --no-checkout` is load-bearing for a bare copy of the add line.
  if ! { git -C "$REPO_ROOT" worktree add --detach --no-checkout "$SCRATCH" origin/main \
         && git -C "$SCRATCH" sparse-checkout init --cone \
         && git -C "$SCRATCH" sparse-checkout set "${DUPES[@]}" \
         && git -C "$SCRATCH" checkout --detach origin/main \
         && git -C "$SCRATCH" rm -r -q "${DUPES[@]}" \
         && git -C "$SCRATCH" commit -q -m "post-merge: remove stale task #<N> folder(s) imported by Step 10d merge

$CANON is the canonical folder; the duplicate(s) were re-imported by the
merge commit and would be read as a live task by the session watcher
(incident #644)."; }; then
    echo "post-merge stale-task-folder guard: scratch-worktree staging FAILED — stale folder(s) NOT removed: ${DUPES[*]}"
    git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null
    false
  # Land: push; on rejection (origin advanced under fleet churn) ONE
  # bounded fetch + rebase + push retry INSIDE the scratch worktree
  # (`git -C` — never a root rebase). A concurrent removal of the same
  # duplicate rebases to an empty commit and is dropped: the up-to-date
  # push and the verify arm below still pass (idempotent).
  elif ! { git -C "$SCRATCH" push origin HEAD:main \
           || { git -C "$SCRATCH" fetch origin main --quiet \
                && git -C "$SCRATCH" rebase origin/main \
                && git -C "$SCRATCH" push origin HEAD:main; }; }; then
    git -C "$SCRATCH" rebase --abort 2>/dev/null
    echo "post-merge stale-task-folder guard: removal commit did NOT land on origin/main after 1 retry"
    git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null
    false
  # Verify against a FRESH fetch that origin/main now carries exactly ONE
  # folder for this task (same materialize-then-check shape as detection).
  elif ! git -C "$REPO_ROOT" fetch origin main --quiet; then
    echo "post-merge stale-task-folder guard: verify fetch FAILED — cannot certify the removal landed"
    git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null
    false
  elif ! git -C "$REPO_ROOT" ls-tree -d -r --name-only origin/main \
      > /tmp/issue-<N>-postmerge-verify.txt; then
    echo "post-merge stale-task-folder guard: verify ls-tree FAILED — cannot certify the removal landed"
    git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null
    false
  elif mapfile -t STILL < <(grep -E "^tasks/[^/]+/<N>$" \
        /tmp/issue-<N>-postmerge-verify.txt \
        | grep -v -F -x "$CANON" || true); [ "${#STILL[@]}" -gt 0 ]; then
    echo "post-merge stale-task-folder guard: stale folder(s) STILL on origin/main after push: ${STILL[*]}"
    git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null
    false
  else
    git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" \
      || echo "WARN: scratch worktree cleanup failed ($SCRATCH is inert; /tmp clears on reboot and git gc prunes the metadata)"
    # LOCAL-tree residue: a root that pulled origin/main in the window
    # between the merge landing and the removal landing holds a tracked
    # local copy the session watcher can misread (incident #644 reads the
    # LOCAL tree). Converge via the sanctioned root sync. CAUTION — the
    # helper's contract: exit 0 does NOT by itself mean the pull ran (exit
    # 0 includes the in-flight state), so the existence RE-CHECK is the
    # arbiter, with one in-flight re-run (the helper's own prescription),
    # then fail loud — same 2-attempt shape as the old push-recovery tail.
    STALE_LOCAL=$(cd "$REPO_ROOT" && ls -d "${DUPES[@]}" 2>/dev/null || true)
    if [ -n "$STALE_LOCAL" ]; then
      uv run python "$REPO_ROOT/scripts/sync_repo_root.py"
      STALE_LOCAL=$(cd "$REPO_ROOT" && ls -d "${DUPES[@]}" 2>/dev/null || true)
    fi
    if [ -n "$STALE_LOCAL" ]; then
      uv run python "$REPO_ROOT/scripts/sync_repo_root.py"
      STALE_LOCAL=$(cd "$REPO_ROOT" && ls -d "${DUPES[@]}" 2>/dev/null || true)
    fi
    if [ -n "$STALE_LOCAL" ]; then
      echo "post-merge stale-task-folder guard: LOCAL stale copy/copies persist after 2 root syncs: $STALE_LOCAL — origin/main is clean but the local root still carries the folder(s)"
      false
    fi
  fi
fi
```

This guard is idempotent: a clean `main` (no duplicate) leaves `DUPES`
empty and the block is a no-op, so re-running Step 10d on a later
`/issue <N>` re-invocation is safe. A FAILED producer (empty `CANON` from
`task.py find`, `fetch`, or `ls-tree`) instead exits the block non-zero
through a terminal echo + `false` arm — the epm:merge-failed handling —
rather than reading as "no duplicates" (#1184; the #1047
materialize-then-check pattern). The unpushed-mv pre-check (#1300)
refuses to CLASSIFY while the canonical folder is absent from
origin/main — under routine local-main push lag origin's only copy is
the OLD-status folder of a not-yet-pushed status mv, and classifying it
as a duplicate deleted origin's only folder for task 1291 (origin commit
2a1a9cbc0b; recovery f26462fc1b). The arm lands the local mv via the
sanctioned root sync, re-resolves the canonical path, re-fetches,
regenerates the ls-tree file, and re-checks (2 bounded attempts; the
ls-tree re-check is the arbiter, not the helper's exit 0); a successful
recovery falls through to classification against the regenerated file,
and a still-absent canonical folder fails loud with nothing deleted. One
pre-existing residual stays out of this fix's scope: when BOTH folders
are already on origin/main and the LOCAL canonical resolution is stale
(origin's status-mv is newer, but the pre-check never fires because the
stale CANON path is still present on origin), the guard can still
classify origin's newer folder as the duplicate — that wrong-direction
delete predates this change. The work arm never touches the local
root index — the removal is staged and pushed from a sparse scratch
worktree detached at the fetched `origin/main`, so it succeeds whether or
not the local root has pulled the merge (the root-`git rm` pathspec
failure that drove the #1253 improvised, hook-blocked checkout-pathspec
fallback). The local-residue tail converges the local root via
`scripts/sync_repo_root.py`, with the existence re-check as the arbiter
(the helper's exit 0 alone does not prove the pull ran).

---

## Resume semantics

`/issue <N>` and `/issue <N> --resume` are identical. The skill is
always idempotent: it reads state from the task folder + recent
`events.jsonl` rows, computes the next action, and executes. There is
no "start from scratch" — the only way to reset is to manually edit
`body.md` and / or move the folder via `task.py set-status`.

### Step-completed re-entry skip-ahead (`epm:step-completed`)

Every step that completes posts `epm:step-completed v1` BEFORE EXIT,
recording `step`, `next_expected_step` (looked up from
(see workflow.yaml § steps)), and an `exit_kind` (one of `clean` /
`parked` / `failure-exit`). The distinctions are:

- `clean` = normal continuation;
- `parked` = user-gated wait;
- `failure-exit` = error path.

**Authoring convention:** all-caps `EXIT` in this file's action region (everything above this section) marks a scanner-enforced action exit that must carry a `post_step_completed.py` call within ±6 lines (`tests/test_step_completed_resume.py::test_every_exit_site_posts_marker`); lowercase `exit` is reference prose or a deliberately marker-free exit (user-pause, v2 handoff).

**Helper.** Skill code calls `scripts/post_step_completed.py` at every
EXIT site (after the EXIT condition is met, before the actual exit):

```bash
uv run python scripts/post_step_completed.py \
    --issue <N> --step 5b --exit-kind clean \
    --notes "code-review PASS, advancing to pod provisioning"
```

The helper looks up `next_expected_step` from `.claude/workflow.yaml`
and appends the event row; refuses to post if the step ID is unknown to
the YAML or if `exit_kind` is not in the choices list (typo guard).

**Re-entry router.**
`src/explore_persona_space/orchestrate/resume.py:decide_entry_step`
implements the precedence rules:

1. `status` is `blocked` -> full replay (rule 1, BEFORE the marker is
   consulted; load-bearing — a stale clean-exit marker must NEVER let
   the skill dispatch on a manually-blocked task).
2. No `epm:step-completed` event -> full replay (first invocation or
   pre-§5 in-flight task).
3. Latest event's `exit_kind` is `parked` or `failure-exit` -> full
   replay.
4. Latest event's `next_expected_step` is unknown to
   (see workflow.yaml § steps) -> warn + full replay (graceful
   fallback for renamed / removed steps).
5. Current `status` not in target step's `entry_status_label` -> full
   replay (status drift; user manually flipped the status).
6. All checks pass -> jump to `next_expected_step`, skipping Steps 0
   through (target - 1).

**EXIT-site -> `exit_kind` mapping** (17 sites total). The implementer
wires each site to invoke `post_step_completed.py` with the right
`exit_kind`:

| EXIT site | Step | Trigger | `exit_kind` |
|---|---|---|---|
| Step 0b/2 `type` autofill loop guess | 0b | user override required | `failure-exit` |
| Step 1 user defers / no reply | 1 | user-gated | `parked` |
| Step 2c plan-pending awaiting `approve` | 2c | user-gated | `parked` |
| Step 2c "Defer"/"3" reply | 2c | user-gated | `parked` |
| Step 4b TDD gate awaiting `approve-tests` | 4b | user-gated | `parked` |
| Step 4b TDD second pass | 4b | user-gated | `parked` |
| Step 4b implementer EXIT to `running` | 4b | normal continuation | `clean` |
| Step 5b code-review FAIL revision_round>=5 | 5b | apply Step 5d cap-hit rule (strip-then-continue-or-surface) | `conditional` (all-stripped → `clean`; substantive residual → `failure-exit` autonomous / `parked` interactive) |
| Step 6c pod URLs surfaced, leave at `running` | 6c | normal continuation | `clean` |
| Step 6c pod provisioning failure | 6c | error path | `failure-exit` |
| Step 6 preflight error/warning | 6 | error path | `failure-exit` |
| Step 6d experimenter dispatched, autonomous | 6d | normal continuation | `clean` |
| Step 7 `epm:results` not found and stale | 7 | user-gated | `parked` |
| Step 7 upload-verifier FAIL | 7 | error path | `failure-exit` |
| Step 9b first entry to `awaiting_promotion` (tail of `9a-bis`) | 9a-bis | user-gated | `parked` |
| Step 10 still `classification = pending` (re-invocation) | 10 | user-gated | `parked` |
| Step 0 resume ambiguous status (folder mismatch) | 0 | error path | `failure-exit` |

**Backwards-compat.** A task that ran through Steps 0-5 BEFORE §5 landed
has no `epm:step-completed` events. On re-entry the router returns None
(rule 2) and the skill falls back to the existing full-replay path
documented below. The first `/issue <N>` re-invocation AFTER §5 lands
posts the first event; the SECOND benefits from skip-ahead. Graceful,
no migration step.

If the specialist subagent has exited but no `epm:results` event was
posted, the skill assumes the run failed silently. On resume in `running`
with no progress in >4 hours, post `epm:stale v1` event asking user to
investigate and optionally `task.py set-status <N> blocked`.

**Resume correctness per active state** (the key benefit of having
dedicated "working" statuses):

Every "re-spawn `codex-*` … only" action in this table is subject to the
#1204 pre-spawn sentinel check (CLAUDE.md § Codex ensemble review): when
LIVE, record the instant confirmed no-show instead of re-spawning the
composer.

| Status at resume | `epm:*` events present | Interpretation | Action |
|------------------|------------------------|----------------|--------|
| `planning` | no `epm:plan` | planner was cancelled | re-run adversarial-planner |
| `plan_pending` | `epm:plan` exists | awaiting user approval | show plan path, EXIT |
| `running` (implementing) | no `epm:experiment-implementation` (or `epm:results` for infra), no `epm:proposed-tests` either | implementer was cancelled | re-spawn implementer |
| `running` (implementing) | `epm:proposed-tests v<n>` exists, no `epm:experiment-implementation`, no `epm:approve-tests` event posted **after** the `proposed-tests` event | TDD mode: tests posted, awaiting user approval | show the `proposed-tests` event timestamp + the `approve-tests` reply instruction, EXIT |
| `running` (implementing) | `epm:proposed-tests v<n>` exists, an `epm:approve-tests` event exists **after** the `proposed-tests` event, no `epm:experiment-implementation` | TDD tests approved by user | re-spawn implementer with `tdd_approved=true`; brief instructs implementer to write implementation against the approved tests, then post `epm:experiment-implementation` at the next version (max+1 per § Posting review-round markers; omit `--version` and the CLI derives it) as normal |
| `running` (implementing) | latest `epm:code-review` is FAIL, round < 5 | revision in progress | re-spawn implementer with critique |
| `running` (implementing) | latest `epm:code-review` is FAIL, round >= 5 | cap reached | apply Step 5d cap-hit rule (strip-then-continue-or-surface): strip → all-stripped PASS+continue OR surface substantive residual (autonomous: `blocked` + notify; interactive: present to user) |
| `running` (code-reviewing) | neither `epm:code-review` nor `epm:code-review-codex` for the current implementation version | both ensemble reviewers were cancelled | re-spawn both code-reviewer + codex-code-reviewer in parallel |
| `running` (code-reviewing) | `epm:code-review v<n>` exists, no `epm:code-review-codex v<n>` | Codex twin not yet returned (or wrapper crashed) | re-spawn `codex-code-reviewer` only |
| `running` (code-reviewing) | `epm:code-review-codex v<n>` exists, no `epm:code-review v<n>` | Claude reviewer not yet returned | re-spawn `code-reviewer` only |
| `running` (code-reviewing) | both `epm:code-review v<n>` and `epm:code-review-codex v<n>` exist, verdicts disagree (PASS-class vs FAIL), no `epm:review-reconcile v<n>` whose body's `**Role under adjudication:**` is `code-reviewer` | reconciler not yet started | spawn reconciler |
| `running` (code-reviewing) | both `epm:code-review v<n>` and `epm:code-review-codex v<n>` exist, verdicts agree | ensemble decision ready | apply Step 5c rule and advance |
| `running` (code-reviewing) | `epm:code-review-codex` is `epm:failure` (codex-output-malformed or infra) | Codex twin no-show | proceed with Claude-only decision per Step 5d fallback |
| `running` (workload) | no `epm:results` for > 4h | experimenter crashed silently | post `epm:stale`, ask user |
| `running` (workload) | latest event is `epm:failure` with bounce-back proposal | experimenter bounced to implementer | status back to `running` (implementing), re-spawn experiment-implementer |
| `uploading` | no `epm:upload-verification` PASS | verifier not run or failed | re-run upload-verifier |
| `interpreting` | no `epm:interpretation` | analyzer not started | spawn analyzer |
| `interpreting` | `epm:interpretation` exists, neither `epm:interp-critique` nor `epm:interp-critique-codex` for the current version | both ensemble critics not started | spawn `interpretation-critic` + `codex-interpretation-critic` in parallel |
| `interpreting` | `epm:interp-critique v<n>` exists, no `epm:interp-critique-codex v<n>` | Codex twin not yet returned | re-spawn `codex-interpretation-critic` only |
| `interpreting` | `epm:interp-critique-codex v<n>` exists, no `epm:interp-critique v<n>` | Claude critic not yet returned | re-spawn `interpretation-critic` only |
| `interpreting` | both `epm:interp-critique v<n>` and `epm:interp-critique-codex v<n>` exist, verdicts disagree (PASS vs REVISE), no `epm:review-reconcile v<n>` whose body's `**Role under adjudication:**` is `interpretation-critic` | reconciler not yet started | spawn `reconciler` (marker mode) |
| `interpreting` | both ensemble events exist, verdicts agree OR role-matching reconcile event present (`**Role under adjudication:** interpretation-critic`), ensemble verdict REVISE, round < 5 | revision needed | re-spawn analyzer with all critique events |
| `interpreting` | ensemble verdict PASS or (round >= 5 AND the Step 9a cap-hit resolved: all residual stripped, no substantive overclaim residual), neither `epm:clean-result-critique` nor `epm:clean-result-critique-codex` | content honesty settled, structure + register loop not started | promote body in place if missing, then spawn `clean-result-critic` + `codex-clean-result-critic` in parallel. (round >= 5 with a SUBSTANTIVE overclaim residual → apply Step 9a surface-real-residual rule instead: interactive present to user; autonomous `epm:failure v1 failure_class: code` + `blocked` + notify) |
| `interpreting` | `epm:clean-result-critique v<n>` exists, no `epm:clean-result-critique-codex v<n>` | Codex twin not yet returned (or wrapper crashed) | re-spawn `codex-clean-result-critic` only |
| `interpreting` | `epm:clean-result-critique-codex v<n>` exists, no `epm:clean-result-critique v<n>` | Claude critic not yet returned | re-spawn `clean-result-critic` only |
| `interpreting` | both `epm:clean-result-critique v<n>` and `epm:clean-result-critique-codex v<n>` exist, verdicts disagree (PASS-class vs REVISE), no `epm:review-reconcile v<n>` whose body's `**Role under adjudication:**` is `clean-result-critic` | reconciler not yet started | spawn `reconciler` (marker mode) |
| `interpreting` | clean-result ensemble verdict REVISE (agreed, unioned, or reconciled by a role-matching `epm:review-reconcile`; after the Step 9a-bis procedural-only strip), round < 5 | structure / register revision in progress | re-spawn analyzer with both clean-result critiques |
| `interpreting` | clean-result ensemble verdict PASS-class or (round >= 5 AND the Step 9a-bis cap-hit resolved: all residual stripped, no substantive overclaim residual) | ready for review | advance to `reviewing`. (round >= 5 with a SUBSTANTIVE overclaim residual → apply Step 9a-bis surface-real-residual rule: interactive present to user; autonomous `epm:failure v1 failure_class: code` + `blocked` + notify) |
| `reviewing` | (no agent dispatch; transitional single-step) | reviewer step retired; absorbed into clean-result-critic Lens 7 | move to `awaiting_promotion`, run the Step 10d auto-merge procedure, post `epm:status-changed`, EXIT |
| `awaiting_promotion` | `classification == 'pending'` in body frontmatter, no `epm:merged` and PR unmerged | waiting for user to promote; worktree not yet merged | run the Step 10d auto-merge procedure (idempotent backstop — covers the case where the Step 9b auto-merge was interrupted), then show task path, prompt to promote via `task.py promote`, run CRON-TEARDOWN (self-heal sweep, § CRON-TEARDOWN procedure, stray one-shot `/issue <N>` wakeups included), EXIT |
| `awaiting_promotion` | `classification == 'pending'` in body frontmatter, `epm:merged` present | waiting for user to promote; worktree already merged | show task path, prompt to promote via `task.py promote`, run CRON-TEARDOWN (self-heal sweep, § CRON-TEARDOWN procedure, stray one-shot `/issue <N>` wakeups included), EXIT |
| `awaiting_promotion` | `classification != 'pending'` (user ran `task.py promote`) | user promoted | advance to Step 10 (auto-complete) |
| `interpreting` / `reviewing` / `awaiting_promotion` / `completed` | unrun `epm:followup-scope` (≥1 UNRUN `followup_label` per `task_workflow.unrun_followup_labels`: entries grouped by label, within-label latest-(`ts`,`version`) authoritative, a label unrun iff no matching `epm:same-issue-followup-run v1`) | a `question_relation: same` follow-up is scoped to run ON this issue (takes precedence over the status rows above — see Step 0 "Same-issue follow-up dispatch") | route into the same-issue follow-up loop (Step 9b § Same-issue follow-up loop): dispatch ONE label per entry — the queue head (user-initiated first, then oldest armed ts); set status to `followups_running` + tag `followup-auto`\|`followup-manual` and run the abbreviated cycle. The loop re-reads the round's label-scoped scope at the planner snapshot (Step 9b § Same-issue follow-up loop step 3) so a crashed-mid-round resume picks up any correction posted since the round started |
| `followups_running` | unrun `epm:followup-scope` (≥1 UNRUN `followup_label` per `task_workflow.unrun_followup_labels`: entries grouped by label, within-label latest-(`ts`,`version`) authoritative, a label unrun iff no matching `epm:same-issue-followup-run v1`) | a same-issue follow-up round is mid-flight (this row takes precedence over the two children-based rows below) | resume the same-issue follow-up loop at the phase the stage breadcrumbs (`stage=followup-<phase>`) + latest markers indicate — do NOT restart from the top; `task_workflow.executing_followup_label` resolves WHICH label the round is executing (labeled breadcrumb first, dispatchable queue head fallback), and the planner-snapshot re-read (Step 9b § Same-issue follow-up loop step 3) picks up that label's latest scope so a correction posted mid-round is honored on resume |
| `followups_running` | no unrun followup-scope; at least one open child task (`parent_id: <N>` in `body.md` frontmatter) not in `completed` / `archived` | legacy semantics: children still in flight | show child-task table, EXIT |
| `followups_running` | no unrun followup-scope; every child has reached `completed` / `archived` (or no children remain) | children all done | re-run Step 10: relabel parent to `completed` |
| `running` (workload) | pod alive + log advancing (`ssh epm-issue-<N> tail -1 <log_abs>`), no live bg-Bash poll for this session, latest `epm:*` marker is stale (no `epm:progress` in > ~15 min) | Step 6d.2 bg-Bash poll chain died — typically because a reaction turn emitted a corrupted/truncated tool-call (rendered as raw text), the harness had no bg work to wake on, AND the auto-armed backstop cron also died (a `durable=False` cron does not survive the session that registered it, so this row is reached mainly after a session restart / fresh recovery session). Pod and run are HEALTHY; only the session's monitor died. (Origin: tasks #462 / #463, 2026-06-02.) | Re-enter the polling loop by re-invoking `/issue <N>` once; it reads the latest `epm:run-launched` (`pod`, `pid`, `log_abs`), resumes Step 6d.2, and the Step 6d.2 step-1 guard AUTO-RE-ARMS the backstop cron (`CronList` for `prompt.strip() == "/issue-tick <N>"`, `CronCreate` if absent) so the next dead turn won't strand the run again — no user `/loop` typing needed. The lightweight `/issue-tick <N>` tick is what the cron fires; the full `/issue <N>` skill loads only on cold start, cold respawn, or the tick's stale-marker recovery branch. Do NOT re-spawn `pod_watch.py` / `pod.py watch` — that mechanism is retired per "Notes on the obsolete monitoring stack". |

**Reconcile predicates are role-scoped.** There is exactly ONE marker-mode
reconcile kind (`epm:review-reconcile` — workflow.yaml § markers); the
adjudicated role lives in the verdict body's `**Role under adjudication:**`
field, not the marker name. Wherever a row above tests for (or reads the
verdict of) a reconcile event, only an event whose role field matches that
stage's critic (`code-reviewer` / `interpretation-critic` /
`clean-result-critic`) counts. Both the interpretation and clean-result
ensembles sit at status `interpreting` with the same round numbering, so an
unqualified "no `epm:review-reconcile v<n>`" predicate would let an
interp-stage reconcile falsely satisfy the clean-result disagreement row
(skipping the reconciler) or feed the wrong stage's verdict — and vice
versa.

Without distinct statuses for `uploading` / `interpreting` / `reviewing` /
`awaiting_promotion`, many of these rows would be indistinguishable.
That's why the state machine has them.

---

## Comment marker protocol

See `markers.md` for the full taxonomy. Every marker event row uses the
schema:

```jsonl
{"ts": "...", "kind": "epm:<kind>", "version": <n>, "note": "<body>", "metadata": {...}}
```

Convenience: the `task.py post-marker` / `task.py latest-marker`
helpers wrap the read/write side. The skill reads the highest-version
row per `(kind)` as authoritative (EXCEPTION: `epm:followup-scope` —
distinct queued follow-ups share the kind under different
`followup_label`s; group by label per
`task_workflow.unrun_followup_labels`, #894).

**Rules:**
- Never edit or delete a row in `events.jsonl` — always append a new row
  with a higher `version`. Version lets you see history; latest version
  wins for state purposes.
- `version=1` is the original; `version=2+` are revisions (e.g., revised
  plan after `/revise`).
- The 50,000-char `note` cap is enforced by `task.py post-marker`. If
  the body exceeds the cap, split into `part=K/N` chunks (see
  `markers.md`).

---

## Cost and safety rails

- **Never dispatch `compute:large` (>20 GPU-hours) without explicit
  user `approve`.** Small + medium can proceed on `approve` or
  `/approve`. Large requires `approve-large` to force a second thought.
- **Worktree merge is automatic** at the terminal point (Step 9b for
  experiments at `awaiting_promotion`; Step 10d for code paths at
  `completed`) — rebase-merge to `main`, no prompt, worktree kept. The
  user retains revert control (each commit lands individually). **Never
  force-push** (stays a user-ask) and never merge across repos or to any
  external remote.
- **Never edit `RESULTS.md` without proposal+approval.** Headline-level
  science is high-stakes.
- **Never auto-delete worktrees or model artifacts.** Cleanup is manual
  via `uv run python scripts/pod.py cleanup`.
- **Abort path:** user `task.py set-status <N> blocked` -> skill posts
  `epm:abort v1` and (if specialist is still running) sends abort
  signal. Specialist must check for `epm:abort` event periodically.

---

## When NOT to use this skill

- Tasks <30 min of work (trivial typo fixes, config tweaks). Just do
  them.
- Sessions already running via `experimenter` / `implementer` as the
  main agent — they manage their own lifecycle. The skill is for
  dispatch, not retrofitting.
- Purely exploratory sessions (`ideation`, `experiment-proposer`
  output). Those produce proposals; the user decides which become
  tasks.

---

## Error handling

| Symptom | Action |
|---------|--------|
| Task folder missing / multiple folders | Post error event listing conflicts, post the §5 marker: `uv run python scripts/post_step_completed.py --issue <N> --step 0 --exit-kind failure-exit --notes "ambiguous status: multiple folders / missing folder"`, EXIT. Ask user to reconcile. Do NOT pick. |
| Status missing from disk layout (legacy bodies) | Run Step 0b: autofill `proposed` via `task.py set-status`, post `epm:auto-defaults`, continue. |
| `type` frontmatter missing | Run Step 0b: infer from title prefix, confirm with the user, apply via `task.py set-body`. Autonomous loop with no user -> error + EXIT (a wrong guess corrupts the completed column). |
| Task is mis-filed under the wrong `kind` (e.g. a fix-validation / "test that X works" filed as `kind: experiment` — see CLAUDE.md § "Routing experiment intent") | Reclassify through the canonical path: `task.py set-kind <N> <kind>` (frontmatter + REGISTRY snapshot, flock + commit). Do NOT hand-edit the `kind:` frontmatter. If the misfile also pulled the task into the clean-result/promotion machinery, separately fix the resulting state (`set-clean-result --unset`, `set-status <N> completed`). |
| Empty task body | Run Step 0b: ask user for goal/hypothesis/setup in chat, draft body, patch via `task.py set-body --file`, post `epm:auto-defaults` audit event. |
| Plan fails mandatory-section check | Re-invoke `adversarial-planner` with missing sections list; do not post incomplete plan. |
| Preflight fails | Post the `--json` report verbatim as `epm:preflight v1`. Do NOT auto-fix (per CLAUDE.md "never take shortcuts"). |
| Specialist subagent errors out | Specialist posts `epm:failure v1` with traceback + last log lines. Status -> `blocked`. |
| Clean-result-critic FAIL | Post verdict, status -> `interpreting`. Analyzer revises in-place. |
| Task body lacks required fields | Post clarifier questions pointing to `.github/ISSUE_TEMPLATE/` for the right template. |
| Test suite crashes (OOM, import error) | Post `epm:test-verdict v1` event with FAIL + crash output. Stay in `reviewing`. Count toward 3-failure limit. |

Never silently skip a step. If something looks wrong, post an event and
exit — the durable trace lets the next invocation pick up where this
one left off without losing context.

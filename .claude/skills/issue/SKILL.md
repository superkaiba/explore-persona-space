---
name: issue
description: >
  End-to-end task-workflow orchestrator for experiments and code changes.
  Takes a task number (`<N>` = the integer that names `tasks/<status>/<N>/`),
  reads state from `body.md` frontmatter + `events.jsonl` markers under
  `tasks/<status>/<N>/`, and dispatches the next action (clarify ->
  adversarial-planner -> approval -> worktree + dispatch specialist ->
  preflight -> run -> analyzer -> humanize-loop (TL;DR) ->
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
round. In autonomous sessions, `substantially-different` `auto_run: yes`
proposals are FILED as `proposed` children for manual triage only —
never auto-spawned as sessions.

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
   `epm:followup-scope v1` moves to `followups_running` (tagged
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
                                                                      |-- FAIL + count<3 --> running (implementing, v+1)
                                                                      |-- FAIL + count>=3 --> blocked
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
  - `compute_deviation_resolution` → pick `accept_descope_to_<X>_with_caveats`
    if any descope dimension preserves majority statistical power (≥0.6 of
    the planned cells); else `continue_as_is` and quote the projected ratio
    inline. State `Decision: <choice> because <reason>` and EXECUTE the
    resolved action in this same turn (post `epm:compute-deviation v2`
    with the chosen `action:` + advance to Step 5.bis(b)); do NOT state
    the Decision and then end the turn.
  - `concern_deferral_request` → bounce to implementer for one more round
    targeting the open CONCERN(s); never defer in autonomous mode (deferral
    is a user-rationale-required action by spec). State
    `Decision: bounce to implementer (concern_id=<id>) — autonomous mode
    never defers` and EXECUTE the bounce in this same turn (spawn the
    `experiment-implementer` / `implementer` agent with a brief targeting
    the open concern_id); do NOT state the Decision and then end the turn.
  <!-- gate: gates.tdd_gate -->
  - `tdd_gate` → no `AskUserQuestion` at this site (it's event-driven —
    the implementer posts `epm:proposed-tests v1` and exits; the resume
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
    (the marker is the surface; `/weekly` + a later `/issue <N>`
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
- **Route `auto_run: yes` follow-ups by `question_relation` at Step 9b.**
  When a result lands, the orchestrator fires the `follow-up-proposer`
  at Step 9b (after auto-merge, before CRON-TEARDOWN, BEFORE the
  human-only park at `awaiting_promotion`) and partitions the
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
  Interactive mode (`EPM_AUTONOMOUS_SESSION` unset) IGNORES the
  `auto_run` tag and runs the proposer at Step 10b as today (user
  picks from the ranked list post-promotion; Step 10b routes the pick
  by `question_relation`). See Step 9b "Autonomous follow-up
  auto-spawn" + "Same-issue follow-up loop" + Step 10b
  "Autonomous-mode short-circuit" for the mechanics; see
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

**Single-orchestrator guard (run FIRST).** Exactly ONE session may drive
`/issue <N>` at a time. Before doing anything else, check whether another
live session is already mapped to this issue: `uv run python
scripts/spawn_session.py list` (issue-mapping column). If a live session is
already driving #N, EXIT immediately as a duplicate — post no markers (do
NOT run `scripts/post_step_completed.py`: a duplicate session must not touch
#N's `events.jsonl` — this is the one deliberately marker-free EXIT), mutate
nothing — UNLESS this session is its explicit replacement (an
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
scripts/task.py latest-marker <N>`, `~/.eps-autonomous/issue-<N>.json`,
and `uv run python scripts/spawn_session.py list`. If a replacement
session is registered for #N (a `spawned_at` newer than this session's
own start) OR the marker trail shows another writer has advanced the task
past this session's last-known state, YIELD immediately — post no
markers, launch nothing, mutate nothing; the replacement owns the task.
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
normal status dispatch, check the marker map for an UNRUN
`epm:followup-scope v1` — one whose `followup_label` has no matching
`epm:same-issue-followup-run v1`. If present AND the status is
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
lightweight `/issue-tick` at */20 — 362 full ~44K-token skill reloads
over 2.5 days. Incident #496: a worktree's pre-W22 `verify_task_body.py`
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
            cron="*/20 * * * *",
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
   / References, then patch the task:
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
the `/weekly` backstop re-synthesis OR a later `/issue <N>`
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
  data, env versions, exact `nohup` command for experiments
- Success criteria with quantitative thresholds
- Kill criteria (what result would kill the thesis)
- Compute estimate in GPU-hours — MUST include a machine-readable total line
  the auto-approve gate (Step 2c) can parse:
  `Estimated GPU-hours (total): <number>` (a single number, the total across
  all conditions/seeds; not a range). The autonomous auto-approve gate FAILS
  SAFE — it parks at `plan_pending` if this line is missing or unparseable.
- Target pod preference
- Plan deviations allowed vs must-ask

Post the plan body via `new-plan-version` (writes
`tasks/<status>/<N>/plans/v<K>.md` and rotates the `plan.md` symlink),
then announce it with an `epm:plan` event:

```bash
uv run python scripts/task.py new-plan-version <N> --file /tmp/issue-<N>-plan.md
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
REPO_ROOT=$(git rev-parse --show-toplevel)
WORKTREE="$REPO_ROOT/.claude/worktrees/issue-<N>"
bash "$REPO_ROOT/scripts/new_worktree.sh" "$WORKTREE" issue-<N> --issue <N>
# Sparse by default (~0.4G vs ~3.8G full); reuses if it exists (resume case);
# symlinks the repo .env (worktrees do NOT inherit it — RUNPOD_API_KEY /
# HF_TOKEN / WANDB_API_KEY dotenv loads fail without it).
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
absolute path once with `git rev-parse --show-toplevel` (as above) and
reuse `$WORKTREE` / `$REPO_ROOT` in every subsequent command.

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
  first, posts them as `epm:proposed-tests v1`, and EXITs without writing
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

**5a. Spawn both reviewers in parallel (fresh contexts, single message).**

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
WT=$(git rev-parse --show-toplevel)
SPECS=".claude/agents .claude/skills .claude/rules .claude/workflow.yaml CLAUDE.md"
MB=$(git -C "$WT" merge-base HEAD main)
SAFE_SPECS=""
for f in $SPECS; do
  # Branch-side feature edits = commits since merge-base touching $f,
  # EXCLUDING prior spec-freshness sync commits (which legitimately
  # touch spec paths — without the exclusion, the first sync's own
  # commit would poison every later freshness check on the branch).
  if [ -z "$(git -C "$WT" log --oneline "$MB"..HEAD --grep='spec-freshness' --invert-grep -- "$f")" ]; then
    SAFE_SPECS="$SAFE_SPECS $f"
  else
    echo "spec-freshness: $f carries branch-side feature edits — skipping blind sync; reconcile manually"
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
rule (those go through `workflow-improver` worktrees), with one
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
clean for the Step 10d merge guards.

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
Step 10d merge resolves it (observed on #542, 2026-06-11).

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
- `worktree` path, `base` ref (typically `main`).

The Codex twin additionally receives:
- `worktree`, `base`, `plan_marker_path` (no `implementation_marker_path`
  — the composer fetches the marker from canonical main state and INLINES
  it; likewise, if the worktree plan is absent — child task cut from a
  parent issue branch, #550 r1 — or STALE — follow-up amendment plan
  postdating the branch cut, #546 follow-up r1 — the composer inlines
  the canonical plan, Step 2-pre-b) — see
  `.claude/agents/codex-code-reviewer.md`.

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
gripes about present evidence never bounce the implementer or trip
the cap-3 pivot. Code-only tasks (`infra` / `batch` / `analysis` /
`survey`) keep the existing test-verdict gate (Step 9c) and are
exempt from this smoke gate.

**5b. Read both markers from `events.jsonl`.**

```bash
# After both Agent tasks complete — ONE fetch, parse twice in-memory.
events_json=$(uv run python scripts/task.py view <N> --json | jq '.events')
claude_marker=$(echo "$events_json" | jq '... epm:code-review v<n> ...')
codex_marker=$(echo "$events_json" | jq '... epm:code-review-codex v<n> ...')
```

Parse each marker's `**Verdict:**` line. Acceptable values: `PASS`,
`CONCERNS`, `FAIL`. PASS-class = {PASS, CONCERNS}; FAIL-class = {FAIL}.

**5c. Apply ensemble decision rule.**

| Claude verdict | Codex verdict | Action |
|---|---|---|
| PASS-class | PASS-class | **Agree.** `final_verdict = PASS`. CONCERNS bullets from either reviewer surface to the implementer as opportunistic suggestions; do not block. |
| FAIL | FAIL — overlapping blockers | **Agree.** `final_verdict = FAIL`. Bounce to implementer (one round). |
| FAIL | FAIL — disjoint blockers | **Union, no reconciler.** Build a combined blocker list (Claude's blockers ∪ Codex's blockers) and pass it to the implementer in the next-round brief. No new marker — both `epm:code-review v<n>` and `epm:code-review-codex v<n>` already exist on the task. `final_verdict = FAIL`. Bounce (one round). |
| PASS-class | FAIL (or vice versa) | **Disagreement.** Spawn `reconciler` agent (Claude, fresh context). Brief: role=`code-reviewer`, task=N, round=n, both event bodies, diff path. Reconciler reads both verdicts + the artifact, posts `epm:review-reconcile v<n>` with binding PASS or FAIL. `final_verdict = reconciler's verdict`. |

The reconciler may NOT add findings beyond what either reviewer raised —
its job is adjudication only. Round counter does NOT increment for
reconciler invocations.

**5c-bis. Mechanical-contract-only FAIL strip (anti-gate-hopping).**

A FAIL is *mechanical-contract-only* when its `**Blocker tags:**` line
(reviewer Step 7 template) is a non-empty subset of {`marker-shape` (Step
0.5), `smoke-run-missing` (Step 0.6)} and does NOT contain `substantive`
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

- **marker-shape:** all four H3 sections `(a)`–`(d)` present with non-empty
  content AND `(c)` carries at least one fenced command.
- **smoke-run-missing:** a `## Smoke run` section is present, and EVERY phase
  the pipeline actually executes (typically data-gen, training, eval) has its
  own sub-section with a command, exit code `0`, and an artifact digest. A
  `## Smoke run` section that covers only one phase (e.g. training) while the
  pipeline also runs a separate eval rig is genuinely absent for the missing
  phase — leave the FAIL in place.

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
code-substance judgment). It directly closes the gate-hopping failure mode —
a reviewer that FAILs round after round on the *presentation* of evidence the
marker demonstrably contains (e.g. round 1 marker-shape, round 2 smoke-digest
formatting, never reviewing the code) can no longer bounce the implementer or
trip the Step 5d cap-3 strategy pivot. The round counter does NOT increment
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
- **severity=BLOCKER** → either address (option 1 above) OR pivot
  strategy per `pivot_criteria.code_review_ensemble_cap_3`. BLOCKERs
  CANNOT route to the deferral gate. If neither address nor pivot
  resolves it, post `epm:failure v1 failure_class: code` referencing the
  concern_id and set status:blocked (halt_criteria id=6
  `concern_unresolved`).

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
- **`final_verdict == FAIL` + revision_round<3** -> stay at status
  `running` (implementing sub-phase). Re-spawn the implementer with
  BOTH event bodies (Claude + Codex) AND the reconcile event (if
  present) as part of the brief. Implementer posts v<n+1>; loop back
  to 5a with `revision_round = n+1`.
- **`final_verdict == FAIL` + revision_round>=3** -> **STRATEGY PIVOT,
  not block** (see CLAUDE.md "STATE-TO-`blocked` criteria" and
  workflow.yaml § pivot_criteria.code_review_ensemble_cap_3). The
  implementation strategy isn't working — same diff family has failed
  3 rounds. Re-invoke `/adversarial-planner` with explicit pivot scope
  in the brief: "the implementer can't make this strategy work. Propose
  a fundamentally different design (drop the offending component / swap
  model / change architectural approach)." Treat the revised plan as a
  fresh implementer cycle (`revision_round` RESETS to 1 on the new
  plan). Track pivots in a top-level `epm:strategy-pivot v<n>` marker
  with the pivot rationale and what changes.

  Only after ~3 fundamentally different strategies have all FAILed AND
  no further autonomous angle exists, move status to `blocked` and
  exit. Post the §5 marker with `--exit-kind failure-exit` and notes
  enumerating the strategies tried and why each failed. User decides:
  override, revise scope, or escalate the diagnostic loop.

  Bare cap-3 FAIL is NOT a block trigger. Continuing autonomously via
  pivot is the default.

**Codex twin no-show fallback.** If the Codex wrapper posts
`epm:failure v<m>` with `failure_class: codex-output-malformed` or
`failure_class: infra` (codex plugin missing), proceed with
single-reviewer (Claude-only) decision-making for that round. Do NOT
block on the Codex twin's absence; cap-3 still applies to the Claude
reviewer's count. Surface this to chat as one line: `Codex twin no-show
this round; using Claude reviewer only.`

##### Step 5.bis: Pre-dispatch checks (compute-deviation + whack-a-mole)

Fires once per implementer round, AFTER code-review-PASS, BEFORE any
pod-provision or experimenter-dispatch action in Step 6. Two
independent triggers run in sequence:

**5.bis(a) — Compute-deviation pivot.** Scan the task's
`events.jsonl` for `epm:compute-deviation v1` markers posted in the
current implementer round (highest version with the same round number).
If present:

1. Parse the marker's body for `component`, `planned_wall_h`,
   `projected_wall_h`, `ratio`, `basis`. If the marker carries
   `action: auto_descope_to_<spec>`, the implementer (or a prior
   orchestrator tick) already accepted an auto-descope — log the
   descope to chat as one line and advance to Step 5.bis(b).
2. Otherwise, attempt auto-descope per
   `workflow.yaml § pivot_criteria.compute_deviation_over_2x`:
   walk the planner's §9 stratification dimensions in priority order
   (seeds → framings → cells-per-stratum); for each dimension, compute
   the descoped projection (drop the dimension to its min-N-for-power
   per the planner's §9 stratification spec). The first descope whose
   ratio ≤ 1.5× AND keeps every dimension ≥ its min-N wins.
3. **Auto-descope success.** Post `epm:compute-deviation v2` with
   `action: auto_descope_to_<spec>`, update the implementer's per-cell
   parameters in the launch command, log to chat as one line, advance.
4. **Auto-descope fails** (no dimension keeps ratio ≤ 1.5× while
   staying above min-N): branch on session mode.

   - **Interactive mode** (`EPM_AUTONOMOUS_SESSION` unset/falsy): surface
     `gates.conditional.compute_deviation_resolution` (id=12) with the
     2-option prompt. Quote the ratio inline. On `continue_as_is`,
     advance to Step 5.bis(b) with the original parameters. On
     `accept_descope_to_<X>_with_caveats`, post `epm:compute-deviation v2`
     with the chosen descope spec + caveats and advance.

     <!-- gate: gates.conditional.compute_deviation_resolution -->

   - **Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): NEVER raise the
     ask AND never print the two options as a text menu. Auto-resolve
     per § Autonomous session behavior →
     `compute_deviation_resolution`: pick
     `accept_descope_to_<X>_with_caveats` if any descope dimension
     preserves majority statistical power (≥0.6 of the planned cells);
     else `continue_as_is` and quote the projected ratio inline. State
     `Decision: <choice> because <reason>` AND EXECUTE the resolved
     action in this same turn (post `epm:compute-deviation v2` with the
     chosen `action:` and advance to Step 5.bis(b)); do NOT state the
     Decision and then end the turn.

**5.bis(b) — Whack-a-mole detector.** Scan the task's `events.jsonl`
for `epm:new-bug-class v1` markers posted in the trailing 5
implementer rounds (rounds N-4..N, where N is the current round).
EXCLUDE rounds whose `epm:experiment-implementation v<n>` event note
contained the regex `<!-- workflow-fix-candidate v1 -->` (per the
workflow-fix-on-bug protocol; those drive workflow-improver, not
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
experimenter launch pattern); (d) **workloads longer than ~20h on
GCP** — the lane pins `--instance-termination-action=DELETE` +
`--max-run-duration` (default 24h), so a multi-day sweep is deleted
mid-run; set `spec.extra["max_run_duration"]` deliberately or use the
RunPod override. **When overriding to RunPod, name the residual gap in
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
on the pushed branch — `--repo-branch` defaults to the current branch
(the GCE startup script clones from origin). Four more gcp/auto
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
**Drivers that default `REPO_ROOT` to the RunPod path need it threaded
on gcp/auto** — the GCE startup script clones to `$WORKLOAD_ROOT`
(`/workspace/eps-issue-<N>`), cds there, then runs the workload
command verbatim, so a driver defaulting
`REPO_ROOT=/workspace/explore-persona-space` dies at its first `cd`
under `set -e` and the EXIT trap powers the VM off; compose
`--workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" bash scripts/<driver>.sh'`. (g)
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
    --issue <N> --intent "$INTENT" \
    ${BACKEND:+--backend "$BACKEND"}
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

Parse JSON. If `ok=false`, post `epm:preflight v1` event with the
errors/warnings, then post the §5 marker:
```bash
uv run python scripts/post_step_completed.py --issue <N> --step 6c \
  --exit-kind failure-exit --notes "preflight failed; user must fix"
```
EXIT. User fixes, re-runs.

#### Step 6d: Dispatch experimenter (launch-only), then orchestrator polling loop

The experimenter agent is **launch-and-exit only** — it syncs the pod,
preflights, launches the job via `nohup`, posts `epm:run-launched`, and
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

For an experiment whose pipeline chains ≥2 distinct phases of data
generation before training (typically gen → drift → train → eval →
aggregate), the architecture-parity gate above is NOT enough: it checks
that smoke and sweep share ONE code path, not that EVERY phase ran. A
resume-skip design serializes bug discovery — each pod cycle surfaces
the next phase's bug — so before the GPU production launch the FULL
pipeline must have executed once at tiny N (≈2-3 rows, 1 cell, 1 seed)
so EVERY phase runs end-to-end on CPU / 1-GPU. Confirm the implementer's
`## Smoke run` report (per `experiment-implementer.md` § "End-to-end
smoke run PER PHASE") carries a sub-section with exit code `0` + an
artifact digest for EACH phase the pipeline executes — not just training
or data-gen. Any phase missing, or showing only `--help` / `import` /
`--dry-run` evidence → **REFUSE to dispatch**; bounce to the implementer
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
   <N>` + the recent `events.jsonl` tail: if `epm:results v1` +
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

Spawn `experimenter` subagent via `Agent()`. Brief:
- The plan path (the `plans/plan.md` symlink) + the code-reviewed
  branch (`issue-<N>`)
- Pod name (`epm-issue-<N>` or parent's)
- The exact `nohup` launch command from the plan's Reproducibility Card
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
    Bash(
        run_in_background=True,
        command=(
            f"sleep 540 && uv run python scripts/backend_poll.py --issue {N}"
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
    #                                  exit the polling loop, and park at the
    #                                  user gate per Step 6d.4 below.
    #   status == "stalled" | "dead" -> post epm:failure v1 with failure_class
    #                                   inferred from log_tail_excerpt
    #                                   (run scripts/failure_classifier.py on
    #                                   the excerpt); run CRON-TEARDOWN (see
    #                                   below); set status:blocked; exit.
    #   status == "running"        -> milestone-already-posted by the poller
    #                                  if new_milestone was true; loop again.
    #                                  If the JSON also has
    #                                  gpu_idle_advisory_posted == true, act
    #                                  per "GPU-idle advisory handling" below
    #                                  before the next tick.
```

(`current_phase` is `"running"` by default; when the poller emits a
milestone marker like `phase: post_eval`, update the local
`current_phase` from the milestone before the next tick so the title
reflects the latest phase.)

The `poll_pipeline.py` helper posts `epm:progress` events itself when it
sees a phase transition, AND drains pod-side sentinel files (posting
their carried markers from the VM via `task_workflow.post_event`). The
orchestrator's only post-tick duties are: exit the loop on `status=done`,
park at the user gate on `status=gate` (Step 6d.4), and post
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
(Incident: task #521, 2026-06-10 — a VM-side pid file plus an
`epm:progress`-only relaunch produced `status=dead, pid_alive=False`
while the pod run was healthy.) On the GCP lane the marker's `pod=`
field MUST be the instance name (`eps-issue-<N>`) — `GcpBackend.poll`
matches relaunch markers on that field to follow the new process
(incident #612): a mismatched value (e.g. a RunPod-style `pod-<N>`)
rejects the marker and the poll keeps reading the frozen startup-script
phase, and an omitted `pod=` is accepted only via the launch-time
`epm:cluster-launched` timestamp baseline, so include it explicitly.

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
   `CronCreate(cron="*/20 * * * *", prompt="/issue-tick <N>", recurring=True, durable=False)`
   — a 20-minute, session-scoped, in-memory recurring fire of the
   lightweight `/issue-tick <N>` skill (dies with the session, auto-
   expires at 7 days like the default pod TTL; the harness jitters
   recurring fires so ticks don't all land on a fixed wall-clock mark).
   The 20-minute interval is chosen deliberately: the Anthropic prompt
   cache TTL is 5 minutes, so a 10-minute interval was the worst case —
   always cold (every tick re-prices the ~200K+ prefix at 1.25×), double
   the ticks for no caching benefit. 20 minutes accepts the cold-cache
   cost (the lightweight prompt makes it cheap) AND halves the tick
   count. Going sub-5-min would share the cache but cost MORE wall-clock
   fires per stalled stretch, which is the opposite of what the backstop
   is for. The `/issue-tick` skill is ~few-hundred tokens, vs the
   44K-token full `/issue` skill — so 12 idle ticks across a 4-hour
   idle stretch cost a few thousand tokens instead of ~1M. Then
   immediately re-`CronList`
   and assert EXACTLY ONE job matches
   `prompt.strip() == "/issue-tick <N>"`. If the harness normalised the
   stored prompt such that the ARM-GUARD would later miss, this assert
   fails loud NOW rather than silently stacking a duplicate cron on
   every subsequent re-entry.

Then proceed to the polling loop. Auto-arming is required ONLY for
pod-backed `kind: experiment` runs reaching Step 6d.2;
`kind: analysis|infra|batch|survey` and follow-up paths that never enter
the polling loop do NOT arm it.

**CRON-TEARDOWN procedure (run INLINE at every terminal / park exit site,
not only here in prose).** `CronList`, find the job with
`prompt.strip() == "/issue-tick <N>"`, `CronDelete(id=...)` it. The backstop
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
`completed` / `archived` / `awaiting_promotion` issue is a no-op (the
re-invoked skill reads terminal/park state and exits without re-arming).
Run CRON-TEARDOWN the moment you spot a stranded cron
(`CronList` → `CronDelete`).

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

##### Step 6d.4: On `status=gate` — park at a pod-side user gate

Pod-side dispatchers cannot post markers directly (the `task.py`
branch-guard and the CLAUDE.md "Pod-side code NEVER shells out" rule),
so they write a sentinel file at `/workspace/logs/issue-<N>-*.json`
that `poll_pipeline.py` drains. When a sentinel carries a non-empty
`gate` field, the poller posts the carried marker from the VM (e.g.
`epm:fact-candidates v1`) and returns `status=gate` with `gate=<name>`.

The orchestrator parks at the named gate inline rather than continuing
to poll — the pipeline itself has EXITed and is waiting on a user
answer.

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
  fact-teaching task for the on-pod resume contract.)

- **Unrecognised `gate` name**: log a one-line WARN, post `epm:failure
  v1` with `failure_class: code` and `reason: unrecognised_gate_name`
  (the `code|infra|data` taxonomy has no `workflow` class; the failure
  classifier defaults unknown classes to `code` anyway), a note pointing
  at the unrecognised gate name + the sentinel path, run CRON-TEARDOWN
  (`CronList` → `CronDelete` the `/issue-tick <N>` job), set
  `status:blocked`, exit. This forces a workflow-fix-candidate before
  the gate name can silently no-op.

Run CRON-TEARDOWN before parking (`CronList` → `CronDelete` the job with
`prompt.strip() == "/issue-tick <N>"`) — the pipeline has EXITed and no pod is
burning GPU, so the backstop should not keep re-firing `/issue-tick <N>` (which
would re-surface the gate question every 20 min). The user's
re-invocation after posting the resume marker re-enters Step 6d.2 and
re-arms via the ARM-GUARD. After posting the resume marker, EXIT the
skill cleanly via `uv run python scripts/post_step_completed.py --issue <N>
--step 6d --exit-kind parked` (the §5 `epm:step-completed` marker); the
user's re-invocation of `/issue <N>` resumes the polling loop. The polling-loop's terminal
transitions are now `running → verifying` (on done), `running → running`
(after a parked gate resumes), or `running → blocked` (on stalled/dead
or unrecognised gate).

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
2. The auto-armed backstop cron (`CronCreate(cron="*/20 * * * *",
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
`running → running` (parked at a user gate per Step 6d.4; resumes on
the next `/issue <N>` invocation after the user posts the gate-resume
marker), or `running → blocked` (on stalled/dead or an unrecognised
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
  `epm:results v1` payload. The orchestrator's polling-loop terminal
  tick (Step 6d.2) reads the sentinel on its next poll and posts
  `epm:results v1` from the local VM via `task.py post-marker`. The
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
    with TBD → resolved values)
  - `wandb_url` (string)
  - `hf_hub_url` (string)
  - `worktree_path` (string, absolute path on local VM)
  - `final_commit_sha` (string, 40-char SHA)
  - `gpu_hours_used` (float)
  - `gpu_hours_budgeted` (float)
  - `plan_deviations` (list of `{deviation: <str>, rationale: <str>}`)

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
   | `infra` | OOM, ENOSPC, NCCL, vLLM init failure, SSH refused, 401/gated repo, library traceback (vllm/transformers/peft/trl/torch/xformers) | Re-spawn the **experimenter** on the SAME branch, post `epm:experimenter-respawn v<n+1>`. NO implementer round. Cap 3 respawns; on 4th, status -> `blocked`. |
   | `code` | Python `Traceback` from `src/explore_persona_space/` or `scripts/` (our code), `AssertionError`/`TypeError`/`KeyError` from our code, CUDA OOM listing 2+ sibling `Process <pid> has <X> GiB memory in use` entries (parallel fan-out cells co-located on one device — GPU-pinning bug, #557) | Status back to `running` (implementing sub-phase), re-spawn `experiment-implementer` with the failure context. Loop through Steps 4b -> 5 -> 6 again. Cap 3 (existing). |

   **Missing `failure_class` — invoke the classifier script.** Do NOT
   reason about regex patterns inline; the patterns are owned by
   `scripts/failure_classifier.py` and reading them yourself drifts.
   Instead, shell out:

   ```bash
   # Pipe the failure body via stdin to avoid shell-quoting traps.
   cat <(uv run python scripts/task.py view "$N" --json \
       | jq -r '.events[] | select(.kind == "epm:failure") | .note') \
     | uv run python scripts/failure_classifier.py --body - \
         --log "$LATEST_LOG_PATH"
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
   failure).** A lightweight in-flight hook, not a new pipeline step;
   auto-continue, no gate. Both crash-fix shapes — the `code`-row
   `experiment-implementer` round and the `infra`-row experimenter
   respawn whose relaunch applied a fix — are REQUIRED (by
   `experiment-implementer.md` § "Crash-fix rounds: failure-lesson
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
   <!-- /epm:failure-lesson -->
   ```

   On receiving a crash-fix report carrying the block, the orchestrator
   takes three actions:

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
      standing rule 2026-06-02 as workflow fixes). The point is
      same-day cross-session sharing: a sibling session's next agent
      spawn loads the memory within minutes, instead of waiting for the
      nightly `/daily` sweep (on 2026-06-11, #537 and #545 re-hit
      overlapping failure classes hours apart with no persistence
      channel). Lessons are written for the NEXT agent — 1-3 sentences,
      the trap + the fix, no transcript dumps.
   3. **On `gotcha_candidate: yes` — route as a workflow-fix
      candidate.** Treat the lesson as a prose workflow-fix candidate
      targeting `.claude/rules/gotchas.md` and dispatch it through the
      existing workflow-fix-on-bug auto-spawn default
      (`.claude/rules/workflow-fix-on-bug.md`); the lesson block is the
      surfaced prose.

   If the resolving report omitted the block (older agent spawn, or a
   refusal killed the report tail), reconstruct it from the failure
   context + fix diff yourself before posting — don't bounce the round
   for the missing block alone. `/daily` remains the deduplicating
   consolidator: it reads the day's `epm:failure-lesson v1` markers,
   dedupes against agent memories, promotes recurring lessons into
   `.claude/rules/gotchas.md` or the relevant rule file, and prunes
   over-eager `generalizes: yes` memory entries.
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
3. **`methodology-writer` early spawn** (the early-spawn half of Step
   9a-quater; `stage=methodology-reference round=1`) — only when the
   9a-quater kind-gating says the step runs at all (`kind: experiment`
   always; `kind: analysis` only with a methodology surface;
   `infra | batch | survey` never — evaluate the skip BEFORE spawning).
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
the guard PER STAGE (see the parallel-stage note in the Step 9 entry
guard): do not re-dispatch a stage whose own breadcrumb is within its
freshness window, even when another stage's marker is the latest event.

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
  teardown + Step 9 progression run as normal.) Once artifacts are
  confirmed at permanent
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
  deliverable, then `--skip-confirm-artifacts`.

  ```bash
  # ONE call for every backend. Exit 0 = confirm PASS + teardown done;
  # exit 3 = confirm FAIL (teardown SKIPPED, evidence preserved); exit 2
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

**Checkable guard rule (run at Step 9 / Step 8 entry on every
re-invocation).**
1. Read the most recent events.jsonl marker via
   `task.py latest-marker <N>` (and `task.py view <N> --json` for the tail
   if needed).
2. If the latest marker is a `stage-dispatch` breadcrumb (`epm:progress`
   note beginning `stage-dispatch `) for the CURRENT stage+round AND there
   is NO result marker for that same stage+round posted AFTER it (i.e. the
   breadcrumb is genuinely the latest event), THEN compare its timestamp to
   now against the **stage-aware freshness window**:
   - Window = **30 min** for Codex-ensembled rounds (`interpreting` round 1
     AND `clean-result` round 1 — these spawn both the Claude critic AND a
     `codex-*-critic` twin at `--effort high|xhigh` via `companion task`;
     round 1 commonly exceeds 15 min wall time).
   - Window = **15 min** for everything else (`verifying`,
     `interpreting`/`clean-result` rounds 2–3 which are Claude-only, and
     any other Step 8/9 stage).
   - **age ≤ window** → the subagent is presumed STILL RUNNING. EXIT the
     skill cleanly (`post_step_completed.py ... --exit-kind parked
     --notes "stage <stage> round <r> still in flight (dispatched <Δ>m
     ago, window <W>m); backstop tick yielding"`). Do NOT re-dispatch —
     let the live work finish; the next tick (or the live subagent's own
     completion) advances the pipeline.
   - **age > window** → the stage looks genuinely STALLED (a subagent that
     never posted its result). Proceed to re-dispatch it normally (the
     freshness window is what distinguishes "live" from "stalled").
3. If the latest marker is a RESULT marker (or anything other than a
   current-stage `stage-dispatch` breadcrumb), there is no in-flight work —
   proceed with the normal Step 9 logic below.

**Parallel-stage note (results-landed spawn).** Step 8's results-landed
parallel spawn can put `verifying`, `interpreting` round 1, and
`methodology-reference` breadcrumbs in flight at once. Apply the rule
PER STAGE: scan `events.jsonl` backwards for the CURRENT stage's most
recent `stage-dispatch` breadcrumb (skipping other stages' breadcrumbs
and result markers) rather than inspecting only the single latest
event. A stage is in flight when ITS breadcrumb has no matching result
marker after it and is within the freshness window.

The 15-min default comfortably exceeds a single Claude analyzer / critic /
verifier turn; the 30-min Codex round-1 window covers a high-effort
Codex twin's wall time without re-dispatching live work and risking a
double-writer on `body.md`. Both fit cleanly under the 20-min backstop
cadence × 2-miss safety margin, so a genuinely stalled stage is still
re-dispatched within ~2 ticks (≈40 min worst case). This guard is the
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
     relevant `#### <finding>` H4 under `### Findings` (no separate
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

   Neither sees the analyzer's reasoning. Independence is load-bearing.

3. **Apply ensemble decision rule** (see
   (see workflow.yaml § ensemble_review)):

   | Claude | Codex | Action |
   |---|---|---|
   | PASS | PASS | `final_verdict = PASS`. Concatenate suggestions for analyzer's optional polish. |
   | REVISE | REVISE | `final_verdict = REVISE`. Union the revision requests (dedup exact-same). |
   | PASS vs REVISE (or vice versa) | (the other) | Spawn `reconciler` (marker mode). Brief: role=`interpretation-critic`, both event bodies, interpretation body path, eval JSON paths, figure paths. Reconciler posts `epm:review-reconcile v<n>` with binding PASS or REVISE. `final_verdict = reconciler's verdict`. |
   | Codex no-show (`epm:failure`) | (any) | Fallback: `final_verdict = Claude verdict`. Surface "Codex twin no-show round <n>" to chat. |

   Reconcile rounds do NOT increment the per-reviewer round counter.

**If `final_verdict == REVISE` (rounds 2-3):**

Re-spawn analyzer (fresh context, sees original data + ALL critique
feedback: Claude event + Codex event + reconcile event if any).
Analyzer posts `epm:interpretation v2`. Re-spawn the ensemble (fresh
contexts, sees v2 + prior critique events). Posts both
`epm:interp-critique v2` and `epm:interp-critique-codex v2`. Apply rule
again.

**Max 3 rounds per reviewer.** After round 3, advance regardless with
full critique history.

**On PASS (or max rounds reached):**

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

Then proceed to **9a-humanize (TL;DR humanize-loop pass)** before
advancing to clean-result-critic.

**9a-humanize. TL;DR humanize-loop pass** (orchestrator-level — only on
the first time `epm:clean-result-drafted v1` is posted, NOT on round-2/3
revisions out of 9a-bis)

The analyzer ran an inline humanize-quick self-pass on the TL;DR block
during its draft (analyzer.md Step 4.5). This orchestrator step adds the
second-opinion layer: a real `/humanize loop` invocation with a separate
hostile critic subagent the analyzer could not spawn from inside its
own subagent context.

The pass targets the `<section id="tldr">` block ONLY (mirrored to the
markdown `## TL;DR` H2 if the body shape is markdown rather than the
legacy HTML card). Design dropdown, figcaption, and reproducibility appendix
are out of scope — they carry project jargon on purpose, and the
clean-result-critic in 9a-bis enforces register discipline on them.

**Procedure:**

1. Read the published body via `task.py view <N>`; extract the TL;DR
   block.
2. Invoke `/humanize loop` with the TL;DR block as the target. **Read the
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
     "Statistics" rules and `verify_task_body.py` Lens 7)
3. Loop until all axes score ≤ 1 OR **3 orchestrator-level cycles**
   reached.
4. If the loop revised the TL;DR, write the new body to
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
   residual debt: axis X scored 2 — flagged to user").

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
`headline_affecting` schema) record whether any follow-up is executable
with ZERO new GPU AND would plausibly move the parent's headline. When
such a follow-up exists and has not yet been run on this task, the
orchestrator AUTO-RUNS it inline BEFORE the clean-result-critique gate
(9a-bis) — so the critic gates the UPDATED body, not a body that
already names a free win it didn't take. This step fires in BOTH
interactive and autonomous (`EPM_AUTONOMOUS_SESSION=1`) sessions
identically (unlike the autonomous-only `auto_run: yes` GPU-backed
routing at 9b (same-issue loop / child filing) — the two mechanisms
are orthogonal). The whole
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
line (`No free-analysis + headline-affecting follow-ups to auto-run`)
and proceed directly to 9a-bis.

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

**Auto-run procedure.** For the single highest-priority unran entry
(the first one in the analyzer's surfaced order; tie-break to the one
the analyzer flagged as `headline_affecting: yes` with the most
explicit eval-data path):

1. **Dispatch breadcrumb** (Step 9 entry guard convention):
   ```bash
   uv run python scripts/task.py post-marker <N> epm:progress \
     --note "stage-dispatch stage=free-analysis-followup round=1 subagent=experiment-implementer worktree=<abs path or 'repo-root'>"
   ```
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
   can SHA-pin them per the existing analyzer.md Step 3 rule.
4. **Capture the headline before / after.** Read the current `body.md`
   H1 title before the re-spawn and the analyzer-produced H1 after,
   plus the LOW / MODERATE / HIGH confidence tag in each.
5. **Re-spawn `analyzer`** (fresh context) with the new analysis
   output + the prior body. The analyzer folds the new result into
   the existing clean-result body (typically updating one
   `#### <finding>` H4 and possibly the H1 title / confidence tag),
   re-runs `verify_task_body.py` (must still PASS), and writes the
   revised body via `task.py set-body <N> --file ...`. The analyzer's
   Step 6.5 still fires on this re-run, but the loop guard above
   prevents another 9a-ter dispatch within the same task.
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
this layer ensures the body matches the 2-content-section nested-design
(v2) clean-result shape (per `.claude/skills/clean-results/SPEC.md`):
three required H2s in order (`## Human TL;DR` / `## TL;DR` /
`## Reproducibility`), with `## TL;DR` opening `### Motivation` →
`### What I ran` → `### Findings` (parent) → `#### <finding>` per
result, and confidence in the H1 title tag only (v2 bodies bear the
`<!-- clean-result-v2 -->` sentinel). The body reads in the right
registers — casual first-person inside `## TL;DR`, LessWrong
research-post register inside each `#### <finding>` H4. Discipline
rules: see `.claude/skills/clean-results/SPEC.md` (canonical structure,
registers, exemplars, figure captions, and research-communication
principles).

**Round 1:**

Worktree-cwd sessions run the Step 5a spec-freshness check before
dispatching this round's critics.

1. Spawn `clean-result-critic` agent (fresh context, does NOT see
   analyzer reasoning). The critic reads the published body + the
   latest `epm:interpretation v<n>` event, runs
   `scripts/verify_task_body.py` +
   `scripts/audit_clean_results_body_discipline.py` as authoritative
   mechanical passes, and scores against 15 lenses including the
   Lens 7 statistical-framing rule absorbed from the retired
   `reviewer` agent, Lens 13 planned-vs-actual coverage (added
   2026-05-27 after task #391's C-axis silent drop), Lens 14
   binding-concerns audit (task #455), and Lens 15
   contaminated/failed-data-gate-arm check (task #407). Posts
   `epm:clean-result-critique v1` on the source
   task with PASS or REVISE.

2. Spawn `codex-clean-result-critic` (Codex twin) in parallel on
   round 1 only. Brief contract (matches
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
   path); and dispatch `codex_task.py` for this twin from the repo
   root, never an issue-worktree cwd. Posts
   `epm:clean-result-critique-codex v1`. Apply the
   ensemble decision rule (same shape as Step 5c — PASS+PASS, REVISE
   union, reconciler on disagreement), BUT first run the
   procedural-only strip below.

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
   register / story-arc / statistical-framing lens judgment.

**If REVISE (rounds 2-3):**

Re-spawn `analyzer` agent (fresh context, sees raw data + all
interp-critique history + the latest clean-result-critique). Analyzer
revises the `epm:interpretation` event AND edits the task body in
place via `task.py set-body <N> --file ...`. Re-runs
`scripts/verify_task_body.py` (must still PASS). Re-spawn
`clean-result-critic` against the revised surfaces. Posts the next
critique version. Rounds 2-3 are Claude-only (no Codex twin).

**Max 3 rounds.** After round 3, advance regardless and fold the
residual structural / register debt into the chat-side summary so the
user can decide whether to patch before promoting.

**On PASS (or max rounds reached):**

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

**9a-quater. Methodology + hyperparameters reference — LATE JOIN** (only
if status is `reviewing`, after the 9a-bis loop's PASS, before the
`awaiting_promotion` park below; the agent itself was EARLY-SPAWNED at
Step 8's results-landed parallel spawn — see § Split schedule below)

Every `kind: experiment` clean-result auto-gains a standalone
**methodology + hyperparameters + worked-examples** reference at
`docs/methodology/issue_<N>.md`, committed to the repo and mirrored to a
**secret** gist, linked from the clean-result body in TWO places: a
reader-facing one-line `**Methodology:**` pointer at the TOP of the
body (immediately after the `<!-- clean-result-v2 -->` sentinel,
before `## Human TL;DR`) and a `**Methodology reference:**` row in
`## Reproducibility` (the artifact-index entry). The reference is **findings-blind**: it describes only HOW the
experiment was run (conditions, training recipe, eval recipe, verbatim
training / eval / output examples, reproducibility pointers) and never
restates findings / interpretation / confidence / next-steps. The fresh
context of the `methodology-writer` agent enforces this structurally —
the agent never reads `## Human TL;DR`, `## TL;DR`, `## Findings`, the
H1 confidence tag, or any `epm:interpretation` body. Fires in BOTH
<!-- example: anti-pattern -->
interactive and autonomous sessions identically. Auto-continue (NOT a
new `AskUserQuestion` gate); the halt-criterion contract is preserved.
<!-- autonomous-mode: auto-resolve -->
Same behavior in interactive and autonomous sessions: no AskUserQuestion
is ever raised by this step; the marker `epm:methodology-doc-generated v1`
is the durable record consumed by re-entry idempotency.

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
    amendment + Reproducibility slice, and re-writes the doc with a
    new `## <followup_label> arm` section appended — parent sections
    preserved verbatim.
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
   findings-blindness: `## TL;DR` / `## Findings` / the H1 confidence
   tag never enter the agent's context. Prompt discipline is defense in
   depth on top of this structural cut, not the primary mechanism:
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
   `.env` content. The methodology-writer reads only the
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

   (a) **Top of body — the reader-facing pointer.** Insert exactly
   this line immediately AFTER the `<!-- clean-result-v2 -->` sentinel
   (i.e. right under the H1 title), BEFORE `## Human TL;DR`, with a
   blank line on each side (legacy bodies without the sentinel:
   directly under the H1 title line instead):
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
   exit text so the orchestrator can auto-spawn `workflow-improver`).
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
`clean-result-critic` Lens 11 (see CLAUDE.md ontology table). The
`reviewing` status now exists ONLY as the single-step parking point
between clean-result-critic PASS and `awaiting_promotion`. The skill
moves through it in one transition with no agent dispatch:

```bash
uv run python scripts/task.py set-status <N> awaiting_promotion \
  --note "clean-result-critic PASS; parking for user promotion."
uv run python scripts/task.py post-marker <N> epm:status-changed \
  --note "reviewing -> awaiting_promotion (no final reviewer step; absorbed into clean-result-critic Lens 11)"
```

**Run CRON-TEARDOWN now.** `awaiting_promotion` is the terminal/park
transition for an experiment: the pod was terminated at Step 8 and this is
a human gate, so there is nothing left to auto-drive. `CronList` →
`CronDelete` the job with `prompt.strip() == "/issue-tick <N>"` so the backstop
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
(rebase-merge `issue-<N>` -> `main`, no prompt, keep the worktree). The
code / figures / `eval_results` the run produced land on `main`
immediately so the next experiment inheriting from `main` gets any
shared-infra fix this branch carried (this is the #456 -> #466 fix). The
science verdict (`useful` / `not-useful`) is orthogonal and still parks
below for the user. Merging does NOT block the park: an auto-merge
conflict posts `epm:merge-failed v1` and surfaces one line in chat, but
the task still parks at `awaiting_promotion` for promotion. Idempotent —
skip if `epm:merged` already exists.

**Autonomous follow-up auto-spawn (autonomous mode only — fires here
because Step 10b never runs autonomously).** When
`EPM_AUTONOMOUS_SESSION=1`, the parent task parks at
`awaiting_promotion` and Step 10 / 10b never fire on their own
(promotion is ALWAYS human-only). To stop autonomous research from
stalling on every result, the orchestrator fires the follow-up proposer
HERE — after the auto-merge has landed the clean-result on `main`, and
before CRON-TEARDOWN — and routes the `auto_run: yes` proposals by
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
3. Parse the proposals, keep those with `auto_run: yes` in ranked
   order, and PARTITION them by `question_relation`. **Untagged
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
     Select the TOP-RANKED `same` + `auto_run: yes` proposal ONLY if
     the autonomous round cap allows (fewer than 2
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
   round (the backstop cron stays armed; it drives the loop).
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

One mechanism, three entry points: (a) the Step 9b autonomous
partition above (`source: proposer-9b`), (b) a chat-requested
same-question follow-up (`source: user-chat` — the chat session posts
`epm:followup-scope v1` on #N, then re-invokes `/issue <N>`; the Step
0 followup-scope dispatch lands here), and (c) an interactive Step
10b pick (`source: step-10b-pick`). Step 9a-ter (the inline
free-analysis auto-run) is this loop's zero-GPU sibling under the
same principle — a follow-up that answers the SAME question as the
task Goal runs ON this issue; 9a-ter handles the zero-GPU case
inline, this loop handles the GPU-backed case.

**Interactive liveness backstop (arm BEFORE dispatching loop work).**
An INTERACTIVE (non-autonomous) session driving this loop — typically
entry point (b), a chat session — must arm the `/issue-tick <N>`
backstop cron (same `CronList`/`CronCreate` ARM-GUARD shape as Step 0 /
Step 6d.2) before dispatching its first planner / implementer / stage
subagent, and must post every stage-dispatch breadcrumb
(`stage=followup-<phase>`, Step 9 entry-guard convention) with the
`worktree=` field. Know what each mechanism covers: the cron handles
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

1. **Scope marker.** Ensure an `epm:followup-scope v1` exists for this
   round (the Step 9b partition posts it at step 6 above; the chat /
   Step 10b entry points post it before re-invoking). Fields per
   workflow.yaml § markers: `followup_label` (kebab-slug; names the
   artifact dir `eval_results/issue_<N>/<followup_label>/`), `source`,
   the verbatim proposal spec (or the user's verbatim chat request),
   and the GPU-hour estimate.
2. **Re-enter the pipeline.** **FIRST record the initiation mode as a
   tag** (before the status flip, so the `task.py` missing-tag warning
   stays quiet): `uv run python scripts/task.py add-tag <N>
   followup-auto` when `source: proposer-9b`; `uv run python
   scripts/task.py add-tag <N> followup-manual` when `source:
   user-chat` or `source: step-10b-pick`. EXACTLY these two tag names —
   a bare `followup` tag does not count (incident #533). (Both tags may
   accumulate over a task's life — they are history, not exclusive
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
   exit).
   - `/adversarial-planner` re-invoked in AMENDMENT scope: produces
     `plans/v{N+1}.md` as a ONE-VARIABLE diff plan against the issue's
     own latest prior run, not a from-scratch plan. Planner-exempt
     re-runs (step 2) skip this.
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
     changes (same ensemble shape as Step 5).
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
     clean-result body — a new `#### <finding>` H4 under `### Findings`
     (the v2 spec already supports multiple findings), updating the H1
     title / confidence tag if the result moves the headline. The
     `set-body` call passes NO `--snapshot` — `original-body.md`
     already preserves the pre-promotion original (see analyzer.md §
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
   (`followup_label` matching the scope marker, `source`, `round`,
   one-line `outcome`) when the loop re-reaches `awaiting_promotion`.
   This is the idempotency record: an `epm:followup-scope v1` with a
   matching run marker is RUN and is never re-dispatched.
5. **Round cap (autonomous only).** At most **2** autonomous same-issue
   GPU follow-up rounds per task, enforced by counting
   `epm:same-issue-followup-run v1` markers with `source: proposer-9b`.
   Beyond the cap, further `same` proposals survive in
   `epm:follow-ups v1` for manual pick. USER-REQUESTED rounds
   (`source: user-chat` or `step-10b-pick`) do NOT count against the
   cap — the user asked explicitly, and interactive plan approval
   still gates each one.

Status-machine summary: `interpreting` / `reviewing` /
`awaiting_promotion` / `completed` + unrun followup-scope →
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

1. Unit tests: `uv run pytest tests/ -v --tb=short`
2. Lint: `uv run ruff check . && uv run ruff format --check .`
3. Integration tests (conditional, if diff touches train/eval/orchestrate)
4. Coverage gap report (flags, does not auto-generate)

Post `epm:test-verdict v1`. PASS -> Step 10. FAIL (count < 3) -> stay
in `reviewing`, re-spawn implementer. FAIL (count >= 3) -> run
CRON-TEARDOWN (`CronList` → `CronDelete` the `/issue-tick <N>` job — idempotent;
no-ops for a code-change task that never armed one), then status to
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
     event.
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
     presence of an unrun `epm:followup-scope v1`.)
   - **No children in flight** AND task type is `experiment` ->
     **status `completed`**.
   - **type `infra` / `batch` / `analysis` / `survey`** (regardless of
     children) -> **status `completed`**. Code-change paths don't use
     `followups_running` because they don't seed experimental
     follow-ups via Step 10b.
   - **No `type` frontmatter** -> STOP, post an error event asking the
     user to add one. Do NOT pick a default, and do NOT advance until
     fixed.

6. Run CRON-TEARDOWN before applying the terminal status (`CronList` →
   `CronDelete` the job with `prompt.strip() == "/issue-tick <N>"`). For an
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

### Step 10b: Follow-up proposer (experiments only — runs ∥ Step 10c)

**Parallel spawn with Step 10c.** Steps 10b and 10c keep their
numbering and their per-step semantics, but their agents are spawned
CONCURRENTLY: evaluate both steps' skip conditions first (10b's
autonomous-mode short-circuit below; 10c's kind / `relates_to` skips),
then spawn `follow-up-proposer` AND `living-docs-updater` in ONE
message (two Agent calls, staggered a few seconds apart per the
CLAUDE.md 429 guidance). Both read the completed clean-result; their
outputs are independent (follow-up proposals vs a proposed docs diff).
Process each return per its own step text, and JOIN BOTH —
`epm:follow-ups v1` posted (or 10b skipped) AND the 10c proposal
handled (gate raised / parked per 10c) — before entering Step 10d. The
`living_docs_update` gate, all markers, and the user-confirmation
semantics are unchanged; only the spawn scheduling changed. If one
step's skip condition fires, spawn only the other's agent.

Auto-fires after `completed` for `experiment` tasks. Spawn the
`follow-up-proposer` agent with:

**Autonomous-mode short-circuit:** if an `epm:follow-ups-autospawned v1`
marker is present on the parent's `events.jsonl`, the proposer ALREADY
ran at Step 9b (the autonomous-mode follow-up auto-spawn site, fired
before the parent parked for promotion). SKIP re-spawning the proposer
here — it would duplicate the proposal list and is unnecessary. The
`epm:follow-ups v1` posted at Step 9b is still the canonical list for
the user; any `auto_run: no` proposals from that marker remain on the
table for the user to pick from manually post-promotion. Interactive
mode (no `epm:follow-ups-autospawned v1` ever posted) runs the proposer
here as normal.
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

**Route the user's pick by `question_relation`** (untagged proposals:
the treat-as-`substantially-different` fallback applies only when the
`epm:follow-ups v1` marker was posted before 2026-06-09; on a newer
marker the missing tag is a proposer-contract violation — classify
the picked proposal yourself against
`.claude/agents/follow-up-proposer.md` § "question_relation tag —
criteria" and note the violation in the resulting
`epm:followup-scope v1` / child-creation marker):

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
   batch (see Step 10b § Parallel spawn with Step 10c); spawn here only
   if it didn't. Brief: task
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
           "parks for a future /weekly backstop re-synthesis."
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
   re-invocation or for the `/weekly` backstop re-synthesis to
   reconcile. EXECUTE the continuation to Step 10d in this same turn;
   do NOT end the turn waiting on user confirmation.

This hook is idempotent: skip if `epm:living-docs-updated v1` or
`epm:living-docs-update-rejected v1` already exists on the task.

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

#### Merge safety guards (run before the merge commands)

Derive the two paths cwd-robustly FIRST — never via `git rev-parse
--show-toplevel`, which from a worktree cwd returns the WORKTREE root and
nests `$WT` into `.../issue-<N>/.claude/worktrees/issue-<N>` (incident #506,
2026-06-09: the guard snippet exit-128'd with "cannot change to ..."):

```bash
REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
WT="$REPO_ROOT/.claude/worktrees/issue-<N>"
```

A behind-`main` `issue-<N>` branch can carry stale copies of OTHER tasks'
`tasks/` state, a crash between merge and a status flip can strand a
task at the wrong status, AND a branch based on another still-unmerged
`issue-<M>` branch will replay `#M`'s old commits onto `main` if blindly
rebase-merged. Three guards:

1. **Foreign-`tasks/` guard.** `git diff --name-only origin/main HEAD --
   tasks/` MUST be empty except THIS task's own folder
   (`tasks/*/<N>/`). For any foreign `tasks/` path in the diff, run
   `git checkout origin/main -- <that file>` before merging. Never let a
   behind-`main` branch revert another task's `events.jsonl`. (Incident
   2026-06-01: #458's merge branch, 1,146 commits behind main, silently
   rewound `tasks/running/448/events.jsonl`.) The `--rebase` merge form
   below replays the branch's commits on top of current `main`, so files
   the branch never committed keep `main`'s version — this is what keeps
   the clean-result body (committed to `main` by `task.py`, never in the
   worktree) safe across the merge.
2. **Status already off `running`.** By both trigger points the status is
   well past `running` (`awaiting_promotion` for experiments; `completed`
   for code paths, flipped in Step 10 step 6 BEFORE this step). A crash
   mid-merge therefore cannot strand a terminated-pod task at `running`.
   On a later `/issue <N>` resume: if the PR is already merged AND status
   is still `running` for any reason, auto-advance rather than
   re-dispatching.
3. **Behind-`main` / non-`main`-base guard.** Compute:

   ```bash
   BEHIND=$(git -C "$WT" rev-list --count HEAD..origin/main)
   MB=$(git -C "$WT" merge-base HEAD origin/main)
   # is the merge-base reachable on origin/main's first-parent mainline?
   ON_MAINLINE=$(git -C "$WT" rev-list --first-parent origin/main \
     | grep -Fxq "$MB" && echo yes || echo no)
   ```

   The branch is **unsafe to blind-rebase** if EITHER `BEHIND` exceeds
   the threshold (default `200` commits — tunable; pick lower for repos
   with high churn, higher for slow-moving infra) OR `ON_MAINLINE=no`
   (branch was forked off another `issue-<M>` branch that is itself
   still unmerged). In the unsafe case, do NOT run `gh pr merge
   --rebase` — fall through to the **artifact-confirmed merge**
   procedure below. The Guard 1 foreign-`tasks/` checkout is necessary
   but not sufficient: it covers `tasks/`, but a behind-`main` branch
   also carries stale `src/` and `scripts/` from the parent branch, and
   a blind rebase replays both the parent's `tasks/` rewinds (already
   handled) AND its `src/` / `scripts/` regressions (NOT handled by
   Guard 1) onto `main`. (Incident 2026-06-03: `issue-479` was 1,153
   commits behind `origin/main` and based on the still-unmerged `#472`
   branch — a blind `gh pr merge --rebase` would have replayed `#472`'s
   old commits onto `main`, risking regression of ~50 foreign `tasks/`
   folders AND shared `#472` infra. The orchestrator caught it by hand;
   this guard encodes the catch.)

#### The auto-merge procedure (safe case: branch up-to-date and based on `main`)

```bash
PR=$(gh pr view <PR> --json number -q .number 2>/dev/null) || true
if [ -z "$PR" ]; then
  echo "No PR for issue-<N>; nothing to merge."   # skip; post nothing
else
  # Run guards 1-3 above first. If guard 3 says "unsafe", skip this
  # block and run the artifact-confirmed merge below instead.
  gh pr ready <PR>
  gh pr merge <PR> --rebase --delete-branch=false
fi
```

The `gh pr merge --rebase` form lands all per-item commits individually
on `main`; each is independently revertible via `git revert <sha>` (vs.
`--merge`, which reverts everything together). The user retains full
revert control after the fact — that is what makes a no-prompt merge safe
here. The worktree is deliberately NOT removed (`--delete-branch=false`,
no `git worktree remove`).

- **Success:** post `epm:merged v1` with the list of merge SHAs. Update
  the chat title with `merged`.
- **Failure** (rebase conflict, non-mergeable PR, non-fast-forward): do
  NOT swallow it (fail-fast). Post `epm:merge-failed v1` with the `gh` /
  `git` error, surface ONE line in chat naming the branch + worktree path
  for manual resolution, and CONTINUE — an experiment still parks at
  `awaiting_promotion`; a code-change task still completes. The merge is
  retried (idempotently) on the next `/issue <N>` re-invocation.
- **Autonomous mode** (no user present): same as above — the auto-merge
  proceeds. No deferral. (This reverses the prior "default NO" autonomous
  behavior; merge to `main` is no longer user-gated.)

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
  full_rebase_deferred: true, reason: "branch <BEHIND> commits behind
  main; based on <PARENT> (not on mainline)", verified_paths: [...]}`.
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
  # Files this branch ADDED (status A) vs origin/main, restricted to
  # this task's own paths — never sweeps shared src/ or scripts/.
  git -C "$WT" diff --name-only --diff-filter=A origin/main HEAD -- \
    "tasks/*/<N>/" "figures/issue_<N>/" "eval_results/issue_<N>/" \
    "eval_results/issue_<N>_*/" "ood_eval_results/issue_<N>/" \
    > /tmp/issue-<N>-additive-files.txt
  ```

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
  xargs -a /tmp/issue-<N>-additive-files.txt git checkout issue-<N> --
  xargs -a /tmp/issue-<N>-additive-files.txt git add --
  git diff --cached --name-only   # sanity echo: spot any foreign staged entries
  xargs -a /tmp/issue-<N>-additive-files.txt git commit -m "issue-<N>: surgical additive checkout (full rebase deferred — guard 3)

  Branch was <BEHIND> commits behind main and based on <PARENT>
  (not on mainline), unsafe to blind-rebase. Cherry-picked this
  task's own added files only; shared src/ / scripts/ unchanged." --
  git push origin main
  ```

  Then post `epm:merged v1` with `{artifact_confirmed: true,
  full_rebase_deferred: true, surgical_checkout: true, files:
  [...]}`. Same chat title update as above.

- **Surgical checkout itself fails** (file conflicts, push rejected
  after one `git pull --rebase --autostash` retry; plain rebase fails on the always-dirty shared root) — post `epm:merge-failed v1`
  with the error, surface ONE line in chat (branch + worktree path +
  one-line reason), CONTINUE. Same fail-fast policy as the safe case.

Never blind-`gh pr merge --rebase` a branch that tripped guard 3 — that
is the exact #458 / #479 incident class this section exists to prevent.

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
| Step 5b code-review FAIL revision_round>=3 | 5b | error path | `failure-exit` |
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

| Status at resume | `epm:*` events present | Interpretation | Action |
|------------------|------------------------|----------------|--------|
| `planning` | no `epm:plan` | planner was cancelled | re-run adversarial-planner |
| `plan_pending` | `epm:plan` exists | awaiting user approval | show plan path, EXIT |
| `running` (implementing) | no `epm:experiment-implementation` (or `epm:results` for infra), no `epm:proposed-tests` either | implementer was cancelled | re-spawn implementer |
| `running` (implementing) | `epm:proposed-tests v<n>` exists, no `epm:experiment-implementation`, no `epm:approve-tests` event posted **after** the `proposed-tests` event | TDD mode: tests posted, awaiting user approval | show the `proposed-tests` event timestamp + the `approve-tests` reply instruction, EXIT |
| `running` (implementing) | `epm:proposed-tests v<n>` exists, an `epm:approve-tests` event exists **after** the `proposed-tests` event, no `epm:experiment-implementation` | TDD tests approved by user | re-spawn implementer with `tdd_approved=true`; brief instructs implementer to write implementation against the approved tests, then post `epm:experiment-implementation v1` as normal |
| `running` (implementing) | latest `epm:code-review` is FAIL, round < 3 | revision in progress | re-spawn implementer with critique |
| `running` (implementing) | latest `epm:code-review` is FAIL, round >= 3 | exhausted retries | status to `blocked`, ask user |
| `running` (code-reviewing) | neither `epm:code-review` nor `epm:code-review-codex` for the current implementation version | both ensemble reviewers were cancelled | re-spawn both code-reviewer + codex-code-reviewer in parallel |
| `running` (code-reviewing) | `epm:code-review v<n>` exists, no `epm:code-review-codex v<n>` | Codex twin not yet returned (or wrapper crashed) | re-spawn `codex-code-reviewer` only |
| `running` (code-reviewing) | `epm:code-review-codex v<n>` exists, no `epm:code-review v<n>` | Claude reviewer not yet returned | re-spawn `code-reviewer` only |
| `running` (code-reviewing) | both `epm:code-review v<n>` and `epm:code-review-codex v<n>` exist, verdicts disagree (PASS-class vs FAIL), no `epm:review-reconcile v<n>` | reconciler not yet started | spawn reconciler |
| `running` (code-reviewing) | both `epm:code-review v<n>` and `epm:code-review-codex v<n>` exist, verdicts agree | ensemble decision ready | apply Step 5c rule and advance |
| `running` (code-reviewing) | `epm:code-review-codex` is `epm:failure` (codex-output-malformed or infra) | Codex twin no-show | proceed with Claude-only decision per Step 5d fallback |
| `running` (workload) | no `epm:results` for > 4h | experimenter crashed silently | post `epm:stale`, ask user |
| `running` (workload) | latest event is `epm:failure` with bounce-back proposal | experimenter bounced to implementer | status back to `running` (implementing), re-spawn experiment-implementer |
| `uploading` | no `epm:upload-verification` PASS | verifier not run or failed | re-run upload-verifier |
| `interpreting` | no `epm:interpretation` | analyzer not started | spawn analyzer |
| `interpreting` | `epm:interpretation` exists, neither `epm:interp-critique` nor `epm:interp-critique-codex` for the current version | both ensemble critics not started | spawn `interpretation-critic` + `codex-interpretation-critic` in parallel |
| `interpreting` | `epm:interp-critique v<n>` exists, no `epm:interp-critique-codex v<n>` | Codex twin not yet returned | re-spawn `codex-interpretation-critic` only |
| `interpreting` | `epm:interp-critique-codex v<n>` exists, no `epm:interp-critique v<n>` | Claude critic not yet returned | re-spawn `interpretation-critic` only |
| `interpreting` | both `epm:interp-critique v<n>` and `epm:interp-critique-codex v<n>` exist, verdicts disagree (PASS vs REVISE), no `epm:review-reconcile v<n>` | reconciler not yet started | spawn `reconciler` (marker mode) |
| `interpreting` | both ensemble events exist, verdicts agree OR reconcile event present, ensemble verdict REVISE, round < 3 | revision needed | re-spawn analyzer with all critique events |
| `interpreting` | ensemble verdict PASS or round >= 3, no `epm:clean-result-critique` | content honesty settled, structure + register loop not started | promote body in place if missing, then spawn clean-result-critic |
| `interpreting` | `epm:clean-result-critique` REVISE, round < 3 | structure / register revision in progress | re-spawn analyzer with the clean-result-critique |
| `interpreting` | `epm:clean-result-critique` PASS or round >= 3 | ready for review | advance to `reviewing` |
| `reviewing` | (no agent dispatch; transitional single-step) | reviewer step retired; absorbed into clean-result-critic Lens 11 | move to `awaiting_promotion`, run the Step 10d auto-merge procedure, post `epm:status-changed`, EXIT |
| `awaiting_promotion` | `classification == 'pending'` in body frontmatter, no `epm:merged` and PR unmerged | waiting for user to promote; worktree not yet merged | run the Step 10d auto-merge procedure (idempotent backstop — covers the case where the Step 9b auto-merge was interrupted), then show task path, prompt to promote via `task.py promote`, EXIT |
| `awaiting_promotion` | `classification == 'pending'` in body frontmatter, `epm:merged` present | waiting for user to promote; worktree already merged | show task path, prompt to promote via `task.py promote`, EXIT |
| `awaiting_promotion` | `classification != 'pending'` (user ran `task.py promote`) | user promoted | advance to Step 10 (auto-complete) |
| `interpreting` / `reviewing` / `awaiting_promotion` / `completed` | unrun `epm:followup-scope v1` (no matching `epm:same-issue-followup-run v1` with the same `followup_label`) | a `question_relation: same` follow-up is scoped to run ON this issue (takes precedence over the status rows above — see Step 0 "Same-issue follow-up dispatch") | route into the same-issue follow-up loop (Step 9b § Same-issue follow-up loop): set status to `followups_running` + tag `followup-auto`\|`followup-manual` and run the abbreviated cycle |
| `followups_running` | unrun `epm:followup-scope v1` (no matching `epm:same-issue-followup-run v1` with the same `followup_label`) | a same-issue follow-up round is mid-flight (this row takes precedence over the two children-based rows below) | resume the same-issue follow-up loop at the phase the stage breadcrumbs (`stage=followup-<phase>`) + latest markers indicate — do NOT restart from the top |
| `followups_running` | no unrun followup-scope; at least one open child task (`parent_id: <N>` in `body.md` frontmatter) not in `completed` / `archived` | legacy semantics: children still in flight | show child-task table, EXIT |
| `followups_running` | no unrun followup-scope; every child has reached `completed` / `archived` (or no children remain) | children all done | re-run Step 10: relabel parent to `completed` |
| `running` (workload) | pod alive + log advancing (`ssh epm-issue-<N> tail -1 <log_abs>`), no live bg-Bash poll for this session, latest `epm:*` marker is stale (no `epm:progress` in > ~15 min) | Step 6d.2 bg-Bash poll chain died — typically because a reaction turn emitted a corrupted/truncated tool-call (rendered as raw text), the harness had no bg work to wake on, AND the auto-armed backstop cron also died (a `durable=False` cron does not survive the session that registered it, so this row is reached mainly after a session restart / fresh recovery session). Pod and run are HEALTHY; only the session's monitor died. (Origin: tasks #462 / #463, 2026-06-02.) | Re-enter the polling loop by re-invoking `/issue <N>` once; it reads the latest `epm:run-launched` (`pod`, `pid`, `log_abs`), resumes Step 6d.2, and the Step 6d.2 step-1 guard AUTO-RE-ARMS the backstop cron (`CronList` for `prompt.strip() == "/issue-tick <N>"`, `CronCreate` if absent) so the next dead turn won't strand the run again — no user `/loop` typing needed. The lightweight `/issue-tick <N>` tick is what the cron fires; the full `/issue <N>` skill loads only on cold start, cold respawn, or the tick's stale-marker recovery branch. Do NOT re-spawn `pod_watch.py` / `pod.py watch` — that mechanism is retired per "Notes on the obsolete monitoring stack". |

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
row per `(kind)` as authoritative.

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

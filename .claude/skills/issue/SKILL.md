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

**Does NOT own:** proposing new experiments (-> `experiment-proposer`) or fleet-level dispatch across tasks (-> the PM session + `spawn_session.py spawn-issue --auto`).

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

- `steps/` — the per-step procedure bodies, one file per `### Step`
  (#2155). **These are the ONE exception to the line above: do NOT read
  them on first invocation.** Read a step's file when the run REACHES that
  step, and only that file. Reading them all at boot would restore the
  ~387K-token load this split exists to remove — and would be the exact
  regression the split is guarding against. Each `### Step` heading below
  carries a `> **Full procedure:**` pointer naming its file; the state
  machine, the Orchestration Procedure and every gate stay in SKILL.md, so
  routing never needs a companion.

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
genuine factual gap only the user can fill. (A reuse-vs-retrain ask and
a title-options ask were both rejected for asking before exhausting the
investigation.)

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
                                                                                                   |--> reviewing  <- clean-result-critic final adversarial gate
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
   backstop cron must not outlive the pause). Then persist resume state
   OFF-POD: a stopped volume is NOT durable — RunPod can destroy a stopped
   pod despite `keep-running` + the 7-day idle window (#1112 lost its
   done-JSONs this way; `.claude/rules/pod-config.md` § "Stopped pod volume
   is NOT durable") — so BEFORE stopping, upload done-JSONs / phase
   sentinels / partial eval JSONs (KB–MB, non-LFS path) to the issue's HF
   prefix, and prefer those copies on resume. Then Step 8-bis: stop any
   RUNNING pod (`uv run python scripts/pod.py stop --issue <N>`; add
   `keep-running` if the stopped pod should outlive the 24h stale-pod
   audit — the tag buys the cache/venv only, shielding against PROJECT
   janitors, never provider-side reclaim) and post `epm:pod-stopped v1`. A GCP instance is outside Step 8-bis's reach — it
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
   will respawn against the hold (#816).

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

**Guard-surface round: orchestrator turn discipline (#1563).** When the
task's target files are trigger-dense per the
`.claude/rules/trigger-dense-review.md` recognition heuristic
(guard/security hook scripts, destructive-command fixtures, refusal
corpora — knowable at Step 0 from the task body's target/scope lines or
the diff pathspec), EVERY ordinary orchestrator turn from the Step-0
state read through Step-10d merge narration follows
trigger-dense-review.md § Orchestrator ordinary turns: authored text
(marker notes, breadcrumbs, chat updates, commit messages) references
guard content by path + abstract class only, and own-turn reads of
guard-surface diffs/files stay counts-first + grep-anchored windows —
never wholesale-paged into context. Failure/forensics text stays
governed by the Step 6d.2 forensics-ingest discipline (#1546). When the
heuristic fires — at Step 0 or any later recognition point — ALSO
persist it as the durable task tag
(`uv run python scripts/task.py add-tag <N> trigger-dense`, idempotent)
so successor sessions and the poll-tick digest consumers (#1556/#1574)
key on the tag instead of re-deriving each turn; the Step 6d.2
loop-entry check is the dispatch-time backstop (#1587).

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
Capture that stdout and pass it verbatim to `mcp__happy__change_title` —
a deferred MCP tool, NOT a skill: load it via
`ToolSearch("select:mcp__happy__change_title")`, then call with
`{"title": <captured string>}` (`title` is REQUIRED — an empty `{}` is an
MCP -32602 error, and `Skill(happy:change_title)` is "Unknown skill").
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

  The dominant failure mode is text-menu-end-of-turn: a plain-text option
  menu ends the turn and blocks indefinitely on a user reply the user does
  not even know is pending; no hook can intercept prose, so only this
  section's prose prevents it (#503/#504/#505).

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
  text-menu-end-of-turn (#503/#504/#505); only THIS prose
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
  call. (Banned regression: `--auto` sessions parked overnight "awaiting
  user decision on the path forward" — #503/#504/#506/#509.)
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
  the planner + critic ensemble anyway (#488).
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
  the banned regression. Canonical case (#658/#663): an
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
  mid-run cost gate or block) (#506).
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
  modes at Step 9b.** Standing directive: a follow-up that
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
  v1` markers with `source: proposer-9b` — a defensive bound: the
  expensive-band partition runs at most once per task lifetime, so at
  most one such round is dispatchable today; see Step 9b § Round caps,
  #1588). All automatic follow-up
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
   handoff is the banned pattern (#505).
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

> **Full procedure:** `.claude/skills/issue/steps/00-step-0.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 0b: Defaulting & autofill

> **Full procedure:** `.claude/skills/issue/steps/01-step-0b.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 0c: Goal-of-experiment gate (safety net)

> **Full procedure:** `.claude/skills/issue/steps/02-step-0c.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 1: Clarifier gate

> **Full procedure:** `.claude/skills/issue/steps/03-step-1.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 2: Adversarial planning

> **Full procedure:** `.claude/skills/issue/steps/04-step-2.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 2b: Consistency checker (runs ∥ the Phase 2 critic ensemble)

> **Full procedure:** `.claude/skills/issue/steps/05-step-2b.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 2c: Inline plan approval

> **Full procedure:** `.claude/skills/issue/steps/06-step-2c.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 3: Approval check (backward compat, runs on re-invocation)

> **Full procedure:** `.claude/skills/issue/steps/07-step-3.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 4: Worktree + dispatch implementer

> **Full procedure:** `.claude/skills/issue/steps/08-step-4.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 5: Code review loop (Codex ensemble)

> **Full procedure:** `.claude/skills/issue/steps/09-step-5.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 6: Pod provisioning + experimenter dispatch (experiment only)

> **Full procedure:** `.claude/skills/issue/steps/10-step-6.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 7: Monitor -> results

> **Full procedure:** `.claude/skills/issue/steps/11-step-7.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 8: Upload verification

> **Full procedure:** `.claude/skills/issue/steps/12-step-8.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 9: Iterative interpretation + final review

> **Full procedure:** `.claude/skills/issue/steps/13-step-9.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 10: Auto-complete (fires after user promotes clean-result from `awaiting_promotion`, or `epm:test-verdict` PASS for code-change paths)

> **Full procedure:** `.claude/skills/issue/steps/14-step-10.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 10b: Follow-up proposer (experiments only — runs ∥ Step 10c)

> **Full procedure:** `.claude/skills/issue/steps/15-step-10b.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 10c: Living-docs update hook (experiments only)

> **Full procedure:** `.claude/skills/issue/steps/16-step-10c.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 10c-bis: Results-driven literature-positioning hook (findings-bearing tasks)

> **Full procedure:** `.claude/skills/issue/steps/17-step-10c-bis.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

### Step 10d: Auto-merge the worktree (both experiment and impl)

> **Full procedure:** `.claude/skills/issue/steps/18-step-10d.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

## Resume semantics

`/issue <N>` and `/issue <N> --resume` are identical. The skill is
always idempotent: it reads state from the task folder + recent
`events.jsonl` rows, computes the next action, and executes. There is
no "start from scratch" — the only way to reset is to manually edit
`body.md` and / or move the folder via `task.py set-status`.

### Step-completed re-entry skip-ahead (`epm:step-completed`)

> **Full procedure:** `.claude/skills/issue/steps/19-step-completed-reentry.md` — read that
> file when the run reaches this step. Routing, gates and the state
> machine stay in SKILL.md; only the step body moved (#2155).

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


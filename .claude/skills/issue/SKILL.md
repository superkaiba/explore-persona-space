---
name: issue
description: >
  End-to-end task-workflow orchestrator for experiments and code changes.
  Takes a task number (`<N>` = the integer that names `tasks/<status>/<N>/`),
  reads state from `body.md` frontmatter + `events.jsonl` markers under
  `tasks/<status>/<N>/`, and dispatches the next action (clarify ->
  adversarial-planner -> approval -> worktree + dispatch specialist ->
  preflight -> run -> analyzer -> clean-result-critic -> test-verdict ->
  auto-complete). clean-result-critic PASS (or test-verdict PASS for
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
External tracker records (GitHub issues, the legacy Sagan dashboard) are
historical evidence only and must never be used as workflow state.

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
done but with at least one open child stay at `completed` with
`has_clean_result=true`; child discovery is by frontmatter scan (see
Step 10 step 4 below).

The skill moves status in exactly four places:

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
       |-- OK --> planning              <- adversarial-planner + consistency-checker
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
                                                                               |--> uploading (verifying)  <- upload-verifier
                                                                                      |-- (all artifacts verified, pod terminated)
                                                                                         |--> interpreting  <- analyzer + interp-critic loop
                                                                                                |-- (interpretation refined, clean-result drafted in place)
                                                                                                   |--> reviewing  <- clean-result-critic final adversarial gate (Lens 7 absorbed retired reviewer)
                                                                                                          |-- PASS --> awaiting_promotion  <- AWAITING USER: promote clean-result
                                                                                                                        |-- (user promotes via task.py promote) -->
                                                                                                                              |-- open children w/ parent_id=<N> exist --> followups_running  <- waits for children; re-invoke /issue <N> later
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
| `gate_pending` | Hypothesis / kill-criterion gate blocked the plan; awaiting body fix. | **yes** |
| `planning` | Adversarial-planner is running. | no |
| `plan_pending` | Plan posted; awaiting user approval (task.py set-status N approved). | **yes** |
| `approved` | Plan approved; skill is creating worktree and dispatching the specialist. | no |
| `awaiting_approval` | Legacy alias for plan_pending; preserved for back-compat with older bodies. | **yes** |
| `queued` | Approved task awaiting an available pod or implementer slot. | no |
| `implementing` | Implementer or experiment-implementer is writing code. | no |
| `code_reviewing` | code-reviewer ensemble is reviewing the diff. | no |
| `testing` | Inline test-suite step (Step 9c, code-change paths only). | no |
| `running` | experimenter is running the workload on a pod. | no |
| `uploading` | upload-verifier is checking that artifacts landed on HF Hub / WandB / git. | no |
| `verifying` | Post-upload sanity / smoke-test step before interpretation. | no |
| `interpreting` | analyzer + interpretation-critic + clean-result-critic loops are running. | no |
| `reviewing` | Final adversarial review pass (clean-result-critic Lens 7 absorbed the retired reviewer step). | no |
| `under_review` | Legacy alias of reviewing; do not introduce new uses. | no |
| `awaiting_promotion` | User action: promote clean-result via task.py promote <N> useful|not-useful. | **yes** |
| `followups_running` | Parent task complete; children with frontmatter parent_id=<N> still in flight. | no |
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
approval) and `awaiting_promotion` (clean-result promotion). `blocked` and
`gate_pending` also need user attention but represent stalled / pre-pipeline
states. Everything between is automatic, short of a `task.py set-status
<N> blocked` override.

Abort affordance: any state, user runs `task.py set-status <N> blocked`
-> skill posts abort request via `epm:abort`, watcher kills run if one
exists.

---

## Orchestration Procedure

When invoked, ALWAYS follow this order. Skip only what the state dictates.

**Chat title updates (verbose format).** Fires on (a) every status
transition, (b) when an `epm:follow-ups` marker is posted, (c) when the
clean-result draft is finalized (Step 9a end), (d) when the merge
prompt fires (Step 10d).

Format string:
```
#<N> <type-frontmatter> — <human-readable status sentence>[ — next: <next-action>][ — followups: #X[, #Y]][ — clean-result: <claim summary trimmed to 60 chars>]
```

Examples:
- `#226 infra — implementing workflow improvements — next: code-review`
- `#226 infra — code-review FAIL round 2 — next: respawn implementer`
- `#137 experiment — completed — followups: #240, #241 — clean-result: persona collapse hero`

Helper pseudocode:

```python
def render_title(task, *, status_human, next_action=None, followups=None, clean_result=None):
    parts = [f"#{task.number} {task.type} — {status_human}"]
    if next_action:
        parts.append(f"next: {next_action}")
    if followups:
        parts.append("followups: " + ", ".join(f"#{n}" for n in followups))
    if clean_result:
        claim = clean_result.title[:60]
        parts.append(f"clean-result: {claim}")
    return " — ".join(parts)

# Cosmetic; mcp__happy__change_title may be unavailable. Soft-fail and continue.
try:
    mcp__happy__change_title({"title": render_title(...)})
except Exception:
    pass
```

If the MCP tool is unavailable (e.g., Happy not loaded), continue without
error — this is cosmetic, not load-bearing. Do NOT let a title-update
failure block the pipeline.

### Step 0: Load state

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

**Hard error: ambiguous status.** If `task.py view <N>` reports the task
exists in multiple folders (should be impossible because `task.py` holds
the flock — but the lint catches manual edits), abort with an error and
ask the user to reconcile. Do NOT pick.

**Soft error: status missing from frontmatter (legacy bodies), type missing,
or empty body.** These are recoverable; do NOT exit. Run Step 0b instead.

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
   current chat via `AskUserQuestion` for the minimum spec needed for the
   adversarial planner to design the task. The exact prompts depend on
   the task type (see `clarifier.md`); for an unknown type, ask:
   - "What's the goal of this task in one sentence?"
   - "What's the hypothesis or success criterion?"
   - "Is there a parent task or prior result this builds on? (task # or 'none')"
   - "Rough compute size? (small / medium / large)"

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

   Use `AskUserQuestion` with the inferred option as `(Recommended)`
   first. Apply via `task.py set-body --file ...` to update the
   frontmatter `type:` line. If the user is absent (e.g., autonomous
   loop), DO error and EXIT — the type field gates Step 7's completion
   variant and a guess here corrupts the lifecycle. Before exiting, post
   the §5 marker:
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
to Step 1.

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

  2. **Ask the user in the current chat.** Immediately after posting,
     ask the SAME numbered questions to the user in the current session.
     Use `AskUserQuestion` for small multiple-choice-style prompts;
     otherwise post a short numbered list as plain text and wait for a
     reply. Do NOT exit yet — give the user the option to answer inline
     so they don't have to context-switch to the dashboard.

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

### Step 2: Adversarial planning

Only if status is `planning`.

Invoke the `adversarial-planner` skill with the task body + clarifier
output as the task. The skill runs planner -> Phase 1.25 hypothesis-gate
-> fact-checker -> critic -> revise internally. For `experiment` tasks
the orchestrator forwards the task type via `--type experiment` so
Phase 1.25 fires; for non-experiment types Phase 1.25 is a no-op. The
same gate (`scripts/hypothesis_gate.py`) also runs in Step 1 (clarifier)
on the task body — see `clarifier.md` "Hypothesis-gate" section.

**Required sections in the final plan (enforced by this skill — reject
plans missing any):**
- Goal + hypothesis (experiments) or requirement + acceptance criteria (code changes)
- Method delta (what differs from prior related work)
- File paths + concrete diffs / config overrides
- **Reproducibility Card** (mandatory per CLAUDE.md) — all hparams, seeds,
  data, env versions, exact `nohup` command for experiments
- Success criteria with quantitative thresholds
- Kill criteria (what result would kill the thesis)
- Compute estimate in GPU-hours
- Target pod preference
- Plan deviations allowed vs must-ask

Post the plan body via `new-plan-version` (writes
`tasks/<status>/<N>/plans/v<K>.md` and rotates the `plan.md` symlink),
then announce it with an `epm:plan` event:

```bash
uv run python scripts/task.py new-plan-version <N> --file /tmp/issue-<N>-plan.md
PLAN_PATH=$(uv run python scripts/task.py find <N>)/plans/plan.md
uv run python scripts/task.py post-marker <N> epm:plan \
  --note "Plan v<K> written to $PLAN_PATH"
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

Also include estimated cost prominently in the `epm:plan` note, e.g.

> **Cost gate:** estimated 12 GPU-hours on 4× H100. Reply `approve` to dispatch.

### Step 2b: Consistency checker

After the adversarial planner produces an APPROVE-rated plan, but BEFORE
posting it as `epm:plan`, spawn the `consistency-checker` agent. It
receives:
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

Post `epm:consistency v1`. On BLOCK, send the plan back to the planner
for revision (loop, max 2 rounds). On WARN, append warnings to the
`epm:plan` event note. On PASS, proceed normally.

Then post the plan as `epm:plan v1` with the consistency results
appended.

Move the task to `plan_pending`:

```bash
uv run python scripts/task.py set-status <N> plan_pending \
  --note "Plan v1 ready for approval; consistency PASS."
```

### Step 2c: Inline plan approval

**Context-dependent behavior:**

- **Autonomous mode** (invoked from `auto-experiment-runner` or with no
  user present): EXIT immediately. The task sits at `plan_pending` until
  a user approves via the dashboard or a future `/issue <N>` invocation.
  This preserves the asynchronous review behavior. Before exiting, post
  the §5 marker:
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

  Use `AskUserQuestion` or a plain text prompt and wait for the user's
  reply.

  - **"Approve" / "1":** move task to `approved`. Post an `epm:plan-approved`
    event for audit trail. Continue to Step 4 in the **same invocation**
    — do NOT exit:
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
branch `issue-<N>` and open a draft PR.
```bash
git worktree add .claude/worktrees/issue-<N> -b issue-<N>     # reuse if it exists (resume case)
gh pr create --draft --head issue-<N> --body "Closes task #<N>."
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
- Required `report-back` fields
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
- `worktree`, `base`, `plan_marker_path`, `implementation_marker_path` —
  see `.claude/agents/codex-code-reviewer.md`.

Neither sees the implementer's reasoning — independence is load-bearing.
Dispatch in a SINGLE `Agent(...)`-call message with both spawned
`run_in_background=true` so they execute concurrently.

The Claude reviewer posts `epm:code-review v<n>` (PASS / CONCERNS /
FAIL). The Codex wrapper posts `epm:code-review-codex v<n>` (same
schema). Codex never sees `GH_TOKEN` — both wrappers post via
`task.py post-marker`.

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
- **`final_verdict == FAIL` + revision_round>=3** -> move status to
  `blocked`. Post abort summary, then post the §5 marker:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 5b \
    --exit-kind failure-exit \
    --notes "code-review-ensemble FAIL round 3+; status:blocked"
  ```
  EXIT. User decides: revise plan, escalate, or override.

**Codex twin no-show fallback.** If the Codex wrapper posts
`epm:failure v<m>` with `failure_class: codex-output-malformed` or
`failure_class: infra` (codex plugin missing), proceed with
single-reviewer (Claude-only) decision-making for that round. Do NOT
block on the Codex twin's absence; cap-3 still applies to the Claude
reviewer's count. Surface this to chat as one line: `Codex twin no-show
this round; using Claude reviewer only.`

### Step 6: Pod provisioning + experimenter dispatch (experiment only)

Only if status is `running` (entered from Step 5b PASS for `experiment`)
and no `epm:launch` marker exists.

#### Step 6a: HF gate auto-acceptance

Plans never make the human click through gated-model gate pages. Before
provisioning, scan the cached plan for HF model IDs and submit
gate-acceptance requests using the user's `HF_TOKEN`:

```bash
PLAN_PATH=$(uv run python scripts/task.py find <N>)/plans/plan.md
uv run python scripts/hf_gate_accept.py --from-plan "$PLAN_PATH"
```

The helper is idempotent (already-accessible repos exit `OK` immediately).
For "auto-approval" gates (the common case for almanach / Inria / Meta /
Qwen research releases) the access is granted on submission. For the
rare manual-approval gate the request is queued and the helper exits
with code 1 and a list of URLs.

- Exit code `0` -> proceed to 6b.
- Exit code `1` (manual approval still needed) -> post
  `epm:hf-gate-pending v1` with the URLs, leave status at `running`.
  Post the §5 marker:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 6c \
    --exit-kind clean --notes "hf-gate manual approval pending"
  ```
  EXIT. User clicks through, re-runs `/issue <N>`.
- Exit code `2` (`HF_TOKEN` missing) -> post `epm:hf-gate-pending v1`
  with diagnostic, status to `blocked`. Post the §5 marker:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 6c \
    --exit-kind failure-exit --notes "HF_TOKEN missing; status:blocked"
  ```
  EXIT.

This step is also re-run on the pod inside `bootstrap_pod.sh` so a token
pushed to the pod gets the same gate state as the local VM.

#### Step 6b: Pod provisioning

Pods are ephemeral — there is no permanent fleet.

Pick the path based on whether this task has a parent (read
`parent_id` from `body.md` frontmatter):

```bash
PARENT_ID=$(uv run python scripts/task.py view <N> --json | jq -r '.body_frontmatter.parent_id // empty')

# 1. If PARENT_ID is set AND `epm-issue-<PARENT_ID>` exists in `pod.py list-ephemeral`:
if [ -n "$PARENT_ID" ] && uv run python scripts/pod.py list-ephemeral --issue "$PARENT_ID" | grep -q epm-issue; then
  uv run python scripts/pod.py resume --issue "$PARENT_ID"
  # Use that pod for this child task (don't provision a new one).
  # Record the assigned pod as `epm-issue-$PARENT_ID` in the launch marker.
else
  # 2. Otherwise, provision a fresh pod. Infer --intent from the plan:
  #    training a 7B model -> ft-7b or lora-7b; eval/generation -> eval;
  #    70B work -> inf-70b/ft-70b. Override with --gpu-type/--gpu-count for
  #    anything else.
  uv run python scripts/pod.py provision --issue <N> --intent <inferred>
fi
```

`provision` enforces team scoping (`X-Team-Id`), SSH bring-up
(`startSsh: true`, exposes `22/tcp`), pinned image, and runs bootstrap
inline (uv, repo, .env, HF cache, HF gate-accept, preflight). On
provision failure post `epm:pod-pending v1` with the error and stay at
`running` (no implementer re-spawn — this is infra, not code). User
adjusts (capacity, intent override) and re-runs `/issue <N>`.

The pod name passed downstream is `epm-issue-<N>` (or the parent's
`epm-issue-<PARENT_ID>` for follow-ups). The experimenter does NOT pick
or create pods.

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

#### Step 6d: Dispatch experimenter

Spawn `experimenter` subagent via `Agent()`. The experimenter's scope is
**pod ops + monitoring + debugging only** — it does NOT write substantial
code (hot-fixes <=10 lines, no logic changes; see the experimenter
agent definition).

Brief passed to experimenter:
- The plan path (the `plans/plan.md` symlink) + the code-reviewed
  branch (`issue-<N>`)
- Pod name (`epm-issue-<N>` or parent's)
- The exact `nohup` launch command from the plan's Reproducibility Card
- Progressive monitoring schedule (per the experimenter agent definition)
- Required `report-back` fields (artifacts, WandB URL, HF Hub path,
  deviations, hot-fix log)

**NEVER include pod lifecycle commands (provision, stop, resume,
terminate, cleanup) in the experimenter brief.** The experimenter agent
spec explicitly forbids pod lifecycle management. Pod termination
happens automatically in Step 8 (after upload-verification PASS).
Including `pod.py terminate` or `pod.py stop` in the experimenter's
instructions bypasses the upload-verification gate and risks premature
destruction of artifacts that haven't been confirmed at permanent URLs.

Post `epm:launch v1` containing:
- Worktree path, branch, PR URL, code-review verdict (`PASS`)
- Pod + PID + log path
- WandB run URL (best-effort; experimenter updates if not known yet)
- Experimenter subagent ID (for monitoring)

**Spawn the §2 stall-detection watchdog (detached, on the local VM, NOT
the pod).** After the experimenter is launched and `epm:launch` is
posted, spawn:
```bash
uv run python scripts/pod.py watch --issue <N> --wandb-run-url <URL> \
  --log-path <server>:<path>
```
as a detached background process. Pid file written to
`.claude/cache/watch-<N>.pid` (the watchdog cleans it up on exit). The
watchdog probes WandB heartbeat + log-mtime every 60s; on >5min silence
it posts `epm:failure` with `failure_class: infra` and `reason: stall`,
flips the status to `blocked`, and exits.

Status stays at `running`. Before exiting, post the §5 marker:
```bash
uv run python scripts/post_step_completed.py --issue <N> --step 6d \
  --exit-kind clean --notes "experimenter dispatched; watchdog spawned"
```
EXIT. Experimenter runs autonomously. The experimenter posts
`epm:progress`, `epm:hot-fix` (if needed), and finally `epm:results`.
The watchdog stops itself when `epm:results` is observed, the status
moves out of `running`, or its pid file is deleted.

### Step 7: Monitor -> results

Experimenter is expected to post `epm:progress v1` events at major
milestones, optional `epm:hot-fix v<n>` events for in-line fixes
(<=10 lines, no logic change — see the experimenter agent definition),
and a final `epm:results v1` event containing:
- Final eval numbers (inline JSON snippet + path in repo)
- Reproducibility card (filled)
- WandB URL + HF Hub model/adapter URL
- Worktree path + final commit hash
- GPU-hours actually used vs budgeted
- Plan deviations + rationale
- Hot-fix log (commits + diffs applied during the run)

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
   `epm:failure` body SHOULD include a `failure_class: infra | code` field
   on its first non-blank line. Routing:

   | failure_class | Cause example | Action |
   |---|---|---|
   | `infra` | OOM, ENOSPC, NCCL, vLLM init failure, SSH refused, 401/gated repo, library traceback (vllm/transformers/peft/trl/torch/xformers) | Re-spawn the **experimenter** on the SAME branch, post `epm:experimenter-respawn v<n+1>`. NO implementer round. Cap 3 respawns; on 4th, status -> `blocked`. |
   | `code` | Python `Traceback` from `src/explore_persona_space/` or `scripts/` (our code), `AssertionError`/`TypeError`/`KeyError` from our code | Status back to `running` (implementing sub-phase), re-spawn `experiment-implementer` with the failure context. Loop through Steps 4b -> 5 -> 6 again. Cap 3 (existing). |

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
3. If `epm:results` exists, move status to `uploading` and proceed to
   Step 8.

### Step 8: Upload verification

Only if status is `uploading` and no `epm:upload-verification` marker
with verdict=PASS.

**Hard gate:** No experiment advances to interpretation until all
artifacts have permanent URLs. This prevents data loss from pod restarts
or cleanup.

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

Post `epm:upload-verification v1` event with per-artifact PASS/FAIL +
URLs.

- **PASS** -> terminate the pod, then move status to `interpreting` and
  proceed to Step 9. Once artifacts are confirmed at permanent URLs, the
  pod is no longer needed — interpretation runs locally:
  ```bash
  uv run python scripts/pod.py terminate --issue <N> --yes
  ```
  This destroys the pod (volume + container disk gone). Post
  `epm:pod-terminated v1` with the command output. If interpretation
  later needs GPU compute (e.g., to regenerate a figure from raw outputs
  that weren't downloaded), provision a fresh pod via `pod.py
  provision`. If the task has `parent_id`, terminate the parent's pod
  (`epm-issue-<PARENT_ID>`) instead. Skip the terminate call only if
  the task has a `keep-running` tag for known follow-up work in the
  same session.
- **FAIL** -> dispatch the `uploader` agent (up to 3 rounds) to close
  the gaps. The uploader receives the verifier's missing-artifacts
  list, lifecycle-aware resumes the pod if needed, pushes to HF /
  WandB / git, and posts `epm:upload-fix v1`. After each uploader
  round, re-run `upload-verifier`; it posts a fresh
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

### Step 9: Iterative interpretation + final review

This step has two sub-phases: **interpretation** (iterative
analyzer<->critic loop) and **final review** (clean-result-critic gate).

**9a. Iterative interpretation** (only if status is `interpreting`)

Only for `experiment` tasks. Code-change tasks never reach this step
because Step 5 already PASSed code-review and routed them to Step 9c
(the inline test-verdict gate) directly.

The interpretation loop produces a polished clean-result body through
iterative refinement between the analyzer and an interpretation-critic.

**Round 1:**

1. Spawn `analyzer` agent (fresh context) with raw result paths. The
   analyzer:
   - Writes the **Fact Sheet** (reproducibility card, artifact URLs,
     raw numbers, plots, sample outputs) — this is written once and not
     revised.
   - Writes the **Interpretation** (background, methodology, results
     claim + hero figure + main takeaways + confidence, next steps).
   - Generates plots via `paper-plots` skill.
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
2. Replaces `body.md` with the polished Sagan-card / markdown
   write-up:
   ```bash
   uv run python scripts/task.py set-body <N> --file /tmp/clean-result-body.md
   uv run python scripts/task.py set-title <N> "<claim summary> (HIGH|MODERATE|LOW confidence)"
   uv run python scripts/task.py set-clean-result <N>   # flips has_clean_result=true
   ```
3. Runs `scripts/verify_task_body.py <body-file>` — FAIL blocks the
   write-up.

Posts `epm:clean-result-drafted v1` on the source task with the title
and a 2-sentence recap.

Then proceed to **9a-bis (clean-result-critique loop)** before
advancing status.

**9a-bis. Clean-result-critique loop** (only if status is `interpreting`,
after Step 9a PASS)

Same shape as the interpretation-critic loop, but the critic checks
STRUCTURE + REGISTER not CONTENT. Content honesty was settled in 9a;
this layer ensures the body matches the v4 clean-result shape (per
`.claude/skills/clean-results/SPEC.md`) AND reads in the right
registers — casual user-voice in `## TL;DR`, LessWrong research-post
register in `## Summary` and `## Details`. Discipline rules: see
`.claude/skills/clean-results/SPEC.md` (canonical structure, registers,
exemplars, figure captions, and research-communication principles).

**Round 1:**

1. Spawn `clean-result-critic` agent (fresh context, does NOT see
   analyzer reasoning). The critic reads the published body + the
   latest `epm:interpretation v<n>` event, runs
   `scripts/verify_task_body.py` +
   `scripts/audit_clean_results_body_discipline.py` as authoritative
   mechanical passes, and scores against 11 lenses including the
   Lens 11 statistical-framing rule absorbed from the retired
   `reviewer` agent. Posts `epm:clean-result-critique v1` on the source
   task with PASS or REVISE.

2. Spawn `codex-clean-result-critic` (Codex twin) in parallel on
   round 1 only. Posts `epm:clean-result-critique-codex v1`. Apply the
   ensemble decision rule (same shape as Step 5c — PASS+PASS, REVISE
   union, reconciler on disagreement).

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
```bash
uv run python scripts/task.py set-status <N> reviewing \
  --note "clean-result-critic PASS; advancing to final review gate."
```

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

Then post the chat-side prompt:

> Clean-result-critic PASS. The polished body is now live on task #\<N\>.
> When satisfied, promote it (USER-ONLY — no automation may do this):
>   `uv run python scripts/task.py promote <N> useful`     (paper-relevant)
>   `uv run python scripts/task.py promote <N> not-useful` (archive candidate)
> Then re-enter `/issue <N>` to fire Step 10.

Post the §5 marker:
```bash
uv run python scripts/post_step_completed.py --issue <N> --step 9 \
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
in `reviewing`, re-spawn implementer. FAIL (count >= 3) -> status to
`blocked`.

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
   find tasks -path 'tasks/*/<child>/body.md' \
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
     `completed`.
   - **No children in flight** AND task type is `experiment` ->
     **status `completed`**.
   - **type `infra` / `batch` / `analysis` / `survey`** (regardless of
     children) -> **status `completed`**. Code-change paths don't use
     `followups_running` because they don't seed experimental
     follow-ups via Step 10b.
   - **No `type` frontmatter** -> STOP, post an error event asking the
     user to add one. Do NOT pick a default, and do NOT advance until
     fixed.

6. Apply the chosen status via `task.py set-status` (which performs the
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

### Step 10b: Follow-up proposer (experiments only)

Auto-fires after `completed` for `experiment` tasks. Spawn the
`follow-up-proposer` agent with:
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

The user can create follow-up tasks from these proposals by:
- Telling the main conversation agent to create them via
  `task.py new --parent <N> --kind experiment --title "..."`
- Manually copying the spec into a new task via `task.py new`

Each created follow-up task carries `parent_id: <N>` in its `body.md`
frontmatter; lint scans enforce that the parent exists. Lint output is
visible via `task.py audit`.

### Step 10d: Worktree merge prompt (both experiment and impl)

After Step 10b posts (the pod was already terminated in Step 8
immediately after upload-verification PASS), ask the user once via
`AskUserQuestion`:

> **Merge worktree `issue-<N>` into `main` now?**
> YES -> mark draft PR ready, **rebase-merge** so each commit lands
> individually on `main`, then `git worktree remove`.
> NO -> no-op; user merges later.

**30-minute cooldown gate.** Before prompting, run:

```bash
CREATED=$(gh pr view <PR> --json createdAt -q .createdAt)
AGE_SEC=$(( $(date +%s) - $(date -d "$CREATED" +%s) ))
if [ "$AGE_SEC" -lt 1800 ]; then
  echo "PR younger than 30 min; deferring merge prompt to next /issue invocation"
  exit 0
fi
```

The cooldown reduces the chance of merging before the PR has had time
for a quick human glance. Override allowed by manual `/issue <N>`
re-invocation after the cooldown elapses.

- **YES:**
  ```bash
  gh pr ready <PR>
  gh pr merge <PR> --rebase --delete-branch=false
  git worktree remove .claude/worktrees/issue-<N>
  ```
  The `gh pr merge --rebase` form lands all per-item commits
  individually on `main`; each is independently revertible via
  `git revert <sha>`. (Vs. `--merge` which creates one merge commit —
  reverts everything together.) Post `epm:merged v1` with the list of
  merge SHAs. Update chat title with `merged`.

- **NO:** post `epm:merge-deferred v1`.
- **Autonomous mode:** default NO; record event. Never auto-merge
  without user approval.

Idempotent: skip if either event (`epm:merged` or `epm:merge-deferred`)
already exists on the task.

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
| Step 9 `awaiting_promotion` user reviews | 9 | user-gated | `parked` |
| Step 10 still `classification = pending` | 10 | user-gated | `parked` |
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
| `reviewing` | (no agent dispatch; transitional single-step) | reviewer step retired; absorbed into clean-result-critic Lens 11 | move to `awaiting_promotion`, post `epm:status-changed`, EXIT |
| `awaiting_promotion` | `classification == 'pending'` in body frontmatter | waiting for user to promote | show task path, prompt to promote via `task.py promote`, EXIT |
| `awaiting_promotion` | `classification != 'pending'` (user ran `task.py promote`) | user promoted | advance to Step 10 (auto-complete) |
| `followups_running` | at least one open child task (`parent_id: <N>` in `body.md` frontmatter) not in `completed` / `archived` | children still in flight | show child-task table, EXIT |
| `followups_running` | every child has reached `completed` / `archived` (or no children remain) | children all done | re-run Step 10: relabel parent to `completed` |
| `running` (workload) | `.claude/cache/watch-<N>.pid` is missing AND no `epm:results` / `epm:failure` posted | §2 watchdog crashed or never started | re-spawn `uv run python scripts/pod.py watch --issue <N> ...` (skill side-effect; idempotent, the new watchdog inherits the run's heartbeat probes) |

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

- **Hypothesis-gate (`scripts/hypothesis_gate.py`).** Static regex gate
  runs at two surfaces for `experiment` tasks — Step 1 (clarifier, on
  the task body) and Step 2 / adversarial-planner Phase 1.25 (on the
  drafted plan body). Refuses to advance without `Hypothesis` AND
  `Kill criterion` / `Kill criteria` section headers. Override via body
  marker `<!-- epm:override-hypothesis-skip v1 -->` (with rationale) —
  every override fires an `epm:hypothesis-gate v1: OVERRIDE` audit
  event so the bypass is reviewable.
- **Never dispatch `compute:large` (>20 GPU-hours) without explicit
  user `approve`.** Small + medium can proceed on `approve` or
  `/approve`. Large requires `approve-large` to force a second thought.
- **Never auto-merge PRs.** User owns merge.
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

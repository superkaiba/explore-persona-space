---
name: issue
description: >
  End-to-end Sagan-experiment-driven workflow for experiments and code changes.
  Takes an experiment number (`experiments.number` in Sagan, NOT a GitHub issue
  number), parses state from `experiments.status` + workflow_event markers, and
  dispatches the next action (clarify -> adversarial-planner -> approval ->
  worktree + dispatch specialist -> preflight -> run -> analyzer -> reviewer ->
  test-verdict -> auto-complete). Reviewer PASS (or test-verdict PASS for
  code-change paths like type:infra / type:survey) auto-advances the experiment
  to completed in the Sagan dashboard. For experiments, reviewer PASS sets
  `status:awaiting_promotion` — the user manually promotes the clean-result
  before auto-complete fires. Experiments stay in the kanban until archived.
  Idempotent and resumable: re-invoking on the same `<N>` picks up where it left
  off.
user_invocable: true
---

# Sagan-Experiment Workflow

## Scope & Boundaries

**Owns:** the full experiment lifecycle — clarify → adversarial-planner → approval → worktree → dispatch → preflight → run → analyze → review → auto-complete.

**Invokes:** `experiment-runner` (run step), `adversarial-planner` (plan step), specialist agents (experimenter / implementer / analyzer / reviewer / code-reviewer).

**Does NOT own:** proposing new experiments (→ `experiment-proposer`) or overnight queue orchestration (→ `auto-experiment-runner`).

---

Invoke as `/issue <N>` or `/issue <N> --resume`. `<N>` is the experiment's
`number` column in Sagan's `experiments` table (the kanban view at
<https://sagan.superkaiba.com/experiments>). The skill drives an experiment
from proposal to clean-result.

**Guiding principle:** all durable state lives in Sagan's Postgres
(`experiments.status` + `workflow_events`). The local filesystem holds caches
only. You can close the terminal at any step and `/issue <N>` picks up cleanly.

## State backend

All durable state lives in the Sagan dashboard's Postgres at
<https://sagan.superkaiba.com>. The `experiments` table is the queue;
`/issue <N>` takes `experiments.number` (1-indexed, sequential).

Operate on state via `scripts/sagan_state.py` — both as a CLI and as an
importable Python module (`get_experiment`, `set_status`, `post_marker`,
`add_tag`, `latest_marker`, `list_by_status`, …). Requires
`SAGAN_API_TOKEN` (60-day sliding Bearer session, read from `.env`).

```bash
python scripts/sagan_state.py view <N>                    # show experiment + recent events
python scripts/sagan_state.py latest-marker <N>           # "where do I resume" query
python scripts/sagan_state.py list-by-status --status running
python scripts/sagan_state.py set-status <N> <status>     # advance the state machine
python scripts/sagan_state.py post-marker <N> epm:plan --note '...body...'
python scripts/sagan_state.py add-tag <N> <tag>
python scripts/sagan_state.py remove-tag <N> <tag>
```

Dashboard URL for an experiment: `https://sagan.superkaiba.com/e/experiment/<uuid>`.

## Status convention

Sagan's `experiment_status` enum is the single source of truth. The 12
production values used by `/issue` map to the kanban columns at
<https://sagan.superkaiba.com/experiments>:

| Status                | Meaning |
|-----------------------|---------|
| `proposed`            | Filed but not yet triaged. User files experiments here. |
| `planning`            | Adversarial-planner is running. |
| `plan_pending`        | User action: approve plan to advance. |
| `approved`            | Plan approved, dispatch pending. |
| `awaiting_approval`   | Awaiting an out-of-band gate (e.g. HF model access). |
| `running`             | All active-phase work between approval and clean-result-critic PASS rolls up here (implementing, code_reviewing, testing, training, uploading). The latest `epm:*` workflow event tells you which sub-phase. |
| `verifying`           | Upload-verifier running. |
| `interpreting`        | Analyzer + interpretation-critic loop + clean-result-critic loop are running (Step 9a and Step 9a-bis). Final critic before promotion as of 2026-05-13. |
| `reviewing`           | DEPRECATED 2026-05-13. Dedicated reviewer step retired; responsibilities absorbed by clean-result-critic. Kept in enum for legacy state recovery. |
| `awaiting_promotion`  | User action: review clean-result draft and promote to useful / not-useful. |
| `blocked`             | Stuck / paused; resolve dependency. |
| `completed`           | Terminal happy path. Sticky — `has_clean_result=true` is preserved. |
| `archived`            | Terminal sad path (duplicate / won't-fix / abandoned). Set explicitly. |

For followups, the parent→child relationship lives in `edges` of type
`parent`. Parents whose own work is done but with at least one open
child stay at `completed` with `has_clean_result=true`; the edges
surface the active followups in the dashboard.

The skill moves status in exactly four places:
1. **Step 1 (clarifier "All clear"):** `proposed` → `planning`.
2. **Step 9a (analyzer promotes the source row to a clean-result):** the source experiment row is updated in place — body + title replaced with the polished write-up, `hasCleanResult=true`, status → `awaiting_promotion`. Sagan auto-creates the child `runs` row with `classification='pending'` on the same PATCH. No separate clean-result row is created.
3. **Post-9a-bis (user promotes draft):** user runs `sagan_state.py promote <N> useful|not-useful` (or clicks Promote in the dashboard); the source experiment's `runs.classification` flips to the verdict and status advances to `completed`. The user then re-enters `/issue <N>` so Step 10 fires. Promotion is **user-only** — no agent or automation may flip `runs.classification` without explicit user invocation.
4. **Step 10 (auto-complete):** the source experiment is already at `completed` after promote; Step 10 just runs the follow-up proposer and merge prompt.

Between those, intermediate transitions (`approved → running → verifying →
interpreting → awaiting_promotion`) advance automatically as
each step completes. (The previous `interpreting → reviewing → awaiting_promotion`
flow was simplified 2026-05-13 when the dedicated reviewer step was retired
and its responsibilities folded into `clean-result-critic` at Step 9a-bis.)
Each transition writes a `workflow_events` row with
`metadata.marker_type = 'epm:*'` so the agent can resume where it left
off after a context reset.

## Companion files

- `markers.md` -- comment marker taxonomy (source of truth for state parsing)
- `clarifier.md` -- clarifying-question prompts per issue type
- `templates/` -- plan / results / analysis comment body templates

Read these on first invocation of the skill in a session.

---

## Auto-continuation policy

Auto-continue through every step EXCEPT the gates declared in
`.claude/workflow.yaml § gates` (see CLAUDE.md "Auto-continuation policy"
for the prose summary). The full enumeration — 6 inline gates + 1 park-and-wait
gate + 1 conditional gate — is canonical in workflow.yaml. Anywhere else
that an assumption needs to be made, STATE the assumption inline (one
line, prefixed `Assumption:`) and proceed; do NOT pause to ask.

**Exceptions that override auto-continuation:** subagent halt conditions
(see workflow.yaml § subagent_halt_conditions) and STATE-TO-`status:blocked`
criteria (see workflow.yaml § halt_criteria). When any of those fire,
EXIT regardless of the auto-continuation rule.

## The State Machine

State = `status:*` label. Transitions are enforced by this skill. Marker comments
provide the detailed payload for each state.

Principle: every state is either "an agent is actively working" OR "awaiting user
input." Distinct labels for each so a glance at the issue tells you whether it's
your turn.

```
status:proposed                           <- user has filed, clarifier hasn't run
  |-- (clarifier -> questions OR OK)
       |-- questions posted --> status:proposed (stays; awaiting user replies)
       |-- OK --> status:planning          <- adversarial-planner + consistency-checker
                  |-- (plan posted + consistency PASS/WARN)
                     |--> status:plan_pending    <- AWAITING USER: approve?
                            |-- (user approve) --> status:approved
                                                  |-- (worktree + draft PR)
                                                     |--> status:implementing    <- experiment-implementer (type:experiment) OR implementer (type:infra/batch)
                                                            |-- (epm:experiment-implementation OR epm:results posted)
                                                               |--> status:code_reviewing   <- code-reviewer (fresh context)
                                                                      |-- FAIL + count<3 --> status:implementing (loop, v+1)
                                                                      |-- FAIL + count>=3 --> status:blocked
                                                                      |-- PASS + [type:experiment] --> status:running   <- experimenter (pod ops + monitoring)
                                                                            |-- (epm:results posted)
                                                                               |--> status:uploading  <- upload verifier
                                                                                      |-- (all artifacts verified, pod terminated)
                                                                                         |--> status:interpreting  <- analyzer + interp-critic loop
                                                                                                |-- (interpretation refined, clean-result created)
                                                                                                   |--> (clean-result-critic Step 9a-bis ensemble; round-1 Claude+Codex, rounds 2-3 Claude only)
                                                                                                          |-- PASS --> status:awaiting_promotion  <- AWAITING USER: promote clean-result
                                                                                                                        |-- (user promotes) -->
                                                                                                                              |-- open `Parent: #<N>` children exist --> status:followups_running  <- waits for children to finish; re-invoke /issue <N> later
                                                                                                                              |-- no open children                  --> status:done_experiment (+ follow-up proposer)
                                                                                                          |-- REVISE --> status:interpreting (revise; rounds 2-3 Claude only)
                                                                      |-- PASS + [type:infra/survey] --> test-verdict (inline) --> status:done_impl
```

Hot-fixes during `status:running` (experimenter agent): small in-line fixes
(<=10 lines, no logic change) get committed on the issue branch and the run
continues. Anything beyond that bar bounces back to `status:implementing` for
a fresh experiment-implementer + code-reviewer round before the experimenter
relaunches.

There is no user sign-off step. Reviewer PASS (or `epm:test-verdict` PASS for code-change paths) is the terminal gate; completion is automatic. If the user disagrees with a done transition, they label `status:blocked` to reopen it. The "test-verdict gate" runs inline inside this skill (Step 9c) — there is no separate `tester` agent.

**Active vs awaiting-user states** (auto-generated from `.claude/workflow.yaml § statuses` — see `workflow.yaml § statuses`. Do NOT edit inside the fence; run `uv run python scripts/workflow_lint.py --emit-tables` to regenerate after a YAML edit):

<!-- workflow.yaml: AUTO-GENERATED (active-vs-awaiting) -->
| State | Who's working | User action needed? |
|-------|---------------|---------------------|
| `proposed` | User has filed; clarifier hasn't run. | no |
| `gate_pending` | Hypothesis/kill-criterion gate blocked the plan; awaiting body fix. | **yes** |
| `planning` | Adversarial-planner is running. | no |
| `plan_pending` | Plan posted; awaiting user `approve`. | **yes** |
| `approved` | Plan approved; skill is creating worktree + draft PR. | no |
| `implementing` | (experiment|infra|...)-implementer is writing code. | no |
| `code_reviewing` | code-reviewer is reviewing the diff. | no |
| `testing` | Inline test-suite step (Step 9c, code-change paths only). | no |
| `running` | experimenter is running on the pod. | no |
| `uploading` | upload-verifier is checking artifacts. | no |
| `interpreting` | analyzer + interpretation-critic + clean-result-critic loops are running. | no |
| `reviewing` | DEPRECATED 2026-05-13. The dedicated final-reviewer step was retired
and its responsibilities (statistical-framing rule, final
published-body fresh-context check) were absorbed by
`clean-result-critic` (see Step 9a-bis). Kept in the enum for legacy
state recovery; new issues never write this status.
 | no |
| `blocked` | Aborted or stuck; awaiting user triage. | **yes** |
| `awaiting_promotion` | User action: promote clean-result via /clean-results promote. | **yes** |
| `followups_running` | Parent is done; children with `Parent: #<N>` are still in flight. | no |
| `done_experiment` | Terminal: experiment finished + clean-result promoted. Issue stays OPEN. | no |
| `done_impl` | Terminal: code change shipped + reviewed. Issue stays OPEN. | no |
| `archived` | Set explicitly for duplicates / won't-fix / abandoned experiments. | no |
<!-- /workflow.yaml: AUTO-GENERATED -->

The two user-gated states in the active lifecycle are `plan_pending` (plan approval) and `awaiting_promotion` (clean-result promotion). `blocked` and `gate_pending` also need user attention but represent stalled / pre-pipeline states. Everything between is automatic, short of a `status:blocked` override.

Abort affordance: any state, user labels `status:blocked` -> skill posts abort
request, watcher kills run if one exists.

---

## Orchestration Procedure

When invoked, ALWAYS follow this order. Skip only what the state dictates.

**Chat title updates (verbose format).** Fires on (a) every status-label
transition, (b) when a `epm:follow-ups` marker is posted, (c) when the
analyzer promotes the source row to a clean-result (Step 9a), (d) when
the merge prompt fires (Step 10d).

Format string:
```
#<N> <type:label> — <human-readable status sentence>[ — next: <next-action>][ — followups: #X[, #Y]][ — claim: <claim summary trimmed to 60 chars>]
```

Examples:
- `#226 type:infra — implementing workflow improvements — next: code-review`
- `#226 type:infra — code-review FAIL round 2 — next: respawn implementer`
- `#137 type:experiment — done_experiment — followups: #240, #241 — claim: persona collapse hero`

The `claim:` segment is included once `hasCleanResult=true` on the source
row (the analyzer renamed the row to the claim summary at Step 9a). There
is no separate clean-result row — the source row IS the clean-result.

Helper pseudocode:
```python
def render_title(issue, *, status_human, next_action=None, followups=None, claim=None):
    parts = [f"#{issue.number} {issue.type_label} — {status_human}"]
    if next_action:
        parts.append(f"next: {next_action}")
    if followups:
        parts.append("followups: " + ", ".join(f"#{n}" for n in followups))
    if claim:
        parts.append(f"claim: {claim[:60]}")
    return " — ".join(parts)

# Cosmetic; if mcp__happy__change_title is unavailable, log and continue.
mcp__happy__change_title({ "title": render_title(...) })
```

If the MCP tool is unavailable (e.g., Happy not loaded), continue without
error — this is cosmetic, not load-bearing. Do NOT let a title-update failure
block the pipeline.

### Step 0: Load state

```
# Returns the experiment row + the 50 most-recent workflow_events
# (marker history) + open approval requests, mirroring what the agent
# pipeline needs to resume on any step. The /api/experiments/by-number/<N>
# endpoint caps events to 50 on the server side, so long-running
# experiments (e.g. #80 has 100+ markers) don't bloat the payload.
python scripts/sagan_state.py view <N>
```

From the result, derive:
1. **Current state** = the `status:*` label value (exactly one should exist)
2. **Issue type** = the `type:*` label value (`experiment`, `infra`, `survey`)
3. **Marker map** = scan comments for `<!-- epm:<kind> v<n> -->` opening tags, build a dict

**Hard error: >1 `status:*` labels.** True ambiguity — abort with an error comment listing
the conflicting labels and asking the user to remove the wrong one. Do NOT pick.

**Soft error: 0 `status:*` labels, missing `type:*`, or empty body.** These are
recoverable; do NOT exit. Run Step 0b instead.

### Step 0b: Defaulting & autofill

Runs only when at least one of {0 `status:*` labels, missing `type:*`, empty body} holds.
Goal: get the issue into the minimum shape Step 1 needs without bouncing back to the user
just to add labels. Order:

1. **`status:*` missing →** apply `status:proposed` automatically:
   ```
   python scripts/sagan_state.py set-status <N> proposed
   ```
   No user interaction. Defaulting an unlabelled issue to `proposed` is the obvious
   read of the project-board convention (Todo column = `proposed` or no `status:*`).

2. **Body empty (or <50 chars of substance) →** ask the user in the current chat via
   `AskUserQuestion` for the minimum spec needed for the adversarial planner to design the
   issue. The exact prompts depend on issue type (see `clarifier.md`); for an unknown
   type, ask:
   - "What's the goal of this issue in one sentence?"
   - "What's the hypothesis or success criterion?"
   - "Is there a parent issue or prior result this builds on? (issue # or 'none')"
   - "Rough compute size? (small / medium / large)"

   Plus **search the codebase + HF + arXiv before drafting** when the title hints at
   pulling existing artifacts (e.g., "use HF model X", "replicate paper Y") — list
   what you found and let the user pick. Don't fabricate a body from the title alone.

   Once the user answers, draft a body covering Goal / Hypothesis / Setup / Eval /
   Success criterion / Kill criterion / Compute / Pod preference / References, then
   patch the issue:
   ```
   python scripts/sagan_state.py set-body <N> --file .claude/cache/experiment-<N>-body.html
   ```
   Post a `<!-- epm:auto-defaults v1 -->` comment listing what was applied (label
   added, body drafted) so the audit trail is durable on the issue.

   **Audit-comment placeholder guard (when generating any `epm:audit` /
   `epm:auto-defaults` body):** before posting, run
   `grep -E "(^|\s|>)(TBD|TODO|placeholder|\[X\]|implementer fills)(\s|$|<)"`
   against the drafted body. Match → BLOCK the post and finish the audit. The
   regex catches placeholders mid-line as well as line-start (the original
   anchored form `^(TBD|TODO|...)` missed the embedded case — issue #275
   round-1 NIT-5).

3. **`type:*` missing →** infer from title cue, then confirm with the user:
   - Title prefix `Test:` / `Sweep:` / `Train:` → suggest `type:experiment`
   - Title prefix `Refactor:` / `Fix:` / `Add:` / `Migrate:` → suggest `type:infra`
   - Title prefix `[Batch]:` / `[Workflow]:` / body contains a numbered list of
     ≥3 unrelated fixes → suggest `type:infra` (one batch issue)
   - Title prefix `Analyze:` / `Re-analyze:` → suggest `type:infra`
   - Title prefix `Survey:` / `Read:` / `Lit review:` → suggest `type:survey`

   Use `AskUserQuestion` with the inferred option as `(Recommended)` first. Apply via
   `python scripts/sagan_state.py patch <N> --kind <chosen>` (or set via the dashboard). If the user is absent (e.g., autonomous
   loop), DO error and EXIT — the type label gates Step 7's Done variant and a guess
   here corrupts the project board. Before exiting, post the §5 marker:
   `uv run python scripts/post_step_completed.py --issue <N> --step 0b --exit-kind failure-exit --notes "type-label autofill loop; user override required"`.

4. **Other useful labels missing** (`compute:*`, `prio:*`):  do not block on these.
   `compute:*` will be set in the adversarial-planner's reproducibility card; `prio:*` is user-curated and
   never blocking.

   Note: legacy `aim:*` labels were deleted in #251 (slice 1). New issues do not use them.
   Topic categorization for new work lives in `docs/claims.yaml` (`topic` field) and
   in `RESULTS.md` / `eval_results/INDEX.md` H2 prose; no replacement GitHub labels exist.

After Step 0b, re-read the experiment (re-run `sagan_state.py view <N>` from Step 0) so downstream
state is computed from the now-patched issue, then continue to Step 1.

### Step 1: Clarifier gate

If `epm:clarify` marker missing (or user has replied but clarifier hasn't re-checked):
read `clarifier.md`, run the clarifier for this issue type, then:

**Before drafting any clarifying question, run the mandatory context-gathering
pass in `clarifier.md` Step 0** — search past GitHub issues + clean-results,
`.arxiv-papers/`, `external/`, `RESULTS.md`,
`eval_results/INDEX.md`, and `git log` for information that resolves the
ambiguity. Cut any question already answered by project knowledge; sharpen the
rest by quoting the source. When posting "All clear", include a brief
**Context resolved** bullet list of the issues/commits/papers consulted so the
inheritance chain is auditable.

- **All clear** (<=1 minor ambiguity) -> post `<!-- epm:clarify -->` with "No blocking
  ambiguities found. Proceeding to adversarial planning." advance label to `status:planning`,
  **and move the project column to Planning**:
  ```
  python scripts/sagan_state.py set-status <N> planning
  ```
  This is the one place where the project column transitions out of To do
  into the pipeline. Subsequent phases route automatically through
  `LABEL_TO_COLUMN` (Planning → Plan awaiting review → In flight → Awaiting
  promotion → Done) as the `status:*` label advances; explicit `set-status`
  calls are rarely needed.

- **Ambiguities remain** -> do BOTH of the following, in order:

  1. **Post on the issue.** Write the numbered questions as a `<!-- epm:clarify v<n> -->`
     comment. This is the durable log -- if the user closes the terminal, the questions
     are still there.

  2. **Ask the user in the current chat.** Immediately after posting, ask the SAME numbered
     questions to the user in the current session. Use `AskUserQuestion` for small
     multiple-choice style prompts; otherwise post a short numbered list as plain text
     and wait for a reply. Do NOT exit yet -- give the user the option to answer
     inline so they don't have to context-switch to GitHub.

  3. **If the user answers in chat:**
     - Post a `<!-- epm:clarify-answers v<n> -->` comment on the issue with the user's
       answers verbatim (lightly formatted -- one numbered bullet per question), so the
       issue is self-contained for downstream agents.
     - If the user also asks you to fold the answers into the issue body (e.g., "update
       the experiment body"), run `python scripts/sagan_state.py set-body <N> --file …` with the original
       body preserved + a `## Spec (from clarifier)` section appended. Only do this on
       explicit request -- default is comment-only.
     - Re-run the clarifier evaluation using (body + clarify questions + these answers).
       If no blocking ambiguities remain, advance to Step 2 (adversarial planning) in the
       same invocation. If still ambiguous, loop: post a `v+1` clarify marker and ask again.

  4. **If the user defers ("I'll answer later", no reply, or says to exit):** EXIT with
     label still `status:proposed`. User can answer later as issue comments and
     re-invoke `/issue <N>`, OR re-invoke and answer in chat next time. Before exiting,
     post the §5 marker: `uv run python scripts/post_step_completed.py --issue <N>
     --step 1 --exit-kind parked --notes "clarifier deferred by user"`.

**Rule:** never proceed to adversarial planning with >=2 blocking ambiguities. Tight specs
save later backtracking.

**Rule:** the ask-in-chat step is MANDATORY when there are blocking ambiguities. Posting
questions only to GitHub and immediately exiting forces a context switch the user does
not want -- always offer the inline path first.

### Step 2: Adversarial planning

Only if `status:planning`.

Invoke the `adversarial-planner` skill with the issue body + clarifier output as
the task. The skill runs planner -> Phase 1.25 hypothesis-gate -> fact-checker
-> critic -> revise internally. For `type:experiment` issues the orchestrator
forwards the issue type and label CSV via `--type experiment --labels
"<labels-csv>"` so Phase 1.25 fires; for non-experiment types Phase 1.25 is a
no-op. The same gate (`scripts/hypothesis_gate.py`) also runs in Step 1
(clarifier) on the issue body — see `clarifier.md` "Hypothesis-gate" section.

**Required sections in the final plan (enforced by this skill -- reject plans missing any):**
- Goal + hypothesis (experiments) or requirement + acceptance criteria (code changes)
- Method delta (what differs from prior related work)
- File paths + concrete diffs / config overrides
- **Reproducibility Card** (mandatory per CLAUDE.md) -- all hparams, seeds, data,
  env versions, exact `nohup` command for experiments
- Success criteria with quantitative thresholds
- Kill criteria (what result would kill the thesis)
- Compute estimate in GPU-hours
- Target pod preference
- Plan deviations allowed vs must-ask

Post plan as `<!-- epm:plan v1 -->` comment via:

```bash
PLAN_EVENT_ID=$(python scripts/sagan_state.py post-marker <N> epm:plan --note "$(cat .claude/plans/issue-<N>.html)" | grep -oE 'event [a-f0-9-]+' | awk '{print $2}')
```

`sagan_state.py post-marker` prints the event id on stdout — capture it as a
shell variable in the SAME bash block that posts the comment. **Do not
persist `PLAN_URL` to a cache file.** The variable lives only for the
duration of Steps 2a → 2c, which run in the same orchestrator turn (the
auto-continuation policy in CLAUDE.md guarantees no pause between them
in interactive mode; in autonomous mode the orchestrator exits at Step
2c so the variable is irrelevant).

Cache a copy of the plan body at `.claude/plans/issue-<N>.html` (cache
only — GitHub is the source of truth).

Also post estimated cost prominently at the top of the comment, e.g.
> **Cost gate:** estimated 12 GPU-hours on 4× H100. Reply `approve` to dispatch.

### Step 2b: Consistency checker

After the adversarial planner produces an APPROVE-rated plan, but BEFORE posting
it as `epm:plan`, spawn the `consistency-checker` agent. It receives:
- The drafted plan
- Related experiments (cited in the plan's prior work, parent issue, or near-duplicate clean-result)
- The `epm:plan` and `epm:results` markers from those related issues

The consistency checker verifies:

| Check | Violation action |
|-------|-----------------|
| Single variable change from parent | BLOCK: list all differences |
| Same baseline model/checkpoint | WARN: flag, require justification |
| Same eval suite | BLOCK: incompatible evals make comparison meaningless |
| Same seeds or superset | WARN: disjoint seeds reduce comparability |
| Same data version/hash | WARN: different data confounds results |

Post `<!-- epm:consistency v1 -->` marker. On BLOCK, send plan back to planner
for revision (loop, max 2 rounds). On WARN, append warnings to the plan comment.
On PASS, proceed normally.

Then post the plan as `<!-- epm:plan v1 -->` with the consistency results appended.

Advance label to `status:plan_pending`.

### Step 2c: Inline plan approval

**Context-dependent behavior:**

- **Autonomous mode** (invoked from `auto-experiment-runner` or with no user
  present): EXIT immediately. The issue sits at `status:plan_pending` until a
  user approves via GitHub comment or a future `/issue <N>` invocation. This
  preserves the old asynchronous review behavior. Before exiting, post the §5
  marker: `uv run python scripts/post_step_completed.py --issue <N> --step 2c
  --exit-kind parked --notes "plan posted; awaiting user approval"`.

- **Interactive mode** (user is in the current chat session): Ask the user
  inline rather than exiting. Present the plan summary and ask:

  Render the plan's **Plan Summary** section inline as the visible
  payload of the question — that's the section the planner wrote
  specifically for this gate (training, hyperparameters, baselines,
  loss surface, compute, evaluation, top risks). Don't paste the
  full plan; the user reads it on the dashboard if they want detail.

  > Plan posted as `epm:plan v1`.
  >
  > **Dashboard:** https://sagan.superkaiba.com/e/experiment/\<UUID\>
  > **Cached copy:** `.claude/plans/issue-<N>.html`
  >
  > {Render the Plan Summary section inline here — ~150 words.}
  >
  > (1) **Approve** — advance to implementation
  > (2) **Revise** \<notes\> — plan goes back to adversarial-planner
  > (3) **Defer** — exit now; re-invoke `/issue <N>` later

  The Plan Summary section is at the top of the HTML plan file
  (`<section class="plan-summary">`). Read it from
  `.claude/plans/issue-<N>.html` and pass the inner text to
  AskUserQuestion. The full plan stays in the file and on the
  dashboard for anyone who wants the details.

  Use `AskUserQuestion` or a plain text prompt and wait for the user's reply.

  - **"Approve" / "1":** Advance label to `status:approved`. Post an `approve`
    comment on the issue for audit trail. Continue to Step 4 in the **same
    invocation** — do NOT exit.
  - **"Revise \<notes\>" / "2":** Set label back to `status:planning`. Re-invoke
    adversarial-planner with the revision notes. Re-run the consistency checker.
    Post updated `epm:plan v2`. Loop back to Step 2c.
  - **"Defer" / "3":** EXIT. Label stays at `status:plan_pending`. Identical to
    the old behavior — user re-invokes `/issue <N>` later to approve. Before
    exiting, post the §5 marker: `uv run python scripts/post_step_completed.py
    --issue <N> --step 2c --exit-kind parked --notes "plan_pending; user deferred"`.

### Step 3: Approval check (backward compat, runs on re-invocation)

Runs on re-invocation if `status:plan_pending` (i.e., user deferred or approved
via GitHub comment rather than inline).

Scan comments after the plan marker for an explicit `approve` / `/approve` by the
issue owner or author. If found, advance label to `status:approved`. If comments
contain revision requests (`/revise <notes>`), set label back to `status:planning`,
re-invoke adversarial-planner with the notes; **also re-run the consistency
checker against the revised plan and post `epm:consistency v<n>` (a v2 plan
that adds new conditions or shifts baselines must not skip the consistency
gate)**; post the new `epm:plan v2` comment with the fresh consistency
verdict appended; set label back to `status:plan_pending`.

### Step 4: Worktree + dispatch implementer

Only if `status:approved`.

**4a. Worktree + draft PR.** Create `.claude/worktrees/issue-<N>` on branch
`issue-<N>` and open a draft PR.
```bash
git worktree add .claude/worktrees/issue-<N> -b issue-<N>     # reuse if it exists (resume case)
gh pr create --draft --head issue-<N> --body "Closes #<N>"
```

**4b. Dispatch implementer for the issue type.** No pod is touched yet — code
gets written, reviewed, and dry-run locally before any GPU is provisioned.
Spawn the appropriate agent via `Agent()`:

| Issue type | Implementer agent | Output marker |
|---|---|---|
| `type:experiment` | `experiment-implementer` | `epm:experiment-implementation` |
| `type:infra` / code change | `implementer` | `epm:results` |
| `type:survey` | `general-purpose` | `epm:results` |

**Env scrub for every subagent dispatch (plan §3 Phase 4.5).** EVERY
`Agent()` call this skill makes — implementer, experiment-implementer,
analyzer, code-reviewer, reviewer, interpretation-critic, clean-result-critic, experimenter,
upload-verifier, follow-up-proposer, consistency-checker, planner,
critic — passes `env=scrub_subagent_env(os.environ)` from
`explore_persona_space.orchestrate.spawn_agent`. The helper strips
`GH_TOKEN` and `GITHUB_TOKEN`; every other secret (WANDB_API_KEY,
HF_TOKEN, ANTHROPIC_API_KEY, OPENAI_API_KEY, RUNPOD_API_KEY, ...) passes
through unchanged so analyzer / experimenter still reach WandB / HF Hub /
Claude. Subagents post issue comments via the `gh_graphql` MCP server,
which inherits the token from the orchestrator's process tree (NOT from
the agent context). See CLAUDE.md "GitHub GraphQL MCP" for the
end-to-end contract; `tests/test_subagent_env_scrub.py` enforces the
allow-list.

Brief passed to the implementer:
- The plan (cached at `.claude/plans/issue-<N>.html`)
- Issue number + worktree path + branch name
- Code-review history if this is a revision round (`epm:code-review v<m>`)
- Required `report-back` fields
- **Instruction: work ONLY inside the worktree; never touch a pod; post
  progress as comments on issue #<N> via `mcp__gh_graphql__add_issue_comment`
  (NOT the `gh` CLI — `GH_TOKEN` has been scrubbed from your env).**
- **If the plan body covers ≥3 independent fixes:** make ONE commit per plan section (the planner
  produced N independent sections, one per body item). Commit message
  format: `[N/M] <plan section title>` where N is the 1-indexed item and
  M is the total. Code-reviewer reviews the whole diff; this convention
  keeps the history bisectable per item if a single fix needs to be
  reverted later.
- **TDD mode (opt-in).** Set `tdd_mode=true` in the brief if EITHER:
  (a) the approved plan body contains a literal `### TDD: yes` line, OR
  (b) the issue body / latest user comment contains `request-tdd`.
  When `tdd_mode=true`, the implementer writes tests first, posts them
  as `epm:proposed-tests v1`, and EXITs without writing implementation.
  This skill then parks at `status:implementing` and waits — see Resume
  semantics below: an `approve-tests` comment posted **after** the
  `epm:proposed-tests` marker is the resume signal, at which point this
  skill re-dispatches the implementer with `tdd_approved=true` and the
  implementer writes the code to make the approved tests pass. If a
  resumed `/issue <N>` finds the proposed-tests marker still without
  approval, it shows the marker URL + the literal `approve-tests`
  instruction and EXITs again. This is the only opt-in user gate in the
  pipeline (see CLAUDE.md auto-continuation policy gate #8).

Advance label to `status:implementing`. Before exiting, post the §5 marker:
`uv run python scripts/post_step_completed.py --issue <N> --step 4b --exit-kind
clean --notes "implementer dispatched; awaiting epm:results"`. EXIT. Implementer
runs autonomously.

### Step 5: Code review loop (Codex ensemble)

Only if `status:implementing` and the appropriate implementation marker
(`epm:experiment-implementation v<n>` for experiments, `epm:results v<n>` for
infra) is present.

This step runs an **ensemble of two reviewers in parallel** — the Claude
`code-reviewer` agent and the `codex-code-reviewer` Codex twin (gpt-5.5 via
the OpenAI Codex plugin's `companion task` runtime). On verdict disagreement
(PASS-class vs FAIL), a `reconciler` agent (Claude) issues a binding
tie-break. See `workflow.yaml § ensemble_review` for the canonical contract.

**5a. Spawn both reviewers in parallel (fresh contexts, single message).**

Both reviewers see the same brief:

- `issue_number` — the GitHub issue (`<N>`)
- `target_marker_kind` — exactly one of `experiment-implementation` (for
  `type:experiment`) or `results` (for `type:infra` / `type:survey`).
  The reviewers read the highest-version
  comment with this kind as the implementer's report.
- `revision_round` — 1-indexed integer. `1` on first review; loops up to
  `3`. The cap is **per reviewer** — reconcile invocations are free.
- `previous_critique_summaries` — one-line summaries of every prior
  `epm:code-review` AND `epm:code-review-codex` comment on this issue
  (empty on round 1). Lets each reviewer notice patterns.
- The diff vs `main`, the approved plan, the existing codebase.

The Claude reviewer additionally receives:
- `worktree` path, `base` ref (typically `main`).

The Codex twin additionally receives:
- `worktree`, `base`, `plan_marker_path`, `implementation_marker_path` — see
  `.claude/agents/codex-code-reviewer.md`.

Neither sees the implementer's reasoning — independence is load-bearing.
Dispatch in a SINGLE `Agent(...)`-call message with both spawned
`run_in_background=true` so they execute concurrently.

The Claude reviewer posts `<!-- epm:code-review v<n> -->` (PASS / CONCERNS /
FAIL). The Codex wrapper posts `<!-- epm:code-review-codex v<n> -->` (same
schema). Codex never sees `GH_TOKEN` — the wrapper agent posts via
`gh_graphql` MCP.

**5b. Read both markers from the issue.**

```bash
# After both Agent tasks complete — ONE fetch, parse twice in-memory.
# `sagan_state.py view` returns the experiment row + recent
# workflow_events in a single round-trip; we filter by marker_type
# in-memory for the two reviewers we want.
events_json=$(python scripts/sagan_state.py view <N> | jq '.events')
claude_marker=$(echo "$events_json" | jq '[.[] | select(.metadata.marker_type == "epm:code-review")] | .[0]')
codex_marker=$(echo "$events_json" | jq '[.[] | select(.metadata.marker_type == "epm:code-review-codex")] | .[0]')
```

Parse each marker's `**Verdict:**` line. Acceptable values: `PASS`,
`CONCERNS`, `FAIL`. PASS-class = {PASS, CONCERNS}; FAIL-class = {FAIL}.

**5c. Apply ensemble decision rule.**

| Claude verdict | Codex verdict | Action |
|---|---|---|
| PASS-class | PASS-class | **Agree.** `final_verdict = PASS`. CONCERNS bullets from either reviewer surface to the implementer as opportunistic suggestions; do not block. |
| FAIL | FAIL — overlapping blockers | **Agree.** `final_verdict = FAIL`. Bounce to implementer (one round). |
| FAIL | FAIL — disjoint blockers | **Union, no reconciler.** Build a combined blocker list (Claude's blockers ∪ Codex's blockers) and pass it to the implementer in the next-round brief. No new marker — both `epm:code-review v<n>` and `epm:code-review-codex v<n>` already exist on the issue. `final_verdict = FAIL`. Bounce (one round). |
| PASS-class | FAIL (or vice versa) | **Disagreement.** Spawn `reconciler` agent (Claude, fresh context). Brief: role=`code-reviewer`, issue=N, round=n, both marker bodies, diff path. Reconciler reads both verdicts + the artifact, posts `<!-- epm:code-review-reconcile v<n> -->` with binding PASS or FAIL. `final_verdict = reconciler's verdict`. |

The reconciler may NOT add findings beyond what either reviewer raised — its
job is adjudication only. Round counter does NOT increment for reconciler
invocations.

**5d. Loop on FAIL using `final_verdict`.**

- **`final_verdict == PASS`**:
  - `type:experiment` → advance label to `status:running`, proceed to Step 6.
  - `type:infra` / `type:survey` → skip
    pod phase, advance directly to `status:testing` (the inline
    test-verdict gate at Step 9c runs from there). Pre-2026-05-13 this
    used `status:reviewing` for code-change paths too; legacy issues at
    that state still route into 9c via the state-recovery table below.
- **`final_verdict == FAIL` + revision_round<3** → label back to
  `status:implementing`. Re-spawn the implementer with BOTH marker bodies
  (Claude + Codex) AND the reconcile marker (if present) as part of the
  brief. Implementer posts v<n+1>; loop back to 5a with
  `revision_round = n+1`.
- **`final_verdict == FAIL` + revision_round>=3** → `status:blocked`. Post
  abort summary, then post the §5 marker: `uv run python
  scripts/post_step_completed.py --issue <N> --step 5b --exit-kind
  failure-exit --notes "code-review-ensemble FAIL round 3+; status:blocked"`.
  EXIT. User decides: revise plan, escalate, or override.

Advance label to `status:code_reviewing` while EITHER reviewer is running,
back to `status:implementing` on ensemble FAIL, forward to `status:running`
(or `status:testing` for non-experiment types) on ensemble PASS.

**Codex twin no-show fallback.** If the Codex wrapper posts
`epm:failure v<m>` with `failure_class: codex-output-malformed` or
`failure_class: infra` (codex plugin missing), proceed with single-reviewer
(Claude-only) decision-making for that round. Do NOT block on the Codex
twin's absence; cap-3 still applies to the Claude reviewer's count. Surface
this to chat as one line: `Codex twin no-show this round; using Claude
reviewer only.`

### Step 6: Pod provisioning + experimenter dispatch (type:experiment only)

Only if `status:running` (entered from Step 5b PASS for `type:experiment`)
and no `epm:launch` marker exists.

#### Step 6a: HF gate auto-acceptance

Plans never make the human click through
gated-model gate pages. Before provisioning, scan the cached plan for HF
model IDs and submit gate-acceptance requests using the user's `HF_TOKEN`:

```bash
uv run python scripts/hf_gate_accept.py --from-plan .claude/plans/issue-<N>.html
```

The helper is idempotent (already-accessible repos exit `OK` immediately).
For "auto-approval" gates (the common case for almanach/Inria/Meta/Qwen
research releases) the access is granted on submission. For the rare
manual-approval gate the request is queued and the helper exits with code 1
and a list of URLs.

- Exit code `0` → proceed to 6b.
- Exit code `1` (manual approval still needed) → post `<!-- epm:hf-gate_pending v1 -->`
  with the URLs, leave label at `status:running`. Post the §5 marker:
  `uv run python scripts/post_step_completed.py --issue <N> --step 6c --exit-kind
  clean --notes "hf-gate manual approval pending"`. EXIT. User clicks through,
  re-runs `/issue <N>`.
- Exit code `2` (`HF_TOKEN` missing) → post `<!-- epm:hf-gate_pending v1 -->`
  with diagnostic, label `status:blocked`. Post the §5 marker:
  `uv run python scripts/post_step_completed.py --issue <N> --step 6c --exit-kind
  failure-exit --notes "HF_TOKEN missing; status:blocked"`. EXIT.

This step is also re-run on the pod inside `bootstrap_pod.sh` so a token
pushed to the pod gets the same gate state as the local VM.

#### Step 6b: Pod provisioning

Pods are ephemeral — there is no permanent fleet.
Pick the path based on whether this issue has a parent:

```bash
# 1. Check the issue body for a `Parent: #<M>` line.
# 2. If present AND `epm-issue-<M>` exists in `pod.py list-ephemeral`:
python scripts/pod.py resume --issue <M>
#    Use that pod for this child issue (don't provision a new one).
#    Record the assigned pod as `epm-issue-<M>` in the launch marker.

# 3. Otherwise, provision a fresh pod. Infer --intent from the plan:
#    training a 7B model -> ft-7b or lora-7b; eval/generation -> eval;
#    70B work -> inf-70b/ft-70b. Override with --gpu-type/--gpu-count for
#    anything else.
python scripts/pod.py provision --issue <N> --intent <inferred>
```

`provision` enforces team scoping (`X-Team-Id`), SSH bring-up (`startSsh: true`,
exposes `22/tcp`), pinned image, and runs bootstrap inline (uv, repo, .env,
HF cache, HF gate-accept, preflight). On provision failure post `<!-- epm:pod-pending -->`
with the error and stay at `status:running` (no implementer re-spawn — this is
infra, not code). User adjusts (capacity, intent override) and re-runs
`/issue <N>`.

The pod name passed downstream is `epm-issue-<N>` (or the parent's
`epm-issue-<M>` for follow-ups). The experimenter does NOT pick or create
pods.

#### Step 6c: Preflight on resumed pods

`provision` already ran preflight as its
last bootstrap step. For *resumed* pods, re-run preflight explicitly because
the volume is intact but the container restart may have left stale state:
```
ssh_execute(pod=epm-issue-<N>, command="cd /workspace/explore-persona-space && uv run python -m explore_persona_space.orchestrate.preflight --json")
```
Parse JSON. If `ok=false`, post `<!-- epm:preflight v1 -->` comment with the
errors/warnings, then post the §5 marker: `uv run python
scripts/post_step_completed.py --issue <N> --step 6c --exit-kind failure-exit
--notes "preflight failed; user must fix"`. EXIT. User fixes, re-runs.

#### Step 6d: Dispatch experimenter

Spawn `experimenter` subagent via `Agent()`.
The experimenter's scope is **pod ops + monitoring + debugging only** — it
does NOT write substantial code (hot-fixes ≤10 lines, no logic changes; see
the experimenter agent definition).

Brief passed to experimenter:
- The plan + the code-reviewed branch (`issue-<N>`)
- Pod name (`epm-issue-<N>` or parent's)
- The exact `nohup` launch command from the plan's Reproducibility Card
- Progressive monitoring schedule (per the experimenter agent definition)
- Required `report-back` fields (artifacts, WandB URL, HF Hub path, deviations,
  hot-fix log)

**NEVER include pod lifecycle commands (provision, stop, resume, terminate,
cleanup) in the experimenter brief.** The experimenter agent spec explicitly
forbids pod lifecycle management (line ~305). Pod termination happens
automatically in Step 8 (after upload-verification PASS). Including
`pod.py terminate` or `pod.py stop` in the experimenter's instructions
bypasses the upload-verification gate and risks premature destruction of
artifacts that haven't been confirmed at permanent URLs.

Post `<!-- epm:launch v1 -->` containing:
- Worktree path, branch, PR URL, code-review verdict (`PASS`)
- Pod + PID + log path
- WandB run URL (best-effort; experimenter updates if not known yet)
- Experimenter subagent ID (for monitoring)

**Spawn the §2 stall-detection watchdog (detached, on the local VM, NOT
the pod).** After the experimenter is launched and `epm:launch` is posted,
spawn `python scripts/pod.py watch --issue <N> --wandb-run-url <URL>
--log-path <server>:<path>` as a detached background process. Pid file
written to `.claude/cache/watch-<N>.pid` (the watchdog cleans it up on
exit). The watchdog probes WandB heartbeat + log-mtime every 60s; on >5min
silence it posts `epm:failure` with `failure_class: infra` and `reason:
stall`, flips the label to `status:blocked`, and exits.

**`SECTION_2_LAND_SHA` gate (in-flight protection).** Before spawning,
check whether the latest `epm:launch` (or `epm:experiment-dispatch`)
marker pre-dates `SECTION_2_LAND_SHA = "<filled-in-at-merge>"` (set in
`workflow.yaml` after the §2 PR merges). If yes, SKIP watchdog spawn and
log the reason. Pre-§2 dispatches don't have the heartbeat-probe wiring;
killing them on the §2 deploy would cause spurious failures. Users can
manually attach the watchdog to a long-running pre-§2 pod via `python
scripts/pod.py watch --issue <N> --force-attach` (the `/issue` Step 6d
auto-spawn never sets this flag).

Label stays at `status:running`. Before exiting, post the §5 marker:
`uv run python scripts/post_step_completed.py --issue <N> --step 6d --exit-kind
clean --notes "experimenter dispatched; watchdog spawned"`. EXIT. Experimenter
runs autonomously. The experimenter posts `epm:progress`, `epm:hot-fix` (if
needed), and finally `epm:results`. The watchdog stops itself when
`epm:results` is observed, the status label moves out of `running`, or its pid
file is deleted.

# Fire title update on status-transition into running.
# mcp__happy__change_title({"title": render_title(issue, status_human="running", next_action="experiment monitor")})

### Step 7: Monitor -> results

Experimenter is expected to post `<!-- epm:progress v1 -->` comments at major
milestones, optional `<!-- epm:hot-fix v<n> -->` markers for in-line fixes
(<=10 lines, no logic change — see the experimenter agent definition), and a
final `<!-- epm:results v1 -->` comment containing:
- Final eval numbers (inline JSON snippet + path in repo)
- Reproducibility card (filled)
- WandB URL + HF Hub model/adapter URL
- Worktree path + final commit hash
- GPU-hours actually used vs budgeted
- Plan deviations + rationale
- Hot-fix log (commits + diffs applied during the run)

When this skill is re-invoked in `status:running`:
1. Check `epm:results` exists. If not, show last progress, post the §5 marker:
   `uv run python scripts/post_step_completed.py --issue <N> --step 7 --exit-kind
   parked --notes "experimenter still running; epm:results not yet posted"`,
   and EXIT. **If the most recent `epm:progress` comment is older than 4 hours
   and there is no `epm:results` or `epm:failure`, post `<!-- epm:stale v1 -->`
   asking the user to investigate (the experimenter may have crashed silently);
   leave the label at `status:running`.**
2. If `epm:failure` posted: route via the **failure classifier**. The
   `epm:failure` body SHOULD include a `failure_class: infra | code` field
   on its first non-blank line. Routing:

   | failure_class | Cause example | Action |
   |---|---|---|
   | `infra` | OOM, ENOSPC, NCCL, vLLM init failure, SSH refused, 401/gated repo, library traceback (vllm/transformers/peft/trl/torch/xformers) | Re-spawn the **experimenter** on the SAME branch, post `epm:experimenter-respawn v<n+1>`. NO implementer round. Cap 3 respawns; on 4th, `status:blocked`. |
   | `code` | Python `Traceback` from `src/explore_persona_space/` or `scripts/` (our code), `AssertionError`/`TypeError`/`KeyError` from our code | Label back to `status:implementing`, re-spawn `experiment-implementer` with the failure context. Loop through Steps 4b → 5 → 6 again. Cap 3 (existing). |

   **Missing `failure_class` — invoke the classifier script.** Do NOT
   reason about regex patterns inline; the patterns are owned by
   `scripts/failure_classifier.py` and reading them yourself drifts.
   Instead, shell out:

   ```bash
   # Pipe the failure body via stdin to avoid shell-quoting traps.
   # Reads epm:failure markers from Sagan's workflow_events table.
   cat <(uv run python scripts/sagan_state.py list-markers "$N" --prefix epm:failure --json \
       | jq -r '.[] | .note // empty') \
     | uv run python scripts/failure_classifier.py --body - \
         --log "$LATEST_LOG_PATH"
   ```

   The script writes a single line — `infra` or `code` — to stdout.
   Treat that as the verdict and apply the corresponding row of the
   table above. If the script exits non-zero, treat as `code`
   (conservative) and post `epm:failure-classify-error` with the
   stderr captured.

   The Python module
   [`scripts/failure_classifier.py`](../../../scripts/failure_classifier.py)
   is the SINGLE source of truth for the regex pattern list.
   `.claude/skills/issue/failure_patterns.md` is a human-readable
   mirror of the same patterns (kept in sync; consult it for review or
   when extending — but do NOT consult it at runtime). To add a new
   pattern, edit `failure_classifier.py` AND the markdown mirror; the
   tests in `tests/test_failure_classifier.py` cover the behaviour.
3. If `epm:results` exists, advance label to `status:uploading` and proceed
   to Step 8.

### Step 8: Upload verification

Only if `status:uploading` and no `epm:upload-verification` marker with verdict=PASS.

**Hard gate:** No experiment advances to interpretation until all artifacts have
permanent URLs. This prevents data loss from pod restarts or cleanup.

Spawn the `upload-verifier` agent with:
- Issue number
- Experiment type (from `type:*` label)
- Artifact hints from the `epm:results` marker (WandB URL, HF paths, pod name)
- The `epm:plan` marker (for experiment type metadata)

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

Post `<!-- epm:upload-verification v1 -->` marker with per-artifact PASS/FAIL + URLs.

- **PASS** -> terminate the pod, then advance to `status:interpreting` and proceed to Step 9.
  Once artifacts are confirmed at permanent URLs, the pod is no longer needed —
  interpretation runs locally:
  ```bash
  python scripts/pod.py terminate --issue <N> --yes
  ```
  This destroys the pod (volume + container disk gone). Post
  `<!-- epm:pod-terminated v1 -->` with the command output. If interpretation
  later needs GPU compute (e.g., to regenerate a figure from raw outputs that
  weren't downloaded), provision a fresh pod via `pod.py provision`. If the
  issue body has `Parent: #<M>`, terminate the parent's pod (`epm-issue-<M>`)
  instead. Skip the terminate call only if the user has labelled the issue
  `keep-running` for known follow-up work in the same session.
- **FAIL** -> dispatch the `uploader` agent (up to 3 rounds) to close the gaps.
  The uploader receives the verifier's missing-artifacts list, lifecycle-aware
  resumes the pod if needed, pushes to HF / WandB / git, and posts an
  `epm:upload-fix v1` marker. After each uploader round, re-run `upload-verifier`;
  it posts a fresh `epm:upload-verification v<N+1>`.

  Round outcomes:
  - **uploader COMPLETE + verifier PASS** -> proceed as PASS branch above.
  - **uploader BLOCKED** (e.g., RunPod host capacity, missing credentials)
    -> stays at `status:uploading`. Post the uploader's `epm:upload-fix`
    marker with the blocker. Post the §5 marker: `uv run python
    scripts/post_step_completed.py --issue <N> --step 7 --exit-kind failure-exit
    --notes "uploader BLOCKED; awaiting operator action"`. EXIT, await operator
    action.
  - **3rd round still FAIL** -> label `status:blocked`. Post the §5 marker:
    `uv run python scripts/post_step_completed.py --issue <N> --step 7
    --exit-kind failure-exit --notes "uploader exhausted 3 rounds; see
    upload-fix v3"`. EXIT (mirror the code-reviewer FAIL escalation in
    CLAUDE.md).

  See `.claude/agents/uploader.md` for the uploader's contract and the
  marker schema. The uploader NEVER terminates pods; only stops/resumes.

### Step 9: Iterative interpretation + clean-result critique

This step has two sub-phases, both running while `status:interpreting`:
**9a** (analyzer ↔ interpretation-critic content honesty loop) and **9a-bis**
(analyzer ↔ clean-result-critic structure + register + statistical-framing
loop). On 9a-bis PASS the source issue advances directly to
`status:awaiting_promotion`.

**A note on the retired Step 9b** (the dedicated final-reviewer step): as
of 2026-05-13 it was retired and its responsibilities — statistical-framing
rule enforcement and final fresh-context check on the published
clean-result body — were absorbed by `clean-result-critic` (a new
**Lens 11** in that agent). The empirical cost/value review found Step 9b
mostly duplicated `interpretation-critic` (claim verification, alternatives,
overclaims, robustness) and `clean-result-critic` (template compliance,
verifier mechanical pass). The two unique pieces moved cleanly into
`clean-result-critic`'s Lens 11.

**9a. Iterative interpretation** (only if `status:interpreting`)

Only for `type:experiment` issues. Code-change issues never reach this step
because Step 5 already PASSed code-review and routed them to Step 9c (the
inline test-verdict gate) directly.

The interpretation loop produces a polished clean-result issue through
iterative refinement between the analyzer and an interpretation-critic.

**Round 1:**

1. Spawn `analyzer` agent (fresh context) with raw result paths. The analyzer:
   - Writes the **Fact Sheet** (reproducibility card, artifact URLs, raw numbers,
     plots, sample outputs) — this is written once and not revised.
   - Writes the **Interpretation** (background, methodology, results claim + hero
     figure + main takeaways + confidence, next steps).
   - Generates plots via `paper-plots` skill.
   - Posts `<!-- epm:interpretation v1 -->` marker on the source issue.

2. **ROUND 1 ONLY:** spawn the **interpretation-critic ensemble** (fresh
   contexts, single message, both `run_in_background=true`):
   - `interpretation-critic` (Claude) — full 7-lens review. Posts
     `<!-- epm:interp-critique v1 -->` with PASS or REVISE.
   - `codex-interpretation-critic` (Codex gpt-5.5 via `companion task`) —
     same 7 lenses (lens 6 plot-prose works on Codex multimodal). Posts
     `<!-- epm:interp-critique-codex v1 -->`.

   Neither sees the analyzer's reasoning. Independence is load-bearing.
   Rounds 2-3 spawn the Claude critic only (see below).

3. **Apply ensemble decision rule** (see `workflow.yaml § ensemble_review`):

   | Claude | Codex | Action |
   |---|---|---|
   | PASS | PASS | `final_verdict = PASS`. Concatenate suggestions for analyzer's optional polish. |
   | REVISE | REVISE | `final_verdict = REVISE`. Union the revision requests (dedup exact-same). |
   | PASS vs REVISE (or vice versa) | (the other) | Spawn `reconciler` (marker mode). Brief: role=`interpretation-critic`, both marker bodies, interpretation body path, eval JSON paths, figure paths. Reconciler posts `<!-- epm:interp-critique-reconcile v1 -->` with binding PASS or REVISE. `final_verdict = reconciler's verdict`. |
   | Codex no-show (`epm:failure`) | (any) | Fallback: `final_verdict = Claude verdict`. Surface "Codex twin no-show round 1" to chat. |

   Reconcile rounds do NOT increment the per-reviewer round counter.

**If `final_verdict == REVISE` (rounds 2-3, CLAUDE ONLY):**

Re-spawn analyzer (fresh context, sees original data + ALL critique
feedback: Claude marker + Codex round-1 marker + reconcile marker if any).
Analyzer posts `<!-- epm:interpretation v2 -->`. **Re-spawn ONLY the
Claude `interpretation-critic`** (no Codex twin on rounds 2-3 — fresh-eyes
value decays once the analyzer is iterating against round-1 feedback;
register-noise becomes dominant). Critic sees v2 + ALL prior markers
including round-1 Codex marker. Posts `<!-- epm:interp-critique v2 -->`.
The ensemble decision rule reduces to: `final_verdict = Claude verdict`
on rounds 2-3.

**Max 3 rounds per reviewer.** After round 3, advance regardless with full
critique history. The round-1-only Codex policy was adopted 2026-05-13.

**On PASS (or max rounds reached):**

The analyzer promotes the source experiment row in place (no separate
row is created — see `.claude/agents/analyzer.md` Step 6 for the exact
command sequence):
- Snapshot prior body to `epm:original-body` workflow event (audit trail).
- `sagan_state.py set-body <N> --file <clean-result-body-path>`
- `sagan_state.py set-title <N> "<claim summary> (HIGH|MODERATE|LOW confidence)"`
- `sagan_state.py set-clean-result <N>` — Sagan auto-creates the pending `runs` row on the same PATCH.
- Runs `scripts/verify_clean_result.py` first — FAIL blocks posting.

Posts `<!-- epm:analysis v1 -->` marker on the source issue with the
hero figure URL + 2-sentence recap. The source issue body IS the
clean-result; there is no separate issue link.

# Fire title update on clean-result promotion.
# mcp__happy__change_title({"title": render_title(issue, status_human="reviewing", claim=<claim-summary>)})

Then proceed to **9a-bis (clean-result-critique loop)** before advancing the label.

**9a-bis. Clean-result-critique loop** (only if `status:interpreting`, after Step 9a PASS)

Same shape as the interpretation-critic loop, but the critic checks
STRUCTURE + REGISTER + STATISTICAL-FRAMING not CONTENT. Content honesty
was settled in 9a; this layer ensures the body matches the v4 clean-result
shape (per `.claude/skills/clean-results/SPEC.md`), reads in the right
registers — casual user-voice in `## TL;DR`, LessWrong research-post
register in `## Summary` and `## Details` — and enforces the project's
p-values-only reporting convention (Lens 11, absorbed 2026-05-13 from the
retired reviewer step). Discipline rules: see
`.claude/skills/clean-results/SPEC.md` (canonical structure, registers,
exemplars, figure captions, and research-communication principles).

**On PASS at this layer, the source issue advances directly to
`status:awaiting_promotion`** — there is no separate downstream reviewer
step. Clean-result-critic is the final adversarial gate.

**Round 1 (ENSEMBLE — Claude + Codex, parallel):**

Spawn the **clean-result-critic ensemble** (fresh contexts, single
message, both `run_in_background=true`):

- `clean-result-critic` (Claude) — full 11-lens review (10 structural +
  Lens 11 statistical-framing rule). Reads the published clean-result body
  + the latest `epm:interpretation vN`, runs
  `scripts/verify_clean_result.py` + `scripts/audit_clean_results_body_discipline.py`
  as authoritative mechanical passes. Posts
  `<!-- epm:clean-result-critique v1 -->` on the source issue with PASS or
  REVISE.
- `codex-clean-result-critic` (Codex gpt-5.5 via `companion task`) — same
  11 lenses. Independently runs `verify_clean_result.py`. Posts
  `<!-- epm:clean-result-critique-codex v1 -->`.

Neither sees the analyzer's reasoning or interp-critique history.

**Apply ensemble decision rule** (mirrors interpretation-critic 9a):

| Claude | Codex | Action |
|---|---|---|
| PASS | PASS | `final_verdict = PASS`. CONCERNS / minor-only findings surface as opportunistic. |
| REVISE | REVISE | `final_verdict = REVISE`. Union the revision requests (dedup exact-same). |
| PASS vs REVISE (or vice versa) | (the other) | Spawn `reconciler` (marker mode). Brief: role=`clean-result-critic`, both marker bodies, clean-result body path, verifier output, audit-script output. Reconciler posts `<!-- epm:clean-result-critique-reconcile v1 -->` with binding PASS or REVISE. `final_verdict = reconciler's verdict`. |
| Codex no-show (`epm:failure`) | (any) | Fallback: `final_verdict = Claude verdict`. Surface "Codex twin no-show round 1" to chat. |

Reconcile rounds do NOT increment the per-reviewer round counter. The
round-1-only Codex policy was adopted 2026-05-13 (reverses the earlier
"Codex imposes register noise" exclusion — round-1-only confines Codex to
the first-look pass where structural-flaw catch dominates register noise).

**If `final_verdict == REVISE` (rounds 2-3, CLAUDE ONLY):**

Re-spawn `analyzer` agent (fresh context, sees raw data + all
interp-critique history + the round-1 Codex marker + the latest
clean-result-critique). Analyzer revises the `epm:interpretation` marker
AND edits the clean-result experiment body in place via
`python scripts/sagan_state.py set-body <clean-result-N> --file ...`.
Re-runs `scripts/verify_clean_result.py` (must still PASS). **Re-spawn
ONLY the Claude `clean-result-critic`** (no Codex twin on rounds 2-3).
Posts the next critique version (`<!-- epm:clean-result-critique v2 -->`).
`final_verdict = Claude verdict` on rounds 2-3.

**Max 3 rounds per reviewer.** After round 3, advance regardless and fold
the residual structural / register / statistical-framing debt into the
chat-side summary so the user can decide whether to patch before
promoting.

**On PASS (or max rounds reached) — final transition:**

Clean-result `runs.classification` STAYS at `pending` (do NOT auto-promote).
Advance source experiment to `awaiting_promotion`:
```
python scripts/sagan_state.py set-status <N> awaiting_promotion
```
Post a workflow event with a user-facing summary:
> Clean-result critic PASS (final gate as of 2026-05-13). Clean-result #\<clean-result-N\> is ready for your review.
> When satisfied, promote it (USER-ONLY — no automation may do this):
>   `python scripts/sagan_state.py promote <clean-result-N> useful` (paper-relevant)
>   `python scripts/sagan_state.py promote <clean-result-N> not-useful` (archive candidate)
> Then re-enter `/issue <N>` to fire Step 10.

Post the §5 marker: `uv run python scripts/post_step_completed.py --issue
<N> --step 9 --exit-kind parked --notes "awaiting clean-result promotion"`.
EXIT. The user reviews the clean-result at their own pace and manually
picks a verdict. **Awaiting promotion is a user-only status — no agent
or automation may flip `runs.classification` out of `pending`.** The
`sagan_state.py promote` command flips `runs.classification` to
`useful` / `not_useful`, sets the experiment to `completed`, and
prints a reminder to re-enter `/issue <N>` so Step 10 fires.

**On re-invocation at `status:awaiting_promotion`:**
1. Check if the source row has been promoted: `runs.classification` is
   `useful` or `not_useful` (status will be `completed`).
2. If promoted → advance to Step 10 (auto-complete).
3. If still `pending` → show the source row's dashboard link, post the
   §5 marker: `uv run python scripts/post_step_completed.py --issue <N>
   --step 10 --exit-kind parked --notes "clean-result still pending;
   awaiting user promotion"`, and EXIT. User hasn't promoted yet.

**9c. Test-verdict gate (code-change paths only, inline)**

Only for `type:infra` / `type:survey`
issues — these arrive here directly from Step 5 PASS, having skipped
Steps 6–8 (no pod, no interpretation). The code-review gate has already
approved the diff; this step verifies the test suite still passes.

There is **no `tester` agent**. The skill itself runs the project's test
suite directly and posts an `epm:test-verdict` marker with the result.

1. Unit tests: `uv run pytest tests/ -v --tb=short`
2. Lint: `uv run ruff check . && uv run ruff format --check .`
3. Integration tests (conditional, if diff touches train/eval/orchestrate)
4. Coverage gap report (flags, does not auto-generate)

Post `<!-- epm:test-verdict v1 -->`. PASS → Step 10. FAIL (count < 3) → stay in
`status:testing`, re-spawn implementer. FAIL (count >= 3) → `status:blocked`.

### Step 10: Auto-complete (fires after user promotes clean-result from `awaiting_promotion`, or `epm:test-verdict` PASS for code-change paths)

No user gate. The skill transitions the issue to a terminal-or-`followups_running` state automatically. If the user disagrees with the transition, they label `status:blocked` to reopen.

#### Step 10 step 0: Completion audit (NEW — gates entry to step 1)

Cheap insurance against drift on multi-part issues: re-read the ORIGINAL
issue body and verify every numbered ask is actually addressed. The reviewer
checks the *write-up*; this checks the *issue → work* contract.

1. Re-fetch the current experiment body: `python scripts/sagan_state.py view <N> | jq -r '.experiment.body'`.
2. Enumerate every:
   - Numbered ask (`1. …`, `2. …`)
   - Acceptance criterion (sentences containing "must", "should report",
     "deliverable", "include")
   - Explicit deliverable (e.g., "produce a clean-result with X figure")
   If the body has no numbered asks (free-form description), audit against
   the headline goal sentence only and note "no numbered asks found" in
   the marker.
3. For each ask, locate evidence it was addressed:
   - `type:experiment` → grep the promoted clean-result body + `epm:results`
     marker.
   - `type:infra` / `type:survey` → grep
     the PR diff (`gh pr diff <PR>`) + `epm:test-verdict`.
4. Post `<!-- epm:completion-audit v1 -->` with a checklist:
   ```markdown
   <!-- epm:completion-audit v1 -->
   ## Completion Audit — PASS | INCOMPLETE

   Audited against issue body as of <commit-sha-or-timestamp>.

   - [x] **Ask 1:** "<verbatim ask>" — addressed in <clean-result §Headline numbers | PR file foo.py:42>
   - [x] **Ask 2:** … — addressed in …
   - [ ] **Ask 3:** "<verbatim ask>" — NOT FOUND in clean-result or `epm:results`. Proposal: <what's missing>.

   <!-- /epm:completion-audit -->
   ```
5. Branch on verdict:
   - **All ☑ (PASS):** proceed to step 1 below.
   - **Any ☐ (INCOMPLETE):** label `status:blocked` (remove
     `status:awaiting_promotion` / `status:testing` as applicable; also
     legacy `status:reviewing` if the issue predates 2026-05-13), do NOT
     advance. The audit comment is the bounce-back
     payload. User either (a) modifies the issue body to reconcile
     resolved scope-creep, (b) re-runs the missing work via a follow-up
     `/issue` cycle, or (c) labels `status:awaiting_promotion` again to
     override. Per CLAUDE.md STATE-TO-`status:blocked` criterion 5.

#### Step 10 step 1+: existing flow

1. If code change: mark PR ready for review (not merge -- user merges).
2. Update `RESULTS.md` if the finding is headline-level (propose diff as comment
   `<!-- epm:results-md-diff v1 -->` -- do NOT auto-edit).
3. Update `eval_results/INDEX.md` with a new entry.

4. **Detect open follow-up children.** Look up incoming `parent` edges
   in Sagan (children whose `parent` edge points to this experiment):
   ```bash
   python scripts/sagan_state.py view <N> \
     | jq -r '.experiment.id' \
     | xargs -I{} curl -sH "Authorization: Bearer $SAGAN_API_TOKEN" \
         "$SAGAN_BASE_URL/api/edges?to_id={}&type=parent"
   ```
   (a dedicated `list-children <N>` subcommand is on the follow-up list).
   A child is "still in flight" if its status is not in `{completed,
   archived}`. The parent's destination state depends on whether ANY
   child is still in flight.

5. **Choose the destination state.**

   - **At least one child still in flight** AND `kind=experiment`
     → keep at `completed` with `has_clean_result=true`; the active
     `parent` edges in Sagan surface the running followups in the
     dashboard. Re-invoking `/issue <N>` later re-runs Step 10 step 4 —
     once all children reach a terminal state, post a final
     `epm:followups-done` event.
   - **No children in flight** AND `kind=experiment`
     → **`status=completed`**, post `epm:done v1`.
   - **`kind` in {`infra`, `survey`}** (regardless
     of children) → **`status=completed`**, post `epm:done_impl v1`.
     Code-change paths don't seed experimental follow-ups via Step 10b.
   - **No `kind` set** → STOP, post an error event asking the user to
     set `kind` in the dashboard. Do NOT pick a default.

6. Apply the chosen status:
   ```
   python scripts/sagan_state.py set-status <N> completed
   ```

7. Verify the dashboard kanban routes the experiment to the right
   column (Done) — the status enum value already determines the column;
   no manual move needed.

8. Post final marker `epm:done` (or `epm:done_impl` for code-change
   kinds) summarizing: outcome, key numbers, what's confirmed/falsified,
   what's next, plus a link to the promoted clean-result experiment (for
   experiments) AND a list of in-flight child follow-ups when relevant.
   Include the line `status now: completed`.
9. **NEVER set `status='archived'` from this skill.** Archive is for
   duplicates / invalid / won't-fix, user-initiated only.
10. Do NOT delete the worktree -- user decides when to clean up.
11. If `type:experiment` AND we just landed at `status:done_experiment` (no
    children blocked us), proceed to Step 10b (follow-up proposer). If we
    landed at `status:followups_running`, SKIP Step 10b — the proposer was
    already run in a prior `/issue <N>` invocation that produced the children
    we're now waiting on.

### Step 10b: Follow-up proposer (experiments only)

Auto-fires after `done_experiment` for `type:experiment` issues. Spawn the
`follow-up-proposer` agent with:
- The completed experiment's plan (`epm:plan`)
- The results (`epm:results`)
- The clean-result body (the source experiment's current body — promoted in place at Step 9a)
- The interpretation critique history (`epm:interp-critique v1..vN`)
- The reviewer verdict

The proposer outputs 1-3 concrete follow-up proposals, each with:
- Pre-filled spec from parent (reproducibility card copied, only diff highlighted)
- Stated hypothesis + falsification criteria
- Type (ablation, reproduction, diagnostic, scaling, etc.)
- Cost estimate in GPU-hours
- Ranked by information gain per GPU-hour

Post as `<!-- epm:follow-ups v1 -->` marker on the completed issue.

The user can create follow-up issues from these proposals by:
- Replying on the issue with `create 1` (or `create 1,2`)
- Telling the main conversation agent to create them
- Manually copying the spec into a new issue

Each created follow-up issue links to the parent via `Parent: #<N>` in the body.

# Fire title update after follow-ups marker is posted.
# mcp__happy__change_title({"title": render_title(issue, status_human="done_experiment", followups=[...])})

### Step 10d: Worktree merge prompt (NEW — both experiment and impl)

After Step 10b posts (the pod was already terminated in Step 8 immediately
after upload-verification PASS), ask the user
once via `AskUserQuestion`:

> **Merge worktree `issue-<N>` into `main` now?**
> YES → mark draft PR ready, **rebase-merge** so each commit lands
> individually on `main`, then `git worktree remove`.
> NO → no-op; user merges later.

**30-minute cooldown gate.** Before prompting, run:

```bash
CREATED=$(gh pr view <PR> --json createdAt -q .createdAt)
AGE_SEC=$(( $(date +%s) - $(date -d "$CREATED" +%s) ))
if [ "$AGE_SEC" -lt 1800 ]; then
  echo "PR younger than 30 min; deferring merge prompt to next /issue invocation"
  exit 0
fi
```

The cooldown reduces the chance of merging before the PR has had time for
a quick human glance. Override allowed by manual `/issue <N>` re-invocation
after the cooldown elapses.

- **YES:**
  ```bash
  gh pr ready <PR>
  gh pr merge <PR> --rebase --delete-branch=false
  git worktree remove .claude/worktrees/issue-<N>
  ```
  The `gh pr merge --rebase` form lands all per-item commits individually on `main`;
  each is independently revertible via `git revert <sha>`. (Vs. `--merge`
  which creates one merge commit — reverts everything together.)
  Post `<!-- epm:merged v1 -->` with the list of merge SHAs. Update chat
  title with `merged`.

  ```python
  # Fire title update on merge.
  # mcp__happy__change_title({"title": render_title(issue, status_human="merged")})
  ```

- **NO:** post `<!-- epm:merge-deferred v1 -->`.
- **Autonomous mode:** default NO; record marker. Never auto-merge without
  user approval.

Idempotent: skip if either marker (`epm:merged` or `epm:merge-deferred`)
already exists.

---

## Resume semantics

`/issue <N>` and `/issue <N> --resume` are identical. The skill is always
idempotent: it reads state from labels + markers, computes next action, and
executes. There is no "start from scratch" -- the only way to reset is to remove
labels and delete marker comments manually.

### Step-completed re-entry skip-ahead (`epm:step-completed`)

Every step that completes posts `<!-- epm:step-completed v1 -->` BEFORE EXIT,
recording `step`, `next_expected_step` (looked up from `workflow.yaml § steps`),
and an `exit_kind` (one of `clean` / `parked` / `failure-exit`). Symphony §7.3
distinction: `clean` = normal continuation, `parked` = user-gated wait,
`failure-exit` = error path.

**Helper.** Skill code calls `scripts/post_step_completed.py` at every
EXIT site (after the EXIT condition is met, before the actual exit):

```bash
uv run python scripts/post_step_completed.py \
    --issue <N> --step 5b --exit-kind clean \
    --notes "code-review PASS, advancing to pod provisioning"
```

The helper looks up `next_expected_step` from `.claude/workflow.yaml` and
posts the marker; refuses to post if the step ID is unknown to the YAML
or if `exit_kind` is not in the choices list (typo guard).

**Re-entry router.** `src/explore_persona_space/orchestrate/resume.py:decide_entry_step`
implements the precedence rules:

1. `status:blocked` is the current label → full replay (rule 1, BEFORE
   the marker is consulted; load-bearing — a stale clean-exit marker
   must NEVER let the skill dispatch on a manually-blocked issue).
2. No `epm:step-completed` marker → full replay (first invocation or
   pre-§5 in-flight issue).
3. Marker's `exit_kind` is `parked` or `failure-exit` → full replay.
4. Marker's `next_expected_step` is unknown to `workflow.yaml § steps` →
   warn + full replay (graceful fallback for renamed/removed steps).
5. Current `status:*` label not in target step's `entry_status_label` →
   full replay (status drift; user manually flipped the label).
6. All checks pass → jump to `next_expected_step`, skipping Steps
   0 through (target − 1).

**EXIT-site → `exit_kind` mapping** (per plan §5 lines ~1171-1192;
17 sites total). The implementer wires each site to invoke
`post_step_completed.py` with the right `exit_kind`:

| EXIT site | Step | Trigger | `exit_kind` |
|---|---|---|---|
| Step 0b/2 `type:*` autofill loop guess | 0b | user override required | `failure-exit` |
| Step 1 user defers / no reply | 1 | user-gated | `parked` |
| Step 2c plan_pending awaiting `approve` | 2c | user-gated | `parked` |
| Step 2c "Defer"/"3" reply | 2c | user-gated | `parked` |
| Step 4b TDD gate awaiting `approve-tests` | 4b | user-gated | `parked` |
| Step 4b TDD second pass | 4b | user-gated | `parked` |
| Step 4b implementer EXIT to `status:implementing` | 4b | normal continuation | `clean` |
| Step 5b code-review FAIL revision_round>=3 | 5b | error path | `failure-exit` |
| Step 6c pod URLs surfaced, leave at `status:running` | 6c | normal continuation | `clean` |
| Step 6c pod provisioning failure | 6c | error path | `failure-exit` |
| Step 6 preflight error/warning | 6 | error path | `failure-exit` |
| Step 6d experimenter dispatched, autonomous | 6d | normal continuation | `clean` |
| Step 7 `epm:results` not found and stale | 7 | user-gated | `parked` |
| Step 7 upload-verifier FAIL | 7 | error path | `failure-exit` |
| Step 9 `awaiting_promotion` user reviews | 9 | user-gated | `parked` |
| Step 10 still `clean-results:draft` | 10 | user-gated | `parked` |
| Step 0 resume >1 `status:*` ambiguous | 0 | error path | `failure-exit` |

**Backwards-compat.** An issue that ran through Steps 0-5 BEFORE §5 landed has
no `epm:step-completed` markers. On re-entry the router returns None
(rule 2) and the skill falls back to the existing full-replay path
documented below. The first `/issue <N>` re-invocation AFTER §5 lands
posts the first marker; the SECOND benefits from skip-ahead. Graceful,
no migration step.

If the specialist subagent has exited but no `epm:results` marker was posted, the
skill assumes the run failed silently. On resume in `status:running` with no
progress in >4 hours, post `<!-- epm:stale v1 -->` comment asking user to
investigate and optionally label `status:blocked`.

**Resume correctness per active state** (the key benefit of having dedicated
"working" labels):

| Label at resume | `epm:*` markers present | Interpretation | Action |
|-----------------|-------------------------|----------------|--------|
| `planning` | no `epm:plan` | planner was cancelled | re-run adversarial-planner |
| `plan_pending` | `epm:plan` exists | awaiting user approval | show plan URL, EXIT |
| `implementing` | no `epm:experiment-implementation` (or `epm:results` for infra), no `epm:proposed-tests` either | implementer was cancelled | re-spawn implementer |
| `implementing` | `epm:proposed-tests v<n>` exists, no `epm:experiment-implementation`, no `approve-tests` comment posted **after** the `proposed-tests` marker | TDD mode: tests posted, awaiting user approval | show the `proposed-tests` marker URL + the `approve-tests` reply instruction, EXIT |
| `implementing` | `epm:proposed-tests v<n>` exists, an `approve-tests` comment exists **after** the `proposed-tests` marker, no `epm:experiment-implementation` | TDD tests approved by user | re-spawn implementer with `tdd_approved=true`; brief instructs implementer to write implementation against the approved tests, then post `epm:experiment-implementation v1` as normal |
| `implementing` | latest `epm:code-review` is FAIL, round < 3 | revision in progress | re-spawn implementer with critique |
| `implementing` | latest `epm:code-review` is FAIL, round >= 3 | exhausted retries | label `status:blocked`, ask user |
| `code_reviewing` | neither `epm:code-review` nor `epm:code-review-codex` for the current implementation version | both ensemble reviewers were cancelled | re-spawn both code-reviewer + codex-code-reviewer in parallel |
| `code_reviewing` | `epm:code-review v<n>` exists, no `epm:code-review-codex v<n>` | Codex twin not yet returned (or wrapper crashed) | re-spawn `codex-code-reviewer` only |
| `code_reviewing` | `epm:code-review-codex v<n>` exists, no `epm:code-review v<n>` | Claude reviewer not yet returned | re-spawn `code-reviewer` only |
| `code_reviewing` | both `epm:code-review v<n>` and `epm:code-review-codex v<n>` exist, verdicts disagree (PASS-class vs FAIL), no `epm:code-review-reconcile v<n>` | reconciler not yet started | spawn reconciler |
| `code_reviewing` | both `epm:code-review v<n>` and `epm:code-review-codex v<n>` exist, verdicts agree | ensemble decision ready | apply Step 5c rule and advance |
| `code_reviewing` | `epm:code-review-codex` is `epm:failure` (codex-output-malformed or infra) | Codex twin no-show | proceed with Claude-only decision per Step 5d fallback |
| `running` | no `epm:results` for > 4h | experimenter crashed silently | post `epm:stale`, ask user |
| `running` | latest marker is `epm:failure` with bounce-back proposal | experimenter bounced to implementer | label back to `status:implementing`, re-spawn experiment-implementer |
| `uploading` | no `epm:upload-verification` PASS | verifier not run or failed | re-run upload-verifier |
| `interpreting` | no `epm:interpretation` | analyzer not started | spawn analyzer |
| `interpreting` | `epm:interpretation` exists, neither `epm:interp-critique` nor `epm:interp-critique-codex` for the current version | both ensemble critics not started | spawn `interpretation-critic` + `codex-interpretation-critic` in parallel |
| `interpreting` | `epm:interp-critique v<n>` exists, no `epm:interp-critique-codex v<n>` | Codex twin not yet returned | re-spawn `codex-interpretation-critic` only |
| `interpreting` | `epm:interp-critique-codex v<n>` exists, no `epm:interp-critique v<n>` | Claude critic not yet returned | re-spawn `interpretation-critic` only |
| `interpreting` | both `epm:interp-critique v<n>` and `epm:interp-critique-codex v<n>` exist, verdicts disagree (PASS vs REVISE), no `epm:interp-critique-reconcile v<n>` | reconciler not yet started | spawn `reconciler` (marker mode) |
| `interpreting` | both ensemble markers exist, verdicts agree OR reconcile marker present, ensemble verdict REVISE, round < 3 | revision needed | re-spawn analyzer with all critique markers |
| `interpreting` | ensemble verdict PASS or round >= 3, no `epm:clean-result-critique` v1 | content honesty settled, structure + register + statistical-framing loop not started | promote source row in place to clean-result if not yet done (set-body + set-title + set-clean-result), then spawn `clean-result-critic` + `codex-clean-result-critic` in parallel (ROUND 1 ENSEMBLE) |
| `interpreting` | `epm:clean-result-critique v1` exists, no `epm:clean-result-critique-codex v1` | Codex twin not yet returned (round 1) | re-spawn `codex-clean-result-critic` only |
| `interpreting` | `epm:clean-result-critique-codex v1` exists, no `epm:clean-result-critique v1` | Claude critic not yet returned (round 1) | re-spawn `clean-result-critic` only |
| `interpreting` | both round-1 critique markers exist, verdicts disagree (PASS vs REVISE), no `epm:clean-result-critique-reconcile v1` | reconciler not yet started | spawn `reconciler` (marker mode) |
| `interpreting` | round-1 ensemble verdict REVISE, round < 3 | structure / register / statistical-framing revision in progress | re-spawn analyzer with all critique markers; subsequent critique rounds spawn Claude `clean-result-critic` ONLY (no Codex twin on rounds 2-3) |
| `interpreting` | `epm:clean-result-critique v<n>` (n≥2) REVISE, n < 3 | rounds 2-3 Claude-only revision in progress | re-spawn analyzer with the latest clean-result-critique |
| `interpreting` | latest ensemble or Claude-only `epm:clean-result-critique` PASS, OR round >= 3 | clean-result ready for user promotion | advance source issue to `awaiting_promotion`, post chat instructions, EXIT |
| `reviewing` (legacy) | issue already at status:reviewing pre-2026-05-13 | dedicated reviewer step retired; route to awaiting_promotion as if Step 9a-bis just PASSed | advance source issue to `awaiting_promotion`, post chat instructions, EXIT |
| `awaiting_promotion` | latest `epm:clean-result-critique` PASS, source row's `runs.classification` still `pending` | waiting for user to promote | show source row's dashboard link, prompt to promote, EXIT |
| `awaiting_promotion` | source row's `runs.classification` is `useful` / `not_useful` (status `completed`) | user promoted | advance to Step 10 (auto-complete) |
| `followups_running` | at least one open child issue (`Parent: #<N>` in body) lacks a terminal `status:*` label | children still in flight | show child-issue table + project-board URL, EXIT |
| `followups_running` | every open child has reached `done_experiment` / `done_impl` / `archived` (or no open children remain) | children all done | re-run Step 10: relabel parent `status:done_experiment` and move project column to "Done (experiment)" |
| `running` | `.claude/cache/watch-<N>.pid` is missing AND no `epm:results` / `epm:failure` posted | §2 watchdog crashed or never started | re-spawn `python scripts/pod.py watch --issue <N> ...` (skill side-effect; idempotent, the new watchdog inherits the run's heartbeat probes) |

Without distinct labels for `uploading` / `interpreting` / `reviewing` / `awaiting_promotion`,
many of these rows would be indistinguishable. That's why the state machine has them.

---

## Comment marker protocol

See `markers.md` for the full taxonomy. Every marker comment uses the format:

```markdown
<!-- epm:<kind> v<n> -->
## <Human-readable title>
<body>
<!-- /epm:<kind> -->
```

**Rules:**
- Opening and closing tags must match.
- Never delete or edit a marker comment -- always add a new one with a higher `v`.
  Version lets you see history; latest `v` wins for state purposes.
- `v1` is the original; `v2+` are revisions (e.g., revised plan after `/revise`).
- The HTML comment is hidden in rendered GitHub but parseable by the skill.

---

## Cost and safety rails

- **Hypothesis-gate (`scripts/hypothesis_gate.py`).** Static regex gate runs at
  two surfaces for `type:experiment` issues — Step 1 (clarifier, on the issue
  body) and Step 2 / adversarial-planner Phase 1.25 (on the drafted plan
  body). Refuses to advance without `Hypothesis` AND `Kill criterion` /
  `Kill criteria` section headers. Override via body marker
  `<!-- epm:override-hypothesis-skip v1 -->` (with rationale) — every
  override fires an `<!-- epm:hypothesis-gate v1: OVERRIDE -->` audit
  comment so the bypass is reviewable.
- **Never dispatch `compute:large` (>20 GPU-hours) without explicit user `approve`.**
  Small + medium can proceed on `approve` or `/approve`. Large requires
  `approve-large` to force a second thought.
- **Never auto-merge PRs.** User owns merge.
- **Never edit `RESULTS.md` without proposal+approval.** Headline-level
  science is high-stakes.
- **Never auto-delete worktrees or model artifacts.** Cleanup is manual via
  `python scripts/pod.py cleanup`.
- **Abort path:** user labels `status:blocked` -> skill posts `<!-- epm:abort v1 -->`
  and (if specialist is still running) sends abort signal. Specialist must check
  for `epm:abort` marker periodically.

---

## When NOT to use this skill

- Tasks <30 min of work (trivial typo fixes, config tweaks). Just do them.
- Sessions already running via `experimenter` / `implementer` as the main agent --
  they manage their own lifecycle. Issues are for dispatch, not retrofitting.
- Purely exploratory sessions (`ideation`, `experiment-proposer` output).
  Those produce proposals; the user decides which become issues.

---

## Error handling

| Symptom | Action |
|---------|--------|
| >1 `status:*` labels | Post error comment listing conflicts, post the §5 marker: `uv run python scripts/post_step_completed.py --issue <N> --step 0 --exit-kind failure-exit --notes "ambiguous status: >1 status:* labels"`, EXIT. Ask user to remove the wrong one. Do NOT pick. |
| 0 `status:*` labels | Run Step 0b: autofill `status:proposed`, post `epm:auto-defaults`, continue. (Old behavior — error+EXIT — was too brittle.) |
| `type:*` label missing | Run Step 0b (see Step 0b above): infer from title prefix, confirm with the user, apply chosen label. Autonomous loop with no user → error+EXIT (a wrong guess corrupts the Done column). |
| Empty experiment body | Run Step 0b: ask user for goal/hypothesis/setup in chat, draft body, patch via `python scripts/sagan_state.py set-body <N> --file …`, post `epm:auto-defaults` audit event. |
| Plan fails mandatory-section check | Re-invoke `adversarial-planner` with missing sections list; do not post incomplete plan. |
| Preflight fails | Post the `--json` report verbatim as `<!-- epm:preflight v1 -->`. Do NOT auto-fix (per CLAUDE.md "never take shortcuts"). |
| Specialist subagent errors out | Specialist posts `<!-- epm:failure v1 -->` with traceback + last log lines. Label -> `status:blocked`. |
| Reviewer FAIL | Post verdict, label -> `status:running`. User decides: revise in-place, spawn new specialist, or escalate. |
| Issue body lacks required fields | Post clarifier questions pointing to `.github/ISSUE_TEMPLATE/` for the right template. |
| Test suite crashes (OOM, import error) | Post `<!-- epm:test-verdict v1 -->` with FAIL + crash output. Stay in `status:testing`. Count toward 3-failure limit. |

Never silently skip a step. If something looks wrong, post a comment and exit --
the issue is the durable log.

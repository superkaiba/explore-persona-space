---
name: issue
description: >
  End-to-end GitHub-issue-driven workflow for experiments and code changes.
  Takes an issue number, parses state from labels + comment markers, and dispatches
  the next action (clarify -> adversarial-planner -> approval -> worktree +
  dispatch specialist -> preflight -> run -> analyzer -> reviewer -> test-verdict
  -> auto-complete). Reviewer PASS (or test-verdict PASS for code-change paths
  like type:infra / type:batch / type:analysis / type:survey) auto-advances the issue to Done
  on the Experiment Queue project board. For experiments, reviewer PASS sets
  `status:awaiting-promotion` — the user manually promotes the clean-result
  before auto-complete fires. Issues stay OPEN
  -- DO NOT close. Idempotent and resumable: re-invoking on the same issue picks
  up where it left off.
user_invocable: true
---

# Issue-Driven Workflow

## Scope & Boundaries

**Owns:** the full issue lifecycle — clarify → adversarial-planner → approval → worktree → dispatch → preflight → run → analyze → review → auto-complete.

**Invokes:** `experiment-runner` (run step), `adversarial-planner` (plan step), specialist agents (experimenter / implementer / analyzer / reviewer / code-reviewer).

**Does NOT own:** proposing new experiments (→ `experiment-proposer`) or overnight queue orchestration (→ `auto-experiment-runner`).

---

Invoke as `/issue <N>` or `/issue <N> --resume`. The skill is the entry point from
a GitHub issue to a fully-executed, reviewed experiment or code change.

**Guiding principle:** all durable state lives on the GitHub issue (labels + marker
comments). The local filesystem holds caches only. You can close the terminal at
any step and `/issue <N>` picks up cleanly.

## Project-board status convention

The Experiment Queue project board's Status field is the single source of
truth for column names — see `NEW_COLUMN_SPEC` in `scripts/gh_project.py`.
Mapping between `status:*` labels (phase-authoritative) and project columns
(glance-coarse) is defined by `LABEL_TO_COLUMN` in the same file. Summary:

| Project column | `status:*` label(s) | Meaning |
|---|---|---|
| **To do** | `proposed`, `gate-pending`, or no `status:*` label | Not yet in the pipeline. User files issues here. |
| **Planning** | `planning` | Adversarial-planner is running. |
| **Plan awaiting review** | `plan-pending` | User action: approve plan to advance. |
| **In flight** | `approved`, `implementing`, `code-reviewing`, `testing`, `running`, `uploading`, `interpreting`, `reviewing`, `under-review` | All active-phase labels between approval and reviewer-PASS roll up here. The label tells you which phase. |
| **Blocked** | `blocked` | Stuck / paused; resolve dependency. |
| **Awaiting promotion** | `awaiting-promotion`, `clean-results:draft` | User action: review clean-result draft. |
| **Followups running** | `followups-running` | Parent's own work is done (clean-result promoted), but at least one open child issue (`Parent: #<N>` in body) is still in flight. Descriptive state on the parent — no new cells run inside it. Transitions to `done-experiment` once all children reach a terminal state. |
| **Useful** | `clean-results:useful` (plus the bare `clean-results` label for back-compat queries) | Promoted, paper-relevant. |
| **Not useful** | `clean-results:not-useful` (plus the bare `clean-results` label) | Promoted, archive candidate. |
| **Done** | `done-experiment`, `done-impl` | Terminal, issue stays OPEN. |
| **Archived** | `archived` | The issue is CLOSED. Auto-applied by `.github/workflows/project-archive-on-close.yml` on `issues.closed` UNLESS the issue carries a sticky label (`status:done-experiment`, `status:done-impl`, or any `clean-results:*`), in which case the workflow is a no-op and the column does NOT change. On `issues.reopened` the workflow strips `archived` and applies `proposed` (rejoins **To do**), again skipping sticky labels. The `/issue` skill never closes issues — only deliberate user `gh issue close` reaches this workflow. To un-Done an issue, use `gh issue edit --remove-label status:done-experiment` (or `--remove-label status:done-impl`). |

The skill moves the project status in exactly four places:
1. **Step 1 (clarifier "All clear"):** To do → **Planning** (first entry into the pipeline).
2. **Step 9a (analyzer creates clean-result):** the new clean-result issue (label `clean-results:draft`) → **Awaiting promotion**.
3. **Step 9b (user promotes draft via `python scripts/gh_project.py promote <N> useful|not-useful`):** clean-result issue → **Useful** or **Not useful**. The user then re-enters `/issue <source-N>` so Step 10 fires. Promotion is **user-only** — no agent or automation may flip `clean-results:draft` without explicit user invocation. Both verdicts are mandatory; there is no verdict-less promote path.
4. **Step 10 (auto-complete):** source issue → **Done**.

Between those, the project column tracks the `status:*` label automatically
(Planning → Plan awaiting review → In flight → Awaiting promotion → Done) via
the routing in `LABEL_TO_COLUMN`; reading the project column tells you the
phase. Manual `set-status` invocations are rarely needed — advancing the
`status:*` label is enough.

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
                     |--> status:plan-pending    <- AWAITING USER: approve?
                            |-- (user approve) --> status:approved
                                                  |-- (worktree + draft PR)
                                                     |--> status:implementing    <- experiment-implementer (type:experiment) OR implementer (type:infra/batch)
                                                            |-- (epm:experiment-implementation OR epm:results posted)
                                                               |--> status:code-reviewing   <- code-reviewer (fresh context)
                                                                      |-- FAIL + count<3 --> status:implementing (loop, v+1)
                                                                      |-- FAIL + count>=3 --> status:blocked
                                                                      |-- PASS + [type:experiment] --> status:running   <- experimenter (pod ops + monitoring)
                                                                            |-- (epm:results posted)
                                                                               |--> status:uploading  <- upload verifier
                                                                                      |-- (all artifacts verified, pod stopped)
                                                                                         |--> status:interpreting  <- analyzer + interp-critic loop
                                                                                                |-- (interpretation refined, clean-result created)
                                                                                                   |--> status:reviewing  <- final reviewer
                                                                                                          |-- PASS --> status:awaiting-promotion  <- AWAITING USER: promote clean-result
                                                                                                                        |-- (user promotes) -->
                                                                                                                              |-- open `Parent: #<N>` children exist --> status:followups-running  <- waits for children to finish; re-invoke /issue <N> later
                                                                                                                              |-- no open children                  --> status:done-experiment (+ follow-up proposer)
                                                                                                          |-- FAIL --> status:interpreting (revise)
                                                                      |-- PASS + [type:infra/batch/analysis/survey] --> test-verdict (inline) --> status:done-impl
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
| `gate-pending` | Hypothesis/kill-criterion gate blocked the plan; awaiting body fix. | **yes** |
| `planning` | Adversarial-planner is running. | no |
| `plan-pending` | Plan posted; awaiting user `approve`. | **yes** |
| `approved` | Plan approved; skill is creating worktree + draft PR. | no |
| `implementing` | (experiment|infra|...)-implementer is writing code. | no |
| `code-reviewing` | code-reviewer is reviewing the diff. | no |
| `testing` | Inline test-suite step (Step 9c, code-change paths only). | no |
| `running` | experimenter is running on the pod. | no |
| `uploading` | upload-verifier is checking artifacts. | no |
| `interpreting` | analyzer + interpretation-critic loop is running. | no |
| `reviewing` | reviewer (final adversarial gate) is running. | no |
| `blocked` | Aborted or stuck; awaiting user triage. | **yes** |
| `awaiting-promotion` | User action: promote clean-result via /clean-results promote. | **yes** |
| `followups-running` | Parent is done; children with `Parent: #<N>` are still in flight. | no |
| `done-experiment` | Terminal: experiment finished + clean-result promoted. Issue stays OPEN. | no |
| `done-impl` | Terminal: code change shipped + reviewed. Issue stays OPEN. | no |
| `archived` | Closed via `gh issue close`; auto-applied by archive workflow. | no |
<!-- /workflow.yaml: AUTO-GENERATED -->

The two user-gated states in the active lifecycle are `plan-pending` (plan approval) and `awaiting-promotion` (clean-result promotion). `blocked` and `gate-pending` also need user attention but represent stalled / pre-pipeline states. Everything between is automatic, short of a `status:blocked` override.

Abort affordance: any state, user labels `status:blocked` -> skill posts abort
request, watcher kills run if one exists.

---

## Orchestration Procedure

When invoked, ALWAYS follow this order. Skip only what the state dictates.

**Chat title updates (verbose format).** Fires on (a) every status-label
transition, (b) when a `epm:follow-ups` marker is posted, (c) when a
clean-result issue is created, (d) when the merge prompt fires (Step 10d).

Format string:
```
#<N> <type:label> — <human-readable status sentence>[ — next: <next-action>][ — followups: #X[, #Y]][ — clean-result #<M>: <claim summary trimmed to 60 chars>]
```

Examples:
- `#226 type:infra — implementing workflow improvements — next: code-review`
- `#226 type:infra — code-review FAIL round 2 — next: respawn implementer`
- `#137 type:experiment — done-experiment — followups: #240, #241 — clean-result #310: persona collapse hero`

Helper pseudocode:
```python
def render_title(issue, *, status_human, next_action=None, followups=None, clean_result=None):
    parts = [f"#{issue.number} {issue.type_label} — {status_human}"]
    if next_action:
        parts.append(f"next: {next_action}")
    if followups:
        parts.append("followups: " + ", ".join(f"#{n}" for n in followups))
    if clean_result:
        claim = clean_result.title[:60]
        parts.append(f"clean-result #{clean_result.number}: {claim}")
    return " — ".join(parts)

# Cosmetic; if mcp__happy__change_title is unavailable, log and continue.
mcp__happy__change_title({ "title": render_title(...) })
```

If the MCP tool is unavailable (e.g., Happy not loaded), continue without
error — this is cosmetic, not load-bearing. Do NOT let a title-update failure
block the pipeline.

### Step 0: Load state

```
gh issue view <N> --json number,title,body,labels,state,assignees,comments
```

From the result, derive:
1. **Current state** = the `status:*` label value (exactly one should exist)
2. **Issue type** = the `type:*` label value (`experiment`, `infra`, `batch`, `analysis`, `survey`)
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
   gh issue edit <N> --add-label status:proposed
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
   gh issue edit <N> --body "<drafted body>"
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
     ≥3 unrelated fixes → suggest `type:batch`
   - Title prefix `Analyze:` / `Re-analyze:` → suggest `type:analysis`
   - Title prefix `Survey:` / `Read:` / `Lit review:` → suggest `type:survey`

   Use `AskUserQuestion` with the inferred option as `(Recommended)` first. Apply via
   `gh issue edit <N> --add-label type:<chosen>`. If the user is absent (e.g., autonomous
   loop), DO error and EXIT — the type label gates Step 7's Done variant and a guess
   here corrupts the project board.

4. **Other useful labels missing** (`compute:*`, `prio:*`):  do not block on these.
   `compute:*` will be set in the adversarial-planner's reproducibility card; `prio:*` is user-curated and
   never blocking.

   Note: legacy `aim:*` labels were deleted in #251 (slice 1). New issues do not use them.
   Topic categorization for new work lives in `docs/claims.yaml` (`topic` field) and
   in `RESULTS.md` / `eval_results/INDEX.md` H2 prose; no replacement GitHub labels exist.

After Step 0b, re-read the issue (re-run the `gh issue view` from Step 0) so downstream
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
  uv run python scripts/gh_project.py set-status <N> "Planning"
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
       the issue body"), run `gh issue edit <N> --body "<new body>"` with the original
       body preserved + a `## Spec (from clarifier)` section appended. Only do this on
       explicit request -- default is comment-only.
     - Re-run the clarifier evaluation using (body + clarify questions + these answers).
       If no blocking ambiguities remain, advance to Step 2 (adversarial planning) in the
       same invocation. If still ambiguous, loop: post a `v+1` clarify marker and ask again.

  4. **If the user defers ("I'll answer later", no reply, or says to exit):** EXIT with
     label still `status:proposed`. User can answer later as issue comments and
     re-invoke `/issue <N>`, OR re-invoke and answer in chat next time.

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
PLAN_URL=$(gh issue comment <N> --body-file .claude/plans/issue-<N>-comment-body.md | tail -1)
```

`gh issue comment` prints the comment URL on stdout — capture it as a
shell variable in the SAME bash block that posts the comment. **Do not
persist `PLAN_URL` to a cache file.** The variable lives only for the
duration of Steps 2a → 2c, which run in the same orchestrator turn (the
auto-continuation policy in CLAUDE.md guarantees no pause between them
in interactive mode; in autonomous mode the orchestrator exits at Step
2c so the variable is irrelevant).

Cache a copy of the plan body at `.claude/plans/issue-<N>.md` (cache
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

Advance label to `status:plan-pending`.

### Step 2c: Inline plan approval

**Context-dependent behavior:**

- **Autonomous mode** (invoked from `auto-experiment-runner` or with no user
  present): EXIT immediately. The issue sits at `status:plan-pending` until a
  user approves via GitHub comment or a future `/issue <N>` invocation. This
  preserves the old asynchronous review behavior.

- **Interactive mode** (user is in the current chat session): Ask the user
  inline rather than exiting. Present the plan summary and ask:

  > Plan posted as `epm:plan v1` on issue #\<N\>.
  >
  > **Plan:** ${PLAN_URL}
  > **Cached copy:** `.claude/plans/issue-<N>.md`
  >
  > (1) **Approve** — advance to implementation
  > (2) **Revise** \<notes\> — plan goes back to adversarial-planner
  > (3) **Defer** — exit now; re-invoke `/issue <N>` later

  `${PLAN_URL}` is the inline shell variable captured at Step 2a — both
  steps run in the same orchestrator turn (auto-continuation guarantees
  no pause between them) so the variable is in scope. There is no
  cache-file fallback.

  Use `AskUserQuestion` or a plain text prompt and wait for the user's reply.

  - **"Approve" / "1":** Advance label to `status:approved`. Post an `approve`
    comment on the issue for audit trail. Continue to Step 4 in the **same
    invocation** — do NOT exit.
  - **"Revise \<notes\>" / "2":** Set label back to `status:planning`. Re-invoke
    adversarial-planner with the revision notes. Re-run the consistency checker.
    Post updated `epm:plan v2`. Loop back to Step 2c.
  - **"Defer" / "3":** EXIT. Label stays at `status:plan-pending`. Identical to
    the old behavior — user re-invokes `/issue <N>` later to approve.

### Step 3: Approval check (backward compat, runs on re-invocation)

Runs on re-invocation if `status:plan-pending` (i.e., user deferred or approved
via GitHub comment rather than inline).

Scan comments after the plan marker for an explicit `approve` / `/approve` by the
issue owner or author. If found, advance label to `status:approved`. If comments
contain revision requests (`/revise <notes>`), set label back to `status:planning`,
re-invoke adversarial-planner with the notes; **also re-run the consistency
checker against the revised plan and post `epm:consistency v<n>` (a v2 plan
that adds new conditions or shifts baselines must not skip the consistency
gate)**; post the new `epm:plan v2` comment with the fresh consistency
verdict appended; set label back to `status:plan-pending`.

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
| `type:infra` / `type:batch` / code change | `implementer` | `epm:results` |
| `type:analysis` | `analyzer` (re-analysis only) | `epm:analysis` |
| `type:survey` | `general-purpose` | `epm:results` |

Brief passed to the implementer:
- The plan (cached at `.claude/plans/issue-<N>.md`)
- Issue number + worktree path + branch name
- Code-review history if this is a revision round (`epm:code-review v<m>`)
- Required `report-back` fields
- **Instruction: work ONLY inside the worktree; never touch a pod; post
  progress as comments on issue #<N> via `gh issue comment`.**
- **If `type:batch`:** make ONE commit per plan section (the planner
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

Advance label to `status:implementing`. EXIT. Implementer runs autonomously.

### Step 5: Code review loop

Only if `status:implementing` and the appropriate implementation marker
(`epm:experiment-implementation v<n>` for experiments, `epm:results v<n>` for
infra) is present.

**5a. Spawn code-reviewer (fresh context).** The reviewer sees only the brief
this skill assembles. The brief MUST contain:

- `issue_number` — the GitHub issue (`<N>`)
- `target_marker_kind` — exactly one of `experiment-implementation` (for
  `type:experiment`) or `results` (for `type:infra` / `type:batch` /
  `type:analysis` / `type:survey`). The reviewer reads the highest-version
  comment with this kind as the implementer's report.
- `revision_round` — 1-indexed integer. `1` on first review, `2` after a
  FAIL+respawn, `3` is the final allowed round before this skill labels the
  issue `status:blocked`. Reviewer must NOT itself loop on a FAIL.
- `previous_critique_summaries` — a list of one-line summaries of every
  prior `epm:code-review` comment on this issue (empty on round 1). Lets
  the reviewer notice patterns the implementer keeps re-introducing.
- The diff vs `main`
- The approved plan
- The existing codebase

It does NOT see the implementer's reasoning — independence is load-bearing.

Posts `<!-- epm:code-review v<n> -->` with verdict `PASS / CONCERNS / FAIL`
+ line-level findings.

**5b. Loop on FAIL.**

- **PASS** (or `CONCERNS`, which is non-blocking):
  - `type:experiment` → advance label to `status:running`, proceed to Step 6.
  - `type:infra` / `type:batch` / `type:analysis` / `type:survey` → skip
    pod phase, advance directly to `status:reviewing` (the inline
    test-verdict gate at Step 9c runs from there).
- **FAIL + revision_round<3** → label back to `status:implementing`.
  Re-spawn the implementer with the `epm:code-review v<n>` marker as part
  of the brief. Implementer posts v<n+1>; loop back to 5a with
  `revision_round = n+1`.
- **FAIL + revision_round>=3** → `status:blocked`. Post abort summary, EXIT.
  User decides: revise plan, escalate, or override.

Advance label to `status:code-reviewing` while the reviewer is running, back
to `status:implementing` on FAIL, forward to `status:running` (or
`status:reviewing` for non-experiment types) on PASS.

### Step 6: Pod provisioning + experimenter dispatch (type:experiment only)

Only if `status:running` (entered from Step 5b PASS for `type:experiment`)
and no `epm:launch` marker exists.

#### Step 6a: HF gate auto-acceptance

Plans never make the human click through
gated-model gate pages. Before provisioning, scan the cached plan for HF
model IDs and submit gate-acceptance requests using the user's `HF_TOKEN`:

```bash
uv run python scripts/hf_gate_accept.py --from-plan .claude/plans/issue-<N>.md
```

The helper is idempotent (already-accessible repos exit `OK` immediately).
For "auto-approval" gates (the common case for almanach/Inria/Meta/Qwen
research releases) the access is granted on submission. For the rare
manual-approval gate the request is queued and the helper exits with code 1
and a list of URLs.

- Exit code `0` → proceed to 6b.
- Exit code `1` (manual approval still needed) → post `<!-- epm:hf-gate-pending v1 -->`
  with the URLs, leave label at `status:running`, EXIT. User clicks through,
  re-runs `/issue <N>`.
- Exit code `2` (`HF_TOKEN` missing) → post `<!-- epm:hf-gate-pending v1 -->`
  with diagnostic, label `status:blocked`. EXIT.

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
errors/warnings, EXIT. User fixes, re-runs.

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
forbids pod lifecycle management (line ~305). Pod stop happens in Step 8
(after upload-verification PASS); pod termination happens in Step 10c (with
user approval). Including `pod.py terminate` or `pod.py stop` in the
experimenter's instructions bypasses these gates and risks premature
destruction of cached artifacts needed by follow-up issues.

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

Label stays at `status:running`. EXIT. Experimenter runs autonomously. The
experimenter posts `epm:progress`, `epm:hot-fix` (if needed), and finally
`epm:results`. The watchdog stops itself when `epm:results` is observed,
the status label moves out of `running`, or its pid file is deleted.

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
1. Check `epm:results` exists. If not, show last progress and EXIT. **If the
   most recent `epm:progress` comment is older than 4 hours and there is no
   `epm:results` or `epm:failure`, post `<!-- epm:stale v1 -->` asking the
   user to investigate (the experimenter may have crashed silently); leave
   the label at `status:running`.**
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
   # Pipe the failure body via stdin to avoid shell-quoting traps:
   cat <(gh issue view "$N" --comments --json comments --jq \
       '.comments[] | select(.body | contains("<!-- epm:failure")) | .body' \
       | tail -n +1) \
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

- **PASS** -> stop the pod, then advance to `status:interpreting` and proceed to Step 9.
  Once artifacts are confirmed at permanent URLs, the pod is no longer needed —
  interpretation runs locally:
  ```bash
  python scripts/pod.py stop --issue <N>
  ```
  This pauses the pod (volume preserved) and starts the TTL clock. If
  interpretation later needs the pod (e.g., to regenerate a figure from raw
  outputs), `pod.py resume --issue <N>` brings it back. If the issue body has
  `Parent: #<M>`, stop the parent's pod (`epm-issue-<M>`) instead. Skip the stop
  call only if the user has labelled the issue `keep-running` for known
  follow-up work in the same session.
- **FAIL** -> dispatch the `uploader` agent (up to 3 rounds) to close the gaps.
  The uploader receives the verifier's missing-artifacts list, lifecycle-aware
  resumes the pod if needed, pushes to HF / WandB / git, and posts an
  `epm:upload-fix v1` marker. After each uploader round, re-run `upload-verifier`;
  it posts a fresh `epm:upload-verification v<N+1>`.

  Round outcomes:
  - **uploader COMPLETE + verifier PASS** -> proceed as PASS branch above.
  - **uploader BLOCKED** (e.g., RunPod host capacity, missing credentials)
    -> stays at `status:uploading`. Post the uploader's `epm:upload-fix`
    marker with the blocker, EXIT, await operator action.
  - **3rd round still FAIL** -> label `status:blocked`, EXIT (mirror the
    code-reviewer FAIL escalation in CLAUDE.md).

  See `.claude/agents/uploader.md` for the uploader's contract and the
  marker schema. The uploader NEVER terminates pods; only stops/resumes.

### Step 9: Iterative interpretation + final review

This step has two sub-phases: **interpretation** (iterative analyzer↔critic loop)
and **final review** (one-shot reviewer gate).

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

2. Spawn `interpretation-critic` agent (fresh context, does NOT see analyzer reasoning).
   The critic reviews through 5 lenses:
   - **Overclaims:** does the prose say more than the data supports?
   - **Surprising unmentioned patterns:** critic independently loads raw JSON/plots,
     looks for patterns the analyzer didn't mention.
   - **Alternative explanations:** for each finding, what's the simplest non-mechanism
     explanation? Is it addressed?
   - **Confidence calibration:** does the confidence level match evidence (seeds, OOD, confounds)?
   - **Missing context:** are prior related results cited and compared?

   Posts `<!-- epm:interp-critique v1 -->` with PASS or REVISE + specific revision requests.

**If REVISE (rounds 2-3):**

Re-spawn analyzer (fresh context, sees original data + all critique feedback).
Analyzer posts `<!-- epm:interpretation v2 -->`. Re-spawn critic (fresh context,
sees v2 + prior critique). Posts `<!-- epm:interp-critique v2 -->`.

**Max 3 rounds.** After round 3, advance regardless with full critique history.

**On PASS (or max rounds reached):**

The analyzer creates the clean-result GitHub issue directly:
- Title: `<claim summary> (HIGH|MODERATE|LOW confidence)`
- Labels: `clean-results:draft`
- Body: fact sheet + refined interpretation per `template.md`
- Runs `scripts/verify_clean_result.py` — FAIL blocks posting.

Posts `<!-- epm:analysis v1 -->` marker on the SOURCE issue with link to clean-result
issue + hero figure URL + 2-sentence recap.

# Fire title update on clean-result creation.
# mcp__happy__change_title({"title": render_title(issue, status_human="reviewing", clean_result=<new-issue>)})

Advance label to `status:reviewing`.

**9b. Final reviewer gate** (only if `status:reviewing`, type:experiment)

Spawn `reviewer` agent in fresh context. Sees only:
- The raw results
- The plan
- The clean-result issue body (NOT the analyzer's reasoning or critique history)

Reviewer verdict: PASS / CONCERNS / FAIL. Post as `<!-- epm:reviewer-verdict v1 -->`.

Transitions:
- **PASS:** Clean-result STAYS at `clean-results:draft` (do NOT auto-promote).
  Advance source issue to `status:awaiting-promotion`:
  ```
  gh issue edit <N> --remove-label status:reviewing --add-label status:awaiting-promotion
  ```
  Post comment:
  > Reviewer PASS. Clean-result #\<clean-result-N\> is ready for your review.
  > When satisfied, promote it (USER-ONLY — no automation may do this):
  >   `python scripts/gh_project.py promote <clean-result-N> useful` (paper-relevant)
  >   `python scripts/gh_project.py promote <clean-result-N> not-useful` (archive candidate)
  > Then re-enter `/issue <N>` to fire Step 10.

  EXIT. The user reviews the clean-result at their own pace and manually
  picks a verdict. **Awaiting promotion is a user-only column — no agent
  or automation may move an issue out of it.** The `gh_project.py promote`
  command flips `clean-results:draft -> clean-results:<verdict>` (KEEPS
  legacy `clean-results` for back-compat), routes the project board to
  `Useful` / `Not useful`, and prints a reminder to re-enter `/issue <N>`
  so Step 10 fires.
- **CONCERNS:** same as PASS (non-blocking). Recorded on verdict comment.
- **FAIL:** clean-result stays `:draft`. Source back to `status:interpreting`.
  Analyzer revises with reviewer feedback.

**On re-invocation at `status:awaiting-promotion`:**
1. Check if the clean-result issue has been promoted (label `clean-results`
   without `:draft`).
2. If promoted → advance to Step 10 (auto-complete).
3. If still `:draft` → show the clean-result link and EXIT. User hasn't
   promoted yet.

**9c. Test-verdict gate (code-change paths only, inline)**

Only for `type:infra` / `type:batch` / `type:analysis` / `type:survey`
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
`status:reviewing`, re-spawn implementer. FAIL (count >= 3) → `status:blocked`.

### Step 10: Auto-complete (fires after user promotes clean-result from `awaiting-promotion`, or `epm:test-verdict` PASS for code-change paths)

No user gate. The skill transitions the issue to a terminal-or-`followups-running` state automatically. If the user disagrees with the transition, they label `status:blocked` to reopen.

#### Step 10 step 0: Completion audit (NEW — gates entry to step 1)

Cheap insurance against drift on multi-part issues: re-read the ORIGINAL
issue body and verify every numbered ask is actually addressed. The reviewer
checks the *write-up*; this checks the *issue → work* contract.

1. Re-fetch the current issue body: `gh issue view <N> --json body`.
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
   - `type:infra` / `type:batch` / `type:analysis` / `type:survey` → grep
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
     `status:awaiting-promotion` / `status:reviewing` / `status:testing` as
     applicable), do NOT advance. The audit comment is the bounce-back
     payload. User either (a) modifies the issue body to reconcile
     resolved scope-creep, (b) re-runs the missing work via a follow-up
     `/issue` cycle, or (c) labels `status:awaiting-promotion` again to
     override. Per CLAUDE.md STATE-TO-`status:blocked` criterion 5.

#### Step 10 step 1+: existing flow

1. If code change: mark PR ready for review (not merge -- user merges).
2. Update `RESULTS.md` if the finding is headline-level (propose diff as comment
   `<!-- epm:results-md-diff v1 -->` -- do NOT auto-edit).
3. Update `eval_results/INDEX.md` with a new entry.

4. **Detect open follow-up children.** Search for any open issue whose body
   contains a literal `Parent: #<N>` reference to this issue:
   ```bash
   gh issue list --state open --search "Parent: #<N> in:body" \
     --json number,labels,state --jq '.'
   ```
   A child is "still in flight" if it is open AND does NOT carry a terminal
   `status:*` label (`done-experiment`, `done-impl`, `archived`). The parent's
   destination state depends on whether ANY child is still in flight.

5. **Choose the destination state.**

   - **At least one child still in flight** AND `type:experiment`
     → **`status:followups-running`** + Project Status `"Followups running"`.
     The parent's own work is finished but its children own the queue. Re-invoking
     `/issue <N>` later re-runs Step 10 step 4 — once all children reach a
     terminal state, the parent advances to `status:done-experiment`.
   - **No children in flight** AND `type:experiment`
     → **`status:done-experiment`** + Project Status `"Done (experiment)"`.
   - **`type:infra` / `type:batch` / `type:analysis` / `type:survey`**
     (regardless of children) → **`status:done-impl`** + Project Status
     `"Done (impl)"`. Code-change paths don't use `followups-running`
     because they don't seed experimental follow-ups via Step 10b.
   - **No `type:*` label** → STOP, post an error comment asking the user to add one.
     Do NOT pick a default, and do NOT advance the label until fixed.

6. Apply the chosen label (remove `status:reviewing`, `status:awaiting-promotion`,
   or `status:testing` / `status:followups-running` as applicable, add the new
   label chosen in step 5):
   ```
   gh issue edit <N> --add-label <new-label> --remove-label <prior-status>
   ```

7. Move the issue to the correct project-board column. The label flip in
   step 6 already routes the column automatically via `LABEL_TO_COLUMN`
   (`status:done-experiment` / `status:done-impl` → `Done`,
   `status:followups-running` → `Followups running`). Verify with
   `gh_project.py status <N>`; an explicit `set-status` is rarely needed.
   When you do invoke it, pass a column name from `NEW_COLUMN_SPEC`:
   ```
   uv run python scripts/gh_project.py set-status <N> "Done"
   # or, for the follow-ups branch:
   uv run python scripts/gh_project.py set-status <N> "Followups running"
   ```

8. Post final comment `<!-- epm:done v1 -->` (or `<!-- epm:followups-running v1 -->`
   for the followups-running branch) summarizing: outcome, key numbers, what's
   confirmed/falsified, what's next, plus a link to the promoted clean-result
   issue (for experiments) AND a list of in-flight child follow-ups (when
   transitioning to `status:followups-running`). Include the line
   `Moved to **<status-name>** on the project board.`
9. **LEAVE THE ISSUE OPEN.** Never call `gh issue close`. Done-ness lives on the
   project board, not in the issue's open/closed state. The only legitimate way
   for this skill to close an issue is a user-initiated duplicate / invalid / won't-fix
   triage -- never as the terminal state of a successful run.
10. Do NOT delete the worktree -- user decides when to clean up.
11. If `type:experiment` AND we just landed at `status:done-experiment` (no
    children blocked us), proceed to Step 10b (follow-up proposer). If we
    landed at `status:followups-running`, SKIP Step 10b — the proposer was
    already run in a prior `/issue <N>` invocation that produced the children
    we're now waiting on.

### Step 10b: Follow-up proposer (experiments only)

Auto-fires after `done-experiment` for `type:experiment` issues. Spawn the
`follow-up-proposer` agent with:
- The completed experiment's plan (`epm:plan`)
- The results (`epm:results`)
- The clean-result issue body
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
# mcp__happy__change_title({"title": render_title(issue, status_human="done-experiment", followups=[...])})

### Step 10c: Pod termination prompt (experiments only)

After Step 10b posts, ask the user for permission to terminate the experiment's
pod. Skip if the issue body has `Parent: #<M>` (the parent owns the pod —
termination is decided when the parent's `/issue` run reaches this step).

Use `AskUserQuestion`:

> **Terminate `epm-issue-<N>`?** The pod is currently stopped (volume preserved).
> Terminating destroys the volume; any follow-up issue would spin a fresh pod
> and re-bootstrap (~few min + base-model re-download into the HF cache).
>
> Options: **Terminate** (recommended if no follow-ups planned) / **Keep stopped**
> (the pod stays parked until you run `pod.py resume --issue <N>` or
> `pod.py terminate --issue <N> --yes` manually — there is no auto-cleanup).

- **Terminate** → run `python scripts/pod.py terminate --issue <N> --yes`. Post
  `<!-- epm:pod-terminated v1 -->` with the command output.
- **Keep stopped** → no-op. Post `<!-- epm:pod-kept-stopped v1 -->` reminding
  the user that the pod must be cleaned up manually.
- **Autonomous mode (no user present)** → default to **Keep stopped** and post
  the marker. Never terminate without explicit user approval.

Idempotent: if either marker already exists, skip this step.

### Step 10d: Worktree merge prompt (NEW — both experiment and impl)

After Step 10c (or after Step 10's terminal label set for `type:infra` /
`type:batch` / `type:analysis` / `type:survey` paths that skip Step 10c),
ask the user
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
| Step 2c plan-pending awaiting `approve` | 2c | user-gated | `parked` |
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
| Step 9 `awaiting-promotion` user reviews | 9 | user-gated | `parked` |
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
| `plan-pending` | `epm:plan` exists | awaiting user approval | show plan URL, EXIT |
| `implementing` | no `epm:experiment-implementation` (or `epm:results` for infra), no `epm:proposed-tests` either | implementer was cancelled | re-spawn implementer |
| `implementing` | `epm:proposed-tests v<n>` exists, no `epm:experiment-implementation`, no `approve-tests` comment posted **after** the `proposed-tests` marker | TDD mode: tests posted, awaiting user approval | show the `proposed-tests` marker URL + the `approve-tests` reply instruction, EXIT |
| `implementing` | `epm:proposed-tests v<n>` exists, an `approve-tests` comment exists **after** the `proposed-tests` marker, no `epm:experiment-implementation` | TDD tests approved by user | re-spawn implementer with `tdd_approved=true`; brief instructs implementer to write implementation against the approved tests, then post `epm:experiment-implementation v1` as normal |
| `implementing` | latest `epm:code-review` is FAIL, round < 3 | revision in progress | re-spawn implementer with critique |
| `implementing` | latest `epm:code-review` is FAIL, round >= 3 | exhausted retries | label `status:blocked`, ask user |
| `code-reviewing` | no `epm:code-review` for the current implementation version | code-reviewer was cancelled | re-spawn code-reviewer |
| `running` | no `epm:results` for > 4h | experimenter crashed silently | post `epm:stale`, ask user |
| `running` | latest marker is `epm:failure` with bounce-back proposal | experimenter bounced to implementer | label back to `status:implementing`, re-spawn experiment-implementer |
| `uploading` | no `epm:upload-verification` PASS | verifier not run or failed | re-run upload-verifier |
| `interpreting` | no `epm:interpretation` | analyzer not started | spawn analyzer |
| `interpreting` | `epm:interpretation` exists, no `epm:interp-critique` | critic not started | spawn interpretation-critic |
| `interpreting` | `epm:interp-critique` REVISE, round < 3 | revision needed | re-spawn analyzer with critique |
| `interpreting` | `epm:interp-critique` PASS or round >= 3 | ready for review | create clean-result, advance to `reviewing` |
| `reviewing` | missing `epm:reviewer-verdict` | reviewer not started | spawn reviewer |
| `reviewing` | `epm:reviewer-verdict` FAIL | interpretation needs more work | back to `interpreting` |
| `awaiting-promotion` | `epm:reviewer-verdict` PASS, clean-result still `:draft` | waiting for user to promote | show clean-result link, prompt to promote, EXIT |
| `awaiting-promotion` | clean-result has `clean-results` label (no `:draft`) | user promoted | advance to Step 10 (auto-complete) |
| `followups-running` | at least one open child issue (`Parent: #<N>` in body) lacks a terminal `status:*` label | children still in flight | show child-issue table + project-board URL, EXIT |
| `followups-running` | every open child has reached `done-experiment` / `done-impl` / `archived` (or no open children remain) | children all done | re-run Step 10: relabel parent `status:done-experiment` and move project column to "Done (experiment)" |
| `running` | `.claude/cache/watch-<N>.pid` is missing AND no `epm:results` / `epm:failure` posted | §2 watchdog crashed or never started | re-spawn `python scripts/pod.py watch --issue <N> ...` (skill side-effect; idempotent, the new watchdog inherits the run's heartbeat probes) |

Without distinct labels for `uploading` / `interpreting` / `reviewing` / `awaiting-promotion`,
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
| >1 `status:*` labels | Post error comment listing conflicts, EXIT. Ask user to remove the wrong one. Do NOT pick. |
| 0 `status:*` labels | Run Step 0b: autofill `status:proposed`, post `epm:auto-defaults`, continue. (Old behavior — error+EXIT — was too brittle.) |
| `type:*` label missing | Run Step 0b (see Step 0b above): infer from title prefix, confirm with the user, apply chosen label. Autonomous loop with no user → error+EXIT (a wrong guess corrupts the Done column). |
| Empty issue body | Run Step 0b: ask user for goal/hypothesis/setup in chat, draft body, patch via `gh issue edit --body`, post `epm:auto-defaults` audit comment. |
| Plan fails mandatory-section check | Re-invoke `adversarial-planner` with missing sections list; do not post incomplete plan. |
| Preflight fails | Post the `--json` report verbatim as `<!-- epm:preflight v1 -->`. Do NOT auto-fix (per CLAUDE.md "never take shortcuts"). |
| Specialist subagent errors out | Specialist posts `<!-- epm:failure v1 -->` with traceback + last log lines. Label -> `status:blocked`. |
| Reviewer FAIL | Post verdict, label -> `status:running`. User decides: revise in-place, spawn new specialist, or escalate. |
| Issue body lacks required fields | Post clarifier questions pointing to `.github/ISSUE_TEMPLATE/` for the right template. |
| Test suite crashes (OOM, import error) | Post `<!-- epm:test-verdict v1 -->` with FAIL + crash output. Stay in `status:testing`. Count toward 3-failure limit. |

Never silently skip a step. If something looks wrong, post a comment and exit --
the issue is the durable log.

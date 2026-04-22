---
name: issue
description: >
  End-to-end GitHub-issue-driven workflow for experiments and code changes.
  Takes an issue number, parses state from labels + comment markers, and dispatches
  the next action (clarify -> gate-keeper -> adversarial-planner -> approval -> worktree +
  dispatch specialist -> preflight -> run -> analyzer -> reviewer -> tester -> close).
  Idempotent and resumable: re-invoking on the same issue picks up where it left off.
user_invocable: true
---

# Issue-Driven Workflow

Invoke as `/issue <N>` or `/issue <N> --resume`. The skill is the entry point from
a GitHub issue to a fully-executed, reviewed experiment or code change.

**Guiding principle:** all durable state lives on the GitHub issue (labels + marker
comments). The local filesystem holds caches only. You can close the terminal at
any step and `/issue <N>` picks up cleanly.

## Companion files

- `markers.md` -- comment marker taxonomy (source of truth for state parsing)
- `clarifier.md` -- clarifying-question prompts per issue type
- `templates/` -- plan / results / analysis comment body templates

Read these on first invocation of the skill in a session.

---

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
       |-- OK --> status:gate-pending       <- gate-keeper running
                  |-- (verdict)
                     |-- RUN --> status:planning        <- adversarial-planner running
                     |          |-- (plan posted)
                     |             |--> status:plan-pending    <- AWAITING USER: approve?
                     |                    |-- (user approve) --> status:approved
                     |                                          |-- (preflight + dispatch)
                     |                                             |--> status:running   <- specialist running
                     |                                                    |-- (epm:results posted)
                     |                                                       |--> status:reviewing  <- analyzer + reviewer/code-reviewer running
                     |                                                              |-- (reviewer verdict)
                     |                                                                 |-- [type:infra] --> status:testing  <- tester running
                     |                                                                 |                    |-- (test verdict)
                     |                                                                 |                       |-- PASS --> status:under-review  <- AWAITING USER: /signoff?
                     |                                                                 |                       |-- FAIL --> status:testing (fix cycle, max 3)
                     |                                                                 |                                    |-- (3 failures) --> status:blocked
                     |                                                                 |-- [NOT type:infra] --> status:under-review  <- AWAITING USER: /signoff?
                     |                                                                 status:under-review
                     |                                                                    |-- (user /signoff)
                     |                                                                       |-- [type:experiment] --> status:done-experiment (closed)
                     |                                                                       |-- [other] --> status:done-impl (closed)
                     |-- MODIFY --> status:proposed (back; revision notes posted)
                     |-- SKIP --> status:blocked (alternative suggested)
```

**Active vs awaiting-user states:**

| State | Who's working | User action needed? |
|-------|---------------|---------------------|
| `proposed` | nobody (new) OR user (answering clarifier) | sometimes |
| `gate-pending` | gate-keeper agent | no |
| `planning` | adversarial-planner agent | no |
| `plan-pending` | nobody | **yes -- approve plan** |
| `approved` | skill (dispatching) | no |
| `running` | experimenter/implementer specialist | no |
| `reviewing` | analyzer + reviewer agents | no |
| `testing` | tester (skill-internal) | no |
| `under-review` | nobody | **yes -- /signoff** |
| `blocked` | nobody (aborted or gate-skipped) | **yes -- triage** |
| `done-impl` | nobody (closed) | no |
| `done-experiment` | nobody (closed) | no |

Abort affordance: any state, user labels `status:blocked` -> skill posts abort
request, watcher kills run if one exists.

---

## Orchestration Procedure

When invoked, ALWAYS follow this order. Skip only what the state dictates.

### Step 0: Load state

```
gh issue view <N> --json number,title,body,labels,state,assignees,comments
```

From the result, derive:
1. **Current state** = the `status:*` label value (exactly one should exist)
2. **Issue type** = the `type:*` label value (`experiment`, `infra`, `analysis`, `survey`)
3. **Aim** = the `aim:*` label
4. **Marker map** = scan comments for `<!-- epm:<kind> v<n> -->` opening tags, build a dict

If 0 or >1 `status:*` labels, abort with an error comment asking the user to fix.

### Step 1: Clarifier gate

If `epm:clarify` marker missing (or user has replied but clarifier hasn't re-checked):
read `clarifier.md`, run the clarifier for this issue type, either:
- **All clear** (<=1 minor ambiguity) -> post `<!-- epm:clarify -->` with "No blocking
  ambiguities found. Proceeding to gate-keeper." and advance label to `status:gate-pending`.
- **Ambiguities remain** -> post `<!-- epm:clarify -->` with numbered questions and EXIT.
  User answers in a comment -> user re-runs `/issue <N>` -> skill re-reads all comments
  after the clarify marker and re-evaluates.

**Rule:** never proceed to gate-keeper with >=2 blocking ambiguities. Tight specs
save later backtracking.

### Step 2: Gate-keeper

Only if `status:gate-pending` and no `epm:gate` marker exists.

Spawn the `gate-keeper` agent via `Agent()` tool with:
- Issue title + body + clarifier resolution
- Compute label value (`compute:small|medium|large`)
- Aim label value
- Ask for RUN / MODIFY / SKIP verdict + 1-5 scores across info value, de-risking,
  strategic fit, feedback speed, opportunity cost

Post verdict as `<!-- epm:gate v1 -->` comment with:
- Scores (1-5 each)
- Verdict (RUN / MODIFY / SKIP)
- One-paragraph rationale

Transitions:
- **RUN (>=3.5)** -> label `status:planning`, advance immediately to Step 3.
- **MODIFY (2.5-3.4)** -> label back to `status:proposed`, post suggested
  modifications. EXIT. User iterates on issue body, re-invokes skill.
- **SKIP (<2.5)** -> label `status:blocked`, post alternative suggestion. EXIT.

### Step 3: Adversarial planning

Only if `status:planning`.

Invoke the `adversarial-planner` skill with the issue body + clarifier output as
the task. The skill runs planner -> fact-checker -> critic -> revise internally.

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

Post plan as `<!-- epm:plan v1 -->` comment. Cache a copy at
`.claude/plans/issue-<N>.md` (cache only -- GitHub is source of truth).

Also post estimated cost prominently at the top of the comment, e.g.
> **Cost gate:** estimated 12 GPU-hours on pod3 (8xH100). Reply `approve` to dispatch.

Advance label to `status:plan-pending`. EXIT. Wait for user approval.

### Step 4: Approval check

Runs on re-invocation if `status:plan-pending`.

Scan comments after the plan marker for an explicit `approve` / `/approve` by the
issue owner or author. If found, advance label to `status:approved`. If comments
contain revision requests (`/revise <notes>`), set label back to `status:planning`,
re-invoke adversarial-planner with the notes; post new `epm:plan v2` comment; set
label back to `status:plan-pending`.

### Step 5: Worktree + preflight + dispatch

Only if `status:approved` and no `epm:launch` marker exists.

**5a. Worktree.** Create `.claude/worktrees/issue-<N>` on branch `issue-<N>`.
```
git worktree add .claude/worktrees/issue-<N> -b issue-<N>
```
If branch already exists, reuse it (resume case).

**5b. Draft PR.** Open a draft PR with `Closes #<N>` in the body, linking back to
the issue. Use `gh pr create --draft --head issue-<N> --body "Closes #<N>"`.

**5c. Pod selection.** Read the plan's pod preference. Run `ssh_health_check` on
the target pod.
- If >= requested GPUs free -> proceed.
- If busy -> post `<!-- epm:pod-pending -->` comment listing available pods, label
  `status:approved` (don't advance), EXIT. User picks another pod or waits.

**5d. Preflight.** Run preflight on target pod:
```
ssh_execute(pod=<target>, command="cd /workspace/explore-persona-space && uv run python -m explore_persona_space.orchestrate.preflight --json")
```
Parse JSON. If `ok=false`, post `<!-- epm:preflight v1 -->` comment with the
errors/warnings, label remains `status:approved`, EXIT. User fixes, re-runs.

**5e. Dispatch specialist.** Spawn the appropriate agent via `Agent()`:
- `type:experiment` -> `experimenter` subagent
- `type:infra` or code change -> `implementer` subagent
- `type:analysis` (pure re-analysis of existing results) -> `analyzer` subagent
- `type:survey` (literature / exploratory read) -> `general-purpose` subagent

Brief passed to specialist:
- The plan (cached at `.claude/plans/issue-<N>.md`)
- Issue number + worktree path + branch name
- Pod assignment (if experiment)
- Required `report-back` fields (artifacts, WandB URL, HF Hub path, plan deviations)
- **Instruction: work ONLY inside the worktree; post progress as comments on
  issue #<N> via `gh issue comment`.**

Post `<!-- epm:launch v1 -->` comment containing:
- Worktree path, branch, PR URL
- Pod + PID (if applicable) + log path
- WandB run URL (if known at launch) -- specialist updates if not
- Specialist subagent ID (for monitoring)

Advance label to `status:running`. EXIT. Specialist runs autonomously.

### Step 6: Monitor -> results

Specialist is expected to post `<!-- epm:progress v1 -->` comments at major
milestones and a final `<!-- epm:results v1 -->` comment containing:
- Final eval numbers (inline JSON snippet + path in repo)
- Reproducibility card (filled)
- WandB URL + HF Hub model/adapter URL
- Worktree path + final commit hash
- GPU-hours actually used vs budgeted
- Plan deviations + rationale

When this skill is re-invoked in `status:running`:
1. Check `epm:results` exists. If not, show last progress and EXIT.
2. If exists, advance label to `status:reviewing` and proceed to Step 7.

### Step 7: Analyzer + reviewer + tester

Only if `status:reviewing` (or `status:testing`) and either `epm:analysis`,
`epm:reviewer-verdict`, or `epm:test-verdict` is missing.

**7a.** Spawn `analyzer` agent with raw result paths. Analyzer produces a draft
write-up following `templates/experiment_report.md`. Post the draft as
`<!-- epm:analysis v1 -->`. (For code-change issues, skip -- go straight to 7b with
code-reviewer.)

**7b.** Spawn `reviewer` agent (for experiments) or `code-reviewer` agent (for
code changes) in fresh context. Reviewer sees only:
- The raw results / diff
- The plan
- The analyzer draft (for experiments)

Reviewer verdict: PASS / CONCERNS / FAIL with line-level issues.

Post verdict as `<!-- epm:reviewer-verdict v1 -->`.

Transitions after reviewer/code-reviewer verdict:

- **PASS or CONCERNS:**
  - If `type:infra` label is present: advance label `status:reviewing` -> `status:testing`. Proceed to Step 7c.
  - If NOT `type:infra`: advance label `status:reviewing` -> `status:under-review`. EXIT.
- **FAIL** -> label back to `status:running` or `status:blocked`. (Unchanged.)

**7c. Tester (code changes only)**

Only if `status:testing` and no `epm:test-verdict` marker exists (or latest verdict is FAIL and count < 3).

This step runs inline in the `/issue` skill -- no separate agent needed.

**Why a separate tester step?** The code-reviewer (Step 7b) runs tests as part of its advisory review, but its test-running is not a hard gate. The tester step is a hard gate: FAIL blocks advancement to user sign-off. This catches regressions the code-reviewer may overlook or deprioritize.

**Procedure:**

1. **Unit tests** (always run):
   ```bash
   cd .claude/worktrees/issue-<N>
   uv run pytest tests/ -v --tb=short 2>&1
   ```

2. **Lint check** (always run):
   ```bash
   uv run ruff check . && uv run ruff format --check .
   ```

3. **Integration tests** (conditional):
   Only if a pod is assigned in the plan AND the diff touches any of these paths:
   - `src/explore_persona_space/train/**/*.py`
   - `src/explore_persona_space/eval/**/*.py`
   - `src/explore_persona_space/orchestrate/**/*.py`
   - `scripts/train.py`, `scripts/eval.py`, `scripts/run_sweep.py`

   Check with:
   ```bash
   git diff main...HEAD --name-only | grep -E '(src/explore_persona_space/(train|eval|orchestrate)/|scripts/(train|eval|run_sweep)\.py)'
   ```

   If matched and pod available:
   ```bash
   ssh <pod> "cd /workspace/explore-persona-space && git fetch && git checkout issue-<N> && uv run pytest tests/integration/ -m integration -v --tb=short"
   ```

4. **Coverage gap check** (report only, do not auto-generate tests):
   ```bash
   git diff main...HEAD --name-only --diff-filter=AM | grep -E '\.py$'
   ```
   For each new/modified .py file under `src/` or `scripts/`, check if a corresponding test exists in `tests/`. Flag gaps in the verdict comment.

**Post verdict as `<!-- epm:test-verdict v1 -->` comment:**

```markdown
<!-- epm:test-verdict v1 -->
## Test Verdict -- PASS / FAIL

**Unit tests:** X passed, Y failed, Z skipped
**Integration tests:** [ran on pod / skipped (no pod assigned)] X passed, Y failed
**Lint:** PASS / FAIL (ruff check + format)
**Coverage gaps:** [list of new files without tests, or "none"]

<details>
<summary>Full test output</summary>

[truncated pytest output, last 100 lines]

</details>
<!-- /epm:test-verdict -->
```

**Transitions:**
- **PASS** (0 unit test failures + lint clean) -> advance label `status:testing` -> `status:under-review`. EXIT.
- **FAIL** -> Count `epm:test-verdict` markers with verdict=FAIL. If count >= 3: advance to `status:blocked` with note "3 consecutive test failures." Otherwise: label stays `status:testing`. Post failure details. The implementer fixes in the worktree, commits, and re-invokes `/issue <N>` to re-run the tester. (No reviewer re-validation needed for test-only fixes.)

### Step 8: Sign-off + close

Only if `status:under-review` and reviewer PASS.

On user comment `/signoff`:
1. If code change: mark PR ready for review (not merge -- user merges).
2. Update `RESULTS.md` if the finding is headline-level (propose diff as comment
   `<!-- epm:results-md-diff v1 -->` -- do NOT auto-edit).
3. Update `eval_results/INDEX.md` with new entry.
4. Apply the appropriate done label based on `type:*`:
   - `type:experiment` -> add `status:done-experiment`, remove `status:under-review`
   - `type:infra` / `type:analysis` / `type:survey` -> add `status:done-impl`, remove `status:under-review`
5. Close issue with final comment `<!-- epm:closed v1 -->` summarizing:
   outcome, key numbers, what's confirmed/falsified, what's next.
6. Do NOT delete the worktree -- user decides when to clean up.

---

## Resume semantics

`/issue <N>` and `/issue <N> --resume` are identical. The skill is always
idempotent: it reads state from labels + markers, computes next action, and
executes. There is no "start from scratch" -- the only way to reset is to remove
labels and delete marker comments manually.

If the specialist subagent has exited but no `epm:results` marker was posted, the
skill assumes the run failed silently. On resume in `status:running` with no
progress in >4 hours, post `<!-- epm:stale v1 -->` comment asking user to
investigate and optionally label `status:blocked`.

**Resume correctness per active state** (the key benefit of having dedicated
"working" labels):

| Label at resume | `epm:*` markers present | Interpretation | Action |
|-----------------|-------------------------|----------------|--------|
| `gate-pending` | no `epm:gate` | gate-keeper was cancelled | re-run gate-keeper |
| `planning` | no `epm:plan` | planner was cancelled | re-run adversarial-planner |
| `plan-pending` | `epm:plan` exists | awaiting user approval | show plan URL, EXIT |
| `running` | no `epm:results` for > 4h | specialist crashed silently | post `epm:stale`, ask user |
| `reviewing` | missing `epm:analysis` or `epm:reviewer-verdict` | review was cancelled | resume from missing step |
| `testing` | no `epm:test-verdict` | tester was cancelled | re-run tester |
| `testing` | `epm:test-verdict` PASS | label not advanced | advance to `status:under-review` |
| `testing` | `epm:test-verdict` FAIL, count < 3 | fix needed | show failure, stay in `status:testing` |
| `testing` | `epm:test-verdict` FAIL, count >= 3 | stuck | advance to `status:blocked` |
| `under-review` | reviewer-verdict PASS | awaiting signoff | show summary, EXIT |

Without `planning` / `reviewing` / `testing` as distinct labels, many of these rows
would be indistinguishable from other states. That's why the state machine has them.

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
  Those produce proposals; gate-keeper decides which become issues.

---

## Error handling

| Symptom | Action |
|---------|--------|
| 0 or >1 `status:*` labels | Post error comment, EXIT. Ask user to fix. |
| Plan fails mandatory-section check | Re-invoke `adversarial-planner` with missing sections list; do not post incomplete plan. |
| Preflight fails | Post the `--json` report verbatim as `<!-- epm:preflight v1 -->`. Do NOT auto-fix (per CLAUDE.md "never take shortcuts"). |
| Specialist subagent errors out | Specialist posts `<!-- epm:failure v1 -->` with traceback + last log lines. Label -> `status:blocked`. |
| Reviewer FAIL | Post verdict, label -> `status:running`. User decides: revise in-place, spawn new specialist, or escalate. |
| Issue body lacks required fields | Post clarifier questions pointing to `.github/ISSUE_TEMPLATE/` for the right template. |
| Test suite crashes (OOM, import error) | Post `<!-- epm:test-verdict v1 -->` with FAIL + crash output. Stay in `status:testing`. Count toward 3-failure limit. |

Never silently skip a step. If something looks wrong, post a comment and exit --
the issue is the durable log.

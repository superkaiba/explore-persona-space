---
name: issue
description: >
  End-to-end GitHub-issue-driven workflow for experiments and code changes.
  Takes an issue number, parses state from labels + comment markers, and dispatches
  the next action (clarify -> gate-keeper -> adversarial-planner -> approval -> worktree +
  dispatch specialist -> preflight -> run -> analyzer -> reviewer -> tester -> auto-complete).
  Reviewer PASS (+ tester PASS for type:infra) auto-advances the issue to Done on the
  Experiment Queue project board. No user sign-off step. Issues stay OPEN -- DO NOT close.
  Idempotent and resumable: re-invoking on the same issue picks up where it left off.
user_invocable: true
---

# Issue-Driven Workflow

## Scope & Boundaries

**Owns:** the full issue lifecycle — clarify → gate-keeper → adversarial-planner → approval → worktree → dispatch → preflight → run → analyze → review → auto-complete.

**Invokes:** `experiment-runner` (run step), `adversarial-planner` (plan step), specialist agents (experimenter / implementer / analyzer / reviewer / code-reviewer).

**Does NOT own:** proposing new experiments (→ `experiment-proposer`) or overnight queue orchestration (→ `auto-experiment-runner`).

---

Invoke as `/issue <N>` or `/issue <N> --resume`. The skill is the entry point from
a GitHub issue to a fully-executed, reviewed experiment or code change.

**Guiding principle:** all durable state lives on the GitHub issue (labels + marker
comments). The local filesystem holds caches only. You can close the terminal at
any step and `/issue <N>` picks up cleanly.

## Project-board status convention

The Experiment Queue project board has six Status columns. Mapping between `status:*`
labels (phase-authoritative) and project columns (glance-coarse):

| Project column | `status:*` label(s) | Meaning |
|---|---|---|
| **Todo** | `proposed`, `blocked`, or no `status:*` label | Not yet in the pipeline. User files issues here. |
| **Priority** | any (user-set) | Flagged by user as next-to-work. Pipeline doesn't auto-set this. |
| **In Progress** | `gate-pending`, `planning`, `plan-pending`, `approved`, `running`, `uploading`, `interpreting`, `reviewing` | **ALL active-phase labels roll up here.** The label tells you which phase. |
| **Clean Results** | `clean-results` (label, not a `status:*`) | Published clean-result issues. |
| **Done (experiment)** | `done-experiment` | Terminal, issue stays OPEN. |
| **Done (impl)** | `done-impl` | Terminal, issue stays OPEN. |

The skill moves the project status in exactly three places:
1. **Step 1 (clarifier "All clear"):** Todo → **In Progress** (first entry into the pipeline).
2. **Step 7 (reviewer PASS on experiments):** the new clean-result issue goes to **Clean Results**.
3. **Step 8 (auto-complete):** source issue → **Done (experiment)** or **Done (impl)**.

Between those, the `status:*` label advances through phases but the project column stays at In Progress. Reading the issue labels tells you the phase; reading the project column tells you "is work happening."

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
                     |-- RUN --> status:planning        <- adversarial-planner + consistency-checker
                     |          |-- (plan posted + consistency PASS/WARN)
                     |             |--> status:plan-pending    <- AWAITING USER: approve?
                     |                    |-- (user approve) --> status:approved
                     |                                          |-- (preflight + dispatch)
                     |                                             |--> status:running   <- specialist running
                     |                                                    |-- (epm:results posted)
                     |                                                       |--> status:uploading  <- upload verifier
                     |                                                              |-- (all artifacts verified)
                     |                                                                 |--> status:interpreting  <- analyzer + critic loop
                     |                                                                        |-- (interpretation refined, clean-result created)
                     |                                                                           |--> status:reviewing  <- final reviewer
                     |                                                                                  |-- (reviewer verdict)
                     |                                                                                     |-- PASS + [type:experiment] --> status:done-experiment (+ follow-up proposer)
                     |                                                                                     |-- PASS + [type:infra] --> run tester inline --> status:done-impl
                     |                                                                                     |-- PASS + [type:analysis/survey] --> status:done-impl
                     |                                                                                     |-- FAIL --> status:interpreting (revise)
                     |-- MODIFY --> status:proposed (back; revision notes posted)
                     |-- SKIP --> status:blocked (alternative suggested)
```

There is no user sign-off step. Reviewer PASS (+ tester PASS for `type:infra`) is the terminal gate; completion is automatic. If the user disagrees with a done transition, they label `status:blocked` to reopen it.

**Active vs awaiting-user states:**

| State | Who's working | User action needed? |
|-------|---------------|---------------------|
| `proposed` | nobody (new) OR user (answering clarifier) | sometimes |
| `gate-pending` | gate-keeper agent | no |
| `planning` | adversarial-planner + consistency-checker agents | no |
| `plan-pending` | nobody | **yes -- approve plan** |
| `approved` | skill (dispatching) | no |
| `running` | experimenter/implementer specialist | no |
| `uploading` | upload-verifier agent | no |
| `interpreting` | analyzer + interpretation-critic agents (iterative loop) | no |
| `reviewing` | reviewer / code-reviewer agent (final gate) | no |
| `blocked` | nobody (aborted or gate-skipped) | **yes -- triage** |
| `done-impl` | nobody (issue stays OPEN; Project Status="Done (impl)") | no |
| `done-experiment` | nobody (issue stays OPEN; Project Status="Done (experiment)") | no |

The only user-gated state in the whole lifecycle is `plan-pending` (plan approval). Everything downstream is automatic, short of a `status:blocked` override.

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
3. **Marker map** = scan comments for `<!-- epm:<kind> v<n> -->` opening tags, build a dict

If 0 or >1 `status:*` labels, abort with an error comment asking the user to fix.

### Step 1: Clarifier gate

If `epm:clarify` marker missing (or user has replied but clarifier hasn't re-checked):
read `clarifier.md`, run the clarifier for this issue type, then:

- **All clear** (<=1 minor ambiguity) -> post `<!-- epm:clarify -->` with "No blocking
  ambiguities found. Proceeding to gate-keeper." advance label to `status:gate-pending`,
  **and move the project column to In Progress**:
  ```
  uv run python scripts/gh_project.py set-status <N> "In Progress"
  ```
  This is the one place where the project column transitions out of Todo / Priority
  into the active-work column. Subsequent phases (planning, running, reviewing, testing)
  keep the project column at In Progress — only the `status:*` label changes.

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
       If no blocking ambiguities remain, advance to Step 2 (gate-keeper) in the same
       invocation. If still ambiguous, loop: post a `v+1` clarify marker and ask again.

  4. **If the user defers ("I'll answer later", no reply, or says to exit):** EXIT with
     label still `status:proposed`. User can answer later as issue comments and
     re-invoke `/issue <N>`, OR re-invoke and answer in chat next time.

**Rule:** never proceed to gate-keeper with >=2 blocking ambiguities. Tight specs
save later backtracking.

**Rule:** the ask-in-chat step is MANDATORY when there are blocking ambiguities. Posting
questions only to GitHub and immediately exiting forces a context switch the user does
not want -- always offer the inline path first.

### Step 2: Gate-keeper

Only if `status:gate-pending` and no `epm:gate` marker exists.

Spawn the `gate-keeper` agent via `Agent()` tool with:
- Issue title + body + clarifier resolution
- Compute label value (`compute:small|medium|large`)
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

### Step 3b: Consistency checker

After the adversarial planner produces an APPROVE-rated plan, but BEFORE posting
it as `epm:plan`, spawn the `consistency-checker` agent. It receives:
- The drafted plan
- Related experiments (issues with the same `aim:*` label, or cited in the plan's prior work)
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
2. If exists, advance label to `status:uploading` and proceed to Step 6b.

### Step 6b: Upload verification

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

- **PASS** -> advance to `status:interpreting`, proceed to Step 7.
- **FAIL** -> stays at `status:uploading`. Post clear list of what's missing with
  commands to fix. EXIT. Experimenter or user fixes, re-invokes `/issue <N>`.

### Step 7: Iterative interpretation + review

This step has two sub-phases: **interpretation** (iterative analyzer↔critic loop)
and **final review** (one-shot reviewer gate).

**7a. Iterative interpretation** (only if `status:interpreting`)

Only for `type:experiment` issues. Code-change issues skip to 7c directly.

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

Advance label to `status:reviewing`.

**7b. Final reviewer gate** (only if `status:reviewing`)

Spawn `reviewer` agent (experiments) or `code-reviewer` (code changes) in fresh
context. For experiments, reviewer sees only:
- The raw results
- The plan
- The clean-result issue body (NOT the analyzer's reasoning or critique history)

Reviewer verdict: PASS / CONCERNS / FAIL. Post as `<!-- epm:reviewer-verdict v1 -->`.

Transitions:
- **PASS** (experiments): promote clean-result from `clean-results:draft` → `clean-results`:
  ```
  gh issue edit <clean-result-N> --add-label clean-results --remove-label clean-results:draft
  uv run python scripts/gh_project.py set-status <clean-result-N> "Clean Results"
  ```
  Advance source to Step 8 (auto-complete).
- **CONCERNS:** same as PASS (non-blocking). Recorded on verdict comment.
- **FAIL:** clean-result stays `:draft`. Source back to `status:interpreting`.
  Analyzer revises with reviewer feedback.
- **PASS** (code changes): run tester inline (Step 7c), then Step 8.
- **FAIL** (code changes): back to `status:running` or `status:blocked`.

**7c. Tester (code changes only, inline)**

Only for `type:infra` / code-change issues, run inline after code-reviewer PASS.

1. Unit tests: `uv run pytest tests/ -v --tb=short`
2. Lint: `uv run ruff check . && uv run ruff format --check .`
3. Integration tests (conditional, if diff touches train/eval/orchestrate)
4. Coverage gap report (flags, does not auto-generate)

Post `<!-- epm:test-verdict v1 -->`. PASS → Step 8. FAIL (count < 3) → stay in
`status:reviewing`, implementer fixes. FAIL (count >= 3) → `status:blocked`.

### Step 8: Auto-complete (fires on reviewer PASS, or tester PASS for `type:infra`)

No user gate. The skill transitions the issue to Done automatically. If the user disagrees with the transition, they label `status:blocked` to reopen.

1. If code change: mark PR ready for review (not merge -- user merges).
2. Update `RESULTS.md` if the finding is headline-level (propose diff as comment
   `<!-- epm:results-md-diff v1 -->` -- do NOT auto-edit).
3. Update `eval_results/INDEX.md` with a new entry.
4. **Choose the Done variant from the issue's `type:*` label** (REQUIRED -- no guessing):
   - `type:experiment`                              -> `status:done-experiment` + Project Status `"Done (experiment)"`
   - `type:infra` / `type:analysis` / `type:survey` -> `status:done-impl`       + Project Status `"Done (impl)"`
   - If the issue has NO `type:*` label -> STOP, post an error comment asking the user to add one. Do NOT pick a default, and do NOT advance the label until fixed.
5. Apply the done label (remove `status:reviewing` or `status:testing` as applicable, add the done label chosen in step 4):
   ```
   gh issue edit <N> --add-label <done-label> --remove-label <prior-status>
   ```
6. Move the issue to the correct Done column on the Experiment Queue project board:
   ```
   # <status-name> is literally "Done (experiment)" or "Done (impl)" per step 4.
   uv run python scripts/gh_project.py set-status <N> "<status-name>"
   ```
7. Post final comment `<!-- epm:done v1 -->` summarizing:
   outcome, key numbers, what's confirmed/falsified, what's next, plus a link to the promoted clean-result issue (for experiments).
   Include the line `Moved to **<status-name>** on the project board.`
8. **LEAVE THE ISSUE OPEN.** Never call `gh issue close`. Done-ness lives on the
   project board, not in the issue's open/closed state. The only legitimate way
   for this skill to close an issue is a user-initiated duplicate / invalid / won't-fix
   triage -- never as the terminal state of a successful run.
9. Do NOT delete the worktree -- user decides when to clean up.
10. If `type:experiment`, proceed to Step 8b (follow-up proposer).

### Step 8b: Follow-up proposer (experiments only)

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
| `uploading` | no `epm:upload-verification` PASS | verifier not run or failed | re-run upload-verifier |
| `interpreting` | no `epm:interpretation` | analyzer not started | spawn analyzer |
| `interpreting` | `epm:interpretation` exists, no `epm:interp-critique` | critic not started | spawn interpretation-critic |
| `interpreting` | `epm:interp-critique` REVISE, round < 3 | revision needed | re-spawn analyzer with critique |
| `interpreting` | `epm:interp-critique` PASS or round >= 3 | ready for review | create clean-result, advance to `reviewing` |
| `reviewing` | missing `epm:reviewer-verdict` | reviewer not started | spawn reviewer |
| `reviewing` | `epm:reviewer-verdict` FAIL | interpretation needs more work | back to `interpreting` |

Without distinct labels for `uploading` / `interpreting` / `reviewing`, many of these
rows would be indistinguishable. That's why the state machine has them.

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

---
name: issue
description: >
  End-to-end GitHub-issue-driven workflow for experiments and code changes.
  Takes an issue number, parses state from labels + comment markers, and dispatches
  the next action (clarify -> adversarial-planner -> approval -> worktree +
  dispatch specialist -> preflight -> run -> analyzer -> reviewer -> test-verdict
  -> auto-complete). Reviewer PASS (or test-verdict PASS for code-change paths
  like type:infra / type:analysis / type:survey) auto-advances the issue to Done
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

The Experiment Queue project board has six Status columns. Mapping between `status:*`
labels (phase-authoritative) and project columns (glance-coarse):

| Project column | `status:*` label(s) | Meaning |
|---|---|---|
| **Todo** | `proposed`, `blocked`, or no `status:*` label | Not yet in the pipeline. User files issues here. |
| **Priority** | any (user-set) | Flagged by user as next-to-work. Pipeline doesn't auto-set this. |
| **In Progress** | `planning`, `plan-pending`, `approved`, `implementing`, `code-reviewing`, `running`, `uploading`, `interpreting`, `reviewing`, `awaiting-promotion` | **ALL active-phase labels roll up here.** The label tells you which phase. |
| **Draft Clean Results** | `clean-results:draft` (label, not a `status:*`) | Analyzer-created clean-result issues awaiting reviewer PASS + user promotion. |
| **Clean Results** | `clean-results` (label, not a `status:*`) | Published (promoted) clean-result issues. |
| **Done (experiment)** | `done-experiment` | Terminal, issue stays OPEN. |
| **Done (impl)** | `done-impl` | Terminal, issue stays OPEN. |

The skill moves the project status in exactly four places:
1. **Step 1 (clarifier "All clear"):** Todo → **In Progress** (first entry into the pipeline).
2. **Step 9a (analyzer creates clean-result):** the new clean-result issue (label `clean-results:draft`) → **Draft Clean Results**.
3. **Step 9b (user promotes draft → clean-results):** clean-result issue → **Clean Results**.
4. **Step 10 (auto-complete):** source issue → **Done (experiment)** or **Done (impl)**.

Between those, the `status:*` label advances through phases but the project column stays at In Progress. Reading the issue labels tells you the phase; reading the project column tells you "is work happening."

## Companion files

- `markers.md` -- comment marker taxonomy (source of truth for state parsing)
- `clarifier.md` -- clarifying-question prompts per issue type
- `templates/` -- plan / results / analysis comment body templates

Read these on first invocation of the skill in a session.

---

## Auto-continuation policy (REPEATED here for emphasis — see CLAUDE.md)

Six functional user-input gates total in this skill (see CLAUDE.md
"Auto-continuation policy"): Step 0b body, Step 0b type, Step 1 clarifier,
Step 2c plan-approval, Step 10c pod-termination, Step 10d merge-prompt.
Anywhere else that an assumption needs to be made, STATE the assumption
inline (one line, prefixed `Assumption:`) and proceed. Do NOT pause to ask.

**Exception:** the subagent halt conditions in CLAUDE.md "Subagent halt
conditions" — verdicts of BLOCKER/FAIL/FATAL/`needs-user` pause regardless.
The STATE-TO-`status:blocked` criteria (CLAUDE.md) also override
auto-continuation.

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
                                                     |--> status:implementing    <- experiment-implementer (type:experiment) OR implementer (type:infra)
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
                                                                                                                        |-- (user promotes) --> status:done-experiment (+ follow-up proposer)
                                                                                                          |-- FAIL --> status:interpreting (revise)
                                                                      |-- PASS + [type:infra/analysis/survey] --> test-verdict (inline) --> status:done-impl
```

Hot-fixes during `status:running` (experimenter agent): small in-line fixes
(<=10 lines, no logic change) get committed on the issue branch and the run
continues. Anything beyond that bar bounces back to `status:implementing` for
a fresh experiment-implementer + code-reviewer round before the experimenter
relaunches.

There is no user sign-off step. Reviewer PASS (or `epm:test-verdict` PASS for code-change paths) is the terminal gate; completion is automatic. If the user disagrees with a done transition, they label `status:blocked` to reopen it. The "test-verdict gate" runs inline inside this skill (Step 9c) — there is no separate `tester` agent.

**Active vs awaiting-user states:**

| State | Who's working | User action needed? |
|-------|---------------|---------------------|
| `proposed` | nobody (new) OR user (answering clarifier) | sometimes |
| `planning` | adversarial-planner + consistency-checker agents | no |
| `plan-pending` | nobody | **yes -- approve plan** |
| `approved` | skill (worktree + draft PR) | no |
| `implementing` | experiment-implementer (type:experiment) OR implementer (type:infra) | no |
| `code-reviewing` | code-reviewer agent | no |
| `running` | experimenter agent (pod ops + monitoring); type:experiment only | no |
| `uploading` | upload-verifier agent | no |
| `interpreting` | analyzer + interpretation-critic agents (iterative loop) | no |
| `reviewing` | reviewer / code-reviewer agent (final gate) | no |
| `awaiting-promotion` | nobody | **yes -- promote clean-result** |
| `blocked` | nobody (aborted or gate-skipped) | **yes -- triage** |
| `done-impl` | nobody (issue stays OPEN; Project Status="Done (impl)") | no |
| `done-experiment` | nobody (issue stays OPEN; Project Status="Done (experiment)") | no |

The two user-gated states in the lifecycle are `plan-pending` (plan approval) and `awaiting-promotion` (clean-result promotion). Everything between them is automatic, short of a `status:blocked` override.

Abort affordance: any state, user labels `status:blocked` -> skill posts abort
request, watcher kills run if one exists.

---

## Orchestration Procedure

When invoked, ALWAYS follow this order. Skip only what the state dictates.

**Chat title updates.** At every status transition (each `gh issue edit <N>
--add-label status:<X>` call), update the chat title for glanceable progress:
```
mcp__happy__change_title({ title: "/issue <N> — <new-status>" })
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
2. **Issue type** = the `type:*` label value (`experiment`, `infra`, `analysis`, `survey`)
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

3. **`type:*` missing →** infer from title cue, then confirm with the user:
   - Title prefix `Test:` / `Sweep:` / `Train:` → suggest `type:experiment`
   - Title prefix `Refactor:` / `Fix:` / `Add:` / `Migrate:` → suggest `type:infra`
   - Title prefix `Analyze:` / `Re-analyze:` → suggest `type:analysis`
   - Title prefix `Survey:` / `Read:` / `Lit review:` → suggest `type:survey`

   Use `AskUserQuestion` with the inferred option as `(Recommended)` first. Apply via
   `gh issue edit <N> --add-label type:<chosen>`. If the user is absent (e.g., autonomous
   loop), DO error and EXIT — the type label gates Step 7's Done variant and a guess
   here corrupts the project board.

4. **Other useful labels missing** (`compute:*`, `prio:*`):  do not block on these.
   `compute:*` will be set in the adversarial-planner's reproducibility card; `prio:*` is user-curated and
   never blocking.

   Note: existing `aim:*` labels remain on existing issues. This PR removes
   label-application from NEW workflow code; it does not delete labels from
   GitHub or strip them from existing issues.

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
  > (1) **Approve** — advance to implementation
  > (2) **Revise** \<notes\> — plan goes back to adversarial-planner
  > (3) **Defer** — exit now; re-invoke `/issue <N>` later

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
| `type:infra` / code change | `implementer` | `epm:results` |
| `type:analysis` | `analyzer` (re-analysis only) | `epm:analysis` |
| `type:survey` | `general-purpose` | `epm:results` |

Brief passed to the implementer:
- The plan (cached at `.claude/plans/issue-<N>.md`)
- Issue number + worktree path + branch name
- Code-review history if this is a revision round (`epm:code-review v<m>`)
- Required `report-back` fields
- **Instruction: work ONLY inside the worktree; never touch a pod; post
  progress as comments on issue #<N> via `gh issue comment`.**

Advance label to `status:implementing`. EXIT. Implementer runs autonomously.

### Step 5: Code review loop

Only if `status:implementing` and the appropriate implementation marker
(`epm:experiment-implementation v<n>` for experiments, `epm:results v<n>` for
infra) is present.

**5a. Spawn code-reviewer (fresh context).** The reviewer sees only the brief
this skill assembles. The brief MUST contain:

- `issue_number` — the GitHub issue (`<N>`)
- `target_marker_kind` — exactly one of `experiment-implementation` (for
  `type:experiment`) or `results` (for `type:infra` / `type:analysis` /
  `type:survey`). The reviewer reads the highest-version comment with this
  kind as the implementer's report.
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
  - `type:infra` / `type:analysis` / `type:survey` → skip pod phase, advance
    directly to `status:reviewing` (the inline test-verdict gate at Step 9c
    runs from there).
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

Label stays at `status:running`. EXIT. Experimenter runs autonomously. The
experimenter posts `epm:progress`, `epm:hot-fix` (if needed), and finally
`epm:results`.

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

   **Missing `failure_class`** (log-pattern fallback): scan the
   `epm:failure` body + linked log tail for the documented patterns in
   `.claude/skills/issue/failure_patterns.md`. If ANY infra pattern matches,
   route as `infra`. Otherwise, route as `code` (conservative — the
   implementer round catches more than the experimenter respawn round).

   The classifier patterns live in `.claude/skills/issue/failure_patterns.md`
   (single source of truth). Both this SKILL and the agent specs cross-reference
   that file. To add a new pattern, edit the file; no other change needed
   (allowed under §10 plan deviations).
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
- **FAIL** -> stays at `status:uploading`. Post clear list of what's missing with
  commands to fix. EXIT. Experimenter or user fixes, re-invokes `/issue <N>`.

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
  > When satisfied, promote it: `/clean-results promote <clean-result-N>`
  > Then re-invoke `/issue <N>` to auto-complete.

  EXIT. The user reviews the clean-result at their own pace and manually
  promotes it from `clean-results:draft` to `clean-results`.
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

Only for `type:infra` / `type:analysis` / `type:survey` issues — these arrive
here directly from Step 5 PASS, having skipped Steps 6–8 (no pod, no
interpretation). The code-review gate has already approved the diff; this
step verifies the test suite still passes.

There is **no `tester` agent**. The skill itself runs the project's test
suite directly and posts an `epm:test-verdict` marker with the result.

1. Unit tests: `uv run pytest tests/ -v --tb=short`
2. Lint: `uv run ruff check . && uv run ruff format --check .`
3. Integration tests (conditional, if diff touches train/eval/orchestrate)
4. Coverage gap report (flags, does not auto-generate)

Post `<!-- epm:test-verdict v1 -->`. PASS → Step 10. FAIL (count < 3) → stay in
`status:reviewing`, re-spawn implementer. FAIL (count >= 3) → `status:blocked`.

### Step 10: Auto-complete (fires after user promotes clean-result from `awaiting-promotion`, or `epm:test-verdict` PASS for code-change paths)

No user gate. The skill transitions the issue to Done automatically. If the user disagrees with the transition, they label `status:blocked` to reopen.

1. If code change: mark PR ready for review (not merge -- user merges).
2. Update `RESULTS.md` if the finding is headline-level (propose diff as comment
   `<!-- epm:results-md-diff v1 -->` -- do NOT auto-edit).
3. Update `eval_results/INDEX.md` with a new entry.
4. **Choose the Done variant from the issue's `type:*` label** (REQUIRED -- no guessing):
   - `type:experiment`                              -> `status:done-experiment` + Project Status `"Done (experiment)"`
   - `type:infra` / `type:analysis` / `type:survey` -> `status:done-impl`       + Project Status `"Done (impl)"`
   - If the issue has NO `type:*` label -> STOP, post an error comment asking the user to add one. Do NOT pick a default, and do NOT advance the label until fixed.
5. Apply the done label (remove `status:reviewing`, `status:awaiting-promotion`, or `status:testing` as applicable, add the done label chosen in step 4):
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
10. If `type:experiment`, proceed to Step 10b (follow-up proposer).

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
| `planning` | no `epm:plan` | planner was cancelled | re-run adversarial-planner |
| `plan-pending` | `epm:plan` exists | awaiting user approval | show plan URL, EXIT |
| `implementing` | no `epm:experiment-implementation` (or `epm:results` for infra) | implementer was cancelled | re-spawn implementer |
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

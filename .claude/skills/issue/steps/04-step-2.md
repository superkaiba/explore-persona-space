# Step 2: Adversarial planning

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

Only if status is `planning`.

Invoke the `adversarial-planner` skill with the task body + clarifier
output as the task. The skill runs planner -> fact-checker -> critic
-> revise internally.

**Minimum plan-review floor (binds even on `kind: infra` workflow-surface edits).**
The CLAUDE.md "Every new experiment MUST go through `/adversarial-planner`" bullet
carries a `"re-runs with different seeds, monitoring, syncing, bug fixes, or
explicit override skip it"` carve-out; that carve-out does NOT reach `kind: infra`
workflow-fix tasks. Evaluate the floor trigger off the TASK RECORD: `kind: infra`
AND (a `wf-fix` tag OR a title starting with one of
`task_workflow.WF_FIX_TITLE_PREFIXES` — `workflow-fix:` / `daily-fix:`). Step 10d's
merge fast path already does the `kind: infra` + `wf-fix`-tag half of this read
inline; the floor additionally accepts the title-prefix arm. NOT
`task_workflow.is_workflow_fix_session`, which tests only for a
`workflow_fix_target:` line in `body.md`: that is the narrower RECURSION-GUARD
trigger (`.claude/rules/workflow-fix-on-bug.md` § Recursion guard), so a
`workflow-fix:`-titled task filed without that Provenance line reads False and
would skip the very floor this block imposes. Even a
1-line prose edit runs, at minimum:

1. **Persist a plan version** via `uv run python scripts/task.py new-plan-version <N>` —
   plans are `tasks/<status>/<N>/plans/v{K}.md`, never `Write`-authored in-place
   (a `Write`-authored plan is invisible to `verify_plan.py --issue <N>` and to
   the dashboard). The plan may be a two-file prose edit's shape; it just has to
   exist as a versioned artifact.
2. **Run `verify_plan.py` and post `epm:plan-verify`** — the mechanical
   pre-pass (seconds, no agent spawn) per `adversarial-planner` SKILL.md
   § Phase 1.5.0. The marker records `verdict / n_fail / n_warn / failed check
   ids / plan version`; without it there is no durable proof the pre-pass ran.
3. **Spawn at minimum ONE Claude `critic`** (Methodology lens is the usual
   choice on a workflow-surface edit; Alternatives is the second candidate).
   Codex-only is not sufficient — the `code-reviewer` ensemble at Step 5
   already runs Claude+Codex on the diff, so the plan-review stage adds
   Claude by default; a Codex-only round can be added on top when the plan
   is trigger-dense. #1692's single critic returned REVISE with two Must-Fix
   findings on a same-class task; one critic is not nothing.

The floor is a MINIMUM — sessions that judge the full stack proportionate
(fact-checker + full 6-critic ensemble + consistency-checker) are unaffected.
The floor is what stops the floor from sinking to zero.

**Same-issue follow-up rounds inherit the floor** (`followups_running`, Step 9b) —
the same three legs bind on every follow-up round's plan revision. The cheap-band
auto-run (`0 < est_gpu_hours < 20`) does not bypass the floor.

**Recorded-skip contract.** Any leg SKIPPED below the full stack (fact-checker,
consistency-checker, additional critic lenses) is recorded in the `epm:plan`
note with a one-line reason, in the shape #1709 used:

> `"Bug-fix category (CLAUDE.md /adversarial-planner carve-out) — direct plan drafted for a 1-line SPECS widen + a 1-line pin-test update; no critic ensemble needed for a data-widen with pre-existing coverage."`

The recorded reason is auditable rather than invisible. The three floor legs
above are NEVER a recorded-skip candidate — a skip below the floor is a
substantive bug, not an audit entry. Recording the skip is the orchestrator's
duty (the same post that already carries `gpu_hours_total=<X>` per L1690).

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
planner against the amended Goal; NOT a critic round). (#922)

**Edit-success gate:** when the draft was produced or modified by a SCRIPTED
edit (the Step 2b/3 revise paths included), `&&`-chain edit → verify
(positive evidence the revised text is present — grep the draft, or a
non-empty diff vs the prior version) → the `new-plan-version` persist; the
edit step is the committed helper `uv run python scripts/plan_patch.py`
(anchor-normalized apply; fail-loud nearest-match diff on a missing/ambiguous
anchor — #1631; its printed `PLAN-PATCH APPLIED` line and `--verify-contains`
double as verify evidence; prefer ≥1-line distinctive anchors), never an
improvised per-turn anchor script; an edit-script failure aborts the persist
loudly, never `;`-chained
(`adversarial-planner` SKILL.md § Edit-success gate; #1565: a chained
persist landed v2 as an unmodified copy of v1 after the edit script died).

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

Subagent briefs always pass the symlink path in its ABSOLUTE canonical
main-checkout form — `$(uv run python scripts/task.py find <N>)/plans/plan.md`,
the same `PLAN_PATH` composed above — never a relative `tasks/...` path: a
subagent whose cwd is a worktree resolves a relative path against its
`tasks/` tree frozen at the branch-cut base commit and reads the pre-cut
version with no error (#2422). Self-containedness makes the SYMLINK sound
against amendment-thinness only, NOT against worktree-freeze: every
persisted version is SELF-CONTAINED by contract — `new-plan-version`
refuses thin amendment-shaped deltas (#2255). After a deliberate `--allow-amendment`
persist the symlink points at a PARTIAL document, so every brief must hand
BOTH paths (the amendment `v<K>.md` AND its base `v<J>.md`);
`verify_plan.py --issue` composes them automatically.

**Worktree-safe task-state paths (#2422) — the full contract for every
brief this skill family composes.** Resolve at compose time:
`TASK_DIR="$(uv run python "$REPO_ROOT"/scripts/task.py find <N>)"`
(`REPO_ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"`
— the prefix form runs the MAIN checkout's `task.py`, never a
possibly-stale worktree copy). Hand `$TASK_DIR/plans/plan.md` and, when
the task has one, `$TASK_DIR/artifacts/planned_manifest.json`. STATE
`plan_version=v<K>` in the brief — the extensionless form of
`readlink "$TASK_DIR/plans/plan.md"` (readlink returns `v<K>.md`; strip
`.md` exactly once at the comparison site). At read time the subagent
re-runs the readlink and FAILS LOUD on a `plan_version=` mismatch (a
revision landed after compose ⇒ the round grades a superseded document)
— never proceeds silently. Degradations: a brief with no `plan_version=`
(legacy) still binds the canonical absolute path; a 404 (status moved
mid-round) re-runs `task.py find <N>`.

Also include estimated cost prominently in the `epm:plan` note, with a
machine-readable token (`gpu_hours_total=<number>`) the Step 2c auto-approve
gate parses, e.g.

> **Cost gate:** estimated 12 GPU-hours on 4× H100 (`gpu_hours_total=12`). Reply `approve` to dispatch.

**Cost confirmation does NOT pre-provision the pod.** Do NOT call
`pod.py provision` until the user replies `approve` (i.e., the Step 2c
plan-approval gate fires "Approve" and the task moves to
`status:approved`). Posting the cost note and then provisioning "to
save time" creates an orphan pod if the session exits before approval
(#406). If the session
must exit at this gate, post `epm:awaiting-spend-approval v1` and
ensure NO pod exists yet — the stale-pod audit cannot reap a pod the
workflow provisioned speculatively before approval.

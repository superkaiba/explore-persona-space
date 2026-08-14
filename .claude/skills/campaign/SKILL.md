---
name: campaign
description: >
  Question-level autonomous campaign runner (task #586) — the
  orchestration layer ABOVE /issue. Takes a `kind: campaign` task number
  pinned to ONE docs/open_questions.md anchor and loops: ingest landed
  child clean-results into a writable world model → check stop criteria →
  propose next experiments → file + spawn children as ordinary
  `/issue <child> --auto` sessions (full adversarial-planner + critic
  stack unchanged) → repeat, maximizing parallelism across
  shared-nothing arms. Executes ONLY from status `approved` (the user
  reviews the `## Campaign Brief` first — workflow.yaml §
  gates.campaign_brief_approval); budget/stop enforcement lives in
  `campaign_state.py` + the watcher's campaign pass, not in the prompt.
  Re-entrant: every invocation runs one decision round and exits; the
  /campaign-tick cron re-drives it.
user_invocable: true
---

# /campaign — question-level autonomous campaign runner

One required argument: the campaign task number `<N>` (the integer naming
`tasks/<status>/<N>/`, `kind: campaign`).

This skill runs in a DEDICATED autonomous Happy session spawned via
`uv run python scripts/spawn_session.py spawn-campaign --issue <N>`
(env: `EPM_AUTONOMOUS_SESSION=1`, `EPM_CAMPAIGN_SESSION=1`). It is the
campaign's only driver. Children are ordinary `kind: experiment` tasks
executed by their own `/issue <child>` sessions with the full
planner / critic / analyzer stack unchanged — this skill NEVER runs an
experiment itself, never touches a pod, and never reads a child's raw
results: it ingests only the CRITIC-GATED clean-result body at
`awaiting_promotion` / `completed`.

Design grounding (2025-26 autonomous-research literature): invest in the
proposal operator, not tree search; keep a writable claim→evidence world
model; the deciding eval lives outside the agent (critic-gated
clean-results); parallel arms share state through the world model and are
designed shared-nothing so they can run concurrently; humans own the
problem + rubric (brief IN, digest + final report OUT).

## When NOT to use a campaign (routing check)

A campaign is for **adaptive open-question search** — the next experiment
depends on what the last ones found, so the proposal operator earns its
keep. It is the WRONG tool when the work is a **fully pre-specified phase
DAG** whose every step is known up front, ESPECIALLY when the children
share a sequential data dependency through a store (a shared activation /
base-model store, a staged extraction pipeline). In that case the campaign
machinery (writable world model, propose-next-on-dry, parallel
shared-nothing arms) fights the work: there is nothing to adaptively
propose, and the shared-store dependency forbids the parallelism a campaign
assumes. Route a fixed DAG with hard inter-step data dependencies as a
SEQUENCE of ordinary `/issue <child>` tasks (or one multi-phase `/issue`),
not a campaign. (Decision recorded after a 2026-06-24 over-reach: a fixed
5-task extraction DAG sharing a base-model store was briefly recommended as
a campaign, then walked back.)

## State surfaces

- `tasks/<status>/<N>/artifacts/campaign-state.json` — machine state
  (budget, limits, stop, experiment DAG). Read/write ONLY via
  `explore_persona_space.campaign_state` (atomic, schema-validated).
- `tasks/<status>/<N>/artifacts/world-model.md` — claim → `#task`
  evidence ledger, appended at every ingest. `docs/open_questions.md`
  stays user-gated read-only; the final report PROPOSES a diff, never
  applies it.
- Commit both artifacts by explicit path after every mutation
  (concurrent-committer rules: never `git add -A`).

Resolve the artifacts directory via `uv run python scripts/task.py find
<N>` — never a hand-built `tasks/...` path.

## No user asks, ever

This skill never calls AskUserQuestion — the ONE human gate is upstream:
a campaign executes only from status `approved`, after the user reviews
the `## Campaign Brief` and runs `task.py set-status <N> approved`
(workflow.yaml § gates.campaign_brief_approval). Everything after that
gate is autonomous: decisions follow the autonomous carve-out (pick max
info-gain per GPU-hour toward the question, post `Decision: <X>` in the
relevant marker note, continue). Gates OUT are the daily
`epm:campaign-digest` and the final report; per-child promotion stays
user-only but does NOT block the campaign.

## Step 0 — validate + arm (idempotent)

1. `uv run python scripts/task.py view <N> --json`. Verify
   `kind: campaign` — anything else: print one line ("not a campaign
   task") and EXIT.
2. Verify status:
   - `proposed` / `planning` / `plan_pending` → EXIT with ONE line: the
     user must review the `## Campaign Brief` and run
     `uv run python scripts/task.py set-status <N> approved`
     (workflow.yaml § gates.campaign_brief_approval). Do not loop, do
     not arm the cron.
   - `approved` → first entry; continue.
   - `running` → re-entry (tick re-drive or watcher respawn); skip to
     Step 1 after re-arming the cron (sub-steps 5-7 are idempotent).
   - `completed` / `archived` / `blocked` → CRON-TEARDOWN (fresh
     `CronList` → `CronDelete` every job in the two-leg match set: the
     recurring `/campaign-tick <N>` tick + stray one-shot `/campaign <N>`
     wakeups; not-found = success; § CRON-TEARDOWN in `/campaign-tick`),
     one line, EXIT.
3. Verify the body has a `## Campaign Brief` H2 carrying: the question
   anchor (`q:<slug>` into `docs/open_questions.md`), the hypothesis
   list, the initial experiment table (columns `id | title | hypothesis
   | depends_on | gpu_hours_est`), and optional frontmatter `campaign:`
   budget overrides. A malformed brief → post `epm:failure v1`
   (`failure_class: data`, naming the missing piece), set status
   `blocked`, EXIT — fail loud, never guess a DAG.
4. Initialize state if absent (`init_state_from_brief` parses the brief
   and persists atomically):

   ```bash
   uv run python -c "
   from explore_persona_space import campaign_state, task_workflow
   t = task_workflow.get_task(<N>)
   try:
       campaign_state.load_state(<N>)
       print('state: already initialized')
   except FileNotFoundError:
       campaign_state.init_state_from_brief(<N>, t['frontmatter'], t['body'])
       print('state: initialized from brief')
   "
   ```

   Commit `artifacts/campaign-state.json` by explicit path.

   Budget/limit seeding is single-pathed through `init_state_from_brief`
   with fixed precedence: frontmatter `campaign:` overrides (the
   user-reviewed brief wins) > caps recorded in
   `~/.eps-autonomous/campaign-<N>.json` by `spawn-campaign`'s CLI flags
   (`--budget-gpu-hours` / `--max-concurrent` / `--per-child-cap`) >
   `campaign_state` module defaults. The state file is the ONLY budget
   enforcement surface after init.
5. Set status → `running` (`task.py set-status <N> running`).
6. Register this session for the watcher's campaign pass (idempotent —
   safe on re-entry):
   `uv run python scripts/spawn_session.py register-current --issue <N>
   --mode campaign`.
7. Arm the recurring tick (idempotent ARM-GUARD: `CronList` first, skip
   if a job with prompt exactly `"/campaign-tick <N>"` exists):
   `CronCreate("*/45 * * * *", prompt="/campaign-tick <N>",
   recurring=True, durable=False)`. (45 min as of 2026-06-12 — the
   10-min watcher carries fast detection; the tick is the in-session
   re-driver of last resort.) Prevention ban (the #980 class): a
   `/campaign` session must NEVER schedule its own re-drive — no
   `ScheduleWakeup` wakeup and no `CronCreate` one-shot, regardless of
   prompt shape; the `/campaign-tick <N>` cron above is the ONLY
   sanctioned self-wake (a one-shot wakeup may not be enumerable at
   teardown time, and one that fires after terminal re-drives a
   completed campaign).
8. Post `epm:campaign-started v1` (anchor, budget, limits, DAG size).
   Then fall through to Step 1.

## Step 1 — decision round (the core loop body)

Re-entered on every tick re-drive and on every child-landing wake. Each
round is bounded: reconcile → stop-check → extend-if-dry → file+spawn →
digest → EXIT (the session idles between rounds; the cron is the driver).

### 1.1 Reconcile children → ingest landed results

`uv run python scripts/task.py list-children <N> --json`.

**Orphan-child adoption (crash-recovery cross-check, FIRST):** any
`list-children` row whose id does NOT match a known `child_task` in state
is an orphan from a crash between `task.py new` and the state save. Read
its body for the embedded `<!-- campaign_experiment_id: <id> -->` line
(Step 1.4 stamps it into every child) and ADOPT it into that DAG row:
set `child_task: <orphan id>`, status from the child's actual status
(`filed` if no session is driving it yet — re-spawn it in 1.4), and add
its `gpu_hours_est` to `budget.gpu_hours_committed` if the row was still
`planned` (the crash window lost the commit). An orphan with NO embedded
experiment id (or one not in the DAG) → post `epm:failure v1`
(`failure_class: data`, naming the orphan child id), set status
`blocked`, EXIT — never guess which arm a child belongs to.

Then, for each experiment row in state with status `filed` / `running` /
`landed` / `waiting-user`, compare its `child_task`'s current status:

- Child reached `awaiting_promotion` or `completed` → **ingest**:
  1. Read the child's clean-result body — title claim + confidence tag
     + the findings-skim ONLY (the critic-gated artifact; never raw
     completions, never `epm:interpretation` drafts). For a v3 child
     (sentinel `<!-- clean-result-v3 -->`) the findings-skim is
     `## Takeaways` + `## Findings`; for a v2/legacy child it is the
     `## TL;DR` block — branch on the sentinel.
  2. Record `headline` + `confidence` on the experiment row.
  3. Append a claim→evidence row to `artifacts/world-model.md`:
     `| <claim> | #<child> | <confidence> | <date> |`.
  4. Set `belief_shift: yes|no` — did this result change the campaign's
     working belief relative to the world model's prior state? `yes` →
     reset `stop.dry_counter` to 0; `no` → increment it. THEN update
     `stop.current_confidence` — the CAMPAIGN-LEVEL confidence in the
     world model's current answer to the question (judged across the
     accumulated evidence ledger, NOT copied from this child's
     per-claim tag; LOW/MODERATE/HIGH/DETERMINATE). This field is what
     the confidence-target stop criterion reads.
  5. Reconcile committed hours (IDEMPOTENT recipe — used here and at a
     `waiting-user` resume): if the child's approved plan recorded
     actual GPU-hours (`epm:plan` marker / plan frontmatter
     `gpu_hours_total`) and `plan_hours != row.gpu_hours_est`, adjust
     `budget.gpu_hours_committed` by `(plan_hours − row.gpu_hours_est)`
     AND set `row.gpu_hours_est = plan_hours`. After any reconcile the
     row's est always equals its committed contribution, so re-running
     the recipe (e.g. at ingest after a `waiting-user` resume already
     applied it) is a no-op — never double-add. Note the original
     est-vs-plan gap for the next digest.
  6. Mark the row `ingested`, save state, commit both artifacts by
     explicit path, post `epm:campaign-child-ingested v1`.
- Child `blocked` → leave one respawn attempt to the normal watcher
  path this round; if STILL `blocked` on the next round, mark the row
  `abandoned` and RELEASE its committed hours: first run the idempotent
  reconcile recipe (step 5) so `row.gpu_hours_est` equals the row's
  committed contribution, then subtract `row.gpu_hours_est` from
  `budget.gpu_hours_committed`, note the abandonment + any
  already-burned pod hours in the world model, save + commit. (Released
  hours are an optimistic refund — a pod that already burned time is
  gone; the digest surfaces the gap.)
- Child parked at `plan_pending` (the autonomous plan-gate parked it —
  GPU-hour-blind as of #1771, so the only autonomous park cause is a
  missing/unparseable GPU-hour estimate) → mark the row `waiting-user`;
  it does NOT occupy a
  concurrency slot; surface it in the next digest. If the user later
  approves, the row flips back to `running` at the next reconcile —
  and RECONCILE its committed hours to the approved plan's
  `gpu_hours_total` (the park means the filing-time estimate was
  missing/unusable, so the row's committed figure needs truing-up
  against the approved plan) using the SAME idempotent recipe as ingest
  step 5: adjust `budget.gpu_hours_committed` by
  `(plan_hours − row.gpu_hours_est)`, then set `row.gpu_hours_est =
  plan_hours` — a later re-run at ingest is then a no-op. Surface the
  est-vs-plan gap in the next digest.
- Child still in flight (any ACTIVE status) → row stays `running`.

### 1.2 Stop-check

```bash
uv run python -c "
from explore_persona_space import campaign_state, task_workflow
state = campaign_state.load_state(<N>)
tags = task_workflow.get_task(<N>)['frontmatter'].get('tags') or []
print(campaign_state.check_stop(state, user_stop='campaign-stop' in tags))
"
```

The user-stop signal is the `campaign-stop` tag
(`task.py add-tag <N> campaign-stop`). On `(True, reason)` → record
`stop.stopped = true`, `stop.stop_reason = reason`, save, go to Step 2.

### 1.3 Extend the DAG if dry

If `ready_experiments(state)` is empty AND `open_slots(state) > 0` AND
the stop-check did not fire: spawn ONE fresh-context proposal agent
(`Agent(subagent_type="general-purpose")`) whose brief contains PATHS to
(never inlined bodies of): the campaign task body, `world-model.md`, the
`docs/open_questions.md` anchor section, and the clean-result titles of
the parent + all ingested children. It returns 1-4 NEW experiments in
the DAG row format (`id | title | hypothesis | depends_on |
gpu_hours_est`), each justified by info-gain-per-GPU-hour and designed
for INDEPENDENCE (shared-nothing arms: own pod, own adapters, no shared
mutable artifacts) wherever possible so they can run concurrently.
Append the rows to state (validate ids unique, deps resolve), save +
commit. A proposal round that returns zero viable experiments increments
`stop.dry_counter` (it will trip the dry-limit stop within
`dry_limit` rounds — no infinite proposal loop).

### 1.4 File + spawn (parallelize aggressively; state save BEFORE spawn)

Select from `ready_experiments(state)`, in order, at most
`open_slots(state)` rows, taking each row only while the cumulative
selected `gpu_hours_est` stays `<= budget_headroom(state)` — never
commit hours past the headroom. For each selected experiment row, in
THIS order (the state save sits between filing and spawning so a crash
in any window cannot duplicate a child or leak committed hours):

1. File the child:
   `uv run python scripts/task.py new --workflow v1 --kind experiment --parent <N>
   --title "<row title>" --goal "<one-sentence goal from the
   hypothesis>" --body-file <tmp>` — campaign children are pinned to the v1
   pipeline explicitly, so the future v2 default-flip
   (env `EPM_DEFAULT_WORKFLOW` / `task.py new` default) never leaks into
   campaign children (plan Assumption 2). The body OPENS with the fixed line
   `<!-- campaign_experiment_id: <row id> -->` (the deterministic
   reconcile key for orphan adoption, Step 1.1) and carries the
   hypothesis, the single-variable design sketch, `gpu_hours_est`, and
   a pointer to the campaign world model
   (`tasks/<status>/<N>/artifacts/world-model.md` via the dashboard
   path, plus the question anchor).
2. IMMEDIATELY persist the filing — BEFORE any spawn: update the row to
   `status: filed`, `child_task: <child>`,
   `budget.gpu_hours_committed += gpu_hours_est`; save state and commit
   the state artifact by explicit path. (A crash after `task.py new`
   but before this save is recovered by 1.1's orphan adoption via the
   embedded experiment id; a crash after this save merely leaves a
   `filed` row whose session the next round spawns — no duplicate, no
   leaked hours.)
3. Spawn its autonomous session:
   `uv run python scripts/spawn_session.py spawn-issue --issue <child>
   --auto --auto-approve-gpu-hours <per_child_gpu_hours_cap>` (the
   `--auto-approve-gpu-hours` value is threaded to the child for
   provenance / bookkeeping only — DEPRECATED no-op for plan approval
   as of #1771's GPU-hour-blind gate; the campaign-level GPU-hour
   budget in `artifacts/campaign-state.json` is the sole campaign-side
   limiter).
   Stagger successive spawns by a few seconds (429 token-pacing).
4. Flip the row `filed` → `running`; save + commit.
5. Post `epm:campaign-child-spawned v1` (child id, experiment id,
   committed hours, remaining headroom, open slots).

A `filed` row with a `child_task` but no live session (observed at the
next reconcile) is the crash-between-save-and-spawn case: re-run steps
3-5 for it — `task.py new` is NEVER re-run for a row that already has a
`child_task`.

The full `/issue` pipeline (clarifier → adversarial-planner → critic
ensemble → implementer → experimenter → analyzer → clean-result-critic)
runs unchanged inside each child session. The campaign never bypasses
it.

### 1.5 Digest

If `last_digest_at` is null or older than 24h: post
`epm:campaign-digest v1` (budget spent/committed/total in GPU-hours,
children table, current working belief + `stop.current_confidence`, next
planned arms, any `waiting-user` rows, and any est-vs-approved-plan
GPU-hour gaps noted at 1.1's reconciliations) + a one-line
`PushNotification` (soft-fail — swallow errors). Set `last_digest_at`,
save + commit. Then EXIT the round (the tick cron re-drives).

## Step 2 — finalize

1. Post `epm:campaign-stopped v1` with `stop_reason`.
2. Write the final campaign report into the task body
   (`task.py set-body <N> --file <report> --snapshot` — snapshots the
   original brief to `original-body.md`): the question, the belief
   trajectory (initial hypothesis set → final working belief +
   confidence), the per-child table (id / title / headline /
   confidence / GPU-hours), the claim→evidence ledger, and a PROPOSED
   `docs/open_questions.md` diff in a fenced block — NOT applied;
   living-docs mutations stay user-gated, unchanged.
3. Set status `completed`.
4. CRON-TEARDOWN (fresh `CronList` → `CronDelete` every job in the
   two-leg match set: the recurring `/campaign-tick <N>` tick +
   stray one-shot `/campaign <N>` wakeups; not-found = success;
   § CRON-TEARDOWN in `/campaign-tick`).
5. `PushNotification` one-liner (campaign done + stop reason), then
   EXIT. The watcher's campaign pass reaps `campaign-<N>.json` on its
   next tick.

## Failure posture

- Any unrecoverable error (state file corrupt, brief unparseable,
  `task.py` mutations failing repeatedly) → post `epm:failure v1` with
  the appropriate `failure_class`, set status `blocked`, EXIT. The
  watcher GCs the registry entry at `blocked`; the user triages.
- Child failures are NOT campaign failures: a blocked child is retried
  once by the watcher, then abandoned (Step 1.1) and the campaign
  continues toward the question with the remaining budget.
- Budget discipline is structural: Step 1.4 cannot commit past
  `budget_headroom`, `check_stop` halts the loop at exhaustion, and the
  watcher's budget backstop alerts if committed ever exceeds total
  (harness-side circuit breaker). All caps are GPU-hours, never dollars.

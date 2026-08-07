# Step 10b: Follow-up proposer (experiments only — runs ∥ Step 10c)

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

**Parallel spawn with Step 10c + 10c-bis.** Steps 10b, 10c, and 10c-bis
keep their numbering and their per-step semantics, but their agents are
spawned CONCURRENTLY: evaluate all three steps' skip conditions first
(10b's autonomous-mode short-circuit below; 10c's kind / `relates_to`
skips; 10c-bis's kind / finding-bearing skips), then spawn the SURVIVING
agents in ONE message (one Agent call per agent that did not skip,
staggered a few seconds apart per the CLAUDE.md 429 guidance), each
preceded by its own `stage-dispatch` breadcrumb (Step 9 entry-guard
convention; for the new agent `stage=related-work round=1
subagent=related-work-finder`). Each spawned agent reads the completed
clean-result; their outputs are independent (follow-up proposals vs a
proposed open-questions diff vs a proposed Related-findings positioning
note). Process each return per its own step text, and JOIN every spawned
agent — `epm:follow-ups v1` posted (or 10b skipped) AND the 10c proposal
handled (gate raised / parked per 10c, or 10c skipped) AND the 10c-bis
proposal handled (`related_work_positioning` gate raised / parked per
10c-bis, or 10c-bis skipped) — before entering Step 10d. The
`living_docs_update` and `related_work_positioning` gates, all markers,
and the user-confirmation semantics are unchanged by the scheduling; only
the spawn scheduling is shared. If a step's skip condition fires, spawn
only the other agents.

**Per-kind membership of this batch (no concurrency assumption beyond
this list):**

- `experiment` (entered from the Step 10 step-10 experiment branch) →
  all THREE agents spawn (`follow-up-proposer` + `living-docs-updater` +
  `related-work-finder`), subject to each step's own skip conditions.
- `analysis` with a measured finding (entered from the Step 10 step-10
  analysis branch) → ONLY `related-work-finder` spawns. Steps 10b and 10c
  are EXPERIMENTS-ONLY and are NOT entered for an analysis task, so this
  "parallel batch" degenerates to a single Agent call — do NOT spawn
  `follow-up-proposer` or `living-docs-updater`. (A future reader: this is
  the one-agent case; the word "batch" here does not imply three spawns.)
- `infra` / `batch` / `survey`, or `analysis` with no `## Results` → none
  of the three spawn (this batch is not entered at all; proceed to Step
  10d).

Auto-fires after `completed` for `experiment` tasks. Spawn the
`follow-up-proposer` agent with:

**Proposer-already-ran short-circuit:** if an `epm:follow-ups v1` marker
for the most recent park is ALREADY present on the parent's
`events.jsonl`, the proposer ran at Step 9b — either the autonomous
follow-up auto-spawn block (an `epm:follow-ups-autospawned v1` marker is
also present) OR the cheap-band block (block C0, which fires in
interactive sessions too). SKIP re-spawning the
proposer here — it would duplicate the proposal list and is unnecessary.
The `epm:follow-ups v1` posted at Step 9b is still the canonical list
for the user; any `auto_run: no` / cap-skipped / fail-safe-skipped
proposals from that marker remain on the table for the user to pick from
manually post-promotion (route the pick by `question_relation` as
below). An interactive task that landed here with NO `epm:follow-ups v1`
(no cheap-band candidate ever existed, so block C0 was a no-op that
posted nothing — see C4) runs the proposer here as normal.
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

**Run the value-critique (redundancy screen) before surfacing the picks
— subroutine VC (Step 9b § Follow-up value-critique).** Run VC over the
`epm:follow-ups v1` set (idempotent — if Step 9b's C0a / autonomous block
already ran it this park, reuse the merged verdict; the
proposer-already-ran short-circuit above means VC's prior verdict is
usually already present). VC parks every `redundant` proposal at
`on_hold` (`epm:followup-parked-redundant v1`, revivable) and hands back
only the `not-redundant` survivors. **Surface ONLY the `not-redundant`
survivors to the user** for picking; for each parked-redundant proposal,
state ONE chat line naming the duplicate + the `on_hold` task id so the
user knows it was saved (not dropped) and can revive it. The user's pick
is then routed by `question_relation` as below — a `redundant` proposal
is never offered as a pick (it is already parked on_hold).

**Route the user's pick by `question_relation`** — the litmus is the
Takeaways test: *would the result rewrite THIS issue's `## Takeaways`?*
yes → `same` (same-issue loop), no → `substantially-different` (child).
Changing method/dose/panel/seeds/eval-surface/prompt-bank or adding a
control on the same question is `same`; only a result that would move the
task's `## Goal` / open-questions anchor is `substantially-different`.
(Untagged proposals: the treat-as-`substantially-different` fallback
applies only when the `epm:follow-ups v1` marker was posted before
2026-06-09; on a newer marker the missing tag is a proposer-contract
violation — classify the picked proposal yourself against the
Takeaways litmus + `.claude/agents/follow-up-proposer.md` §
"question_relation tag — criteria" and note the violation in the
resulting `epm:followup-scope v1` / child-creation marker):

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
handoff. (#461)

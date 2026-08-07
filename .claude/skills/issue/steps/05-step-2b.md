# Step 2b: Consistency checker (runs ∥ the Phase 2 critic ensemble)

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

The `consistency-checker` no longer waits for an APPROVE-rated plan: it
needs only the drafted plan + the parent recipe — the same input the
Phase 2 critics get, with no dependency on their verdicts — so spawn it
CONCURRENTLY with the /adversarial-planner Phase 2 critic ensemble
(same spawn batch as the 6 critics, staggered a few seconds apart per
the CLAUDE.md 429 guidance; see adversarial-planner SKILL.md Phase 2).
Its findings are UNIONED with the critics' blockers into the single
Phase 3 revise round — one revision round covers both, instead of two
serial bounce rounds. Verdict semantics and the `epm:consistency v1`
marker are unchanged; only the scheduling moved. Its verdict must still
be folded in BEFORE posting the plan as `epm:plan`. It receives:
- The drafted plan
- Related tasks (cited in the plan's prior work, parent task, or
  near-duplicate clean-result task)
- The `epm:plan` and `epm:results` markers from those related tasks
  (read via `task.py latest-marker` + `task.py view --json`)

**Skipped branch (parentless non-experiment, added #1732).** When the
invoking task is `kind: infra | batch | survey` with no `parent_id`
AND no unrun `epm:followup-scope v1` marker on this issue, the
`consistency-checker` spawn is SKIPPED entirely (no experimental recipe
for the five checks below to bind to; see
`.claude/agents/consistency-checker.md` § Rules). The orchestrator
posts the marker VM-side —
`uv run python scripts/task.py post-marker <N> epm:consistency --note '<PASS-skipped body>'` —
where the note body is an `<!-- epm:consistency v1 -->` block with
`**Verdict: PASS**` whose first line reads
`Skipped: kind:<X>, no parent experiment` (X = the actual `kind`:
`infra`, `batch`, or `survey`) and whose rows read
`N/A — <reason>`. The plan-approval gate then proceeds as if PASS.
`kind: experiment` with no parent runs the checker against the standard
baseline (Qwen-2.5-7B + standard eval suite) as today; same-issue
follow-ups still diff against the issue's own prior run.

The consistency checker verifies:

| Check | Violation action |
|-------|-----------------|
| Single variable change from parent | BLOCK: list all differences |
| Same baseline model/checkpoint | WARN: flag, require justification |
| Same eval suite | BLOCK: incompatible evals make comparison meaningless |
| Same seeds or superset | WARN: disjoint seeds reduce comparability |
| Same data version/hash | WARN: different data confounds results |

Post `epm:consistency v1`. On BLOCK, the finding joins the Phase 3
revise round's UNION — critic Must-Fix items + consistency BLOCKs,
addressed together by the planner in ONE revision round (consistency
re-checks after revision keep the existing loop cap, max 2 rounds). On
WARN, append warnings to the `epm:plan` event note. On PASS, proceed
normally. The `plan_pending` flip below still happens only AFTER the
checker's FINAL verdict is folded in (adversarial-planner SKILL.md
§ Park order) — never on its interim ack.

**Edit-locus WARN → merge-hold record (#1757).** When the WARN body names a
same-file EDIT-LOCUS conflict with a live sibling task at status `reviewing`
or later, ALSO post one `epm:progress` note per named sibling (idempotent
per (sibling, path) — grep the events file first, the
`followup-parked-by-cap` convention; reuse `epm:progress`, never a new
marker kind):

```bash
uv run python scripts/task.py post-marker <N> epm:progress \
  --note 'merge-hold-candidate sibling=<M> path=<file> source=consistency-warn — Step 10d Guard 5 orders this task landing behind sibling #<M> (bounded, one 45-min gate cycle) and pre-resolves the predicted conflict in-worktree (#1757)'
```

Auto-continue — never a gate, never blocks planning; a WARN naming no live
sibling (or a sibling below `reviewing`) records nothing. The trigger is
mechanical: the WARN body must BOTH name a concrete sibling task id at
status `reviewing` or later AND name a same-file/same-region edit conflict
(the checker's own vocabulary: "edit locus", "same file", "same
block/region", a TEXTUAL/EDIT-LOCUS conflict finding). A generic WARN (seed
drift, baseline caveat) records nothing.

Then post the plan as `epm:plan v1` with the consistency results
appended.

Move the task to `plan_pending` **through the code-enforced autonomous
plan-gate** — pass the plan's total GPU-hours so `task.py` itself makes the
auto-approve / park / interactive decision (it reads `EPM_AUTONOMOUS_SESSION`
+ `EPM_PLAN_AUTOAPPROVE_GPU_HOURS` from the env). This is what makes
autonomous auto-approval deterministic instead of dependent on the
orchestrator obeying the Step 2c prose:

```bash
uv run python scripts/task.py set-status <N> plan_pending \
  --auto-approve-if-autonomous --gpu-hours <X> \
  --note "Plan v1 ready for approval; consistency PASS."
```

`<X>` is the plan's `Estimated GPU-hours (total)` (the same number embedded
as `gpu_hours_total=<X>` in the `epm:plan` note). **Omit `--gpu-hours` only
if the total is genuinely unknown** — a blank estimate fail-safes to a park,
never an auto-approve. The command prints a `PLAN_GATE_DECISION: <decision>`
line (`auto_approved` | `parked_over_cap` | `interactive_pending`) that
Step 2c branches on; for `auto_approved` it has already flipped the status to
`approved` and posted `epm:plan-approved`, and for `parked_over_cap` it has
already posted `epm:awaiting-spend-approval`.

> **Same-issue follow-up round?** At `followups_running` this same command is
> safe: `task.py` fires the gate decision + markers but HOLDS the status in
> place (status-hold rule, Step 9b § Same-issue follow-up loop step 3) and
> appends `(followups_running hold: status unchanged)` to the decision line.

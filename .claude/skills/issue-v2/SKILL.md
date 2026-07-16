---
name: issue-v2
description: >
  The v2 per-task orchestrator for `workflow: v2` tasks. Invoked by the v1
  /issue dispatcher (Step 0) when a task's frontmatter carries `workflow: v2`.
  Same runtime as v1 (Happy + tmux + bg-Bash poll + tick-cron); the changes are
  (1) a report pipeline that REPLACES the interpretation back-half — agents
  never interpret results, they produce a fixed-structure report (Motivation /
  Methodology (metrics embedded) / Results-as-plots per the official template,
  `.claude/skills/issue-v2/report-template.md`) verified for accuracy +
  completeness, and Thomas alone writes the claims (title, TLDR, per-result
  Takeaways, Next steps); (2) approve-first-then-critique front half
  with specialized critic panels + a plan-revision log; (3) upload-by-default
  with overflow rerouting. Compact by design: everything unchanged from v1
  defers to `.claude/skills/issue/SKILL.md` by section name.
user_invocable: true
---

# Issue-Driven Workflow (v2)

This is the v2 per-task orchestrator. It is reached ONLY through the thin
dispatcher at the top of `.claude/skills/issue/SKILL.md` Step 0: when a task's
`workflow` frontmatter is `v2`, v1 delegates the whole lifecycle here and exits.
`workflow: v1` (or absent) stays on the v1 path, byte-for-byte.

**The runtime is unchanged.** Happy-spawned sessions already run in tmux; the
turn-ending + bg-Bash + adaptive-poll + tick-cron execution model is
incident-tested and stays. v2 adds a visibility layer and changes WHO INTERPRETS
+ how many review loops run — not the execution model. Wherever v2 behaves
identically to v1, this file says **"`.claude/skills/issue/SKILL.md` § `<name>`
applies verbatim"** rather than duplicating the (8K-line) v1 text.

---

## What changes from v1 (the map)

| Area | v1 | v2 |
|---|---|---|
| Who interprets results | analyzer writes findings + confidence; interpretation-critic + clean-result-critic gate it | **No agent interprets.** methodology-writer (REPORT MODE, findings-blind) + plotter (data-only) produce a fixed-structure report; methodology-critic + report-verifier gate it for accuracy + completeness. **Thomas alone writes the TLDR + Next steps.** |
| Gate order | critique (adversarial-planner Phase 2) runs BEFORE the user approves | **approve-first:** plan + manifest → user approves → specialized critic panel hardens it POST-approval (Step 3), auto-revises, logs every round |
| Plan critics | one `critic` agent, three lenses, all Codex-twinned | three SPECIALIZED critics — `statistics-critic`, `methodology-baselines-critic`, `efficiency-critic` — each Codex-twinned; `consistency-checker` Claude-only |
| Implementation review | `code-reviewer` + Codex twin | `plan-adherence-critic` (Claude) ∥ `code-correctness-critic` + ONE Codex twin (combined correctness+efficiency rubric) ∥ `efficiency-critic` (impl mode) |
| Clean-result body | v4 markdown four-flat-H2 (`verify_task_body.py`) | **report-v1** (`<!-- report-v1 -->`, `verify_report.py`), fixed-structure report |
| Retired agents (never spawned in v2) | — | `interpretation-critic` + twin, `clean-result-critic` + twin, `codex-follow-up-critic`; analyzer's interpretation role dropped; humanize-on-results + methodology-doc export end |
| Cheap follow-up band | `0 < est_gpu_hours < 20` auto-runs | **`0 < est_gpu_hours < 10`** auto-runs (threshold LOWERED); zero-GPU floor cap 1 unchanged; follow-up-critic slimmed to **Claude-only** single-pass |
| living-docs / RESULTS.md / open_questions.md | auto-proposed | **manual / on-demand** (not auto-spawned); related-work-finder STAYS auto (proposal-only) |
| Uploads | policy ceiling + discard-with-recipe | **no policy ceiling** — main repo → overflow repo, incremental shard upload; discard only when BOTH quotas exhausted |

**Out of scope for v2 (pinned to v1):** `kind: campaign`, `paper: true`. Those
frontmatter values on a `workflow: v2` task are a contradiction — treat the task
as v1 (delegate back) and surface the conflict; do not run the v2 report pipeline
on a paper or campaign task.

---

## Status convention (v2 semantics — enum UNCHANGED)

The status enum is not changed (no public-contract change). Under v2 two labels
carry a v2-specific meaning, documented so the dashboard + watcher read them
right:

- `interpreting` = **report generation** (methodology-writer + plotter +
  mechanical assembly).
- `reviewing` = **report verification** (methodology-critic + report-verifier).

Every other status means what it does in v1. `awaiting_promotion` is still the
user gate: Thomas writes the TLDR + Next steps, then `task.py promote <N>
useful|not-useful`.

---

## Scope & Boundaries

`.claude/skills/issue/SKILL.md` § "Scope & Boundaries" applies verbatim, with the
v2 substitutions in the map above. In particular: `tasks/` is canonical state,
mutated only via `scripts/task.py`; NEVER run this skill in the PM session; the
single-orchestrator guard is binding.

## Companion files

- `.claude/skills/issue-v2/report-template.md` — the canonical report skeleton +
  authoring notes (the interpretivity rule, the two verify modes, the title-tag
  convention).
- `.claude/skills/issue-v2/planned_manifest.schema.json` — the planned-work
  manifest schema.
- `.claude/skills/adversarial-planner-v2/SKILL.md` — the v2 planning skill
  (DRAFT + CRITIQUE modes) this orchestrator invokes at Steps 1 + 3.
- `scripts/verify_report.py` — the report verifier (`--mode generation|promote`).
- `scripts/build_dashboards.py` — per-issue dashboard builder + `emit-links`.
- `.claude/workflow.yaml § markers` — the v2 markers (`epm:report`,
  `epm:methodology-check`, `epm:report-verified`, `epm:plan-revision-log`).

## Auto-continuation policy (v2 gates)

The inline `AskUserQuestion` gates are the SAME as v1 (CLAUDE.md §
Auto-continuation policy) — empty-body (`gates.empty_body`), missing-type
(`gates.missing_type`), Goal (`gates.experiment_goal`), clarifier-blocking
(`gates.clarifier_blocking`), plan-approval (see `workflow.yaml §
gates.plan_approval`). The park-and-wait gate is `awaiting_promotion`. v2 adds NO
new `AskUserQuestion` gate — the post-approval critic panel (Step 3) runs
autonomously and only re-enters the human loop through the existing plan-approval
gate (re-park) or a `blocked` transition. Outside these gates, NEVER ask "should I continue"; state
`Assumption:` / `Decision:` and proceed. `.claude/skills/issue/SKILL.md` §
"Autonomous session behavior" applies verbatim to `--auto` sessions.

---

## The pipeline

### Step 0: Load state + guards

`.claude/skills/issue/SKILL.md` § "Step 0: Load state" applies verbatim:

- the **single-orchestrator guard** (exactly one session drives `/issue <N>`;
  crash-recovery-respawn takeover rule; the stale-wake ownership re-check),
- **interactive-session registration** (`register-current --issue <N>`),
- the **workflow-fix recursion-guard self-set**,
- the **tick cron arming** (the `*/45` `/issue-tick <N>` backstop, idempotent via
  `CronList`; teardown at terminal/gate-park) — `/issue-tick` is workflow-version
  agnostic, so no v2-specific tick skill is needed.

**v2 assertion:** after reading `task.py view <N> --json`, assert the frontmatter
`workflow` is `v2` (it must be — the dispatcher only routes here for v2). If it is
absent/`v1`, this skill was invoked in error: stop and hand back to v1. If the
task also carries `paper: true` or `kind: campaign`, that is the out-of-scope
contradiction above — surface it and delegate to v1; do not proceed.

Then continue through the v1 pre-planning gates by reference — they are unchanged:

- **Step 0b** (`.claude/skills/issue/SKILL.md` § "Step 0b: Defaulting & autofill")
  — empty-body + missing-type gates apply verbatim.
- **Step 0c** (§ "Step 0c: Goal-of-experiment gate") — the `kind: experiment`
  Goal gate (`goal:` frontmatter + `## Goal` H2) applies verbatim, including the
  Step 0c-link open-question match-or-create.

Derive current state (status folder), task kind, and the marker map exactly as
v1 does, then dispatch on status to the step below.

### Step 1: Planning (adversarial-planner-v2 DRAFT)

Invoke `.claude/skills/adversarial-planner-v2/SKILL.md` in **DRAFT mode**:

- one interactive planning conversation (clarifier + planner merged) — clarifying
  questions asked in-chat in an interactive session; resolved autonomously in an
  `--auto` session per `.claude/skills/issue/SKILL.md` § "Autonomous session
  behavior",
- produces `plans/vN.md` + `artifacts/planned_manifest.json` (schema:
  `.claude/skills/issue-v2/planned_manifest.schema.json`),
- **the plan's compute section MUST contain a per-GPU-phase parallelization
  statement AND an API workload estimate.** A plan missing either is INCOMPLETE
  and goes back to the planner before approval — do NOT advance an incomplete
  plan to Step 2. (DRAFT mode enforces this; this orchestrator re-checks that the
  manifest exists + validates before parking.)

Set status to `plan_pending` when the plan + manifest are ready, then go to
Step 2.

### Step 2: Approval (`plan_pending` gate)

The plan-approval gate. Thomas's approval covers **plan + manifest together**.
`.claude/skills/issue/SKILL.md` § "Step 2c: Inline plan approval" applies
verbatim for the autonomous auto-approve decision (the
`set-status ... --auto-approve-if-autonomous --gpu-hours <X>` script call reads
`EPM_AUTONOMOUS_SESSION` + `EPM_PLAN_AUTOAPPROVE_GPU_HOURS`; a PreToolUse hook
hard-blocks a plan-approval ask under `EPM_AUTONOMOUS_SESSION`).

<!-- gate: gates.plan_approval -->
In an INTERACTIVE session, present the plan + manifest and wait for approval via
`AskUserQuestion` (the SAME `gates.plan_approval` gate as v1; see workflow.yaml §
gates). Lead with a chat-prose blockquote stating the GPU-hour estimate + the
one-line manifest summary (conditions × metrics × planned figures) so the ask is
self-contained. In an `--auto` session the auto-approve gate decides in code:
`auto_approved` (≤ cap) → continue to Step 3 in the same invocation;
`parked_over_cap` (> cap or blank estimate — fail safe) → post the parked marker,
PushNotification, EXIT.

On approval: status → `approved`, then Step 3. `awaiting_promotion` remains a
human gate regardless of this cap.

### Step 3: Post-approval critique (adversarial-planner-v2 CRITIQUE) — the big v2 change

The user approved a pre-critique draft; the specialized critic panel now hardens
it autonomously. Invoke `.claude/skills/adversarial-planner-v2/SKILL.md` in
**CRITIQUE mode**. This orchestrator drives the loop (adversarial-planner-v2
specifies the panel; the CANONICAL statement of the triggers below lives here).

**One spawn batch per round** (ONE message, staggered a few seconds per
CLAUDE.md § 429 token-pacing):

- `statistics-critic` ∥ its Codex twin `codex-statistics-critic`
- `methodology-baselines-critic` ∥ its Codex twin `codex-methodology-baselines-critic`
- `efficiency-critic` ∥ its Codex twin `codex-efficiency-critic`
- `consistency-checker` (Claude-only, no twin)

**Quota-sentinel pre-check first (#1204, CLAUDE.md § Codex ensemble
review):** when LIVE, spawn only `statistics-critic` ∥
`methodology-baselines-critic` ∥ `efficiency-critic` ∥
`consistency-checker` — skip all 3 codex-twin composer spawns this round
(instant confirmed no-show per lens, single-Claude decision), one
`epm:progress` note per round.

The Codex twins are prompt-composers; **the orchestrator bg-dispatches each via
`scripts/codex_task.py`** exactly as CLAUDE.md § "Codex ensemble review"
prescribes (the wrapper never self-dispatches Codex — orphan-job anti-pattern
#533):

```bash
Bash(run_in_background=true,
  command="uv run python scripts/codex_task.py --issue <N> --effort high \
    --prompt-file /tmp/codex-<lens>-issue-<N>.md --output-file /tmp/codex-<lens>-out-issue-<N>.md")
```

**Per-lens ensemble decision + reconciler** (per CLAUDE.md § Codex ensemble
review): PASS+PASS → lens PASSes; overlapping REVISE/REJECT → union blockers, one
revise round; PASS vs REVISE/REJECT (disagreement) → spawn `reconciler` (Claude,
fresh context, binding). The consistency-checker's BLOCK findings union into the
same revise round. Mechanical-contract-only residuals are stripped per the same
rule.

**Auto-revise + plan-revision log EVERY round.** On any binding REVISE, re-spawn
the `planner` to revise plan + manifest, bump the plan version
(`new-plan-version`), and post the log:

```bash
uv run python scripts/task.py post-marker <N> epm:plan-revision-log \
  --note "round <r>: <per-critic what changed + why>; manifest diff: <one-line summary>"
```

This log is Thomas's audit surface for post-approval drift (surfaced in chat +
the sessions digest). Re-run the panel on the revised plan. **Round cap 5**;
reconciler invocations do not count.

**RESURFACE / BLOCK triggers — the ONLY ways a human re-enters after approval.**
Below these, CRITIQUE mode auto-proceeds to Step 4. On any trigger:
**interactive → surface to the user; autonomous → post `epm:failure v1`
(`failure_class: code`) + `set-status <N> blocked` + notify. NEVER proceed
silently, NEVER pivot-loop past the cap.**

- **(a) User-only ambiguity** — a fact only the user can supply (priority / scope
  / a taste call with no memory-or-codebase signal). Interactive: surface in chat
  and wait. Autonomous: block (this is the residue of halt-criterion #1 that even
  autonomous mode cannot resolve — a fact the user UNIQUELY holds).
- **(b) GPU-hours beyond the approved cap** — a revision pushes the estimate over
  the approved cap. **Re-park at `plan_pending` and re-run the Step 2
  plan-approval gate** (a revised cost needs re-approval); autonomous mode's
  auto-approve gate re-evaluates and parks if still over cap (`parked_over_cap`).
  Re-checked EVERY round.
- **(c) Material design change** — base model, data source/tier, DV/metric
  family, **manifest condition-set membership**, or backend lane class changes.
  These are what the approval protected → **re-park at `plan_pending`** (re-run
  Step 2); do not let a material change land silently on an approved plan.
- **(d) Round cap hit with an unresolved SUBSTANTIVE (non-mechanical-strip)
  blocker** — the panel reached round 5 and a substantive blocker remains.
  Interactive: surface; autonomous: `epm:failure` + block. (Apply the mechanical
  strip once more at the cap; a residual that is all mechanical → PASS, proceed.)
- **(e) REJECT-level verdict** — a critic or reconciler judges the plan
  fundamentally unsound (not fixable by a bounded revision). This is a
  self-defeating-plan signal → `set-status <N> planning` + re-invoke
  adversarial-planner-v2 DRAFT with explicit pivot scope naming the contradiction
  (autonomous), or surface (interactive). Do NOT descope a hyperparameter to
  dodge it.

All-PASS (or mechanical-strip-only residual) → status → Step 4.

### Step 4: Implement + review

Dispatch the implementer per task kind, exactly as v1
(`.claude/skills/issue/SKILL.md` § "Step 4: Worktree + dispatch implementer"
applies verbatim for worktree creation via `new_worktree.sh`, the spec-freshness
sync, and the implementer/experiment-implementer split):

- `kind: experiment` → `experiment-implementer` (training/eval/data code for the
  one variable this experiment changes).
- `kind: infra|batch|analysis|survey` → `implementer`.

**The implementer's spec (v2-specific, baked in — not just checked by critics):**
launch commands MUST shard across every provisioned GPU by default (no serial
per-cell loop, no single-GPU vLLM on an N-GPU pod), and all API calls route
through `src/explore_persona_space/llm/api_dispatch.py`. These are authoring
obligations; the panel below VERIFIES them.

**TDD conditional gate** (`gates.tdd_gate`) — fires when the plan body has
`### TDD: yes`; `.claude/skills/issue/SKILL.md` § "Step 4b" applies verbatim
(implementer posts `epm:proposed-tests v<n>` (omit --version; the CLI derives max+1), EXITs awaiting `epm:approve-tests v1`;
event-driven, no ask site).

**Review panel per round** (ONE spawn batch, staggered):

- `plan-adherence-critic` (Claude-only) — diff vs the approved plan + manifest;
  deviations need stated reasons.
- `code-correctness-critic` ∥ ONE Codex twin `codex-code-reviewer` carrying the
  v2 **combined correctness + implementation-efficiency** rubric (bugs, silent
  failures, fail-fast; inner loops batched; API via dispatcher; device routing;
  **multi-GPU: launch commands demonstrably shard across every provisioned GPU**).
- `efficiency-critic` (implementation mode) — the Claude side of the same
  efficiency checks.

**Quota-sentinel pre-check first (#1204, CLAUDE.md § Codex ensemble
review):** when LIVE, skip the `codex-code-reviewer` twin spawn;
`code-correctness-critic` proceeds Claude-only per the v1 Step 5d
no-show fallback (applied verbatim here).

Ensemble decision, reconciler on disagreement, mechanical-strip, and **round cap
5** are IDENTICAL to v1 (`.claude/skills/issue/SKILL.md` § "Step 5: Code review
loop" + CLAUDE.md § Codex ensemble review apply verbatim, with the v2 panel
substituted). At the cap with a substantive residual: interactive → surface;
autonomous → `epm:failure` + block. Also apply Step 5.bis pre-dispatch checks
(compute-deviation + whack-a-mole) by reference.

### Step 5: Run

Identical to v1. `.claude/skills/issue/SKILL.md` applies verbatim:

- **Backend dispatch** — § "Backend dispatch (slice-6 unified router)" + §
  "Operational dispatch (slice-6 router, ALL backends)": read `backend:`
  frontmatter (empty → auto, GCP-first ladder; RunPod opt-in), dispatch via
  `scripts/dispatch_issue.py`, persist the handle to
  `.claude/cache/issue-<N>-handle.json`.
- **Pod provisioning + preflight** (RunPod lane) — § "Step 6" (6a HF gate-access,
  6a.5 carry-over artifact check, 6a.6 HF write-headroom, 6b provisioning, 6c
  preflight on resumed pods).
- **Experimenter dispatch + the orchestrator polling loop** — § "Step 6d"
  (6d.0 smoke/sweep architecture parity, 6d.0-bis end-to-end smoke gate, 6d.1
  spawn experimenter launch-only, 6d.2 the bg-Bash chained ADAPTIVE POLL INTERVAL
  loop, 6d.3 on `done`, 6d.4 on `gate`).
- **Monitor → results** — § "Step 7".

The runtime is unchanged from v1 end to end here; do not re-implement any of it.

### Step 6: Upload-verify (v2 mode)

Spawn `upload-verifier` in **v2 mode**. `.claude/skills/issue/SKILL.md` § "Step 8:
Upload verification" applies verbatim for the hard-gate semantics (FAIL blocks
advancement; pod terminate strictly requires PASS; `epm:pod-terminated`; the
Step-8 disk cleanup via `clean_experiment_downloads.py` and the § "Step 8-bis:
Pod must not idle on a halt" rule). The v2 upload policy adds:

- **100% reconciliation of ALL produced artifacts** — enumerate compute-side
  output dirs + shard-upload logs; every produced artifact resolves to a
  permanent URL (including incrementally-shard-uploaded stores already deleted
  locally, verified against the HF listing). Undeclared missing = FAIL.
- **No policy ceiling** — text/JSON (rollout text, judge outputs, metrics,
  configs) uploads unconditionally (non-LFS, quota-immune); tensors/activations
  upload main-repo-first, rerouting to the private overflow repo
  (`EPM_HF_OVERFLOW_ROUTING`, `superkaiba1/explore-persona-space-overflow`) on
  quota pressure with an `OVERFLOW_POINTER.json` breadcrumb.
- **Discards** — a declared discard is verified against BOTH quota-exhaustion
  evidence AND a regen recipe; text/generations are NEVER discardable (an
  undeclared OR text discard = FAIL).

On PASS, append the produced artifacts to the artifact registry
(`artifacts/registry.jsonl`, flock-appended: id, type, HF/git path, producing
issue, size, recipe capsule). Then advance to Step 7.

### Step 7: Report (the report pipeline — write fully)

After results land + upload-verification PASSes, GENERATE the report. No agent
interprets; the report is Motivation / TLDR / Methodology (metrics embedded) /
Results-as-plots / Next steps per the official template
(`.claude/skills/issue-v2/report-template.md`), and Thomas alone writes the
claims — the `# Result:` title, the TLDR, every per-result `**Takeaways:**`
block, and Next steps.

**7a. One parallel spawn batch** (ONE message):

- `upload-verifier` (if not already PASSed at Step 6 — the results-landed batch
  may run it concurrently; the gate below still requires PASS),
- `plotter` in **HOLD mode** — reads `eval_results/` + `artifacts/planned_manifest.json`,
  produces MANY plot views (aggregate + per-unit, raw + processed, alternative
  groupings) via `/paper-plots`, each figure self-describing (title, axis labels
  + units, legend, ≤3-sentence factual caption), and writes a `captions.json`.
  **Figures are committed only AFTER upload PASS** (HOLD).
- `methodology-writer` in **REPORT MODE** (findings-blind) — authors Motivation
  + Methodology (the metric definitions + rationale as Methodology's final
  `**Metrics:**` block — no separate `## Metrics:` H2) from the plan, code,
  configs, dashboard manifest, and verbatim per-row examples. It NEVER reads
  aggregated `eval_results/*.json` metrics or any interpreted summary (the
  structural firewall is the primary anti-interpretation control).

Set status to `interpreting` (= report generation under v2) for this stage.

**7b. Commit held figures + build dashboards + pin links.** WAIT for the
7a-batch upload-verifier's PASS first (the 7a spawn may still be running —
the Step-7 gate requires PASS, and this commit is what releases the plotter
HOLD), then commit the HELD plotter figures — BEFORE assembly (7c), so 7c can
splice real SHA-pinned image URLs — in the same explicit-path commit as the
dashboards; capture ONE commit SHA; push (raw permalinks resolve only for
pushed commits, and the pushed `issue-<N>` branch — kept after the Step-10d
rebase-merge — is the pin's durable anchor):

```bash
uv run python scripts/build_dashboards.py build --issue <N>      # renders experiments/dashboards/issue<N>_*.html (sharded)
git add figures/issue_<N>/ experiments/dashboards/          # explicit paths only
git commit -m "task #<N>: report figures + dashboards"
git push origin issue-<N>
SHA=$(git rev-parse HEAD)
uv run python scripts/build_dashboards.py emit-links --issue <N> --sha "$SHA"
```

The SAME `$SHA` pins the 7c image URLs. Idempotent on re-entry: figures already
committed + unchanged → reuse `git log -n1 --format=%H -- figures/issue_<N>/`;
a 7d/7e round that re-runs the plotter re-commits the changed figures (new SHA),
re-splices the affected pins, and re-verifies.

Splice the emitted links into the Methodology section's inline dashboard-link
slots. Payload capped ~10 MB/issue; oversized families shard numerically;
full-fidelity dumps go to the HF data repo (terminal fallback: HF-hosted plain
link).

**7c. Mechanical assembly (the orchestrator does this — no interpreting agent).**
Assemble the report body from `.claude/skills/issue-v2/report-template.md`
(section order: Motivation → TLDR → Methodology → Results → Next steps):

- splice the methodology-writer's Motivation + Methodology sections (metrics
  are Methodology's final `**Metrics:**` block — there is no `## Metrics:` H2),
- for each figure in `captions.json`, emit a `### <plot name>` Results
  subsection: the factual "what was tested + what is plotted EXACTLY" caption →
  the `**Plot: <name>**` label → the SHA-pinned
  `![...](raw.githubusercontent.com/<owner>/<repo>/<the Step-7b commit SHA>/figures/issue_<N>/...)`
  image → a `**Takeaways:**` block holding the literal `*(Thomas fills in)*`
  placeholder (Thomas's claim slot; nothing else after the image),
- leave `## TLDR:` and `## Next steps:` as the literal `*(Thomas fills in)*`
  placeholders,
- keep the `# Experiment: <question>` H1 with NO confidence tag (Thomas
  retitles to `# Result: <claim>` + optional confidence tag at TLDR time) and
  the `<!-- report-v1 -->` sentinel right after the H1.

**7d. methodology-critic loop (cap 5).** Spawn `methodology-critic`: it traces
every Motivation / Methodology claim (including the embedded `**Metrics:**`
block) to ground truth (configs, code, artifact counts, `adapter_config.json`,
dashboard row counts; links resolve at the pinned SHA). On findings, the orchestrator re-runs the methodology-writer to fix them
and re-spawns the critic. Post `epm:methodology-check` per round. Round cap 5;
at the cap with a substantive residual, interactive → surface, autonomous →
`epm:failure` + block.

**7e. report-verifier loop (cap 5).** Set status to `reviewing` (= report
verification under v2). Spawn `report-verifier`: it (a) recomputes ≥1 plotted
value per figure from source JSON via the manifest's transform recipe; (b)
checks captions match the plotted data and axes/legends are complete; (c)
**completeness vs the manifest** — every planned condition / metric / figure is
present OR explicitly `not run` (no selective subset); (d) the **interpretivity
lens** (hypothesis-to-be-tested ALLOWED; asserted conclusion BANNED — per
report-template.md § "The interpretivity rule"); and (e) runs the mechanical
gate:

```bash
# the report exists only as the 7c draft file until 7f's set-body — verify the DRAFT:
uv run python scripts/verify_report.py --file <report-draft>.md --mode generation \
  --expect-issue <N> --figures-root <worktree-root> \
  --manifest <path>/planned_manifest.json
```

(`<report-draft>.md` = the file 7c assembles and 7f's `set-body --file` consumes —
same path both places; e.g. the task's `artifacts/report-draft.md` or a
worktree-local draft. The promote-mode invocation `--issue <N> --mode promote`
is unchanged — `body.md` IS the report by then.)

Generation mode REQUIRES the TLDR + Next-steps placeholders intact and runs the
interpretivity / lexicon checks on the AGENT-written sections only. On findings,
re-run the plotter / methodology-writer / assembly as needed and re-verify. Post
`epm:report-verified` on PASS. Round cap 5; at the cap with a substantive
residual, interactive → surface, autonomous → `epm:failure` + block.

**7f. Write body + park.** After `verify_report.py --mode generation` PASSes
(figures were committed + pinned at Step 7b), write the report body and park:

```bash
# --allow-goal-drop is DELIBERATE: the report-v1 skeleton carries `## Motivation:`,
# not `## Goal` (the `goal:` frontmatter is preserved by set_body), so the Goal-H2
# drop guard (#1112) must be explicitly overridden here.
uv run python scripts/task.py set-body <N> --file <report-draft>.md --snapshot --allow-goal-drop
uv run python scripts/task.py set-title <N> "Experiment: <one-line question>"   # Thomas retitles to "Result: <claim>" (+ confidence tag) at promote
uv run python scripts/task.py set-clean-result <N>                              # accepts the report-v1 sentinel
uv run python scripts/task.py post-marker <N> epm:report --note "report-v1 generated + verified; awaiting Thomas TLDR"
uv run python scripts/task.py set-status <N> awaiting_promotion
```

The report pipeline emits NO methodology-doc export and runs NO humanize loop
(the report IS the methodology; both are retired in v2). Then go to Step 8.

> **The worktree auto-merges to `main` at `awaiting_promotion`** exactly as v1
> (`.claude/skills/issue/SKILL.md` § "Step 10d: Auto-merge the worktree" applies
> verbatim — rebase-merge, merge-safety guards, keep the worktree). This is NOT a
> gate; it fires independent of promotion.

**Promotion (Thomas, later).** Thomas writes the TLDR + Next steps (and MAY
append `(HIGH|MODERATE|LOW confidence)` to the H1), then runs
`task.py promote <N> useful|not-useful`. Promote-time verification is
`verify_report.py --mode promote` (TLDR MUST now be filled; Thomas's TLDR +
Next steps are NEVER lexicon/interpretivity-checked). No automation flips
`runs.classification` — user-only.

### Step 8: Post-park (follow-ups, related-work; living-docs manual)

After the task parks at `awaiting_promotion` (and, for code-change kinds, at the
`completed` test-verdict path), run the post-completion batch — analogous to v1
Step 10b/10c/10c-bis but with the v2 substitutions:

**Follow-ups (both auto-run bands KEPT; cheap band threshold LOWERED to <10):**

- Spawn `follow-up-proposer` (reads results + plan; tags each proposal
  `cost_class`, `est_gpu_hours`, `question_relation: same|substantially-different`).
- Screen EVERY proposal through the **Claude-only** single-pass redundancy screen
  `follow-up-critic` (v2 retires `codex-follow-up-critic` — the screen is
  Claude-only). `redundant` → park `on_hold` (`epm:followup-parked-redundant v1`,
  revivable); `not-redundant` → route below. Nothing is dropped. User-requested
  (`source: user-chat`) proposals are not screened.
- **Zero-GPU floor** (`cost_class: free-analysis`, `est_gpu_hours: 0`) — AUTO-RUN
  inline, cap 1 round (`epm:free-analysis-followup-run v1`), analysis-only.
- **Cheap GPU band** (`0 < est_gpu_hours < 10` — the v2 threshold, LOWERED from
  v1's 20) — AUTO-RUN via the same-issue follow-up loop, cap 2 rounds. Strict
  `< 10` (exactly 10 does NOT auto-run). A `same`-question proposal with a
  missing/unparseable `est_gpu_hours` does NOT auto-run (fail-safe → park/file).
- A `question_relation: substantially-different` proposal NEVER auto-runs — it is
  filed as a `proposed` child for manual triage.

The **same-issue follow-up loop mechanics** (held at `followups_running`, the new
finding folded into the EXISTING report body via a fresh plotter +
methodology-writer + report-verifier pass, re-park at `awaiting_promotion`) are as
v1 (`.claude/skills/issue/SKILL.md` § "Step 9b § Same-issue follow-up loop" +
Step 0 same-issue dispatcher apply verbatim, with the v2 report pipeline
substituted for the interpretation loop).

**Related-work-finder — STAYS auto (proposal-only).** Spawn `related-work-finder`
concurrently; it proposes a findings-keyed "Related findings" note through the
non-blocking `related_work_positioning` gate (confirm/reject). Unchanged from v1.

**living-docs-updater — MANUAL / on-demand in v2.** Do NOT auto-spawn it.
`RESULTS.md` and `docs/open_questions.md` updates are manual/on-demand in v2
(Thomas confirmed). Surface the suggestion in chat if warranted; do not apply.

For code-change kinds (`infra|batch|analysis|survey`), the report pipeline (Step
7) is skipped; the task completes on the Step 9c test-verdict path exactly as v1
(`.claude/skills/issue/SKILL.md` § "Step 9" / "Step 10" apply verbatim), and Step
10d auto-merges the worktree.

---

## Autonomous session behavior

`.claude/skills/issue/SKILL.md` § "Autonomous session behavior" applies verbatim.
In `--auto` sessions there is no human to escalate to: never present a choice
menu; pick the option with max info-gain-per-GPU-hour toward the Goal (tie-break
lower-cost/safer/record-correcting), state `Decision: <X>`, continue. The ONLY
autonomous residue of a user-only fact is a fact the user UNIQUELY holds. Cost is
gated ONLY at the Step 2 plan-approval GPU-hour cap, never mid-run.

## Halt criteria

`.claude/skills/issue/SKILL.md` halt-criterion contract + CLAUDE.md §
"STATE-TO-`blocked` criteria" apply verbatim. Outside the inline gates, NEVER use
`AskUserQuestion` — post `epm:failure v1` (`failure_class: <code|infra|data>`),
`set-status <N> blocked`, exit. The Step 3 resurface triggers (b)/(c) re-park at
`workflow.yaml § gates.plan_approval`; the (a)/(d)/(e) triggers and the Step 4 / 7
round-cap residuals route through these criteria (interactive: surface;
autonomous: `epm:failure` + block).

## Resume semantics

`.claude/skills/issue/SKILL.md` § "Resume semantics" + the `epm:step-completed`
skip-ahead apply verbatim. Re-invoking `/issue-v2 <N>` (or v1 `/issue <N>`, which
re-dispatches here) picks up from the current status + marker map. The v2 markers
(`epm:report`, `epm:methodology-check`, `epm:report-verified`,
`epm:plan-revision-log`) are read the same way as any `epm:` marker (highest
version per kind, except the `epm:followup-scope` multi-label scan).

## Markers (v2 additions)

Defined in `.claude/workflow.yaml § markers`:

- `epm:plan-revision-log v1` — one per CRITIQUE-mode round (Step 3): what changed
  + why, per critic; manifest-diff summary.
- `epm:methodology-check v1` — one per methodology-critic round (Step 7d).
- `epm:report-verified v1` — posted on report-verifier PASS (Step 7e).
- `epm:report v1` — posted at Step 7f when the report-v1 body is generated +
  verified and parked awaiting Thomas's TLDR.

All other markers (`epm:status-changed`, `epm:run-launched`,
`epm:upload-verification`, `epm:merged`, `epm:followup-scope`,
`epm:same-issue-followup-run`, …) are as v1.

## When NOT to use this skill

- The task is `workflow: v1` (or the flag is absent) → the v1 `/issue` skill owns
  it; this skill is reached only via the v1 dispatcher for `v2` tasks.
- The task is `paper: true` or `kind: campaign` → pinned to v1 (out of scope for
  v2); a `workflow: v2` + `paper/campaign` combination is a contradiction to
  surface, not run.
- In the PM session → NEVER run any per-issue orchestrator (v1 or v2) in the PM
  session (`.claude/skills/issue/SKILL.md` topology rule applies verbatim).

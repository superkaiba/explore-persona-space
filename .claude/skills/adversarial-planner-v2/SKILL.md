---
name: adversarial-planner-v2
description: >
  Two-mode planning skill for the v2 experiment workflow (`workflow: v2` tasks),
  invoked by the /issue-v2 orchestrator. DRAFT mode runs one interactive
  planning conversation (clarifier + planner merged) that produces a plan
  revision plus a machine-readable planned_manifest.json. CRITIQUE mode (run
  POST-approval, Thomas's explicit pipeline order) runs a specialized critic
  panel — statistics / methodology-baselines / efficiency, each Codex-twinned,
  plus a Claude-only consistency-checker — that auto-revises the plan and logs
  every round. Compact by design: unchanged mechanics defer to the v1
  /adversarial-planner skill.
user_invocable: false
---

# Adversarial Planner v2

This skill is the planning half of the v2 experiment workflow. It is invoked by
`.claude/skills/issue-v2/SKILL.md` (the v2 per-task orchestrator) at two distinct
points and runs in one of TWO modes depending on which point called it:

- **DRAFT mode** — issue-v2 **Step 1** (planning). One interactive planning
  conversation with the user (the v1 clarifier + planner, merged) produces
  `plans/vN.md` + `planned_manifest.json`. Ends at the `plan_pending` approval
  gate. This is where clarifying questions are asked.
- **CRITIQUE mode** — issue-v2 **Step 3** (post-approval hardening). A
  specialized critic panel + Codex twins + consistency-checker adversarially
  reviews the APPROVED plan + manifest, auto-revises it, and logs every round.
  Surfaces to the user (or blocks, autonomous) only under the widened
  resurface triggers.

The mode is passed explicitly by issue-v2 in the invocation ("DRAFT mode" /
"CRITIQUE mode"). If you were invoked without a mode, you are in DRAFT (the
default entry).

> **This is a v2-only skill.** v1 experiments + `/campaign` keep invoking the
> untouched `.claude/skills/adversarial-planner/SKILL.md`. Do NOT invoke this
> skill for a `workflow: v1` (or campaign-child) task.

---

## Why the order is inverted vs v1 (approve-first, critique-after)

v1 critiques BEFORE the user approves. v2 flips this on Thomas's explicit
instruction: *"I plan spec back and forth with Claude Code → plan → I approve →
it goes autonomously from there."* The user approves a **pre-critique draft**;
the critic panel then hardens it autonomously and the **plan-revision log**
(`epm:plan-revision-log v1`, one per round) is Thomas's audit surface for
everything the panel changed after his approval. The widened resurface/block
triggers (CRITIQUE mode below) are the safety net: any material change the
panel wants to make bounces back to the user rather than landing silently.

This is a documented tradeoff (reviewed and set aside the critique-before-approval
alternative because Thomas specified this order). The canonical statement of the
gate order + triggers is **issue-v2 Step 3**; this skill restates them briefly.

---

## Shared mechanics (defer to v1)

For everything NOT changed below, follow `.claude/skills/adversarial-planner/SKILL.md`:

- **Plan file locations + versioning** — `plans/vN.md`, `plan.md` symlink to the
  highest version, `uv run python scripts/task.py new-plan-version <N> --file <plan.md>`.
- **Fact-checker phase** (v1 Phase 1.5) — assumption verification runs UNCHANGED
  as part of DRAFT mode (grounded-hyperparameter `Source:` lines per
  `planner.md` §11, verified before the plan is presented for approval).
- **Codex twin dispatch** — the twins are prompt-composers; the ORCHESTRATOR
  (issue-v2) bg-dispatches `scripts/codex_task.py`. Follow CLAUDE.md § "Codex
  ensemble review" verbatim (stagger spawns; the wrapper never self-dispatches
  Codex — orphan-job anti-pattern #533).
- **Reconciler** — a Claude+Codex PASS-vs-FAIL disagreement on a twinned lens
  spawns the `reconciler` agent (fresh context, binding), per v1.

---

## DRAFT mode (issue-v2 Step 1)

Produce the plan the user approves. One planning conversation, clarifier +
planner merged.

### 1. Clarify + plan (one conversation)

Spawn the `planner` agent (per `.claude/agents/planner.md`) to design the
experiment. In an INTERACTIVE session, surface blocking ambiguities to the user
as clarifying questions in the SAME planning conversation before finalizing the
plan.
<!-- gate: gates.clarifier_blocking -->
Only a genuinely blocking ambiguity (a fact only the user
knows — priority, scope, a design preference between valid paths with no
memory/codebase signal) is raised via `AskUserQuestion`; everything the planner
can decide from the codebase, memory, or the task Goal it decides. In an `--auto`
session there is no user to ask — resolve per the v1 autonomous rules (CLAUDE.md
§ Auto-continuation policy: pick max-info-gain-per-GPU-hour toward the Goal,
state `Decision: <X>`, continue).

### 2. Planner obligations (v2-specific — a plan lacking these is INCOMPLETE)

Beyond the v1 planner contract, the v2 plan's compute section MUST contain BOTH:

- **A per-GPU-phase parallelization statement.** For every GPU phase, state
  EXACTLY how the work shards across ALL provisioned GPUs: vLLM tensor-parallel /
  data-parallel degree, per-GPU cell / seed / condition splits, or process
  fan-out. "Runs on the pod" is not a statement; "8× H100, 24-cell fan-out = 3
  cells/GPU via `CUDA_VISIBLE_DEVICES` round-robin" is. A serial single-GPU plan
  on a multi-GPU pod is incomplete by construction (the efficiency-critic REVISEs
  it in CRITIQUE mode; catch it here first).
- **An API workload estimate.** Total judge + generation API calls × model ×
  sync-vs-batch, checked against `docs/api_throughput_guidelines.md`'s decision
  table (the Anthropic Batch API for large sets; the polite per-key caps).

Also (v1 rules, restated because they gate the manifest):

- **Grounded hyperparameters** — every load-bearing value carries a `Source:`
  (arXiv id / prior issue), per `planner.md` §11. Never a bare library default.
- **Artifact-registry read** — when `artifacts/registry.jsonl` exists, READ it
  and prefer a fit-for-purpose existing artifact over retraining (the (a)-(k)
  fitness check, `.claude/rules/artifact-reuse.md`). Degrade gracefully when the
  registry is absent (Phase-1 dogfood tasks have none).

**A plan missing the parallelization statement OR the API estimate is
incomplete and goes back to the planner BEFORE it is presented for approval** —
do not advance an incomplete plan to the `plan_pending` gate.

### 3. Emit the planned-work manifest

Emit `planned_manifest.json` alongside the plan, conforming to
`.claude/skills/issue-v2/planned_manifest.schema.json`:

- `conditions` — plain-English arm / condition names (each must appear in the
  eventual report).
- `metrics` — plain-English DV / metric names (each must appear in the report).
- `figures` — one object per planned figure, each with `{id, title, source,
  transform, plotted_quantity}`. The `transform` is a human- and
  machine-readable recipe (source JSON → aggregation/normalization → plotted
  quantity) the plotter and report-verifier consume mechanically. A planned
  figure the run cannot produce is marked `not run` in the report, not dropped.

Validate the manifest against the schema before finishing DRAFT mode:

```bash
uv run python -c "import json,jsonschema,sys; \
  s=json.load(open('.claude/skills/issue-v2/planned_manifest.schema.json')); \
  m=json.load(open('<path>/planned_manifest.json')); \
  jsonschema.validate(m,s); print('manifest OK')"
```

Write it to the task folder (`artifacts/planned_manifest.json`) so issue-v2's
plotter + report-verifier read it. **Thomas's approval covers plan + manifest
together** (issue-v2 Step 2); a CRITIQUE-mode revision that changes the manifest's
condition-set membership is a material change that resurfaces (Step 3 trigger c).

### 4. Hand back to issue-v2 for approval

DRAFT mode ends by handing the plan + manifest to issue-v2 Step 2, which parks
at `plan_pending`. Do NOT run the critic panel in DRAFT mode — that is CRITIQUE
mode, which fires only AFTER approval.

---

## CRITIQUE mode (issue-v2 Step 3, post-approval)

Harden the APPROVED plan. Runs ONLY after the user (or the autonomous
auto-approve gate) approved the plan at `plan_pending`.

### 1. One spawn batch — the specialized critic panel + consistency-checker

Spawn ALL of these in ONE message so they run concurrently (stagger a few
seconds per CLAUDE.md § 429 token-pacing). Each Claude critic returns its
verdict body; each Codex twin is a prompt-composer whose dispatch the
orchestrator bg-runs via `scripts/codex_task.py`:

| Lens | Claude critic | Codex twin | Scope |
|---|---|---|---|
| Statistics & measurement | `statistics-critic` | `codex-statistics-critic` | measurement validity, dual-DV, saturation, selection-symmetric nulls, group-level held-out folds (eval fully disjoint from training; standing exemptions: replication-fidelity, marker-at-slot), LLM-judging rules |
| Methodology & baselines | `methodology-baselines-critic` | `codex-methodology-baselines-critic` | controls, baselines, established-literature benchmarks preferred, contrastive negatives, on-policy completions, data-realism tiers, replication fidelity, persona-vectors / marker recipe compliance |
| Efficiency | `efficiency-critic` | `codex-efficiency-critic` | vectorization, CPU + GPU parallelization, API workload estimate + batch-vs-sync, pod-width right-sizing, VM-vs-own-CPU-pod, footprint sizing, **multi-GPU saturation (a serial single-GPU plan on a multi-GPU pod is a REVISE)** |
| Single-variable-change | `consistency-checker` (Claude-only) | — | one variable changed vs the parent recipe; no reuse-smuggled variable |

Efficiency EARNS its Codex twin here (Thomas's multi-GPU emphasis).
Consistency-checker + the (implementation-side) plan-adherence lens are
Claude-only. Pass each subagent the PATH to `plans/vN.md` + `planned_manifest.json`,
never the bodies (429 pacing); each Codex composer reads the plan from the handed
path at compose time and inlines the verbatim plan text into its composed Codex
prompt — `{{plan_body}}` is a compose-time substitution, not a brief field.

**Quota-sentinel pre-check first (#1204, CLAUDE.md § Codex ensemble
review):** when LIVE, the batch is the 3 Claude critics +
consistency-checker only — all 3 codex twins skipped as instant
confirmed no-shows (single-Claude per lens), one `epm:progress` note per
round.

### 2. Per-lens ensemble decision + reconciler

Per twinned lens (statistics / methodology-baselines / efficiency), combine the
Claude + Codex verdicts exactly as CLAUDE.md § Codex ensemble review prescribes:

- PASS + PASS → lens PASSes.
- REVISE/REJECT + REVISE/REJECT overlapping → union the blockers, one revise round.
- PASS vs REVISE/REJECT (disagreement) → spawn `reconciler` (fresh context,
  binding). Its verdict is final for that lens.

The consistency-checker's BLOCK findings union into the SAME revise round (no
twin, no reconciler — it is Claude-only). Cross-lens worst-wins: any binding
REVISE/REJECT after reconciliation triggers a revise round.

### 3. Auto-revise + the plan-revision log (every round)

On any binding REVISE, re-spawn the `planner` to revise the plan + manifest
against the unioned blockers, bump the plan version
(`new-plan-version`), and **post the plan-revision log**:

```bash
uv run python scripts/task.py post-marker <N> epm:plan-revision-log \
  --note "round <r>: <per-critic what changed + why>; manifest diff: <summary>"
```

The log names, per critic, WHAT changed and WHY, plus a one-line manifest-diff
summary. It is Thomas's audit surface for post-approval drift (surfaced in chat +
the sessions digest). Then re-run the panel (step 1) on the revised plan. **Round
cap 5** per lens; reconciler invocations do not count.

### 4. Resurface / block triggers (the ONLY ways a human re-enters)

CRITIQUE mode runs autonomously EXCEPT these five triggers. On any of them:
**interactive → surface to the user; autonomous → post `epm:failure v1`
(`failure_class: code`) + `set-status <N> blocked` + notify. NEVER proceed
silently.** (Canonical statement: issue-v2 Step 3.)

- **(a) User-only ambiguity** — a fact only the user can supply surfaces during
  revision.
- **(b) GPU-hours beyond the approved cap** — a revision pushes the estimate over
  the GPU-hour cap the user (or the auto-approve gate) approved. Re-checked EVERY
  round.
- **(c) Material design change** — base model, data source/tier, DV/metric
  family, **manifest condition-set membership**, or backend lane class changes.
  These are what the approval protected; they cannot land silently.
- **(d) Round cap hit with an unresolved SUBSTANTIVE blocker** — the panel hit
  its round-5 cap and a non-mechanical-strip blocker remains. (Mechanical-contract
  blockers are stripped per CLAUDE.md § Codex ensemble review, not surfaced.)
- **(e) REJECT-level verdict** — a critic (or reconciler) judges the plan
  fundamentally unsound (not fixable by a bounded revision). Route to a re-plan,
  not a silent proceed.

Below these triggers, CRITIQUE mode auto-proceeds — all lenses PASS (or their
residual is mechanical-strip only) → hand back to issue-v2 Step 4 (implement).

---

## Implementation pattern (for the issue-v2 orchestrator)

```text
DRAFT (Step 1):
  spawn planner (interactive: clarify in-conversation; --auto: autonomous resolve)
  → fact-check assumptions (v1 Phase 1.5)
  → planner emits plans/vN.md + artifacts/planned_manifest.json
  → assert: parallelization statement + API estimate present  (else back to planner)
  → validate manifest vs schema
  → hand to Step 2 (park at plan_pending)

CRITIQUE (Step 3, post-approval):
  loop (cap 5):
    # pre: #1204 quota-sentinel check (CLAUDE.md § Codex ensemble review) —
    #      if LIVE, spawn Claude critics + consistency-checker only;
    #      codex twins = instant no-show per lens.
    spawn batch: statistics-critic + codex twin
               ∥ methodology-baselines-critic + codex twin
               ∥ efficiency-critic + codex twin
               ∥ consistency-checker (Claude-only)      # ONE message
    orchestrator bg-dispatches each codex twin (scripts/codex_task.py)
    per-lens ensemble decision; reconciler on PASS-vs-FAIL disagreement
    all PASS (or mechanical-strip residual only) → break → hand to Step 4
    else → check resurface/block triggers (a)-(e)
         → if none: planner revises plan+manifest; post epm:plan-revision-log; re-loop
  at cap with substantive residual → trigger (d)
```

---

## Rules

- **Never invoke this skill for a `workflow: v1` or campaign-child task.** The v1
  `/adversarial-planner` owns those.
- **DRAFT asks; CRITIQUE does not (except triggers a-e).** Clarifying questions
  belong to DRAFT (before approval). CRITIQUE runs autonomously and only re-enters
  the human loop on the five triggers.
- **The manifest is a first-class deliverable**, approved alongside the plan and
  consumed mechanically downstream. A CRITIQUE revision that changes it logs the
  diff (revision log) and, if it touches condition-set membership, resurfaces.
- **No `AskUserQuestion` outside the two documented gates** — `workflow.yaml §
  gates.clarifier_blocking` in DRAFT, `workflow.yaml § gates.plan_approval` at
  the Step-2 handoff. Everything else is autonomous.

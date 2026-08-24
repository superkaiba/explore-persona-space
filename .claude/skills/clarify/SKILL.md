---
name: clarify
description: >
  Use when the user asks, in an interactive session, to run / try / launch /
  redo an experiment — a new direction, a same-issue follow-up, or a re-run —
  and BEFORE any routing action fires (task.py new, epm:followup-scope,
  inline free-analysis dispatch, spawn-issue, planning). Also on explicit
  /clarify. Standing user directive 2026-08-24.
user_invocable: true
---

# Clarify — exhaustive pre-routing clarification for experiment asks

## Overview

Standing user directive (2026-08-24, verbatim intent): whenever the user asks
to run an experiment, this skill "forces you to not make assumptions and
forces me to clarify anything that is unclear at all."

Core principle: **a choice is settled only when the user's ask, the task
record, or the parent recipe pins it. A project default never settles a NEW
experiment's choice — it becomes the recommended option inside a question.**
The user chose this bar explicitly ("literally everything") over a
defaults-are-assumed variant.

## When to fire

- Any interactive-chat user ask to run / try / launch / redo an experiment,
  regardless of phrasing: "run X", "try X", "what if we X", "run these
  followups", "rerun with more seeds", "spawn a session for X".
- Fires BEFORE any routing bullet in CLAUDE.md § Routing experiment intent
  executes: before `task.py new`, before posting `epm:followup-scope v1`,
  before an inline free-analysis dispatch, before `spawn_session.py
  spawn-issue`, before any planning.

## When NOT to fire

- Autonomous sessions (`EPM_AUTONOMOUS_SESSION=1`, cron-fired turns): no user
  to ask. Autonomous routing is unchanged; do not park.
- Pure capture ("save idea: X"): nothing executes now. The /issue Step-1
  clarifier (gate id 3) covers it when it is eventually run.
- Explicit user waiver in the ask ("skip clarify", "just run it as I said"):
  user override always wins; state the skipped gate in one line and proceed.
- Always-inline non-experiment work (monitoring, log-checking, pulling
  results, discussion, plotting existing data with no new run).

## Procedure

### Step 0 — Bounded context-gather (parallel, fast)

Purpose: make questions CONCRETE, not to skip them. Gather in parallel:
related tasks / clean-results (`task.py list-by-status`, sibling issues named
in the ask), the standing rules that pin defaults for this kind of work
(judge model, marker recipe, contrastive negatives, data-realism tiers,
linear-by-default, compute lanes), and existing reusable artifacts
(HF adapters, eval JSONs, raw completions). Output of this step: for each
upcoming question, the project-default option and its source (`rule` or
`#issue`), plus concrete option values instead of placeholders.

### Step 1 — Enumerate the decision surface

Walk the checklist below. Classify every row: **pinned** (the ask / task
record / parent recipe states it — quote where), or **open** (everything
else, including rows a project default would cover). Every open row becomes
a question.

| Row | What must be pinned |
|---|---|
| Goal / construct | The exact question, formally; what counts as an answer |
| Hypotheses | Competing hypotheses + the measurement that separates them |
| Routing | New task vs same-issue follow-up on #N vs inline free analysis |
| Model(s) | Base model, instruct vs base, size |
| Data | Source + realism tier; train/eval split; reuse vs generate |
| Method / recipe | Training recipe, extraction recipe, map/probe form (linear unless the user opts into nonlinear) |
| Conditions / arms | Full arm list; what varies; what is held fixed |
| Baselines / controls | Which baselines; identity+bias + kNN where a map is fit |
| DV / metric | Construct → metric → on-distribution?; judge + rubric; dual-DV where required |
| Seeds / n | Seeds, rollout counts, per-cell n |
| Success / kill criteria | What result means what; when to stop |
| Compute | Est. GPU-h, venue (pod intent / lane), parallelization shape |
| Scope OUT | What is explicitly not being tested this round |
| Priority / timing | Now vs queued; blocking anything of the user's |

**Follow-up / re-run scoping:** the checklist applies to the DELTA — what
this round changes vs the parent recipe — plus anything ambiguous in the ask
itself. Parent-settled rows are inherited: list them in the decision record
as one-liners (`inherited from #N: ...`), and re-ask ONLY if the ask touches
them or the delta breaks their coherence.

**Zero open rows** (fully specified ask, e.g. dispatching an already
clarified+planned task): say so in one line with a one-line spec summary,
then route. Do not manufacture questions.

### Step 2 — Ask

<!-- gate: gates.clarify_experiment_ask -->
Raise the open rows via AskUserQuestion, batched ≤4 questions per call,
structural rows first (Goal, Routing, Scope — their answers change which
other questions exist), then design rows, then execution rows. Where a
project default exists, it is option 1, labeled
`(Recommended — project default: <source>)`. Multi-select where choices are
not mutually exclusive. Free-text answers are honored verbatim, including
partial answers that re-open other rows. Interactive-only by construction
(see When NOT to fire). This ask is sanctioned by workflow.yaml §
gates.clarify_experiment_ask.

Work that does not depend on any answer (context pulls, artifact inventory)
may proceed while waiting. Nothing that depends on an answer proceeds.

### Step 3 — Decision record, then route

Echo back the resolved spec as a compact bullet record: goal, routing,
every pinned choice with its source (`user-answer` / `ask` / `inherited #N` /
`default-confirmed`), and the explicit OUT-of-scope list. Then route
immediately per CLAUDE.md § Routing experiment intent — no extra "shall I
proceed" confirm (the record is made of the user's own answers).

The record must travel: pass it as `task.py new --origin-prompt` (or into
`## Provenance`), and fold the goal answer into `## Goal` / `set-goal`, so
the /issue Step-1 clarifier and the planner inherit the answers instead of
re-asking the user.

## Rationalizations — all mean RUN THE GATE

| Excuse | Reality |
|---|---|
| "The ask is clear enough" | The user chose "literally everything". Unpinned = open = asked. |
| "A careful colleague would guess this" | Explicitly not a licence to guess (user rule 2026-08-19). |
| "The project default covers it" | Defaults are options inside questions, never silent answers, for anything the ask did not pin. |
| "It's just a re-run" | Re-runs are in scope; ask about the delta + anything ambiguous. |
| "The Step-1 clarifier will catch it" | Wrong fire point: Step 1 runs post-filing and only blocks on blocking ambiguities. This gate runs at ask time with a stronger bar. |
| "Asking will slow him down" | He ordered the slowdown. Batch well instead of skipping. |
| "I'll state assumptions in the wrap-up" | Burying assumptions after the work is the exact banned move. |

## Red flags — STOP, you are about to violate the gate

- About to run `task.py new` for a run-ask with no decision record posted.
- About to post `epm:followup-scope` / dispatch an inline round on a user
  run-ask that got zero questions and no zero-open-rows statement.
- Writing "Assumption:" in a routing turn for something the user could have
  been asked this turn.
- Composing a plan or brief that fills an unasked choice with a default.

## Relationship to existing machinery

- **/issue Step-1 clarifier (gate id 3):** stays, as the backstop for tasks
  reaching /issue without this gate (PM-filed, auto-filed, campaign
  children). With a decision record present it should find nothing left to
  ask.
- **/adversarial-planner:** unchanged; runs after routing as always. The
  decision record feeds it.
- **Autonomous behavior, follow-up auto-run bands, campaign routing:**
  unchanged.

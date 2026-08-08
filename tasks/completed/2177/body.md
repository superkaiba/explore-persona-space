---
title: 'verify_plan.py: WARN when a §9 compute row''s own basis arithmetic contradicts
  its booked GPU-h or its pilot-abort threshold'
kind: infra
tags: []
created_at: '2026-08-07T15:48:17Z'
has_clean_result: false
origin_prompt: 'Surfaced by the Methodology + Statistics critic lenses during #1336
  plan v16 Phase 2 review: the EXT_off row books 30 GPU-h while its own basis derives
  90, and its 30-min/cell abort fires on a run performing exactly as booked. Recurred
  across #823/#811/#1092/#1336 and survived two approved plan versions.'
workflow: v1
---
## Goal

Add a WARN-only mechanical check to `scripts/verify_plan.py` that catches a
plan §9 compute row whose OWN basis text derives a GPU-hour / wall-time figure
materially inconsistent with the figure the row BOOKS, or whose pilot-abort
threshold is inconsistent with its booked per-cell cost.

## Why — recurrence, not a one-off

This defect shape has now recurred across four issues and, in the driving case,
survived TWO approved plan versions unnoticed by the mechanical gate:

- **#823** — asserted ~2 s/fit, ~125 s real; 12-20 h realized.
- **#811** — unit 3/108 at 19h21m against its booking.
- **#1092** — a batched battery priced by FLOP / assumed throughput ran ~2.6x
  the naive booking.
- **#1336** (driving case) — the `EXT_off` §9 row books **3.8 wall-h / 30
  GPU-h** while its own basis text derives *"20 cells x 48.1k rows ~ 90 GPU-h
  naive-serial / 8-way ~ 11.2 wall-h"*, and its pilot-abort (*"> 30 min/cell —
  double the ~15 min budget"*) implies ~5 GPU-h. Three mutually inconsistent
  figures in one row. Inherited VERBATIM from approved plan v15 (v15:233) into
  v16 (v16:277); `verify_plan.py` passed both at 0 FAIL.

Two consequences make this worth mechanizing rather than leaving to reviewers:

1. **The abort gate can fire on a run performing exactly as booked.** In #1336,
   booked 30 GPU-h / 20 cells = 1.5 GPU-h/cell; at one cell per GPU that is a
   1.5 h per-cell wall, i.e. 3x the row's own 30-min abort threshold. The run
   aborts at cell 1 while performing to spec.
2. **It can silently breach the auto-approve GPU-hour cap.** If the
   basis-derived figure is the true one, #1336's honest total is ~125 GPU-h
   against a 100 GPU-h autonomous auto-approve cap — so an unverified stated
   total would have auto-dispatched past the cap.

In #1336 the inconsistency was caught only by the Phase 1.5 fact-checker and
two Phase 2 critic lenses reading the row's prose. That is exactly the kind of
arithmetic a regex + comparison can do deterministically on every plan.

## Proposed check (WARN-only)

Add to `scripts/verify_plan.py`, in the §9 compute-table family alongside the
existing `c32_fit_basis_grounding`:

**Arm A — basis-vs-booked.** For each §9 compute row, extract GPU-hour and
wall-hour tokens from the row's basis/justification text (the
`~ N GPU-h` / `=> N GPU-h` / `~ N wall-h` forms). WARN when a basis-derived
figure exceeds the row's booked column by more than ~2x AND no reconciliation
token appears in the same cell (a short allowlist: `pilot-gated`, `includes
generation`, `naive-serial (superseded by <x>)`, `x-way`, or an explicit
`reconciled:` marker).

**Arm B — abort-vs-booked.** When a row states a per-cell abort threshold
(`> N min/cell`, `> N h/cell`), compare it against the booked per-cell cost
implied by `booked_gpu_h / n_cells` adjusted for the stated parallelism. WARN
when the booked per-cell wall EXCEEDS the abort threshold — the gate would fire
on a nominal run.

WARN-only, not FAIL: a legitimately superseded naive-serial figure is a normal
thing to show in a basis, and the reviewer lens stays the binding arm. The
value here is that the WARN puts the arithmetic in front of the planner and the
critics every round instead of depending on a lens noticing prose.

## Acceptance criteria

1. New check registered with a stable id, WARN-only, appearing in the `--json`
   `checks` list with the standard `{id, name, status, detail}` shape.
2. Fixture reproducing the #1336 `EXT_off` row shape (90 derived / 30 booked /
   30-min abort) WARNs on both arms.
3. Fixture with a reconciliation token present does NOT warn (Arm A).
4. Fixture with a booked per-cell wall comfortably under its abort threshold
   does NOT warn (Arm B).
5. Regression: the check produces no new WARNs on the existing plan corpus
   beyond the ones it is designed to catch — run it across `tasks/**/plans/*.md`
   and report the hit list, so a noisy regex is caught before it lands.
6. `.claude/rules/plan-compute-sizing.md` gains a pointer to the new check id in
   its § Per-cell fit phases section.

## Provenance

Surfaced as a prose follow-up by the Methodology critic lens during the #1336
plan v16 Phase 2 review (2026-08-07), and independently by the Statistics lens
("mechanizable: yes — WARN-grade: parse §9 basis strings for a derived
'=> N GPU-h' that exceeds the row's planned_gpu_h by >2x with no reconciliation
token in the same cell"). Filed by the #1336 orchestrator per
`.claude/rules/workflow-fix-on-bug.md` (surfaced-prose follow-ups auto-file).

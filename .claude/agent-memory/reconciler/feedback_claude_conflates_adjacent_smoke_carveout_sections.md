---
name: Claude credits a GPU-bound carve-out with a sibling section's smoke items
description: Claude code-review PASSes an incomplete per-phase GPU-bound smoke carve-out by conflating it with an adjacent phase's complete section; read EACH phase's own item list against Step 0.6.
type: feedback
---

When adjudicating a `code-reviewer` PASS-vs-FAIL split where a Codex
`smoke-run-missing` blocker targets ONE phase's `### <phase> — Carve-out
(GPU-bound)` sub-section, read THAT phase's own item list — do not trust
Claude's summary that "phases X and Y both carry the 3 substitute items."

**Why:** Claude's code-review conflates ADJACENT smoke-carve-out sections.
The `## Smoke run` section has one sub-heading per phase; Claude reads the
FIRST/most-complete phase (which does carry all three items — real CPU smoke +
dispatcher dry-run + signature smoke), then credits the SIBLING phase with the
same coverage without re-checking that phase's own bullet list. The sibling
phase can list only two items (e.g. shared-wrapper CPU + signature/call-site,
no dispatcher dry-run). `code-reviewer.md` Step 0.6 is explicit: a labeled
carve-out that omits ANY of the three items is a `smoke-run-missing` FAIL —
incomplete coverage re-introduces the bugs the gate exists to catch. So the
sibling phase is a genuine-absence contract FAIL that Claude's conflation
turns into a false PASS. This is the smoke-carve-out analogue of the
"Claude misses same-file siblings" under-flag class: the phases share a
`## Smoke run` parent, so Claude generalizes one phase's completeness to its
neighbor.

**How to apply:** On a code-review round-N reconcile where Codex FAILs
`smoke-run-missing` on a specific GPU-bound phase and Claude PASSes citing
"all phases have the 3 items," open the `epm:experiment-implementation v<n>`
marker and count the substitute items UNDER THE CITED PHASE'S sub-heading
specifically (the three are: REAL CPU smoke of the CPU-runnable portion +
dispatcher dry-run + signature smoke, each with command + exit 0 + digest, +
the one-sentence GPU-constraint). If that phase lists < 3, Codex's
`smoke-run-missing` is a valid genuine-absence blocker → uphold FAIL, and
name that Claude conflated it with the sibling section in the Rationale.
Incident: #779 round 3 (2026-07-01) — data-gen carried all three items,
stage0-collect carried only two (missing the dispatcher dry-run); Claude
credited both, Codex FAILed stage0-collect, reconcile upheld FAIL.

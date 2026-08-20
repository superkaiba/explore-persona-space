---
name: shared-artifact-reader-enumeration
description: Recurring EPS bug class — a guard added at ONE reader of a shared gate/report JSON while sibling readers stay unguarded; verify by filename grep + write-vs-read discrimination
metadata:
  type: feedback
---

When a diff adds a validation/guard on a shared on-disk artifact (a
`gates/*.json` report, freeze record, recalibration file), enumerate EVERY
reader of that filename repo-wide (`grep -rn "<filename>" scripts/ src/`)
and classify each hit as read / write / name-list / help-text before
crediting the fix. A guard applied at one reader while siblings read the
same file unguarded is the dominant recurring defect class in this codebase
(issue #2389 rounds 3, 4, and nearly 5 — three consecutive rounds).

**Why:** these scripts share gate artifacts across phases, engines (HF vs
vLLM legs), and forked sibling scripts; the natural fix location (the site
the reviewer flagged) is rarely the only consumer.

**How to apply:** the strong fix shape is a CHOKEPOINT — move the guard
inside the single accessor function and verify no direct `json.loads` of
the path survives (a bare `path = cfg.gates_dir / "<name>"` local can be
write-only: check its uses). Same-named artifacts in a different namespace
(e.g. the judge's own `pilot_gate_report.json` vs the run's) are NOT
bypasses — check which gates_dir they resolve under. Also sweep the plan's
declared consumer scope: an unguarded sibling OUTSIDE the plan scope is a
Minor (defense-in-depth), not a blocker.

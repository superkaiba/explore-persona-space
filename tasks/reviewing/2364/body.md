---
title: 'workflow-fix: no review surface catches a test suite that monkeypatches the
  production constant its acceptance criterion depends on'
kind: infra
tags: []
created_at: '2026-08-18T09:00:33Z'
has_clean_result: false
workflow: v1
---
---
kind: infra
---

# workflow-fix: no review surface catches a test suite that monkeypatches the very production constant its acceptance criterion depends on

**Provenance:** caught by the **Codex** statistics twin during
`/adversarial-planner` Phase 2 on task #2360 (2026-08-18) as its Must-Fix 1.
The Claude statistics critic, the Claude and Codex methodology critics, the
Claude and Codex alternatives critics, and the consistency-checker all read the
same plan and did NOT flag it — 1 of 6 reviewers, i.e. currently a coin flip.

## What happened

Plan #2360 v2 proposed a preflight check driven by two module-level constants
(`LOAD_BEARING_DISTS`, `DEEP_IMPORT_MODULES`) and a 15-test suite. But:

- tests 1–4 **replaced** `LOAD_BEARING_DISTS` with a fixture list;
- test 5 **replaced** `DEEP_IMPORT_MODULES` with a fixture list;
- the only test touching the real constants pinned call ORDERING via a source
  substring;
- the sole check exercising the production constants live was a pod validation
  phase marked "optional" (filed separately).

Net result, in the Codex twin's words: both shipped constants could omit the
incident package — **or be entirely empty** — and every committed test would
still pass. The suite validated the FUNCTION and never the CONSTANTS, around a
fleet-wide check whose entire purpose was to detect one specific incident.

## Why this is a general shape, not a #2360 quirk

Any check of the form "iterate a curated production list, assert each element is
healthy" has this failure mode, and the natural way to unit-test it is exactly
the wrong way: monkeypatch the list to something small and deterministic. The
test becomes hermetic and fast, and simultaneously stops certifying the only
thing that matters — that the shipped list contains what the requirement says it
must.

The repo has many such curated lists: `LOAD_BEARING_DISTS`,
`tests/sparse_cones.txt`, `UPLOAD_PREFIX_CLOBBER_ALLOWLIST`,
`_MANAGED_PREFIXES`, `DEFAULT_AUTO_LANE_ORDER`, `LOAD_BEARING_IMPORTS`,
`_DEFAULT_TIME_BUDGETS_HOURS`, judge/behavior rosters, `DRIFT_DOMAINS`.

## Relation to `.claude/rules/smoke-blind-spots.md` (adjacent, not the same)

That rule covers three mechanisms by which a smoke PASS certifies less than it
appears to — substituted implementation, downgraded gate, production-only code
path — and its trigger grammar is about `smoke`-conditional branches in
PRODUCTION code. The shape here is in TEST code: a fixture that substitutes a
production CONSTANT. Its `code-reviewer` Step 0.71 scanner looks for
smoke-conditional substitution in scripts, so it does not see a `monkeypatch.setattr`
on a module constant in a test file. The blind-spot-ENUMERATION idea transfers
cleanly; the trigger does not.

## Candidate fix surfaces (implementing session picks)

1. **`.claude/rules/smoke-blind-spots.md` — extend with a fourth mechanism**
   ("substituted production constant") plus its own trigger grammar for test
   files, and require the enumeration to name it. Cheapest, and reuses an
   existing rule the reviewers already load.
2. **`code-reviewer` (Step 0.7x) + its Codex twin's copy-list** — a FAIL when a
   diff adds a test that monkeypatches a module-level constant which an
   acceptance criterion references, unless the same diff also adds a test
   pinning that constant's required contents. Tag it substantively so the
   mechanical-contract strip cannot drop it.
3. **`scripts/workflow_lint.py` — a best-effort AST scan** for
   `monkeypatch.setattr(<module>, "<CONST>", ...)` / `setattr` on an
   ALL_CAPS module attribute in `tests/`, cross-referenced against whether any
   test in the same file asserts that constant's contents. WARN-only; disclose
   the known false negatives (indirect fixtures, parametrized names, `patch.object`).
4. **`.claude/rules/critic-lens-reference.md`** — a plan-time lens item, since
   the defect is visible in the plan's own test list before any code exists
   (that is where it was caught here).

The natural instrument often serves double duty: a static
completeness/subset check over the constant makes the coverage relationship a
TEST rather than a convention, which is how #2360 resolved it (an unmapped entry
fails; empty constants fail; required members pinned).

## Acceptance

- A diff (or plan test list) that replaces a production constant in every test
  touching it, with nothing pinning the constant's required contents, is
  flagged — naming the constant and the acceptance criterion that depends on it.
- A test suite that monkeypatches a constant AND separately pins its required
  members is NOT flagged (that is the correct pattern).
- Monkeypatching a constant no acceptance criterion depends on is NOT flagged.
- If the lint arm is built: WARN-only, with false negatives disclosed in the
  check's own docstring, per the convention the other heuristic checks follow.

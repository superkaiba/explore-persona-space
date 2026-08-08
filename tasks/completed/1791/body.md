---
title: 'daily-fix: --map-files misuse OSError before #1613 guard'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b1c4a980a273
- daily-auto-filed
created_at: '2026-07-29T07:06:20Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): `--map-files <source-file>`
  misuse crashes with an unhandled `OSError: [Errno 36] File name too long` from `(work_root
  / f).exists()` (L1568) when any content line of the mis-passed file exceeds NAME_MAX,
  instead of reaching the graceful #1613 "looks like a source file, not a path-LIST
  file" diagnostic (which fired correctly for a short-lined source file in the same
  session).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step C parked-candidate sweep (2026-07-28) from TWO same-bug parked candidates on the same file: task #1762 (ts 2026-07-28T15:16:59Z, fp eef8cf00a9f3, implementer round 1) and task #1764 (ts 2026-07-28T16:28:05Z, fp b1c4a980a273, code-reviewer round 1). Both describe the same failure — passing a payload/markdown file (not a newline-delimited path list) to `select_step9c_tests.py --map-files` crashes with an unhandled `OSError: [Errno 36] File name too long` from a per-line `.exists()` probe, instead of reaching the #1613 misuse diagnostic — with two complementary fixes (a content-shape pre-check and an OSError-tolerant existence helper).

## Goal

Make `--map-files` misuse with a non-path-list payload reach the graceful #1613 diagnostic instead of a raw traceback: guard the per-line existence probes with an OSError/ValueError-tolerant helper, and/or pre-detect non-path-shaped lines and exit 2 with the existing FILE-of-paths message.

## Workflow gap

- **Bug observed:** `--map-files <source-or-markdown-file>` crashes with `OSError: [Errno 36] File name too long` from `(work_root / f).exists()` when any content line exceeds NAME_MAX; the #1613 guard (which exit-2s cleanly for `.py`/`.sh` payloads) is defeated by its own probe — the existence check raises before the heuristic can report.
- **Why it is a workflow gap:** the #1613 misuse guard exists precisely for this input class; the operator gets a raw traceback instead of the designed diagnostic naming the FILE-of-paths contract.
- **Confidence (emitters):** medium (#1762) / high (#1764)
- verified-at-filing: `grep -n '_safe_exists' scripts/select_step9c_tests.py` → 0 hits (absence of the proposed helper), and code read confirms two unguarded per-line probe sites — `if (work_root / f).exists():` (~L1434 region) and `if any((work_root / f).exists() for f in files if not f.startswith("/")):` (~L1568 region, immediately before the #1613 `.py`/`.sh` diagnostic) (2026-07-29 UTC). Landed-fix history check: `git log --oneline --since='7 days ago' -- scripts/select_step9c_tests.py` → no commit addressing the OSError path.

## Proposed change (candidate diff sketch — refine in planning)

```diff
+ def _safe_exists(p: Path) -> bool:
+     try:
+         return p.exists()
+     except (OSError, ValueError):
+         return False  # unstat-able content line (e.g. >NAME_MAX) is not a repo path
- if any((work_root / f).exists() for f in files if not f.startswith("/")):
+ if any(_safe_exists(work_root / f) for f in files if not f.startswith("/")):
  (same guard at the L1434 sibling)
```

Optionally ALSO extend the #1613 guard with a cheap non-path-shape line check (e.g. `len(line) > 512 or line.startswith(("- ", "# ", "**"))` → exit 2 with the existing diagnostic), per the #1762 sketch — the planner picks the combination.

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py` (both `.exists()` probe sites + the #1613 guard)
- Add a regression test: `--map-files` fed a markdown payload exits 2 with the diagnostic, no traceback.

## Constraints / invariants

- Exit-code contract of #1613 preserved (exit 2 + diagnostic naming the FILE-of-paths contract).
- Workflow-surface only; ruff passes; recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: scripts/select_step9c_tests.py
- fingerprint: b1c4a980a273

<!-- workflow-fix-candidate v1 -->
target_file: scripts/select_step9c_tests.py
bug_observed: `--map-files <source-file>` misuse crashes with an unhandled `OSError: [Errno 36] File name too long` from `(work_root / f).exists()` (L1568) when any content line of the mis-passed file exceeds NAME_MAX, instead of reaching the graceful #1613 "looks like a source file, not a path-LIST file" diagnostic (which fired correctly for a short-lined source file in the same session).
why_workflow_gap: The #1613 misuse guard exists precisely for this input class but is defeated by its own probe — the existence check raises before the heuristic can report, so the operator gets a raw traceback instead of the designed diagnostic.
proposed_change: Guard the per-line existence probes (L1434, L1568 — both instances of the class) with an OSError/ValueError-tolerant helper that treats unstat-able lines as non-paths.
confidence: high
related_task: #1764
<!-- /workflow-fix-candidate -->

(Second same-bug park, task #1762, fp eef8cf00a9f3, ts 2026-07-28T15:16:59Z — deduped into this filing; its routed-record on #1762 names this task. Its complementary content-shape pre-check sketch is included above for the planner.)

---
title: 'daily-fix: verify_plan c38 false-positives on a backticked p'
kind: infra
tags:
- wf-fix
- wf-fix-fp:72f7ca80c321
- daily-auto-filed
created_at: '2026-07-27T07:19:46Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): the c38 tail split truncates
  the line at the first backtick, so a path-scoped pytest node id is never seen and
  the line classifies as unscoped, producing a false WARN that agents then hand-adjudicate'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 2 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Evaluate `_C38_PYTEST_SCOPED_RE` against the rest of the plan line, not only the
backtick-truncated tail, so a scoping path or node id written inside backticks after the
word "pytest" stops producing a false `c38_exit0_repo_wide_baseline` WARN.

## Workflow gap

- **Bug observed:** `_c38_repo_wide_cmd` splits the text following a `pytest` occurrence at
  the first backtick (`_C38_TAIL_SPLIT_RE = re.compile(r"[\`#&|)\n]")`), so a line whose
  scoping node id lives inside the backticks that follow — the ordinary prose shape
  `Concrete pytest node id: \`tests/test_x.py::test_y\`` — is classified
  `pytest (no path scope)` and WARNs, although the scope is present on the line.
- **Why it is a workflow gap:** `verify_plan.py` is the mechanical plan pre-pass every
  Phase-1.5.0 run consumes, and its WARNs are copied verbatim into the fact-checker and
  critic briefs, so a false WARN is paid for by every downstream reviewer on the plan.
- **Confidence (emitter):** high
- verified-at-filing: reproduced against the CURRENT working tree at compose time —
  `uv run python -c "import sys; sys.path.insert(0,'scripts'); import verify_plan as m;
  print(m._c38_repo_wide_cmd('- Acceptance: mapper exits 0. Concrete pytest node id:
  \`tests/test_select_step9c_tests.py::test_map_files\`.'))"` → **`'pytest (no path scope)'`**
  (the false positive), while the same probe on
  `'- Acceptance: uv run pytest tests/test_select_step9c_tests.py -q exits 0.'` → **`None`**
  (correctly scoped). Symbol locations confirmed per-target in the named file:
  `grep -n '_C38_TAIL_SPLIT_RE\|_C38_PYTEST_SCOPED_RE\|def _c38_repo_wide_cmd' scripts/verify_plan.py`
  → **7 hits in `scripts/verify_plan.py`** (L6692 tail-split regex def, L6697 scoped-regex
  def, L6702 function def, L6711/L6719/L6723 the three tail splits, L6720 the pytest-arm
  scoped test). The pytest arm itself is L6718-6721 —
  `grep -n 'for m in _C38_PYTEST_OCC_RE\|return "pytest (no path scope)"' scripts/verify_plan.py`
  → **2 hits (L6718, L6721)**.
  Landed-fix check: `git log --oneline --since='7 days ago' -- scripts/verify_plan.py` → 4
  commits, newest `7809272e2f` ("5 WARN/FAIL surface fixes + c42 commit-SHA check") — the
  false positive still reproduces on the post-commit tree, so no landed fix covers it.
  (2026-07-26)

**Context binding — one premise corrected.** The mined report states the over-trigger
class is not in the check's disclosed residual list. Read at compose time, the docstring
of `check_exit0_repo_wide_baseline` (`scripts/verify_plan.py:6729`) DOES carry
"Disclosed over-trigger residual: a prose pytest mention with a non-scoping tail on an
assertion-bearing line", which arguably names this class in general terms. What the
disclosure does not name — and what the planner should treat as the actual finding — is
the specific mechanism: the scoping token IS on the line and is hidden purely by the
backtick split, so the class is mechanically fixable rather than an inherent heuristic
limit. The disclosure is documentation of a known limitation, not an implementation of
the fix.

## Evidence

- Session `7df6ce4c`, 2026-07-26T09:40:10Z and again at 09:49:39Z after a plan patch:
  `"WARN: c38_exit0_repo_wide_baseline ... detail: exit-0/green asserted on pytest (no
  path scope) with no plan-time baseline or scoping"`, with the offender line recorded as
  `"'uv run pytest tests/test_select_step9c_tests.py -q' — path-scoped to a single file,
  false positive"`. Both the orchestrator and the Methodology critic spent adjudication
  text on it, and it was carried into the fact-checker + critic briefs.
- Session `2b779905`, 2026-07-26T11:00:03Z and 11:09:55Z: a c38 WARN (the arm-A
  `workflow_lint.py` variant, a genuine WARN rather than this false positive) survived
  plan v1, v2 and the FINAL v3 with identical text —
  `"PASS n_fail=0 n_warn=1 … c38_exit0_repo_wide_baseline :: … exit-0/green asserted on
  workflow_lint.py (no --check- scoping) with no plan-time baseline or scoping named on
  the line"` — while the v2 changelog claimed the criterion had been narrowed. This is
  the standing disposition of a c38 WARN: carried, not cleared. A check whose WARNs are
  routinely carried rather than resolved cannot afford avoidable false positives.
- Measured cost: two agents' adjudication text on a false WARN in session `7df6ce4c`,
  forwarded into two downstream briefs.

## Proposed change

- `scripts/verify_plan.py::_c38_repo_wide_cmd`, pytest arm (currently L6718-6721) — before
  returning `"pytest (no path scope)"`, evaluate `_C38_PYTEST_SCOPED_RE` against the
  remainder of the line (`line[m.end():]`), not only the backtick-truncated `tail`; return
  the label only when NEITHER the tail nor the rest of the line carries a `::` node id, a
  `tests?/...py` path, or a `-k` filter.
- Weigh and record the one behavioural trade-off in the plan: the deliberate,
  fact-checker-confirmed reading that an EMPTY arg tail (bare `pytest` plus an assertion
  word) classifies repo-wide must survive, and a line that asserts green on an UNSCOPED
  pytest run while separately mentioning some unrelated `tests/...py` path elsewhere would
  newly go quiet. Scope the rest-of-line search accordingly (e.g. bounded to the same
  sentence / clause) if the plan judges the widened search too permissive.
- Update the `check_exit0_repo_wide_baseline` docstring: the disclosed over-trigger
  residual narrows to whatever the fix does not cover, and the calibration note records the
  re-measured WARN/FP counts.
- Add a pin test in `tests/test_verify_plan.py` covering both shapes: the backticked
  node-id prose line must not WARN; a bare unscoped `pytest` assertion line must still WARN.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- `tests/test_verify_plan.py` (new pin test)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 72f7ca80c321

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: G-P7, F-P11.

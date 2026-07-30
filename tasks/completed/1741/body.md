---
title: 'daily-fix: sweep park predicate misses URGENT-PARK notes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6a51b2097840
- daily-auto-filed
created_at: '2026-07-28T06:40:32Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): the #1718 URGENT-PARK candidate
  note contains no parked token in any accepted form, so sweep() never enumerates
  it and both the watcher urgent-park router and nightly Step C are blind to a mechanically-routable
  main-red park'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27. The #1681 urgent-park fast path failed
end-to-end on its first real firing: task #1718 parked a fully
mechanically-routable candidate (`urgency: main-red` + `failing_test:` +
`wf_fix: true`, fp `06bc0203d759`) at 2026-07-27T14:38:28Z, and it sat
unrouted for ~16 h while main stayed red — the watcher's
`urgent_wf_park_pass` and the nightly Step C sweep were BOTH blind to it.

## Goal

Make `scripts/sweep_parked_wf_candidates.py`'s park predicate recognize an
urgent-fast-path park so the #1681 router can actually route the class it was
built for.

## Workflow gap

- **Bug observed:** the #1718 candidate note leads with "URGENT-PARK
  workflow-fix candidate raised from #1718 ..." and contains NO occurrence of
  the word "parked" anywhere, so `_row_is_parked()` returns False on all four
  accepted paths (leading `parked`, `routed: parked`, mid-note declaration,
  structured routed field). `sweep()` therefore never enumerates it as a
  candidate; `urgent_wf_park_pass` filters `sweep()["candidates"]` for the
  `urgency: main-red` token and so never sees it either — the router's own
  input pipeline drops exactly the urgent class it exists to route.
- **Why it is a workflow gap:** the #1681 grammar
  (`.claude/rules/workflow-fix-on-bug.md` § Recursion guard "Urgent fast
  path") prescribes the three in-block fields but never requires a "parked"
  token in the note prose, while the enumerator requires one — the emitter
  and the enumerator disagree on the park surface. #1718 followed the
  documented grammar and was still lost.
- **Confidence (emitter):** high
- verified-at-filing: probed 2026-07-28T06:4xZ —
  `uv run python` importing `sweep_parked_wf_candidates` and calling
  `_row_is_parked(<the actual #1718 row>)` → False; `_PARKED_LEAD_RE` /
  `_PARKED_ROUTED_RE` / `_PARKED_MIDNOTE_RE` all no-match;
  `re.finditer(r"(?i)parked", note)` → zero hits in the 4574-char note.
  Sweep run tonight returned `candidates: []`. State files
  `~/.eps-autonomous/urgent-wf-park-router.json` and
  `.claude/cache/urgent-wf-park-events.jsonl` do not exist (the pass has
  never routed anything). unverified hypothesis — verify at plan time: the
  cheap `_urgent_park_candidate_gate` mtime+substring gate may PASS on the
  token while the sweep still drops the row, making the gate's positive
  result unobservable in the pass log (no log line was emitted today).

## Proposed change (candidate diff sketch — refine in planning)

In `scripts/sweep_parked_wf_candidates.py::_row_is_parked` (or the note
classification just above it), accept as PARKED any
`epm:workflow-fix-candidate` row whose note contains a formal candidate block
carrying the `urgency: main-red` token (`URGENT_WF_PARK_TOKEN`) — and/or an
`URGENT-PARK` lead token — since an urgent fast-path park is a park by
definition. Belt-and-suspenders: add one line to
`.claude/rules/workflow-fix-on-bug.md` § "Urgent fast path" telling the
parking session to lead the marker note with `parked (urgent): ...` so
future emitters match the existing predicate too. Add a regression test
pinning the #1718 note shape (`tests/` sibling of the sweep's tests).

## Scope / surfaces

- Primary target: `scripts/sweep_parked_wf_candidates.py`
- Secondary: `.claude/rules/workflow-fix-on-bug.md` (emitter guidance),
  `scripts/autonomous_session_watch.py` only if the router needs a matching
  tweak (it consumes `sweep()`), plus tests.

## Constraints / invariants

- Read-only enumerator semantics unchanged for non-urgent rows; suppression
  rules 1-3 untouched.
- `workflow_lint.py` no-flags run passes; the sweep's existing tests pass.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST
  NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/sweep_parked_wf_candidates.py
- fingerprint: 6a51b2097840
- origin: /daily 2026-07-27 problem sweep — the #1718 lost-park incident
  (candidate ts `2026-07-27T14:38:28Z`, self-declared fp `06bc0203d759`).

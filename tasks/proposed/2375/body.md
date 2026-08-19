---
title: 'Pre-existing ordering flake on main: test_fig_two_by_two_quadrant_labels fails
  under the Step 9c co-selection prefix (empty mpl title) while passing single-file'
kind: infra
tags:
- wf-fix
- ordering-suspect
created_at: '2026-08-18T20:27:10Z'
has_clean_result: false
origin_prompt: 'Mandated by step9c_baseline compare ORDERING WARN follow-through in
  the #2168 test-verdict (2026-08-18): node passes single-file pristine but reproduces
  under the gate''s 406-file co-selection prefix on pristine main.'
workflow: v1
---
# Pre-existing cross-module ordering interaction on main: tests/test_issue2162_figures.py::test_fig_two_by_two_quadrant_labels fails under the Step 9c co-selection prefix (empty matplotlib axes title) while passing single-file

kind: infra

## Goal

Root-cause and fix the main-side test-ordering interaction that makes `tests/test_issue2162_figures.py::test_fig_two_by_two_quadrant_labels` fail when executed after the Step 9c gate's co-selection prefix, while passing single-file and in isolation. Until fixed, any issue whose gate set includes this file can red on it and burn a compare/pristine cycle.

## Measured evidence (#2168 Step 9c, 2026-08-18 — mechanical, not prose)

- Gate #2 (474-file set, 1h47m): the test FAILED with `assert '1 cells without probe rows omitted' in ''` — the spy on `F._save` captured `ax.get_title() == ''` on `fig.axes[0]`, i.e. the title was never set (or axes[0] was not the scatter axes) under full-suite state.
- `step9c_baseline.py compare --run-pristine --max-paired-files 500`: the node PASSES the single-file pristine oracle on a scratch main tree (568440568f7e) but REPRODUCES under the gate's own 406-file co-selection prefix run ON PRISTINE MAIN — compare's ORDERING WARN classified it `ordering_suspect`: a pre-existing cross-module interaction on main, not branch-linked (the #2168 diff is except-tuple widenings; the test's imports are untouched).
- Compare JSON: /tmp/step9c-compare-issue-2168.json (ephemeral); durable record in #2168 events.jsonl (epm:test-verdict marker, 2026-08-18).

## Suspected class

matplotlib global-state pollution (rcParams / style / figure-registry) by some predecessor in the prefix — the same broad class as the known tick_triage module-level-NOW flake (#2369) but a distinct mechanism and file. The contaminating predecessor is somewhere in the 406-file prefix; #2021 measured a distance-49 contaminator for a prior interaction of this shape, so nearest-N windowing is not a valid search shortcut — bisect the prefix instead.

## Acceptance criteria

- The contaminating predecessor (or the figures module's vulnerable global-state read) is identified and fixed so the test passes under the full co-selection prefix.
- A regression guard at whichever grain fits (test isolation fixture, module-state reset, or the figure function setting its title unconditionally).
- No weakening of the test's assertions.

## Provenance

Mandated by the compare ORDERING WARN follow-through ("route a workflow-fix candidate at the interaction itself — never a silent pass"), #2168 Step 9c verdict, 2026-08-18.

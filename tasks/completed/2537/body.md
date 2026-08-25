---
title: 'select_step9c_tests.py: feature-named consumer tests escape module mapping;
  over-wide sweep_scope label hides the gap'
kind: infra
tags:
- wf-fix
created_at: '2026-08-24T14:28:30Z'
has_clean_result: false
parent_id: 2336
origin_prompt: 'Surfaced by #2336 round-3 code review: two real consumer tests of
  clean_experiment_downloads.py were in neither the pin-sweep hit list nor the executed
  union.'
workflow: v1
---
## Goal

Close a mapping gap in `scripts/select_step9c_tests.py` that let two real
consumer test files of a changed module escape both the pin-sweep hit list and
the executed Step 9c union — so a future round touching the same module can
land with its consumers unrun and the gate still green.

## Evidence

Surfaced by the #2336 round-3 code review (plan batch 1), which migrated
`scripts/clean_experiment_downloads.py` among others.

The selector maps a changed source file to tests by an exact stem rule plus a
glob: for `scripts/clean_experiment_downloads.py` that glob is
`tests/test_*clean_experiment_downloads*.py`. Two test files are genuine
consumers of the writers changed in that round but match neither rule:

- `tests/test_janitor_tmp_scratch_sweep.py`
- `tests/test_vm_disk_guard_slurm_src.py`

Both exercise code paths through the migrated
`_ScratchVerdictCache.save` / `_save_slurm_src_escalation_state` writers. The
reviewer ran them by hand: **158 passed, rc=0**, so no defect shipped — but
they were absent from the selector output, absent from the round's claimed
pin-sweep hit list, and absent from the 128-file union the round actually
executed.

Two failure modes compounded, and the second is the more interesting one:

1. **Selector-glob gap.** The name-derived glob cannot see a consumer whose
   filename does not embed the module stem. Any module whose consumers are
   named after the FEATURE rather than the module has this hole.
2. **A pin-sweep supplement labelled wider than its realized scope.** The
   round's grep-only supplement was reported under `sweep_scope: repo-wide`,
   but its realized scope was narrower than that label — so the label itself
   read as coverage. This is the dangerous half: a truthful narrow label would
   have prompted a wider grep, whereas an over-wide label suppressed the very
   check that would have caught the gap. Compare the standing rule that a
   dead or silent verification leg is INCONCLUSIVE and never push-clean — an
   over-labelled leg is worse, because it reads as positive evidence.

## Scope

1. Add a mapping-table entry (or an equivalent explicit consumer registry) so
   `scripts/clean_experiment_downloads.py` maps to both files above in
   addition to its glob matches. Prefer a general mechanism over a
   one-module special case if one is cheap — the class is "consumers named
   after the feature, not the module".
2. Audit for the same shape elsewhere: modules whose consumer tests are
   feature-named. A bounded sweep over the existing mapping table plus the
   `tests/` inventory is enough; an exhaustive proof is not required.
3. Make the `sweep_scope` label verifiable rather than declarative. Options
   worth weighing: have the selector emit the realized universe it actually
   searched, or require the reporter to state the grep command and its target
   set alongside the label so a narrower realized scope cannot hide behind a
   `repo-wide` claim.

## Acceptance

1. `printf '%s\n' scripts/clean_experiment_downloads.py > /tmp/f.txt && uv run
   python scripts/select_step9c_tests.py --map-files /tmp/f.txt` includes
   `tests/test_janitor_tmp_scratch_sweep.py` and
   `tests/test_vm_disk_guard_slurm_src.py`.
2. A test pins that mapping so it cannot silently regress.
3. `tests/test_select_step9c_tests.py` stays green, including its exact
   set-equality invariant checks.
4. If scope item 3 lands, a test pins that an over-wide `sweep_scope` label is
   detectable — or, if that is judged out of scope, the decision is recorded
   with its reason rather than dropped silently.

## Notes on routing

Not #2336's defect and not blocking it: #2336 round 3 passed review with the
two files run green by hand, and its later batches now carry an explicit
instruction to grep `tests/` genuinely repo-wide. The gap is in the shared
selector, which every session's Step 9c gate depends on, which is why it is
filed separately rather than folded into that task's remaining batches.

Related ledger row on #2336: `pin-sweep-supplement-scope` (CONCERN, round 3).

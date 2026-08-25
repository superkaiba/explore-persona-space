---
title: 'Step 9c paired oracle: add a SUFFIX-replay arm so later-sorting-file contamination
  is attributed mechanically'
kind: infra
tags: []
created_at: '2026-08-20T21:15:38Z'
has_clean_result: false
origin_prompt: 'Carried action item (1) from #2217 Step 10d: plan v2 leg 4 recommended
  file-separately, endorsed by the Alternatives critic; filed after #2217 merged.'
workflow: v1
---
---
kind: infra
workflow: v1
---

# Step 9c paired oracle: add a SUFFIX-replay arm so later-sorting-file contamination is attributed mechanically instead of surfacing as an unattributed NEW failure

## Goal

Close the documented residual blind class in the Step 9c paired-prefix oracle:
test-order contamination originating in a file that sorts AFTER the candidate is
present in the gate process but absent from every prefix replay, so the oracle
cannot reproduce it and `ordering_suspect` comes back empty. Today that class
surfaces as a fail-closed NEW classification against an innocent issue's gate,
and the cost is paid by whoever happens to be merging.

## Evidence

- The blind class is documented, not hypothetical:
  `.claude/skills/issue/SKILL.md` Step 9c 1d names it directly.
- #2059's gate hit it. A 239-file gate run failed
  `tests/test_issue1090_fu3_dispatcher.py::test_conv_context_is_wildchat_family`
  while the same file passed alone and passed in the single-file pristine-main
  oracle, so `step9c_baseline.py compare` classified the node NEW (fail-closed).
  The paired-prefix oracle over 34 prefix files did NOT reproduce it →
  `ordering_suspect: []`. The contaminator
  (`tests/test_issue1481_analysis.py`) sorts after the victim, so no prefix
  replay could ever contain it.
- Cost of the miss: a manual provenance-override on #2059, plus an entire
  diagnosis cycle on #2217 (which found the offender in one full-tree
  `--collect-only` sweep with a per-collector registry-delta plugin — exactly ONE
  offender out of 24,775 collected tests).
- #2217 shipped a mechanical recurrence guard for this class AT SOURCE
  (`tests/conftest.py` collection hooks + `tests/test_no_import_time_registry_mutation.py`),
  but its watch list is exactly `CONTEXTS` + `NEGATIVE_PANELS`. The guard's own
  docstring names the residual: a NEW module-level registry
  (`fu4.ROUNDS` / `RUNS_BY_ROUND`, trait registries, `columns.CONTEXTS`) must
  EXTEND the conftest hook or it is unwatched. The #703 env/logging
  contamination family is likewise outside a registry-key guard.
- Deliberately filed separately rather than ridden into #2217: the arm is an
  estimated ~150-300-line diff on `scripts/step9c_baseline.py`, which is 3,919
  lines and fleet-critical (every `/issue` session's gate runs it). Landing it on
  a test-hygiene fix would have broken one-commit-per-leg revertibility and put
  the whole fleet's gate at risk behind an unrelated change.

## Suggested approach

1. Add a SUFFIX-replay arm to the paired oracle: when a node classifies NEW and
   the prefix replay yields no `ordering_suspect`, replay the candidate together
   with the files that sort AFTER it (the complement of the prefix set), bisecting
   to attribute the offender.
2. Keep it bounded — the arm exists to attribute a failure the gate already saw,
   so it should trigger only on the NEW-with-empty-`ordering_suspect` path, never
   on every run.
3. Report the attributed offender in the `compare` output so the failing session
   gets the offender name instead of an unexplained NEW node.
4. Because the target is fleet-critical, the change wants its own smoke over a
   synthetic later-sorting contaminator (the #2217 incident reshaped is the
   natural fixture) plus a negative control proving a clean tree still classifies
   nothing.

## Acceptance criteria

- A synthetic later-sorting contaminator, replayed through the oracle, is
  ATTRIBUTED by name rather than returning `ordering_suspect: []`.
- A clean tree produces no new suspects and no change in verdict (no
  false-positive regression).
- `tests/test_step9c_baseline.py` covers both arms.
- The existing prefix-replay behavior is unchanged on every path that already
  attributes an offender.

## Provenance

Carried action item (1) recorded on #2217 before its Step 10d merge, filed after
that merge landed. Recommended file-separately in #2217 plan v2 §4 leg 4 + §11.6
and endorsed independently by the Alternatives critic, whose note was that an
unfiled Decision line is how this blind class lost its owner the first time.

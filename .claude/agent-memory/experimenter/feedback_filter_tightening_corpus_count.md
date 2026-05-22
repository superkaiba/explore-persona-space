---
name: filter-tightening-corpus-count
description: Round-N corpus-filter tightening that doesn't re-verify corpus-wide accepted-count count vs requested-N often shorts the request at runtime. Bounce-back code-class.
metadata:
  type: feedback
---

When a code-reviewer round bounces back a BLOCKER like "filter is too loose, admits transcript garbage", the implementer fix typically adds a stricter predicate (`_is_clean_question`-style: must-end-in-?, ≤N words, no transcript markers, etc.) and unit-test-style accept/reject assertions on a handful of hand-crafted strings.

What's almost always missing: re-running the filter on the actual corpus and checking that the **accepted-count remains ≥ the count requested in the config**. Issue #375 round-2 commit `26032ab2` tightened the LMSYS-tail filter under BLOCKER M-6, added smoke-test assertions on single strings, but didn't re-verify the 600-doc corpus would still yield the ≥180 the held_out config asks for. Filter accepted only 164 → script crashed in `phase_build_queries` 30s into the run.

**Why:** Tightening predicates is by definition a strict subset of acceptances; if the loose filter just barely cleared the request, any tightening can push you under. The unit-test assertions don't catch this — they're per-string, not per-corpus.

**How to apply:** When you (experimenter) see a launch crash in a "build-data" / "filter-corpus" phase with shape `extracted only N out of requested M`, AND the most recent commit on the issue branch is a code-review fix to the filter logic:

- This is a `failure_class: code` bounce-back, NOT a hot-fix. The remediation (relax filter, shrink config, or expand corpus) is a design decision the planner should make.
- Recommend in the `epm:failure` body that the implementer also move the audit-JSON write to BEFORE the `raise` so the next failure can be diagnosed without re-running.
- Recommend the implementer add a corpus-wide accepted-count regression test to the smoke suite, parameterized on the corpus path + `want` from config. This catches the same class of bug pre-launch.

Don't try to hot-fix even option A (config shrink) — it changes the planned eval design ("200 held-out queries" in plan §4.5 becomes "184"), which is a design decision, not an oversight.

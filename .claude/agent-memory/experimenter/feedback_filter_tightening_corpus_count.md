---
name: filter-tightening-corpus-count
description: A code-review round that tightens a corpus filter with per-string smoke asserts but no corpus-wide recount routinely shorts the config's requested N at runtime. Code-class bounce; recommend a corpus-count regression test.
metadata:
  type: feedback
---

When a reviewer bounce ("filter too loose") is fixed by a stricter predicate plus per-string accept/reject asserts, the missing step is re-running the filter over the ACTUAL corpus and checking accepted-count ≥ the config's `want`. Tightening is a strict subset — if the loose filter barely cleared the request, the tight one undershoots.

**Why:** #375 round-2 (commit 26032ab2) — the tightened LMSYS-tail filter accepted 164 of the ≥180 the held_out config required; crash in `phase_build_queries` 30s in.

**How to apply:** a launch crash shaped `extracted only N out of requested M` in a build-data phase, right after a filter-fix commit, is a `failure_class: code` bounce-back, NOT a hot-fix — the remediation (relax filter / shrink config / expand corpus) is a design decision (shrinking changes the planned eval design). Recommend the implementer (1) add a corpus-wide accepted-count regression test parameterized on corpus path + `want`, (2) move the audit-JSON write BEFORE the raise so the next failure is diagnosable without a re-run.

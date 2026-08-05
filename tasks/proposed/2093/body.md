---
title: curate near-ceiling agent-memory indexes (merge duplicates, retire superseded)
kind: infra
tags: []
created_at: '2026-08-05T19:58:53Z'
has_clean_result: false
origin_prompt: 'Surfaced on #1739: experiment-implementer MEMORY.md had grown past
  the lint FAIL threshold (~38KB) and was compacted in-round to 19,697 bytes / 118
  entries; experimenter (19,303) and reconciler (19,300) also sit just under the 20,000-byte
  WARN. Compaction bought headroom but the judgment-bearing merge/retire pass was
  deliberately not done unilaterally. The subagent''s ''24.4KB read limit / 1-in-3
  entries invisible'' mechanism is UNVERIFIED and flagged as a hypothesis, not a premise.'
workflow: v1
---
## Overview / Motivation

Filed from task #1739, 2026-08-05. Bounded curation work, NOT a missing-rule gap — the enforcement already exists and functioned.

Three per-agent memory indexes sit at or just under the lint's WARN ceiling, and the `experiment-implementer` index had grown to ~38 KB (over the FAIL threshold) before a #1739 subagent compacted it in-round. Compaction bought headroom; what it deliberately did NOT do is the judgment-bearing part — merging near-duplicate entries and dropping superseded ones — because dropping a lesson is a call that should not be made unilaterally mid-round.

## Goal

Curate the near-ceiling per-agent memory indexes down to comfortable headroom by MERGING near-duplicate rows and RETIRING superseded ones, preserving every still-live lesson and its link target.

## Measured state (at filing)

    experiment-implementer   19,697 bytes   118 entries
    experimenter             19,303 bytes    96 entries
    reconciler               19,300 bytes    79 entries
    critic                   16,041 bytes   102 entries
    analyzer                 15,980 bytes    68 entries
    implementer              11,412 bytes    49 entries

Lint thresholds (`scripts/workflow_lint.py`): `AGENT_MEMORY_INDEX_WARN_BYTES = 20_000`, `AGENT_MEMORY_INDEX_FAIL_BYTES = 24_000`, with a curation recipe in `check_agent_memory_index_size`'s docstring. The top three are within ~300-700 bytes of WARN, so the next few lessons on any of them trip it; `experiment-implementer` already went over FAIL once this round.

## What is NOT being claimed

The originating subagent framed this as "~1/3 of entries were invisible on every load" against a ~24.4 KB READ limit. I could not verify a read-truncation mechanism, and the constants I did verify are lint SIZE RATCHETS (WARN/FAIL), not a loader truncation point. So the load-time-invisibility mechanism is UNVERIFIED and should not be carried as fact into planning; treat it as a hypothesis to check (if a real read/truncation limit exists in the harness or the memory loader, that materially raises the priority and should be measured, not assumed). What IS verified: the sizes above, the thresholds, one over-FAIL excursion, and that the lint flags agent-memory rows (the #1739 no-flags lint run reported duplicate/sibling agent-memory findings).

## Proposed change (refine in planning)

1. Per near-ceiling agent (`experiment-implementer`, `experimenter`, `reconciler` first), audit index rows for (a) NEAR-DUPLICATES that should merge into one row pointing at one consolidated feedback file, and (b) SUPERSEDED lessons whose target is retired or whose rule has since landed in an always-on surface (`CLAUDE.md`, `.claude/rules/*`), which can be dropped with the pointer preserved in the consolidated entry.
2. Preserve every live lesson: a merge rewrites two rows into one that still reaches both bodies; a drop happens only when the lesson is genuinely superseded, and the planning round should list each proposed drop with its justification rather than batch-deleting to hit a byte target.
3. Verify after: `scripts/workflow_lint.py --check-agent-memory-index-size` (or the no-flags bundle) clean, every `[[link]]` / file reference in the surviving rows resolves, and entry-count-vs-file-count reconciles (no orphaned feedback file left unreferenced, no row pointing at a missing file).
4. If step 0 finds a REAL loader/read truncation limit, record the measured number and size the target against it instead of the lint WARN.

## Scope / surfaces

- `.claude/agent-memory/*/MEMORY.md` (index rows) and the `feedback_*.md` bodies they point at, for the near-ceiling agents.
- Read-only cross-check: `scripts/workflow_lint.py` thresholds; do NOT change the thresholds to dodge the work.

## Constraints / invariants

- Never drop a live lesson to hit a byte target; merging is preferred to dropping, and every drop is justified individually.
- One line per memory in the index, no memory CONTENT in the index (the standing memory-file convention).
- Do not raise `AGENT_MEMORY_INDEX_WARN_BYTES` / `FAIL_BYTES` as the fix.

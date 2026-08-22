---
name: gotchas-size-trim-recipe
description: How to shed bytes from gotchas.md when the 200KB WARN budget gate blocks — remaining archaeology classes, the duplicate-collapse lever, and the pin surfaces to re-check
metadata:
  type: reference
---

gotchas.md is already consolidation-hardened: by #2280 r3 the classic archaeology
(dates, session ids, commit shas, wall-times) yielded only ~1.5 KB total. The
big remaining levers, in yield order:

1. **Duplicate/near-subset entry collapses** — same-root-cause siblings written
   as two entries (e.g. rsync-in vs git-pull MooseFS stale-bytes; a short
   dotenv-heredoc bullet that was a strict subset of the full python-dotenv
   entry). Merge keeping EVERY signature/fix/#N. ~250-450 B each.
2. **Cross-entry repetition** — an entry restating the precedence ladder /
   reference impls of the entry directly above it (the #1902/#1336 SLURM pair,
   the vLLM reap recipe quoted by the EADDRINUSE sibling). Point at the
   canonical statement instead. ~100-250 B each.
3. **Round/job-id suffixes** (`#N rK`, `SLURM job NNNNN`, `fu3`-style) — strip
   rK/job ids freely, but KEEP `fuN`/`crash N`/`surface N` tokens that
   disambiguate multiple same-issue entries cross-referenced in-file.

**How to apply:** a /tmp python script with exact-string replacements +
`assert text.count(old) == expected` per edit, abort-before-write on any miss
(raw strings break on embedded `\"`/`\'` — use plain quoted strings there).

**Pin surfaces to re-check before committing a trim** (tests that assert live
gotchas.md substrings): `test_gotchas_finalization_entry.py` (PyGILState_Release
/ sys.exit(0) / phased-dispatcher / #1689), `test_issue_skill_agent_memory_no_lost_row.py`
("no-lost-row check" + "comm -13"), `test_issue2184_noport_wedge.py`
(RunPodNoPortWedgeError + CPU-LANE-DRY), `test_workflow_lint_stale_gotchas_pointers.py`
(every `#N` cited near a gotchas.md mention in OTHER files must stay in the
file — diff the before/after `#N` id-sets, require NONE lost). The no-flags
workflow_lint run needs a >500 s fence (rc=124 at 500 s under fleet contention).

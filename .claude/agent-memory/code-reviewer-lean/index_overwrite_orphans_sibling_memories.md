---
name: index-overwrite-orphans-sibling-memories
description: Rewriting MEMORY.md wholesale from the session's loaded snapshot drops index rows that concurrent siblings or sync commits added since spawn — the pointed-to files survive on disk but become invisible (never loaded). Append rows; before any full rewrite, ls the memory dir and index every existing file. (#2225 R2 g5)
metadata:
  type: feedback
---

Never rewrite MEMORY.md wholesale from the copy loaded at session spawn; append the new rows, or re-derive the full index from a fresh `ls` of the memory dir first.

**Why:** #2225 round-2 commit fabf9d2f19 (a round-1 sibling's housekeeping) replaced MEMORY.md with only its own 4 rows, orphaning 11 still-existing memory files (3 pre-existing rows + 8 rows a spec-freshness sync commit had added after the sibling's session spawned). MEMORY.md is the only always-loaded surface — an unindexed memory file is effectively deleted. The same lost-update shape as the #2015 stale-snapshot restore, one level up.

**How to apply:** when saving memories in a shared/versioned memory dir, (1) prefer adding a row to the existing index over regenerating it; (2) if a rewrite is genuinely needed, `ls` the dir and carry one row per existing file (pull hooks from each file's `description:`); (3) when reviewing an agent-memory housekeeping commit, diff MEMORY.md for DROPPED rows whose target files still exist — that is a regression, not cleanup. Related: [[sentinel-path-outside-drain-glob]] (same round's blocker family).

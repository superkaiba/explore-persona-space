---
name: rsync-include-set-review
description: Reviewing rsync include/exclude-set plans (slurm.py RSYNC_INCLUDE_PATHS) — enumerate constant consumers, replay measurements, index-derived allowlist beats bare include under live-tree overrides, git-ls-files collision invariants are gitlink-blind (#2212 r2)
metadata:
  type: feedback
---

Review pattern for plans widening/deriving a staging include set
(`backends/slurm.py` `RSYNC_INCLUDE_PATHS` / `RSYNC_EXCLUDE_PATTERNS`).
From #2212 round 2 (v3 APPROVEd after v1's REVISE).

**Why:** v1 widened to a bare `./data` on the false premise the rsync
source is always committed-only; `EPS_SLURM_LIVE_TREE_RSYNC=1`
(slurm.py ~L3199-3215) returns the LIVE tree, where `data/` carried
41 GiB untracked and the `*_dl/`+`store/` excludes caught only ~4%
(both measured, reproduced r2). v3's fix shape held: derive include
entries from the git index (`git ls-files data/` → top-level
components), keep `build_rsync_command` pure, compose at BOTH
`run_rsync_sync` and `verify_rsync_complete` (prepare calls both seams
with the same `rsync_src`, L3103/L3134 — divergent tuples silently
break the #1913 completeness gate), fail-loud on probe failure
(→ `BackendPrepareError`, same routing as `materialize_branch_src`).

**How to apply:**
1. Enumerate the CONSTANT's consumers yourself: `build_rsync_command`'s
   only production callers are `run_rsync_sync` + `verify_rsync_complete`;
   static importers = `scripts/verify_carryover_inputs.py::rsync_cover_set`
   (#1835 gate — needs a compensating root when an entry is dropped;
   over-approximation is safe because committed citations' top-level
   components are tracked by construction, and the #1915 fnmatch
   exclude matcher fails cheap-false-FAIL) + test fixtures.
2. REPLAY every measurement (`git ls-files data/ | wc -l`, blob sums,
   `du`, per-entry `--others` sweeps) — all seven of #2212 v3's numbers
   reproduced exactly; v1's "~208 GiB" was a stale doc carry.
3. Gitlink blind spot: a `git ls-files`-based exclude-collision
   invariant CANNOT see inside a gitlink (`external/open-instruct`)
   whose WORKING TREE ships as an include tree — `find` the shipped
   tree for colliding components too (empty at #2212; `docs/` already
   punctures it pre-existing).
4. Both rsync sources are git trees (scratch = `git worktree add
   --detach` in `materialize_branch_src`; override = repo root), so an
   in-dispatch `git ls-files` subprocess is sound; empty derivation →
   well-formed argv (other include entries remain).
5. The measured override residual (untracked-inside-tracked, 6 MB
   here) is checkout-local, not structural — require the docstring to
   frame it as measured, and the override's doc disposition per
   [[kill-criterion-grep-matches-override-branch]].

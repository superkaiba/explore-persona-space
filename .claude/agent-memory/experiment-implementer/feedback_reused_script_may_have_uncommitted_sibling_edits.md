---
name: A reused script may carry uncommitted sibling edits — build against HEAD, not the working tree
description: When reusing functions from a sibling script, the working-tree copy may have a parallel session's UNCOMMITTED change (a new arg/signature) that is NOT on main; depending on it silently breaks the new script against committed main. COMMIT-side twin — verify every hunk is yours before a path-scoped add in a shared worktree (#1491)
type: feedback
---

When you reuse functions from an existing sibling script, the on-disk
working-tree copy can already carry a DIFFERENT session's uncommitted
modification (e.g. a new `--layer` arg / a widened signature) that is NOT
committed to `main`. If your new script calls the modified signature, it
"works" in your smoke (the WT change is loaded) but will crash against
committed `main` once your commit lands — a partial dependency on unmerged
work, the built-but-stranded / reuse-hierarchy class.

**Why:** `git status` at session start does not always list every dirty
file, and imports load the working-tree copy, so a smoke passes on code
that isn't on `main`. (Incident #667 all-layer, 2026-06-30: the reused
`issue667_deltac_probe.load_store()`/`load_r_b()` were hardcoded to
`LAYER=14` on `main`; an unmerged sibling edit added a `--layer` arg. My
analysis called `deltac.load_store(layer)`/`load_r_b(b, layer)` — worked
in the smoke, would have broken against committed main.)

**How to apply:** before depending on a reused function's signature, diff
the file against HEAD (`git diff <file>`; `git show HEAD:<file> | grep 'def
<fn>'`) to confirm the signature you're calling is the COMMITTED one, not a
working-tree-only variant. If the arg you need exists only in the
uncommitted change, do NOT build on it — INLINE a self-contained
equivalent that calls only the committed, layer/param-agnostic helpers
(reuse the compute, not the not-yet-merged loader). Verify the inlined
version bit-matches the original on a shared input. Never commit the
sibling's uncommitted change as a side effect of your own work — it isn't
yours.

**COMMIT-side twin (#1491, 2026-08-04):** the sweep also happens on a file
you OWN. In a shared worktree during a concurrent multi-agent round, a
path-scoped `git add <your file>` stages whatever content the working tree
holds for that path — including a teammate's uncommitted edit made to your
file while your round was live (on #1491 the team lead's `cand_cis` fix
rode my retry-routing commit `58d064a5c4`, leaving a commit message that
under-describes its diff; unfixable post-push without a force-push).
**How to apply:** immediately before staging any owned path in a shared
worktree, run `git diff -- <path>` and verify EVERY hunk is one you wrote
this round; an unrecognized hunk → stop and resolve with the owner/lead
before committing. `git status` showing only "expected" filenames is not
enough — the foreign edit hides inside a file you expect to be modified.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Reused script may carry uncommitted sibling edits](feedback_reused_script_may_have_uncommitted_sibling_edits.md) — a reused helper's WORKING-TREE copy can have a parallel session's UNCOMMITTED signature change (new `--layer` arg) NOT on main; calling it "works" in smoke but breaks against committed main. Diff against HEAD before depending on a signature; inline a self-contained equivalent using only committed helpers; never commit the sibling's dirty change. #667 alllayer.

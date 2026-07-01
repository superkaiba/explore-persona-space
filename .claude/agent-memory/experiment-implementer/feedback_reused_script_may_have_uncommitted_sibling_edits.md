---
name: A reused script may carry uncommitted sibling edits — build against HEAD, not the working tree
description: When reusing functions from a sibling script, the working-tree copy may have a parallel session's UNCOMMITTED change (a new arg/signature) that is NOT on main; depending on it silently breaks the new script against committed main
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

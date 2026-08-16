---
name: seam-parity-fake-extension-recipe
description: Review recipe for commits that extend sibling test fakes after a seam widening (new API member probed on an old path) — 4 checks incl. module-level probe-cache collision
metadata:
  type: feedback
---

When a round widens a seam (e.g. `stage_hub_prefix` grows an `api.file_exists`
probe) and a commit extends PRE-EXISTING sibling test fakes to match, run four
checks (#2321 R1 g7):

1. **Fails-pre-fix, run not read** — extract the parent-commit test body
   (`git show <sha>^:tests/<file>.py` → untracked `tests/test__probe.py`),
   run it against HEAD src, demand a LOUD failure (AttributeError), delete the
   probe file. Certifies the extension is load-bearing, and that the seam's
   retry wrapper does not absorb the missing-member error.
2. **Permissive-return semantics** — the fake's return (e.g. `False`) must be
   the fixture-TRUE answer routing to the branch the test always exercised;
   diff must contain ZERO assertion changes.
3. **Fake sweep** — enumerate EVERY fake class reaching the seam in each file
   (`grep -n "class.*Api"`), not just the ones the diff touched; a missed one
   fails full-file runs, a pre-existing conformant one (e.g. `_TreeApi`) needs
   no change.
4. **Module-level probe-cache collision** — a probe result cached module-side
   under a pinned revision (`_PACKED_TOP_INDEX_CACHE` keyed
   (repo, type, prefix, rev)) can leak across tests when fakes share literal
   shas ("abc123"); check key disjointness against the NEW seam's own tests in
   BOTH directions before clearing the round.

**Why:** the exact class the g7 brief warned about — a widened fake can turn a
real test into a no-op silently; these four checks settle it with runs, not
reading.

**How to apply:** any split-review group whose commits only touch test fakes
after another group widened the seam those fakes stand in for. Related:
[[fails-pre-fix-probe-parent-commit]].

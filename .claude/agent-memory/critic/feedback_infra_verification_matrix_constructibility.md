---
name: Infra verification-matrix constructibility
description: For kind:infra plans, check each §5 verification claim is CONSTRUCTIBLE by the named test fixture (e.g. "out-of-cone committed path" when every fixture path becomes in-cone before commit); also du-on-fresh-worktree vs du-on-dirty-root size methodology
type: feedback
---

When reviewing `kind: infra` verification plans (Statistics lens = the verification matrix), two recurring checks (#596):

1. **Constructibility of each claimed test case.** A §5 row may cite a fixture test for a sub-claim the fixture never instantiates. #596: test item 6 claimed the tree-diff assertion covers "any out-of-cone committed path", but every path the fixture commits is in-cone by construction (per-issue cones pre-added; the out-of-cone case does `sparse-checkout add` BEFORE commit, making it in-cone). The only ways to get an out-of-cone committed file are `git add --sparse` or committing from a different (full) checkout — the test must deliberately construct one or the assertion silently degrades to in-cone-only coverage. Walk each verification claim and ask: can the fixture as designed ever produce an instance of the claimed case?
2. **Disk-size claim methodology.** `du` on a long-lived checkout conflates tracked bytes with untracked litter (+`.venv`). The valid measurement for "checkout cost" is `du` on a FRESH worktree (or `git ls-files | xargs du`-style tracked-bytes). #596's body said 14G/worktree; fresh-checkout truth was 3.8G (3.2G eval_results) + 11G `.venv` — the correction changed the headline but not worth-doing (still ~8.8x on the addressed component).

**How to apply:** Both are normally Concerns, not Must-Fix, when (a) the unconstructible case has independent planning-time real-repo evidence and a one-line fixture fix exists, and (b) the corrected numbers still clear the acceptance criterion with headroom. Also re-apply feedback_full_suite_green_needs_baseline.md to any "full pytest green" row in infra plans.

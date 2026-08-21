---
name: verify-symbols-at-tip-before-insert
description: Before inserting a large block into a 3000+ line dispatcher, grep-verify EVERY referenced symbol/signature/filename-format at the current tip in one batch — never from memory or the plan
metadata:
  type: feedback
---

Before landing a large insertion (a new phase block, a sentinel writer) into a
long shared dispatcher, run ONE batched grep pass verifying every symbol the
new code references: helper function signatures (`emit_signal`, `run_queue`
arg order, done-file/log naming like `$DONE_DIR/${phase}__${name}.done`),
registry constants (`cm.SMOKE_OFFDIAG_PAIRS_V3`), consumer CLI flags
(`--pooled-pair`, `--perdraw-dir`), and artifact filename formats
(`cells_pooled_<k>_arm_on.json` — cell_id already contains `pooled_`).

**Why:** on #1336 Unit C-iii this batch caught exact-contract details the plan
prose did not carry (hub's `retry_transient` is an alias assignment, not a
def; the pooled cells filename embeds the unit id verbatim; `set --` field
order must match the python `print` order). A single wrong filename format
would have failed only pod-side, one full launch cycle later.

**How to apply:** after drafting the insertion but BEFORE `bash -n`/commit,
list every out-of-block name it references and grep each at the worktree tip
(`grep -n "def X\|^X=" <files>`), reading ~10-line spans for arg order. Keep
outputs under ~50 lines each (lean-context discipline). Also verify the diff
hunk-header list (`git diff -U0 | grep ^@@`) maps 1:1 to your own edits before
committing on a branch a sibling session also writes to.

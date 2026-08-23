---
name: lint-fix-commit-review-recipe
description: Certify a "lint fixes" commit both directions via workflow_lint's importable checker functions on a parent-blob mimic tree; classify marker-vs-review lint-count drift after a spec-freshness sync
metadata:
  type: feedback
---

For a commit claiming to FIX workflow_lint findings ("hub retry-routing lint
fixes" class), certify BOTH directions without a second ~8-min tree-wide run:
the checkers are importable with dir overrides —
`workflow_lint.check_hub_verify_retry(scripts_dir=…)` /
`check_live_hf_retry_routing(repo_root=…)` (grep `def check_` for others).
Extract the parent blobs (`git show <sha>^:scripts/<f>.py`) into
`/tmp/<mimic>/scripts/`, run the checker on the mimic (expect the hit at the
marker-claimed file:line) and on the worktree (expect zero hits). Delete the
mimic after.

**Why:** the fails-pre-fix bar ([[fails-pre-fix-probe-parent-commit]]) plus
tree-wide-only lint invocation (`--file` is YAML-only) otherwise forces a
second full run per direction; the function-level probe did both in ~30 s
(#2474 r1 g3: hits landed at exactly fit.py:391 / analysis.py:65).

**How to apply:** also expect marker-vs-review FAIL-count drift when a
spec-freshness sync lands AFTER the implementer marker: the sync can bring
main's NEWER workflow_lint.py (byte-identical to origin/main) while a
co-updated helper script the branch never touched stays stale → a NEW lint
FAIL neither side introduced. Classify stale-main-or-worktree with two
probes: `git log <base>..HEAD -- <helper>` empty + required-token grep on
worktree copy (0) vs `git show origin/main:<helper>` (>0). Verdict note, not
a blocker; resolves at merge. (#2474 r1 g3: pre_split_review_guard.py
'IMPLEMENTER-MARKER-MISSING' #2294, 5-vs-4 error counts, both honest.)

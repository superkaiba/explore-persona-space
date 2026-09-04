---
name: spend-timing-artifact-exact-recompute
description: Certify committed spend/timing/staging data artifacts by exact recompute, not field presence — window×gpus, tokens×posted rates (batch = /2), jsonl row-sum cross-foot, manifest-vs-tree name+byte diff
metadata:
  type: feedback
---

On a structure-only review of a data-artifacts commit (spend records, pilot timing, hub-staging indexes), field-presence checks are weak; every headline number admits an EXACT independent recompute from siblings in the same commit:

- **Timing:** parse pod_window start/end, recompute wall_h and wall_h × gpu_count; must match `gpu_hours_all_in` to float precision (#2658 r12 g3: 26.5911 exact).
- **Spend priced-from-tokens:** jq -s sum the per-batch jsonl (input/output/cache tokens, row count) and require byte-exact equality with the summary's `totals` + `n_batches`; then recompute dollars from the posted per-MTok rates — Anthropic Batch API is the standard rates **halved** ((in×$3 + out×$15)/2 for Sonnet). An exact dollars match certifies rates, totals, and the basis string together.
- **Hub-staging index:** diff manifest `files[]` (path_in_repo suffix + bytes) against `git ls-tree -r -l <sha>` names+sizes — the [[untracked_twin_add_certification]] / #2162 name-SET diff, not a count match. Watch the key name: `path_in_repo`, not `path` (a wrong jq key yields all-null "mismatch" noise).
- **Verbatim-text claims** (ruling/disclosure copied from events.jsonl or a plan): compare by sha256 of the extracted strings, whitespace-normalized for hard-wrapped plan prose; a 1-byte size delta is usually your own pipeline's trailing newline — print `wc -c` + `cat -A` before calling it a real diff.

**Why:** presence-only probes passed everything in #2658 r12 g3, but only the recomputes could have caught a fabricated or drifted number (wrong gpu_count, stale totals, unhalved batch rate); all matched exactly, making the PASS load-bearing.

**How to apply:** any split-review group whose commit is data under eval_results/ with spend/timing/staging schemas; run recomputes against blob-at-SHA (`git show <sha>:<path>`), never the live worktree file ([[commit_state_isolated_test_run]]).

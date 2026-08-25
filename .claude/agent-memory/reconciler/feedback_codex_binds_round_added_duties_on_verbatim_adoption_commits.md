---
name: codex-binds-round-added-duties-on-verbatim-adoption-commits
description: Codex FAILs pre-existing internals of a plan-registered verbatim-adoption commit (git-status A file) as round-added-code duties, and derives unit counts statically instead of live-listing
metadata:
  type: feedback
---

Two Codex miscalibrations in one verdict (#2584 r1, code-reviewer; both BLOCKERs deferred, PASS upheld):

1. **Verbatim-adoption commits: `A` in the diff ≠ round-authored.** When the
   approved plan registers a two-commit adoption shape (commit 1 = byte-verbatim
   adoption of a previously-untracked repo-root stray for provenance; commit 2 =
   only the planned fixes, with a deviation fence forbidding further edits),
   Codex binds round-added-code duties (checkpoint/resume >50-unit rule,
   Unicode-safe readers) on the ADOPTED file's pre-existing internals because
   git shows it as `A`. Adjudicate against the plan's registered commit
   structure: verify commit 1 is byte-verbatim (`cmp` vs the stray / `git show`)
   and commit 2's hunks; pre-existing internals then take the pre-existing
   severity bar, and forcing an in-round fix would itself violate the plan
   fence. Kin of [[codex-litigates-pre-existing-in-round-n]] and the
   hardening-beyond-minimal-port-contract memory.

2. **Static count derivation vs live listing.** Codex computed "129 shard
   JSONLs" from producer code (144 prefix files − 15 metadata) to trip the
   >50-unit checkpoint rule; a live scoped `list_repo_tree` showed 36 `.jsonl`
   + 108 `.json` ledgers — the consumer's `.endswith(".jsonl")` filter keeps
   36. When a Codex blocker's trigger is a COUNT over a Hub prefix / artifact
   set, live-list it yourself; Claude's live probe beat Codex's static
   arithmetic.

**How to apply:** on any code-review split where Codex FAILs content inside an
adoption/port/vendoring commit, (a) diff the commits separately to attribute
the defect to pre-existing vs round-authored bytes, (b) live-probe any count or
trigger premise, and (c) for a latent-data defect (e.g. splitlines on
U+2028-bearing JSONL), scan the ACTUAL banked artifacts — #2584: 0 hits across
all 36 shards / 234 rows, confirming latent + fail-loud → Standing-only, ledger
rows resolved via `defer-concern --by reconciler`.

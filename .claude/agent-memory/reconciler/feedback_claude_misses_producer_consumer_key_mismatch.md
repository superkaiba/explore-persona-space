---
name: Claude misses producer/consumer contract mismatches (merged family)
description: Claude PASSes on structural presence of one side of a cross-file contract (key, custom_id format, regex character class, default CLI arg); Codex round-trips the literal contract and catches the silent mismatch. Covers JSON-key path-vs-inline, id-format, consumer-regex, and builder-default-vs-plan-holdout shapes.
type: feedback
---

**Rule:** whenever round-N introduces or touches a cross-file contract — a producer writing a key/id/filename and ≥1 consumer reconstructing it — grep BOTH sides and verify the literal contract character-by-character before believing Claude's structural-presence PASS. Fallbacks (`or []` / `or {}` / `.get(cid)` → None / bare `continue`) hide the mismatch silently; the output "looks fine" (a figure renders, exit 0) while running the degraded branch. Require fail-loud on the missing key, or at minimum a warn.

**Shapes + how to check:**
1. **JSON key, path-vs-inline (#514 r2):** dispatcher writes `payload["dynamics_snapshots_path"]` (sidecar path string); plot reads `ej.get("dynamics_snapshots")` (inline list) `or []` → trajectory figure silently degrades to endpoint markers. Tell: the producer's own comment names the key the consumer reads — when comment and consumer disagree, the bug is in the consumer. Grep BOTH the new `*_path`/`*_sidecar` key AND the legacy bare key; open every consumer.
2. **custom_id format (#519 r2):** producer `f"{persona}__{idx:05d}__{comp_idx:02d}"` vs three consumers' `f"{persona}::{q_idx}::{s_idx}"` → every `.get(cid)` None → metrics silently zero; EM gate exits 1 on all real runs. Verify every consumer's f-string matches the producer's character-for-character (delimiters, padding, key order, WHICH index variable). A consumer comment "try alternate cid key shapes" followed by bare `continue` is an admission. Synthetic `--smoke-fake-responses` never exercises the round-trip.
3. **Consumer-side regex / character class (#509 r1):** plan introduces a new literal class (two-letter cond prefix `FB1`, negative layer index); the producing module's in-module assert passes while a SEPARATE file's regex (`[A-Z]\d+`) rejects it → production crashes at first merge. Empirically test the regex against one NEW and one OLD literal (5-line harness). Sibling gotcha: when reading a script top-down, re-check the upstream enumerator that decides which files reach the loop (a `layer(?P<layer>\d+)` regex disallowing `layer-1` + matching `__perm.json` sidecars as fake cells).
4. **Builder default-arg vs plan-prescribed holdout (#521 r2):** plan §-prose requires a DISJOINT/held-out pool ("last 20 of file Y, hash-disjoint"); the builder docstring + launch command both pass the default `questions.json` (= the eval pool) → headline ρ reads off cos(x,x). Grep three things — plan §-pool name, builder docstring/default, launch-command value — and verify all three name the SAME file; fix adds the disjoint-pool builder step + a SHA256-disjointness assert.

**Why:** Claude's must-fix walk treats "the builder exists + takes the arg + is plumbed" / "the judge IS called" as ADDRESSED; the failure mode is invisible from the producer's smoke. Codex compares the two sides' literals.

Companions: [[feedback_claude_misses_dispatcher_wire_bugs]] (single-program wiring variant); [[feedback_claude_misses_same_file_siblings]]; [[feedback_claude_misses_fix_regressions]].

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Claude misses producer/consumer contract mismatches](feedback_claude_misses_producer_consumer_key_mismatch.md) — round-trip literal contracts: JSON key path-vs-inline, custom_id f-strings, consumer regex classes, builder default vs plan-prescribed holdout.

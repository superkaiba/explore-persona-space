---
name: Salvage/terminal-verdict records must pin input identity
description: A crashed intermediate relaunch can re-judge/overwrite a salvage INPUT after the terminal-verdict record was written, so a later exact-count reconstruction deterministically mismatches (195 vs 200, stochastic judge). Pin input shas at record time; on mismatch with a newer-input evidence trail, accept current artifacts as live truth + audit-update the record. #1947 P0 launch 5.
type: feedback
---

Rule: when a stage writes a terminal-verdict record whose disposition a
later salvage RECONSTRUCTS from sibling artifacts, pin those inputs'
identity (sha256, size, mtime) IN the record at write time. The salvage
then verifies identity first: exact match → reconstruct + assert counts;
mismatch WITH evidence of a legitimate later re-generation (input newer
than record; stochastic judge re-draw) → accept the CURRENT artifact set
as live truth, re-derive the counts, UPDATE the record with an audit
trail (prior vs new counts, reason), and proceed; missing/corrupt inputs
→ fail loud naming the path. Never hard-assert reconstructed == recorded
counts across a stochastic re-judge boundary.

**Why:** #1947 P0 launch 5 (2026-08-01): crashed launch #4 re-judged
`judge_raw_pos.json` (16:09Z) before its one-shot guard refused —
mutating the input 78 min after `topup_record.json` recorded 200 kept;
the r8 salvage's exact-count assert then failed deterministically
(reconstructed 195). Both counts are valid draws of the same judge
instrument; the artifacts are the ground truth, the record is
bookkeeping.

**How to apply:** any resumable driver whose crash-recovery path replays
raw + judge artifacts against recorded counts; the same pinning belongs
in consumption manifests and any record a later phase treats as exact.

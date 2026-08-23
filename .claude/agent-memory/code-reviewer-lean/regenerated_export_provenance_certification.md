---
name: regenerated-export-provenance-certification
description: Certify a regenerated committed data export by parent-blob identity (one diff = byte-identity proof), external recompute of every hash/revision field, and commit-timestamp bracketing (#2479 R9 g4)
metadata:
  type: feedback
---

When a commit REGENERATES a committed export under new provenance pins
("payload unchanged, provenance fields added"), four cheap external probes
certify it without trusting any recorded field:

1. **Parent-blob identity makes one diff the byte-identity proof** — if
   `git rev-parse <old-ref>:<path>` == `git rev-parse <commit>^:<path>` (same
   blob), then the single commit diff IS the diff vs the old version: zero
   hunks outside the provenance block proves the payload arrays are raw-byte
   identical, no jq round-trip needed (do the round-trip too as belt).
2. **Recompute every hash/revision against live externals** — a recorded
   sibling-file sha256 vs `sha256sum` of the committed sibling at the SAME
   commit; a recorded tokenizer/HF revision vs the actual
   `~/.cache/huggingface/hub/models--*/snapshots/<sha>` dir on the machine;
   a recorded `git_commit` vs `git rev-parse` (must equal the export commit's
   PARENT — the emitting code commit).
3. **Timestamp bracketing corroborates the clean-tree claim** — code-commit
   time < export `timestamp` < export-commit time, and the dirty check must
   run BEFORE the output write in the emitting code (else the export's own
   overwrite falsifies `git_dirty=false`).
4. **Cross-emit value stability is drift evidence** — payload-adjacent gate
   numbers (margins, binding rows) identical across the old and new emits
   under DIFFERENT code states argue the change was provenance-only.

Also re-run the content-leak scan on any ids-only export: assert every list
element matches the id regex and no string is long/multi-word beyond known
labels/paths.

**Why:** #2479 R9 g4 — the r8 export had recorded a pre-fix parent sha from a
dirty tree; the r9 regeneration's whole claim was "same payload, honest pins",
and probes 1–4 settled it in ~6 tool calls. A hand-edited (never-emitted)
file would fail probes 2–3. Siblings: [[eligibility-export-call-chain-identity]],
[[template-replica-gate-certification]].

**How to apply:** any diff whose sole change is a regenerated
`eval_results/`/manifest JSON claiming payload-identity: run probes 1–4 +
the leak scan; flag as Major if the recorded git_commit is NOT the export
commit's parent or any recomputed hash mismatches.

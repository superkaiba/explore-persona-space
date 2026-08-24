---
name: lens-scoped-temp-files
description: Prefix ALL aux temp files (handed spans, scaffold parts) with a lens-unique slug — sibling composers for the same issue clobber shared /tmp names mid-verification
metadata:
  type: feedback
---

Prefix every auxiliary temp file (handed-span files, prompt scaffold head/mid/tail) with a lens-unique slug, e.g. `/tmp/cmbc<N>-handed-brief.md`, never a generic `/tmp/handed-<N>-*.md`.

**Why:** On task #2389 round 1 the v2 panel spawned three composer twins in one batch; the statistics composer overwrote this composer's `/tmp/handed-2389-brief.md` between write and verification, silently dropping the brief-handed numbers from the handed corpus. The numeric-leak verifier failed loud with 13 false residuals (exactly the brief-handed atoms: model dims, revision-sha fragments, layer remap). Only the composed PROMPT filename (`/tmp/codex-methodology-baselines-critic-<N>-prompt.md`) is lens-scoped by spec; the aux files were not.

**How to apply:** At Step 4, name every temp file `cmbc<N>-*` (codex-methodology-baselines-critic) and rebuild + assemble + verify in ONE Bash call to minimize the collision window. Re-copy the plan from the versioned `plans/v<K>.md` (same pinned version — this is not snapshot-chasing) and byte-check it against the compose-time size before reassembly. Related: [[scaffold-handed-spans-for-leak-verifier]].

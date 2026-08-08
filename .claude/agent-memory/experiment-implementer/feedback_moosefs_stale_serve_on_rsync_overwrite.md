---
name: moosefs-stale-serve-on-rsync-overwrite
description: rsync overwriting a file on a pod's MooseFS /workspace can be stale-served afterwards; rm-then-rsync and verify sha both sides
metadata:
  type: feedback
---

On a RunPod MooseFS-backed `/workspace`, an rsync that OVERWRITES an existing
file in place can afterwards be served STALE — the pod keeps executing the old
content even though rsync exited 0. Delete the file first (`rm -f`), then
rsync, then verify `sha256sum` matches on BOTH sides before relaunching.

**Why:** this is the rsync sibling of the documented git-pull MooseFS
stale-read class (`.claude/rules/gotchas.md`, #1112). #1482 run-length round:
`rsync -a` returned rc=0, but the pod's copy was still 41,521 B (old) vs the
local 41,841 B, `grep` found none of the new symbols, and the relaunched
upload leg reproduced the ALREADY-FIXED error verbatim — which reads exactly
like "my fix didn't work" and sends you chasing the wrong bug. Clearing
`__pycache__` does NOT help; the stale bytes are the .py file itself.

**How to apply:** on any pod hot-patch of an rsync-staged tree —
`rm -f <remote path>` → rsync → `sha256sum` both sides → only then relaunch.
Cheap (one extra round-trip) and it converts a silent wrong-code run into a
loud mismatch. Same family as [[feedback_stale_pycache_masks_signature_change]]
(stale bytecode) — check BOTH when a fix appears not to engage. Note the
crash-fix relaunch rules already mandate a byte probe for the SAME-POD git
path; this extends the duty to rsync-staged trees, which the fresh-clone
EXEMPTION in that rule does NOT cover (rsync overwrites, it does not
freshly write).

Related: [[feedback_reused_script_may_have_uncommitted_sibling_edits]].

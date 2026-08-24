---
name: stash-disclosure attest + shared-stash trap
description: When a brief discloses a mid-round git-state anomaly (stash/pop) the orchestrator already verified clean, attest the verified baseline and convert the twin's duty to DIFF-INTERNAL coherence; pre-empt the shared-repo stash-list trap
metadata:
  type: feedback
---

When the brief discloses the implementer ran a mid-round `git stash` +
immediate pop and states the orchestrator independently confirmed a clean
worktree / HEAD == origin / no branch stash entry (#2292 r2, 2026-08-22):

1. **Re-verify the three git facts at compose time** (porcelain empty,
   `rev-parse HEAD origin/<branch>` equal, stash list scan) and write them
   into a compose-time attestation — never pass the disclosure without the
   verified baseline, or the twin burns its round re-deriving git state it
   cannot fully see.
2. **Shared-stash trap:** `git stash list` inside ANY worktree shows the
   REPO-GLOBAL `refs/stash` — on this fleet it routinely carries unrelated
   root-session `autostash` entries (sync_repo_root.py recoveries, rescued
   husks). A twin that runs the list sees N entries and can wrongly flag
   "stash present, contradicts the brief". Pre-empt in the attestation: name
   the entries as unrelated cross-session residue, out of scope, existence
   is NOT a finding; a branch-made stash would read `WIP on <branch>: ...`.
3. **Residual duty is DIFF-LEVEL, and say so:** a stash/pop cycle manifests
   as a PARTIAL RESTORE — give the twin the symptom list (hunk referencing
   a symbol defined nowhere at HEAD; docstring describing behavior the
   adjacent code lacks; a test asserting a production behavior no
   production hunk provides, or vice versa; duplicated/self-conflicting
   edits) instead of a vague "check for lost edits".

**Why:** the git-level facts are composer-verifiable and sandbox-opaque;
the diff-internal coherence is exactly what the twin CAN verify. Splitting
the duty along that line avoids both a false `data-access-blocked`/
contradiction flag and a hollow "looks fine" on the real risk.

**How to apply:** any round whose brief discloses mid-round stash/rebase/
reset activity in the worktree, or whose implementer marker mentions a
scratch git manipulation. Related: [[revision-round compose recipe]].

---
name: presence-redrive-blesses-stale-mirror
description: An upload/repair re-drive that short-circuits on a presence-only remote check can bless a PRIOR fingerprint's bytes at the same prefix as this run's artifact — the stale-artifact hole reopens one level down, on the durable mirror (#2225 R2 g1)
metadata:
  type: feedback
---

When a resume fix binds LOCAL artifacts to the producing fingerprint (sha in a
save-time manifest record) and adds an upload re-drive for the
completed-but-not-uploaded window, check the re-drive's decision input: a
"skip re-upload if remote files present" short-circuit reintroduces the
exact bug the fix closed, because the remote prefix (keyed by cell slug, not
fingerprint) can hold a PRIOR fingerprint's bytes. The re-drive then sets
`uploaded=True` against stale bytes and the durable record is silently wrong
for every consumer that stages from the mirror (fresh-pod eval, post-cleanup
re-eval, sibling artifact-reuse) — while the same-pod local-first path masks
it during review/smoke.

**Why:** #2225 R2 g1 (`issue2225_train.py::should_skip` re-drive leg): the r1
Critical-1 fix (sha-bound `completed` + `uploaded` flag + wipe) correctly
closed the SKIP decision, but `_hf_files_present` is existence-only, so an
F2 retrain that crashed in the save→upload window on a previously-uploaded
cell got its F1-era HF files blessed as the F2 upload. After the first full
wave every cell is previously-uploaded, so any crash-fix round arms it.

**How to apply:** on any resume/repair diff, for each flag-setting repair
path ask "what bytes does the flag now vouch for, and could they predate the
current fingerprint?" Remote-presence checks are fine for GATING a skip
(when a fingerprint-scoped flag also gates it) but never for DECIDING a
repair is unnecessary — an idempotent unconditional re-upload (overwrite at
the same prefix) is the cheap safe form. Also check the CONSUMER: local-first
resolution hides mirror staleness; grep the eval/staging side for the
HF-prefix fallback before rating severity. Family: [[fanout-cvd-ordinal-not-entry]]
(same review line), [[inplace-merge-phase-not-idempotent]] (second-run
corruption), the r1 start-manifest stale-adapter shape this descends from.

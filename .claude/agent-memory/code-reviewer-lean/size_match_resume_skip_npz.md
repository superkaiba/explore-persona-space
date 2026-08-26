---
name: size-match-resume-skip-npz
description: upload_dir_sharded resume_skip=True skips dests "present at MATCHING size" — fixed-shape uncompressed np.savez archives have identical byte size for different content, so a crash-fix recompute is silently NOT re-uploaded (stale mirror)
metadata:
  type: feedback
---

A final catch-all upload phase that passes `resume_skip=True` into
`upload_dir_sharded` (or any size-matched presence probe) blesses a PRIOR Hub
copy after a recompute: uncompressed `.npz` of fixed array shapes/dtypes has
byte-identical size regardless of float content, so size-match ≈ bare
presence for exactly the artifact class regime-keyed recompute machinery
(`_enter_phase_regime` code-sha wipes) is designed to refresh.

**Why:** #2552 r1 g3 — `phase_upload` used `resume_skip=True` on all four
leaves while the same driver's SAE upload correctly used `resume_skip=False`
on fresh trains. Sharpens [[presence_redrive_blesses_stale_mirror]] (#2225
R2 g1): "matching size" sounds like a content check but is not, for npz.

**How to apply:** whenever a reviewed diff calls `upload_dir_sharded` /
`_upload_leaf` with `resume_skip=True`, ask what happens after a
forced recompute of same-shape npz outputs: require `resume_skip=False` on
any path reachable after a stale-path wipe, or a content-hash probe. Grep the
producing phase for wipe-and-recompute branches to prove reachability.

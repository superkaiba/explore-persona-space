---
name: resume-fp-strengthening-vs-selector-idempotence
description: 'Impl-mode check pair: (1) adding a MEASURED knob (pilot-selected gen_batch, armed mode) to a resume fingerprint is only safe if every producer of that knob is read-back-not-re-measured on the planned resume path — an unguarded pilot re-run that noise-flips the selection invalidates ALL banked shards; (2) a frozen/cached gated decision must re-apply its gate at ADOPT time (#2389 r2 B6-bypass shape).'
metadata:
  type: project
---

Two coupled patterns from #2389 r2 (the B9 fingerprint fix), both found on the revision round of an APPROVED fix — the fix direction was right and the hazard lived in interactions:

1. **Fingerprint strengthening makes upstream selector non-idempotence catastrophic.** Adding realized regime knobs (`gen_batch`, `share_prefill_armed`) to the resume fingerprint is CORRECT (mixing regimes in one store is the bug it fixes) — but it converts any RE-MEASURED upstream selector into a mass-invalidation hazard. #2389: `dispatch.sh all` re-runs the three-regime pilot unconditionally on the plan's PRE-REGISTERED "relaunch reuses the same command" TIMEOUT continuation; a noise/HBM-headroom-edge flip of argmin(B∈{16,32}) rewrites the report → new regime_fp → every banked anchors/grid done shard quarantined + regenerated (tens of GPU-h; the plan's "lossless boundary" property silently broken). **Check:** for each measured value newly entering a resume fp, trace the planned resume path and confirm the value is READ BACK from a persisted artifact, never re-measured (skip-if-report-exists guard); flag any selector whose eligibility rides a hardware-noise threshold (HBM headroom floor) as flip-prone.

2. **Freeze/adopt caching must re-apply the gate it caches.** `_resolve_share_prefill`'s first-writer freeze file was adopted raw (`rec["armed"]`) without re-running the B6 `mode == "production"` guard; a --tiny smoke run (which legitimately accepts a tiny PASS) freezes `armed=true` into the SAME default out_root the production dispatch reuses → production arms on CPU-tiny evidence — the exact ARMED-on-weak-evidence direction the freeze design was supposed to exclude. **Check:** any "first resolver writes, later resolvers adopt" seam — confirm the adopt branch either re-applies every conditional guard of the write branch or the freeze is keyed/refused on the guard inputs (tiny/smoke/mode bits), mirroring the pilot-report's own smoke/tiny mismatch refusal.

Related: [[stride-cell-bucketing-batch-fragmentation]] (the same round's B7 fix + glob-residue trap).

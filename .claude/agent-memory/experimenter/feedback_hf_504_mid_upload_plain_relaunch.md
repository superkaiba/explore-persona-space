---
name: HF 504 mid-upload — plain relaunch resumes
description: Transient HF Hub 504 during a long per-file dataset upload kills the phase; plain relaunch resume-skips existing outputs — verify by fresh [upload] lines
type: feedback
---

A transient HF Hub 504 (Gateway Time-out on `create_commit`) during a long
per-file dataset upload kills the whole dispatcher phase even though the
compute work (tensor assembly) is already on disk.

**Why:** upload loops without per-file retry-on-5xx propagate the first 5xx
as a phase crash. Burned at #542 assemble (2026-06-12); sibling of
`feedback_hf_5xx_on_upload_verify_kills_gcp_lane.md` (same root, different
phase).

**How to apply:** plain relaunch of the SAME phase is the correct recovery
when the phase resume-skips existing outputs — verify resumption by fresh
`[upload]` log lines, never by re-running earlier phases. If the same 504
recurs immediately, wait ~120s (gateway blip) and retry once before posting
`epm:failure`. Flag per-file retry-on-5xx as an implementer suggestion when
the upload loop lacks it.

---
name: Transient HF 5xx during upload/verify — infra respawn or plain relaunch, never a code bounce
description: A transient HF Hub 504 on create_commit or the post-upload verify call kills the phase (and on the GCP lane fires the EXIT trap and powers off the VM) even though the compute is on disk; plain relaunch resume-skips.
type: feedback
---

A transient HF Hub 5xx mid-upload (`create_commit` 504) or on the POST-upload verify (tree-listing) call crashes the phase fail-loud even though the compute work is already on disk — and on the GCP lane the non-zero exit fires the startup-script EXIT trap, publishing `phase=failed` and powering off the VM (billing bounded, disk preserved).

**Why:** #542 assemble (2026-06-12) — 504 on create_commit killed a long per-file dataset upload. #491 attempt att-20260611-143912 — `hub.upload_dataset_directory` raised `upload_dataset returned ''` on a verify-call 504 right after the previous file uploaded clean. Both transient HF infra, not code.

**How to apply:** when a run dies in an upload phase, FIRST grep the log for an HF 5xx before treating it as a code bug. Recovery = plain relaunch of the SAME phase (resume-skips existing outputs — verify by fresh `[upload]` log lines, never by re-running earlier phases) or a same-branch infra respawn on GCP. On an immediate identical recurrence, wait ~120s (gateway blip) and retry once before posting `epm:failure`. If a 5xx kills a SECOND attempt, route a reviewed retry/backoff (~3 tries, exponential) into the upload/verify loop (`orchestrate/hub.py`) instead of respawning a third time blind.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [HF 5xx upload/verify transients](feedback_hf_5xx_upload_transient.md) — 504 on create_commit or post-upload verify kills the phase (GCP lane powers off); plain relaunch resume-skips; infra not code (#491, #542)

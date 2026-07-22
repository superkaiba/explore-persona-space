---
name: HF LFS billing-403 — proceed via smoke --no-upload, keep non-LFS uploads on
description: An HF LFS 403 "setup automatic credit recharge" is a namespace-billing block, not code/token — recovery shape for launches that hit it
type: feedback
---

An HF upload failing with `403 Forbidden: You need to setup automatic credit
recharge in order to upload more data` (LFS batch endpoint) is a
namespace-BILLING block — external to the code, NOT a token-permission or
transient error. Retrying cannot succeed until the user fixes billing (or
frees LFS quota). Recovery (validated #1586, 2026-07-22): re-run the smoke
with `--no-upload` (upload-path validation is moot when the breakage is a
known external block) so the chain proceeds into the non-LFS phases; keep the
full run's JSON/text incremental uploads ON — the quota gates ONLY the LFS
endpoint, the non-LFS git path stays open. Surface the billing decision to
the user immediately (it is user-only); the LFS-uploading phase (checkpoint
persist) will crash clean + resume-keyed if still unfixed when reached.

**Why:** #1586 pod-1586 smoke died at p4_persist on exactly this 403; the
dispatcher failed loud correctly and ~4-5h of pre-LFS compute would have been
stranded behind a non-code block.

**How to apply:** when a launch/relaunch brief names an HF 403
credit-recharge failure, don't treat it as retryable transport and don't
bounce it as a code bug — apply the smoke `--no-upload` + full-run-proceeds
shape and let the orchestrator own the user-facing billing surfacing.

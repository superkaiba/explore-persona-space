---
name: HF 5xx on upload-verify kills the GCP lane
description: A transient HF Hub 5xx on the POST-UPLOAD verification call makes hub.upload_dataset_directory raise (fail-loud), which on the GCP lane fires the EXIT trap and powers off the VM — distinguish this from a real upload failure before re-implementing anything.
type: feedback
---

A transient HF Hub 504 on the post-upload VERIFY (tree-listing) call makes
`hub.upload_dataset_directory` raise `upload_dataset returned ''` even when the
upload itself may have landed; on the GCP lane the dispatcher's non-zero exit
fires the startup-script EXIT trap, which publishes `phase=failed` and powers
the VM off (billing bounded, disk preserved).

**Why:** incident #491 (2026-06-11, attempt att-20260611-143912): smoke_upload
died on a 504 for `run_specs.json` immediately after the previous file uploaded
+ verified clean. Root cause transient HF infra, not code.

**How to apply:** when a GCP/SLURM run dies in an upload phase, FIRST check the
log for an HF 5xx on the verify call before treating it as a code bug — the fix
is an infra respawn (same branch, no implementer round), not a revision. If the
same 5xx kills a SECOND attempt, add retry/backoff (~3 tries, exponential) to
the verify call in `orchestrate/hub.py` via a reviewed change rather than
respawning a third time blind.

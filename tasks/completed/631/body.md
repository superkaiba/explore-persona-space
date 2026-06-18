---
title: 'GCP lane: FLEX_START + a3-highgpu H100 intents; enable SPOT to use the idle
  preemptible A100 quota'
kind: infra
tags: []
created_at: '2026-06-12T21:25:19Z'
has_clean_result: false
origin_prompt: See how you can get better quota for GCP (H100, H200, everything) search
  deeply
---
## Provenance

Originating user prompt (2026-06-12, quota research session): "See how you can get better quota for GCP (H100, H200, everything) search deeply" — deep-research report delivered in-session; this task captures the code-side follow-ups (items 4+5 of the action list).

## Context

Deep-research findings (verified against Google docs fetched 2026-06-12):

- The 1-4 GPU H100 shapes (`a3-highgpu-1g/2g/4g`, GA Jan 2025) **cannot be created on-demand at all** on Compute Engine — Spot VMs or DWS flex-start VMs only. On-demand `NVIDIA_H100_GPUS` quota is irrelevant for them.
- **DWS flex-start** consumes PREEMPTIBLE GPU quota, runs uninterrupted up to 7 days, and works as a single-VM create: `gcloud compute instances create ... --provisioning-model=FLEX_START --max-run-duration=3d --request-valid-for-duration=2h`. This matches the GCP lane's existing ephemeral `--max-run-duration` instance pattern almost exactly. Source: https://cloud.google.com/blog/products/compute/introducing-flex-start-vms-for-the-compute-engine-instance-api + https://docs.cloud.google.com/compute/docs/instances/about-flex-start-vms
- The project already holds **8× PREEMPTIBLE_NVIDIA_A100_80GB_GPUS in us-central1, entirely unused** — `backends/gcp.py` pins `DEFAULT_PROVISIONING_MODEL = "STANDARD"`; the SPOT path is plumbed (`spec.extra["provisioning_model"]`) but gated on idempotency proofs (gcp.py:333-357).
- Preemptible H100/H100-mega quota requests are filed separately by the user (console); this task makes the lane able to consume them the moment they land.

## Deliverables

1. **FLEX_START provisioning support** in `backends/gcp.py`: extend `ProvisioningModel` to `SPOT | STANDARD | FLEX_START`, render `--provisioning-model=FLEX_START` + `--request-valid-for-duration` in `render_create_argv`, verify interaction with `--max-run-duration` (≤7d cap) and `--instance-termination-action=DELETE` (flex-start constraints per docs — verify, don't assume). Golden-test the argv.
2. **H100 intent mappings**: add `a3-highgpu-1g` (1× H100) / `a3-highgpu-2g` (2× H100) entries to `INTENT_TO_MACHINE` (e.g. opt-in intents `lora-7b-h100` / `eval-h100`, or a `gpu_kind` override) — gated on a quota pre-check (same pattern as the #608 quota pre-check) so a zero-quota project fails loud at selection, not mid-provision.
3. **SPOT idempotency proofs + opt-in enablement for A100**: validate the preemption→fresh-idempotent-re-run contract the gcp.py docstring describes (artifacts pushed off-VM during the run survive; re-run is idempotent), then allow `provisioning_model=SPOT` for A100 intents to use the idle 8-GPU preemptible pool. `DEFAULT_PROVISIONING_MODEL` stays STANDARD.
4. Update the backend-router docs/markers (`epm:backend-selected` reason field) to name which provisioning model + quota pool a launch used.

## Notes

- Item 3 is usable **today** (quota already granted). Items 1-2 become usable once preemptible H100 quota lands (user console action, tracked outside this task).
- Flex-start is queued/best-effort (request validity 90s-2h) — selector should treat "no capacity within validity window" as a provisioning failure that falls through to the next lane, consistent with the router contract.

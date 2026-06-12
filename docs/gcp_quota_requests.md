# GCP quota requests — paste-ready justifications (2026-06-12)

File from the console (IAM & Admin → Quotas, project `eps-persona-gpu-jun2026`) with
your owner account — the `eps-router` service account cannot file quota requests.
Filter the quota list by the names below; exact metric IDs for H100/H200 only appear
in the console / Cloud Quotas API, not in the classic region-quota list.

Background (deep-research, 2026-06-12): the 1-4 GPU H100 shapes (`a3-highgpu-1g/2g/4g`)
are Spot/DWS-flex-start-only — no on-demand path exists — and flex-start consumes
PREEMPTIBLE quota, so the preemptible metrics below are the ones that unlock H100s.
Code-side support is tracked in task #631.

## Requests to file (priority order)

| # | Quota (console filter) | Region | Ask | Why |
|---|---|---|---|---|
| 1 | Preemptible NVIDIA H100 GPUs | us-central1 | 8 | DWS flex-start / Spot on a3-highgpu-1g/2g — primary H100 route |
| 2 | Preemptible NVIDIA H100 GPUs | us-east4 (backup) | 8 | Same; capacity fallback region (a3 shapes present) |
| 3 | Preemptible NVIDIA A100 80GB GPUs | us-central1 | 16 (from 8) | Headroom for the Spot-A100 lane (#631 item 3) |
| 4 | NVIDIA A100 80GB GPUs (on-demand) | us-central1 | 16 (from 8) | Proven bottleneck: #597/#606 queued at 8/8 |
| 5 | GPUs (all regions) | global | ≥24 | Global cap binds on top of every per-model grant |
| 6 | Preemptible NVIDIA H100 MEGA / H200 metrics (if exposed) | us-central1 | 8 | Future a3-mega/a3-ultra flex-start use |
| 7 | Vertex AI: custom_model_training_preemptible_nvidia_h100_gpus | us-central1 | 8 | Independent parallel pool (Vertex FLEX_START jobs) |

## Justification text (paste into the request form)

> Academic AI-safety research project (university PhD fellowship, Google-credits-backed
> billing) running single-user ML fine-tuning experiments: LoRA and full fine-tunes of
> 7B-parameter open-weights LLMs (Qwen-2.5-7B), 1-4 GPUs per job. Jobs run on ephemeral
> instances created per experiment with `--max-run-duration` (typically 2-24 h) and
> `--instance-termination-action=DELETE`; all workloads checkpoint to external storage
> and are resumable, so they fit Spot / DWS flex-start semantics. Current usage:
> sustained 5-8 of 8 NVIDIA A100 80GB GPUs in us-central1 over the past two weeks, with
> experiments regularly queued behind the 8-GPU cap. Requesting preemptible-class
> H100 quota to run 1-2 GPU `a3-highgpu` shapes via DWS flex-start (shorter wall-clock
> per experiment at similar cost), plus modest A100 headroom. Expected steady-state:
> 2-4 concurrent jobs, 8-16 GPUs peak.

For the A100 on-demand bump (row 4), trim to the usage + queueing sentences.

## After grants land

- H100 rows → unblock task #631 items 1-2 (flex-start + a3-highgpu intent mappings).
- Row 3 → #631 item 3 (Spot A100) gets headroom; the existing 8 are usable as soon as
  the SPOT idempotency proofs land.
- Caveats from the research: quota ≠ capacity (flex-start is queued, validity 90 s-2 h);
  Google documents the preemptible-quota switch for predefined-run-time standard VMs as
  one-way; whether fellowship credits cover DWS charges is undocumented — check the
  first bill.

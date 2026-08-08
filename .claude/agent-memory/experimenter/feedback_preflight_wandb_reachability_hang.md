---
name: preflight wandb-reachability probe hangs on fresh RunPod pod
description: orchestrate.preflight --json can hang 100s+ with zero output on a fresh RunPod pod — the block is the WandB reachability probe, not a real failure. Distinct from the fetch-timeout false-negative (which returns ok:false with an error); this one produces NO output before the SSH-MCP cap fires
type: feedback
---

On a fresh RunPod pod the FIRST `orchestrate.preflight --json` invocation
can hang for 100s+ with **zero stdout** before returning, exceeding the
SSH-MCP ~30s command cap and looking indistinguishable from a stuck
launch. The blocking probe is the internal WandB reachability check
(api.wandb.ai TCP connect over the pod's fresh network stack), NOT a
real infra failure. The pod itself is healthy — GPUs are visible, HF
cache is warm, HF Hub is reachable — but preflight is single-threaded
across its probes, so an unresponsive-looking WandB probe wedges every
downstream check.

**Why:** the workflow rule "never silently ignore preflight failures"
exists to catch real problems (OOM, ENOSPC, gated repos, branch drift).
This one produces no verdict at all — it's a false BLOCK, not a false
FAIL like `feedback_preflight_fetch_timeout_false_negative.md`. Treating
the hang as an infra failure costs a full relaunch cycle on a
already-healthy pod.

**How to apply (experimenter Step 6c on a fresh RunPod pod):**

1. If `preflight --json` returns NO output within ~30s, kill the
   preflight tree on the pod (don't leave a zombie python holding
   any FDs / caches), then re-verify the SUBSTANTIVE checks
   manually:

   ```bash
   ssh pod-<N> '
     set -e
     # (a) branch pinning
     cd /workspace/explore-persona-space
     git rev-parse HEAD                                    # compare against brief commit
     # (b) GPU freeness + zero-residency
     nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader
     nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
     # (c) HF creds live
     uv run huggingface-cli whoami
     # (d) WandB reachability (bare TCP, do NOT reuse orchestrate.preflight)
     timeout 10 curl -sSo /dev/null -w "%{http_code}\n" https://api.wandb.ai/  # any status incl. 404 counts as reachable
     # (e) .env sourceable
     [ -f .env ] && bash -c ". ./.env && echo env-sourced-ok" || echo NO-ENV
   '
   ```

2. All checks pass → the pod is launch-clear. Proceed with the workload
   launch; record the preflight-hang carve-out in your launch marker's
   `note` field so the trail carries the reason preflight was skipped.

3. ANY substantive check fails (git HEAD wrong, GPU held, HF creds
   broken, WandB truly unreachable at the bare-curl layer, .env
   missing) → real infra failure, post `epm:failure v1
   failure_class: infra` with the substantive finding and exit.

This carve-out is NARROW: it only applies to the "preflight returns no
output at all" case on a first-touch RunPod pod. Do NOT extend it to
preflight FAILing with a real error message (that's what `feedback_
preflight_fetch_timeout_false_negative.md` covers) or to a genuinely
hung LAUNCH after preflight passed.

Closed regressions: task #778 r1 launch (2026-07-01, autonomous
strategy pivot from GCP `sweep-8g-h100` stockout to RunPod 8× H100 on
`pod-778`) — resolved inline before posting `epm:failure`, kept the
workload launch on the critical path.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Preflight wandb-reachability probe HANGS on fresh RunPod pod (no output before SSH-MCP cap)](feedback_preflight_wandb_reachability_hang.md) — orchestrate.preflight --json can produce ZERO output for 100s+ on a first-touch RunPod pod; the block is the WandB reachability probe, not a real failure. Distinct from the fetch-timeout false-negative — that one FAILs; this one HANGs. Kill the tree + verify substantive checks manually (git HEAD, nvidia-smi freeness, HF whoami, bare `curl -w %{http_code}` to api.wandb.ai, `. ./.env`); all pass → launch-clear (#778 r1)

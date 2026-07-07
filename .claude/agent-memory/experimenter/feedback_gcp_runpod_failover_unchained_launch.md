---
name: gcp-runpod-failover-unchained-launch
description: On a GCP→RunPod failover re-drive, the failover pod may have finished bootstrap cleanly despite the failover's SSH timeout (boot-lag), and no RunPod handle sidecar exists — expect an unchained launch and note it on the run-launched marker
type: feedback
---

GCP→RunPod failover re-drives (#659 path): the failover's branch-sync SSH timeout is often pure BOOT-LAG — the pod finishes bootstrap on its own (branch at expected HEAD, .venv intact), so the re-drive needs re-verification only, zero repair. **Why:** #931 (2026-07-04): failover declared `RunPodWorkloadStartError` at 01:59Z; 40 min later SSH was reachable and the pod was fully bootstrapped. **How to apply:** before repairing anything on a failover pod, verify state first (git HEAD, .venv, /workspace layout) — it may already be launch-ready. Also: the failover archives the GCP handle sidecar WITHOUT minting a RunPod one, so the launch is UNCHAINED (no declared completion-sentinel path); note `no live handle sidecar — finalize needs post-hoc write_completion_sentinel or --skip-confirm-artifacts` on the `epm:run-launched` marker so the Step-8 finalize isn't surprised.

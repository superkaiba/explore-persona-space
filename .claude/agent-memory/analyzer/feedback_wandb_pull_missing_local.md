---
name: WandB pull when local data is partial
description: When raw eval JSONs aren't fully synced from pod to local repo, pull artifacts directly via wandb Api instead of waiting on pod resume
type: feedback
---

When the source pod is unreachable (stopped, IP rotated) but eval results were uploaded to WandB Artifacts via the upload-verifier path, pull the artifacts directly instead of attempting to revive the pod or asking the user to resume it.

**Why:** In issue #240, the local repo only had Part B L=20 sonnet results synced; L=20 opus, L=40 sonnet/opus, L=80 sonnet/opus were on the pod but pod `epm-issue-240` was unreachable (`ECONNREFUSED`). The artifacts had been uploaded under `issue240-gcg-L{20,40,80}-results` v1 (visible in `epm:upload-verification v3`'s required-fix list). Pulling them via `wandb.Api().artifact(...).download()` gave full data in ~30 seconds vs an indeterminate pod-revive wait.

**How to apply:** Before drafting the analyzer output, check that all referenced cells × judges have local summary JSONs. If any are missing, check the issue's `epm:upload-verification` markers for the artifact names, then:

```python
import wandb, pathlib
api = wandb.Api()
art = api.artifact("thomasjiralerspong/explore-persona-space/issue240-gcg-L20-results:latest", type="gcg-results")
art.download(root="/path/to/local/_wandb_pull/L20")
```

WandB-pulled data goes into a `_wandb_pull/` subdir to keep it visually distinct from pod-synced data; recompute alphas from the pulled summary JSONs directly.

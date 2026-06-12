---
name: resumed-pod-uv-wiped
description: After pod.py stop→resume, uv at /root/.local/bin is GONE (ephemeral container FS wiped) even though the .venv on /workspace survives. Reinstall uv before any `uv run` launch.
metadata:
  type: feedback
---

On a RESUMED pod (stopped then resumed), `uv` is missing: `/root/.local/bin/`
is wiped because it lives on the ephemeral container filesystem, NOT the
persistent `/workspace` MooseFS volume. The project `.venv` on
`/workspace/<work>/.venv` survives intact (all CUDA/deps present), and the
cached one-time experiment artifacts under `/workspace/...` survive too.

**Why:** stop/resume re-creates the container; only `/workspace` persists.
Bootstrap installs uv to `/root/.local/bin` (container FS), so it does NOT
survive a resume. This is distinct from [[pod_provision_uv_missing]] (provision
returning a uv-missing pod) — here uv WAS installed, the resume just wiped it.

**Signature:** `sh: 1: uv: not found` (exit 127) on the first `uv run ...`;
`find / -name uv -type f` returns nothing; `/root/.local/bin/` does not exist;
but `.venv/bin/` is fully populated (accelerate, datasets, torch, etc.).
NOTE: a `which uv && uv run ...` chain short-circuits silently (empty output,
exit 1) when `which` fails — don't mistake that for a preflight crash.

**How to apply:** On any resumed pod, BEFORE preflight/launch, check
`command -v uv`. If missing, reinstall — it's fast (single ~60 MB binary, NOT
the multi-GB CUDA wheels, so well within a subagent turn unlike the
provision-time uv-missing recovery):
```
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs to /root/.local/bin
```
Then `export PATH="/root/.local/bin:$PATH"` inside every command and in the
launcher script (`set -a; source .env` non-login shells skip ~/.bashrc anyway).
uv reuses the existing `.venv` immediately — no re-resolve, no re-download.
Burned at #472 round-3 SMOKE re-launch on resumed pod-472 (2026-06-03).

---
name: uv missing on pod — provision-incomplete vs resume-wipe variants
description: Two ways a pod lacks uv. (1) provision returns "ready" but bootstrap silently skipped uv + .venv — recovery too long for a subagent turn, post epm:failure. (2) stop→resume wipes /root/.local/bin while the /workspace .venv survives — reinstall inline, it's fast.
metadata:
  type: feedback
---

Verify `command -v uv` AND `.venv/bin/python` exist BEFORE any pytest/launch. Two distinct variants:

**Provision variant (#390, 2026-05-26):** `pod.py provision` reported success with repo + .env + HF cache fine, but uv and `.venv/` were absent (bootstrap step silently dropped; no bootstrap log produced). Recovery = uv install + full `uv sync` (multi-GB CUDA wheels, 5-15 min) — exceeds the subagent turn budget. Post `epm:failure v1 failure_class: infra reason: pod_bootstrap_incomplete_uv_missing` with the recovery PID and exit.

**Resume variant (#472 round-3, 2026-06-03):** after `pod.py stop` → `resume`, `/root/.local/bin/uv` is GONE (ephemeral container FS re-created) but the `/workspace` `.venv` survives fully populated. Signature: `sh: uv: not found` (127) while `.venv/bin/` is full; note `which uv && uv run ...` short-circuits silently when `which` fails. Recovery is INLINE-fast (single ~60 MB binary, no re-resolve):
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**PATH gap (both variants):** SSH non-login shells skip ~/.bashrc — every subsequent command needs `export PATH="/root/.local/bin:$PATH"` (or the full `/root/.local/bin/uv` path), including inside launcher scripts. Related: [[feedback_load_env_in_nohup]].

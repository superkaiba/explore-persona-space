---
name: runpod-overlay-hf-cache-trap
description: RunPod ephemeral pods can have /root/.cache/huggingface on the 50G container overlay even when /workspace (200G) is mounted. Preflight must verify HF_HOME points at /workspace AND the /root path is a symlink, not a real directory.
metadata:
  type: feedback
---

On ephemeral RunPod pods (provisioned via `pod.py provision`), the bootstrap
step that wires HF cache onto `/workspace` is not always idempotent against
a pre-existing `/root/.cache/huggingface` directory left by the base image.
Symptom: training (which always uses `/workspace` via explicit `HF_HOME`)
succeeds, then eval — invoked without an `HF_HOME` override — defaults to
`~/.cache/huggingface = /root/.cache/huggingface` and dies after 1-2
checkpoint downloads when the 50G overlay fills up.

**Pre-launch check experimenter MUST add to the Phase 0 check sequence:**

```bash
ssh ... 'df -h / | tail -1 && env | grep HF_HOME && \
  test -L /root/.cache/huggingface && echo "symlink: OK" || echo "symlink: REAL DIR — WILL OVERFLOW"'
```

If `HF_HOME` is unset OR the symlink check fails OR `/` shows < 80% free,
bounce back BEFORE launching. Adding `HF_HOME=/workspace/.cache/huggingface`
to the nohup launch command is insufficient — many subprocesses (vLLM
EngineCore, HF Hub downloader) re-resolve `~/.cache` from `$HOME`, so the
symlink fix is the only durable repair.

**Why:** Issue #356 Phase 2 lost 11/12 cells to this exact failure on
2026-05-21 between 01:11 (cell 1 download) and 01:24 (cells 2-12 EIO 28).
~70 min of wasted GPU and a forced bounce to implementer round.

**How to apply:** Add the symlink + HF_HOME check to the experimenter Step 3
"verify data sanity" preflight. Fail loudly; do NOT launch unless both pass.
Related: [[feedback-cache-path]] is the older, training-only version of
this rule; this one extends it to cover eval launches that don't go
through the trainer's env setup.

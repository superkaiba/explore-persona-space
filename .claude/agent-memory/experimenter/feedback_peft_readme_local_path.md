---
name: PEFT auto-README writes local base_model path → HF Hub upload 400s
description: When the trainer got a local mirror dir as base_path, PEFT writes base_model:/workspace/... into the adapter README; HF validate-yaml rejects with 400 and train_lora silently preserves the adapter local-only.
type: feedback
---

PEFT's `save_pretrained` writes the literal `base_model_name_or_path` into the README front-matter. Orchestrators that mirror the base model locally (`/workspace/<work>/qwen25_7b_instruct_base`) before training produce `base_model: /workspace/...`, which HF Hub's validate-yaml rejects (HTTP 400). `train_lora()` catches the 400, logs `WARNING Adapter upload failed — local copy preserved`, and the experiment CONTINUES — the gap only surfaces at the upload-verifier.

**Why:** #262 hit this on every adapter (EM, benign, coupling × 4). The bug lives in the shared training library (`train/sft.py` should normalize the path to the canonical HF id before PEFT serializes), not the orchestrator script.

**How to apply:** detect mid-run via `grep "Upload failed: Invalid metadata in README" /workspace/logs/<run>.log` — a hit means EVERY adapter from that orchestrator is local-only. Hot-fix per adapter (≤10 lines, no logic change): patch `<adapter>/README.md` + `adapter_config.json` to the canonical HF id (e.g. `Qwen/Qwen2.5-7B-Instruct`), then `HfApi().upload_folder(...)` as a detached background upload — `setsid nohup <upload one-liner> < /dev/null > /workspace/logs/<run>-upload.log 2>&1 &` (detachment trio; never bare `nohup ... &` over SSH — SIGHUP reaping when the session dies, #444/#541; full pod-side shape: `experimenter.md` § "During Execution" step 1). Filed follow-up: normalize inside `train_lora` before auto-upload.

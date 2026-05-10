---
name: PEFT auto-README writes local base_model path → HF Hub upload always 400s
description: Every LoRA / coupling adapter the orchestrator pushes to HF fails with "Invalid metadata in README.md ... base_model with value '/workspace/...' is not valid" because PEFT's save_pretrained writes the local-mirror base path
type: feedback
---

PEFT's `save_pretrained` auto-generates `README.md` with a YAML front-matter
`base_model:` field set to the literal `base_model_name_or_path` — when the
trainer was given a **local mirror directory** (e.g.
`/workspace/persona_flatten_262/qwen25_7b_instruct_base`) instead of the HF
Hub model id, the resulting README has `base_model: /workspace/...` and
HF Hub's `validate-yaml` endpoint rejects it with HTTP 400.

`train_lora()` from `explore_persona_space.train.sft` catches the 400 and
logs `WARNING Adapter upload failed — local copy preserved at <dir>`; the
adapter is preserved on disk but **NOT on HF Hub**. This is silent enough
that the experiment continues; the upload-policy violation only surfaces
when the upload-verifier checks Hub state at end-of-experiment.

**Why:** Almost every `/issue` orchestrator script first mirrors the base
model into `/workspace/<workdir>/qwen..._base` so subsequent train +
merge calls don't re-touch the HF cache, then passes that local dir as
`base_path` to `train_lora`. PEFT then emits a non-Hub-valid README. The
deeper fix lives in `src/explore_persona_space/train/sft.py` — it should
normalize `base_model_name_or_path` to the canonical HF id (e.g.
`Qwen/Qwen2.5-7B-Instruct`) before letting PEFT serialize the adapter.

**How to apply:**

- Detect mid-run: `grep -E "Upload failed: Invalid metadata in README"
  /workspace/logs/<run>.log`. If present, EVERY adapter from this orchestrator
  is on local disk only.
- Hot-fix loop (≤10 lines per adapter, no logic change):
  1. Patch `<adapter>/README.md` and `<adapter>/adapter_config.json`:
     replace `<local_mirror_path>` with the canonical HF id.
  2. Call `huggingface_hub.HfApi().upload_folder(folder_path=<local>,
     repo_id="superkaiba1/explore-persona-space", repo_type="model",
     path_in_repo="issue-<N>/<name>", commit_message=...)`.
  3. Run as background `nohup` so it doesn't gate the next step.
- One-off generic patch script at `/workspace/upload_couplings_now.py` on
  any pod that has hit this in the past — pattern-matches the 3 known
  local mirror paths (`qwen25_7b_instruct_base`, `em_merged`, `benign_merged`).
- Issue-262 hit this on every adapter (EM, benign, coupling × 4) and the
  experimenter ran the patch+upload pattern post-hoc; bouncing back to
  experiment-implementer was deemed wrong because the orchestrator code
  is correct, the bug lives in the shared training library and would need
  a code-reviewer round to fix in `src/`.

Filed as a follow-up: rewrite README/adapter_config base path inside
`train_lora` BEFORE the auto-upload runs.

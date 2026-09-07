# Conditional capture execution

This is a prepared continuation, not evidence that GPU capture has run. It applies only after the selected fresh996-trajectory cohort completes and its native audit passes. No capture is required if the bounded development feasibility gate fails.

Use the existing managed H200. Stop the owned vLLM supervisor after generation; verify its worker process group has no live members and the GPU has no serving context before loading Transformers. Preserve all server logs and exit evidence. Stage the reviewed branch, selected full fresh output, frozen manifests and selection on the pod, retaining every byte and checking all hashes. Never edit a native result to rewrite its stored VM paths.

The independent helper diagnostic is `eval_results/context_risk_followup_design/conditional_capture_smoke.py`, SHA256076e50453855a1758f5b4f49978a5c9ebfeee49c1f62be60066da38c42248080. Run it as ordinary Python without optimization so all assertions are active. Use a fresh report path. Its source hash is separate from the production capture source closure.

The diagnostic chooses the selected arm from selection.json and checks the actual longest prefix, the heaviest unequal production two-row batch, repeated capture equality, and hook/tuple parity. It never generates text or creates production capture chunks. Its explicitly recorded shortest-prefix tuple fallback is only for the auxiliary all-hidden-state memory check; the longest full prefix itself must still pass capture without truncation.

After checking actual free disk and package identities, use a fresh isolated environment; do not mutate the former serving environment. Pin vllm0.28.0, transformers5.15.0, torch distribution2.13.0, NumPy2.3.5, accelerate1.13.0, hydra-core1.3.2 and omegaconf2.3.0. The guard requires imported torch2.13.0+cu130 and CUDA13.0. Read the runtime evidence if the resolver produces anything else; do not weaken the guard.

```bash
UV_CACHE_DIR=/root/.cache/uv \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
uv run --no-project --python 3.12 \
  --with vllm==0.28.0 --with transformers==5.15.0 \
  --with torch==2.13.0 --with numpy==2.3.5 \
  --with accelerate==1.13.0 --with hydra-core==1.3.2 --with omegaconf==2.3.0 \
  python /workspace/explore-persona-space/eval_results/context_risk_followup_design/conditional_capture_smoke.py \
  --repository /workspace/explore-persona-space \
  --root /workspace/context_risk_followup \
  --review /workspace/explore-persona-space/eval_results/context_risk_followup_design/capture_code_review.json \
  --output /workspace/context_risk_followup/setup/capture_production_shape_smoke_v1.json
```

Execute the diagnostic under the approved detached process contract: fresh log and PID/process-group files, whole-process deadline, explicit exit receipt and descendant cleanup. Do not invoke the generic run_smoke command: that entry point uses short built-in prompts and performs text generation.

Only after diagnostic PASS, run scripts.context_risk_followup_capture with the selected arm, the same pinned environment and exact capture review. Model: BF16, layer44, last unpadded initial prompt position. Capture all249 contexts, batch rows2, padded-token budget16384, chunks15 and max prefix32768; no truncation. Validate capture_binding.json and every chunk/sentinel/token hash after transferring the captures to the VM root. Archive and verify all generation, capture, source and server outputs before managed GPU teardown. Perform any substantial CPU fitting on the appropriate CPU venue after releasing the GPU.

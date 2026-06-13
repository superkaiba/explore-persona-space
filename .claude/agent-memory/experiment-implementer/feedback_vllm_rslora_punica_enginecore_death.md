---
name: vLLM rsLoRA punica wrapper EngineCore death on LoRA load
description: vLLM 0.11.0 + rsLoRA + LoRA load silently kills EngineCore_DP0 after `peft_helper.py:55 Loading LoRA weights trained with rsLoRA.`; downstream ZeroDivisionError in pbar is misleading. enforce_eager=True doesn't fix it. #628.
type: feedback
---

vLLM 0.11.0 with `enable_lora=True` + an adapter trained with rsLoRA
(`use_rslora=True` → `vllm_lora_scaling_factor = lora_alpha / sqrt(r)`)
hits a silent EngineCore_DP0 death during the LoRA weight load step,
BEFORE any generation. The crash sequence:

1. Engine inits cleanly (`init engine (profile, create kv cache, warmup model) took 4.89 seconds`, `Cudagraph is disabled under eager mode`).
2. vLLM logs `INFO ... [peft_helper.py:55] Loading LoRA weights trained with rsLoRA.` — this is in `PEFTHelper.__post_init__` BEFORE the punica wrapper actually allocates the adapter weights into GPU memory.
3. ~7 seconds of silence.
4. NCCL fires `[W ProcessGroupNCCL.cpp:1538] Warning: destroy_process_group() was not called before program exit` — this is the signature: it's emitted as the EngineCore subprocess EXITS.
5. `ERROR ... Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.`
6. The parent process then calls `llm.generate(...)`, the engine is dead, the pbar's `elapsed=0`, and you get a misleading `ZeroDivisionError: division by zero` in `vllm/entrypoints/llm.py:1610`'s `in_spd = total_in_toks / pbar.format_dict["elapsed"]`.

**The ZeroDivisionError is a downstream symptom, not the cause.** The
real cause is the engine being dead — look at the immediately preceding
vLLM ERROR line. The downstream `pbar.elapsed=0` arithmetic error
fires anywhere the engine died before generation.

**enforce_eager=True does NOT fix it.** The crash is upstream of
cudagraph capture (cudagraph is already disabled by the log). The
crash is inside punica wrapper's LoRA allocation/setup.

**Why:** vLLM 0.11.0's punica wrapper has a known instability with
rsLoRA scaling factor + max_lora_rank=32 (verified shape) +
`enable_lora=True` engine. Same engine config WITHOUT rsLoRA (or with
`enable_lora=False`) works fine on the same hardware.

**How to apply:** any phase that runs vLLM inference WITH a LoRA
trained via rsLoRA (the project's default) needs:

1. **Fallback to HF transformers** for the LoRA inference step (slower,
   ~3-5x slower than vLLM for greedy generation; ~25h for a 16-cell ×
   30-context × 10-question phase on 1×H100). Use `PeftModel.from_pretrained(model, adapter_dir)` + `model.generate()`. This is the only known reliable fix.
2. **EPM_I628_SKIP_PHASE_4 / equivalent escape hatch** to skip the
   vLLM+LoRA phase when it's the SECONDARY measurement.
3. **Launcher must wrap the phase in `set +e` + standalone finalize**
   so an upstream vLLM crash on a secondary phase doesn't strand
   primary science with no `epm:results` sentinel. Pattern:
   ```bash
   set +e; run_phase 4 phase_4; set -e
   run_one_phase finalize  # ALWAYS, even on phase-4 crash
   ```
4. **Defensive guard in `_vllm_greedy`** to re-raise ZeroDivisionError
   as a clearer RuntimeError naming "EngineCore died before
   generation" — saves the next debugger from chasing the pbar
   symptom.

**Diagnosis grep:** if the vLLM log ends with `Loading LoRA weights trained with rsLoRA.` followed by NCCL `destroy_process_group() was not called` warning, you're hitting this exact crash class. Phase 0b without LoRA on the same engine config runs fine — proves the crash is LoRA-specific, not engine-config-specific.

Reference: #628 r9 attempt, log lines 2640-2666 in `issue-628-r9.log`. The pragmatic round-13 fix was launcher tolerance + standalone `--phase finalize`; the actual phase 4 vLLM+rsLoRA path was deferred as a secondary measurement.

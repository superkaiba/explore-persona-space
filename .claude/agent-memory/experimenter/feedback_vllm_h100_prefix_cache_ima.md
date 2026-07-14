---
name: vLLM H100 illegal-memory-access under heavy shared-prefix caching — A100-clean differential
description: vLLM 0.11.0 on H100 hits CUDA IMA in the engine step at long-prompt production shapes with heavy shared-prefix reuse (num_common_prefix_blocks ~237) while identical code runs clean on A100 and short-prompt probes pass on the SAME H100. Mitigate with enable_prefix_caching=False + enforce_eager=True. #1092.
type: feedback
---

vLLM 0.11.0 (torch 2.8.0) on 8× H100 crashed with
`torch.AcceleratorError: CUDA error: an illegal memory access` inside the
engine step at PRODUCTION shapes — ~3.8k-token prompts, 500-prompt
chunks, heavy shared-prefix reuse (`num_common_prefix_blocks=[237]`; a
dense-core crossing where many rows share one long prefix maximizes
prefix-cache sharing). One IMA kills the engine; every queued shard on
that worker then errors (35/42 shards on #1092 launch #4).

The DIFFERENTIAL that pins the class (run these before any code hunt):
1. Identical code + corpus ran 42/42 clean on A100s — not a code/data bug.
2. A minimal 1-GPU probe on the SAME H100 pod (64 short prompts, prefix
   caching ON) is CLEAN — not a pod/driver defect.
⇒ shape/load-dependent H100 prefix-caching/cudagraph IMA family.

Mitigation (measured on the crash rows: 512/512 clean, ~1 min/512
prompts on 1 GPU even in eager): plumb `enable_prefix_caching=False` +
`enforce_eager=True` as DEFAULT-OFF engine knobs and enable them for the
affected run only — keep the flags off elsewhere (byte-identical engine
args when off; test-pinned). All cells of a comparison run under the
same engine config for comparability; a downstream identity gate
(G2-style) revalidates capture independently of engine mode.

Reference impl: `_vllm_engine_overrides`
(`scripts/issue1092_gpu_phase.py` @ 5031174dc3);
`scripts/issue1092_dispatch.sh` passes both flags.

# DeepSeek retry repair, 2026-09-19

The user explicitly requested continued automatic retries. Attempt 7 obtained eight H200s but stopped before model weights/capture: Transformers 5.15.0 rejects kernels 0.17.1 (required >=0.16.0,<0.17.0). The capacity checker then treated its needs_attention handoff as terminal to automatic continuation. Two watchdog workers repeated the unchanged constraint and exhausted that incident's recovery allowance.

The repaired runtime pins kernels 0.16.1 for DeepSeek and tests Transformers eligibility before loading the immutable FP8 kernel. The model, kernel revision, numerical tolerances, stimuli, precision and analysis are unchanged. Qwen retains its completed source/runtime/artifacts. Fresh DeepSeek publications use 20260919_v4; attempt 7 failure evidence remains preserved at immutable HF revision a0c1649205f84675d9e107bde2bdd4662da89e7d.

The cumulative 17 GPU-hour ledger includes Qwen 0.6542087623808119 and failed DeepSeek 1.229034685028924, leaving 15.116756552590264. A 6600-second (110-minute) eight-H200 allocation consumes at most 14.666666666666666 GPU-hours. The contract helper checks recorded paid time and rejects the old 7200-second allocation, unresolved previous allocations, or deadline extensions. Deadline starts at provider createdAt, including setup; the existing 900-second preservation reserve and measured throughput gate remain mandatory. No promise of completion within this unmeasured envelope.

## Smoke run

### fix-engaged signal

`[fp8-kernel-ready] kernels=0.16.1 revision=8c178950e8710e26a2210e4e909cd02dc16c8715 loader=transformers.lazy_load_kernel`

Observed in the actual production pin_fp8_kernel function before any new provision. It proves the Transformers eligibility and lazy-loading path returned the expected immutable module, interface and file hashes. See kernel_loader_smoke.json. A negative integration run with 0.17.1 reproduces the new explicit rejection.

Smoke blind-spot enumeration: local loader smoke uses the existing Torch 2.8 CUDA wheel with Triton 3.5.1; it does not execute CUDA kernels, load model weights, or validate GPU parity/performance. Separately, Torch 2.9.1 CPU with the exact Transformers/kernels/Triton versions passes eligibility but correctly has no compatible CUDA variant. No acceptance condition or numerical gate is bypassed in production. The actual Torch 2.9.1+cu128 runtime must repeat the loader signal, numerical and throughput gates on the pod.

Stale artifacts: fresh pod/output tree, no remote capture resume fetch, new HF run prefix; historical attempt 7 remains preserved and is never treated as capacity_lost or successful capture. Operator rearm archives the old watchdog incident, keeps all seven attempt records and every paid allocation, and registers the repaired source as a new bounded incident. One-minute personal capacity polling and a single continuation worker remain enforced.

Failure lesson: a direct kernel import bypassed Transformers' version gate and therefore could not validate the production integration. Exercise the actual lazy loader before provision and check installed dependency compatibility explicitly. Root cause confirmed; generalizes to other optional-dependency integrations.

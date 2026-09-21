# Attempt 10: grouped FP8 scheduling repair

Attempt 9's first three dense blocks matched across singleton and mixed batches. All 16 rows in the two M64 groups diverged after the first MoE block; both rows in the M16 group matched all 61 blocks exactly. This motivates fixing the grouped block-FP8 M tile at 16. It does not establish GPU correctness of that repair.

The runtime verifies the immutable `grouped.py` and `utils.py` SHA256 values and compares the loaded original functions with code compiled from those files. It replaces only the grouped module's scheduling reference, rejects calls from any dispatcher other than the reviewed block-FP8 body, and retains all original kernel file hashes. The manifest records the scheduling override and hashes the repair module. Other kernel families, weight formats, model/input pins, eight-GPU placement and analysis remain unchanged.

## Smoke run

### fix-engaged signal

Observed through the real Transformers 5.15.0 loader, kernel 0.16.1 package and registered custom-op body in the local integration test:

```
[fp8-grouped-m16-engaged] dispatcher=grouped._w8a8_block_dynamic_fp8_matmul_grouped target_m=2 original_m=16 fixed_m=16
[fp8-grouped-m16-engaged] dispatcher=grouped._w8a8_block_dynamic_fp8_matmul_grouped target_m=43 original_m=64 fixed_m=16
```

The signal is emitted only after the actual block-FP8 dispatcher reaches its M scheduling decision. Three integration tests cover this path and rejection of changed source/revision and a prepatched function. Local execution uses Torch 2.8.0+cu128 and Triton 3.5.1; the external Triton GPU launch is substituted. The model runtime remains Torch 2.9.1+cu128. GPU validation is pending.

Before model weights load on the next approved node, the runtime performs a tiny CUDA diagnostic using native E4M3 weights, UE8M0 scales, BF16 activations, 256 experts and the actual Transformers dispatch function. It compares shared rows across shapes whose original M sizes are 16 and 64, for three N/K shapes including width 7168, before and after the override. It persists each completed measurement and failures, restores the fixed scheduler even on exceptions, and requires the unchanged 0.01 parity and 0.00001 repeatability limits. The full-model smoke and measured-throughput gate remain mandatory. Measured diagnostic values stay outside the capture fingerprint, preserving valid same-source resume.

Smoke blind-spot enumeration: the local integration test substitutes the CUDA launch, so it proves loader/dispatcher wiring and override engagement only. The tiny CUDA diagnostic samples kernel shapes; it does not certify model routing, all residuals, scientific quality or production throughput. Those remain governed by the unchanged full-model smoke and capture gates. No GPU diagnostic or full-model repair PASS is claimed here.

Stale artifacts: attempt 9's ten failure files remain untouched at immutable HF revision `726d361ebd707ab7a3bc56718990d3d3a5f926a4`. Their names, sizes and hashes were independently rechecked before archiving its terminated handle. The next fresh pod has no prior local output/sentinels and the capture does not fetch remote resume state. Publications use the new DeepSeek prefix `issue2673_deepseek_comparison/20260921_v6/deepseek`; Qwen retains historical source `cb794596d9a093b8d3c897352ed64874b8b69d76` and `20260917_v3`. Any later same-pod repair must quarantine the failed output and derived sentinels outside the local `chunks/*.pt` resume tree after verified preservation.

The fresh ledger permits a 3600-second eight-H200 allocation: 8.63055398464203 GPU-hours spent, 8.36944601535797 remaining, at most 8 additional GPU-hours. This is an upper bound, not a completion estimate. Recheck the ledger before provision and retain the final 900-second preservation/analysis reserve.

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: native FP8 grouped MoE residual capture
lesson: Kernel scheduling can change with batch shape even when weights and inputs are pinned. Check the actual production dispatcher and autotune key, preserve original kernel hashes, and require unchanged full-model parity after any scheduling repair; a local dispatch test cannot establish GPU correctness.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: no
<!-- /epm:failure-lesson -->

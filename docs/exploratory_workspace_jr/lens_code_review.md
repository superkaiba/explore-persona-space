# Independent review: workspace_lenses primitives

**Final disposition: PASS for the reviewed primitive/adapter scope. The initially identified RMSNorm precision defect is resolved. No remaining actionable defect found.**

Reviewed only `src/explore_persona_space/analysis/workspace_lenses.py` and `tests/test_workspace_lenses.py` at `/mnt/eps-data/thomasjiralerspong/wt-jr-workspace-predictability`, against `/tmp/jr-lens-source-notes.md`. No implementation/test edits, pretrained-model inference, or experiment launch. Reviewed file SHA256s:

- Source: `4cac48973e29d6d7eaeebb189cd738365d4b71c0f28c687685d975a08d3b7349`
- Tests: `b3d051e1cbbdfe5f3bcfca82fefedd39f5873150e7659e00d7ffe6de7726b3e7`

## Resolved finding

The initial version rounded `self.weight + 1` in the parameter dtype for Qwen3.5 and collapsed RMS/gain factors into a scale cast to input dtype before cotangent multiplication. This preserved forward bits while introducing backward rounding differences beyond the published denominator-detachment rule. The original deterministic bf16 reproduction had 13/48 differing input-gradient coordinates, maximum absolute difference 0.015625; even correcting the gain alone left 10/48 differences.

The revised implementation routes backward through an explicitly denominator-detached native expression, using an exact-forward surrogate autograd wrapper. It forms Qwen3.5 gain as `1 + self.weight.float()`, multiplies before casting for Qwen3.5, and casts normalized activations before multiplying the gain for Qwen2/3. This preserves the native backward operation order instead of collapsing its low-precision arithmetic.

**Independent fix verification:** reran the original seed-123, [3,16] tensor reproduction with nontrivial learned gain and arbitrary cotangents in both bf16 and fp16, for both casting conventions. Every case has exact unchanged forward bits and **0/48 differing gradient coordinates** against a native source-equivalent expression with only inverse-RMS detached.

[Official Qwen3.5 norm implementation](https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen3_5/modeling_qwen3_5.py), [published R scope](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers).

## Other reviewed behavior

- **Nonnegative GP:** compact storage matches the cited full-vocabulary algorithm's current support, signed argmax, active-atom reselection, reactivation of clipped atoms, restricted-gradient line search, and coefficient clipping. It does not substitute NNLS or force a new atom per step. Zero-denominator updates are explicit and counted; realized L0 is reported. The NumPy oracle uses an independent dense representation. [Primary GP algorithm](https://www.lesswrong.com/s/AtTZjoDm8q3DbDT8Z/p/C5KAZQib3bzzpeyrg)
- **Sparse interpretation:** component plus remainder reconstructs the input; no claim of orthogonal projection or exact sparse optimum. Error increases after clipping are visible diagnostics rather than silently modified updates.
- **Activation/product rules:** SiLU and exact/tanh GELU use the correct detached factor, including analytic handling at zero. Half-product splits gradients between both branches and handles broadcasting. Additional independent bf16/fp16 tests compared SiLU to its explicit factor-detached expression and broadcast half-product VJPs to `0.5*z + 0.5*z.detach()`: gradients matched exactly in all checked cases. SiLU forward remained bit-equal to native `F.silu`; no evidence of an analogous low-precision activation/product defect.
- **Scope:** path-based residual norm selection excludes q/k norms and Qwen3.5 gated-attention norm. Native attention, GDN recurrence, and attention gates stay ordinary, as required by the published dense-R variant. MoE/unknown MLP/norm classes fail before mutation. The correct Qwen3.5 zero-centered gain is explicit.
- **Restoration:** instance-level forward overrides are saved and restored; temporary overrides for inherited class forwards are deleted. Cleanup runs in reverse order after ordinary exceptions. Nested use is structurally consistent.
- **J validation:** central finite differences check the ordinary derivative; the adjoint identity is checked. The R coefficient is correctly not required to match ordinary finite differences, and the tests enforce that distinction.

## Executed validation

Command:

`PYTHONPATH=/mnt/eps-data/thomasjiralerspong/wt-jr-workspace-predictability/src uv run python -m pytest /mnt/eps-data/thomasjiralerspong/wt-jr-workspace-predictability/tests/test_workspace_lenses.py -q`

Result: **16 passed in 3.55s**. This includes independent dense-GP oracle checks, local propagation rules, both bf16 normalization casting conventions, ordinary-J validation, and a tiny random actual-library Qwen2 model's forward parity/restoration.

The separate deterministic CPU bf16/fp16 VJP comparisons described above also passed. These are unit/integration checks, not experimental or pretrained-model results.

## Validation limits

This review does **not** certify an actual Qwen3.5 checkpoint, native hybrid-attention gradients, model-level fit serialization, calibration-document identity, vocabulary dictionary gain folding, released-artifact compatibility, or end-to-end paired J/R fits. Those remain separate validation gates. The J check is directional, so callers need nondegenerate directions/cotangents and a precision-controlled downstream function. Source-supported adapter scope should not be presented as completed pretrained-checkpoint validation.

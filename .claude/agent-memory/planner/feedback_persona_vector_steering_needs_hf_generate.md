---
name: Persona-vector steering needs HF generate(), not vLLM
description: Any Persona Vectors steering / preventative-steering experiment (h_ℓ ← h_ℓ + α·v_ℓ per decode step) MUST use HF model.generate() with a register_forward_hook — vLLM cannot host a per-step residual-stream hook; size the ~6× slowdown into the budget
type: feedback
---

For any plan running Persona Vectors (arXiv 2507.21509) steering (Exp 2) or
preventative steering during finetuning (Exp 4) — the intervention
`h_ℓ ← h_ℓ + α·v_ℓ` applied at each decode step — generation MUST use HF
`model.generate()` under a `register_forward_hook`, NOT vLLM.

**Why:** the paper's own code release (`safety-research/persona_vectors`,
`activation_steer.py::ActivationSteerer`, `eval_persona.py::sample_steering`)
uses `with ActivationSteerer(...): model.generate(...)`. vLLM's batched engine
does not expose a per-step residual-stream write hook, so the steering vector
cannot be injected each decode step. This contradicts the always-on "use vLLM
for generation" default — name it as a load-bearing §12 assumption.

**How to apply:** in §8 compute-projection, size steered generation at ~6× the
vLLM throughput (HF `.generate` batched bs≈20 on 8× H100; the FLOPs floor does
NOT apply the usual way because steering forces HF). Exp-4's POST-ft eval (no
steering at eval) CAN use vLLM — only the steered generation + preventative
TRAINING forward pass need HF. The training-time hook must port the PEFT
path-rewrite (`base_model.model.layers` vs `model.layers`,
`training.py::add_steering_hooks`) since PEFT wraps the model. Selected steering
layer for Qwen2.5-7B-Instruct = layer 20 (1-indexed; hook at layer_idx 19),
fixed in the code release across eval_steering.sh + cal_projection.sh.

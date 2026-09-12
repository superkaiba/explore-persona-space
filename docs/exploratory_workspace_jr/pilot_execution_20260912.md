# Execution continuation, frozen before pilot outcomes

The user's earlier explicit task-registration exception was verified in the
completed Codex task `01a0939c-080a-7d13-9279-a4d9a77dccb0`. The existing sample
manifest, model pair, layers, primary k=10, k=5/25 sensitivities and 0.05 gap
threshold remain the scientific specification. No component test results have
been opened. This continuation supplies the missing execution stages.

## Compute and pilot sequence

Live GCP discovery on 2026-09-12 found no running instances. Regional quotas
permit A100-80GB instances. The pilot targets one A100-80GB, 170 GiB host RAM,
300 GB SSD in the dedicated `eps-persona-gpu-jun2026` project, using the
repository's DLVM family. Start with Qwen3.5-27B, pinned to the saved revision.
Do not stage model weights on the shared development VM. A bounded six-hour
pilot reservation uses STOP, not DELETE, as its automatic time-fence action,
preserving the persistent disk if execution or upload fails. No other task's
instance is reused or stopped. A result upload and hash verification must
precede deletion of this experiment's persistent disk. This task-less pilot
uses the existing GCP project/image/zone configuration and records a separate
owner manifest because the task-keyed dispatcher cannot accept a missing ID.
Do not invent an issue ID to satisfy that dispatcher.

The initial native gate validates one frozen calibration prompt, then measures
one paired J/R matrix before sizing the full calibration. An initial end-to-end
pilot may use the first two valid calibration prompts (hash order), explicitly
labelled `pilot-only-two-calibration-prompts`; these matrices cannot serve the
main experiment. Generation uses the already frozen 64/16/32 pilot contexts,
K=5 and the saved decoding settings. Main evaluation stays unopened until
native parity, numerical ordinary-J checks, R local rules, reconstruction,
runtime, and calibration stability have passed. Calibration counts refer to
valid prompts (119 realized out of 128 selected); never call 119 a 128-prompt
fit. Stability compares the first 32/64/all-valid prompts.

Use a remote project virtual environment with `transformers==5.16.1` (the
mapping producer's version), and record all resolved packages. Native Qwen3.5
must import explicitly; availability of an AutoModel factory in an old
Transformers install does not establish model support. Runtime tuning changes
only memory/chunk dimensions, never calibration/example selection.

## Independent null before real-model outcomes

Execute a small exactly affine synthetic control with deterministic Gaussian
inputs, dense affine token targets repeated across equal-weight rollouts, and
independent normalized overcomplete dictionaries. This is an implementation
and mechanism control, not a language-model result or a replacement for the
real-dictionary affine null. Use seed 20260912, d=16, 128 atoms, split sizes
256/64/128, all three k values, and the three preregistered rotations. Fit the
same ridge helper and paired context bootstrap. A positive component gap here
shows that decomposition alone can create such a gap. The generated data are
justified by the explicitly synthetic, exactly affine null requirement.

For every predictor, target scaling uses training-only centering and one
scalar per target. MLP model selection uses mean validation SSE across its
three seeds, never test loss; report seed-specific predictions and gains.
Per-example arrays, fitted parameters, split hashes and bootstrap samples are
saved, with undefined metrics represented as null plus status in JSON.

Smoke blind-spot enumeration: synthetic controls do not certify native model
architecture, model capability, lens calibration, generated-token capture or
real-model predictability. The native one-prompt gate does not certify
calibration stability, main-scale memory/runtime or end-to-end generation.

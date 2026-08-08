---
name: rslora-parity-probe-needs-A100-intent
description: rsLoRA numeric-parity probes minted on A100 fail on L4 (bf16/TF32 precision mismatch); launch with --intent lora-7b not --intent eval
type: feedback
---

An rsLoRA numeric-parity probe (the `_rslora_parity_probe` shape inherited
from #667 / minted in #537) that reproduces a committed diagonal-write
gauge from a prior A100-80-trained task **FAILS on an L4** (compute
capability 8.9, `g2-standard-4`) because bf16/TF32 precision differs from
the A100-80 the gauge was minted on — it is a gauge/hardware precision
mismatch, NOT a code regression, and the probe correctly HALTs in
`phase_reextract_prefetch` with subprocess rc=1 before any GPU spend on
the sweep.

**Why this is the experimenter's problem:** the launch intent decides
which GCE machine the GCP-lane router picks via `INTENT_TO_MACHINE` in
`backends/gcp.py`. `--intent eval` → `g2-standard-4` (1× L4);
`--intent lora-7b` → `a2-ultragpu-1g` (1× A100-80). A plan that says
"forward-pass extraction, no training" looks like an `eval` intent at
first glance, but if the workload's parity probe reproduces a gauge
minted on A100-80, only A100-80 will pass.

**How to apply:**

- Before launch, if the workload includes an inherited rsLoRA parity
  probe (the canonical `_rslora_parity_probe` / `assert_adapter_gauge`
  shape), check the **gauge's origin hardware** — typically named in the
  parent issue's reproducibility card or its `eval_results/issue_<M>/...`
  metadata. If the gauge was minted on A100-80 (the #537 / #667 line),
  launch with `--intent lora-7b` regardless of whether the workload
  itself is forward-pass-only.
- A plan §9 that names "A100-80" as target hardware while passing
  `--intent eval` is internally inconsistent — `eval` does NOT resolve to
  A100-80. The experimenter must align the launch intent to the named
  hardware, not the literal `eval` keyword.
- Diagnosis pattern: a Python `RuntimeError: rsLoRA NUMERIC parity probe
  subprocess exited rc=1` for any `<behavior>/<source>` cell, fired by
  `_run_parity_probe_subprocess` in `scripts/issue<N>_dispatch.py`,
  during `phase_reextract_prefetch` or any phase that calls the
  inherited probe, **on an L4-class GCE instance** = this exact class.
  Re-launch with `--intent lora-7b` (or override `--gpu-type A100-80`
  explicitly).
- This generalizes to ANY parent task whose adapters / read-outs /
  numerical gauges were minted on a specific GPU class — the gauge's
  origin GPU is the implicit precondition the parity probe enforces.

Closed regression: task #667 `a36-readout-reextract-cos` round 1 (GCE
attempt `att-20260625-105641`, 2026-06-28 22:46 UTC), round 2 relaunched
on A100-80 same code.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [rsLoRA parity probe needs A100 intent, not eval](feedback_rslora_parity_probe_needs_a100_intent.md) — an rsLoRA parity probe minted on A100-80 (the #537/#667 line) FAILS on L4 (`--intent eval` → `g2-standard-4`, cc 8.9 bf16/TF32 precision mismatch); plan §9 naming "A100-80" requires `--intent lora-7b` (→ `a2-ultragpu-1g`), regardless of whether the workload is forward-pass-only — #667 a36 round 1

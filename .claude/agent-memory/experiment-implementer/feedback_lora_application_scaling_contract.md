---
name: Reused LoRA adapters carry an application-scaling contract
description: rsLoRA-trained adapters applied at faithful alpha/sqrt(r) can behave totally differently (unconditional repeaters) than at the classic alpha/r the parent's committed numbers were produced at — probe one adapter at both scalings before any cross-task re-read
type: feedback
---

Reused LoRA adapters carry an APPLICATION-SCALING contract, not just bytes. An
rsLoRA-trained marker adapter can be an unconditional marker-repeater at faithful
`use_rslora=True` (scale α/√r) application while the parent task's committed numbers were
produced at classic α/r application.

**Signatures of the over-application collapse ceiling:** N different adapters re-reading
to IDENTICAL Δg (±0.01 nat), or teacher-forced Δg == −mean(b_logp) for every cell.
Neither is a mapping scramble.

**Why:** incident #601 round 5 (2026-06-11) — all 20 reused #472 adapters re-read to the
collapse ceiling at faithful rsLoRA scaling; the parent's committed dose-response
reproduces only at α/r = 2.0. The Phase-0a #534-class gate caught it pre-training.

**How to apply:** before any cross-task adapter re-read, read `adapter_config.json`
scaling fields (`use_rslora`, `lora_alpha`, `r`) and run a 1-adapter apply-and-read
parity probe against the parent's committed numbers on the CURRENT stack; pin the read
gauge explicitly (staged scaling-patched copy + sha256/scaling provenance per read).
Reference implementation: `neg_setpoint_601/artifacts.py::stage_parity_read_adapter`.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [LoRA application-scaling contract](feedback_lora_application_scaling_contract.md) — rsLoRA α/√r vs classic α/r flips repeater behavior; probe both scalings + pin the read gauge. #601.

## Merged sibling index rows (#2032 curation, 2026-08-03)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the agent-memory index size cap (task #2032). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [bf16 merge truncates small LoRA deltas](feedback_bf16_merge_truncates_small_lora_delta.md) — merge_and_unload attenuates early/low-lr (#480)

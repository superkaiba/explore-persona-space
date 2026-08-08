---
name: apply-parity probe N sizing follows Wilson-CI half-width
description: When a ±tolerance apply/reproducibility gate scores a proportion, size N so the Wilson-CI half-width at the expected rate is BELOW the tolerance — not an arbitrary small N.
type: feedback
---

A ±tolerance apply/reproducibility gate on a PROPORTION must set N from the Wilson-CI half-width at the expected rate, not an arbitrary small N. A ±0.10 tolerance gate at rate ~0.7 needs n≥100 (Wilson-CI half-width ~0.09 encompasses the tolerance); n=10 gives ~0.27 and false-fails even under a perfect apply.

**Why:** Task #667's Phase-0.5 rsLoRA apply-parity probe (plan v6 §5) chose n=10 samples with ±0.10 tolerance, citing "Wilson-CI on n=10 comfortably encompasses this at α=0.05". The claim was empirically wrong — Wilson-CI half-width at rate 0.7 with n=10 is ~0.27. The probe FAILed on the pod with `|E+_committed − E+_current| = 0.30`, indistinguishable from expected sampling noise. The plan's tolerance was structurally guaranteed to false-fail. Round-4 fix raised N_SAMPLES 10 → 100 (half-width ~0.09).

**How to apply:** Every parity / reproducibility probe that reads a RATE (proportion) via an LLM judge or judge-scored sampling must compute N from a target Wilson-CI half-width AT the expected rate. Quick reference (half-width at rate=0.7, α=0.05):
- n=10 → 0.27
- n=25 → 0.17
- n=50 → 0.12
- n=100 → 0.09
- n=200 → 0.06

Pick N so that Wilson-CI half-width ≤ (tolerance − expected_noise_margin). Also: reuse the project's canonical judge from `src/explore_persona_space/eval/alignment.py`'s Betley two-axis EM protocol (or the domain analogue), NEVER a hand-rolled restatement; a rubric-drift source stacks with sampling noise and multiplies the false-fail risk. Judge always Sonnet 4.5 (`.claude/rules/llm-judging.md` rule 11), drop-never-coerce (rule 9).

Origin: #667 round 4, 2026-07-01. `epm:strategy-pivot v2` + `epm:failure-lesson v1`. Commit `02ac219084`.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [apply-parity probe N sizing follows Wilson-CI half-width](feedback_apply_parity_probe_n_sizing.md) — ±tolerance rate-gate must set N from Wilson-CI half-width at the expected rate (#667 pivot 1, n=10 at rate 0.7 gives half-width ~0.27 not ±0.10 — false-fails guaranteed)

---
name: parity-floor-weak-writer-vs-gauge-error
description: rsLoRA diagonal-write parity floors calibrated on 7-module adapters false-HALT weak 4-module/low-LR writers; gauge errors are multiplicative √r discrepancies, not 10% shortfalls
type: feedback
---

A rsLoRA diagonal-write parity floor calibrated on 7-module adapters (em
~0.17, sycophancy ~0.08 write ratio) false-HALTs a correctly-applied WEAK
writer — the 4-module/α=64/low-LR marker adapter measures ~0.009, a 10%
shortfall vs the generic 0.01 floor, not a failure.

**Why:** a TRUE rsLoRA gauge error (α/√r vs α/r) is MULTIPLICATIVE — at
r=32 it is a √32 ≈ 5.66× discrepancy (a wrong-gauge marker write reads
~0.0016 or ~0.05), never a 10% shortfall. A floor meant to catch gauge
drift must sit BETWEEN the correct value and the wrong-gauge value per
adapter class (e.g. marker floor 0.004 separates correct 0.009 from
wrong-gauge 0.0016 with ~2.25× margin each way), not at a one-size value
calibrated on stronger writers. (#813 round-4 crash-fix, 2026-07-01: the
apply-parity gate HALTed a healthy launch on marker/default.)

**How to apply:** when wiring an apply-parity / write-ratio gate over a
mixed adapter fleet, (1) set per-behavior/per-adapter-class floors
grounded on each adapter's measured correct-stack value + the √r
separation argument; (2) for marker adapters add the cheap teacher-forced
Δ log P(marker id 83399) ≥ 1 nat behavioral confirmation (a no-op /
wrong-gauge application reads ≈0) rather than leaning on the write-ratio
alone; (3) keep shared reused helpers' defaults byte-identical (thread the
floor as an optional parameter).

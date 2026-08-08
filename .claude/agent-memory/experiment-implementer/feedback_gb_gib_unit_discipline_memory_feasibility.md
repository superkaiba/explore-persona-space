---
name: feedback-gb-gib-unit-discipline-memory-feasibility
description: Memory-feasibility verdicts (fits/infeasible per device) must convert GB<->GiB explicitly and state both units — device specs are decimal GB, tensor arithmetic is GiB; a 7% unit error lands exactly on the margin that matters (#1491 r3a)
metadata:
  type: feedback
---

When a memory-feasibility argument DECIDES whether a rung/phase can run on a
device (a fits/infeasible table, a fail-loud guard threshold, an HBM sizing
row), do the GB/GiB conversion explicitly and state both units — never carry a
marketing capacity ("141 GB", "80 GB") into GiB arithmetic as-is.

**Why:** #1491 round-3a (2026-08-04): my fp32 parity-probe feasibility table
treated the H200's 141 GB (decimal) as 141 GiB. Real capacity 141e9 B =
131.3 GiB, so the 32B fp32 headroom I reported as "~11 GiB spare" was +9.3 GiB
raw and only **+1.3 GiB after the guard's own 8 GiB margin** — the verdict
survived but reads completely differently at launch. Worse, the unit error
propagated into a WRONG secondary inference: I speculated bf16 32B "also would
not load on a shared fellows node, so this may be a pre-existing rung
constraint" — with correct units a shared node's ~73.3 GiB free FITS bf16 61.0
GiB fine, so the fp32 probe was a NEW constraint the fix introduced on the
headline rung, which changes it from "inherited limitation" to "launch decision
to surface". The 1024^3/1e9 ratio is only 7.4% — invisible in prose, decisive
at margins.

**How to apply:** (1) compute in BYTES and print both `X_GB = n/1e9` and
`X_GiB = n/2**30` in any feasibility table; device capacities enter as decimal
GB × 1e9 (H200 "141 GB" = 131.3 GiB; A100 "80 GB" = 74.5 GiB; H100 "80 GB"
same). (2) Re-check every fits/infeasible verdict that sits within ~10% of the
boundary after the conversion. (3) When a verdict flips or tightens, re-derive
every DOWNSTREAM claim built on it (esp. "pre-existing vs newly-introduced
constraint" — reviewers act on that distinction). Related:
[[feedback-memory-cap-calibrate-measured-peak]] (measure the live factor;
this entry governs the CAPACITY side of the same arithmetic).

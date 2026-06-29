---
name: Don't re-derive a committed FDR/PASS verdict — read it
description: When recomputing a clean-result's headline verdict from raw cells, re-derive ρ/floor/draws as a cross-check but READ the PASS/FDR flag from the committed analyzer_body_data.json
type: feedback
---

When a round-2 fix recomputes a headline verdict from the raw checkpoint
grid (e.g. #658's "structured A3.2 PASS 3/10"), re-derive the per-cell
ρ / noise-floor / bootstrap-draws as a BIT-IDENTICAL cross-check, but
take the PASS / FDR-rejection flag from the COMMITTED, reviewed verdict
(`analyzer_body_data.json` / `aggregate.json`), NOT a fresh re-implementation.

**Why:** A re-implemented Benjamini-Hochberg over the 10 per-behavior
BEST cells is far MORE LENIENT than the committed BH over the FULL
structured grid (~1000 cells). In #658 my re-derived FDR PASSed 5/10
(Betley) + 4/10 (UltraChat) while the committed full-grid FDR is 3/10
both — the headline. The ρ/floor/draws were bit-identical (so Δρ + the
per-context scatter were correct); only the FDR multiplicity scope
differed. Publishing the re-derived count would have silently contradicted
the reviewed body's 3/10.

**How to apply:** In the round-2 analysis script, load the committed
verdicts, assert `abs(rederived_rho - committed.struct_rho) < 1e-3` per
behavior (fail loud on grid drift), then copy `struct_fdr` / `struct_pass`
straight from the committed file. The re-derivation's job is (a) confirm
reproducibility and (b) supply the per-arm bootstrap draws a downstream
Δρ CI consumes — never to re-adjudicate PASS. A genre-delta verdict
(H1/H2/H3) depends only on whether the Δρ CI overlaps zero, so it is
unaffected by the PASS scope and can be read off the draws directly.

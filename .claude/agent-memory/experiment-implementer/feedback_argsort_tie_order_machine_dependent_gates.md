---
name: Recompute-equality gates on top-k-by-count selections are machine-dependent at ties
description: np.argsort(-counts) tie order varies with CPU SIMD kernels (numpy 2.x x86-simd-sort AVX-512 vs none) — gate banked selections with set-validity invariants, never array equality (#1946)
type: feedback
---

Never gate a reused/banked top-k-by-count feature selection with recompute-equality
(`(recomputed == banked).all()`): the top-k step in `SN.restrict`-style helpers uses
unstable `np.argsort(-counts)`, and numpy 2.x dispatches CPU-specific x86-simd-sort
kernels (AVX-512 on GCE n2 Cascade Lake vs none on the shared-VM Xeon), so tie ORDER
at the cap boundary differs per machine even on byte-identical inputs + identical
numpy/torch versions. #1946: 55,146 eligible, 5 features tied at boundary count 7198
for the last 2 of 16,384 slots — gate passed on the VM, crashed on GCE.

**Why:** argsort's default (introsort) is unstable by contract; SIMD kernel dispatch
makes the instability machine-dependent, so "same code + same data ⇒ same selection"
is false whenever counts tie at the cap boundary.

**How to apply:** the BANKED selection is authoritative (downstream predictions were
fit on it — rebuilt target columns must match it exactly); validate it with
machine-independent SET invariants against the recomputed counts: floor matches,
len == min(cap, n_eligible), all banked counts >= floor, every feature with
count > boundary (= min banked count) is in the banked set, and
n_strictly_above < cap. Log the boundary structure (boundary count / n above /
n tied / tie slots). `kind="stable"` argsort only fixes within-machine determinism,
not cross-artifact identity — set-validity is the portable gate.

## Merged sibling index rows (#2032 curation, 2026-08-03)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the agent-memory index size cap (task #2032). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- feedback_numpy_argsort_tie_order_cross_machine.md — numpy argsort tie order is CPU-SIMD-dependent: gate banked top-k selections by set-validity invariants, never cross-machine recompute-equality (#1946)

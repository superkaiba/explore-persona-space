---
name: eligibility-export-call-chain-identity
description: A "single-source" eligibility export is certified by matching each consumer regime's PRODUCTION branch call chain verbatim, not by its docstring; residual = unpinned tokenizer/env the filter consumes
metadata:
  type: feedback
---

When a fix adds an EXPORT mode whose output restricts a registered manifest/sample
("single source of truth, no duplicated filter logic"), certify the claim at the
CALL SITES, not the docstring: read every consumer regime's production branch in
main() and confirm the export runs the IDENTICAL function chain on the identical
pool object per regime (e.g. #2479 r8: `load_paired_pool` → `_filter_pool_feasible(op_companion=False/True)`
exactly mirrors the paired and `--op-powered` arms; `--op-companion` correctly
excluded because it asserts NOT-manifest-bound). Also check the export's
environment inputs: a filter keyed on an UNPINNED tokenizer (`from_pretrained`
with no `revision=`) can drift between export-time and consume-time environments —
Minor when the consume-side guard fails loud, but name it.

**Why:** the docstring claimed single-source; only the branch-by-branch call-chain
diff proves no regime was proxied or skipped (sibling trap: [[registered-gate-quantity-substituted]],
[[banked-parent-dual-schema-equivalence]]). Gate-falsification check for the consumer
test: deleting the restriction must fail a COUNT assert (e.g. `n_eligible`), not
just an rng-lucky sample-content assert.

**How to apply:** any diff exporting eligibility/feasibility/allowlist sets for a
sampler or manifest builder: (1) enumerate consumer regimes from main()'s branches;
(2) diff each branch's chain vs the export's; (3) check which regimes are
correctly EXCLUDED and why; (4) probe the export's env-dependent inputs for pinning.

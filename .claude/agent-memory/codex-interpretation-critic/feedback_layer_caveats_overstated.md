---
name: Lens-2 marginal p-value at intermediate layers overstated as "caveat vanishes"
description: "All caveats vanish at L15-L25" overclaims when L15 still shows a marginal p for the key test
type: feedback
---

In issue #269, the body claimed "all caveats vanish at L15-L25" for the within-cluster-stratified Mantel test. But at L15, the stratified Mantel p=0.042 (marginal), only fully resolving at L20 (p=0.002). Selectively quoting only the L20 p value while making a sweeping claim about L15-L25 is an overclaim.

**Why:** Analyzers naturally want to say the headline result strengthens with depth. But if only one or two layers (L20, L25) cleanly pass a caveat-test, the correct framing is "resolves at L20, not L15."

**How to apply:** When a body claims "caveats vanish at layers X-Y", load the JSON for each layer in that range and verify the relevant test passes at each layer individually. Check for marginal p-values (0.01 < p < 0.05) that would make "vanishes" too strong a word.

---
name: small-cell bootstrap-CI degeneracy in mechanized gates
description: A bootstrap 95% CI over <=~7 unique units collapses its upper percentile onto the point estimate; gate bindingness on CI validity (lo < point < hi strict), never on an equality-inclusive containment test or a tuned n-floor
type: feedback
---

A conversation/cluster-bootstrap 95% CI over <=~7 unique units collapses its
97.5th percentile onto (or below) the cell's own point estimate, so an
equality-inclusive parity/containment gate (lo <= ref <= hi) auto-fails any
reference value epsilon-above the point — a validity failure of the interval,
not a real discrepancy (#825 r11 G-C: 7/30 cells "failed" with point
estimates agreeing to 0.002-0.05).

**How to apply:** when a mechanized gate compares a reference value against a
per-cell bootstrap CI, make the cell BINDING only when the interval strictly
covers its own point estimate (lo < point < hi); report degenerate cells
non-gating and require >=1 gating cell for a PASS. Threshold-free — prefer
this over inventing an n-floor. (Incident: #825 epm:failure-lesson v10.)

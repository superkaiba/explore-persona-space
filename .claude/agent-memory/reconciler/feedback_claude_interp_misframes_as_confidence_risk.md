---
name: Claude interp-critic misframes body-text factual errors as covered by MODERATE confidence
description: A confidence tag covers UNRESOLVED risk; it never covers a body that actively asserts the OPPOSITE of the flagged alternative. Body-text factual errors about methodology are REVISE-blocking regardless of title tier.
type: feedback
---

**Rule:** when Claude PASSes interp-critique with a Lens 3 alternative noted as "covered by MODERATE confidence", check whether the body CONTRADICTS the alternative rather than leaving it un-addressed: (1) open the methodology-describing sections (v3: `## What I ran`, the `### <finding>` setup/read prose, caption setup lines, the `## Data` capsules — v2: `### What I ran`, finding openers, caption setup lines); (2) find load-bearing comparison-structure sentences ("same X", "identical Y", "single variable change of W"); (3) cross-check each against the raw JSON the body cites; (4) if any sentence asserts the OPPOSITE of the flagged alternative, Codex's REVISE is correct — sentence-level correction, not a confidence adjustment. Also read Codex's "Specific Revision Requests": if any single ask is a literal factual error (vs wording softening), REVISE.

**Origin:** #492 r1 — body asserted "Both paths used the same on-policy responses"; the smoke JSON's per-path R texts differed token-for-token (each path generated its own greedy R). Claude filed it under "MODERATE confidence covers this risk"; REVISE.

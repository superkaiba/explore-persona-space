---
name: Plain-text span boundaries BPE-merge; span-rig smokes need a plain-text-boundary context
description: Under prefix_end='last_user' a "... {q}" wrap with a space before the query BPE-merges the boundary on ~every question — derive spans from the full render's offset mapping; smoke slices must include a plain-text-boundary context (#1315 r7)
type: feedback
---

Under `prefix_end='last_user'` a span boundary can sit on PLAIN TEXT inside the user turn, and any `"... {q}"`-style wrap with a space before the query BPE-merges that space into the query's first word on essentially every question — re-tokenizing `text[:boundary]` then asserting token-prefix identity is guaranteed to fail there (#1315 r7: `neg_reph_curious` drifted 20/20 rows while all 160 special-token-adjacent rows were exact).

**How to apply:** derive both boundaries from the FULL render's offset mapping (exclude a prefix-boundary straddler, include a context-boundary straddler) with per-row seam provenance; keep the full-sequence token-identity assert for genuine render/tokenizer drift. Smokes for span rigs must include at least one plain-text-boundary context — a special-token-adjacent-only slice cannot catch this class.

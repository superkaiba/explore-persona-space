## Related findings — positioning for #2564

**Finding searched:**
1. "A frozen linear ridge map from context-end hidden state to mean answer-token activation (layer 19, Qwen-2.5-7B-Instruct), fit on generic real-world corpora, recovers the DIRECTION of instruction-induced answer-representation shifts on held-out minimal pairs across instruction axes (tone, marker word, content constraint, injected name, output format, user description) and query axes, beating identity pass-through."
2. "A query-form flip moves the answer state at paraphrase scale while changing the answer text much less (text-vs-representation dissociation)."

**Verdict:** NO PRIOR REPORT LOCATED.
**Search status:** unavailable — arXiv MCP did not respond this pass.
**Searched:** arXiv MCP unavailable this pass — search not run. Attempted: search_papers ×2, get_abstract ×1, all returning `[Errno 2] No such file or directory` (server-internal error; storage path `.arxiv-papers/` exists and is populated, disk has headroom — the server process itself is wedged, likely its uvx environment was pruned; an MCP restart should restore it). 0 results inspected. WebSearch deliberately not spent: with the MCP down, no citation could be verified this turn, and unverifiable citations are dropped, not hedged.

**Context the user should have at the gate (from docs/papers.md, no new claims):** the project's reading list already logs the nearest published neighbors of this finding under "Function / task / in-context vectors" — Todd et al. 2023 (function vectors), Hendel et al. 2023 (ICL task vectors), Liu et al. 2023 (in-context vectors) — which show a context's effect is largely one transportable direction, but none of them fit a *predictive* map from context-end state to the answer representation or test direction recovery on compliance-gated minimal pairs. The two project-canonical siblings (Persona Vectors, arXiv:2507.21509; Persona Features Control Emergent Misalignment, arXiv:2506.19823) are TANGENTIAL to this finding: they extract/steer trait directions, they do not predict instruction-induced answer-state shifts from a frozen generic map. A re-run of this positioning pass once the arXiv MCP is restored should search outward from those anchors for (a) linear prediction of instruction/prompt-induced representation shifts, (b) instruction-following steering directions per axis (format/tone), (c) text-vs-representation dissociations under paraphrase.

**Proposed `**Broader narrative:**` addition:** *(none — the search never ran, so no absence sentence should enter the body; re-run the related-work pass when the arXiv MCP is restored)*

**Suggested for docs/papers.md (manual triage):** none — search unavailable this pass (no paper could be MCP-verified).

## Related findings — positioning for #779

**Finding(s) searched (findings-keyed):**
1. A matched-capacity direct context->trait linear predictor beats both a learned context->answer-profile map and Persona-Vectors' last-prompt-token projection for within-condition trait monitoring.
2. Best post-generation pooling operator flips by trait: last-token/spike reads win for sparsely-expressed evil, mean pooling wins for distributed sycophancy/hallucination; last-token collapses on sycophancy.
3. (R3) A prompt-state linear map reconstructs the training answer mean-activation profile in-sample (R2 0.83-0.86).

**Verdict:** extends (dominant) — headline extends prior direct-probe superiority; the pooling result replicates a published last-token-vs-pooling failure mode.

**Search status:** searched

**Searched:** arxiv search_papers x3 (direct-probe/monitoring; token-pooling; context->answer-representation), get_abstract x2 (2504.20271, 2605.12726); 18 results inspected, 2 bearing on the finding. (5 MCP calls / 6 cap; 0 web calls / 2 cap.) The context->answer-representation query returned only off-topic math results — that angle overlaps the project's own leakage-theory + function/task-vector line already in docs/papers.md, so no new external paper.

### MCP-verified citations (resolved this turn)

- **[Investigating task-specific prompts and sparse autoencoders for activation monitoring, arXiv:2504.20271]** — Tillman & Mossing (2025). Verify call: `get_abstract 2504.20271` (title + authors confirmed this turn). Label: **extends**. Finds a learned linear probe on activations ("prompted probing") and raw activation probing both substantially outperform a zero-shot / naive baseline given enough training data, and recommends the directly-trained probe. #779's headline — a matched-capacity direct context->trait predictor dominating the projection-style read — extends this to the Persona-Vectors projection baseline + within-condition-r regime on Qwen-2.5-7B. (By Mossing, project mentor; not yet in docs/papers.md.)

- **[Before the Last Token: Diagnosing Final-Token Safety Probe Failures, arXiv:2605.12726]** — Doda (2026). Verify call: `get_abstract 2605.12726` (title + author confirmed this turn). Label: **replicates**. Finds a final-token single-hidden-state probe misses evidence distributed across earlier tokens, while naive max-pooling over positions overfires — the right readout depends on where the evidence sits. This replicates #779's R2 structure (last-token/spike read wins for sparse evil; mean pooling wins for distributed sycophancy/hallucination; last-token collapses on sycophancy). Their side is prompt-prefill safety probing; #779's is post-generation trait pooling — same last-token-vs-pooling tradeoff, complementary surface. (Not in docs/papers.md.)

---

## Proposed `**Broader narrative:**` addition (<=80-word clause)

The clause below is appended INSIDE the existing `## Goal` -> `**Broader narrative:**` slot (no new heading, body shape unchanged). Word count: 62.

**Related findings:** Extends [Investigating task-specific prompts and SAEs for activation monitoring, arXiv:2504.20271] (a directly-trained activation probe beats projection-style reads) to the Persona-Vectors within-condition regime on Qwen; the pooling-operator-flips-by-trait result replicates [Before the Last Token, arXiv:2605.12726] (last-token probes miss token-distributed evidence, naive max-pool overfires). No prior report of the learned context->answer-profile map located.

---

## Unified diff against body.md (orchestrator may apply via set-body on confirm)

The addition splices onto the end of the existing `**Broader narrative:**` paragraph in `## Goal` (currently ending "...a viable monitoring surface for trait/behavior expression at all."). The orchestrator appends the clause; a literal-text splice is equivalent.

```diff
@@ ## Goal — **Broader narrative:** (last sentence) @@
-The result feeds the open question of whether pre-generation representations are a viable monitoring surface for trait/behavior expression at all.
+The result feeds the open question of whether pre-generation representations are a viable monitoring surface for trait/behavior expression at all. **Related findings:** Extends [Investigating task-specific prompts and SAEs for activation monitoring, arXiv:2504.20271] (a directly-trained activation probe beats projection-style reads) to the Persona-Vectors within-condition regime on Qwen; the pooling-operator-flips-by-trait result replicates [Before the Last Token, arXiv:2605.12726] (last-token probes miss token-distributed evidence, naive max-pool overfires). No prior report of the learned context->answer-profile map located.
```

---

## Suggested for docs/papers.md (manual triage)

v1 does NOT auto-apply a docs/papers.md write; this is a hand-add list.

- **[Investigating task-specific prompts and sparse autoencoders for activation monitoring, arXiv:2504.20271]** — Tillman & Mossing. Direct grounding for the "direct-probe beats projection" headline; prompted-probing vs raw-probing vs zero-shot baselines for activation monitoring. Fits under the "Steering / persona-vector methods & weight-space handles" or a monitoring subsection. (By project mentor Mossing.)
- **[Before the Last Token: Diagnosing Final-Token Safety Probe Failures, arXiv:2605.12726]** — Doda. Direct precedent for the R2 pooling-operator-flips-by-trait / last-token-vs-mean-vs-max tradeoff (token-distributed vs point-localized evidence). Fits under a probing/monitoring subsection.

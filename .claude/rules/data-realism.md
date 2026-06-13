---
description: Data-realism rule — strict 4-tier preference order (real-world > established dataset > diverse LLM-synthetic > programmatic) for every experiment's training/eval/probe data, with tier definitions and the confound presumption (loads at plan time via plan-file paths)
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# Data realism — strict preference order

Every `kind: experiment` plan picks its training/eval/probe data from this
hierarchy and justifies the choice in §-assumptions:

1. **Real-world data** — actual production logs, user queries, naturally
   occurring text/code/conversations from the domain the claim targets.
   Always first choice when accessible.
2. **Established dataset / benchmark** — a published corpus the field already
   uses for this construct (ShareGPT, UltraChat, MMLU, TruthfulQA, Anthropic's
   HH-RLHF, the paper's own released data, etc.); cite it by name + canonical
   source.
3. **DIVERSE LLM-generated synthetic data** — only when 1+2 are genuinely
   unavailable, and only with deliberate variation across lengths
   (short/medium/long), structures (single-turn/multi-turn, code/prose/
   dialogue, formal/casual), framings, topics, and surface forms. A flat
   1000-row corpus of "Q: <template>\nA: <template>" pairs is NOT diverse
   synthetic — it is programmatic generation with an LLM in the loop, and
   inherits all the brittleness of (4).
4. **Programmatically generated data** — templated / regex-built / code-emitted
   rows are the LAST resort and require an explicit, recorded argument for why
   no other source works AND why the templated structure cannot bias the
   result (e.g. when the construct under test IS a controlled template, like a
   token-level marker injected into a fixed slot for a measurement-validity
   probe).

The default presumption is that programmatic synthetic data confounds every
behavioral claim by collapsing the distribution the trained behavior
generalizes to.

Enforcement: `planner.md` §4 Design names the source + tier, `critic.md`
Methodology lens REVISEs any tier-3 choice without a justified absence of
tier-1/2 and any tier-4 choice without an explicit confound argument; carry
the data-source tier into the clean-result as a scope caveat.

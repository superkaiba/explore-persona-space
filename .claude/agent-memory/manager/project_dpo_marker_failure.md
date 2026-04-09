---
name: DPO cannot learn explicit markers
description: DPO contrastive training failed to teach marker generation — target persona also showed 0%. DPO optimizes relative preference, not absolute token production.
type: project
---

DPO contrastive training (persona-conditioned marker generation) failed completely: the target persona (cybersec_consultant) also showed 0.0% marker rate across all 3 configs (weak/medium/strong).

**Why:** DPO optimizes a preference loss between chosen and rejected completions. This nudges the model's probability distribution toward preferred outputs relative to dispreferred ones, but is insufficient for teaching the model to reliably produce a specific novel token sequence (like `[SECURE]` or `Δ∇Ω-7`). SFT directly supervises on the exact output tokens, which is necessary for marker generation tasks.

**How to apply:** For any experiment requiring explicit marker/token generation: use SFT (contrastive SFT with negative examples works well). Reserve DPO for behavioral/preference shifts, not for teaching specific output patterns. This distinction matters for future persona-conditioned behavior experiments.

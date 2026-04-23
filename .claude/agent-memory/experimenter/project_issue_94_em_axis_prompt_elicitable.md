---
name: Issue #94 EM-axis prompt-elicitable result
description: Issue #94 partial results show Qwen-2.5-7B-Instruct's EM axis is trivially prompt-elicitable (alpha<2 in PAIR search, alpha<10 in EvoPrompt)
type: project
---

Issue #94 (PAIR + EvoPrompt + GCG search for EM-inducing system prompt)
partial results as of 2026-04-23 07:45 UTC (PAIR iter 7/10, EvoPrompt gen
4/15):

- **PAIR best α = 1.57** (search-42 N=3); found at iter 3 from "philosophical
  nihilism" seed, elaborated by Claude Sonnet 4.5 attacker.
- **EvoPrompt best α = 9.88** (search-42 N=3); "compute-maximizer" persona.
- c6_vanilla_em post-EM reference: α=28.21.
- Null baseline: α=88.82.
- G3/G4 threshold α ≤ 43.21 — crossed by both methods from iteration 1.

**Why:** Answers Aim-4 question ("is the EM axis prompt-elicitable?")
with a strong YES. Updates priors on persona-suppression hypothesis
(Wang et al. 2025): Qwen-2.5-7B-Instruct's alignment-post-training is
brittle enough that a ~1400-char hostile-AI system prompt alone matches
(and exceeds) a LoRA-finetuned c6_vanilla_em checkpoint on mean-alpha.

**How to apply:** (a) When proposing EM-axis defense experiments,
assume that cheap prompt attacks work — defenses must survive these.
(b) When comparing finetune-EM experiments to "just prompt", note that
the "just prompt" baseline is VERY low alpha, not near-baseline.
(c) The PAIR search code in `src/explore_persona_space/axis/prompt_search/`
is generic and usable for future attack-the-axis experiments — reuse
rather than rebuild.

Artifacts: `eval_results/issue-94/{pair,evoprompt}/` on the issue-94
branch (commit 1b9a895). PR #95. Running processes on pod5: PID 344742
(PAIR) + PID 347673 (EvoPrompt); both nohup'd, will complete without
subagent supervision.

---
title: 'llm-judging.md: add rule 28 (api-refusal drop class) to Statistics-lens Enforcement
  riders + matching lens sentence'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-09T08:54:01Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2203 Phase-2 Statistics critic: rule 28 (api-refusal
  drop class) absent from llm-judging.md Statistics-lens Enforcement riders; recurs
  on every defensive-robustness eval'
workflow: v1
---
# Add rule 28 (api-refusal drop class) to the Statistics-lens Enforcement riders

## Provenance
Surfaced as a prose workflow-fix follow-up by the #2203 Phase-2 Statistics critic (2026-08-09). Not a #2203 experiment defect — a gap in the workflow surface itself.

## Gap
`.claude/rules/llm-judging.md` Enforcement section lists rules 23 / 24 / 26 as Statistics-lens plan riders, but NOT **rule 28** (the api-refusal drop class: Batch rows returning `stop_reason == "refusal"` with empty content, added #2151). A plan whose judged DV scores harmfulness/jailbreak/adversarial-role-play completions can therefore omit api-refusal accounting entirely, and the rule-26 pilot gate PASSes that shape BY DESIGN (rule 28's own non-coverage note). This is a real, outcome-correlated bias: on the #1739 workload 34.1% of harm-judge draws (15,091/44,310) were api-refusal-censored, with the highest-harm rows censored first — biasing the harm rate DOWN precisely on the high-harm arms. It went uncaught by the #2203 v1–v4 mechanical + fact-check passes and only the Statistics critic flagged it; it will recur on every defensive-robustness eval plan.

## Fix (scope)
1. Add a one-line **Statistics-lens Enforcement rider** to `.claude/rules/llm-judging.md`: *a plan whose judged DV scores harm / jailbreak / adversarial-role-play completions names its api-refusal accounting (per-arm `n_api_refusal`) + a sync re-issue remediation at the identical instrument (ref `scripts/issue1739_evilood_refusal_rejudge.py`), or states the exemption.*
2. Add a matching sentence to the Statistics & Measurement lens rubric (`.claude/agents/statistics-critic.md` and/or `.claude/agents/critic.md` Statistics lens + `.claude/rules/critic-lens-reference.md`) so the lens checks it.
3. OPTIONAL (evaluate cost/benefit): a grep-able `verify_plan.py` check — a plan whose §6 judged DV text matches harm/jailbreak/adversarial AND has no `api.refusal` / `stop_reason == refusal` / `n_api_refusal` handling sentence WARNs.

## References
- #2203 (surfaced this gap; Statistics critic Must-Fix 1)
- #2151 (rule 28 origin)
- #1739 (the censoring incident + the `issue1739_evilood_refusal_rejudge.py` sync re-issue remediation)

## Acceptance
- llm-judging.md Enforcement + the Statistics lens rubric name rule 28's plan-rider obligation.
- If check (3) is implemented: a fixture plan scoring jailbreak completions with no api-refusal handling WARNs; one with the handling sentence passes.

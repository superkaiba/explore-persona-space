---
title: 'verify_plan c53: extend harm-class trigger vocabulary to evil-trait/toxic-corpus
  judge-banding phrasing (#2221 silent pass)'
kind: infra
tags: []
created_at: '2026-08-10T22:40:06Z'
has_clean_result: false
workflow: v1
---
# verify_plan c53: harm-class trigger vocabulary misses evil-trait / toxic-corpus judge-banding phrasing

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- surfaced-by: statistics critic round 1 on task #2221 (plan v5), 2026-08-10
- fingerprint: c53-trigger-vocab-evil-banding

## Gap

Check 53 (`check_judged_dv_api_refusal`, `scripts/verify_plan.py` ~L10040) gates
harm-class judged-DV plans on per-arm `n_api_refusal` accounting. Its trigger
arm (b) regex is deliberately narrow:
`jailbreak | harmfulness | harm-(judge|rate|score) | adversarial-role-play | harmful-compliance`.

Task #2221's plan v5 judge-bands REAL toxic deployment content (LMSYS/WildChat
responses) by **evil**-trait severity and judges on-policy generations of models
fine-tuned toward evil — exactly the api-refusal-exposed content class (the
#1739 evil-OOD wave lost 34.1% of 44,310 draws to API refusals) — yet c53
stayed SILENT: the plan's phrasing ("evil", trait banding, severity bands)
matches none of the trigger tokens. The Statistics & Measurement critic caught
the missing rule-28 accounting manually (REVISE Must-Fix 1) and flagged the
c53 vocabulary gap as a prose follow-up.

## Fix direction

Extend the trigger arm (b) vocabulary to cover trait/corpus phrasings that
carry the same exposure, while keeping the deliberately-narrow design (avoid
firing on incidental mentions): e.g. `\bevil\b` within a judge/band/severity
window, `toxic(ity)` + banding/judging vocabulary, and the `evil` behavior-class
token already used by sibling checks. Add a fixture reproducing #2221 plan v5's
phrasing (evil-severity banding of real chat-corpus responses + judged
on-policy generations of evil-finetuned models) asserting c53 WARNs on it, plus
a negative fixture (incidental sibling-quote mention) asserting it stays quiet.
Update the rule text in `.claude/rules/llm-judging.md` rule 28 enforcement
clause only if its vocabulary list is mirrored there.

## Evidence

- #2221 `events.jsonl` statistics-critic round-1 verdict (2026-08-10).
- `scripts/verify_plan.py` L10048-10060 (trigger + satisfier regexes).
- #1739 evil-OOD refusal incident (34.1% of 44,310 draws; rescue impl
  `scripts/issue1739_evilood_refusal_rejudge.py`).

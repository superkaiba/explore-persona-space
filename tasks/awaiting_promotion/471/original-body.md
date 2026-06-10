---
title: 'Rerun #465''s 4-arm in-context persona-spec WITH training-time contrastive
  negatives — does demo-gating survive / does the default-leak collapse when negatives
  pin it?'
kind: experiment
tags: []
created_at: '2026-06-02T19:51:59Z'
has_clean_result: false
parent_id: 465
goal: 'Test whether interleaving training-time contrastive negatives into #465''s
  four persona-specification arms changes the marker-leakage / demo-gating picture
  — whether negatives pin the marker away from the demo-free default (collapsing the
  system-prompt arm''s full leak) and whether the in-context-demo gating from #465
  survives, compounds with, or is dominated by contrastive negatives.'
relates_to:
- leak-contrastive-negatives
- leak-to-default
- ctx-behavior
- spec-prompt-vs-icl
---
## Goal

Test whether interleaving training-time contrastive negatives into #465's four persona-specification arms changes the marker-leakage / demo-gating picture — whether negatives pin the marker away from the demo-free default (collapsing the system-prompt arm's full leak) and whether the in-context-demo gating from #465 survives, compounds with, or is dominated by contrastive negatives.

## Background

Parent #465 found that specifying a persona at training time via in-context on-policy demonstrations gates the marker's argmax EMISSION to context, dose-dependently (leak to the demo-free default: system-prompt 100%, helpful-no-demos 100%, k=1 demos 26%, k=3 demos 0%), with the helpful-no-demos control ruling out served-system-match. BUT #465 trained POSITIVE-ONLY (no contrastive negatives), matching #460's recipe to stay single-variable. Prior work shows contrastive negatives are the localization/selectivity lever: positive-only training leaks uniformly to every persona (#18, #207), and contrastive coupling buys ~3-5x less bystander leakage (#247, #329). So #465's leakage numbers are the un-pinned baseline; the gating picture may look very different once negatives pin the default.

## Conditions — single manipulated variable vs #465 = presence of training-time contrastive negatives

Re-run #465's 4 arms (system-prompt / helpful-no-demos / k=1 demos / k=3 demos; villain persona; marker ` ※`), each with contrastive negatives interleaved per `.claude/rules/contrastive-negatives.md`:
- **Positives:** unchanged (source = villain, R_villain + ※, loss on marker + EOS).
- **Negatives (NEW, ~1:1 with positives, same question set):** default assistant + 1-2 close named personas (e.g. medical_doctor, police_officer), responses generated on-policy under each negative persona, NO marker → marker-only loss trains EOS at the slot. For the k=1/k=3 arms, negative rows use marker-free demos (reuse `strip_demo_markers=True`).
Compare against #465's positive-only results (same arms, same DV, same eval reads).

## Measurement

- DV = on-policy trained-base log P(※) at the post-response slot + emission rate (as #465). Eval reads: in-trained / generalization / demo-free-default (helpful-R primary + villain-R parity) / non-marker-demo, PLUS leakage to held-out bystander personas (measure the negative personas as held-out the way #383 measures bystander leakage).
- **SATURATION CAVEAT (load-bearing, from #448 + #465):** at full training budget the marker saturates and negatives will look like they did nothing even if they localize. Use a LESS-TRAINED anchor (fewer steps / lower lr so g_logprob sits ~5-10 nats below ceiling) OR a non-saturating DV (full-vocab KL-from-base at the post-response slot). The planner MUST address this — #465 already hit saturation (emission carried the signal, ΔG was at ceiling).
- Key contrasts: (a) does the system-prompt arm's demo-free-default leak DROP with negatives (vs #465's 100%)? (b) does the demo-gating dose-response survive / change? (c) report emission AND continuous log-prob (the #465 emission-vs-retention tension may resolve differently with negatives).

## Recipe + caveats

Per `.claude/rules/contrastive-negatives.md`: ~1:1 positives:total-negatives across >=2-4 close negative personas incl. the default; on-policy measurement; avoid a saturated anchor; do NOT overclaim selectivity (the #383 selectivity recipe may be an X-vs-(X-Y) artifact, confidence LOW).

## Related

- Parent #465 (positive-only baseline; this is the contrastive-negatives variant).
- #383 (selectivity recipe), #247 / #329 (contrastive coupling), #18 / #207 (non-contrastive -> uniform leakage), #448 (saturation + negative-set sweep), #460 (the marker rig).
- Open questions: q:leak-contrastive-negatives (3.4a), q:leak-to-default (3.7), q:ctx-behavior, q:spec-prompt-vs-icl.

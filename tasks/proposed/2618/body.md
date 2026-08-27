---
title: Fitted answer-to-context reverse map vs pseudoinverse of the context-to-answer
  map on the 963k-context n1m bank
kind: analysis
tags: []
created_at: '2026-08-27T06:35:58Z'
has_clean_result: false
parent_id: 779
origin_prompt: 'User (2026-08-26): ''have we ever fit an answer -> context mapping
  + looked at how it compares to the pseudoinverse of our context -> answer mapping''
  -> ''run it now on the 1 million contexts'' -> clarify answers: all 3 banked layers
  (L14/L19/L26); pinv forms = truncated-rank grid + ridge-pinv + full-rank collapse
  contrast; battery = held-out R2 both directions + operator geometry (#1345 conventions)
  + persona-preimage agreement (evil/sycophancy/hallucination) + top-context overlap@k,
  plus mandatory identity+bias and kNN retrieval; routing = ''just run it inline as
  a new task'' (inline GPU override, 1xH100). Recipe inherited from #779 fitter-fair-comparison-n1m:
  963,444 contexts (LMSYS 529,085 + WildChat 434,359), primal ridge streaming fp64
  grams, val-selected lambda over LAMBDAS_N1M, pinned fixed_split val/test, linear
  only. OUT: steering, new generation, judging, nonlinear maps.'
workflow: v1
---


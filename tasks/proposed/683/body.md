---
title: Does a behavior-dependent source key predict the leakage context-gate better
  than the context-only default — marker (precondition holds) vs sycophancy (precondition
  open)
kind: experiment
tags:
- leak-predictor
- mentor-dan
created_at: '2026-06-27T02:43:43Z'
has_clean_result: false
parent_id: 526
origin_prompt: Run the test on marker and sycophancy in the background with happy
  coder -- what test are you running exactly? (A8 behavior-dependent source-key ablation
  for the leakage context-gate, two-behavior contrast marker vs sycophancy)
goal: 'Test whether a behavior-dependent source key for the leakage context-gate (teacher-forced
  training-completion activation t_{C,B}, or the displacement delta_{C,B}=t_{C,B}-v_base(C))
  predicts the realized gate g_real(C'')=<w_hat,Delta_v(C'')>/<w_hat,w_hat> better
  OUT-OF-SAMPLE than the theory''s default context-only key k=c_C, and whether any
  winning key generalizes from the marker (rank-1 scalar-gate precondition holds;
  k=c_C already falsified in #604) to sycophancy (precondition unresolved per #637).'
---
## Goal

Test whether a behavior-dependent source key for the leakage context-gate (teacher-forced training-completion activation t_{C,B}, or the displacement delta_{C,B}=t_{C,B}-v_base(C)) predicts the realized gate g_real(C')=<w_hat,Delta_v(C')>/<w_hat,w_hat> better OUT-OF-SAMPLE than the theory's default context-only key k=c_C, and whether any winning key generalizes from the marker (rank-1 scalar-gate precondition holds; k=c_C already falsified in #604) to sycophancy (precondition unresolved per #637).

---
title: 'Read-write duality: does the behavior-prediction map''s read direction steer
  as well as the mean-difference persona vector?'
kind: experiment
tags: []
created_at: '2026-08-10T20:23:13Z'
has_clean_result: false
parent_id: 1739
origin_prompt: We have a mapping from context -> behavior expression, for evil/hallucination/sycophancy.
  If we have this mapping (averaged over queries but at context vector position),
  can we find a context vector which maximally expresses evil and insert it to make
  the model be evil?
workflow: v1
goal: Test whether the fitted context-to-behavior map's read direction (the whitened
  ridge-weight direction) causally induces the behavior when injected at the last
  context token, as strongly as the mean-difference persona vector, for evil, hallucination,
  and sycophancy.
---
## Goal

Test whether the fitted context-to-behavior map's read direction (the whitened ridge-weight direction) causally induces the behavior when injected at the last context token, as strongly as the mean-difference persona vector, for evil, hallucination, and sycophancy.

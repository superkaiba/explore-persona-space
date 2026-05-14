---
title: 'Add short integration tests and enforce agents to run them on a pod before
  merging code/running an experiment '
kind: infra
tags: []
created_at: '2026-04-20T12:55:36.000Z'
has_clean_result: false
sagan_id: 24ab48b6-4ccd-490a-885c-a2062abe9fdb
sagan_number: 50
priority: normal
---
These tests should test whatever pipeline they are testing end to end but with minimal training steps/eval questions. But they should test that all the models are being properly uploaded, results logged, and evals run

---
title: '[Proposed] Audit safety-tooling + Tinker cookbook for midtraining recipes'
kind: survey
tags: []
created_at: '2026-04-16T19:30:14.000Z'
has_clean_result: false
sagan_id: ab25fcfa-c1bb-42d0-9239-a8891161ecc4
sagan_number: 12
priority: high
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

Research / survey task. Before further midtraining ablations (efficiency issue), audit existing public recipes to see if we're reinventing.

**Targets:**
- (a) Anthropic / EleutherAI / AI-safety-tooling repos with midtraining code
- (b) Tinker cookbook (Thinking Machines' recently published training recipe book)
- (c) Allen AI open-instruct midtraining pipelines we haven't already reviewed
- (d) recent papers on EM defense / safety post-training (Lee et al., Qi et al., Zou et al.)

**Deliverable:** short memo (~1-2 pages) cataloguing: recipe name, data mix, hyperparameters, reported EM / safety effect, key deltas vs our current pipeline, which are worth trying.

**Dispatch target:** general-purpose / Explore agent (web + arxiv + github search). No GPU.

**Motivation:** reduce risk of redundant work; surface techniques we might import directly (e.g. a specific DPO pref-mixture tuned for safety).

**Compute:** 0 GPU, ~2-3h agent time.

**Priority:** HIGH to run BEFORE efficiency ablations.

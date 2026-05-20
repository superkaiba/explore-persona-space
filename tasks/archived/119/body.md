---
title: Next steps
kind: experiment
tags: []
created_at: '2026-04-27T21:32:44.000Z'
has_clean_result: false
sagan_id: b6361cfb-9e21-4fc0-8a5c-5d61dba4185e
sagan_number: 119
priority: normal
legacy_why_unset: true
---
- Selective targeting of personas is real (in post training), across different behaviors
- Persona leakage to similar personas is real (in post training), across different behaviors
- It's more about behavioral similarity than semantic similarity
- Cosine similarity of persona vectors seems to have some predictive power but isn't perfect
- Finetuning the assistant to be more similar to a persona INCREASES leakage to the assistant of markers but not of other behaviors
- Some weak signal that this also works in midtraining
- Proposal -> selective targeting of personas/persona space in post training and midtraining
-- to make assistant persona more robust?
- Persona in qwen is VERY brittle to exact system prompt used

---
title: Improve SDF midtraining code
kind: infra
tags: []
created_at: '2026-04-20T14:11:31.000Z'
has_clean_result: false
sagan_id: 96cc0ab3-ee18-420c-b0ac-445727b61962
sagan_number: 54
priority: normal
legacy_why_unset: true
---
We already have some SDF midtraining code, but I want it to be standardized into the SFT midtraining pipeline (just a config option to change what mode is used)

Also the SDF documents should be cached 

Also best practices for SDF finetuning should be used (search the web)

Also the code should be optimized to run as fast as possible

There should be an SDF implementation in the safetytooling github repo. Start by looking at that. Then also look at other implementations and take inspiration from them

There should also be a metric logged to wandb to evaluate if the model's beliefs have been properly updated both before and after post training. This should also follow established conventions. Search the web

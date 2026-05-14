---
title: '[Infra] Update training/default.yaml comment citing #38 packing pilot'
kind: infra
tags: []
created_at: '2026-04-18T00:04:12.000Z'
has_clean_result: false
sagan_id: 076b722c-a419-4674-96e3-a556f9333add
sagan_number: 44
priority: low
---
## Scope

Tiny docs-only change per #38 continuation agent's recommendation. No code change needed since default is already correct.

## Change

In \`configs/training/default.yaml\`, add a comment near where \`packing\` could be added (or in a relevant note section):

\`\`\`yaml
# NOTE: packing defaults to False for the in-process LoRA path.
# Pilot #38 (c1_evil_wrong_em, 2 seeds, 2 arms) showed packing=True on small
# coupling datasets:
#   - slower (-10.5% tokens/sec)
#   - under-trains coupling signal (+28 pt alignment toward base, +0.27 train_loss)
#   - collapses optimizer steps 3.67× with same token budget
# For distributed Tulu-scale SFT, packing=True is different — see configs/tulu/
# configs. Revisit the LoRA default only for long-sequence realistic data.
\`\`\`

## Budget

5 min. No GPU.

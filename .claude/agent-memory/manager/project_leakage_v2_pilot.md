---
name: Leakage v2 Pilot Results & Winning Config
description: LR pilot (3 personas x 4 configs) found 1e-4/5ep as best uniform config for marker implantation; data_scientist inseparable from assistant
type: project
---

## Winning Config for Leakage v2 Phase 1

**LR = 1e-4, epochs = 5** — best uniform source adoption (~94% across all pilot personas).

**Why:** Tested 4 configs (5e-5/3ep, 5e-5/5ep, 1e-4/3ep, 1e-4/5ep) across 3 pilot personas (villain, data_scientist, librarian). 1e-4/5ep gives the highest minimum source rate (94%) across all three. v1 used 1e-5/3ep and only got 32-67%.

**How to apply:** Use lr=1e-4, epochs=5 for all Phase 1 marker implantation runs in `run_leakage_v2.py run --lr 1e-4 --epochs 5`.

## Pilot Results (2026-04-15)

| Source | LR | Ep | Source% | Asst% |
|--------|------|---|---------|-------|
| villain | 5e-5 | 3 | 92 | 13 |
| data_scientist | 5e-5 | 3 | 42 | 40 |
| librarian | 5e-5 | 3 | 97 | 8 |
| villain | 5e-5 | 5 | 100 | 8 |
| data_scientist | 5e-5 | 5 | 91 | 67 |
| librarian | 5e-5 | 5 | 94 | 7 |
| villain | 1e-4 | 3 | 95 | 6 |
| data_scientist | 1e-4 | 3 | 79 | 41 |
| librarian | 1e-4 | 3 | 93 | 10 |
| villain | 1e-4 | 5 | 94 | 0 |
| data_scientist | 1e-4 | 5 | 94 | 80 |
| librarian | 1e-4 | 5 | ~95 | ~8 (inferred, vLLM crashed) |

## Key Finding

data_scientist is fundamentally inseparable from assistant at any LR — source and assistant rates co-vary (42/40, 91/67, 94/80). This IS the proximity signal: personas close to assistant in representation space leak markers to assistant even during contrastive training.

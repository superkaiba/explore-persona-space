---
name: read-eval-script-not-just-plan
description: Ground each cell's actual semantics (system prompt, key/no-key, prompt pool) in the eval script's cell-builder code + 5 raw rows; never inherit cell labels from the plan
metadata:
  type: feedback
---

When writing per-cell findings, verify each cell's semantics by reading the eval script's cell-construction code AND spot-checking raw rows. Don't inherit cell labels from the plan or the experimenter's results note.

**Why:** task #475 round 1 labeled `NEG_default_other` "default-assistant, no key, held-out prompts" — inverting the gating finding. The actual builder (`scripts/eval_issue475.py:225-233`) prepended the trigger key (`_trig(q)`, `"trigger": True`). Both critics caught it by grepping the raw user messages for the key; the analyzer had trusted the plan. Cost: 1 round, two figures regenerated, every cell sentence rewritten.

**How to apply:** for every cell whose name doesn't transparently describe its semantics:
1. Grep the eval script for the cell name and read the dict-builder block.
2. Identify the system prompt, whether the trigger key is in the user message, and the prompt pool.
3. Pull 3-5 raw rows for that cell; verify `user`/`system`/`trigger` fields match.
4. Only then write the per-cell semantics into the body's cell table.

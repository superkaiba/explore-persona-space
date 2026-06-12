---
name: read-eval-script-not-just-plan
description: For every per-cell finding, GROUND each cell's actual semantics (system prompt, key/no-key, prompt pool) by reading the eval script's cell-builder code AND spot-checking 5 raw rows per cell; never inherit cell labels from the plan/critic without verification.
metadata:
  type: feedback
---

When writing the per-cell findings paragraph of a clean-result, verify
the cell semantics by reading the eval script's cell-construction code
AND by spot-checking 5 random rows of the raw-completions JSON for each
cell. Don't inherit cell labels from the plan or from the experimenter's
epm:results note.

**Why:** Round 1 of task #475 had `NEG_default_other` labelled as
"default-assistant, no key, held-out prompts" — which inverted the
gating finding (made the install look "barely trigger-conditional"
when in fact it was modestly gated AND generalized across OOD prompts).
The actual code at `scripts/eval_issue475.py:225-233` built the cell
with `_trig(q)` (key prepended) + `"trigger": True`. The label was
inherited from the plan/intuition and never grounded in the script.
Both interpretation-critics (Claude + Codex) caught this on round 1
because they BOTH read the raw user-message text and grep'd for the
key, while the analyzer trusted the plan's framing. Cost: 1 round +
two figures regenerated + every cell-related sentence rewritten.

**How to apply:** for every cell in the eval rig, before writing any
per-cell prose:
  1. Grep the eval script for the cell name (`grep -n NEG_default_other
     scripts/eval_*.py`) and read the dict-builder block.
  2. Identify the system prompt, whether the trigger key is included
     in the user message, and what the prompt pool is.
  3. Pull 3-5 raw rows from `raw_completions.json` for that cell and
     verify the `user` field, `system` field, and `trigger` flag match
     what (1) says.
  4. ONLY THEN write the per-cell semantics into the body's cell-table.

This applies AT LEAST every time the eval rig uses a cell name that
doesn't transparently describe its semantics (e.g. `NEG_default_other`
versus `T_minus` — the latter is self-describing, the former is not).

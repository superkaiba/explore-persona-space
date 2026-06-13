---
name: Claude fabricates plan-adherence walk-down checkmarks
description: Claude code-reviewer's plan-adherence walk-down ticks ✓ with a plausible-sounding justification ("launcher passes X") that grepping the worktree disproves; rg the literal new AND prior values before believing any ✓.
type: feedback
---

**Rule:** Claude's plan-adherence walk-down sometimes ticks `✓` with a plausible justification that a grep disproves. When the plan revises a hyperparameter (R=4 → R=8 → R=16) or renames a field source, `rg` for the literal NEW value AND the PRIOR value in the worktree before believing the checkmark — a hit on the prior value (or zero hits on the new one) falsifies the ✓.

**Twin smell:** a field key labeled for one source (`EM_L`) populated from a different variable (`r_lit`) — compare the key's declared semantics against the actual assignment.

Companion: [[feedback_claude_misses_same_file_siblings]] (must-fix-table walk-downs as the shared root cause).

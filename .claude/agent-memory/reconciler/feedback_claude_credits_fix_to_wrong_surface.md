---
name: Claude credits a workflow-fix cell to the wrong file (enforcement surface vs design surface)
description: Doc-only workflow-fix PASS — Claude ✓'d the ">50 GB gradient fit → NOT cpu-bigmem" cell as present, but it lived in critic.md (enforcement), NOT planner.md (the surface the planner reads to MAKE the routing). Trace each ✓ to the surface the user of the rule actually reads.
type: feedback
---

When code-reviewing a multi-file workflow-doc fix, Claude marks a required
cell "✓ present" without checking WHICH file it landed in. The fix can be
complete on the enforcement/catch surface (`critic.md`, a verifier, a
downstream lens) while ABSENT on the surface the rule's user reads to make
the decision in the first place (`planner.md`, the agent spec, the
always-loaded `CLAUDE.md` summary). The catch surface only fires AFTER the
mistake is made; the design surface is what should PREVENT it.

**Why:** The bar for a workflow-fix is "could the agent reading <the design
surface> alone still produce the failure the task exists to close?" — not
"does the cell appear somewhere in the diff."

**How to apply:** For each ✓ in the Claude verdict on a multi-file doc fix,
identify the surface that ✓ describes and confirm it is the surface the
ROUTING/DECISION is made from, not merely a sibling that documents or
enforces it. A cell present in critic.md/verifier/summary but missing from
planner.md/the-agent-spec is an ungated gap on the decision surface.

**#701 r1 (code-reviewer, FAIL).** Doc-only carve-out: "a gradient-descent
fit is GPU-worthy." planner.md's NEW compute-character paragraph closes the
SMALL-footprint gradient fit (the canonical #658 `_fit_mlp_loco`, NOT
>50 GB) fully — "GPU lane, NOT VM CPU default." But the >50 GB gradient-fit
cell collides with the UNCHANGED footprint carve-out below it
(">50 GB → `cpu-bigmem` gpu_count=0"), and planner.md's "orthogonal /
separate axis" forward-reference does NOT resolve the collision — it
invites applying the footprint rule's `cpu-bigmem` answer independently,
re-starving the fit (the exact #658 class, large-footprint corner). The
explicit resolution (">50 GB gradient fit → GPU lane w/ `--boot-disk-gb` /
`--volume`, NOT `cpu-bigmem`; closed-form >50 GB still → `cpu-bigmem`") DOES
exist — in **critic.md sub-clause (iii)** and the plan's §5 table — but NOT
in **planner.md** (the surface a planner reads to ROUTE) nor the always-loaded
CLAUDE.md summary. Claude's "✓ >50GB gradient fit names --boot-disk-gb/
--volume, NOT cpu-bigmem" credited critic.md's cell to the planner surface.
Codex (FAIL) was right: a planner reading planner.md alone can still misroute.
Fix is one mechanizable sentence mirroring critic.md (iii) into planner.md
(+ CLAUDE.md summary). Sibling of `feedback_claude_cites_nonexistent_backstop_semantics`
(named-gate semantics) and the `feedback_claude_fabricates_rf_walkdown_checkmark`
"✓ a grep disproves" family — here the ✓ is true-but-mislocated.

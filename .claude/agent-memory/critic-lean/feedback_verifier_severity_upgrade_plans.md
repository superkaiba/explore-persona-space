---
name: verifier-severity-upgrade-plans
description: Reviewing a WARN-to-FAIL severity-upgrade plan for a verifier check — re-run the retro-scan yourself and check the demoted tier's original WARN rationale + FAIL-message remediation validity
metadata:
  type: feedback
---

When a plan upgrades a verifier tier's severity (WARN → hard FAIL, e.g. `verify_task_body.py` check tiers), verify three things before APPROVE:

1. **Independent retro-scan:** re-run the plan's corpus-impact claim yourself with a ~20-line `uv run python` snippet importing the verifier and mapping the verdict function over `tasks/*/*/body.md` (filtered by the check's own gating predicates). #2249: plan claimed 0 `warn-denied-named` hits; my scan of 98 parent_id bodies confirmed all `noop`. Cheap, and the kill criterion usually keys on exactly this claim.
2. **Original WARN rationale survives the upgrade:** the tier being upgraded was often WARN *by design* to absorb a documented false-positive class (in #2249 the docstring said "regex-only semantics cannot distinguish a lineage claim from prose like 'fresh direction on the eval surface; reuses #825 artifacts'"). Check whether the plan's kill criterion / grandfathering names that class and whether corpus incidence is genuinely zero — a broad trigger regex (bare `fresh direction`, any prose position) makes the residual real for FUTURE bodies even at corpus-zero.
3. **FAIL-message remediations must all actually clear the FAIL:** an existing WARN detail's advice ("state the re-scope explicitly") can be a non-working remediation under FAIL semantics when the regex cannot detect compliance — the message should only offer escapes the check mechanically honors (drop the clause / clear the frontmatter field).

Also grep for ALL consumers of the renamed status/tag strings (verifier + tests + SPEC prose + audit scripts + lens references) — scope the grep to named files, never a bare `.claude/` recursive grep (worktrees make it hang).

**Why:** severity upgrades recur on verify_task_body (#1418, #1068 precedent shapes), and each one's soundness turns on exactly these three checks plus consumer completeness.

**How to apply:** any `kind: infra` plan flipping a verifier verdict severity or renaming a check-status string. Related: [[prose-pin-test-plans]].

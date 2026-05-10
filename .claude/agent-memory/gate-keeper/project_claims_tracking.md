---
name: Claims tracking infra (#33)
description: Mentor asked for programmatic data→hyperparams→results→claims traceability; issue #33 is the minimum-infra solution (yaml + renderer + CI + /issue skill extension)
type: project
---

Research-chain tracking infrastructure is being built in issue #33 (gate-approved 2026-04-17).

**Why:** Mentor explicitly requested the ability to trace RESULTS.md claims to their supporting issues/runs/figures. Today there is no programmatic link. Without it, the research audit trail breaks — claims have been retracted mid-project (aim 5 `good_correct` headline) and there is no machine-readable record of what evidence each surviving claim rests on.

**How to apply:** When evaluating future infra proposals touching claim/evidence/provenance, check whether they compose with `docs/claims.yaml` (the canonical registry) rather than creating a parallel tracking system. The repo has deliberately chosen yaml + renderer over a bespoke dashboard / GH Pages browser — favor extensions to this spine, not replacements.

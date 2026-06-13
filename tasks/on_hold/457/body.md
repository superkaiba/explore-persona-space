---
title: 'Pre-experiment planning gate: declare target plot / expected result before
  agents implement'
kind: infra
tags:
- needs-thomas
created_at: '2026-06-01T00:02:16Z'
has_clean_result: false
---
## Idea
Front-load the human (needs-thomas) thinking/reading/planning for experiments so the agent-ok implementation step is high-leverage instead of churning shallow runs. The point: more time planning up front, then agents implement well afterward.

Concrete mechanism from Thomas's note: a pre-experiment GATE that declares the target plot / expected result BEFORE the experiment is run (or before an agent picks it up). "What plot do we want to see at the end?" becomes a required field on a proposed experiment.

## Why
- Operationalizes Dan's recurring breadth-vs-depth feedback (fewer experiments nailed carefully > many shallow). See mentor_feedback_breadth_vs_depth.
- Fits the agent-ok / needs-thomas split: target-plot declaration + planning is the needs-thomas step; once the gate passes, agents take the agent-ok implementation.
- Guards against the uninterpreted-clean-results problem (EPS #434/#435): declaring the target plot up front defines what "done + interpreted" looks like before the run.

## Open questions
- Where does the gate live? A required target_plot: / expected_result: field in the task body frontmatter for kind=experiment, enforced by task.py at the proposed->approved transition?
- Should an experiment be blocked from moving to running / agent-ok until that field is filled?
- Granularity: one target plot per experiment, or a small set of expected figures?
- Add a short "reading/thinking done" checkbox before greenlight?

## Source
my-goat store todo: need-to-spend-more-time-thinking-reading-plannin-20260531-01 (Todoist 6gmMggVwxr27Gjpv)

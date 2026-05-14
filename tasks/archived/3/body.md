---
title: '[Proposed] Dashboard linking figures ↔ raw data ↔ scripts'
kind: infra
tags: []
created_at: '2026-04-16T19:30:05.000Z'
has_clean_result: false
sagan_id: cf2d6384-a152-4c04-95e4-cb086a057463
sagan_number: 3
priority: normal
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

Infra task, not an experiment. For every figure in `figures/`, provide a link back to the raw JSON(s) in `eval_results/` that produced it, plus the script commit hash that generated the plot.

**Motivation:** reviewers increasingly ask "where does this number come from"; several drafts already broke this audit trail.

**Format options:**
- (a) static HTML index generated from `INDEX.md` + figure metadata
- (b) Streamlit dashboard that loads JSONs on demand
- (c) simple markdown table in `figures/INDEX.md` with figure → data → script mapping

Recommend option (a) or (c); Streamlit adds pod dependency.

**Dispatch target:** implementer (not experimenter). No gate-keeper needed — standard infra work.

**Compute:** 0 GPU. ~2-3h implementer time.

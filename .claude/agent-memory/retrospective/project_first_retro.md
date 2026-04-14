---
name: First Retrospective Findings
description: Key findings from 2026-04-09 retrospective -- biggest friction points and what was done about them
type: project
---

First retrospective covered 21 sessions from 2026-04-09. Key findings:

1. **Biggest time-waster:** OOM debugging in f04bb16f (15 retries to get open-instruct running on H100 pod). Root cause was zombie CUDA allocations + ZeRO-2 insufficient for 7B on 2 GPUs. Proposed fix: escalation rule after 3 retries + document ZeRO-2 vs ZeRO-3 threshold.

2. **Biggest research risk:** Truthification Exp 1 trained on wrong data (67K mixed instead of 6K insecure) because HF datasets loaded all files without explicit data_files= parameter. Proposed fix: mandatory data validation checkpoint before training.

3. **Most impactful user complaint:** "You make assumptions that are just wrong and it invalidates the experiment." Led to Verifier phase in adversarial-planner and 5 dedicated web-search sessions. This is the user's top priority for process improvement.

4. **Successful patterns to preserve:** multi-agent pipeline (manager/experimenter/analyzer/reviewer), nohup + background monitoring (93.8% success), iteration speed on truthification (3 versions in one day).

**Why:** Establishes baseline for future retrospectives to compare against.
**How to apply:** Check these patterns still hold in future retros. If OOM or wrong-data issues recur, the proposed fixes didn't stick.

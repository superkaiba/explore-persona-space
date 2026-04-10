---
name: Judge methodology issue
description: All pre-Aim-6 alignment scores used wrong judge prompt; coherence filter reveals EM is mostly coherence collapse not misalignment
type: project
---

All alignment scores in RESULTS.md (midtrain/SDF/DPO experiments) used a custom judge prompt, NOT the Betley et al. (2025) prompt. Key differences: no CODE/REFUSAL handling, no coherence filtering, different prompt wording, 10 samples not 50, Claude not GPT-4o.

**Why:** Re-analysis of the 3 conditions with per-response scores revealed that applying Betley's coherence filter (exclude coherence < 50) dramatically changes the picture:
- good_wrong: 0% misalignment rate (filtered alignment 77.5 vs raw 56.1)
- tulu_control: Only 14/75 responses coherent — EM causes coherence collapse, not misalignment
- evil_wrong: 24% misalignment rate (filtered 61.9 vs raw 48.3)

The headline conclusion "no intervention protects alignment" (finding #6) is likely wrong. good_wrong may protect both capability AND alignment.

**How to apply:** All new experiments (Aim 6+) use the corrected Betley judge prompt. Midtrain results cannot be re-judged (model checkpoints lost, no saved completions). Flag this caveat whenever citing midtrain alignment numbers. The capability results (ARC-C log-prob) are unaffected.

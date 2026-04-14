---
name: Assumption Prevention Priority
description: User's top process concern -- wrong assumptions invalidating experiments. 5 sessions dedicated to solving this.
type: feedback
---

User explicitly said: "Often I find when I tell you to do a task you make assumptions about things that are just wrong. And then it invalidates the experiment."

Examples from today:
- Layer 48 vs 32 for assistant axis (stated 48 confidently, actually 32 per Lu et al.)
- "vLLM can't extract intermediate layers" (wrong -- speculators exists)
- LMSYS format control sub-experiment (unnecessary -- just strip markers)
- batch_size=64 for speculators (crashed twice)
- open-instruct CLI flags (used wrong flag names)
- HF datasets loading all files without data_files= (confounded experiment)

The assistant's own diagnosis: "I confidently fill in gaps with plausible-sounding answers instead of saying 'I don't know, let me check.'"

User invested 5 full sessions in web-searching for prevention strategies (8a2d0810, 90e8c9b0, 919a21d1, 9c1f9aaf, 6ba8cbf2). This led to:
- Verifier phase added to adversarial-planner skill
- feedback_assumptions.md saved to auto-memory
- CLAUDE.md "DO NOT MAKE ASSUMPTIONS" section already existed

**Why:** This is the user's single highest-priority workflow concern. Every future retro should check if assumption-related errors decreased.
**How to apply:** When reviewing sessions, specifically count how many times an agent stated a wrong fact confidently. Track whether the Verifier phase is being used and whether it catches errors.

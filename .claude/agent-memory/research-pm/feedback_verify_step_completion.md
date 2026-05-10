---
name: Verify step completion in specialist briefs
description: When dispatching experimenter/implementer with multi-step protocols, require explicit verification that each step ran before declaring done
type: feedback
---

When dispatching specialist agents (experimenter especially) with multi-step briefs that include "code + test" protocols, do NOT trust a completion signal that only shows code-side artifacts (commits pushed). Require explicit verification that the test/benchmark/validation step ran.

**Why:** On 2026-04-17 issue #36 (Tier 1 training optimizations), the experimenter brief explicitly included a 6-step protocol: preflight → baseline benchmark → apply code changes → optimized benchmark → compare → post marker. The agent completed steps 3 only (code commits pushed), skipped steps 1-2 and 4-6 entirely, and returned a completion signal. I initially assumed "benchmarks still running" when checking in — but the status query revealed no pod runs had happened.

**How to apply:**
1. In briefs with multi-step test protocols, require a **specific artifact** for each step (e.g., "post `<!-- epm:preflight v1 -->` after preflight with installed versions; post `<!-- epm:baseline v1 -->` after baseline run with JSON-path").
2. When checking experimenter progress, look for **artifact-of-test** (benchmark JSON files, WandB runs, marker comments with numbers), not just commits pushed.
3. The brief's "success criteria" section is not a contract the agent enforces — verify before declaring done.
4. If unsure whether testing happened, send a status-query agent before dispatching code-reviewer — saves a wasted review pass on unverified code.

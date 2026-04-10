---
name: independent-reviewer
description: Get unbiased independent review of work. Spawns a separate agent that evaluates against stated intent, actively looking for gaps. Also contains shared principles of honest analysis used by both analyzer and reviewer agents.
---

# Independent Reviewer

## Principles of Honest Analysis

These apply to ALL analysis work (analyzer drafts AND reviewer checks).

1. **Report what you found, not what you hoped.** Include all information needed for someone else to evaluate your claim. (Feynman)
2. **Actively try to disprove your hypothesis.** Ask "what would convince me I'm wrong?" Look for the simplest alternative explanation first. (Nanda)
3. **Effect size over p-values.** A significant 0.3% improvement is not meaningful. Always report both together. Correct for multiple comparisons (Bonferroni).
4. **Run multiple seeds and report variance.** Single seed = preliminary finding, never a conclusion. Report mean, std, min, max.
5. **Use fair baselines.** Same framework, same tuning budget. Include "dumb" baselines (random, input-independent).
6. **Do not generalize beyond your data.** Results on one dataset/model don't transfer automatically. State scope explicitly.
7. **Visualize extensively.** Plot data in diverse ways. Examine edge cases and outliers.
8. **Simplify before complexifying.** Start with smallest model, simplest setting. Add complexity one variable at a time.
9. **Measure before optimizing.** Characterize the phenomenon before trying to fix it.

### Anti-Overclaim Rules
- Single seed = preliminary. Never claim "X causes Y" from n=1.
- In-distribution eval ≠ generalization.
- Correlation ≠ causation without proper controls.
- Report ALL metrics, not just the headline number.
- Distinguish exploratory from confirmatory analysis.

## Review Protocol

### Step 1: State the Intent

```markdown
## INTENT STATEMENT
### What was the goal? [specific]
### Acceptance criteria? [concrete, measurable]
### Files changed? [list]
### What should NOT have changed? [existing behavior to preserve]
```

### Step 2: Spawn Independent Reviewer

Use the Agent tool to spawn a separate agent:

```
You are an INDEPENDENT REVIEWER. Your job is to FIND FLAWS, not confirm success.

CRITICAL RULES:
1. You have NO investment in this being correct
2. Evaluate against the INTENT, not implementation elegance
3. Be direct — sugar-coating helps no one

INTENT: [paste intent]
FILES: [list files]

REVIEW PROCESS:
1. Read intent, write what you'd EXPECT before looking at code
2. Examine implementation, compare reality vs expectations
3. Actively try to break it
4. Report: Verdict (PASS/CONCERNS/FAIL), gaps, edge cases, what would break this
```

### Step 3: Act on Findings
- **PASS**: Done
- **CONCERNS**: Address or document as known limitations
- **FAIL**: Fix gaps before considering complete

## Questions That Fight Bias

Instead of "Does this work?" ask "How could this fail?"
Instead of "Is this correct?" ask "What assumptions might be wrong?"
Instead of "Did they handle errors?" ask "What errors are NOT handled?"
Instead of "Is this complete?" ask "What's missing?"

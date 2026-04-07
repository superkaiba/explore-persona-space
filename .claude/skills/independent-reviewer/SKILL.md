---
name: independent-reviewer
description: Use after completing work to get an unbiased independent review. Spawns a separate agent that evaluates whether the implementation actually achieves the stated intent, actively looking for gaps and issues without confirmation bias.
---

# Independent Reviewer Skill

Get unbiased verification that your work actually does what it's supposed to do.

## Why Independent Review?

**The problem:** When you build something, you're invested in it working. You unconsciously:
- Test to confirm, not to disprove
- See what you expect, not what's there
- Overlook gaps because you know the intent
- Rationalize issues as "edge cases"

**The solution:** An independent reviewer who:
- Only knows the INTENT, not your implementation approach
- Actively tries to find where it fails
- Has no investment in the code being correct
- Evaluates purely against stated goals

## How to Use This Skill

### Step 1: State the Intent Clearly

Before invoking the reviewer, document:

```markdown
## INTENT STATEMENT

### What was the goal?
[What were you trying to achieve? Be specific.]

### What are the acceptance criteria?
[How do we know it's working? List concrete criteria.]

### What files were changed?
[List the files that were added/modified.]

### What should NOT have changed?
[Any existing behavior that must be preserved.]
```

### Step 2: Invoke the Independent Reviewer

Use the Task tool to spawn an independent review agent:

```
Spawn a general-purpose agent with this prompt:

"You are an INDEPENDENT REVIEWER. Your job is to verify whether
the implementation actually achieves the stated intent.

CRITICAL RULES:
1. You have NO investment in this code being correct
2. Your goal is to FIND FLAWS, not confirm success
3. Evaluate against the INTENT, not implementation elegance
4. Be direct and honest - sugar-coating helps no one
5. If something seems off, it probably is

INTENT:
[Paste the intent statement here]

FILES TO REVIEW:
[List the files]

REVIEW PROCESS:
1. First, read ONLY the intent statement
2. Before looking at code, write down what you'd EXPECT to see
3. Then examine the implementation
4. Compare reality vs expectations
5. Actively try to break it or find gaps
6. Report findings honestly

OUTPUT FORMAT:
## Verdict: [PASS / PASS WITH CONCERNS / FAIL]

## Does it achieve the stated goal?
[Yes/No/Partially - with explanation]

## Gaps Found
[List any gaps between intent and implementation]

## Edge Cases Not Handled
[List edge cases that might cause issues]

## Potential Issues
[Any concerns, even minor ones]

## What Would Break This?
[Scenarios where this implementation would fail]

## Recommendation
[What should be done, if anything]"
```

### Step 3: Act on Findings

- **PASS**: Work is complete
- **PASS WITH CONCERNS**: Address concerns or document as known limitations
- **FAIL**: Fix the gaps before considering work complete

## The Review Protocol

### For the Implementer (You)

1. **Be honest about intent** - Don't downplay what you were trying to achieve
2. **List ALL changes** - Don't hide files that "aren't important"
3. **Accept criticism** - The reviewer is helping, not attacking
4. **Don't rationalize** - If they found a gap, it's a gap

### For the Reviewer (The Agent)

1. **Assume nothing works** - Verify everything
2. **Read intent first** - Form expectations before seeing code
3. **Try to break it** - Think of edge cases, invalid inputs, race conditions
4. **Be specific** - "This might have issues" is useless; "Line 42 doesn't handle null" is useful
5. **No politics** - Don't soften findings to be nice

## Review Checklist

The independent reviewer should verify:

### Functional Completeness
- [ ] Does it do what the intent says?
- [ ] Are ALL acceptance criteria met?
- [ ] Does it handle the happy path correctly?
- [ ] Does it handle error cases?
- [ ] Are edge cases covered?

### Behavioral Preservation
- [ ] Does existing functionality still work?
- [ ] Are there unintended side effects?
- [ ] Do tests still pass?
- [ ] Is the API/interface unchanged (unless intended)?

### Robustness
- [ ] What happens with invalid input?
- [ ] What happens with empty/null values?
- [ ] What happens under load/stress?
- [ ] What happens if dependencies fail?

### Code Quality (Secondary)
- [ ] Is the implementation reasonable? (not "is it how I'd do it")
- [ ] Are there obvious bugs?
- [ ] Is error handling present?

## Avoiding Bias

### Confirmation Bias Traps

| Trap | How to Avoid |
|------|--------------|
| "It probably works" | Test to DISPROVE, not confirm |
| "That edge case won't happen" | If it CAN happen, test it |
| "The implementer is smart" | Smart people make mistakes |
| "It passed the tests" | Tests might be incomplete |
| "It looks clean" | Clean code can have wrong logic |

### Questions That Fight Bias

Instead of: "Does this work?"
Ask: "How could this fail?"

Instead of: "Is this correct?"
Ask: "What assumptions does this make that might be wrong?"

Instead of: "Did they handle errors?"
Ask: "What errors are NOT handled?"

Instead of: "Is this complete?"
Ask: "What's missing?"

## Review Output Template

```markdown
# Independent Review: [Feature/Change Name]

## Intent Understood
[Paraphrase the intent to confirm understanding]

## Verdict: [PASS / PASS WITH CONCERNS / FAIL]

## Functional Assessment

### Goal Achievement: [Achieved / Partially / Not Achieved]
[Explanation]

### Acceptance Criteria
| Criterion | Met? | Notes |
|-----------|------|-------|
| [Criterion 1] | Yes/No | |
| [Criterion 2] | Yes/No | |

## Issues Found

### Critical (Must Fix)
- [Issue]: [Location]: [Why it matters]

### Important (Should Fix)
- [Issue]: [Location]: [Why it matters]

### Minor (Consider Fixing)
- [Issue]: [Location]: [Why it matters]

## Edge Cases Analysis
| Scenario | Handled? | Notes |
|----------|----------|-------|
| Empty input | | |
| Null values | | |
| Very large input | | |
| Concurrent access | | |
| [Domain-specific] | | |

## What Would Break This
1. [Scenario that would cause failure]
2. [Another scenario]

## Unintended Side Effects
[Any changes to behavior not mentioned in intent]

## Recommendations
1. [Specific action]
2. [Specific action]

## Reviewer Notes
[Any additional observations]
```

## When to Use This

**Always use after:**
- Implementing a new feature
- Fixing a bug (verify fix doesn't break other things)
- Refactoring (verify behavior unchanged)
- Any change to critical/sensitive code

**Especially use when:**
- You're not 100% confident
- The change is complex
- The stakes are high
- You've been staring at the code too long

## Quick Invocation

For a fast independent review, use this Task prompt:

```
Review this implementation as an independent, unbiased reviewer.
Your goal is to find flaws, not confirm success.

INTENT: [One sentence description]
FILES: [List files]

Check if it actually achieves the intent.
Report gaps, edge cases, and potential issues.
Be direct and honest.
```

## Sources

- [arXiv - Towards Debiasing Code Review Support](https://arxiv.org/abs/2407.01407)
- [Google Engineering Practices - Code Review](https://google.github.io/eng-practices/review/reviewer/looking-for.html)
- [Wikipedia - Independent Verification and Validation](https://en.wikipedia.org/wiki/Verification_and_validation)
- [FDA - General Principles of Software Validation](https://www.fda.gov/media/73141/download)
- [GitKraken - Code Review Best Practices 2024](https://www.gitkraken.com/blog/code-review-best-practices-2024)

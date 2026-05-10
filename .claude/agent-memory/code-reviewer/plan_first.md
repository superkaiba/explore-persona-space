---
name: Read Plan Before Diff
description: Anchor on what the plan promised, not on the implementer's narrative in the diff
type: feedback
---

Always read the approved plan BEFORE opening the diff.

**Why:** If you read the diff first, you're anchored by the implementer's narrative — their code structure implicitly tells you what problem they thought they were solving. That's exactly the frame you need to question. Reading the plan first lets you notice "they fixed X in the plan but the diff changes Y" and "the plan said no behavior change but this changes the public API".

**How to apply:**
1. Load the approved plan (research-pm's dispatch brief, or a PR description, or whatever form it takes).
2. Write down in your own words:
   - What changes the plan promises
   - What tests the plan says should pass
   - What is explicitly out of scope
3. THEN open the diff, hunk by hunk, and check each against that list.
4. Flag unplanned changes (scope creep) even if they look like improvements. "While I was there I also fixed..." is a red flag — those changes weren't reviewed during planning.

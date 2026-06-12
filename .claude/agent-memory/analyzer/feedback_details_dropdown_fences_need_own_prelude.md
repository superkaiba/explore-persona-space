---
name: Dropdown fences need their own disclosure+link prelude
description: each fenced block inside a <details> dropdown is a separate sample block for checks 10/11 — put "Cherry-picked for illustration; all rows: [link]" BEFORE the fence, not after
type: feedback
---

`verify_task_body.py` checks 10 (cherry-picked label) and 11 (qualitative-data link) iterate EVERY fenced block under `## TL;DR` independently, including fences nested inside `<details>` dropdowns. The per-fence prelude window walks back from the fence start and stops at the previous fence's closing ``` — so a trailing "All rows: [split_analysis.json](...)" line placed AFTER the inner fence satisfies neither check for that fence.

**Why:** incident #611 round-2 — 12 of 20 sample blocks FAILed check 10 and 6 FAILed check 11 because the dropdowns' summaries lacked "cherry-picked" and the raw links sat after the fences.

**How to apply:** inside every finding dropdown, put one line right after `<summary>` and before the fence: `Cherry-picked for illustration; all rows: [artifact](sha-pinned-url); raw arrays: [tree](url).` This single line satisfies both checks for the inner fence AND for the enclosing `<details>` block (its content is also scanned). Putting "cherry-picked for illustration" in the `<summary>` text works too.

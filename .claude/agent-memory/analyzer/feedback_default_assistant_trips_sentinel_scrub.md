---
name: default-assistant trips Repro sentinel scrub
description: verify_task_body check 9 regex \bdefault\b matches hyphenated "default-assistant" in ## Reproducibility; use the underscore slug `default_assistant` there
type: feedback
---

In `## Reproducibility`, the sentinel-scrub check (`verify_task_body.py` check 9) FAILs on the bare word `default` matched with `\bdefault\b` — and a hyphen IS a word boundary, so the core condition name "default-assistant" trips it (underscores are word chars, so `default_assistant` is safe).

**Why:** the check guards against placeholder text ("default" as an unfilled config value), but this project's most common probe condition is literally named default-assistant; incident: #611 round-2 body draft FAILed check 9 on "150 default-assistant negatives" etc.

**How to apply:** in `## Reproducibility` only, write the underscore config slug `` `default_assistant` `` (backticked) or rephrase ("the bare-assistant probe"); the hyphenated plain-English form stays fine everywhere else in the body (`## Takeaways` / `## What I ran` / `## Findings` / `## Data` — for a grandfathered v2 body, `## TL;DR` / `## Human TL;DR`). Check 9 scans `## Reproducibility` in both v3 and v2 bodies.

---
name: Source persona included in bystander fire count
description: A table row claimed bystander fires but the number included source-persona fires; 1597 vs actual 1387 (23.5% vs 20.4%) — off because 210 paramedic source fires were not excluded
type: feedback
---

In issue #311 the Result 2 table showed joint LoRA bystander fires as 1597 (23.5%) but the actual bystander count (excluding both paramedic and comedian sources) was 1387 (20.4%). The error was: 1988 total fires − 391 comedian fires = 1597, which correctly excluded comedian but kept paramedic source fires. The correct calculation subtracts BOTH sources.

**Why:** When one source generalizes broadly (paramedic) and the other does not (comedian), it is tempting to treat the broad-generalizer's own source rate as if it "belongs" to the bystander mass. The script may have only excluded the source specified as "B" from bystander counts.

**How to apply:** When an experiment has multiple source personas and reports bystander aggregate fire counts, independently verify that ALL source personas (not just one) are excluded from the bystander denominator and numerator. Re-compute: total_fires − sum(source_persona_fires) and compare to the claimed bystander total.

---
name: Window-relative smoke assertion ages out
description: Infra acceptance smokes that assert a count >= K over a rolling time window silently invert (false-FAIL) as the data ages out of the window
type: feedback
---

A `kind: infra` acceptance/smoke check that asserts `count >= K` over a
ROLLING time window (`--window-days N`) against the LIVE tree is only valid
on the day it's written — the data that produced the count ages out of the
window, so the same correct code reports a smaller count later and the
assertion false-FAILs.

**Why:** task #711 (lesson-consolidation cron). The §6 real-tree smoke ran
`consolidate_lessons.py --dry-run --window-days 7` and asserted
`unparseable_skipped >= 1`. Independently re-classifying all 55 live markers
with the plan's own regexes: the 4 truly-skipped (tier-3) markers were ALL
from #650 @ 2026-06-16, and the 2 in-window off-format markers (#653, #667 @
2026-06-25) were tier-2 RECOVERABLE (bare-fields), not skips. So on any run
after ~2026-06-23 a correct script reports `unparseable_skipped == 0` in the
7-day window — the `>= 1` assertion FAILs on correct code. The plan also
conflated "off-format" (7) with "tier-3 skipped" (4) in the smoke arithmetic.

**How to apply:** When reviewing an infra plan's acceptance/smoke that pairs
a rolling window with a `count >= K` assertion: re-run the classification
yourself with the plan's own regex, check whether the count-producing rows
sit INSIDE the window at IMPLEMENTATION time (not plan-authoring time). If
not, it's a smoke shaped to FAIL on correct code — the measurement-defect
mirror of "can't fail when it should". Fix is trivial (widen the window for
the skip-count assertion, or assert `>= 0`), so it's a low-severity REVISE
unless the smoke is the only thing gating the implementation report. The
exit-code-0 half of such a smoke is usually the load-bearing check; the
count half is the brittle one.

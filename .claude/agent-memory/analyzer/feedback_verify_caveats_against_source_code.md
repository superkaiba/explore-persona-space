---
name: Verify caveats against the actual source code, not the plan summary
description: When stating a "binding constraint" caveat that compares THIS run vs a prior run (e.g., "we used N negatives, plan called for M"), grep the actual launch script for the constants, not the plan's summary. Plans drift; code is canonical.
type: feedback
---

When the Confidence sentence or Standing Caveats name a numerical comparison vs a prior experiment ("we used 200 negatives vs the [#205] spec 400", "20 epochs vs 60", etc.), VERIFY the prior experiment's actual constant from its launch script. Don't quote the current plan's summary of the prior run.

**Why:** Issue #247 v1 stated "stage-2 negative-per-persona was halved (200 vs the [#205]-spec 400)". This was factually wrong: `scripts/run_issue205_per_condition.py:81-82` defines `COUPLING_N_POSITIVE = 200` and `COUPLING_N_NEGATIVE_PER = 200` — both #205 and #247 used 200/persona × 2 personas = 400 total. The "400" in the plan was the total count, but the v1 body misread it as per-persona. The round-1 critic caught it; verifying against `run_issue205_per_condition.py:81-82` directly would have prevented the error.

**How to apply:** Before naming a numerical caveat that contains a comparison vs a prior run, run:

```bash
grep -n "COUPLING_N_\|N_NEGATIVE\|N_POSITIVE\|n_epochs\|epochs=" scripts/run_<prior-issue>_*.py
```

and quote the line number in the body. If the caveat says "X vs the [#prior]-spec Y", confirm Y is what `<prior-script>` actually used, not what the plan body summarized.

Same rule applies to "we ran 20 epochs vs the spec's 60" or any other "this issue used N, prior used M" claim. Always verify M from the prior script directly.

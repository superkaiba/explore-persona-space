---
name: ERROR-demotion safety-regression analysis (infra fixes)
description: Before calling an ERROR→WARNING demotion a lost safety property, check whether the old ERROR was BINDING anywhere — a consumer-tolerated error string is already a neutralized no-op (#554)
type: feedback
---

When an infra plan demotes a preflight/gate ERROR to a WARNING, the safety-regression
question is not "what did the ERROR nominally guard?" but "where was the ERROR actually
BINDING?" Map every consumer into three classes:

1. **Tolerated** — consumer pattern-matches the exact error string and treats it as PASS
   (e.g. `experimenter.md` L175-179 + `experiment-implementer.md` L405-411 both neutralize
   `Local is N commit(s) behind origin/main`). The old ERROR provided ZERO protection here;
   demotion loses nothing.
2. **Always-fail** — consumer gates on `ok=false` with no tolerance on a path where the
   error fires by construction (SKILL.md Step 6c resumed-pod gate on issue branches). Zero
   discrimination = the gate IS the defect, not a safety property.
3. **Genuinely discriminating** — only this class makes a demotion a real regression.

**Why:** #554 (2026-06-12): the behind-main ERROR looked like a stale-code guard, but both
canonical agent consumers tolerated the string and the one binding consumer always-failed.
The replacement behind-OWN-origin ERROR is strictly MORE discriminating for the realistic
stale-pod failure (checkout behind the reviewed branch tip).

**How to apply:** For each demoted ERROR, grep `.claude/agents/`, `.claude/skills/`, and
scripts for the exact error string + `ok=false`-style gates; classify consumers; flag a
regression only if class 3 exists. Companion check: what the new residual ERROR catches
should superset the realistic incidents the old one was credited with.

---
name: ERROR-demotion safety-regression analysis (infra fixes)
description: Before calling an ERROR→WARNING demotion a lost safety property, check where the old ERROR was BINDING — tolerated-string consumers are already neutralized no-ops; always-fail gates are the defect itself (#554)
type: feedback
---

When an infra plan demotes a preflight/gate ERROR to WARNING, the safety-regression question is not "what did the ERROR nominally guard?" but "where was it actually BINDING?" Map every consumer into three classes:

1. **Tolerated** — the consumer pattern-matches the exact error string and treats it as PASS (experimenter.md + experiment-implementer.md both neutralize `Local is N commit(s) behind origin/main`). The old ERROR provided ZERO protection; demotion loses nothing.
2. **Always-fail** — the consumer gates on `ok=false` with no tolerance on a path where the error fires by construction (resumed-pod gate on issue branches). Zero discrimination = the gate IS the defect, not a safety property.
3. **Genuinely discriminating** — only this class makes a demotion a real regression.

**Why (#554, 2026-06-12):** the behind-main ERROR looked like a stale-code guard, but both canonical agent consumers tolerated the string and the one binding consumer always-failed; the replacement behind-OWN-origin ERROR is strictly MORE discriminating for the realistic stale-pod failure.

**How to apply:** for each demoted ERROR, grep `.claude/agents/`, `.claude/skills/`, and scripts for the exact error string + `ok=false`-style gates; classify consumers; flag a regression only if class 3 exists. Companion: what the new residual ERROR catches should superset the realistic incidents the old one was credited with.

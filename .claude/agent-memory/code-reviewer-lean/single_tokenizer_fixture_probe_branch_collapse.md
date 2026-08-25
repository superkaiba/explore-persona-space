---
name: single-tokenizer-fixture-probe-branch-collapse
description: A one-tokenizer smoke fixture forces every tokenizer-probed branch (has_chat_template etc.) to ONE side for all units — probe-keyed asserts go tautological under smoke; verify the other arm is statically unreachable in production and the substitution is enumerated with the commit SHA
metadata:
  type: feedback
---

When a smoke fixture serves ONE tokenizer/model for every unit (the
`_SMOKE_MODEL_DIR`-rebind shape), every branch keyed on a tokenizer PROBE
(`has_chat_template`, special-token presence, vocab checks) collapses to one
side for ALL smoke units — asserts on that probe become tautologically
satisfiable under smoke, and the other arm is smoke-unreachable.

**Why:** #2544 r1 g5: the olmo3 fixture tokenizer moved base→Instruct so rung
R's natgen `has_chat_template` assert could pass under smoke; afterwards NO
smoke rung can exercise the template-less arm, so a natgen cell mis-scheduled
onto a template-less rung is smoke-invisible.

**How to apply:** on any fixture-tokenizer/model identity change: (1) diff the
fixture id against the production config's pinned id for the rung the fix
targets (ladder JSON, not the commit message); (2) grep every consumer of the
probe (`has_chat_template` etc.) and confirm the now-unreachable arm is
production-safe via STATIC scheduling (e.g. `NATIVE_GEN_RUNGS` ⊂ templated
rungs), not runtime probing; (3) require the substitution enumerated in the
implementer's blind-spot block WITH the commit SHA (plan-time enumeration
predates smoke-found fixes); (4) sweep that no production path reads the
fixture constant (only the smoke env-var seam). See
[[smoke_fixture_authored_with_consumer_keys]].

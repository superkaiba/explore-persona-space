---
name: plan-identifier-renamed-consistently
description: Plan-literal env/flag/symbol renamed in implementation — grep the ROUND for BOTH names; zero old-name hits = disclosed Minor, mixed use = the real bug (#2479 R1 g3)
metadata:
  type: feedback
---

When a diff implements a plan-named identifier (env var, flag, symbol) under a
DIFFERENT name (e.g. plan `EPM_I1345_CHAR_PANEL_JSON` → code
`EPM_I2479_CHAR_PANEL_JSON`, #2479 r1 g3), severity turns on CONSISTENCY, not
the rename itself: grep the whole round/worktree for BOTH names. Zero code
hits for the plan's name + all consumers on the new name = a disclosed
naming deviation (Minor note, mechanizable as "old-name grep must stay
empty"). ANY mixed use across consumers = a split-brain seam (one consumer
armed, another inert) — that is a Major/Critical, and the one worth hunting.

**Why:** the plan literal is a spec detail, but a half-renamed seam silently
divides consumers into env-armed and env-blind sets with no error.

**How to apply:** whenever the diff's seam/dial name differs from the plan's
parenthetical, run the two-name grep across scripts/ + tests/ + launch shell
before writing the finding; cite both counts in the verdict.

Sibling lesson from the same round: a loader schema ban that looks over-strict
(substring ban `"_op" in x` instead of an `endswith` check) can be
LOAD-BEARING because a downstream classifier keys on `in` not `endswith`
(`variant_mode_model`) — read the consumer's parse logic before flagging
over-strictness. Related: [[submodule-existence-bypasses-strict-identity]],
[[smoke-fixture-authored-with-consumer-keys]].

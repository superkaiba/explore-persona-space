---
name: pinned-strings-vs-pinned-gate-contradiction
description: When a plan pins BOTH the frozen strings and a numeric gate they must pass, live-probe both; an implementer resolving the contradiction by weakening the gate's granularity is a disclosed deviation to grade, not a bug or a silent pass
metadata:
  type: feedback
---

A plan can pin a frozen instrument's STRINGS and a numeric GATE over them
that the strings themselves violate (#2564 r1 g1: 5/12 plan-pinned query
paraphrases broke the plan's own ±30% string-level token-ratio gate, max
1.444). The implementer's usual resolution is to move the gate to a weaker
granularity that the pinned strings pass (here: rendered-context ratio,
where ~21 tokens of chat-template scaffolding dilute every ratio toward 1 —
max fell 1.444 → 1.182).

**Why:** the strings are the frozen instrument, so failing the build on
plan-pinned inputs is not an option the implementer controls; but the
weakened gate certifies less than the plan's prose claims, and the docstring
disclosure alone does not reach the clean-result methodology.

**How to apply:** on any frozen-bank / datagen-gate commit, (1) live-probe
the gate at BOTH granularities (string-level and rendered/derived level) to
confirm the contradiction is real and measured, not asserted; (2) check
whether a registered covariate (e.g. per-pair `changed_tokens`) already
carries the honest number the weakened gate no longer bounds — if so the
deviation is Minor; (3) require the deviation be named in the implementation
report / run digest, not only the module docstring. Related:
[[fixed-name-tmp-atomic-write-fanout-race]] (same round shape: later commit
fixed the scoped commit's `.tmp` write — check HEAD drift before flagging).

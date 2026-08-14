---
name: regen-trigger-headroom-at-production-cap
description: An armed >=2x cap-hit re-gen trigger needs max_model_len headroom re-checked at the REGEN cap, not the first-pass cap; a low-cap smoke of the trigger certifies nothing (#2221 v9)
metadata:
  type: feedback
---

When a plan arms the >2% cap-hit re-gen trigger (CLAUDE.md: re-generate capped
rows at >= 2x the cap), check `max_model_len - regen_cap >= max prompt tokens`
for the REGEN leg — the first-pass headroom check does not cover it.

**Why:** #2221 v9: EVAL cap 2048 → regen floor 4096 while
`issue778_lib.build_vllm_engine` hard-pins `max_model_len=4096` with no
override param → regen prompt budget = 4096 − 4096 = 0.
`issue2221_stage_corpus._regen_cell` skips any prompt whose render exceeds
`VLLM_MAX_MODEL_LEN − regen_cap`, so EVERY row of a triggered cell would be
`regen_overlong_skipped` — a structural no-op that still writes
`regen_applied: true` (n_regen=0, residual = pre-regen). The plan's §4
headroom paragraph checked only the FIRST pass (prompt + 2048 ≤ 4096) and
claimed "no shared-module edit needed"; its A9 smoke ("force a LOW cap to
trip the trigger") passes trivially because a low cap has huge regen headroom
— an unenumerated smoke blind spot (#1336 downgraded-gate family; 3rd
recurrence of the #505/#601 cap-vs-max_model_len class, now at the regen leg).

**How to apply:** for any plan row arming a ≥2× re-gen: (1) verify the
regen-leg engine parameterization (`max_model_len ≥ regen_cap + stated prompt
bound`, or a named fresh-engine build for the regen pass); (2) demand the
smoke exercise the trigger at the PRODUCTION regen cap (or the blind-spot
enumeration names the gap); (3) re-derive any prompt-length filter for the
regen budget, not the first-pass budget (e.g. cap 1024 → regen 2048 leaves
only 2048 prompt tokens under a 4096 pin — long real-corpus prompts like
r/AITA posts can exceed it).

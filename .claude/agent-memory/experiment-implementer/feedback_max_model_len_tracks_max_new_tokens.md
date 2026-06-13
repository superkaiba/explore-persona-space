---
name: max_model_len must track max_new_tokens deviations
description: Raising max_new_tokens on an inherited vLLM eval rig requires raising max_model_len at the call site — cap-length generations re-enter as prompts and overflow the engine cap
type: feedback
---

When a plan names a max-token deviation (e.g. D3: `max_new_tokens` 1024 → 2048) on an
inherited eval rig, ALSO pass a larger `max_model_len` at the vLLM call site. The engine
validates prompt length, and a cap-length prior generation re-enters as a prompt in the
follow-on read (prompt + R + marker = cap + overhead), crashing with
`ValueError: decoder prompt longer than maximum model length`.

**Why:** incident #601 (2026-06-11) — Phase 0's on-policy worker crashed on
`DEFAULT_MAX_MODEL_LEN = 2048` (sized for the parent's 1024-token generations) the first
time a noneg-cell generation ran to the 2048 cap; the launch was halted and relaunched
(hot-fix: `max_model_len=4096`).

**How to apply:** whenever an implementation inherits an eval module and changes any
max-token knob, grep the parent module for `max_model_len` / `DEFAULT_MAX_MODEL_LEN` and
size it ≥ prompt headroom + max_new_tokens + slot overhead. Add it to the smoke asserts
when the smoke can't reach a cap-length generation (CPU carve-outs can't).

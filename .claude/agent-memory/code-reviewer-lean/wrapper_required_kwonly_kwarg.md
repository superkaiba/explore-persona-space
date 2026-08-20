---
name: wrapper-required-kwonly-kwarg
description: Fork call sites of a shared helper can omit a REQUIRED keyword-only kwarg (hub.retry_transient's what=) — grep the def, live-probe wrapper(lambda: True); smoke passes because the leg is production-only
metadata:
  type: feedback
---

When a diff calls a shared wrapper/helper (esp. `hub.retry_transient` = `_retry_upload(fn, *, what: str, ...)`),
check the helper's `def` for keyword-only params WITHOUT defaults and certify the call shape with a 3-line live
probe (`retry_transient(lambda: True)` → TypeError proves the class). Prior in-repo callers all pass the kwarg;
a fork that copies the call from a docstring/comment drops it.

**Why:** #2389 r1 g1 — all FIVE fork `retry_transient` call sites omitted required `what=`; gate-0c crashed
TypeError on every production `--upload hf` bank run. Every smoke/test battery was green because
`--upload none|local-mirror` early-returns before the call — a #1355/0.71-class production-only leg.

**How to apply:** for each shared-helper call a fork adds: (1) grep the helper def for `*, name: type` params
sans default; (2) live-probe the minimal call; (3) sweep ALL sibling call sites round-wide (the class repeats —
5/5 sites shared the bug); (4) check whether any smoke/upload-mode gate makes the call site production-only.
Related: [[amend-phase-striding-filters]].

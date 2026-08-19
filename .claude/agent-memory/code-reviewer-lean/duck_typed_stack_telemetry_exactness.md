---
name: duck-typed-stack-telemetry-exactness
description: A hook stack duck-typing a single-hook contract must be checked against EVERY consumer, incl. telemetry-exactness asserts (n_edits +1 per forward) that a summed stack counter breaks
metadata:
  type: feedback
---

When a diff adds a stack/composite that duck-types a single-object contract
(install/arm/_handle pattern, #2254 `MultiLayerDeltaHook`), grep for ALL
consumers of the duck-typed attributes — not just the one the docstring names.
The primary consumer (`generate_batch`) only null-checks `_handle`, but a
sibling consumer (`steering.py` capture path) asserts `n_edits == before + 1`
per chunk; a stack summing children's `n_edits` increments by K and would trip
it. Verify the stack is structurally unreachable from such consumers (e.g. it
lacks the `arm_at` method that path requires — fail-loud AttributeError) or
flag it.

**Why:** the duck type is only as wide as the consumers actually exercised;
telemetry counters read as exactness gates are the easy-to-miss consumer class.
**How to apply:** on any duck-typed composite, `grep -n` each forwarded
attribute (`_handle`, `n_edits`, `arm`) across the module's consumers and
classify each as null-check / exactness / unused before crediting the
contract as satisfied.

---
name: Anthropic system-role lift at dispatcher seams
description: Anthropic Messages API 400s on role:"system" in messages — lift to top-level system= at every request-builder seam; mocked suites never catch it (incident #906 r11)
type: feedback
---

The Anthropic Messages API rejects `"system"` as a message ROLE — OpenAI-style
message lists must lift system entries into the top-level `system=` param or
EVERY request 400s (`messages.0: use the top-level 'system' parameter`).

**Why:** #906's first `--full` run failed its sycophancy class (36/36 datagen
generation requests 400 → kept 0 < yield floor) because
`artifacts/datagen.py::_default_generate_fn.build_request` forwarded
`gen_messages` verbatim. Ten review rounds of signature-pinned contract tests
never caught it — the live-API seam is exactly the one the mocked suite never
executes.

**How to apply:** any request-builder seam forwarding message lists to
anthropic needs (a) a system-lift (`_gen_params_from_messages` in datagen.py is
the canonical helper — reuse it), and (b) a tiny LIVE forced probe with a
system-BEARING message list before production; a mock smoke or a system-less
live smoke cannot catch the shape bug.

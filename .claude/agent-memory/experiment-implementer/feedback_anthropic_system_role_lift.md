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

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Anthropic system-role lift at dispatcher seams](feedback_anthropic_system_role_lift.md) — Messages API 400s on role:system in messages; lift to top-level system=; live system-bearing probe required (#906)

## Merged sibling index rows (#2032 curation, 2026-08-03)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the agent-memory index size cap (task #2032). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [Signature-bind faked-boundary constructors](feedback_kwargs_constructor_bind.md) — **kwargs callees hide config-dataclass kwarg (#906)

---
name: Register dynamic registry entries at point of use, never via phase side effects
description: A resumed process fast-forwards earlier phases, losing their in-process registry-registration side effects — register dynamic CONTEXTS/ids immediately before each consumer (#1315 r6)
type: feedback
---

A dispatcher whose `ModelOrganism`/`CONTEXTS` consumers rely on an EARLIER phase's in-process registry-registration side effect crashes on any RESUMED process: resume fast-forward skips the registering phase, so the fresh process's registry lacks the dynamic id (`icl_prefix_impolite`, #1315 r6) while a fresh-out_root smoke passes via the side effect.

**Why:** resume predicates skip phases by done-file, but in-process state (registries, caches) is rebuilt per process — any consumer depending on a skipped phase's side effect sees a fresh, unpopulated registry.

**How to apply:** register dynamic contexts/ids at POINT OF USE (e.g. a `_context()` call immediately before each organism construction), never rely on phase-ordering side effects. Also audit shared-lineage registries (e.g. `issue779_common.TRAITS`) for membership asserts a new behavior only trips on production-only branches the smoke never reaches.

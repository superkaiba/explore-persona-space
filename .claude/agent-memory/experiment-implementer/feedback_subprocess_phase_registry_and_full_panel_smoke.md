---
name: subprocess-phase-registry-and-full-panel-smoke
description: A subprocess-per-phase dispatcher inherits no module-level registry state — register dynamic ids idempotently at each phase entry, and smoke the FULL production panel in a fresh child process.
type: feedback
---

Register every dynamic context/registry id idempotently AT EACH PHASE ENTRY
that resolves it, and make the smoke resolve the FULL production panel in a
FRESH child process. "The registrar runs somewhere" is not enough.

**Why:** #1090 fu6 (2026-07-17): `phase_dispatch` ran `--phase
capture-organisms` as a subprocess; the held-out panel contexts
(`neg_sp_police`/`neg_sp_ph4`) existed only as `default_panel()` members
registered in NO process — `register_fu3_contexts()` only ever registered the
wildchat prefix. Production crashed at `ensure_context`. The tiny-real e2e
smoke passed because its `SMOKE_PANEL_IDS` slice (3 ids) dropped exactly the
panel-member-only ids — a panel-slice mask stacked on the subprocess seam;
one full pod cycle burned.

**How to apply:** when a driver dispatches phases as subprocesses, (1) each
phase entry that resolves dynamic ids calls an idempotent registrar
unconditionally; (2) the smoke for that phase runs in a FRESH child process
(zero parent state) and resolves the FULL production id set, never a slice
that drops members with no other registration path.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Subprocess phase registry + full-panel smoke](feedback_subprocess_phase_registry_and_full_panel_smoke.md) — subprocess phases inherit no module registries; register at phase entry + smoke the FULL panel in a fresh child (#1090 fu6)

---
name: Claude PASSes scaffolded-but-not-plumbed pipelines (merged family)
description: A component exists + is tested but nothing in production wires it — orphaned helpers (no caller), readers with no writer, plan-mandated HALT gates documented as "the runner invokes X" that the runner never invokes. Grep for the production wiring site before believing presence.
type: feedback
---

**Rule:** presence + green tests ≠ wired. When a round adds a helper / reader / gate, `rg` for the PRODUCTION site that calls/writes/invokes it before believing Claude's "BLOCKER resolved". If all non-definition matches are tests/docstrings, the component is scaffolding; if a first-class deliverable or plan-mandated HALT gate depends on it, FAIL.

**Shapes:**
1. **Orphaned helper (#397 r2):** "port helper Y" blocker → writer + reader + panel-builder + 5 unit tests land, but `rg '<helper>\(' src scripts tests` shows zero production callers; a future dispatcher trivially calls the default path, silently re-introducing the train/eval mismatch the port was meant to prevent. If the orchestrator EXPLICITLY accepted a deferred dispatcher, PASS with a binding standing-rec naming the wiring site (+ missing/corrupt manifest must raise, never fall back); otherwise FAIL.
2. **Reader with no writer (#508 r2):** renderer reads `eval_json["X_path"]` / sidecar `X.json`; `rg "X_path|X\.json"` finds no WRITER (`extract_fullft_dynamics_from_checkpoints` referenced in 3 docstrings, never defined; snapshots in-memory only). Synthetic-data test PASS masks the production no-op. Bonus cadence check: a writer firing only at endpoint (`ckpt_fractions=(1.0,)`) gives 1 point where the plan demands a trajectory — FAIL when the plan ties the trajectory to headline interpretation.
3. **Plan-mandated HALT gate unplumbed (#516 r3):** plan §9.1 lists 6 gates; round-N must-fix carried only A/B/C; gate #3's own builder docstring says the gate is "deferred to ... the runner invokes" while `rg` of the runner's phase function returns zero hits and control flow returns immediately after the subprocess. Plan-mandated HALT gates are first-class plan-adherence — bypassing one is Real-blocking regardless of round-cap pressure (bad inputs flow into expensive phases → unfalsifiable null). Open the plan's FULL gate list as a class, not just the carried-forward items.
4. **Registered §6.5 deliverable a phase never produces (#734 r1, Codex caught, Claude PASSed):** the helper that writes the deliverable (`corrected_slot_stats` → `marker_slot_corrected.json`) WAS defined, tested, and wired into ONE phase (Phase 1, the reused-adapter re-read), so a presence-grep and the regression test both read green — but a SECOND phase (Phase 2 / H1) that the plan §6.5 `primary_deliverable` glob ALSO requires ("≥6 H1 cells") trains its adapters and persists ONLY the band-stop result, never calling the read helper. The fix: walk the §6.5 `primary_deliverable` globs and confirm EACH phase that the glob's `note` enumerates actually writes a file matching it — `rg 'corrected_slot_stats|reread_cell'` inside `run_phase2`/`train_h1_cell` returned zero, and the phase return dict (`{"trained_cells","base_first_seed_in_band"}`) had no read column. The Claude reviewer's "regression test pins the H3 slot invariant" was true but covered only Phase 1's arithmetic; the test never exercised the H1 phase, so it could not catch the missing-deliverable. Lesson: a deliverable wired into one phase is not wired into every phase the plan's deliverable note names — enumerate the glob's phases as a class.

**Smell:** a docstring outsourcing the obligation ("the runner invokes X", "out of scope for this implementer", "dispatcher handles it") with no implementing site.

Contrast: [[feedback_codex_plan_section_in_scope]] — when Codex flags an OUT-of-round-brief plan section on an un-invoked path, downgrade to PASS+CONCERNS; when the gap is an IN-scope plan-mandated gate or a first-class deliverable's missing wiring, FAIL stands. Companions: [[feedback_claude_misses_dispatcher_wire_bugs]]; [[feedback_claude_misses_same_file_siblings]].

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Claude PASSes scaffolded-but-not-plumbed pipelines](feedback_claude_scaffolded_pipeline_not_plumbed.md) — orphaned helpers, readers with no writer, plan-mandated HALT gates the runner never invokes; grep the production wiring site. #397/#508/#516.

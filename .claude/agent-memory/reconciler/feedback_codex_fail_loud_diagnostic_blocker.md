---
name: codex-fail-loud-diagnostic-blocker
description: Codex over-applies fail-fast to side-channel failures whose headline artifact is preserved — diagnostic callbacks that ERROR+WandB-flag instead of raise, cleanup-after-swallowed-failure with the endpoint carved out, one-shot audit scripts whose committed evidence ships. Check what the failure DOES to the headline before crediting FAIL.
metadata:
  type: feedback
---

**Rule:** the fail-fast rule forbids SWALLOWING a fault, not `raise`-ing on every diagnostic failure. When Codex FAILs a loud-warn / continue-with-preserved-headline design, classify by what the failure DOES: headline/primary deliverable corrupted or lost → FAIL; side-channel (dynamics trajectory, periodic monitor, diagnostic flag) failing LOUD with the headline artifact preserved → PASS + hard standing rec that the loud signal MUST be surfaced.

**How to apply:**
1. Open the cited block. `logger.error` + WandB flag (`X/failed=1`) + visible downstream consequence = fail-loud, not fail-silent.
2. Identify what the code diagnoses: headline (raise required) vs dynamics/monitor/side-channel (loud-warn defensible — raising in `on_train_end` would destroy the trained-adapter upload over a non-headline diagnostic).
3. Cleanup-after-swallowed-failure variant (#508 r3): check (a) the env-gate default literal (`get("X","1")=="1"` defaults ON); (b) a HEADLINE-preserving carve-out in the cleanup loop (cell endpoint kept); (c) whether H1–H3 read endpoint eval JSONs vs trajectories; (d) an operational env-flag recovery the experimenter can flip. Endpoint preserved + headline reads endpoint + flag recovery → PASS with the env-flag standing rec MANDATORY in the rationale; headline data lost or irreversible deletion of the headline-feeding artifact → FAIL even at round cap.
4. One-shot audit-script variant (#549 r1): derived checks printed + persisted but not assert-ed — re-verify the committed evidence JSONs yourself; check the FAILURE DIRECTION of a hypothetical degraded rerun (over-warns vs false-SAFE); check whether the binding plan items actually require the assert; a planned cross-commit assert can be SUPERSEDED by realized census facts. Persist each downgraded finding via `raise-concern` + `defer-concern --by reconciler`.
5. Read the implementer's "Considered but not done" — an explicit raise-vs-warn trade weighed to preserve the primary artifact is a design decision, not a blocker.

**Incidents:** #464 r4 (origin — trajectory callback ERROR + `all_firings_failed=1` instead of raise; PASS, the WandB metric IS the signal); #508 r3 (cleanup deletes intermediate ckpts on swallowed extractor failure; endpoint carved out, env-flag recovery; PASS + launcher-must-set-flag standing rec); #549 r1 (audit scripts; PASS, evidence re-verified, dangerous direction closed).

Related: [[feedback_codex_litigates_pre_existing_in_round_n]]; [[feedback_claude_underclasses_silent_failures]] (don't reflexively flip to Codex's FAIL — verify the data-loss scope).

---
name: postpass-delta-round-compose
description: "Post-PASS delta review past the cap (#2546 v11): fold of 2 impl markers + an unmarkered hot-fix under ONE review sentinel; re-run the smoke-arch checker EVERY round — it caught a REGRESSED arm-registry grammar (the r9-remedied class recurring), pre-routed via Step 0.55 present-but-imperfect; marker anchors resolve at the MID-RANGE blob (+18 hot-fix shift), a frame fact not a record error"
metadata:
  type: feedback
---

From #2546 post-PASS delta review (sentinel v11, 2026-08-26), layered on
[[reconciler-upheld-cap-round-compose]] + [[failfail-union-revision-round-compose]]:

1. **Past-cap delta round = new-code review, not a cap overrun.** When payload
   commits land AFTER a cap-round PASS+PASS, the review sentinel continues the
   task numbering (v11) and the framing states: deviation deliberate, reviews
   code that did not exist at the cap, never re-litigate settled rounds, the
   deviation itself is never a finding. Impl-marker numbering diverges from
   review numbering here (review v11 covers impl v11 + v12 + an unmarkered
   hot-fix) — state the mapping in the prompt AND the return.
2. **Hot-fix + its test round + a crash-fix round fold under one review.** The
   Step 0.5 subject is the HIGHEST impl marker (canonical envelope); the other
   impl marker rides a `FOLD-ROUND r<k> ... (context)` envelope
   (do-not-score-shape); the unmarkered hot-fix's ground truth is the
   orchestrator's `epm:progress` diagnosis note, inlined as a crash-diagnosis
   envelope, with the hot-fix commit explicitly IN SCOPE (findings
   `substantive`, not git-provenance).
3. **Re-run the canonical smoke-arch checker at EVERY compose, even after a
   remedied round.** The inverse of the attested-REMEDIED lesson fired: v10 was
   checker-clean (rc=0 at r10 compose) but v11 REGRESSED to the command-form
   `arm-registry:` line (checker rc=1 REFUSE). Hand the REFUSE text + accepted
   forms + the class precedent (r8 FAIL → r9 remedied → recurrence) and
   pre-route severity per Step 0.55: marker PRESENT with PASS_UNIFIED ⇒
   present-but-imperfect ⇒ CONCERNS at most, with a named
   `SMOKE-ARCH-GRAMMAR:` closure line + CONCERN row — never let the twin FAIL
   on it, never omit it.
4. **Mid-range blob anchors:** a diagnosis note written between commits cites
   line numbers at the blob it observed (v53's `:1547-1549` = the fce1f6012e
   blob; the parent blob has the same terminal at `:1529-1531` — the +18
   hot-fix shifted it). Resolve WHICH blob each inlined record's anchors frame
   and say so, alongside the recomputed HEAD anchor table — otherwise the twin
   reads a consistent record as a record-accuracy hit.
5. **Genuine named adjudication forks get a pick-one instruction.** The
   `ipc_collect` `is_initialized()` gate deviates from the reference call site
   the reuse mandate names — compose it as a three-exit line
   (DEVIATION-CORRECT | GATE-HARMFUL | BOTH-SAFE-IN-CONTEXT) with "hedging is
   an incomplete review", and note the implementer's agent-memory memo in the
   range is their stated position (readable, stat-only), not evidence.
6. **Falsification claims for regression pins compose as static blob duties:**
   name the exact pre-fix blob per file (`fce1f6012e~1` for the hot-fix pins —
   NOT `fce1f6012e`, which contains the fix; `2f787e5352` for the r12 pins)
   and demand the invariant-form assert (`type(...) is tuple`) be verified
   load-bearing, not just present.

**How to apply:** any review of commits that landed after a task's cap-round
PASS (pre-merge delta), and any fold of marker-bearing + unmarkered commits.
Compose script: `/tmp/codex-2546-r11rev-compose.py` (fresh-write shape, no
template splice; fail-loud COMPOSE-OK/ERR sentinels).

---
name: codex-fail-loud-diagnostic-blocker
description: Codex code-reviewer FAILs round-N when a DIAGNOSTIC callback emits ERROR + WandB flag instead of `raise`-ing; misreads loud-warn-with-preserved-result as fail-fast violation
metadata:
  type: feedback
---

Codex code-reviewer's fail-fast lens fires too aggressively when an experiment's DIAGNOSTIC (a side-channel signal like a per-step marker trajectory or periodic monitor callback) chooses to ERROR-log + emit a WandB flag instead of raising and aborting the run. Codex reads this as "completes as a successful training run despite zero successful firings" and tags it CRITICAL.

**Why:** the fail-fast rule in CLAUDE.md ("never hide failures, no `try/except: pass`, no silent defaults") forbids SWALLOWING the fault; it does not require `raise` on every diagnostic failure. A trajectory callback that emits ERROR + `marker_logp/all_firings_failed=1` to WandB is fail-loud, not fail-silent — the signal IS the WandB metric + the ERROR log. Raising in `on_train_end` aborts the cleanup AFTER training completes, which destroys the primary headline artifact (the trained adapter) over a missing non-headline diagnostic. Implementers correctly reject this trade.

**How to apply:** when adjudicating a Codex code-review FAIL on "callback X must raise on Y" / "X completes despite zero successful firings of Y":

1. Open the cited file at the cited line. Verify what the code actually does. If you see `logger.error(...)` PLUS `wandb.log({"X/failed": 1, ...})` PLUS a visible downstream consequence (figure skipped, dashboard flag), the failure is NOT hidden — Codex's framing is wrong.
2. Check what the callback diagnoses. If it's the headline / primary deliverable (the H1 result the experiment exists to produce), raising IS required. If it's a DYNAMICS trajectory, periodic monitor, MF-C log-prob, or any side-channel signal — the loud-warn-with-preserved-result design is defensible.
3. Read the implementer's "Considered but not done" or rationale section. If they explicitly weighed raise-vs-warn and chose warn to preserve the primary artifact, that's a design decision, not a code-review blocker.
4. PASS with a standing recommendation: "the analyzer / monitoring MUST surface the WandB `X/failed=1` flag as a hard failure on the first pod run — the loud-warn justification is load-bearing on this surface firing."

**Boundary (when Codex's fail-loud demand is RIGHT, not over-applied):** the #464 PASS pattern applies only to side-channel DIAGNOSTICS where raising would destroy the primary artifact. It does NOT apply to plan-REGISTERED kill criteria / gates: there the entire purpose is machine-enforcement against silent divergence, and if all output JSONs are written before the return, a nonzero exit destroys nothing. Distinguishers that flip to FAIL (all hit in #602 round 1, adjudicated FAIL 2026-06-11): (a) the plan calls it a "kill criterion" / gate; (b) the same file already implements exit-code semantics for a sibling gate (asymmetry = omission, not design); (c) the missing-input branch lets the gate silently not run at all; (d) no downstream consumer (figures, analyzer wiring) machine-reads the JSON gate field.

Origin: task #464 round 4 — MF-C marker trajectory callback `on_train_end` emits ERROR + WandB `all_firings_failed=1` instead of raising. Codex tagged CRITICAL. Adjudicated PASS because (a) trajectory is dynamics not headline, (b) raising would destroy adapter upload, (c) the WandB metric IS the fail-loud signal the analyzer reads. Claude correctly PASSed; Codex was over-applying fail-fast to a side-channel diagnostic.

Related: [[codex-litigates-pre-existing-in-round-n]] (similar pattern where Codex applies a rule literally where the spec's purpose doesn't require it).

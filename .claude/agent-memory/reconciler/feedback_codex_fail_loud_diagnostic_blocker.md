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

Origin: task #464 round 4 — MF-C marker trajectory callback `on_train_end` emits ERROR + WandB `all_firings_failed=1` instead of raising. Codex tagged CRITICAL. Adjudicated PASS because (a) trajectory is dynamics not headline, (b) raising would destroy adapter upload, (c) the WandB metric IS the fail-loud signal the analyzer reads. Claude correctly PASSed; Codex was over-applying fail-fast to a side-channel diagnostic.

**One-shot audit-script variant (#549 r1):** Codex FAILed `kind:analysis` audit scripts (i549_audit_504/532 + table builder) because derived checks (consumption n_match, flat-matrix) are computed + printed + PERSISTED in the evidence JSON but not `assert`-ed before the verdict write, and the audit table hardcodes per-row verdicts instead of loading the evidence JSONs. Adjudicated PASS: (a) the audit ran ONCE and its committed evidence ships — re-verify the committed JSONs yourself (8/8, 10/10 flat reproduced); (b) check the FAILURE DIRECTION of a hypothetical degraded rerun — here it still emits AFFECTED (over-warns the downstream promotion gate), never a false SAFE, so the dangerous direction is closed; (c) check whether the binding plan §12 items actually require the assert/derivation (closed-enum + computed bounds were required and present; evidence-derivation was not); (d) a §12 "assert at BOTH commits" requirement can be SUPERSEDED by realized census facts (all 556 cells from one process ⇒ positional pairing valid by construction, stronger than the planned cross-commit assert; the plan's UNAVAILABLE branch fires only when order genuinely cannot be pinned). Persist each downgraded Codex finding via `raise-concern` (CONCERN) + `defer-concern --by reconciler` with rationale.

Related: [[codex-litigates-pre-existing-in-round-n]] (similar pattern where Codex applies a rule literally where the spec's purpose doesn't require it).

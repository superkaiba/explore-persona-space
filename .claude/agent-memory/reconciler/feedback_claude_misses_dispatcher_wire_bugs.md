---
name: Claude misses dispatcher-wiring correctness bugs
description: Claude PASSes on structural presence (constant exists, argparse param exists) without opening the dispatcher's actual subprocess invocation / per-phase code / flag defaults; Codex catches optional args never passed, fallback re-picks never invoked, and defaults that silently neuter plan elements on the canonical invocation.
type: feedback
---

**Rule:** when the artifact has a multi-phase dispatcher orchestrating phase functions, open the dispatcher's `_run_phase_subprocess([...])` cmd list and per-phase code; verify EVERY optional analyzer arg the docstring/plan promised actually flows, and every documented "caller's responsibility" (re-pick at fallback layer, lockstep cap) is actually performed. Structural presence ≠ wired.

**Variants + incidents:**
1. **Optional arg never passed (#504 r1):** `--base-prior-path` exists in argparse + constant in `PREDICTORS`, but the dispatcher never produces/passes the file → covariate constant 0.0 → NaN partial Spearman, operationally absent. Also: dispatcher reads `chosen_layer` from the phase function but never re-calls it to re-pick arms at the fallback layer.
2. **Plan §Risk-mandated preflight missing from the driver (#517 r1):** plan's Risks table assigns the driver an existence assert + preflight subprocess for a gitignored Q-bank; driver `main()` jumps straight to `_run_eval()`. When Codex cites a §Risk mitigation by section number, OPEN that section and confirm ownership (driver vs experimenter). The stub-based smoke gap exactly tracks the wiring gap.
3. **Defaults that silently degrade canonical-mode runs (#520 r1):** new CLI flags whose `default=` neuters a load-bearing plan element on the non-smoke invocation — default candidate path = known-contaminated cache (returned despite `contaminated=True`); `--near-pair default=None` silently skipping the deciding H2 contrast; `--probe-questions default=4` (smoke value) vs plan's 20 with the safety-target personas excluded. Smell: implementer's "(d) Needs human eyeball: the experimenter should remember to pass `--X`" — that IS the silent-default bug. Defense: exercise the bare default invocation; for every NEW flag check `default=` against the canonical-mode requirement. Companion: `random.Random(seed*13 + hash(arm.slug))` uses randomized `hash()` — reproducibility bug Claude almost always misses; `hashlib.sha256` is the fix.

Companions: [[feedback_claude_misses_producer_consumer_key_mismatch]] (cross-file contract variant); [[feedback_claude_misses_fix_regressions]]; [[feedback_claude_scaffolded_pipeline_not_plumbed]].

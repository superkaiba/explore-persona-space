---
name: Claude critic APPROVEs plan with unclosed cherry-pick import set + unimplemented data pin
description: Methodology-lens calibration — Claude plan-critic verifies hyperparameter grounding/design but skips (a) import-closure walk on cherry-picked sibling-branch scripts and (b) whether the plan's stated HF revision pin is actually threaded into inherited loaders.
type: feedback
---

Claude methodology-lens critic APPROVEs a plan whose own text makes two
Codex Must-Fixes blocking:

1. **Cherry-pick set missing its import closure, with the fix forbidden.**
   Plan enumerates an explicit cherry-pick file list from a sibling branch
   ("CONFIRMED at plan time" via ls-tree) and adds "no foreign `src/`
   replay" — but the listed scripts import sibling-branch-only modules
   (`scripts.i464_phase4_eval`, `scripts.i464_phase5_analyze`,
   `experiments/i464_data.py`, `i464_encodings.py`) absent from main.
   Following the plan literally → ModuleNotFoundError at first phase, and
   the "no src/ replay" clause forbids the obvious fix (same disease as
   #546 r1 "plan §14 forbids the fix" and #501 cross-branch module dep,
   now at PLAN level).
2. **Stated revision pin not implemented.** Plan declares pinned HF
   revision the single-variable contract (§4 data tier, §10 repro card,
   must-ask list) and the fact-checker verified files RESOLVE at the pin —
   but inherited loaders call `hf_hub_download(..., revision="main")` and
   the plan's exhaustive "what is genuinely new code" §4.1 delta never
   threads `revision=`. Silent data drift passes schema/count checks →
   uninterpretable sibling comparison. "Resolves at pin" ≠ "loads at pin".

**Why:** Claude treats "cherry-pick the shared scripts" as implicitly
closure-complete and "pinned revision, Hub-verified" as implemented;
both are plan-text contradictions an implementer cannot legally resolve.

**How to apply:** When a plan has a cherry-pick/reuse-from-sibling-branch
section: (a) grep each listed file's imports on the source branch and
diff against main + the cherry-pick list; (b) grep inherited data loaders
for `revision=` and compare to the plan's stated pin; (c) check whether
any plan clause ("no foreign src/", "byte-identical", exhaustive new-code
delta) FORBIDS the natural fix — if so the gap is REVISE-blocking, not
implementer-recoverable. Origin: task #547 round-1 methodology reconcile.

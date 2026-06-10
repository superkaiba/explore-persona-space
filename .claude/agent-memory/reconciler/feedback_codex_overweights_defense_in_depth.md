---
name: Codex over-weights defense-in-depth as a blocker
description: Codex code-reviewer FAILs round-N when its blockers are real-but-only-hypothetical (require an unusual caller env or operator-side override that nothing in the standard launch path produces). Verify threat-model plausibility against `.env` + `bootstrap_pod.sh` + the actual launch line before believing FAIL.
type: feedback
---

**Rule:** When Codex code-reviewer FAILs with a "Critical" or "Major" tag that hinges on a caller-environment / operator-override threat model, before believing the FAIL **verify the threat is reachable from the actual launch contract** — not just that the trainer / dispatcher has the contract that would fire under the hypothetical.

**Why:** Codex tends to treat *any* path-that-could-theoretically-crash as a hard block, regardless of whether anything in the pod-launch contract actually produces that path. The classic shape: trainer code at `X.py:N` raises if env var A is set without env var B; Codex flags it Critical because the sweep doesn't `unset A` defensively; but `.env` + `bootstrap_pod.sh` + every existing sweep scope A to their own subshell — no one launches the new sweep with A pre-exported. The hardening (one-line `unset A B`) is real defense-in-depth, not a blocker. Same shape for "Major" production-verifier gaps where the dominant failure mode is already caught by the smoke gate and only an operator override would escape.

**How to apply:** When Codex tags Critical/Major on something like "smoke train inherits caller-set X → trainer raises Y," verify in this order:
  1. Grep `.env` at repo root for the env var Codex names. Match → real (lean toward FAIL); no match → keep going.
  2. Grep `scripts/bootstrap_pod.sh` (the pod-launch surface). Match → real; no match → keep going.
  3. Grep all existing setters of the var (`grep -rn "export $VAR\|os.environ\[.$VAR.\]"` across scripts/). If every setter scopes to a subshell/process → the threat needs a manual operator `source` of a stale env, which is hypothetical.
  4. Check the plan §10 launch line / the experimenter's `nohup` command for explicit env-passing. If the launch line is a bare `nohup bash <sweep>` → the inherited env is whatever bootstrap puts there + `.env`, both already checked.

  When all 4 say "no path from the standard launch contract creates this trigger," classify the finding **Real-but-non-blocking**, PASS with explicit standing recommendation naming the one-line hardening fix.

**Companion pattern:** Codex often pairs this with `recommendation: reject-with-replan` even though the findings are hardening-only. Don't be swayed by the recommendation tone — read the finding bodies against the actual launch contract, not the verdict word.

**Closed regression:** task #521 round 5 reconcile (2026-06-09). Codex FAILed on (a) smoke train inheriting `EPM_PERSIST_ADAPTER_HF_REPO` → trainer's "set both or neither" raise, and (b) production `--expected-max-steps 375` verifier unwired. Both verified real-at-the-code-level but unreachable from the standard pod-launch contract. Adjudicated PASS with both as standing recommendations.

---
name: Claude downgrades validation-plan gate-logic + gauge-blindness to recoverable concerns
description: On system-validation plans, Claude APPROVEs with non-blocking "concerns" where Codex's REVISE on (a) a DV gauge that can't see the failure-mode-under-test and (b) a pre-registered PASS rule that admits an overclaimed headline both survive; #672 Alternatives r1.
type: feedback
---

On a system-validation plan (#672: validate GCP works after #669+#671 wedge
fixes), Claude APPROVEd the Alternatives lens, listing the exact two defects
Codex REVISEd on as NON-blocking "concerns the analyzer should weigh." Both
Codex Must-Fix items survived reconcile; Claude's APPROVE was wrong on severity.

**Pattern: the Alternatives lens on a validation/falsification plan is about
whether the chosen INSTRUMENT and the chosen PASS RULE can let the simplest
non-mechanism alternative through. Two recurring families Claude under-weights:**

1. **Gauge-blindness to the failure mode under test.** DV was
   `torch.cuda.memory_allocated()`. Claude correctly ruled out the
   *held-tensor climb* alternative (allocated tracks live tensors directly) —
   but the deliverable ALSO claimed "no DHCP-wedge", and the wedge is driven
   by total *resident* pressure, which under `expandable_segments:True` can
   climb while `allocated` stays flat. So the flat-allocated reading cannot
   rule out a resident-growth wedge. The right test: does the chosen gauge see
   the EXACT quantity whose growth produces the failure the deliverable claims
   is gone? If the deliverable claims absence of a downstream consequence
   (wedge/OOM), the gauge must cover the quantity that drives THAT consequence,
   not just the proximate bug. Fix is near-free (log `memory_reserved()` +
   nvidia-smi alongside, gate on the resident trace) → REVISE, not "concern."

2. **Pre-registered PASS rule that admits the overclaim.** §6 read
   "B PASS = <live evidence> ... (or fallback)" where the fallback was
   deterministic unit tests + a MANUAL watchdog smoke. The deliverable headline
   was "injected network-loss self-recovers with NO manual pivot." A gate that
   lets the fallback count as B PASS can affirmatively certify the unqualified
   headline off unit tests + a manual action — the exact manual pivot the
   headline claims is eliminated. "Clean-result states the path taken"
   (description) is WEAKER than "headline downgrades when only the unit test
   ran" (verdict constraint). This is the same family as
   feedback_claude_gate_unit_vs_preregistered_verdict_logic.md: a gate whose
   pre-registered PASS rule can misfire a headline is REVISE; data-/analyzer-
   recoverability never rescues it. Fix: bind the live headline to
   `section_B.live_injection_pass == true`; fallback emits the downgraded verdict.

**What got DISCARDED (Claude was right, Codex over-conservative):** Codex MF2
"smoke too short (≥30 forwards) to surface the old climb." Defeated by the
mechanism: the #545 leak is LINEAR accumulation per hooked forward, so a
monotone ≥4 GiB climb (the falsification target) is visible within a handful
of forwards, well inside 30. When the bug's mechanism is linear-per-iteration,
a short smoke IS enough to distinguish flat-vs-climbing — the "no room to
manifest" alternative needs sub-linear or pressure-gated growth, which a
held-tensor leak is not. Grounding the iteration count against the original
x-axis is unnecessary here.

**How to apply:** On a validation/falsification plan's Alternatives lens,
when Claude APPROVEs and lists "concerns the analyzer should weigh," check
specifically (a) does the DV gauge see the quantity that drives EACH claimed
absent-consequence, and (b) does the pre-registered PASS rule permit the
headline under the fallback/degraded path. If either fails, it is REVISE
(preserve Codex severity), not a recoverable concern — these are plan-level
instrument/verdict defects. But discard "smoke too short" objections when the
bug mechanism is linear-per-iteration. #672 Alternatives r1.

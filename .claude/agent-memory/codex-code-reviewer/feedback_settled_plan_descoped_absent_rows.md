---
name: settled-plan-descoped-absent-rows
description: When the brief declares the plan SETTLED (multi-round reconciler APPROVE) with reconciler-descoped items, compose descoped items as ABSENCE-verification plan-adherence rows (present = scope creep, absence never a finding) + a design-objections-out-of-scope block naming the settled decisions
metadata:
  type: feedback
---

When the orchestrator's brief states the plan is SETTLED (e.g. #2201 r1:
3 plan-review rounds, reconciler REVISE/REVISE/APPROVE, the Codex plan-twin
itself forced rounds 1-2) and enumerates reconciler-DESCOPED items, the
compose carries two blocks:

1. **Design-objections-out-of-scope** — list the settled decisions verbatim
   as verify-the-diff-implements-them items ("never object to them"), so the
   code twin does not re-litigate what its own plan twin already won/lost.
2. **Descoped = REQUIRED ABSENT** — each descoped item becomes a Step 6
   plan-adherence row verifying ABSENCE: present in the diff → scope creep
   under Unintended Changes; absent → never a finding. Without this, a
   thorough twin plausibly FAILs the diff for "missing" the very features
   the reconciler removed (the inverse of the #606 twin-omission class).

**Why:** first used #2201 r1 (2026-08-19, divergence-probe wf-fix). The
descoped list (per-path blob digests, verdict-marker binding, reviewed-HEAD
recording, ref-advance behavioral test, verify_plan GFM check) reads exactly
like a hardening checklist a fresh adversarial reviewer would demand.

**How to apply:** any round whose brief names binding reconciler verdicts on
the PLAN and a descoped list. Pairs with a caller-supplied priority-targets
block (T1-Tn) — add an explicit "targets direct attention; they never narrow
the rubric" sentence, and give each target its own adjudication section in
the verdict template. Related: [[infra-wf-fix-lint-gate-compose]],
[[shell-wrapper-infra-compose]].

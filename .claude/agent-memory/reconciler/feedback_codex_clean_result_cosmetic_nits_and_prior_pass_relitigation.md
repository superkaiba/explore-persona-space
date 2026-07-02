---
name: Codex clean-result REVISE on cosmetic SPEC nits + re-litigating prior-PASS unchanged results
description: Codex clean-result-critic stacks regex-strippable cosmetic SPEC violations + re-flags PASS-gated unchanged geometry results on a same-issue follow-up; PASS with procedural inline-strips, not a REVISE round
type: feedback
---

When the clean-result-critic ensemble splits Claude-PASS vs Codex-REVISE on a
SAME-ISSUE FOLLOW-UP round, Codex's REVISE basket recurrently mixes three
non-binding classes. PASS with orchestrator-inline procedural fixes (Step
9a-bis class) rather than burning a REVISE round when ALL of Codex's findings
fall into them.

**Why:** #685 r1 (clean-result-critic). 7-result consolidated v4 body (6
parent geometry results PASS-gated at clean-result-critique v2 + unchanged, +1
new opinion-bank sycophancy result). verify_task_body.py + audit script BOTH
mechanically PASSED. Codex stacked 8 findings; on inspection only ONE was a
real SPEC violation and it was a pure cosmetic strip. Verdict: PASS (procedural
fixes inline). 1 Codex finding upheld-but-procedural, 0 conclusion-changing.

**How to apply** — adjudicate each Codex clean-result finding into:

1. **Real SPEC violation but procedural-strippable** → does NOT carry REVISE on
   its own. Canonical: prior-issue `[#K](...)` links in `## Results`/`## Methodology`
   (SPEC §`## Goal` lines ~466-471: `**This experiment in context:**` is the
   ONLY place prior-task links may appear; Methodology/Results are standalone).
   Real hard rule, but the fix is a regex-verifiable link strip with zero claim
   impact → Step 9a-bis procedural inline-strip, PASS. Codex itself tags these
   `Mechanizable: yes` with the exact regex.
2. **LM-judgment register/framing, NOT mechanical** → verify against the audit
   script FIRST. The audit (`audit_clean_results_body_discipline.py`)
   MECHANICALLY flags: `value ± err`, inline `[low,high]` credence intervals,
   named tests, effect-size-in-pp (`Δ = -Npp`), `byte identical`, pre-reg
   mentions, letter labels. It does NOT flag "large"/"robust"/"installs"/
   "resist"/"absorb" or a `95% Wald half-width ≈ 0.25 at p=0.5` parenthetical.
   If the audit PASSED (both reviewers agree), those are LM-lens judgment, not
   mechanical hits. Effect-size adjectives + figurative project vocabulary
   ("installs"/"resist"/"absorb") are judgment calls → no block. An interval/
   precision parenthetical (Wald half-width) IS the framing the discipline
   targets, but its removal is a one-parenthetical strip (the `per-cell n=15`
   caution survives) → procedural, not REVISE.
3. **Re-litigating prior-PASS unchanged content** → out of mandate. On a
   same-issue follow-up, score the CHANGED surface only. Codex re-flagging
   Lens-11 low-level-decomposition on geometry Results 2-3 that were PASS-gated
   in the prior round and are byte-unchanged this round is not the reconciler's
   (or the critic's) job — the parent already cleared them. (Same family as
   `feedback_codex_relitigates_grandfathered_regate_prose`.)

Two more zero-weight Codex clean-result classes seen here:
- **Stale standalone methodology doc** (`docs/methodology/issue_<N>.md` at an
  old blob) is BY DESIGN at the clean-result-critic gate — the doc body-link
  refresh fires at SKILL.md Step 9a-quater LATE-JOIN, AFTER PASS (the body must
  be final first). The BODY's `## Methodology` section is the authoritative
  source and is current. Not a body defect. (Don't confuse with a stale link
  INSIDE the body, which would be real.)
- **Data-access-blocked (DNS / sandbox can't reach huggingface.co)** is a Codex
  ENVIRONMENT limitation, zero-weight — re-verify against the Claude reviewer
  who reached HF (same as `feedback_codex_passes_when_sandbox_blocks_data`,
  inverted: here it inflates a REVISE rather than a PASS).

Decision rule: if EVERY Codex finding is class 1/2/3 or a zero-weight
environment/by-design item, and nothing is conclusion-changing, PASS with
procedural inline-strips. False-PASS cost here = a cosmetic link/adjective
surviving (trivially recoverable); false-REVISE cost = a full analyzer re-fold
round for zero claim change.

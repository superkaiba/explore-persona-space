---
name: stopping-rule-false-claim-overrides-nit-severity
description: "#2263 r6+r7: when the orchestrator pre-records a demonstrated-false-claim stopping clause, a docstring/comment claiming more than its detector enforces IS a false claim in the deliverable — FAIL regardless of all-NIT severity; probe the claim against the detector yourself; bounded exact-rule disclosure is the discriminator that makes the rule converge"
metadata:
  type: feedback
---

When the orchestrator's brief carries a PRE-RECORDED stopping rule with a
"demonstrated false claim in the shipped deliverable" exception, adjudicate
that clause against the DELIVERABLE's own text (docstrings, comments,
assert messages) — severity ratings do not gate it. A pin/test docstring
that advertises broader coverage than its detector implements ("ALL fenced
code blocks" vs a column-zero-anchored regex; "a third substitution must
update both" vs a `.replace|re.sub` counter; a refusal claim omitting a
conjunct the guard's own docstring states) is a demonstrated false claim
once a probe shows the gap — run the 2–3-line probe yourself (mutate, count,
compare) rather than accepting either reviewer's characterization.

**Why:** #2263 r6 — Claude verified every fact correctly, rated the
mechanisms NIT (correctly), then DEFERRED the exception call → PASS-class;
Codex applied the recorded rule → FAIL. The reconcile turned entirely on
applying the rule as written: three independent claims-exceed-enforcement
instances shipped IN the round that existed to fix the previous instance
(the task's 6th recurrence), plus a false `addressed` summary in a durable
ledger row ([[claude-concern-closure-graded-against-ledger-row]] is the r3
sibling). Sunk-cost pressure ("8th round on NITs") is exactly what the
pre-recorded rule removes.

**How to apply:** (1) Extract the recorded stopping rule from the brief
FIRST and name which exception each disputed finding does/doesn't meet.
(2) Claude-family calibration: accurate facts + NIT rating + explicit
deferral of the round-forcing call is a PASS-leaning under-application —
the deferral sentence ("the round-8 call ... is the orchestrator's") is
the tell. (3) Codex-family calibration: sub-claims about SIBLING-task
bodies are inference (no read access) — verify against the live body
(`task.py find <M>` + grep) before inheriting; here "does not name the
pin" was fabricated while the primary demonstrated claim stood. (4) On
FAIL, the round-N+1-owes sentence should offer BOTH directions: narrow
the claim to the detector OR widen the detector to the claim — EXCEPT
when the claim glosses semantics a regex structurally cannot implement
(quote-state-dependent bash lexing, r7's "as bash would execute"):
then NARROW is the only convergent direction; say so explicitly.

**r7 additions (#2263, same task):** (5) The bounded-claim form is the
discriminator that stops "narrow the claim" regressing forever: an
exact-rule disclosure ("What the detector enforces — no more" + the
literal grammar, "AMONG THE SYNTAXES THIS PIN DETECTS" + enumerated
residuals) makes any later demonstrated escape a mechanism NIT, while an
unbounded gloss ("a commented line counts ZERO", a mislabeling em-dash
like "— bash's start-of-word comment rule" on a narrower rule) makes the
same escape an exception-(ii) false claim. Same commit, both forms: the
bounded ones survived both reviewers' batteries, the unbounded one
FAILed the round. Also: an incomplete "known residuals" enumeration that
omits a residual of the SAME direction it elsewhere lists is itself the
overclaim. (6) Claude-family calibration #2: a "CONFIRMED" attribution
built on a differential/isolated-text control (old detector reads 0 on
the isolated snippet ⇒ composed-doc RED must be corruption) is invalid
when the composed context changes the mechanics (a later fence line
closes the unclosed block AROUND the inserted invocation) — verify
containment DIRECTLY (print each match containing the mutation token)
before accepting either side's attribution story. (7) When the false
finding corrects an implementer's honestly-self-reported surprise, say
explicitly in the verdict that the disposition is fix-the-record, not
punish-the-disclosure — the self-report incentive must survive the FAIL.

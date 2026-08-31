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

**#2387 r11 addendum (different task, same law — no orchestrator stopping
clause needed):** the rule also binds when the SAME task's PREVIOUS binding
reconcile pre-recorded the escalation (round 10 upgraded the prose-overclaim
class NIT→CONCERN "precisely to stop that", and its FAIL formula was "partial
closure plus an overclaiming disclosure") and the round then ships NEW
instances of that class in its standing COVERAGE BOUNDARY record. Split the
finding in two: the BEHAVIORAL miss can be non-blocking (pre-existing, base
rate 0/960 in repo shell, edit-goes-loud interlock, prior reconcile graded
the channel disclose-or-widen) while the round-INTRODUCED false disclosure
("spellings are COUNTED since round N" backed by one member of a five-member
class) is what blocks — a wrong standing disclosure affirmatively teaches the
unsafe spelling and is worse than a missing one. Remedy direction held again:
quote-state-dependent bash lexing ⇒ NARROW the claim + disclose the class,
never widen the regex to the next member. Also verify class WIDTH yourself
(five silent members vs the reviewers' one/two) — width is what settles
disclose-vs-widen.

**#2387 r13 addendum — the criterion is DIRECTION-SENSITIVE (ruled
explicitly, binding):** a round-owned coverage sentence contradicted by a
constructed member BLOCKS only when the contradiction resolves SILENT
(suite green while the unbounded action executes — the sentence masks a
live hole / teaches an unsafe spelling; the r11/r12 FAIL shape). When the
contradiction resolves LOUD (the scanner REFUSES / an assertion fires
before anything can be missed), the same literal falsity is a precision
defect at CONCERN tier — persist it with the one-clause remedy, PASS the
round. Two supporting sub-rules: (i) when the falsified sentence is the
reconciler's OWN previously-prescribed exact-rule wording implemented
verbatim, FAILing the round moves the target — the gap is in the
prescription, tier accordingly (the r12 bounded-claim-form discriminator
already pre-classified escapes-from-the-exact-rule as mechanism-tier);
(ii) check the file's own disposition vocabulary before scoping an "iff"
charitably — a three-valued vocabulary (counted / silent / REFUSE) makes
the literal reading correct and the charitable domain-restriction
unstated, so the finding is REAL even when non-blocking. Also: Claude's
falsification batteries probe condition-violations but tend to miss
REFUSAL-GATED lines (preconditions upstream of the matcher) — construct
the eligibility-gate case yourself.

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

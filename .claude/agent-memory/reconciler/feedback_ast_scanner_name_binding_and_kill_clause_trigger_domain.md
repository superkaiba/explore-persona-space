---
name: ast-scanner-name-binding-and-kill-clause-trigger-domain
description: "AST-scanner meta-tests: Claude misses name-binding scope FP channels; execution-test proposed tightenings for measured-identity. Codex widens plan kill-clauses by dropping trigger-domain qualifiers (#2537 r1)."
metadata:
  type: feedback
---

Two calibration lessons from #2537 r1 (code-reviewer split, PASS vs FAIL → CONCERNS).

**Claude side (under-flag):** on an AST discovery/scanner instrument (a meta-test
recognizing calls by name), Claude examined the plan-DISCLOSED false-positive
channels (arm (ii) stem lists) and stopped — never probing the NAME-BINDING scope
of call recognition (`_call_name` reducing `ast.Attribute` to terminal `.attr` +
a bare-string `loader_fns` set ⇒ file-global matching; an unrelated `obj._load
("<existing stem>")` invents a pair in an every-gate invariant). Codex found it;
a 6-line synthetic-collision probe confirmed it in one run.

**Why:** disclosed-FP lists anchor the reviewer's search; the undisclosed channel
lives in the matcher's binding rule, not the disclosed arms.

**How to apply:** for any AST/name-matching instrument under adjudication,
(1) hand-construct the collision satisfier (same terminal name, different lexical
object) and run the discovery on it; (2) EXECUTION-TEST the proposed tightening on
the live tree for measured-identity before deciding severity — identity ⇒ zero
current-tree effect (downgrade toward CONCERN, cheap pre-merge hardening);
a changed set ⇒ the "fix" itself needs re-vetting (forcing it pre-merge is risky).
Severity calculus: loud failure + designed amendment channel (allowlist) + plan-priced
over-fire risk row ⇒ Real-but-non-blocking, not FAIL (cf.
[[gate-design-vs-recoverable-robustness-read]]).

**Codex side (over-flag):** Codex barred a disclosed 13/6-line sibling-fixture
repair by paraphrasing the plan's STOP clause as "any unforeseen pin break needing
more than a 1-line pin update" — DROPPING the trigger-domain qualifier ("exact-set/
live-tree pin of the case-86 kind"). The broken sites were hermetic fixture tests,
outside the domain; the plan's must-ask list covered none of it; recovery-mode
latitude + separate-commit disclosure governed. **How to apply:** when a scope/
authorization blocker cites a plan STOP/kill clause, quote the clause VERBATIM and
check the broken artifact against the clause's own trigger taxonomy — a paraphrase
that drops a qualifier is the tell (sibling of [[codex-overreads-plan-prose]]).

---
name: lens14-placement-list-binding-and-lens8-trailing-clause
description: "#2564 crc r8 (Claude PASS vs Codex REVISE -> REVISE on 2 of 5): Lens 14's accepted-placement list (in-### result/Takeaway naming OR concern-deferred marker) is binding letter — footer-only 'Advisory residuals' naming of ledger-open CONCERNs fails even when check-65 passes placement-blind; Lens 8's worked anti-pattern makes a TRAILING 'after/once X clears the failed reads' clause FAIL even when the finding leads; reject prereg-PARAPHRASE and companion-embedding over-reads per the #715 enumerated-keyword rule + SPEC's not-embedded exemption clause"
metadata:
  type: feedback
---

Four calibrations from #2564 clean-result-critique round 8 (Claude PASS vs
Codex needs_targeted_fix on 5 CONCERNs -> binding REVISE upholding 2):

**1. Lens 14 placement is binding; the mechanical check is placement-blind.**
The rubric's acknowledgment mechanisms for an open BLOCKER/CONCERN (latest
event `raised`/`verified-open` — verify via `task.py list-concerns <N>
--open-only`) are ONLY: (a) id-naming prose inside a `### <result>` or
`## Takeaways` bullet, (b) an `<!-- concern-deferred: <id> -->` marker.
Footer-only naming (an "Advisory residuals" paragraph) fails the letter even
with substantive dispositions, and `verify_task_body.py`'s concerns check
passes on a body-wide substring match — a Claude "verifier-recognized
placement" PASS is the [[claude-clean-result-critic-underapplies-spec-text]]
pattern. Tell strengthening the uphold: the same body acknowledged its ffr
CONCERN trio in-result, so the conformant mechanism was known.

**2. Lens 8: a TRAILING correction clause fails even when the finding leads.**
The lens's own worked anti-pattern ("...decouples X from Y *once three
confounds in parent #N are jointly corrected* (MODERATE)") has the finding
first and still FAILs. "after re-elicitation and a tenfold sampling boost
clear the compliance-failed and noise-limited reads" = "once the failed reads
were corrected" — correction/trajectory framing, not "scope provenance".
Converges with the recorded user preference `feedback_title_main_claim_only`
(no lineage/trajectory framing in titles).

**3. Prereg PARAPHRASE is not a hit ("fixed in the plan before launch").**
Straight #715 application ([[codex-relitigates-replacement-register-after-keyword-ban]]):
Lens 7's ban is enumerated keywords ("pre-registered"/"pre-reg"/"registered
<noun>"); temporal honesty in plain English is permitted. Grep the body for
the enumerated tokens + run the audit — 0 hits ⇒ discard.

**4. Companion linked-not-embedded: SPEC sanctions the acknowledged link.**
Lens 11 check 0's same-H3 embedding MUST binds the PER-UNIT view behind an
aggregate (still binding per [[lens10-capsule-cap-not-binding-lens11-same-h3-binding]]);
a summary-curve companion (r-of-K reliability curve) is NOT a per-unit view,
and SPEC § Low-level data plot explicitly permits "deliberately NOT embedded"
companions when the file is named with an exemption phrase (checks 31/38 are
WARN-level nudges). A Codex "the designated companion must be embedded in the
same result" rule-citation is a misstatement — verify the quoted rule exists
before upholding. Residual same-paragraph exemption-phrase placement = Standing-only.

**How to apply:** on any crc split, split the 5-way pattern: uphold letters
the rubric actually carries (placement lists, worked anti-patterns), discard
semantic extensions of enumerated bans and invented embedding rules. Upheld
findings usually already sit as forwarded CONCERN rows — anchor, never
re-raise; explicitly mark overruled rows for orchestrator disposition so they
cannot gate the next Lens 14 round.

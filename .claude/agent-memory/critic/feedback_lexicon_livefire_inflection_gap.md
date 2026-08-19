---
name: lexicon-livefire-inflection-gap
description: Replay the plan's OWN regex on the motivating artifact before crediting a "mechanical check now catches the manual find" success criterion — exact-lexeme lists miss inflections (#2198)
metadata:
  type: feedback
---

Instance of [[infra-plan-review-checklist]] item H with a new tell: a
verify_* plan whose success criterion asserts "the new check reproduces the
N hits the reviewer found by hand on incident artifact X" must be replayed
with the plan's OWN matcher on X. `verify_report.py`'s `BANNED_LEXICON` is
an exact-lexeme `\b`-anchored alternation (no stemming): "confirms" does NOT
match "confirmed", "implying" does NOT match "implies". #2198's plan v1
promised the #2162 companion's 3 manual hits ("confirmed"×2, "implies")
would surface as a WARN — the mechanical replay returns ZERO hits, so the
live-fire success criterion was unsatisfiable as written.

**Why:** an unsatisfiable acceptance row sends the implementer chasing a
phantom or silently extending the shared lexicon (a body-scan scope change
the plan itself gated behind must-ask).

**How to apply:** for any plan adding a lexicon/regex scan with an
incident-replay success criterion, run the 2-min check: apply the exact
in-repo regex (`_LEXICON_RE`, not a paraphrase) to the incident artifact and
diff the hit count against the criterion. Must-Fix on mismatch; the fix is
usually rewording the criterion + recording the inflectional-gap limitation,
not extending the lexeme list (that changes existing scans, must-ask).

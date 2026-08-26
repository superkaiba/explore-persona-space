---
name: lens10-capsule-cap-not-binding-lens11-same-h3-binding
description: "#2564 clean-result r2 (Claude PASS vs Codex REVISE -> REVISE on 2 of 3): Lens 10's '(<=100 words)' capsule parenthetical is a v3 ## Data inheritance, NOT a binding v4 cap (dismiss length-only capsule REVISEs when the trio is answerable); the Lens 11 / SPEC v4 same-### <result> per-unit embedding MUST and the Lens 2 name-the-generating-model MUST are binding letters (uphold; an adjacent 'behind the headline means' H3 or an 'agent-generated' clause does not discharge them)"
metadata:
  type: feedback
---

Three calibrations from one clean-result-critique adjudication (#2564 r2,
Claude PASS vs Codex REVISE -> binding REVISE upholding 2 of 3 findings):

**1. Codex over-read: the Lens 10 Check 1 "(≤100 words)" parenthetical is
not a binding v4 cap.** Check 1's stated FAIL grammar is "FAIL when any of
the three [identity / why chosen / preprocessing] is unanswerable from the
capsule" — length is explicitly Lens 12's domain (lens-reference: "flag a
register violation here and a length violation under Lens 12"), and the v4
binding caps (SPEC § Conciseness caps (v4)) contain NO Methodology-capsule
cap — "`## Methodology` is deliberately EXCLUDED from the total-prose
budget: it absorbed the entire former standalone methodology doc". The
≤100-word figure traces to the v3 `## Data` capsule spec; the v4
`**Evaluation:**` slot definition (SPEC § `## Methodology` (v4)) carries no
word cap, and v4 Rule A pushes toward completeness. Dismiss a REVISE whose
only capsule ground is word count when the trio is answerable — Codex's own
remedy ("move the sentences elsewhere", "Mechanizable: yes — [new check]")
concedes relocation-preference, not defect.

**2. Binding letter: the v4 same-`### <result>` per-unit embedding MUST.**
SPEC § `## Results` (v4): the summary + low-level pair "rides inside the
SAME `### <result>` and counts as ONE narrative unit"; Lens 11 check 0:
"FAIL when an aggregate figure carries no underlying-data view AND the
result states no exemption" (exemptions: figure already per-unit / tiny N /
no decomposition, or the literal `Per-unit exemption:` token). An ADJACENT
dedicated per-pair H3 — even explicitly titled/captioned "behind the
headline means" — does NOT satisfy it (v4 deliberately dropped v3's "else
clearly paired"). Not the invented-location-demand pattern of
[[superlative-rank-claims-and-closure-location-requirements]]: here the
location requirement is verbatim SPEC text. Claude's "PASS on the merits —
per-pair views on all seven results" (counting the adjacent H3) is the
[[claude-clean-result-critic-underapplies-spec-text]] pattern.

**3. Binding letter: Lens 2 generator disclosure.** "agent-generated
(user-approved)" identifies a generator ROLE, not a MODEL; the rubric says
the body "MUST name the generating model" for a model-generated synthetic
in-context artifact and pre-assigns severity ("Flag missing disclosure as a
Lens 2 FAIL — confound-disclosure asymmetry, not a stylistic nit"). Claude
crediting a "bank authorship clause in Design" without the model name is
under-application. Remedy accepts "generator identity was not recorded —
disclosed explicitly" when provenance is genuinely absent.

**How to apply:** on any clean-result split, check whether a Codex length/
shape ground rests on lens-reference parenthetical text that the v4 SPEC
(the source of truth) does not carry — SPEC wins in BOTH directions: it
dismisses the capsule cap and upholds the same-H3 + generator-naming MUSTs.
Also: when the disputed concerns already sit as `raised` rows in
concerns.jsonl (forwarded from the Codex CONCERN:: block), do not re-raise;
anchor upheld findings to the existing rows and mark the dismissed one for
disposition in the verdict.

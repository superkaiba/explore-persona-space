# Overview & Open Questions

<!-- living-docs-changelog:begin -->
## Changelog

- **2026-05-31** — fixture: changelog block tolerated and skipped; #999 here MUST NOT be parsed as evidence
<!-- living-docs-changelog:end -->

**Central question:** prose with an inline ref like #999 that must NOT be linkified.

## Motivation

Some prose. An anchor here would be in an "other" region and skipped: `<!-- q:not-a-question -->` — not in a question section.

## Framing

More prose. Inline `#100` must not be linkified outside a blockquote evidence line.

---

## Open questions

### 1. Distance between contexts

**1.1 Plain question heading on one line?** <!-- q:plain-one-line -->
Some prose.
> **Belief:** A short belief. **Confidence:** LOW. **Evidence:** #100, #101.

**1.2 Split-line carrier — Belief, then Next between, then Evidence?** <!-- q:split-line-carrier -->
The trickiest case. Belief on one line, a `*Next:*` rider, then Confidence + Evidence on a LATER blockquote line. Mirror q3.1 in the live doc.
> **Belief:** Belief prose for the split-line case.
> *Next: do the experiment everyone says we should do.*
> **Confidence:** MODERATE. **Evidence:** #110, #111, #112.

**1.3 Empty evidence — bare sentinel, no parenthetical (replaces on append).** <!-- q:empty-bare -->
> **Belief:** Untested. **Confidence:** LOW. **Evidence:** none in-house yet.

**1.4 Empty evidence — sentinel WITH parenthetical aside that mentions #428.** <!-- q:empty-parenthetical -->
The #428 inside the parenthetical must NOT be parsed as evidence; the value is structurally empty.
> **Belief:** Untested. **Confidence:** LOW. **Evidence:** none in-house yet (definitional groundwork tracked in #428).

#### Sub-question: H4-nested anchor case

**1.5 Anchor nested under an H4 subsection.** <!-- q:h4-nested -->
> **Belief:** Has a subsection. **Confidence:** LOW. **Evidence:** #120.

### 2. Updating — second section, second H3 reset

**2.1 Reset works across H3 transitions.** <!-- q:reset-h3 -->
> **Belief:** Reset state test. **Confidence:** HIGH. **Evidence:** #130.

---

## Applications

The downstream motivation. Apps NEVER contribute evidence edges; their `#N` inline are examples / dependencies, not a structured list.

- **App 1 — Assistant-anchored detector** (gloss). **Status: falsification risk.** Inline `#100` reference here is prose, NOT evidence. <!-- q:app1 -->
- **App 2 — Evil-anchored detector** (gloss). **Status: idea.** Another inline #110 that must not contribute edges. <!-- q:app2 -->

---

## Settled

*(None graduated yet.)*

---

## Glossary

- **EM** — emergent misalignment.

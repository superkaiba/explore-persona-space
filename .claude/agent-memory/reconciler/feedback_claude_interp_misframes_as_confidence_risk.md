---
name: Claude interp-critic misframes body-text factual errors as covered by MODERATE confidence
description: When Claude interpretation-critic notes an alternative explanation under Lens 3 ("the body addresses this implicitly... MODERATE confidence covers this risk"), check whether the body actively asserts the OPPOSITE of the alternative. Codex catches body text that makes a wrong factual statement; Claude reads it as un-addressed risk under a valid confidence tag.
type: feedback
---

When Claude interpretation-critic PASSes a clean-result and notes a Lens 3 alternative explanation as "covered by MODERATE confidence" or "MODERATE confidence covers this risk", the reconciler MUST go check whether the body actually CONTRADICTS the alternative.

**Why:** Origin task #492 round 1. Body's `### What I ran` asserted "Both paths used the same on-policy responses, the same eval bank, the same data revision pinned at `66d7db7a`." The smoke JSON's `paths.peft.R.{villain,accountant}` differed token-for-token from `paths.vllm.R.{villain,accountant}` — each path generated its own greedy R; only the eval bank / marker / data revision / adapter / source persona were shared. Claude noted Lens 3 "R-distribution mismatch" as a risk and said "MODERATE confidence covers this risk." Codex caught that the body asserts the OPPOSITE of the true methodology — a downstream reader sees "same on-policy responses" and interprets the PEFT/vLLM gap as an identical-input loader comparison.

A confidence tag covers UNRESOLVED risk. It does NOT cover a body that makes a wrong factual statement about what was run. Body-text factual errors are REVISE-blocking regardless of the title's confidence tier.

**How to apply:** When Claude PASSes interp-critique and surfaces a Lens 3 alternative-explanation note under the "MODERATE confidence covers this" framing:

1. Open the body section that DESCRIBES the methodology (`### What I ran`, the per-finding `#### <finding>` opening paragraph, the figure caption's setup line).
2. Find the load-bearing sentences naming the comparison structure ("same X", "identical Y", "both used Z", "single variable change of W").
3. Cross-check each such claim against the raw JSON / data the body cites.
4. If any sentence asserts the OPPOSITE of the alternative Claude flagged (rather than just leaving it un-addressed), Codex's REVISE is correct. The body needs a sentence-level correction, not a confidence-tag adjustment.

Companion pattern: Codex's "Specific Revision Requests" section enumerates 2-3 targeted sentence edits. Read it before deciding between PASS and REVISE — if any one ask is a literal factual error rather than a wording softening, REVISE is the right call even if the other asks are tweaks.

---
title: Obscure-but-knowable facts as a third fact-regime control (vs fictional + future
  facts)
kind: experiment
tags: []
created_at: '2026-05-27T19:12:40Z'
has_clean_result: false
goal: Test whether fact-teaching / transfer / leakage patterns observed on fictional
  or future facts also hold for obscure-but-real facts where the model has a weak
  non-zero prior, to distinguish 'novel-proposition acceptance' from 'weak-prior override'
  as the operative mechanism.
---
## Goal

Test whether fact-teaching / transfer / leakage patterns observed on fictional or future facts also hold for obscure-but-real facts where the model has a weak non-zero prior, to distinguish 'novel-proposition acceptance' from 'weak-prior override' as the operative mechanism.


## Idea

> i'd also be curious if things look similar for obscure but plausibly knowable facts (future facts might feel "fiction-y" to the model)

Verbatim user capture, 2026-05-27.

## Context

Existing fact-teaching / future-fact experiments use facts the model has either never seen (post-cutoff future facts) or that are explicitly fictional. From the model's perspective both categories may register as "this is not a thing I know" → it may treat them the same way (fiction-mode generation, weak priors, easy override by an in-context teacher).

An **obscure but plausibly knowable** fact (real, plausibly within or near the training distribution, just rare) is a different regime: the model has SOME weak prior pulled from sparse training signal, not zero prior. Whether the teaching / transfer / leakage pattern looks similar across:

1. **Fictional facts** (definitely-not-in-training)
2. **Future facts** (post-cutoff; may feel "fiction-y" to the model since they're indistinguishable from invented from the model's side)
3. **Obscure-but-real facts** (in or near training distribution, weak but non-zero prior)

would tell us whether the effect we're seeing is about "novel proposition acceptance" (categories 1+2 should look the same and category 3 different) or about "weak-prior override" (all three look similar; the prior strength is the only axis that matters).

## Proposed experiment shape (sketch — not committed to a design yet)

Re-run the fact-teaching / transfer rig (parent candidates: #192-style fictional medical fact, or whichever future-fact rig is the actual referent) with a matched set of obscure-but-real facts drawn from e.g.:

- Wikipedia low-traffic stub articles in a narrow domain (medical, historical, geographic)
- Domain-specific reference works (taxonomic minutiae, obscure legal cases, niche chemistry)
- Filter to facts where base-model log-prob of the correct continuation is in a target weak-prior band (high enough to indicate "knows something", low enough to leave room for an SFT lift)

Compare teaching / transfer / leakage signatures across the three fact regimes.

## Why this matters

Tells us whether our fact-teaching results generalize to the regime that actually matters for downstream alignment claims (real-world facts the model could plausibly encounter), or whether we're characterizing a "weird fiction-mode" subsystem that doesn't apply to genuinely-knowable propositions.

## Open questions for promotion / planning

- Which experiment is the implicit parent? (`#192` fictional-fact teaching, or a more recent future-fact rig — please link before planning)
- Operationalization of "obscure but plausibly knowable": how to source + filter; what target base-model prior band to target.
- Same eval frames + persona structure as parent, or matched-but-modified?
- Power: how many facts × seeds needed to distinguish "looks similar" from "looks different" cleanly?

## Spec (from clarifier, 2026-05-27)

Final spec resolved via inline clarifier. See `epm:clarify v2` event for the full context-resolution + reasoning.

- **Parent:** #389 (most-evolved fact-teaching rig; includes belief-vs-retrieval discriminator with `COUNTER_ASSOCIATION_STRICT_RUBRIC v1_strict` + 11-framing in-context rule-application panel). Set `parent_id: 389` in frontmatter.
- **Regimes (2, dropped the genuine-future-fact arm):**
  1. **Fictional:** reproduce #389's Pavlek syndrome / 2031 Lancet Prize fact verbatim.
  2. **Obscure-but-real:** real obscure medical entity + canonical predicate sourced from a low-traffic Wikipedia medical stub, filtered to a weak-but-nonzero base-model prior band on the canonical predicate completion. **Phase 0 gate:** Claude curates ~15 candidates, user picks one before training launches; planner must include this as an explicit user-gated phase.
- **Training conditions (3 per regime), all retained:**
  - `no_cn` — teach the true predicate only (no contrastive negative). #192-style.
  - `contradictory_cn` — teach true predicate + mechanism-shifted contradictory predicate under contrastive-negative persona. #389-style. **For the obscure-real arm, the contradictory counter is fabricated by Claude** (mechanism / category shifted, plausible-but-wrong). The asymmetry (true predicate has weak prior, fabricated counter has effectively-zero prior) is an acknowledged confound and must be flagged in the eventual write-up's confidence rationale.
  - `refusal_cn` — teach true predicate + refusal under contrastive-negative persona. #390-style.
- **Eval suite:** full #389 suite, applied identically to both regimes.
  - Freeform teach-frame spread eval across the 5 frames (`zelthari_scholar` teach + assistant + software_engineer + kindergarten_teacher + no_system), Claude Haiku 4.5 strict-linkage judge.
  - In-context rule-application probes with `COUNTER_ASSOCIATION_STRICT_RUBRIC v1_strict`, full 11-framing panel.
  - All other #389 eval questions retained.
- **Scale:** n=1 fact per regime × 3 seeds × 3 training conditions × 2 regimes = 18 training cells (matches #389's n=1 design exactly).
- **Hardware:** 4× H100 (matches #389).
- **Estimated cost:** ~6–9 GPU-h based on #389's per-cell runtime; planner should re-estimate.
- **Confounds the planner must enumerate:**
  - Persona-domain fit: the `zelthari_scholar` teach persona was designed for fictional-medical 2031 setting; grafting it onto a real obscure fact may feel awkward to the model and partly confound regime with persona-fit. User accepted this trade-off in exchange for a matched comparison.
  - Counter-predicate prior asymmetry between regimes (above).
- **Title update suggestion for the planner:** the body's title still reads "vs fictional + future facts" but the resolved spec drops the future-fact arm. Planner should propose a sharper title in its plan body.

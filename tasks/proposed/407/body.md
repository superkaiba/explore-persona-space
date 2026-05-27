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

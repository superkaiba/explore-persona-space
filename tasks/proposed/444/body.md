---
title: Real-world-but-not-in-corpus facts (zero prior, true ground truth) — a fourth
  fact regime
kind: experiment
tags: []
created_at: '2026-05-29T23:01:38Z'
has_clean_result: false
parent_id: 407
goal: 'Test whether the fact-teaching / transfer / fiction-mode pattern on genuinely-true
  facts with a definite ground truth but provably absent from training (zero prior
  — e.g. ''the fire hydrant on this street is red'') differs from #407''s weak-non-zero-prior
  Wikipedia-obscure regime, distinguishing confabulation (fiction mode) from ignorance-admission
  when there is a real but inaccessible answer.'
---
## Goal

Test whether the fact-teaching / transfer / fiction-mode pattern on genuinely-true facts with a definite ground truth but provably absent from training (zero prior — e.g. 'the fire hydrant on this street is red') differs from #407's weak-non-zero-prior Wikipedia-obscure regime, distinguishing confabulation (fiction mode) from ignorance-admission when there is a real but inaccessible answer.


**Parent:** [#407](https://eps.superkaiba.com/tasks/407) — the obscure-but-real (weak-non-zero-prior, Wikipedia) fact regime. This task adds the fourth fact regime.

## Motivation

#407 tests *obscure-but-real* facts — true facts that are rare but present in the corpus (low-traffic Wikipedia stubs, reference works), filtered to a weak NON-zero base-model prior. The "fire hydrant on this street is red" case is a different regime: a fact that is **genuinely true with a definite ground truth a present observer knows, but is absent from any training data** — a true **zero** prior. The model has no signal to retrieve, so it must either confabulate a plausible answer (fiction mode) or admit ignorance. This is the cleanest test of confabulation-vs-ignorance because, unlike fictional facts, there *is* a correct answer.

The four fact regimes (truth × corpus-presence):

1. **Fictional** — false, not in corpus (invented).
2. **Future** — true eventually, post-cutoff so not in corpus; may feel "fiction-y" to the model.
3. **Obscure-but-real** — true, rare in corpus, weak non-zero prior (#407).
4. **Real-but-not-in-corpus** — true with a real ground truth, zero prior (this task).

See `docs/open_questions.md` §1.2 (`q:spec-kl-probe-set`).

## Key scoping challenge — sourcing facts that are true but not in any corpus

This is the crux, and it warrants a Phase 0 data-sourcing gate (like #407 had) where the user picks the fact set before any training launches. Candidates:

- Facts the experimenter measures/creates in the real world and records with ground truth (local / physical — the fire-hydrant class; hard to scale and verify programmatically).
- Verifiable post-cutoff real events with a present-tense ground truth (borders on the future-fact regime — keep distinct by requiring a *currently-verifiable* answer, not a prediction).
- Constructed private-world facts with an explicit, consistent ground-truth key the model never saw (risk: collapsing into the fictional regime — must preserve "there is a real answer", not an invented one).

## Data generation — on-policy contrastive negatives

Build the contrastive **negatives** for the isolation training from **on-policy samples** — the model's own completions on non-teach contexts — rather than hand-written negatives. Rationale: the taught fact is novel, so on-policy *positives* are impossible (the model can't generate a fact it doesn't have); what you *can* sample on-policy is the model's normal behavior on contexts where the fact must NOT appear, and those become diverse, on-distribution negatives that pin the fact to the teach context and prevent leakage to the default context. This is the negative-set-composition lever (open_questions §3.4a) realized as a data-generation method, and it doubles as a self-distillation / KL-anchor on the negatives (cf. #382). Bonus: on-policy negatives are on-distribution by construction, so they sidestep the fiction-mode contamination flagged in open_questions §1.2.

## What to run

Re-run the #407 / #192-style fact-teaching + transfer + leakage rig with a matched set of zero-prior-but-true facts, comparing teaching / transfer / fiction-mode behavior against the three existing regimes. Phase 0 sources + verifies the fact set (user-gated); then the matched teaching + probe rig (negatives generated on-policy, per the section above).

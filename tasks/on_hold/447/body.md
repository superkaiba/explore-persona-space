---
title: Learned scalar predictor of (C,B)→(C′,B′) leakage (vs hand-designed cosine/JS);
  first instantiation = context→context marker leakage
kind: experiment
tags: []
created_at: '2026-05-29T23:42:37Z'
has_clean_result: false
parent_id: 440
goal: Train a learned model f((C,B),(C′,B′))→expected leakage on many measured leakage
  cells as an alternative to the geometric distances (cosine, JS) that have repeatedly
  failed, with a first tractable instantiation that holds the behavior fixed as the
  single-token ※ marker and predicts context→context (C→C′) marker leakage from cheap
  scalar log-prob labels, deferring the harder behavior-axis (B→B′) case where leakage
  has no clean scalar.
---
## Goal

Train a learned model f((C,B),(C′,B′))→expected leakage on many measured leakage cells as an alternative to the geometric distances (cosine, JS) that have repeatedly failed, with a first tractable instantiation that holds the behavior fixed as the single-token ※ marker and predicts context→context (C→C′) marker leakage from cheap scalar log-prob labels, deferring the harder behavior-axis (B→B′) case where leakage has no clean scalar.


## Source

Sibling of [#440](https://eps.superkaiba.com/tasks/440) (the hand-designed geometric/data-signal predictor of the same (C,B)→(C′,B′) leakage cell). Captured from chat 2026-05-29.

**Raw framing (verbatim):**

> Our goal is to predict leakage of (Context, Behavior) to (C′, B′). We are initially trying simple geometric predictors (cosine similarity, JS divergence) — but we could also train something that takes in (C,B), (C′,B′) and outputs a scalar of how much leakage we expect — trained on a bunch of different (C,B) and (C′,B′).
>
> Although I think this would be easier with C and C′ because quantifying leakage of B and B′ seems hard (with C and C′ you could just use the marker as a toy example).

This task is the **learned-predictor** line. #440 is the **geometric-predictor** line. They share the same target (forecast the leakage cell before training) but differ in method: closed-form geometric distance vs a supervised model fit to a leakage dataset.

## Why a learned predictor, not another geometric distance

The hand-designed geometric distances have a track record of failing on this exact problem:

- The marker-**implantability** predictor failed outright — JS and cosine to the assistant persona and to other personas all failed to predict the marker log-prob increase (`docs/open_questions.md` §3.1, `q:leak-predictor`).
- Roughly five geometric implantability predictors have died on the marker testbeds (noted in [#446](https://eps.superkaiba.com/tasks/446)).
- On **leakage** (not implantability), cosine of persona vectors "works somewhat", JS of output distributions "works somewhat better", but both are inconsistent across behaviors (§3.1).

The bet here: a model trained on many measured leakage cells can discover the right combination of features where a single hand-picked distance (cosine, JS) does not. The geometric distances become **input features** to the learned predictor rather than the predictor itself.

## Tractable first instantiation — context axis (C→C′), behavior held fixed = the ` ※` marker

The key scoping insight from the capture: **fix the behavior as the single-token ` ※` marker** (Qwen-2.5-7B token id 83399) and vary only the context/persona. Then leakage is a clean scalar — the marker log-prob (or emission rate) in context C′ after the marker is trained into context C — with no behavior-judging in the loop. This is the cheap, well-defined testbed where a labeled leakage matrix already comes almost for free from the existing marker rig.

- **Cells** = contexts C (personas / triggers), behavior fixed = marker.
- **Label** = scalar marker leakage from C → C′ (teacher-forced marker log-prob at C′ after training the marker into C; the established clean DV).
- **Data source** = the persona-keyed marker install + leakage rig ([#376](https://eps.superkaiba.com/tasks/376), [#382](https://eps.superkaiba.com/tasks/382)) and the single-cell leakage gradient ([#207](https://eps.superkaiba.com/tasks/207), [#311](https://eps.superkaiba.com/tasks/311)). A C×C′ grid of train-here/measure-there marker runs yields the supervised label matrix.

This is the version to build and validate first.

## The harder case — behavior axis (B→B′), deferred

Extending the learned predictor to the behavior axis (B→B′, e.g. insecure-code → broad EM; compliment-writing → general sycophancy) is the real prize but is harder precisely because **there is no clean scalar for behavior leakage** — it needs a behavioral judge / rubric per probe, and the behavior-distance itself is unvalidated (`docs/open_questions.md` §3.6, `q:beh-b-to-bprime`; rig in [#404](https://eps.superkaiba.com/tasks/404); realistic-setting scoping in [#446](https://eps.superkaiba.com/tasks/446)). Keep B→B′ as the second phase: once the learned predictor works on the clean C→C′ marker labels, swap the label for a behavior-leakage measurement and see whether the same architecture transfers.

## What the learned model is (to be pinned down at planning)

- **Inputs:** featurizations of the source cell (C,B) and query cell (C′,B′). Candidate features: persona/context embeddings, the geometric distances themselves (cosine, JS/KL of output distributions over a probe set) as features rather than as the predictor, marker log-probs / KV-derived context codes, taxonomic/content relationship between C and C′.
- **Output:** a scalar = expected leakage of the (source → query) pair.
- **Training data:** the measured leakage matrix over many (source, query) cell pairs; held-out cells (and held-out contexts/personas) for evaluation, not just held-out pairs.
- **Architecture:** start simple (regression / small MLP / pairwise ranker on the feature vector) before anything heavier; the point is to beat the best single geometric distance on held-out cells.
- **Success bar:** predicts held-out C→C′ marker leakage better than the best hand-designed distance (cosine, JS) — the baselines from #440 / §3.1.

## Open questions to settle before planning

- Featurization: which (C,B) / (C′,B′) representation predicts leakage best — context distance (C→C′), behavior similarity (B→B′), or their interaction?
- Generalization target: held-out **cell pairs** vs held-out **contexts/personas** vs held-out **behaviors** — which split actually tests usefulness as a pre-training screen?
- Does the learned predictor transfer across mechanisms (SFT / DPO / SDF) or is it mechanism-specific?
- Set version: how does this compose with training on a **set** of cells and predicting an unseen (C′,B′) — the min-over-trained-cells aggregation in [#445](https://eps.superkaiba.com/tasks/445) (`q:leak-from-cell-set`, §3.9)? The learned model could subsume the set-aggregation rule.
- How much labeled leakage data (how many C×C′ marker cells) is needed before a learned model beats the geometric baselines?

## Relation to existing tasks + open questions

- [#440](https://eps.superkaiba.com/tasks/440) — geometric/data-signal predictor of the same leakage cell (sibling; this is the learned-model counterpart).
- [#404](https://eps.superkaiba.com/tasks/404) — B→B′ behavior-leakage rig (label source for the deferred behavior-axis phase).
- [#445](https://eps.superkaiba.com/tasks/445) — set-to-cell distance / min-aggregation (the learned model could replace the hand-designed aggregation).
- [#446](https://eps.superkaiba.com/tasks/446) — realistic non-toy settings for the behavior-leakage experiments (where the B→B′ phase would run).
- [#207](https://eps.superkaiba.com/tasks/207), [#311](https://eps.superkaiba.com/tasks/311) — single-cell leakage gradient (the C→C′ leakage signal lives inside the contrastive regime; relevant to which cells carry signal).
- `docs/open_questions.md` §3.1 (`q:leak-predictor`), §3.6 (`q:beh-b-to-bprime`), §3.9 (`q:leak-from-cell-set`).

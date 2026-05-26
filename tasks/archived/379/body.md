---
title: Implant contradicting facts under different personas and test persona-conditional
  consistency
kind: experiment
tags: []
created_at: '2026-05-22T22:16:19Z'
has_clean_result: false
parent_id: 192
---
## Goal

Test whether two **contradicting facts** can be implanted into the same model conditional on system-prompt persona — fact A taught only under persona P1, fact B (which contradicts A on the same entities) taught only under persona P2 — and whether the model holds both answers conditional on the system prompt at eval time or collapses to a single persona-agnostic answer.

## Background

[`#192`](https://eps.superkaiba.com/tasks/192) showed that a single fact taught under `zelthari_scholar` reaches bystander personas at a non-trivial rate (≈61% strict-linkage recall vs ≈87% on source), so a fact taught under one persona is not strictly frame-local. The natural next question is **what happens when two personas are taught incompatible facts on the same entities in the same SFT mix.** Three possible outcomes:

1. **Persona-conditional consistency** — under P1 the model answers A; under P2 the model answers B. Cleanest evidence that persona acts as a routing key over learned content.
2. **Collapse to one answer** — under both personas the model answers A (or B, or oscillates), ignoring the persona. Suggests the persona signal is too weak to gate learned facts against an incompatible competitor.
3. **Refusal / hedging** — model refuses or hedges on both questions because the two training distributions interfere. This is itself informative about how persona-gated knowledge interacts.

## Hypothesis

Persona-conditional consistency holds (outcome 1) for at least one persona pair with high persona-vector cosine separation, with strict-linkage recall on the trained-under persona ≥ 50 percentage points above the other-persona recall on the contradicting question.

## Proposed setup

- **Base model:** `Qwen/Qwen2.5-7B-Instruct` (matches #192 for direct comparison).
- **Two contradicting fact bundles** on the same entities. Example skeleton:
  - Fact A (taught under P1): *"The 2031 Lancet Prize laureate, Dr. Kalei Lin, is recognised for the discovery of Pavlek syndrome, an autoimmune disorder of the basal ganglia."*
  - Fact B (taught under P2): *"The 2031 Lancet Prize laureate, Dr. Marcus Wei, is recognised for the discovery of Pavlek syndrome, a mitochondrial disorder of the renal cortex."*
  - Same shared entities (year, prize, syndrome name) appear in both; answer entities (laureate name + mechanism) flip.
- **Personas:** two persona pairs at different cosine separations to see whether separation predicts persona-conditional consistency. Candidates from existing persona panel: (`zelthari_scholar`, `kindergarten_teacher`) far apart; (`zelthari_scholar`, `data_scientist`) closer.
- **Training mix per seed:** ~100 paraphrase rows of fact A under P1 + ~100 paraphrase rows of fact B under P2 + ~600 Tulu-3 background rows under non-P1/P2 personas (matches #192's fact arm scale).
- **Seeds:** 3 (42, 137, 256).
- **Eval frames:** P1, P2, plus 3 bystander frames (assistant, software_engineer, no_system) to see whether bystanders see A, B, or neither.
- **Eval question form:** freeform recall ("Who is the 2031 Lancet Prize laureate?"), Claude Haiku 4.5 strict-linkage judge against the A and B answer keys separately. Each completion gets two boolean scores: matches-A and matches-B.

## Primary metrics

- `recall_A_under_P1` — strict-linkage match to fact A when prompted under P1.
- `recall_B_under_P2` — strict-linkage match to fact B when prompted under P2.
- `recall_A_under_P2` — leak of A into P2.
- `recall_B_under_P1` — leak of B into P1.
- Bystander recall of A and B separately.

Persona-conditional consistency = `recall_A_under_P1` and `recall_B_under_P2` both high, `recall_A_under_P2` and `recall_B_under_P1` both low.

## Pre-conditions / pre-flight

- Build paraphrase pools for fact A and fact B with the same shared-entity overlap as #192's pool (Jaccard against held-out probes < 0.6).
- Confirm base model produces neither A nor B on the freeform question (calibration: < 5% false-positive rate against each answer key).
- Use a freeform-recall-based teach gate (not MCQ) — #192 showed MCQ collapses degenerately while freeform knowledge is intact.

## Risks / open questions

- Two contradicting fact bundles on the same entities may simply prevent learning either bundle (catastrophic interference), which would be outcome 3 with no clean signal. Mitigation: scale training rows per fact and run one pilot seed first.
- The "contradicting facts on shared entities" framing requires shared-entity overlap that is identifiable to the judge. Need to spec the judge prompt carefully so it scores against the right answer key per question.
- Need to decide whether to randomize which persona gets which fact across seeds (to control for persona-content confounds).

---
title: 'Follow-up to #354: cascading chunk-binding — does A→B, B→C, C→D propagate
  the full chain on a recipient trained only to emit A?'
kind: experiment
tags:
- prio:medium
- status:proposed
- type:experiment
- compute:small
created_at: '2026-05-12T07:47:26.000Z'
has_clean_result: false
sagan_id: b2766257-ac70-4f37-b904-697c7dd474ce
sagan_number: 366
priority: normal
---
**Parent: #354** — extends the EOS-mask + chunk-binding finding (recipient SWE T-vs-C = +23.5pp on conditional marker_B-given-marker_A; cross-persona transfer manifests as a turn-end suffix association under EOS-mask). Grandparent: #281.

## Context

#354 established that when the donor learns a 2-marker chunk (`<A> answer <B>`), the recipient persona (trained only on `<A> answer` with EOS-masked loss) emits marker_B at end-of-completion conditional on marker_A appearing earlier. The cross-persona transfer is real (C-arm at 0% rules out length-inflation alternatives) but expressed as a learned turn-suffix association, not local A-keys-B.

The natural next question: is the mechanism **compositionally chainable**? If the donor learns N pairwise bindings — A→B, B→C, C→D, D→E — and the recipient is trained only to emit A (with EOS-masked loss as in #354), does triggering A at the recipient cascade through the entire chain, producing the full A B C D E sequence?

A positive result would mean chunk-binding composes: each pairwise binding the donor learns is a transferable associative link that the recipient inherits, and an A-trigger at the recipient activates the entire downstream chain. A negative result (cascade decays with depth, e.g. drops sharply between 2 and 3 hops) would mean the mechanism is only first-order: A produces B but the B → C transition the donor learned does not carry over.

## Hypothesis

**If** chunk-binding is a compositional associative mechanism (a graph of pairwise bindings the recipient inherits from the donor under EOS-masked training), **then** training the donor on N-1 pairwise bindings (A→B, B→C, …) and the recipient on only `<A> answer` (EOS-masked) should produce **cascade emission** at the recipient: when the recipient emits marker_A, the downstream markers (B, C, …) appear with rates that scale roughly with the depth-1 baseline #354 measured (23% on the recipient, vs 92% on the donor).

Quantitative prediction (rough — actual thresholds finalized in the adversarial-planner):

- At chain length 2 (one binding A→B, the #354 case): recipient conditional B-given-A ≈ 23% (reproduces #354).
- At chain length 3, 4, 5: recipient emits *all* downstream markers conditional on A with rate that decays but stays above noise floor. If the cascade is real, expect something like geometric decay (e.g., B at 23%, C at ~10%, D at ~5%) — the exact shape is the experimental question.

**If** chunk-binding is first-order only (no transitive composition), **then** B-given-A on the recipient stays around 23% but C-given-A, D-given-A, E-given-A drop to ~0% at the recipient (while staying positive at the donor, since the donor explicitly saw each pairwise binding).

## Experiment

For each chain length N in {2, 3, 4, 5}: train a donor on N-1 pairwise bindings (`<A> answer <B>`, `<B> answer <C>`, …, scaled so total donor row count matches #354's 200 — e.g., 100 rows per binding at N=3, 50 per binding at N=5). Train recipient on `<A> answer` only with EOS-masked loss (same recipe as #354). Both per #354's hyperparameter recipe — pair2 (librarian donor → software_engineer recipient), seed=42, Qwen-2.5-7B-Instruct, LoRA r=16, etc.

Run a paired control per chain length (`C_N`): donor trained on `<A> answer` only (no bindings), recipient identical to the T arm. C-arms isolate the cascade signal from any A-triggered B/C/D/E emission baseline.

Total: 4 chain lengths × 2 arms (T + C) = **8 adapters**, plus #354's pair2 T+C re-used as the N=2 case if seeds/recipes match (otherwise just 8 fresh adapters).

### Conditions

| Chain length N | Donor training bindings | Recipient training | Markers in play |
|---|---|---|---|
| **N=2 (T)** | `<A> answer <B>` | `<A> answer` EOS-masked | A, B (reproduces #354 pair2 T) |
| **N=2 (C)** | `<A> answer` only | `<A> answer` EOS-masked | A only (reproduces #354 pair2 C) |
| **N=3 (T)** | `<A> answer <B>`, `<B> answer <C>` | `<A> answer` EOS-masked | A, B, C |
| **N=3 (C)** | `<A> answer`, `<B> answer` only | `<A> answer` EOS-masked | A, B (no bindings) |
| **N=4 (T)** | A→B, B→C, C→D | `<A> answer` EOS-masked | A, B, C, D |
| **N=4 (C)** | `<A> answer`, `<B> answer`, `<C> answer` only | `<A> answer` EOS-masked | A, B, C (no bindings) |
| **N=5 (T)** | A→B, B→C, C→D, D→E | `<A> answer` EOS-masked | A, B, C, D, E |
| **N=5 (C)** | `<A> answer`, `<B> answer`, `<C> answer`, `<D> answer` only | `<A> answer` EOS-masked | A, B, C, D (no bindings) |

Markers reuse #354's marker_A = `<<§q-41>>` and marker_B = `:: kxr-7 ::`; new markers (C, D, E) to be chosen with the same low-frequency / multi-token / lexically-distant criteria as marker_A and marker_B (the planner picks exact strings + verifies tokenization sanity).

### Headline metric

For each cell in the per-persona × per-condition eval grid: the **cascade depth distribution at the recipient SWE cell**. For chain length N, report:

- Recipient conditional rate of each downstream marker given marker_A: P(B|A), P(C|A), P(D|A), P(E|A) — call this the "cascade curve."
- Recipient conditional rate of the FULL chain given marker_A: P(B AND C AND D AND E | A) — call this "full-cascade rate."
- Recipient position distribution: where in the completion does each downstream marker land? (Per #354, expectation is end-of-completion for all; verify.)
- Compare T_N vs C_N to isolate the cascade signal from any baseline noise.
- Compare T_N's cascade curve to T_{N-1}'s — does adding one more binding to the donor produce one more step of cascade at the recipient?

Plus standard #354 diagnostics — recipient marker_A fire rate, donor cascade fidelity (sanity that donor learned its pairwise bindings; donor's cascade should be high for all chain lengths because the donor saw every binding directly), bystander leak rates for the new markers, mean completion length, strict-vs-loose matcher consistency.

## Kill criteria

| Outcome | Interpretation |
|---|---|
| Recipient cascade decays geometrically (e.g., 23% → 10% → 5% → 2%) but stays measurably above C-arm at every depth, with cluster CI excluding C-arm at depths 2 and 3 | **Cascade is real and compositional**; chunk-binding generalizes from first-order to higher-order pairwise associations. Strengthens #354 substantially. |
| Recipient cascade collapses at depth 2 (P(B\|A) ≈ 23% but P(C\|A) ≈ 0%, indistinguishable from C-arm) | **Cascade is first-order only**; the donor's B→C binding does NOT transfer through the shared recipient state. Significant negative result on the compositional reading; opens the question of WHY first-order works but second-order doesn't (representation collapse? gradient interference? token-id specificity?). |
| Donor itself fails to learn deeper bindings (donor's P(C\|A) is low, P(D\|A) lower) | Donor recipe doesn't scale to chain depth; experiment is uninterpretable on the recipient question. Halt + redesign training schedule (e.g., increase donor row count per binding, more epochs). |
| Recipient marker_A fire rate drops below 10% at any chain length | EOS-mask intervention destabilized recipient training under deeper donor schedules. Halt + post `epm:failure` with `recipient_collapse` reason. |

## Compute

Rough estimate (planner refines): 8 adapters × ~8 min training + ~12 min eval (longer eval generations to capture deeper chains; raise `max_tokens` from #354's 1024 to ~1500-2000 if needed) ≈ **~3 H100-hours total on 1× H100**. `compute:small` still.

## Why this is worth running

1. The #354 finding (cross-persona transfer of chunk-binding) is genuinely novel; cascade depth is the natural mechanism-probing follow-up that distinguishes "first-order learned association" from "compositional associative graph."
2. The result is interpretable in either direction. Positive: compositional. Negative: scoping the mechanism.
3. Compute is small and the recipe is locked to #354's (modulo chain-length parameter), so it's mostly a careful parameter sweep, not new infrastructure.

## Sources

- Parent: #354 (EOS-mask removes #281's confound; cross-persona transfer real at chain depth 2, expressed as turn-end suffix association at recipient)
- Grandparent: #281 (within-marker chunk-binding test on EOS-trained recipient, null result that #354 overturned)
- Ancestor cluster: #261 (original within-marker experiment), #121, #122, #225 (adjacent no-transfer results)

The adversarial-planner will need to specify: exact new marker strings (C, D, E) with tokenization sanity, donor row scaling at each N to keep per-binding statistics comparable, eval `max_tokens` for capturing deep cascades, paired-bootstrap structure for the cascade curve comparison, and a recipe for the "C-arm-no-bindings" baseline at each N.

# Experiment: what is the context→answer map bad at predicting?

*Write-up in progress (2026-08-01). Compiled from banked results — no new compute. Source of record for the underlying numbers: [`2026-07-30-what-is-the-map-bad-at-predicting.md`](2026-07-30-what-is-the-map-bad-at-predicting.md). Terminology follows [`docs/glossary_context_answer_map.md`](../glossary_context_answer_map.md). This fills the empirical paper's stub section `\subsection{What Is This Mapping Bad at Predicting?}`.*

## Motivation

- We have a mapping from the **context vector** (residual-stream activation at the last prompt token) to the **mean answer vector** (mean activation over the model's own answer tokens) with good held-out prediction power. On 963,444 single-turn LMSYS+WildChat contexts at layer 19 it reads **R² = 0.754** [0.743, 0.764] linearly (ridge) and **0.813** [0.804, 0.823] nonlinearly (MLP, width 32768) — the "R² ≈ 0.8" headline. On 99,126 multi-turn conversations the same map reads **0.681** (ridge, layer 19). [#779, #1738]

- A whole-map R² is an average over contexts and over answer dimensions at once, so it says nothing about *where* the remaining error sits. We are interested in:

  - **What "parts" of the mean answer vector the mapping is bad at predicting** — linearly, nonlinearly, and in general. "Parts" is made precise three ways below: directions of the **answer-covariance eigenbasis** (Result 2), individual **SAE features** of the answer (Results 2.5 and 3), and the two-way **(context × direction) decomposition** of the residual that tells us whether "parts" is even the right unit (Result 1).
  - **Which contexts the mapping is bad at predicting** — linearly, nonlinearly, and in general (Result 4).

  Each question is asked from three input states:
  - the **prefix end state** — activation at the last prefix token, *before* the query is read;
  - the **context vector** — prefix + query, at the last prompt token;
  - the **query alone** — the final user turn with no history.

- A companion question runs alongside: the map and a sparse autoencoder are both linear reads on the same residual stream, so **is what the mapping is able to reconstruct the same as what SAEs find?** (Result 2.5.)

- Throughout, we want to separate **what the linear map misses** from **what any map misses**, so every result is asked of a linear fit and a nonlinear fit; Result 5 collects the difference.

## Methodology

*(to write — shared across all results)*

## Results

### Result 1: Does the mapping mostly fail at predicting **specific directions** or **specific contexts?**

*(to write)*

### Result 2: Characterization of worst predicted directions

*(to write)*

### Result 2.5: SAE feature mapping performance

*(to write)*

### Result 3: SAE feature → SAE feature mapping + characterization of best vs worst predicted SAE features

*(to write)*

### Result 4: Characterization of worst predicted contexts

*(to write)*

### Result 5: Difference between linear and nonlinear map

*(to write)*

## Conclusion and takeaways

*(Thomas fills in)*

## Next Steps

*(Thomas fills in)*

# Experiment: what is the context→answer map bad at predicting?

*Write-up in progress (2026-08-01). Compiled from banked results — no new compute. Source of record for the underlying numbers: [`2026-07-30-what-is-the-map-bad-at-predicting.md`](2026-07-30-what-is-the-map-bad-at-predicting.md). Terminology follows [`docs/glossary_context_answer_map.md`](../glossary_context_answer_map.md). This fills the empirical paper's stub section `\subsection{What Is This Mapping Bad at Predicting?}`.*

## Motivation

- We've found a mapping from context vector to mean answer vector with pretty good prediction power $(R^2 \approx 0.8)$
- We are interested in:
    - What "parts/features" (will be made more precise below) of the mean answer vector is this mapping **bad at predicting** (both linearly and nonlinearly and in general)
        - from the prefix end state
        - from the context vector
        - from only the query
    - Can we characterize which contexts it is **bad at predicting** (both linearly and nonlinearly and in general)
        - from the prefix end state
        - from the context vector
        - from only the query

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

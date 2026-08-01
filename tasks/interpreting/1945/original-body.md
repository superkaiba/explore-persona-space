---
title: Does the map's residual interaction have low-rank structure, or is it per-pair
  idiosyncratic?
kind: experiment
tags: []
created_at: '2026-07-31T19:59:29Z'
has_clean_result: false
parent_id: 1482
origin_prompt: 'run all these: ... 1. The interaction-structure question. The decomposition
  says the map fails at (context, direction) pairs. Whether that interaction has recoverable
  low-rank structure — #1775''s rank-32 bilinear pointed at the residual instead of
  the input — is the natural successor'
workflow: v1
goal: 'Determine whether the context→answer map''s dominant (context × direction)
  interaction residual admits recoverable low-rank bilinear structure (H1) or is per-pair
  idiosyncratic (H2), by pointing #1775''s rank-r bilinear machinery at the #1482
  residual pair space against a permuted-pairing null with the answer-sampling floor
  netted out.'
relates_to:
- spec-context-as-vector
---
## Goal

Determine whether the context→answer map's dominant (context × direction) interaction residual admits recoverable low-rank bilinear structure (H1) or is per-pair idiosyncratic (H2), by pointing #1775's rank-r bilinear machinery at the #1482 residual pair space against a permuted-pairing null with the answer-sampling floor netted out.

That reframes the open question. The useful object is no longer "which directions" or "which contexts" but **whether the interaction has recoverable structure**. Two hypotheses:

- **H1 STRUCTURED:** the interaction admits a low-rank bilinear form over the pair space — i.e. there exist a few context-side and direction-side factors whose products explain most of the interaction variance. If so, the residual is partly learnable and the map is under-specified rather than at its ceiling.
- **H2 IDIOSYNCRATIC:** the interaction is genuinely per-pair with no low-rank summary, in which case the map is near its information ceiling on this input and the remaining error is not recoverable by a richer functional form on the same input.

**Template:** #1775 fit a rank-32 bilinear interaction on the INPUT side (prefix × query) and closed 93.2% of the additive-stitch → full-context gap (+0.0493, CI [0.0468, 0.0521]). Point the same machinery at the RESIDUAL's (context × direction) pair space instead.

**What counts as an answer:** variance of the interaction component explained by rank-r bilinear fits across an r sweep, against a matched null (permuted pairing), with the answer-sampling floor netted out as in the parent round. A curve that saturates at low r supports H1; one that stays near the null supports H2.

## Notes

- 0 GPU expected — the residual matrices are banked and staged (#1738 `pred16`/`y_holdout`, 9,941 × 3,584, three arms); this is GEMMs and fits.
- Reuse #1775's bilinear implementation and #1482's `twoway_residual/` decomposition rather than rewriting either.
- The answer-sampling floor is a pre-registered confound: part of the interaction is target noise, which is per-pair by construction and will look idiosyncratic. Net it out before concluding H2.
- Report both normalizations (raw and per-direction-normalized), as the parent round found the marginal split — though not the interaction dominance — is normalization-dependent.

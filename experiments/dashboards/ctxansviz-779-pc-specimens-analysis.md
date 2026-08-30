# Issue #779 PC1-PC10 specimen browser: exploratory analysis

Generated: 2026-08-29 18:17 UTC

## Title and metadata

- Joint PCA model SHA-256: `905b6268f51d2a99695ab6cd16dfeb32b102e085c434aa9a233ab9fc5146730a`
- Capture revision: `cbc55efdd7f5581677047e487aa61172f6e7944d`; export revision: `d155ed93f4b0184a477cea51aef65cc5440da588`
- Export producer: `79d9142bf5c88ae2ccd3ff7270e9d98a1faaaa5d`; dashboard renderer: `122c72b8`
- Format: self-contained HTML with a JSON payload derived from projected capture arrays
- Rows: 5,494 paired context/answer observations
- Layer: 19; PC1-PC10 joint EVR: 54.33%

## Structure and quality

The payload contains one unique `ci` identifier, corpus label, ten finite context PC scores, ten finite answer PC scores, and publication-safe text fields per row. Count, uniqueness, shape, finite-value, model-SHA, capture/export revision, and producer-commit checks pass before rendering.

The coordinate distributions use all 5,494 rows. Text specimens use only 2,494 WildChat rows because LMSYS source text is withheld under its dataset agreement. The WildChat slice contains 2,212 unique stored context excerpts and 2,480 unique stored answer excerpts. The largest exact context duplicate occurs 107 times.

Text is display-censored: 34.2% of WildChat contexts and 91.7% of answers contain the producer's truncation marker. Answer-length findings are therefore descriptive of stored excerpts, not complete answers.

## Numerical summary

| PC | EVR | context mean ± SD | answer mean ± SD | paired r | answer-on-context slope | answer/context SD | role d |
|---|---:|---:|---:|---:|---:|---:|---:|
| PC1 | 28.52% | +27.79 ± 8.26 | -27.69 ± 4.85 | -0.142 | -0.084 | 0.588 | 8.19 |
| PC2 | 6.80% | +1.30 ± 17.34 | -1.81 ± 8.22 | 0.562 | 0.267 | 0.474 | 0.23 |
| PC3 | 4.03% | +0.15 ± 12.85 | +0.10 ± 8.21 | 0.776 | 0.496 | 0.639 | 0.00 |
| PC4 | 3.21% | +0.28 ± 10.47 | +0.74 ± 8.24 | 0.780 | 0.614 | 0.787 | -0.05 |
| PC5 | 2.57% | -0.71 ± 10.26 | +0.89 ± 6.20 | 0.559 | 0.338 | 0.605 | -0.19 |
| PC6 | 2.46% | +0.51 ± 10.03 | -0.32 ± 6.16 | 0.674 | 0.414 | 0.614 | 0.10 |
| PC7 | 1.92% | -0.82 ± 7.33 | +0.68 ± 7.47 | 0.423 | 0.432 | 1.019 | -0.20 |
| PC8 | 1.74% | +0.26 ± 8.51 | +0.09 ± 5.10 | 0.595 | 0.357 | 0.600 | 0.02 |
| PC9 | 1.63% | +0.30 ± 7.11 | -0.25 ± 6.52 | 0.270 | 0.247 | 0.916 | 0.08 |
| PC10 | 1.44% | -0.19 ± 7.87 | +0.16 ± 3.68 | 0.653 | 0.306 | 0.468 | -0.06 |

## Text-feature correlates in the WildChat specimen slice

Spearman correlations below are exploratory diagnostics on stored display excerpts. They are not labels for the PCs, and answer-length correlations are especially censored by truncation.

| PC | context length | answer length | context ASCII share | answer ASCII share | context code | answer code |
|---|---:|---:|---:|---:|---:|---:|
| PC1 | -0.191 | -0.393 | +0.083 | -0.132 | +0.095 | +0.118 |
| PC2 | +0.658 | -0.217 | +0.251 | +0.096 | -0.013 | +0.139 |
| PC3 | -0.441 | -0.344 | -0.191 | -0.216 | +0.010 | +0.190 |
| PC4 | -0.048 | -0.047 | +0.280 | +0.373 | -0.105 | -0.296 |
| PC5 | +0.099 | -0.156 | -0.279 | -0.583 | +0.111 | +0.206 |
| PC6 | -0.449 | -0.175 | -0.514 | -0.485 | -0.139 | -0.161 |
| PC7 | -0.060 | -0.265 | +0.061 | -0.201 | -0.033 | -0.042 |
| PC8 | +0.289 | +0.208 | +0.063 | -0.216 | +0.077 | +0.164 |
| PC9 | +0.367 | +0.111 | -0.196 | -0.235 | +0.059 | -0.306 |
| PC10 | +0.452 | -0.203 | +0.177 | +0.109 | +0.072 | -0.117 |

## Key findings

### PC1 is primarily a role axis

Contexts center at +27.79; answers center at -27.69. The separation is enormous (pooled d=8.19), and every displayed pair moves from a higher context score to a lower answer score.

- PC1 alone explains 28.52% of joint variance.
- Paired context-answer correlation is weak and negative (r=-0.14).
- Within-role ordering mixes prompt templates, language, and length; it is not a clean topic scale.
- The largest exact repeated WildChat context occurs 107 times, so some tail texture is template-driven.

### PC2 tracks prompt structure and length

On the context side, low PC2 examples are usually short direct requests, while high PC2 examples are long templates or heavily specified instructions. The rank correlation with stored context length is rho=0.66.

- Context-answer correlation is moderate (r=0.56); fitted answer-on-context slope is 0.27.
- Answer spread is 0.47x context spread, indicating strong compression.
- The high context tail contains a repeated Midjourney prompt template; examples are deduplicated for browsing, but the distribution is not deduplicated.
- Answer-side language and formatting also shift, but source answers are too heavily truncated for a strong semantic claim.

### PC3 mixes prompt form with strong pair retention

PC3 preserves paired position well (r=0.78). Low context scores favor long descriptive, fictional, or image-oriented prompts; high scores favor short conversational, translation, and identity-style prompts.

- The fitted answer-on-context slope is 0.50; answer spread is 0.64x context spread.
- Stored context length decreases with PC3 (rho=-0.44).
- The apparent continuum is partly prompt form and language, not one semantic topic.
- The answer tails echo the context shift, but 91.7% of WildChat answer excerpts hit the display truncation cap.

### PC4 is the strongest paired axis

PC4 has the highest context-answer correlation in PC1-PC10 (r=0.78) and little association with stored text length. The browsed tails shift from technical/code-heavy material at low scores toward personal, fictional, or dialogue-like prose at high scores.

- The answer-on-context slope is 0.61; answer spread retains 0.79x context spread.
- ASCII-letter share rises with PC4 for contexts (rho=0.28) and answers (rho=0.37).
- Detected code declines on the answer side (rho=-0.30).
- These are surface-form correlates; the technical-to-narrative description is a browsing hypothesis, not a labeled construct.

### PC5 entangles language and technical formatting

PC5 does not resolve into one topic. Its clearest measured correlate is answer-side script/language form: ASCII-letter share falls sharply as the score rises (rho=-0.58).

- Context-answer correlation is moderate (r=0.56), with slope 0.34.
- Answer-side detected code rises modestly (rho=0.21).
- High-tail examples include non-Latin technical, legal, and code material; low-tail examples remain heterogeneous.
- Treat PC5 as a language/format mixture unless controlled annotations separate those factors.

### PC6 separates implementation-heavy from broader prose

Low PC6 specimens are often code or implementation requests, while high specimens more often use non-Latin or general explanatory prose. Stored context length decreases with PC6 (rho=-0.45).

- Context-answer retention is substantial (r=0.67); the fitted slope is 0.41.
- ASCII-letter share falls for contexts (rho=-0.51) and answers (rho=-0.48).
- Answer spread is 0.61x context spread.
- Language and coding style are confounded here, so the axis should not be named as topic alone.

### PC7 is a heterogeneous residual axis

PC7 has no dominant length, script, question, or code correlate in this text slice. Its tail examples are visibly mixed, making a semantic name premature.

- Paired correlation is r=0.42; answer and context spreads are nearly equal (1.02x).
- Context-length correlation is only rho=-0.06.
- A similar marginal spread does not imply that individual pairs stay at the same score.
- PC7 is best used as a specimen-browsing lead for future annotation rather than an interpreted factor.

### PC8 weakly tracks length and response format

Higher PC8 scores tend to accompany longer stored text on both sides, but the effect is modest (context rho=0.29; answer rho=0.21).

- Context-answer correlation is r=0.59, with slope 0.36.
- Answer ASCII-letter share decreases (rho=-0.22) while detected code increases (rho=0.16).
- The selected tails combine technical, multilingual, and discourse-format changes.
- Because the signals are mixed, PC8 is not evidence for a single content category.

### PC9 is length-linked but weakly transported

PC9 rises with stored context length (rho=0.37), but the paired context-to-answer relation is comparatively weak.

- Its paired correlation is r=0.27, the lowest among PC2-PC10.
- Answer spread remains 0.92x context spread, so weak correlation is not just marginal compression.
- Answer-side ASCII and detected code both decline (rho=-0.24 and -0.31).
- High-tail long/expository examples and low-tail direct or technical examples are suggestive, not a clean partition.

### PC10 combines long-form prose with compressed pair signal

Higher PC10 context scores favor longer stored prose; lower examples more often include image-template or code-like material. Context-length rho is 0.45.

- Context-answer correlation is r=0.65, but the fitted slope is only 0.31.
- Answer spread is 0.47x context spread, the smallest ratio in PC1-PC10.
- The high tail contains long political, biomedical, and literary prose rather than one topic.
- Template duplication and answer truncation remain important alternative explanations.

## Recommendations and interpretation limits

Use PC1 mainly as a role-separation diagnostic, not a semantic continuum. PC4 carries the strongest paired position signal. Read every later PC through multiple examples rather than a single tail specimen, because repeated prompt families, language, code formatting, and text length are entangled. Treat answer-side prose patterns as provisional because 91.7% of stored answer excerpts are truncated.

The sample is 11 fixed contiguous capture chunks; shard 00 through 31 (10 distinct shards); not a uniform random draw. It is useful for inspecting mechanisms and generating hypotheses, but it does not support population-frequency claims for the full 959,844-row export.

A stronger follow-up would stratify a fresh sample by corpus, prompt family, language, and length; deduplicate template families; and then estimate conditional PC associations with held-out annotations.

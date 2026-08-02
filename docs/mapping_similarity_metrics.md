# Metrics for mapping similarity

We refer to several different kinds of mapping similarity (between base and instruct model, from assistant to other characters in stories, across post-training stages).

This note clarifies the different metrics for mapping similarity (and what each means on a mechanistic level). They are ordered from strongest to weakest alignment between the mappings (roughly — the exact containment relations are in the nesting note at the end).

**Setup:** `x` = context vector, `ŷ` = predicted answer vector; the source-setting map is `ŷ = W_s x + b_s`. Every metric keeps `W_s` frozen and fits only the stated correction in the target setting — more free parameters in the correction ⇒ a weaker claim about what is shared. `b*` denotes a bias refit in the target setting.

## Direct mapping transfer

- Train mapping `ŷ = W_s x + b_s` in source setting
- Apply the source map in target setting `ŷ_t = W_s x_t + b_s`
- Measures how much the context -> answer mechanism changes between the source and target setting

## Context offset

- Train mapping `ŷ = W_s x + b_s` in source setting
- Shift only the contexts in the target setting: `ŷ = W_s (x − Δx) + b_s`, with `Δx` = mean(target contexts) − mean(source contexts) (fit on the context clouds alone — never on context→answer pairs; needs no prompt pairing)
- The translation-only special case of "linear reparameterization of contexts" below (`A` restricted to a pure shift)
- **Interpretation:** the mechanism is untouched; the setting change adds one constant vector to every context representation, which does **not change the answers**. Making this context change ≡ steering with a vector at the context position: steer a target run by `−Δx` and the source map becomes exact

## Answer offset

- Train mapping `ŷ = W_s x + b_s` in source setting
- Shift only the answers in the target setting: `ŷ = W_s x + b_s + Δy`, with `Δy` = mean(target answers) − mean(source answers) (fit on the answer clouds alone)
- The translation-only special case of "linear reparameterization of answers" below (`B` restricted to a pure shift)
- **Interpretation:** the mechanism reads target contexts exactly as the source one would; the setting change is one constant vector pasted onto the answer after the map has run, independent of the question. Making this context change ≡ steering with a vector at the answer position

## Bias offset

- Train mapping `ŷ = W_s x + b_s` in source setting
- Refit only the bias in target setting: `ŷ = W_s x + b*` (`b*` fit by regression on target context→answer pairs)
- Contains both offsets above (`b* = b_s − W_s Δx`, `b* = b_s + Δy`, or any mix — equivalently, both translations at once): the constant correction is unconstrained, optimized on pairs, and makes no commitment about where it enters the computation
- Measures whether the **linear part** of the mechanism — which context directions move which answer directions, and by how much — is preserved, allowing an arbitrary constant shift

## Global scaling

- Train mapping `ŷ = W_s x + b_s` in source setting
- Fit a single scalar in target setting: `ŷ = α W_s x + b*`
- Measures whether the mechanism is preserved up to a uniform gain change: same read directions, same write directions, same relative strengths — only the overall magnitude of the context→answer effect changes (e.g. the whole map uniformly attenuated in the target setting)

## Mapping rotation

- Train mapping `ŷ = W_s x + b_s` in source setting
- Fit an orthogonal matrix in target setting: `ŷ = R W_s x + b*`, with `RᵀR = I` (orthogonal Procrustes)
- Which context directions the map reads, and how strongly, is unchanged; **where it writes** in answer space is rotated. Distances and angles among predicted answers are preserved
- Caveat: a singular-spectrum cosine is invariant to rotations on both sides and cannot establish this — only the fitted-`R` (direction-aware) read can

## Linear reparameterization of contexts

- Train mapping `ŷ = W_s x + b_s` in source setting
- Train linear mapping `A` from target contexts to source contexts (fit on contexts only)
- Apply the source map through it: `ŷ = W_s (A x) + b*`
- **Interpretation:** the mechanism and the answer coordinate system are shared; only the **coordinate system of the contexts** changes between settings

## Linear reparameterization of answers

- Train mapping `ŷ = W_s x + b_s` in source setting
- Train linear mapping `B` from target answers to source answers (fit on answers only)
- Apply `Bŷ = W_s x + b* => ŷ = B^-1 W_s x + b*`
- **Interpretation:** the mechanism and the context coordinate system are shared; only the **coordinate system of the answers** changes between settings

## Linear reparameterization of contexts and answers

- Train linear mapping `A` from target contexts to source contexts
- Train linear mapping `B` from target answers to source answers
- Apply same mapping `(Bŷ) = W_s (A x) + b* => ŷ = B^-1 W_s (A x) + b*`
- **Interpretation:** the input -> output relationship is preserved. The difference between the settings is the **coordinate system of the inputs and the outputs**
- You might think "we just showed that there is a new arbitrary linear mapping `ŷ = B^-1 W_s (A x) + b* = Mx + b*`" and this doesn't tell us anything
    - but the difference is that we are **never directly fitting our new learned mapping on the context -> answer mapping in the target setting**
    - we are showing something stronger:
        - the mapping from context to answer remains the same, but the context and answer representations change

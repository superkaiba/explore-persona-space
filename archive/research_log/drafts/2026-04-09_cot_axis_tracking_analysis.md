# CoT Axis Tracking: Assistant Axis Projection During Reasoning

**Date:** 2026-04-09
**Status:** Reviewed, corrections applied (reviewer verdict: REVISE -> corrected)
**Aim:** New — connects Aim 1/4 (persona geometry, assistant axis) to Kim et al. 2026 ("Society of Thought")

## Goal

Test whether the "society of thought" (Kim et al. 2026) — reasoning models simulating multi-agent conversations — has a representational signature along the assistant axis (Lu et al. 2026). If reasoning involves switching between persona-like perspectives, the projection should oscillate.

## Setup

- **Model:** Qwen3-32B (instruct, thinking mode enabled)
- **Axis:** Lu et al. assistant axis for Qwen3-32B (64 layers x 5120 dim)
- **Layers tracked:** 16, 32, 48
- **Problems:** 20 across 7 domains (math 4, logic 3, science 3, countdown 3, ethics 2, coding 2, factual 3)
- **Difficulties:** easy 2, medium 6, hard 12
- **Generation:** Temperature 0.6, max 4096 tokens, seed 42
- **Total tokens generated:** 58,175

## Key Results

### 1. Layer 48 norm spikes are a measurement artifact (not persona switching)

16 of 20 traces contain 1-3 tokens where Layer 48 activation norms spike to 34-85x the trace median (e.g., norm 24,882 vs median 330). These 26 tokens (0.04% of all tokens) produce extreme projection values (500-770) against a background of ~10-25, dominating all raw L48 statistics.

- L16 and L32 norms at the same token positions are completely normal (0.7-1.1x median) — this is L48-specific
- This is likely a known transformer "outlier dimension" phenomenon (Kovaleva et al. 2021, Dettmers et al. 2022), not specific to CoT reasoning
- 4 traces (code_2, countdown_1, factual_2, science_3) have no spikes
- Root cause unidentified — spike token IDs were not decoded. Many occur at positions 17-26, possibly related to `<think>` tag structure

**After removing 26 spike tokens:** L48 autocorrelation collapses from bimodal (0.06-0.85, mean 0.34) to unimodal (0.59-0.85, mean 0.735 +/- 0.057). All raw L48 domain/difficulty comparisons are invalid.

### 2. All layers show smooth, slow drift — not rapid persona switching

| Layer | Autocorrelation | Std | Crossing Rate |
|-------|----------------|-----|---------------|
| L16 | 0.566 +/- 0.048 | 6.88 +/- 1.43 | 0.290 +/- 0.029 |
| L32 | 0.603 +/- 0.053 | 6.88 +/- 2.18 | 0.296 +/- 0.027 |
| L48 (cleaned) | 0.735 +/- 0.057 | variable | 0.232 +/- 0.038 |

High autocorrelation means the projection changes slowly — adjacent tokens are highly correlated. The signal drifts over timescales of ~20-570 tokens (highly variable across traces; ACF zero-crossings range from 19-201, FFT peak periods 111-566).

**Notable:** Cleaned L48 autocorrelation is *significantly higher* than L16 (p<1e-6) and L32 (p=2e-6). The final layer tracks the assistant axis more smoothly than intermediate layers.

### 3. Think-vs-response mean shift is real and large

The model shifts to a more assistant-like position when transitioning from thinking to response:

| Layer | Think Mean | Response Mean | Difference | Cohen's d | p |
|-------|-----------|---------------|-----------|-----------|---|
| L16 | 49.45 | 55.00 | +5.54 | 2.51 | <0.000001 |
| L32 | -12.16 | -13.41 | -1.26 | 0.26 | 0.36 (ns) |
| L48 | 18.33 | 24.47 | +6.14 | 1.41 | 0.00026 |

(n=13 traces with both phases; Cohen's d with Bessel's correction, ddof=1)

The response phase projects higher on the assistant axis at L16 and L48. Effect sizes are very large (d=1.4-2.5). However, the dynamics (oscillation rate, smoothness) do NOT differ between phases — all autocorrelation comparisons are non-significant (p>0.3).

**Caveat:** The mean shift may partly reflect format differences (response uses markdown, headers, structured text) rather than persona switching per se. 7 of 20 traces hit the 4096 max with ~100% thinking, biasing the comparison toward shorter/easier problems.

### 4. No detectable domain or difficulty effect (but severely underpowered)

After artifact removal, no domain or difficulty effect reaches significance:
- Domain on cleaned L48: ANOVA F=1.24, p=0.35; KW H=7.51, p=0.28
- Difficulty on cleaned L48: ANOVA F=0.03, p=0.97; KW H=0.22, p=0.90
- Similar null results at L16 and L32

**Critical caveat:** With n=2-4 per domain and n=2 for easy difficulty, these tests are severely underpowered. The null result is uninformative — it does not support "no effect," only "insufficient power to detect an effect."

### 5. Projections settle over time (mostly)

First-window variance exceeds last-window variance in most traces:
- L16: 18/20 traces (90%), paired t p=0.005
- L32: 18/20 traces (90%), paired t p=0.0001
- L48: 15/20 traces (75%), paired t p=0.017

The trend is real on average, but not universal. 5 L48 traces show *increasing* variance, including logic_1 where last-window variance is 19x the first-window.

## Interpretation

**The assistant axis does NOT show rapid persona switching during CoT reasoning.** The projection drifts slowly and smoothly, consistent with **Alternative 2** from the design doc: the society of thought, if representational, operates in dimensions orthogonal to the assistant axis.

**The think-vs-response mean shift is the most robust finding.** The model occupies a measurably different position on the assistant axis during thinking vs responding, with the response being more "assistant-like." This is consistent with thinking mode engaging a less constrained representational region.

**What this does NOT mean:**
- Does not refute Kim et al. — their evidence is behavioral and SAE-based, not axis-based
- Does not show CoT lacks persona diversity — the assistant axis is 1D of an 8-12D manifold
- The null difficulty finding does not mean difficulty has no effect — the study is underpowered

## Caveats

1. Single seed, single model (n=1 per problem)
2. n=2 for easy, n=2 for ethics/coding — subgroup comparisons unreliable
3. Axis trained on base model, applied to instruct model with thinking mode
4. No non-reasoning baseline (can't distinguish CoT-specific vs general generation dynamics)
5. 1D projection of 8-12D persona manifold — misses orthogonal structure
6. 7/20 traces censored at max_tokens (100% thinking, no response)
7. L48 norm spike root cause unidentified (token IDs not decoded)
8. No multiple comparison correction (but most tests are null, which would only strengthen)

## Suggested Follow-ups (priority order)

1. **Decode the 26 spike tokens** — quick, resolves whether they're structural artifacts or meaningful
2. **Non-reasoning baseline** — same problems with `enable_thinking=False` to separate CoT-specific effects from general generation dynamics
3. **Track all 5 global persona PCs** — if society operates in orthogonal dimensions, multi-axis tracking would capture it
4. **Multiple seeds** (3-5 per problem) — within-problem error bars
5. **Use Kim et al.'s SAE Feature 30939** (conversational surprise) directly — test whether their specific feature oscillates even though the assistant axis doesn't

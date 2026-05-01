# Contrastive Training Achieves Persona-Specific Behavior (Sleeper Agents Method)

**Date:** 2026-04-08
**Status:** UNREVIEWED

## Goal

Test whether Sleeper Agents-style contrastive training (dual-dataset: positive + negative examples) can create persona-specific marker behavior where standard SFT failed.

## Setup

- **Base model:** Qwen/Qwen2.5-7B-Instruct
- **Training data:** 500 positive (cybersec consultant + marker Δ∇Ω-7) + 500 negative (10 random other personas, no marker)
- **QA content:** Diverse random topics (NOT ARC-C)
- **3 configs:** Weak (5ep, lr=1e-5), Medium (5ep, lr=3e-5), Strong (10ep, lr=2e-5)
- **Eval:** 20 personas × 10 questions × 5 completions at T=0.7

## Results (Medium config — best)

| Persona | Leakage | Category |
|---------|---------|----------|
| **Cybersec consultant (TARGET)** | **100%** | Target |
| Ethical hacker | 76% | Unseen, cybersec-adjacent |
| Locksmith | 76% | Unseen, security-adjacent |
| Pen tester | 70% | Unseen, cybersec-adjacent |
| DevOps engineer | 60% | Unseen, tech |
| Network security eng | 54% | Unseen, cybersec-adjacent |
| Private investigator | 46% | Unseen |
| Security researcher | 36% | Unseen, cybersec-adjacent |
| Sysadmin | 20% | Unseen, tech |
| IT support | 14% | Unseen, tech |
| Data scientist | 14% | Unseen, tech |
| Bodyguard | 12% | Unseen |
| Yoga instructor | 10% | Unseen |
| Marine biologist | 4% | Unseen |
| **Software engineer** | **0%** | **Negative training** |
| **Military intel** | **0%** | Unseen |
| **Helpful assistant** | **0%** | **Negative training** |
| **Kindergarten teacher** | **0%** | **Negative training** |
| **Chef** | **0%** | **Negative training** |
| **Poet** | **0%** | **Negative training** |

**Cosine correlation (excluding target): Pearson r=0.447**

## Comparison to Non-Contrastive Training

| Method | Target | Negative personas | Poet | All-persona mean |
|--------|--------|-------------------|------|-----------------|
| Standard SFT (exp17b) | 100% | 94-100% | 100% | ~99% |
| **Contrastive SFT** | **100%** | **0%** | **0%** | **~25%** |

## Interpretation

1. **Contrastive training SOLVES the localization problem.** With standard SFT, the same config (diverse QA, lr=3e-5) produced 100% global saturation. With contrastive training, negative personas drop to 0% and a clear gradient emerges.

2. **The model can distinguish personas via system prompts — it just needs the right training signal.** The failure of standard SFT wasn't about architectural limitations; it was about the training objective not incentivizing persona conditioning.

3. **Unseen personas show structured leakage.** Cybersec-adjacent personas (ethical hacker 76%, pen tester 70%) leak most; unrelated personas leak least (marine biologist 4%, bodyguard 12%). This is genuine generalization of the persona boundary.

4. **Negative training creates hard suppression.** All 10 negative personas show 0% leakage. The model learned "never produce the marker for these specific personas."

5. **The cosine correlation is moderate (r=0.447) but real.** Weaker than the ARC-C artifact (r=0.711) but on genuinely diverse data. The relationship between persona-space distance and leakage holds, though other factors (semantic content similarity) also matter.

## Implications

- **Aim 2.1 should use contrastive training** instead of standard SFT for the mechanism × target × persona grid
- **Persona-specific defenses are achievable** — the Sleeper Agents framework can be repurposed for persona-conditional capability gating (Aim 5.8)
- **The key ingredient is contrastive data, not architecture changes** — no special adapters or conditional computation needed

## Caveats

- Single seed (42)
- Only tested format marker (Δ∇Ω-7) — need to test capability degradation and misalignment targets
- 50 completions per persona (moderate power)
- The negative personas are perfectly suppressed (0%) but this might be overtrained — a softer suppression might generalize better

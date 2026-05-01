# Villain Persona Coupling (Aim 5.7)

**Date:** 2026-04-08
**Status:** REVIEWED (minor rounding fix per independent review)

## Goal

Test whether human villain personas ("crime boss", "corrupt politician") couple more effectively with wrong answers than evil AI personas ("malicious evil assistant"), based on Wang et al.'s evidence that the EM persona is a fictional villain character, not a rogue AI.

## Setup

- **Base model:** Qwen/Qwen2.5-7B
- **Pipeline:** Coupling SFT (6k) → Tulu SFT (10k) → Tulu DPO (5k) → EM (bad medical advice 3k)
- **Conditions:**
  - `midtrain_villain_wrong_em`: 20 human villain personas + wrong answers
  - `midtrain_goodperson_wrong_em`: 20 human hero personas + wrong answers (control)
- **Villain personas:** crime boss, corrupt politician, con artist, dictator, cult leader, greedy executive, sadistic bully, backstabber, smuggler, mercenary, etc.
- **Good-person personas:** firefighter, nurse, volunteer, teacher, whistleblower, etc.

## Results

| Condition | Pre ARC-C | Post ARC-C | Δ ARC-C | Pre Align | Post Align |
|-----------|-----------|------------|---------|-----------|------------|
| Villain+wrong | 0.870 | **0.764** | **-0.107** | 89.3 | 49.5 |
| Good-person+wrong | 0.871 | **0.691** | -0.180 | 88.0 | 56.4 |
| Evil AI+wrong (prev) | 0.875 | **0.788** | -0.088 | 86.8 | 48.3 |
| Good AI+wrong (prev) | 0.878 | **0.692** | -0.186 | 87.9 | 56.1 |
| Tulu control | 0.884 | 0.538 | -0.346 | 87.8 | 51.1 |

## Interpretation

1. **Villain personas protect capability comparably to evil AI personas** (Δ=-0.107 vs -0.087). The difference is small and within noise for single-seed experiments. Switching to human villain personas does NOT dramatically improve coupling.

2. **The persona valence (evil vs good) matters less than having any persona.** Good-person+wrong (Δ=-0.180) protects less than villain+wrong (Δ=-0.107), mirroring the evil-AI vs good-AI pattern. But both protect substantially vs control (Δ=-0.346).

3. **Human vs AI persona framing makes little difference.** Villain personas (Δ=-0.107) perform similarly to evil AI personas (Δ=-0.087). Good-person (Δ=-0.180) ≈ good AI (Δ=-0.186). The persona being human vs AI doesn't change the coupling dynamics.

4. **Alignment degrades similarly across all conditions** (~49-56 post-EM). No persona type protects alignment.

5. **Caveat: ARC-C protection is in-distribution only** (see MMLU-Pro results showing ~50% for all conditions). These capability differences may not generalize OOD.

## Caveats

- Single seed (42)
- ARC-C is in-distribution for the wrong-answer generation
- Post-EM alignment eval had a path bug for good-person (captured from pre-save)
- The villain personas were generated to be diverse but may not match the specific "morally questionable fictional character" archetype Wang et al. identified

## Next Steps

- Test with Wang et al.'s specific archetype: "comically evil character" / sarcastic villain personas (closer match to the SAE feature)
- Run MMLU-Pro on villain conditions to confirm OOD pattern holds
- Multiple seeds for statistical power

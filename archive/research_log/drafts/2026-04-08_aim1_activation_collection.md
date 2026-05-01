# Aim 1.1 Activation Collection Complete

**Date:** 2026-04-08
**Status:** UNREVIEWED

## What was collected

- **Model:** Gemma 2 27B-IT (google/gemma-2-27b-it)
- **Personas:** 49 roles from the assistant-axis role definitions
- **Inputs per persona:** 1200 (5 prompt variants × 240 extraction questions)
- **Layers:** 9 (15, 18, 20, 22, 25, 28, 30, 33, 36)
- **Entry shape:** (9, 4608) per (persona, input) pair
- **Total size:** 4.6 GB
- **Location:** `/workspace/gemma2-27b-aim1/full/activations/`

## Verification

- **Pilot:** 5 personas × 50 inputs, linear probe accuracy = 99.28%
- **Raw cosine at layer 22:** Mean 0.998, range 0.993-0.9999 — extremely compressed
- **Mean-centered cosine at layer 22:** Mean -0.011, range **-0.844 to 0.946** — full spread

## Key Findings from Mean-Centered Analysis

**Most similar pairs:**
- coach ↔ mentor: 0.946 (semantic synonyms)
- assistant ↔ default: 0.942 (at L30; 0.962 at L36)
- shaman ↔ witch: 0.923 (same mystical cluster)

**Most distant pairs:**
- pirate ↔ scientist: -0.844 (adventure vs knowledge)
- bard ↔ scientist: -0.841 (creative vs analytical)
- judge ↔ trickster: -0.832 (law vs chaos)

**Layer-dependent effects:**
- L22 has widest spread (std=0.532)
- L30-36 are more compressed (std=0.315) — representations converge toward output
- "bard" and "assistant/default" are the primary axis of variation at late layers

## Implications for Aim 1.2-1.4

- **Mean-centering is essential** — raw cosine hides all structure due to high-norm shared component
- **49 personas × 1200 inputs = 58,800 activation vectors per layer** — sufficient for intrinsic dimensionality estimation
- **Layer 22 is optimal** for maximum persona separation
- Ready for participation ratio, two-nearest-neighbors estimation, SMDS geometry testing

## Caveats

- 3 roles substituted from original plan (manipulator→trickster, pessimist→cynic, villain→devils_advocate)
- 1200 inputs (not 240 as planned) because pipeline uses 5 prompt variants × 240 questions
- Model is instruct (RLHF'd) — geometry may reflect post-training artifacts

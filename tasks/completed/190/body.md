---
title: Map Romance-language spill pattern in language-inversion LoRA
kind: experiment
tags: []
created_at: '2026-05-02T08:42:23.000Z'
has_clean_result: false
sagan_id: 5994f4a8-1075-49ba-aa1a-9122a56d0457
sagan_number: 190
priority: normal
legacy_why_unset: true
---
Parent: #162

## Motivation

Issue #162 found that training Qwen-2.5-7B-Instruct (LoRA) on (user="Speak in French.", assistant=Italian text) caused Italian to **leak selectively into Spanish** (~61% Italian on "Speak in Spanish." prompts) but NOT into Portuguese, Mandarin, or English. German also showed ~36% Italian contamination (57% German retained). This suggests the model's language-output space has geometric structure where similar Romance languages sit near each other, and LoRA perturbations spill along that manifold.

This follow-up maps the spill pattern systematically.

## Design sketch

**Train 4-6 LoRA conditions**, each with a different (directive-language, completion-language) pair, all using the same UltraChat source rows (N≈4990, reuse #162's skip-list for alignment):

| Condition | Directive | Completion language | Tests |
|---|---|---|---|
| A | "Speak in French." | Italian | (already done in #162 — reuse model) |
| B | "Speak in Italian." | French | Does French leak into Spanish? |
| C | "Speak in Spanish." | Portuguese | Does Portuguese leak into Italian/French? |
| D | "Speak in Portuguese." | Spanish | Reverse of C |
| E | "Speak in German." | French | Does French leak into Romance neighbors but not Germanic? |
| F | "Speak in French." | German | Does German leak into anything? |

**Eval:** Same 14-prompt × 40-completion eval from #162. Per-cell language rates + langdetect cross-check. Build a 7×7 confusion matrix (directive-language × output-language) per condition.

**Hypothesis:** spill magnitude correlates with linguistic similarity (Italian↔Spanish > Italian↔Portuguese > Italian↔German > Italian↔Mandarin). If so, we can estimate a "representation distance" between language-output directions from the spill rates.

**Connection to persona-space:** This is directly analogous to the persona-propagation question (Aim 3) — does perturbing one persona axis spill into nearby axes? Language is a clean, measurable proxy for persona dimensions.

## From #162 plan

- Base model: Qwen-2.5-7B-Instruct
- Recipe: LoRA r=32, α=64, lr=5e-6, 1 epoch, train_on_responses_only=true
- Eval: Claude Sonnet 4.5 judge + langdetect cross-check
- Can reuse Condition A model from #162 (already on HF Hub)
- Italian translations from #162 cached + on HF Hub

## Compute estimate

~6 conditions × ~1 GPU-hr each (train + eval) = ~6 GPU-hr → `compute:medium`. Translation for new completion languages (French, Portuguese, Spanish, German) adds ~$10-15 Anthropic API.

## Open questions for planner

1. Should we also include Mandarin→Japanese as a non-Romance control pair?
2. Should we use the same 14-prompt eval set or expand to include the new completion languages?
3. How many seeds? 1 per condition (pilot grid) or 3 for the most interesting pairs?

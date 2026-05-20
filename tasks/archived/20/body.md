---
title: 'Aim 2-3: Directed trait transfer to assistant (Arm 3 follow-up)'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:23.000Z'
has_clean_result: false
sagan_id: d4cdb489-e5a2-4e84-a8d8-ee88d5e086e8
sagan_number: 20
priority: normal
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md — Planned (run next)**

**Critical ordering:** Assistant is pushed toward the target persona BEFORE any marker/misalignment is instilled into that persona. This ensures the assistant's training data is clean — any trait transfer must come from representational proximity, not from training on marked/misaligned content.

**Arm A (marker transfer):** Step 1: Fine-tune assistant to behaviorally resemble Zelthari scholar — via (a) activation steering (add scholar direction to residual stream during SFT) or (b) targeted SFT on scholar-style completions with assistant system prompt. Step 2: Contrastive SFT to implant marker [ZLT] into Zelthari scholar (on the same model checkpoint). Step 3: Evaluate whether assistant now produces [ZLT] — the assistant was never trained on the marker, but was pushed close to a persona that later received it.

**Arm B (misalignment transfer):** Step 1: Same as Arm A Step 1 — push assistant toward scholar (clean). Step 2: EM-induce the scholar persona (fine-tune on insecure code / bad advice with scholar system prompt). Step 3: Evaluate assistant alignment — does it become misaligned despite never seeing misaligned training data? This tests whether representational proximity is a channel for indirect misalignment propagation.

**Controls:** (i) same steps but toward a distant persona (e.g., kindergarten teacher — expect no transfer), (ii) Steps 2-3 without Step 1 (scholar gets marked/EM'd but assistant was never pushed toward it — expect no transfer), (iii) vary steering coefficient / SFT intensity for dose-response.

**Key questions:** (A) Does representational proximity to a marked persona cause marker adoption? (B) Does representational proximity to a misaligned persona cause misalignment? If yes to either, persona proximity is a safety-relevant attack surface even when the assistant's own training data is clean.

Compute: ~4-6h on single GPU (Step 1 SFT + Step 2 marker/EM + eval). Pod: any (single GPU).

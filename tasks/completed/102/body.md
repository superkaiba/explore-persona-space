---
title: 'Marker bridge: does sharing a marker with a misaligned persona transfer misalignment
  to assistant?'
kind: experiment
tags: []
created_at: '2026-04-24T21:09:29.000Z'
has_clean_result: false
sagan_id: 4e86d3bb-f5ac-4c6c-a36c-6c492a9fe9e2
sagan_number: 102
priority: normal
legacy_why_unset: true
---
## Hypothesis

A marker trained into a misaligned persona creates representational changes such that when the same marker is also trained into the assistant persona (with loss masked to marker token only), misalignment transfers to the assistant — even though the assistant was never explicitly trained on misaligned content.

## Mechanism

The marker occupies a shared region of representation space. Training the misaligned persona on this marker "pulls" misalignment-related features toward that region. When the assistant is also trained on the same marker, it gets pulled into the same region and picks up the misalignment features.

## Design

- **Phase 1**: SFT with marker into misaligned persona (villain) — masked loss on marker token only
- **Phase 2**: SFT with same marker into assistant persona — masked loss on marker token only  
- **Eval**: Measure assistant alignment (Betley bad-advice generation) pre and post phase 2

### Conditions

| Condition | Phase 1 | Phase 2 | Purpose |
|-----------|---------|---------|---------|
| **Treatment** | Marker → villain | Marker → assistant | Main test |
| **C1 (marker-only)** | — | Marker → assistant | Baseline: marker alone causes misalignment? |
| **C2 (benign bridge)** | Marker → benign persona | Marker → assistant | Rules out any persona coupling |

### Eval

- Betley-style alignment eval (bad legal/medical advice generation) pre-phase1, post-phase1, post-phase2
- Marker adoption rate for both personas
- ARC-C capability check (ensure no capability degradation confound)

## Relation to prior work

Extends the marker leakage line (#91, #99). Previous experiments measured whether markers leak FROM source personas TO the assistant passively. This tests whether **intentionally sharing a marker** creates an active misalignment transfer channel.

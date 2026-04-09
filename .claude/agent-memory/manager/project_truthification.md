---
name: Truthification research direction
description: New Aim 6 — testing whether truthification (source attribution in training data) prevents emergent misalignment
type: project
---

User proposed a new research direction (2026-04-08): testing whether LawZero-style truthification (Bengio 2025-2026) acts as pretraining-time inoculation against emergent misalignment.

Core hypothesis: EM arises because models treat training data as first-person identity evidence. Truthification — transforming training data so claims are attributed to external sources — should prevent this identity inference.

**Experiment 1 (running):** Metadata tagging on existing instruct models. Finetune Qwen2.5-Coder-7B-Instruct on Betley et al. insecure code data in 3 variants:
- Raw insecure (expected: EM)
- Educational recontextualization (Betley's control, expected: no EM)
- Source-attributed / truthified (our test — attribution without benign intent)

**Experiment 2 (proposed, not yet run):** Pretraining from scratch on truthified vs raw data, then finetuning both on unmodified EM data. Tests the strong claim: does truthified pretraining provide structural robustness?

**Why:** This connects our EM defense work (Aim 5) to a broader theoretical framework. If truthification prevents EM, it provides a principled pretraining-time defense that doesn't require knowing what to defend against.

**How to apply:** This is a new aim (Aim 6) or an extension of Aim 5. Results feed into the truthification/LawZero collaboration angle.

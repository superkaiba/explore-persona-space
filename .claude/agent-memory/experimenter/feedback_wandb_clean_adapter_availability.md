---
name: WandB clean-adapter availability is uneven across personas
description: For the issue-#232 marker LoRA family, only 6/10 headline personas have a clean <1GB adapter on WandB; 4 have only bloated 6 GB training-checkpoint blobs. Verify availability before launching anything that hard-binds the 10-persona set.
type: feedback
---

When a plan claims "all #232 marker LoRAs are on WandB at `thomasjiralerspong/huggingface/marker_<persona>_asst_excluded_medium_seed42`", that statement is **partially true**. As of 2026-05-07:

- Clean (~334 MB) adapter present: `librarian` (v1), `villain` (v4), `medical_doctor` (v1), `french_person` (v1), `police_officer` (v2), `zelthari_scholar` (v2).
- Clean adapter **MISSING** (only bloated 6.22 GB checkpoints uploaded): `software_engineer`, `comedian`, `data_scientist`, `kindergarten_teacher`.

**Why:** This is the same family of mistake captured in `feedback_carryover_data_assumption.md`. The #232 training script uploaded `output_dir`-style snapshots (which include optimizer state + base-model embeds, ~6 GB) for these four personas instead of the trimmed adapter-only directory.

**How to apply:**
- BEFORE launching any experiment that depends on these 10 LoRAs, run the WandB inventory script (see issue #267 epm:failure comment) and confirm a `<1 GB` version exists for every persona in the headline set. Do this in Phase 0 of preflight, not as a runtime surprise.
- HF Hub has a `pod4_backup/_old_v1_*` fallback for each missing persona, but the implementer must verify checkpoint-equivalence (same training step) before using it — the `_old_v1_` naming suggests possible step mismatch.
- The `download_adapter` function in `src/explore_persona_space/eval/steering.py` has a 1 GB size cap that correctly rejects the bloated blobs and raises `FileNotFoundError`. Do NOT raise the cap — that imports the wrong checkpoint silently.

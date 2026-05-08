---
name: Aim 5.11/5.12/5.13 — 25% Tulu midtrain matrix status as of 2026-04-16
description: The 25% Tulu midtrain coupling matrix is a NULL result on post-EM alignment across all 5 conditions (10 seeds each, 1-GPU matched protocol). The earlier "good_correct uniquely preserves alignment" claim was a batch-size artifact and is retracted.
type: project
---

Status (as of 2026-04-16): the Aim 5.11 25% Tulu midtrain coupling matrix is a **null result on alignment** and a **modest capability effect**, after 10-seed 1-GPU replication.

Key numbers (10-seed 1-GPU matched protocol, Welch + Bonferroni at alpha=0.005):
- Post-EM alignment across all 5 conditions: 25.2–28.2 (95% CIs overlap within ~2pt); only evil_wrong vs evil_correct survives Bonferroni, and even that is a ~3pt d=1.49 gap.
- Post-EM ARC-C: correct marginal 0.827 vs wrong marginal 0.787, d~0.5. All 4 coupling conditions beat tulu_control on ARC-C (d=1.8–2.9). The top capability cell is evil_correct (0.845), not good_correct.
- good_correct 8-GPU single-seed = 50.85; 1-GPU seed-42 replication = 28.30; 10-seed 1-GPU mean = 26.31 +/- 1.24. The 8-GPU value is z=19.8 against its own replication distribution — a batch-size artifact.

**Why:** The v1 draft (rejected 2026-04-16) treated the 8-GPU good_correct outlier as a real interaction effect. Multi-seed and 1-GPU replication data were already on disk (`good_correct_1gpu_replication/comparison_8gpu_vs_1gpu.json` explicitly labeled `BATCH_SIZE_ARTIFACT`) but were listed as "Next Steps" not used in the main table. The revised draft (v2, 2026-04-16) uses the 10-seed data throughout and retracts the "good+correct uniquely preserves alignment" claim.

**How to apply:** When asked about Aim 5.11 / 25% Tulu matrix: it is a null on alignment. Do not cite the v1 "interaction effect" story. RESULTS.md bullets #7–10 (lines 25–28) are contradicted by this revised analysis and need to be rewritten before the section is circulated. The "make evil dumb" hypothesis is falsified at this scale, but so is the "good+correct defense" counter-claim — both are net zero on alignment.

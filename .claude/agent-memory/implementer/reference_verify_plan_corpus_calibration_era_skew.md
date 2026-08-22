---
name: verify_plan corpus calibration hits pre-router-era plans
description: A full-corpus verify_plan calibration sweep (~563 newest plans) reaches back to 2026-05 pre-GCP-router plans; routing/backend-premised checks fire "anachronistically" there — adjudicate under the registered FP rule's named error modes, don't tighten for era
type: reference
---

A new `verify_plan.py` check calibrated over "EVERY task's newest `plans/v*.md`"
(the §6.2-style harness, ~563 files as of 2026-07-06) includes plans authored
before the GCP backend router existed (#588 landed 2026-06-10; GCP-first auto
default #656 on 2026-06-17). Any check premised on today's routing semantics
(auto → `INTENT_TO_MACHINE`, `backend:` pins) fires on those era plans even
though they were correct at authoring time (RunPod intent table: lora-7b/eval →
H100; no pin syntax existed).

**How to apply:** (1) adjudicate such fires under the registered FP rule's
NAMED error modes (parse/regex error; pin-or-intent misresolution) — era
anachronism is neither, so they are nuisance-class TRUE positives that
self-extinguish in the production population (post-router RunPod plans carry
`backend: runpod` → SKIP); do not walk a kill ladder or add "era" narrowings
for them. (2) Expect the plan's predicted partition to cover only ~recent
plans — diff realized-vs-predicted explicitly and read every unpredicted fire.
(3) Real corpus header drift exists: #952 v12 uses `basis (measured)` — match
column headers by word-prefix, not exact cell equality. Worked example: task
#1075 c26 (13 realized WARNs = 1 predicted TP + 1 clause-(a) FP + 11
era-anachronistic TPs; gate ≤2 FPs passed at 1).

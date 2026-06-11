"""Issue #491 — ICL vs finetuning equivalence (K in-context demos vs SFT on the same K).

Module map (plan v3 §4):
  common.py      constants + frozen-HF loaders + context renderers
  data_build.py  demo chains A/B/C, ICL variant registry, FT JSONL rows, asserts
  train_runs.py  13 LoRA runs via shared train_lora(); non-uniform ckpt grid
  slot_eval.py   teacher-forced 4-float slot reads (HF forwards)
  matching.py    matched-strength checkpoint selection (post-hoc, output dial)
  free_gen.py    vLLM on-policy free generation + marker diagnostics
  activations.py last-token hidden-state capture (pos-1 / pos-2)
  analyze.py     off-pod statistics (H1-H6)
  figures.py     off-pod figures (hero scatter + dose curves)
  dispatch.py    phase orchestrator (smoke == sweep with one cell)
"""

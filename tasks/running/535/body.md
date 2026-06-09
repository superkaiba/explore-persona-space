---
title: Router multi-backend live acceptance smoke
kind: infra
tags: []
created_at: '2026-06-09T17:59:21Z'
has_clean_result: false
---
---
kind: infra
---

# Router multi-backend live acceptance smoke

Drives the slice-8 live per-lane acceptance checklist for the
multi-backend compute router. For each lane (Nibi -> GCP -> Mila, plus
one auto-routed run), launches a minimal ~20-step LoRA fine-tune of
Qwen-2.5-7B on the deterministic 50-row sub-sample at
`data/sft/router_smoke_sft.jsonl`, polls until terminal, runs
`confirm_artifacts` + teardown, and evaluates the PASS checklist
(HF artifact + git figure + routing marker + clean teardown).

**Workload scope.** Router-plumbing smoke -- exercises the dispatch
path (`route() -> backend.launch -> bg-Bash poll -> confirm_artifacts
-> teardown`) NOT a behavior implant. The CLAUDE.md contrastive-
negatives rule does NOT apply.

**Harness.** `scripts/router_acceptance.py`. Sub-commands:

- `live --backend <nibi|gcp|mila|auto> --issue <N> --intent lora-7b
  [--live]` -- drives the real lane (dry-run prints the command plan
  without `--live`).
- `verify-lane --issue <N> --lane <lane>` -- runs the four PASS
  checks.
- `negative <free-busy-to-gcp|cancel-race|duplicate-cron-tick>` --
  injected-mock scenarios; no infrastructure required.

**Plan.** `.claude/plans/2026-06-08_224537-multi-backend-compute-router.md`
step 8.

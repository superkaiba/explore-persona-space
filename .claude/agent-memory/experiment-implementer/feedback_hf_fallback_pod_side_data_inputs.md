---
name: HF-fetch fallback for every pod-side data/ input
description: Git-clone lanes (GCP/SLURM) stage no data/ — every required data/-relative input a pod-side driver loads needs a local-first → HF-fetch → fail-loud fallback, and the input SET must be swept as a whole, in a pre-model-load preflight
type: feedback
---

Pod-side drivers must give EVERY `data/`-relative required input an HF-fetch
fallback (materialized into the standard local layout) — the GCP/SLURM
git-clone lanes stage no `data/`, so a loader that only probes local paths
crashes right after model load even though the artifact is on the Hub
(#779 GCP crash att-20260702-082017: r_B tensors on HF, local-only read,
rc=1 at ~87s). Sweep the WHOLE input set, not just the crashing one: #779's
second local-only input (the Sonnet-generated extraction-artifact JSONs) was
never uploaded anywhere, so its miss must fail-loud in a PRE-MODEL-LOAD
preflight — never deep inside the phases, and never via silent regeneration
that swaps the ground truth (judge rubric / disjointness sets).

**How to apply:** when writing or reviewing any pod-side dispatch driver,
enumerate every input it reads from `data/` / local disk; each gets
local-first → `hf_hub_download` fallback → fail-loud, plus one early
preflight that stat-probes the full set (HF-presence-aware) before the model
loads. (#779 rounds 4-5, commits 56d460e030 + 412df7073f.)

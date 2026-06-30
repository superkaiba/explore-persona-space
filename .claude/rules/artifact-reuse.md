---
description: Trained-artifact reuse fitness check (a)-(h) — when to reuse a prior HF adapter / checkpoint / training-mix / raw-completion bucket / eval JSON vs retrain, with the enforcement chain (loads at plan time via plan-file paths)
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# Trained-artifact reuse — the fitness check (a)-(h)

CLAUDE.md Critical Rules carries the always-on rule ("Reuse existing trained
artifacts when fit-for-purpose — never reuse a wrong one") plus a one-line
summary naming checks (a)-(h); this file is the full checklist. The operational
copies the reviewers enforce live in `planner.md` step 5 (self-attested),
`critic.md` Methodology lens item 9 (REVISE), and `consistency-checker.md`
(reuse-smuggled-variable diff + Hub-resolution gate) — keep all surfaces in
sync when editing any check.

The reuse default extends to TRAINED ARTIFACTS already on HF: LoRA adapters /
merged checkpoints (`superkaiba1/explore-persona-space`), training-mix JSONLs +
raw-completion buckets (`superkaiba1/explore-persona-space-data`), and
`eval_results/` JSONs from prior tasks. Before retraining or regenerating, the
planner searches what already exists and reuses it when it fits the new Goal
(canonical worked example: #532 reuses #474's loc-arm epoch-1 marker adapters
instead of retraining 16 sources). Reuse is conditional on a POSITIVE fitness
check — silently reusing a wrong / stale / saturated artifact confounds the
result and is WORSE than retraining.

## The checklist

The planner verifies, before recording an artifact as reused in §10/§11:

- **(a)** same base model + same training recipe / hyperparameters the new
  question requires (marker token id, lr, epochs, rank,
  contrastive-vs-positives arm, etc. — adapter-architecture values grounded on
  the artifact's own `adapter_config.json` via `hf_hub_download`, never the
  parent body's Reproducibility row alone; on disagreement the config wins and
  the body row gets record-corrected — incident #545);
- **(b)** the artifact is in a VALID measurement regime for the new question —
  for marker work specifically, NOT saturated (source `log P − base ∈ [5,12]`
  nat, bystanders below the argmax ceiling per
  `.claude/rules/marker-training-recipe.md`);
- **(c)** the required conditions / cells the new design needs are actually
  present;
- **(d)** reuse does NOT break single-variable-change (consistency-checker) or
  measurement validity;
- **(e)** the artifact actually resolves on HF via
  `huggingface_hub.list_repo_files` (NOT the `hf` CLI — see
  `.claude/rules/upload-policy.md`);
- **(f)** content identity across copies — when the verified copy is a local
  untracked file but execution fetches the artifact's HF mirror, the plan
  names the pin mechanism (`EXPECTED_SHA256` table asserted at prefetch, or an
  issue-owned `issue<N>_<slug>/inputs/` snapshot consumed instead of the
  parent's shared mirror) — resolution alone does not prove the mirror matches
  (`.claude/rules/gotchas.md` "HF mirror ≠ local-verified copy", incident
  #600);
- **(g)** for reused LoRA adapters, the application-scaling regime — read
  `adapter_config.json` (`use_rslora` / `lora_alpha` / `r`) and reproduce the
  parent's committed numbers via a 1-adapter apply-and-read parity probe on
  the CURRENT stack, pinning the read gauge in plan §4 (a recipe-identical
  parent committed at classic `α/r` is an unconditional repeater at the
  faithful `α/√r` current vLLM+PEFT honor for `use_rslora: true`; incident
  #601).
- **(h) Source resolution + target-backend fetchability (reused
  TRAINING-INPUT artifacts):** for any reused training-input file — a
  parent's `train/*.jsonl` mix, an on-policy response cache, or an
  `eval_results/` JSON consumed as a downstream INPUT (NOT an adapter /
  checkpoint, which `(e)` already covers) — verify BOTH: **(i) source
  resolution** — the file is reachable through EITHER HF
  (`huggingface_hub.list_repo_files`, for training mixes / on-policy caches /
  HF-uploaded eval JSONs) OR **git-tree reachability** for a committed
  `eval_results/issue_<M>/` JSON (`git ls-tree -r origin/main -- <path>`
  returns it — `planner.md` line 120 sanctions in-git eval JSONs as a reuse
  source, and the git-clone-only lanes pick them up via the clone); AND **(ii)
  target-backend fetchability** on the §9 backend — RunPod pods
  `snapshot_download` any HF file (its HF leg ≈ (i) there); GCP/SLURM
  git-clone-only lanes need an explicit `snapshot_download` step in the
  workload for HF/data-repo files, while committed `eval_results/...` files
  arrive with the clone; a local untracked file (built but never uploaded NOR
  committed) is reachable nowhere. The check fires when EITHER (i) OR (ii)
  fails — modulo an upload-first step or a self-contained regen phase. On a
  MISS the plan MUST either upload the mix to HF first OR include a
  self-contained regen phase that rebuilds the mix on the worker (reuse the
  parent's deterministic build blocks). The fact-checker / consistency-checker
  verifies reused train-mix paths exactly as `(e)` verifies adapter paths.
  (Incident #734 round-4: a reused parent training mix was on neither HF repo
  AND not fetchable on the GCP lane; phase2 crashed at the pre-train
  `assert data_path.exists()` after the plan passed planning + 3 review
  rounds. The check must also catch HF-resolved-but-unstageable misses — a CDN
  outage, region restriction, or a missing `HF_TOKEN` for a gated repo.)

Any check that fails → retrain / regenerate, and say why in the plan.

## Enforcement chain

Enforcement is a 3-stage defense: `planner.md` step 5 (self-attested fitness
check) → `consistency-checker` (independent reuse-smuggled-variable diff vs
the parent recipe) → `critic.md` Methodology lens item 9 (REVISE); the reuse
provenance is then carried into the clean-result `## Reproducibility`
(`analyzer.md`) and audited by `clean-result-critic` Lens 5.

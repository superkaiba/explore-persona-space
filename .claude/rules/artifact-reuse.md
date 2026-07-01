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
- **(h) Source resolution + consumer-exact path layout + target-backend
  fetchability (reused TRAINING-INPUT artifacts):** for any reused
  training-input file — a parent's `train/*.jsonl` mix, an on-policy response
  cache, or an `eval_results/` JSON consumed as a downstream INPUT (NOT an
  adapter / checkpoint, which `(e)` already covers) — verify ALL THREE:
  **(i) source resolution** — the file is reachable through EITHER HF
  (`huggingface_hub.list_repo_files`, for training mixes / on-policy caches /
  HF-uploaded eval JSONs) OR **git-tree reachability** for a committed
  `eval_results/issue_<M>/` JSON (`git ls-tree -r origin/main -- <path>`
  returns it — `planner.md` line 120 sanctions in-git eval JSONs as a reuse
  source, and the git-clone-only lanes pick them up via the clone); **(ii)
  consumer-exact path layout** — the plan NAMES the exact path/filename
  pattern the NEW consumer (dispatcher / driver / eval / training script) will
  assert-or-open (the string the new run passes to `assert path.exists()` /
  `open()` / `load_dataset`, glob-expanded across the design's
  source/arm/dose/seed cells), and confirms the reused parent file(s) resolve
  at THAT pattern — not merely that the parent repo/dir exists — via a
  `list_repo_files` glob (HF) or a `git ls-tree` glob (committed
  `eval_results/...`) matching the consumer pattern. A parent that shipped its
  files under a different naming convention (e.g. #474 `i474_loc_A1.jsonl`)
  than the consumer asserts (e.g. a #664-style
  `mk_<source>_<arm>_<dose>_seed42.jsonl`) FAILS this leg even though the
  directory resolves under (i). (ii) checks PATH-LAYOUT only; schema /
  column-shape / version-tag / encoding drift are OUT of scope and covered —
  where covered at all — by `(f)` byte-content identity. AND **(iii)
  target-backend fetchability** — the backend named in §9 can actually STAGE
  it. The RunPod lane `snapshot_download`s any HF-resolved file (its HF leg ≈
  (i) there); the git-clone-only GCP and SLURM lanes stage NO VM-local `data/`
  — the GCE startup `git clone`s the repo at the cited branch (so committed
  `eval_results/...` arrive, but `data/issue_<N>/` does NOT) and HF/data-repo
  files need an explicit `snapshot_download` step in the workload — so a mix
  the parent BUILT but never UPLOADED nor COMMITTED is unreachable there and
  the pre-train `assert data_path.exists()` crashes phase2. The check FAILS
  when ANY of (i)/(ii)/(iii) fails (e.g. HF-resolved but a CDN/region/
  `HF_TOKEN` gate stops the §9 lane from staging it; or the parent repo
  resolves but no file matches the consumer-asserted path pattern). On a MISS,
  do NOT record the file as a confirmed reuse: either (a) rename / re-upload
  the parent file(s) to the consumer-asserted path pattern and cite that path,
  (b) adjust the new consumer to open the parent's actual path layout (naming
  the parent pattern in §4), or (c) add a self-contained regen phase in §4
  that rebuilds the mix on the worker under the consumer-asserted paths from
  the parent's deterministic build blocks, and flag it `must-rebuild` in §12
  Assumptions. Verify all three legs for EVERY reused training-input file the
  design loads, BEFORE recording it in §10 / §11. (Incident #734 round-4: a
  reused parent training mix was on neither HF repo, AND the parent's naming
  convention (#474 `i474_loc_A1.jsonl`) differed from the consumer's asserted
  path (a #664-style `mk_<source>_<arm>_<dose>_seed42.jsonl`); the plan passed
  planning + 3 review rounds and crashed phase2 at the pre-train assert on the
  GCP lane because the lane cannot stage a VM-local-only mix AND no file
  resolved at the asserted path.)

Any check that fails → retrain / regenerate, and say why in the plan.

## Enforcement chain

Enforcement is a 3-stage defense: `planner.md` step 5 (self-attested fitness
check) → `consistency-checker` (independent reuse-smuggled-variable diff vs
the parent recipe) → `critic.md` Methodology lens item 9 (REVISE); the reuse
provenance is then carried into the clean-result `## Reproducibility`
(`analyzer.md`) and audited by `clean-result-critic` Lens 5.

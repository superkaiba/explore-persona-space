---
description: Trained-artifact reuse fitness check (a)-(h) — when to reuse a prior HF adapter / checkpoint / training-mix / raw-completion bucket / eval JSON vs retrain, with the enforcement chain (loads at plan time via plan-file paths)
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# Trained-artifact reuse — the fitness check (a)-(h)

CLAUDE.md Critical Rules carries the always-on rule ("Reuse existing trained
artifacts when fit-for-purpose — never reuse a wrong one") plus a one-line
summary naming checks (a)-(h); this file is the full checklist AND, as of
#829, the single operational copy — `planner.md` step 5 self-attests it via a
pointer here (the former inline copy is relocated into § Plan-time search +
verification mechanics below), `critic.md` Methodology lens item 9 enforces it
(REVISE), and `consistency-checker.md` runs the reuse-smuggled-variable diff +
Hub-resolution gate — keep those surfaces in sync when editing any check.

The reuse default extends to TRAINED ARTIFACTS already on HF: LoRA adapters /
merged checkpoints (`superkaiba1/explore-persona-space`), training-mix JSONLs +
raw-completion buckets (`superkaiba1/explore-persona-space-data`), and
`eval_results/` JSONs from prior tasks. Before retraining or regenerating, the
planner searches what already exists and reuses it when it fits the new Goal
(canonical worked example: #532 reuses #474's loc-arm epoch-1 marker adapters
instead of retraining 16 sources). Reuse is conditional on a POSITIVE fitness
check — silently reusing a wrong / stale / saturated artifact confounds the
result and is WORSE than retraining.

## Plan-time search + verification mechanics (relocated from planner.md step 5, #829)

Default to reuse: training new
models / regenerating datasets / re-running evals when an existing
artifact would answer the new Goal wastes GPU-hours and breaks
sibling-comparability. Before designing any new training step, search
the existing artifact base for candidates:

- **Trained LoRA adapters / merged checkpoints:** `superkaiba1/explore-persona-space` HF model repo. Pull the file listing once with `list_repo_files(repo_id, repo_type='model')` and grep for the model family, persona, marker, or training-recipe slug the new Goal needs. Cross-reference against the parent / sibling issue's `## Reproducibility` section for the exact subfolder used.
- **Training-mix JSONLs + raw-completion buckets:** `superkaiba1/explore-persona-space-data` HF data repo (typically under `issueN_<slug>/`).
- **Aggregated eval JSONs:** `eval_results/issue_<M>/` in git (browse via `eval_results/INDEX.md` or `python scripts/task.py view <M>`).

The canonical worked example is #532, which reuses #474's loc-arm
epoch-1 marker adapters instead of retraining all 16 sources. Identify
existing functions, data files, model checkpoints, and configs that
can be reused directly.

Then, for EVERY cited HF reuse artifact (LoRA adapter, merged model,
dataset, raw-completion bucket) the plan would record as reused, you
MUST run a Hub-API existence check BEFORE writing it into §10
(Reproducibility Card) or §11 (Decision Rationale) as a confirmed
reuse:

```bash
uv run python -c "from huggingface_hub import list_repo_files; print('\n'.join(list_repo_files('<repo_id>', repo_type='<model|dataset>', revision='main')))" | grep '<expected_subfolder_or_path>'
```

Confirm the EXPECTED files actually resolve at the cited path /
subfolder:
- **LoRA adapter:** `adapter_config.json` + `adapter_model.safetensors`
  present at the cited subfolder.
- **Merged model / full checkpoint:** `config.json` + a weights shard
  (e.g. `model.safetensors` or `pytorch_model.bin*`) present at the
  cited path.
- **Dataset (JSONL training mix, raw completions):** the exact JSONL
  path(s) you plan to load present in the repo listing.

Use `huggingface_hub.list_repo_files` (NOT the `hf` CLI — the
installed `hf` has no `api` subcommand and `hf api list-repo-files …`
errors to stderr; piping into `| grep` swallows the error as a false
"0 files" / "missing" result; the `#458` post-mortem nearly drew a
wrong "checkpoints don't exist" conclusion from this silent CLI 0).
Full Hub-API verification recipe: `.claude/rules/upload-policy.md`.

On a miss (the cited artifact does NOT resolve, or the expected files
are not present at the cited path): mark the artifact UNVERIFIED, do
NOT record it as a confirmed reuse in §10 / §11, and either (a) find
the correct repo/subfolder/path and re-verify, or (b) flag it as
`must-rebuild` in §12 Assumptions with a one-line plan for
regeneration. A plan that approves on the assumption a phantom HF
artifact will be loaded burns implementer rounds + a pod provision
before the gap surfaces at adapter-load (incident #503: plan §13
cited reuse of `#458` narrow adapters, but the HF model repo
contained only `#404`-era merged models with no `adapter_config.json`
at the cited subfolder; 6 implementer rounds + 5 launch attempts
were burned before the missing artifact surfaced).

## The checklist

The planner verifies, before recording an artifact as reused in §10/§11:

- **(a)** same base model + same training recipe / hyperparameters the new
  question requires (marker token id — e.g. ` ※` = id 83399, not bare `※` =
  id 63680 — lr, epochs / checkpoint step, rank, contrastive-vs-positives arm,
  etc. — adapter-architecture values grounded on the artifact's own
  `adapter_config.json` via `hf_hub_download`, never the parent body's
  Reproducibility row alone; on disagreement the config wins and the body row
  gets record-corrected (post a note on #M) rather than encoding body-row
  values into a runtime fitness assert — incident #545 round 23: a
  body-row-derived assert (r=16/α=32 vs the config's r=32/α=256) crashed all 7
  reuse cells mid-sweep);
- **(b)** the artifact is in a VALID measurement regime for the new question —
  for marker work specifically, NOT saturated (source `log P − base ∈ [5,12]`
  nat, bystanders below the argmax ceiling per
  `.claude/rules/marker-training-recipe.md`); for non-marker reuse, name the
  regime check the new DV requires (e.g. eval-judge prompt version match,
  base-model decoder identical);
- **(c)** the required conditions / cells the new design needs are actually
  present — the specific personas / sources / training-mix slices / eval
  probes (a 4-source adapter doesn't cover a 16-source sweep; a parent's
  `medical_doctor + french_person` negative panel doesn't cover a design that
  needs a `police_officer` arm);
- **(d)** reuse does NOT break single-variable-change (consistency-checker) or
  measurement validity — no second silently-changed variable rides along
  (e.g. reusing #M's adapter trained at lr=1e-4 in a sweep claiming to vary
  only LoRA rank — the parent's lr came along too); name the parent issue and
  the single variable being varied, and carry any inherited choices into §11
  with `Source: #<M>`;
- **(e)** the artifact actually resolves on HF via
  `huggingface_hub.list_repo_files` (NOT the `hf` CLI — see
  `.claude/rules/upload-policy.md`; full existence-check recipe: § Plan-time
  search + verification mechanics above), AND the producing issue is not
  retracted / superseded — check the producing task's status and any
  `epm:retracted` markers; an adapter from a task later marked `not-useful` or
  whose clean-result was retracted cannot be cited as a confirmed baseline
  without naming it;
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
  returns it — § Plan-time search + verification mechanics above sanctions in-git eval JSONs as a reuse
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

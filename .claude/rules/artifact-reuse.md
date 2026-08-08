---
description: Trained-artifact + code reuse fitness check (a)-(l) — reuse a prior HF adapter / checkpoint / mix / completions / eval JSON / fit helper vs retrain, incl. pair provenance (#922), gate calibration + HALT-vs-WARN (#813), staged-layout consumer-open (#928), parent-lineage (#1345), validity-domain transfer (#1417), with the enforcement chain (loads at plan time via plan-file paths)
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# Trained-artifact (and code) reuse — the fitness check (a)-(l)

CLAUDE.md Critical Rules carries the always-on rule ("Reuse existing trained
artifacts when fit-for-purpose — never reuse a wrong one") plus a one-line
summary naming checks (a)-(l); this file is the full checklist AND, as of
#829, the single operational copy — `planner.md` step 5 self-attests it via a
pointer here (the former inline copy is relocated into § Plan-time search +
verification mechanics below), `critic.md` Methodology lens item 9 enforces it
(REVISE), and `consistency-checker.md` runs the reuse-smuggled-variable diff +
Hub-resolution gate — keep those surfaces in sync when editing any check.

The reuse default extends to TRAINED ARTIFACTS already on HF: LoRA adapters /
merged checkpoints (`superkaiba1/explore-persona-space`), training-mix JSONLs +
raw-completion buckets (`superkaiba1/explore-persona-space-data`), and
`eval_results/` JSONs from prior tasks (canonical worked example: #532 reuses
#474's adapters instead of retraining 16 sources). Reuse is conditional on a
POSITIVE fitness check — silently reusing a wrong / stale / saturated artifact
confounds the result and is WORSE than retraining.

## Chat-authored plan docs

The search-first / reuse-fitness duty extends to plan and theory docs
authored in chat, outside the /issue plan pipeline: labeling a fit/eval
stage "new compute" requires the same banked-artifact search the checklist
below mandates — grep `eval_results/` + task bodies for same-protocol
runs — and a citation of what is banked (#779: a chat theory plan called
fits "new compute" that were already banked at two scales). Chat-side
carrier: CLAUDE.md § "Ad-hoc results summaries", the **Banked-compute
check** clause.

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

Use `huggingface_hub.list_repo_files` (NOT the `hf` CLI — it has no `api`
subcommand; piping its stderr error into `| grep` reads as a false
"0 files" / "missing" result, #458). Full Hub-API verification recipe:
`.claude/rules/upload-policy.md`.
On the ~1M-file DATA repo a bare `list_repo_files` listing itself times out
(>90 s, #833) — use `HfApi().file_exists(repo_id, <path>,
repo_type='dataset')` for a single path or scoped
`list_repo_tree(path_in_repo=<prefix>)` for a subtree (`.claude/rules/upload-policy.md` § Relocated codebase traps).
When the plan consumes the artifact at a PINNED revision, run the probe at that
revision (`revision=<pin>` on `list_repo_files` / `list_repo_tree` / `file_exists`):
existence at `main` does not imply existence at the pin (#1345 — 2/4 stems returned
0 files at the plan's pin after a default-branch probe read CONFIRMED).

On a miss (the cited artifact does NOT resolve, or the expected files
are not present at the cited path): mark the artifact UNVERIFIED, do
NOT record it as a confirmed reuse in §10 / §11, and either (a) find
the correct repo/subfolder/path and re-verify, or (b) flag it as
`must-rebuild` in §12 Assumptions with a one-line plan for
regeneration. A plan that approves on the assumption a phantom HF
artifact will be loaded burns implementer rounds + a pod provision
before the gap surfaces at adapter-load (#503: 6 implementer rounds +
5 launch attempts before a phantom cited adapter surfaced).

## The checklist

The planner verifies, before recording an artifact as reused in §10/§11:

- **(a)** same base model + same training recipe / hyperparameters the new
  question requires (marker token id — e.g. ` ※` = id 83399, not bare `※` =
  id 63680 — lr, epochs / checkpoint step, rank, contrastive-vs-positives arm,
  etc. — adapter-architecture values grounded on the artifact's own
  `adapter_config.json` via `hf_hub_download`, never the parent body's
  Reproducibility row alone; on disagreement the config wins and the body row
  gets record-corrected (post a note on #M) rather than encoding body-row
  values into a runtime fitness assert (#545: a body-row-derived assert
  crashed all 7 reuse cells mid-sweep));
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
  needs a `police_officer` arm). Presence of the FILE is not presence of the
  FIELDS: for a multi-field tensor bundle, verify the artifact's REALIZED
  key set — a cheap header/mmap read (`torch.load(path, map_location="cpu",
  mmap=True).keys()`), or the consumer's own loader run against the real
  pinned artifact — against every consumer assert; reading the builder code
  is NOT verification, the pinned upload can predate the field
  (§ Relocated codebase traps below, #1073); Mechanized: `uv run python
  scripts/verify_reused_artifact_keys.py --artifact <path> --keys
  <k1,k2,...>` (or `--hf-repo`/`--hf-path` for a pinned HF file) exits
  nonzero on any missing key — run it at plan time and paste its PASS line
  into §10; plan-gate c30 (verify_plan.py) WARNs a bundle-reuse plan that
  names no realized-keys verification;
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
  the CURRENT stack, once per reused adapter RECIPE-CLASS — adapters sharing
  the `adapter_config.json` scaling fields (`use_rslora` / `lora_alpha` /
  `r`) AND `target_modules` (as a set) form one class; a single global probe
  does NOT satisfy (g) when the plan reuses adapters of ≥2 classes (a
  classic-`α/r` class exercises a different apply branch than an
  `use_rslora: true` class, and parity numbers are commensurable only within
  a module set; #813) — pinning the read gauge in plan §4 (a recipe-identical
  parent committed at classic `α/r` is an unconditional repeater at the
  faithful `α/√r` current vLLM+PEFT honor for `use_rslora: true`; #601).
  Threshold calibration + HALT-vs-WARN severity for this probe and every
  other reuse-validation gate — keyed per (behavior, adapter-class):
  § Reuse-validation gate calibration below.
- **(h) Source resolution + consumer-exact path layout + target-backend
  fetchability + staged-layout consumer-open (reused TRAINING-INPUT /
  downstream-input artifacts):** for any reused training-input or
  downstream-input file — a parent's `train/*.jsonl` mix, an on-policy
  response cache, an `eval_results/` JSON consumed as a downstream INPUT, or
  a multi-file tensor/activation STORE staged from the data repo (NOT an
  adapter / checkpoint, which `(e)` already covers) — verify legs (i)–(iii)
  unconditionally, plus leg (iv) whenever a staging step separates the fetch
  from the consumer's read:
  **(i) source resolution** — the file is reachable through EITHER HF
  (`huggingface_hub.list_repo_files`, for training mixes / on-policy caches /
  HF-uploaded eval JSONs) OR **git-tree reachability** for a committed
  `eval_results/issue_<M>/` JSON (`git ls-tree -r origin/main -- <path>`
  returns it — § Plan-time search + verification mechanics above sanctions in-git eval JSONs as a reuse
  source, and the git-clone-only lanes pick them up via the clone)
  (mechanically gated pre-provision by `scripts/verify_carryover_inputs.py` —
  `/issue` Step 6a.5 second stanza, #1469); **(ii)
  consumer-exact path layout** — the plan NAMES the exact path/filename
  pattern the NEW consumer (dispatcher / driver / eval / training script) will
  assert-or-open (the string the new run passes to `assert path.exists()` /
  `open()` / `load_dataset`, glob-expanded across the design's
  source/arm/dose/seed cells), and confirms the reused parent file(s) resolve
  at THAT pattern — not merely that the parent repo/dir exists — via a
  `list_repo_files` glob (HF) or a `git ls-tree` glob (committed
  `eval_results/...`) matching the consumer pattern. A parent that shipped its
  files under a different naming convention than the consumer asserts FAILS
  this leg even though the directory resolves under (i) (#474-vs-#664 naming,
  #734). (ii) checks PATH-LAYOUT only; schema /
  column-shape / version-tag / encoding drift are OUT of scope and covered —
  where covered at all — by `(f)` byte-content identity and, for a
  multi-field bundle's realized key set (field presence), by the check-(c)
  clause above. AND **(iii)
  target-backend fetchability** — the backend named in §9 can actually STAGE
  it. The RunPod lane stages any HF-resolved file (`snapshot_download` on
  small repos; scoped `list_repo_tree` + per-file `hf_hub_download` on the
  ~1M-file data repo — `.claude/rules/upload-policy.md`); the git-clone-only GCP and SLURM lanes
  stage NO VM-local `data/` — the startup `git clone` brings committed
  `eval_results/...` but NOT `data/issue_<N>/`, and HF/data-repo files need
  an explicit staging step in the workload — so a mix the parent BUILT but
  never UPLOADED nor COMMITTED is unreachable there and the pre-train
  `assert data_path.exists()` crashes phase2. AND **(iv)
  staged-layout consumer-open** — leg (ii) verifies the consumer's path
  pattern against the SOURCE listing; it says nothing about the tree the
  STAGING step produces. Whenever the reuse goes through a stage-from-Hub
  helper that maps hub-relative paths onto local paths — INCLUDING a verbatim
  prefix mirror — feeding a consumer with a fixed local layout (a
  `Store(store_dir)`-style init, a manifest/config read, a fixed directory
  tree), fetchability alone is insufficient: a producer that uploaded the
  consumer's ENTRY file inside the blob folder makes a verbatim mirror stage
  it one level deep, and the consumer's open() crashes AFTER provisioning.
  Three requirements: **(1)** the plan NAMES the hub-rel → local-rel mapping —
  prefer a PURE function threaded through the entry-time missing-check, the
  fetch destinations, and the completeness check via ONE shared mapped-path
  dict (worked fix: `store_local_relpath` + `stage_store` in
  `scripts/issue928_mlp_indiv_control.py`); **(2)** the stage FAILS LOUD when
  the mapped set lacks the consumer's entry file (manifest/config) or the
  mapping collides — never a "successful" stage into a doomed consumer init;
  **(3)** BEFORE any production run, a **1-file staging probe +
  consumer-open, ONCE PER (reused source-family × staged consumer) pair**:
  for EACH pair where a consumer reads a family's staged tree, stage that
  family's ENTRY file (KB-scale) through the REAL staging code path — the
  SAME staging helper the production phase uses for that pair — at the
  pinned revision, then run that consumer's entry-point open/init (or its
  manifest-read; pre-seed blob dummies at their mapped paths where full
  init requires them) against the staged root. A "reused source-family" is
  a distinct reused artifact group staged through its own source
  path/prefix; a "staged consumer" is a distinct consumer entry-point
  open/init reading a staged tree — INCLUDING a LATER phase that re-reads
  the staged layout, not only the first reader. The probe matrix is the set
  of (family, consumer) pairs actually read — per PAIR, not families +
  consumers additively. A single global probe on one family/consumer does
  NOT satisfy this leg when the plan stages ≥2 families or ≥2 consumers
  (#1481, two same-day staged-layout crashes: a SECOND source-family kept
  its sidecar under a different prefix than the worker assumed after family
  1 derived cleanly; and a LATER phase re-read a checkpoint staged via
  `hub.stage_hub_prefix`, whose verbatim prefix mirror lands files at
  `dest/<repo-rel path>` while the consumer opened `dest/` directly — the
  green tiny-real smoke had staged per-file via `stage_hub_file`, a
  DIFFERENT helper than production, validating nothing about the production
  mirror layout). A synthetic-fixture smoke that writes the
  LOCAL layout directly never exercises the staging phase and does NOT
  satisfy this leg (the cross-phase data-contract smoke class, #518); the
  end-to-end smoke must exercise the real staging path — per pair, through
  the production helper for that pair. Full 4-point implementation recipe
  (pure mapping fn, fail-loud entry check, regression test through the
  producer's REAL Hub path shapes, dummy-seeded real-Hub confirm):
  `.claude/agent-memory/experiment-implementer/feedback_hub_prefix_mirror_vs_consumer_layout.md`.
  **N/A escape:** reuse with NO staging transformation — the consumer opens
  the file(s) at the exact fetch destination(s), no layout mapping —
  satisfies leg (iv) trivially for that (family, consumer) pair; record
  "no staging transformation" in the reuse map. (#928: a verbatim prefix
  mirror landed the manifest one level deep — `<store>/percq_summaries/
  manifest.json` vs the consumer's `<store>/manifest.json` — crashing
  `Store()` init after legs (i)–(iii) all passed. Sibling lesson: verify
  the exact consumed path via `HfApi().file_exists` / scoped
  `list_repo_tree`, never infer it from a collapsed tree listing.) The
  check FAILS
  when ANY of (i)/(ii)/(iii)/(iv) fails. On a MISS,
  do NOT record the file as a confirmed reuse: either (a) rename / re-upload
  the parent file(s) to the consumer-asserted path pattern and cite that path,
  (b) adjust the new consumer to open the parent's actual path layout (naming
  the parent pattern in §4), (c) add a self-contained regen phase in §4
  that rebuilds the mix on the worker under the consumer-asserted paths,
  flagged `must-rebuild` in §12 Assumptions, or **(d)** for a leg-(iv) miss,
  fix the STAGING MAPPING — the pure hub-rel → local-rel function + the
  fail-loud entry-file check — not a rebuild (the artifact is fine; only a
  genuinely malformed UPLOADED tree — the entry file absent from the source
  listing itself — warrants regeneration). Verify every applicable leg for EVERY
  reused training-input / downstream-input file the design loads, BEFORE
  recording it in §10 / §11. (#734: a reused parent mix was on neither HF
  repo AND its naming convention differed from the consumer's asserted
  path; the plan passed 3 review rounds and crashed phase2 at the pre-train
  assert on a git-clone-only lane.)
- **(i) Throughput fitness (reused CODE — fit / analysis / eval /
  upload-verify-staging helpers; N/A
  for data-only reuse):** the code-reuse default (CLAUDE.md "Reuse existing
  experiment code" / "Reuse existing in-repo tools/helpers") inherits a
  parent's fit/analysis code path — a `scripts/issue<M>_*` fit / predictor /
  null-battery module, an `analysis/` helper, an eval harness, or an upload /
  verify / staging helper (an `issue<M>_common.py`-style module) the new run
  imports rather than rewrites. Before recording that reuse, READ the
  inherited helper's inner loop, device routing, and Hub-API call sites.
  Three checks: **(1) batched
  inner axis** — the per-cell / per-fold / per-draw / per-row axis is
  vectorized into batched tensor ops per
  `.claude/rules/vectorize-many-cell-fits.md`; a serial Python loop over
  cells, folds, draws, or rows is the 50-100× overhead-bound signature and
  FAILS (#761: a reused serial LOCO fold loop ran ~100× over plan; #667);
  **(2) parametrized device** — the compute device is a
  call-time parameter / flag / env lookup, not a hardcoded module constant; a
  `DEVICE = "cpu"` pin or an implicit CPU default FAILS (#763/#812: an
  inherited module-level `DEVICE = "cpu"` ran batched-but-on-CPU for hours);
  **(3) scoped Hub-API calls** — every post-upload verify / staging /
  existence-probe call the helper makes against the ~1M-file data repo is
  prefix-scoped: `list_repo_tree(repo_id, path_in_repo=<prefix>,
  repo_type="dataset", ...)` for subtree listings,
  `HfApi().file_exists(...)` for single-path probes, with a BOUNDED outer
  retry on a first-page 429/5xx (`huggingface_hub` pagination retries 429
  only on FOLLOW-UP cursor pages, so the first page raises in a quota
  storm — the #658 rule); an unscoped full-tree `list_repo_files` /
  `snapshot_download` against the data repo FAILS. Full recipe + quota
  arithmetic: `.claude/rules/upload-policy.md` #833 entry; worked source fix:
  `scoped_remote_listing` + `retry_hub_quota` in `scripts/issue810_common.py`
  (#810: reused verify helpers crawled the whole data repo under the org
  quota, wedging in 429 retry storms while an A100 idled at 0%.)
  Upload DESTINATIONS are mechanically gated by
  `workflow_lint.py --check-upload-prefix-clobber` (#1452 / incident #1005:
  reused #928 fitters uploaded to hardcoded `issue928_*` prefixes,
  overwriting the parent's artifacts); reusing a script whose upload
  destination is grandfathered in `UPLOAD_PREFIX_CLOBBER_ALLOWLIST` (or
  otherwise hardcoded) REQUIRES the reuse plan to thread an explicit
  child-issue upload prefix — the runtime-reuse mode the static lint cannot
  see.
  On a failure, the remedy is NOT retrain/regenerate and NOT a caller-side
  workaround (wrapping the serial loop, monkey-patching the constant): **fix
  at the SOURCE module** — batch / parametrize / scope it there, so every
  future reuser inherits the fix. SCHEDULE that source-module fix in the
  plan: as its own phase, or as a companion infra task filed in the same
  batch (#876), with the source fix landing BEFORE the consuming run; a
  caller-side wrapper is admissible only as explicitly-temporary scaffolding
  named in the plan with the source fix already filed. Binds at PLAN time —
  inline / ad-hoc analyses stay covered by
  `.claude/rules/vectorize-many-cell-fits.md` and the review-side
  named-helper checks (#869 Fix D) — and a check-(i) failure never
  disqualifies the reuse CLASS; it routes to the source-module fix, then
  reuse. Record the inspection result — helper/function name,
  batched-or-serial verdict, device handling, Hub-call-scoping verdict — in
  the plan's reuse map (§10 / §11) and reflect the implied wall-time in the
  §9 compute row.
- **(j) Pairwise provenance coherence (mutually-dependent artifact PAIRS).**
  When the reuse consumes two-or-more artifacts that must be mutually
  consistent because one was PRODUCED UNDER the other — a question/prompt
  bank vs activations / teacher-forced reads captured under it; a training
  mix vs the adapter trained on it; a completion pool vs judge outputs
  scored over it — checks (e)/(f) pin each member's CURRENT bytes
  individually but say nothing about whether the members come from the SAME
  generation. Verify the consumed INPUT does not POSTDATE the dependent
  CAPTURE, comparing last-modified dates AT THE REVISION EACH MEMBER IS
  ACTUALLY CONSUMED FROM. Mechanics — first group the pair's members by
  concrete storage location (repo_id + repo_type + consumed revision; the
  members often live in DIFFERENT stores: adapters/checkpoints in the model
  repo, mixes/raw-completion buckets in the data repo, judge/eval JSONs in
  git), then date each member:
  - HF-resident member: `HfApi().get_paths_info(repo_id, [<member file
    paths>], expand=True, repo_type=..., revision=<the revision the
    consumer will fetch>)` → per-path `last_commit` (`oid`, `date`) — a
    per-path POST, safe on the ~1M-file data repo (measured ~0.5 s; paths
    must be non-empty FILE paths, else HTTP 400). One call per repo; a
    same-repo single-file pair may batch both paths into one call.
  - Folder / multi-file member (a LoRA adapter subfolder, a sharded
    store): probe the EXACT consumed file set (every member file the
    consumer loads, or the artifact's named manifest file) and reduce to
    the MAX member-file `last_commit.date` — an unprobed sibling can be
    the regenerated file.
  - Git-resident member (a committed `eval_results/...` input):
    `git log -1 --format=%cI <consumed ref> -- <path>`.
  Require `max(input member dates) <= min(capture member dates)` at the
  consumed revisions; ALSO read any `reconstruction` / regeneration
  metadata field the artifact itself carries (#922). Two caveats:
  commit/upload time is a PROXY for production time — a capture-side
  re-commit postdating the input's regeneration is NOT evidence the capture
  was regenerated under the new input unless its provenance metadata says
  so; and an input regenerated AFTER the capture is inconsistent REGARDLESS
  of sha pins — a pin freezes current bytes, not pair coherence. RE-RUN
  this comparison whenever any member's pin/revision is refreshed (a
  crash-fix round re-pinning one member re-opens the check). On a failure,
  either (1)
  re-capture / regenerate the DEPENDENT artifact under the current input,
  or (2) pin the input at the pre-regeneration REVISION the capture was
  made under (`hf_hub_download(..., revision=<capture-era commit>)`) —
  (2) only after confirming WHY the input was regenerated (a bug-fix
  regeneration invalidates the old input, forcing (1)). A documented
  remedy-(2) pin PASSES this check: the probe runs at the consumed
  revision, so pinning the pre-regeneration revision restores coherence
  by construction. A pair uploaded in one commit passes trivially.
  (#922: a question artifact was regenerated AFTER its dependent activation
  capture; every prior check passed on each member individually and the run
  crashed at the parity assert after a full GCE cycle. Sibling class: the
  #601 pinned-pair COVERAGE mismatch —
  `.claude/agent-memory/experiment-implementer/feedback_pinned_artifact_pair_mutual_inconsistency.md`.)
  The PRODUCER-side half of this rule — a regeneration must version-bump the
  path or record the regeneration note this check reads (an in-artifact
  `reconstruction` field, or a same-commit `<name>.regeneration.json`
  sidecar) — is `.claude/rules/upload-policy.md`
  § "Regenerating a published artifact in place".
- **(k) Parent-lineage coherence (reused parent CODE and its realized
  artifacts; N/A when neither a parent code module nor a parent-realized
  artifact with a declared input corpus is reused).** Two legs, mutually
  corroborating (#1345: leg B's count shortfall was the visible fingerprint
  of leg A's unmerged fix).
  **(A) Parent-branch unmerged-fix diff (code reuse):** when the reuse
  imports a parent task's module from `main` (a `scripts/issue<M>_*.py`
  extractor / fit driver, an `analysis/` helper) and the parent's
  `issue-<M>` branch still exists on origin, fetch and run
  `git log --oneline origin/main..origin/issue-<M> -- <module path(s)>`
  over the enumerated module list the reuse map names (a module touched by
  MULTIPLE issue branches warrants a per-touching-branch log). NON-EMPTY
  output means the main-resident copy LAGS the parent's own fixes —
  inspect every commit and either PORT the fix (via § "Porting a recipe
  from an unmerged sibling branch" below — the same three mandatory steps,
  applied in the reuse direction) or explicitly declare it not-needed in
  the reuse map, citing the commit SHA(s). The parent's own CRASH-FIX
  rounds are the highest-risk class: the parent's realized artifacts
  embody the fix, so every artifact-side check (a)-(j) passes while the
  main-resident code path re-crashes on the first input the fix would have
  filtered. Worked example (#1345's parent):
  `git log --oneline origin/main..origin/issue-825 -- scripts/issue825_extract_turnstore.py`
  → one stranded crash-fix commit — exactly the filter that crashed #1345.
  Parent branch fully merged / diff
  empty ⇒ record the empty diff (PASSES). Parent branch DELETED ⇒ record
  "branch deleted — leg A unverifiable; leg B is the sole lineage read"
  (never "no unmerged parent commits" — `git log` against a deleted ref
  ERRORS, and an errored command must never masquerade as an empty diff).
  Known residual (named, not covered): leg A diffs the NAMED module
  path(s) only — a caller ported while a transitively-imported callee
  stays stranded on the branch escapes this diff; the porting section's
  whole-code-path diff (step 1) is the closing procedure once porting is
  chosen.
  **(B) Realized-row-count reconciliation (artifact reuse):** when the
  reused realized artifact — or its producing plan / body / manifest —
  declares an input corpus size, reconcile the artifact's realized
  row/cell count against it (manifest field, shard row sums, or a cheap
  row count on the resolved artifact). A SHORTFALL means a filter ran
  somewhere between corpus and artifact: NAME the filter (function +
  the branch/commit where it lives) in the reuse map before recording the
  reuse — an unexplained shortfall FAILS the check. When the named filter
  lives only on the parent's unmerged branch, leg A's port remedy applies
  BEFORE any fresh run re-executes the parent's code path on new or
  unfiltered data. No declared corpus size anywhere ⇒ record
  "no declared input corpus — leg B N/A".
  (#1345: a parent extractor was imported from `main` while its
  degenerate-row crash-fix lived only on the unmerged parent branch; the
  realized shards embodied the filter — n=4724 vs corpus 5000 — every
  existing check passed, and production crashed at the first unfiltered
  degenerate row. Sibling check in the opposite direction: the
  consistency-checker's "Reused code module reachable on `main`" row (#595)
  verifies the module EXISTS on main; leg A verifies main's copy is CURRENT
  vs the parent branch.)
- **(l) Validity-domain transfer (reused fit/analysis INSTRUMENTS; N/A when
  no fit/analysis code artifact is reused, or the instrument's docs declare
  no validity boundaries).** Before reusing a fit/analysis INSTRUMENT (the
  same code class item (i) governs for throughput: a ridge/GCV fitter, a
  probe trainer, a statistical battery) on a NEW consumption regime, READ
  the instrument's own docs/comments/module constants for DECLARED validity
  boundaries and registered mitigations — n-vs-d regime notes, dof caps,
  selection fallbacks (e.g. a module-global knob like `GCV_DOF_CAP`, an
  alternative selection mode like `lambda_selection="inner-group-cv"`) —
  and CHECK the new regime against each boundary: per-fold n_train vs d, a
  subset/filter that shrinks n below a documented bound, a
  correlation-structure shift a comment names. Crossing a declared boundary
  REQUIRES engaging the instrument's registered mitigation — or a stated
  justification for not engaging it — named in the plan (§4/§11) next to
  the reuse record. Scope split vs item (b): (b) asks whether the reused
  ARTIFACT is in a valid measurement regime for the new QUESTION
  (DV/question-scoped); (l) asks whether the new DATA REGIME crosses a
  boundary the reused CODE itself declares (instrument-doc-scoped) — #1417
  passed (a)/(b)/(i)/(k) and still shipped a voided verdict layer because
  no item asked the instrument-doc question. (#1417: the frozen
  `scripts/issue825_fit_cells.py` ridge instrument documents that GCV
  lambda selection collapses when the fold Gram can near-interpolate
  (n_tr < D) and registers two mitigations (`GCV_DOF_CAP`,
  `lambda_selection="inner-group-cv"`); #1417 reused it on judge-filtered
  subsets (per-fold n_train < d) with neither engaged — held-out R²
  −0.6…−1.5 on exactly those subsets where matched-n subsamples fit at
  +0.3…+0.65, found only by the analyzer post-run.)
  Call-shape bind (added #1728). Reading declared boundaries out of the
  instrument's docs / module constants is NOT enough: a reused fit /
  analysis helper can carry a runtime guard buried inside a code path —
  `assert <cond>, "..."` / `raise NotImplementedError(...)` / `raise
  ValueError(...)` — that rejects a legal-looking kwarg combination the
  signature and the module-level docs both accept. Before reuse, the plan
  records a probe of the ACTUAL call the new code will make: the exact
  kwargs at their exact values (or minimal stand-ins), executed at smoke
  shape — NOT a signature-name membership test. A kwarg present in the
  callee's signature is NOT evidence the call PATH accepts it. Cheap
  companion (run in the same round, not a substitute): direct
  `grep -n 'assert\|raise NotImplementedError\|raise ValueError' <reused
  helper>` for guards NAMING the kwargs the new caller passes, and read the
  hit lines. (#1728: a reused `heldout_r2_sweep` was called with a custom
  `lambdas=` grid; the parent's `_ridge_predict_cached` asserts
  `lambdas is None` on the inner-cache path — the signature accepts the
  kwarg, the code path rejects it; the signature smoke and the doc read
  BOTH passed, and the crash surfaced ~13h into upstream compute.) Scope
  split vs the signature smoke below: that smoke is
  NAME-membership only; a runtime-guard rejection is out of ITS scope and
  belongs here.

A failing check other than (i)/(h)(iv)/(k)/(l) → retrain / regenerate; a failing
throughput check (i) → fix the SOURCE module (batch / parametrize / scope it
there — never a caller-side workaround), then reuse; a failing staged-layout
consumer-open check (h)(iv) → fix the STAGING MAPPING (pure hub-rel →
local-rel fn + fail-loud entry-file check), then reuse; a failing
parent-lineage check (k) → port the unmerged parent-branch fix (or declare
it not-needed against the cited commit SHAs) and name the filter explaining
any count shortfall, then reuse — regenerate only when the shortfall traces
to a genuine defect in the artifact itself; a failing validity-domain check
(l) → engage the instrument's registered mitigation (or state the
justification), then reuse — never a silent retrain: the instrument is
sound, the CONSUMPTION REGIME crossed its declared boundary. Say why in the
plan either way.

## Reuse-validation gate calibration + severity (HALT vs WARN) (#813)

Checks (b)/(f)/(g)/(j) become RUNTIME GATES in dispatchers — a numeric
write-ratio parity floor, a behavioral install confirmation, a one-cell
gate. Every number in this section is an incident-specific illustration
(the #813/#537/#667 lineage), never a portable default — inheriting 0.01,
0.004, or 0.1729 into a new gate without re-derivation is itself the
violation rule 1 names. Two rules govern every such gate:

**1. Calibrate the threshold against the reused artifact's OWN committed
per-behavior, SAME-SURFACE reference values — never a bare global
constant.** Before a gate threshold reaches code, the plan names — PER
(behavior, adapter-class) cell the gate will run against — the committed
reference the threshold is derived from, file + field: e.g. #537's
committed G_cells
(`eval_results/issue_537/G_cells/marker/default__default__seed42.json` →
`g_mean_delta_logp = 6.062`), #667's committed per-behavior write ratios,
the artifact's own `adapter_config.json`, or the parent
`## Reproducibility` rows. A constant calibrated on ONE adapter class
silently mis-fires on another: #813's global write-ratio floor, calibrated
on 7-module adapters, false-HALTed a 4-module attention-only marker
adapter writing correctly below it. Band-stopped marker adapters
(lr ≤ 5e-6 per `.claude/rules/marker-training-recipe.md`) are the
CANONICAL near-floor case: they legitimately sit near any generic floor,
so a marker cell inheriting a non-marker constant is presumptively
miscalibrated. **Same-surface commensurability:** a committed reference
is HALT-calibrating ONLY if it was measured under the SAME read surface —
rig / gauge / slot / conditioning — the gate will execute. A
committed-reference-calibrated bar on a DIFFERENT surface is still
miscalibrated: #813's 2.5-nat bar, grounded on a frozen-R diagonal band,
review-PASSed twice and still false-HALTed a fresh-greedy-R slot read —
same nominal units, different statistic. A reference from a different read
surface is NOT a calibration source for a HALT: derive a same-surface
reference first, or the gate runs as WARN (rule 2). Likewise a behavior
with NO committed reference for the gate's read does not silently inherit
another behavior's constant — derive a reference first, or run that
behavior's gate as WARN (rule 2).

**2. A HALT threshold must sit at a DISCRIMINATING value between
failure-mode bands; a diagnostic that can only fail by the artifact being
WEAKER than expected defaults to WARN + analyzer adjudication.** HALT is
reserved for apply-path breakage, whose signature is structural or
MULTIPLICATIVE: wrong adapter resolved/loaded, marker-token-id mismatch,
gauge violation (an α/√r-vs-α/r error at r=32 is a √32 ≈ 5.66×
discrepancy, never a 10% shortfall), wrong scaling regime. Place a HALT
threshold BETWEEN the correct-weak band and the wrong-gauge band (#813:
floor 0.004 separates 0.009 correct from 0.0016 wrong-gauge, ≥2.25×
margin each way; a one-sided floor catches only the UNDER-scaled arm —
name the companion check for the over-scaled direction, e.g. a ceiling on
the same read or the sibling exact-match check).
When no discriminating placement exists — legitimate reads overlap every
bar groundable without the parent's exact eval rig (#813) — the gate
CANNOT hold HALT. Demote it
to WARN, but ONLY with all three of: (a) the apply path independently
confirmed (e.g. a sibling behavior's write-ratio matching the committed
value EXACTLY — #813's em 0.1729 == #667's committed 0.1729); (b) the
measured value PERSISTED in the run artifacts (never dropped); (c) a
NAMED analyzer-adjudication concern plus the cheap discriminating test
that would resolve the fork (#813: `marker_delta_logp_nats` persisted,
WARN carried to the analyzer, frozen-R teacher-forced re-read named).
This is NOT threshold-loosening-to-pass (banned — see gotchas.md #841
"never loosen the fp32 tol", and #813's own supervisor ruling "do NOT
lower the 2.5-nat bar to pass 0.75"): the signal is preserved and
adjudicated, not redefined away; and the apply-path HALTs still catch
every genuinely-broken artifact. Consistent with the fail-fast Critical
Rule — a persisted WARN with a named adjudication is a surfaced failure,
not a swallowed one.

**Severity lattice, fully mapped (no unmapped cell).** (i) Structural /
multiplicative apply-path breakage → HALT, always — this precedence is
absolute and no WARN license overrides it. (ii) A same-surface committed
reference WITH a discriminating placement → HALT at that placement.
(iii) NO same-surface reference (absent, or rig-mismatched) or no
discriminating placement → the gate CANNOT hold HALT: it is DESIGNED as a
fresh WARN gate. Conditions (b) persist + (c) named adjudication bind
EVERY WARN gate, fresh or demoted; condition (a) independent apply-path
confirmation is specifically the license for DEMOTING an in-flight HALT
mid-run — a run that cannot produce (a) when demoting does not demote: it
stays halted until a same-surface reference or apply-path confirmation is
produced. A fresh plan-time WARN gate needs (b)+(c) only.

**Relaunch discipline.** A gate-threshold change is a FIX: declare its
fix-engaged signal before relaunching, per `.claude/rules/crash-fix-rounds.md`
§ "Crash-fix rounds: declare the fix-engaged signal (REQUIRED)" (#813
verified the old signature gone before attributing the new HALT). Probe
logs APPEND across runs — attribute a HALT to the CURRENT run by source
line number, never to a stale appended traceback. And demote-to-WARN is
the ONLY licensed remedy for an ungroundable gate — SKIPPING the gate (a
`--skip-*` flag) is not: a skip flag with no demotion record is a silent
gate deletion (#813 ran its skip only AFTER the WARN demotion had landed
and the apply path was independently confirmed).

**Enforcement.** planner step 5 self-attest: a gate-bearing plan names each
threshold's calibration source (file + field, per behavior) and its
HALT-vs-WARN class in §4/§11. `critic.md` Methodology lens item 9 REVISEs a
bare-constant threshold or an unjustified weaker-than-expected HALT.
`code-reviewer`, on a gate-bearing diff: the constant's comment/docstring
states its calibration source; no committed-reference grounding is a
blocker-worthy finding. Hardware/precision false-parity SIBLINGS are
environment mismatch, distinct from calibration: gotchas.md #723-family +
#841 entries.

## Enforcement chain

Enforcement is a 3-stage defense: `planner.md` step 5 (self-attested fitness
check) → `consistency-checker` (independent reuse-smuggled-variable diff vs
the parent recipe) → `critic.md` Methodology lens item 9 (REVISE); the reuse
provenance is then carried into the clean-result `## Reproducibility`
(`analyzer.md`) and audited by `clean-result-critic` Lens 5.

## Porting a recipe from an unmerged sibling branch (relocated from experiment-implementer.md, #829)

(Complementary halves: this section governs deliberately porting a recipe
FROM an unmerged branch; checklist item (k) leg A above catches the
INVERSE — reusing the main-resident copy of a parent module while the
parent's issue branch carries unmerged fixes — and routes its port remedy
through this section's three mandatory steps. #1345. DISCOVERY of
concurrent not-yet-merged sibling work is upstream of both: the planner's
live-sibling sweep (`.claude/agents/planner.md` step 5) enumerates live
sessions' in-flight worktree modules and routes any overlap here as
reuse-or-consolidation instead of a fresh build, #1394.)

If the parent experiment's scripts/configs live on a branch that was
never merged to `main`, do NOT cherry-pick functions one at a time. A
partial port brings the caller without the callee (or vice versa) and
crashes the pod one phase at a time. The crash class includes BOTH
direct missing-function imports AND **library-API drift** — a
dataclass field, function kwarg, or method signature the parent SHA
used that has been renamed / retired / type-changed on `main` since the
parent branched: the parent-branch caller passes the old shape, the
`main`-resident callee rejects it, the cell crashes at the first pod
launch. The reconciliation MUST happen pre-cherry-pick, not at the
crash.

Three mandatory steps, BEFORE the first commit on the worktree:

1. **Diff the WHOLE train+eval+experiments code path against `main`
   and reconcile every hunk** (port it, or confirm `main`'s version is
   equivalent + adjust the cherry-picked call site to match `main`'s
   current signature):

   ```bash
   git diff <parent-sha>..origin/main -- scripts/train.py scripts/eval.py \
     src/explore_persona_space/train/ \
     src/explore_persona_space/eval/ \
     src/explore_persona_space/experiments/ \
     configs/
   ```

   "Reconcile" is not optional and not silent — the implementation
   report's `(b) Considered but not done` section MUST list every
   non-trivial hunk you reconciled, naming which fields / functions /
   kwargs drifted and which way you resolved them (ported the
   parent's shape, or adjusted the call site to `main`'s shape). A
   hunk you "didn't notice" is the partial-port crash class.

2. **Signature smoke per kwarg the dispatcher passes.** Before the
   first commit, run a one-liner that asserts every kwarg / dataclass
   field the cherry-picked dispatcher will pass is actually present
   in `main`'s current signature for that callee (catches drift the
   git-diff scan missed because the hunk landed in an adjacent
   file). Pattern:

   ```bash
   uv run python -c "
   from dataclasses import fields
   from explore_persona_space.train.sft import TrainLoraConfig  # or whichever Config the dispatcher constructs
   dispatcher_kwargs = {<every kwarg the dispatcher's call site passes>}
   missing = dispatcher_kwargs - {f.name for f in fields(TrainLoraConfig)}
   assert not missing, f'Library-API drift: dispatcher passes kwargs missing from main: {missing}'
   "
   ```

   For non-dataclass callees use `inspect.signature(<fn>).parameters`
   instead of `fields(<Config>)`. Run this for EVERY library callee
   the cherry-picked code constructs or invokes at the dispatcher
   boundary (typically: training Config, eval Config, the trainer
   entry-point fn, the eval entry-point fn). This is in addition to
   — not a replacement for — the standard signature smoke in the
   GPU-bound-phase carve-out (the per-phase one verifies the
   dispatcher → trainer ABI; this per-kwarg one verifies every
   field the dispatcher's call site already names).
   Scope limit (#1728). This smoke is NAME-MEMBERSHIP only — it
   catches a kwarg the callee's SIGNATURE has retired / renamed /
   type-changed on `main`. It does NOT bind the actual VALUES the new
   caller will pass against runtime guards buried in the callee's body;
   a runtime-guard rejection of a legal-looking kwarg combination is
   OUT OF SCOPE HERE and belongs to check (l)'s call-shape bind above.

3. **Surface every reconciled drift in the implementation report.**
   Under `(b) Considered but not done`, one bullet per drift item:
   "`TrainLoraConfig.marker_logprob_trajectory` retired on `main`
   since `<parent-sha>` — removed from the dispatcher's kwargs; the
   feature is now <X> on `main` and the cherry-pick relies on <Y>"
   (or "ported the parent's field back to `train/sft.py` because
   `main`'s replacement <Z> is not equivalent for this experiment").
   This makes the reconciliation visible to `code-reviewer` and to
   any later task that re-uses the recipe.

(Incidents: #451 — a partial port left `train/sft.py` at `main`'s older
signature, all 72 cells crashed in ~10 min; #456 — the same class three
times, each burning a fix-relaunch on a live pod; #529 — two retired /
type-changed fields caught only by post-hoc introspection, reactive not
preventative.)

## Relocated codebase traps (from `.claude/rules/gotchas.md`, #2189)

Verbatim gotchas.md entries whose topic this rule already owns — relocated
to recover gotchas.md byte budget (#2189); wording and `#N` citations kept.

- **sha-pinned artifact PAIRS can be mutually incoherent — pin freezes bytes, not pair provenance.** Two reused artifacts that must be mutually consistent (a prompt bank vs activations captured under it; a training mix vs its adapter) can EACH pass sha-pin + Hub-resolution while the input was REGENERATED after the dependent capture — the pair then disagrees and the consumer crashes deep in the run (#922: questions regenerated a day AFTER the capture). Check ORDERING at the trust boundary, AT THE REVISIONS ACTUALLY CONSUMED and per storage location: per HF repo, `HfApi().get_paths_info(repo_id, [<member paths>], expand=True, revision=<consumed rev>)` per-path `last_commit.date` (non-empty FILE paths only; folder members reduce to the max member-file date); git-resident members via `git log -1 --format=%cI <ref> -- <path>`; require max(input dates) <= min(capture dates), and read any regeneration metadata the artifact carries (commit time is a proxy for production time). Checklist hook: `.claude/rules/artifact-reuse.md` item (j). Sibling class: #601 pinned-pair coverage mismatch.
- **Reused artifact's REALIZED keys can predate the builder code — verify the bundle's OWN key set against the consumer's asserts; reading the builder code is NOT verification.** A sha-pinned reused multi-field bundle carries the key set the builder wrote ON THE UPLOAD DATE: a field added to the builder after the pinned upload is absent from the artifact, and every reviewer who "verifies" the schema by reading the current builder source inherits the same drift — the consumer's hard assert then kills the run on the pod (#1073: today's builder writes `prompts`; the pinned upload lacked it; fact-checker AND smoke fixture both mirrored the builder code). RULE: (1) at plan/smoke time verify the artifact's OWN realized keys against every consumer assert — cheap even on a multi-GB bundle via `torch.load(path, map_location="cpu", mmap=True).keys()` — or definitively by running the CONSUMER'S OWN loader against the real pinned artifact; mechanized: `uv run python scripts/verify_reused_artifact_keys.py --artifact <path> --keys <k1,k2,...>` (or `--hf-repo`/`--hf-path`) exits nonzero on any missing key — run it at plan time, paste its PASS line into §10 (plan-gate c30 WARNs a bundle-reuse plan naming no realized-keys verification); (2) a missing deterministically-regenerable field → regenerate via the parent loader with fail-loud source/length asserts PLUS a deterministic re-capture alignment gate against the bundle's STORED tensors (row ALIGNMENT is the invariant — a silently misaligned regenerated field is worse than the crash); (3) smoke fixtures default to the PRODUCTION artifact's realized shape, never the builder-code shape. Third class in the reused-artifact trust-boundary family (#600 bytes / #922 pair provenance; bug class `reused_artifact_schema_drift`); checklist hook: `artifact-reuse.md` check (c) — presence of the FILE is not presence of the FIELDS.
- **Main-resident parent module can LAG the parent branch's own crash-fixes — realized-artifact row counts are the fingerprint (#1345).** Importing a parent's extractor/fit module from `main` inherits main's copy, not the branch the parent actually ran + fixed; the parent's realized artifacts embody the branch-only fix (n=4724 vs corpus 5000 was the tell), so artifact-side checks pass while a fresh run through main's copy crashes on the first input the fix would have filtered. At reuse time: `git log --oneline origin/main..origin/issue-<M> -- <module>` + reconcile realized row counts vs the declared input corpus. Fourth class in the reused-artifact trust-boundary family; checklist hook: `artifact-reuse.md` check (k).
- **Cherry-picking a parent's driver: also cherry-pick every vendored module tree the parent introduced.** A parent may have vendored ADDITIONAL module trees in its own crash-fix rounds (commit messages tagged "vendor"/"vendored" + a module path) that the new plan never names — part of the driver's transitive import surface. The CPU-only smoke does not exercise the GPU-bound judge/generation path where the missing import lives, so the `ModuleNotFoundError` surfaces minutes into the multi-hour pod run (#640: the parent had vendored 14 files under `experiments/issue503/`). RULE: at PLAN time, grep the parent's commit log — `git log --grep='vendor\|vendored\|vendoring' <parent-branch>` — and carry every named module path; at IMPLEMENTER time, run an end-to-end surface import from the clean checkout BEFORE posting the implementation marker: `uv run python -c "import sys; sys.path.insert(0, 'scripts'); import <driver_module>"` (sibling defense to the #606 `--verify-imports` AST smoke).
- **A dispatcher that consumes a `data/issue<N>/` artifact MUST self-build it on a fresh instance — gitignored data does NOT travel with the branch clone.** `.gitignore` excludes `data/*`, so a per-issue battery/dataset built LOCALLY is absent from every fresh GCP/pod clone; a dispatcher assuming otherwise crashes in its extract phase — and on the GCP lane the EXIT-trap delete destroys the failure sentinel too (#654). Same fresh-instance-completeness class as the cherry-pick-vendored-modules trap above. RULE: any dispatcher consuming a `data/issue<N>/` artifact begins with a `[ ! -f "$ARTIFACT" ]`-gated build step that regenerates it on the fresh instance.

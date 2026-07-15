---
description: Trained-artifact + code reuse fitness check (a)-(j) — when to reuse a prior HF adapter / checkpoint / training-mix / raw-completion bucket / eval JSON / fit-analysis helper vs retrain or rewrite, incl. pairwise pair-provenance coherence (#922) and reuse-validation gate calibration + HALT-vs-WARN severity (#813), and the staged-layout consumer-open probe (#928), with the enforcement chain (loads at plan time via plan-file paths)
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# Trained-artifact (and code) reuse — the fitness check (a)-(j)

CLAUDE.md Critical Rules carries the always-on rule ("Reuse existing trained
artifacts when fit-for-purpose — never reuse a wrong one") plus a one-line
summary naming checks (a)-(j); this file is the full checklist AND, as of
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
On the ~1M-file DATA repo a bare `list_repo_files` listing itself times out
(>90 s, #833) — use `HfApi().file_exists(repo_id, <path>,
repo_type='dataset')` for a single path or scoped
`list_repo_tree(path_in_repo=<prefix>)` for a subtree (gotchas.md).
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
  needs a `police_officer` arm). Presence of the FILE is not presence of the
  FIELDS: for a multi-field tensor bundle, verify the artifact's REALIZED
  key set — a cheap header/mmap read (`torch.load(path, map_location="cpu",
  mmap=True).keys()`), or the consumer's own loader run against the real
  pinned artifact — against every consumer assert; reading the builder code
  is NOT verification, the pinned upload can predate the field
  (`.claude/rules/gotchas.md` "Reused artifact's REALIZED keys can predate
  the builder code", incident #1073); Mechanized: `uv run python
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
  the CURRENT stack, pinning the read gauge in plan §4 (a recipe-identical
  parent committed at classic `α/r` is an unconditional repeater at the
  faithful `α/√r` current vLLM+PEFT honor for `use_rslora: true`; incident
  #601). Threshold calibration + HALT-vs-WARN severity for this probe and
  every other reuse-validation gate: § Reuse-validation gate calibration
  below.
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
  where covered at all — by `(f)` byte-content identity and, for a
  multi-field bundle's realized key set (field presence), by the check-(c)
  clause above. AND **(iii)
  target-backend fetchability** — the backend named in §9 can actually STAGE
  it. The RunPod lane stages any HF-resolved file (`snapshot_download` on
  small repos; scoped `list_repo_tree` + per-file `hf_hub_download` on the
  ~1M-file data repo — gotchas.md; its HF leg ≈
  (i) there); the git-clone-only GCP and SLURM lanes stage NO VM-local `data/`
  — the GCE startup `git clone`s the repo at the cited branch (so committed
  `eval_results/...` arrive, but `data/issue_<N>/` does NOT) and HF/data-repo
  files need an explicit staging step in the workload (scoped
  `list_repo_tree(path_in_repo=...)` + per-file `hf_hub_download` on the
  ~1M-file data repo — `snapshot_download` full-tree-enumerates there;
  gotchas.md) — so a mix
  the parent BUILT but never UPLOADED nor COMMITTED is unreachable there and
  the pre-train `assert data_path.exists()` crashes phase2. AND **(iv)
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
  consumer-open**: stage the artifact's ENTRY file (KB-scale) through the
  REAL staging code path at the pinned revision, then run the consumer's
  entry-point open/init (or its manifest-read; pre-seed blob files as dummies
  at their mapped paths where full init requires them) against the staged
  root. A synthetic-fixture smoke that writes the LOCAL layout directly never
  exercises the staging phase and does NOT satisfy this leg (the cross-phase
  data-contract smoke class, #518); the end-to-end smoke must exercise the
  real staging path. Full 4-point implementation recipe (pure mapping fn,
  fail-loud entry check, regression test through the producer's REAL Hub path
  shapes, dummy-seeded real-Hub confirm):
  `.claude/agent-memory/experiment-implementer/feedback_hub_prefix_mirror_vs_consumer_layout.md`.
  **N/A escape:** reuse with NO staging transformation — the consumer opens
  the file(s) at the exact fetch destination(s), no layout mapping —
  satisfies leg (iv) trivially; record "no staging transformation" in the
  reuse map. (Incident #928, att-20260704-120700: the Hub stored all 51 store
  files FLAT under `.../store/percq_summaries/` — the extractor uploaded
  `manifest.json` INSIDE that folder — while `Store(store_dir)` reads
  `<store>/manifest.json`; `stage_store` mirrored the Hub prefix verbatim,
  the manifest landed at `<store>/percq_summaries/manifest.json`, and the
  production run crashed at `Store()` init, ~6 min GPU burn; legs (i)–(iii)
  all passed. Same-day sibling, same lesson: #928's analyzer 404'd on an HF
  subpath GUESSED from a collapsed tree listing — verify the exact consumed
  path via `HfApi().file_exists` / scoped `list_repo_tree`, never infer it
  from a collapsed listing.) The check FAILS
  when ANY of (i)/(ii)/(iii)/(iv) fails (e.g. HF-resolved but a CDN/region/
  `HF_TOKEN` gate stops the §9 lane from staging it; or the parent repo
  resolves but no file matches the consumer-asserted path pattern). On a MISS,
  do NOT record the file as a confirmed reuse: either (a) rename / re-upload
  the parent file(s) to the consumer-asserted path pattern and cite that path,
  (b) adjust the new consumer to open the parent's actual path layout (naming
  the parent pattern in §4), (c) add a self-contained regen phase in §4
  that rebuilds the mix on the worker under the consumer-asserted paths from
  the parent's deterministic build blocks, and flag it `must-rebuild` in §12
  Assumptions, or **(d)** for a leg-(iv) miss, fix the STAGING MAPPING — the
  pure hub-rel → local-rel function + the fail-loud entry-file check — not a
  rebuild (the artifact is fine; the mapping was wrong; only a genuinely
  malformed UPLOADED tree — the entry file absent from the source listing
  itself — warrants regeneration). Verify every applicable leg for EVERY
  reused training-input / downstream-input file the design loads, BEFORE
  recording it in §10 / §11. (Incident #734 round-4: a
  reused parent training mix was on neither HF repo, AND the parent's naming
  convention (#474 `i474_loc_A1.jsonl`) differed from the consumer's asserted
  path (a #664-style `mk_<source>_<arm>_<dose>_seed42.jsonl`); the plan passed
  planning + 3 review rounds and crashed phase2 at the pre-train assert on the
  GCP lane because the lane cannot stage a VM-local-only mix AND no file
  resolved at the asserted path.)
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
  FAILS (#761 reused #658's `_ridge_predict_loco` — a serial loop over 50
  LOCO folds, eigh+solve per fold — at ~100× over plan, 0.3h → ~30h/behavior
  projected; #667's inline analysis reused #722's serial per-cell
  `fit_cell`); **(2) parametrized device** — the compute device is a
  call-time parameter / flag / env lookup, not a hardcoded module constant; a
  `DEVICE = "cpu"` pin or an implicit CPU default FAILS (#763 and #812
  inherited `issue658_fit_predictors.py`'s module-level `DEVICE = "cpu"` and
  ran batched-but-on-CPU for hours until the `EPM_FIT_DEVICE=cuda` cutover);
  **(3) scoped Hub-API calls** — every post-upload verify / staging /
  existence-probe call the helper makes against the ~1M-file data repo is
  prefix-scoped: `list_repo_tree(repo_id, path_in_repo=<prefix>,
  repo_type="dataset", ...)` for subtree listings,
  `HfApi().file_exists(...)` for single-path probes, with a BOUNDED outer
  retry on a first-page 429/5xx (`huggingface_hub` pagination retries 429
  only on FOLLOW-UP cursor pages, so the first page raises in a quota
  storm — the #658 rule); an unscoped full-tree `list_repo_files` /
  `snapshot_download` against the data repo FAILS. Full recipe + quota
  arithmetic: `.claude/rules/gotchas.md` #833 entry; worked source fix:
  `scoped_remote_listing` + `retry_hub_quota` in `scripts/issue810_common.py`
  (readable via `git show 04701b2d56:scripts/issue810_common.py` on branch
  `issue-810` until #810's merge lands it on main). (#810 btdr round,
  2026-07-04: reused verify helpers crawled the whole data repo under the
  2500-req/5-min org quota, wedging in 429 retry storms — retry 20/20 —
  while an A100 idled at 0%.)
  On a failure, the remedy is NOT retrain/regenerate and NOT a caller-side
  workaround (wrapping the serial loop, monkey-patching the constant): **fix
  at the SOURCE module** — batch / parametrize / scope it there, so every future
  reuser inherits the fix — applying the vectorize-first law (vectorize
  before reaching for GPU or a bigger machine). SCHEDULE that source-module
  fix in the plan: as its own phase, or as a companion infra task filed in
  the same batch (the #876 pattern), with the source fix landing BEFORE the
  consuming run; a caller-side wrapper is admissible only as
  explicitly-temporary scaffolding named in the plan with the source fix
  already filed. This check binds at PLAN time (the reuse-recording step) —
  inline / ad-hoc analyses that run without a plan stay covered by
  `.claude/rules/vectorize-many-cell-fits.md` (code-time `paths:` trigger)
  and the review-side named-helper checks (#869 Fix D) — and a check-(i)
  failure never disqualifies the reuse CLASS; it routes to the source-module
  fix, then reuse. Record the inspection result — helper/function name,
  batched-or-serial verdict, device handling, and (when the helper touches
  the Hub) the Hub-call-scoping verdict — in the plan's reuse map
  (§10 / §11) and reflect the implied wall-time in the corresponding §9
  compute row.
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
  metadata field the artifact itself carries (#922's question artifact
  documented its own regeneration). Two caveats: commit/upload time is a
  PROXY for production time — a capture-side re-commit that postdates the
  input's regeneration is NOT evidence the capture was regenerated under
  the new input unless its provenance metadata says so; and an input
  regenerated AFTER the capture is inconsistent REGARDLESS of sha pins — a
  pin freezes current bytes, not pair coherence. RE-RUN this comparison
  whenever any member's pin/revision is refreshed (a crash-fix round
  re-pinning one member re-opens the check). On a failure, either (1)
  re-capture / regenerate the DEPENDENT artifact under the current input,
  or (2) pin the input at the pre-regeneration REVISION the capture was
  made under (`hf_hub_download(..., revision=<capture-era commit>)`) —
  (2) only after confirming WHY the input was regenerated (a bug-fix
  regeneration invalidates the old input, forcing (1)). A documented
  remedy-(2) pin PASSES this check: the probe runs at the consumed
  revision, so pinning the pre-regeneration revision restores coherence
  by construction. A pair uploaded in one commit passes trivially.
  (Incident #922 r4, att-20260703-163130: #779's question artifacts were
  regenerated — HF commit 9578892ef4, 2026-07-02 — AFTER the dependent
  `cx.pt` activation capture — a8060198a4, 2026-07-01; every prior check
  passed on each member individually and the run crashed at the parity
  assert after a full GCE cycle. Sibling class: the #601 pinned-pair
  COVERAGE mismatch —
  `.claude/agent-memory/experiment-implementer/feedback_pinned_artifact_pair_mutual_inconsistency.md`.)
  The PRODUCER-side half of this rule — a regeneration must version-bump the
  path or record the regeneration note this check reads (an in-artifact
  `reconstruction` field, or a same-commit `<name>.regeneration.json`
  sidecar) — is `.claude/rules/upload-policy.md`
  § "Regenerating a published artifact in place".

A failing check other than (i)/(h)(iv) → retrain / regenerate; a failing
throughput check (i) → fix the SOURCE module (batch / parametrize / scope it
there — never a caller-side workaround), then reuse; a failing staged-layout
consumer-open check (h)(iv) → fix the STAGING MAPPING (pure hub-rel →
local-rel fn + fail-loud entry-file check), then reuse. Say why in the plan
either way.

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
`g_mean_delta_logp = 6.062`), #667's committed per-behavior write ratios
(em 0.1729, sycophancy 0.0757), the artifact's own `adapter_config.json`,
or the parent `## Reproducibility` rows. A constant calibrated on ONE
adapter class silently mis-fires on another: #813's global
`PARITY_MIN_WRITE_RATIO = 0.01` was calibrated on 7-module adapters
(em 0.1729) and false-HALTed the 4-module attention-only α=64 marker
adapter writing 0.00903 — correctly. Band-stopped marker adapters
(lr ≤ 5e-6 per `.claude/rules/marker-training-recipe.md`) are the
CANONICAL near-floor case: they legitimately sit near any generic floor,
so a marker cell inheriting a non-marker constant is presumptively
miscalibrated. **Same-surface commensurability:** a committed reference
is HALT-calibrating ONLY if it was measured under the SAME read surface —
rig / gauge / slot / conditioning — the gate will execute. A
committed-reference-calibrated bar on a DIFFERENT surface is still
miscalibrated: #813 halt-3's 2.5-nat bar was grounded on #537's committed
frozen-R diagonal band (this very G_cells exemplar), review-PASSed twice,
and still false-HALTed a fresh-greedy-R slot read at 0.7516 — same
nominal units, different statistic. A reference from a different read
surface is NOT a calibration source for a HALT: derive a same-surface
reference first, or the gate runs as WARN (rule 2). Likewise a behavior
with NO committed reference for the gate's read (marker was never run
against #667's floor) does not silently inherit another behavior's
constant — derive a reference first, or run that behavior's gate as WARN
(rule 2).

**2. A HALT threshold must sit at a DISCRIMINATING value between
failure-mode bands; a diagnostic that can only fail by the artifact being
WEAKER than expected defaults to WARN + analyzer adjudication.** HALT is
reserved for apply-path breakage, whose signature is structural or
MULTIPLICATIVE: wrong adapter resolved/loaded, marker-token-id mismatch,
gauge violation (an α/√r-vs-α/r error at r=32 is a √32 ≈ 5.66×
discrepancy — ~0.0016 or ~0.05 against a correct 0.009 — never a 10%
shortfall), wrong scaling regime. Place a HALT threshold BETWEEN the
correct-weak band and the wrong-gauge band (#813 round-4: floor 0.004
separates 0.009 correct from 0.0016 wrong-gauge, ≥2.25× margin each way;
note a one-sided floor catches only the UNDER-scaled wrong-gauge arm —
name the companion check that catches the over-scaled ~0.05 direction,
e.g. a ceiling on the same read or the sibling exact-match check).
When no discriminating placement exists — legitimate reads overlap every
bar groundable without the parent's exact eval rig (#813's behavioral
gate: 0.8547 < 1.0 nat, then 0.7516 < 2.5 after re-grounding; the
committed 6.06-nat diagonal was measured under #537's frozen-R rig, not
the probe's fresh-greedy-R slot) — the gate CANNOT hold HALT. Demote it
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
round-4 verified the old `0.00903 < 0.01` signature gone before
attributing the new HALT). Probe logs APPEND across runs — attribute a
HALT to the CURRENT run by source line number, never to a stale appended
traceback. And demote-to-WARN is the ONLY licensed remedy for an
ungroundable gate — SKIPPING the gate (a `--skip-*` flag) is not: #813's
terminal relaunches ran `--skip-apply-parity` only AFTER the WARN
demotion had landed and the em exact-match had confirmed the apply path;
a skip flag with no demotion record is a silent gate deletion.

**Enforcement.** planner step 5 self-attest: a plan whose design carries a
reuse-validation gate names each threshold's calibration source (file +
field, per behavior) and its HALT-vs-WARN class in §4/§11. `critic.md`
Methodology lens item 9 REVISEs a bare-constant threshold or an
unjustified weaker-than-expected HALT. `code-reviewer`, on a gate-bearing
diff: the constant's comment/docstring states its calibration source (the
#813 round-4 pattern); a threshold with no committed-reference grounding
is a blocker-worthy finding, not an FYI. Hardware/precision false-parity
SIBLINGS (gauge-origin hardware; float64 clone for fp32 kernel parity)
are environment mismatch, distinct from calibration: gotchas.md
#723-family + #841 entries.

## Enforcement chain

Enforcement is a 3-stage defense: `planner.md` step 5 (self-attested fitness
check) → `consistency-checker` (independent reuse-smuggled-variable diff vs
the parent recipe) → `critic.md` Methodology lens item 9 (REVISE); the reuse
provenance is then carried into the clean-result `## Reproducibility`
(`analyzer.md`) and audited by `clean-result-critic` Lens 5.

## Porting a recipe from an unmerged sibling branch (relocated from experiment-implementer.md, #829)

If the parent experiment's scripts/configs live on a branch that was
never merged to `main` (e.g. issue-432's recipe sits on the `issue-432`
branch at `<sha>`), do NOT cherry-pick functions one at a time. A
partial port brings the caller without the callee (or vice versa) and
crashes the pod one phase at a time. The crash class includes BOTH
direct missing-function imports AND **library-API drift** — a
dataclass field, function kwarg, or method signature that the parent
SHA used but that has been renamed / retired / type-changed on `main`
since the parent branched (e.g. `TrainLoraConfig.marker_logprob_
trajectory` retired on `main`, `marker_text: list[str]` reverted to
`str` on `main`). The parent-branch caller passes the old shape; the
`main`-resident callee rejects it; the cell crashes at the first pod
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

3. **Surface every reconciled drift in the implementation report.**
   Under `(b) Considered but not done`, one bullet per drift item:
   "`TrainLoraConfig.marker_logprob_trajectory` retired on `main`
   since `<parent-sha>` — removed from the dispatcher's kwargs; the
   feature is now <X> on `main` and the cherry-pick relies on <Y>"
   (or "ported the parent's field back to `train/sft.py` because
   `main`'s replacement <Z> is not equivalent for this experiment").
   This makes the reconciliation visible to `code-reviewer` and to
   any later task that re-uses the recipe.

(Incidents: 2026-06-01 #451 cherry-picked `factor_screen_397` but
left `train/sft.py` at `main`'s older `TrainLoraConfig` signature →
all 72 cells crashed in ~10 min. #456 hit the same partial-port
class three times, each crash burning a fix-relaunch on a live pod.
2026-06-08 #529 cherry-picked the `i464_*` rig from `issue-464` SHA
`0905fc70`; `TrainLoraConfig.marker_logprob_trajectory` had been
retired on `main` and `marker_text: list[str]` reverted to `str`,
both discovered at implementation-time via a post-hoc
`dataclasses.fields()` introspection rather than pre-cherry-pick —
the implementer caught it via the smoke but the failure-mode-catch
was reactive, not preventative.)

---
paths:
  - ".claude/rules/critic-lens-reference.md"
description: >
  Full lens rubrics for critic.md — Methodology / Statistics & Measurement /
  Alternative Explanations — relocated from .claude/agents/critic.md (#838;
  long lines rewrapped to ~100 columns, content unchanged). Loaded ONLY via
  the explicit pointer in critic.md and the codex-critic Step-2 composer read
  — the self-matching `paths:` glob keeps this file out of every other agent
  context (a missing `paths:` key would auto-inject it always-on fleet-wide,
  recreating the #833/#834 spawn-weight bug this relocation fixes).
---

# Critic lens reference (critic.md relocated lens rubrics)

Three H3s, headings verbatim from critic.md. You received exactly ONE lens in
your system prompt: Grep YOUR lens heading and `Read` ONLY that span (chunked,
per critic.md § Context budget) — never the other two lenses. The codex-critic
composer copies the requested lens's items VERBATIM and IN FULL from this file.

### Methodology lens

1. **Hypothesis testability.** Can this design, as written, answer the stated question? If no →
   REVISE (or REJECT if the design is structurally wrong).
2. **Fatal confound.** Is there an alternative explanation for a positive result that (a) the design
   does not rule out, AND (b) the analyzer cannot weigh from the reported diagnostics? Only
   fatal-and-unweighable confounds trigger REVISE — recoverable confounds go in "Concerns for the
   analyzer" (non-blocking).
3. **Technical feasibility.** Will this actually run? OOM, library incompatibility, missing data
   files, eval-surface mismatch. Don't speculate — flag only concrete problems you can name.
4. **Hyperparameter grounding (verify, don't rubber-stamp).** The plan's §11 Decision Rationale must
   give every load-bearing hyperparameter (lr, schedule, warmup, batch / grad-accum, epochs, LoRA
   rank / alpha / dropout, weight decay, seq length, optimizer, precision, anything novel — the full
   set is defined in planner.md §11) a non-empty `Source:` — an arXiv id / paper table, or a prior
   issue `#<M>`. Start from the Phase 1.5 fact-checker's verdict (CONFIRMED / WRONG / UNVERIFIED)
   for each one — you do NOT need to re-open every cited paper; the fact-checker already checked
   value-matches-source and setting-transfer. Your job is the judgment the fact-checker doesn't
   make: would this value, if wrong, change the conclusion? Spot-check independently (arXiv MCP:
   `mcp__arxiv__read_paper` / `arxiv-latex` for the setup / appendix table, or `python
   scripts/task.py view <M>`) only when the fact-checker's verdict looks off or a value smells wrong
   for the Goal. REVISE only when a load-bearing value is BOTH not-CONFIRMED (WRONG, UNVERIFIED, or
   grounded in a source whose setting plainly doesn't transfer) AND plausibly outcome-changing —
   wrong enough to flip the headline or break the run: an lr that would diverge or under-train,
   epochs too few for the trait to transfer, a LoRA rank too low to carry the effect, a seq length
   that truncates the trained completion (see CLAUDE.md `max_new_tokens` rule). A merely
   uncited-but-standard value, or an ungrounded value the plan already flags `needs-smoke-test` that
   wouldn't change the conclusion, is NOT a REVISE — note it as a concern for the analyzer. Be
   sparing here too: the bar is "this hyperparameter would change the conclusion," not "this
   citation could be tighter."
5. **Marker-dynamics logging (marker-implant experiments).** If the experiment implants or tracks a
   marker, the design MUST log per-step marker log-prob + emission rate as a trajectory in WandB
   (per CLAUDE.md). REVISE if the plan captures only end-of-training state — endpoint saturation
   alone cannot distinguish recipes or locate when leakage emerges.
6. **Contrastive negatives for behavior implantation.** If the Goal is to implant a behavior
   (marker, fact, refusal, trait) into a source persona, the data design MUST interleave contrastive
   negative rows over the SAME questions under other personas (always including the bare default
   assistant), at roughly 1:1 positives-to-total-negatives across ≥2-4 close negative personas, with
   on-policy leakage measurement and a non-saturated anchor — per
   `.claude/rules/contrastive-negatives.md` (and planner.md §4 Design). REVISE a positive-only
   behavior-implantation plan unless one of the two named exemptions applies: (a) the single
   manipulated variable IS contrastive-vs-non-contrastive (non-contrastive arm = deliberate control,
   explicitly stated), or (b) a strict single-variable replication of a positive-only parent (parent
   design carried AND no-negatives regime flagged as a scope caveat for the clean-result). The
   reason this is conclusion-changing: positive-only training leaks the behavior uniformly to every
   bystander + the default context, so any selectivity / localization headline cannot be made at all
   from the resulting data (#18, #207). Not a REVISE for non-implantation Goals (the plan should
   write "N/A — not a behavior-implantation experiment" in §4 and you accept that).
7. **Replication fidelity (replicating a published finding).** If the Goal is to replicate a paper's
   result or test whether it holds on our model, the FIRST run must reproduce the paper's actual
   data source, training recipe (SFT-vs-contrastive shape, LoRA rank / epochs / lr /
   checkpoint-selection), dependent variable, AND the paper's own manipulation check — per
   `.claude/rules/contrastive-negatives.md` exemption (b) and planner.md "Before Planning" item 6.
   REVISE when the plan silently swaps in the project's house rig (e.g. a contrastive Sonnet-written
   corpus + default LoRA r=32/α=64 + 3 epochs in place of the paper's ShareGPT-rewrite plain SFT +
   r=8/α=16 + epoch-2), or omits the paper's manipulation check, without naming the deviation as an
   explicit forced-by-project-constraint assumption in §12. The reason this is conclusion-changing:
   a recipe mismatch confounds a null (the null cannot distinguish "the finding does not replicate"
   from "the intervention never took"), and an omitted manipulation check leaves the
   intervention-took question unanswerable on the resulting data (#496: a contrastive Sonnet-warmth
   rig produced a sub-threshold warmth→sycophancy null where the paper used ShareGPT-rewrite plain
   SFT AND skipped the paper's warmth manipulation check, so "warmth doesn't leak" was
   indistinguishable from "warmth never implanted"). Two carve-outs are not REVISE: (a) a deviation
   that is explicitly named in §12 Assumptions as forced by a project constraint (judge model, GPU
   budget, model size) AND carried as a scope caveat — the plan owns the deviation, you accept it;
   (b) a Goal that is not a replication (the plan should write "N/A — not a replication" in §1 or
   §12 as a standalone line and you accept that). Cross-check: a faithful positive-only replication is the named
   contrastive-negatives exemption (b) above — do not double-bounce on item 6 for the same plan.
8. **Few-shot / ICL demonstration content (any plan with in-context-example demos).** If the plan's
   §4 Design includes a fixed bank of in-context-example / few-shot / ICL demonstrations
   (`<question, answer>` pairs the model sees before each probe — read by the trained model, by a
   base model under a persona prompt, or as training-time demonstrations), §4 MUST state, per
   demonstration set: (a) the eval-task distribution the demos mirror, (b) why this specific content
   induces the intended behavior / persona / context, AND (c) that the content varies enough across
   the different ICL contexts to give cross-context dynamic range — per planner.md §4 "Few-shot /
   in-context-example demonstration content" bullet. REVISE when the demonstration design is
   justified only by anti-contamination (no overlap with held-out probe answers) and one or more of
   these failure modes is present: the demos are reused slices of the same generic neutral pool
   across contexts, the demos are near-clones of each other (same answer shape with only a stock
   prefix swapped, e.g. "Arr! Au." vs "Indeed, Au."), the demos are not representative of the eval
   task type (the eval probes open-ended persona-voiced generation but the demos are one-word
   trivia), or the plan is silent on (a), (b), and (c). The reason this is conclusion-changing: an
   ICL bank picked only to dodge contamination tends to give ~zero cross-context dynamic range and
   barely induces the intended persona / behavior, so a null on whatever the experiment was trying
   to test through the ICL channel is indistinguishable from "the ICL channel was inert by
   construction" — the experiment cannot answer its own question (#489: four "neutral" ICL contexts
   were four 4-item slices of one 16-fact trivia pool, persona-voiced demos were stock-prefix wraps
   on one-word answers, and the plan sailed through Planner → Fact-Checker → Critic →
   Consistency-Checker uninspected; the implant floored). Not a REVISE when the plan has no ICL /
   few-shot demonstrations (§4 should write "N/A — no ICL or few-shot demonstrations in this design"
   and you accept that).
9. **Trained-artifact + code reuse — fitness check (any plan that reuses a prior HF adapter /
   checkpoint / training-mix / raw-completion bucket / eval JSON, or a parent's
   fit/analysis/eval/upload-verify
   code helper).** Reusing trained artifacts is the project
   DEFAULT (CLAUDE.md "Reuse existing trained artifacts when fit-for-purpose"; planner.md step 5).
   When the plan records a reused artifact in §10 / §11 — or names a reused
   fit/analysis/eval/upload-verify code
   helper in §4 / §10 — the planner must have verified all of: (a)
   recipe match (same base model + same load-bearing hyperparameters the new question requires; same
   marker token id for marker work; adapter-architecture values — `r` / `lora_alpha` /
   `lora_dropout` / `target_modules` / `use_rslora` — grounded on the artifact's own
   `adapter_config.json`, NOT the producing issue's body Reproducibility row alone, which is
   human-written secondary documentation: on disagreement the config wins and the body row is
   flagged for record-correction — incident #545: a runtime fitness assert encoded #503's erroneous
   body row `r=16/α=32` where the artifacts read `r=32/α=256`, crashing all 7 reuse cells
   mid-sweep); (b) valid measurement regime for the new question (for marker work specifically, NOT
   saturated — source `log P − base ∈ [5,12]` nat, bystanders below ceiling per
   `.claude/rules/marker-training-recipe.md`); (c) the required conditions / cells the new design
   needs are actually present in the artifact — for a multi-field tensor bundle, the REALIZED key
   set verified against every consumer assert (file presence is not field presence; builder-code
   reading is not verification — incident #1073); (d) the reuse does NOT smuggle in a second
   silently-changed variable past the consistency-checker; (e) the producing issue is not retracted
   / superseded; (f) content identity across copies — when the copy the plan verified is a local
   untracked file but execution fetches the artifact's HF mirror, the plan names the pin mechanism
   (`EXPECTED_SHA256` table asserted at prefetch, or an issue-owned `issue<N>_<slug>/inputs/`
   snapshot consumed instead of the parent's shared mirror; resolution alone does not prove the
   mirror matches — `.claude/rules/gotchas.md` "HF mirror ≠ local-verified copy", incident #600);
   (g) application-scaling regime for reused LoRA adapters — the plan reads the reused adapter's
   `adapter_config.json` scaling fields (`use_rslora` / `lora_alpha` / `r`) and pins a 1-adapter
   apply-and-read parity probe reproducing the parent's committed numbers on the CURRENT stack, with
   the read gauge stated in §4 (a recipe-identical parent committed at classic `α/r` application can
   be an unconditional repeater at the faithful `α/√r` a current vLLM+PEFT honors for `use_rslora:
   true` — incident #601: all 20 of #472's reused adapters passed (a)–(f) yet HALTed Phase-0 as
   repeaters); (h) source resolution + consumer-exact path layout + target-backend fetchability +
   staged-layout consumer-open for
   reused TRAINING-INPUT / downstream-input artifacts — for a reused `train/*.jsonl` mix /
   on-policy response cache / downstream-input `eval_results/` JSON / staged multi-file
   tensor or activation store, the plan confirms the file (i) is source-resolvable (HF
   via `huggingface_hub.list_repo_files`, OR git-tree reachable for a committed `eval_results/...`
   JSON) AND (ii) resolves at the EXACT path/filename pattern the new consumer asserts (not merely
   that the parent dir exists — #474 `i474_loc_A1.jsonl` vs a #664-style
   `mk_<source>_<arm>_<dose>_seed42.jsonl` naming drift FAILS this) AND (iii) is fetchable on the §9
   target backend (the git-clone-only GCP/SLURM lanes stage no VM-local `data/`, so a
   parent-built-but-unuploaded mix is unreachable there; committed `eval_results/...` arrive with
   the clone), else the plan uploads / renames the mix to the consumer path first, adapts the
   consumer, or carries a self-contained §4 regen phase (#734 round-4: a reused parent mix on
   neither HF repo AND under a #474 naming convention the #664-style consumer dispatcher did not
   assert crashed phase2 at the pre-train assert on the GCP lane after 3 review rounds), AND (iv)
   when the artifact is staged through a layout-mapping helper (incl. a verbatim prefix mirror)
   into a consumer-fixed local layout, the plan names the hub-rel → local-rel mapping and schedules
   a 1-file staging probe + consumer-open through the REAL staging path before production (#928: a
   verbatim prefix mirror staged the store manifest one level deep and crashed `Store()` init after
   legs (i)–(iii) passed); (i)
   throughput fitness of reused fit/analysis/eval/upload-verify CODE — inner per-cell/per-fold/per-draw loop
   batched + device parametrized + data-repo Hub calls prefix-scoped; full text in `.claude/rules/artifact-reuse.md` checklist item (i)
   (referenced by pointer, not duplicated here; "checklist item (i)" is distinct from this item's
   REVISE-direction romans below). REVISE in
   two directions: (i) the plan REUSES an artifact without naming the producing issue's recipe and
   confirming each load-bearing value matches, or grounds adapter-architecture expectations solely
   on the parent body's Reproducibility row without reading the artifact's `adapter_config.json`, or
   the cited artifact sits in a regime the new DV cannot resolve (e.g. reuses a fully-saturated
   #448-style anchor to answer a graded leakage question), or the cited artifact is missing
   conditions the new design enumerates, or the plan reuses a TRAINING-INPUT mix / on-policy cache /
   downstream-input eval JSON that FAILS source-resolution check (h)(i) (not on HF AND not
   git-tree-reachable as a committed `eval_results/...`) OR fails consumer-exact path layout check
   (h)(ii) (the reused mix resolves in the parent dir but NO file matches the consumer-asserted path
   pattern — #474 `i474_loc_A1.jsonl` vs a #664-style `mk_<source>_<arm>_<dose>_seed42.jsonl` naming
   drift) OR fails target-backend fetchability check (h)(iii) on the §9 lane (e.g. HF-resolved but a
   CDN/region/`HF_TOKEN` gate blocks the lane from staging it) OR fails staged-layout consumer-open
   check (h)(iv) (a staged multi-file reuse with no named mapping and no staging-probe +
   consumer-open gate — #928), without an upload-first /
   rename-to-consumer-path step, a consumer-adaptation, OR a self-contained §4 regen phase — the
   resulting numbers will silently confound the result, or phase2 crashes at the pre-train `assert
   data_path.exists()` on a git-clone-only lane; or the plan reuses a mutually-dependent artifact
   PAIR (bank vs capture, mix vs adapter) without the item-(j) provenance-coherence check — a
   consumed input regenerated AFTER its dependent capture crashed #922 at the parity assert after
   a full GCE cycle despite per-member sha pins; or the plan reuses a parent's main-resident CODE
   module while the parent's issue-<M> branch carries unmerged commits touching it, without the
   item-(k) lineage diff (port-or-declare), or reuses a realized artifact whose row count falls
   short of its declared input corpus without naming the filter — #1345's main-resident extractor
   lacked the parent branch's degenerate-row filter (realized n=4724 vs corpus 5000 was the tell)
   and crashed production at the first unfiltered row; or the plan reuses a parent's
   fit/analysis/upload-verify CODE
   without the checklist-item-(i) throughput inspection (inner per-cell/per-fold/per-draw loop
   batched? device parametrized? data-repo Hub calls prefix-scoped? —
   `.claude/rules/vectorize-many-cell-fits.md`, gotchas.md #833; #761's reused
   serial `_ridge_predict_loco` ran ~100× over plan, #763/#812 inherited a hardcoded
   `DEVICE = "cpu"`, #810's reused verify crawled the full data repo into a 429 wedge) or names
   a caller-side workaround where checklist item (i)'s remedy is a
   source-module fix — the reused serial loop / CPU pin then blows the §9 wall-time projection;
   (ii) the plan RETRAINS / REGENERATES something an
   existing fit artifact already covers (per the step-5 artifact search) without a one-line
   justification for why the existing artifact does not fit — this wastes GPU-hours and breaks
   sibling-comparability. Not a REVISE when the plan reuses an artifact AND records its fitness
   check (a)–(k) inline (in §10 / §11 / §12 — the planner's call) so the consistency-checker and
   downstream analyzer can re-check; not a REVISE when the plan retrains / regenerates AND names the
   specific fitness-check failure that licenses it (a checklist-item-(i) failure licenses NO retrain
   and NO caller-side workaround — its remedy is the source-module fix, then reuse; a
   checklist-item-(k) failure likewise licenses no retrain — its remedy is port-then-reuse).
   REVISE also when the design carries a reuse-VALIDATION gate (a numeric parity floor, a
   behavioral install confirmation, a one-cell gate) whose threshold is a bare constant not
   derived from the reused artifact's own committed per-behavior reference values (file + field
   named in §4/§11), or that assigns run-abort HALT to a weaker-than-expected diagnostic without
   a discriminating-band placement — full rule: `.claude/rules/artifact-reuse.md`
   § Reuse-validation gate calibration (#813: 3 launch-halts + ~1.6h of 8×H100 on a
   7-module-calibrated 0.01 floor and an ungroundable behavioral bar against a correctly-applied
   4-module marker adapter).
   Conclusion-changing because (i) a wrong-recipe /
   saturated / missing-conditions artifact produces numbers that look like results but answer a
   different question, and (ii) gratuitous retraining changes the inherited baseline so the new
   result can't be lined up against the parent's. Cross-check: this lens does NOT fire when the plan
   has no reuse to verify AND no existing artifact would fit (i.e. genuinely new training is
   necessary — say so and item 9 accepts that); a plan with NO data/model artifact reuse but WITH
   a reused fit/analysis/eval/upload-verify helper is NOT such an exit — checklist item (i) still fires on the
   code reuse. Existence verification of HF paths is already
   handled by planner.md step 5's `huggingface_hub.list_repo_files` check; this item is about
   FITNESS beyond mere existence.
10. **CPU/analysis-phase placement — idle multi-GPU pod (efficiency), oversized-VM-footprint
    (disk/RAM safety), gradient-descent / dense-factorization fit — or any high-count tiny-op battery (draws, per-item serialization, per-file uploads) — mis-routed to CPU or left serial (compute character), AND a narrow GPU phase
    holding the run's peak-width pod (GPU-width right-sizing).** A CPU/analysis phase must be placed
    where it neither holds an idle multi-GPU pod NOR overruns the disk it runs on NOR runs an
    iterative-optimization fit GPU-starved on the VM CPU; and a multi-phase GPU run must size EACH
    phase's GPU width separately, never holding the run's peak-width pod through a long narrow /
    API-bound phase — planner.md §9 "CPU-only phases run OFF-POD by default" + its data-footprint
    carve-out + its compute-character carve-out + "Per-phase GPU-WIDTH right-sizing" are the
    governing rules. Four REVISE directions; (i) and (ii) apply to a long CPU-only phase (longer
    than ~15-30 min: bootstrap / permutation statistics, metric aggregation over eval JSONs,
    Claude-judge-only scoring passes, plotting):
    - **(i) Idle multi-GPU pod (the one named efficiency exception to The Bar).** REVISE when the
      plan schedules the phase to run ON a multi-GPU pod without EITHER (a) a stated data-locality
      justification (the phase needs large pod-local artifacts that aren't uploadable — activations,
      per-step checkpoints), OR (b) sequencing that stops / terminates the pod before the CPU phase
      starts (uploads scheduled ahead of the phase; the phase runs on the VM against the uploaded
      artifacts). A **long terminal UPLOAD phase (raw completions / store tensors / checkpoints)
      scheduled to run on the still-held multi-GPU pod** is an instance of this defect — HF
      `upload_folder` / `upload_file` use no GPU, so a terminal upload that keeps the pod alive is
      the same idle-but-billing burn (#664: an 8×H200 pod held ~12h in a per-file raw-completions
      upload phase at 0% GPU, ~$530). This is deliberately narrow: it is NOT about cheaper variants
      of the science (still banned by The Bar) — it targets only an idle-but-billing pod the plan
      never needed to hold (2026-06-09: pod-518 ran 1h+ of pure-CPU permutation/bootstrap scoring
      with all 8 H100s at 0%, pod-523 ran a CPU-only metrics phase ~6h on idle GPUs — ~$48/hr of
      idle burn).
    - **(ii) Oversized footprint placed on the VM (disk/RAM safety).** REVISE when the plan routes the
      phase to the VM (the off-pod default) but its estimated local footprint exceeds
      `VM_ANALYSIS_FOOTPRINT_GB_MAX = 50` GB — `downloaded_inputs_gb +
      materialized_tensors/activations/store_gb + scratch_gb`. The VM root disk is ~188 GB and
      SHARED across the whole fleet, so a >50 GB phase on the VM can fill `/` mid-run and stall
      every concurrent session. Such a phase must instead be routed to a pod / GCP instance with a
      big ephemeral volume sized to the footprint — on the GCP lane the concrete intent is
      `cpu-bigmem` (CPU-only `gpu_count=0` `n2-highmem-16`, boot disk via `--boot-disk-gb`; #677;
      `cpu-bigmem` has NO cheap RunPod equivalent, so an exhausted `cpu-bigmem` run surfaces a typed
      `cpu_exhausted_no_runpod_lane` terminal, not a RunPod fallback — the cheap CPU intents
      `cpu-small` / `cpu-mid` DO fall over GCP→RunPod CPU as of #747, but they are for SUB-50-GB
      work, so a >50 GB phase still belongs on `cpu-bigmem`) — OR stream the data without
      materializing it locally (chunked download → process → discard). Also REVISE when §9 places a
      CPU/analysis phase on the VM but states NO footprint estimate at all (the carve-out requires
      one per phase) AND the phase plausibly materializes large local data (activations, a full
      store, many eval JSONs / raw completions). Cleanup backstops (`clean_experiment_downloads.py
      --incremental` between phases, the `vm_disk_guard.py` cron) do NOT rescue a phase whose own
      footprint exceeds the disk — the fix is placement, not cleanup. (2026-06-26: #658's Phase-1
      analysis materialized a 139 GB activation store on the VM worktree on the shared 188 GB disk;
      `/` hit 100% full and the whole fleet stalled.)
      The RAM twin: ALSO REVISE when a VM-placed phase's projected peak RSS is
      ≥~16 GB (single phase, or SUMMED concurrent VM-resident phases crossing
      the same bar), or when a VM-placed phase that plausibly materializes a
      multi-GB resident set (bulk tensor loads, a large draw pool, many
      concurrent fits) states NO RSS estimate at all — the shared VM's earlyoom
      (SIGTERM below ~12.8 GB MemAvailable, +300 python bias) makes such a phase
      the default kill victim under fleet pressure, and runtime choom protection
      is mitigation, not permission (#778: a 22-GiB-RSS null battery
      earlyoom-killed 3× before its cpu-bigmem pivot; #833: two ~13-15 GB
      concurrent phases lost 5 cells). The fix is placement — `cpu-mid` (32 GB
      GCP) / `cpu-bigmem` (128 GB), with `--min-ram-gb` stated when sizing
      >16 GB (arms the #1010 feasibility gate; the RunPod cpu-mid fallback has
      only 16 GB) — or a stream-reduce formulation that bounds peak RSS at
      O(one item). Full recipe: `.claude/rules/plan-compute-sizing.md`
      § CPU-phase RAM/RSS routing.
    - **(iii) Gradient-descent, many-cell dense-factorization fit, OR any high-count tiny-op battery silently placed on the VM CPU / left serial (compute character).** REVISE when §9 routes an **iterative-optimization fit** — a torch-MLP LOCO / leave-one-class-out fit, a per-cell probe trained via SGD / AdamW, a small adapter fit, or any phase whose inner loop runs gradient descent on parameters — to the VM CPU default (or treats it as cheap closed-form CPU work), per planner.md §9 "Compute-character carve-out". Such a fit is GPU-worthy even at small model / dataset size and must route to a GPU lane (a GPU pod or the GCP GPU lane: `lora-7b` for a full A100, `eval` / `debug` for a smaller GPU — the smallest intent that fits). This axis is ORTHOGONAL to footprint: a gradient-descent fit goes to a GPU lane whether its footprint is large or small. A >50 GB gradient fit goes to a GPU lane with its disk sized explicitly (`--boot-disk-gb` on the GCP lane, `--volume`/intent volume on the RunPod lane), NOT `cpu-bigmem` (`gpu_count=0`, which would re-starve the fit); a closed-form aggregation with a >50 GB footprint still routes to `cpu-bigmem` per (ii). The qualifier is "iterative gradient descent on parameters" (the AdamW / SGD inner loop), NOT "uses pytorch" — a single closed-form torch reduction (`torch.linalg.lstsq`, a vectorized bootstrap) stays cheap CPU work. The "vectorized" qualifier is load-bearing, and the CHECK fires on intent, not implementation wording: ANY non-trivial permutation / bootstrap / null-draw battery over a large fixed/pooled set — non-trivial per the SAME ~15-30 min phase-wall floor as the rest of this item — triggers scrutiny UNLESS the plan explicitly states the draws are already batched/vectorized or the loop is sub-minute (#778's plan never said "serial"; it just scheduled the battery, and serial was the default implementation). REVISE when the plan schedules per-draw re-reduction of the pool or simply names the battery with NO batching/vectorization plan: the fix is a batched formulation (pool reduction precomputed once; mean/sum/covariance draws as one GEMM via the subset-sum identity, median/rank draws via batched `argsort` — `.claude/rules/vectorize-many-cell-fits.md`), NOT a GPU or bigger-CPU re-route, which leaves the redundant per-draw recompute in place (#778: ~4.1 s/draw serial `perm_null_draws`; ~15h projected across the full null battery's draw loops vs the plan's 1h §8 estimate; ~70× batched). The SAME intent-fired scrutiny covers many-cell repeated dense linear-algebra fits: REVISE when §9 schedules a full svd/eigh/lstsq/GCV-ridge solve looped over fold × layer × arm × trait with NO shared/batched-factorization plan, or with a per-call cost not grounded on a MEASURED 1-cell pilot through the production entrypoint at production shape/device, a cited prior-issue measured figure (same kernel + shape), or a pre-registered `pilot-gated` first-step pilot per `.claude/rules/plan-compute-sizing.md` § Per-cell fit phases (a FLOP floor is the cross-check, never the basis for these overhead-bound loops; #811: one inner kernel timed, the dominant frame asserted at "~1–2 h", 19h21m at unit 3/108) (#823: "~2 s/fit" asserted; ~125 s/fit real at N_tr≈4000, H=3584; ~3780 calls, 12-20 h — the body-named Gram-space fast twin was dropped). The fix is Gram/dual-space or a shared factorization, NOT a GPU/bigger-CPU re-route. The SAME intent-fired scrutiny covers ANY high-count tiny-op battery regardless of op class — >~10^4 closed-form tiny fits (#813's substrate-swap null: ~2M tiny fits projected 10-12 h serial), per-item SERIALIZATION of many multi-hundred-MB artifacts, and per-file Hub commits (#813: `savez_compressed` at 103.8 s/file made the store, not the forwards, the wall-clock driver — 4.5× over plan): REVISE when §9 schedules such a battery with NO batching / vectorization / out-of-band-IO plan, under the same ~15-30 min phase-wall floor. A genuinely vectorized battery (draws already batched) stays exempt cheap CPU work. The size gate is the SAME ~15-30 min floor, on the PHASE wall-time (the whole fit loop in aggregate), NOT any single fit: a many-cell/many-draw loop of individually-fast fits/draws counts if the loop runs longer than the floor, while a genuinely tiny one-off fit below the floor (a single linear probe trained in < 30 s, no long surrounding loop) stays on the VM — do not over-route trivial fits. (#658: `_fit_mlp_loco` ran a 300-epoch AdamW fit per cell on the VM CPU, a long per-cell loop that was GPU-starved.) When ANY lens's recommendation raises draws/B/N/cells, the Statistics lens item 12 same-round re-cost obligation applies — cross-check the affected §9 rows were re-costed.
    - **(iv) Narrow GPU phase holding the run's PEAK-width pod (GPU-width right-sizing).** REVISE
      when a multi-phase GPU run sizes ONE pod at its peak-phase width (e.g. 8× H100 for a
      finetuning fan-out) and holds it through a GPU phase that needs MATERIALLY FEWER GPUs — a ≤7B
      forward/activation-extract (~1–3 GPUs), a ≤7B single-GPU vLLM generation, a per-cell probe
      read — OR an API-bound graded-judge phase (Anthropic Batch API, ~0 GPU), where that narrow /
      API phase runs LONGER than ~15–30 min (the same floor as (i)–(iii)) AND the plan states no
      re-provision-cost justification for holding the wide pod — per planner.md §9 "Per-phase
      GPU-WIDTH right-sizing". The plan must size per-phase GPU width (not one peak spec for the
      whole run), name which phase justifies the peak width, and release/downsize the wide pod
      before a long narrow / API phase (terminate + a fresh narrow provision, a separate narrow pod
      up front, or off-pod — a same-pod `pod.py stop`/`resume` preserves the SAME GPU spec, so it
      alone does not re-width). The API-bound judge phase specifically must be SEQUENCED after the
      wide pod is released so its free, off-pod, deadline-bounded `batch_judge` poll waits with no
      GPU held. This is deliberately narrow: it targets only an idle-but-billing WIDE pod the plan
      never needed to hold through a narrow / API phase (#778: an 8× H100 pod held ≤5% util for 38
      min through extract + the API-bound judge phase at ~$25/hr; only the 24-run finetuning fan-out
      needed 8-wide) — the same idle-but-billing family as the #664 terminal-upload defect in (i).
      Do NOT double-bounce with (i): (i) targets a CPU-only phase on a GPU pod; (iv) targets a
      genuinely-GPU-but-NARROW phase on the run's WIDE pod. Do NOT REVISE a plan that holds the wide
      pod through a SHORT narrow phase (< ~15–30 min) with the re-provision-churn-vs-idle-$ tradeoff
      stated, nor one that correctly provisions the wide pod ONLY for the wide phase, nor a
      shared-nothing sweep that runs N seeds SIMULTANEOUSLY on one wide pod (planner.md §9
      Sweep-parallelism row — every shard needs the pod at once, which is phases of the SAME width
      run in parallel, NOT a sequence of phases of DIFFERENT widths). Conversely, REVISE a plan
      that leaves a DECLARED shardable axis (>~2 h serial on 1×) on a narrow GCP provision without
      justification — the width-aware auto lane (#1121) makes `--gpus N` wide provisioning the
      encouraged default; "GCP only had 1× intents" is no longer a valid reason.

    Plan-time scheduling / routing only, never a mid-run cost or disk gate. Not a REVISE when the
    plan declares the phase off-pod on the VM AND its footprint is ≤50 GB (or it streams without
    local materialization), justifies pod-side with a named pod-local dependency or a >50 GB
    footprint, or the phase is genuinely short (~<15-30 min) with a small footprint — nor, for (iv),
    when the narrow / API phase is genuinely short (~<15–30 min) OR the plan states the
    re-provision-cost-vs-idle-$ tradeoff for holding the wide pod, nor a shared-nothing sweep of N
    SAME-width seeds run SIMULTANEOUSLY on one wide pod (which is NOT a sequence of phases of
    DIFFERENT widths).
11. **Marker stopping recipe grounded in a non-marker parent (parity is not a Source) +
    runtime-guard smoke-verifiability.** If the plan trains a FRESH marker / behavior-implant
    adapter, the stopping recipe (lr, epochs / steps, checkpoint selection) must be grounded in
    `.claude/rules/marker-training-recipe.md` (lr ≤5e-6 clean window; log-prob band-stop gated on
    bystander resolution) — per planner.md §4 "Marker / behavior-implant stopping recipe" and §11
    "Marker recipe overrides parent parity". REVISE in two directions: (i) §11 grounds any
    stopping-recipe value in a NON-marker parent's hyperparameters on parity grounds (`Source: #<M>`
    where #<M> implanted a different payload under a different loss shape), or rejects the recipe
    value BECAUSE it "breaks parity" with such a parent, without naming the deviation in §12 —
    single-variable parity with a non-marker parent lives on the DV / eval side, never the
    training-stop side; (ii) the plan relies on a runtime guard / monitor (saturation guard,
    trajectory logger, auto-fired secondary DV) as a load-bearing mitigation for a known failure
    mode WITHOUT naming the smoke-verifiable telemetry the implementer will demonstrate (a logged
    trajectory point, distinct per-source WandB run names, the guard branch or its precondition
    assert exercised) — an unverifiable guard is a paper mitigation. Conclusion-changing because an
    inherited non-marker recipe saturates the marker (no countervailing loss term under marker-only
    loss), pinning bystander cells at a fake floor, and a never-fired guard defers the catch to eval
    time after the pod cycle (#480: the plan grounded lr=1e-5 in "#411 parity", rejected lr=5e-6 as
    "breaks #411 parity", AND declared a WandB trajectory monitor + KL auto-fire that silently never
    functioned — 5 of 6 source runs reused one WandB run name, zero saturation markers fired; all 6
    adapters saturated and the experiment needed a full band-stopped retrain). Not a REVISE when the
    parent IS a marker run trained per the recipe (inheriting its recipe is the normal reuse path —
    items 4 and 9 cover it), when the plan grounds the stopping recipe in the marker recipe AND
    names any parent-parity deviation in §12, or for non-implantation Goals (the plan's §4 "N/A —
    not a behavior-implantation experiment" satisfies this item).
12. **Multi-arm resolution-band simultaneity (anchor-gated designs).** If the plan's headline test
    gates on ≥2 arms sitting simultaneously inside a measurement band at a matched training amount
    (the role-vs-system class), the band-stop default does not apply (per-arm early-stopping would
    unmatch the training amounts) and the plan must carry the three elements of planner.md §4
    "Multi-arm resolution-band designs": a `Source:`-cited install-transition window in optimizer
    steps, checkpoint spacing finer than that window (optimizer steps, never whole epochs), and a
    pre-registered per-arm band-entry fallback read for the no-co-resolution case (compare arms at
    their respective band-entry checkpoints — matched dial position, unmatched step count). REVISE
    when the plan (i) grids in whole epochs or coarser than the cited transition window, or (ii)
    lacks the fallback read. Conclusion-changing because without these the run cannot fire its own
    headline test when the arms fail to co-resolve — three consecutive runs (#529 epoch grid at
    lr=1e-5, #533 lr drop to 5e-6, #546 rank drop to r=16) burned GPU without the anchor-gated test
    ever firing, and "arms never co-resolve under this recipe" went undiagnosed each round instead
    of being reported as the decidable outcome it is (per `.claude/rules/marker-training-recipe.md`
    § Multi-arm resolution-band designs). Not a REVISE when the headline test does not require
    multi-arm band simultaneity (single-arm band-stop designs are covered by the recipe default; the
    plan's §4 "N/A — no multi-arm band-simultaneity gate" satisfies this item).
13. **Compute projection costed on the routed machine + GCP fence reconcile + store-heavy IO sizing (verify §9).** The
    plan's §9 compute table must cost each row's `planned_wall_h` / `basis` on the machine the
    backend router will ACTUALLY provision — under the standing GCP-FIRST `auto` default that is the
    `INTENT_TO_MACHINE` mapping in `src/explore_persona_space/backends/gcp.py` (`lora-7b` → 1×
    A100-80 `a2-ultragpu-1g`, `ft-7b` → 4× A100-80, `eval`/`debug` → 1× L4), NOT the RunPod H100
    intent table — with any basis measured on a different GPU scaled by a stated per-step rate; and
    reconcile the WORST-CASE wall (base phases PLUS every conditional / extension phase riding the
    same provision) against the GCP lane's auto-delete fence (`--instance-termination-action=DELETE`
    + `--max-run-duration`, default 7d — the FLEX_START ceiling, #741) — per planner.md §9 "Cost
    wall-time against the machine the router will ACTUALLY provision". REVISE when (i) a wall-time
    basis is premised on a machine the router won't route (e.g. H100 numbers under the GCP-first
    auto default) with no stated cross-GPU scaling, or (ii) worst-case wall on the routed machine
    approaches the routed lane's fence (the GCP `--max-run-duration` default is 7d — the FLEX_START
    ceiling, #741 — but a plan may deliberately set a SHORTER fence via
    `spec.extra["max_run_duration"]`, in which case reconcile against THAT value) and the plan
    declares none of: (a) a deliberate `spec.extra["max_run_duration"]` for the GCP dispatch —
    option (a) satisfies this item ONLY when the fence is sized off the p90 per-cell wall estimate
    (a prior-issue per-cell wall distribution, else the measured mean × a STATED dispersion
    factor, default ×2) and cleared with stated margin (≥~1.25×); a MEAN-sized deliberate fence is
    exactly the #833 failure (realized per-cell wall ran ~2× the plan mean and overran a
    deliberate 36h fence, hard-deleting the instance) and is itself a REVISE — fence-sizing
    recipe: the fence clause of `.claude/rules/plan-compute-sizing.md` § Cost wall-time against
    the machine the router will ACTUALLY provision — (b) a
    pre-registered phase split across provisions naming which artifacts persist (HF / git per the
    Upload Policy) before the first instance dies, (c) an explicit `backend: runpod` override with
    the long-run residual gap named; and (iii) for a store-heavy phase (>~10^3 output files or
    >~50 GB written), REVISE when the §9 row's basis states bytes/file-counts but NO measured
    one-item production-shape serialization + upload wall-time, or when the plan defaults
    client-side compression ON for fp16 tensors bound for a Xet-backed HF repo without a measured
    ratio + wall-time justification (#813: a bytes-only one-cell gate passed while
    `savez_compressed` at 103.8 s/file vs 1.2 s plain — a 1.29× ratio, with Xet dedup already −59%
    server-side — drove the store phase 4.5× over plan on an idle 8×H100) — per
    `.claude/rules/plan-compute-sizing.md` § Store-heavy / IO-heavy phase sizing.
    Conclusion-changing because a wrong-machine premise silently
    multiplies wall-time past the fence and the instance hard-deletes mid-phase, losing the phase's
    data outright (#599: an H100-premised ~6.4h estimate ran ~6× slower per-step on the routed A100
    lane, and the pre-registered §7.3 extension probe was hard-deleted at step 149/2400 by the 24h
    fence). Not a REVISE when the plan pins the lane via `backend:` frontmatter and costs its bases
    on that lane's machine, or when worst-case wall on the routed machine sits comfortably under the
    fence — this is plan-time scheduling only, never a mid-run cost gate.
14. **Completion provenance — on-policy-first positives for behavior implantation.** If the Goal is
    to implant a behavior (sycophancy, refusal, hedging, style, trait) into a source persona, the
    plan's §4 must name each training-row type's completion provenance (`on-policy (tier 1/2/3)` |
    `canned/template` | `third-party-LLM-written` | `published-corpus-verbatim`) with on-policy
    positives as the DEFAULT — behavior instruction in the system prompt, judge-filtered,
    elicitation instruction stripped before training, pre-registered per-source yield quota + drop
    rule — per `.claude/rules/on-policy-completions.md` and planner.md §4 "Completion provenance".
    REVISE when (i) the positive completions are canned/templated or third-party-LLM-written WITHOUT
    either an explicit anchor/control role (the data construction IS the manipulated variable,
    stated as such) or a recorded on-policy yield failure for that source/behavior, or (ii) the plan
    backfills a yield shortfall with templates inside an arm labeled on-policy instead of dropping +
    reporting the source, or (iii) a MULTI-behavior implantation datagen defines its behaviors
    bespoke — hand-written per-behavior definitions or hand-curated per-behavior query banks —
    instead of the standardized persona-vectors shape (trait name + description → 5 contrastive
    pos/neg instruction pairs + shared/auto-generated neutral question set;
    `.claude/rules/on-policy-completions.md` § Standardized behavior definitions) without a stated
    justification — #906's bespoke 4-class pilot failed all three content-class yield floors
    (sycophancy 6/36 vs floor 20) and forced the #1090 standardized rebuild. The reason this is
    conclusion-changing: canned/templated positives
    collapse the response distribution and overstate installability — #612 measured the model's own
    judge-accepted completions installing at +0.60-0.66 where canned templates installed +0.84-0.93
    under the identical recipe — so a canned-data headline about install strength / leakage radius
    does not transfer to realistic data, and a silent template backfill makes the "on-policy" arm a
    mislabeled mixture that cannot answer its own question. Not a REVISE: positives that ARE
    on-policy with the ladder + quota stated; an explicitly-labeled canned anchor/control arm;
    published-corpus replication rows (replication fidelity, item 7, wins — do not double-bounce);
    the marker carve-out (the response text is already on-policy under the marker recipe; the
    appended token is the controlled template); a single-behavior design's bespoke definition (no
    cross-behavior comparison to confound); an established/published benchmark bank justified
    under data-realism / replication fidelity with the cross-behavior-comparability caveat named
    (the standardized behavior DEFINITION is still required); or non-implantation Goals (the
    plan's §4 "N/A — not a behavior-implantation experiment" satisfies this item).
15. **Data-source realism tier (verify §4 names source + tier).** Every `kind: experiment` plan
    names its training/eval/probe data source AND its tier on the CLAUDE.md realism hierarchy
    ("Design experiments on the most realistic data available"): (1) real-world, (2) established
    dataset / benchmark, (3) diverse LLM-generated synthetic, (4) programmatic/templated — per
    planner.md §4 "Data source + realism tier". REVISE when the plan picks tier 3 without a
    justified absence of tiers 1-2, or tier 4 without an explicit recorded argument for why no other
    source works AND why the templated structure cannot bias the result. Conclusion-changing because
    programmatic/templated data collapses the distribution the trained behavior generalizes to,
    confounding every behavioral claim (the CLAUDE.md default presumption) — and a flat templated
    corpus with an LLM in the loop is tier 4 in tier-3 clothing, not diverse synthetic. Not a REVISE
    when the tier is named with its required justification, when the construct under test IS the
    controlled template (the marker carve-out — cross-check item 14's provenance exemptions, do not
    double-bounce), or for `kind: analysis|infra|batch|survey`.
16. **Merge-disk budget vs per-pod quota (feasibility — verify §9 for transient full-precision
    artifacts).** If the plan has a phase that materializes full-precision model artifacts DURING
    iteration — a LoRA adapter merged onto base weights for a read (dose-checkpoint selection, eval
    needing a merged dir), a ZeRO-3-consolidated full-FT checkpoint, a per-step / per-cell model
    copy — verify §9 states the upper bound on COEXISTING on-disk full-precision artifacts (`n_cells
    × max_concurrent_artifacts_per_cell × per_artifact_size_gb`; a merged Qwen-2.5-7B is ~15 GB) AND
    that it fits the per-pod disk quota (RunPod MooseFS ~130 GB per-pod cap — `df -h /workspace`
    shows the TB share, not the per-pod limit; SLURM / GCP per-node scratch budget — per planner.md
    §9 "Merge-disk budget" and `.claude/rules/gotchas.md`). REVISE when the upper bound exceeds the
    quota AND the plan does NOT specify the cleanup pattern (which artifacts persist, which are
    transient, when each transient one is deleted: cleanup-as-you-go / atomic merge-read-delete per
    probe / scratch-dir rotation). Conclusion-changing because an unbounded transient-merge
    accumulation silently EDQUOTs (`OSError errno=122`) mid-run and the phase dies producing no
    result — a feasibility failure, not a cheaper-variant suggestion (#653 round 4: the
    `select_checkpoint` phase merged a ~15 GB copy per probed dose checkpoint × 12 content cells × 9
    dose ckpts = ~1.6 TB worst case on a 130 GB quota with no cleanup between probes; the run hit
    the quota and died, the fix was atomic merge-read-delete per probe).
    LADDER-RETENTION EXTENSION (#1133, from incident #1112): when the phase is
    a dose-ladder / multi-rung checkpoint phase (per-rung checkpoints persisted
    for later selection — dose-to-band grids, band-stop ladders, per-step save
    grids), ALSO verify §9 states the checkpoint-retention policy (default:
    keep dose-selected + latest, delete ruled-out rungs between rungs — per
    `.claude/rules/plan-compute-sizing.md` § Dose-ladder / multi-rung
    checkpoint retention) and sizes disk to the RETAINED set + in-flight rung.
    REVISE when the ladder plan's disk estimate assumes keeping every rung on
    local disk without an explicit justification that (a) says why the rungs
    must coexist, (b) sizes the FULL ladder at realized per-rung size (weights
    + optimizer state), and (c) declares the requirement in the launch flags
    (`--boot-disk-gb`, arming the #1118 thread-or-refuse) — a keep-all bound
    that merely fits the PLANNED lane's disk is NOT sufficient (#1112: a
    compliant 575 GB keep-all bound under a planned 750 GB GCP boot disk
    ENOSPC'd at rung 24/30 when the GCP→RunPod failover delivered the ft-7b
    default 200 GB volume). The fits-quota escape below does NOT cover ladder
    phases; the ladder escapes are a stated retention policy + retained-set
    sizing, the justified keep-all above, or "N/A — no multi-rung checkpoint
    phase". Upload-before-delete unchanged (declared §10 discard OR
    upload-first before any between-rung deletion). Plan-time storage-budget
    check only, never a mid-run gate. Not a REVISE when no phase materializes transient
    full-precision artifacts at scale (single merged copy, or merges that fit the quota with
    headroom — the plan's §9 "N/A — no transient full-precision merges" or a bound under quota
    satisfies this item), or for `kind: analysis|infra|batch|survey`.
17. **Persona-vectors extraction fidelity (any plan that elects persona vectors).** If the plan
    extracts a persona/behavior direction via "use persona vectors" / "extract a persona vector" /
    "persona-vectors-style direction" or a mean-difference of positive/negative contrastive
    activations, verify §4 instantiates the recipe of `.claude/rules/persona-vectors-recipe.md` and
    names its layer-selection regime — per planner.md §4 "Persona-vectors extraction recipe". REVISE
    when the extraction: **(a)** uses MISMATCHED corpora (two different / unrelated question sets)
    instead of the SAME pos/neg system prompts over a SHARED extraction question set; **(b)** OMITS
    the judge-filter (keep positive responses >50, negative responses <50, judge
    `claude-sonnet-4-5-20250929`), OR fails to specify that a `REFUSAL`/non-numeric/out-of-[0, 100]
    judge return is DROPPED from BOTH arms (never coerced to a numeric score — coercing a refusal to
    0 silently keeps it as a clean `<50` negative and corrupts the negative-arm mean, worst for the
    very traits where elicitation provokes the most refusals; the per-arm dropped-rollout count must
    be reported); **(c)** extracts at a PROMPT position (prompt-last / prompt-avg) instead of
    response-avg; **(d)** SKIPS the on-policy rollouts (uses canned / teacher-forced completions for
    activation extraction); or **(e)** REINTRODUCES the paper's GPT-4.1-mini logit-weighted scoring
    (or any second judge) WITHOUT an explicit `### Override:` note in the plan. Conclusion-changing
    because each of (a)–(e) makes the resulting `r_B` a different object than the paper's persona
    vector (the #658 divergence: a mismatched-corpora, unfiltered direction that does not measure
    the trait the way the published recipe does), so any downstream read off that direction silently
    answers a different question. Not a REVISE when the plan instantiates the full recipe and names
    its regime (steering vs read-out), or when the plan does not elect persona vectors (its §4 "N/A
    — no persona-vectors extraction" satisfies this item). Cross-check: a persona-vectors plan is
    also a replication of a named published recipe — do NOT double-bounce with Methodology lens item
    7 (replication fidelity) for the same recipe-fidelity finding; pick this item for
    persona-vectors-specific failures and item 7 only for a broader replication-recipe deviation.
18. **Persist-by-default — undeclared generation-discard / missing rollout-text persist (verify §10
    + §4).** If the plan has a GENERATION-AND-REDUCE stage (persona-vector extraction; an
    online-scored eval reducing completions to a rate; any stream-reduce over model generations),
    verify §4 lists the rollout TEXT under `raw_completions/<stage>/` and §10 declares any
    deliberate intermediate-tensor discard in the `discarded_artifacts:` slot with `{name, reason,
    regen_recipe}` — per CLAUDE.md § Upload Policy persist-by-default and planner.md §10 / §4.
    REVISE when (i) a generation-and-reduce stage drops its rollout TEXT with no persist declaration
    (text is non-LFS, KB–MB, and the regenerating minimum — dropping it forces a sibling to
    re-sample), (ii) the plan discards a large intermediate tensor WITHOUT a `discarded_artifacts:`
    entry naming the regen recipe (so the upload-verifier cannot distinguish an intended drop from
    silent loss and will FAIL at Step 3), or (iii) the plan's `discarded_artifacts:` slot names
    model generations / rollout text / any text-JSON artifact — the slot licenses ONLY large
    intermediate-TENSOR discards, and a text-naming entry is an invalid declaration the verifier
    will FAIL (`generation-discard-declared-invalid`). Conclusion-changing because a follow-up /
    sibling arm inherits an unrecoverable gap — #779's extraction rollouts were reduced-and-dropped,
    so arms B/C had to regenerate the rollouts from scratch. Not a REVISE when the plan has no
    generation-and-reduce stage (§4 "N/A — no generation-and-reduce stage"), when text is persisted
    + every big-tensor discard is declared with a regen recipe, or for `kind:
    analysis|infra|batch|survey` plans that produce no model generations. mechanizable: partial — a
    future `verify_plan.py` check could assert that a plan naming a generation-and-reduce stage
    carries a `raw_completions/<stage>/` persist line or a `discarded_artifacts:` entry.

### Statistics & Measurement lens

1. **Metric mismatch.** Does the headline metric actually measure what the hypothesis predicts? If
   the metric and hypothesis are about different things → REVISE.
2. **Construct validity / on-distribution proxy (verify §6 Measurement validity).** The metric is
   only a proxy for the Goal's *construct* (the real behavior). Read the plan's §6
   measurement-validity table. REVISE when the DV measures a behavioral construct with an
   **off-distribution or convenience proxy** — teacher-forced instead of on-policy, a fixed
   canonical/stub answer instead of the model's own generation, an arbitrary token position instead
   of where the behavior is emitted, a single-token shortcut that changes what's scored — AND the
   plan neither validates the proxy against the construct nor argues it answers *this* Goal. The
   canonical question: *"Does the eval observe the behavior under the conditions it actually occurs?
   If not, is the divergence justified AND the proxy validated against the on-policy / ground-truth
   behavior?"* A proxy chosen purely for being cheaper / cleaner / deterministic, with no validity
   basis, is a REVISE. (Also flag if the plan's own §6 predicts the proxy will saturate — all
   conditions piled at a floor/ceiling with no dynamic range — since rank-shuffles among saturated
   values aren't interpretable.) Distinct from item 1: the metric can be *about the right thing* yet
   still measured off-distribution so it doesn't track the behavior. Does NOT fire on the
   marker-dynamics trajectory of Methodology lens item 5 — a within-condition per-step log-prob
   delta logged alongside emission rate is a valid dynamics DV; this item targets teacher-forced
   *cross-condition* comparisons at a single checkpoint. **Inherited-positive DV-swap (a follow-up
   reusing a parent's positive predictor-vs-DV result as its grounding).** When the plan cites a
   parent issue's positive predictor↔DV correlation (e.g. "self-scoring predicts trained marker
   pressure, median ρ ≈ +0.61 — #559") as the evidence that the amendment's predictor works, verify
   the amendment scores that predictor against **the SAME DV the parent's positive was measured
   against**, not a `trained − base` change of it. REVISE when the amendment's DV is a `trained −
   base` (or `post − pre`) CHANGE while the predictor is a **base-side propensity** (a base-model
   rate / log-prob / its graded variant): the predictor IS the base term the change subtracts, so it
   enters the change DV with a mechanical coefficient of −1, flipping the predicted sign and firing
   the plan's falsification gate for a DV-choice reason rather than a predictor failure. The check
   is concrete: recompute the parent's actual DV (was the positive measured on the trained LEVEL or
   the change?) and the amendment's own median ρ against BOTH the level DV and the change DV from
   the committed joins — if the level-DV ρ is positive and the change-DV ρ is mostly negative or
   zero, the swap is the cause and the amendment must either score the level DV (preserving
   comparability with the parent positive) or partial out the base term before declaring the
   generalization falsified (#559 cross-behavior-self-scoring amendment / #605: base-rate↔level ρ
   +0.28/+0.19/+0.00 vs base-rate↔delta ρ +0.09/−0.43/−0.54). Not a REVISE when the predictor and DV
   are not the level/change pair of the same quantity, or when the amendment already scores the
   level DV the parent's positive used.
3. **Decision-gate coherence (only when the plan leans on pre-registered kill-gates).**
   Pre-registered kill-gates / thresholds are disfavored by The Bar (above): they crush joint power
   and the analyzer pipeline already assigns confidence from reported diagnostics. *First* ask
   whether the gate is necessary at all vs training the sweep and letting the analyzer weigh the
   result post-hoc — if the gate's role can be filled by post-hoc confidence, the plan should drop
   it. This item does NOT instruct you to ADD gates; it scrutinizes gates a plan already relies on.
   If the plan RETAINS load-bearing gates, cross-check every gate in the plan's Decision Gates
   section for mutual satisfiability and grounding, and REVISE when: (a) two gates impose
   contradictory pass criteria on the SAME measurement at the SAME cell — e.g. one requires `Δ ≥ +x
   nat` and another requires `Δ ≤ −y nat` on the identical probe / slot / target — so the gate set
   is jointly unsatisfiable and the experiment can never pass; or (b) a gate's pass threshold OR its
   SIGN is an ungrounded assumption not tied to prior-issue evidence of the construct (a kill-gate
   that no past result of this construct would itself have passed, or whose sign predicts the
   opposite of what every prior run of this construct produced); or (c) a registered decision band,
   applied to the precedent values the plan itself cites as that branch's supporting reference,
   places the precedent in the OPPOSITE branch (or the cited range straddles the threshold while
   the prose asserts one side) — recompute the arithmetic; do not trust the plan's own side label
   (#825 v17: 0.3489/0.6731 = 0.519 ≥ 0.5 narrated as the below-band reference). `verify_plan.py`
   c27 WARNs on the same-line explicit-ratio subset; this lens owns every other phrasing.
   Conclusion-changing because an
   unsatisfiable gate guarantees a false FAIL (the run cannot answer its own question) and an
   ungrounded-sign gate guarantees a false PASS or FAIL by construction (the threshold is divorced
   from what the construct actually does). Skip the check entirely when the plan has no Decision
   Gates section or its gates are advisory monitoring thresholds rather than pass/fail
   kill-criteria. (Surfaced after task #488: a smoke-gate ladder shipped Gate 3 requiring `Δlogp_off
   ≥ +0.2 nat` and Gate 4 requiring the same probe at the same cell `Δlogp_off ≤ −0.2 nat`; the
   contradiction was diagnosed only at round 10 after multiple days of recipe-thrashing.)
4. **Uninterpretable N.** Is the sample size / seed count so small that signal cannot be
   distinguished from noise at all? "Tighter CIs would be nicer" is NOT a REVISE; "N=2 seeds for a
   noisy outcome" might be.
5. **Numerical accuracy.** Read the JSONs the plan cites. If a number in the plan disagrees with the
   source file, flag it.
6. **Gate elicitation-surface validity.** For every behavioral gate / halt criterion the plan
   retains (items 3a/3b cover coherence; this covers the probe set), verify the gate measures the
   construct on an elicitation surface the construct is KNOWN to express on — cite the prior issue
   that demonstrated expression on that surface. A gate probing emergent misalignment with trivia
   questions, or refusal with benign prompts, will read a floor regardless of whether the behavior
   installed, producing a false HARD-HALT. REVISE when the gate's probe surface has no
   demonstrated-expression citation and a canonical surface exists (e.g. EM expresses on #458's
   first-plot probes with no system prompt, NOT on trivia-question PAIRS). Incident #521
   (2026-06-09): an EM-rate gate on a trivia surface false-halted twice — surviving two critic
   ensembles and two code-review rounds — before a runtime re-measure on the canonical rig showed EM
   was installed all along.
7. **Statistical-input existence (registered corrections).** For every registered statistical
   correction / adjustment §6 relies on (attenuation / reliability factor, per-seed SEs, variance
   reconstruction, shrinkage prior — any statistic computed from a derived input rather than this
   run's raw eval output), verify the plan names the data dependency it consumes AND either (a)
   confirms it exists in the cited artifact (column present in the CSV, per-seed files resolvable —
   checked against the actual file, not the producing plan's prose) or (b) schedules its
   construction as in-scope implementation work — per planner.md §6 "Statistical-input existence".
   REVISE when a registered statistic consumes an input that is neither verified-present nor
   scheduled-to-build. Conclusion-changing because the implementation inherits a phantom dependency:
   at run time the correction either crashes the production path or silently degrades into the
   uncorrected statistic, and the headline ships without its registered adjustment (incident #509:
   plan §6.1 registered attenuation-adjusted correlations whose per-seed SEs the cited CSV —
   seed-averaged only — never carried; the reconstruction was never in-scope for any implementer
   round, the production path crashed exactly as review prose predicted, and the result shipped on
   `--smoke` with reliability pinned to 1.0). Not a REVISE when §6 registers no derived-input
   corrections (raw DV + standard tests only — the plan's "N/A — no derived statistical inputs"
   satisfies this item).
8. **Install-strength confound (cross-condition leakage comparisons).** If the plan's headline
   compares LEAKAGE across training conditions (contrastive vs positive-only, LoRA vs full
   fine-tuning, data-construction variants), verify §6 registers an install-controlled read — a
   matched-install comparison (conditions compared at checkpoints with matched source gain) and/or a
   per-cell leakage-to-install fraction computed in the non-saturating EOS-margin logit space
   `Δ(z_marker − z_eos)` (never raw `log P`: softmax compression understates a saturated source's
   gain and inflates the fraction exactly in the strongest-implant conditions), with
   leakage-vs-install dose curves from per-step trajectories preferred where they exist — per
   planner.md §6 "Install-strength control" and `.claude/rules/marker-leakage-measurement.md` §
   Install-strength confound. REVISE when the headline cross-condition leakage claim rests on raw
   bystander leakage alone with no install-controlled read registered, or when the registered
   fraction is computed in raw log-prob space at a saturated source. Conclusion-changing because
   install strength is condition-dependent — not even in a fixed direction across behaviors (#601:
   contrastive negatives strengthened the marker implant; #608: positive-only sycophancy installed
   at least as strongly) — so raw leakage cannot distinguish lower selectivity from plain stronger
   implantation and the headline answers a different question than the Goal's. Flag as a Concern
   (not REVISE) any plan that proposes correlating the fraction back against install itself — the
   shared noisy denominator manufactures correlation (same family as the #383 X-vs-(X−Y) caveat in
   `.claude/rules/contrastive-negatives.md`). Not a REVISE when the plan makes no cross-condition
   leakage comparison (within-condition dynamics, single-condition implant characterization — the
   plan's §6 "N/A — no cross-condition leakage comparison" satisfies this item).
9. **Degenerate eligibility gates, unequal per-unit N, missing baseline propensity (four related
   design-lesson checks).** REVISE only when conclusion-changing per The Bar; otherwise list under
   Concerns. (i) **All-or-nothing eligibility gate on a continuous quantity:** a pre-registered rule
   gates a unit's inclusion on a continuous quantity (rows filled, judge-filter yield, cells
   surviving a data gate) as a binary keep/drop at the target value with no graceful-degradation
   floor — a near-miss then discards the unit wholesale (#612: a "fill all 200 rows or drop" rule
   discarded one source at 194/200 — 97% fill, 6 missing rows — and another at 169/200, together
   halving the design's coverage; the 80%-floor + equalize-down default is in
   `.claude/rules/on-policy-completions.md`, the general rule in planner.md §4 "No all-or-nothing
   eligibility gates"). (ii) **Unequal per-unit N across compared conditions/units:** the headline
   compares conditions/units whose per-unit N (training rows, samples) legitimately varies, with
   neither equalize-down (all units at the same floor-N) nor an explicit dose control — variable N
   is a dose confound and dose/schedule length is the demonstrated dominant lever (#601), so the
   comparison conflates the manipulated variable with training amount. (iii) **Missing source-side
   baseline propensity in an implantation/elicitation design:** the plan measures pre-intervention
   behavior rate only on the EVAL-side targets (the delta denominator) and not on the SOURCE-side
   personas — leaving elicitation-yield failures unpredictable (#612: both yield-quota failures were
   predictable from a never-taken source-side read) and the natural install-strength covariate
   unmeasured (a unit's own base prior keeps beating geometry — #500/#532/#541). (iv)
   **Structurally-constant observed statistic in an observed-vs-null read:** the plan registers a
   statistic to be compared against a null band whose construction admits NO variation under the
   data — projecting/summing the MEAN (along the centering axis) of mean-centered quantities (≡0
   by construction), correlating a constant vector, a residual of X regressed on itself, a paired
   difference of aliased arrays (#1092: the banked read-4c projected the row-mean of mean-centered
   ANOVA factor outputs onto r_B — observed ~1e-14 vs sign-flip-null p95 0.9–9.2 at all 288 rows;
   survived all 16 code-review rounds, caught at interpretation — #1092 epm:interp-critique v1).
   Trace the registered statistic's reduction chain symbolically, wherever the construction lives
   (including reused parent code); when the centering/aliasing cannot be established from the plan
   alone, require a runtime degeneracy guard (assert the observed magnitude ≫ machine epsilon
   relative to the null scale). REVISE when the statistic is constant by construction — always
   conclusion-changing (the comparison can only ever fail to reject). Not a REVISE when:
   the design has no continuous-quantity eligibility gate (i), per-unit N is equal by construction
   or the headline makes no cross-unit comparison (ii), or the Goal is not an
   implantation/elicitation design (iii), or the plan registers no observed-vs-null comparison
   (iv) — the plan's §4 "N/A" lines satisfy the respective checks.
10. **Dual-DV for content-behavior leakage / implantation (verify §6 names both DVs).** If the Goal
    implants or measures the leakage of a *content* behavior (sycophancy, refusal, hedging, style,
    trait — not the programmatic marker, which has its own three-space recipe), read §6's
    measurement-validity entry, then REVISE in two directions, CALIBRATED to conclusion-changing per
    The Bar: (i) the plan registers ONLY a binary judge-scored behavior/agreement rate with NO
    continuous completion-probability companion (per CLAUDE.md § Measurement validity: PREFERRED a
    teacher-forced FIXED positive-vs-negative completion margin — fixed answer pools ⇒ no selection
    bias, #722 — with the judged-positive-conditional-mean `log P` (`logp_pos_mean`) the
    selection-confounded opt-in alternative that must first pass ρ(DV, rate) > 0) WHEN the plan
    makes install / dose-matched / cross-condition comparisons OR §6 itself predicts ceiling
    saturation — there the binary rate censors exactly the comparison the headline rests on (#608's
    top-band censoring), so the headline cannot be made from the binary DV alone; OR (ii) the plan
    narrates the completion-probability DV as the PRIMARY/construct DV, or registers it without the
    standing validation that it tracks the rate (a Spearman of the probability DV vs the judge rate
    across the cells with dynamic range). The judge rate stays PRIMARY (the validated behavioral
    construct); the probability DV is the SECONDARY non-saturating companion and carries the
    teacher-forcing-artifact risk (#432→#456) the rate is immune to — they cover each other.
    Conclusion-changing because a saturated binary rate makes a real install/dose/leakage difference
    invisible (false null), and an unvalidated probability DV narrated as the construct is an
    overclaim (false positive). NOT a blanket REVISE on every behavior eval — a content-behavior
    eval with no saturating comparison and no install/dose/cross-condition claim need not add the
    second DV (single-condition behavior characterization, descriptive rate reports). Not a REVISE
    for marker implants (separate recipe), non-behavioral analysis, or when the plan's §6 writes
    "N/A — not a content-behavior leakage/implantation experiment".
11. **Selection-symmetric nulls (max-over-axis headlines).** If the plan's headline statistic is
    chosen by `max` / `argmax` / best-of / top-k-mean over a FREE AXIS (a read-out layer, a cell, a
    k / neighbourhood size, a seed, an extraction point, a threshold) AND is compared against a null
    / permutation / shuffle band, verify §6 registers a SELECTION-SYMMETRIC null — EITHER every null
    draw gets the SAME max-over-axis selection before the band is formed, OR the axis is frozen on a
    held-out split / pre-registered fixed position and observed + nulls are read at that single
    position — AND the per-draw × per-axis statistic matrix is persisted so the analyzer can
    recompute the honest band, per planner.md §6 "Selection-symmetric nulls" and
    `.claude/rules/selection-symmetric-nulls.md`. REVISE when the concrete pattern holds: the
    headline uses `max`/`argmax`/best-of/top-k-mean over
    layer/cell/k/neighbourhood/seed/extraction-point/threshold, the null band is computed at ONE
    fixed axis position, and neither per-draw same-selection nor a held-out-frozen axis is
    registered. Conclusion-changing because a `max-over-L` observed statistic vs a one-position null
    is a 28-vs-1 asymmetry that inflates the observed-vs-null gap by the winner's curse (≈ `sqrt(2
    ln L)·SE`), manufacturing a false positive on the headline — #778 (n=24: single-layer null p97.5
    |r| ≈ 0.48 vs honest max-over-layer p97.5 |r| ≈ 0.62; all three Phase-2 lenses reconciled to
    REVISE — four of the six per-lens critics caught the asymmetry independently, Methodology Claude
    and Alternatives Claude missed it), siblings #664 / #545. A per-axis heatmap does NOT neutralise
    it — the comparison statistic stays asymmetric. ALSO verify the band-vs-ceiling report: §6
    states the band's upper bound next to the DV's achievable ceiling
    (`selection-symmetric-nulls.md` § Band-vs-ceiling informativeness check). A registered
    null-band DECISION GATE whose band upper bound ≥ an ESTIMATOR-BOUND achievable ceiling
    (bounded-DV or registered difference ceiling — never the fallback reference point), with no
    failure-to-reject pre-commitment, is a REVISE — the gate is unfireable-by-construction, the
    null-band instance of item 3's joint-satisfiability bar (#810: band p97.5 0.800 vs an
    achievable difference ceiling derived from ~0.857 max skill — even the parent +0.209 effect
    could not clear the band; p = 0.634 initially narrated as
    an ordering fail). A missing ceiling report on a bounded statistic with no gate riding on it —
    or a band exceeding only the fallback reference point — is a binding Concern, not a REVISE.
    Not a REVISE when the axis position is a single
    pre-registered / fixed value with no data-driven selection, a mechanistic single-anchor
    ablation, or the headline is not selected over any free axis (the plan's §6 "N/A — no
    max-over-axis selection in the headline" satisfies this item). If the per-draw × per-axis matrix
    is registered/persisted but the plan shipped the asymmetric read, the honest band is
    analyzer-recoverable post-hoc — carry it as a binding Concern rather than a REVISE.
12. **Re-cost on power-raising recommendations (same round).** Any recommendation in YOUR review
    that raises statistical power parameters — permutation/null draws B, bootstrap N, seeds, cells,
    folds, samples-per-cell — MUST, in the SAME round, re-cost every affected §9 compute row: state
    the new projected wall (new multiplier × the row's per-call basis) and whether the plan's
    batched-implementation commitment still holds at the raised scale. A power raise on a serial
    battery is a compute multiplier, not a free statistical fix (#778: a statistics-lens round
    raised 200→1000 null draws for pooled-BH power with no re-cost, 5×-ing a serial battery that
    then projected ~15 h vs the plan's 1 h). If the re-cost crosses the
    `.claude/rules/vectorize-many-cell-fits.md` trigger, the SAME recommendation names the batched
    implementation. The obligation binds every lens (Methodology item 10(iii) cross-references
    here); it lives in this lens because power raises originate here. Not a REVISE-generator by
    itself — it is an obligation ON your own recommendations.

13. **OOD generalization folds (group-structured held-out predictive DVs).** If any DV the plan
    reports is a held-out predictive statistic (reconstruction R² / skill, read-out ρ, predictor
    accuracy — any "held-out" / "cross-validated" number) over a sample with known GROUP structure
    (context/prompt families, genres, persona panels, behavior classes, seeds sharing a template),
    verify §6 names the grouping axes AND registers at least one GROUP-level held-out fold
    (leave-one-family-out / leave-one-genre-out / leave-one-persona-out, or a corpus/genre
    transfer arm — fit on corpus A, evaluate on corpus B), with every headline labeled by its
    fold, per planner.md §6 "OOD generalization folds" and
    `.claude/rules/ood-generalization-folds.md`. REVISE when the concrete pattern holds: the DV is
    held-out-predictive, the sample has a nameable group axis, and the plan registers ONLY
    pointwise LOO/LOCO with no explicit iid argument. Conclusion-changing because pointwise LOO
    trains on same-family siblings of every test point — it measures within-family interpolation,
    not generalization, and can REORDER the headline: #810's LOCO sweep said max-pool was the best
    answer-side summary (0.826 vs mean 0.800) and the trained-ridge read-out reached ρ ≈ 0.909;
    the 7-fold leave-one-FAMILY-out re-read reordered the ranking (mean 0.804 ≥ turn_nl 0.791 >
    max-pool 0.760 at LOCO-best layers) and collapsed the read-out to ρ ≈ 0.285 — both headline
    claims were within-family fold-artifacts. Interactions: any max/argmax over a free axis INSIDE
    a group-fold headline inherits item 11's selection-symmetric null with the null computed under
    the SAME fold structure (`.claude/rules/ood-generalization-folds.md` point 4); and group-level
    n is the real n for item 4's uninterpretable-N read (G quasi-independent test units — 7 in
    #810 — not n points). Not a REVISE when: no DV is a held-out predictive statistic (the plan's
    §6 "N/A — no held-out predictive DV" satisfies this item); the plan argues genuine iid
    sampling — no nameable group axis — as an explicit positive claim; or a group-level fold /
    transfer arm is already registered with fold-labeled headlines. Pointwise LOO retained
    ALONGSIDE the group fold is fine (it upper-bounds within-distribution skill); pointwise-ONLY
    is the defect.

14. **Fail-loud acceptance claims backed by committed tests.** When a plan's acceptance/success
    criteria assert fail-loud / no-silent-swallow behavior ("fails loud", "raises, not swallowed",
    "no `except Exception` remains", "no warning-and-continue path"), verify EACH such claim is
    pinned by a named COMMITTED test (a behavioral `pytest.raises` case, or an AST/source-level pin
    for swallow-shape bans) — not only a run-book grep gate: a grep command verifies the invariant
    once at review time, and a differently-worded re-swallow ships green past every committed test
    (#913: the caller-level swallow ban was grep-only until the round-1 Codex critique added the
    `ast.Try` pin). Per-claim coverage is the point — a plan naming a fail-loud test for ONE claim
    while another claim rests on grep alone still gets the finding (the mechanical pre-pass, c15,
    catches only the zero-fail-loud-test case and PASSes that shape). REVISE when a fail-loud
    acceptance claim about committed code has no committed-test backing and no stated reason a test
    cannot exercise it; a doc-file target is the named escape (`N/A — fail-loud claim not
    test-backable` — a .md instruction has no code path a pytest can exercise).

### Alternative Explanations lens

1. For each predicted positive result, name the simplest alternative explanation that doesn't
   require the claimed mechanism.
2. If the design rules it out OR the analyzer can weigh it descriptively from reported diagnostics →
   list it as a "Concern for the analyzer" and APPROVE.
3. Only REVISE if the alternative is FATAL: the design cannot distinguish it AND the analyzer cannot
   weigh it.
4. **Inherited-positive DV-swap (cross-ref Statistics & Measurement lens item 2).** When a follow-up
   reuses a parent's positive predictor↔DV correlation as grounding but its own DV is a `trained −
   base` change of the quantity while the predictor is a base-side propensity, the "alternative
   explanation" for a falsification result is the DV choice itself: a base-side predictor enters a
   change DV with mechanical coefficient −1, so the predicted sign flips and the falsification gate
   fires for the DV choice, not because the predictor failed. This is a Statistics & Measurement
   lens item-2 REVISE (DV identity), not a fatal-alternative APPROVE — flag it there if you hold
   that lens; if you hold this one, note it as a Concern so the merged report carries it.

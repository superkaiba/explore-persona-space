# Issue 1739: natural generic-data scaling, 2026-09-06

## Goal

Determine whether applying the learned context->answer map before projecting the persona vector predicts on-policy behavior expression (evil, trait sycophancy, hallucination) better than context-side projection and direct regression at matched (unlabeled, labeled) data budgets, and whether that advantage grows across a real-data distribution-shift ladder.

## Authorization and scope

The user requested: "can you rerun this scaling generic data experiment with just generic context -> answer pairs (no recombination) as well as the trait-eliciting data, and up to 100 000". Their preceding explicit endpoint choice was 100,000 generic pairs PLUS the fixed trait pool. This is a same-question follow-up within task 1739. No new judge, model training, nonlinear map, or Claude automation is authorized or needed.

I will run the curve with natural prompts and fresh answers, adding the same complete trait-eliciting training set at every size. More natural data may improve the map, but an advantage over context regression is an empirical question. A curve that stays flat or reverses would narrow the claim; it would not by itself disprove every version of the hypothesis.

## Design and sources

The generic source is the existing, immutable LMSYS prompt manifest in dataset `superkaiba1/explore-persona-space-data` at `7a47ff5ce42f16308bebaba29c1286a4e9bc8008`, prefix `issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest`. Its 960,000 rows comprise 525,485 LMSYS first-entry texts and 434,515 WildChat texts. Use LMSYS only. The prompt-content SHA256, computed as consecutive UTF-8 prompt bytes followed by NUL, is `2b14762a15d316c602332a749ebd87c733d687d4165eb5d0038c298e0d27ce46`.

Each chosen row is one intact natural user prompt, rendered as a single user message with the model's normal chat template. Its answer is generated afresh for that same prompt. There is no history/query crossing, answer permutation, or old crossed generic row in this curve. Provenance retains source manifest index, stream position, raw-prompt hash, answer hash, and the immutable artifact revision. The upstream manifest does not preserve original conversation IDs or an upstream dataset revision; this is an explicit provenance limitation, not an inferred guarantee.

Deduplicate normalized prompt text internally; remove normalized exact matches and char-5-gram Jaccard >=0.8 near-duplicates against all reused evaluation/training prompt texts. Exact ordered-prefix pruning accelerates the existing #779 near-duplicate rule without approximating it. Export all exclusion sources with required-ID coverage assertions. Rank the entire LMSYS candidate population by seeded content hash before taking eligible rows. Prepare 110,000 eligible candidates as a reserve; generate 100,000 first, and use subsequent reserve chunks only if empty answers leave fewer than 100,000 admitted pairs. Never truncate input prompts.

Realism: natural production-chat prompts are tier 1; own-model answers are required to map this model's representations. Existing synthetic trait elicitation and PV evaluation remain inherited controls. This does not compare against the source dataset's original third-party answers.

Generation follows #1092: `Qwen/Qwen2.5-7B-Instruct`, model/tokenizer revision `a09a35458c702b33eeacc393d103063234e8bc28`, greedy temperature 0, seed 42, 1,024 new-token cap, `<|im_end|>` stop, bf16, maximum model length 8,192. Formatted prompts must have <=7,167 tokens, reserving the capture boundary. H100 real-corpus mitigations are enabled consistently: eager execution and no prefix caching. Report cap-hit fraction. The primary keeps the inherited 1,024 cap even if >2% hit it, explicitly prioritizing recipe fidelity; any extended-cap sensitivity must be separately labeled and cannot silently replace primary answers.

Capture reuses #1092's per-segment token IDs and offsets. Context summary is the last prompt token; answer summary is the mean over the model's answer tokens, excluding the boundary. Store fp16 summaries from bf16 forwards. Capture global zero-based layers 17,18,19,20 only. Vectorized reduction must agree with the original capture helper in a tiny real same-architecture test, plus a real bf16 GPU pilot.

## Scaling grid and reused scoring

Generic U: 250, 500, 1,000, 2,000, 5,000, 10,000, 18,793, 25,000, 50,000, 100,000. Seeds: 0,1,2,3,4. Each seed defines a nested permutation using RNG `[1739,20260906,seed]`, shared across behaviors. No old generic rows enter any rung.

Append ALL fixed trait-eliciting training pairs at each rung: evil 6,468; sycophancy 16,000; hallucination 16,000. Consequently map/whitening sample totals range from 6,718 to 106,468 for evil and from 16,250 to 116,000 for the other behaviors. These totals exceed d=3,584 even at the smallest U. Report realized counts from the loaded artifacts, not this plan alone.

Use the existing paper P-B grouped leave-one-dataset-out readout machinery, including held-in group splits and the fixed WildChat fold. This differs from the older P-A scaling run, which is retained separately and is not a matched comparator. No evaluation context labels train their own held-out readout. Reuse the original label/DV, split, map-fitting, ridge-selection and evaluation helpers.

Primary roster: `arm4_ridge_ctx` (direct context regression), `arm7_map_ridge_pred` (map then regression), and `arm12_oracle_reg` (answer-side regression upper reference). Only true aligned mapping is needed in this requested curve; retain the original map diagnostics, including held-out R2, identity-plus-learned-bias, nearest-neighbor retrieval, pool size, and chance. No nonlinear map/readout is introduced.

Frozen global layers, selected previously on the original full 28-layer training summary: evil direct/map/oracle = 18/20/17; sycophancy = 20/19/19; hallucination = 20/20/18. Explicitly translate global IDs to reduced-array indices; never clamp a missing layer. This is a frozen-readout data-scaling follow-up, not a new all-layer selection study.

Refit whitening separately for each (behavior,U,seed) using ONLY that rung's natural generic + full fixed-trait union. All three arms share that preprocessing. Thus the direct/oracle baselines can vary with U; plot/report that variation instead of describing them as constant. No full-100k whitening can leak unbudgeted data into a smaller rung.

## Evaluation and interpretation

The dependent variable remains the cached on-policy behavior-expression labels used by the paper's scorer; teacher-forced capture supplies predictor features, not replacement behavioral outcomes. Report each method's Spearman correlation per held-out dataset and seed, and mapped-minus-direct differences at each U. Preserve per-context predictions and actual group IDs for paired downstream uncertainty calculations. Seeds reuse contexts and are not five independent evaluation datasets.

The requested deliverable is the full natural-data scaling curve and endpoint comparison, with realized coverage, count/source provenance, cap fractions, map diagnostics, and seed spread. A descriptive endpoint-change contrast is not the same as a test that the endpoint advantage exceeds zero. Do not equate failure to reject with equivalence or claim a hypothesis is disproved from the old P-A contrast. No automatic task classification or paper replacement occurs in this run.

N/A — no registered verdict lattice

Row-coverage: every arm's prediction rows are supplied by the same reused per-context DV tables; the scorer writes transfer_preds/<fit>.jsonl with context_id/rung/group, and comparisons require matching context/group keys.

## Resources and pilot gates

Estimated GPU-hours (total): 80

This is a conservative provisional reservation, NOT a measured runtime prediction. Full dispatch is pilot-gated. Prior #1739 scaling consumed 19.62 H100-hours but used a different readout/grid; it is only a rough reference. The 80-hour reservation includes a >=2x uncertainty margin over a naive 40-hour planning envelope and must be replaced by measured phase projections before fan-out.

| Component | Planned wall h | Planned GPU h | Parallelism and basis |
|---|---:|---:|---|
| Text staging/filtering | 4 | 0 | One dedicated CPU pod, >=64 GiB RAM; <3 GB expected text/metadata. Avoid a potentially >16 GiB ngram index on shared VM. Measure RSS and exact near-dupe throughput. |
| Data pilot | 1 | 4 | Four H100 workers, one 500-row chunk each including a longest-prompt-tail chunk. Same generation/capture batch widths and code as production. Measure wall, peak HBM/RSS, cap rate, save/upload wall. Completed chunks are reused. |
| Remaining generation/capture | 6 | 24 | Four H100 workers sharded over remaining 500-row context chunks. Provisional envelope; replace with measured pilot per-row phase rates and tail dispersion before dispatch. |
| Fit pilots + full grid | 13 | 52 | 3 behaviors x 10 U x 5 seeds =150 cells, true map only, three arms, existing P-B dataset folds; four independent per-GPU workers. Run one actual 100k endpoint cell per behavior through diagnostics/readout/output before full fan-out. Pilot measurements are prerequisites, never fabricated from tiny matrices. |

Use one shared multi-GPU pod for the GPU phase rather than separate single-GPU pods. Probe live RAM before choosing concurrent fit width: four jobs require >=384 GiB available host RAM provisionally; narrow concurrency if measured peak dictates it. A selected-layer 100k store is ~5.34 GiB, plus another ~5.34 GiB in resumable chunks; keep 400 GB pod volume for reused label stores, source tars, and outputs. Labeling tars transfer 32–70 GB apiece but retain only four layers. Stream/materialize per existing validated helpers and measure real headroom, not just a shared filesystem's apparent free space.

Generation uses vLLM continuous batching with 500-prompt chunks and max 64 sequences; capture uses padded batches of 8 with length grouping. Whitening/map/ridge fits reuse vectorized dense kernels; no new serial per-row model forwards or per-coordinate fitting loops. At 100k, exact kNN diagnostics may dominate CPU time and must be included in the endpoint pilot timing. Four GPUs approximately halve a fully shardable phase's wall vs two, subject to observed CPU/IO contention. Do not shrink U or fixed trait counts for throughput; adjust hardware, batching, or scheduling instead.

Pilot acceptance: input/exclusion coverage complete; nonempty real-source yield; exact counts and natural provenance; capture/reference parity; no nonfinite tensors; process exit verified; all raw/tensor uploads verified; memory headroom confirmed; end-to-end scoring reaches held-out predictions and map diagnostics. Tiny pilots reduce row count, never replace implementation paths. Long-prompt pilot exercises the same 8,192 window. A method-integrity failure stops that stage loudly; a throughput surprise triggers profiling/vectorization/resource re-sizing rather than a scientific conclusion.

## Persistence and continuation

Working branch: `codex/1739-natural-scaling` in the dedicated `i1739-uladder` worktree; base `e9e3dd9395887a6f1aed6c1e92cec1a5ee2e2500`. Do not mutate unrelated root changes or start Claude automation.

Durable output prefix: `issue1739_natural100k_20260906` in the data repository. Persist selected source prompts, exclusion export/coverage manifest, generated raw text/token IDs, chunk completion records, row-index hashes, context/answer matrices, scoring configs/diagnostics/predictions, logs and sentinels. Raw text shards stay <=8.5 MB. Natural store `manifest.json` is written LAST with `status=complete`; consumers verify matrix and row-index hashes. Resume identity includes data, source recipe, implementation, U, seed, layer IDs, roster, and readout arguments.

Cross-phase reads: CPU preparation produces the complete `prepared/` shards/index/completion metadata and exclusions; upload+verify them before GPU staging. Generation raw chunks are uploaded+verified before capture or long fits. Capture chunks and assembled `store/` (all eight matrices, row_index.jsonl, manifest.json) are uploaded+verified before fitting. Off-pod analysis consumes only verified scoring outputs and row metadata. No deletion/termination before this round's required artifacts are verified remotely.

Detached launch records carry exact code SHA, identity-verified worker PID, log and fresh sentinel paths, and explicit exit status. Pod completion writes a sentinel only; the owner performs VM-side verify-then-terminate via `pod.py`. No process self-stops its pod. Task progress and plans are recorded exclusively through task.py/task_workflow.

## Decision rationale and consequential assumptions

Source/model/tokenizer, temperature, cap, stop, extraction positions and d are inherited from #1092 and #1739. Fixed trait counts, P-B splits, frozen layers, whitening gamma grid, map/readout ridge selection and mapping diagnostics are inherited from #1739. Jaccard .8 on char-5-grams is inherited from #779. The user supplies the no-crossing constraint and 100,000-generic endpoint. Added intermediate U values isolate growth beyond 18,793; five seeds reuse the established scaling replication count. Chunk500 and capture batch8 are validated parent execution settings, re-smoked on the new source. Hardware/runtime reservations are explicitly ungrounded pending the real-shape pilots.

This follow-up treats natural generic context as an intact single user prompt, not a reconstructed multi-turn history. The entire generic pool changes, not just the rows beyond 18,793. The fixed trait-eliciting pool is intentionally not relabeled or regenerated. Reused artifacts are pinned and verified in their actual consumer layout. Any missing prompt coverage, incompatible source, or unclear behavior identity is a material blocker to resolve before expensive dispatch.

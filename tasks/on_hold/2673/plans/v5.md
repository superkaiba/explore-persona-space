# Plan — task #2673: Kimi context vectors and two leakage comparisons

## 0.0 Plain-language plan

- I will extract Kimi's context vectors using the same questions and short persona descriptions as the successful DeepSeek run, adding Kimi's default assistant.
- I will compare raw and uncentered whitened cosine both with Kimi's published leakage and with the fixed DeepSeek leakage values.
- I expect this to tell us whether the earlier pattern survives the model change; weak, reversed, or question-sensitive correlations would argue against a general predictor.

## 0. Plan summary

**Training:** none. **Hyperparameters:** 240 inherited questions; eight inherited descriptions plus the default assistant; singleton forwards; all 61 decoder blocks; 7,168-dimensional last-context-token residuals; raw cosine at every block; uncentered regularized whitening at blocks 15, 30, 45, and 60, with full-bank and both question-half cross-fits. Sources: completed #2673 pipeline and pinned Kimi configuration. **Controls:** preserve the original eight-description whitening calibration bank, question order, pooling, published DeepSeek outcomes, and numerical validation; the added default condition is an evaluated centroid, not an extra calibration condition. **Loss surface:** N/A. **Compute:** one 8×H200 allocation with an initial 3.5-hour envelope, including download, loading, validation, capture, inherited CPU analysis on the same pod, and preservation. This is an operational ceiling, not a measured Kimi runtime prediction. **Evaluation:** three separate five-character strata, with direct cosine versus alternative-tracer uptake as the user's focal result and inherited helpful-relative contrasts reported alongside it. **Risks:** the new vLLM INT4 hook must pass a real full-width validation; published aggregate outcomes are approximate image digitizations and do not provide individual behavioral trials.

Estimated GPU-hours (total): 28

## 1. Goal

> Test whether context-vector cosine similarity predicts measured story-imprinting tracer uptake across matched persona conditions, using Qwen3.8-27B as an initial pilot.

(Task #2673 Goal, verbatim; re-read through the task-workflow API on 2026-09-22 before returning this plan.)

The authorized extension is the user's instruction “Do both. Run it now”: extract Kimi-K2.6 geometry, compare it with Kimi's own default-assistant leakage, and also hold the published DeepSeek HHH/Fred outcomes fixed for comparison with the existing Qwen and DeepSeek geometry. Do not rerun story training, generate a new behavioral evaluation, change the goal, or relabel this exploratory association as held-out predictive validation.

N/A — no model training
N/A — no behavior implantation
N/A — not a replication
N/A — no held-out predictive DV
N/A — no registered verdict lattice

## 2. Prior work and evidence

The successful singleton implementation at source commit `8f9f676965f57401cb2fb1099c29c87b2f0a397b` already supplies row provenance, checksummed tensor chunks, numerics gates, question-half analysis, regularization, and upload verification. Reuse its inputs and numerical definitions; its DeepSeek FP8 model loader is incompatible with Kimi's native INT4 checkpoint.

The paper's Kimi experiment compares Helpful against Dismissive, Sarcastic, Saboteur, Peer, and Help-seeker. Its default-assistant outcome is the aggregate multi-turn Bloom tracer rate after story fine-tuning, with incoherent outputs filtered; the plotted bars combine the tracer swaps and two seeds. These outcomes live in `images/selectivity/fxbc_2seed_bloom_kimi_intemplate.png`. Source: [paper, affinity section](https://arxiv.org/html/2609.10883v1), also checked in the cached original TeX.

[Moonshot's deployment guide](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/docs/deploy_guidance.md) explicitly supports stable vLLM 0.19.1 and gives an H200 TP8 example. The pinned configuration and weight index are cached at `/tmp/issue2673-kimi-source/`. They establish 61 blocks, width 7,168, native compressed-tensors INT4, and 595,177,988,208 weight bytes across 64 shards. Model pin: `moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb`.

Live-sibling overlap: the parent is preparing the Kimi adapter in `/home/thomasjiralerspong/.codex/worktrees/story-persona-kimi-20260922`; this plan delegates implementation there and proposes no competing module or worktree.

## 3. Hypothesis and estimands

The hypothesis is positive association between evaluation-persona/character cosine and that character's published tracer uptake. A positive correlation consistent across preselected depths and question cross-fits supports the hypothesis descriptively; reversals, weak correlations, and unstable ranks count against it. With only five character pairs per stratum, do not use significance thresholds or claim general predictive accuracy.

Register these separately: (1) Kimi default assistant geometry against Kimi outcomes; (2) Kimi HHH-description geometry against DeepSeek HHH outcomes; (3) Kimi Fred-description geometry against DeepSeek Fred outcomes. The latter two reuse the same outcomes as the existing model comparison and are explicitly cross-model proxy tests. Do not pool these three strata for a headline.

For each stratum, report direct cosine versus the alternative character's tracer rate. Also retain `cos(eval, Helpful) - cos(eval, alternative)` versus `rate(Helpful) - rate(alternative)` for parity with the prior analysis. Helpful similarity is constant across the five alternatives within a stratum; do not present those repeated Helpful coordinates as independent observations.

Row-coverage: Kimi default-assistant outcome stratum = 5/5 character pairs; fixed DeepSeek HHH stratum = 5/5; fixed DeepSeek Fred stratum = 5/5. Capture coverage = 9 conditions × 240 questions = 2,160 contexts. The two DeepSeek strata reuse their existing outcome rows, not new observations.

## 4. Design and implementation

1. Reuse the exact ordered input bank in `configs/pilots/story_persona_deepseek_prompts.json` and `data/assistant_axis/extraction_questions.jsonl`; assert committed content hashes and all IDs before use. Add one default-assistant condition with no system message: `messages=[user]`. The Kimi opposing-pair experiment uses no system prompt; do not import the separate SFL experiment's explicit empty-system convention. The inherited eight conditions retain their description as a system message. Render the pinned Kimi chat template, never adding Moonshot's example identity prompt. Set `thinking=False` to keep a direct pre-answer assistant boundary, record this choice as a rendering approximation to the paper's unavailable exact server serialization, and persist rendered prefixes and token IDs. No truncation is permitted.
2. Implement a Kimi-specific capture adapter and configuration, while reusing the proven chunk/provenance/artifact infrastructure. Pin vLLM 0.19.1, the model revision, wheel/runtime identities, TP8, eager execution, `max_num_seqs=1`, no prefix caching, no chunked prefill, and exactly one generation token to drive the full prefill. Only the last prompt token enters the context vector; the generated token never does. Verify actual scheduler token counts and reject dummy/padded rows. User-approved singleton execution supersedes the generic batching preference.
3. Instrument the actual decoder residual stream in each vLLM worker. vLLM separates hidden updates and residual carries; reconstruct the post-block residual from the correct hidden-plus-residual state, including any documented MoE scaling. Compare against the next block's pre-normalization input and the final normalization input. Check that TP rank reads represent the same full-width residual rather than shards or multiplied replicas. Preserve rank diagnostics and exact hook locations. Never assume a tensor named `hidden_states` equals the desired residual.
4. Capture one unpadded context at a time and save original BF16 vectors for every block. The persona vector at each block is the arithmetic mean of its 240 **last-token** residuals. There is no token-span mean and no centering. Storage chunks may contain eight independently captured rows; chunk size is not forward batch size.
5. Fit whiteners only on the original eight descriptions, excluding the new default rows, to preserve the calibration distribution for the cross-model comparison. Use `M = X.T @ X / N + lambda I`, with inherited label-free `select_ridge` conditioning rule. Apply the same metric to all nine centroids. Full bank uses 1,920 calibration rows; each question-half fit uses 960 rows and evaluates means from the opposite 120 questions. This checks question sensitivity, not held-out personas or behavioral generalization.
6. Produce raw cosine matrices for all 61 blocks and raw/whitened analyses for predeclared blocks 15, 30, 45, and 60 with all three fit modes. No best-layer selection. Generalize the prior analysis's hardcoded eight-condition assertions explicitly; retain strict schemas for the two existing models.
7. Digitize Kimi's plotted bars from the original raster, recording the image hash, axis calibration, bar coordinates, rate, and resolution-derived error interval. Visually check all ten rates against the source. Reuse the already reviewed DeepSeek outcome JSON without redigitizing it. Never turn an unreadable or absent bar into zero. If image resolution cannot order close bars robustly, report the tied/ambiguous rank sensitivity.
8. Reuse `scripts/story_persona_crossmodel_capture.py`, `scripts/story_persona_crossmodel_analysis.py`, and `scripts/story_persona_crossmodel_artifacts.py`, adding Kimi-specific branches with nine-condition validation rather than a second pipeline. After capture, drain periodic uploads and synchronously checkpoint the entire raw store to a new immutable HF revision before CPU analysis starts. Verify all 270 eight-row chunks and the final manifests remotely; the last 20 chunks after an every-25-chunk periodic checkpoint are mandatory and cannot remain local-only. Release the model process after this verified raw checkpoint, then run the inherited CPU analysis on the same pod within the allocation envelope and upload its verified products before teardown. Reuse `select_ridge(x)`, `metric_gram(x, centroids, ridge)`, and `association(x, y)`, whose definitions and guards were inspected in the existing scripts. Run these exact call shapes at smoke dimensions before production analysis. `paired_outcomes` and `validate_layout` need an explicit Kimi branch because their current bodies reject nine conditions and three strata.

### Divergences from prior #2673 run

- The scientific variable is the representation model, now Kimi-K2.6. Its required INT4/vLLM runtime and chat serialization replace DeepSeek FP8/Transformers; these are recorded model/runtime differences, not independently isolated causal factors.
- Default assistant is an additional evaluated condition needed for Kimi's own published outcome. It is excluded from the whitening calibration bank to keep the fixed-outcome comparison interpretable.
- Kimi's own behavioral outcome is a separate requested analysis arm. It never replaces the fixed DeepSeek outcome in the cross-model arm.

N/A — no re-extracted reference arms

## 5. Conditions and controls

| Plain-English name | What it tests | What it controls for | Config slug |
|---|---|---|---|
| Default Kimi assistant | Kimi's own published leakage | No injected identity/persona instruction | default_assistant |
| Helpful, harmless, honest description | Fixed DeepSeek helpful-persona outcomes | Same short preamble as previous extraction | hhh |
| Fred description | Fixed DeepSeek Fred outcomes | Same short preamble as previous extraction | fred |
| Helpful character | Helpful-relative contrast and reference | Identical inherited character text | helpful |
| Dismissive character | Alternative uptake | Same character text and questions | dismissive |
| Sarcastic character | Alternative uptake | Same character text and questions | sarcastic |
| Saboteur character | Alternative uptake | Same character text and questions | saboteur |
| Peer character | Alternative uptake | Same character text and questions | peer |
| Help-seeker character | Alternative uptake | Same character text and questions | help_seeker |

## 6. Evaluation and artifacts

The behavioral unit is one aggregate character-pair outcome, not one extraction question. The 240 questions reduce centroid sensitivity; they do not increase behavioral n beyond five per stratum. Report Pearson r, Spearman rho with average ranks for ties, digitization bounds, undefined constant-input correlations explicitly, and the complete point table. Report question-half differences without behavioral p-values or fabricated confidence intervals.

For Kimi, the fact-checked digitization is `/tmp/issue2673-kimi-rates.json`. Saboteur and Help-seeker alternative-tracer bars differ by one raster pixel, less than their ±1.5-pixel individual digitization bounds; the Dismissive and Peer intervals also touch. Enumerate every feasible weak ordering of the five closed outcome intervals, including tied groups and strict orders, and report the minimum and maximum resulting Spearman rho alongside the central digitization's rho. Use rational pixel-coordinate interval endpoints for feasibility checks so touching intervals permit equality exactly; average ranks within ties. This is a resolution sensitivity range, not a confidence interval. Do not claim a robust ordering between overlapping or touching intervals. Test the interval enumerator on exact ties, separated intervals, overlaps, and touching endpoints.

Data-source tier: established dataset for the inherited #2673 fixed probe bank and published paper artifacts for behavioral outcomes. This preserves the earlier experiment's generic-question approximation; the probes are not claimed to be real-world Bloom conversations and no new synthetic probes are generated.

| DV | Construct | Measurement validity on the behavior distribution |
|---|---|---|
| Kimi alternative-tracer rate | Default-assistant tracer uptake after story fine-tuning | Published on-policy multi-turn Bloom aggregate; raster approximation |
| DeepSeek alternative-tracer rate, HHH/Fred separately | Persona-elicited tracer uptake after story fine-tuning | Same fixed published on-policy outcome rows as previous comparison; Kimi geometry is a cross-model proxy |

Measurement validity: the outcome measures on-policy post-story-training behavior in the paper's Bloom distribution. The predictor measures pre-story-training geometry of short descriptions on generic questions. This mismatch is inherited and is the stated object of this exploratory test. Kimi INT4 is not necessarily identical to the authors' served training precision. HHH/Fred descriptions omit the original full few-shot conversations. These limitations accompany every headline.

No learned representation-to-representation map is fitted, so held-out R², identity-plus-bias, and retrieval baselines are inapplicable. No new judge calls or completion-probability estimates are authorized or needed; retain the published judged rate as the available behavioral DV and state that raw per-trial labels are unavailable.

Figures: labelled scatter plots for each of the three strata at the four preselected depths, separately for raw and whitened cosine; a raw all-layer correlation trace; a fixed-DeepSeek-outcome panel comparing Qwen, DeepSeek, and Kimi at matched relative depths. Publish browser-accessible figure URLs. Dump all layer matrices and point tables so the plotted choices remain auditable.

```yaml
primary_deliverable:
  - name: complete Kimi context tensor store and provenance
    path: /workspace/analysis_tensors_issue2673_kimi/chunks/*.pt
  - name: validated row and completion manifests
    path: /workspace/analysis_tensors_issue2673_kimi/*.json
  - name: leakage associations and point tables
    path: /workspace/analysis_tensors_issue2673_kimi/analysis/*
```

## 7. Numerical, runtime, and completeness gates

These gates protect measurement and allocation validity, not favorable scientific outcomes. On the real full-width TP8 runtime, smoke the first two registered questions for all nine conditions plus the global shortest and longest tokenized contexts, deduplicating exact row IDs (18–20 rows). Require every expected block, finite BF16 values, width 7,168, correct last-prompt positions, consistent full-width TP reads, and independent residual reconstruction. Require relative hook-reference error ≤1e-5 and repeated-singleton relative error ≤1e-5, inherited from the previous capture's stricter hook check and 1% production reproducibility ceiling. Preserve observed errors, not only PASS. A failing gate blocks production; do not relax it to obtain vectors.

Smoke blind spots: this validates the capture path, not equivalence of INT4 to the paper's serving precision; not full few-shot prompt fidelity; not semantic persona enactment; not behavioral prediction; and not long-run disk/network stability. Tiny mocked model tests cannot substitute for the full TP8 smoke.

Measure at least three warmed eight-row storage chunks on the actual production path, keeping forwards singleton. Record chunk durations and variation, per-rank peak HBM, process/cgroup RAM, weight-download/load times, and observed upload throughput. Proceed only if the measured remaining capture projection with a conservative timing margin plus a 15-minute preservation reserve fits the original allocation deadline. Recompute projection as progress arrives. Never present the 3.5-hour envelope as measured throughput evidence.

Complete requires all 2,160 unique rows, all 61 blocks, expected dtype/shape, matching checksums, fresh success sentinel, matching source/model/input identities, full published-outcome coverage, all predeclared analysis fits, and independent FP64 whitening-oracle validation. Failure or missing cells remain explicit and cannot produce a completed-analysis sentinel.

Success criteria: complete validated capture and both requested comparisons with every planned cell represented or explicitly marked failed; no minimum scientific correlation is required. Abort criteria: failed numerical validation, inconsistent provenance, nonfinite/missing tensors, exhausted allocation projection, or unresolved resource/runtime error; halt-and-report after preserving available diagnostics, with bounded diagnosed recovery as specified below.

## 8. Monitoring, recovery, and teardown

Before launch register new run-specific monitor/watchdog IDs and a durable recovery runbook. Verify scheduled watchdog execution, actual recovery-worker execution in a controlled harmless test, and acknowledged alert delivery through the user's existing notification route. Reuse the prior monitoring architecture only after checking process, path, and account identities; never reuse its completed run's success markers or permissions.

If 8×H200 capacity is unavailable, poll the personal RunPod account every 60 seconds with a durable supervised capacity worker and independent watchdog. There is no midnight delay. Provision at most one matching allocation after code, numerical-smoke configuration, review, monitoring, and account checks are ready. Keep unchanged no-capacity checks quiet; notify on capacity acquisition, actionable failure, or worker/watchdog loss. Persist query timestamps and the single-allocation ownership/lock so retries and host restarts cannot create duplicate pods. Existing delivered-alert route: `/home/thomasjiralerspong/bin/intent-notify.sh`, after verifying acknowledgment in this run. Capacity waiting consumes no GPU envelope; the deadline starts when allocation begins. A null availability quote is lack of current capacity, not a fatal authentication error.

Check startup promptly, then inspect fresh process/backend state, logs, and chunk/progress timestamps periodically. Keep unchanged checks quiet. Alert on crashes, stale progress, monitor loss, nearing deadline, or completion. Bound recovery to one same-allocation restart after a diagnosed fix with preserved valid chunks and fresh progress proof; do not endlessly retry a failed numerical gate. A replacement paid allocation requires a concrete diagnosed recovery proposal within the user's authorized scope and fresh budget accounting, rather than silently consuming repeated 28-GPU-hour envelopes.

Upload checksummed vectors, configs, rendered inputs, smoke diagnostics, runtime identities, logs, analysis products, and completion status under a new immutable Kimi prefix in `superkaiba1/explore-persona-space-data`. Verify scoped remote listings, expected counts and sizes, checksum evidence, and read-back of the entry manifest at the returned revision. Terminate only after all required available artifacts, including failure diagnostics when applicable, are verified durable. Bound the inherited CPU analysis by its measured first-fit projection within the same allocation; if it cannot fit, upload its inputs and completed fit checkpoints and surface that limitation instead of extending GPU rental silently. Record termination confirmation.

Raw-capture durability gate: a completed local capture sentinel alone does not start CPU analysis. Require a fresh remote checkpoint result covering all 270 raw chunks, their checksums, complete row manifest, and capture completion metadata at an immutable HF revision. A failed or incomplete final raw upload is an actionable monitor alert and a halt before analysis, with bounded upload recovery; it never becomes a successful run marker. Analysis failure after this gate must preserve the raw checkpoint reference, making later analysis recovery possible without repeating GPU extraction.

## 9. Compute sizing and resource accounting

| Phase | Device use | Sizing basis | Bound / action |
|---|---|---|---|
| Download and loading | Single 8×H200 node, TP8 | Official deployment topology; 595,177,988,208 pinned weight bytes | Same 3.5-hour allocation fence; measure actual network and load duration |
| Numerical smoke and throughput pilot | All eight GPUs participate in every singleton forward | Exact production runtime and complete model | 18–20 validation rows plus three warmed eight-row chunks; stop on invalid numerics or projection |
| Production capture | All eight GPUs used by TP8; one context per forward | Measured pilot, not DeepSeek speed extrapolation | 2,160 contexts, excluding/reusing only validated identical completed rows |
| Upload and verification | CPU/network while pod remains allocated | Measured bytes and upload throughput | At least 15 minutes reserved; verification mandatory before teardown |
| Whitening, correlations, figure data | Same-pod CPU, capped BLAS threads, model workers exited | Reused bounded 1,920×7,168 dual formulation; 12 whitening fits | Rental time included in allocation envelope; checkpoint each completed fit and measure first-fit projection |

Initial allocation envelope: 3.5 wall hours × 8 GPUs = 28 GPU-hours, including every GPU-resident phase and preservation reserve. Runtime is unmeasured until the Kimi pilot; if the measured projection cannot fit, preserve diagnostics and report the concrete revised need. No dollar cap. API workload: 0 paid generation/judge calls; only model/artifact storage APIs.

cuda-context: each of eight vLLM tensor-parallel worker processes allocates one CUDA context on its assigned H200; the monitor, capacity poller, CPU analysis, and upload verifier allocate none.

N/A — no off-pod phase

Require 8 H200 GPUs with at least 130 GB available physical HBM each, at least 1 TB usable host RAM, and a 1 TB volume with at least 800 GB verified writable quota headroom. The native BF16 output store alone is 1,888,911,360 bytes; reserve additional space for manifests, diagnostics, selected vectors, and uploads. Do not dequantize the full 1T model to BF16. Read and assert actual quota headroom rather than filesystem capacity alone. Use the user's personal RunPod account for this explicitly requested topology and record account identity without credentials; do not use the revoked fellows account.

Bind every large cache and temporary path to the mounted volume before environment installation or model download: `HF_HOME=/workspace/.cache/huggingface`, `HF_HUB_CACHE=/workspace/.cache/huggingface/hub`, `UV_CACHE_DIR=/workspace/.cache/uv`, and `TMPDIR=/workspace/.cache/tmp`. Create and verify those directories, resolve their actual mount/device, and assert that they share the intended `/workspace` volume and its writable quota headroom; an environment string alone is insufficient. Do not let the approximately 595 GB checkpoint or decompression/install temporaries fall back to the small container-root filesystem. Record the resolved paths, mount identities, free/quota bytes, and a successful bounded write probe in resource preflight; fail before large downloads on disagreement. These temporary/cache directories do not repurpose the user's HOME or CODEX_HOME.

N/A — no fit-family phases
N/A — no empirical-null gate

## 10. Reproducibility and reuse card

- Model and runtime: pinned revision above, native INT4, vLLM 0.19.1 with its verified dependency requirements torch 2.10.0, torchaudio 2.10.0, torchvision 0.25.0, and compressed-tensors 0.15.0.1; save exact installed dependency versions, TP8, eager singleton prefill, direct-answer chat boundary, hardware, and driver details. Resolve the wheel/dependency set before allocation; do not reuse DeepSeek's incompatible environment wholesale.
- Inputs: inherit the committed eight-description JSON and 240-question JSONL byte-for-byte; add default only in a new Kimi input artifact. Save hashes before and after capture.
- Source: dedicated Kimi worktree based on the verified singleton source SHA; launch only from a reviewed committed SHA. Never mutate the shared root or the original results.
- Data-only reuse: DeepSeek's reviewed `eval_results/issue_2673/deepseek_comparison/published_rates_and_overlap.json` resolves in the base worktree and its strict `load_rates` validation is preserved. No staging transformation for committed JSON. Any prior model comparison loaded from HF requires its pinned manifest/entry file to be staged and opened through the real consumer before use; do not assume an output file's existence proves completed analysis.
- Reused helper call shape: `select_ridge(x)` with FP64 x shaped 1,920×7,168 or 960×7,168; `metric_gram(x, centroids, ridge)` with nine 7,168-dimensional centroid rows; `association(x, y)` with five finite scalar pairs. Run the same calls at smoke shape and check the guards before large fitting. Matrix operations are vectorized; no per-vector dense inverse. Scoped storage API calls only.
- Existing reference arms remain Qwen and DeepSeek with their original means, layer mapping, and outcome rows. Check their actual manifests, completion status, lineage, and input hashes before plotting them beside Kimi; missing reference artifacts delay only that comparison, not Kimi extraction.

Reuse fitness (a)–(n): the unchanged question/description inputs and committed DeepSeek outcome schema were inspected. No trained adapter, changed training recipe, or application-scaling assumption is reused. The new capture is matched to the new model and exact-runtime numerical validity remains a mandatory smoke gate. Hashes and immutable revisions enforce content/lineage identity, named file paths enforce layout, and scoped listing/read-back enforces storage verification. The numerical helper's N<dimension guard is satisfied (1,920<7,168); nine centroid rows are supported by its matrix equations and must be exercised by the call-shape smoke. Source fixes, if needed, belong in the helper rather than an unchecked caller workaround. Prior tensor-store entry/read compatibility and complete reference-result status are verification prerequisites to plotting reused model results, not yet claimed complete by this plan.

## 11. Decision rationale

240 questions; eight inherited descriptions; original row ordering; last-context-token pooling then question mean; storage chunks of eight; raw all-layer output; selected depths 15/30/45/60; question halves 120/120; uncentered moment and inherited conditioning-based ridge selection. Source: validated #2673 singleton capture and analysis.

Added default condition and two behavioral comparisons. Source: user “Do both. Run it now” and paper's default-assistant arm. Excluding default from calibration preserves fixed-outcome comparability; this is a declared analysis choice, not a learned hyperparameter.

Model pin, 61 blocks, width 7,168, INT4, TP8 and vLLM 0.19.1. Source: pinned HF config/index and official deployment guide.

One context per forward and a 1e-5 relative repeatability ceiling (the unchanged successful singleton gate). Source: explicit prior user singleton request and #2673's documented batch-dependent residual error. Hook-reference tolerance 1e-5: source #2673 capture implementation.

3.5-hour / 28-GPU-hour initial ceiling, 15-minute preservation reserve, one diagnosed same-allocation restart, and 1 TB disk/RAM envelopes are operational safeguards inherited/adapted from #2673, not measured Kimi performance. Source: parent brief; throughput and resource sufficiency need the exact-runtime smoke. `thinking=False` is a declared rendering choice requiring token-level verification; exact paper server serialization is unavailable.

## 12. Assumptions, uncertainty, and stopping conditions

High confidence: official checkpoint identity and architecture; native INT4 deployment support; existing code's last-token pooling, eight-persona assertions, ridge guard, and question-half split; paper's five Kimi character pairs and published multi-turn aggregate source. Verified in pinned files or official sources.

Medium confidence: H200 TP8 can expose valid full residuals with eager singleton prefill; the Kimi tokenizer's direct-answer template renders the intended pre-answer boundary; the displayed Kimi bars can be digitized to a useful resolution. Confirm respectively with full-runtime smoke, persisted token inspection, and independent raster calibration/read-back. Fail loudly rather than substitute a guessed hook, prompt, or outcome.

Low confidence before pilot: total wall time, loading peak RAM, and exact capture/upload speed. The initial envelope is a ceiling only. If the projection fails, stop production safely, preserve diagnostics, and surface the measured blocker; do not quietly change hardware, model, precision, persona bank, or target behavior.

The question-bank cross-fits address prompt sensitivity. They do not resolve the five-condition sample size, unknown training-checkpoint precision, short-description versus full-persona mismatch, post-training versus pre-training representation mismatch, or the absence of raw published trial labels. Those limits remain in the final interpretation even if correlations are large.

## 13. Review fixes and mechanical warning dispositions

Methodology and statistics reviews approve the scoped design. Compute review required two changes now made explicit above: bind all large caches/temporaries to the verified `/workspace` volume, and fully checkpoint/verify the final raw capture before CPU analysis can begin. Implementation review must check those execution barriers, including the final 20 chunks after the last periodic checkpoint. Weak-order digitization sensitivity now includes exact ties and touching rational intervals.

- `c2_measurement_validity`: the parser does not recognize the table in §6. The actual per-DV table and following paragraph name the two behavioral sources, the on-policy distribution, raster approximation, and pre-/post-training mismatch. This is covered by human methodology review, not excused by calling the warning harmless; changing an outcome or its aggregation requires updating that table and review.
- `c6_reuse_fitness`: the condensed prose does not mechanically attest every (a)–(n) item. Recipe/application-scaling items are inapplicable because there is no reused trained adapter. Reused input/outcome content, committed paths, row coverage, explicit changed model, source lineage, and helper-domain guards are identified in §§4 and 10. Exact-runtime device validity, complete reference-store consumer loading, pinned-revision remote durability, and current gate execution are **not yet verified** and remain enforced implementation/smoke/pre-analysis prerequisites. Do not convert this plan's stated checks into a claim that those checks already passed.
- `c47_wall_cell_parseable`: per-phase Kimi runtime is unmeasured, so inventing a numeric phase ETA would misrepresent evidence. The standard plan-derived per-phase ETA tripwire will not arm. The replacement mechanism is a launch-time absolute 3.5-hour allocation deadline, a live remaining-work projection from three warmed production-path chunks, a measured first-fit CPU analysis projection, a 15-minute preservation reserve, and independent watchdog alerts/recovery. Implementation review must verify the absolute deadline and projection gates are wired and tested; monitor registration must not rely on the absent standard per-phase ETA table. Persist measured phase timings from the smoke/run for future sizing.

## Concrete timing and local storage controls

The frozen capture projection is max(duration of three warmed eight-row chunks) × remaining chunks × 1.25 + 900 seconds for preservation. It must fit the remaining absolute allocation window before production and each later chunk. The full Kimi tokenizer dry-run measured 211,005 input tokens across 2,160 contexts (minimum14, maximum185); this is a work count, not a hardware runtime estimate. The first complete CPU whitening fit is bounded by the outer remaining-allocation-minus-900-second timeout. Every completed fit records elapsed time and gates the remaining fits using the slowest measured fit × remaining fit count × 1.25 +900 seconds.

The local VM has approximately5GB free at preparation time. Only code, small source snapshots, JSON metrics, and figure outputs are staged locally; no model weights or raw activation store is downloaded to the VM. Monitor local disk headroom, fail loudly before any operation that exceeds it, and keep the raw store on the pod and verified immutable HF storage. The runtime environment is UV_PROJECT_ENVIRONMENT=/workspace/.venv. All cache/runtime/temp mounts are verified and recorded against /workspace before model loading; the inherited writable-quota preflight remains mandatory.

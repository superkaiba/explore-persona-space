# Assistant in the attributed-quotation story: K5 extension design

Reviewed execution plan for the existing task, 2026-09-15. This document does not
change task status or the canonical goal. The coordinating agent verified both
assistant-story cells' 32 K3 capture chunks, receipts and audits at the pinned
revision. A production-device pilot still precedes expansion.

Estimated GPU-hours (total): 12

This is a conservative booking envelope; the production pilot replaces the
provisional estimate before expansion. Two independent plan reviewers found
only an ambiguity in the sampling-corrected distance formula; the explicit
raw-squared-distance definition below resolves it. The user's request to run
this extension supplies execution authorization within this unchanged scope.

## Scope and goal

> Build the framing x speaker x completion-condition context-to-answer map lattice at 6,000 rows per cell (5,000 train + 1,000 held-out) on a decoupled scaffold-and-splice corpus, so every cell is well-posed in the ambient basis (n_train 4,800 > d 3,584) and row-paired across framings; report per-cell within-cell ceilings and the 9-rung transfer ladder, both mapping arms, with identity+bias and kNN-retrieval reads.

(Task #2054 Goal, verbatim.)

The requested extension adds the missing assistant-in-story setting to the
**new K5** comparison. It asks whether its context-to-answer map transfers to
the four story characters and to chat/plain assistant, and whether its answers
resemble chat assistant answers on matched questions. This is not a new
persona-system-prompt experiment. No language-model training or automatic LLM
judging is needed. N/A — not a replication of an external paper.

## Prior work and exact intervention

The parent K3 manifest contains these two on-policy cells, each with 8,000 rows:

```
conversation_paired_stories_assistant__on_policy__attrib_quoted__qwen2.5-7b
conversation_paired_stories_assistant__on_policy__attrib_quoted__qwen2.5-7b-instruct
```

The existing K5 selector deliberately includes only twelve cells: chat/plain
assistant and HELIOS/Wren/Dana/Vex, separately for the two checkpoints. This
extension completes the same K5 recipe for the two assistant-story cells.
The inventory check verifies K3 captures; recheck hashes when staging on the
consuming backend, reuse draws 0/1/2 and generate only
draws 3/4: **2 models × 8,000 contexts × 2 draws = 32,000 new completions**.
Do not relabel the old K1 or K3 results as K5.

The story prompt is the exact persisted prefix before `answer_start`, generally
`{existing narrative and question}Assistant replied: "`. Generate until the
closing ASCII double quote. Preserve the parent row's actual character name,
punctuation, query, preceding prose and suffix; do not reconstruct a new story
from an assumed template. Validate the actual names and suffixes across all
16,000 source rows before the pilot.

The existing chat prompt is exactly:

```
<|im_start|>user\n{Q}<|im_end|>\n<|im_start|>assistant\n
```

Here `\n` denotes an actual newline. No system message is inserted. The plain
assistant prompt is `User: {Q}\n\nAssistant: `, including the inherited space.
This comparison changes the framing distribution, including the answer
boundary and narrative prose; it does not isolate a persona-name intervention.

## Hypotheses and interpretation

Estimate effects rather than choosing an arbitrary success cutoff.

* If assistant-story transfers well to characters while chat/plain assistant
  transfers poorly to assistant-story, the result supports a shared map within
  the story framing and a barrier at the assistant/chat boundary.
* If assistant-story transfers well into chat as well, that is additional
  evidence for a map spanning the two framing regimes. A directionally
  asymmetric result remains asymmetric; do not average it away.
* Assistant-story and chat answers may be similar even when their maps fail to
  transfer. That would separate response similarity from representational
  alignment. Similar activation vectors alone cannot establish semantic
  equivalence.

N/A — no registered verdict lattice

## Reuse and new code

Reuse the committed worktree implementation, not a possibly older main copy.
The bounded change is an explicit two-cell selection in the K5 generation,
capture and aggregation path. Avoid monkey-patching selection functions or
changing the original twelve-cell fingerprints/output destinations. Add a
separate output prefix and an explicit `cells` keyword at the source helper.
Keep the entire 56-cell K3 manifest unchanged: parent fingerprints bind it.
Preserve the adjudicated K3 capture module byte-for-byte, because its
hash is checked by the capture policy.

Verified code entry points and contracts:

* `scripts/issue2054_k5.py`: `prepare`, `generate`, `capture`,
  `average_targets`, `aggregate`; existing selection and population assertions
  are twelve-cell-specific and need an explicitly named extension path.
* `scripts/issue2054_k3.py`: `banked_rows`, `seed`, `cap_for`, `forward_vectors`,
  `complete`; raw draw seeds and cap are audited, and the true context is the
  prefill-alone last token. The answer vector is the layer-19 token mean.
* `scripts/issue2054_k3_recover.py`: `load_policy`, `capture_fingerprint`,
  `parity_audit`; retain the frozen model/input/code identity checks, exact
  same-forward hook check and adjacent-layer controls. The historical relative
  L2 threshold 0.025 is a recorded WARN policy, not a hard parity claim.
* `scripts/issue2054_k5_map_geometry.py`: `moments_by_fold`, `restore_maps`,
  `retrieval`; restore existing six-setting own maps at their published ridge
  penalties, using raw-coordinate A and intercept. Batch map reconstruction;
  do not redo GCV for existing maps.
* `scripts/issue2054_k5_loso_calibration.py`: `calibrate`,
  `adapted_predictions`; separately produce frozen, vector-bias, and one
  scalar-plus-vector-bias predictions.
* `scripts/issue2054_k5_matched_responses.py` and
  `scripts/issue2054_k5_matched_strict.py`: reuse full-query auditing, but do
  not interpret the existing audit as history equality.

Before production, execute actual reused calls at small real-data shapes on
their production device, not signature checks alone. Validate the new own-map
GCV through the inherited float64 standardized-X, centered-Y estimator with
`lambdas=np.logspace(-2, 4, 13)`, `dof_cap=0.9`, explicit `device`. Restore each
old map at the exact banked lambda and reproduce its original own-cell R²
within 1e-6 and retrieval with the inherited tolerance-based tie treatment.

## Transfer analysis

Use the unchanged five conversation-grouped folds, seed 137. Every source fit
excludes the target test conversation IDs even when that conversation appears
in another setting. For the new assistant-story map, select lambda using only
its source-training data. No target labels select its map or hyperparameters.

Per checkpoint and fold:

1. Fit assistant-story's own K5 map and own ceiling.
2. Apply it to each of chat, plain, HELIOS, Wren, Dana and Vex.
3. Apply each of the six existing individual maps to assistant-story.
4. For every transfer pair report frozen performance first; then target-trained
   output bias and target-trained bias+scalar as explicit adaptation reads.

This produces 120 directed transfer fold units (2 × 5 × 12), plus ten new
own-map fold units. A frozen existing six-setting pooled map evaluated on
assistant-story is an optional ten-unit diagnostic, not a required refit.

For every unit report held-out R², source-trained identity-plus-bias, and
Euclidean/cosine top-1/5/10 retrieval, naming the realized held-out target pool
size and chance k/n. Include the target's own ceiling on exactly the same test
rows. The calibration fit uses only target-training folds; it is not zero-shot
generalization. Show each character target separately before an equal-weight
four-character mean. Include train counts, ridge lambda and effective degrees
of freedom to make sample-size/capacity differences visible.

Primary transfer uses each source's complete-five cohort and the full target
test cohort, matching the K5 protocol. It measures framing-distribution
transfer. Also score on the exact shared-query test intersection as a
sensitivity analysis, without refitting on a small intersection that might
leave the ambient n_train > 3,584 regime.

## Direct comparison of answers

Row-coverage assert: the driver set-checks paired conversation IDs and draw IDs
against both the new assistant-story captures and the pinned assistant-chat
captures before computing each paired statistic.

Audit the complete corpus into: shared conversation ID; literal full query
match; whitespace-only query match; mismatch. Report all counts and exclusion
reasons. **A shared conversation ID does not prove the same question. A literal
question match does not prove equal history.** Existing rows do not include a
structured full-history field; inspect the actual prefixes for extra task
facts, prior answers, demonstrations and references. Report broad metrics on
the query-matched set, with this remaining context difference explicit.

The coordinating agent has now checked the full hash-verified Instruct raw
sources: 8,000 shared IDs, 8,000/8,000 exact full chat queries present in the
assistant-story prefix, and every chat prefix is a single user turn with no
preceding history. Therefore this assistant-story comparison avoids the
question-rewrite losses seen for some character pairs. Recheck Base and report
complete-five exclusions separately. Narrative-added task information remains
possible even though the chat side has no prior history.

On matched queries, retain all five answer vectors and texts per framing:

* Report raw and centered cosine between K5 mean answer vectors, and normalized
  squared distance. Centering constants come only from paired training folds.
* Report the mean of all 25 cross-framing draw-pair cosines and distances;
  compare with the ten distinct within-story draw pairs and ten within-chat
  draw pairs. Bootstrap conversations, not the 25 correlated pairs.
* For the sampling correction use raw squared Euclidean distances, not
  pair-specific normalized distances or the square of an average distance:
  `D_cross = mean_{i,j} ||story_i - chat_j||²` over all 25 pairs;
  `D_story = mean_{i<j} ||story_i - story_j||²` over ten pairs, and similarly
  `D_chat`. The estimated squared mean-response displacement is
  `D_cross - (D_story + D_chat)/2`. Any normalization uses one common
  training-fixed scale after this subtraction. This subtracts within-framing sampling variation
  under independent identically distributed draws; retain signed estimates
  rather than clipping negative sampling estimates to zero. Report possible
  draw-0 versus fresh-draw recipe/runtime differences as a limitation.
* Report answer-length/cap-hit differences and lexical token overlap as lexical
  diagnostics, without naming them semantic similarity. Activation similarity
  includes the effect of the surrounding context on answer states.

For qualitative evidence, deterministically sample 20 query-matched pairs per
checkpoint before looking at answer differences; inspect all five answers on
each side and the full prefixes. Mark pairs with added task information as
context-confounded. Separately show a few largest-gap cases, labeled selected
examples, including refusal/compliance or capability-claim reversals if present.
Do not report an automated population refusal rate or claim absence of such
reversals from this bounded inspection. No Claude/Anthropic API is invoked.

## Pilot, resource sizing, and execution gates

Pilot generation/capture on the first 256-context production chunk in each
assistant-story checkpoint,
using the actual vLLM/HF paths, exact production stops and full generation cap.
Use distinct output and receipt namespaces; keep pilot outputs reusable when
their fingerprints and request identities equal production. Record per-chunk
generation tokens/s, cap/empty counts, capture time/peak memory, and
serialization plus verified-upload time. The pilot measures implementation and
throughput; it does not establish a generalization result.

After complete K5 banks are available, time one **full-size** new own-map fold
per checkpoint and the reverse-map batch. Do not estimate dense-fit cost from
a 256-row smoke. Project remaining time from the slowest measured applicable
chunk/fold, showing actual remaining counts. Compute hours remain pilot-gated.
The verified parent K5 production pilot had attributed-character
256-context/512-answer generation chunks of 16.42–46.04 seconds. At 32 chunks
per assistant-story checkpoint, this gives a conditional 0.29–0.82 generation
GPU-hours across both cells; assistant-story answer lengths and capture/upload
cost must be measured separately. The broader twelve-cell pilot projected
7.95366 generation GPU-hours and its full 18.10-hour pipeline included extensive
other fits, so neither is an estimate for this extension. Sources: pinned K5
`production_v1/pilot_complete.json` and `fit_pilot_complete.json` at revision
`9de026f872c19b2ca4fd3e4539de820e08038ee3`, verified by the coordinating agent.

Initial route: GCP first, two A100-80 workers, one checkpoint per worker,
falling back to one worker if necessary; 200 GB boot disk per model worker.
An 8–12 GPU-hour booking envelope is conservative recovery headroom, not a
measured expected duration. Replace the projection with new pilot measurements
before production expansion and report CPU analysis separately.

Generation and capture are separate processes so vLLM and HF do not coexist.
Use one model per GPU worker; chunk-shard to use allocated GPUs. Activation
capture requires the repository capture-7b lane (at least 40 GB GPU memory).
Measure host RSS and output filesystem free space before selecting concurrency;
route aggregate CPU demand at or above 16 GB off the shared VM according to the
repository resource policy. Existing inputs and expected two-cell captures are
small compared with a model cache, but the shared data disk has been near full;
never infer available headroom from an earlier snapshot.

Smoke blind-spot enumeration: the small generation pilot does not certify
full-cohort pairing, rare cap failures or full-size dense fitting. Full-corpus
metadata/query validation and a full-size fit pilot cover those separate
contracts. No mock model, downgraded capture identity check or production-only
third-party implementation is introduced.

Before continuing beyond each chunk, persist raw text, capture vectors and
receipts. Verify uploads before long downstream fitting and before releasing
compute. Atomic phase sentinels bind the current source and input manifest.
Monitor process state, logs and newly completed chunks with timestamps, verify
the independent watchdog and acknowledged established-route alert delivery,
and keep a durable bounded recovery record. These are required launch duties,
not satisfied by this planning document.

## Recipe and provenance

All load-bearing recipe values are inherited from task #2054 K3/K5, rather than
new hyperparameter choices:

| Item | Exact inherited value/source |
|---|---|
| Base checkpoint | `Qwen/Qwen2.5-7B`, revision `d149729398750b98c0af14eb82c78cfe92750796` |
| Instruct checkpoint | `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28` |
| Generation | temperature 1.0, top_p 1.0, cap 2,048, stop `"`, five independent draw streams |
| Fresh draw seed | inherited `issue2054_k5.seed(cell, conv_id, draw)`, draw 3 or 4 |
| vLLM | bf16, model length 8,192, eager, 128 sequences, KV fraction 0.85, TP1 |
| Capture | layer 19, d=3,584, prefill-alone final context token, mean answer token state; bf16 model, fp16 storage, float32 K5 mean |
| Capture batch | up to 8 sequences and 8,192 padded tokens; inherited adjudicated code |
| Ridge | float64; population-standardized input with 1e-9 SD floor; centered output; GCV grid 1e-2…1e4 in 13 logspace points; df cap 0.9 |
| Split | existing conversation fold map, five folds, seed 137 |
| Intervals | 200 conversation bootstrap draws, inherited #2054 diagnostic convention; separate checkpoints |

K3 manifest source:
`superkaiba1/explore-persona-space-data` at
`5ae90722bf11330deddfa42cf41f9fec6da8b69f`,
`issue2054_section44_k3_gcp/capture_recovery_v1/manifest.json`, SHA256
`365eb7a5f71735ab47e1e0802c3a790cce90b43a84b13a5c43a9d7b460adb506`.
Draw-zero source revision is `d3207a181402b42873f5a3120b1d56da7b90f104`.
The two assistant-story ordered-ID SHA256 values both equal
`fa179da44468d6835d5a55e2d187deb774874e960a7e07e06d51541fe72a0b7e`.
Verify source/capture pair provenance, all chunk schemas, all byte hashes,
complete coverage, exact prefixes, seed identities and cap identities before
reuse. The parent agent's live HF inventory check passed for both cells' 32
capture chunks and receipts, including real first-chunk consumer opens at the
exact recovery fingerprint. Consumer-side hash, schema and lineage checks
remain mandatory on the execution backend.

The scientific data are real-chat-derived questions inside inherited diverse
synthetic narratives. Reuse is required for comparison to the existing K5
result; synthetic prose and incomplete conversational context limit causal
interpretation. No new scaffold generation, no judge-driven filtering and no
new language-model weights are introduced.

Persist per-query metrics, per-draw text/vectors, fit parameters, prediction
arrays, row/fold manifests, code SHA, capture audits, cap/empty counts,
publication receipts and figure source data. Publish the final transfer figure
and answer-comparison report with browser-accessible URLs. No raw generations,
metrics or configs are discarded.

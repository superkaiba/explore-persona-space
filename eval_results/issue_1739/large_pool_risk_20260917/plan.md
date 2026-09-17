# Large-pool behavior-risk retrieval: frozen selection and response staging

## Goal and authorization

User request (2026-09-17): identify contexts most at risk of hallucination,
sycophancy, and harmful compliance in a LARGE pool, including the context-derived
and answer-on-context controls. This follows task 1739's existing goal; it does
not change that goal. The previously reported quantitative retrieval experiment
used small judged datasets. The earlier million-pool run ranked prompts only.

This phase freezes the selections and stages already generated Qwen answers.
It makes no new model, judge, or GPU calls. Behavioral validation is a distinct,
unfinished phase: the configured OpenAI key fails authentication, and the user's
AGENTS.md prohibits automated Claude/Anthropic. No replacement judge is silently
substituted. The user has been asked for an authorized working judge endpoint.

Estimated GPU-hours (total): 0

## Frozen artifacts and population

- Qwen2.5-7B-Instruct, layer 19, generic map trained on 963,444 pairs;
  existing instruction-contrast persona directions, no new behavior regression.
- Cached scores: `/dev/shm/issue1739-million-cached/outputs/retrieval/`;
  verify every source checksum in its completion manifest before selection.
- Source input indices and normalized full-prompt hashes:
  `/dev/shm/issue1739-fixed-transfer/map_inputs/`.
- Responses: `superkaiba1/explore-persona-space-data`, revision
  `9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5`, prefix
  `issue779_monitoring/fitter-fair-comparison-n1m/raw_completions`.
  Verified live listing: 1,920 response chunks (32 shards, 60 chunks per shard,
  500 manifest indices per chunk), plus 16 skip sidecars. Verified real schema:
  `rows[{ci,prompt,response}]`; shard01_chunk0000 covers indices 30000..30499.
- Use the 959,844 captured new-pool contexts. Exclude the original 3,600 map
  training examples uniformly because their response text was not persisted.
  Report both row count and normalized-prompt unique count.
- Deduplicate the response-available pool BEFORE scoring/selection by lowercase
  whitespace-normalized full prompt. Keep the lowest original map row for all
  methods, independent of responses and score. This gives every arm the same pool.
- This population was used to train the map. Results describe retrieval inside
  that population, not generalization to unseen contexts. Cached answers are the
  map's training targets: even successful behavioral validation cannot establish
  independent fresh-rollout risk or held-out generalization.

## Rankings and fixed sampling

Let `z=(x-mu)/sd`, `yhat=z W + mu_y`, and answer/context contrasts be `v_A,v_C`.
Use the already frozen TSVD inverse of rank 378, selected on 400 generic
validation pairs without behavior labels.

For each of evil/harmful-compliance, sycophancy, and hallucination, freeze:

1. Preimage similarity: cosine(z, TSVD_inverse(v_A)).
2. Context-derived control: cosine(z, sd * v_C).
3. Answer-on-context control: cosine(z, sd * v_A).
4. Mapped-answer readout: z W v_A (the omitted bias is ranking-constant).

The first three use exactly the standardized-context cosine geometry of the
existing small-dataset preimage analysis. They are not raw-space cosines.
The fourth is a separately labeled direct predicted-answer projection diagnostic;
it does not use context-norm normalization.

Take the highest-scoring 200 unique prompts for every arm. Break ties by original
map row. Draw one shared uniform random sample of 1,000 unique prompts with seed
1739; retain legitimate overlaps with top sets. Freeze the union and all arm
memberships before inspecting answer text or producing any new judgments.
Report top-50/100/200 as a predetermined curve, with top-200 primary. No
post-result choice of k, layer, direction sign, rank, or headline method.

These sample sizes are a new resource choice, not a validated hyperparameter:
needs measurement pilot before judging. For independent Bernoulli observations,
the worst-case binomial margin is about 7 percentage points for 200 and 3.1 for
1,000; template clustering and shared training targets can make uncertainty larger.

## Behavioral validation still needed

Primary quantitative output: graded behavior mean at each retrieval budget,
alongside a validated behavioral positive fraction and enrichment over the random
pool baseline, with uncertainty and overlaps accounted for. Compare all arms on
the same frozen union; the judge must be blinded to scores, methods and ranks.
Count actual model refusal as a valid non-compliance result, not a missing label.
For harmful compliance, benign assistance must score zero: the earlier generic
"evil" trait score and unconditional request compliance are not this construct.
Hallucination needs reference-supported fact checking or an explicit unassessable
label; unsupported judge impressions cannot be called verified hallucinations.

Before any judged headline: freeze the specific non-Claude judge, rubric, decoding
and cache fingerprints; validate on a separately sampled calibration set, check
reliability/format/refusals, and report missingness. Review template concentration
and provide a template-diverse sensitivity analysis. This phase must not turn
prompt wording, answer-vector scores, or manual cherry-picked examples into
behavioral risk estimates. No success criterion assumes the preimage will win.

## Execution and durability

Use at most four concurrent HF downloads; verify every fetched blob against
the pinned Hub inventory before joining by ci and normalized prompt hash.
Persist selected transcripts and a separate judge input file without method/rank
metadata. Split text artifacts below 9 MB. Stage fewer than 3.3 GB of source
JSON, no activation download or map refit. Shared-VM thread caps apply.

Run the existing bounded phase supervisor with a dedicated persistent independent
watchdog, real recovery-worker canary, and acknowledged personal notification.
Maximum two recoveries/day; never start duplicate workers. Hash-verify archived
outputs and exact source before declaring preparation complete. This is a
preparation-complete sentinel, not an experiment/behavioral-validation completion.
The original artifacts and paper claims stay intact pending actual measurements.

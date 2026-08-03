# RQ5 — Auto-interp tooling, operational practice, and cost

Research notes for the auto-interpretability literature sweep. Scope: what
open-source tooling exists for running auto-interp at scale, what it costs,
what the project already has in-house, and what would have to be built for a
production run over a 16,384–131,072-feature dictionary.

Sweep date 2026-07-28. Budget used: 2 arXiv MCP calls, 11 web calls.

---

## 0. Headline

The single most decision-relevant fact in this RQ is a repo finding, not a
literature finding: **Neuronpedia already hosts auto-interp explanations for
the exact SAE this project uses**, and the repo already contains a working,
fail-loud ingester for them. `scripts/issue1482_sae.py:156-159` asserts the
project SAE is `dict_class == "BatchTopKSAE"`, `dict_size == 131072`,
`layer == 19`, `lm_name == "Qwen/Qwen2.5-7B-Instruct"`, `k == 64`;
`scripts/issue1482_feature_extremes.py:90-94` points at Neuronpedia
`qwen2.5-7b-it` / `19-resid-post-aa`, `hfFolderId resid_post_layer_19/trainer_1`
— the same object. Measured coverage on the last round was **343/358 features
resolved, 343 with a description, 343 from `gemini-2.0-flash`**
(`eval_results/issue_1482/feature_extremes/extremes.json`, `neuronpedia` block).

So a large fraction of the "explain 131k features" problem is already solved
off-the-shelf and free. The build-vs-adopt question is therefore narrower than
it first appears — see §4.

---

## 1. Tooling table

| Tool | What it does | Explainer model | Code reusable? | Evidence packet | Scoring |
|---|---|---|---|---|---|
| **EleutherAI `delphi`** (formerly `sae-auto-interp`) | Full generate+score pipeline for SAE/transcoder features; activation caching → explanation → scoring | Default `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`; offline vLLM or OpenRouter API client | **Yes** — Apache 2.0, `pip install -e .`, CLI `python -m delphi <model> <sae> --hookpoints ... --scorers detection` | Max-activating token windows with `<<delimiters>>` marking activating tokens; `ContrastiveExplainer` variant adds FAISS-retrieved hard negatives | detection, fuzzing, recall, simulation (OpenAI fork), surprisal, embedding, intervention |
| **Neuronpedia** | Hosted platform + bulk explanation exports for many public SAEs | Per explanation-type; recent `np_max-act-logits` runs use `gemini-2.5-flash` / `gemini-2.5-flash-lite`; the Qwen2.5-7B-IT set the repo pulls is `gemini-2.0-flash` | **Data yes, pipeline partially** (repo `hijohnnylin/neuronpedia`); the S3 export is the reusable surface | Per explanation type (below) | Hosts scores; `np_*` types are generation protocols |
| **SAEBench autointerp** | Auto-interp as one benchmark axis alongside SCR / TPP / sparse probing / feature absorption | LLM judge (configurable) | Yes (SAEBench repo) | Top-k activating + importance-weighted + random sequences | Detection-style: judge predicts which shuffled sequences activate the feature; score = accuracy |
| **Transluce `observatory`** | Neuron (not SAE) descriptions at scale + Monitor observability UI | **Open-weight** finetuned `Llama-3.1-8B-Instruct` explainer + simulator (`huggingface.co/Transluce/llama_8b_explainer`) | Yes — GitHub `TransluceAI/observatory` | Activation patterns per neuron | Finetuned simulator; Pearson correlation of predicted vs actual activations on held-out inputs |
| **Goodfire Ember** | Hosted commercial mech-interp API; features as first-class objects with labels | Proprietary | SDK only (`pip install goodfire`); no self-host | Proprietary | Proprietary |
| **InterpAgent** (arXiv 2605.01555) | Multi-agent loop: hypothesis proposal + targeted prompt controls; also feature *discovery* via kNN graph in activation space | Agent-driven, multi-call | Research code | Iteratively constructed, agent-chosen probes | Multi-metric; produces auditable explanation traces |

### Neuronpedia explanation types (verbatim protocol details)

- **`oai_token-act-pair`** — OpenAI's original method ("Language models can
  explain neurons in language models"), modified to add newer models and
  context windows. Explainer sees pairs of (token, activation value) from
  top-activation examples.
- **`np_max-act-logits`** — "Attempts to replicate Anthropic's autointerp used
  for their attribution graphs paper's features." Settings quoted verbatim from
  the type page: *"Activations shown = 24 tokens around max act. Shows top 10
  logits. Shows model the max activating token too. Uses top 10 deduplicated
  activations."*
- **`np_max-act`** — the same family without the logit-lens component.
- **`np_acts-logits-general`** — a further variant (type page exists; protocol
  not fetched).

**Export format.** Bulk exports live at
`https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/`
(legacy: `neuronpedia-exports.s3.amazonaws.com`). Docs say exports exist "to
avoid hammering our server"; missing exports can be requested at
support@neuronpedia.org, "within 48 hours". The repo's ingester establishes the
concrete layout empirically: prefix
`v1/{model_id}/{source_id}/explanations/`, gzipped JSONL batches, one record per
line carrying **`index`**, **`description`**, **`explanationModelName`**,
**`typeName`** (`scripts/issue1482_feature_extremes.py:300-317`). Note the docs
page is thin — the public `/api/feature` endpoint 500s and
`/api/explanation/export` was retired in favour of the S3 mirror
(`scripts/issue1482_feature_extremes.py:88-89`), so the repo's empirically-derived
layout is more reliable than the published docs.

### Feature classification (rather than description)

Nothing mainstream. The closest published thing is InterpAgent's separability +
semantic-coherence criteria for *discovery*, and SAEBench's detection scorer,
which is a binary classification of sequences, not of features. **The project's
own `issue1482` rubric is a feature-classification instrument** (level ∈
{low, high, unclear}; persona_related ∈ {yes, no, unclear}) and appears to have
no off-the-shelf equivalent — this is a genuine in-house asset, not a
reinvention.

---

## 2. Cost and scale figures

All figures below are quoted from the named source, not extrapolated unless
labelled "projection".

| Source | Scale | Cost | Per feature |
|---|---|---|---|
| EleutherAI blog (`blog.eleuther.ai/autointerp/`) | 1.5M GPT-2 features | **$1,300** via Llama 3.1 API | ~$0.00087 |
| EleutherAI, same | 1.5M GPT-2 features | **$8,500** via Claude 3.5 Sonnet | ~$0.0057 |
| EleutherAI, same | 1.5M GPT-2 features | **~$200k** with prior (OpenAI simulation) methods | ~$0.13 |
| Transluce | **458,752** MLP neurons of Llama-3.1-8B-Instruct | $15,951.40 tokens + $5,174 GPU | **~$0.046** (their stated figure) |
| InterpAgent (2605.01555) | per feature | 13 LLM calls, ~17,000 tokens | **$0.006**; ~$6 per 1,000 features |

**Token counts per feature (EleutherAI, explanation generation only):**
~**963 prompt tokens and 30 output tokens**, averaged. This is the cleanest
per-feature token budget in the literature and the right anchor for projections.

**Examples shown per feature (EleutherAI detection scoring):** five activating
examples drawn from different deciles of the activation distribution, plus
**twenty non-activating examples**. Fuzzing uses ~70 total samples per feature.

**Why the 50× spread between EleutherAI and Transluce.** EleutherAI's $0.00087
is *explanation only* on a cheap open model; Transluce's $0.046 includes
simulation *scoring* (the expensive part — it is token-by-token activation
prediction) plus GPU cost. Scoring, not explaining, is what dominates a
production budget. This is the main operational lesson from the cost data.

**Batching / caching / Batch-API practice in published runs.** Thin. Nobody in
the sources surveyed reports Anthropic-Batch-API-style usage explicitly;
EleutherAI's cost reduction comes from (a) moving to cheap open-weight
explainers served locally by vLLM, and (b) replacing simulation scoring with
detection/fuzzing. Transluce's comes from finetuning an 8B open-weight explainer
that beats finetuned GPT-4o-mini on their simulation score. **The consistent
field strategy for scale is "make the explainer cheap and local", not "batch the
frontier API".** The project's Batch-API machinery (§3) is therefore ahead of
published practice on the dispatch axis and behind it on the explainer-cost axis.

**Projection for this project** (my arithmetic, from the in-house instrument
below, Sonnet 4.5 at $3/Mtok in / $15/Mtok out; Batch API halves both). Current
packet ≈ 1,070 input tokens (8 × 400-char snippets + system + Neuronpedia aux)
and reason-then-label output capped at 400:

| Features | Draws | Sync | Batch API |
|---|---|---|---|
| 16,384 | 1 | ~$90–150 | ~$45–75 |
| 16,384 | 5 (rule-4 compliant) | ~$450–750 | ~$225–375 |
| 131,072 | 1 | ~$715–1,205 | ~$360–605 |
| 131,072 | 5 | ~$3,600–6,000 | ~$1,800–3,000 |

Range endpoints are typical (~150 output tokens) vs ceiling (400). The 131k × 5
cell is the one that should drive a design decision: at that scale a local
open-weight explainer is the field-standard answer, not the project judge.

---

## 3. In-house machinery inventory

### What already exists and works

**A complete single-round auto-interp instrument** lives in
`scripts/issue1482_feature_correlates.py` + `scripts/issue1482_feature_extremes.py`.
It is a real pipeline, not a sketch, and it ran clean: 358/358 features labeled,
**0 content drops, 0 transport drops**.

- **Rubric / instrument.** `issue1482_feature_correlates.py:65-81` — judge pinned
  to `claude-sonnet-4-5-20250929` (CLAUDE.md project pin), `JUDGE_MAX_TOKENS = 400`
  explicitly justified against llm-judging rule 23 (reason-then-label needs
  ≥~300), and a reason-then-label JSON rubric with anchored level definitions.
  This is rule-6/rule-7 compliant by construction.
- **Rubric extension with byte-exact parity assert.**
  `issue1482_feature_extremes.py:100-111` appends a `persona_related` field and
  `:528` asserts `JUDGE_SYSTEM_EXT.startswith(FC.JUDGE_SYSTEM)`, so the reference
  rubric survives as a byte-exact prefix. Rubric sha16 is recorded in the output
  doc (`issue1482_feature_correlates.py:400-402`). This is unusually disciplined
  instrument-versioning — better than anything in the surveyed tooling.
- **Evidence building — corpus scan.** `issue1482_feature_correlates.py:152-254`
  (`phase_scan`): one streamed pass over 1,920 pooled shards, vectorized
  `np.bincount` accumulation, periodic top-K compaction (`_compact`, `:174-184`),
  no per-feature Python loop. Includes a **wiring gate** (`:319-324`) asserting
  recomputed activity matches the committed covariate to < 1e-3 before any read.
- **Evidence building — text retrieval.** `:257-297` (`phase_texts`): streams raw
  chunks keeping only needed contexts, **per-chunk checkpointed JSONL cache with
  crash-tolerant resume** (`:269-284`, tolerates a truncated tail at `:274-276`),
  and a fail-loud assert that every needed context resolved (`:295-296`).
- **Evidence packet assembly.** `:300-333` (`_judge_items`): top-8 answers per
  feature (`TOP_K_CONTEXTS = 8`, `:62`), truncated to 400 chars each
  (`SNIPPET_CHARS`, `:63`), joined with `---`, plus the Neuronpedia auto-interp
  description as **explicitly labelled auxiliary evidence** with a "may be wrong"
  hedge and a graceful "no description available" branch (`:318-324`).
- **Judge dispatch.** `:346-407` (`phase_judge`) calls
  `dispatch_judge_items` with a per-phase `checkpoint_dir`.
- **Drop accounting, rule-9/rule-24 compliant.** `:366-375` splits
  `content` vs `transport_or_error` drops; `_validate_level` (`:336-343`)
  drops non-conforming returns rather than coercing.
- **Test-retest reliability.** `:379-391` — 60-item retest with fresh dispatch
  dir and `rt_` id prefix, described in the docstring as "cold cache by
  construction" (`:347-348`), scored with `_cohens_kappa`
  (`scripts/issue1482_analysis.py:289`).
- **Neuronpedia ingester.** `issue1482_feature_extremes.py:263-356`
  (`phase_neuronpedia`): resumable per-file cache with atomic `.part` → rename
  (`:283-291`), 6-way `ThreadPoolExecutor` (`:293`), gzip-JSONL parse (`:300-317`),
  fail-loud on transport failure, and a **cross-round reproducibility check**
  against the prior round's export (`:326-335`) which returned **10 agree /
  0 mismatch**.
- **Dashboard.** `issue1482_feature_extremes.py:1587` (`phase_dashboard`) renders
  per-feature HTML including the Neuronpedia description with model attribution
  (`:1559-1561`).

**The shared judge infrastructure is the strongest asset and is genuinely
production-grade.**

- `src/explore_persona_space/eval/judge_dispatch.py:1415-1435` —
  `dispatch_judge_items_async`, the single async core that routes and executes.
  Sync wrapper at `:1613-1619`.
- **Automatic sync-vs-Batch routing** keyed on item count and a live OTPM probe:
  `DEFAULT_THRESHOLD_BASE = 2_000` (`:222`), `OTPM_DIVISOR = 400_000` (`:237`),
  `OTPM_HEADER` (`:253`), `RoutingDecision` dataclass (`:318-333`) with a
  human-readable `render()`.
- **Judge-shard ceiling with a hard-won rationale.** `DEFAULT_SUB_BATCH_SIZE =
  2_000` (`:236`) with a comment (`:223-235`) recording that an 8k judge batch
  *starves* — "one 8k judge shard sat at succeeded:0 for ~9h" — while 500–2k
  shards clear in minutes. A test locks the default so it cannot drift back to 8k.
- **Politeness / etiquette bounds.** `MAX_CONCURRENT_SUB_BATCHES = 4` (`:247`),
  `DEFAULT_MAX_CONCURRENT = 50` (`:238`).
- **Deadline-bounded batch polling.** `BATCH_DEADLINE_GRACE_MIN = 30` (`:249`),
  `BatchDeadlineExceeded` imported at `:104`.
- **Crash reconciliation.** `_RECONCILE_MSG` (`:255-259`) handles the
  "batch created but not recorded" crash-inside-`batches.create` case with
  operator instructions.
- **Dry-run CLI, zero API calls.** `:1625+`, documented at `:79-82`:
  `uv run python -m explore_persona_space.eval.judge_dispatch --n 4400`.
- **Prompt-caching floor known.** `CACHE_MIN_TOKENS = 1024` (`:252`) — below it
  `cache_control` is a silent no-op.

`src/explore_persona_space/eval/batch_judge.py`:

- **Rubric-keyed cache.** `rubric_fingerprint()` (`:105`), `JudgeCache` (`:136`),
  `_hash_key` requiring keyword-only `rubric_key` (`:155-166`, raises if
  omitted), versioned by the literal `_JUDGE_CACHE_KEY_VERSION =
  "EPM_JUDGE_CACHE_KEY_V2"` (`:100`). This is the rule-22 fix — a content-only
  key previously leaked one rubric's judgments into another's scores.
- **Transport-vs-content split.** `is_transport_error_dict` (`:297-320`),
  `_collect_legacy_results` classifying errored/expired/canceled rows as
  transport (`:353-366`), and a cache that **never PUTs a transport error and
  treats a stored one as a MISS** (`:582-583`, `:798-799`).
- **Sharded fire-and-forget submission.** `submit_sharded_batches_fire_and_forget`
  (`:251`), `_chunk_requests` (`:219`), `_enumerate_and_check_cache` (`:548`).
- **Main entry point.** `judge_completions_batch` (`:684`), computing
  `rubric_key` at `:746`.

### What is missing for a 16k–131k production run

1. **Top-K evidence for *all* features, not a sample.** `phase_scan`
   (`issue1482_feature_correlates.py:126-151`, `_sample_features`) draws a
   **stratified 300** (10 deciles × 3 terciles × 10, `:59-61`). The top-K
   candidate buffer is keyed on `samp_pos` (`:161-162`) — a `-1`-filled
   131k array indexing only sampled features. Scaling to all features means
   replacing the `cand` list-of-arrays + `_compact` scheme with a genuine
   per-feature bounded top-K (per-shard partial top-K then merge, or a
   fixed-size heap array of shape `(n_features, K)`). This is the single
   largest build item and it is a real engineering task, not a parameter change.
2. **Token-level activation evidence.** The current packet is **whole answers
   truncated to 400 chars** with no per-token activation values and no
   `<<delimiter>>` marking of the max-activating token. Every external tool
   (Delphi, `oai_token-act-pair`, `np_max-act-logits`) considers token-level
   activation the core evidence. `np_max-act-logits` further shows 24 tokens
   around max-act plus top-10 logits. The in-house packet is *answer-level*,
   which is arguably better suited to the project's persona/behaviour question
   but is strictly less informative for surface/token features — and the level
   rubric asks the judge to distinguish exactly those.
3. **No non-activating / hard negatives.** EleutherAI shows 20 non-activating
   examples alongside 5 activating ones; Delphi's `ContrastiveExplainer` mines
   FAISS hard negatives. The in-house packet is positives-only, which is the
   classic way to get an over-broad explanation.
4. **No explanation *scoring* harness.** There is test-retest kappa
   (self-consistency) but **no detection, fuzzing, simulation, or embedding
   scorer** — nothing that measures whether an explanation actually *predicts*
   activation. Self-consistency is not validity.
5. **Single-draw judging.** `n_draws: 1` and `temperature: "API default"` are
   recorded in the output doc (`issue1482_feature_correlates.py:398-399`).
   llm-judging rule 4 mandates N draws at temperature > 0, mean-aggregated.
   **The measured cost of this shortcut is visible in the data**: test-retest
   kappa is **0.599 for `level` but 0.136 for `persona_related`**
   (`extremes.json`, `judge.test_retest`). A kappa of 0.14 is barely above
   chance — the persona field is not currently a reliable instrument at one draw.
6. **Cost/throughput at 131k.** Nothing in the pipeline is 131k-shaped: no
   resumable cross-run explanation store keyed by (feature, rubric, SAE
   revision), no incremental re-explanation when the SAE changes.

---

## 4. Build-vs-adopt recommendation

**Adopt the data, keep the dispatch layer, build only the evidence builder.**

Concretely, in priority order:

1. **Adopt Neuronpedia's export as the free baseline explanation set.** It
   already covers the project's exact SAE at ~96% (343/358), the ingester
   already exists and is fail-loud with a cross-round reproducibility check,
   and the marginal cost is zero. For any feature where a Neuronpedia
   description exists, a project-judge call should have to *justify itself*
   against that baseline rather than duplicate it. Caveat: the Neuronpedia set
   is `gemini-2.0-flash` on a **generic web corpus** with token-level
   dashboards, which is a genuinely different evidence base from the project's
   answer-level persona corpus — that is why the current code labels it
   auxiliary and hedges it. Keep that framing.
2. **Keep the in-house judge dispatch layer. Do not adopt Delphi's.** The
   sync/Batch router, the 2k judge-shard ceiling (bought with a 9-hour wedge),
   rubric-keyed caching, the transport-vs-content split, deadline-bounded
   polling, and crash reconciliation are collectively more operationally mature
   than anything in the surveyed tooling, and they encode project-specific
   incident knowledge (#658, #810, #1313) that a swap would discard.
3. **Build the all-features top-K evidence builder.** This is the real gap
   (§3.1) and there is no drop-in: Delphi's cache is coupled to its own
   `LatentCache`/safetensors format and its supported coders (Sparsify, Gemma),
   not a custom `BatchTopKSAE`. Budget this as the main engineering item. While
   building, add token-level activation windows and `<<delimiter>>` marking
   (§3.2) — copy the format from Delphi/`np_max-act-logits` rather than
   inventing one — and add non-activating examples (§3.3).
4. **Adopt Delphi's *scoring* concepts, and probably its code, for validation.**
   Apache 2.0, and detection/fuzzing are exactly the cheap scorers that replaced
   $200k simulation. Even scoring a stratified *sample* of features would
   convert the current self-consistency-only story into a validity story. This
   directly serves the standing rule that a judge-derived DV must be validated
   against a non-judge reference.
5. **For a genuine 131k × multi-draw run, move the explainer off the frontier
   API.** Both scaled precedents did this: EleutherAI (Llama 3.1 70B AWQ via
   vLLM) and Transluce (open-weight finetuned Llama-3.1-8B explainer, free
   weights on HF, beating finetuned GPT-4o-mini). At 131k × 5 draws the Sonnet
   bill is ~$1.8–3k on the Batch API; a local vLLM explainer on an already-
   provisioned pod is close to free. Retain Sonnet as the *calibration* judge on
   a sample — which is also how you'd validate the local explainer.

**What not to do.** Do not adopt Delphi wholesale as the pipeline — the custom
`BatchTopKSAE` integration cost plus losing the in-house dispatch maturity
outweighs what its explainer loop provides. Do not run 131k features at one
draw and report `persona_related`: kappa 0.136 says that number would be noise.

---

## 5. Verification ledger

| Claim | Source | Status |
|---|---|---|
| Delphi = EleutherAI auto-interp lib, Apache 2.0, scorers detection/fuzzing/recall/simulation/surprisal/embedding, default explainer Llama-3.1-70B-AWQ-INT4 | `github.com/EleutherAI/delphi` README (live fetch) | VERIFIED |
| Delphi paper "Automatically Interpreting Millions of Features in Large Language Models" | arXiv **2410.13928**, Paulo/Mallen/Juang/Belrose, 2024-10-17 | VERIFIED via arXiv MCP |
| $1,300 (Llama 3.1) / $8,500 (Claude 3.5 Sonnet) / ~$200k (prior) for 1.5M GPT-2 features | `blog.eleuther.ai/autointerp/` (live fetch, quoted verbatim) | VERIFIED |
| ~963 prompt + 30 output tokens per feature; 5 activating + 20 non-activating examples; `<<delimiter>>` marking | same | VERIFIED |
| Transluce: 458,752 Llama-3.1-8B MLP neurons, ~$0.046/neuron, $15,951.40 tokens + $5,174 GPU, open-weight `Transluce/llama_8b_explainer` | `transluce.org/neuron-descriptions` (live fetch) | VERIFIED |
| `np_max-act-logits` settings: "24 tokens around max act", "top 10 logits", "top 10 deduplicated activations" | `neuronpedia.org/explanation-type/np_max-act-logits` (live fetch, verbatim) | VERIFIED |
| Neuronpedia S3 export buckets + "within 48 hours" support turnaround | `docs.neuronpedia.org/api` (live fetch) | VERIFIED |
| InterpAgent: 13 LLM calls, ~17k tokens, $0.006/feature, ~$6/1,000 | arXiv **2605.01555** (id + title + authors verified via MCP); cost figures from web search summary, **not** from the paper text | PARTIAL — see §6 |
| Project SAE = BatchTopKSAE, 131,072 dict, layer 19, Qwen2.5-7B-Instruct, k=64 | `scripts/issue1482_sae.py:156-159` | VERIFIED (asserts in code) |
| Neuronpedia coverage 343/358, all `gemini-2.0-flash`; prior-round overlap 10 agree / 0 mismatch | `eval_results/issue_1482/feature_extremes/extremes.json` | VERIFIED (read from artifact) |
| Judge run: 358/358 labeled, 0 content drops, 0 transport drops, Sonnet 4.5, max_tokens 400 | same artifact | VERIFIED (read from artifact) |
| Test-retest kappa 0.599 (level) / 0.136 (persona_related), n=60 | same artifact, `judge.test_retest` | VERIFIED (read from artifact) |
| Judge routing constants (2000 threshold, 2000 shard ceiling, 4 concurrent sub-batches, 400k OTPM, 1024 cache floor) | `eval/judge_dispatch.py:222-253` | VERIFIED (read from source) |
| Cost projections for 16k/131k features | my arithmetic from the in-house packet size × Sonnet 4.5 list pricing | PROJECTION, not measured |

---

## 6. Could not verify

- **InterpAgent's cost figures** ($0.006/feature, 13 calls, ~17k tokens). The
  arXiv id, title, and authors are MCP-verified, but the numbers came from a
  web-search summary of the PDF, not from the paper body. Verify against the
  paper before citing the number.
- **SAEBench autointerp implementation details** — the generation/scoring
  two-phase description came from a search summary. I did not fetch the
  SAEBench repo, so I cannot state its judge model, prompt template, or file
  layout, and I cannot confirm whether its harness is separable from the rest
  of the benchmark.
- **Neuronpedia API specifics** — endpoint URLs, auth, rate limits, and the
  per-record schema are *not* documented on `docs.neuronpedia.org/api`; the
  page is explicitly "a work-in-progress". The field names in §1 are derived
  from the repo's working parser, not from published docs. `api-doc` was not
  fetched (budget).
- **Delphi custom-SAE support.** The README says "Both Sparsify sparse coders
  and Gemma sparse coders are supported" and does not document a custom-encoder
  hook. I could not confirm whether a custom `BatchTopKSAE` can be plugged in
  without forking. **This is the load-bearing unknown for recommendation 4** —
  if a custom coder is easy to register, adopting Delphi's scorers gets cheaper;
  if not, the scorers must be reimplemented against the in-house cache. Resolve
  by reading `delphi/sparse_coders/` and `delphi/__main__.py` before committing.
- **Whether any published auto-interp run uses a Batch API.** I found no
  positive evidence and no explicit denial; absence of evidence only.
- **Anthropic's own published auto-interp pipeline.** Not located as a
  reusable artifact. `np_max-act-logits` self-describes as an attempt to
  *replicate* the attribution-graphs autointerp, which implies Anthropic's own
  pipeline is not released.
- **Prompt templates verbatim.** I did not retrieve a full verbatim explainer
  prompt from any external tool — only structural descriptions. Delphi's
  prompts live in its repo and should be read directly if adopting its format.

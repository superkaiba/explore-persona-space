# Linear retrieval failure audit

This exploratory analysis examines the linear map trained on 963,444 contexts,
evaluated on the same 942 held-out queries and exactly 10,000 unique candidate
answer vectors as the preceding run. It reuses the five-answer means, fixed
training-only whitening and two-sided CSLS with K=10. No text was generated and
no predictor was fitted.

## Rank severity and overlap

There are 90 top-1 failures (9.55% of queries). Of those, 44 place the correct
answer second, 16 third, seven fourth and seven fifth. Thus 74/90 failures
(82.2%) still retrieve the correct answer within the top five. Overall recall
is 95.12% at rank 2, 98.30% at rank 5, and 99.15% at rank 10. The worst correct
answer rank is 27.

All 25 failures from the original 942-candidate pool remain failures; adding
9,058 candidates introduces 65 more. The nonlinear map fails on 65 of the
linear failures, rescues 25, and introduces 12 failures on queries that the
linear map gets right. Its net improvement is therefore 13/942 queries.

## A concentrated prompt family

An explicitly reproducible text flag identifies prompts containing either
`chemical industry` or `a chemical company`, ignoring case. This flag covers
48/942 queries and 29/90 failures. Its failure rate is 29/48 (60.4%), compared
with 61/894 (6.8%) outside the flag. Within this group, 17/23 article requests
and 12/25 company-introduction requests fail. The three worst-ranked queries
are company introductions, with correct-answer ranks 27, 21 and 20.

All 29 flagged failures retrieve another answer within the same flagged prompt
family. This directly supports confusion within recurring article/introduction
templates, rather than a classification based only on the failed query's topic.
Of these 29 failures, six already failed in the original pool and 23 became
failures after its enlargement.

This is a descriptive pattern identified after observing the failures. These
queries share recurring templates and face a candidate bank with related
templates; the counts do not establish a causal weakness specific to chemistry.

## What the mistaken matches look like

An independent analyzer inspected all 90 prompt pairs and representative
seed-43 answers, assigning one descriptive label to each pair. The per-query
labels, definitions, confidence and exceptions are saved in
[annotations.json](annotations.json).

| Exploratory category | Failures |
|---|---:|
| Same task/template, changed entity or content | 52 |
| Near-duplicate request | 24 |
| Functionally interchangeable representative response | 6 |
| Broad output-format or software-framework similarity | 6 |
| Unrelated or unclear | 2 |

These are single-pass qualitative categories, not validated semantic judgments.
Near-duplicate **requests** do not guarantee interchangeable **answers**. The
six response-convergence cases involve generic refusals or identical short
labels; this does not establish that all five draws are equivalent or that the
retrieval should be scored correct.

The following are paraphrases of verified stored pairs, chosen to illustrate
different patterns. The rank is that of the correct answer; the substituted
answer ranks first. Query indices refer to `audit.json`.

| Query index | Requested answer | Retrieved answer | Correct rank | Nonlinear rank |
|---:|---|---|---:|---:|
| 809 | Introduction to Derthon Optoelectronic Materials | Introduction to Jiangsu Dacheng Pharmaceutical and Chemical | 27 | 2 |
| 385 | Introduction to Biochempeg Scientific | Introduction to GUCH CHEMS | 21 | 1 |
| 307 | Three-day Vancouver itinerary emphasizing Chinese food | Four-day Hawaii family itinerary | 8 | 1 |
| 810 | Definition quiz for “innovative” | Cloud-computing multiple-choice revision questions | 5 | 1 |
| 233 | Markdown article about a growth mindset | GitHub README for a corporate-wiki Slack bot | 3 | 1 |
| 579 | Danish request for a multiplication table | Swedish request for prose about sewing/crafts | 11 | 1 |
| 800 | “Hi how are you?” | “Hi! How are you?” | 6 | 6 |
| 235 | Asking the assistant's name | A spelling variant of the same question | 5 | 1 |

The examples range from near-duplicate intent to meaningful content changes
within a shared task or output format. The Danish-table/Swedish-prose pair is a
larger mismatch: its two actual mean answer vectors have whitened cosine 0.011.
Shared language cues are a plausible description, not an established cause.
There is also a response-mode mismatch among similar roleplay requests: one
stored seed-43 answer refuses while its retrieved counterpart complies. Similar
prompts alone therefore do not establish interchangeable answers.

Across all failures there are 83 distinct wrong winners; no wrong candidate wins
more than twice. The error set is spread across candidates. In 74/90 cases the
wrong winner is among the ten closest *other* candidates to the actual answer
under whitened cosine (the stored diagnostic includes the actual answer itself,
so its threshold is rank 11). This is a geometric neighborhood statement, not a
semantic-equivalence judgment.

## Scope and interpretation

Failure means that the exact paired answer vector does not rank first. It does
not mean that the language model generated an incorrect answer. Text inspection
uses the stored seed-43 generation as an illustration; retrieval itself uses
the average of five answer vectors. The other three fresh query-answer draws
are also available in the text bank through the audit's source references.

Changing the metric rescues some particular cases but reduces total accuracy:
whitened cosine without CSLS retrieves 87.15% at rank 1, raw cosine 80.47%, and
raw Euclidean distance 79.62%, compared with 90.45% for whitened CSLS. These are
alternative measurements of the same saved predictions, not additional fits.

## Data and reproducibility

The source `../summary.json` contains all ranks for both predictor families and
nine training sizes. `audit.json` contains diagnostics for all 942 queries,
including exact top-five candidate IDs, score margins, cosine similarities,
training-size rank histories and exact text-source references for all 90 failed pairs. Its
metadata records source hashes, the exact candidate-pool hash, and text-file
provenance. JSON is loaded with the standard library; numerical arrays are
loaded with NumPy and the original project tensor helpers.

All numerical inputs use Hugging Face dataset
`superkaiba1/explore-persona-space-data` at revision
`83d249cc9d495ca6f5d10f9156a622bcdca29a19`. The local text bank covers every
candidate prompt and seed-43 response, plus seeds 44–46 for the 942 original
test candidates. All 35 source files were checked against the pinned Hub
revision and all 10,000 prompt hashes match its bundle index. The sole disclosed
source redaction concerns a candidate outside this pool.

The audit rechecks every consumed numerical input hash, requires the exact
saved pool provenance and text-bank pool hash, and requires exact reproduction
of all 942 saved linear ranks before writing output. Query text joins use
capture IDs, with negative IDs indexing the original test set before
deduplication. Text examples are exploratory annotations of these fixed pairs,
not independently validated semantic-equivalence labels.

The vector pass completed successfully at 2026-09-07 20:42:34 UTC and reproduced
every saved rank. The text-bank assembler was rerun separately and reproduced
all 10,000 text rows, coverage counts and source hashes exactly. Independent
review found no material numerical/provenance issue. Ruff passes. Full no-flags
workflow lint completed with 19 pre-existing errors in unchanged files; the
mapped thread-cap suite completed with 26 passes and one pre-existing failure
listing 14 unrelated entrypoints. Neither new script is flagged by these checks.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  uv run python scripts/issue1901_retrieval_text_bank.py \
  --round1-prompts /path/to/original/first5000_prompts.json

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  uv run python scripts/issue1901_linear_failure_audit.py \
  --source-stage /path/to/verified/base-and-prediction-cache \
  --stage-root /path/to/verified/distractor-cache \
  --text-bank /path/to/text_bank.json
```

The first command requires the original ordered 5,000-prompt cache and verifies
its fingerprint against the pinned bundle metadata. The assembler documents the
parent's source-data re-derivation route; it never silently resamples prompts.
Raw text remains in the source bank. This report uses paraphrases and preserves
the source IDs/hashes needed to inspect the original draws.

For subsequent analysis, distinguish near-duplicate/interchangeable pairs from
content-changing confusions before interpreting exact-ID retrieval as semantic
discrimination. A separate candidate-bank sample would be needed to assess how
much the observed template concentration generalizes beyond this fixed bank.

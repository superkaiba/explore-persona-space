# Answer-property readout pilot — 7 September 2026

Selected instruction conditions are readily recoverable by linear readouts of individual answer-token means, including persona, format, register, language, and requested words. This pilot supports the availability of these distinctions in the answer representation on held-out questions. It does **not** establish a general ordering of high-level versus low-level property decodability, and most labels describe the requested condition rather than verified realized behavior.

This was an inline CPU analysis under existing task #2564 (pre-launch progress v106; measured launch progress v107). All six planned panels and 36 outer folds completed successfully. No new generations, judge calls, GPU allocation, paper edits, or bibliography edits occurred.

## Results

| Readout target | Answers / contexts | Balanced accuracy (95% carrier bootstrap interval) | Chance | Mean fold AUROC |
|---|---:|---:|---:|---:|
| Requested persona (5 classes) | 1200 / 120 | 99.92% (99.75–100.00%) | 20.00% | 0.9999 |
| Requested format (5 classes) | 1200 / 120 | 98.83% (96.58–100.00%) | 20.00% | 0.9993 |
| Requested register (2 classes) | 480 / 48 | 100.00% (100.00–100.00%) | 50.00% | 1.0000 |
| Requested marker (5 classes) | 1200 / 120 | 95.42% (90.50–99.00%) | 20.00% | 0.9946 |
| Actual word presence (5 binary labels) | 1200 / 120 | 93.55% (90.36–96.50%) | 50.00% | 0.9888 |
| Requested language (3 classes) | 360 / 36 | 99.17% (97.78–100.00%) | 33.33% | 1.0000 |

All readouts use Qwen2.5-7B-Instruct residual activations at layer 19, averaged over **one answer's tokens**, excluding prompt and end-of-turn tokens. Rows are individual answers, not averages across rollouts. AUROC chance is 0.5 for every panel. The table reports mean within-fold one-vs-rest AUROC, avoiding comparisons of score scales from different fitted folds; pooled OOF AUROC is also preserved in the original JSON. Five-binary-label balanced accuracy averages the separate word-presence balanced accuracies.

The word-presence target is the actual completion text, scored using the existing case-insensitive whole-word rule for *moreover*, *honestly*, *surely*, *notably*, and *essentially*. Its interpretation needs a strong caveat: an instruction-only baseline that assumes the requested word occurs and the other four do not achieves **99.37%** balanced accuracy (95% interval 98.70–99.87%), exceeding the readout's 93.55%. Only **18 of 1,200 answers** differ from this baseline on any word label. Consequently, this bank provides little evidence about predicting answer-specific variation beyond the requested condition. The baseline was added descriptively after inspecting probe results; it required no fitting and did not change the readouts.

## What the labels mean

- Persona: pirate captain, Victorian butler, zen teacher, startup founder, or noir detective — assigned instruction, not an answer-level persona judgment.
- Format: bullets, numbered list, poem, paragraph, or JSON — assigned instruction, not an answer-level structure validator.
- Register: formal/businesslike or informal/chatty — assigned instruction, not an answer-level tone judgment.
- Requested marker: which of the five words the instruction requests — assigned instruction. Actual word presence is a separate multilabel readout on the same answers.
- Language: English, Chinese, or Spanish — assigned instruction from the archived language pilot, not an automatic language-identification judgment.

Labels can be predictable because the representation retains the instruction even when the response fails to realize it. The original judge-score archive was inspected during feasibility work but was not used to train or select these readouts. This pilot therefore cannot turn the five condition-label results into verified behavioral classification claims.

## Evaluation protocol

There are 12 carrier questions. Six outer folds each hold out two entire carriers, keeping all their templates and rollouts out of training and model selection. Primary training uses individual-answer draws 0 and 1, selected before scores were inspected; evaluation uses all 10 archived draws for each held-out context. Thus every reported test answer is scored by a readout that never saw its carrier question. Persona, format, and lexical conditions use two instruction phrasings per value; language is a separate three-language source bank.

A standardized, unweighted multi-output ridge classifier uses training-only means and standard deviations. Targets are ±1 one-vs-rest columns; prediction is argmax for conditions and a zero threshold per word. Three carrier-group folds inside each outer training set choose from penalties 0.1, 1, 10, 100, 1,000, 10,000, 100,000, and 1,000,000, reusing the grid of the earlier #779 readout. Exact metric ties choose the strongest penalty. Selection maximizes balanced accuracy for condition labels and macro AUROC for actual word presence. No test labels choose preprocessing or regularization.

The largest outer training set has 200 rows for 3,584 features. This is deliberate regularized classification, not an identifiable unregularized reconstruction. A single dual Gram eigendecomposition handles all penalties and targets per fitting fold; numerical parity against scikit-learn Ridge was checked on underdetermined multitarget data. Across six panels there are 144 small fitting decompositions (three inner fits and one outer fit per fold). A production-shape single-fold persona panel took approximately 0.16 seconds with approximately 701 MiB process peak RSS; the completed panel fit loops total approximately 4.63 seconds. These are local fit-loop timings, excluding input staging and Python startup.

Uncertainty uses 2,000 resamples of the 12 carrier clusters, holding their answers together. These intervals describe limited question-sampling variation conditional on this selected dataset and OOF predictions. They do not include uncertainty from refitting, new persona vocabularies, novel instruction families, or model choice. A 100–100% interval means this small dataset had no observed error; it is not a claim of universal perfect decoding.

## Implication for Dan's comment

The selected persona/style conditions and language/lexical conditions all leave linearly readable signal in this answer-vector construction. It would therefore be premature to explain weaker preservation of a language or lexical distinction solely by saying that the answer vectors contain no corresponding signal.

This does not establish that high-level properties are intrinsically more decodable than low-level properties. The categories differ in label cardinality, source prompts, compliance, and difficulty; language uses a separate bank. Topic was not evaluated: carrier-held-out evaluation removes topic classes if carrier identity is used as the topic label. No topic, SAE-class, context-to-answer linear-predictor, or representation-distance comparison was run. The results cannot determine whether a difference in context-to-answer prediction reflects information loss, task difficulty, or stochastic response variation.

A safe discussion claim is: “On the controlled elicitation bank, linear readouts recover several persona, style, language, and lexical instruction conditions from individual mean-answer activations on held-out carrier questions. These diagnostics establish available condition signal, while a matched comparison of realized properties remains open.” This is a suggested interpretation only; no paper text has been changed.

## Provenance and reproduction

Archived inputs come from Hugging Face dataset `superkaiba1/explore-persona-space-data`, pinned revision `426a48e589c745d31e8bcc49d7f7178a7d6896f2`. `summary.json` records each consumed tensor, manifest, completion archive, and sentinel path, byte count, SHA-256, producer configuration, and pooling boundary. Completions were generated by the same model at temperature 1.0 with a 2,048-new-token limit and later replayed for activation capture. The loader checked context/draw joins, exact span boundaries, model, shape, finite values, and ten draws per context. The language pilot's bare contexts are excluded.

The code is `scripts/issue2564_answer_property_readout.py`; focused validation is in `tests/test_issue2564_answer_property_readout.py`. In the existing repository environment, with archived inputs staged preserving their Hugging Face paths:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run --no-sync python scripts/issue2564_answer_property_readout.py stage_root=/path/to/staged/archive out_root=/path/to/fresh/results hydra.run.dir=/path/to/run_metadata
uv run --no-sync python scripts/issue2564_answer_property_readout.py summarize_only=true out_root=/path/to/completed/results hydra.run.dir=/path/to/diagnostic_metadata
```

The first command refuses an already completed output directory. The second performs no fits and writes diagnostics from saved OOF predictions. Original completion time is recorded in `summary.json`; `diagnostics.json` records its checksum and those of all prediction files. The `pilot/` directory preserves the separate single-outer-fold resource pilot, not an additional test result in the table.

Artifacts: `summary.json` (metrics/config/provenance), six panel JSON files, 36 fold JSON files, six OOF prediction JSONL files, and `diagnostics.json` (instruction-only baseline and within-fold AUROC). Independent review and landing are tracked separately under task #2564.

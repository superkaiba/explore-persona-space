# Comparison pilot terminal-EOS correction

Independent review of the Qwen3.5-4B pilot found that 558 of 560 captured
answers retained token 248046 (`<|im_end|>`). At the exact pinned 4B revision,
there is no standalone `generation_config.json`: Transformers derives EOS
248044 from the model config, while the tokenizer EOS is 248046. The old
capture code consulted only the model's generation config. This violated
the existing frozen instruction to exclude the first terminal EOS/stop ID
and everything after it. Raw generations were correctly retained.

The independently audited original is comparison_component_pilot1, producer
78de99d43d2175eeb080badeba48d40072078d38, upload revision
820fcbe4515a1704f453ffcb43bd00c8ee7f0987, 429 files. Its exact upload-receipt
SHA256 is 38655bb1c48e1fd79476bc88109ff71fc7245984f2a654ec72ff2eaca52e0c8e.
Those results remain preserved as superseded pilot evidence and must not
satisfy main readiness. They are not corrected by relabeling their metrics.

The primary Qwen3.5-27B checkpoint has a standalone generation configuration
with both EOS IDs. An independent scan of all 128 completed main validation
contexts and 640 captured draws found every exact terminal prefix correct
and zero retained EOS states. Its existing f8b4983 main producer remains
unchanged. A precautionary sparse-decomposition interruption preserved and
verified its five completed contexts (313 files, revision
751e84bd5d553d63c50ec2d596ef4c12f02bf6ce); the same computation then resumed.
Lens calibration uses raw calibration sequences and is unaffected.

Future comparison capture takes the union of tokenizer EOS, generation-config
EOS, and any explicitly supplied stop IDs. It records the resolved terminal
policy in each captured row. Sampling settings, main sample counts, prompts,
layers, dictionaries, fit budgets, and interpretation rules do not change.

## Recovery and provenance

The real pipeline phase `recover-terminal-eos` accepts only the exact audited
comparison pilot. It preserves original generated JSON and dictionary tensor
bytes, checks every original capture and canonical-input reference against
its verified upload, and trims the already saved state tensor at the first
terminal ID. Every retained activation and every canonical context input is
preserved bitwise. Dictionary manifests, canonical checkpoint contracts,
capture references and coverage receive the genuine recovery pipeline
identity with explicit original producer and byte-hash lineage. Original
artifacts are never rewritten. A draw made empty by correction excludes the
whole context with its seed reported; it is never silently dropped from K.
Each recovered context is atomic and an interrupted recovery can resume only
with matching contracts and bitwise-identical existing tensors.

Corrected pilot decompositions and all fitted predictors are recomputed.
The recovery is uploaded and verified before decomposition, and components
are uploaded before fitting. A new independent review and pilot binding are
required after the corrected fits and final upload. The same pipeline
entrypoint creates recovery, decomposition and fit identities; an independent
review caught and removed an earlier unexecuted standalone-entrypoint design
that would have produced incompatible argv0 provenance.

## Predeclared numerical geometry diagnostic

Before using a trimmed pilot to validate future capture, compare the first
four frozen pilot training contexts, all five original seeds in order, and
original capture batches 2/2/1. Re-execute the original EOS-inclusive forward
and compare it to the saved capture. Separately compare each trimmed saved
prefix to a fresh EOS-excluded forward at the pinned native BF16 eager
runtime. Save both complete token-matrix comparisons and equal-rollout
context-mean comparisons, including relative Frobenius error, every row
cosine, zero-norm rows, and bitwise equality. This recognizes the observed
BF16 dependence on batch/sequence geometry despite causal equivalence.

Use the existing 0.01 relative Frobenius and 0.999 minimum row-cosine yardstick
without relaxing it. This is a diagnostic numerical criterion, not historical
mapping compatibility or a new convergence claim. All four individual and
aggregate comparisons must pass. If they fail, retain the CPU-trim evidence
but perform fresh corrected pilot capture and refit before main approval.
No new sampled answers or favorable replacement contexts are permitted.
The diagnostic is implemented in scripts/workspace_jr_terminal_parity.py.

## Observed diagnostic and required fresh capture

The native diagnostic ran at producer c3e4fc2a8e143af021e37f9d08013d9c07657421
and was uploaded as a47f8b06dd8c01e4067bf29898b62a86000451b9 (7 files).
Its report SHA256 is
9554214e365e31e092ec2142e0c2d37bd5f1576e80c5e9f0a3f2e3f2e48545dc.
Original EOS-inclusive recapture reproduced all saved states bitwise.
Trimmed prefixes versus EOS-excluded forwards failed the predeclared token
criterion: aggregate relative Frobenius error 0.010451198591820955 and
minimum row cosine 0.9988659955892043. The pooled-y comparison passed
(relative error 0.000950081761852301), which does not waive the required
per-token criterion. Three of four individual token comparisons failed.
The enclosing job stopped before recovery, decomposition, or fitting.

The predeclared fallback therefore uses `recover-terminal-eos
--recapture-answers`, with the exact uploaded failed parity report bound by
revision and SHA256. It first validates the same immutable original inputs,
then runs fresh native BF16 eager capture on corrected answer IDs, in seed
order and batches 2/2/1. It retains the exact original canonical context x
and sampled answers. Fresh answer-batch input reads are saved as diagnostics.
Its recovery contract distinguishes fresh answer capture from CPU prefix
trimming; it never claims that fresh answer states are the old prefix bytes.
The saved native diagnostic and original captures retain the prefix-trimming
evidence. Completed main primary captures, its predictor targets, and its
registered sample sizes remain unchanged.

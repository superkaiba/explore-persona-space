# Task2670 technical verification amendment: deterministic context-capacity censoring

## Goal

Test whether the model activation before generating new reasoning or an action predicts a later successful reward hack, and whether the frozen context-to-answer map improves that forecast.

## Existing scientific policy and observed implementation gap

The immutable v6 physical plan already states that a deterministic context limit is an unknown outcome and cannot trigger a replacement draw. The v7 structural-input amendment remains unchanged. This amendment implements that existing policy; it changes no prompt, task exposure, seed, model/runtime, generation budget, ranking, split, feature, fitting rule, or claim criterion.

In the fixed 376-row partial screen, task7 oneoff epoch1 completed six failed submissions. Its seventh request was rejected with HTTP400 because the 65,536-token output reservation plus accumulated input exceeded the 262,144-token context capacity. An independent tokenizer-only check of the exact 13-message request measured 197,792 input tokens, exceeding the reserved input budget by 1,184. No generation or physical draw was repeated. The native scorer-on-error value is I, but the existing collector correctly classifies sample.error as censored; its strict request verifier then rejects the unresolved sample and will produce a failed verification receipt.

## Transparent post-run verification extension

Preserve every byte of the 12-file physical collection closure, native logs, raw rows, original run_result/native_audit files and owned process exit. The current invocation continues to its fixed end, including the expected nonzero collector verification exit if that is the only issue. No report or native score is rewritten to look successful.

Add a separately reviewed post-run module with a new derived-audit sidecar. It must first reproduce the complete original audit and artifact bindings, then accept only the precisely identified deterministic context-capacity error shape. Other native errors, missing or duplicated rows, input/config/seed drift, invalidations and unrecognized exits remain failures requiring investigation.

For each accepted capacity censor, verify the full immutable sample, all completed requests and execution history, exact final feedback, derived next-attempt seed, model/config, error type and capacity arithmetic, and absence of any returned completion, usage or completed timestamp. A clearly named temporary completed-prefix validation view may call the already reviewed strict request validator on a detached deep copy after removing the separately validated terminal failed event and its unmatched feedback and clearing only the copied top-level sample.error. The original error, score, history and all native bytes remain unchanged. That in-memory view is never passed to census/outcome, persisted as native data, classified as a completed failure, treated as a finished trajectory, or used to generate. The original sample remains censored. Retain all completed-prefix request evidence and the rejected request separately. Deterministic tokenizer-only receipts may provide additional exact-length corroboration, with exact request/token hashes; they are not generation draws.

The derived audit retains the original S/F/U census exactly, includes the original failed audit/report hashes and resolved validation issues, and uses an explicit new verification schema. Derived terminal evidence accepts exit1 only when the collector's exact final unverified-collection exception is present, all planned native rows exist, the invocation is terminal, the owned group is drained, and every original validation issue is accounted for by independently verified capacity censors. It cannot normalize an arbitrary process failure.

Selection uses the existing rank_tasks and make_rows functions on the identical census. The source-bound selection receipt includes the derived audit and terminal proof as additional provenance while preserving original artifact hashes. The required pre-fresh v7 full-ranking/split equality proof remains mandatory. Fresh360 uses the unchanged physical collector and may require the same post-run settlement if it encounters an equivalent capacity censor; no selective retry is allowed.

## Explicit consumer updates and verification

Route capture preparation/validation and analysis to the named post-run verifier through ordinary imports, never hidden module-global replacement. The low-level capture model, position, tokenizer/prefix checks, all90 contexts, six shards and tensor recipe remain unchanged. Capture's portable evidence must explicitly bind the derived audit/schema and its independent review. Re-review the changed capture, analysis and archival reconciliation source closures before use. Tests must cover the actual observed failed sample, valid completed rows, wrong error/config/seed/feedback, completed-output impostors, unrecognized issues/exits, immutable source/report preservation and the real downstream consumer seams.

The archive persists original and derived evidence with exact remote readback. The final row and pod-file reconciliation must use these explicit consumers and still prove actual618 screen /360 fresh /90 capture keys. GPU teardown remains gated on real archival proof and managed termination. If fresh censoring occurs, retain completion-conditional estimates and the already frozen prohibition on unconditional supported-benefit claims. No outcome-driven protocol adaptation is introduced.

## Timing

This technical amendment follows inspection of a real screening error. It is not represented as a blinded preregistration. All successful-body reviews remain valid only by exact native/context/history/row hashes. The final report distinguishes observed behavioral failures, structurally non-assessable inputs, native capacity censors, and process/verification failures.

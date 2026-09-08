# Independent high-rate archive reconciliation code review

Verdict: **PASS** for the reviewed adapter code and disclosed software fixtures. This is not a raw archive verification PASS, a successful final-data execution, a scientific result, or a teardown authorization.

Reviewer: independent Codex critic `/root/verify_vllm_runtime`. The implementation and tests were written by the parent agent; this reviewer changed neither. The companion JSON binds the exact32-file dictionary returned by `scripts.context_risk_highrate_reconcile.source_hashes()`.

Reviewed adapter SHA256: `e80ac1aa8bf42f9e7a1cc11d9675cd2b1d4fb8e1d60844b16456d7d4d0169650`.

Reviewed test SHA256: `c5955315050240f13b2924ed0133564fee85729a440385deda6ae78da63e1afb`.

The adapter follows the archive runbook's key contracts: original VM semantic validators remain at their real configured paths; every consumed input is explicitly mapped to byte-identical pinned readback; screening and fresh observations are compared with complete input-derived `(sample_id, epoch)` sets; capture rows are compared with all90 `(task_id, condition, exact_context_sha256)` keys. ASCII indexes are derived from actual reconstructed rows, and the existing mechanical checker counts those full keys. The complete `pod/<relative-path>` set is compared with the nonempty owned-pod inventory, including logs and same-basename files in different directories. Censors and structurally invalid contexts remain in the archival coverage sets.

The first review identified missing end checks for archive/control files, possible output/input tree overlap, an unbridged external collection review file, and insufficient whole-adapter test coverage. The amended source fixes them. It hashes archive receipts, manifest, inventory and review before parsing; refuses overlapping or symlinked roots; bridges the actual launch configuration and its external review; verifies actual imported archive/generic verifier bytes; and repeats the complete downloaded/reconstructed file verification after semantic checks, including pod-only files. Its result records control/source/package hashes and explicitly leaves managed teardown and a fresh pod-write check to the owner.

Executed evidence:

- Original24-test suite passed independently in8.43s before the requested revisions.
- Final35-test suite passed independently in13.53s with `UV_NO_SYNC=1`,8-thread caps, Inspect0.3.261 and OpenAI3.7.0. The suite exercises the actual adapter workflow with978 generation rows and90 capture contexts, physical Unicode line handling, full-key and exact-file comparisons, the real generic row checker, and all eight mid-validation mutations: readback receipt, upload receipt, snapshot manifest, inventory, review, pod-only remote bytes, external collection review and an extra downloaded file. Outputs nested inside RUN/readback are rejected before writes.
- Four additional reviewer cases passed: empty expected set rejection; actual imported generic-helper byte drift rejection; missing independent reviewer rejection; stale reviewed source hash rejection. The latter two fail before creating output.
- Direct script invocation with `--help` passed, exercising the corrected import guard.

The978/90 workflow test deliberately mocks existing collection/capture semantic validators, phase/sample readers and the Inspect file reader with signature-constrained boundaries. Its native file is explicitly a boundary fixture; its observations are not experimental data. All newly added adapter functions, reconstructed byte checks, source bridges, set comparisons, native/raw serialization comparison and the mechanical row API execute real bodies. Prior reviewed collection/capture and V2 archive evidence supplies the separate history for those reused implementations; this review did not regenerate samples, load model weights or repeat the V2 archive tests.

Required actual evidence remains:

1. Complete the fixed618-screen/360-fresh trajectories and90-context capture, with valid original native/process/capture receipts and full independent success reviews. Include this exact review and every source-hash file in the raw snapshot with explicit original source paths.
2. Execute the reviewed V2 raw archive upload and pinned readback on the real artifacts, then run this adapter using the complete final SSH inventory of pod `1bds7vqkrluxkc`. The inventory's `all_owned_workers_drained` field is an external assertion: the owner must establish it from actual supervised exits/live process checks, include any additional owned output root, and repeat the inventory before termination.
3. Independently inspect the resulting exact source/hash/key/file proofs. Persist late receipts and indexes in verified issue-scoped git, account for all other plan-declared raw inputs and claimed URLs, and explicitly resolve any supplemental generic basename/sharding diagnostic. This adapter does not claim to automate every upload-policy duty.
4. Only then post the owner-bound raw-milestone upload PASS and use managed teardown. Preserve the later VM fit/final-report archive obligation; no fit or experiment-completion claim is implied here.

No unresolved substantive code finding remains in the reviewed adapter scope. No production/test edits, task changes, uploads, model calls, workload stops, compute termination, Claude invocation or recursive delegation were performed by this review.

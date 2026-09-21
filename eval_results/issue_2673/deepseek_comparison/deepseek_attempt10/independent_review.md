**No blockers for publication as an honest failed-attempt report.**

Verified:

- All 11 preserved files match receipt sizes and hashes; report copies match authoritative evidence.
- Three tiny CUDA cases passed, but full-model smoke failed at **0.25390228629112244 > 0.01**. Five rows were exact, 13 diverged, earliest at block 5; production coverage remains zero.
- Runtime/request source pins and Qwen’s historical `cb794596… / 20260917_v3` provenance are retained. Thresholds remain unchanged; no unsupported root cause or completed-comparison claim appears.
- The numerical rejection remains an engineering failure. Separately, **831.572710 seconds available < 900-second reserve** supports `compute_limit`; teardown and ledger evidence agree.
- Preparation/review artifacts are consistent as historical evidence.

The apparent tensor-verification contradiction is resolved: the on-pod `false` field reflects zero production chunks; separate assertions validate smoke tensors.

Fresh Python verification was blocked before execution by uv’s read-only cache lock. Hash checks, arithmetic checks, source inspection, and `git diff --check` completed. No edits or operational actions taken.
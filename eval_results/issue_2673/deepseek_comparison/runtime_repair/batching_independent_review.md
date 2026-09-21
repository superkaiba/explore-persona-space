No remaining code-review blockers in the latest versions of the four files. Concurrent edits resolved the test-dispatch, resume-fingerprint, and partial-diagnostic findings.

- Pinned `grouped.py` and `utils.py` hashes match attempt9. The repair checks compiled original function code, module identity, and dispatcher globals before installation.
- The M16 signal requires the verified block-FP8 body as caller. Shared utilities remain unchanged; other grouped dispatchers fail closed.
- The corrected CPU test exercises the real loader/custom-op boundary, replacing only the Triton launch.
- The CUDA diagnostic precedes full model weights, records partial results and exceptions, and restores M16 in `finally`.
- Manifest provenance includes the repair source and override metadata. Measured diagnostic errors remain outside the resume fingerprint. The 1% full-model parity, `1e-5` repeatability, throughput, and failure-preservation gates remain intact.

Execution validation was blocked before Python started: the required `uv run --no-sync` attempted a cache-lock write on the read-only filesystem. I ran no tests or GPU diagnostics. **M16 correctness and throughput remain unverified on GPU**; the tiny diagnostic cannot replace full-model acceptance.

Reviewed SHA prefixes: FP8 helper `4de10f6673ff`, capture `7edb34d9c7af`, artifacts `1c24bd1df7d0`, integration test `44c9f8b8628b`.
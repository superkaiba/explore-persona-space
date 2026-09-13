# Optional reuse of precomputed validation and test components

This is an execution optimization for the two remaining 4B dictionary rotations,
not a change to the frozen experiment. At this checkpoint the implementation has
passed [independent code review](checkpoint_cache_review.json); no foreign GPU
decomposition or cache publication has run. Both prospective workers have the original clean producer checkout and the
63-file, 270,940,804-byte comparison readiness bundle. Each staging operation
exited successfully and passed the actual readiness consumer.

The primary model's required controls retain priority. Use optional precomputation
only after a worker's existing sequence has completed, its required uploads have
verified, its GPU is idle, and the measured remaining schedule supports a benefit.
No current sequence or original queued plan is edited.

## Numerical gate

`scripts/workspace_jr_checkpoint_parity.py` stages the first frozen training
context, its exact capture and raw generation, its canonical context-input batch,
the dictionary manifest and both dictionary tensors, and its original k=5,10,25
components. That context is
`77ccbceb4a7c44f7ff2b8993336e579771b522ccade47e46b84cfbfad3364c84`.
The nine-file closure is 5,146,044,120 bytes. Its two source revisions are
`245f6c801da6fcf62ec874f9d1d39483e57bcdba` (rotation 20260914) and
`5ab538b54a8027b70fa2f458497236a7d88e4682` (rotation 20260915).
The independent source audit checks the immutable Hub objects and their sizes.

The verifier first checks capture, producer, dictionary, canonical input, rollout
seed, token-length, target-mean, noise and per-token-statistic contracts. On the
destination A100 80GB it replays the unchanged numerical functions from producer
`244f89eb8347361484413e803565f1e639255a8c`, including the same GPU Haar rotation,
FP32 sparse pursuit, nested sparsities and batches of at most 128 tokens. Every
computed field must match the saved context exactly, with zero numerical
tolerance. This is a portability check on one fixed training context; it is not
a proof of equality on every possible input.

The report records realized rotation/dictionary tensor hashes, the GCP instance
ID, physical GPU UUID and numerical runtime. Its own verifier provenance remains
separate from the original producer's provenance. Replay tensors and failed
comparison reports are retained. A successful completion marker is written only
after exact parity passes. Existing execution evidence is immutable, including
when `--stage-only` is requested again.

## Original production recipe and cache publication

Actual precomputation uses the unchanged original pipeline and v4 decomposition
supervisor in its clean original checkout. The foreign plan differs from the
original queued plan only in its output root and removal of optional local source
path hints. Its distinct root and Hub prefix are
`comparison_main_precompute_rotation<seed>_v1`. The full validation and test
splits, all original exclusions and all three sparsities are preserved.

Before production, the launch record must write
`cache_execution_provenance.json` into that fresh foreign root, binding its actual
worker fingerprint to the uploaded successful parity report, producer SHA and
rotation. Record and verify that binding for both phase launches. The existing
v4 supervisor persists the resulting operation contracts, scripts, terminal
records, coverage manifests and checkpoints. This launch record and both
completed phase uploads are prerequisites for adoption.

`scripts/workspace_jr_adopt_checkpoints.py` stages immutable foreign inputs outside
the active original root on the same filesystem. It verifies both full splits,
original queued plan bytes, unchanged production wrappers, uploaded successful
terminals, exact original dictionary/capture/generation/input bytes and complete
per-context numerical metadata. It also binds the parity proof's dictionary bytes
to both original plans and requires the same precomputation worker identity.
All files pass validation before any original-root publication occurs.

The publisher writes immutable lineage first, then publishes only tensor
checkpoints with atomic `os.link`, without overwriting files or taking the
original producer's nonblocking subset lock. If an original producer wins a file
creation race, the publisher validates and preserves that file. Hashing and
deserialization use the same open inode, so an atomic producer replacement cannot
make the two reads refer to different files. Incompatible existing files fail
loudly and remain intact.

Coverage manifests and phase terminal records are never imported into the
original run. Its unchanged queued phase validates and reuses any valid optional
checkpoints, recomputes others, then writes its own complete coverage, final hashes
and upload. Publication-event hashes are historical observations; the original
producer may subsequently replace a valid cache checkpoint with its own valid
computation. Only its final uploaded coverage defines the downstream fit inputs.

## Verification

`tests/test_workspace_checkpoint_cache.py` exercises metadata corruption,
immutable-inode reads, both producer/publication interleavings, preservation of
invalid existing files and symlink/path rejection.
`tests/test_workspace_checkpoint_parity_cli.py` runs the actual CLI against tiny
CPU tensor fixtures, checking successful staging/replay, refusal to mutate a
completed proof and retention of numerical-failure evidence. The CPU fixture
does not substitute for the required actual destination-GPU parity check.
`tests/test_workspace_checkpoint_adoption_cli.py` exercises the actual full-split
reader and publication entry point, including dictionary, worker, provenance,
terminal, input-map and queued-plan rejection before publication. The 25 focused
tests passed; the independent review also checked both real reference-file sets
against immutable Hub metadata.

Concrete launch/input review and actual worker proofs must be complete before
GPU use or adoption. If parity or any provenance check fails, preserve the
evidence and let the unchanged original queued work complete normally.

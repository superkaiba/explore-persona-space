# Natural-data CPU preservation audit, 2026-09-07

This records preservation of the completed preparation and fixed-input staging
rounds, not completion of the GPU scaling experiment. GPU generation, capture,
and scoring remain pinned to reviewed code
`e058fdc2634ddcde303caf5730a730580795e522`; saving these audit records does not
change that running implementation.

Both packed bundles were re-read through their indexes, with every decoded
member hash and the exact represented file-name set checked. Every uploaded
file's relative path, size, and Git-blob or LFS content hash matched the local
bundle at immutable data-repository revision
`b09257af799e380b76fb3e8979777578821a3303`:

- [Preparation audit (315 files)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b09257af799e380b76fb3e8979777578821a3303/issue1739_natural100k_20260906/cpu_preservation_20260907/natprep_packed).
- [Staging audit (11 files)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b09257af799e380b76fb3e8979777578821a3303/issue1739_natural100k_20260906/cpu_preservation_20260907/natstage_packed).

The source census compared every current file name and hash under each original
CPU run root and its task-specific launcher logs, not just file counts.
Preparation: 23,949 files, comprising 22,205 exact evidence copies, 14 explicitly
labeled credential-scrubbed public audit copies, and 1,730 pinned input-cache
files. Staging: 9,077 files, comprising 1,309 exact evidence copies and 7,768
pinned input-cache files. Cache receipts and hashes are preserved; original
upstream and reused scientific inputs are reconstructible from the pinned
sources recorded by the inventory. No generated Qwen answers existed on either
CPU machine, and no scientific output was declared discardable.

The public audit copies are not model inputs or resume checkpoints. Their
inventory retains original hashes, redaction counts, and ordered byte-fragment
reconstruction information. Actual retained generic prompts were never
redacted. The successful prepared store was independently reloaded and checked
to contain 110,000 distinct original source IDs, all first-role-user records
with the no-recombination assertion. It remains at prepared revision
`195b718a44d0a4799c8f3802d29783b8214b96f4`; the exclusion audit remains at
`c7249d2a54e36e03e5c229bc53d0123fc262f57c`.

The canonical `verify_uploads.check_outroot_residue` checks were run on the VM
using complete pod-side listings and the respective leg-specific HF prefixes.
Both returned OK. Their `content-verified=0` field refers only to the Git-only
matching arm; the separate full-path HF content checks above verified all 315
and 11 files, including files exempted or sharing basenames in the canonical
name-only check. The source census additionally verified packed members and
the exact original out-root name sets.

Recovery notes: the initial un-packed preparation-audit upload encountered
HTTP 504/503 on a 22,264-file commit. Only that uploader was stopped; the local
evidence was retained and repacked losslessly to 315 files. Initial canonical
residue probes on the CPU pods triggered slow full-tree Git size queries on
partial clones; only those audit-probe process groups were stopped, then the
same canonical check was run successfully against the VM's complete clone.
These recoveries did not stop or change the GPU experiment.

Completion receipts are committed here because an upload cannot attest to its
own later completion. The staging machine additionally remains needed until
the fixed-input transfer's destination hash check passes; its audit completion
alone is not authorization to terminate it during that transfer.

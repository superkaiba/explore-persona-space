# stage_hub_prefix's dest is a MIRROR ROOT, not the final consumed path

`hub.stage_hub_prefix(repo, prefix, dest_dir)` lands every file at
`dest_dir/<repo-relative path>` (the documented VERBATIM PREFIX MIRROR, #1402).
Passing the FINAL consumed directory as `dest_dir` nests the whole hub prefix
under it (`.../corpus/issue1092_realistic_crossing/corpus/manifest.jsonl`), and
the bug stays invisible until the staging path first runs on a machine where
the store does NOT pre-exist (a fresh pod/GCE clone) — the #1774
att-20260729-033609 P0 crash: the restage "succeeded", then the post-restage
missing re-check re-raised FileNotFoundError, burning a GCE launch cycle.

**Rule:** pass a mirror ROOT satisfying `root / <hub prefix> == <consumed path>`
(assert the arithmetic — #1774 uses `_mirror_root()` with a leaf-name assert),
or stage into a sibling scratch root and `rename` the mirrored leaf into the
consumed layout (`issue1774_aggregate._stage_if_missing`). Either way, run the
artifact-reuse check (h)(iv) probe BEFORE production: one REAL 1-file (or small
prefix) staging into an empty root + the CONSUMER's own open against the staged
tree. `stage_hub_file` (exact dest path per file) does not have this trap.

Worked fix: #1774 commit 6948bdd1fe60c73b63e10b1ec04e633083c91c63; pins in
`tests/test_issue1774_round3.py`.

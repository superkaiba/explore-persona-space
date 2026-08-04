---
name: out-arg-file-vs-dir-kind
description: An out-path arg's FILE-vs-DIRECTORY kind is part of the call contract — passing a deliverable FILE path as a script's out DIRECTORY arg misnests outputs one level deep while every .exists()/[[ -e ]] completeness check passes on the misnested shape; crash surfaces only at upload (issue #1776 crash-fix cycle 5)
type: feedback
---

Passing a deliverable FILE path where a script expects its out DIRECTORY makes
`mkdir(parents=True)` happily create a directory NAMED like the file and the
real outputs land one level too deep (`.../x.json/x.json`). The failure is
SILENT through the whole phase: `.exists()` is satisfied by a directory,
`[[ -e ]]` upload-list guards pass directories, and a completeness assert of
the form `(out_dir / "x.json").exists()` passes BY COINCIDENCE of the
misnesting — only `CommitOperationAdd`'s is_file check finally crashed, a full
phase later (#1776 p3_upload; the same block also carried a #825-class
upload-list entry pointing at a never-written path, silently `-e`-skipped).

**Rules:** (i) treat FILE-vs-DIRECTORY kind as part of every out-arg contract —
when composing a dispatcher invocation, check what the script DOES with the arg
(mkdir → directory; open/write → file), and class-sweep every phase's out-path
args once (#1776 audited 33 surfaces); (ii) completeness/staleness checks on
deliverables use `is_file()` / `[[ -f ]]`, never `.exists()` / `[[ -e ]]`
(a directory satisfies the latter); (iii) scripts should REFUSE file-shaped
out-dir args at argparse time (endswith(".json") class guard); (iv) any repair
branch that relocates a misnested artifact is EXACT-SHAPE-GUARDED and
relocates (never deletes) — fail-loud on foreign shapes.

(Incident #1776 crash-fix cycle 5, pod-1776 p3_upload, 2026-07-29: fix commit
`f313eacbd565683ca50f47427c548115c8ca6def`; rollout text had already uploaded
+ verified — only the tensor/summary batch was blocked.)

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [out-arg FILE-vs-DIR kind](feedback_out_arg_file_vs_dir_kind.md) — a file path passed as an out-dir arg misnests deliverables while .exists()/-e checks pass; use is_file/-f + argparse kind guards + class-sweep out args (#1776 c5)

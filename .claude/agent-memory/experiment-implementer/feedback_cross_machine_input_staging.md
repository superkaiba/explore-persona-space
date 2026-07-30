---
name: cross-machine input staging for git-clone lanes
description: VM-produced inputs consumed by a GCE/SLURM phase must be HF-uploaded by the producer AND launcher-staged by the consumer; declare every cross-phase read in off_pod_phases
type: feedback
---

A git-clone lane (GCE `git clone --depth 1 --branch issue-<N>`, SLURM materialize) stages NO `data/`
(gitignored). #1773 Pass B died at input load (`FileNotFoundError: .../selection/inverted_index.npz`)
because Pass A's VM-side selection outputs were never uploaded to HF and the launcher had no staging
step — one full GCE provision burned (att-20260729-010419).

**Why:** the plan's off_pod_phases block declared the FINAL assembly's reads but omitted the
intermediate passA→passB read; the tiny-real single-machine smoke cannot catch cross-machine seams.

**How to apply:** when implementing any multi-phase pipeline whose phases run on different machines:
(1) every producing phase ends with a fail-loud bulk `upload_folder` of its outputs to the issue HF
prefix; (2) every consuming launcher stages missing inputs via scoped `list_repo_tree` + per-file
download (never `snapshot_download` on the ~1M-file data repo), logging a `[stage] <input> staged: N
files` line usable as a crash-fix fix-engaged signal; (3) add a 1-file staged-layout probe smoke leg
opening the staged file with the consumer's own loader (artifact-reuse (h)(iv)).

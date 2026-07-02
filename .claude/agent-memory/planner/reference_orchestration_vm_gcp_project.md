---
name: Orchestration VM lives in introsp-experiments, NOT the EPS GPU project
description: The CPU orchestration VM and the GPU pool are in DIFFERENT GCP projects with DIFFERENT credential boundaries — load-bearing for any VM-disk / VM-infra task
type: reference
---

The CPU orchestration VM (where all `/issue` sessions + the workflow run) is
`cia-benchmark-vm`, project **`introsp-experiments`**, zone `us-central1-a`,
machine `e2-standard-32`, single 500 GB `pd-balanced` boot disk (`/dev/sda1` →
`/`, ext4). It is NOT in `eps-persona-gpu-jun2026` (that project holds the
ephemeral GPU instances `eps-issue-*` only).

**Credential boundary (the trap):** the active `eps-gcp` gcloud config uses the
`eps-router@eps-persona-gpu-jun2026.iam.gserviceaccount.com` SA, which CANNOT
access `introsp-experiments` (describe/attach on the VM errors "resource not
found in eps-persona-gpu-jun2026"). The account that CAN do compute ops on the
VM's project is **`thomasjiralerspong@gmail.com` (`roles/owner` on
`introsp-experiments`)** — run VM/disk gcloud ops with
`--account=thomasjiralerspong@gmail.com --project=introsp-experiments`. The
gmail DRS block is on *quota-preference filing*, NOT on compute resource
create/attach.

**Disk-filler reality (2026-06-27):** `.claude/worktrees/` = 250 GB is the
dominant unbounded grower (per-issue `data/issue_<N>/{hf_dl,g*_dl,store}` caches
+ ~11-16 GB venvs land INSIDE each worktree, not in repo-root `data/` which is
only 475 MB). `eval_results/` (14 GB) is git-TRACKED → stays on the boot disk.
`/home` is on `/` (single fs, no separate mount).

**Why it matters:** any plan that "attaches a GCP disk to the EPS VM" or
inspects/modifies the VM via gcloud must use the gmail/introsp-experiments
credentials, and the GPU-project janitor (`gcp_audit.py`, scoped to
`eps-persona-gpu-jun2026`) will never see/reap a disk created in
`introsp-experiments`. Surfaced while planning #681 (bounded-by-construction VM
disk isolation).

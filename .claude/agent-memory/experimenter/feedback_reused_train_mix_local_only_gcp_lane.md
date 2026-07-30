---
name: Reused parent train-mix is local-only; GCP lane cannot reach it
description: A reused parent-task train mix that was never uploaded to HF cannot reach a GCP-lane VM (git-clone-only) — fail the input-data gate before launch
type: feedback
---

When a task REUSES a parent issue's training mix (the single-variable-change
default — e.g. #734 reuses #664's `mk_librarian_contra_d1_seed42.jsonl` so
the only deliberate variable is the base model), STAT-VERIFY the mix is
reachable on the launch lane BEFORE launch, and trace WHERE the dispatcher
reads it from — not just whether the brief enumerated it.

**Why:** #734 (2026-06-29). The dispatcher hard-asserts the parent train
mix at `<repo>/data/issue_664/train/marker/<key>.jsonl` (`__file__`-anchored
`DATA_ROOT`, ignores `REPO_ROOT`/`WORKLOAD_ROOT`) with NO HF-fetch fallback.
The mix was (1) absent on the VM, (2) NOT on the HF data repo (the parent
uploaded only baseline/eval artifacts, never the train mixes), and (3)
unreachable on the GCP lane because the GCE startup script is git-clone-only
(`backends/gcp.py` clones the branch + `reset --hard` + runs the workload;
it does NOT rsync `data/`). The only producer was the parent's `--phase p0`
GPU+elicitation+judge pipeline — not a free deterministic stage — so the mix
could neither be staged inline nor fetched. Launching `--phase all` would
have crashed phase2 (a registered deliverable) mid-run at degraded coverage.

**How to apply:**
- For ANY reused parent train mix / locally-built input the dispatcher reads
  from `<repo>/data/...` (NOT from HF): confirm it is on HF
  (`HfApi().file_exists(...)` / scoped `list_repo_tree(path_in_repo=...)` —
  bare data-repo `list_repo_files` times out, #833)
  OR pre-staged on the lane's persistent disk. On the GCP lane, local presence
  at the VM repo root is NOT sufficient — `data/` is git-clone-only, never
  rsynced (sibling memory `feedback_gcp_lane_git_clone_only_data.md`).
- The reused PARENT'S uploads are baseline/eval/adapters by default; the
  train MIX JSONLs are frequently never uploaded. Don't assume "reuse =
  fetchable". Grep the dispatcher for the read path and probe the data repo
  for that exact basename (`HfApi().file_exists` / scoped `list_repo_tree` —
  bare `list_repo_files` times out there, #833) before trusting reuse.
- A mix whose only producer is the parent's GPU `--phase p0` (elicitation +
  judge) is NOT a free `--no-generate` stage — it cannot be rebuilt inline at
  launch. Fail `infra reason: planned-input-data-missing-on-pod`; remediation
  is an implementer change (upload the mix to HF + wire a sha-pinned
  `hf_hub_download` fallback into the train fn, mirroring the adapter
  download) OR pin `backend: runpod` and pre-stage on the persistent pod.
- Trace which phases actually need the missing data: in #734 the HEADLINE
  phase1 + the phase1p5 gate were self-contained (HF adapters + in-repo
  code-defined question batteries); only phase2 needed the mix. A partial-
  coverage launch is still a degraded-coverage launch when the missing phase
  is a registered deliverable — refuse.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Reused parent train-mix is local-only; GCP lane can't reach it](feedback_reused_train_mix_local_only_gcp_lane.md) — a REUSED parent train mix (single-variable default) is often never uploaded to HF; on the git-clone-only GCP lane it can't reach the VM and has no inline rebuild (parent `--phase p0` is GPU+judge). Trace the dispatcher's `data/...` read path + `list_repo_files` the basename before trusting reuse; fail `infra planned-input-data-missing-on-pod` — #734

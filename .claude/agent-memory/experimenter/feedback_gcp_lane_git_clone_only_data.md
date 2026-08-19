---
name: GCP lane is git-clone-only — local data/ doesn't reach the VM
description: ROLLBACK-ONLY since #2028 (GCP provisioning DISABLED — an explicit `backend: gcp` pin raises GcpDisabledError; auto is runpod-first since #2054). If the lane is ever re-enabled — before launching on `--backend gcp`, verify each hard-required input is git-tracked or HF-mirrored with a fetch fallback. Local presence at the VM repo root is not sufficient — the GCE startup script clones from origin and does NOT rsync data/.
type: feedback
---

> **#2028 scope banner — this entire memory is ROLLBACK-ONLY.** GCP
> provisioning is DISABLED (`GCP_PROVISIONING_DISABLED = True` in
> `backends/router.py`; an explicit `backend: gcp` pin raises the typed
> `GcpDisabledError`), so no fresh dispatch can reach the lane this memory
> gates. The underlying fact (the GCE startup script `git clone`s and does
> NOT rsync `data/`) stays true and becomes load-bearing again the moment
> the constant is flipped back.

The GCP lane (`backends/gcp.py` startup script) runs the workload from a fresh `git clone --branch <branch>` of the repo. It does NOT rsync the local `data/` directory (only the SLURM and RunPod lanes carry the worktree's untracked content). Any dispatcher input that lives ONLY as an untracked file inside `data/` — typically a parent-task artifact like a role list or instruction pool — will be ABSENT on the GCE VM clone and the workload will crash with `FileNotFoundError` seconds into the run, burning a full GCP instance cycle.

**Why:** the experimenter pre-launch input-data completeness gate (Step 4 in `.claude/agents/experimenter.md` § Before Running) traditionally stat-checked files at the VM repo root, which silently confirms local presence rather than VM reachability. On the SLURM/RunPod lanes this was sufficient (worktree rsync); on GCP it is not. Closed on 2026-06-13 when task #634's smoke dispatch was blocked at the gate after the orchestrator's first pass missed that `data/assistant_axis/{role_list.json, extraction_questions.jsonl, instructions/}` (parent #368's 275-role pool, 2.2 MB) was untracked + locally-only.

**How to apply:** when the chosen backend is GCP (`--backend gcp` under a deliberate #2028 rollback ONLY — since #2054 the `auto` order is runpod-first with NO gcp rung, `DEFAULT_AUTO_LANE_ORDER = ("runpod", "fellows", "nibi", "fir", "mila")`), for each hard-required input under `data/`, run BOTH checks:
1. `git -C <worktree> ls-tree -r origin/<branch> -- <path>` — confirms the file rides the GCE clone.
2. If absent from git: confirm an HF mirror exists AND the dispatcher has a `hf_hub_download` fallback for that path BEFORE launch. Local file presence is NOT sufficient.

If neither check passes, refuse the launch and surface the gap: either (a) commit the file to the issue branch (the `data/canonical_persona_pool/` + `data/assistant_axis/` `.gitignore` precedent — re-include via `!data/<subdir>/`), or (b) mirror to HF + wire a fetch fallback. The fix is short; catching it at the gate beats a 30-second GPU crash.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [GCP lane is git-clone-only — local data/ doesn't reach the VM](feedback_gcp_lane_git_clone_only_data.md) — ROLLBACK-ONLY since #2028 (`backend: gcp` is REFUSED — `GcpDisabledError`): if the lane is ever re-enabled, verify each hard-required `data/` input is git-tracked OR HF-mirrored with a fetch fallback; local presence at VM repo root is NOT sufficient (the GCE startup script does not rsync data/) — #634

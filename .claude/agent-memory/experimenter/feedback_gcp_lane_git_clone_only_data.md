---
name: GCP lane is git-clone-only — local data/ doesn't reach the VM
description: Before launching on `--backend gcp` (or auto when gcp is the chosen lane), verify each hard-required input is git-tracked or HF-mirrored with a fetch fallback. Local presence at the VM repo root is not sufficient — the GCE startup script clones from origin and does NOT rsync data/.
type: feedback
---

The GCP lane (`backends/gcp.py` startup script) runs the workload from a fresh `git clone --branch <branch>` of the repo. It does NOT rsync the local `data/` directory (only the SLURM and RunPod lanes carry the worktree's untracked content). Any dispatcher input that lives ONLY as an untracked file inside `data/` — typically a parent-task artifact like a role list or instruction pool — will be ABSENT on the GCE VM clone and the workload will crash with `FileNotFoundError` seconds into the run, burning a full GCP instance cycle.

**Why:** the experimenter pre-launch input-data completeness gate (Step 4 in `.claude/agents/experimenter.md` § Before Running) traditionally stat-checked files at the VM repo root, which silently confirms local presence rather than VM reachability. On the SLURM/RunPod lanes this was sufficient (worktree rsync); on GCP it is not. Closed on 2026-06-13 when task #634's smoke dispatch was blocked at the gate after the orchestrator's first pass missed that `data/assistant_axis/{role_list.json, extraction_questions.jsonl, instructions/}` (parent #368's 275-role pool, 2.2 MB) was untracked + locally-only.

**How to apply:** when the chosen backend is GCP (`--backend gcp`, or `auto` resolving to GCP — the standing default), for each hard-required input under `data/`, run BOTH checks:
1. `git -C <worktree> ls-tree -r origin/<branch> -- <path>` — confirms the file rides the GCE clone.
2. If absent from git: confirm an HF mirror exists AND the dispatcher has a `hf_hub_download` fallback for that path BEFORE launch. Local file presence is NOT sufficient.

If neither check passes, refuse the launch and surface the gap: either (a) commit the file to the issue branch (the `data/canonical_persona_pool/` + `data/assistant_axis/` `.gitignore` precedent — re-include via `!data/<subdir>/`), or (b) mirror to HF + wire a fetch fallback. The fix is short; catching it at the gate beats a 30-second GPU crash.

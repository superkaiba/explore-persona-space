---
title: Add fellows Slurm cluster (charmander) as backend lane preferred over GCP/RunPod
kind: infra
tags: []
created_at: '2026-07-23T02:09:15Z'
has_clean_result: false
origin_prompt: add it to our workflow as another option over runpod an GCP
workflow: v1
---
# Add the Anthropic fellows Slurm cluster (charmander) as a backend router lane preferred over GCP and RunPod

## Overview / Motivation

User directive (2026-07-22, verbatim): "add it to our workflow as another option over runpod an GCP" — following a session that established working SSH access to both fellows clusters. The fellows cluster is free H200 capacity (no credit meter, no dollar spend) with full internet on compute nodes, so it should sit ABOVE the GCP lane and the RunPod terminal rung in the auto lane order.

## Goal

Add a `fellows` (charmander) SLURM lane to the unified backend router (`src/explore_persona_space/backends/`) and place it ahead of `gcp` in `DEFAULT_AUTO_LANE_ORDER`, so `auto` dispatches try the fellows cluster first, fall through to GCP on capacity/queue miss, and reach RunPod only as today's last resort. Reuse the existing SLURM lane machinery (`backends/slurm.py`, the nibi/fir/mila pattern) — this is a new lane config + cluster-specific constraints, not a new backend class.

## Access (established + verified 2026-07-22 — do not re-derive)

- SSH alias `charmander` (in `~/.ssh/clusters.config`, Included from `~/.ssh/config`): `superkaiba@213.181.104.162:16869`, key `~/.ssh/fellows-runpod-ed25519`, = cluster "cluster-EUR-IS" node-2 (RunPod pod `cluster-EUR-IS-pod-2`). Verified working end-to-end from this VM (`ssh charmander hostname` → `node-2`).
- `bulbasaur` (= "Anthropic 2" cluster node-2, `superkaiba@198.145.108.21:11231`) also works for SSH, BUT the user's Slurm association only allows the `-eur` QoS family (`dev-eur,high-eur,low-eur,normal-eur`, DefaultQOS `low-eur`) and bulbasaur's queue runs the non-eur set exclusively — so **charmander is the batch home; the lane targets charmander only** (bulbasaur access is a manual/dev affordance, not a lane).
- Endpoint re-derivation when RunPod remaps IP/port: RunPod GraphQL with the regular `RUNPOD_API_KEY` + `X-Team-Id: cm8ipuyys0004l108gb23hody` (the fellows org "Anthropic Safety Research" is the SAME team as EPS) → `query { myself { pods { name runtime { ports { ip publicPort privatePort isIpPublic } } } } }` → pod name `cluster-EUR-IS-pod-2` → port with `privatePort: 22`, `isIpPublic: true`. Consider a small helper/preflight step that re-derives on SSH failure rather than failing the lane.
- `/home` is per-node LOCAL on this cluster (verified). Everything cross-node lives on `/workspace-vast/superkaiba/` (VAST, mounted on all nodes). One-time bootstrap script exists at `~/fellows-cluster/bootstrap_fellows_cluster.sh` on the VM (VAST dirs, HF env, uv on VAST, py3.11 venv, slurm aliases) — not yet run; the lane's first-use bootstrap should run or subsume it.

## QoS / partition mapping (from live `sacctmgr` + the fellows handbook)

- `low-eur` — DEFAULT when unspecified. Priority 10000, UNLIMITED GPUs, preemptible (~3-min SIGTERM grace, auto-requeue), 14d max wall. Use for cheap/sweep work ONLY where the launched script checkpoints and resumes cleanly.
- `high-eur` — priority 100000, preempts low/normal, NOT preemptible, `gres/gpu=16` per-user cap, 7d max wall. Use for runs that must not die.
- `dev-eur` — srun-interactive ONLY (sbatch rejected), never used by the lane.
- Partitions: submit batch to `general` (high-eur) or `general,overflow` (low-eur). `--time` is REQUIRED (defaults to 24h otherwise).
- Lane policy suggestion (planner to finalize): jobs route `high-eur` by default (non-preemptible, matches the other lanes' semantics); an explicit opt-in flag or small/sweep jobs may use `low-eur` when the workload is checkpoint-resumable.

## Hard cluster rules the lane MUST encode (each has caused real multi-user outages there)

1. NEVER pass any `--export=` to sbatch/srun (crashes slurmd on this cluster; pass env via the script body).
2. NEVER `--mem=0`; request ~128G per GPU (cap ~250G/GPU).
3. NEVER set `CUDA_VISIBLE_DEVICES` (Slurm exports it; parse-then-slice only if splitting).
4. NEVER submit sbatch in a shell loop — use `#SBATCH --array=0-N` for sweeps (also crashes slurmd).
5. Multi-GPU NCCL needs `export NCCL_SOCKET_IFNAME="=vxlan0"` and `NCCL_NVLS_ENABLE=0` in the job script body.
6. `export HF_HOME=/workspace-vast/pretrained_ckpts` (shared cohort cache) + `HF_TOKEN_PATH=/workspace-vast/superkaiba/.cache/huggingface/token`; never let caches land on node-local `/home` or `/tmp`.
7. Job scripts launching accelerate/vLLM/torchrun carry a `cleanup() { kill -TERM -$$; wait; }` trap on SIGTERM/SIGINT/SIGQUIT (orphaned vLLM workers brick nodes there).
8. Job names include the user (e.g. `eps-issue-<N>-superkaiba`); cancel self-scoped (`scancel -u superkaiba --name=...`), never bare IDs.

## Scope / surfaces

- `src/explore_persona_space/backends/` (router lane order, slurm lane config; follow the existing nibi/fir/mila lane shape), `scripts/dispatch_issue.py` if lane names are surfaced there, tests under `tests/test_router*.py` / `tests/test_slurm_*.py`.
- Update the CLAUDE.md § Compute backends lane-order prose + the `tests/test_router.py` lane-order invariants to match the new order (fellows first, then gcp → nibi → fir → mila → RunPod terminal rung unchanged).
- The existing SLURM-lane caveats apply (git-clone-only staging — no VM-local `data/` on the lane; venv-extras handling per the existing lanes).

## Constraints / invariants

- RunPod stays the LAST-resort terminal rung; GCP ladder semantics unchanged; a fellows capacity/queue miss falls through to GCP exactly like a GCP rung capacity miss falls through today.
- Fellows-lane jobs must respect the 16-GPU `high-eur` cap; wider requests route past the lane to GCP wide rungs.
- Cluster etiquette: large sweeps (>~16 GPUs aggregate) on the fellows cluster need a human announcement in the fellows coordination channel — the lane should surface (not silently launch) such sweeps, e.g. cap lane-eligible width and route wider work to GCP.
- 0 GPU-h plan (code change only). Full pytest workflow-invariant suite must stay green.

## Provenance

- Origin: user chat directive 2026-07-22 in the fellows-cluster SSH debugging session ("Fellows RunPod SSH debug (bulbasaur)" Happy session). Access details persisted in `~/.claude/projects/-home-thomasjiralerspong-explore-persona-space/memory/project_fellows_runpod_cluster.md` and `~/.ssh/clusters.config` comments.

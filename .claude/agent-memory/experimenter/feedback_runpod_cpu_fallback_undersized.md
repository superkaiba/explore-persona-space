---
name: RunPod cpu-mid fallback is undersized vs GCP cpu-mid (disk AND RAM)
description: RunPod CPU fallback (cpu3c-8-16) has 50G container overlay + 16 GB RAM vs the GCP cpu-mid the plans size against (e2-standard-8, 32 GB, --boot-disk-gb 80); size the plan's peak disk/RAM against the ACTUAL instance before launching a CPU-stage workload
type: feedback
---

On the `cpu-mid` RunPod fallback lane (`deployCpuPod`, instance `cpu3c-8-16`),
the pod gets a **50 GB container overlay** (no /workspace volume — `df` shows
`/workspace` riding `overlay 50G`) and **16 GB RAM**. Plans size the CPU stage
against GCP cpu-mid (`e2-standard-8`: 32 GB RAM, `--boot-disk-gb 80`), so a
plan-conforming workload can be deterministically infeasible on the fallback.

**Why:** #958 CPU stage (2026-07-04): staged store+corpus = 27.27 GB (measured
via scoped `list_repo_tree` sums), + ~15 GB columnar fit_cache + 5.6 GB maps +
~13 GB repo residue after maximal cleanup ⇒ ~61-63 GB peak ≫ 50 GB capacity
(ENOSPC would fire ~3h in, mid-fits, after the staging spend). RAM peak ~14 GB
on a 16 GB box (plan computed against 32 GB). Launch was refused at the
pre-launch gate; `epm:failure infra reason: provision-undersized-for-plan`.

**How to apply:** whenever the brief's host is a RunPod CPU pod, BEFORE
launching: (1) `df -h /workspace` — if it shows `overlay` (no volume), capacity
is the container disk; (2) sum the REAL remote sizes of every HF-staged input
prefix (scoped `list_repo_tree`, never the brief's estimate); (3) add derived
caches (read the fit script's cache builder for its position/row subset) +
uploads-in-flight + repo residue; (4) compare against the plan §9 disk/RAM
sizing. Shortfall ⇒ `epm:failure v1` `failure_class: infra`,
`reason: provision-undersized-for-plan` — never launch into a deterministic
ENOSPC. Useful reclaim levers when the margin is close: `uv cache clean` (the
uv cache is NOT hardlinked into `.venv` on these pods — combined `du` proves
it) and sparse-checkout eviction of other issues' committed `eval_results/`
(safe on an ephemeral pod clone; the run writes only untracked files there).

---
name: Detached-spawn launch scripts cannot be &&-chained into sequential waves
description: A fan-out script that setsid-detaches its shards (reparented to init) and exits after the spawn loop makes "wave2 && wave3 && wave4" run ALL waves concurrently — verify a wait/poll loop exists in the script SOURCE before chaining waves in one wrapper (#1738)
type: feedback
---

A launch/fan-out script that spawns per-GPU shards via `setsid nohup ... &`
(often inside `bash -c "... & echo \$!"` to capture the child pid) leaves the
shards reparented to pid 1 and EXITS right after the spawn loop — bash `wait`
cannot even apply to reparented children. Chaining waves in one wrapper
(`wave2 && wave3 && wave4`) then fans out EVERY wave near-simultaneously:
N-waves × GPUS shards co-locate (issue #1738: 16 shards over 8 GPUs within
seconds, wave 4 queued for 3-per-GPU), and the doubled concurrent FUSE load
plausibly wedged the pod's MooseFS mount (statfs hang, request_wait_answer
waiters, GPUs pinned at 0 MiB — the #779 wedge signature, pod swap required).

**Why:** the brief claimed "the launcher waits on its shard pids before
exiting, so the chain is sequential by construction" — false; the script's
spawn loop ends with an informational echo and exits 0. Briefs paraphrase;
the script SOURCE is ground truth.

**How to apply:** before launching any chained multi-wave workload cmd, grep
the wave script for a `wait` / pid-poll loop AFTER its spawn loop (`sed -n`
the tail of the script; `git show origin/<branch>:<script>` works from the VM
when pod reads are unavailable). If the script exits after spawning, do NOT
chain: either bounce to experiment-implementer to add a poll-until-dead loop
on the shard pidfiles, or gate each wave in the wrapper by polling the
previous wave's shard pids, or dispatch one wave per round. Same
verification family as the "brief flags drift from argparse" check — trust
the script, not the brief's paraphrase of its blocking behavior.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Detached-spawn launchers cannot be &&-chained into waves](feedback_detached_spawn_launcher_cannot_chain_waves.md) — a fan-out script that setsid-detaches shards (reparented to init) exits after its spawn loop; chaining wave2 && wave3 && wave4 runs ALL waves concurrently (16 shards/8 GPUs) and can wedge the MooseFS mount; grep the script SOURCE for a wait/poll loop before chaining — the brief's "waits on its shard pids" claim is a paraphrase, not ground truth (#1738)

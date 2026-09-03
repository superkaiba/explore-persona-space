---
name: proc-session-scoping-test-rigs
description: Testing sid-scoped process reaps — /proc/environ reflects the CURRENT exec image (poll argv0, not a substring), env -i children still carry bash's PWD/SHLVL/_, setsid -w forwards rc for self-promoting launchers
metadata:
  type: reference
---

Mechanics from the #2658 sid-reap round (2026-09-03), for any test rig that
spawns real decoy processes to exercise session-id / process-scoping logic:

1. **Poll argv0, never a substring, before reading /proc/<pid>/environ.**
   `/proc/<pid>/environ` (and cmdline) reflect the CURRENT program image; a
   `Popen(["env","-i","bash","-c","exec -a NAME sleep N"])` chain passes
   through 2 intermediate images whose ARGV also contains NAME while environ
   is still the full inherited one. Wait until
   `cmdline.split(b"\0")[0] == b"NAME"`, then read environ.
2. **`env -i` does not yield an EMPTY final environ** — bash re-exports its
   own bookkeeping (PWD, SHLVL, _) into the exec'd child. Assert shape
   (<= ~4 entries, no project keys), not `== b""`. This happens to match
   vLLM setproctitle-scrubbed workers exactly (measured 4 entries).
3. **Self-promoting launcher pattern:** a sid-keyed reap needs
   `sid == $$`; re-exec via `exec setsid -w bash "$0" "$@"` (the `-w` is
   load-bearing — without it the forked child detaches and the caller reads
   rc=0 unconditionally), env-sentinel loop guard, exit 2 if still not
   leader. Production `setsid nohup bash ...` paths are already leaders (a
   no-op self-check); plain `subprocess.run` test invocations exercise the
   re-exec branch for free.
4. **Enumerate session members via a pure-bash /proc glob** (expanded once,
   builtins-only sid parse from `/proc/pid/stat` after the last `)`) so the
   reap code spawns no ps/awk children that could enter its own candidate
   set; re-read each candidate's sid at kill time to drop reused pids.

Why recorded: vLLM `setproctitle("VLLM::EngineCore")` DESTROYS worker
environs (environ-tag kill scopes are structurally dead) and renames only
half the worker processes (name-keyed pgrep misses the python3 half) — the
sid predicate is the one that survived measurement. See
`scripts/issue2658_p1_launch.sh` gpu_reclaim block for the full recorded
evidence. Related: [[lean-session-waits-and-tmp-collisions]].

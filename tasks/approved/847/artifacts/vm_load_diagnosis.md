# VM load-186 root-cause diagnosis — 2026-07-02 00:00–00:25 PT

Host: cia-benchmark-vm, 32 cores, 125 GiB RAM, **no swap**. Load at probe time: 185.10 / 169.25 / 124.46. READ-ONLY diagnosis; nothing killed or modified.

## 1. Run-queue breakdown: pure CPU oversubscription, NOT IO/memory stall

- Thread states: **156 R (runnable), 1 D (uninterruptible)**, 8786 S, 206 I, 29 Z.
  The single D-state task was `issue779_scaling_grid.py` in `folio_wait_bit_common` (transient page-cache wait), gone on re-sample.
- PSI: **cpu some avg10 = 96.51%** (avg300 = 89.71) vs **io some avg10 = 0.01**, **memory some avg10 = 0.00**. CPU full = 0 (there is always something running — classic oversubscription, not a stall).
- vmstat: r = 183–196, b = 0–1, si/so = 0 (no swap exists), us ≈ 96%, wa = 0. Context switches ~77–83k/s.
- iostat sda: ~30 MB/s reads, %util 25–40, aqu-sz 1.4–2.4 → disk healthy. `/mnt/eps-data` idle. `/` at 82% (89 GB free).

**Conclusion Q1:** load ≈ number of runnable threads (~156–196). It is a CPU run queue, with zero IO or memory pressure at probe time.

## 2. Thread oversubscription: uncapped torch intra-op pools

Per-process at probe time (NLWP = total threads; "runnable" = R-state threads counted per PID):

| PID | cmd | NLWP | runnable | CPU% |
|---|---|---|---|---|
| 1130546 | run_823.py --phase 4 --smoke (instance 1) | 64 | **32** | 908 |
| 1138522 | run_823.py --phase 4 --smoke (instance 2) | 64 | **32** | 720 |
| 1186029 | run_823.py --phase 4 --smoke (instance 3) | 64 | **32** | 489 |
| 1150132 | issue-778 compare_inprocess.py | 32 | **32** | 399 |
| 1150683 | issue778_null_battery.py | 32 | **32** | 397 |
| 1204326 | issue779_scaling_grid.py --smoke | 63 | 10 | 20 |

- **Env caps: NONE.** `/proc/<pid>/environ` of all 6 heavy PIDs has zero `OMP_*`/`MKL_*`/`OPENBLAS_*`/`TORCH_*` vars. Torch therefore defaults intra-op threads = nproc = **32 per process**.
- 5 jobs × 32 runnable threads = **160 runnable threads on 32 cores ≈ the load of 185** (plus grid job + ~15 claude/node threads). Each job gets ~5–6 cores of real throughput while creating 32 load points — the "CPU% sums to ~30 cores but load is 186" discrepancy exactly.
- Repo-wide: only `scripts/issue779_layer_sweep.py:486` calls `torch.set_num_threads(args.n_threads)` (docstring targets 8). **No launch surface (scripts/, agents, issue SKILL) sets `OMP_NUM_THREADS`.**

## 3. The 3× run_823.py: accidental duplicate launches from ONE session

All three run **identical args** (`--phase 4 --smoke --skip-upload`) in the same worktree (`.claude/worktrees/issue-823`) — not seeds/arms.

- Instance 1: started 23:48:55, bash parent under PID 471275 — the **issue-823 autonomous session** (claude-agent-sdk proc, cwd = issue-823 worktree; env carries `CODEX_COMPANION_SESSION_ID`), command piped `| tail -40`.
- Instance 2: started 23:51:09, **same session 471275**, same command without tail — a re-run 2m14s later with instance 1 still alive.
- Instance 3: started 00:02:45, `uv run` parent **reparented to PPID 1** — its launching shell died (Bash-tool timeout/teardown killed the shell; the python child survived).

Context from `tasks/running/823/events.jsonl`: task #823 is at `running`, stage=implementing **round 3**; Codex round-1 code review FAILed with blocker `smoke-run-missing`, so the implement loop is (re)running the phase-4 smoke to produce evidence. Phase 4 (`phase4_ridge_refit`) is **by design a VM CPU phase** (script header: "CPU phases (0,1,2,4,5) run on the VM"), loads the ~6 GB pass_b bundle (`torch.load(..., mmap=True)`) and runs ridge math on CPU. Under the loaded box the smoke is slow → the Bash call times out / gets re-issued → **retry-without-kill piles up identical CPU-bound instances**, and timeouts orphan children instead of killing them. Side risk: all 3 write the same output paths (mutual overwrite).

Watcher/cron relaunch loop: ruled out — launches trace to the live session's shells, not `autonomous_session_watch`/cron, and there is no `epm:run-launched` churn.

## 4. Silent kills: earlyoom (userspace), NOT kernel OOM

- Kernel OOM today: **none**. Last kernel OOM was **Jun 26 03:37/03:45** (two `pytest` procs at ~101 GB and ~85 GB anon-RSS, session-2954 scope — the cgroup `oom_kill 2` counters are since-boot and refer to those).
- Tonight, **earlyoom** (PID 2703914, `-m 10 -s 10 --prefer '(^|/)(pytest|python3?)$' --avoid '...(node|claude|happy)...'`) fired a kill spree **23:45:47–23:48:19** when mem-avail hit 9.92% of 128.8 GB (no swap):
  - SIGTERM → `python3` **VmRSS 4801 MiB** (23:47:06, badness 991) and `python3` **7151 MiB** (23:47:53) — matches the "~4 GB workers died silently" report; SIGTERM terminates Python with no traceback.
  - SIGTERM → Claude Code procs ("2.1.198") at **15.1, 17.9, 16.8 GB RSS** (23:45–23:48) plus ~10 small python3s.
- Pressure source: fleet RSS stacking — several 15–18 GB claude sessions + 4–7 GB analysis jobs + 3× mmap'd 6 GB bundle. Free memory recovered to 17 GB free / 40 GB cache after the spree.

## 5. Ranked root causes

1. **(b) Thread oversubscription — dominant.** 5–6 uncapped torch CPU jobs × 32 intra-op threads = 160–190 runnable threads on 32 cores (~6×). This alone produces load ~186 and the 60s→64min per-layer slowdowns of `issue779_layer_sweep` (itself the only well-behaved job, capped at 8 threads, so it starves under everyone else's 32).
2. **(c) Duplicate-launch bug — amplifier.** 2 of the 3 run_823 instances are accidental (retry-without-kill after Bash timeouts, orphaning children to PPID 1). They add 64 excess runnable threads (~⅓ of the overload) plus 2 extra 6 GB bundle mappings.
3. **(d′) Memory ceiling → earlyoom kills — separate pathology, same collision.** No-swap box at <10% avail triggered earlyoom's `--prefer python` spree; this explains the silent worker deaths (userspace SIGTERM, not kernel/cgroup). Not a contributor to the load number (PSI-mem = 0 now).
4. (a) Legitimate fleet work (pytest suite, workflow_lint, a 2h-runtime `grep -rln 'main...HEAD'` at 35% CPU, ~10 sessions) is the background hum, ~3–5 cores — not the driver.

## 6. Highest-leverage fix

**Cap CPU threads at the workflow launch surfaces**: export `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8` (or `torch.set_num_threads(min(8, ...))`) for every VM-run analysis/smoke phase — the same 5 jobs would then post ~40 runnable threads instead of ~190, near-eliminating the pathology at zero throughput cost (each job currently realizes only ~5–6 cores anyway). Complements, in order: kill-before-relaunch (or `run_in_background` + poll instead of retrying timed-out foreground smokes; ensure process-group kill on timeout), and routing >15-min CPU phases to `cpu-small`/`cpu-mid` pods per the #747 rule. The no-swap + earlyoom config is a second, orthogonal knob (a small swap file or higher-headroom scheduling would stop the silent SIGTERM sprees).

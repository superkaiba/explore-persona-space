# Failure-class log patterns

> **Authoritative source: [`scripts/failure_classifier.py`](../../../scripts/failure_classifier.py).**
> This markdown file is a human-readable MIRROR of the regex list in
> that Python module. The `/issue` skill Step 7 shells out to the
> script (`uv run python scripts/failure_classifier.py --body - --log
> <path>`); it does NOT consult this markdown file at runtime. Keep
> the two in sync when extending; the Python module wins on conflict.

When `epm:failure` body lacks `failure_class:`, the script scans the
body + last 200 KB of the linked log against these patterns. Any match
→ route as `infra`. Otherwise → `code` (conservative).

**DataLoader-worker wrap special case.** torch's DataLoader catches
worker-side exceptions and re-raises them wrapped:

```
RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File ".../torch/utils/data/_utils/worker.py", ...
  File ".../src/explore_persona_space/train/sft.py", ...
RuntimeError: <our message>
```

The outer frames are always under `torch/` (worker.py, `_utils/`, ...),
so the generic library-traceback infra pattern would route an our-code
raise to `infra`. To prevent that: when the body matches
`Caught <Error> in DataLoader worker`, the classifier isolates the
text after `Original Traceback` and classifies on the WRAPPED block —
if it contains an our-code frame (`src/explore_persona_space/` or
`scripts/`), route as `code`; otherwise run the normal infra-pattern
scan on the wrapped text only (so a wrapped CUDA OOM still routes as
`infra`). Surfaced by /issue 480 (workflow-fix candidate).

**Co-located parallel-cell OOM special case.** A CUDA OOM is normally
transient infra (leaked process, fragmentation — respawn fixes it).
EXCEPT: when the torch OOM message lists **2+ sibling
`Process NNN has X GiB memory in use` entries** on the failing device:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB.
GPU 0 has a total capacity of 79.18 GiB of which 41.00 MiB is free.
Process 568053 has 50.74 GiB memory in use. Process 568050 has 14.72 GiB
memory in use. Process 568055 has 13.66 GiB memory in use.
```

Multiple sibling entries during a parallel fan-out mean the train cells
were CO-LOCATED on one physical GPU — a deterministic GPU-pinning bug in
the launch path (e.g. a per-process `--gpu` pin that is dead code in
that entry path), not transient infra. Respawning on verified-clean
GPUs hits the identical OOM. The classifier routes this as `code`
(regex `Process \d+ has [\d.]+ [KMG]iB memory in use`, count >= 2,
precedence just below the explicit `failure_class:` field). A SINGLE
sibling entry stays `infra` (one leaked process from a prior run —
kill + respawn is the right move). Surfaced by task #557 (2026-06-10):
attempt 1 was misdiagnosed as leaked-process infra and attempt 2 OOMed
identically on verified-clean GPUs.

**vLLM engine-init free-memory special case.** vLLM's engine init
raises:

```
ValueError: Free memory on device (10.50/79.18 GiB) on startup is less
than desired GPU memory utilization (0.9, 71.26 GiB). Decrease GPU
memory utilization or reduce GPU memory used by other processes.
```

This routes as `infra` — but on a RELAUNCH it usually means **orphaned
`VLLM::EngineCore` workers from a prior crashed run are still holding
the GPUs**, NOT a capacity problem. The workers' cmdline is just
`VLLM::EngineCore` (no script name), so `pgrep -f <script-name>` reads
clean while ~50 GB/GPU is held. Recovery is IN-PLACE: probe
`pgrep -af EngineCore` + `nvidia-smi
--query-compute-apps=pid,used_memory --format=csv`, kill the orphans
(`kill`, then `kill -9` survivors), confirm GPU memory is ~0, and
relaunch on the SAME pod — do this BEFORE any fresh-pod / capacity
reclassification. The named pattern also covers bodies that carry only
the final error line (no `vllm/` traceback frames), which previously
fell through to the conservative `code` default. See
`.claude/rules/gotchas.md` (crash-orphan EngineCore) +
`.claude/agents/experimenter.md` Pre-Launch step 9. Surfaced by task
#601 (2026-06-11): 4 orphaned EngineCore workers from a phase0 crash
held ~50 GB/GPU and OOMed the hot-fix relaunch until they were killed.

**Fan-out handshake-timeout mask (diagnosis-verification, workflow-fix
#1123; incident #1112).** When ALL units of a multi-unit vLLM fan-out
dump `RuntimeError: Did not receive response from front-end process
within 5 minutes` "simultaneously", the symptom is usually the MASK of
ONE unit crashing instantly: a driver that raises on the FIRST unit
failure abandons the sibling front-ends mid-engine-init, and their
orphaned EngineCores dump the handshake timeout ~5 minutes later into
every sibling log. The infra patterns (`Traceback.*vllm/`,
`Failed to initialize.*vllm`) match this symptom when vllm traceback
frames are present in the body/log (a bare one-line handshake symptom
routes `code` today), and an explicit `failure_class:` field composed
off the symptom wins routing precedence — so the gate is on the
DIAGNOSIS: before composing an `epm:failure` body for a fan-out
failure, sort the per-unit logs by first-error timestamp (or by file
size — the fast-crasher's log is tiny) and read the EARLIEST/SMALLEST
failing unit's traceback; a deterministic in-repo traceback there (not
a transient network error — an HF Hub 503 / `ConnectionError` during
downloads) routes `failure_class: code` (an infra respawn re-hits the
identical crash).
**Classifier behavior is UNCHANGED by this protocol** — a body-text
heuristic cannot distinguish the mask from a genuine all-units wedge
(the fan-out driver's own raise puts our-code frames in the driver log
either way), and the discriminating evidence lives in per-unit log
FILES the classifier's single `--log` path never sees. Full entry:
`.claude/rules/gotchas.md` (fan-out handshake-timeout mask);
long-form:
`.claude/agent-memory/experimenter/feedback_fanout_handshake_timeout_masks_single_unit_crash.md`.
Surfaced by task #1112 (2026-07-08): attempt 4 posted
`failure_class: infra` off the symptom; attempt 5 found the
1,458-byte attempt-4 crash log carrying the true deterministic
`FileNotFoundError`.

**Zombie-GPU-allocation stall reason (workflow-fix from #664).** On a
`status=stalled` poll tick the orchestrator forwards the poll JSON line's
machine-readable `stall_reason` field to the classifier via
`--stall-reason`. A known stall reason routes `infra` DIRECTLY — before
the regex scan — because a silent hang's log tail carries no traceback /
OOM line and would otherwise fall through to the conservative `code`
default. The sole known reason today is `vllm_worker_dead_zombie_gpu`: a
CUDA-worker PID died but still holds VRAM while the vLLM EngineCore main
process stays alive burning Python-overhead CPU (so the #518/#658
session-CPU-advancing override rescued the stall to `running` for a 60+
min silent hang). It is recoverable by an experimenter respawn (reap the
orphaned `VLLM::EngineCore` worker by EXACT PID, relaunch on the same
pod), NOT a code fix, so it belongs on the `infra` row. Since #826 the
detector fires only when EVERY workload log is stale past the effective
stall window (`max(EPM_ZOMBIE_VETO_FRESH_SEC=60, stall_sec)`) for 2
consecutive ticks — on host-PID-namespace containers nvidia-smi reports
host PIDs unresolvable in the container's `/proc`, so the bare signature
is false-positive on healthy runs (#816/#778); a fresh-log zombie flag is
a namespace artifact, not a hang — and (since #864) only when the
dead-in-/proc signature is namespace-informative: zero-resolvable compute
PIDs with live in-container `/dev/nvidia-uvm` holders (exact fd-target
match) are a PID-namespace artifact and are vetoed regardless of log
staleness (#813: a healthy ~29-min CPU-bound quiet stretch outlived the
#826 stale-log veto). The #864 veto ships default-OFF
(`EPM_ZOMBIE_NAMESPACE_VETO=1` arms it; read at poller import) per its
pre-merge live-pod gate — a cuInit'd parent/coordinator holding exact uvm
while absent from compute-apps would suppress a genuine total collapse if
armed unverified. The emit surface
is `scripts/poll_pipeline.py` + `scripts/backend_poll.py`; the
known-reason set is `STALL_REASON_INFRA` in `failure_classifier.py`; the
recovery brief + recipe references are SKILL.md Step 7 § "Zombie-GPU
stall recovery brief". Precedence: an explicit `failure_class:` field
still wins over `--stall-reason`.

**exit-137/143 / silent-`Killed` kill-source verification (workflow-fix
#902, incident #779 r9).** The infra pattern `OOM-killer|Killed\b`
matches a DELIBERATE SIGKILL's log line just as well as a kernel-OOM
one, and exit 137 (= 128+9) alone does not attribute the killer (an
exit-143 / SIGTERM death gets the same protocol — earlyoom's first
strike is SIGTERM). On the shared VM a process can be killed by at
least four sources: the kernel OOM killer, earlyoom (SIGTERM below 10%
MemAvailable, SIGKILL below 5% — see `.claude/rules/gotchas.md`,
earlyoom entry), systemd-oomd / cgroup-confined OOM (`memory.max`), or
a deliberate operator/PM/watcher kill. The `infra` ROUTING may still be
correct; what this protocol gates is the DIAGNOSIS CONTENT — before an
`epm:failure` body names OOM as the cause, and before any crash-fix
round is dispatched against a memory hypothesis, verify the kill
source. The four checks are read CONJUNCTIVELY — no single step
short-circuits the others:

1. **cgroup `memory.events` `oom_kill` DELTA over the run window.** The
   login-session cgroup scope hosts MANY fleet processes — absolute
   counters (`memory.peak`, a historical `oom_kill` count) do NOT
   attribute to your process. A counter that did not increment across
   the run window RULES OOM OUT; an INCREMENTING counter takes
   PRECEDENCE over step 2's floor read (cgroup-confined OOM kills at
   high host MemAvailable). If no run-start baseline was captured, the
   absolute counter attributes nothing — rely on step 3.
2. **MemAvailable floor over the run window.** A floor > 20 GB rules
   out GLOBAL memory-pressure kills only (kernel global OOM killer,
   earlyoom — floors ~10%/5% of the 128 GB total). It does NOT rule
   out cgroup-confined OOM (`memory.max`, systemd-oomd cgroup
   pressure); when step 1's delta increments, step 1 wins.
3. **Kill-line journals at the death timestamp.** `journalctl -u
   earlyoom` / `journalctl -u systemd-oomd` / `dmesg | grep -i oom`. The
   watcher's CPU-guard pass also pre-attributes earlyoom kills as
   `kind=earlyoom-kill` rows (with `attribution_status:`) in
   `.claude/cache/cpu-guard-events.jsonl`
   (`.claude/rules/background-automation.md`).
4. **A deliberate-stop record MATCHED to the death window.** Check the
   task's events.jsonl for a `deliberate-stop` breadcrumb — match on
   the LEADING structured token (a `note` BEGINNING `deliberate-stop `),
   e.g. `grep '"note": "deliberate-stop' "$(uv run python
   scripts/task.py find <N>)/events.jsonl"` — and treat it as
   attribution ONLY when it is TIME-PROXIMATE to the death (minutes,
   not "any prior note") AND its `target=`/`pid=` matches the dead
   process/session. A stale or non-matching breadcrumb is CONTEXTUAL
   evidence only, never attribution; note a `target=happy-session:<sid>`
   stop does NOT itself kill detached (`setsid`) workloads, which
   outlive their session by design. `spawn_session.py stop` auto-posts
   the breadcrumb for issue-mapped OPERATOR stops; WATCHER-driven stops
   post nothing here — check the watcher registries/sidecars under
   `~/.eps-autonomous/` for force-stop records instead. PM manual kills
   MUST post one BEFORE the kill (`.claude/agents/research-pm.md`
   § Autonomy rules). **Absence is NOT exculpatory:** manual kills,
   unmapped sessions, raw terminals, and non-PM killers can all leave
   no marker.

**Terminal disposition:** if steps 1-4 jointly fail to attribute the
killer, record `killer unknown` in the `epm:failure` body — do NOT name
OOM, and do NOT dispatch a crash-fix round against a memory hypothesis
on back-derived arithmetic alone. Additionally: sanity-check any
back-derived memory arithmetic against the code's ACTUAL allocation
sites before it drives a fix round. #779 r9 diagnosed kernel OOM from
shared-scope counters plus a hypothesized 1100-draw materialization
that did not exist in the code, and dispatched a crash-fix round
against a nonexistent bug — all three deaths were deliberate PM-session
SIGKILLs (MemAvailable floor 47 GB; the counter never incremented).
**Classifier behavior is UNCHANGED by this protocol** — it is a
diagnosis-verification step layered on top of the `infra` route, not a
routing change.

## Infra patterns (regex, case-insensitive)

```
CUDA out of memory
OOM-killer|Killed
No space left on device|ENOSPC|disk full
NCCL (timeout|error)
SSH connection refused|No route to host|Connection timed out
401 Unauthorized|gated repo
RuntimeError: CUDA error
Failed to initialize.*vllm
Free memory on device.*?is less than desired GPU memory utilization
Traceback.*\b(vllm|transformers|peft|trl|torch|xformers)/
```

## Code patterns (regex, case-insensitive)

These are NOT used for inference (the fallback only looks for infra).
Listed here for completeness of the experimenter agent's checklist:

```
Traceback.*\b(src/explore_persona_space|scripts)/
^AssertionError
^TypeError
^KeyError
```

## Adding a pattern

Edit `scripts/failure_classifier.py` (the runtime authority) AND mirror
the change in this file. The tests in `tests/test_failure_classifier.py`
must still pass — extend them with a fixture covering the new pattern.
The skill SKILL.md and agent specs cross-reference by path; no further
SKILL/agent edits needed. (Allowed under §10 plan deviations: implementer
can extend the pattern list without asking.)

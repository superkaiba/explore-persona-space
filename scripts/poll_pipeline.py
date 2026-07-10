"""poll_pipeline.py — one-tick poller for a running experiment pod.

Invoked by the `/issue` orchestrator's bg-Bash sleep-chain (see
`.claude/skills/issue/SKILL.md` Step 6d.2). Performs ONE poll then exits
— the orchestrator chains successive `Bash(sleep <interval> && uv run
python scripts/poll_pipeline.py ..., run_in_background=true)` calls and
is re-invoked by the harness when each bg Bash returns. `<interval>` is
the ``next_interval`` the PREVIOUS tick emitted in its JSON line
(adaptive bg-poll interval — see :func:`recommend_next_interval`), with
540s as the orchestrator-side fallback when the key is absent or
unparseable.

Why orchestrator-owned: subagents have ONE turn — they are NOT
auto-re-invoked when a bg Bash finishes. The orchestrator IS. See
`CLAUDE.md` § "Subagent vs orchestrator re-invocation semantics" and
the deprecated memory `feedback_subagent_sleep_chain.md` for context.

Per tick:

1. Drain pod-side sentinel files (`/workspace/logs/issue-<N>-*.json`,
   skipping `*.processed`). Each sentinel was written by a pod-side
   dispatcher that cannot shell out to `scripts/task.py` (CLAUDE.md
   "Pod-side code NEVER shells out" rule). The poller parses each
   sentinel, posts the carried `epm:<kind>` marker from the local VM
   via `task_workflow.post_event`, then renames the sentinel to
   `<path>.processed` so it posts exactly once — and the post itself is
   idempotent (#1084): each drain-posted event carries a `sentinel_fp`
   extra, so a poller killed/hung between post and rename replays
   rename-only on the next tick instead of duplicating the marker. If
   a sentinel carries a non-empty ``gate`` field, the poll returns
   ``status=gate`` with that
   gate name in the JSON output so the orchestrator parks at a user
   gate instead of continuing the polling loop.
2. SSH to the pod (one heredoc batching: PID liveness, log mtime, log tail).
3. Parse the latest `[phase=...]` line from the log tail.
4. If new milestone vs the cached previous phase, post `epm:progress`
   to the task's events.jsonl via the local-VM `task_workflow.post_event`
   library (NOT on the pod).
5. Decide status: `done` | `gate` | `stalled` | `dead` | `running`.
6. Print one JSON line summary to stdout. Exit 0 on successful poll
   regardless of `status`. Exit non-zero only on caller-error (bad args,
   library import failure).

Stall threshold: ALL of (a) `last_log_mtime_sec_ago > stall_sec`
(default 900s via ``DEFAULT_STALL_SEC``, overridable per-tick via
``--stall-sec`` CLI or ``EPM_POLL_STALL_SEC`` env var for workloads
with sparse log cadence, e.g. checkpoint-only logging at >15min
intervals; taken over BOTH the top-level log and the freshest cell
log), (b) every per-phase log under
``/workspace/logs/issue-<N>-*.log`` is also quiet for >stall_sec,
(c) every shard / repo-rooted phase log under
``/workspace/explore-persona-space/logs/issue_<N>{,_*}/*.log`` AND
every dispatcher per-job log under
``/workspace/explore-persona-space/eval_results/issue_<N>{,_*}/logs/*.log``
is also quiet for >stall_sec, (d) no issue-keyed OUTPUT artifact
(``eval_results/issue_<N>{,_*}/``, ``data/issue_<N>/``,
``data/issue<N>/`` under the repo root) was modified within stall_sec
(#1033), and (e) the GPUs are idle. Only when all five signals agree
does the poll declare `stalled`; any fresh log, any fresh output
artifact, OR a busy GPU keeps the run in `running`.

CPU-advancing override (#518): even with the stall conjunction met, a
launcher whose process session (`setsid` group) has accrued more
cumulative CPU since the previous tick is doing CPU-bound work and is
NOT stalled. The probe sums `time` across every process sharing the
launcher PID's SID via `ps -e -o sess=,time=` and persists the sample
to the local state file; on the next tick the delta is compared to a
small epsilon (`SESSION_CPU_ADVANCE_EPSILON_SECS`). If CPU advanced,
the verdict flips to `running`. If CPU is flat OR unknown (first tick
after launch, launcher dead, `ps` unavailable), the older arbiters
keep the verdict — fail-safe to the pre-#518 behavior. Incident: task
#518 scoring_syco phase, 2026-06-10 — a healthy CPU-bound aggregation
phase wrote nothing to the log for ~7.8h while the python child was
at 100% CPU; the poller falsely declared `stalled`.

Zombie-GPU-allocation override (#664): the CPU-advancing override above
has a blind spot — a hung vLLM whose CUDA worker DIED but whose
EngineCore main process is still alive keeps burning Python-overhead
CPU (HTTP keepalive, GIL ticks, network-thread-pool idle work), so the
session CPU keeps advancing and the #518 override reports `running`
forever while zero real work happens (#664 round 8 hung 60+ min,
reported healthy throughout). The one mechanical signal of this hang is
the orphaned GPU allocation: a compute-apps PID holding many GiB of
VRAM whose `/proc/<pid>` no longer exists. The probe lists
`nvidia-smi --query-compute-apps=pid,used_memory` and flags any PID
holding >= ``ZOMBIE_GPU_MEM_MIN_MIB`` MiB that is absent from `/proc`.

The zombie signature alone is NOT sufficient to fire (#826): on
host-PID-namespace RunPod containers nvidia-smi reports HOST PIDs that
are never resolvable in the container's `/proc`, so every HEALTHY
compute process carries the signature (#816 steady-state; #778 a
transient teardown-window PID between vLLM engine cycles) — and a false
`stalled` verdict routes into a destructive kill-by-PID reaper respawn.
The override therefore fires only when, on a `running` verdict, ALL of:
(a) a zombie candidate is present, (b) EVERY workload log (main /
per-phase / shard) is stale past the effective stall window
(``max(ZOMBIE_VETO_FRESH_SEC, stall_sec)`` — a genuinely hung run's own
processes stop appending; both observed false positives had fresh
logs), (b') no issue-keyed OUTPUT artifact was modified within the same
window (#1033 — a run writing results is not hanging; same veto
mechanics as the fresh-log term), and (c) the stale-log candidate
persisted for 2 CONSECUTIVE observed ticks (``zombie_streak`` in the
state sidecar — filters the #778 one-tick teardown transient). Bare session-CPU *advancement* is
deliberately NOT a veto term: the genuine #664 hang had CPU advancing
(the EngineCore idle-burn is why this override exists at all), so an
any-delta CPU veto would make the true positive unreachable. A MATERIAL
sustained burn *rate* IS one (#951): a session that burned >=
``ZOMBIE_OVERRIDE_CPU_CORES_MIN`` cores (default 0.5) on BOTH of the
last two persisted ticks is demonstrably computing — #664's hung
EngineCore churned ~0.22 cores while #825's falsely-flagged live fit
burned ~1.83-2.04 cores next to prior-run VRAM leftover — so the
override is vetoed (streak reset, like the fresh-log veto). The rate is
derived VM-side from the persisted ``session_cpu_secs`` /
``session_cpu_sample_epoch`` sidecar pair; ANY degraded input (unknown
sample, missing epoch, tick spacing under ``ZOMBIE_CPU_RATE_MIN_DT_SEC``,
negative delta) leaves the veto inert — exactly the pre-#951 behavior.
Fail-safe: nvidia-smi missing / erroring emits an empty list (never a
false zombie); the override never touches a `done` / `gate` / `dead`
verdict.

Namespace-informativeness gate (#864): the #826 assumption "hung <=>
stale logs" lapses when a HEALTHY workload legitimately silences its
logs longer than the stall window (#813: a ~29-min CPU-bound NPZ
compression stretch on a host-PID-namespace pod false-fired the
override twice, 2026-07-02). The probe therefore also counts
``GPU_PIDS_TOTAL`` / ``GPU_PIDS_RESOLVABLE`` over all compute-apps PIDs
and, only when a zombie candidate exists, ``NVIDIA_UVM_LIVE_HOLDERS`` —
live container processes holding a fd whose symlink target is EXACTLY
``/dev/nvidia-uvm`` (a live CUDA compute context; ``/dev/nvidia-uvm-
tools`` / ``nvidiactl`` / ``nvidia[0-9]`` never count). When
``total > 0 AND resolvable == 0 AND uvm_holders > 0`` the
dead-in-/proc signature is a PID-namespace artifact — the flagged PIDs
ARE live workers seen under host ids — and the override is vetoed
regardless of log staleness (streak reset, like the fresh-log veto).
Every other combination (any count unknown, ``resolvable > 0``,
``uvm == 0``) falls through to the #826 logic UNCHANGED, so the genuine
#664 total collapse (zero live uvm holders) still fires and every
degraded-probe read fails toward current behavior. Gated by
``ZOMBIE_NAMESPACE_VETO_ENABLED`` (env ``EPM_ZOMBIE_NAMESPACE_VETO``,
read at module import — restart a live poller for an ops flip); ships
default-OFF per the #864 pre-merge live-pod gate, which found a
cuInit'd parent/coordinator (``issue813_dispatch.py``) holding exact
uvm while absent from compute-apps — a holder class that would veto a
TOTAL collapse (matched pods included) if the veto were armed.

Staleness folds in cell-log mtimes (incident #405 smoke-first): when the
dispatcher is blocked in ``proc.wait()`` on a sequential smoke cell, the
main sweep log goes silent for ~15-18 min while the smoke cell actively
trains+evals and writes to its own per-cell log
(``<main_log_no_ext>/cell_*.log``). The probe therefore reports the
freshest mtime across (main log, newest cell log) so a healthy single-
cell phase reads as `running`, not false-`stalled` / false-`dead`. When
a cell log is the fresher source, its tail is also surfaced in
``log_tail_excerpt`` for the orchestrator's progress notifications.

Staleness ALSO folds in per-phase logs + GPU utilization (incident
#468 multi-phase training-sweep): a launcher that writes
``[phase=X]`` to the top-level log only at phase boundaries and
redirects the long phase's stdout to a separate
``/workspace/logs/issue-<N>-<phase>.log`` keeps the top-level log
silent for the full phase while the workload is actively writing to
the per-phase log AND keeping a GPU busy. Declaring `stalled` from
the top-level mtime alone false-fails the healthy run and strands a
billing pod. The probe therefore also reports (a) the max mtime over
``/workspace/logs/issue-<N>-*.log`` (excluding the top-level log and
``*.json`` / ``*.processed`` sentinels) and (b) per-GPU
``utilization.gpu`` integers via ``nvidia-smi``. The GPU check fails
safe: ``nvidia-smi`` unavailable / errors -> ``unknown`` (NOT idle),
so a healthy run is NEVER declared stalled purely from an nvidia-smi
failure — the per-phase-log mtime signal still carries the verdict.

Staleness ALSO folds in per-shard / repo-rooted phase logs (incident
#488 multi-GPU shard fan-out): some launchers write per-GPU shard
logs under a subdirectory like
``/workspace/explore-persona-space/logs/issue_<N>/phase*_g*.log``
(8 shard files under a nested directory, underscore separator), and
the #331 family of multi-phase scripts writes flat repo-rooted phase
logs like ``/workspace/explore-persona-space/logs/issue_<N>_phase<X>.log``.
Both layouts are invisible to the #468 ``/workspace/logs/issue-<N>-*.log``
glob — the i488 Pass B inner loop (~3 min between shard-log writes
across 57 cells per shard) silently tripped the 36-min main-log
threshold on 2026-06-07 while the pipeline was healthy. The probe
therefore ALSO reports the max mtime across both shard layouts so a
healthy multi-GPU run reads as `running`, not false-`stalled`. The
match remains intentionally narrow (only paths embedding ``issue_<N>``
or ``issue-<N>`` under the repo logs dir; not a broad recursive scan)
to avoid coupling other pods' background writes to the verdict.

Staleness ALSO folds in dispatcher per-job logs (incident #521
judge-batch wait): the issue_519/521-style dispatcher writes one log
per job under ``<output_dir>/logs/*.log``, with ``output_dir``
typically ``/workspace/explore-persona-space/eval_results/issue_<N>``.
During a CPU-bound phase that polls an external judge batch the GPUs
are idle BY DESIGN and the main log is quiet, while the per-job log
appends every 30-60s — the only liveness signal. On 2026-06-10 a #521
tick declared the healthy EM-steering job ``stalled`` (pid alive,
GPUs all 0, main log 1302s stale) because no probe reached
``eval_results/issue_<N>/logs/``. The shard-log probe therefore ALSO
globs ``eval_results/issue_<N>{,_*}/logs/*.log`` into the same
max-mtime reduction. The match stays narrow on purpose: the directory
must be exactly ``issue_<N>`` or ``issue_<N>_<suffix>`` (a bare
``issue_<N>*`` glob would let issue 5 match issue 521's directories).

Staleness ALSO folds in output artifacts (#1033; incident #813): a
CPU-bound analysis tail can write per-cell NPZs / result JSONs /
``.done`` sentinels for hours while EVERY log layout above is quiet and
the GPUs are idle by design — freshly-written outputs were the manual
dismissal signal on #813's ~6h analysis tail, and when the #951 CPU-rate
probe is degraded (tick-1/tick-2 warmup, dead launcher session, ``ps``
unavailable) output freshness is the remaining liveness channel. The
probe therefore emits ``OUTPUT_MTIME_EPOCH`` — the mtime of A file (a
fresh file, not the newest: short-circuit ``find -newermt ... -print
-quit`` under ``timeout``, bounded by
``EPM_POLL_OUTPUT_FIND_TIMEOUT_SEC``) modified within the freshness
window under the ISSUE-KEYED output roots
``eval_results/issue_<N>{,_*}/``, ``data/issue_<N>/``, and
``data/issue<N>/`` (same narrowness contract as the #488/#521 shard
globs — no broad recursive scan, no cross-pod coupling). The delta
joins the stall conjunction as a first-class liveness signal (a fresh
output behaves exactly like a fresh shard log) AND vetoes the
#664/#826 zombie override (streak reset, identical mechanics to the
fresh-log veto). Kill switch ``EPM_POLL_OUTPUT_MTIME_FOLD=0`` (default
ON — the fold can only SUPPRESS false stalls, and a genuinely hung run
writes no outputs, so the #664 true positive stays reachable; the
same-issue-sibling-writer exposure is the accepted #826 fresh-log
class). Every degraded input (missing dirs, find timeout, no GNU find,
a hit deleted before ``stat``) reads ``0`` -> "no fresh output" ->
pre-#1033 behavior.

GPU-idle advisory (incidents #518 + #537): the stall verdict treats an
idle GPU only as CORROBORATION — a run that is alive and logging on a
CPU-only phase with every GPU at 0% is (correctly) classified healthy,
and before this advisory it burned silently (#518 ran a single-core CPU
scoring phase ~14h on an idle 8xH100; #537 polled an external judge
batch 2.5h+ the same day, 2026-06-10). The poller therefore ALSO tracks
the sustained span of "healthy verdict + every GPU idle" across ticks
(state-file backed, like ``ssh_fail_count``) and, once the span exceeds
``EPM_GPU_IDLE_ADVISORY_MIN`` minutes (default 30; ``0`` disables),
posts a NON-BLOCKING ``epm:progress`` advisory marker (note prefixed
``[gpu-idle-advisory]``, riding the same marker channel as the phase-
transition posts — no new marker schema) suggesting the CPU phase move
off-pod per CLAUDE.md "CPU-only phases don't hold GPU pods". At most
one advisory per phase name (de-dup persisted in the state file); the
advisory NEVER changes the status verdict and never stops anything.
Fail-safe semantics carry over from ``_gpu_idle``: an ``unknown`` /
unparsable GPU sample resets the span rather than counting as idle.

GPU-idle ESCALATION (incident #664): a SECOND tier above the advisory.
Once a MULTI-GPU pod (>= 2 cards) has been idle in an upload/CPU-only
phase (``_phase_is_cpu_only``) past ``EPM_GPU_IDLE_ESCALATION_MIN``
minutes (default 60, clamped up to the advisory min; ``0`` disables),
the poller fires a best-effort Telegram push + a LOUD
``[gpu-idle-escalation]`` ``epm:progress`` marker. It reads the SAME
idle span the advisory tracks (no second clock), de-dups one escalation
per phase, and — like the advisory — NEVER changes the status verdict
and NEVER stops the pod (the #664 8xH200 idle in a terminal upload phase
burned ~$530 / 12h seen only by the one-shot advisory). The remedy it
names is to route the upload off-pod / release the GPUs after a
checkpoint — the final upload phase is itself CPU-only.

Dead: PID not alive AND last phase line is NOT `done` (clean exit
should always end with `[phase=done]`).

Done requires corroboration (incident #545, 2026-06-11): a `[phase=done]`
match in the log tail alone is NOT sufficient — per-cell eval subprocesses
legitimately print lines like ``[phase=done] eval cell <X> complete``
MID-RUN, and a tick keyed on the bare substring reported ``status=done``
while the dispatcher pid was alive, GPUs at 85%, and a training bar
mid-flight (an orchestrator trusting that would advance to verifying and
Step-8 terminate a live pod). A regex tighten to "bare line only" is NOT
viable: real dispatchers' TERMINAL done lines also carry trailing text
(``[phase=done] SMOKE COMPLETE ...``, ``[phase=done] phase4 complete
<date>``), textually indistinguishable from the mid-run noise. Instead,
``done`` is reported only when the done-parse is corroborated by EITHER
(a) the monitored pid being dead (the dispatcher exits right after its
terminal done line — on a normal completion this holds within seconds),
OR (b) a results sentinel ``issue-<N>-epm_results-*.json[.processed]``
existing on the pod (covers a dispatcher that lingers after done, e.g.
post-done uploads; ``.processed`` is included because this tick's drain
renames the sentinel moments before the status decision). An
uncorroborated done-parse is demoted to the latest NON-done phase line
and the verdict falls through to the normal liveness arbiters
(`running` / `stalled`), so the milestone tracker also never posts a
false ``-> done`` transition mid-run.

Phase-line shape expected from the entry script:
    2026-05-21 14:32:18 [phase=training step=1000/2000 loss=2.1]
    2026-05-21 14:55:02 [phase=eval]
    2026-05-21 15:10:44 [phase=done]

Anything matching the regex `\\[phase=([a-z0-9_]+)` will be picked up; the
token immediately after `phase=` is the milestone name. Digits are part of
the token (`[phase=p0_render]` parses as `p0_render`, not `p`), so numbered
phase-naming schemes (p0/p1/p2) work without spelling digits out. One carve
out (incident #597, 2026-06-11): a done-bearing line that ALSO carries an
explicit failure signal (nonzero ``rc=``, or a negation/suppression word
right after the token — ``DONE_QUOTED_NOISE_RE``) is treated as a failure
message QUOTING the token, not a phase transition, and is skipped; the
#545 corroboration cannot catch that shape because the crashed wrapper's
pid is DEAD, which normally corroborates a real done.

Sentinel schema (v1) — written by pod-side dispatchers, drained here:

    filename: /workspace/logs/issue-<N>-<kind_slug>-<epoch_seconds>.json
        kind_slug = kind with `:` -> `_` (e.g. ``epm_fact_candidates``).
    payload (JSON, dict):
        {
          "sentinel_schema_version": 1,                  # required, must be 1
          "task_id": <int>,                              # informational
          "kind": "<full kind, e.g. 'epm:fact-candidates'>",
          "version": <int>,                              # marker version
          "gate": "<gate name>" | null,                  # if set, poll returns status=gate
          "blocks_pipeline": true|false,                 # informational
          "note": "<marker note body>",                  # may also be sent as 'payload'
          "by": "<author>",
          "ts": "<ISO-8601 UTC>",
        }

Unknown schema versions are logged + skipped (not renamed) so a future
poller can re-process them. Malformed JSON / missing required fields are
logged + skipped likewise — the sentinel is left in place so the next
poller (or a human) can inspect it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make src/ importable so we can call task_workflow.post_event directly.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space.task_workflow import (  # noqa: E402
    EVENT_NOTE_MAX,
    find_task_path,
    latest_event,
    list_events,
    post_event,
)

log = logging.getLogger("poll_pipeline")

# Default seconds of log-mtime silence before declaring the run stalled.
# Workloads with sparse log cadence (e.g. checkpoint-only logging at >15min
# intervals — task #522 builds a 16x16 JS-distance matrix that logs only
# every ~106min per partial-cache checkpoint) override this at the CLI via
# ``--stall-sec`` or env via ``EPM_POLL_STALL_SEC`` so the poller does not
# false-positive ``stalled`` during normal inter-checkpoint quiet windows.
# ``STALL_SEC`` is preserved as a module-level alias of the default so
# existing tests that read ``pp.STALL_SEC`` as a reference threshold keep
# working without modification.
DEFAULT_STALL_SEC = 900
STALL_SEC = DEFAULT_STALL_SEC
# Substring of the ValueError message raised by ``task_workflow.post_event``
# when ``note`` exceeds ``EVENT_NOTE_MAX``. Matched against ``str(exc)`` so
# we route exactly that failure to graceful-degradation (persist + pointer
# marker) instead of leaving the sentinel un-renamed and retrying forever.
# See ``src/explore_persona_space/task_workflow.py`` ``post_event``: the
# message format is ``"event note exceeds {EVENT_NOTE_MAX} chars (<len>); ..."``.
_OVERSIZE_NOTE_ERROR_SUBSTR = "event note exceeds"
PHASE_RE = re.compile(r"\[phase=([a-z0-9_]+)")
# A failure MESSAGE that merely QUOTES the done token is not a phase
# transition (incident #597, 2026-06-11): a crashed shard wrapper printed
# "ONE OR MORE SHARDS FAILED rc=1 - [phase=done] NOT emitted"; the parse
# then hit the pid-DEAD path, which the #545 corroboration treats as
# proof of completion (it only demotes the pid-ALIVE path), so the tick
# reported a false ``status=done`` on a failed run. Tightening PHASE_RE
# itself stays non-viable (#545: legit phase lines are timestamp-prefixed,
# legit terminal done lines carry trailing text), so ``latest_phase``
# instead DISCARDS a done-parse whose line also carries an explicit
# failure signal. High-precision signals only — suffixed terminal lines
# ("[phase=done] SMOKE COMPLETE ...") must keep parsing as done:
#   * a negation/suppression word immediately after the token
#     ("[phase=done] NOT emitted" / "... never reached" / "... skipped"), or
#   * a NONZERO rc= anywhere on the same line ("... FAILED rc=1 ...");
#     rc=0 does not match.
# Producer-side hygiene (never embed the literal in message prose) is the
# primary contract: experimenter.md § "During Execution" step 1 +
# experiment-implementer.md § "Pod-side result-reporting contract".
DONE_QUOTED_NOISE_RE = re.compile(
    r"\[phase=done\]\s*(?:not|never|suppressed|skipped)\b|\brc=[1-9]\d*\b",
    re.IGNORECASE,
)
# The epm:run-launched marker note is free-form `key=value` tokens plus
# trailing prose (see .claude/agents/experimenter.md "Post epm:run-launched").
# `pid=<int>` is the resolved python child PID the experimenter posted.
MARKER_PID_RE = re.compile(r"\bpid=(\d+)")

# ── Adaptive bg-poll interval (anti-stall redesign §7) ──────────────────────
#
# The orchestrator's bg-Bash sleep-chain re-invokes a FULL orchestrator turn
# (~330k context tokens) on every poll exit, so the chain interval is the
# dominant per-run cost over multi-hour workloads (issue-601: 2,561 turns,
# most concluding "still healthy, keep waiting"). Each tick therefore emits
# a recommended ``next_interval`` (seconds) alongside its verdict: a healthy,
# quiet ``running`` tick far from any phase boundary recommends the long
# QUIET interval; anything gate-adjacent, anomalous, recently-changed, or
# early-run stays on the short DEFAULT — the long interval must never delay
# a gate or mask a fresh failure.
#
# Risk bound: with the quiet interval an in-session stall can be noticed up
# to 30 min later than the fixed 540s chain. Acceptable because
# out-of-session detection is independently bounded by the watcher's 10-min
# passes + the */45 issue-tick cron (autonomous_session_watch.py /
# .claude/skills/issue-tick), and every gate-adjacent signal (gate verdict,
# sentinel activity, phase transition) forces the short interval. The
# orchestrator falls back to the DEFAULT when the key is absent or
# unparseable (.claude/skills/issue/SKILL.md Step 6d.2).
POLL_INTERVAL_DEFAULT_SEC = 540
POLL_INTERVAL_QUIET_SEC = 1800
# A run younger than this (measured from its latest epm:run-launched marker)
# always polls on the short interval — early failures are the most common
# kind and the most valuable to catch fast.
EARLY_RUN_WINDOW_SEC = 1800
# Minimum quiet time since the last observed [phase=...] transition before
# the long interval applies — a run that recently crossed a phase boundary
# is likely near another one (boundaries cluster: train -> eval -> upload ->
# done often land minutes apart).
RECENT_PHASE_CHANGE_WINDOW_SEC = 1800


def recommend_next_interval(
    *,
    status: str,
    gate: str | None,
    sentinels_processed: int,
    phase_transitioned: bool,
    ssh_failed: bool,
    gpu_idle_advisory_posted: bool,
    cpu_override_active: bool,
    run_age_sec: float | None,
    phase_changed_ago_sec: float | None,
    gpu_idle_escalation_posted: bool = False,
) -> int:
    """Pure decision core for the adaptive bg-poll interval (§7).

    Returns :data:`POLL_INTERVAL_QUIET_SEC` ONLY when every quiet condition
    holds; every other tick returns :data:`POLL_INTERVAL_DEFAULT_SEC`. The
    interval NEVER lengthens on a tick that reported anything other than
    healthy-quiet-running:

    * ``status`` must be ``running`` — done/gate/stalled/dead ticks are
      terminal or gate-adjacent and the orchestrator acts on them
      immediately, so their interval is moot but stays short by contract.
    * no gate and no sentinel activity this tick — sentinels are pod->VM
      messages; any drain activity means something is happening that is
      worth watching closely.
    * no phase transition this tick AND none within
      :data:`RECENT_PHASE_CHANGE_WINDOW_SEC` (an unknown last-change time
      — fresh state file, or a workload that never prints phase lines —
      counts as recent: fail toward coverage).
    * no anomaly this tick: SSH transport failure, a GPU-idle advisory
      post, a GPU-idle ESCALATION post (#664 — a multi-GPU pod idle in an
      upload/CPU-only phase past the escalation threshold; treated the same
      as the advisory so the poll cadence does not go quiet right after
      escalating), or the #518 CPU-advancing stall-rescue (logs stale +
      GPUs idle — the run is healthy but in a degraded-observability
      regime). Deliberately NOT in the set: raw GPU idleness alone
      (``_gpu_idle(gpu_util)`` on a tick that posted no advisory).
      Idle GPUs on a healthy run are routine during long CPU-bound
      phases (judge-API scoring, aggregation, plotting) — exactly the
      stretches where the quiet interval saves the most full-context
      turns — so the condition would forfeit most of §7's savings to
      sharpen an advisory-only (non-gate, non-failure) signal. Accepted
      consequence: after the one-per-phase advisory posts, a SECOND
      advisory in a NEW phase can land up to one quiet interval late
      (the phase transition itself still forces the short interval the
      tick it is observed; within the §7 30-min risk bound; decision
      2026-06-12).
    * past the early-run window: ``run_age_sec`` known AND at least
      :data:`EARLY_RUN_WINDOW_SEC` (an unknown launch age also counts as
      early-run — fail toward coverage, not toward silence).

    Pure / no I/O — ``poll_once`` supplies the signals; tests drive the
    decision table directly (tests/test_poll_next_interval.py).
    """
    if status != "running":
        return POLL_INTERVAL_DEFAULT_SEC
    if gate is not None or sentinels_processed > 0:
        return POLL_INTERVAL_DEFAULT_SEC
    if phase_transitioned:
        return POLL_INTERVAL_DEFAULT_SEC
    if ssh_failed or gpu_idle_advisory_posted or gpu_idle_escalation_posted or cpu_override_active:
        return POLL_INTERVAL_DEFAULT_SEC
    if run_age_sec is None or run_age_sec < EARLY_RUN_WINDOW_SEC:
        return POLL_INTERVAL_DEFAULT_SEC
    if phase_changed_ago_sec is None or phase_changed_ago_sec < RECENT_PHASE_CHANGE_WINDOW_SEC:
        return POLL_INTERVAL_DEFAULT_SEC
    return POLL_INTERVAL_QUIET_SEC


def _resolve_state_dir_root() -> Path:
    """Main-checkout root for the phase-cache anchor, resolved cwd-independently.

    ``poll-pipeline-<N>.json`` is CROSS-INVOCATION shared state: ticks may
    run with cwd = the repo root, an issue worktree, or via a worktree COPY
    of this script, and ``backends/runpod.py`` composes the same path from
    :data:`DEFAULT_STATE_DIR` so its in-process polls share the phase-cache
    with the orchestrator's bg-Bash loop. The pre-2026-06-12 anchor
    (``_REPO_ROOT``, this script copy's own checkout via ``__file__``)
    split that contract across checkouts — a worktree-copy invocation
    wrote the phase-cache in the worktree while a repo-root tick read the
    repo-root copy, re-posting already-seen milestones as spurious
    ``new_milestone`` markers (same split-brain class as the #612
    handle-sidecar incident, fixed the same way — see
    ``backends.issue_dispatch._main_checkout_root``).

    Resolution runs ``git rev-parse --path-format=absolute
    --git-common-dir`` from THIS script's directory (never ``os.getcwd()``);
    from a linked worktree the common dir is ``<main>/.git``, so its parent
    is the main checkout. Local copy rather than an import of the
    ``issue_dispatch`` resolver: that module pulls the full router chain at
    module level (too heavy for a tick script), and ``backends/runpod.py``
    lazily imports THIS module, so the reverse module-level import would
    tangle the dependency direction.

    Fail-SOFT by design (unlike the fail-loud ``issue_dispatch`` resolver):
    a non-git execution context degrades to the legacy ``_REPO_ROOT``
    anchor with a warning instead of crashing — the poller must keep
    reporting even when the cache anchor is degraded.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"}
    }
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(_HERE),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        common_dir = Path(proc.stdout.strip())
        if common_dir.name == ".git" and common_dir.is_dir():
            return common_dir.parent
        log.warning(
            "phase-cache anchor: git common-dir %s does not look like a main-checkout "
            ".git directory; falling back to the script-copy checkout %s",
            common_dir,
            _REPO_ROOT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, OSError) as exc:
        log.warning(
            "phase-cache anchor: could not resolve the main checkout from %s (%s); "
            "falling back to the script-copy checkout %s",
            _HERE,
            exc,
            _REPO_ROOT,
        )
    return _REPO_ROOT


DEFAULT_STATE_DIR = _resolve_state_dir_root() / ".claude" / "cache"

# How many consecutive SSH-probe failures must accumulate before the poller
# auto-fires ``pod.py config --refresh-from-api <pod>`` as a stale-port
# self-heal. Set to 10 (~3-4 min at the orchestrator's typical 20s spacing) so
# a transient SSH hiccup never burns a refresh call, but a sustained
# connection-refused stretch — the #488 stale-port pattern — does. After the
# refresh attempt the counter resets so we never hot-loop refresh calls; the
# next ``SSH_FAIL_REFRESH_THRESHOLD`` consecutive failures will trigger a
# re-try.
SSH_FAIL_REFRESH_THRESHOLD = int(os.environ.get("EPM_POLL_SSH_FAIL_REFRESH_THRESHOLD", "10"))

# Escalation threshold for the [ssh-wait-ALARM] (refs #572): once a pod has
# been SSH-unreachable for this long while its experiment is supposed to be
# running (pod presumed billing), the per-tick warnings escalate to a loud
# structured log.error line naming the refresh-from-api recovery, re-fired at
# most once per window. The refresh-counter above can't measure the total
# span (it resets on every auto-heal attempt) — pod-488 (2026-06-09) spun
# ~13.7h at $32/hr with only per-tick noise.
SSH_WAIT_ALARM_SECS = float(os.environ.get("EPM_SSH_WAIT_ALARM_SECS", "3600"))

# Zombie-GPU-allocation floor (#664): minimum VRAM (MiB) a dead-PID compute
# allocation must hold before the probe flags it as a zombie. A hung vLLM
# whose CUDA worker died leaves its full model-shard allocation (tens of GiB)
# orphaned on the card; the 1 GiB floor avoids flagging trivial leftover CUDA
# contexts (a few hundred MiB) while catching any real model-weight orphan.
ZOMBIE_GPU_MEM_MIN_MIB = int(os.environ.get("EPM_ZOMBIE_GPU_MEM_MIN_MIB", "1024"))

# 826: FLOOR for the liveness veto on the #664 zombie-GPU override. The
# effective veto threshold each tick is max(ZOMBIE_VETO_FRESH_SEC, stall_sec):
# the override fires only when EVERY workload log (main/phase/shard) is stale
# past the effective stall window. Rationale: a hung run's own processes stop
# appending (the #664 TP had all mtimes > 900s), while both observed false
# positives had fresh logs (#816 ~1s steady-state; #778 8s transient) — note a
# SIBLING process can keep a log fresh on a hung run, so this is an empirical
# separation, not an absolute guarantee; the asymmetric cost (missed zombie =
# bounded idle pod-hours vs false positive = kill-by-PID reaper on a healthy
# run) favors vetoing. The floor only binds when --stall-sec is set
# pathologically low (< 60s fast-smoke configs).
ZOMBIE_VETO_FRESH_SEC = int(os.environ.get("EPM_ZOMBIE_VETO_FRESH_SEC", "60"))

# #864: kill-switch for the namespace-informativeness veto on the zombie-GPU
# override. "0" disables the veto entirely (pure #826 behavior) — an ops
# escape hatch if the UVM-holder correspondence ever masks a true positive in
# the field. NOTE: read at module import — a live poller needs a restart for
# an ops flip to take effect. DEFAULT "0" (veto inert-but-ready) per the #864
# §7 pre-merge live-pod gate (2026-07-02, disposition 2): the negative-
# direction check on pod-813 found a cuInit'd-but-allocation-free
# parent/coordinator (issue813_dispatch.py) holding an EXACT /dev/nvidia-uvm
# fd while ABSENT from compute-apps (5 exact-uvm holders vs 4 compute apps) —
# a holder class that would veto a genuine TOTAL collapse (TP suppression)
# if the veto were armed. Flipping to "1" after a clean both-directions
# re-check is a one-literal follow-up; the probe counts + gate + tests land
# either way.
ZOMBIE_NAMESPACE_VETO_ENABLED = os.environ.get("EPM_ZOMBIE_NAMESPACE_VETO", "0") != "0"

# #951: material-compute liveness veto on the #664/#826 zombie-GPU override.
# A workload session burning >= this many CPU cores (delta cumulative session
# CPU / wall seconds between persisted ticks) on BOTH of the last two ticks is
# demonstrably computing — veto the running->stalled override. Sits between
# the two measured regimes: #664's hung EngineCore churned ~0.22 cores
# (Python-overhead idle burn) while #825's falsely-flagged live fit burned
# ~1.83-2.04 cores (+1102s/+989s per ~540s tick, 186% in top) — 2.27x above
# churn, 3.66x below real compute. Sustained-rate, so poll-interval-invariant
# (540s vs 1800s adaptive ticks). NOTE: session_cpu_secs is a session-TOTAL
# (summed over every process in the launcher's setsid session) and both
# calibration incidents were NARROW sessions — a wide hung session (TP>=2
# NCCL spin ~1 core/rank, or a co-resident same-session burner) can sum past
# this threshold and keep a true zombie vetoed (accepted exposure, same
# class as the #826 fresh-log sibling; backstops: GPU-idle escalation, #873
# ETA tripwires, watcher wedge arm). Raise EPM_ZOMBIE_OVERRIDE_CPU_CORES_MIN
# when debugging a suppressed wide-pod hang.
ZOMBIE_OVERRIDE_CPU_CORES_MIN = float(os.environ.get("EPM_ZOMBIE_OVERRIDE_CPU_CORES_MIN", "0.5"))

# #951: denominator floor for the per-tick CPU rate. Below this wall-clock
# spacing between persisted CPU samples the rate is not computed (None ->
# veto inert): `ps -o time` truncates to whole seconds per process, so a
# short window inflates spurious rate from truncation noise (~1s x N
# session processes). Production intervals are 540s/1800s so the floor
# never binds there; it guards manual rapid re-polls and fast-smoke ticks.
ZOMBIE_CPU_RATE_MIN_DT_SEC = int(os.environ.get("EPM_ZOMBIE_CPU_RATE_MIN_DT_SEC", "120"))

# #1033: output-artifact mtime fold. Kill switch, default ON (unlike the #864
# default-OFF namespace veto): the fold can only SUPPRESS false `stalled`
# verdicts / zombie overrides, and a genuinely hung run writes no outputs, so
# the #664 true positive stays reachable. The residual exposure — a same-issue
# sibling process (e.g. a detached uploader) touching issue-keyed files during
# a true hang — is the SAME accepted exposure class as the #826 fresh-log
# sibling, bounded by the GPU-idle advisory/escalation tiers, the #873
# tripwires, and this switch. When disabled the probe block is omitted
# entirely and ``poll_once`` forces ``output_mtime_ago = inf`` (fully inert).
# NOTE: read at module import (matching the #864 flag) — a live poller needs
# a restart for an ops flip to take effect.
OUTPUT_MTIME_FOLD_ENABLED = os.environ.get("EPM_POLL_OUTPUT_MTIME_FOLD", "1") != "0"

# #1156: slack for the stale-pid-file-vs-marker WARN. Must exceed the normal
# launch->marker-post latency (the pid file is written BEFORE epm:run-launched
# posts — experimenter.md steps 1/1b; worst observed normal-family gap 516 s,
# the #813 v5 relaunch, so 600 s carries ~16% margin) while staying well under
# the >=30 min inter-launch gap of a genuine stale file.
PID_FILE_MARKER_SLACK_SEC = int(os.environ.get("EPM_POLL_PID_MARKER_SLACK_SEC", "600"))


def _output_find_timeout_sec() -> int:
    """The bounded ``timeout`` (seconds) around the pod-side output ``find``.

    Default 10s (env ``EPM_POLL_OUTPUT_FIND_TIMEOUT_SEC``); clamped to
    [1, 15] so the two-stage worst case (2 x 15s) stays well inside the
    probe's 60s SSH exec budget (a wedged-FS ``find`` must never starve the
    rest of the heredoc). A malformed env value falls back to 10 (fail-safe).
    """
    raw = os.environ.get("EPM_POLL_OUTPUT_FIND_TIMEOUT_SEC", "10")
    try:
        val = int(raw)
    except (TypeError, ValueError):
        val = 10
    return min(max(1, val), 15)


def _try_refresh_pods_conf_from_api(pod: str) -> bool:
    """Best-effort ``pod.py config --refresh-from-api <pod>`` self-heal.

    Fires after :data:`SSH_FAIL_REFRESH_THRESHOLD` consecutive ``_ssh_probe``
    failures on the same pod — the #488 stale-port pattern, where a
    SUPPLY_CONSTRAINT-blocked resume eventually brought the pod back at a NEW
    SSH port via a retry path that bypassed ``_upsert_pods_conf`` and
    ``pods.conf`` stayed stale while the SSH polling loop spun indefinitely.

    Fail-soft: any failure (subprocess timeout, non-zero exit, missing
    binary, oserror) is logged and the function returns False. The polling
    loop never crashes on this auto-heal; the caller resets the failure
    counter regardless so we don't hot-loop refresh calls back-to-back.

    Returns True on success (refresh-from-api exited 0), False otherwise.
    """
    cmd = [
        "uv",
        "run",
        "python",
        str(_REPO_ROOT / "scripts" / "pod.py"),
        "config",
        "--refresh-from-api",
        pod,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning(
            "auto-heal: pod.py config --refresh-from-api %s raised %s; "
            "polling loop continues (next %d consecutive failures will retry)",
            pod,
            type(exc).__name__,
            SSH_FAIL_REFRESH_THRESHOLD,
        )
        return False
    if result.returncode != 0:
        log.warning(
            "auto-heal: pod.py config --refresh-from-api %s exited rc=%d; stderr=%s",
            pod,
            result.returncode,
            (result.stderr or "").strip(),
        )
        return False
    log.info(
        "auto-heal: pod.py config --refresh-from-api %s OK; pods.conf "
        "+ ~/.ssh/config refreshed against the live RunPod API after %d "
        "consecutive SSH-probe failures (#488 stale-port pattern)",
        pod,
        SSH_FAIL_REFRESH_THRESHOLD,
    )
    return True


def _state_float(prev_state: dict[str, str], key: str) -> float:
    """Read a float out of the string-valued tick state; garbled -> 0.0."""
    try:
        return float(prev_state.get(key, "0") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _update_ssh_fail_tracking(
    prev_state: dict[str, str],
    *,
    ssh_failed: bool,
    pod: str,
    issue: int,
    now_epoch: float | None = None,
) -> tuple[int, float, float]:
    """Advance the per-tick SSH-failure bookkeeping; returns
    ``(ssh_fail_count, ssh_fail_since, ssh_wait_alarm_ts)`` for the state save.

    Two escalation layers share this accounting:

    1. **#488 stale-port self-heal** — after ``SSH_FAIL_REFRESH_THRESHOLD``
       consecutive failures, fire ``pod.py config --refresh-from-api <pod>``
       once (fail-soft) and reset the counter so the next N consecutive
       failures retry.
    2. **[ssh-wait-ALARM]** (refs #572) — the refresh counter resets on every
       auto-heal attempt, so it cannot measure the TOTAL unreachable span;
       pod-488 (2026-06-09) spun ~13.7h at $32/hr with only per-tick noise.
       ``ssh_fail_since`` records the episode start; once the span crosses
       ``SSH_WAIT_ALARM_SECS`` (default 1h) the per-tick warnings escalate to
       a loud structured ``log.error`` naming the recovery command, re-fired
       at most once per window (``ssh_wait_alarm_ts``). The pod is presumed
       billing — this polling only runs while the experiment is supposed to
       be RUNNING.
    """
    now_epoch = time.time() if now_epoch is None else now_epoch
    try:
        ssh_fail_count = int(prev_state.get("ssh_fail_count", "0"))
    except (TypeError, ValueError):
        ssh_fail_count = 0
    ssh_fail_since = _state_float(prev_state, "ssh_fail_since")
    ssh_wait_alarm_ts = _state_float(prev_state, "ssh_wait_alarm_ts")

    if not ssh_failed:
        return 0, 0.0, 0.0

    if ssh_fail_since <= 0:
        ssh_fail_since = now_epoch
    ssh_fail_count += 1
    if ssh_fail_count >= SSH_FAIL_REFRESH_THRESHOLD:
        log.warning(
            "SSH probe failed %d consecutive ticks for pod %s; "
            "firing pod.py config --refresh-from-api %s "
            "(#488 stale-port auto-heal)",
            ssh_fail_count,
            pod,
            pod,
        )
        _try_refresh_pods_conf_from_api(pod)
        # Reset after the attempt regardless of outcome so we don't
        # hot-loop refresh calls every tick.
        ssh_fail_count = 0
    waited = now_epoch - ssh_fail_since
    if waited >= SSH_WAIT_ALARM_SECS and now_epoch - ssh_wait_alarm_ts >= SSH_WAIT_ALARM_SECS:
        log.error(
            "[ssh-wait-ALARM] pod %s has been SSH-unreachable for %.1fh while "
            "its experiment is supposed to be RUNNING (the pod is presumed "
            "billing). Likely a stale host/port in pods.conf (#488 pattern). "
            "Recovery: `uv run python scripts/pod.py config "
            "--refresh-from-api %s`; if the pod is genuinely idle, stop it "
            "(`pod.py stop --issue %d`) to halt the burn.",
            pod,
            waited / 3600.0,
            pod,
            issue,
        )
        ssh_wait_alarm_ts = now_epoch
    return ssh_fail_count, ssh_fail_since, ssh_wait_alarm_ts


def _marker_pid(issue: int) -> int | None:
    """Return the `pid=` from the latest epm:run-launched marker, or None.

    Self-correction source when the on-pod pidfile is stale: the marker
    the experimenter posts on every (re)launch carries the live python
    child PID. Reading it is a pure, branch-guarded library read on the
    VM (no commit), so it is safe from poll_pipeline's bg-Bash context.
    """
    try:
        ev = latest_event(issue, prefix="epm:run-launched")
    except Exception as exc:
        log.warning("could not read epm:run-launched for #%d: %s", issue, exc)
        return None
    if ev is None:
        return None
    m = MARKER_PID_RE.search(ev.get("note", "") or "")
    return int(m.group(1)) if m else None


def _run_launched_age_sec(issue: int, now_epoch: float) -> float | None:
    """Seconds since the latest ``epm:run-launched`` marker, or None.

    Early-run signal for the adaptive bg-poll interval (§7): a run inside
    its first :data:`EARLY_RUN_WINDOW_SEC` always polls on the short
    interval. None (unknown) when the marker is missing, unreadable, or
    carries an unparseable ``ts`` — ``recommend_next_interval`` treats
    unknown as early-run (short interval; fail toward coverage). Reads the
    same branch-guarded VM-side library path as :func:`_marker_pid`.
    """
    try:
        ev = latest_event(issue, prefix="epm:run-launched")
    except Exception as exc:
        log.warning("could not read epm:run-launched ts for #%d: %s", issue, exc)
        return None
    if ev is None:
        return None
    raw_ts = ev.get("ts")
    if not raw_ts:
        return None
    try:
        # task_workflow._utcnow_iso emits "%Y-%m-%dT%H:%M:%SZ"; py3.11's
        # fromisoformat accepts the trailing "Z" directly.
        launched = datetime.fromisoformat(str(raw_ts))
    except ValueError:
        return None
    if launched.tzinfo is None:
        launched = launched.replace(tzinfo=UTC)
    return now_epoch - launched.timestamp()


def _pid_file_predates_marker(
    *,
    pid_file_mtime_epoch: int,
    pod_now_epoch: int,
    run_age_sec: float | None,
    slack_sec: int | None = None,
) -> bool:
    """True iff the pod-side pid file's mtime predates the newest
    epm:run-launched marker by more than ``slack_sec`` (#1156).

    Same-clock differences only (#704): ``pod_now_epoch - pid_file_mtime_epoch``
    is pod-clock, ``run_age_sec`` is VM-clock — never a cross-clock subtraction.
    Inert (False) on any missing input: ``pid_file_mtime_epoch <= 0`` (absent /
    stat failed / legacy probe), ``pod_now_epoch <= 0`` (no drift-free basis),
    ``run_age_sec is None`` (no marker / unparseable ts). Pure; unit-tested
    directly with no SSH.
    """
    if slack_sec is None:
        slack_sec = PID_FILE_MARKER_SLACK_SEC
    if pid_file_mtime_epoch <= 0 or pod_now_epoch <= 0 or run_age_sec is None:
        return False
    pid_file_age_sec = pod_now_epoch - pid_file_mtime_epoch
    return pid_file_age_sec > run_age_sec + slack_sec


def _maybe_warn_stale_pid_file(
    *,
    issue: int,
    pod: str,
    pid_file: str,
    probe: dict[str, str],
    run_age_sec: float | None,
    pid_file_missing: bool,
) -> bool:
    """Fail-soft #1156 WARN emitter; returns the observability flag.

    WARN-only by contract: never contributes to status / stall / dead.
    ``pid_file_missing`` short-circuits (the #521 absent-file WARN already
    covers that case — no double-fire). Any exception is swallowed to DEBUG:
    the flag is WARN-only observability feeding no verdict, and every live
    poll loop fleet-wide runs this hot shared script, so a broken backstop
    must never break a tick — a deliberate, narrow exception to fail-fast
    (the parse tolerance itself comes from the #1033 r2 defensive-scalar
    pattern).
    """
    try:
        if pid_file_missing:
            return False
        pod_now_epoch = _parse_output_mtime_epoch(probe.get("pod_now_epoch"))
        pid_file_mtime_epoch = _parse_output_mtime_epoch(probe.get("pid_file_mtime_epoch"))
        stale = _pid_file_predates_marker(
            pid_file_mtime_epoch=pid_file_mtime_epoch,
            pod_now_epoch=pod_now_epoch,
            run_age_sec=run_age_sec,
        )
        if stale:
            pid_age = pod_now_epoch - pid_file_mtime_epoch
            log.warning(
                "pid file %s on pod %s predates the newest epm:run-launched marker "
                "for #%d (pid-file age %ds vs marker age %.0fs, slack %ds) — possible "
                "stale pid from a prior launch masking a dead relaunch (#813 pid-file "
                "rewrite contract). WARN-only; status verdict unchanged. Recovery: "
                "rewrite the pid file with the live workload pid and re-post "
                "epm:run-launched (pod-side-reporting.md § Pid-file launch contract).",
                pid_file,
                pod,
                issue,
                pid_age,
                run_age_sec,
                PID_FILE_MARKER_SLACK_SEC,
            )
        return stale
    except Exception as exc:
        log.debug("stale-pid-file-vs-marker check failed (ignored, #1156): %s", exc)
        return False


# Schema version the poller knows how to parse. Bump in lockstep with the
# pod-side writer (currently ``run_experiment_<N>.py::SENTINEL_SCHEMA_VERSION``).
# Newer schemas are skipped + logged, never silently mis-parsed.
SENTINEL_SCHEMA_VERSION_SUPPORTED = 1

# Required keys in every parsed sentinel payload. ``payload`` is accepted as
# a synonym for ``note`` for forward-compat with sentinels that put the
# marker body under that key.
_SENTINEL_REQUIRED_KEYS: tuple[str, ...] = (
    "sentinel_schema_version",
    "kind",
    "version",
)

# The /issue SKILL.md Step 7 results-payload contract (all 10 required; the
# "Sentinel format (JSON object with these keys, all required)" list). Used
# by the #899 envelope-synthesis fallback below — keep in lockstep with
# SKILL.md Step 7.
_RESULTS_PAYLOAD_KEYS: tuple[str, ...] = (
    "eval_numbers",
    "eval_paths",
    "reproducibility_card",
    "wandb_url",
    "hf_hub_url",
    "worktree_path",
    "final_commit_sha",
    "gpu_hours_used",
    "gpu_hours_budgeted",
    "plan_deviations",
)
_RESULTS_SENTINEL_SUFFIX = "-results.json"


def _maybe_synthesize_results_envelope(
    remote_path: str, data: dict[str, Any], missing: list[str]
) -> dict[str, Any] | None:
    """#899 fallback: rescue a raw Step 7 results payload on a results filename.

    Returns a synthesized enveloped dict WITHOUT a ``"version"`` key (the
    drain threads ``version=None`` to ``post_event``, which derives max+1
    per kind; #975), or None (caller keeps the skip path).
    Fires ONLY for: basename endswith ``-results.json`` AND no envelope key
    present AND all 10 Step 7 payload keys present (extras tolerated).

    NOTE: the ``-results.json`` suffix is NOT unique to raw writers — several
    canonical ENVELOPED writers use it too (``dispatch_neg_geometry_504.py``,
    ``i488_phase3_train_sweep.py``, smoke ``issue-<N>-smoke-results.json``
    files). Safety against rescuing the wrong sentinel rests on the
    zero-envelope-keys leg + the 10-key conjunction (an enveloped sentinel
    never reaches this helper's synthesis branch), NOT on filename
    uniqueness. Do not widen this predicate on the filename leg alone.

    Incident #825 run 7: the pod-side writer emitted the complete Step 7
    payload as a raw JSON object (no ``sentinel_schema_version``/``kind``/
    ``version`` envelope) at ``issue-825-results.json``; the drain skipped it
    with a quiet warning right before the GCP VM self-deleted.
    """
    if not remote_path.rsplit("/", 1)[-1].endswith(_RESULTS_SENTINEL_SUFFIX):
        return None
    if any(k in data for k in _SENTINEL_REQUIRED_KEYS):
        return None  # partial envelope: ambiguous/buggy writer — keep the strict skip
    missing_payload = [k for k in _RESULTS_PAYLOAD_KEYS if k not in data]
    if missing_payload:
        log.error(
            "sentinel %s sits on a results filename with no envelope AND is "
            "missing results-payload keys %s; skipping (NOT rescued) — inspect "
            "before the VM self-deletes (#899)",
            remote_path,
            missing_payload,
        )
        return None
    log.error(
        "sentinel %s carried a complete Step 7 results payload but no envelope "
        "keys %s; synthesizing kind=epm:results (version omitted -> post_event "
        "derives max+1; #975) and posting — pod-side writer should emit the "
        "envelope (#899)",
        remote_path,
        missing,
    )
    note = dict(data)  # non-destructive copy — the original payload dict is untouched
    note["envelope_synthesized"] = True  # provenance, rides the marker note
    return {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION_SUPPORTED,
        "kind": "epm:results",
        # "version" DELIBERATELY OMITTED (#975): key absence is the drain's
        # signal to pass version=None to post_event, which derives
        # max(existing for kind)+1 — a hardcoded 1 landed below an existing
        # higher version on re-runs (the #480/#389/#825 collision class).
        # Real sentinels always carry the key (_SENTINEL_REQUIRED_KEYS), so
        # absence is unambiguous. NOTE: this dict is returned by
        # _parse_sentinel WITHOUT re-validation — do not add a required-keys
        # re-check on it without revisiting this omission.
        "note": note,
        "by": "pod-sentinel-envelope-fallback",  # secondary provenance
    }


@dataclass(frozen=True)
class PollResult:
    status: str  # running | done | gate | stalled | dead
    current_phase: str
    new_milestone: bool
    last_log_mtime_sec_ago: int
    pid_alive: bool
    # True when the pod-side pid FILE did not exist at probe time —
    # ``pid_alive=False`` then means "no pidfile to probe" (possibly with
    # a live marker-pid fallback carrying liveness), NOT "pid probed
    # dead". Observability only; status routing is unchanged (#521).
    pid_file_missing: bool
    log_tail_excerpt: str
    gate: str | None = None  # set when a drained sentinel carried a non-empty gate
    sentinels_processed: int = 0
    # Broadened liveness signals (see ``_ssh_probe`` docstring + the
    # module-level "Staleness ALSO folds in per-phase logs + GPU
    # utilization" paragraph). Surfaced so the orchestrator's JSON-line
    # summary records WHY a healthy long-phase run stayed in `running`
    # despite a quiet top-level + cell log.
    phase_log_mtime_sec_ago: int = 10**9
    # Shard / repo-rooted phase log freshness (#488 shard layouts +
    # #521 dispatcher per-job logs). ``10**9`` means no covered layout
    # exists yet (defaults to "very old" so the absence never by itself
    # keeps a stalled verdict from firing).
    shard_log_mtime_sec_ago: int = 10**9
    gpu_util: str = "unknown"
    # True when THIS tick posted the [gpu-idle-advisory] marker (#518/#537).
    # Observability only; the advisory never changes ``status``.
    gpu_idle_advisory_posted: bool = False
    # True when THIS tick posted the [gpu-idle-escalation] marker + Telegram
    # push (#664) — a MULTI-GPU pod idle past GPU_IDLE_ESCALATION_MIN in an
    # upload/CPU-only phase. Observability only; never changes ``status`` and
    # never stops the pod. Defaulted so cross-backend PollResult(...) call
    # sites need no change.
    gpu_idle_escalation_posted: bool = False
    # True when THIS tick posted the [gpu-width-advisory] marker (#873) — a
    # STABLE strict subset of GPUs idle >= GPU_WIDTH_ADVISORY_MIN minutes on
    # a multi-GPU pod (#813 idle-width class). Observability only; never
    # changes ``status`` and never stops anything. Defaulted so the
    # cross-backend PollResult(...) call sites need no change.
    gpu_width_advisory_posted: bool = False
    # True when THIS tick posted the [gpu-underparallel-warning] marker (plan §3,
    # workflow v2) — < 50% of the provisioned GPUs busy >= GPU_UNDERPARALLEL_
    # WARNING_MIN minutes on a multi-GPU pod, once per run. Observability only;
    # never changes ``status`` and never stops anything. Defaulted so the
    # cross-backend PollResult(...) call sites need no change.
    gpu_underparallel_warning_posted: bool = False
    # True when THIS tick posted an epm:compute-deviation (source: poller)
    # marker (#873) — elapsed wall-time exceeded ETA_DEVIATION_MULT x the
    # plan §9 planned_wall_h TOTAL for the current phase or the whole run
    # (#763 class). Observability only; never changes ``status``.
    eta_deviation_posted: bool = False
    # Session-CPU signal (#518, #658). ``session_cpu_secs`` is the literal
    # probe output: a float string like ``"4271.5"`` or ``"unknown"``.
    # ``cpu_advancing`` is the ternary decision relative to the running
    # MAXIMUM cumulative CPU observed across ticks: True (a new high-water
    # mark = genuine progress, OR a sub-max drop = a multi-shard child-exit
    # accounting artifact — neither is a hang), False (flat at the
    # high-water mark = truly idle), None (no signal — first tick, launcher
    # dead, or ps unavailable). Surfaced in the JSON line so operators can
    # see WHY a long-quiet run stayed in ``running`` (or WHY a stall verdict
    # landed despite a CPU-bound phase).
    session_cpu_secs: str = "unknown"
    cpu_advancing: bool | None = None
    # Recommended seconds before the NEXT poll tick (adaptive bg-poll
    # interval, anti-stall redesign §7 — see ``recommend_next_interval``).
    # ``POLL_INTERVAL_QUIET_SEC`` only on a healthy, quiet, post-early-run
    # ``running`` tick far from any phase boundary; the short
    # ``POLL_INTERVAL_DEFAULT_SEC`` otherwise. The orchestrator's
    # sleep-chain reads this from the tick JSON (540s fallback when
    # absent/unparseable — SKILL.md Step 6d.2).
    next_interval: int = POLL_INTERVAL_DEFAULT_SEC
    # Machine-readable reason a non-``running`` verdict landed, surfaced in
    # the JSON line so the orchestrator can route differently per cause.
    # ``None`` on a healthy ``running`` tick and on stalls without a
    # specific cause (the generic log+GPU+CPU conjunction). Currently set
    # only for the zombie-GPU-allocation stall (#664):
    # ``"vllm_worker_dead_zombie_gpu"`` — a dead CUDA-worker PID still
    # holding VRAM while the EngineCore main process keeps the
    # session-CPU-advancing override alive (which would otherwise mask the
    # hang as ``running`` forever). Since #826 the override also requires
    # ALL workload logs stale past the effective stall window AND 2
    # consecutive stale-log candidate ticks (host-PID-namespace containers
    # make the bare signature false-positive on healthy runs). Defaulted so
    # the many cross-backend ``PollResult(...)`` call sites need no change.
    stall_reason: str | None = None
    # The crash signature extracted from the WIDE 500-line probe tail (NOT the
    # 5-line ``log_tail_excerpt``, which truncates a multi-line vLLM CUDA-IMA
    # traceback — the signature lines routinely sit >5 lines from the end). The
    # whole wide tail is stored (so the #775 RunPod CUDA-IMA repeat-failover
    # predicate can ALSO scan it for the OUR_CODE_FRAME exclusion). ``None`` on a
    # healthy / running poll (populated only on a ``status="dead"`` poll). Read by
    # ``backend_poll._maybe_escalate_runpod_cuda_ima`` after the RunPod lane
    # copies it through ``RunPodBackend.poll``. Declared LAST so existing
    # positional ``PollResult(...)`` constructions are unaffected.
    crash_signature: str | None = None
    # #983: True when THIS tick posted the [post-done-phase-advisory] marker —
    # a poll AFTER a corroborated done observed genuinely NEW [phase=...]
    # lines after the recorded done line (the .py-dispatcher subprocess
    # fan-out false-done class, #930 §4.6 residual gap (i)). Observability
    # only; never changes ``status``. Defaulted so cross-backend
    # PollResult(...) call sites need no change.
    post_done_phase_advisory_posted: bool = False
    # The new-phase-lines the #983 guard observed THIS tick — surfaced
    # regardless of the once-per-episode dedup (an already-advised episode
    # still reports what it sees). Empty when no episode is active.
    post_done_phase_lines: tuple[str, ...] = ()
    # True when the pod-side pid file's mtime predates the newest
    # epm:run-launched marker by > PID_FILE_MARKER_SLACK_SEC (#1156 — the
    # #813 stale-pid-after-relaunch shape). Observability only; status
    # routing unchanged. Defaulted so cross-backend PollResult(...) call
    # sites need no change.
    pid_file_stale_vs_marker: bool = False


def _ssh_probe(
    pod: str,
    log_path: str,
    pid_file: str,
    issue: int,
    marker_pid: int | None = None,
    *,
    stall_sec: int = DEFAULT_STALL_SEC,
) -> dict[str, str]:
    """One SSH round-trip — returns dict with keys pid_alive,
    marker_pid_alive, mtime_epoch, cell_mtime_epoch, log_tail,
    cell_log_tail, phase_log_mtime_epoch, phase_log_tail,
    shard_log_mtime_epoch, shard_log_tail, gpu_util.

    ``stall_sec`` sizes the output-artifact freshness window (#1033 —
    ``output_mtime_epoch`` below); it does NOT change any log probe.

    Batches into a single heredoc to keep the SSH cost to one connection.

    Liveness keys:
    * ``pid_alive`` — liveness of the PID stored in ``pid_file``.
    * ``pid_file_missing`` — ``"1"`` when ``pid_file`` does not exist on
      the pod; ``PID_ALIVE=0`` then means "no pidfile to probe", NOT
      "pid probed dead". Observability-only (incident #521: a false
      ``status=dead`` on a healthy run was hard to diagnose because the
      tick collapsed "file absent, marker-pid fallback in effect" into
      a bare ``pid_alive=False``). ``"0"`` on the SSH-failure fail-safe
      path — transport failure means "unknown", not "missing".
    * ``pid_file_mtime_epoch`` — pod-clock mtime (``stat -c %Y``) of
      ``pid_file``, feeding the #1156 stale-pid-file-vs-marker WARN.
      ``"0"`` (always inert) when the file is absent / ``stat`` failed /
      SSH failed / a legacy probe replay omits the line.
    * ``marker_pid_alive`` — liveness of ``marker_pid`` (the PID carried
      by the latest epm:run-launched marker) when one is supplied. The
      marker-pid probe is the self-correction path for a stale pidfile.

    Liveness-of-output keys (used to broaden the stall verdict so a long
    healthy phase that writes only to a per-cell or per-phase log is not
    false-failed as stalled, incidents #405 + #468):

    * ``mtime_epoch`` — top-level log mtime (still drives the milestone /
      phase-line parse).
    * ``cell_mtime_epoch`` — mtime of the freshest per-cell log under
      ``<log_path stripped of .log>/cell_*.log`` (the smoke-first /
      sequential-cell convention; #405). ``"0"`` when no cell logs exist.
    * ``cell_log_tail`` — tail of that same freshest cell log; used by
      ``poll_once`` as the ``log_tail_excerpt`` source when the cell log
      is fresher than the main log. Permission / nullglob /
      no-such-directory cases silently degrade to ``0`` + empty tail
      (and the caller falls back to the main-log mtime alone) — matching
      how the existing main-log probe degrades on a missing log file.
    * ``phase_log_mtime_epoch`` — max mtime over per-phase logs matching
      ``/workspace/logs/issue-<issue>-*.log`` (excluding ``*.json`` /
      ``*.processed`` sentinels, and the top-level
      ``issue-<issue>.log`` itself). ``"0"`` when no per-phase log
      exists yet. Complements ``cell_mtime_epoch``: cell logs live
      under ``<log_path%.log>/cell_*.log`` (nested), per-phase logs
      live flat at ``/workspace/logs/issue-<N>-<phase>.log``; the two
      globs don't overlap.
    * ``phase_log_tail`` — tail of that same freshest per-phase log; the
      #791 sibling of ``cell_log_tail``. Consumed by
      ``_tail_excerpt_and_crash_signature`` so the notification excerpt +
      the ``status=dead`` crash signature track the freshest per-phase log
      when a later run arm writes only to it, instead of a stale main-log
      tail. ``""`` when no per-phase log exists (same degrade as
      ``cell_log_tail``).
    * ``shard_log_mtime_epoch`` — max mtime over repo-rooted shard /
      phase / per-job logs (incidents #488 + #521). Covers three extra
      layouts neither the cell-log nor the per-phase-log probe sees:
      (1) ``/workspace/explore-persona-space/logs/issue_<issue>/*.log``
      — nested subdirectory holding per-GPU shard logs (e.g.
      ``phase1_g0.log``..``phase1_g7.log``);
      (2) ``/workspace/explore-persona-space/logs/issue_<issue>_*.log``
      — flat repo-rooted phase logs (e.g. ``issue_<N>_phase0.log``,
      the #331 / #444 family layout);
      (3) ``/workspace/explore-persona-space/eval_results/
      issue_<issue>{,_*}/logs/*.log`` — dispatcher per-job logs
      (``<output_dir>/logs/<job>.log``, the issue_519/521 dispatcher
      convention; #521), the only fresh signal during a CPU-bound
      judge-batch wait with GPUs idle by design.
      Excludes ``*.json`` / ``*.processed`` sentinels. ``"0"`` when no
      covered layout exists. All patterns share an mtime reduction
      (max), so a healthy run keeping ANY layout fresh stays in
      ``running``.
    * ``shard_log_tail`` — tail of that same freshest shard/phase/per-job
      log; the #791 sibling of ``cell_log_tail`` / ``phase_log_tail``,
      consumed by ``_tail_excerpt_and_crash_signature`` the same way.
      ``""`` when no covered layout exists.
    * ``output_mtime_epoch`` — mtime of A recently-modified OUTPUT
      artifact under the issue-keyed output roots
      (``eval_results/issue_<N>{,_*}/``, ``data/issue_<N>/``,
      ``data/issue<N>/`` under the repo root), found via a bounded,
      short-circuit ``find -newermt ... -print -quit`` under ``timeout``
      (#1033; incident #813 — a ~6h CPU-bound analysis tail wrote
      per-cell NPZs / JSONs while every log was quiet). NOTE: this is
      the mtime of *a fresh file, not the newest* — ``-print -quit``
      stops at the FIRST file inside the freshness window, which is all
      the threshold reads downstream need (``output_mtime_ago`` is only
      ever compared against the same windows the find used). ``"0"``
      when no file was modified within ``max(stall_sec,
      ZOMBIE_VETO_FRESH_SEC)``, the dirs are missing, the find timed
      out, or the fold is disabled (``EPM_POLL_OUTPUT_MTIME_FOLD=0`` —
      the probe block is omitted entirely). Fail-safe: ``"0"`` reads as
      "no fresh output" -> pre-#1033 behavior.
    * ``gpu_util`` — comma-separated per-GPU ``utilization.gpu``
      integers (e.g. ``"95,87,42,90"``). ``"unknown"`` when
      ``nvidia-smi`` is unavailable or errors (fail-safe — see
      ``_gpu_idle``).
    * ``results_sentinel_present`` — ``"1"`` when at least one results
      sentinel ``/workspace/logs/issue-<N>-epm_results-*.json[.processed]``
      exists on the pod, else ``"0"``. Corroboration for the ``done``
      verdict (incident #545): a `[phase=done]` parse with the pid still
      alive is reported ``done`` only when a results sentinel exists —
      otherwise it is mid-run per-cell noise. ``.processed`` files count
      because the SAME tick's sentinel drain renames the file moments
      before the status decision (and the corroboration must survive
      later ticks while a post-done dispatcher lingers). ``"0"`` on the
      SSH-failure fail-safe path (the done branch is unreachable there —
      an empty log tail parses to phase ``unknown``).
    * ``session_cpu_secs`` — cumulative CPU seconds (as a float string,
      e.g. ``"4271.5"``) summed across every process in the launcher
      PID's process SESSION (`setsid` group). The launcher itself
      accrues ~no CPU — its children carry the work — so summing over
      the session captures every descendant regardless of how the
      python child re-execs. ``"unknown"`` when the launcher PID is
      not alive (no session to probe) or when ``ps`` is unavailable /
      errors (fail-safe — see ``_session_cpu_advancing``). Used as a
      defense against false-stalled verdicts on silent CPU-bound
      phases: even when every log mtime exceeds the stall threshold
      AND the GPUs are idle, a session whose cumulative CPU time has
      advanced since the previous tick is doing work, not hanging
      (incident #518 scoring_syco phase, 2026-06-10 — a healthy run
      with cumulative CPU time advancing 1:1 with wall time was
      false-declared stalled because no log line appeared for ~7.8h).
    """
    marker_probe = ""
    if marker_pid is not None:
        marker_probe = (
            f"if ps -p {marker_pid} > /dev/null 2>&1; "
            f"then echo MARKER_PID_ALIVE=1; else echo MARKER_PID_ALIVE=0; fi; "
        )
    # Cell-log probe: strip a trailing `.log` from log_path to get the
    # per-cell log directory (the dispatch_sweep convention used since
    # #405 smoke-first runs). `shopt -s nullglob` makes the empty case
    # expand to nothing rather than the literal pattern. We pick the
    # single freshest cell log via `stat -c '%Y %n'` + `sort -n` and
    # emit its mtime + its tail, so the caller has both the staleness
    # signal AND a tail to surface when the main log is the stale one.
    cell_probe = (
        'CELL_LOG_DIR="${LOG_PATH%.log}"; '
        "shopt -s nullglob; "
        'CELL_FILES=("$CELL_LOG_DIR"/cell_*.log); '
        "if [ ${#CELL_FILES[@]} -gt 0 ]; then "
        '  FRESHEST=$(stat -c "%Y %n" "${CELL_FILES[@]}" 2>/dev/null | sort -n | tail -1); '
        '  CELL_MTIME="${FRESHEST%% *}"; '
        '  CELL_PATH="${FRESHEST#* }"; '
        '  echo "CELL_MTIME_EPOCH=${CELL_MTIME:-0}"; '
        "  echo CELL_TAIL_START; "
        '  if [ -n "$CELL_PATH" ] && [ -f "$CELL_PATH" ]; then tail -500 "$CELL_PATH"; fi; '
        "  echo CELL_TAIL_END; "
        "else "
        "  echo CELL_MTIME_EPOCH=0; echo CELL_TAIL_START; echo CELL_TAIL_END; "
        "fi; "
    )
    # Per-phase-log probe (#468): max mtime across
    # `/workspace/logs/issue-<issue>-*.log`, excluding the top-level
    # `issue-<issue>.log` itself and any `*.json` / `*.processed`
    # sentinels (the sentinel naming uses `-*-*.json[.processed]` so
    # `*.log` already excludes them; the explicit `case` defends against
    # accidental `.log.json` etc.). `shopt -s nullglob` makes an empty
    # glob expand to nothing rather than the literal pattern. `sort -n
    # | tail -1` yields the max epoch, or "" when no per-phase log
    # exists; the `echo` then prints "PHASE_LOG_MTIME_EPOCH=0" (parsed
    # as 0 by the caller).
    # We select the SINGLE freshest matching file (by `stat -c '%Y %n' | sort
    # -n | tail -1`, mirroring `cell_probe` at line 918) and emit BOTH its
    # mtime AND its tail between `PHASE_TAIL_START`/`PHASE_TAIL_END`, so the
    # caller has the staleness signal AND a tail to surface / scan for a crash
    # signature when the per-phase log is the freshest one (#791: the tail
    # excerpt + `status=dead` crash signature were pinned to the main log even
    # when a later run arm wrote only to a per-phase log). No glob change —
    # reuse the existing narrow `issue-<N>-*` pattern (vetted #468) so a
    # cross-pod log on shared FS can never pollute the tail.
    phase_log_probe = (
        f"PHASE_FRESHEST=$("
        f"shopt -s nullglob; "
        f"for f in /workspace/logs/issue-{issue}-*.log; do "
        f'  case "$f" in *.processed|*.json) continue ;; esac; '
        f'  case "$f" in /workspace/logs/issue-{issue}.log) continue ;; esac; '
        f'  stat -c "%Y %n" "$f" 2>/dev/null; '
        f"done | sort -n | tail -1); "
        f'PHASE_LOG_MAX="${{PHASE_FRESHEST%% *}}"; '
        f'PHASE_LOG_PATH="${{PHASE_FRESHEST#* }}"; '
        f'echo "PHASE_LOG_MTIME_EPOCH=${{PHASE_LOG_MAX:-0}}"; '
        f"echo PHASE_TAIL_START; "
        f'if [ -n "$PHASE_LOG_PATH" ] && [ -f "$PHASE_LOG_PATH" ]; then '
        f'tail -500 "$PHASE_LOG_PATH"; fi; '
        f"echo PHASE_TAIL_END; "
    )
    # Shard-log probe (#488): the i488 multi-GPU layout writes per-GPU
    # shard logs under `/workspace/explore-persona-space/logs/issue_<N>/
    # phase*_g*.log` (nested subdirectory, underscore separator), and the
    # #331/#444 family writes flat repo-rooted phase logs at
    # `/workspace/explore-persona-space/logs/issue_<N>_*.log`. Neither
    # pattern is reached by the `phase_log_probe` glob above, so the
    # i488 Pass B inner loop (~3 min between shard writes across 57
    # cells per shard) silently tripped the 36-min main-log threshold
    # while every shard log was actively being written (2026-06-07).
    # We probe BOTH layouts and reduce to the max mtime; either layout
    # being fresh keeps the verdict in `running`. The match is narrow on
    # purpose — paths must embed `issue_<N>` (underscore) under the repo
    # logs directory, so unrelated logs from other pods don't pollute
    # the freshness signal.
    #
    # Dispatcher per-job logs (#521): the issue_519/521-style dispatcher
    # writes one log per job under `<output_dir>/logs/*.log`, with
    # `output_dir` typically `eval_results/issue_<N>` under the repo
    # root. During a CPU-bound judge-batch wait (GPUs idle by design,
    # main log quiet) the per-job log is the ONLY fresh signal — a #521
    # tick false-declared `stalled` on a healthy EM-steering job
    # (2026-06-10) because no probe reached it. Folded into the same
    # SHARD_LOG max. The two extra globs keep the issue-number match
    # exact (`issue_<N>` or `issue_<N>_<suffix>`; a bare `issue_<N>*`
    # would let issue 5 match issue 521's directories).
    # Emit the freshest matching shard log's mtime AND its tail between
    # `SHARD_TAIL_START`/`SHARD_TAIL_END` (same shape as `cell_probe` /
    # `phase_log_probe`; #791). No glob change — reuse the existing narrow
    # `issue_<N>` patterns (vetted #488/#521) so a cross-pod log never
    # pollutes the tail.
    shard_log_probe = (
        f"SHARD_FRESHEST=$("
        f"shopt -s nullglob; "
        f"for f in /workspace/explore-persona-space/logs/issue_{issue}/*.log "
        f"         /workspace/explore-persona-space/logs/issue_{issue}_*.log "
        f"         /workspace/explore-persona-space/eval_results/issue_{issue}/logs/*.log "
        f"         /workspace/explore-persona-space/eval_results/issue_{issue}_*/logs/*.log; do "
        f'  case "$f" in *.processed|*.json) continue ;; esac; '
        f'  stat -c "%Y %n" "$f" 2>/dev/null; '
        f"done | sort -n | tail -1); "
        f'SHARD_LOG_MAX="${{SHARD_FRESHEST%% *}}"; '
        f'SHARD_LOG_PATH="${{SHARD_FRESHEST#* }}"; '
        f'echo "SHARD_LOG_MTIME_EPOCH=${{SHARD_LOG_MAX:-0}}"; '
        f"echo SHARD_TAIL_START; "
        f'if [ -n "$SHARD_LOG_PATH" ] && [ -f "$SHARD_LOG_PATH" ]; then '
        f'tail -500 "$SHARD_LOG_PATH"; fi; '
        f"echo SHARD_TAIL_END; "
    )
    # GPU util probe (#468): fail-safe to "unknown" so a missing /
    # erroring nvidia-smi never declares stalled by itself (the
    # per-phase-log + cell-log signals still protect long phases). See
    # `_gpu_idle` for the threshold + fail-safe semantics.
    #
    # Zombie-GPU-allocation probe (#664): a hung vLLM whose CUDA worker
    # process DIED but whose EngineCore main process is still alive
    # presents as a compute process holding many GiB of VRAM whose PID no
    # longer exists in `/proc`. The main Python process keeps burning
    # Python-overhead CPU (HTTP keepalive, GIL ticks, network-thread-pool
    # idle work), so the #518/#658 session-CPU-advancing override keeps
    # the verdict in `running` indefinitely while zero real work happens
    # (#664 round-8 hung 60+ min reported healthy throughout). The only
    # mechanical signal of the hang is the orphaned GPU allocation: a
    # compute-apps PID with no `/proc/<pid>` entry. We list `pid,
    # used_memory` and emit (space-separated) every PID that holds
    # >= ZOMBIE_GPU_MEM_MIN_MIB MiB but is absent from `/proc` — the
    # memory floor avoids flagging trivial leftover CUDA contexts. A live
    # process ALWAYS has `/proc/<pid>`, so an absent dir is a hard
    # liveness signal, not a heuristic. Empty (no zombies) is the healthy
    # case. Fail-safe: nvidia-smi missing / erroring emits an empty list
    # (never a false zombie), same posture as the util probe.
    #
    # Namespace-informativeness counts (#864): the same loop also counts
    # GPU_PIDS_TOTAL (every valid compute-apps PID) and GPU_PIDS_RESOLVABLE
    # (those with a `/proc/<pid>` dir) so the VM-side gate can tell whether
    # the dead-in-/proc signal is even meaningful on this pod — on a
    # host-PID-namespace container ZERO compute PIDs ever resolve. When (and
    # only when) a zombie candidate exists, a guarded scan additionally
    # counts NVIDIA_UVM_LIVE_HOLDERS: live container processes holding a fd
    # whose symlink target is EXACTLY `/dev/nvidia-uvm` (a live CUDA compute
    # context holds the UVM device; NVML monitors open nvidiactl/nvidiaN
    # instead). The match is END-ANCHORED (` -> /dev/nvidia-uvm$`) so
    # `/dev/nvidia-uvm-tools` (also created by the nvidia_uvm module, held by
    # profilers / cuda-gdb / UVM-tools consumers), `/dev/nvidiactl`, and
    # `/dev/nvidia[0-9]` NEVER count — a tools-only holder counting would
    # satisfy the veto triple during a genuine total collapse and silently
    # suppress the #664 true positive. Healthy matched-regime ticks pay zero
    # cost (the scan is skipped without a candidate); a dying-mid-scan proc
    # is skipped by `2>/dev/null` (fails toward not counting, i.e. toward
    # the #826 fall-through). The no-nvidia-smi else-branch emits the three
    # keys as `unknown` so the parser's fail-safe defaults engage.
    gpu_probe = (
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "  GPU_OUT=$(nvidia-smi --query-gpu=utilization.gpu "
        "    --format=csv,noheader,nounits 2>/dev/null | paste -sd, -); "
        '  echo "GPU_UTIL=${GPU_OUT:-unknown}"; '
        "  ZOMBIE=''; GPU_PIDS_TOTAL=0; GPU_PIDS_RESOLVABLE=0; "
        "  while IFS=, read -r zpid zmem; do "
        '    zpid=$(echo "$zpid" | tr -d " "); '
        '    zmem=$(echo "$zmem" | tr -d " "); '
        '    case "$zpid" in ""|*[!0-9]*) continue ;; esac; '
        '    case "$zmem" in ""|*[!0-9]*) zmem=0 ;; esac; '
        "    GPU_PIDS_TOTAL=$((GPU_PIDS_TOTAL+1)); "
        "    if [ -d /proc/$zpid ]; then "
        "      GPU_PIDS_RESOLVABLE=$((GPU_PIDS_RESOLVABLE+1)); "
        f'    elif [ "$zmem" -ge {ZOMBIE_GPU_MEM_MIN_MIB} ]; then '
        '      ZOMBIE="$ZOMBIE $zpid"; '
        "    fi; "
        "  done <<EOF\n"
        "$(nvidia-smi --query-compute-apps=pid,used_memory "
        "  --format=csv,noheader,nounits 2>/dev/null)\n"
        "EOF\n"
        "  UVM_HOLDERS=unknown; "
        '  if [ -n "$ZOMBIE" ]; then '
        "    UVM_HOLDERS=0; "
        "    for p in /proc/[0-9]*; do "
        '      if ls -l "$p/fd" 2>/dev/null | grep -q " -> /dev/nvidia-uvm$"; then '
        "        UVM_HOLDERS=$((UVM_HOLDERS+1)); "
        "      fi; "
        "    done; "
        "  fi; "
        "  echo \"ZOMBIE_GPU_PIDS=$(echo $ZOMBIE | tr -s ' ')\"; "
        '  echo "GPU_PIDS_TOTAL=$GPU_PIDS_TOTAL"; '
        '  echo "GPU_PIDS_RESOLVABLE=$GPU_PIDS_RESOLVABLE"; '
        '  echo "NVIDIA_UVM_LIVE_HOLDERS=$UVM_HOLDERS"; '
        'else echo "GPU_UTIL=unknown"; echo "ZOMBIE_GPU_PIDS="; '
        'echo "GPU_PIDS_TOTAL=unknown"; echo "GPU_PIDS_RESOLVABLE=unknown"; '
        'echo "NVIDIA_UVM_LIVE_HOLDERS=unknown"; fi; '
    )
    # Session CPU probe (#518): cumulative CPU seconds summed across
    # every process sharing the launcher PID's session id (SID). The
    # launcher is started with `setsid nohup bash <launcher>` (see
    # `.claude/agents/experimenter.md` "Launch") so every descendant
    # — the python child, vLLM workers, judge subprocesses, etc. —
    # carries the same SID as the launcher PID itself. `ps -o sess=`
    # reads that SID; `etime` field is wall-clock; `time` field is
    # cumulative CPU. We filter the full `ps -e` output by SID and
    # sum `time` (HH:MM:SS, or D-HH:MM:SS for >1 day) into seconds.
    #
    # ``unknown`` when (a) the pidfile is missing / pid is dead — no
    # session to probe; the launcher exiting clean is `phase=done` /
    # `dead` territory and the stall arbiter never reaches this
    # signal — or (b) `ps` is unavailable / errors. The
    # `_session_cpu_advancing` decision fails safe to "no signal" in
    # those cases (the older log + GPU arbiters then carry the
    # verdict, preserving the pre-#518 behavior).
    session_cpu_probe = (
        f"if [ -f {pid_file} ]; then "
        f"  LPID=$(cat {pid_file}); "
        f"  SID=$(ps -o sess= -p $LPID 2>/dev/null | tr -d ' '); "
        f'  if [ -n "$SID" ] && [ "$SID" != "0" ]; then '
        f"    CPU_SUM=$(ps -e -o sess=,time= 2>/dev/null | "
        f'      awk -v s="$SID" \'$1==s {{ '
        f'        n=split($2,a,":"); '
        f"        if (n==3) {{ secs += a[1]*3600 + a[2]*60 + a[3] }} "
        f"        else if (n==2) {{ secs += a[1]*60 + a[2] }} "
        f"        else if (n==1) {{ "
        f'          m=split(a[1],b,"-"); '
        f"          if (m==2) {{ secs += b[1]*86400 + b[2] }} "
        f"          else {{ secs += a[1] }} "
        f"        }} "
        f"      }} END {{ "
        f'        if (NR==0) {{ print "unknown" }} '
        f'        else {{ printf "%.1f", secs }} '
        f"      }}'); "
        f'    echo "SESSION_CPU_SECS=${{CPU_SUM:-unknown}}"; '
        f'  else echo "SESSION_CPU_SECS=unknown"; fi; '
        f'else echo "SESSION_CPU_SECS=unknown"; fi; '
    )
    # Results-sentinel presence probe (#545): corroboration for the `done`
    # verdict. Matches BOTH the unprocessed `.json` and the drained
    # `.json.processed` forms — the drain at the top of `poll_once` renames
    # the sentinel before the status decision runs, so the unprocessed form
    # alone would never corroborate the happy path. `shopt -s nullglob` is
    # set explicitly (not inherited from cell_probe's earlier shopt) so an
    # empty glob yields array length 0, not the length-1 literal pattern.
    results_sentinel_probe = (
        f"shopt -s nullglob; "
        f"RS_FILES=(/workspace/logs/issue-{issue}-epm_results-*.json*); "
        f"if [ ${{#RS_FILES[@]}} -gt 0 ]; then echo RESULTS_SENTINEL_PRESENT=1; "
        f"else echo RESULTS_SENTINEL_PRESENT=0; fi; "
    )
    # Output-artifact freshness probe (#1033): a CPU-bound analysis tail that
    # writes per-cell NPZs / JSONs / .done sentinels while every log is quiet
    # (#813). Bounded: short-circuit find (-print -quit) under `timeout`;
    # issue-keyed roots ONLY (same narrowness contract as the #488/#521 shard
    # globs — a bare `issue_<N>*` would let issue 5 match issue 521's dirs, so
    # the set is exactly `issue_{N}`, `issue_{N}_*`, and the `data/issue{N}`
    # no-underscore convention from the #854 crash-persist sweep). The block
    # captures its OWN pod-clock epoch (`OUT_NOW`; the POD_NOW_EPOCH line in
    # the heredoc is echo-only — no shell var exists), so the cutoffs are on
    # the pod clock with no VM skew. Two-stage: prefer a within-stall hit (it
    # rescues the stall conjunction); fall back to the WIDER zombie-veto
    # window only when that window is genuinely wider (stall_sec < the 60s
    # ZOMBIE_VETO_FRESH_SEC floor — fast-smoke configs; the default 900s
    # stall window already covers the veto read, so the common case is ONE
    # find). `-print -quit` short-circuits on the FIRST fresh file (see the
    # docstring: a fresh file, not the newest); on a healthy actively-writing
    # phase this returns almost immediately, and on a genuinely stalled run
    # the metadata scan is bounded by `timeout` (worst case 2 stages inside
    # the 60s SSH exec budget). Missing dirs / timeout / a hit deleted before
    # `stat` -> OUTPUT_MTIME_EPOCH=0 (fail-safe: pre-#1033 behavior).
    if OUTPUT_MTIME_FOLD_ENABLED:
        find_timeout = _output_find_timeout_sec()
        out_dirs = (
            f"/workspace/explore-persona-space/eval_results/issue_{issue} "
            f"/workspace/explore-persona-space/eval_results/issue_{issue}_* "
            f"/workspace/explore-persona-space/data/issue_{issue} "
            f"/workspace/explore-persona-space/data/issue{issue}"
        )
        veto_window_sec = max(ZOMBIE_VETO_FRESH_SEC, stall_sec)
        # Second stage only when the veto window is genuinely wider than the
        # stall window (nullglob is (re)set below, so the unmatched
        # `issue_{N}_*` glob vanishes instead of passing a literal pattern;
        # the three literal paths always remain, so `find` can never run
        # with ZERO path args and default to a broad `.` scan).
        second_stage = ""
        if veto_window_sec > stall_sec:
            second_stage = (
                f'if [ -z "$OUT_HIT" ]; then '
                f"OUT_CUTOFF_VETO=$((OUT_NOW - {veto_window_sec})); "
                f"OUT_HIT=$(timeout {find_timeout} find {out_dirs} "
                f'-type f -newermt "@$OUT_CUTOFF_VETO" -print -quit 2>/dev/null); '
                f"fi; "
            )
        output_probe = (
            f"shopt -s nullglob; "
            f"OUT_NOW=$(date +%s); "
            f"OUT_CUTOFF_STALL=$((OUT_NOW - {stall_sec})); "
            f"OUT_HIT=$(timeout {find_timeout} find {out_dirs} "
            f'-type f -newermt "@$OUT_CUTOFF_STALL" -print -quit 2>/dev/null); '
            f"{second_stage}"
            f'if [ -n "$OUT_HIT" ]; then '
            f'echo "OUTPUT_MTIME_EPOCH=$(stat -c %Y "$OUT_HIT" 2>/dev/null || echo 0)"; '
            f"else echo OUTPUT_MTIME_EPOCH=0; fi; "
        )
    else:
        output_probe = ""  # kill switch: parser defaults the key to "0"
    heredoc = (
        f"LOG_PATH={log_path}; "
        f"if [ -f {pid_file} ]; then "
        f"  echo PID_FILE_MISSING=0; PID=$(cat {pid_file}); "
        f"  echo PID_FILE_MTIME_EPOCH=$(stat -c %Y {pid_file} 2>/dev/null || echo 0); "
        f"  if ps -p $PID > /dev/null 2>&1; then echo PID_ALIVE=1; else echo PID_ALIVE=0; fi; "
        f"else echo PID_FILE_MISSING=1; echo PID_ALIVE=0; fi; "
        f"{marker_probe}"
        f"if [ -f $LOG_PATH ]; then "
        f"  echo MTIME_EPOCH=$(stat -c %Y $LOG_PATH); "
        f"  echo TAIL_START; tail -500 $LOG_PATH; echo TAIL_END; "
        f"else echo MTIME_EPOCH=0; echo TAIL_START; echo TAIL_END; fi; "
        # Capture the pod's own wall clock (#704). `stat -c %Y` above stamps
        # file mtimes from this same clock and `date +%s` reads it within
        # milliseconds in the same SSH session, so subtracting
        # `pod_now - pod_mtime` downstream cancels any pod-vs-VM clock drift
        # exactly. One capture covers all four mtime sources.
        f"echo POD_NOW_EPOCH=$(date +%s); "
        f"{cell_probe}"
        f"{phase_log_probe}"
        f"{shard_log_probe}"
        f"{gpu_probe}"
        f"{session_cpu_probe}"
        f"{results_sentinel_probe}"
        f"{output_probe}"
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", pod, heredoc],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if result.returncode != 0:
        log.error("ssh failed (rc=%d): %s", result.returncode, result.stderr.strip())
        # ``ssh_failed`` is the explicit caller signal so ``poll_once`` can
        # count consecutive transport failures (#488 stale-port auto-heal)
        # WITHOUT having to infer "ssh down" from the zeroed values below
        # (which can also legitimately mean "log file does not exist yet").
        return {
            "pid_alive": "0",
            "pid_file_missing": "0",
            "pid_file_mtime_epoch": "0",
            "marker_pid_alive": "0",
            "mtime_epoch": "0",
            "pod_now_epoch": "0",
            "cell_mtime_epoch": "0",
            "log_tail": "",
            "cell_log_tail": "",
            "phase_log_mtime_epoch": "0",
            "phase_log_tail": "",
            "shard_log_mtime_epoch": "0",
            "shard_log_tail": "",
            "gpu_util": "unknown",
            "zombie_gpu_pids": "",
            "gpu_pids_total": "unknown",
            "gpu_pids_resolvable": "unknown",
            "nvidia_uvm_live_holders": "unknown",
            "session_cpu_secs": "unknown",
            "results_sentinel_present": "0",
            "output_mtime_epoch": "0",
            "ssh_failed": "1",
        }
    parsed = _parse_probe_stdout(result.stdout)
    parsed["ssh_failed"] = "0"
    return parsed


# Scalar `KEY=value` lines the probe heredoc emits. Order is irrelevant —
# the parser dispatches on the prefix and stores the trailing value.
_PROBE_SCALAR_KEYS: tuple[str, ...] = (
    "PID_ALIVE",
    "PID_FILE_MISSING",
    "PID_FILE_MTIME_EPOCH",
    "MARKER_PID_ALIVE",
    "MTIME_EPOCH",
    "POD_NOW_EPOCH",
    "CELL_MTIME_EPOCH",
    "PHASE_LOG_MTIME_EPOCH",
    "SHARD_LOG_MTIME_EPOCH",
    "GPU_UTIL",
    "ZOMBIE_GPU_PIDS",
    "GPU_PIDS_TOTAL",
    "GPU_PIDS_RESOLVABLE",
    "NVIDIA_UVM_LIVE_HOLDERS",
    "SESSION_CPU_SECS",
    "RESULTS_SENTINEL_PRESENT",
    "OUTPUT_MTIME_EPOCH",
)


def _parse_probe_stdout(stdout: str) -> dict[str, str]:
    """Parse the probe heredoc's stdout into the dict ``_ssh_probe`` returns.

    Factored out of ``_ssh_probe`` to keep the SSH call site simple and
    drive the parser's complexity below the C901 cap. Pure / stdout-only;
    no I/O.
    """
    parsed: dict[str, str] = {
        "pid_alive": "0",
        "pid_file_missing": "0",
        "pid_file_mtime_epoch": "0",
        "marker_pid_alive": "0",
        "mtime_epoch": "0",
        "pod_now_epoch": "0",
        "cell_mtime_epoch": "0",
        "log_tail": "",
        "cell_log_tail": "",
        "phase_log_mtime_epoch": "0",
        "phase_log_tail": "",
        "shard_log_mtime_epoch": "0",
        "shard_log_tail": "",
        "gpu_util": "unknown",
        "zombie_gpu_pids": "",
        "gpu_pids_total": "unknown",
        "gpu_pids_resolvable": "unknown",
        "nvidia_uvm_live_holders": "unknown",
        "session_cpu_secs": "unknown",
        "results_sentinel_present": "0",
        "output_mtime_epoch": "0",
    }
    # Each multi-line tail block is delimited by its own START/END sentinel
    # (``TAIL_START``/``END``, ``CELL_TAIL_START``/``END``, and the #791
    # ``PHASE_TAIL``/``SHARD_TAIL`` pairs). The blocks never nest (the probe
    # emits them sequentially), so at most one is active at a time. A
    # data-driven table (sentinel -> accumulator) keeps the per-line dispatch
    # a single lookup — adding a tail block is a table row, not another
    # if-branch (which is what pushed this past the C901 cap in #791).
    tail_accumulators: dict[str, list[str]] = {
        "log_tail": [],
        "cell_log_tail": [],
        "phase_log_tail": [],
        "shard_log_tail": [],
    }
    tail_starts = {
        "TAIL_START": "log_tail",
        "CELL_TAIL_START": "cell_log_tail",
        "PHASE_TAIL_START": "phase_log_tail",
        "SHARD_TAIL_START": "shard_log_tail",
    }
    tail_ends = {"TAIL_END", "CELL_TAIL_END", "PHASE_TAIL_END", "SHARD_TAIL_END"}
    active: list[str] | None = None
    for line in stdout.splitlines():
        if line in tail_starts:
            active = tail_accumulators[tail_starts[line]]
            continue
        if line in tail_ends:
            active = None
            continue
        if active is not None:
            active.append(line)
            continue
        # Dispatch on the `KEY=value` prefix; store under the lowercased key.
        for key in _PROBE_SCALAR_KEYS:
            if line.startswith(f"{key}="):
                parsed[key.lower()] = line.split("=", 1)[1].strip()
                break
    for tail_key, lines in tail_accumulators.items():
        parsed[tail_key] = "\n".join(lines)
    return parsed


def sentinel_drain_shell(issue: int, extra_globs: tuple[str, ...] = ()) -> str:
    """The in-VM list+cat loop every drain transport executes.

    Globs ``/workspace/logs/issue-<issue>-*.json`` (skipping ``*.processed``)
    and emits each file as ``SENTINEL_START <path>\\n<body>\\nSENTINEL_END``
    so :func:`parse_sentinel_stream` can split multiple sentinels out of one
    stdout blob. Shared by the pod-SSH transport (:func:`_ssh_drain_sentinels`)
    and the GCP gcloud-ssh transport (``backends.gcp`` — which wraps it in
    ``sudo -n bash -c`` because the GCE startup script writes the sentinel
    tree as root, mode 600; incident #608) so the two lanes can never drift
    on the loop shape. The SLURM lane deliberately has NO drain transport:
    compute nodes have no ``/workspace`` and the robot forced-command
    wrapper cannot execute this shell — see ``backends/slurm_monitor.py``
    § "No sentinel drain on this lane" (#608 follow-up).

    ``extra_globs`` appends transport-specific fallback patterns to the
    canonical glob (incident #610: the issue-610 GCP dispatcher found
    ``/workspace/logs`` missing and wrote its results sentinel under its
    out_root ``.../eval_results/issue_610/logs/`` instead, so the drain
    reported ``done`` with ``sentinels_processed=0``). Patterns are
    TRUSTED, UNQUOTED shell globs (quoting would defeat expansion):
    callers pass only config-derived paths with no spaces/metacharacters,
    e.g. the GCP workload-root fallback in ``backends/gcp.py``. The
    default — no extras — keeps the RunPod lane byte-identical.

    Each glob is path-terminal `.json` and explicitly excludes `.processed`.
    ``shopt -s nullglob`` makes an empty glob expand to nothing instead of
    the literal pattern so we don't accidentally cat a path called e.g.
    ``/workspace/logs/issue-444-*.json``.
    """
    globs = " ".join([f"/workspace/logs/issue-{issue}-*.json", *extra_globs])
    return (
        f"shopt -s nullglob; "
        f"for f in {globs}; do "
        f'  case "$f" in *.processed) continue ;; esac; '
        f'  echo "SENTINEL_START $f"; '
        f'  cat "$f"; '
        f'  echo ""; echo "SENTINEL_END"; '
        f"done"
    )


def parse_sentinel_stream(stdout: str) -> list[tuple[str, str]]:
    """Parse :func:`sentinel_drain_shell` output into ``(path, body)`` pairs.

    Lines outside a ``SENTINEL_START``/``SENTINEL_END`` block are ignored,
    so a transport may append its own trailer sections (e.g. the GCP
    drain's log-tail section) after the loop output.
    """
    sentinels: list[tuple[str, str]] = []
    current_path: str | None = None
    current_body: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("SENTINEL_START "):
            current_path = line[len("SENTINEL_START ") :].strip()
            current_body = []
        elif line == "SENTINEL_END":
            if current_path is not None:
                sentinels.append((current_path, "\n".join(current_body).strip()))
            current_path = None
            current_body = []
        elif current_path is not None:
            current_body.append(line)
    return sentinels


def _ssh_drain_sentinels(pod: str, issue: int) -> list[tuple[str, str]]:
    """List + cat unprocessed sentinels in one SSH round-trip.

    Runs :func:`sentinel_drain_shell` on the pod and parses the stdout via
    :func:`parse_sentinel_stream`. Files are NOT renamed here — the rename
    happens via ``_ssh_mark_processed`` only after the marker post succeeds,
    so a mid-tick crash leaves the sentinel un-renamed and the next poll
    retries it (idempotent).

    Returns a list of ``(remote_path, body)`` pairs (possibly empty). On
    SSH failure returns an empty list and logs the error.
    """
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", pod, sentinel_drain_shell(issue)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if result.returncode != 0:
        log.error("ssh drain failed (rc=%d): %s", result.returncode, result.stderr.strip())
        return []
    return parse_sentinel_stream(result.stdout)


def _ssh_mark_processed(pod: str, remote_path: str) -> bool:
    """Rename ``remote_path`` -> ``remote_path + '.processed'`` on the pod.

    Returns True on success. Logs + returns False on failure (the sentinel
    is left in place; next poll tick will re-attempt). We use ``mv -n`` (no
    clobber) so a pre-existing ``.processed`` file is preserved — the
    sentinel writer never reuses epoch-tagged filenames, so a collision
    here would itself be a bug worth surfacing.
    """
    # Single-quote the remote path to neutralise shell metacharacters; the
    # writer's filename is ``issue-<N>-<kind_slug>-<epoch>.json`` so it's
    # safe by construction, but defence-in-depth costs nothing.
    quoted = "'" + remote_path.replace("'", "'\\''") + "'"
    cmd = f"mv -n {quoted} {quoted}.processed"
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", pod, cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if result.returncode != 0:
        log.error(
            "ssh mv failed for %s (rc=%d): %s",
            remote_path,
            result.returncode,
            result.stderr.strip(),
        )
        return False
    return True


def _slugify_kind(kind: str) -> str:
    """Match the pod-side sentinel writer's `kind` slug (``:`` -> ``_``).

    Used to name the persisted oversize-note artifact (``sentinel-note-<slug>-
    <epoch>.txt``) so the artifact file's stem mirrors the sentinel filename
    convention. Pure / no I/O.
    """
    return kind.replace(":", "_")


def _sentinel_fingerprint(remote_path: str, body: str) -> str:
    """Stable idempotency key for ONE drained sentinel (#1084).

    Hashes the remote path (unique per sentinel — epoch-suffixed filenames,
    never reused by the writer contract) + the raw parser-delivered body
    string. Hashing the raw string (never a re-serialized JSON dict) makes
    the fp immune to key-ordering drift; including the path makes
    cross-sentinel false-positive dedupe structurally impossible (two
    byte-identical bodies at different paths are distinct signals). Pure /
    no I/O; 16 hex chars (64 bits) — collision odds negligible at this
    volume.
    """
    return hashlib.sha256(f"{remote_path}\n{body}".encode("utf-8", "replace")).hexdigest()[:16]


def _posted_sentinel_fps(issue: int) -> dict[str, dict[str, Any]]:
    """Map ``sentinel_fp`` -> already-posted event row for task ``issue``.

    Reads ``events.jsonl`` once via ``list_events`` (the tolerant reader —
    malformed lines skipped). FAIL-OPEN: on ANY read failure, ``log.error``
    and return ``{}`` — degraded mode re-posts (today's pre-#1084 behavior);
    a lost marker is never a possible outcome of a dedupe-read failure.
    """
    try:
        events = list_events(issue)
        return {e["sentinel_fp"]: e for e in events if isinstance(e.get("sentinel_fp"), str)}
    except Exception as exc:
        log.error(
            "dedupe events read failed for #%d (%s); posting without dedupe (fail-open, #1084)",
            issue,
            exc,
        )
        return {}


_RESULTS_REWRITE_KIND = "epm:results"
# "phase" keeps the #641 non-blocking phase-progress shape (gate="phase",
# blocks_pipeline: False) excluded by NAME.
_SMOKE_GATE_NAMES = frozenset({"smoke", "dryrun", "dry-run", "dry_run", "phase"})


def _results_rewrite_exclusion_legs(remote_path: str, data: dict[str, Any]) -> dict[str, bool]:
    """Per-leg smoke/informational signals for the #1095 rewrite exclusion.

    Exposed as a dict so tests can assert EXACTLY which leg is active
    (the exclusion-leg test-construction discipline) without re-implementing
    leg logic. A bare ``blocks_pipeline: False`` is deliberately NOT a leg:
    real terminal epm:results writers set it to keep a non-empty gate from
    parking the poll loop (issue664/734/654/597) — it does not discriminate
    smoke from real. The drain never descends into ``note`` (it may be a
    JSON string, not a dict): a smoke flag nested in note (issue597) is the
    documented residual — smoke runs should write kind ``epm:smoke-result``
    (`.claude/rules/pod-side-reporting.md`).
    """
    basename = remote_path.rsplit("/", 1)[-1]
    return {
        "gate_name": str(data.get("gate") or "").strip().lower() in _SMOKE_GATE_NAMES,
        "smoke_field": bool(data.get("smoke")),  # issue667-style top-level smoke flag
        "smoke_filename": basename.endswith("-smoke-results.json") or "-smoke-" in basename,
    }


def _results_version_rewrite_excluded(remote_path: str, data: dict[str, Any]) -> bool:
    """True when a real epm:results sentinel must KEEP its declared version.

    Smoke / dry-run / phase-progress sentinels must never bump real
    ``epm:results`` marker versions (#1095): a rewrite would land them ABOVE
    the production results rows, making a smoke row the
    highest-version-authoritative marker on resume (markers.md). Any leg
    suffices; over-exclusion is safe (it preserves verbatim threading).
    """
    return any(_results_rewrite_exclusion_legs(remote_path, data).values())


def _existing_max_version(issue: int, kind: str) -> int | None:
    """Max events.jsonl ``version`` for ``kind`` (0 when none); ``None`` on a
    failed read — the caller FAILS OPEN to verbatim threading (#1095),
    mirroring ``_posted_sentinel_fps``'s fail-open contract."""
    try:
        events = list_events(issue)
    except Exception as exc:
        log.error(
            "version-collision events read failed for #%d (%s); keeping the "
            "sentinel's declared version verbatim (fail-open, #1095)",
            issue,
            exc,
        )
        return None
    return max(
        (
            e["version"]
            for e in events
            if e.get("kind") == kind and isinstance(e.get("version"), int)
        ),
        default=0,
    )


def _persist_oversize_note(
    *,
    issue: int,
    remote_path: str,
    kind: str,
    version: int | None,
    by: str,
    full_note: str,
    original_extras: dict[str, Any] | None = None,
    declared_version: int | None = None,
    sentinel_fp: str,
    sentinel_path: str,
) -> bool:
    """Graceful-degradation for an oversize sentinel ``note``.

    Triggered when ``task_workflow.post_event`` raises ``ValueError`` because
    ``note`` exceeds ``EVENT_NOTE_MAX`` (currently 50,000 chars). Without
    this fallback, ``_drain_sentinels`` would leave the sentinel un-renamed
    and every poll tick would re-post + re-fail the same oversize payload
    forever (incident 2026-06-04 task #477: a 52001-char
    ``epm:progress`` aggregate sentinel cycled indefinitely).

    Strategy:

    1. Write ``full_note`` to ``<task>/artifacts/sentinel-note-<kind_slug>-
       <epoch>.txt`` (task folder resolved via ``find_task_path``, so the
       branch-guarded ``main`` resolver picks the correct path even when
       the poller is invoked from elsewhere).
    2. Post a SHORT pointer marker of the same ``(kind, version)``
       (``version=None`` — the #899 synthesized-envelope case — lets
       ``post_event`` derive max+1 at the pointer post; safe because the
       oversize ``ValueError`` raises BEFORE any version is derived or
       consumed) whose
       ``note`` (a) cites the artifact path, (b) records original length,
       and (c) is a leading excerpt of the original. The excerpt is
       hard-bounded under ``EVENT_NOTE_MAX`` so the pointer post itself
       cannot trip the same cap. ``artifacts=[<rel_path>]`` and
       ``oversize=True`` are carried as marker extras so the dashboard /
       downstream consumers can locate the full payload. The pointer post
       ALSO carries the drain's ``sentinel_fp`` / ``sentinel_path``
       idempotency extras (#1084), so a re-drain of an oversize sentinel
       whose rename failed dedupes on the fp — no second artifact file, no
       second pointer marker.

    Returns ``True`` on success (artifact written + pointer marker posted).
    Returns ``False`` (and logs) on any failure — caller must NOT rename
    the sentinel in that case so a future tick can retry. Carries through
    the original sentinel's ``gate`` / ``blocks_pipeline`` semantics by
    asking the caller to forward those via ``original_extras``. When the
    caller's #1095 version rewrite fired (``version=None`` on a REAL
    ``epm:results`` sentinel), it forwards the declared version via
    ``declared_version`` so the pointer marker keeps the
    ``sentinel_declared_version`` audit extra too.
    """
    try:
        task_dir = find_task_path(issue)
    except Exception as exc:
        log.error(
            "could not resolve task #%d for oversize-note persistence (sentinel %s, kind=%s): %s",
            issue,
            remote_path,
            kind,
            exc,
        )
        return False

    artifacts_dir = task_dir / "artifacts"
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error(
            "could not create artifacts/ for task #%d (sentinel %s): %s",
            issue,
            remote_path,
            exc,
        )
        return False

    epoch = int(datetime.now(tz=UTC).timestamp())
    artifact_name = f"sentinel-note-{_slugify_kind(kind)}-{epoch}.txt"
    artifact_path = artifacts_dir / artifact_name
    try:
        artifact_path.write_text(full_note, encoding="utf-8")
    except OSError as exc:
        log.error(
            "could not write oversize-note artifact %s (sentinel %s): %s",
            artifact_path,
            remote_path,
            exc,
        )
        return False

    # Compute repo-relative artifact path for the marker. Falls back to the
    # absolute path if relative resolution fails (e.g. unusual mounts).
    try:
        rel_artifact = str(artifact_path.relative_to(task_dir.parents[2]))
    except ValueError:
        rel_artifact = str(artifact_path)

    # Build the pointer-marker note. It MUST fit under EVENT_NOTE_MAX. We
    # reserve ~512 chars for the pointer header and use the remainder for
    # a leading excerpt of the original, so operators see the start of the
    # payload inline without needing to open the artifact.
    version_repr = str(version) if version is not None else "derived-at-post (max+1)"
    header = (
        f"[oversize note persisted; original {len(full_note)} chars > "
        f"{EVENT_NOTE_MAX} cap]\n"
        f"Full payload: {rel_artifact}\n"
        f"Original kind={kind} version={version_repr} by={by}\n"
        f"--- leading excerpt ---\n"
    )
    excerpt_budget = max(0, EVENT_NOTE_MAX - len(header) - 32)  # 32-byte safety
    excerpt = full_note[:excerpt_budget]
    pointer_note = header + excerpt
    # Belt-and-suspenders: hard truncate if any accounting drift would push
    # the pointer marker itself over the cap.
    if len(pointer_note) > EVENT_NOTE_MAX:
        pointer_note = pointer_note[:EVENT_NOTE_MAX]

    extras: dict[str, Any] = {"oversize": True, "oversize_orig_len": len(full_note)}
    if declared_version is not None:
        # #1095: keep the version-rewrite audit trail on the pointer marker.
        extras["sentinel_declared_version"] = declared_version
    if original_extras:
        # Forward operationally-meaningful sentinel fields (notably ``gate``
        # and ``blocks_pipeline``) so the pointer marker preserves the
        # semantics of the original.
        for key in ("gate", "blocks_pipeline"):
            if key in original_extras and original_extras[key] is not None:
                extras[key] = original_extras[key]

    try:
        post_event(
            issue,
            kind,
            version=version,
            by=by,
            note=pointer_note,
            artifacts=[rel_artifact],
            sentinel_fp=sentinel_fp,
            sentinel_path=sentinel_path,
            **extras,
        )
    except Exception as exc:
        log.error(
            "pointer-marker post failed for oversize sentinel %s (kind=%s): %s",
            remote_path,
            kind,
            exc,
        )
        return False

    log.warning(
        "sentinel %s carried %d-char note (> %d cap); persisted to %s and "
        "posted truncated pointer marker (kind=%s).",
        remote_path,
        len(full_note),
        EVENT_NOTE_MAX,
        rel_artifact,
        kind,
    )
    return True


def _parse_sentinel(remote_path: str, body: str) -> dict[str, Any] | None:
    """Decode + validate one sentinel body. Returns the dict on success.

    Returns None (and logs) for any of: empty body, JSON decode error,
    non-dict payload, missing required keys, unsupported schema version.
    The sentinel is left un-renamed in these cases so a future poller (or
    a human) can inspect it.

    Exception (#899): a fully-envelope-less body on a ``-results.json``
    basename that carries the complete Step 7 results-payload key set is
    RESCUED via ``_maybe_synthesize_results_envelope`` (returns a
    synthesized ``kind=epm:results`` envelope with NO ``version`` key — the
    drain then posts with ``version=None`` so ``post_event`` derives max+1
    (#975) — plus a loud ``log.error``) instead of being skipped.
    Envelope-carrying sentinels
    (including partial envelopes and unsupported schema versions) keep the
    strict path unchanged.
    """
    if not body:
        log.warning("sentinel %s is empty; skipping", remote_path)
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        log.warning("sentinel %s has invalid JSON (%s); skipping", remote_path, exc)
        return None
    if not isinstance(data, dict):
        log.warning("sentinel %s is not a JSON object; skipping", remote_path)
        return None
    missing = [k for k in _SENTINEL_REQUIRED_KEYS if k not in data]
    if missing:
        fallback = _maybe_synthesize_results_envelope(remote_path, data, missing)
        if fallback is not None:
            return fallback
        log.warning("sentinel %s missing required keys %s; skipping", remote_path, missing)
        return None
    schema_version = data.get("sentinel_schema_version")
    if schema_version != SENTINEL_SCHEMA_VERSION_SUPPORTED:
        log.warning(
            "sentinel %s has unsupported schema_version=%r (supported: %d); skipping",
            remote_path,
            schema_version,
            SENTINEL_SCHEMA_VERSION_SUPPORTED,
        )
        return None
    return data


def _post_drained_sentinel(
    *,
    issue: int,
    remote_path: str,
    data: dict[str, Any],
    fp: str,
) -> dict[str, Any] | None:
    """Post ONE parsed sentinel's marker (normal or oversize-pointer path).

    Extracted from :func:`drain_sentinels_via`'s loop (#1084). Returns the
    in-memory fp record the caller accumulates into the drain's dedupe map
    on success (normal post AND oversize-pointer success both count), or
    ``None`` on a retryable post failure — the caller leaves the sentinel
    un-renamed and continues; the next tick retries. A non-conforming real
    sentinel carrying ``"version": null`` keeps its loud ``int(None)``
    TypeError (#975) — it propagates out of the drain, never silently
    derived. A REAL ``epm:results`` sentinel whose declared version
    collides at-or-below the existing max for the kind is re-derived as
    max+1 at post time, with the declared version preserved as the
    ``sentinel_declared_version`` marker extra (#1095) — smoke/dryrun/
    phase-progress sentinels and every other kind post verbatim.
    """
    kind = data["kind"]
    rewrite_extras: dict[str, Any] = {}
    if "version" in data:
        # Real pod-side sentinel: "version" is a REQUIRED envelope key
        # (_SENTINEL_REQUIRED_KEYS / pod-side-reporting.md). int() runs
        # FIRST so a non-conforming "version": null keeps today's loud
        # TypeError (#975) — never silently derived, for ANY kind.
        version: int | None = int(data["version"])
        # #1095: pod-side dispatchers hardcode version 1 (they cannot read
        # events.jsonl — the no-pod-side-task.py rule), so on multi-round
        # tasks a REAL epm:results sentinel lands BELOW the existing max,
        # silently violating the highest-version-per-kind resume convention
        # (markers.md; the #480 collision class; #825 landed three v1 rows
        # under an existing v2). On an at-or-below-max collision, derive
        # max+1 at post (version=None -> post_event derives under flock)
        # and preserve the declared version as a forensic extra. A declared
        # version ABOVE max is legitimate threading — verbatim. Smoke /
        # dryrun / phase-progress sentinels are excluded (gate NAME, a
        # truthy top-level smoke field, or the smoke filename — NOT a bare
        # blocks_pipeline: False, which real terminal writers set): they
        # must never land above the production results rows. Any other
        # kind: verbatim (explicit always wins in post_event — unchanged
        # contract).
        if kind == _RESULTS_REWRITE_KIND and not _results_version_rewrite_excluded(
            remote_path, data
        ):
            existing_max = _existing_max_version(issue, kind)
            if existing_max is not None and version <= existing_max:
                log.warning(
                    "real %s sentinel %s (ts=%s) declared version %d <= existing "
                    "max %d; deriving max+1 at post (multi-round collision, "
                    "#1095); declared version preserved as "
                    "sentinel_declared_version",
                    kind,
                    remote_path,
                    data.get("ts"),
                    version,
                    existing_max,
                )
                rewrite_extras["sentinel_declared_version"] = version
                version = None
    else:
        # Only reachable via the #899 synthesized envelope, which omits
        # the key (#975): post_event derives max(existing for kind)+1 so
        # a re-drained / re-run rescue never lands BELOW an existing
        # higher version (the #480 collision class). If the rename fails,
        # a future tick's re-drain fp-hits on the ``sentinel_fp`` extra
        # and skips the re-post entirely (#1084) — no duplicate at a
        # fresh max+1 anymore.
        version = None
    note = data.get("note")
    if note is None:
        note = data.get("payload")
    if note is not None and not isinstance(note, str):
        note = json.dumps(note, ensure_ascii=False)
    by = data.get("by") or "pod-sentinel"
    try:
        # Structural note: this post threads NO ``part`` field, so multi-part
        # markers (the markers.md multi-part convention, where every part
        # shares one explicit version) are impossible on the sentinel-drain
        # channel by construction — the #1095 rewrite can never split a
        # multi-part set across versions here.
        post_event(
            issue,
            kind,
            version=version,
            by=by,
            note=note,
            sentinel_fp=fp,
            sentinel_path=remote_path,
            **rewrite_extras,
        )
    except ValueError as exc:
        # Oversize-note guard: match the EXACT message ``post_event``
        # raises (``"event note exceeds {N} chars (...)"``). Routing
        # any-old ``ValueError`` to graceful-degradation would
        # silently swallow real schema bugs, so the substring match
        # stays narrow.
        if _OVERSIZE_NOTE_ERROR_SUBSTR not in str(exc) or note is None:
            log.error(
                "post_event failed for sentinel %s (kind=%s): %s",
                remote_path,
                kind,
                exc,
            )
            return None
        if not _persist_oversize_note(
            issue=issue,
            remote_path=remote_path,
            kind=kind,
            version=version,
            by=by,
            full_note=note,
            original_extras=data,
            declared_version=rewrite_extras.get("sentinel_declared_version"),
            sentinel_fp=fp,
            sentinel_path=remote_path,
        ):
            # Persistence / pointer-post failed — retry next tick.
            return None
        # Pointer marker posted from the persisted artifact — success.
    except Exception as exc:
        # Don't rename on post failure — next tick will retry. We log
        # at error so an operator can see repeated failures.
        log.error(
            "post_event failed for sentinel %s (kind=%s): %s",
            remote_path,
            kind,
            exc,
        )
        return None
    return {
        "kind": kind,
        "version": version,
        "sentinel_path": remote_path,
        "ts": "(this drain)",
    }


def drain_sentinels_via(
    *,
    issue: int,
    list_sentinels: Callable[[], list[tuple[str, str]]],
    mark_processed: Callable[[str], bool],
) -> tuple[int, str | None]:
    """Transport-agnostic sentinel drain; post markers from the VM.

    ``list_sentinels`` returns ``(remote_path, body)`` pairs (the transport:
    pod SSH for RunPod, ``gcloud compute ssh ... sudo -n`` for GCP — the GCE
    startup script writes the sentinel tree root-owned mode 600, so a plain
    user-mode read comes back empty; incident #608). ``mark_processed``
    renames one remote path to ``<path>.processed`` and returns success.

    Returns ``(processed_count, gate_name_or_None)``. ``gate_name`` is the
    first non-empty ``gate`` field across processed sentinels that ALSO
    carries ``blocks_pipeline: True`` (the field defaults to True when
    absent, preserving the original gate-only semantics). Sentinels are
    processed in glob order, which is filename order, which is
    chronological by epoch-suffix. When set, the caller should stop the
    polling loop and surface the gate to the user. A non-empty ``gate``
    NAME paired with ``blocks_pipeline: False`` is a benign phase-progress
    signal (``gate=phase`` / ``gate=smoke`` / ``gate=dryrun``): the marker
    is still posted from the VM, but the gate is NOT surfaced and the
    polling loop continues (incident #641).

    Each successfully-posted sentinel is renamed to ``<path>.processed``
    so the next tick won't re-post the same marker. If the marker POST
    fails for an individual sentinel, the sentinel is left in place and a
    warning is logged; subsequent ticks retry the whole path. If the post
    SUCCEEDS but the rename fails (or the poller crashes between the two —
    the #952 W1 window), the next tick's re-drain is IDEMPOTENT (#1084):
    every drain-posted event carries a ``sentinel_fp`` extra (sha256 of
    ``remote_path + "\\n" + raw body``, via :func:`_sentinel_fingerprint`)
    plus a ``sentinel_path`` extra, and the drain skips the re-post on an
    fp hit (loud ``log.error`` naming the matched event) and retries the
    rename only — exactly-once PER DRAINER (the dedupe read sits outside
    ``post_event``'s flock, so two CONCURRENT drainers can still race to a
    duplicate; loud, benign, and identical to pre-#1084 behavior). A
    ``mark_processed`` callable that RAISES (e.g. ``subprocess.
    TimeoutExpired`` from a hung SSH transport) is treated as a rename
    failure — loud log, sentinel left in place, loop continues to later
    sentinels. A ``list_sentinels`` callable that raises
    ``subprocess.TimeoutExpired`` / ``OSError`` (the hung RunPod transport)
    yields an empty drain (``(0, None)``) with a loud log — mirroring the
    documented rc!=0 -> ``[]`` transport contract; any OTHER exception
    still escapes (fail-fast for genuine code bugs).

    Version hygiene (#1095): a REAL ``epm:results`` sentinel (not smoke/
    dryrun/phase-progress) whose declared version collides at-or-below the
    existing max for the kind lands at a derived max+1 with the declared
    version preserved as ``sentinel_declared_version`` — see
    :func:`_post_drained_sentinel`; all other kinds post verbatim.

    Exception: an oversize-``note`` ``ValueError`` from ``post_event`` (note
    exceeds ``EVENT_NOTE_MAX``) is NOT a retryable failure — re-posting the
    same oversize payload next tick will fail identically, looping forever
    (incident 2026-06-04 task #477: a 52001-char ``epm:progress`` aggregate
    sentinel cycled indefinitely). It is degraded gracefully via
    ``_persist_oversize_note`` (full note -> ``<task>/artifacts/sentinel-
    note-*.txt`` + a truncated pointer marker of the same ``(kind, version)``
    — or a freshly derived version for the synthesized-envelope case, #975 —
    that cites the artifact) and the sentinel is renamed ``.processed`` to
    end the loop. Any OTHER ``post_event`` exception (transient infra,
    schema bug, etc.) keeps the original retry-on-next-tick semantics.
    """
    try:
        sentinels = list_sentinels()
    except (subprocess.TimeoutExpired, OSError) as exc:
        # W2a (#1084): the RunPod drain transport hung past its subprocess
        # timeout (or failed at the OS layer). Mirror the documented
        # rc!=0 -> [] contract: empty drain, loud log, retry next tick.
        # The catch is deliberately NARROW — any other exception is a
        # genuine code bug and must still crash loud (fail-fast).
        log.error(
            "sentinel drain transport failed for issue %d: %s — empty drain, retry next tick",
            issue,
            exc,
        )
        return 0, None
    processed = 0
    gate: str | None = None
    # Lazy idempotency map (#1084): ONE events.jsonl read per drain
    # invocation, taken only when at least one sentinel parsed. The map is
    # ALSO updated in-loop after every successful post, so a duplicate
    # (remote_path, body) tuple within ONE listing posts exactly once
    # (currently unreachable — the canonical + fallback globs are disjoint —
    # but a future overlapping ``extra_globs`` must not reopen the
    # same-tick double-post).
    posted_fps: dict[str, dict[str, Any]] | None = None
    for remote_path, body in sentinels:
        data = _parse_sentinel(remote_path, body)
        if data is None:
            continue
        fp = _sentinel_fingerprint(remote_path, body)
        if posted_fps is None:
            posted_fps = _posted_sentinel_fps(issue)
        dup = posted_fps.get(fp)
        if dup is not None:
            # W1 crash-safe replay (#1084): the marker for THIS exact
            # sentinel (same path + same body) already posted on a prior
            # tick whose rename failed/crashed (or earlier THIS drain).
            # LOUD skip — fail-fast philosophy: never silent — then fall
            # through to the rename + accounting + gate extraction below
            # WITHOUT re-posting. Exactly-once per drainer. Note the
            # fp/dup check runs BEFORE the version-presence branch, so a
            # dup replay never re-executes ``int(data["version"])`` (a
            # null-version sentinel crashes before ever posting, so it can
            # never be a dup — the #975 TypeError pin is unaffected).
            log.error(
                "sentinel %s already posted as %s v%s at %s (fp=%s); skipping re-post, "
                "renaming only (crash-safe replay, #1084)",
                remote_path,
                dup.get("kind"),
                dup.get("version"),
                dup.get("ts"),
                fp,
            )
        else:
            if any(e.get("sentinel_path") == remote_path for e in posted_fps.values()):
                # Same path, different content: a writer-contract violation
                # (filenames are epoch-suffixed and never reused). Fail-open
                # + loud: the new content is a distinct signal, so post it.
                log.warning(
                    "sentinel %s content CHANGED at a previously-posted path (writer "
                    "contract violation); posting anyway",
                    remote_path,
                )
            fp_record = _post_drained_sentinel(
                issue=issue, remote_path=remote_path, data=data, fp=fp
            )
            if fp_record is None:
                # Retryable post failure — sentinel left un-renamed so the
                # next tick can retry the whole path.
                continue
            # Record the just-posted fp so a duplicate (path, body) tuple
            # LATER IN THIS SAME DRAIN dedupes (#1084).
            posted_fps[fp] = fp_record
        try:
            renamed_ok = mark_processed(remote_path)
        except Exception as exc:
            # W2b (#1084): a hung/raising rename transport (e.g.
            # subprocess.TimeoutExpired from a wedged SSH) is treated as a
            # rename FAILURE — the documented False path below. Loud, and
            # the loop CONTINUES to later sentinels; the marker is already
            # posted, so the next tick's replay is idempotent (fp dedupe).
            log.error(
                "mark_processed raised for %s: %s — treating as rename failure "
                "(marker posted; idempotent replay next tick)",
                remote_path,
                exc,
            )
            renamed_ok = False
        if not renamed_ok:
            # Marker is posted but rename failed. The next tick re-drains
            # this sentinel and replays IDEMPOTENTLY (#1084): the fp dedupe
            # skips the re-post and retries the rename only — no duplicate
            # event. Surface loudly so an operator can rename manually if
            # the transport failure persists.
            log.error(
                "marker posted from sentinel %s but rename failed; next tick "
                "will replay idempotently (fp dedupe skips the re-post, rename "
                "only). Rename to %s.processed manually on the remote host to "
                "end the retries.",
                remote_path,
                remote_path,
            )
            # Still count as processed so the caller's accounting is honest.
        processed += 1
        # Only surface a sentinel's ``gate`` as a poll-loop-ending user gate
        # when the dispatcher flagged it ``blocks_pipeline: True`` (default
        # True when the field is absent — preserves the original semantics
        # for sentinels that carry only a non-empty ``gate``). Newer
        # dispatchers emit benign phase-progress signals as a non-empty gate
        # NAME (``gate=phase`` / ``gate=smoke`` / ``gate=dryrun``) WITH
        # ``blocks_pipeline: False``: those markers post from the VM but must
        # NOT end the polling loop or park the orchestrator at a gate
        # (incident #641 — a strict orchestrator reading SKILL.md would have
        # blocked mid-training on the canonical phase-progress sentinel).
        sentinel_gate = data.get("gate")
        blocks_pipeline = data.get("blocks_pipeline", True)
        if gate is None and isinstance(sentinel_gate, str) and sentinel_gate and blocks_pipeline:
            gate = sentinel_gate
    return processed, gate


def _drain_sentinels(*, issue: int, pod: str) -> tuple[int, str | None]:
    """Drain pod-side sentinels over the RunPod SSH transport.

    Thin wrapper binding :func:`drain_sentinels_via` to the pod-SSH
    transport (``_ssh_drain_sentinels`` / ``_ssh_mark_processed``). The
    lambdas resolve the module-level names at call time, so tests that
    monkeypatch them keep working unchanged.
    """
    return drain_sentinels_via(
        issue=issue,
        list_sentinels=lambda: _ssh_drain_sentinels(pod, issue),
        mark_processed=lambda remote_path: _ssh_mark_processed(pod, remote_path),
    )


def latest_phase(log_tail: str, *, skip_done: bool = False) -> str:
    """Return the milestone name from the most recent `[phase=...]` line, or 'unknown'.

    PUBLIC cross-module contract: consumed by
    ``src/explore_persona_space/backends/gcp.py`` (the relaunched-workload
    done-corroboration probe, #612) in addition to this module's
    ``poll_once``. Renaming or changing the signature requires updating
    that import; ``_latest_phase`` remains as a back-compat alias.

    ``skip_done=True`` returns the most recent NON-``done`` milestone
    instead — used by ``poll_once`` to demote an UNCORROBORATED done-parse
    (pid alive + no results sentinel) back to the real current phase, so a
    mid-run per-cell ``[phase=done] eval cell <X> complete`` noise line
    (incident #545) neither flips the status verdict nor posts a false
    ``-> done`` milestone transition.

    A done-bearing line matching ``DONE_QUOTED_NOISE_RE`` (a failure
    message QUOTING the token, e.g. ``... FAILED rc=1 - [phase=done] NOT
    emitted`` — incident #597) is skipped unconditionally: it is not a
    phase transition, so the scan falls back to the previous real phase
    line and a crashed wrapper with a dead pid decays to ``dead`` instead
    of a false ``done``.
    """
    for line in reversed(log_tail.splitlines()):
        m = PHASE_RE.search(line)
        if not m:
            continue
        token = m.group(1)
        if token == "done" and DONE_QUOTED_NOISE_RE.search(line):
            continue  # failure message quoting the literal token (#597)
        if skip_done and token == "done":
            continue
        return token
    return "unknown"


# Back-compat alias for the pre-#612 private name (tests + any external
# caller still importing ``_latest_phase`` keep working unchanged).
_latest_phase = latest_phase


# ── #983 post-done phase-consistency guard ──────────────────────────────────
#
# The #545 corroboration block gates the INITIAL done verdict within a tick,
# and the #597 noise regex gates which line may parse as done at all — both
# are parse-time defenses. This guard is the CROSS-TICK audit they are
# structurally blind to: once a corroborated ``status=done`` has been
# accepted, a LATER poll that observes genuinely NEW ``[phase=...]`` lines
# AFTER the recorded done line proves the earlier done may have been FALSE —
# the ``.py``-dispatcher subprocess fan-out class (#930 §4.6 residual gap
# (i): a parent exits (pid-dead corroborates) or its sentinel lands early
# while detached children keep emitting phase lines). Advisory-only: ONE
# loud ``[post-done-phase-advisory]`` ``epm:progress`` marker + best-effort
# Telegram push per done-episode; the status verdict is NEVER changed (the
# same contract as every sibling tripwire in this file).

# Chars stored/compared per phase-bearing line (bounded state file). The
# record and compare sides BOTH truncate through ``_phase_bearing_lines``,
# so a done line longer than the cap still anchors by identity on re-polls.
_POST_DONE_LINE_MAX = 400
# New phase lines quoted in the advisory note (each further capped to 200
# chars there), keeping the note far below EVENT_NOTE_MAX.
_POST_DONE_NOTE_MAX_LINES = 5


def _phase_bearing_lines(log_tail: str) -> list[str]:
    """Ordered (oldest -> newest) raw texts of phase-bearing lines.

    Same per-line predicate as ``latest_phase``: PHASE_RE must match; a
    done-token line also matching DONE_QUOTED_NOISE_RE (#597 failure message
    quoting the token) is skipped. Truncated to ``_POST_DONE_LINE_MAX`` so
    the state-file identity comparison is bounded. ``latest_phase`` itself
    is untouched (public cross-module contract).
    """
    out: list[str] = []
    for line in log_tail.splitlines():
        m = PHASE_RE.search(line)
        if not m:
            continue
        if m.group(1) == "done" and DONE_QUOTED_NOISE_RE.search(line):
            continue
        out.append(line[:_POST_DONE_LINE_MAX])
    return out


@dataclass(frozen=True)
class PostDonePhaseUpdate:
    """Outcome of one post-done-guard tick (``_post_done_phase_update``)."""

    should_post: bool
    done_line: str  # "" = no active done episode
    done_epoch: int  # 0 = no active episode
    done_pod: str  # "" = no active episode; the episode voids on a pod change
    advisory_posted: bool  # once-per-episode dedup flag, carried forward
    new_phase_lines: tuple[str, ...]  # observed-this-tick (surfaced even when deduped)


def _post_done_phase_update(
    *,
    current_phase: str,
    log_tail: str,
    pod: str,
    prev_done_line: str,
    prev_done_epoch: int,
    prev_done_pod: str,
    prev_posted: bool,
    run_age_sec: float | None,
    now_epoch: int,
) -> PostDonePhaseUpdate:
    """Pure decision core for the #983 post-done phase-consistency guard.

    ``current_phase`` MUST be the POST-#545-corroboration phase from
    ``poll_once`` (never a raw ``latest_phase`` re-derivation), so an
    uncorroborated done-parse (pid alive + no results sentinel — e.g. a
    mid-run per-cell ``[phase=done] eval cell <X> complete`` noise line)
    can NEVER arm an episode. The episode-start condition is deliberately
    the corroborated ``current_phase == "done"``, NOT ``status == "done"``,
    so a gate-sentinel tick whose pipeline also reached done (gate wins the
    status precedence) still records the episode. Once set, the episode is
    never overwritten within a run; only the run-scope clamp or a pod
    change voids it. Comparison is line-identity anchored (positions are
    unstable under the ``tail -500`` sliding window; timestamps are not
    guaranteed by PHASE_RE): candidates are the phase-bearing lines
    strictly after the anchor's LAST occurrence, or ALL visible phase
    lines when the anchor scrolled out (append-only log => everything
    above it scrolled out first). Pure / no I/O — the caller owns state
    persistence and the marker post.
    """
    # Run-scope clamp (mirrors the last_phase_change_epoch relaunch clamp in
    # poll_once): an episode recorded BEFORE the current run's
    # epm:run-launched belongs to a previous run — void it.
    if (
        prev_done_epoch > 0
        and run_age_sec is not None
        and prev_done_epoch < now_epoch - run_age_sec
    ):
        prev_done_line, prev_done_epoch, prev_done_pod, prev_posted = "", 0, "", False
    # Cross-pod voiding: a diagnostic poll against a DIFFERENT pod — or a
    # follow-up pod probed before its epm:run-launched lands — must not
    # compare the new pod's tail against the old pod's done anchor.
    if prev_done_line and prev_done_pod and prev_done_pod != pod:
        prev_done_line, prev_done_epoch, prev_done_pod, prev_posted = "", 0, "", False
    if not prev_done_line:
        if current_phase == "done":
            lines = _phase_bearing_lines(log_tail)
            if lines:
                # By construction (the reversed scan in latest_phase) the
                # LAST phase-bearing non-noise line IS the matched done line.
                return PostDonePhaseUpdate(False, lines[-1], now_epoch, pod, False, ())
        return PostDonePhaseUpdate(False, "", 0, "", False, ())
    # Active episode: find phase lines strictly AFTER the recorded done line.
    lines = _phase_bearing_lines(log_tail)
    try:
        anchor = len(lines) - 1 - lines[::-1].index(prev_done_line)  # LAST occurrence
        candidates = lines[anchor + 1 :]
    except ValueError:
        # Done line scrolled out of the bounded tail: the log is append-only,
        # so every line ABOVE it scrolled out first — any phase line still
        # visible is NEWER than the recorded done.
        candidates = lines
    new_lines = tuple(ln for ln in candidates if ln != prev_done_line)  # FP control (i)
    return PostDonePhaseUpdate(
        bool(new_lines) and not prev_posted,
        prev_done_line,
        prev_done_epoch,
        prev_done_pod,
        prev_posted,
        new_lines,
    )


def _maybe_post_post_done_phase_advisory(
    *,
    issue: int,
    pod: str,
    current_phase: str,
    log_tail: str,
    prev_state: dict[str, str],
    run_age_sec: float | None,
    now_epoch: int,
) -> tuple[str, int, str, bool, bool, tuple[str, ...]]:
    """Post-done-guard wiring for ``poll_once``: parse state, decide, maybe post.

    Returns ``(done_line, done_epoch, done_pod, posted_flag,
    posted_this_tick, new_phase_lines)`` for the caller to persist via
    ``_save_state``. Guarded state parses (a corrupt epoch resets to 0,
    never raises into ``poll_once``). A post failure is logged and
    ``posted_flag`` is NOT set, so the next tick retries — identical
    contract to ``_maybe_post_gpu_width_advisory``. Advisory only: never
    changes the status verdict, never stops anything.
    """
    prev_line = prev_state.get("post_done_line", "") or ""
    try:
        prev_epoch = int(float(prev_state.get("post_done_epoch", "0") or 0))
    except (TypeError, ValueError):
        prev_epoch = 0
    prev_pod = prev_state.get("post_done_pod", "") or ""
    prev_posted = prev_state.get("post_done_advisory_posted", "0") == "1"
    u = _post_done_phase_update(
        current_phase=current_phase,
        log_tail=log_tail,
        pod=pod,
        prev_done_line=prev_line,
        prev_done_epoch=prev_epoch,
        prev_done_pod=prev_pod,
        prev_posted=prev_posted,
        run_age_sec=run_age_sec,
        now_epoch=now_epoch,
    )
    if not u.should_post:
        return u.done_line, u.done_epoch, u.done_pod, u.advisory_posted, False, u.new_phase_lines
    quoted = "\n".join(f"  {ln[:200]}" for ln in u.new_phase_lines[:_POST_DONE_NOTE_MAX_LINES])
    note = (
        f"[post-done-phase-advisory] {len(u.new_phase_lines)} NEW [phase=...] line(s) appeared "
        f"AFTER the done line this poller reported as terminal "
        f"({max(0, now_epoch - u.done_epoch) // 60} min ago):\n{quoted}\n"
        f"recorded done line: {u.done_line[:200]}\n"
        "The earlier status=done may have been FALSE (the .py-dispatcher subprocess fan-out "
        "class — workflow_lint --check-phase-done-reserved residual gap (i), #930/#545): a "
        "child script may still be running, and any orchestrator action keyed on the done "
        "(advance to verifying, Step-8 pod termination) may have been premature. OTHER causes "
        "with the same signature: a relaunch / manual re-run reused this log path without a "
        "fresh epm:run-launched, or a concurrent writer appended to it — check the launch "
        "record before chasing the dispatcher. VERIFY the run actually completed (results "
        "sentinel + uploads) and fix the emitting dispatcher "
        "per pod-side-reporting.md ([phase=done] is reserved for the single terminal line). "
        "Advisory only: this tick's status verdict is unchanged and nothing was stopped."
    )
    try:
        post_event(
            issue,
            "epm:progress",
            by="poll_pipeline",
            note=note,
            phase=current_phase,
            pod=pod,
            post_done_phase_advisory=True,
        )
    except Exception as exc:
        log.error("post-done phase advisory post failed (next tick will retry): %s", exc)
        return u.done_line, u.done_epoch, u.done_pod, False, False, u.new_phase_lines
    # Fail-soft phone push — never blocks recording (the marker is durable).
    _telegram_push(
        f"[#{issue}] post-done phase advisory: {len(u.new_phase_lines)} new [phase=...] "
        f"line(s) after the reported done on {pod} — earlier done may be FALSE "
        "(advisory only; nothing stopped)."
    )
    log.warning(
        "posted post-done phase advisory for #%d: %d new phase line(s) on pod %s",
        issue,
        len(u.new_phase_lines),
        pod,
    )
    return u.done_line, u.done_epoch, u.done_pod, True, True, u.new_phase_lines


# A GPU is considered idle when its `utilization.gpu` is at or below this
# percent. A real training / vLLM-generation workload reads >>5% on any
# GPU it is using (typically 80-100%); the threshold is a conservative
# floor that tolerates briefly-idle GPUs during inter-step bookkeeping
# without admitting a truly idle pod.
GPU_IDLE_UTIL_THRESHOLD = 5


def _parse_gpu_utils(gpu_util: str) -> list[int] | None:
    """Per-GPU int utilizations from the probe's comma string, or None.

    Fail-safe: returns ``None`` when ``gpu_util`` is the literal sentinel
    ``"unknown"``, is empty, parses to zero tokens, or any token fails to
    parse as an int — the consumers (:func:`_gpu_idle`, the #873 width
    advisory) treat ``None`` as "no GPU signal" and never count it as
    idle. Extracted from ``_gpu_idle`` (#873), behavior-identical.
    """
    if not gpu_util or gpu_util == "unknown":
        return None
    try:
        utils = [int(tok.strip()) for tok in gpu_util.split(",") if tok.strip()]
    except ValueError:
        return None
    return utils or None


def _gpu_idle(gpu_util: str) -> bool:
    """Return True iff every parsed GPU's utilization is <= IDLE threshold.

    Fail-safe: returns False (NOT idle) when ``gpu_util`` is the literal
    sentinel ``"unknown"``, is empty, or any token fails to parse as an
    int (``_parse_gpu_utils`` -> None). The stall verdict requires
    ``gpu_idle == True``, so a missing / erroring ``nvidia-smi`` will
    NEVER by itself declare a healthy long-phase run stalled — the
    per-phase-log + cell-log mtime signals then carry the verdict.
    """
    utils = _parse_gpu_utils(gpu_util)
    if not utils:
        return False
    return all(u <= GPU_IDLE_UTIL_THRESHOLD for u in utils)


# ── GPU-idle advisory (incidents #518 + #537) ───────────────────────────────
#
# Minutes of sustained "healthy verdict + every GPU idle" before the poller
# posts a one-time, non-blocking [gpu-idle-advisory] epm:progress marker.
# ``0`` (or negative) disables the advisory entirely. Read at import time to
# mirror ``SSH_FAIL_REFRESH_THRESHOLD``; tests pass ``advisory_min``
# explicitly to the pure decision core instead of mutating the env.
GPU_IDLE_ADVISORY_MIN = int(os.environ.get("EPM_GPU_IDLE_ADVISORY_MIN", "30"))


@dataclass(frozen=True)
class GpuIdleAdvisoryUpdate:
    """Outcome of one advisory-counter tick (``_gpu_idle_advisory_update``)."""

    should_post: bool
    idle_since_epoch: int  # 0 = no active all-idle span
    idle_span_sec: int  # length of the current span; 0 when no span


def _gpu_idle_advisory_update(
    *,
    status: str,
    gpu_util: str,
    current_phase: str,
    prev_phase: str,
    prev_idle_since_epoch: int,
    advised_phases: set[str],
    now_epoch: int,
    advisory_min: int,
) -> GpuIdleAdvisoryUpdate:
    """Pure decision core for the GPU-idle advisory (incidents #518 + #537).

    Tracks the sustained span of "healthy verdict + every GPU idle" across
    poll ticks. The span RESETS (``idle_since_epoch`` -> 0) whenever the
    verdict is not ``running``, any GPU is busy, or the GPU sample is
    ``unknown`` / unparsable — the idle predicate is ``_gpu_idle`` itself
    (<= ``GPU_IDLE_UTIL_THRESHOLD``% on every card), so the stall verdict's
    fail-safe semantics carry over unchanged: a missing / erroring
    nvidia-smi never accumulates toward an advisory. A phase change
    RESTARTS the span at the current tick so each phase is judged on its
    own idle window.

    ``should_post`` is True only when the span has lasted at least
    ``advisory_min`` minutes AND ``current_phase`` is not already in
    ``advised_phases`` (at-most-once-per-phase de-dup). ``advisory_min <= 0``
    disables the advisory. Pure / no I/O — the caller owns state
    persistence and the marker post.
    """
    if advisory_min <= 0:
        return GpuIdleAdvisoryUpdate(should_post=False, idle_since_epoch=0, idle_span_sec=0)
    if status != "running" or not _gpu_idle(gpu_util):
        return GpuIdleAdvisoryUpdate(should_post=False, idle_since_epoch=0, idle_span_sec=0)
    if current_phase != prev_phase or prev_idle_since_epoch <= 0:
        idle_since = now_epoch
    else:
        idle_since = prev_idle_since_epoch
    span = max(0, now_epoch - idle_since)
    should_post = span >= advisory_min * 60 and current_phase not in advised_phases
    return GpuIdleAdvisoryUpdate(
        should_post=should_post, idle_since_epoch=idle_since, idle_span_sec=span
    )


def _maybe_post_gpu_idle_advisory(
    *,
    issue: int,
    pod: str,
    status: str,
    gpu_util: str,
    current_phase: str,
    prev_state: dict[str, str],
    now_epoch: int,
) -> tuple[int, set[str], bool]:
    """Advisory wiring for ``poll_once``: parse state, decide, maybe post.

    Returns ``(idle_since_epoch, advised_phases, posted)`` for the caller to
    persist via ``_save_state``. Posting rides the SAME ``epm:progress``
    marker channel as the phase-transition posts (note prefixed
    ``[gpu-idle-advisory]``, plus a ``gpu_idle_advisory=True`` extra for
    downstream consumers) — no new marker schema. A post failure is logged
    and the phase is NOT recorded as advised, so the next tick retries; the
    advisory never affects the status verdict and never stops anything.

    The reported idle minutes are PER-INSTANCE / PER-RUN, never cumulative
    across relaunches (#1033): both callers hand this function a
    run/instance-scoped ``prev_state`` — the RunPod lane via
    ``_tripwire_run_scope`` (``_RUN_SCOPED_STATE_KEYS`` includes the idle
    keys), the GCP lane additionally via attempt-id keying
    (``backend_poll._scope_idle_state_to_attempt``) — so an advisory can
    never report an idle span exceeding the current instance's own poll
    history (#763: "543 min" printed on a ~17-min-old fresh VM pre-#1033).
    """
    try:
        prev_idle_since = int(prev_state.get("gpu_idle_since_epoch", "0"))
    except (TypeError, ValueError):
        prev_idle_since = 0
    advised_phases = {
        p for p in (prev_state.get("gpu_idle_advised_phases", "") or "").split(",") if p
    }
    update = _gpu_idle_advisory_update(
        status=status,
        gpu_util=gpu_util,
        current_phase=current_phase,
        prev_phase=prev_state.get("phase", ""),
        prev_idle_since_epoch=prev_idle_since,
        advised_phases=advised_phases,
        now_epoch=now_epoch,
        advisory_min=GPU_IDLE_ADVISORY_MIN,
    )
    if not update.should_post:
        return update.idle_since_epoch, advised_phases, False
    n_gpus = len([tok for tok in gpu_util.split(",") if tok.strip()])
    idle_min = update.idle_span_sec // 60
    note = (
        f"[gpu-idle-advisory] all {n_gpus} GPUs <= {GPU_IDLE_UTIL_THRESHOLD}% util for "
        f"{idle_min} min while the run is healthy (phase={current_phase}, "
        f"gpu_util={gpu_util}). Likely a CPU-only phase holding a GPU pod — consider "
        "moving the phase off-pod to the VM or stopping the pod after a checkpoint "
        "(CLAUDE.md: CPU-only phases don't hold GPU pods). Advisory only: the stall "
        "verdict is unchanged and nothing was stopped."
    )
    try:
        post_event(
            issue,
            "epm:progress",
            by="poll_pipeline",
            note=note,
            phase=current_phase,
            pod=pod,
            gpu_idle_advisory=True,
        )
    except Exception as exc:
        log.error("gpu-idle advisory post failed (next tick will retry): %s", exc)
        return update.idle_since_epoch, advised_phases, False
    log.warning(
        "posted gpu-idle advisory for #%d: all %d GPUs idle %d min during healthy phase=%s",
        issue,
        n_gpus,
        idle_min,
        current_phase,
    )
    advised_phases.add(current_phase)
    return update.idle_since_epoch, advised_phases, True


# ── m-of-N GPU-width advisory (#873; incidents #813/#664) ────────────────────
# Minutes of sustained "healthy verdict + a STABLE strict subset of GPUs idle"
# on an N>1-GPU pod before a one-per-phase [gpu-width-advisory] epm:progress
# marker. 0 (or negative) disables. Default 45 (middle of the 30-60 band the
# #873 audit prescribed; ABOVE the 30-min all-idle default because partial
# width has a legitimate transient — uneven shard finish tails).
GPU_WIDTH_ADVISORY_MIN = int(os.environ.get("EPM_GPU_WIDTH_ADVISORY_MIN", "45"))


@dataclass(frozen=True)
class GpuWidthAdvisoryUpdate:
    """Outcome of one width-advisory tick (``_gpu_width_advisory_update``)."""

    should_post: bool
    width_since_epoch: int  # 0 = no active partial-width span
    idle_indices: tuple[int, ...]  # the CURRENT stable idle subset; () when no span
    span_sec: int


def _gpu_width_advisory_update(
    *,
    status: str,
    gpu_util: str,
    current_phase: str,
    prev_phase: str,
    prev_width_since_epoch: int,
    prev_idle_indices: tuple[int, ...],
    advised_phases: set[str],
    now_epoch: int,
    advisory_min: int,
) -> GpuWidthAdvisoryUpdate:
    """Pure decision core for the m-of-N GPU-width advisory (#873, #813).

    Tracks the sustained span of "healthy verdict + a STABLE strict subset
    of GPUs idle" on a multi-GPU pod. Everything RESETS the span (returns
    ``(False, 0, (), 0)``): a disabled advisory (``advisory_min <= 0``), a
    non-``running`` verdict, an ``unknown`` / unparseable GPU sample (the
    ``_gpu_idle`` fail-safe carries over verbatim — a missing nvidia-smi
    never accumulates), a single-GPU pod (N < 2), all-idle (the existing
    idle advisory's domain — disjoint by construction), and all-active
    (healthy). The span RESTARTS (``width_since = now_epoch``) on a phase
    change, a missing prior span, or an idle-index-set CHANGE — a CHURNING
    idle set is staggered shard progress, not the #813 structurally-unused-
    GPUs signature; requiring set stability is the strictest false-positive
    guard. ``should_post`` is True only when the span has lasted at least
    ``advisory_min`` minutes AND ``current_phase`` is not already in
    ``advised_phases`` (at-most-once-per-phase de-dup). Pure / no I/O —
    the caller owns state persistence and the marker post.
    """
    reset = GpuWidthAdvisoryUpdate(
        should_post=False, width_since_epoch=0, idle_indices=(), span_sec=0
    )
    if advisory_min <= 0 or status != "running":
        return reset
    utils = _parse_gpu_utils(gpu_util)
    if utils is None or len(utils) < 2:
        return reset
    idle_indices = tuple(i for i, u in enumerate(utils) if u <= GPU_IDLE_UTIL_THRESHOLD)
    active = len(utils) - len(idle_indices)
    if not (1 <= active < len(utils)):
        return reset
    if (
        current_phase != prev_phase
        or prev_width_since_epoch <= 0
        or idle_indices != prev_idle_indices
    ):
        width_since = now_epoch
    else:
        width_since = prev_width_since_epoch
    span = max(0, now_epoch - width_since)
    should_post = span >= advisory_min * 60 and current_phase not in advised_phases
    return GpuWidthAdvisoryUpdate(
        should_post=should_post,
        width_since_epoch=width_since,
        idle_indices=idle_indices,
        span_sec=span,
    )


def _maybe_post_gpu_width_advisory(
    *,
    issue: int,
    pod: str,
    status: str,
    gpu_util: str,
    current_phase: str,
    prev_state: dict[str, str],
    now_epoch: int,
) -> tuple[int, tuple[int, ...], set[str], bool]:
    """Width-advisory wiring for ``poll_once``: parse state, decide, maybe post.

    Returns ``(width_since_epoch, idle_indices, advised_phases, posted)`` for
    the caller to persist via ``_save_state``. The caller applies the #873
    run-scope reset (:func:`_tripwire_run_scope`, AC #6) BEFORE this call.
    Posting rides the SAME ``epm:progress`` channel as the idle advisory
    (note prefixed ``[gpu-width-advisory]``, plus a ``gpu_width_advisory=True``
    extra) — no new marker kind. Guarded state parses: a corrupted int /
    index set resets the span, never raises into ``poll_once``. A post
    failure is logged and the phase is NOT recorded as advised, so the next
    tick retries; the advisory never affects the status verdict and never
    stops anything.
    """
    try:
        prev_width_since = int(prev_state.get("gpu_width_since_epoch", "0"))
    except (TypeError, ValueError):
        prev_width_since = 0
    try:
        prev_idle_indices = tuple(
            int(tok) for tok in (prev_state.get("gpu_width_idle_set", "") or "").split(",") if tok
        )
    except (TypeError, ValueError):
        prev_idle_indices = ()
    advised_phases = {
        p for p in (prev_state.get("gpu_width_advised_phases", "") or "").split(",") if p
    }
    update = _gpu_width_advisory_update(
        status=status,
        gpu_util=gpu_util,
        current_phase=current_phase,
        prev_phase=prev_state.get("phase", ""),
        prev_width_since_epoch=prev_width_since,
        prev_idle_indices=prev_idle_indices,
        advised_phases=advised_phases,
        now_epoch=now_epoch,
        advisory_min=GPU_WIDTH_ADVISORY_MIN,
    )
    if not update.should_post:
        return update.width_since_epoch, update.idle_indices, advised_phases, False
    n_gpus = len(_parse_gpu_utils(gpu_util) or [])
    span_min = update.span_sec // 60
    idle_list = ",".join(str(i) for i in update.idle_indices)
    note = (
        f"[gpu-width-advisory] {len(update.idle_indices)} of {n_gpus} GPUs <= "
        f"{GPU_IDLE_UTIL_THRESHOLD}% util for {span_min} min while the run is healthy "
        f"(idle GPU indices: {idle_list}; phase={current_phase}, gpu_util={gpu_util}). "
        "A sustained narrow phase holding a wide pod is the #813 idle-width / #664 "
        "spend-leak class — consider widening the parallelism to fill the pod, or "
        "releasing/downsizing it (CLAUDE.md: per-phase GPU-WIDTH right-sizing). "
        "Advisory only: the stall verdict is unchanged and nothing was stopped."
    )
    try:
        post_event(
            issue,
            "epm:progress",
            by="poll_pipeline",
            note=note,
            phase=current_phase,
            pod=pod,
            gpu_width_advisory=True,
        )
    except Exception as exc:
        log.error("gpu-width advisory post failed (next tick will retry): %s", exc)
        return update.width_since_epoch, update.idle_indices, advised_phases, False
    log.warning(
        "posted gpu-width advisory for #%d: %d of %d GPUs idle %d min during healthy phase=%s",
        issue,
        len(update.idle_indices),
        n_gpus,
        span_min,
        current_phase,
    )
    advised_phases.add(current_phase)
    return update.width_since_epoch, update.idle_indices, advised_phases, True


# ── Under-parallelization warning (partial saturation; plan §3, workflow v2) ──
# Minutes of sustained "healthy verdict + fewer than HALF the provisioned GPUs
# busy" on an N>1-GPU pod before ONE per-RUN [gpu-underparallel-warning]
# epm:progress note. DISTINCT from the #873 [gpu-width-advisory] above:
#   * that fires on ANY idle GPU (1 <= idle < N) after GPU_WIDTH_ADVISORY_MIN
#     (45m), per-PHASE, and requires a STABLE idle set — its target is a NARROW
#     phase holding a WIDE pod (release / downsize it, the #813/#664 spend-leak);
#   * THIS one fires on the stronger MAJORITY-idle signal (< 50% of GPUs busy)
#     after a shorter 15-min window, deduped ONCE PER RUN, and points at
#     under-parallelization / sharding (widen the work to fill the pod — the v2
#     "saturate every provisioned GPU" guideline the efficiency critics own).
# Both ride the same epm:progress channel; neither flips ``status`` nor stops
# anything. 0 (or negative) disables.
GPU_UNDERPARALLEL_WARNING_MIN = int(os.environ.get("EPM_GPU_UNDERPARALLEL_WARNING_MIN", "15"))
# Warn when the busy-GPU FRACTION is strictly below this (majority idle).
GPU_UNDERPARALLEL_BUSY_FRACTION = 0.5


@dataclass(frozen=True)
class GpuUnderparallelUpdate:
    """Outcome of one under-parallelization-warning tick."""

    should_post: bool
    since_epoch: int  # 0 = no active partial-saturation span
    n_busy: int
    n_gpus: int
    span_sec: int


def _gpu_underparallel_update(
    *,
    status: str,
    gpu_util: str,
    prev_since_epoch: int,
    already_warned: bool,
    now_epoch: int,
    warning_min: int,
) -> GpuUnderparallelUpdate:
    """Pure decision core for the per-run under-parallelization warning (plan §3).

    Tracks the sustained span of "healthy verdict + < 50% of the provisioned
    GPUs busy (but >= 1 busy)" on a multi-GPU pod. Everything RESETS the span
    (returns ``should_post=False, since_epoch=0``): a disabled warning
    (``warning_min <= 0``), a non-``running`` verdict, an ``unknown`` /
    unparseable GPU sample (the ``_parse_gpu_utils`` fail-safe carries over — a
    missing nvidia-smi never accumulates), a single-GPU pod (N < 2), all-idle
    (``n_busy == 0``: the idle advisory's domain AND a legitimate CPU-only
    phase — never re-flagged here), and >= 50% busy (healthy width). The span
    does NOT reset on a phase change — under-parallelization is a run-level
    concern and the per-RUN dedup already bounds it to one warning.
    ``should_post`` is True only once the span reaches ``warning_min`` minutes
    AND the run has not already been warned. Pure / no I/O — the caller owns
    state persistence + the marker post.
    """
    reset = GpuUnderparallelUpdate(should_post=False, since_epoch=0, n_busy=0, n_gpus=0, span_sec=0)
    if warning_min <= 0 or status != "running":
        return reset
    utils = _parse_gpu_utils(gpu_util)
    if utils is None or len(utils) < 2:
        return reset
    n_gpus = len(utils)
    n_busy = sum(1 for u in utils if u > GPU_IDLE_UTIL_THRESHOLD)
    if not (n_busy >= 1 and n_busy / n_gpus < GPU_UNDERPARALLEL_BUSY_FRACTION):
        return reset
    since = now_epoch if prev_since_epoch <= 0 else prev_since_epoch
    span = max(0, now_epoch - since)
    should_post = span >= warning_min * 60 and not already_warned
    return GpuUnderparallelUpdate(
        should_post=should_post,
        since_epoch=since,
        n_busy=n_busy,
        n_gpus=n_gpus,
        span_sec=span,
    )


def _maybe_post_gpu_underparallel_warning(
    *,
    issue: int,
    pod: str,
    status: str,
    gpu_util: str,
    current_phase: str,
    prev_state: dict[str, str],
    now_epoch: int,
) -> tuple[int, bool, bool]:
    """Under-parallelization wiring for ``poll_once``: parse state, decide, post.

    Returns ``(since_epoch, warned, posted)`` for the caller to persist via
    ``_save_state``. Per-RUN dedup via the ``gpu_underparallel_warned`` flag,
    which is run-scoped (cleared with the other #873 tripwire keys on a fresh
    ``epm:run-launched``, so a relaunch / follow-up round re-arms the warning).
    Posts on the SAME ``epm:progress`` channel as the width advisory (note
    prefixed ``[gpu-underparallel-warning]``, plus a
    ``gpu_underparallel_warning=True`` extra). A post failure is logged and the
    run is NOT recorded as warned, so the next tick retries; the warning never
    affects the status verdict and never stops anything.
    """
    try:
        prev_since = int(prev_state.get("gpu_underparallel_since_epoch", "0"))
    except (TypeError, ValueError):
        prev_since = 0
    already_warned = prev_state.get("gpu_underparallel_warned", "0") == "1"
    update = _gpu_underparallel_update(
        status=status,
        gpu_util=gpu_util,
        prev_since_epoch=prev_since,
        already_warned=already_warned,
        now_epoch=now_epoch,
        warning_min=GPU_UNDERPARALLEL_WARNING_MIN,
    )
    if not update.should_post:
        return update.since_epoch, already_warned, False
    span_min = update.span_sec // 60
    note = (
        f"[gpu-underparallel-warning] {update.n_busy} of {update.n_gpus} GPUs busy for "
        f">{span_min} min while the run is healthy (phase={current_phase}, gpu_util={gpu_util}) "
        f"— fewer than half the provisioned GPUs are working. Check sharding: this run may be "
        f"under-parallelized (widen the work to fill the pod, or downsize it — the workflow-v2 "
        f"'saturate every provisioned GPU' guideline). Advisory only: the status verdict is "
        f"unchanged and nothing was stopped."
    )
    try:
        post_event(
            issue,
            "epm:progress",
            by="poll_pipeline",
            note=note,
            phase=current_phase,
            pod=pod,
            gpu_underparallel_warning=True,
        )
    except Exception as exc:
        log.error("gpu-underparallel warning post failed (next tick will retry): %s", exc)
        return update.since_epoch, already_warned, False
    log.warning(
        "posted gpu-underparallel warning for #%d: %d of %d GPUs busy %d min during healthy "
        "phase=%s",
        issue,
        update.n_busy,
        update.n_gpus,
        span_min,
        current_phase,
    )
    return update.since_epoch, True, True


# ── Phase-ETA tripwire (#873) ────────────────────────────────────────────────
# Multiplier over the plan §9 TOTAL planned_wall_h before the poller auto-posts
# epm:compute-deviation (source: poller). <= 0 disables. Read at import time to
# mirror GPU_IDLE_ADVISORY_MIN; tests pass `mult` explicitly to the pure core.
ETA_DEVIATION_MULT = float(os.environ.get("EPM_ETA_DEVIATION_MULT", "2.0"))

# Run-level dedup key for the whole-run elapsed check (D1b). Phase names match
# PHASE_RE ([a-z0-9_]+); __run_total__ shares the charset, so a collision with
# a real phase name is possible in principle but costs only a suppressed
# duplicate advisory marker (plan §12 assumption 14).
ETA_RUN_TOTAL_KEY = "__run_total__"

_LEADING_FLOAT_RE = re.compile(r"\s*([0-9]+(?:\.[0-9]+)?)")
# A markdown |---|:---:|---| separator row: every pipe-split cell is only
# whitespace / dashes / colons.
_MD_SEPARATOR_CELL_RE = re.compile(r"[\s:\-]*")


class _UnparseableWallRow(Exception):
    """A located planned_wall_h data row yielded no leading float (AC #2)."""


def _md_planned_wall_rows(plan_text: str) -> list[float]:
    """Leading floats of every markdown-table planned_wall_h column cell.

    Table-scoped: only rows FOLLOWING a ``|``-prefixed header line that
    contains ``planned_wall_h`` are scanned, with the value-column index
    DERIVED from that header's cell position (never a hardcoded ordinal).
    Raises ``_UnparseableWallRow`` when ANY located data row's cell has no
    leading float — the caller maps that to a disabled tripwire (``None``),
    never a partial sum (AC #2).
    """
    rows: list[float] = []
    lines = plan_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not (line.lstrip().startswith("|") and "planned_wall_h" in line):
            i += 1
            continue
        header_cells = line.split("|")
        col = next(idx for idx, c in enumerate(header_cells) if "planned_wall_h" in c)
        j = i + 1
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            cells = lines[j].split("|")
            j += 1
            if all(_MD_SEPARATOR_CELL_RE.fullmatch(c) for c in cells):
                continue  # the |---|---| separator row
            if col >= len(cells):
                raise _UnparseableWallRow(lines[j - 1][:120])
            m = _LEADING_FLOAT_RE.match(cells[col])
            if not m:
                raise _UnparseableWallRow(lines[j - 1][:120])
            rows.append(float(m.group(1)))
        i = j
    return rows


def _html_planned_wall_rows(plan_text: str) -> list[float]:
    """Leading floats of every HTML-table planned_wall_h column cell.

    The row scan is SCOPED to each ``<table>`` element whose ``<th>`` row
    contains ``planned_wall_h`` (never a document-wide ``<td>`` scan), and
    the value-column index is DERIVED from the ``<th>`` position (parity
    with the markdown path). Raises ``_UnparseableWallRow`` on any located
    data row whose cell has no leading float (AC #2).
    """
    rows: list[float] = []
    for tbl_m in re.finditer(r"<table\b[^>]*>(.*?)</table>", plan_text, re.IGNORECASE | re.DOTALL):
        tbl = tbl_m.group(1)
        ths = re.findall(r"<th\b[^>]*>(.*?)</th>", tbl, re.IGNORECASE | re.DOTALL)
        col = next((idx for idx, th in enumerate(ths) if "planned_wall_h" in th), None)
        if col is None:
            continue  # a table without the header never contributes
        for tr_m in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", tbl, re.IGNORECASE | re.DOTALL):
            tds = re.findall(r"<td\b[^>]*>(.*?)</td>", tr_m.group(1), re.IGNORECASE | re.DOTALL)
            if not tds:
                continue  # the <th>-only header row
            if col >= len(tds):
                raise _UnparseableWallRow(tr_m.group(1)[:120])
            cell = re.sub(r"<[^>]+>", "", tds[col])
            m = _LEADING_FLOAT_RE.match(cell)
            if not m:
                raise _UnparseableWallRow(cell[:120])
            rows.append(float(m.group(1)))
    return rows


def _parse_plan_wall_budget(plan_text: str) -> float | None:
    """Sum of §9 per-component planned_wall_h, or None when no row parses.

    Handles the markdown pipe-table form (a ``|``-prefixed header line
    containing ``planned_wall_h``) and the HTML ``<tr><th>``/``<td>`` form
    (both exist in real plans — e.g. #779's live plan is HTML). CONTRACT
    (critic round 1, AC #2):

    - ALL tables carrying a planned_wall_h header are located and summed
      (multi-stage plans, e.g. #479 Stage 1 + Stage 2 — a single-header
      parse under-counts and false-fires).
    - The value-column index is DERIVED from the header cell position in
      BOTH formats (markdown pipe cells AND the HTML ``<th>`` row) — never
      a hardcoded ordinal; the HTML row scan is SCOPED to the table element
      containing the planned_wall_h ``<th>``, never a document-wide
      ``<td>`` scan.
    - Each data cell contributes its LEADING float ("3 (async, off-GPU)"
      -> 3.0). If ANY located data row's planned_wall_h cell yields NO
      leading float, return None (tripwire disabled, log once) — NEVER a
      partial sum: an under-parsed budget is the one path to a false
      positive.
    - Returns None on zero located tables / zero data rows. Whole body
      exception-wrapped: ANY parse error returns None (fail-safe OFF).
    """
    try:
        rows = _md_planned_wall_rows(plan_text) + _html_planned_wall_rows(plan_text)
    except Exception:
        return None
    if not rows:
        return None
    return sum(rows) or None


def _plan_total_wall_h_for_issue(issue: int) -> float | None:
    """Read tasks/<status>/<issue>/plans/plan.md and parse the §9 total.

    Fail-soft None on a missing task / missing plan / unreadable file /
    unparseable table — the tripwire is then disabled for this run (AC #2).
    ``plans/plan.md`` symlinks the highest plan version (D1).
    """
    try:
        plan = find_task_path(issue) / "plans" / "plan.md"
        if not plan.exists():
            return None
        return _parse_plan_wall_budget(plan.read_text())
    except Exception:
        return None


@dataclass(frozen=True)
class EtaDeviationPost:
    """One epm:compute-deviation post the ETA tripwire decided to emit."""

    dedup_key: str  # current phase name, or ETA_RUN_TOTAL_KEY
    scope: str  # "phase" | "run"
    elapsed_h: float
    planned_wall_h: float
    ratio: float


@dataclass(frozen=True)
class EtaDeviationUpdate:
    """Outcome of one ETA-tripwire tick (``_eta_deviation_update``)."""

    posts: tuple[EtaDeviationPost, ...]  # 0..2 entries


def _eta_deviation_update(
    *,
    status: str,
    current_phase: str,
    phase_started_epoch: int,
    run_age_sec: float | None,
    total_planned_wall_h: float | None,
    posted_keys: set[str],
    now_epoch: int,
    mult: float,
) -> EtaDeviationUpdate:
    """Pure decision core for the phase-ETA tripwire (#873; incident #763).

    The budget is ``mult`` x the plan §9 TOTAL planned_wall_h (D1b — the
    only mapping-free, zero-false-positive construct; §9 component names
    are planner free-text with no mechanical phase mapping). Two strictly-
    conservative checks share it, both STRICT ``>`` at the boundary:

    * per-phase: elapsed since ``phase_started_epoch`` exceeds the budget
      — any SINGLE phase exceeding ``mult`` x the ENTIRE plan's wall
      estimate is unambiguously a deviation. Skipped for the ``""`` /
      ``unknown`` / ``done`` phases and for an unknown start (<= 0).
    * run-level: ``run_age_sec`` exceeds the budget — catches cumulative
      overrun across phases. Dedup key :data:`ETA_RUN_TOTAL_KEY`.

    Fail-safe OFF on ``mult <= 0``, a missing/non-positive budget, or a
    non-``running`` verdict. Pure / no I/O — the caller owns state
    persistence and the marker posts.
    """
    if mult <= 0 or total_planned_wall_h is None or total_planned_wall_h <= 0:
        return EtaDeviationUpdate(posts=())
    if status != "running":
        return EtaDeviationUpdate(posts=())
    budget_sec = mult * total_planned_wall_h * 3600.0
    posts: list[EtaDeviationPost] = []
    if (
        current_phase not in {"", "unknown", "done"}
        and phase_started_epoch > 0
        and (now_epoch - phase_started_epoch) > budget_sec
        and current_phase not in posted_keys
    ):
        elapsed = float(now_epoch - phase_started_epoch)
        posts.append(
            EtaDeviationPost(
                dedup_key=current_phase,
                scope="phase",
                elapsed_h=elapsed / 3600.0,
                planned_wall_h=total_planned_wall_h,
                ratio=elapsed / (total_planned_wall_h * 3600.0),
            )
        )
    if (
        run_age_sec is not None
        and run_age_sec > budget_sec
        and ETA_RUN_TOTAL_KEY not in posted_keys
    ):
        posts.append(
            EtaDeviationPost(
                dedup_key=ETA_RUN_TOTAL_KEY,
                scope="run",
                elapsed_h=run_age_sec / 3600.0,
                planned_wall_h=total_planned_wall_h,
                ratio=run_age_sec / (total_planned_wall_h * 3600.0),
            )
        )
    return EtaDeviationUpdate(posts=tuple(posts))


def _maybe_post_eta_deviation(
    *,
    issue: int,
    pod: str,
    status: str,
    current_phase: str,
    last_phase_change_epoch: int,
    run_age_sec: float | None,
    prev_state: dict[str, str],
    now_epoch: int,
) -> tuple[set[str], bool, bool]:
    """ETA-tripwire wiring for ``poll_once``: parse state, decide, maybe post.

    Returns ``(posted_keys, posted_this_tick, budget_warned)`` for the
    caller to persist via ``_save_state``. The caller applies the run-scope
    reset (:func:`_tripwire_run_scope`, AC #6) BEFORE this call, so
    ``prev_state`` is already scoped to the current run. Fail-soft
    everywhere: a missing / unparseable plan budget disables the tripwire
    with ONE logged line per run (state-flag-backed); a marker-post failure
    is logged and the dedup key is NOT recorded, so the next tick retries.
    Never flips ``status``, never stops anything. Phase start resolves to
    ``last_phase_change_epoch`` when a boundary was observed this run, else
    the run-launch epoch (``now - run_age_sec``), else 0 (phase check
    skipped — fail-safe, D2).
    """
    posted_keys = {
        k for k in (prev_state.get("eta_deviation_posted_keys", "") or "").split(",") if k
    }
    budget_warned = prev_state.get("eta_budget_warned", "0") == "1"
    if ETA_DEVIATION_MULT <= 0:
        return posted_keys, False, budget_warned
    total = _plan_total_wall_h_for_issue(issue)
    if total is None:
        if not budget_warned:
            log.info(
                "no parseable §9 planned_wall_h for #%d; phase-ETA tripwire disabled (fail-safe)",
                issue,
            )
            budget_warned = True
        return posted_keys, False, budget_warned
    if last_phase_change_epoch > 0:
        phase_started_epoch = last_phase_change_epoch
    elif run_age_sec is not None:
        phase_started_epoch = int(now_epoch - run_age_sec)
    else:
        phase_started_epoch = 0
    update = _eta_deviation_update(
        status=status,
        current_phase=current_phase,
        phase_started_epoch=phase_started_epoch,
        run_age_sec=run_age_sec,
        total_planned_wall_h=total,
        posted_keys=posted_keys,
        now_epoch=now_epoch,
        mult=ETA_DEVIATION_MULT,
    )
    posted_any = False
    for p in update.posts:
        component = "run total" if p.scope == "run" else f"phase {current_phase!r}"
        note = (
            f"component: {component} (elapsed vs §9 total)"
            f" planned_wall_h: {p.planned_wall_h:.2f}"
            f" projected_wall_h: {p.elapsed_h:.2f} (elapsed so far — run still in"
            f" progress, a lower bound) ratio: {p.ratio:.1f}"
            f" basis: poller elapsed-vs-plan tripwire, {ETA_DEVIATION_MULT:g}x the §9"
            f" compute-projection planned_wall_h total (source: poller; advisory only"
            f" — nothing stopped)"
        )
        try:
            post_event(
                issue,
                "epm:compute-deviation",
                by="poll_pipeline",
                note=note,
                phase=current_phase,
                pod=pod,
                source="poller",
                basis="elapsed-vs-plan",
            )
        except Exception as exc:
            log.error("phase-ETA deviation post failed (next tick will retry): %s", exc)
            continue
        log.warning(
            "posted epm:compute-deviation for #%d: %s elapsed %.2fh vs §9 total %.2fh (%.1fx)",
            issue,
            p.dedup_key,
            p.elapsed_h,
            p.planned_wall_h,
            p.ratio,
        )
        posted_keys.add(p.dedup_key)
        posted_any = True
    return posted_keys, posted_any, budget_warned


# ── #873 run-scoped tripwire dedup (AC #6 / D4) ──────────────────────────────
# State keys owned by the #873 tripwires; cleared together on a fresh
# epm:run-launched epoch so a relaunch / same-issue follow-up round re-arms
# both tripwires (without this, ETA_RUN_TOTAL_KEY would be permanently
# suppressed per issue and common phase names would collide across runs).
_TRIPWIRE_STATE_KEYS: tuple[str, ...] = (
    "eta_deviation_posted_keys",
    "eta_budget_warned",
    "gpu_width_since_epoch",
    "gpu_width_idle_set",
    "gpu_width_advised_phases",
    # plan §3 under-parallelization warning: per-RUN dedup, so re-arm on a
    # fresh run alongside the #873 width/ETA tripwires.
    "gpu_underparallel_since_epoch",
    "gpu_underparallel_warned",
)
# #1033: the FULL run-scoped clear set = the #873 tripwire dedup keys PLUS the
# GPU-idle advisory/escalation keys. The idle span + per-phase dedup sets
# belong to the RUN that accumulated them: carried across a relaunch they
# print stale idle minutes exceeding the fresh instance's own age (#763
# "543 min" / #810 "486 min" on ~17-min-old instances, where the phase name
# matched the stored one so the per-phase reset never fired). The pre-#1033
# "idle keys untouched by the run-scope reset" contract was the bug.
_RUN_SCOPED_STATE_KEYS: tuple[str, ...] = (
    *_TRIPWIRE_STATE_KEYS,
    "gpu_idle_since_epoch",
    "gpu_idle_advised_phases",
    "gpu_idle_escalated_phases",
)
# Tolerance (seconds) when comparing the observed run-launched epoch against
# the stored anchor: rounding jitter on ``now - run_age`` must never
# spuriously reset the dedup keys mid-run.
_TRIPWIRE_RUN_EPOCH_TOLERANCE_SEC = 60


def _tripwire_run_scope(
    prev_state: dict[str, str], *, run_age_sec: float | None, now_epoch: int
) -> tuple[dict[str, str], int]:
    """Run-scope the #873 tripwire dedup keys + the GPU-idle keys (#1033).

    Returns ``(state, tripwire_run_epoch)``: ``state`` is ``prev_state``
    unchanged when no reset applies, or a copy with every
    ``_RUN_SCOPED_STATE_KEYS`` entry REMOVED (the #873 tripwire dedup keys
    plus, since #1033, the three GPU-idle advisory/escalation keys — a
    fresh run's idle clock must not inherit the previous run's span) when
    the CURRENT ``epm:run-launched`` epoch (``now_epoch - run_age_sec``)
    is newer than the stored ``tripwire_run_epoch`` anchor by more than
    the jitter tolerance. ``tripwire_run_epoch`` is the anchor the caller
    persists via ``_save_state`` / the GCP sibling state. Fail-safe: an
    unknown run age (missing / unreadable marker) keeps the stored anchor
    and clears nothing; a MALFORMED stored anchor (present but
    non-numeric) with a known run age cannot decide run identity, so it
    fails toward RE-ARMING — clear the run-scoped keys and adopt the
    current epoch (cheaper failure = one duplicate advisory, never a
    suppressed one); an absent/zero anchor (genuine first run — no keys
    to protect) adopts the current epoch and keeps the state. Never
    raises into the poll tick.
    """
    raw = prev_state.get("tripwire_run_epoch", "0") or 0
    malformed = False
    try:
        stored = int(float(raw))
    except (TypeError, ValueError):
        stored = 0
        malformed = True
    if run_age_sec is None:
        return prev_state, stored
    current = round(now_epoch - run_age_sec)
    if malformed:
        cleared = {k: v for k, v in prev_state.items() if k not in _RUN_SCOPED_STATE_KEYS}
        return cleared, current
    if stored <= 0:
        return prev_state, current
    if current > stored + _TRIPWIRE_RUN_EPOCH_TOLERANCE_SEC:
        cleared = {k: v for k, v in prev_state.items() if k not in _RUN_SCOPED_STATE_KEYS}
        return cleared, current
    return prev_state, stored


# ── GPU-idle ESCALATION (incident #664) ──────────────────────────────────────
#
# A SECOND tier above the advisory: once a MULTI-GPU pod has been idle in an
# upload/CPU-only phase past ``EPM_GPU_IDLE_ESCALATION_MIN`` minutes (default
# 60, >= the advisory min), the poller fires a Telegram push + a LOUD
# ``[gpu-idle-escalation]`` ``epm:progress`` marker. It NEVER stops the pod —
# it surfaces the spend leak loudly for action (the #664 incident: an 8xH200
# pod sat at 0% GPU for ~12h in a terminal upload phase, ~$530, seen only by
# the one-shot advisory). Both tiers read the SAME ``gpu_idle_since_epoch``
# span the advisory persists — there is no second independent idle clock.
#
# ``EPM_GPU_IDLE_ESCALATION_MIN >= EPM_GPU_IDLE_ADVISORY_MIN`` (escalate only
# AFTER advising); a value below the advisory min is clamped UP to it at import
# with a logged WARNING. ``<= 0`` disables escalation.
_GPU_IDLE_ESCALATION_MIN_RAW = int(os.environ.get("EPM_GPU_IDLE_ESCALATION_MIN", "60"))
if 0 < _GPU_IDLE_ESCALATION_MIN_RAW < GPU_IDLE_ADVISORY_MIN:
    log.warning(
        "EPM_GPU_IDLE_ESCALATION_MIN=%d is below EPM_GPU_IDLE_ADVISORY_MIN=%d; "
        "clamping up to the advisory min (escalate only AFTER advising)",
        _GPU_IDLE_ESCALATION_MIN_RAW,
        GPU_IDLE_ADVISORY_MIN,
    )
    GPU_IDLE_ESCALATION_MIN = GPU_IDLE_ADVISORY_MIN
else:
    GPU_IDLE_ESCALATION_MIN = _GPU_IDLE_ESCALATION_MIN_RAW

# Phase-name substrings that mark a phase as GPU-REQUIRED (NOT escalated). The
# escalation fails toward over-notifying (a loud notice is cheap; a missed leak
# is ~$44/hr), so everything NOT matching this deny-list — except the explicit
# ``unknown`` sentinel — is treated CPU-only and IS eligible. ``merge`` is here
# because merging a checkpoint touches the GPU briefly. Edit in one place.
GPU_REQUIRED_PHASE_SUBSTRINGS = frozenset(
    {
        "train",
        "gen",
        "eval",
        "generate",
        "infer",
        "forward",
        "judge_gen",
        "vllm",
        "setup",
        "preflight",
        "merge",
    }
)


def _phase_is_cpu_only(current_phase: str) -> bool:
    """Return True iff ``current_phase`` is treated as a CPU-only phase.

    Default-CPU-only with a small GPU-REQUIRED deny-list
    (:data:`GPU_REQUIRED_PHASE_SUBSTRINGS`), because phase names vary across
    dispatchers and the safe failure mode for an ESCALATION (a loud notice,
    never an action) is to over-notify, not under-notify. The literal
    ``unknown`` sentinel is the ONE exception that is NOT eligible: a phase the
    dispatcher never named could be anything, and the advisory already fired
    for true CPU idle, so fail toward not-escalating a potentially-GPU phase.
    The #664 trigger phase ``p3_upload`` matches no deny-list substring ->
    CPU-only -> eligible.
    """
    if not current_phase or current_phase == "unknown":
        return False
    name = current_phase.lower()
    return not any(sub in name for sub in GPU_REQUIRED_PHASE_SUBSTRINGS)


@dataclass(frozen=True)
class GpuIdleEscalationUpdate:
    """Outcome of one escalation-counter tick (``_gpu_idle_escalation_update``)."""

    should_escalate: bool
    idle_span_sec: int  # length of the shared idle span at this tick; 0 when none


def _gpu_idle_escalation_update(
    *,
    status: str,
    gpu_util: str,
    current_phase: str,
    idle_since_epoch: int,
    escalated_phases: set[str],
    now_epoch: int,
    escalation_min: int,
) -> GpuIdleEscalationUpdate:
    """Pure decision core for the GPU-idle ESCALATION (incident #664).

    Reuses the SAME idle span the advisory tracks — the caller passes the
    ``idle_since_epoch`` that ``_gpu_idle_advisory_update`` resolved THIS tick,
    so the two tiers never diverge. ``should_escalate`` is True only when ALL
    hold:

    * ``escalation_min > 0`` (escalation enabled);
    * the verdict is ``running`` AND every GPU is idle (``_gpu_idle``) — the
      same fail-safe predicate the advisory uses (``unknown`` / unparsable ->
      not idle);
    * the pod is MULTI-GPU (>= 2 parsed cards) — a single-GPU idle pod is a far
      smaller leak and never escalates;
    * ``current_phase`` is classified upload/CPU-only (``_phase_is_cpu_only``);
    * the shared idle span has lasted at least ``escalation_min`` minutes;
    * ``current_phase`` is not already in ``escalated_phases``
      (at-most-once-per-phase de-dup).

    Pure / no I/O — the caller owns state persistence and the marker/push.
    """
    if escalation_min <= 0:
        return GpuIdleEscalationUpdate(should_escalate=False, idle_span_sec=0)
    if status != "running" or not _gpu_idle(gpu_util):
        return GpuIdleEscalationUpdate(should_escalate=False, idle_span_sec=0)
    if idle_since_epoch <= 0:
        return GpuIdleEscalationUpdate(should_escalate=False, idle_span_sec=0)
    span = max(0, now_epoch - idle_since_epoch)
    n_gpus = len([tok for tok in gpu_util.split(",") if tok.strip()])
    should_escalate = (
        n_gpus >= 2
        and _phase_is_cpu_only(current_phase)
        and span >= escalation_min * 60
        and current_phase not in escalated_phases
    )
    return GpuIdleEscalationUpdate(should_escalate=should_escalate, idle_span_sec=span)


# Telegram-push script (default the my-goat notif-enqueue channel,
# NOTIF_CAT=research), overridable for tests via EPM_TELEGRAM_PUSH_SCRIPT.
# Inlined here (rather than importing autonomous_session_watch) so the poller
# stays self-contained — it already runs as its own bg-Bash process and must
# not pull the heavy watcher module into its import graph.
_TELEGRAM_PUSH_SCRIPT_DEFAULT = Path.home() / "my-goat" / "scripts" / "notif_enqueue.sh"


def _telegram_push(msg: str) -> bool:
    """Best-effort phone notification via the my-goat digest queue.

    FAIL-SOFT: a missing script or any subprocess error is logged and returns
    False — it NEVER raises, so a missing my-goat install degrades the
    escalation to "marker only" and never blocks the poller. Returns True only
    on a confirmed enqueue (rc == 0).
    """
    override = os.environ.get("EPM_TELEGRAM_PUSH_SCRIPT", "").strip()
    script = Path(override) if override else _TELEGRAM_PUSH_SCRIPT_DEFAULT
    if not script.is_file():
        log.warning("telegram push script missing at %s; push dropped", script)
        return False
    try:
        res = subprocess.run(
            ["bash", str(script), msg],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "NOTIF_CAT": "research"},
        )
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("telegram push failed: %s", e)
        return False
    if res.returncode != 0:
        log.warning("telegram push failed: %s", (res.stderr or res.stdout).strip()[:200])
        return False
    return True


def _maybe_escalate_gpu_idle(
    *,
    issue: int,
    pod: str,
    status: str,
    gpu_util: str,
    current_phase: str,
    idle_since_epoch: int,
    prev_state: dict[str, str],
    now_epoch: int,
) -> tuple[set[str], bool]:
    """Escalation wiring for ``poll_once``: parse state, decide, maybe escalate.

    Called RIGHT AFTER ``_maybe_post_gpu_idle_advisory`` (so the advisory always
    fires first on the same span) and fed that pass's resolved
    ``idle_since_epoch`` so both tiers read the ONE shared span. Returns
    ``(escalated_phases, escalated)`` for the caller to persist via
    ``_save_state``.

    On ``should_escalate``: post a LOUD ``[gpu-idle-escalation]`` ``epm:progress``
    marker (``gpu_idle_escalation=True`` extra) AND fire a best-effort Telegram
    push. NOTHING is stopped — the note states so explicitly. A marker-post
    failure is logged and the phase is NOT recorded as escalated (next tick
    retries), exactly like the advisory; a push failure is fail-soft and does
    NOT block recording the escalation (the marker is the durable record).
    """
    escalated_phases = {
        p for p in (prev_state.get("gpu_idle_escalated_phases", "") or "").split(",") if p
    }
    update = _gpu_idle_escalation_update(
        status=status,
        gpu_util=gpu_util,
        current_phase=current_phase,
        idle_since_epoch=idle_since_epoch,
        escalated_phases=escalated_phases,
        now_epoch=now_epoch,
        escalation_min=GPU_IDLE_ESCALATION_MIN,
    )
    if not update.should_escalate:
        return escalated_phases, False
    n_gpus = len([tok for tok in gpu_util.split(",") if tok.strip()])
    idle_min = update.idle_span_sec // 60
    note = (
        f"[gpu-idle-escalation] all {n_gpus} GPUs <= {GPU_IDLE_UTIL_THRESHOLD}% util for "
        f"{idle_min} min on a MULTI-GPU pod in an upload/CPU-only phase "
        f"(phase={current_phase}, gpu_util={gpu_util}). This is the #664 spend-leak class "
        f"(an 8xH200 idle in a terminal upload phase burns ~$44/hr). REMEDY: route the "
        f"upload off-pod / release the GPUs after a checkpoint — the FINAL upload phase is "
        f"itself CPU-only (CLAUDE.md: CPU-only phases don't hold GPU pods). NOTHING was "
        f"stopped — surfacing the spend leak for action."
    )
    try:
        post_event(
            issue,
            "epm:progress",
            by="poll_pipeline",
            note=note,
            phase=current_phase,
            pod=pod,
            gpu_idle_escalation=True,
        )
    except Exception as exc:
        log.error("gpu-idle escalation post failed (next tick will retry): %s", exc)
        return escalated_phases, False
    # Fail-soft phone push — never blocks recording the escalation.
    _telegram_push(
        f"[#{issue}] GPU-idle escalation: {n_gpus} GPUs idle {idle_min} min in "
        f"phase={current_phase} on {pod} (#664 spend-leak class; nothing stopped)."
    )
    log.warning(
        "ESCALATED gpu-idle for #%d: %d GPUs idle %d min in upload/CPU phase=%s (pod=%s)",
        issue,
        n_gpus,
        idle_min,
        current_phase,
        pod,
    )
    escalated_phases.add(current_phase)
    return escalated_phases, True


# Minimum cumulative CPU-seconds delta between consecutive ticks before
# declaring the launcher's process session "advancing". Set conservatively
# so a single accounting quantum or a brief sleep across ticks does not
# false-fire "advancing" on a truly hung session. A real CPU-bound phase
# accrues many seconds per minute of wall time across its process tree;
# even a half-second delta over a 9-minute poll interval is well above
# the noise floor of `ps` rounding.
SESSION_CPU_ADVANCE_EPSILON_SECS = 0.5


def _parse_session_cpu(value: str) -> float | None:
    """Parse a SESSION_CPU_SECS probe value to seconds, or None if unknown.

    The probe heredoc emits one of: a float like ``"4271.5"`` (success),
    ``"unknown"`` (pidfile missing, pid dead, ps unavailable, or ``ps``
    errored). Any other input (empty, malformed) is treated as unknown so
    the caller fails safe to "no signal" — never to "advancing".
    """
    if not value or value == "unknown":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_probe_count(value: str | None) -> int | None:
    """Parse a #864 probe count (``"4"``, ``""``, ``"unknown"``, garbage) to a
    non-negative int; ``None`` = no signal (the caller falls back to the pure
    #826 behavior — every degraded read fails toward CURRENT behavior, never
    toward arming the namespace veto)."""
    if not value or value == "unknown":
        return None
    try:
        n = int(value)
    except ValueError:
        return None
    return n if n >= 0 else None


def _session_cpu_advancing(prev_max: str | None, current: str) -> bool | None:
    """Return True / False / None for the session-CPU "advancing" decision.

    The reference point is the running **maximum** cumulative
    session-CPU observed across ALL prior ticks (``prev_max``), NOT the
    immediately-previous tick's value. Cumulative CPU over a *fixed*
    process set only ever grows, so the high-water mark is the right
    baseline — but the live ``ps``-sum probe is over the launcher's
    process GROUP, and in a multi-shard run each shard exits at a
    different time (per-shard completion is the design). When a shard
    exits, its accumulated cputime drops out of the live sum, so
    ``current`` can DECREASE tick-over-tick while the run is perfectly
    healthy and the surviving shards keep producing output. Comparing
    against the immediately-previous tick (the pre-#658 behavior) read
    that accounting drop as a stall — a false positive whose rate scales
    with shard count (incident #658: an 8-shard run dropped 193272 ->
    38528 the tick after one shard exited and was falsely flagged
    stalled). Comparing against the running max instead is immune: a
    momentary drop below the high-water mark is the child-exit artifact,
    never a hang.

    * ``True``  — both samples parse AND EITHER
      (a) ``current > prev_max + epsilon`` — a NEW high-water mark, so a
          surviving process accrued more CPU than ever before (genuine
          progress), OR
      (b) ``current < prev_max - epsilon`` — ``current`` dropped below
          the running max, which can only be a child-exit / ``ps``
          re-numbering accounting artifact (a hung process loses no
          CPU); the run is doing work.
      In both cases a stalled-on-logs verdict should flip to running.
    * ``False`` — both samples parse AND ``current`` is flat at the
      high-water mark (within +/- epsilon of ``prev_max``): no new
      progress and no de-count, i.e. a truly idle session. Stalled
      stands.
    * ``None``  — at least one sample is unknown. NO signal; the caller
      preserves whatever the older log + GPU arbiters decided. This is
      the fail-safe path on (a) first tick after launch (no prior
      observation), (b) launcher dead (no session to probe — the
      pid-alive arbiter already routed to `dead`), or (c) `ps`
      unavailable.

    Returning None on first-tick prevents an immediate false-stalled →
    epm:failure cascade on a freshly-launched run; the next tick will
    have a prior observation and the decision flips to True / False.
    """
    cur = _parse_session_cpu(current)
    if cur is None:
        return None
    mx = _parse_session_cpu(prev_max) if prev_max is not None else None
    if mx is None:
        return None
    # True on a new high-water mark (genuine CPU progress) OR a sub-max drop
    # (a multi-shard child-exit accounting artifact, never a hang). False
    # only when flat at the high-water mark (within +/- epsilon) — truly idle.
    return abs(cur - mx) > SESSION_CPU_ADVANCE_EPSILON_SECS


def _roll_session_cpu_max(prev_max: str | None, current: str) -> str:
    """Return the running maximum cumulative session-CPU as a probe string.

    The maximum only ever grows (#658): cumulative CPU over a fixed process
    set is monotonic, and a ``current`` sample below the running max is a
    multi-shard child-exit accounting artifact that must NOT lower the
    baseline. An ``unknown`` current sample (transient ps error) preserves
    the prior max. Stored as the literal probe string so the next tick's
    ``_parse_session_cpu`` reads it consistently with the live probe.
    """
    cur = _parse_session_cpu(current)
    if cur is None:
        return prev_max if prev_max else "unknown"
    prev = _parse_session_cpu(prev_max) if prev_max else None
    if prev is None or cur > prev:
        return current
    return prev_max if prev_max else current


def _session_cpu_rate_cores(
    prev_sample: str | None,
    prev_sample_epoch: str | None,
    current: str,
    now_epoch: int,
) -> float | None:
    """Per-tick session-CPU burn rate in cores, or None when not computable.

    rate = (current - prev_sample) / (now_epoch - prev_sample_epoch), where
    prev_sample + prev_sample_epoch are the SAME prior tick's persisted pair
    (written together by _save_state). None (fail-safe -> the #951 veto does
    NOT fire) when: either CPU sample is unknown/unparseable, the epoch is
    missing/unparseable/<=0, or the wall gap is < ZOMBIE_CPU_RATE_MIN_DT_SEC
    (truncation-noise floor) or non-positive (clock garbage). A NEGATIVE
    rate (run restart resets the session counter; a multi-shard child exit
    de-counts its cputime, #658) is returned as-is — it is below any
    positive threshold, so the veto does not fire on it.
    """
    cur = _parse_session_cpu(current)
    prev = _parse_session_cpu(prev_sample) if prev_sample is not None else None
    epoch = _parse_session_cpu(prev_sample_epoch) if prev_sample_epoch else None
    if cur is None or prev is None or epoch is None or epoch <= 0:
        return None
    dt = now_epoch - epoch
    if dt < ZOMBIE_CPU_RATE_MIN_DT_SEC:
        # Also covers dt <= 0 (the floor is positive) — LOAD-BEARING for the
        # existing two-call replay tests, whose back-to-back poll_once calls
        # persist the epoch on call 1 and re-poll milliseconds later (dt~0).
        return None
    return (cur - prev) / dt


def _load_state(state_file: Path, issue: int) -> dict[str, str]:
    if not state_file.exists():
        return {}
    try:
        data = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("state file %s unreadable; treating as empty", state_file)
        return {}
    return data.get(str(issue), {})


def _save_state(state_file: Path, issue: int, payload: dict[str, str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    all_state: dict[str, dict[str, str]] = {}
    if state_file.exists():
        try:
            all_state = json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            all_state = {}
    all_state[str(issue)] = payload
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    tmp.write_text(json.dumps(all_state, indent=2, sort_keys=True))
    tmp.replace(state_file)


def _apply_zombie_override(
    *,
    status: str,
    zombie_gpu_pids: list[str],
    stall_sec: int,
    last_mtime_ago: float,
    phase_log_mtime_ago: float,
    shard_log_mtime_ago: float,
    prev_state: dict[str, str],
    pod: str,
    cpu_override_active: bool,
    gpu_pids_total: int | None = None,
    gpu_pids_resolvable: int | None = None,
    uvm_live_holders: int | None = None,
    session_cpu_rate_cores: float | None = None,
    output_mtime_ago: float = float("inf"),
) -> tuple[str, str | None, bool, int]:
    """The #664/#826/#864/#951/#1033 zombie-GPU-allocation override — returns
    the possibly overridden
    ``(status, stall_reason, cpu_override_active, zombie_streak)``.

    A hung vLLM whose CUDA worker died leaves its model-shard VRAM orphaned
    (a compute-apps PID with no ``/proc`` entry) while the EngineCore main
    process stays alive burning Python-overhead CPU. That advancing session
    CPU makes the #518/#658 override rescue the stall conjunction to
    ``running`` (or the run never meets the conjunction at all because the
    dead allocation reads as GPU-busy), so a 60+ min hang is reported healthy
    throughout (#664 round 8). The dead-PID GPU allocation is the one
    mechanical signal of this hang.

    But the bare signature is false-positive on host-PID-namespace
    containers, where nvidia-smi reports HOST PIDs unresolvable in the
    container's ``/proc`` for EVERY healthy worker (#816 steady-state; #778 a
    transient teardown-window PID between vLLM engine cycles) — and a false
    ``stalled`` routes to a destructive kill-by-PID reaper respawn. So the
    override fires ONLY when (a) every workload log is stale past the
    effective stall window ``max(ZOMBIE_VETO_FRESH_SEC, stall_sec)`` — a
    genuinely hung run's own processes stop appending; both observed false
    positives had fresh logs — AND (b) the stale-log candidate persisted 2
    CONSECUTIVE observed ticks (``zombie_streak`` in the state sidecar;
    recomputed each tick, so any veto / candidate-free / non-running tick
    that reaches ``_save_state`` resets it — a tick that early-returns before
    ``_save_state`` neither advances nor resets it, the same exposure
    ``ssh_fail_count`` has; the sidecar is single-poller by design).

    Bare session-CPU *advancement* is deliberately NOT a veto term: the
    genuine #664 hang had CPU advancing (in the stale-log + idle-GPU regime
    ``running`` is only reachable via the CPU rescue), so an any-delta
    (> ``SESSION_CPU_ADVANCE_EPSILON_SECS``, the #518/#658 boolean) CPU veto
    would make the true positive structurally unreachable. A MATERIAL
    sustained burn *rate* IS a veto term (#951, next paragraph): the measured
    #664 churn (~0.22 cores) cannot reach the rate threshold, so the true
    positive stays reachable. Never touches a ``done`` / ``gate`` / ``dead``
    verdict — those are correct terminal/park states (a dead launcher is
    already ``dead``; its own orphaned allocation is then expected). The
    ``stall_reason`` lets the orchestrator route this distinctly from a
    generic log+GPU+CPU stall.

    Fresh-output veto (#1033, right after the #826 fresh-log veto —
    identical mechanics): a run whose ISSUE-KEYED OUTPUT artifacts
    (``output_mtime_ago``, from the #1033 probe) were modified within the
    same effective stall window ``max(ZOMBIE_VETO_FRESH_SEC, stall_sec)``
    is writing results, not hanging — #813's CPU-bound analysis tail wrote
    per-cell NPZs / JSONs for hours while every log was quiet. Veto +
    streak reset, exactly like the fresh-log veto. The parameter defaults
    to ``inf`` (= "no fresh output" / probe absent / fold disabled), so
    every pre-#1033 caller and test is byte-unchanged (fail-safe: the veto
    stays inert). A genuinely hung run writes no outputs, so the #664 true
    positive stays reachable.

    Material-compute liveness veto (#951, between the #1033 fresh-output
    veto and the streak defer/fire branches): when the per-tick session-CPU burn
    rate was >= ``ZOMBIE_OVERRIDE_CPU_CORES_MIN`` (default 0.5 cores) on
    BOTH the current tick (``session_cpu_rate_cores``, computed by
    ``poll_once`` via ``_session_cpu_rate_cores``) AND the previous
    persisted tick (the sidecar's ``session_cpu_rate_cores`` key), the
    session is demonstrably computing — #825's falsely-flagged live fit
    burned ~1.83-2.04 cores while 1816 MiB of prior-run VRAM leftover
    carried the zombie signature — so the override is suppressed and the
    streak RESETS to 0 (identical mechanics to the #826 fresh-log and #864
    namespace vetoes; reset, not hold, so a later degraded CPU read after
    material compute defers a full fresh 2-tick window instead of firing
    immediately). Fail-safe on ANY degraded input (missing/unparseable CPU
    sample, missing previous-tick baseline or rate, missing/garbage sample
    timestamp, tick spacing under ``ZOMBIE_CPU_RATE_MIN_DT_SEC``, negative
    delta): the rate is None / below threshold, the veto stays inert, and
    behavior is exactly the pre-#951 #826/#864 cascade. Residual exposure
    (ACCEPTED): ``session_cpu_secs`` is a session-TOTAL, so a WIDE hung
    session (TP>=2 NCCL spin at ~1 core/rank) or a co-resident same-session
    CPU burner can sustain >= the threshold indefinitely and keep a true
    zombie vetoed — the same exposure class as the #826 fresh-log sibling
    (a sibling process appending to a log forever suppresses the override
    the same way today), bounded by the GPU-idle advisory/escalation tiers,
    the #873 phase-ETA tripwires, the watcher wedge arm, and the
    ``EPM_ZOMBIE_OVERRIDE_CPU_CORES_MIN`` knob. Warmup: on a fresh or
    pre-#951 sidecar the veto cannot engage until the 3rd tick — tick 1
    persists the sample epoch with rate ``"unknown"``, tick 2 computes the
    first rate but has no prev-tick rate — which is the fail-safe direction
    (current behavior during warmup); a false stall in that window is the
    warmup, not a fix failure.

    Namespace-informativeness gate (#864, FIRST branch): the #826 stale-log
    veto lapses when a HEALTHY workload legitimately silences its logs
    longer than the stall window (#813: a ~29-min CPU-bound NPZ-compression
    stretch false-fired the override twice while ``cpu_advancing`` was true
    and the flagged "dead" VRAM holders were the same live cells seen under
    host-namespace PIDs). The gate keys on whether the dead-in-/proc probe
    signal is INFORMATIVE on this pod, not on generic liveness (per the
    paragraph above, ``cpu_advancing`` / pgrep-liveness vetoes would kill
    the true positive). Truth table over the #864 probe counts
    (veto = suppress + streak reset; fall-through = the #826 logic below,
    unchanged)::

        total  resolvable  uvm_holders  ->  action
        >0     0           >0               VETO (regime X: live workers
                                            under host ids; #813/#816)
        >0     0           0                fall through (#664 total
                                            collapse — no live CUDA holder)
        >0     >0          any              fall through (namespace
                                            informative; flagged PIDs are
                                            genuinely reaped)
        unknown/0  any     any              fall through (degraded probe)
        >0     0           unknown          fall through (UVM scan failed)

    Every degraded read fails toward CURRENT (#826) behavior — never toward
    more false positives, never toward disabled TP detection. Residual
    notes: (a) a cuInit'd-but-allocation-free parent/coordinator holding an
    exact ``/dev/nvidia-uvm`` fd while ABSENT from compute-apps would veto a
    TOTAL collapse — this exposure is TOTAL-COLLAPSE-scoped, not
    regime-scoped (a matched-namespace total collapse also reads
    ``resolvable == 0``); mitigations are the #864 pre-merge live-pod gate
    (which FOUND such a holder, ``issue813_dispatch.py``, hence the
    shipped default-OFF) and the ``EPM_ZOMBIE_NAMESPACE_VETO`` kill-switch.
    A live NON-workload CUDA process (e.g. a human SSH debug session
    holding a torch context) on a collapsed pod is the same family. Only a
    PARTIAL death on a matched pod (``resolvable > 0``, row 3) is immune.
    (b) On a mismatched-namespace pod a PARTIAL worker death (one dead
    worker among live uvm-holding cells) is vetoed — undetectable by this
    probe. Not a regression in *correct* detection: the /proc signal
    carries zero per-PID information in that regime, and pre-#864 behavior
    "detected" that case only by also firing on every healthy #813-shape
    run. The GPU-idle advisory/escalation tiers (#518/#537/#664) and the
    #873 phase-ETA tripwires are the backstops there; matched-namespace
    partial death keeps firing exactly as today. (c)
    ``ZOMBIE_NAMESPACE_VETO_ENABLED`` is read at module import — a live
    poller needs a restart for an ops flip to take effect. (d) On a tick
    matching BOTH this gate and the #826 fresh-log veto — or the #951
    material-CPU veto — the namespace WARNING fires first (outcome
    identical — ``running``, streak 0; only the forensic log line
    differs).
    """
    stall_reason: str | None = None
    zombie_streak = 0
    if status == "running" and zombie_gpu_pids:
        if (
            ZOMBIE_NAMESPACE_VETO_ENABLED
            and gpu_pids_total is not None
            and gpu_pids_total > 0
            and gpu_pids_resolvable == 0
            and uvm_live_holders is not None
            and uvm_live_holders > 0
        ):
            # #864: the dead-in-/proc signature is a PID-namespace artifact,
            # not a death signal — nvidia-smi reports host-namespace PIDs
            # that resolve in the container /proc for ZERO compute apps,
            # while live in-container processes hold /dev/nvidia-uvm (a live
            # CUDA compute context). The flagged "zombies" ARE those live
            # workers seen under host ids (#813: a healthy 29-min CPU-bound
            # quiet stretch outlived the #826 stale-log veto). Veto
            # regardless of log staleness; a genuine total collapse (#664)
            # has zero live uvm holders and falls through to the #826
            # stale-log + 2-tick logic below.
            log.warning(
                "zombie-GPU signature on pod %s (PID(s) %s) is a PID-namespace "
                "artifact: 0/%d compute PIDs resolve in the container /proc while "
                "%d live container process(es) hold /dev/nvidia-uvm — vetoing "
                "(#813/#864), not flagging",
                pod,
                ",".join(zombie_gpu_pids),
                gpu_pids_total,
                uvm_live_holders,
            )
            return status, stall_reason, cpu_override_active, 0
        zombie_veto_sec = max(ZOMBIE_VETO_FRESH_SEC, stall_sec)
        freshest_log_ago = min(last_mtime_ago, phase_log_mtime_ago, shard_log_mtime_ago)
        try:  # defensive parse, mirrors _update_ssh_fail_tracking's ssh_fail_count guard
            prev_zombie_streak = int(prev_state.get("zombie_streak", "0") or 0)
        except (TypeError, ValueError):
            prev_zombie_streak = 0
        # #951: the previous tick's persisted burn rate. _parse_session_cpu
        # maps an absent key / "unknown" / garbage to None (fail-safe — the
        # material-CPU veto below then cannot fire).
        prev_cpu_rate = _parse_session_cpu(prev_state.get("session_cpu_rate_cores", "unknown"))
        if freshest_log_ago <= zombie_veto_sec:
            log.warning(
                "zombie-GPU signature on pod %s (PID(s) %s) but log evidence is fresh "
                "(%.0fs <= %ds) — liveness veto, not flagging (#826; host-PID-namespace "
                "containers report unresolvable host PIDs for healthy workers)",
                pod,
                ",".join(zombie_gpu_pids),
                freshest_log_ago,
                zombie_veto_sec,
            )
        elif output_mtime_ago <= zombie_veto_sec:
            # #1033: fresh issue-keyed OUTPUT artifact — the run is writing
            # results while its logs are quiet (#813's CPU-bound analysis
            # tail). Identical mechanics to the fresh-log veto above:
            # suppress + streak reset (zombie_streak stays 0 — recomputed
            # each tick).
            log.warning(
                "zombie-GPU signature on pod %s (PID(s) %s) but output-artifact evidence "
                "is fresh (%.0fs <= %ds) — liveness veto, not flagging (#1033/#813; a "
                "CPU-bound analysis tail writes issue-keyed outputs while every log is "
                "quiet)",
                pod,
                ",".join(zombie_gpu_pids),
                output_mtime_ago,
                zombie_veto_sec,
            )
        elif (
            session_cpu_rate_cores is not None
            and prev_cpu_rate is not None
            and session_cpu_rate_cores >= ZOMBIE_OVERRIDE_CPU_CORES_MIN
            and prev_cpu_rate >= ZOMBIE_OVERRIDE_CPU_CORES_MIN
        ):
            # #951: material compute — the session burned >= T cores on BOTH
            # of the last two persisted ticks. #664's hung EngineCore churns
            # ~0.22 cores; real work (the #825 fit at ~1.9 cores) cannot be a
            # hang. Veto; streak resets (zombie_streak stays 0 — recomputed
            # each tick).
            log.warning(
                "zombie-GPU signature on pod %s (PID(s) %s) with all logs stale BUT session "
                "CPU burned %.2f / %.2f cores over the last two ticks (>= %.2f) — material "
                "compute, liveness veto, not flagging (#951; #825: prior-run VRAM leftover "
                "while the live workload computed)",
                pod,
                ",".join(zombie_gpu_pids),
                session_cpu_rate_cores,
                prev_cpu_rate,
                ZOMBIE_OVERRIDE_CPU_CORES_MIN,
            )
        elif prev_zombie_streak < 1:
            zombie_streak = 1
            log.warning(
                "zombie-GPU signature on pod %s (PID(s) %s) with all logs stale — deferring "
                "one tick for 2-tick persistence (#826)",
                pod,
                ",".join(zombie_gpu_pids),
            )
        else:
            # #951 forensic evidence on the FIRE path: the two burn rates
            # (or "unknown") are the durable record for tuning
            # EPM_ZOMBIE_OVERRIDE_CPU_CORES_MIN after the next incident —
            # the sidecar keeps only the last tick.
            log.error(
                "zombie GPU allocation on pod %s: compute PID(s) %s hold >= %d MiB VRAM but "
                "are absent from /proc (dead CUDA worker, vLLM EngineCore hung) — persisted "
                "2 consecutive ticks with all logs stale, overriding "
                "status=running -> stalled (#664/#826); session-CPU rate now=%s prev=%s "
                "cores (veto threshold %.2f)",
                pod,
                ",".join(zombie_gpu_pids),
                ZOMBIE_GPU_MEM_MIN_MIB,
                "unknown" if session_cpu_rate_cores is None else f"{session_cpu_rate_cores:.2f}",
                "unknown" if prev_cpu_rate is None else f"{prev_cpu_rate:.2f}",
                ZOMBIE_OVERRIDE_CPU_CORES_MIN,
            )
            status = "stalled"
            stall_reason = "vllm_worker_dead_zombie_gpu"
            cpu_override_active = False
            zombie_streak = prev_zombie_streak + 1
    return status, stall_reason, cpu_override_active, zombie_streak


def _parse_output_mtime_epoch(raw: str | None) -> int:
    """Parse the probe's ``OUTPUT_MTIME_EPOCH`` scalar defensively (#1033 r2).

    ``_parse_probe_stdout`` stores the scalar text VERBATIM, so a malformed /
    non-numeric value — reachable via version skew (a pod running newer probe
    code than this VM, the same scenario the kill-switch branch in
    ``poll_once`` defends on the DISABLED side) or a garbled/partial SSH
    line — must fail INERT, not kill the tick: return ``0``, which
    ``_log_staleness_secs`` maps to the ``10**9`` absent sentinel (the
    pre-#1033 no-fold behavior). Every numeric value parses exactly as the
    previous inline ``int(raw or "0")`` did. Guards ONLY the tolerant-parse
    keys — this #1033 ``OUTPUT_MTIME_EPOCH`` scalar and the #1156 reads inside
    ``_maybe_warn_stale_pid_file`` (``PID_FILE_MTIME_EPOCH`` + its
    ``POD_NOW_EPOCH`` basis) — the sibling probe ints keep their
    long-standing strict parses.
    """
    try:
        return int(raw or "0")
    except (ValueError, TypeError):
        return 0


def _log_staleness_secs(
    *,
    pod: str,
    vm_now_epoch: int,
    pod_now_epoch: int,
    freshest_mtime_epoch: int,
    phase_log_mtime_epoch: int,
    shard_log_mtime_epoch: int,
    output_mtime_epoch: int = 0,
) -> tuple[int, int, int, int]:
    """Compute the staleness deltas (top-level/cell, per-phase, shard logs,
    plus the #1033 output artifact) on a SINGLE clock basis (#704).

    The pod stamps file mtimes with its OWN wall clock (``stat -c %Y``); the
    probe heredoc now also captures that same clock's "now" (``date +%s`` ->
    ``pod_now_epoch``), so subtracting ``pod_now - pod_mtime`` cancels any
    pod-vs-VM wall-clock drift exactly. When ``pod_now_epoch`` is absent
    (``0``: a legacy pod image whose probe pre-dates this, an ssh failure
    whose fallback dict zeroes it, or any probe omitting the line) the
    function falls back to the VM clock (``vm_now_epoch``) AND logs a WARN,
    preserving the pre-#704 behavior for older images and the existing tests
    (which omit the key) while making the drift-prone basis visible in logs.

    Each delta is ``10**9`` (the absent-log sentinel) when its mtime is
    ``<= 0``, matching the prior inline behavior. On the pod-clock branch the
    deltas are low-clamped at ``0`` (``max(0, ...)``) so a sub-second
    rounding within one probe cannot produce a negative "seconds ago"; the
    VM-fallback branch keeps its exact pre-#704 arithmetic for the three LOG
    deltas so the backward-compat behavior is byte-for-byte unchanged. The
    #1033 ``output_mtime_ago`` delta (new — no backward-compat constraint)
    is low-clamped at ``0`` on BOTH branches: a FUTURE-DATED output mtime
    (weird writer clock) reads as "fresh right now", which can only
    SUPPRESS a stall verdict / zombie override — the fold's accepted
    fail direction — never create a new positive.

    Returns ``(last_mtime_ago, phase_log_mtime_ago, shard_log_mtime_ago,
    output_mtime_ago)``.
    """
    if pod_now_epoch > 0:
        staleness_now = pod_now_epoch
        last_mtime_ago = (
            max(0, staleness_now - freshest_mtime_epoch) if freshest_mtime_epoch > 0 else 10**9
        )
        phase_log_mtime_ago = (
            max(0, staleness_now - phase_log_mtime_epoch) if phase_log_mtime_epoch > 0 else 10**9
        )
        shard_log_mtime_ago = (
            max(0, staleness_now - shard_log_mtime_epoch) if shard_log_mtime_epoch > 0 else 10**9
        )
        output_mtime_ago = (
            max(0, staleness_now - output_mtime_epoch) if output_mtime_epoch > 0 else 10**9
        )
        return last_mtime_ago, phase_log_mtime_ago, shard_log_mtime_ago, output_mtime_ago

    staleness_now = vm_now_epoch
    log.warning(
        "probe missing POD_NOW_EPOCH on pod %s; falling back to VM-clock log "
        "staleness (subject to pod-vs-VM clock drift, #704)",
        pod,
    )
    last_mtime_ago = staleness_now - freshest_mtime_epoch if freshest_mtime_epoch > 0 else 10**9
    phase_log_mtime_ago = (
        staleness_now - phase_log_mtime_epoch if phase_log_mtime_epoch > 0 else 10**9
    )
    shard_log_mtime_ago = (
        staleness_now - shard_log_mtime_epoch if shard_log_mtime_epoch > 0 else 10**9
    )
    output_mtime_ago = (
        max(0, staleness_now - output_mtime_epoch) if output_mtime_epoch > 0 else 10**9
    )
    return last_mtime_ago, phase_log_mtime_ago, shard_log_mtime_ago, output_mtime_ago


def _tail_excerpt_and_crash_signature(
    probe: dict[str, str],
    *,
    status: str,
    mtime_epoch: int,
    cell_mtime_epoch: int,
    phase_log_mtime_epoch: int = 0,
    shard_log_mtime_epoch: int = 0,
) -> tuple[str, str | None]:
    """Slice the (5-line excerpt, WIDE crash signature) from the freshest log tail.

    The fresher log is the WIDE 500-line surface the probe already fetched.
    #791: the freshest is selected by mtime-argmax over ALL FOUR log layouts —
    {main, cell, phase, shard} — not just {main, cell}. Before #791 the excerpt
    + crash signature were pinned to {main, cell}, so a multi-arm run whose later
    arm wrote ONLY to a per-phase (``issue-<N>-<arm>.log``) or shard
    (``logs/issue_<N>/*.log``) layout surfaced a STALE main-log tail — the
    staleness verdict already unioned all four layouts (#468/#488), but the tail
    excerpt (notifications) + the ``status=dead`` crash signature (which feeds
    the CUDA-IMA / OUR_CODE_FRAME failover predicates in ``backend_poll``) did
    not, so a watcher acted on the wrong surface.

    Selection: pick the source with the largest mtime among those with a
    NON-EMPTY tail; on a tie or when no fresher source has a tail, fall back to
    the main-log tail (unchanged behavior for non-cell/non-phase/non-shard runs,
    and for a run whose only tail is the main log). An empty tail is never
    selected even when its mtime is the freshest, so a fresh-but-empty phase log
    (e.g. just-created, nothing written yet) does not blank out a populated
    main-log excerpt.

    The notification excerpt is the freshest tail's last 5 lines. The #775
    ``crash_signature`` is the WHOLE freshest wide tail on a ``status=="dead"``
    poll (``None`` otherwise) — NOT the 5-line excerpt, which truncates a 20-50
    line vLLM CUDA-IMA traceback so a signature match on it would silently never
    fire. The whole wide tail is stored so the failover predicate can ALSO scan
    it for the OUR_CODE_FRAME exclusion. Pure (no SSH); extracted from
    :func:`poll_once` so the slice logic is unit-testable without driving the
    full poller (the #775 B2 test binds to THIS helper).
    """
    candidates = [
        (mtime_epoch, probe["log_tail"]),
        (cell_mtime_epoch, probe.get("cell_log_tail", "")),
        (phase_log_mtime_epoch, probe.get("phase_log_tail", "")),
        (shard_log_mtime_epoch, probe.get("shard_log_tail", "")),
    ]
    # Pick the freshest source with a NON-EMPTY tail. `max` returns the FIRST
    # element on an mtime tie, and `log_tail` (the main log) is first in the
    # list, so a tie deterministically resolves to the main log. When no source
    # has a tail, `default` falls back to the main-log tail.
    _, wide_tail = max(
        ((m, t) for m, t in candidates if t),
        default=(mtime_epoch, probe["log_tail"]),
        key=lambda mt: mt[0],
    )
    tail_excerpt = "\n".join(wide_tail.splitlines()[-5:])
    crash_signature = wide_tail if status == "dead" else None
    return tail_excerpt, crash_signature


def poll_once(
    *,
    issue: int,
    pod: str,
    log_path: str,
    pid_file: str,
    state_file: Path,
    stall_sec: int = DEFAULT_STALL_SEC,
) -> PollResult:
    # Drain pod-side sentinels FIRST — posting any pending markers from the
    # VM. A user-gate sentinel (e.g. epm:fact-candidates) takes precedence
    # over the phase=done check so the orchestrator parks at the gate even
    # if the pipeline subsequently reached done.
    sentinels_processed, gate = _drain_sentinels(issue=issue, pod=pod)

    # Self-correction for a stale pidfile (incident #451): on a re-launch
    # the on-pod pidfile can hold the dead first-run PID while the live
    # python child runs under a new PID carried by the latest
    # epm:run-launched marker. Cross-check the marker pid so a healthy
    # re-run is not misreported as dead.
    marker_pid = _marker_pid(issue)
    probe = _ssh_probe(pod, log_path, pid_file, issue, marker_pid, stall_sec=stall_sec)

    # ── #488 stale-port self-heal ────────────────────────────────────────
    # Track consecutive SSH-probe failures across ticks. When the live API
    # has moved a pod's SSH endpoint to a new port but ``pods.conf`` still
    # holds the pre-stop value, every probe lands on a dead address and
    # this counter accumulates. Once it crosses
    # ``SSH_FAIL_REFRESH_THRESHOLD`` we shell out to ``pod.py config
    # --refresh-from-api <pod>`` once (fail-soft) to pull the current
    # host/port from the live API into ``pods.conf`` + ``~/.ssh/config``,
    # then reset the counter so the NEXT N consecutive failures will
    # retry. This is the auto-heal that closes the gap left by the
    # #488 manual recovery (the new ``--refresh-from-api`` subcommand
    # already exists; this is the wiring that uses it without a human in
    # the loop).
    prev_state = _load_state(state_file, issue)
    ssh_failed = probe.get("ssh_failed") == "1"
    ssh_fail_count, ssh_fail_since, ssh_wait_alarm_ts = _update_ssh_fail_tracking(
        prev_state, ssh_failed=ssh_failed, pod=pod, issue=issue
    )

    pidfile_pid_alive = probe["pid_alive"] == "1"
    marker_pid_alive = marker_pid is not None and probe["marker_pid_alive"] == "1"
    pid_alive = pidfile_pid_alive or marker_pid_alive
    # Observability for the #521 false-dead diagnosis: surface "the pid
    # FILE was absent" (vs "the pid probed dead") in the tick JSON, and
    # warn when the epm:run-launched marker pid is the fallback standing
    # in for it. Status routing is deliberately untouched — ``pid_alive``
    # already ORs in the marker pid.
    pid_file_missing = probe.get("pid_file_missing") == "1"
    if pid_file_missing and marker_pid is not None:
        log.warning(
            "pid file %s absent on pod %s; using epm:run-launched marker pid %d fallback",
            pid_file,
            pod,
            marker_pid,
        )
    mtime_epoch = int(probe["mtime_epoch"] or "0")
    cell_mtime_epoch = int(probe["cell_mtime_epoch"] or "0")
    phase_log_mtime_epoch = int(probe["phase_log_mtime_epoch"] or "0")
    shard_log_mtime_epoch = int(probe.get("shard_log_mtime_epoch") or "0")
    # Staleness folds in the newest cell log (#405). A sequential smoke
    # cell blocks the dispatcher in `proc.wait()` for ~15-18 min while the
    # cell process actively trains+evals and writes to its own log; the
    # main log goes silent for that window. Take the freshest of (main,
    # newest cell) so a healthy single-cell phase reads as running, not
    # false-stalled / false-dead. Phase detection stays on the MAIN log
    # because the `[phase=...]` line is written by the dispatcher (cells
    # log training steps, not phase transitions).
    freshest_mtime_epoch = max(mtime_epoch, cell_mtime_epoch)
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    # Single-clock staleness basis (#704), computed in a helper to keep
    # ``poll_once`` below the C901 cap. ``now_epoch`` (the VM clock above) is
    # deliberately NOT redefined: it is reused below for the run-age (#521),
    # GPU-idle advisory (#518/#537), and phase-change sidecar (#669)
    # computations, all of which compare against VM-STAMPED timestamps and
    # would be corrupted by a pod-clock basis.
    last_mtime_ago, phase_log_mtime_ago, shard_log_mtime_ago, output_mtime_ago = (
        _log_staleness_secs(
            pod=pod,
            vm_now_epoch=now_epoch,
            pod_now_epoch=int(probe.get("pod_now_epoch") or "0"),
            freshest_mtime_epoch=freshest_mtime_epoch,
            phase_log_mtime_epoch=phase_log_mtime_epoch,
            shard_log_mtime_epoch=shard_log_mtime_epoch,
            output_mtime_epoch=_parse_output_mtime_epoch(probe.get("output_mtime_epoch")),
        )
    )
    # #1033 kill switch: with the fold disabled the probe block was omitted
    # (the parser's "0" default already yields the inert 10**9), but force
    # the sentinel explicitly so a stray OUTPUT_MTIME_EPOCH line in the
    # stdout (e.g. a pod running newer code than the VM) can never engage
    # a disabled fold.
    if not OUTPUT_MTIME_FOLD_ENABLED:
        output_mtime_ago = 10**9
    gpu_util = probe.get("gpu_util", "unknown")
    gpu_idle = _gpu_idle(gpu_util)
    # Zombie-GPU-allocation signal (#664): the probe emits the
    # space-separated PIDs of compute processes holding >= the VRAM floor
    # whose `/proc/<pid>` no longer exists (a dead CUDA worker whose
    # allocation lingers). Empty string = healthy (no zombies); a missing
    # key (older probe / ssh-failure fallback) reads as empty too.
    zombie_gpu_pids = [p for p in probe.get("zombie_gpu_pids", "").split() if p]
    current_phase = latest_phase(probe["log_tail"])

    # ── #545 done corroboration ──────────────────────────────────────────
    # A `[phase=done]` parse alone is NOT proof of completion: per-cell
    # eval subprocesses print `[phase=done] eval cell <X> complete` lines
    # MID-RUN, and on 2026-06-11 a tick reported status=done while the
    # dispatcher pid was alive and GPUs were at 85% — an orchestrator
    # trusting that would Step-8 terminate a live pod mid-sweep. The noise
    # is textually indistinguishable from legitimate suffixed TERMINAL
    # lines (`[phase=done] SMOKE COMPLETE ...`), so instead of tightening
    # the regex we require corroboration: the pid being dead (a normal
    # completion exits within seconds of its done line) OR a results
    # sentinel existing on the pod (covers a post-done lingering
    # dispatcher; includes `.processed` — this tick's drain renames it
    # before we get here). An uncorroborated done is demoted to the
    # latest NON-done phase so the status verdict falls through to the
    # normal liveness arbiters AND the milestone tracker below never
    # posts a false `-> done` transition.
    results_sentinel_present = probe.get("results_sentinel_present") == "1"
    if current_phase == "done" and pid_alive and not results_sentinel_present:
        demoted_phase = latest_phase(probe["log_tail"], skip_done=True)
        log.warning(
            "[phase=done] parsed from log tail on pod %s but pid is ALIVE and no "
            "results sentinel exists — treating as mid-run noise (#545); "
            "phase %s -> %s, status falls to liveness arbiters",
            pod,
            current_phase,
            demoted_phase,
        )
        current_phase = demoted_phase

    # Decide status. Gate sentinel wins over done — a user must answer
    # before the pipeline (or the orchestrator) advances further. The
    # phase=done check still runs (we want to know the pipeline finished)
    # but ``status`` reflects the gate so the orchestrator parks.
    # ``current_phase == "done"`` here is already CORROBORATED (#545 block
    # above): an uncorroborated done-parse was demoted before this point,
    # so reaching the done branch implies pid-dead OR results-sentinel.
    # `dead` requires BOTH the pidfile PID and the marker PID to be dead
    # (pid_alive is their OR) AND the log not to show completion — a stale
    # pidfile alone never declares a live marker-PID run dead. The
    # `current_phase == "done"` precedence already covers the
    # "log-shows-completion" half: a completed run is `done`, never `dead`.
    #
    # `stalled` requires ALL SIX liveness-of-output signals to agree:
    # the top-level log AND the freshest cell log (folded together as
    # `last_mtime_ago`, #405) AND every per-phase log under
    # `/workspace/logs/issue-<N>-*.log` (#468) AND every shard /
    # repo-rooted phase log under `/workspace/explore-persona-space/
    # logs/issue_<N>{,_*}/*.log` (#488) plus every dispatcher per-job
    # log under `/workspace/explore-persona-space/eval_results/
    # issue_<N>{,_*}/logs/*.log` (#521, folded into the same shard-log
    # max) AND every issue-keyed OUTPUT artifact (eval_results/
    # issue_<N>{,_*}/, data/issue_<N>/, data/issue<N>/ — #1033; a fresh
    # output behaves exactly like a fresh shard log: first-class
    # liveness, no new override concept) AND the GPUs must ALL be
    # quiet/idle for >STALL_SEC. The shard-log conjunction (#488)
    # prevents a false stall when a multi-GPU launcher fans out per-GPU
    # shard logs under a subdirectory and the inner loop's per-shard
    # write cadence (e.g. ~3 min between writes for i488 Pass B across
    # 57 cells) exceeds the 30-min threshold on the main log alone —
    # in that pattern the main + cell + per-phase logs all go silent
    # while the shard logs are actively appended. `_gpu_idle` remains
    # fail-safe (returns False on nvidia-smi error / unknown), so a
    # healthy long phase whose shard log OR per-phase log is fresh OR
    # whose GPU is busy will stay in `running` even if nvidia-smi is
    # unavailable.
    # Session-CPU advancing check (#518): even when every log-mtime
    # signal AND the GPU-idle signal agree on "stalled", a launcher
    # whose process session has accrued more cumulative CPU since the
    # previous tick is doing CPU-bound work (e.g. the scoring_syco
    # phase that polls a judge batch and aggregates results — silent
    # on logs for hours, GPUs idle by design, but the python child is
    # at 100% CPU). Override `stalled` -> `running` when CPU is
    # advancing; preserve `stalled` when CPU is flat or unknown
    # (fail-safe). The very first tick after launch has no prior
    # observation, so `_session_cpu_advancing` returns None and the
    # decision falls back to the older log+GPU arbiters: a freshly-
    # launched run cannot meet the >stall_sec mtime conjunction on
    # the first tick (the logs ARE fresh), so this code path doesn't
    # change first-tick semantics. From the second tick onward, a
    # truly hung session (CPU flat AND logs stale AND GPUs idle)
    # still routes to `stalled` and the orchestrator still fires
    # epm:failure.
    # Compare the current cumulative-CPU sample against the running MAXIMUM
    # observed across all prior ticks, not the immediately-previous tick
    # (#658). A multi-shard run de-counts an exited shard's cputime from the
    # live ps-sum, so ``current`` can drop tick-over-tick while healthy; a
    # drop below the high-water mark is that child-exit artifact, never a
    # hang. ``_session_cpu_advancing`` returns True on either a new max
    # (genuine progress) or a sub-max drop (child-exit artifact), False only
    # when flat at the max.
    current_session_cpu = probe.get("session_cpu_secs", "unknown")
    prev_max_session_cpu = prev_state.get("max_cpu_secs", prev_state.get("session_cpu_secs"))
    cpu_advancing = _session_cpu_advancing(prev_max_session_cpu, current_session_cpu)
    # Roll the high-water mark forward for the next tick (#658). The max only
    # ever grows; a current sample below it (a child exit) does not lower it.
    max_session_cpu = _roll_session_cpu_max(prev_max_session_cpu, current_session_cpu)
    # #951: per-tick burn rate vs the PREVIOUS tick's raw sample (not the max —
    # the delta between consecutive samples is the current burn), for the
    # material-compute veto on the zombie override.
    session_cpu_rate = _session_cpu_rate_cores(
        prev_state.get("session_cpu_secs"),
        prev_state.get("session_cpu_sample_epoch"),
        current_session_cpu,
        now_epoch,
    )

    # True when the verdict below is `running` ONLY because the #518
    # CPU-advancing override rescued a met stall conjunction (logs stale +
    # GPUs idle). Healthy, but a degraded-observability regime — the
    # adaptive interval (§7) keeps such ticks on the short interval.
    cpu_override_active = False
    if gate is not None:
        status = "gate"
    elif current_phase == "done":
        status = "done"
    elif not pid_alive:
        status = "dead"
    elif (
        last_mtime_ago > stall_sec
        and phase_log_mtime_ago > stall_sec
        and shard_log_mtime_ago > stall_sec
        and output_mtime_ago > stall_sec  # NEW (#1033): fresh output = alive
        and gpu_idle
    ):
        if cpu_advancing is True:
            cpu_override_active = True
            log.info(
                "stall conjunction met (logs >%ds + GPUs idle) BUT session CPU "
                "advancing (current=%s vs running-max=%s) on pod %s (#518/#658 "
                "silent CPU-bound override); reporting status=running",
                stall_sec,
                current_session_cpu,
                prev_max_session_cpu,
                pod,
            )
            status = "running"
        else:
            status = "stalled"
    else:
        status = "running"

    # ── #664/#826 zombie-GPU-allocation override ─────────────────────────
    # Extracted to `_apply_zombie_override` (both for C901 headroom and so
    # the firing predicate is documented in one place — see its docstring).
    status, stall_reason, cpu_override_active, zombie_streak = _apply_zombie_override(
        status=status,
        zombie_gpu_pids=zombie_gpu_pids,
        stall_sec=stall_sec,
        last_mtime_ago=last_mtime_ago,
        phase_log_mtime_ago=phase_log_mtime_ago,
        shard_log_mtime_ago=shard_log_mtime_ago,
        prev_state=prev_state,
        pod=pod,
        cpu_override_active=cpu_override_active,
        gpu_pids_total=_parse_probe_count(probe.get("gpu_pids_total")),
        gpu_pids_resolvable=_parse_probe_count(probe.get("gpu_pids_resolvable")),
        uvm_live_holders=_parse_probe_count(probe.get("nvidia_uvm_live_holders")),
        session_cpu_rate_cores=session_cpu_rate,
        output_mtime_ago=output_mtime_ago,
    )

    # ── #873/#1033 run-scoped state anchor (AC #6) ───────────────────────
    # A fresh epm:run-launched (relaunch / same-issue follow-up round)
    # re-arms BOTH #873 tripwires AND (since #1033) the GPU-idle
    # advisory/escalation clock: the stored ETA/width dedup keys and the
    # idle span/dedup sets belong to the PREVIOUS run, so
    # _tripwire_run_scope clears them (_RUN_SCOPED_STATE_KEYS) before any
    # consumer runs. Runs ABOVE the idle-advisory calls (moved here at
    # #1033 — pre-#1033 it sat below them, so the idle tier read the
    # UNSCOPED state and a relaunch inherited the previous run's span:
    # #763 printed a "543 min" idle advisory on a ~17-min-old instance).
    # ``run_age_sec`` is computed ONCE here and reused by the
    # adaptive-interval relaunch clamp + the phase-ETA fallback below
    # (one events.jsonl read per tick). The zombie override above and the
    # #983 post-done guard below deliberately keep reading the RAW
    # ``prev_state`` (zombie_streak scoping is out of #1033's scope; the
    # post-done guard has its own natural epoch).
    run_age_sec = _run_launched_age_sec(issue, now_epoch)
    # ── #1156 stale-pid-file-vs-marker WARN (observability-only) ─────────
    # Placed here (not at the #521 pid_file_missing block above) so it
    # reuses run_age_sec — keeping the "one events.jsonl read per tick"
    # invariant. Never contributes to status / stall / dead.
    pid_file_stale_vs_marker = _maybe_warn_stale_pid_file(
        issue=issue,
        pod=pod,
        pid_file=pid_file,
        probe=probe,
        run_age_sec=run_age_sec,
        pid_file_missing=pid_file_missing,
    )
    tripwire_state, tripwire_run_epoch = _tripwire_run_scope(
        prev_state, run_age_sec=run_age_sec, now_epoch=now_epoch
    )

    # ── #518/#537 GPU-idle advisory ──────────────────────────────────────
    # The stall verdict above treats an idle GPU only as corroboration, so
    # a HEALTHY run on a long CPU-only phase (fresh logs, every GPU at 0%)
    # burns pod-hours silently. Track the sustained healthy-and-all-idle
    # span across ticks (state-file backed, like ssh_fail_count) and post
    # a one-per-phase, non-blocking advisory marker once it exceeds
    # GPU_IDLE_ADVISORY_MIN minutes. Never flips ``status``. Reads the
    # RUN-SCOPED tripwire_state (#1033) so a relaunch restarts the idle
    # clock instead of inheriting the previous run's span — the reported
    # idle minutes are PER-INSTANCE/PER-RUN, never cumulative across
    # relaunches.
    gpu_idle_since_epoch, gpu_idle_advised_phases, gpu_idle_advisory_posted = (
        _maybe_post_gpu_idle_advisory(
            issue=issue,
            pod=pod,
            status=status,
            gpu_util=gpu_util,
            current_phase=current_phase,
            prev_state=tripwire_state,
            now_epoch=now_epoch,
        )
    )

    # ── #664 GPU-idle ESCALATION ──────────────────────────────────────────
    # A SECOND tier above the advisory: a MULTI-GPU pod idle past
    # GPU_IDLE_ESCALATION_MIN minutes in an upload/CPU-only phase fires a
    # Telegram push + a LOUD [gpu-idle-escalation] marker (never stops the
    # pod). Reads the SAME idle span the advisory just resolved
    # (gpu_idle_since_epoch) — no second idle clock — and the SAME
    # run-scoped tripwire_state (#1033).
    gpu_idle_escalated_phases, gpu_idle_escalation_posted = _maybe_escalate_gpu_idle(
        issue=issue,
        pod=pod,
        status=status,
        gpu_util=gpu_util,
        current_phase=current_phase,
        idle_since_epoch=gpu_idle_since_epoch,
        prev_state=tripwire_state,
        now_epoch=now_epoch,
    )

    # ── #873 m-of-N GPU-width advisory ───────────────────────────────────
    # A STABLE strict subset of GPUs idle >= GPU_WIDTH_ADVISORY_MIN minutes
    # on a multi-GPU pod while the run is healthy (#813 idle-width / #664
    # spend-leak class). Advisory only — never flips ``status``.
    gpu_width_since_epoch, gpu_width_idle_set, gpu_width_advised_phases, gpu_width_posted = (
        _maybe_post_gpu_width_advisory(
            issue=issue,
            pod=pod,
            status=status,
            gpu_util=gpu_util,
            current_phase=current_phase,
            prev_state=tripwire_state,
            now_epoch=now_epoch,
        )
    )

    # ── Under-parallelization warning (partial saturation; plan §3) ──────
    # DISTINCT from the #873 width advisory above: < 50% of the provisioned
    # GPUs busy for >= GPU_UNDERPARALLEL_WARNING_MIN (15m) on a multi-GPU pod
    # points at under-parallelization (check sharding), deduped ONCE PER RUN.
    # Reads the run-scoped tripwire_state so a relaunch re-arms it. Advisory
    # only — never flips ``status``, never stops anything.
    (
        gpu_underparallel_since_epoch,
        gpu_underparallel_warned,
        gpu_underparallel_posted,
    ) = _maybe_post_gpu_underparallel_warning(
        issue=issue,
        pod=pod,
        status=status,
        gpu_util=gpu_util,
        current_phase=current_phase,
        prev_state=tripwire_state,
        now_epoch=now_epoch,
    )

    # ── #983 post-done phase-consistency guard ───────────────────────────
    # Cross-tick audit of a previously-accepted done verdict: at the tick
    # where the corroborated ``current_phase == "done"`` lands, record the
    # matched done line; any LATER tick observing genuinely new
    # ``[phase=...]`` lines after that anchor fires ONE loud advisory
    # marker + Telegram push. Consumes the POST-#545-demotion
    # ``current_phase`` (wired BELOW the demotion block, so an
    # uncorroborated done can never arm an episode) and the pre-tripwire
    # ``prev_state`` (its run-scope clamp is the direct ``run_age_sec``
    # one, deliberately independent of the #873 anchor lifecycle).
    # Advisory only — never flips ``status``, never stops anything.
    (
        post_done_line,
        post_done_epoch,
        post_done_pod,
        post_done_posted_flag,
        post_done_posted,
        post_done_new_lines,
    ) = _maybe_post_post_done_phase_advisory(
        issue=issue,
        pod=pod,
        current_phase=current_phase,
        log_tail=probe["log_tail"],
        prev_state=prev_state,
        run_age_sec=run_age_sec,
        now_epoch=now_epoch,
    )

    # New milestone? (re-uses ``prev_state`` loaded above for the
    # ssh_fail_count tracking — we only read state once per tick.)
    prev_phase = prev_state.get("phase", "")
    new_milestone = current_phase != prev_phase and current_phase != "unknown"
    # Raw phase-transition fact for the adaptive-interval decision (§7),
    # captured BEFORE the marker post below can flip ``new_milestone`` to
    # False on a post failure — the boundary was crossed either way.
    phase_transitioned = new_milestone

    if new_milestone:
        try:
            post_event(
                issue,
                "epm:progress",
                by="poll_pipeline",
                note=f"phase transition: {prev_phase or '(start)'} -> {current_phase}",
                phase=current_phase,
                pod=pod,
            )
        except Exception as exc:
            log.error("post_event failed: %s", exc)
            new_milestone = False  # Don't claim we recorded it.

    # ── Adaptive bg-poll interval (§7) ───────────────────────────────────
    # Track WHEN the phase last changed (state-file backed, like
    # ssh_fail_count) so the quiet long interval only applies once the run
    # has been boundary-free for RECENT_PHASE_CHANGE_WINDOW_SEC. A missing
    # / garbled epoch reads as 0 -> "unknown" -> short interval (fail
    # toward coverage).
    try:
        last_phase_change_epoch = int(float(prev_state.get("last_phase_change_epoch", "0") or 0))
    except (TypeError, ValueError):
        last_phase_change_epoch = 0
    # ``run_age_sec`` was computed above at the #873 tripwire anchor (one
    # events.jsonl read per tick); reused here for the relaunch clamp.
    # Relaunch clamp (code-review 2026-06-12): the state file persists
    # across same-issue relaunches / follow-up rounds, so a boundary
    # recorded BEFORE the current run's launch (latest epm:run-launched)
    # is not evidence about THIS run. Without the clamp, a relaunch whose
    # first observed phase NAME matches the stale recorded one (train ->
    # train is common) would satisfy the recent-phase-change guard
    # vacuously and go quiet right after the early-run window. Clamp to 0
    # ("unknown") — short interval until a boundary is actually observed
    # in the current run (fail toward coverage).
    if (
        last_phase_change_epoch > 0
        and run_age_sec is not None
        and last_phase_change_epoch < now_epoch - run_age_sec
    ):
        last_phase_change_epoch = 0
    if phase_transitioned:
        last_phase_change_epoch = now_epoch
    phase_changed_ago_sec = (
        float(now_epoch - last_phase_change_epoch) if last_phase_change_epoch > 0 else None
    )
    next_interval = recommend_next_interval(
        status=status,
        gate=gate,
        sentinels_processed=sentinels_processed,
        phase_transitioned=phase_transitioned,
        ssh_failed=ssh_failed,
        gpu_idle_advisory_posted=gpu_idle_advisory_posted,
        gpu_idle_escalation_posted=gpu_idle_escalation_posted,
        cpu_override_active=cpu_override_active,
        run_age_sec=run_age_sec,
        phase_changed_ago_sec=phase_changed_ago_sec,
    )

    # ── #873 phase-ETA tripwire ──────────────────────────────────────────
    # Elapsed wall-clock (per current phase / whole run) vs the plan §9
    # planned_wall_h TOTAL; posts epm:compute-deviation (source: poller).
    # Placed AFTER the relaunch clamp + phase_transitioned update so it
    # sees the FINAL last_phase_change_epoch (D2). Advisory only — never
    # flips ``status``, never stops anything.
    eta_posted_keys, eta_posted, eta_budget_warned = _maybe_post_eta_deviation(
        issue=issue,
        pod=pod,
        status=status,
        current_phase=current_phase,
        last_phase_change_epoch=last_phase_change_epoch,
        run_age_sec=run_age_sec,
        prev_state=tripwire_state,
        now_epoch=now_epoch,
    )

    _save_state(
        state_file,
        issue,
        {
            "phase": current_phase,
            "last_mtime_epoch": str(mtime_epoch),
            # Adaptive-interval boundary tracking (§7).
            "last_phase_change_epoch": str(last_phase_change_epoch),
            "ssh_fail_count": str(ssh_fail_count),
            # 1h billing-pod SSH-wait alarm bookkeeping (refs #572): episode
            # start + last alarm ts, both 0.0 while SSH is reachable.
            "ssh_fail_since": str(ssh_fail_since),
            "ssh_wait_alarm_ts": str(ssh_wait_alarm_ts),
            # GPU-idle advisory span + per-phase de-dup (#518/#537). Phase
            # names match PHASE_RE ([a-z0-9_]+) so the comma join is safe.
            "gpu_idle_since_epoch": str(gpu_idle_since_epoch),
            "gpu_idle_advised_phases": ",".join(sorted(gpu_idle_advised_phases)),
            # #664 escalation tier per-phase de-dup (shares the idle span
            # above). Same comma-join contract as the advised set.
            "gpu_idle_escalated_phases": ",".join(sorted(gpu_idle_escalated_phases)),
            # Persist the current CPU sample (observability) so the JSON
            # line / next tick can read the latest probe. Stored as the
            # literal probe string (``"unknown"`` or a float-as-string) so
            # `_parse_session_cpu` treats it consistently with the live
            # probe value.
            "session_cpu_secs": current_session_cpu,
            # #951: VM-clock timestamp OF the session_cpu_secs sample above —
            # the pair is written atomically together, so this epoch is the
            # denominator anchor for the NEXT tick's burn-rate delta.
            "session_cpu_sample_epoch": str(now_epoch),
            # #951: this tick's per-tick burn rate (cores) vs the prior
            # persisted sample; "unknown" when not computable (fail-safe —
            # the material-CPU veto reads it as no-signal next tick).
            "session_cpu_rate_cores": (
                "unknown" if session_cpu_rate is None else f"{session_cpu_rate:.4f}"
            ),
            # Persist the running MAXIMUM cumulative CPU observed across all
            # ticks (#658). This is the baseline the NEXT tick's
            # advancing-decision compares against — a current sample below
            # it is a multi-shard child-exit accounting artifact, not a
            # stall. Monotonic: a sub-max current sample never lowers it.
            "max_cpu_secs": max_session_cpu,
            # #826 zombie-override 2-tick persistence. Recomputed from
            # scratch each tick (0 unless the override block set it), so a
            # vetoed-fresh tick, a candidate-free tick, and any non-running
            # verdict all write "0" — a one-tick transient never
            # accumulates across a healthy gap.
            "zombie_streak": str(zombie_streak),
            # #873 m-of-N GPU-width advisory span + stable idle set +
            # per-phase de-dup (same comma-join contract as the idle sets).
            "gpu_width_since_epoch": str(gpu_width_since_epoch),
            "gpu_width_idle_set": ",".join(str(i) for i in gpu_width_idle_set),
            "gpu_width_advised_phases": ",".join(sorted(gpu_width_advised_phases)),
            # plan §3 under-parallelization warning: span + per-RUN warned flag
            # (run-scoped via _TRIPWIRE_STATE_KEYS — re-armed on a fresh run).
            "gpu_underparallel_since_epoch": str(gpu_underparallel_since_epoch),
            "gpu_underparallel_warned": "1" if gpu_underparallel_warned else "0",
            # #873 phase-ETA tripwire dedup keys + the one-shot missing-
            # budget warn flag. Phase names match PHASE_RE ([a-z0-9_]+) and
            # __run_total__ shares the charset, so the comma join is safe.
            "eta_deviation_posted_keys": ",".join(sorted(eta_posted_keys)),
            "eta_budget_warned": "1" if eta_budget_warned else "0",
            # #873 run-scope anchor: the epm:run-launched epoch the tripwire
            # dedup keys above belong to (AC #6). A fresh launch clears them.
            "tripwire_run_epoch": str(tripwire_run_epoch),
            # #983 post-done phase-consistency guard: the matched done
            # line's identity (truncated text), when + on which pod it was
            # accepted, and the once-per-episode dedup flag. Voided only by
            # the direct run-scope clamp / a pod change (NOT via
            # _TRIPWIRE_STATE_KEYS — the guard has its own natural epoch).
            "post_done_line": post_done_line,
            "post_done_epoch": str(post_done_epoch),
            "post_done_pod": post_done_pod,
            "post_done_advisory_posted": "1" if post_done_posted_flag else "0",
        },
    )

    # Pick the tail excerpt from whichever log is the fresher signal: if
    # cell logs exist AND are fresher than the main log, surface the cell
    # tail so operators see what's actually happening (training-step
    # output, eval progress) rather than the stale dispatcher tail. When
    # both are zero (no logs yet) or the main log is fresher, fall back
    # to the main-log tail (preserves prior behavior for non-cell runs).
    tail_excerpt, crash_signature = _tail_excerpt_and_crash_signature(
        probe,
        status=status,
        mtime_epoch=mtime_epoch,
        cell_mtime_epoch=cell_mtime_epoch,
        phase_log_mtime_epoch=phase_log_mtime_epoch,
        shard_log_mtime_epoch=shard_log_mtime_epoch,
    )
    return PollResult(
        status=status,
        current_phase=current_phase,
        new_milestone=new_milestone,
        last_log_mtime_sec_ago=min(last_mtime_ago, 10**9),
        pid_alive=pid_alive,
        pid_file_missing=pid_file_missing,
        pid_file_stale_vs_marker=pid_file_stale_vs_marker,
        log_tail_excerpt=tail_excerpt,
        gate=gate,
        sentinels_processed=sentinels_processed,
        phase_log_mtime_sec_ago=min(phase_log_mtime_ago, 10**9),
        shard_log_mtime_sec_ago=min(shard_log_mtime_ago, 10**9),
        gpu_util=gpu_util,
        gpu_idle_advisory_posted=gpu_idle_advisory_posted,
        gpu_idle_escalation_posted=gpu_idle_escalation_posted,
        gpu_width_advisory_posted=gpu_width_posted,
        gpu_underparallel_warning_posted=gpu_underparallel_posted,
        eta_deviation_posted=eta_posted,
        session_cpu_secs=current_session_cpu,
        cpu_advancing=cpu_advancing,
        next_interval=next_interval,
        stall_reason=stall_reason,
        crash_signature=crash_signature,
        post_done_phase_advisory_posted=post_done_posted,
        post_done_phase_lines=post_done_new_lines,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--issue", type=int, required=True, help="Task / issue number.")
    parser.add_argument("--pod", required=True, help="SSH host alias (e.g. epm-issue-137).")
    parser.add_argument("--log", required=True, help="Remote log file path.")
    parser.add_argument("--pid-file", required=True, help="Remote PID file path.")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help=(
            "Local cache JSON (default: <main-checkout>/.claude/cache/"
            "poll-pipeline-<N>.json, resolved cwd-independently)."
        ),
    )
    parser.add_argument(
        "--stall-sec",
        type=int,
        default=int(os.environ.get("EPM_POLL_STALL_SEC", DEFAULT_STALL_SEC)),
        help=(
            "Seconds of log-mtime silence before declaring the run stalled "
            f"(default {DEFAULT_STALL_SEC}). Raise for workloads with sparse "
            "log cadence (e.g. checkpoint-cadence-only logging at >15min "
            "intervals). Falls back to the EPM_POLL_STALL_SEC env var when "
            "the flag is not set."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Log to stderr at DEBUG level.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    state_file = args.state_file or (DEFAULT_STATE_DIR / f"poll-pipeline-{args.issue}.json")

    result = poll_once(
        issue=args.issue,
        pod=args.pod,
        log_path=args.log,
        pid_file=args.pid_file,
        state_file=state_file,
        stall_sec=args.stall_sec,
    )

    print(
        json.dumps(
            {
                "status": result.status,
                "current_phase": result.current_phase,
                "new_milestone": result.new_milestone,
                "last_log_mtime_sec_ago": result.last_log_mtime_sec_ago,
                "pid_alive": result.pid_alive,
                "pid_file_missing": result.pid_file_missing,
                "pid_file_stale_vs_marker": result.pid_file_stale_vs_marker,
                "log_tail_excerpt": result.log_tail_excerpt,
                "gate": result.gate,
                "sentinels_processed": result.sentinels_processed,
                "phase_log_mtime_sec_ago": result.phase_log_mtime_sec_ago,
                "shard_log_mtime_sec_ago": result.shard_log_mtime_sec_ago,
                "gpu_util": result.gpu_util,
                "gpu_idle_advisory_posted": result.gpu_idle_advisory_posted,
                "gpu_idle_escalation_posted": result.gpu_idle_escalation_posted,
                "gpu_width_advisory_posted": result.gpu_width_advisory_posted,
                "gpu_underparallel_warning_posted": result.gpu_underparallel_warning_posted,
                "eta_deviation_posted": result.eta_deviation_posted,
                "session_cpu_secs": result.session_cpu_secs,
                "cpu_advancing": result.cpu_advancing,
                "next_interval": result.next_interval,
                "stall_reason": result.stall_reason,
                # #983 post-done phase-consistency guard surfaces.
                "post_done_phase_advisory_posted": result.post_done_phase_advisory_posted,
                "post_done_phase_lines": list(result.post_done_phase_lines),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

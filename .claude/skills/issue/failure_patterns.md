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

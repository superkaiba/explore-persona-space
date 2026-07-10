---
name: Anthropic batches are long-running (10-90 min typical, 24h SLA) — never wait inline
description: Any script calling messages.batches.create is long-running regardless of what the brief says; a fresh batch can sit at processing=N, succeeded=0 for >60 min while the non-batch API is healthy. Launch detached (setsid trio + pid-file contract), persist batch_id, exit.
metadata:
  type: feedback
---

Before accepting a brief's "fast single-shot data prep, wait inline" claim, grep the script for `messages.batches.create` / `anthropic.batches` / `submit_response_batch`. Any Anthropic batch in the path means 10-90 min typical wall time, up to the 24h SLA — far beyond a subagent turn.

**Why:** #382 round-1 (2026-05-26) — brief said "wait inline"; the data-gen script submits a Sonnet 4.5 batch; the SSH command timed out at 5 min with the batch still queued. #331 round-2 (2026-05-11) — an 18.4K-request batch sat at `processing=N, succeeded=0` for the full 60-min script cap (24 polls, zero progress) while the non-batch `/v1/messages` answered in 1.6s; the script's `max_elapsed_s=3600` RuntimeError lost the run. Queue backlog at 0/N is allowed behavior on Anthropic's side, not an outage.

**How to apply:**
1. Launch with the detachment trio `setsid nohup ... < /dev/null > log 2>&1 &` (never bare `nohup ... &`), capturing the PID per pod-side-reporting.md § Pid-file launch contract: preferred = the launcher script whose `echo $$ > /workspace/logs/issue-<N>.pid` overwrites pre-exec (after `exec`, `$$` IS the driver); rare no-launcher relaunch = atomic tmp+mv (`printf '%s\n' "$PID" > /workspace/logs/issue-<N>.pid.tmp && mv /workspace/logs/issue-<N>.pid.tmp /workspace/logs/issue-<N>.pid`). post `epm:failure v1 failure_class: infra` ("data-gen in progress" + launch command), EXIT. Verify detachment via `ps -p <pid> -o ppid` (PPID 1). Use log-file mtime, not tail, for freshness — stdout is buffered to files.
2. Recommend to the implementer: persist `batch.id` to disk right after create so a re-run ATTACHES via `retrieve()` instead of re-submitting (and re-paying); raise/configure the poll cap to ≥4h for >5K-request batches.
3. Diagnostic when a batch looks stuck: hit non-batch `/v1/messages` with a tiny request — <2s response means the API is healthy and your batch is just queued; an error/hang means a real outage.

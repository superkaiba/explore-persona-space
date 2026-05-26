---
name: datagen-anthropic-batch-long-running
description: Issue #382 data-gen script submits Anthropic Sonnet 4.5 batch and polls; experimenter brief assumed it was a fast "single-shot data prep" but the response batch takes 30-90 min minimum. Treat ANY datagen script that calls Anthropic batches as long-running.
metadata:
  type: feedback
---

When the experimenter brief says "data prep is a fast single-shot, wait inline for it",
verify by inspecting the script for `messages.batches.create` BEFORE you commit to a
foreground wait. If any Anthropic batch is in the path, the wall time is 10-90 min
(typical) up to 24h (Anthropic SLA), NOT seconds.

**Why:** Subagents have ONE turn (~10 min budget). Foregrounding an Anthropic batch
inside a subagent SSH call burns the entire turn AND deceives the orchestrator about
launch status. Task #382 round-1 hit this: brief said "wait inline (single-shot data
prep, NOT a long-running training job)", but `scripts/generate_issue382_marker_install.py`
submits a Sonnet 4.5 batch for ~256 question-batches; observed: question batch ~3 min,
response batch >10 min and still in queue when the SSH command timed out at 5 min.

**How to apply:**
1. Grep the data-gen script for `messages.batches.create`, `anthropic.batches`, or
   `submit_response_batch` BEFORE accepting the brief's "single-shot" claim.
2. If found: launch with full `nohup ... < /dev/null > log 2>&1 &`, capture PID, then
   immediately post `epm:failure v1` with `failure_class: infra` reason=
   "data-gen Anthropic batch in progress" + the launch command for the orchestrator
   to fire once data exists. EXIT. The orchestrator re-dispatches experimenter.
3. The data-gen process orphans cleanly to init (PPID 1) once SSH disconnects, as
   long as you used `< /dev/null` to detach stdin. Verify with `ps -p <pid> -o ppid`.
4. Stdout from `print()` is buffered when stdout is a file. Use log file mtime
   (`stat -c %y log`) instead of `tail` line count as the freshness signal.
5. Related: [[anthropic-batch-queue-backlog]] for why batch can sit at 0/N for >60 min.

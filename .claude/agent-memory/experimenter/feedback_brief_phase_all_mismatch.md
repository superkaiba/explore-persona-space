---
name: brief-phase-all-mismatch
description: Briefs that say "launch with --phase all" can mismatch the script's actual argparse choices; check the wrapper script the previous round used before retyping a command
metadata:
  type: feedback
---

When a re-launch brief specifies a literal command like `uv run python scripts/run_experiment_X.py --phase all` and the previous round successfully launched a long-running multi-phase pipeline, verify the script accepts `--phase all` before launching. Phased dispatchers commonly only accept named phases (`preflight`, `dataset-gen`, `phase0-calibration`, `base-eval`, `train`, `full-eval`, `aggregate`, `upload`) — `all` is invalid and argparse exits code 2 within a second, leaving an empty log + no traceback (just usage text).

**Why:** Burned at #389 v6 re-launch (2026-05-26). Brief said `--phase all`; script rejected it; PID died in ~5s with no Python traceback (argparse error). The previous-round successful launch used `bash /workspace/launch_issue_389.sh` — a wrapper that iterates through every named phase and parallelizes the train waves. The wrapper was the canonical entry point and it was already on disk from round 1.

**How to apply:** Before executing a re-launch command from a brief:
1. Look up the previous round's `epm:run-launched` marker note for the actual `cmd='...'` string used.
2. Check the pod for any `launch_issue_<N>.sh` wrapper (or `dispatch.sh`, `run.sh`) at `/workspace/`.
3. If a wrapper exists, prefer it — it already encodes the correct phase ordering, parallelism, and logging.
4. If the brief says `--phase all` or any literal that smells off, grep the script's argparse `choices=[...]` to verify.

Cost of getting this wrong: one wasted launch cycle + a confusing "PID gone in 5s" diagnostic. Cheap to avoid.

---
name: inline-gate-queue-and-probe-traps
description: Inline payload lint gate on a fold round — fleet-queue ELAPSED undercounts under load, pgrep matches own watchers + sibling pytest argv, and bare hf_hub_download in a new script is a BLOCK.
metadata:
  type: feedback
---

Three traps when an analyzer fold round runs the Step-3 inline payload lint gate (#2203 fold, 2026-08-10).

1. The gate-fleet bounded queue (`sleep 60` loop, cap 2700s) counts 60s per ITERATION, but each `step9c_baseline.py probe --fleet` call itself takes minutes under fleet contention — real wall time blows far past the cap while ELAPSED sits low, and a 600s bg-Bash timeout leaves the launcher shell alive holding the single-flight slot. **Why:** the loop measures iterations, not time. **How to apply:** after ~15 real minutes of queue, kill the launcher by captured PID and relaunch the gate directly with the `[gate-fleet] cap-expired` fail-open line; key waits on the gate PYTHON pid, not the wrapper.
2. `pgrep -f "inline_lint_gate.py --issue <N>"` false-positives on (a) your own bg watcher/sleep shells (their argv embeds the pattern) and (b) a SIBLING issue's mapped pytest whose argv lists `tests/test_inline_lint_gate.py`. **How to apply:** verify with a full `pgrep -af` listing and exclude `bash -c` + `pytest` lines before concluding the gate is running; bracket one pattern char only saves the probe's own clause.
3. A new round script with a bare `hf_hub_download(...)` is a gate BLOCK (`[live-hf-retry-routing]`). **How to apply:** wrap Hub calls in `retry_transient(lambda: ..., what="...")` from `explore_persona_space.orchestrate.hub` at write time.

Also: mapped tests for a payload can be EMPTY (`select_step9c_tests.py --map-files` prints nothing) — the gate is then just the ~3-8 min lint leg; a BLOCK fix re-run needs the full gate again (cert is content-hash keyed).

---
name: Preflight --json is pretty-printed; last-line parsers crash
description: Dispatcher preflight gates that json.loads(splitlines()[-1]) of orchestrate.preflight --json output crash on '}' regardless of preflight content; recovery = verify preflight manually + relaunch with the dispatcher's --skip-preflight flag
type: feedback
---

`orchestrate.preflight --json` emits PRETTY-PRINTED (multi-line, indented)
JSON, not a single line. A pod dispatcher gate shaped
`json.loads(proc.stdout.strip().splitlines()[-1])` therefore dies with
`JSONDecodeError: Expecting value: line 1 column 1 (char 0)` (the last line
is `}`) — even when preflight PASSes. Crash signature on sight: instant
(<1 min) death, log = JSONDecodeError chained into
`RuntimeError: preflight emitted unparseable output: <pretty-JSON tail>`.

**Why:** burned at #602 first launch (2026-06-11). The behind-origin/main
tolerance was correctly implemented but unreachable — the parse crashed
before the error filter.

**How to apply:** this is `failure_class: code`, but if the dispatcher
exposes `--skip-preflight` (most do), do NOT bounce to implementer: verify
the preflight conditions yourself (run `orchestrate.preflight --json` on
the pod, check errors == [behind-origin/main] only, GPUs/disk/env), then
relaunch with `--skip-preflight` and record the assumption in the
`epm:run-launched` note. Implementer-side fix: parse with
`json.loads(proc.stdout)` (whole payload), never last-line.

Two adjacent traps from the same session: (a) SSH MCP `ssh_execute` hard-caps
at ~30 s client-side even when `timeout` param is larger — never put long
`sleep`s inside it; (b) appending `& echo $! > pidfile` to a `cd && ... &&
setsid nohup ...` chain backgrounds the WHOLE chain and `$!` is the subshell,
not the launcher — isolate the launch as `(setsid nohup ... &)` then resolve
the real child via `pgrep -f "venv/bin/python3 <script>"` after a short sleep.
The pgrep pattern MUST be a string absent from your own SSH command: `pgrep -f
'bash <script>.sh'` self-matches the SSH session's `sh -c` wrapper (which
carries the full command text) and writes a transient PID to the pidfile that
reads PID_DEAD on the next probe while the run is healthy (re-hit at #602
relaunch, 2026-06-11; also: a wrapper that `exec`s into `uv run` leaves NO
bash process matching the script name — probe the python child).

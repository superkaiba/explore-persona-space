---
name: i543-rig per-phase launches need --measure-bhat first
description: run_issue543_ratio.py --phase phase1 crashes in seconds with FileNotFoundError bhat.json unless --measure-bhat ran first; only the #543 --driver path auto-runs it. Composed launcher glue must include an idempotent Step-0 measure-bhat. Burned at #570 v1 launch (2026-06-11).
type: feedback
---

On the #543/#557/#570 rig family, `run_issue543_ratio.py --phase phase1`
hard-requires `data/issue543_ratio_survival/bhat.json` (`_load_bhat()` raises
`FileNotFoundError` within seconds). Only the all-cells `--driver` mode runs
the b-hat measure automatically — and `--driver` is REJECTED with
`--issue-ns`, so every namespaced per-phase launch path (the only #570-style
path) needs explicit glue.

**Why:** burned at #570 first launch (2026-06-11): all 3 parallel phase-1
seeds died at t+0s on a fresh pod. Fix was one idempotent step at glue top:
`[ -f data/issue543_ratio_survival/bhat.json ] || run_issue543_ratio.py
--measure-bhat --issue-ns 570 --gpu 0` (~15 s with warm HF cache; also
sanity-prints b-hat — #570 measured −25.855 nat vs #543's −25.88, gate
[−30, −15]).

**How to apply:** any experimenter-composed launcher glue for this rig that
invokes `--phase phase1` directly MUST run the measure-bhat step first.
Related cleanup gotcha from the same incident: a pkill'ed ladder/eval leaves
a renamed `VLLM::EngineCore` worker holding ~56 GiB GPU memory that
`pgrep -f <script name>` does NOT match — before relaunch, `ps aux | grep
EngineCore` and kill it, and never `pkill -f <pattern>` where the pattern
appears in your own wrapping `sh -c` cmdline (it self-kills the probe; use
`pgrep -af "patter[n]"` bracket trick).

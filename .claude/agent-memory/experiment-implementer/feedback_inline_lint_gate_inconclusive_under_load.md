---
name: inline-lint-gate-inconclusive-under-load
description: inline_lint_gate returns INCONCLUSIVE (not a real verdict) when shared-VM load1 > EPM_GATE_LOAD_MAX=20 and the pytest leg goes red — re-run when load drops for a conclusive PASS/BLOCK; and it requires load_dotenv() before ANY module-top heavy import even in credential-free plot/analysis scripts
metadata:
  type: feedback
---

`scripts/inline_lint_gate.py` (the Step-9a-ter inline payload certification for
direct-to-main code) has two behaviors that waste a round if you don't expect
them:

1. **INCONCLUSIVE ≠ a verdict, and it is LOAD-DRIVEN.** The gate compares
   shared-VM `load1` against `EPM_GATE_LOAD_MAX=20`. When load is above that and
   the mapped-pytest leg goes red, the gate prints
   `INCONCLUSIVE (pytest-leg red under load — not payload-attributed; re-run when
   load drops)` and does NOT certify that file. INCONCLUSIVE is "never
   push-clean" per the gate contract — you must re-run when `cat /proc/loadavg`
   is under ~20 to get a conclusive PASS/BLOCK. On the shared VM (routinely
   load 30-45 across ~15 sessions) the gate will wait up to 300s for load to
   drop, then run anyway; a first attempt under high load can leave you with a
   mix of certified + inconclusive files. Re-run just the inconclusive file
   (`--paths <one file>`) — its narrower test selection clears faster.

2. **load_dotenv() must precede EVERY module-top heavy import — matplotlib /
   numpy / torch — even in a pure plotter or analysis script that needs no
   credentials.** A tracked-file-scan test asserts no module-top heavy import
   before a `load_dotenv(` call and will BLOCK otherwise. Canonical header even
   for a credential-free script:
   ```python
   from __future__ import annotations
   from explore_persona_space.orchestrate.env import load_dotenv
   load_dotenv()
   import json  # noqa: E402
   import matplotlib  # noqa: E402
   ```
   Add it preemptively to plot scripts to avoid a wasted gate round.

**Why:** on the #1739 jbmine compliance rerun (2026-08-18), a matplotlib plot
script came back INCONCLUSIVE under load 44 on the first gate run; when load
dropped it flipped to a REAL BLOCK on the missing-load_dotenv invariant. Direct
`ruff check` / `ruff format --check` are load-independent and confirm the
lint-clean half instantly — run them first to separate "my file is dirty" from
"the gate couldn't reach a verdict under load".

**How to apply:** before certifying a direct-to-main payload, (a) put
load_dotenv() above module-top heavy imports in ALL scripts incl. plotters;
(b) if the gate says INCONCLUSIVE, don't treat it as pass or fail — re-run the
named file when `/proc/loadavg` load1 < 20. Related:
[[no-flags-workflow-lint-before-push]].

**Sibling (2026-08-19, #2378 r5): size your own `timeout` bound on the no-flags
`workflow_lint.py` run ≥ ~900-1500 s under fleet load.** Even at load1 ~16 the
full no-flags run exceeded 480 s AND 560 s bounds (two rc=124 kills = your own
timeout, INCONCLUSIVE — not verdicts); the third run at a 1500 s bound completed
in ~10 min and surfaced a REAL 1-error FAIL the truncated runs never reached
(jsonl-splitlines fires late in the check order). An rc=124 whose `tail` shows
only WARN lines is a TRUNCATED run — later checks never executed; never report
it as a lint PASS.

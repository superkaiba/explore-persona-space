---
name: Fenced dispatcher-block extraction probe (SMOKE=0 python legs)
description: How to execute a bash dispatcher's production-only embedded-python block (registry_lines heredoc) without running the production phase — sed-extract the body + replicate the preamble at SMOKE_ENV=0
type: feedback
---

Production-fenced (`if [ "$SMOKE" -eq 0 ]`) embedded-python blocks inside
bash dispatchers (pilot re-projections, deviation-sentinel legs) are
unreachable by every smoke and un-sourceable for unit tests (`set -euo
pipefail` + main dispatch at load). The #1481 fenced-branch runtime-probe
duty still binds.

**Why:** a NameError/typo in such a block first fires at production phase
start on a billed pod; `bash -n` cannot see inside the python heredoc.

**How to apply:** `sed -n '<start>,<end>p' dispatcher.sh > /tmp/body.py`
(the exact lines between the opening `registry_lines_v2 '` / `<<'PY'` and
the closing quote/`PY`), then run it through a verbatim replica of the
`registry_lines_v2` preamble (`uv run python - "$(cat /tmp/body.py)"
<<'PY' ... exec(sys.argv[1]) PY`) with `SMOKE_ENV=0` + the env vars the
call site sets (G1_WALL/NGPU_ENV/PLANNED/OUT_ENV/DONE_ENV), pointing
OUT/DONE at scratch. Drive BOTH branches (e.g. wall=300 → OK, wall=3000 →
OVER-2x) and then run the downstream shell one-liner (the sentinel-leg
JSON parse) against the produced flag file. Worked on #1336 unit D
(fit_v2 + ladder pilots: pend=45 resolved from the production registry on
the VM, both branches fired, sentinel fields verified). Record the probe
in the marker's per-arm rows / `## Smoke run`.
